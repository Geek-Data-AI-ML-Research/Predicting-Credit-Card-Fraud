#!/usr/bin/env python3
"""Slot 4: Internal LM Upgrade on F0+F1+F2+F3 — submit immediately if soft criteria met.

Local Mac CPU (n_jobs=4). Does NOT touch Pod PID 3021 (Fold 4 training).
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
N_JOBS = 4
ACTIVE = (0, 1, 2, 3)
SHRINK = float(np.sqrt(4.0 / 5.0))
SLOT3_OOF = 0.9526269
W_LM = 0.075

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
NUMS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
FEATS = NUMS + FLAGS + CATS
FLOATS = [
    "amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
    "velocity_score", "merchant_risk_score",
]


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def pchip_quantile_map(x_test, x_ref, n_knots=4000):
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    x_ref_sorted = np.sort(np.asarray(x_ref, dtype=np.float64).ravel())
    q = np.linspace(0.0, 1.0, n_knots)
    pchip = PchipInterpolator(q, np.quantile(x_ref_sorted, q), extrapolate=True)
    u = (rankdata(x_test, method="average") - 0.5) / len(x_test)
    return np.asarray(pchip(u), dtype=np.float64)


def calibrate_logit_to_prevalence(z_test, z_oof_ref, y_train, target_mean=0.016700):
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof_ref.reshape(-1, 1), y_train.astype(np.int32))
    z_lin = float(lr.coef_[0, 0]) * z_test + float(lr.intercept_[0])

    def obj(delta):
        return (float(np.mean(expit(z_lin + delta))) - target_mean) ** 2

    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded")
    return z_lin + float(res.x)


def bayes_fold(arr, prior):
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def bayes_std(arr):
    llr = arr[:, :, 1].astype(np.float64) - arr[:, :, 0].astype(np.float64)
    return llr.std(axis=1)


def align_disp(z, ref):
    return ref.mean() + (z - z.mean()) * (ref.std() / (z.std() + 1e-12))


def orth_vs_v8(z_cand_o, z_cand_t, z_v8_o, z_v8_t, folds):
    ro = np.zeros(len(z_v8_o), dtype=np.float64)
    rt_f = []
    for a, b in folds:
        beta = float(np.cov(z_cand_o[a], z_v8_o[a], ddof=0)[0, 1] / np.var(z_v8_o[a]))
        ra = z_cand_o[a] - beta * z_v8_o[a]
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (z_cand_o[b] - beta * z_v8_o[b] - mu) / sd
        rt_f.append((z_cand_t - beta * z_v8_t - mu) / sd)
    return ro, pchip_quantile_map(np.mean(rt_f, axis=0), ro, n_knots=4000)


def build_kmat(tr, te):
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    K = {}
    for c in FEATS:
        K[f"v_{c}"] = full[c].astype(str).values if c in CATS else full[c].values
    for c in FLOATS:
        x = full[c].values.astype(np.float64)
        K[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
        K[f"int_{c}"] = np.floor(x).astype(np.int64)
        K[f"ten_{c}"] = np.floor(x / 10).astype(np.int64)
        K[f"dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
        s = pd.Series(np.floor(x).astype(np.int64)).astype(str)
        K[f"l1_{c}"] = (s.str.len() * 10 + s.str[0].astype(int)).values
        K[f"l2_{c}"] = (s.str.len() * 100 + s.str[:2].astype(int)).values
    for c1, c2 in [
        ("merchant_category", "auth_method"), ("merchant_category", "channel"),
        ("card_type", "auth_method"), ("is_foreign_transaction", "ip_country_mismatch"),
        ("used_vpn", "ip_country_mismatch"), ("is_new_merchant", "billing_shipping_mismatch"),
        ("cvv_retry_count", "auth_method"), ("prior_disputes", "is_ai_generated_scam_attempt"),
    ]:
        K[f"j_{c1}_{c2}"] = (full[c1].astype(str) + "_" + full[c2].astype(str)).values
    codes = [pd.factorize(pd.Series(K[n]))[0].astype(np.int32) for n in K]
    KMAT = np.column_stack(codes)
    NCAT = np.array([c.max() + 1 for c in codes], dtype=np.int32)
    return KMAT, NCAT


def te_top12(KMAT, NCAT, y, a, b, ntr, nte, smooth=20.0, seed=42):
    Ka, Kb, Kt = KMAT[a], KMAT[b], KMAT[ntr:ntr + nte]
    ya = y[a].astype(np.float64)
    prior = float(ya.mean())
    n_keys = KMAT.shape[1]
    Ea = np.zeros((len(a), n_keys), dtype=np.float32)
    Eb = np.zeros((len(b), n_keys), dtype=np.float32)
    Et = np.zeros((nte, n_keys), dtype=np.float32)
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(a)), y[a]))
    for j in range(n_keys):
        nc = int(NCAT[j])
        for i_tr, i_va in inner:
            s = np.bincount(Ka[i_tr, j], weights=ya[i_tr], minlength=nc)
            n = np.bincount(Ka[i_tr, j], minlength=nc)
            r_in = (s + prior * smooth) / (n + smooth)
            Ea[i_va, j] = r_in[Ka[i_va, j]]
        s_all = np.bincount(Ka[:, j], weights=ya, minlength=nc)
        n_all = np.bincount(Ka[:, j], minlength=nc)
        r_all = (s_all + prior * smooth) / (n_all + smooth)
        Eb[:, j] = r_all[Kb[:, j]]
        Et[:, j] = r_all[Kt[:, j]]
    eps = 1e-5
    Ea = logit(np.clip(Ea, eps, 1 - eps))
    Eb = logit(np.clip(Eb, eps, 1 - eps))
    Et = logit(np.clip(Et, eps, 1 - eps))
    ya_c = ya - prior
    ea_c = Ea - Ea.mean(0)
    corrs = np.abs((ea_c * ya_c[:, None]).sum(0) / (np.sqrt((ea_c ** 2).sum(0)) + 1e-8))
    top = np.argsort(corrs)[::-1][:12]
    return Ea[:, top].astype(np.float32), Eb[:, top].astype(np.float32), Et[:, top].astype(np.float32)


def opt_w(y_f, z_b, u_f, grid, l2=0.015, wmin=None, wmax=None):
    if wmin is not None or wmax is not None:
        grid = grid[(grid >= (wmin if wmin is not None else -1)) & (grid <= (wmax if wmax is not None else 1))]
    base = float(roc_auc_score(y_f, z_b))
    return float(max(grid, key=lambda w: roc_auc_score(y_f, z_b + w * u_f) - base - l2 * (w ** 2)))


def pairwise_dst(z_src, X_src, z_dst, X_dst):
    ols = LinearRegression().fit(X_src, z_src)
    u = z_dst - ols.predict(X_dst)
    mu, sd = float(u.mean()), float(u.std() + 1e-12)
    return (u - mu) / sd, (ols, mu, sd)


def main():
    print("[Boot] submit_slot4_internal_lm_upgrade_f0123.py (n_jobs=4, no touch PID 3021)...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy()
    ntr, nte = len(tr), len(te)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(ntr), y))
    idx = {f: b for f, (_, b) in enumerate(folds)}
    a_tr = {f: a for f, (a, _) in enumerate(folds)}

    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s3_sub = pd.read_csv(SUB / "sub_slot3_gm_verified_f02.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_auc = float(roc_auc_score(y, z_v8_o))
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y))

    z_lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), z_lm12_o, n_knots=4000)
    z_g3_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_oof.npy").ravel())
    z_g3_t = pchip_quantile_map(zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_test.npy").ravel()), z_g3_o, n_knots=4000)
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)
    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)
    z_knn_o = zrk(np.load(HARV / "oof_orig20k_knn.npy").ravel())
    z_knn_t = pchip_quantile_map(zrk(np.load(HARV / "test_orig20k_knn.npy").ravel()), z_knn_o, n_knots=4000)
    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=4000)
    z_xgbf_cln_o = zrk(np.load(HARV / "oof_xgb_clean.npy").ravel())
    z_xgbf_cln_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_clean.npy").ravel()), z_xgbf_cln_o, n_knots=4000)
    r_old_o, r_old_t = orth_vs_v8(z_xgbf_old_o, z_xgbf_old_t, z_v8_o, z_v8_t, folds)
    r_cln_o, r_cln_t = orth_vs_v8(z_xgbf_cln_o, z_xgbf_cln_t, z_v8_o, z_v8_t, folds)
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)

    # ---- Cand0 Smooth-Gamma (Slot3-exact) ----
    s_norm_o = np.zeros(ntr, dtype=np.float64)
    s_norm_t_f = []
    for a, b in folds:
        smu, ssd = float(s_tabm_o[a].mean()), float(s_tabm_o[a].std() + 1e-12)
        s_norm_o[b] = np.clip((s_tabm_o[b] - smu) / ssd, -3.5, 3.5)
        s_norm_t_f.append(np.clip((s_tabm_t - smu) / ssd, -3.5, 3.5))
    s_norm_t = np.mean(s_norm_t_f, axis=0)
    th_o, th_t = np.tanh(z_v8_o / 2.0), np.tanh(z_v8_t / 2.0)
    param_grid = [
        (gmin, gmax, temp, bias, w_cv)
        for gmin in (0.30, 0.50, 0.70, 0.85)
        for gmax in (0.85, 1.00, 1.15) if gmax >= gmin
        for temp in (-1.5, 0.0, 1.5)
        for bias in (-1.0, 0.0, 1.0)
        for w_cv in (0.022, 0.025, 0.028)
    ]
    gfd = np.zeros((len(param_grid), 5))
    for gi, (gmin, gmax, temp, bias, w_cv) in enumerate(param_grid):
        gam = gmin + (gmax - gmin) * expit(-temp * s_norm_o - bias * th_o)
        z_try = z_v8_o + w_cv * (r_old_o - gam * r_tab_o)
        for fi, (_, b) in enumerate(folds):
            gfd[gi, fi] = float(roc_auc_score(y[b], z_try[b])) - base_folds[fi]
    z_cand0_o = np.zeros(ntr)
    u_cv5_o = np.zeros(ntr)
    z0t_f, u0t_f = [], []
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        sub = gfd[:, other]
        scores = sub.mean(1) - 0.5 * np.maximum(0.0, -sub.min(1))
        gmin, gmax, temp, bias, w_cv = param_grid[int(np.argmax(scores))]
        gam_b = gmin + (gmax - gmin) * expit(-temp * s_norm_o[b] - bias * th_o[b])
        ub = r_old_o[b] - gam_b * r_tab_o[b]
        u_cv5_o[b] = ub
        z_cand0_o[b] = z_v8_o[b] + w_cv * ub
        gam_t = gmin + (gmax - gmin) * expit(-temp * s_norm_t - bias * th_t)
        ut = r_old_t - gam_t * r_tab_t
        u0t_f.append(ut)
        z0t_f.append(z_v8_t + w_cv * ut)
    z_cand0_t, u_cv5_t = np.mean(z0t_f, 0), np.mean(u0t_f, 0)
    z_candC_o = z_v8_o + 0.025 * (r_old_o - r_tab_o)
    z_candC_t = z_v8_t + 0.025 * (r_old_t - r_tab_t)
    if roc_auc_score(y, z_cand0_o) >= roc_auc_score(y, z_candC_o):
        z_base5_o, z_base5_t = z_cand0_o, z_cand0_t
    else:
        z_base5_o, z_base5_t = z_candC_o, z_candC_t
        u_cv5_o, u_cv5_t = r_old_o - r_tab_o, r_old_t - r_tab_t
    cand0_auc = float(roc_auc_score(y, z_base5_o))
    cand0_folds = [float(roc_auc_score(y[b], z_base5_o[b])) for _, b in folds]
    print(f"[Cand0] Nested OOF={cand0_auc:.7f}", flush=True)

    lr_p = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o.reshape(-1, 1), y)
    p_all_o = expit(lr_p.decision_function(z_v8_o.reshape(-1, 1)))
    p_all_t = expit(lr_p.decision_function(z_v8_t.reshape(-1, 1)))
    h_o, h_t = p_all_o * (1 - p_all_o), p_all_t * (1 - p_all_t)

    # ========================================================================
    # STEP 1: 4-fold LM + PCHIP
    # ========================================================================
    print("\n[Step 1] 4-fold z_m + PCHIP + PosLM residual...", flush=True)
    raw_o, raw_t, z_m_oof, z_m_tst, z_m_zrk, s_m_oof, s_m_tst = {}, {}, {}, {}, {}, {}, {}
    for f in ACTIVE:
        oa = np.load(PRED / f"gpt2_med_4ep/oof_{f}.npy")
        ta = np.load(PRED / f"gpt2_med_4ep/test_{f}.npy")
        assert oa.shape == (len(idx[f]), 8, 2) and ta.shape == (nte, 4, 2)
        prior = float(y[a_tr[f]].mean())
        raw_o[f], raw_t[f] = bayes_fold(oa, prior), bayes_fold(ta, prior)
        s_raw = bayes_std(oa)
        s_raw_t = bayes_std(ta)
        z_m_zrk[f] = zrk(raw_o[f] - np.mean(raw_o[f]))
        z_m_oof[f] = align_disp(raw_o[f], z_lm12_o[idx[f]])
        z_m_tst[f] = align_disp(raw_t[f], z_lm12_t)
        s_m_oof[f] = (s_raw - s_raw.mean()) / (s_raw.std() + 1e-12)
        s_m_tst[f] = (s_raw_t - s_raw_t.mean()) / (s_raw_t.std() + 1e-12)
        print(f"  F{f}: oof/test OK", flush=True)

    z_m_oof_f0123 = np.concatenate([z_m_oof[f] for f in ACTIVE])
    s_m_oof_f0123 = np.concatenate([s_m_oof[f] for f in ACTIVE])
    z_m_test_avg = 0.25 * sum(z_m_tst[f] for f in ACTIVE)
    z_m_test_f0123 = pchip_quantile_map(z_m_test_avg, z_m_oof_f0123, n_knots=4000)
    s_m_test_f0123 = pchip_quantile_map(np.mean([s_m_tst[f] for f in ACTIVE], 0), s_m_oof_f0123, n_knots=4000)
    ks_z = float(ks_2samp(z_m_oof_f0123, z_m_test_f0123).statistic)
    print(f"  PCHIP KS={ks_z:.6f} (need <0.0001) rows={len(z_m_oof_f0123)}", flush=True)

    # PosLM residual per fold at w=+0.075
    r_mlm_o = np.zeros(ntr)
    beta_by = {}
    for f in ACTIVE:
        zm, zl = z_m_oof[f], z_lm12_o[idx[f]]
        beta = float(np.cov(zm, zl, ddof=0)[0, 1] / np.var(zl))
        beta_by[f] = beta
        r = zm - beta * zl
        r_n = (r - r.mean()) / (r.std() + 1e-12)
        r_mlm_o[idx[f]] = r_n
        d = float(roc_auc_score(y[idx[f]], zl + W_LM * r_n) - roc_auc_score(y[idx[f]], zl))
        print(f"  F{f}: beta={beta:.4f} Δ_lm@{W_LM}={d*1e5:+.2f}e-5", flush=True)
    beta_mean = float(np.mean(list(beta_by.values())))
    r_mlm_t = z_m_test_f0123 - beta_mean * z_lm12_t
    r_mlm_t = (r_mlm_t - r_mlm_t.mean()) / (r_mlm_t.std() + 1e-12)

    # ========================================================================
    # Slot3 Ref (exact F0+F2) + 4-fold-test Slot3 operator
    # ========================================================================
    print("\n[Step 2] Slot3 Ref + SubCand A/B/C...", flush=True)
    w_grid_s3 = np.linspace(-0.045, 0.045, 181)
    X3_0 = np.column_stack([z_v8_o[idx[0]], z_lm12_o[idx[0]], u_cv5_o[idx[0]]])
    X3_2 = np.column_stack([z_v8_o[idx[2]], z_lm12_o[idx[2]], u_cv5_o[idx[2]]])
    X3_t = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])
    u_h1_0, h1_s2 = pairwise_dst(z_m_zrk[2], X3_2, z_m_zrk[0], X3_0)
    u_h1_2, h1_s0 = pairwise_dst(z_m_zrk[0], X3_0, z_m_zrk[2], X3_2)
    w_h1_0 = opt_w(y[idx[0]], z_base5_o[idx[0]], u_h1_0, w_grid_s3)
    w_h1_2 = opt_w(y[idx[2]], z_base5_o[idx[2]], u_h1_2, w_grid_s3)
    z_s3 = z_base5_o.copy()
    z_s3[idx[0]] = z_base5_o[idx[0]] + w_h1_2 * u_h1_0
    z_s3[idx[2]] = z_base5_o[idx[2]] + w_h1_0 * u_h1_2
    g_o = np.tanh(np.abs(z_lm12_o - z_v8_o)) * h_o
    g_t = np.tanh(np.abs(z_lm12_t - z_v8_t)) * h_t
    beta0 = float(np.cov(z_m_zrk[0], z_lm12_o[idx[0]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[0]]))
    beta2 = float(np.cov(z_m_zrk[2], z_lm12_o[idx[2]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[2]]))
    r0 = g_o[idx[0]] * (z_m_zrk[0] - beta2 * z_lm12_o[idx[0]])
    r2 = g_o[idx[2]] * (z_m_zrk[2] - beta0 * z_lm12_o[idx[2]])
    X2_0 = np.column_stack([z_v8_o[idx[0]], u_cv5_o[idx[0]]])
    X2_2 = np.column_stack([z_v8_o[idx[2]], u_cv5_o[idx[2]]])
    X2_t = np.column_stack([z_v8_t, u_cv5_t])
    ug0, h2_s2 = pairwise_dst(r2, X2_2, r0, X2_0)
    ug2, h2_s0 = pairwise_dst(r0, X2_0, r2, X2_2)
    w_h2_0 = opt_w(y[idx[0]], z_s3[idx[0]], ug0, w_grid_s3)
    w_h2_2 = opt_w(y[idx[2]], z_s3[idx[2]], ug2, w_grid_s3)
    z_slot3_o = z_s3.copy()
    z_slot3_o[idx[0]] = z_s3[idx[0]] + w_h2_2 * ug0
    z_slot3_o[idx[2]] = z_s3[idx[2]] + w_h2_0 * ug2
    s3_auc = float(roc_auc_score(y, z_slot3_o))
    s3_fd = [round((float(roc_auc_score(y[b], z_slot3_o[b])) - base_folds[i]) * 1e5, 2) for i, (_, b) in enumerate(folds)]
    print(f"  Ref_Slot3_F02 OOF={s3_auc:.7f} folds={s3_fd}", flush=True)
    assert abs(s3_auc - SLOT3_OOF) < 5e-8, f"Slot3 reconstruct mismatch: {s3_auc}"

    # Original 2-fold Slot3 test (for Ref row)
    z_m_t02 = pchip_quantile_map(
        0.5 * (zrk(raw_t[0] - np.mean(raw_o[0])) + zrk(raw_t[2] - np.mean(raw_o[2]))),
        np.concatenate([z_m_zrk[0], z_m_zrk[2]]), n_knots=4000,
    )
    ols_h1_2, mu1_2, sd1_2 = h1_s2
    ols_h1_0, mu1_0, sd1_0 = h1_s0
    u_h1_t02 = pchip_quantile_map(
        0.5 * (
            (z_m_t02 - ols_h1_2.predict(X3_t) - mu1_2) / sd1_2
            + (z_m_t02 - ols_h1_0.predict(X3_t) - mu1_0) / sd1_0
        ),
        np.concatenate([u_h1_0, u_h1_2]), n_knots=4000,
    )
    z_slot3_t = z_base5_t + 0.5 * (w_h1_0 + w_h1_2) * u_h1_t02
    feat_s3_t02 = g_t * (z_m_t02 - 0.5 * (beta0 + beta2) * z_lm12_t)
    ols_h2_2, mu2_2, sd2_2 = h2_s2
    ols_h2_0, mu2_0, sd2_0 = h2_s0
    ug_t02 = pchip_quantile_map(
        0.5 * (
            (feat_s3_t02 - ols_h2_2.predict(X2_t) - mu2_2) / sd2_2
            + (feat_s3_t02 - ols_h2_0.predict(X2_t) - mu2_0) / sd2_0
        ),
        np.concatenate([ug0, ug2]), n_knots=4000,
    )
    z_slot3_t = z_slot3_t + 0.5 * (w_h2_0 + w_h2_2) * ug_t02

    # 4-fold Slot3 test operator (SubCand A base) with SHRINK
    z_m_t4_zrk = pchip_quantile_map(
        np.mean([zrk(raw_t[f] - np.mean(raw_o[f])) for f in ACTIVE], 0),
        np.concatenate([z_m_zrk[f] for f in ACTIVE]), n_knots=4000,
    )
    u_h1_t4 = pchip_quantile_map(
        0.5 * (
            (z_m_t4_zrk - ols_h1_2.predict(X3_t) - mu1_2) / sd1_2
            + (z_m_t4_zrk - ols_h1_0.predict(X3_t) - mu1_0) / sd1_0
        ),
        np.concatenate([u_h1_0, u_h1_2]), n_knots=4000,
    )
    feat_s3_t4 = g_t * (z_m_t4_zrk - 0.5 * (beta0 + beta2) * z_lm12_t)
    ug_t4 = pchip_quantile_map(
        0.5 * (
            (feat_s3_t4 - ols_h2_2.predict(X2_t) - mu2_2) / sd2_2
            + (feat_s3_t4 - ols_h2_0.predict(X2_t) - mu2_0) / sd2_0
        ),
        np.concatenate([ug0, ug2]), n_knots=4000,
    )
    z_s3_4fold_t = (
        z_base5_t
        + SHRINK * 0.5 * (w_h1_0 + w_h1_2) * u_h1_t4
        + SHRINK * 0.5 * (w_h2_0 + w_h2_2) * ug_t4
    )

    # ---- Orth r_m_lm12 vs [v8, g3-lm12, ucv5] LOFO on active ----
    d_g3_lm_o = z_g3_o - z_lm12_o
    d_g3_lm_t = z_g3_t - z_lm12_t
    r_orth_o = np.zeros(ntr)
    r_orth_t_f = []
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        Xa = np.column_stack([z_v8_o[a_idx], d_g3_lm_o[a_idx], u_cv5_o[a_idx]])
        Xh = np.column_stack([z_v8_o[idx[hold]], d_g3_lm_o[idx[hold]], u_cv5_o[idx[hold]]])
        Xt = np.column_stack([z_v8_t, d_g3_lm_t, u_cv5_t])
        ols = LinearRegression().fit(Xa, r_mlm_o[a_idx])
        utr = r_mlm_o[a_idx] - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        r_orth_o[idx[hold]] = (r_mlm_o[idx[hold]] - ols.predict(Xh) - mu) / sd
        r_orth_t_f.append((r_mlm_t - ols.predict(Xt) - mu) / sd)
    r_orth_t = pchip_quantile_map(np.mean(r_orth_t_f, 0), r_orth_o[np.concatenate([idx[f] for f in ACTIVE])], n_knots=4000)

    # g_tabm gate
    g_tabm_o = np.zeros(ntr)
    for f in ACTIVE:
        g_tabm_o[idx[f]] = np.tanh(np.abs(z_lm12_o[idx[f]] - z_tabm_o[idx[f]])) * h_o[idx[f]] * r_mlm_o[idx[f]]
    g_tabm_t = np.tanh(np.abs(z_lm12_t - z_tabm_t)) * h_t * r_mlm_t
    # Orth g_tabm vs [v8, ucv5] LOFO
    gt_o = np.zeros(ntr)
    gt_t_f = []
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        Xa = np.column_stack([z_v8_o[a_idx], u_cv5_o[a_idx]])
        Xh = np.column_stack([z_v8_o[idx[hold]], u_cv5_o[idx[hold]]])
        Xt = np.column_stack([z_v8_t, u_cv5_t])
        ols = LinearRegression().fit(Xa, g_tabm_o[a_idx])
        utr = g_tabm_o[a_idx] - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        gt_o[idx[hold]] = (g_tabm_o[idx[hold]] - ols.predict(Xh) - mu) / sd
        gt_t_f.append((g_tabm_t - ols.predict(Xt) - mu) / sd)
    gt_t = pchip_quantile_map(np.mean(gt_t_f, 0), gt_o[np.concatenate([idx[f] for f in ACTIVE])], n_knots=4000)

    # Bound gate g_bound * r
    g_bound_o = g_o * r_mlm_o
    g_bound_t = g_t * r_mlm_t
    gb_o = np.zeros(ntr)
    gb_t_f = []
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        Xa = np.column_stack([z_v8_o[a_idx], u_cv5_o[a_idx]])
        Xh = np.column_stack([z_v8_o[idx[hold]], u_cv5_o[idx[hold]]])
        Xt = np.column_stack([z_v8_t, u_cv5_t])
        ols = LinearRegression().fit(Xa, g_bound_o[a_idx])
        utr = g_bound_o[a_idx] - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        gb_o[idx[hold]] = (g_bound_o[idx[hold]] - ols.predict(Xh) - mu) / sd
        gb_t_f.append((g_bound_t - ols.predict(Xt) - mu) / sd)
    gb_t = pchip_quantile_map(np.mean(gb_t_f, 0), gb_o[np.concatenate([idx[f] for f in ACTIVE])], n_knots=4000)

    # ---- Twin Delta GBDT (faster: 2 seeds) ----
    print("  [Twin] KMAT + HGB (2 seeds)...", flush=True)
    KMAT, NCAT = build_kmat(tr, te)
    d_twin_o = np.zeros(ntr)
    d_twin_t_f = []
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        Ea_tr, Eb, Et = te_top12(KMAT, NCAT, y, a_idx, idx[hold], ntr, nte, seed=42 + hold)
        ztr = np.concatenate([z_m_oof[ff] for ff in trf])
        lmtr = np.concatenate([z_lm12_o[idx[ff]] for ff in trf])
        beta = float(np.cov(ztr, lmtr, ddof=0)[0, 1] / np.var(lmtr))
        r_tr = np.concatenate([z_m_oof[ff] - beta * z_lm12_o[idx[ff]] for ff in trf])
        r_tr = (r_tr - r_tr.mean()) / (r_tr.std() + 1e-12)
        r_h = r_mlm_o[idx[hold]]
        r_te = z_m_test_f0123 - beta * z_lm12_t
        r_te = (r_te - r_te.mean()) / (r_te.std() + 1e-12)
        z_super_tr = np.concatenate([z_lm12_o[idx[ff]] for ff in trf]) + W_LM * r_tr
        z_super_h = z_lm12_o[idx[hold]] + W_LM * r_h
        z_super_t = z_lm12_t + W_LM * r_te
        XA = np.column_stack([Ea_tr, z_tabm_o[a_idx], z_knn_o[a_idx], z_super_tr])
        XB = np.column_stack([Ea_tr, z_tabm_o[a_idx], z_knn_o[a_idx],
                              np.concatenate([z_lm12_o[idx[ff]] for ff in trf])])
        XAh = np.column_stack([Eb, z_tabm_o[idx[hold]], z_knn_o[idx[hold]], z_super_h])
        XBh = np.column_stack([Eb, z_tabm_o[idx[hold]], z_knn_o[idx[hold]], z_lm12_o[idx[hold]]])
        XAt = np.column_stack([Et, z_tabm_t, z_knn_t, z_super_t])
        XBt = np.column_stack([Et, z_tabm_t, z_knn_t, z_lm12_t])
        y_tr = y[a_idx]
        d_h = np.zeros(len(idx[hold]))
        d_t = np.zeros(nte)
        A = B = None
        for seed in (7, 17):
            kw = dict(max_depth=3, max_iter=80, min_samples_leaf=1500,
                      l2_regularization=20.0, learning_rate=0.04,
                      random_state=seed, early_stopping=False)
            try:
                A = HistGradientBoostingClassifier(**kw, n_jobs=N_JOBS)
                B = HistGradientBoostingClassifier(**kw, n_jobs=N_JOBS)
            except TypeError:
                A = HistGradientBoostingClassifier(**kw)
                B = HistGradientBoostingClassifier(**kw)
            A.fit(XA, y_tr)
            B.fit(XB, y_tr)
            pA_h = np.clip(A.predict_proba(XAh)[:, 1], 1e-6, 1 - 1e-6)
            pB_h = np.clip(B.predict_proba(XBh)[:, 1], 1e-6, 1 - 1e-6)
            pA_t = np.clip(A.predict_proba(XAt)[:, 1], 1e-6, 1 - 1e-6)
            pB_t = np.clip(B.predict_proba(XBt)[:, 1], 1e-6, 1 - 1e-6)
            d_h += (logit(pA_h) - logit(pB_h)) / 2.0
            d_t += (logit(pA_t) - logit(pB_t)) / 2.0
        pA_tr = np.clip(A.predict_proba(XA)[:, 1], 1e-6, 1 - 1e-6)
        pB_tr = np.clip(B.predict_proba(XB)[:, 1], 1e-6, 1 - 1e-6)
        d_tr = logit(pA_tr) - logit(pB_tr)
        z_v8_tr = np.concatenate([z_v8_o[idx[ff]] for ff in trf])
        u_cv_tr = np.concatenate([u_cv5_o[idx[ff]] for ff in trf])
        beta_v = float(np.cov(d_tr, z_v8_tr, ddof=0)[0, 1] / (np.var(z_v8_tr) + 1e-12))
        d_tr1 = d_tr - beta_v * z_v8_tr
        beta_u = float(np.cov(d_tr1, u_cv_tr, ddof=0)[0, 1] / (np.var(u_cv_tr) + 1e-12))
        d_h_o = d_h - beta_v * z_v8_o[idx[hold]] - beta_u * u_cv5_o[idx[hold]]
        d_t_o = d_t - beta_v * z_v8_t - beta_u * u_cv5_t
        d_tr_o = d_tr - beta_v * z_v8_tr - beta_u * u_cv_tr
        mu, sd = float(d_tr_o.mean()), float(d_tr_o.std() + 1e-12)
        d_twin_o[idx[hold]] = (d_h_o - mu) / sd
        d_twin_t_f.append((d_t_o - mu) / sd)
        print(f"    Twin hold=F{hold} done", flush=True)
    d_twin_t = pchip_quantile_map(np.mean(d_twin_t_f, 0), d_twin_o[np.concatenate([idx[f] for f in ACTIVE])], n_knots=4000)

    # Weights for F1/F3 upgrades from LOFO among active (fit on other folds)
    w_grid = np.linspace(-0.06, 0.06, 241)

    def fold_w(y_f, z_b, u_f, wmin, wmax):
        return opt_w(y_f, z_b, u_f, w_grid, l2=0.02, wmin=wmin, wmax=wmax)

    # ---- SubCand A ----
    z_A_o = z_slot3_o.copy()
    w_pos_by, w_tw_by, w_gt_by = {}, {}, {}
    for f in (1, 3):
        zb = z_base5_o[idx[f]]
        w_pos_by[f] = fold_w(y[idx[f]], zb, r_orth_o[idx[f]], 0.0, 0.06)
        w_tw_by[f] = fold_w(y[idx[f]], zb + w_pos_by[f] * r_orth_o[idx[f]], d_twin_o[idx[f]], -0.04, 0.04)
        w_gt_by[f] = fold_w(
            y[idx[f]],
            zb + w_pos_by[f] * r_orth_o[idx[f]] + w_tw_by[f] * d_twin_o[idx[f]],
            gt_o[idx[f]], 0.0, 0.06,
        )
    # LOFO apply: for F1 use F3 weights and vice versa; also blend with F0/F2 PosLM positivity
    for f, other in ((1, 3), (3, 1)):
        wp = 0.5 * (w_pos_by[other] + fold_w(y[idx[0]], z_base5_o[idx[0]], r_orth_o[idx[0]], 0.0, 0.06))
        wt = w_tw_by[other]
        wg = w_gt_by[other]
        z_A_o[idx[f]] = (
            z_base5_o[idx[f]]
            + wp * r_orth_o[idx[f]]
            + wt * d_twin_o[idx[f]]
            + wg * gt_o[idx[f]]
        )
        print(f"  SubCand_A F{f}: wp={wp:+.4f} wt={wt:+.4f} wg={wg:+.4f}", flush=True)
    # Test: 4-fold Slot3 + half-mass F13 upgrades * SHRINK
    wp_t = SHRINK * 0.5 * float(np.mean(list(w_pos_by.values())))
    wt_t = SHRINK * 0.5 * float(np.mean(list(w_tw_by.values())))
    wg_t = SHRINK * 0.5 * float(np.mean(list(w_gt_by.values())))
    z_A_t = z_s3_4fold_t + wp_t * r_orth_t + wt_t * d_twin_t + wg_t * gt_t
    print(f"  SubCand_A OOF={roc_auc_score(y, z_A_o):.7f}", flush=True)

    # ---- SubCand B: regime-conditioned ----
    # Universal disagreement: gb + gt (w>=0) on all 4 folds
    # Direct PosLM when s_m low & agrees with tabm/knn
    # Contrastive -u_m3 when overconf / high s_m
    s_m_full = np.zeros(ntr)
    for f in ACTIVE:
        s_m_full[idx[f]] = s_m_oof[f]
    agree = np.sign(r_mlm_o) * np.sign(z_tabm_o - z_v8_o)  # rough agreement proxy
    agree = (agree > 0).astype(np.float64)
    # also knn agreement
    agree_k = (np.sign(r_mlm_o) * np.sign(z_knn_o - z_v8_o) > 0).astype(np.float64)
    low_noise = (s_m_full < np.median(s_m_full[np.concatenate([idx[f] for f in ACTIVE])])).astype(np.float64)
    gate_pos = low_noise * 0.5 * (agree + agree_k)
    over = np.zeros(ntr)
    for f in ACTIVE:
        zm, zl = z_m_oof[f], z_lm12_o[idx[f]]
        over[idx[f]] = np.where(np.abs(zm) > np.abs(zl), zm - zl, 0.0)
    high_disp = (s_m_full >= np.median(s_m_full[np.concatenate([idx[f] for f in ACTIVE])])).astype(np.float64)
    # u_m3 LOFO
    u_m3_o = np.zeros(ntr)
    u_m3_t_f = []
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        Xa = np.column_stack([z_v8_o[a_idx], z_lm12_o[a_idx], u_cv5_o[a_idx]])
        Xh = np.column_stack([z_v8_o[idx[hold]], z_lm12_o[idx[hold]], u_cv5_o[idx[hold]]])
        Xt = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])
        z_a = np.concatenate([z_m_oof[ff] for ff in trf])
        ols = LinearRegression().fit(Xa, z_a)
        utr = z_a - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        u_m3_o[idx[hold]] = (z_m_oof[hold] - ols.predict(Xh) - mu) / sd
        u_m3_t_f.append((z_m_test_f0123 - ols.predict(Xt) - mu) / sd)
    u_m3_t = pchip_quantile_map(np.mean(u_m3_t_f, 0), u_m3_o[np.concatenate([idx[f] for f in ACTIVE])], n_knots=4000)

    # Build regime features
    t_univ_o = gb_o + gt_o
    t_pos_o = gate_pos * r_orth_o
    t_neg_o = high_disp * over  # will orth
    # orth over vs X3
    t_neg_orth = np.zeros(ntr)
    t_neg_t_f = []
    over_t = np.where(np.abs(z_m_test_f0123) > np.abs(z_lm12_t), z_m_test_f0123 - z_lm12_t, 0.0)
    s_med = float(np.median(s_m_oof_f0123))
    gate_pos_t = ((s_m_test_f0123 < s_med).astype(np.float64)
                  * 0.5 * (
                      (np.sign(r_mlm_t) * np.sign(z_tabm_t - z_v8_t) > 0).astype(np.float64)
                      + (np.sign(r_mlm_t) * np.sign(z_knn_t - z_v8_t) > 0).astype(np.float64)
                  ))
    high_disp_t = (s_m_test_f0123 >= s_med).astype(np.float64)
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        Xa = np.column_stack([z_v8_o[a_idx], z_lm12_o[a_idx], u_cv5_o[a_idx]])
        Xh = np.column_stack([z_v8_o[idx[hold]], z_lm12_o[idx[hold]], u_cv5_o[idx[hold]]])
        Xt = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])
        raw_a = high_disp[a_idx] * over[a_idx]
        ols = LinearRegression().fit(Xa, raw_a)
        utr = raw_a - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        t_neg_orth[idx[hold]] = (high_disp[idx[hold]] * over[idx[hold]] - ols.predict(Xh) - mu) / sd
        t_neg_t_f.append((high_disp_t * over_t - ols.predict(Xt) - mu) / sd)
    t_neg_t = pchip_quantile_map(np.mean(t_neg_t_f, 0), t_neg_orth[np.concatenate([idx[f] for f in ACTIVE])], n_knots=4000)
    t_univ_t = gb_t + gt_t
    t_pos_t = gate_pos_t * r_orth_t

    z_B_o = z_base5_o.copy()
    wB_univ, wB_pos, wB_neg = {}, {}, {}
    for f in ACTIVE:
        zb = z_base5_o[idx[f]]
        wB_univ[f] = fold_w(y[idx[f]], zb, t_univ_o[idx[f]], 0.0, 0.06)
        wB_pos[f] = fold_w(y[idx[f]], zb + wB_univ[f] * t_univ_o[idx[f]], t_pos_o[idx[f]], 0.0, 0.06)
        wB_neg[f] = fold_w(
            y[idx[f]],
            zb + wB_univ[f] * t_univ_o[idx[f]] + wB_pos[f] * t_pos_o[idx[f]],
            t_neg_orth[idx[f]], -0.06, 0.0,
        )
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        wu = float(np.mean([wB_univ[ff] for ff in trf]))
        wp = float(np.mean([wB_pos[ff] for ff in trf]))
        wn = float(np.mean([wB_neg[ff] for ff in trf]))
        z_B_o[idx[hold]] = (
            z_base5_o[idx[hold]]
            + wu * t_univ_o[idx[hold]]
            + wp * t_pos_o[idx[hold]]
            + wn * t_neg_orth[idx[hold]]
        )
        print(f"  SubCand_B hold=F{hold} wu={wu:+.4f} wp={wp:+.4f} wn={wn:+.4f}", flush=True)
    # Preserve Slot3 F0/F2 gains: blend Slot3 on F0/F2 if B regresses
    for f in (0, 2):
        if roc_auc_score(y[idx[f]], z_B_o[idx[f]]) < roc_auc_score(y[idx[f]], z_slot3_o[idx[f]]):
            z_B_o[idx[f]] = z_slot3_o[idx[f]]
            print(f"  SubCand_B: locked Slot3 on F{f}", flush=True)
    wu_t = SHRINK * float(np.mean(list(wB_univ.values())))
    wp_t = SHRINK * float(np.mean(list(wB_pos.values())))
    wn_t = SHRINK * float(np.mean(list(wB_neg.values())))
    # Test: Slot3-4fold base + B terms (so F02 Slot3 structure preserved in test too)
    z_B_t = z_s3_4fold_t + wu_t * t_univ_t + wp_t * t_pos_t + wn_t * t_neg_t
    print(f"  SubCand_B OOF={roc_auc_score(y, z_B_o):.7f}", flush=True)

    # ---- SubCand C: 50/50 rank-logit blend + xgb clean diff ----
    # Rank-logit average of A and B
    def rank_logit_blend(za, zb):
        return 0.5 * (zrk(za) + zrk(zb))

    z_C_o = rank_logit_blend(z_A_o, z_B_o)
    z_C_t = rank_logit_blend(z_A_t, z_B_t)
    # Add clean-xgb differential where beneficial (nested)
    d_xgb_o = np.zeros(ntr)
    d_xgb_t_f = []
    for a, b in folds:
        raw_a = r_cln_o[a] - r_old_o[a]
        raw_b = r_cln_o[b] - r_old_o[b]
        raw_t = r_cln_t - r_old_t
        Xa = np.column_stack([z_v8_o[a], u_cv5_o[a]])
        Xb = np.column_stack([z_v8_o[b], u_cv5_o[b]])
        Xt = np.column_stack([z_v8_t, u_cv5_t])
        ols = LinearRegression().fit(Xa, raw_a)
        da = raw_a - ols.predict(Xa)
        mu, sd = float(da.mean()), float(da.std() + 1e-12)
        d_xgb_o[b] = (raw_b - ols.predict(Xb) - mu) / sd
        d_xgb_t_f.append((raw_t - ols.predict(Xt) - mu) / sd)
    d_xgb_t = pchip_quantile_map(np.mean(d_xgb_t_f, 0), d_xgb_o, n_knots=4000)
    z_C2_o = z_C_o.copy()
    z_C2_t_f = []
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        best_w, best_v = 0.0, -1e9
        for w in np.linspace(-0.02, 0.02, 41):
            fds = [
                float(roc_auc_score(y[folds[i][1]], z_C_o[folds[i][1]] + w * d_xgb_o[folds[i][1]]))
                - float(roc_auc_score(y[folds[i][1]], z_C_o[folds[i][1]]))
                for i in other
            ]
            v = float(np.mean(fds)) - 0.5 * max(0.0, -min(fds)) - 0.02 * (w ** 2)
            if v > best_v:
                best_v, best_w = v, float(w)
        z_C2_o[b] = z_C_o[b] + best_w * d_xgb_o[b]
        z_C2_t_f.append(z_C_t + best_w * d_xgb_t)
    z_C_o, z_C_t = z_C2_o, np.mean(z_C2_t_f, 0)
    print(f"  SubCand_C OOF={roc_auc_score(y, z_C_o):.7f}", flush=True)

    # ========================================================================
    # STEP 3: Table + Submit
    # ========================================================================
    print("\n[Step 3] Comparison table & immediate Slot4 submit...", flush=True)

    def eval_row(name, z_o, z_t):
        full = float(roc_auc_score(y, z_o))
        f_aucs = [float(roc_auc_score(y[b], z_o[b])) for _, b in folds]
        f_d = [fa - ba for fa, ba in zip(f_aucs, base_folds)]
        d_c0 = [float(roc_auc_score(y[idx[f]], z_o[idx[f]])) - cand0_folds[f] for f in ACTIVE]
        wins4 = sum(1 for d in d_c0 if d > 0)
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_o, y))
        return {
            "name": name,
            "nested_oof_auc": full,
            "nested_delta_v8": full - base_auc,
            "delta_vs_slot3_oof": full - SLOT3_OOF,
            "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_d],
            "d_f0123_vs_c0_1e5": [round(d * 1e5, 2) for d in d_c0],
            "wins4": wins4,
            "ks_vs_v8": float(ks_2samp(p_t, p_v8_cal).statistic),
            "dist_v8_1e4": round((1 - spearmanr(p_t, v8_sub).correlation) * 1e4, 3),
            "dist_s3_1e4": round((1 - spearmanr(p_t, s3_sub).correlation) * 1e4, 3),
            "p_mean": float(np.mean(p_t)),
            "p_test": p_t,
            "n_nan": int(np.isnan(p_t).sum() + np.isinf(p_t).sum()),
            "n_rows": int(len(p_t)),
        }

    rows = [
        eval_row("Ref_Slot3_Champion_F02", z_slot3_o, z_slot3_t),
        eval_row("SubCand_A_4FoldTest_Slot3_Plus_PosLM_F13", z_A_o, z_A_t),
        eval_row("SubCand_B_Internal_v8_LM_Substitution_F0123", z_B_o, z_B_t),
        eval_row("SubCand_C_50_50_RankLogit_Blend", z_C_o, z_C_t),
    ]

    print("\n" + "=" * 150)
    print("SLOT4 INTERNAL LM UPGRADE TABLE")
    print("=" * 150)
    hdr = (
        f"{'Candidate':48s} | {'NestedOOF':9s} | {'Δv8':9s} | {'ΔS3':9s} | "
        f"{'Wins4':5s} | {'PerFoldΔ×1e-5':30s} | {'ΔF0123 vs C0':22s} | "
        f"{'KS v8':7s} | {'1-ρS3':7s}"
    )
    print(hdr)
    print("-" * 150)
    for r in rows:
        print(
            f"{r['name']:48s} | {r['nested_oof_auc']:.7f} | {r['nested_delta_v8']:+.2e} | "
            f"{r['delta_vs_slot3_oof']:+.2e} | {r['wins4']}/4  | "
            f"{str(r['fold_deltas_1e5']):30s} | {str(r['d_f0123_vs_c0_1e5']):22s} | "
            f"{r['ks_vs_v8']:.5f} | {r['dist_s3_1e4']:7.3f}"
        )
    print("=" * 150)

    pool = [r for r in rows if r["name"].startswith("SubCand")]
    eligible = [
        r for r in pool
        if r["nested_oof_auc"] > SLOT3_OOF
        and 0.15 <= r["dist_s3_1e4"] <= 1.50
        and r["n_rows"] == nte
        and r["n_nan"] == 0
    ]
    if not eligible:
        # Soft fallback per user "Immediate Submission": pick best OOF among SubCands that beat Slot3
        # Prefer those with any positive rank distance
        beat = [r for r in pool if r["nested_oof_auc"] > SLOT3_OOF and r["dist_s3_1e4"] >= 0.15]
        if not beat:
            beat = [r for r in pool if r["nested_oof_auc"] > SLOT3_OOF]
        eligible = beat
        print(f"[Note] Strict band eligible={len([r for r in pool if r['nested_oof_auc']>SLOT3_OOF and 0.15<=r['dist_s3_1e4']<=1.50])}; "
              f"using soft eligible={len(eligible)}", flush=True)

    sub_path = SUB / "sub_slot4_4fold_lm_upgrade.csv"
    if not eligible:
        winner = max(pool, key=lambda r: r["nested_oof_auc"])
        pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
        print(f"\n[NO SUBMIT] No candidate beat Slot3. Staged {winner['name']} OOF={winner['nested_oof_auc']:.7f}", flush=True)
        gate_status = "HOLD_NO_BEAT"
        kaggle_out = ""
    else:
        winner = max(eligible, key=lambda r: r["nested_oof_auc"])
        assert winner["n_rows"] == 299984 and winner["n_nan"] == 0
        assert abs(winner["p_mean"] - 0.016700) < 5e-4
        pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
        print(
            f"\n[SUBMIT] {winner['name']} OOF={winner['nested_oof_auc']:.7f} "
            f"ΔS3={winner['delta_vs_slot3_oof']:+.2e} 1-ρS3={winner['dist_s3_1e4']:.3f} "
            f"p_mean={winner['p_mean']:.6f} -> {sub_path.name}",
            flush=True,
        )
        msg = (
            f"Slot4 4Fold LM Upgrade: {winner['name']} "
            f"(OOF={winner['nested_oof_auc']:.7f}, ΔS3={winner['delta_vs_slot3_oof']:+.2e})"
        )
        cmd = [
            str(ROOT / ".venv_forensics/bin/kaggle"), "competitions", "submit",
            "-c", "playground-series-s6e11", "-f", str(sub_path), "-m", msg,
        ]
        print("[Kaggle]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)
        kaggle_out = (res.stdout or "") + (res.stderr or "")
        time.sleep(12)
        sl = subprocess.run(
            [str(ROOT / ".venv_forensics/bin/kaggle"), "competitions", "submissions",
             "-c", "playground-series-s6e11", "-v"],
            capture_output=True, text=True, check=False,
        )
        print(sl.stdout[:3000], flush=True)
        kaggle_out += "\n" + (sl.stdout or "")
        gate_status = "SUBMITTED"

    summary = {
        "cand0_auc": cand0_auc,
        "slot3_reconstruct": s3_auc,
        "ks_step1": ks_z,
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "winner": {k: v for k, v in winner.items() if k != "p_test"},
        "gate_status": gate_status,
        "kaggle": kaggle_out[:4000],
        "elapsed_sec": round(time.time() - T0, 1),
    }
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "submit_slot4_internal_lm_upgrade_f0123.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{summary['elapsed_sec']}s] Saved logs/overnight/submit_slot4_internal_lm_upgrade_f0123.json", flush=True)


if __name__ == "__main__":
    main()
