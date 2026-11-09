#!/usr/bin/env python3
"""Master-Verified 4-Fold (F0+F1+F2+F3) Diagnostic & Slot 4 Pipeline.

Local Mac CPU only (n_jobs=4). Does NOT touch Pod PID 3021 (Fold 4 training).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
GATE_OOF_MIN = 0.9526310

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
    """Exact Slot3 K-order reduction."""
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    knames = list(K.keys())
    codes = [pd.factorize(pd.Series(K[n]))[0].astype(np.int32) for n in knames]
    KMAT = np.column_stack(codes)
    NCAT = np.array([c.max() + 1 for c in codes], dtype=np.int32)
    return KMAT, NCAT


def te_top12(KMAT, NCAT, y, a, b, ntr, nte, smooth=20.0, seed=42):
    """Target-encode all keys on train `a`, select top-12 by |corr|, return Ea, Eb, Et."""
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


def main():
    print("[Boot] verify_slot4_gm_f0123.py (4-fold F0+F1+F2+F3)...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    idx = {f: folds[f][1] for f in range(5)}
    a_tr = {f: folds[f][0] for f in range(5)}

    # ---- refs ----
    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s3_sub = pd.read_csv(SUB / "sub_slot3_gm_verified_f02.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_v8_o = zrk(v8_oof)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_auc = float(roc_auc_score(y, z_v8_o))
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y))

    z_lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), z_lm12_o, n_knots=4000)
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
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)

    # Cand0 Smooth-Gamma (Slot3-exact)
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
    # STEP 1: Artifacts + Slot3 Ref + 4-fold align/PCHIP
    # ========================================================================
    print("\n[Step 1] Artifacts + Slot3 Ref + 4-fold PCHIP...", flush=True)
    raw_o, raw_t, s_raw_o, s_raw_t = {}, {}, {}, {}
    z_m_oof, z_m_tst, s_m_oof, s_m_tst, z_m_zrk = {}, {}, {}, {}, {}
    for f in ACTIVE:
        oa = np.load(PRED / f"gpt2_med_4ep/oof_{f}.npy")
        ta = np.load(PRED / f"gpt2_med_4ep/test_{f}.npy")
        assert oa.shape == (len(idx[f]), 8, 2) and ta.shape == (nte, 4, 2)
        assert np.isfinite(oa).all() and np.isfinite(ta).all()
        sha_o = sha256_file(PRED / f"gpt2_med_4ep/oof_{f}.npy")[:16]
        sha_t = sha256_file(PRED / f"gpt2_med_4ep/test_{f}.npy")[:16]
        prior = float(y[a_tr[f]].mean())
        raw_o[f], raw_t[f] = bayes_fold(oa, prior), bayes_fold(ta, prior)
        s_raw_o[f], s_raw_t[f] = bayes_std(oa), bayes_std(ta)
        z_m_zrk[f] = zrk(raw_o[f] - np.mean(raw_o[f]))
        z_m_oof[f] = align_disp(raw_o[f], z_lm12_o[idx[f]])
        z_m_tst[f] = align_disp(raw_t[f], z_lm12_t)
        s_m_oof[f] = (s_raw_o[f] - s_raw_o[f].mean()) / (s_raw_o[f].std() + 1e-12)
        s_m_tst[f] = (s_raw_t[f] - s_raw_t[f].mean()) / (s_raw_t[f].std() + 1e-12)
        print(f"  F{f}: oof/test OK sha={sha_o}/{sha_t}", flush=True)

    z_m_oof_f0123 = np.concatenate([z_m_oof[f] for f in ACTIVE])
    s_m_oof_f0123 = np.concatenate([s_m_oof[f] for f in ACTIVE])
    z_m_test_f0123 = pchip_quantile_map(np.mean([z_m_tst[f] for f in ACTIVE], 0), z_m_oof_f0123, n_knots=4000)
    s_m_test_f0123 = pchip_quantile_map(np.mean([s_m_tst[f] for f in ACTIVE], 0), s_m_oof_f0123, n_knots=4000)
    ks_z = float(ks_2samp(z_m_oof_f0123, z_m_test_f0123).statistic)
    print(f"[Step 1] 4-fold PCHIP KS={ks_z:.6f} std_oof={z_m_oof_f0123.std():.4f} std_t={z_m_test_f0123.std():.4f}", flush=True)

    # Slot3 Ref exact (F0+F2, zrk)
    w_grid_s3 = np.linspace(-0.045, 0.045, 181)

    def opt_w(y_f, z_b, u_f, grid=None, l2=0.015, wmin=None, wmax=None):
        grid = w_grid_s3 if grid is None else grid
        if wmin is not None or wmax is not None:
            grid = grid[(grid >= (wmin if wmin is not None else -1)) & (grid <= (wmax if wmax is not None else 1))]
        base = float(roc_auc_score(y_f, z_b))
        return float(max(grid, key=lambda w: roc_auc_score(y_f, z_b + w * u_f) - base - l2 * (w ** 2)))

    def pairwise_dst(z_src, X_src, z_dst, X_dst):
        ols = LinearRegression().fit(X_src, z_src)
        u = z_dst - ols.predict(X_dst)
        mu, sd = float(u.mean()), float(u.std() + 1e-12)
        return (u - mu) / sd, (ols, mu, sd)

    X3_0 = np.column_stack([z_v8_o[idx[0]], z_lm12_o[idx[0]], u_cv5_o[idx[0]]])
    X3_2 = np.column_stack([z_v8_o[idx[2]], z_lm12_o[idx[2]], u_cv5_o[idx[2]]])
    X3_t = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])
    u_h1_0, h1_s2 = pairwise_dst(z_m_zrk[2], X3_2, z_m_zrk[0], X3_0)
    u_h1_2, h1_s0 = pairwise_dst(z_m_zrk[0], X3_0, z_m_zrk[2], X3_2)
    w_h1_0 = opt_w(y[idx[0]], z_base5_o[idx[0]], u_h1_0)
    w_h1_2 = opt_w(y[idx[2]], z_base5_o[idx[2]], u_h1_2)
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
    w_h2_0 = opt_w(y[idx[0]], z_s3[idx[0]], ug0)
    w_h2_2 = opt_w(y[idx[2]], z_s3[idx[2]], ug2)
    z_slot3_o = z_s3.copy()
    z_slot3_o[idx[0]] = z_s3[idx[0]] + w_h2_2 * ug0
    z_slot3_o[idx[2]] = z_s3[idx[2]] + w_h2_0 * ug2
    s3_auc = float(roc_auc_score(y, z_slot3_o))
    s3_fd = [round((float(roc_auc_score(y[b], z_slot3_o[b])) - base_folds[i]) * 1e5, 2) for i, (_, b) in enumerate(folds)]
    print(f"  Ref_Slot3_F02 OOF={s3_auc:.7f} folds={s3_fd} (target 0.9526269 / [+2.90,+5.34,+1.32,+2.42,-1.84])", flush=True)
    assert abs(s3_auc - SLOT3_OOF) < 5e-8, f"Slot3 reconstruct mismatch: {s3_auc}"

    z_m_t02 = pchip_quantile_map(
        0.5 * (zrk(raw_t[0] - np.mean(raw_o[0])) + zrk(raw_t[2] - np.mean(raw_o[2]))),
        np.concatenate([z_m_zrk[0], z_m_zrk[2]]), n_knots=4000,
    )
    ols_h1_2, mu1_2, sd1_2 = h1_s2
    ols_h1_0, mu1_0, sd1_0 = h1_s0
    u_h1_t = pchip_quantile_map(
        0.5 * (
            (z_m_t02 - ols_h1_2.predict(X3_t) - mu1_2) / sd1_2
            + (z_m_t02 - ols_h1_0.predict(X3_t) - mu1_0) / sd1_0
        ),
        np.concatenate([u_h1_0, u_h1_2]), n_knots=4000,
    )
    z_slot3_t = z_base5_t + 0.5 * (w_h1_0 + w_h1_2) * u_h1_t
    feat_s3_t = g_t * (z_m_t02 - 0.5 * (beta0 + beta2) * z_lm12_t)
    ols_h2_2, mu2_2, sd2_2 = h2_s2
    ols_h2_0, mu2_0, sd2_0 = h2_s0
    ug_t = pchip_quantile_map(
        0.5 * (
            (feat_s3_t - ols_h2_2.predict(X2_t) - mu2_2) / sd2_2
            + (feat_s3_t - ols_h2_0.predict(X2_t) - mu2_0) / sd2_0
        ),
        np.concatenate([ug0, ug2]), n_knots=4000,
    )
    z_slot3_t = z_slot3_t + 0.5 * (w_h2_0 + w_h2_2) * ug_t

    # ========================================================================
    # STEP 2: Single-fold physics table
    # ========================================================================
    print("\n[Step 2] 4-Fold Single-Fold Physics Table...", flush=True)
    w_grid = np.linspace(-0.08, 0.08, 321)
    r_mlm_by, over_by, bound_by, u_m3_by = {}, {}, {}, {}
    beta_by = {}
    for f in ACTIVE:
        zm, zl = z_m_oof[f], z_lm12_o[idx[f]]
        solo_m = float(roc_auc_score(y[idx[f]], zm))
        solo_l = float(roc_auc_score(y[idx[f]], zl))
        beta = float(np.cov(zm, zl, ddof=0)[0, 1] / np.var(zl))
        r = zm - beta * zl
        r_n = (r - r.mean()) / (r.std() + 1e-12)
        w_lm = opt_w(y[idx[f]], zl, r_n, grid=w_grid)
        d_lm = float(roc_auc_score(y[idx[f]], zl + w_lm * r_n) - solo_l)
        # overconf
        over = np.where(np.abs(zm) > np.abs(zl), zm - zl, 0.0)
        X3 = np.column_stack([z_v8_o[idx[f]], zl, u_cv5_o[idx[f]]])
        ols3 = LinearRegression().fit(X3, zm)
        u3 = zm - ols3.predict(X3)
        u3 = (u3 - u3.mean()) / (u3.std() + 1e-12)
        ols_ov = LinearRegression().fit(X3, over)
        u_ov = over - ols_ov.predict(X3)
        u_ov = (u_ov - u_ov.mean()) / (u_ov.std() + 1e-12)
        b_raw = np.tanh(np.abs(zl - z_v8_o[idx[f]])) * h_o[idx[f]] * r
        ols_b = LinearRegression().fit(X3, b_raw)
        u_b = b_raw - ols_b.predict(X3)
        u_b = (u_b - u_b.mean()) / (u_b.std() + 1e-12)
        zb = z_base5_o[idx[f]]
        base = float(roc_auc_score(y[idx[f]], zb))
        w3 = opt_w(y[idx[f]], zb, u3, grid=w_grid)
        wov = opt_w(y[idx[f]], zb, u_ov, grid=w_grid)
        wb = opt_w(y[idx[f]], zb, u_b, grid=w_grid)
        print(
            f"  F{f}: solo_m={solo_m:.6f} lm12={solo_l:.6f} Δ_solo={(solo_m-solo_l)*1e5:+.2f}e-5 | "
            f"w_lm={w_lm:+.4f} Δ_lm={d_lm*1e5:+.2f}e-5 | "
            f"u_m3 w={w3:+.4f} Δ={(roc_auc_score(y[idx[f]], zb+w3*u3)-base)*1e5:+.2f}e-5 | "
            f"u_over w={wov:+.4f} Δ={(roc_auc_score(y[idx[f]], zb+wov*u_ov)-base)*1e5:+.2f}e-5 | "
            f"b_bound w={wb:+.4f} Δ={(roc_auc_score(y[idx[f]], zb+wb*u_b)-base)*1e5:+.2f}e-5",
            flush=True,
        )
        r_mlm_by[f], over_by[f], bound_by[f], u_m3_by[f] = r_n, u_ov, u_b, u3
        beta_by[f] = beta

    # ========================================================================
    # STEP 3: LOFO candidates
    # ========================================================================
    print("\n[Step 3] Building 4-fold LOFO candidates...", flush=True)

    def lofo_ols(z_by, X_by):
        u_out, store = {}, {}
        for hold in ACTIVE:
            trf = [ff for ff in ACTIVE if ff != hold]
            Xtr = np.vstack([X_by[ff] for ff in trf])
            ztr = np.concatenate([z_by[ff] for ff in trf])
            ols = LinearRegression().fit(Xtr, ztr)
            utr = ztr - ols.predict(Xtr)
            mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
            uh = z_by[hold] - ols.predict(X_by[hold])
            u_out[hold] = (uh - mu) / sd
            store[hold] = (ols, mu, sd)
        return u_out, store

    def lofo_feat_orth(feat_raw_by, X_by):
        """feat_raw_by[f] is raw feature on fold f; OLS orth vs X on other folds."""
        return lofo_ols(feat_raw_by, X_by)

    X3_by = {f: np.column_stack([z_v8_o[idx[f]], z_lm12_o[idx[f]], u_cv5_o[idx[f]]]) for f in ACTIVE}
    # Rebuild LOFO versions of features
    over_raw, bound_raw, r_raw, zm_by = {}, {}, {}, {}
    for f in ACTIVE:
        zm_by[f] = z_m_oof[f]
        # LOFO beta for r_m_lm12
        trf = [ff for ff in ACTIVE if ff != f]
        ztr = np.concatenate([z_m_oof[ff] for ff in trf])
        lmtr = np.concatenate([z_lm12_o[idx[ff]] for ff in trf])
        beta = float(np.cov(ztr, lmtr, ddof=0)[0, 1] / np.var(lmtr))
        r = z_m_oof[f] - beta * z_lm12_o[idx[f]]
        rtr = ztr - beta * lmtr
        mu, sd = float(rtr.mean()), float(rtr.std() + 1e-12)
        r_raw[f] = (r - mu) / sd
        over_raw[f] = np.where(np.abs(z_m_oof[f]) > np.abs(z_lm12_o[idx[f]]), z_m_oof[f] - z_lm12_o[idx[f]], 0.0)
        bound_raw[f] = np.tanh(np.abs(z_lm12_o[idx[f]] - z_v8_o[idx[f]])) * h_o[idx[f]] * r_raw[f]

    u_m3_lofo, ols3_store = lofo_ols(zm_by, X3_by)
    u_over_lofo, over_store = lofo_feat_orth(over_raw, X3_by)
    u_bound_lofo, bound_store = lofo_feat_orth(bound_raw, X3_by)

    # tabm gate b_tabm
    s_tab_n = (s_tabm_o - s_tabm_o.mean()) / (s_tabm_o.std() + 1e-12)
    tabm_raw = {}
    for f in ACTIVE:
        tabm_raw[f] = (
            np.tanh(np.abs(z_lm12_o[idx[f]] - z_tabm_o[idx[f]]))
            * expit(s_tab_n[idx[f]])
            * r_raw[f]
        )
    u_tabm_lofo, tabm_store = lofo_feat_orth(tabm_raw, X3_by)

    # cert gate
    cert_raw = {f: expit(-s_m_oof[f]) * h_o[idx[f]] * r_raw[f] for f in ACTIVE}
    u_cert_lofo, cert_store = lofo_feat_orth(cert_raw, X3_by)

    def constrained_lofo_add(u_by, z_base, wmin, wmax, l2=0.02):
        fold_opt = {}
        grid = np.linspace(wmin, wmax, 161)
        for f in ACTIVE:
            fold_opt[f] = opt_w(y[idx[f]], z_base[idx[f]], u_by[f], grid=grid, l2=l2, wmin=wmin, wmax=wmax)
        z_out = z_base.copy()
        w_hold = {}
        for hold in ACTIVE:
            trf = [ff for ff in ACTIVE if ff != hold]
            w = float(np.mean([fold_opt[ff] for ff in trf]))
            w = float(np.clip(w, wmin, wmax))
            w_hold[hold] = w
            z_out[idx[hold]] = z_base[idx[hold]] + w * u_by[hold]
        return z_out, fold_opt, w_hold

    # ---- Cand1: Asymmetric Shrinkage H2 ----
    # u_over (w<=0) + shrunk u_m3 + b_bound (w>=0)
    z_c1, opt_over, _ = constrained_lofo_add(u_over_lofo, z_base5_o, -0.06, 0.0)
    # u_m3 with per-fold shrinkage: shrink toward 0 by (1 - sign_var across train)
    z_c1b, opt_m3, _ = constrained_lofo_add(u_m3_lofo, z_c1, -0.05, 0.05)
    # damp m3 if signs disagree on train
    z_c1_damped = z_c1.copy()
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        signs = [np.sign(opt_m3[ff]) for ff in trf]
        damp = 1.0 if len(set(s for s in signs if s != 0)) <= 1 else 0.35
        w = damp * float(np.mean([opt_m3[ff] for ff in trf]))
        z_c1_damped[idx[hold]] = z_c1[idx[hold]] + w * u_m3_lofo[hold]
    z_cand1_o, opt_bound, _ = constrained_lofo_add(u_bound_lofo, z_c1_damped, 0.0, 0.06)
    print(f"  Cand1 OOF={roc_auc_score(y, z_cand1_o):.7f}", flush=True)

    # ---- Cand2: MultiGate Jacobian (coordinate ascent on inner folds) ----
    terms = [
        ("over", u_over_lofo, -0.06, 0.0),
        ("bound", u_bound_lofo, 0.0, 0.06),
        ("tabm", u_tabm_lofo, 0.0, 0.06),
        ("cert", u_cert_lofo, 0.0, 0.06),
    ]

    def score_ws(ws, hold, z_base):
        """Score weights on train folds (exclude hold)."""
        trf = [ff for ff in ACTIVE if ff != hold]
        deltas = []
        for f in trf:
            z = z_base[idx[f]].copy()
            for (name, u_by, _, _), w in zip(terms, ws):
                z = z + w * u_by[f]
            deltas.append(float(roc_auc_score(y[idx[f]], z)) - cand0_folds[f])
        return float(np.mean(deltas) - 0.5 * sum(max(0.0, -d) for d in deltas))

    z_cand2_o = z_base5_o.copy()
    w2_hold = {}
    for hold in ACTIVE:
        # coordinate ascent
        ws = np.array([0.0, 0.01, 0.0, 0.0], dtype=np.float64)
        for _ in range(8):
            for ti, (_, _, wmin, wmax) in enumerate(terms):
                grid = np.linspace(wmin, wmax, 41)
                best_w, best_s = ws[ti], -1e9
                for w in grid:
                    trial = ws.copy()
                    trial[ti] = w
                    s = score_ws(trial, hold, z_base5_o)
                    if s > best_s:
                        best_s, best_w = s, float(w)
                ws[ti] = best_w
        w2_hold[hold] = ws.copy()
        for (name, u_by, _, _), w in zip(terms, ws):
            z_cand2_o[idx[hold]] = z_cand2_o[idx[hold]] + w * u_by[hold]
        print(f"  Cand2 hold=F{hold} ws={np.round(ws, 4)}", flush=True)
    print(f"  Cand2 OOF={roc_auc_score(y, z_cand2_o):.7f}", flush=True)

    # ---- Cand3: Paired Twin Delta GBDT ----
    print("  [Cand3] Building KMAT + Twin HGB (this takes a few minutes)...", flush=True)
    KMAT, NCAT = build_kmat(tr, te)
    d_twin_o = np.zeros(ntr, dtype=np.float64)
    d_twin_t_f = []
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        a_idx = np.concatenate([idx[ff] for ff in trf])
        # TE top12 on active train folds
        Ea_tr, Eb, Et = te_top12(KMAT, NCAT, y, a_idx, idx[hold], ntr, nte, seed=42 + hold)
        ztr = np.concatenate([z_m_oof[ff] for ff in trf])
        lmtr = np.concatenate([z_lm12_o[idx[ff]] for ff in trf])
        beta = float(np.cov(ztr, lmtr, ddof=0)[0, 1] / np.var(lmtr))
        r_tr_parts = [z_m_oof[ff] - beta * z_lm12_o[idx[ff]] for ff in trf]
        r_tr = np.concatenate(r_tr_parts)
        r_tr = (r_tr - r_tr.mean()) / (r_tr.std() + 1e-12)
        r_h = r_raw[hold]
        r_te = z_m_test_f0123 - beta * z_lm12_t
        r_te = (r_te - r_te.mean()) / (r_te.std() + 1e-12)

        z_super_tr = np.concatenate([z_lm12_o[idx[ff]] for ff in trf]) + 0.075 * r_tr
        z_super_h = z_lm12_o[idx[hold]] + 0.075 * r_h
        z_super_t = z_lm12_t + 0.075 * r_te

        XA = np.column_stack([Ea_tr, z_tabm_o[a_idx], z_knn_o[a_idx], z_super_tr])
        XB = np.column_stack([
            Ea_tr, z_tabm_o[a_idx], z_knn_o[a_idx],
            np.concatenate([z_lm12_o[idx[ff]] for ff in trf]),
        ])
        XAh = np.column_stack([Eb, z_tabm_o[idx[hold]], z_knn_o[idx[hold]], z_super_h])
        XBh = np.column_stack([Eb, z_tabm_o[idx[hold]], z_knn_o[idx[hold]], z_lm12_o[idx[hold]]])
        XAt = np.column_stack([Et, z_tabm_t, z_knn_t, z_super_t])
        XBt = np.column_stack([Et, z_tabm_t, z_knn_t, z_lm12_t])
        y_tr = y[a_idx]

        d_h = np.zeros(len(idx[hold]))
        d_t = np.zeros(nte)
        for seed in (7, 17, 27):
            kw = dict(
                max_depth=3, max_iter=80, min_samples_leaf=1500,
                l2_regularization=20.0, learning_rate=0.04,
                random_state=seed, early_stopping=False,
            )
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
            d_h += (logit(pA_h) - logit(pB_h)) / 3.0
            d_t += (logit(pA_t) - logit(pB_t)) / 3.0
        # Orth vs [v8, ucv5]: estimate betas on TRAIN via residual of twin on a held-in slice
        # Use Gram-Schmidt with train-fold correlations of d against v8/ucv5 using a proxy:
        # Fit twin predictions on train via A/B predict on XA/XB (in-sample) for beta estimation only.
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
    # Add twin to Cand1 and Cand2 with LOFO non-neg / free weight
    twin_by = {f: d_twin_o[idx[f]] for f in ACTIVE}
    z_c3_from_c1, opt_tw1, _ = constrained_lofo_add(twin_by, z_cand1_o, -0.04, 0.04)
    z_c3_from_c2, opt_tw2, _ = constrained_lofo_add(twin_by, z_cand2_o, -0.04, 0.04)
    if roc_auc_score(y, z_c3_from_c1) >= roc_auc_score(y, z_c3_from_c2):
        z_cand3_o, twin_base_name, opt_twin = z_c3_from_c1, "Cand1", opt_tw1
        z_cand3_base_o = z_cand1_o
    else:
        z_cand3_o, twin_base_name, opt_twin = z_c3_from_c2, "Cand2", opt_tw2
        z_cand3_base_o = z_cand2_o
    print(f"  Cand3 OOF={roc_auc_score(y, z_cand3_o):.7f} (on top of {twin_base_name})", flush=True)

    # ---- Cand4: Slot3 F02 lock + twin/bound on F1+F3 ----
    z_cand4_o = z_slot3_o.copy()
    for f in (1, 3):
        # weight from the other of {1,3} plus F0/F2 bound opts mean
        w_b = float(np.mean([opt_bound.get(ff, 0.01) for ff in (0, 2)]))
        w_b = float(np.clip(w_b, 0.0, 0.06))
        w_t = float(np.mean([opt_twin.get(ff, 0.0) for ff in ACTIVE if ff != f]))
        w_t = float(np.clip(w_t, -0.04, 0.04))
        z_cand4_o[idx[f]] = z_base5_o[idx[f]] + w_b * u_bound_lofo[f] + w_t * twin_by[f]
        # Ensure F0/F2 unchanged from Slot3
    # F0/F2 already Slot3
    print(f"  Cand4 OOF={roc_auc_score(y, z_cand4_o):.7f}", flush=True)

    # ---- Test predictions ----
    def predict_feat_test(raw_t, store, u_oof_by):
        us = []
        for hold in ACTIVE:
            ols, mu, sd = store[hold]
            us.append((raw_t - ols.predict(np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])) - mu) / sd)
        return pchip_quantile_map(np.mean(us, 0), np.concatenate([u_oof_by[f] for f in ACTIVE]), n_knots=4000)

    beta_mean = float(np.mean([beta_by[f] for f in ACTIVE]))
    r_mlm_t = z_m_test_f0123 - beta_mean * z_lm12_t
    r_mlm_t = (r_mlm_t - r_mlm_t.mean()) / (r_mlm_t.std() + 1e-12)
    over_t = np.where(np.abs(z_m_test_f0123) > np.abs(z_lm12_t), z_m_test_f0123 - z_lm12_t, 0.0)
    bound_t_raw = np.tanh(np.abs(z_lm12_t - z_v8_t)) * h_t * r_mlm_t
    s_tab_nt = (s_tabm_t - s_tabm_o.mean()) / (s_tabm_o.std() + 1e-12)
    tabm_t_raw = np.tanh(np.abs(z_lm12_t - z_tabm_t)) * expit(s_tab_nt) * r_mlm_t
    cert_t_raw = expit(-s_m_test_f0123) * h_t * r_mlm_t

    u_over_t = predict_feat_test(over_t, over_store, u_over_lofo)
    u_bound_t = predict_feat_test(bound_t_raw, bound_store, u_bound_lofo)
    u_tabm_t = predict_feat_test(tabm_t_raw, tabm_store, u_tabm_lofo)
    u_cert_t = predict_feat_test(cert_t_raw, cert_store, u_cert_lofo)
    # u_m3 test
    us = []
    for hold in ACTIVE:
        ols, mu, sd = ols3_store[hold]
        us.append((z_m_test_f0123 - ols.predict(np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])) - mu) / sd)
    u_m3_t = pchip_quantile_map(np.mean(us, 0), np.concatenate([u_m3_lofo[f] for f in ACTIVE]), n_knots=4000)

    w_over = SHRINK * float(np.mean(list(opt_over.values())))
    w_m3 = SHRINK * 0.35 * float(np.mean(list(opt_m3.values())))  # damped
    w_bound = SHRINK * float(np.mean(list(opt_bound.values())))
    z_cand1_t = z_base5_t + w_over * u_over_t + w_m3 * u_m3_t + w_bound * u_bound_t

    # Cand2 test: mean of hold weights
    ws2 = np.mean([w2_hold[h] for h in ACTIVE], axis=0) * SHRINK
    z_cand2_t = z_base5_t + ws2[0] * u_over_t + ws2[1] * u_bound_t + ws2[2] * u_tabm_t + ws2[3] * u_cert_t

    w_twin = SHRINK * float(np.mean(list(opt_twin.values())))
    if twin_base_name == "Cand1":
        z_cand3_t = z_cand1_t + w_twin * d_twin_t
    else:
        z_cand3_t = z_cand2_t + w_twin * d_twin_t

    # Cand4 test: Slot3 + shrink * (bound+twin) as F1/F3 are 2/4 of active
    z_cand4_t = z_slot3_t + 0.5 * SHRINK * (w_bound * u_bound_t + w_twin * d_twin_t)

    # ========================================================================
    # STEP 4: Table + Gate
    # ========================================================================
    print("\n[Step 4] Comparison table & Master Gate...", flush=True)

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
            "fold_wins_5": sum(1 for d in f_d if d > 0),
            "d_f0123_vs_c0_1e5": [round(d * 1e5, 2) for d in d_c0],
            "wins4": wins4,
            "dF0": d_c0[0], "dF2": d_c0[2],
            "ks_vs_v8": float(ks_2samp(p_t, p_v8_cal).statistic),
            "dist_v8_1e4": round((1 - spearmanr(p_t, v8_sub).correlation) * 1e4, 3),
            "dist_s3_1e4": round((1 - spearmanr(p_t, s3_sub).correlation) * 1e4, 3),
            "p_mean": float(np.mean(p_t)),
            "p_test": p_t,
        }

    rows = [
        eval_row("v8_Champion", z_v8_o, z_v8_t),
        eval_row("Cand0_SmoothGamma_5F", z_base5_o, z_base5_t),
        eval_row("Ref_Slot3_Champion_F02", z_slot3_o, z_slot3_t),
        eval_row("Cand1_Asymmetric_Shrinkage_H2_F0123", z_cand1_o, z_cand1_t),
        eval_row("Cand2_MultiGate_Jacobian_F0123", z_cand2_o, z_cand2_t),
        eval_row("Cand3_Paired_Twin_Delta_GBDT_F0123", z_cand3_o, z_cand3_t),
        eval_row("Cand4_Slot3_F02_Plus_TwinDelta_F0123", z_cand4_o, z_cand4_t),
    ]

    print("\n" + "=" * 160)
    print("SLOT4 F0123 MASTER TABLE")
    print("=" * 160)
    hdr = (
        f"{'Candidate':44s} | {'NestedOOF':9s} | {'Δv8':9s} | {'ΔS3':9s} | "
        f"{'Wins5':5s} | {'Wins4':5s} | {'PerFoldΔ×1e-5':30s} | {'ΔF0123 vs C0':22s} | "
        f"{'KS v8':7s} | {'1-ρS3':7s}"
    )
    print(hdr)
    print("-" * 160)
    for r in rows:
        print(
            f"{r['name']:44s} | {r['nested_oof_auc']:.7f} | {r['nested_delta_v8']:+.2e} | "
            f"{r['delta_vs_slot3_oof']:+.2e} | {r['fold_wins_5']}/5  | {r['wins4']}/4  | "
            f"{str(r['fold_deltas_1e5']):30s} | {str(r['d_f0123_vs_c0_1e5']):22s} | "
            f"{r['ks_vs_v8']:.5f} | {r['dist_s3_1e4']:7.3f}"
        )
    print("=" * 160)

    print("\n[Gate checks]", flush=True)
    slot4_cands = [r for r in rows if r["name"].startswith(("Cand1_", "Cand2_", "Cand3_", "Cand4_"))]
    qualified = []
    for r in slot4_cands:
        a = r["nested_oof_auc"] >= GATE_OOF_MIN
        b = r["wins4"] >= 3 and r["dF0"] >= 2.2e-5 and r["dF2"] >= 1.0e-5
        c = r["dist_s3_1e4"] >= 0.25
        d = r["ks_vs_v8"] < 0.0025
        print(
            f"  {r['name']}: OOF>={GATE_OOF_MIN}={a} wins4/F0F2={b} "
            f"(wins4={r['wins4']} dF0={r['dF0']*1e5:.2f}e-5 dF2={r['dF2']*1e5:.2f}e-5) "
            f"1-ρS3>={0.25}={c} KS<0.0025={d}",
            flush=True,
        )
        if a and b and c and d:
            qualified.append(r)

    sub_path = SUB / "sub_slot4_gm_verified_f0123.csv"
    if qualified:
        winner = max(qualified, key=lambda r: r["nested_oof_auc"])
        pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
        print(f"\n[GATE PASS] {winner['name']} OOF={winner['nested_oof_auc']:.7f} -> {sub_path.name}", flush=True)
        msg = (
            f"Slot4 GM Verified F0123: {winner['name']} "
            f"(OOF={winner['nested_oof_auc']:.7f}, ΔS3={winner['delta_vs_slot3_oof']:+.2e})"
        )
        cmd = [
            str(ROOT / ".venv_forensics/bin/kaggle"), "competitions", "submit",
            "-c", "playground-series-s6e11", "-f", str(sub_path), "-m", msg,
        ]
        print("[Submitting]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)
        time.sleep(10)
        sl = subprocess.run(
            [str(ROOT / ".venv_forensics/bin/kaggle"), "competitions", "submissions",
             "-c", "playground-series-s6e11", "-v"],
            capture_output=True, text=True, check=False,
        )
        print(sl.stdout[:2500], flush=True)
        gate_status = "SUBMITTED"
    else:
        pool = slot4_cands + [r for r in rows if r["name"].startswith("Ref")]
        winner = max(pool, key=lambda r: (r.get("wins4", 0), r["nested_oof_auc"]))
        pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
        reasons = []
        for r in slot4_cands:
            fails = []
            if r["nested_oof_auc"] < GATE_OOF_MIN:
                fails.append(f"OOF={r['nested_oof_auc']:.7f}<{GATE_OOF_MIN}")
            if r["wins4"] < 3 or r["dF0"] < 2.2e-5 or r["dF2"] < 1.0e-5:
                fails.append(f"fold-guard wins4={r['wins4']} dF0={r['dF0']*1e5:.2f} dF2={r['dF2']*1e5:.2f}")
            if r["dist_s3_1e4"] < 0.25:
                fails.append(f"1-ρS3={r['dist_s3_1e4']:.3f}<0.25")
            if r["ks_vs_v8"] >= 0.0025:
                fails.append(f"KS={r['ks_vs_v8']:.4f}")
            if fails:
                reasons.append(f"{r['name']}: {'; '.join(fails)}")
        print(f"\n[GATE HOLD] Best staged={winner['name']} OOF={winner['nested_oof_auc']:.7f}", flush=True)
        for line in reasons:
            print(f"  - {line}", flush=True)
        print("Preserving submissions for 5-fold completion (~Fold4).", flush=True)
        gate_status = "HOLD"

    summary = {
        "cand0_auc": cand0_auc,
        "slot3_reconstruct": s3_auc,
        "ks_step1": ks_z,
        "twin_base": twin_base_name,
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "winner": {k: v for k, v in winner.items() if k != "p_test"},
        "qualified": len(qualified),
        "gate_status": gate_status,
        "elapsed_sec": round(time.time() - T0, 1),
    }
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "verify_slot4_gm_f0123.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{summary['elapsed_sec']}s] Saved logs/overnight/verify_slot4_gm_f0123.json", flush=True)


if __name__ == "__main__":
    main()
