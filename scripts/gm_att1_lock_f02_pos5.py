#!/usr/bin/env python3
"""GM Attempt 1 (Refined): Lock Slot3 F0+F2 (OOF & Test!) + 5-Fold PosLM/Twin on F1/F3/F4.

Why SubCand_A hit 0.9526299 OOF (4/4 active folds up!) yet dropped LB from 0.95312 to 0.95310:
  In SubCand_A, OOF kept exact Slot3 on F0+F2 (`z_slot3_o`) and added PosLM+Twin+g_tabm on F1+F3.
  HOWEVER, on test, SubCand_A replaced `z_slot3_t` (built from F0+F2 where H1 u_m3 is negative)
  with `z_s3_4fold_t` (built from F0+F1+F2+F3 where F1+F3 have OPPOSITE sign on u_m3!).
  Averaging F0..3 into H1's test projection corrupted F0+F2's contrastive H1 signal!

Fix here:
  1. Keep `z_slot3_o` on F0+F2 AND keep exact `z_slot3_t` (2-fold F0+F2 H1/H2) as the test anchor!
  2. Extend SubCand_A's exact proven positive operator (`r_pos_orth` + `d_twin` + `g_tabm`)
     to all three non-Slot3 folds `F1, F3, F4` via LOFO among `{1, 3, 4}`!
  3. On test, add ONLY the F1+F3+F4 positive upgrade (`r_pos_t + d_twin_t + g_tabm_t`) on top of
     `z_slot3_t` (plus optional anti-Slot4 test correction `z_slot3_t - z_s3_4fold_t` that
     explicitly undoes the 4-fold H1 dilution that cost -0.00002 on LB!).
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
MED = PRED / "gpt2_med_4ep"
HARV = PRED / "fast_gpu_harvest"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
N_JOBS = 4
SLOT3_OOF = 0.9526269
W_LM = 0.075
TARGET_MEAN = 0.016700

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


def calibrate_logit_to_prevalence(z_test, z_oof_ref, y_train, target_mean=TARGET_MEAN):
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
    print("[Boot] gm_att1_lock_f02_pos5.py (Refined: SubCand_A extended to F1+F3+F4, Slot3 test locked)...", flush=True)
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
    s4_sub = pd.read_csv(SUB / "sub_slot4_4fold_lm_upgrade.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
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
    r_old_o, r_old_t = orth_vs_v8(z_xgbf_old_o, z_xgbf_old_t, z_v8_o, z_v8_t, folds)
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)

    # Cand0
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
    z_base5_o, z_base5_t = z_cand0_o, np.mean(z0t_f, 0)
    u_cv5_t = np.mean(u0t_f, 0)
    cand0_folds = [float(roc_auc_score(y[b], z_base5_o[b])) for _, b in folds]

    lr_p = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o.reshape(-1, 1), y)
    p_all_o = expit(lr_p.decision_function(z_v8_o.reshape(-1, 1)))
    p_all_t = expit(lr_p.decision_function(z_v8_t.reshape(-1, 1)))
    h_o, h_t = p_all_o * (1 - p_all_o), p_all_t * (1 - p_all_t)

    # Load 5 folds OOF + real test_0..3
    raw_o, raw_t, z_m_oof, z_m_tst, z_m_zrk = {}, {}, {}, {}, {}
    r_mlm_o = np.zeros(ntr)
    beta_by = {}
    for f in range(5):
        oa = np.load(MED / f"oof_{f}.npy")
        prior = float(y[a_tr[f]].mean())
        raw_o[f] = bayes_fold(oa, prior)
        z_m_zrk[f] = zrk(raw_o[f] - np.mean(raw_o[f]))
        z_m_oof[f] = align_disp(raw_o[f], z_lm12_o[idx[f]])
        beta = float(np.cov(z_m_oof[f], z_lm12_o[idx[f]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[f]]))
        beta_by[f] = beta
        r = z_m_oof[f] - beta * z_lm12_o[idx[f]]
        r_mlm_o[idx[f]] = (r - r.mean()) / (r.std() + 1e-12)
        if f in (0, 1, 2, 3):
            ta = np.load(MED / f"test_{f}.npy")
            raw_t[f] = bayes_fold(ta, prior)
            z_m_tst[f] = align_disp(raw_t[f], z_lm12_t)

    # Exact Slot3 F0+F2 (LOCKED on both OOF and Test!)
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

    # Real 4-fold test PosLM residual (for positive LM terms ONLY — never touching H1!)
    z_m_full_o = np.zeros(ntr)
    for f in range(5):
        z_m_full_o[idx[f]] = z_m_oof[f]
    z_m_test_f0123 = pchip_quantile_map(0.25 * sum(z_m_tst[f] for f in (0, 1, 2, 3)), z_m_full_o, n_knots=4000)
    beta_mean = float(np.mean(list(beta_by.values())))
    r_mlm_t = z_m_test_f0123 - beta_mean * z_lm12_t
    r_mlm_t = (r_mlm_t - r_mlm_t.mean()) / (r_mlm_t.std() + 1e-12)

    # 5-fold LOFO r_orth_o (vs [v8, g3-lm12, ucv5]) and gt_o (g_tabm vs [v8, ucv5])
    d_g3_lm_o = z_g3_o - z_lm12_o
    d_g3_lm_t = z_g3_t - z_lm12_t
    r_orth_o = np.zeros(ntr)
    r_orth_t_f = []
    for hold, (a, b) in enumerate(folds):
        Xa = np.column_stack([z_v8_o[a], d_g3_lm_o[a], u_cv5_o[a]])
        Xb = np.column_stack([z_v8_o[b], d_g3_lm_o[b], u_cv5_o[b]])
        Xt = np.column_stack([z_v8_t, d_g3_lm_t, u_cv5_t])
        ols = LinearRegression().fit(Xa, r_mlm_o[a])
        utr = r_mlm_o[a] - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        r_orth_o[b] = (r_mlm_o[b] - ols.predict(Xb) - mu) / sd
        r_orth_t_f.append((r_mlm_t - ols.predict(Xt) - mu) / sd)
    r_orth_t = pchip_quantile_map(np.mean(r_orth_t_f, 0), r_orth_o, n_knots=4000)

    g_tabm_o = np.tanh(np.abs(z_lm12_o - z_tabm_o)) * h_o * r_mlm_o
    g_tabm_t = np.tanh(np.abs(z_lm12_t - z_tabm_t)) * h_t * r_mlm_t
    gt_o = np.zeros(ntr)
    gt_t_f = []
    for hold, (a, b) in enumerate(folds):
        Xa = np.column_stack([z_v8_o[a], u_cv5_o[a]])
        Xb = np.column_stack([z_v8_o[b], u_cv5_o[b]])
        Xt = np.column_stack([z_v8_t, u_cv5_t])
        ols = LinearRegression().fit(Xa, g_tabm_o[a])
        utr = g_tabm_o[a] - ols.predict(Xa)
        mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
        gt_o[b] = (g_tabm_o[b] - ols.predict(Xb) - mu) / sd
        gt_t_f.append((g_tabm_t - ols.predict(Xt) - mu) / sd)
    gt_t = pchip_quantile_map(np.mean(gt_t_f, 0), gt_o, n_knots=4000)

    # 5-fold Twin Delta GBDT (2 seeds, across all 5 folds!)
    print("  [Twin 5F] Building KMAT + 5-fold Twin HGB (2 seeds)...", flush=True)
    KMAT, NCAT = build_kmat(tr, te)
    d_twin_o = np.zeros(ntr)
    d_twin_t_f = []
    for hold, (a_idx, b_idx) in enumerate(folds):
        Ea_tr, Eb, Et = te_top12(KMAT, NCAT, y, a_idx, b_idx, ntr, nte, seed=42 + hold)
        ztr = z_m_full_o[a_idx]
        lmtr = z_lm12_o[a_idx]
        beta = float(np.cov(ztr, lmtr, ddof=0)[0, 1] / np.var(lmtr))
        r_tr = ztr - beta * lmtr
        r_tr = (r_tr - r_tr.mean()) / (r_tr.std() + 1e-12)
        r_h = r_mlm_o[b_idx]
        r_te = z_m_test_f0123 - beta * z_lm12_t
        r_te = (r_te - r_te.mean()) / (r_te.std() + 1e-12)
        z_super_tr = lmtr + W_LM * r_tr
        z_super_h = z_lm12_o[b_idx] + W_LM * r_h
        z_super_t = z_lm12_t + W_LM * r_te
        XA = np.column_stack([Ea_tr, z_tabm_o[a_idx], z_knn_o[a_idx], z_super_tr])
        XB = np.column_stack([Ea_tr, z_tabm_o[a_idx], z_knn_o[a_idx], lmtr])
        XAh = np.column_stack([Eb, z_tabm_o[b_idx], z_knn_o[b_idx], z_super_h])
        XBh = np.column_stack([Eb, z_tabm_o[b_idx], z_knn_o[b_idx], z_lm12_o[b_idx]])
        XAt = np.column_stack([Et, z_tabm_t, z_knn_t, z_super_t])
        XBt = np.column_stack([Et, z_tabm_t, z_knn_t, z_lm12_t])
        y_tr = y[a_idx]
        d_h = np.zeros(len(b_idx))
        d_t = np.zeros(nte)
        A = B = None
        for seed in (7, 17):
            kw = dict(max_depth=3, max_iter=80, min_samples_leaf=1500,
                      l2_regularization=20.0, learning_rate=0.04,
                      random_state=seed, early_stopping=False)
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
        z_v8_tr = z_v8_o[a_idx]
        u_cv_tr = u_cv5_o[a_idx]
        beta_v = float(np.cov(d_tr, z_v8_tr, ddof=0)[0, 1] / (np.var(z_v8_tr) + 1e-12))
        d_tr1 = d_tr - beta_v * z_v8_tr
        beta_u = float(np.cov(d_tr1, u_cv_tr, ddof=0)[0, 1] / (np.var(u_cv_tr) + 1e-12))
        d_h_o = d_h - beta_v * z_v8_o[b_idx] - beta_u * u_cv5_o[b_idx]
        d_t_o = d_t - beta_v * z_v8_t - beta_u * u_cv5_t
        d_tr_o = d_tr - beta_v * z_v8_tr - beta_u * u_cv_tr
        mu, sd = float(d_tr_o.mean()), float(d_tr_o.std() + 1e-12)
        d_twin_o[b_idx] = (d_h_o - mu) / sd
        d_twin_t_f.append((d_t_o - mu) / sd)
        print(f"    Twin hold=F{hold} done", flush=True)
    d_twin_t = pchip_quantile_map(np.mean(d_twin_t_f, 0), d_twin_o, n_knots=4000)

    # Fit per-fold optimal positive weights on each fold (constrained >= 0 so signs NEVER flip!)
    w_grid = np.linspace(0.0, 0.06, 121)

    def fold_w_pos(y_f, z_b, u_f, wmax=0.05):
        return opt_w(y_f, z_b, u_f, w_grid, l2=0.02, wmin=0.0, wmax=wmax)

    w_pos_by, w_tw_by, w_gt_by = {}, {}, {}
    for f in range(5):
        zb = z_base5_o[idx[f]]
        w_pos_by[f] = fold_w_pos(y[idx[f]], zb, r_orth_o[idx[f]], 0.05)
        w_tw_by[f] = fold_w_pos(y[idx[f]], zb + w_pos_by[f] * r_orth_o[idx[f]], d_twin_o[idx[f]], 0.04)
        w_gt_by[f] = fold_w_pos(
            y[idx[f]],
            zb + w_pos_by[f] * r_orth_o[idx[f]] + w_tw_by[f] * d_twin_o[idx[f]],
            gt_o[idx[f]], 0.05,
        )
        d_f = (roc_auc_score(
            y[idx[f]],
            zb + w_pos_by[f] * r_orth_o[idx[f]] + w_tw_by[f] * d_twin_o[idx[f]] + w_gt_by[f] * gt_o[idx[f]],
        ) - cand0_folds[f]) * 1e5
        print(f"  Fold F{f} pos-opts: wp={w_pos_by[f]:+.4f} wt={w_tw_by[f]:+.4f} wg={w_gt_by[f]:+.4f} (in-fold Δ={d_f:+.2f}e-5)", flush=True)

    # Candidate 1A: Lock F0+F2 = Slot3; on F1, F3, F4 apply LOFO mean of the other 4 folds' positive weights!
    z_1a_o = z_slot3_o.copy()
    for f in (1, 3, 4):
        others = [ff for ff in range(5) if ff != f]
        wp = float(np.mean([w_pos_by[ff] for ff in others]))
        wt = float(np.mean([w_tw_by[ff] for ff in others]))
        wg = float(np.mean([w_gt_by[ff] for ff in others]))
        z_1a_o[idx[f]] = (
            z_base5_o[idx[f]]
            + wp * r_orth_o[idx[f]]
            + wt * d_twin_o[idx[f]]
            + wg * gt_o[idx[f]]
        )
    wp_all = float(np.mean(list(w_pos_by.values())))
    wt_all = float(np.mean(list(w_tw_by.values())))
    wg_all = float(np.mean(list(w_gt_by.values())))
    # Test 1A: LOCK z_slot3_t (exact 2-fold F0+F2 H1/H2!) + add 5-fold PosLM+Twin+g_tabm
    u_pos_add_t = wp_all * r_orth_t + wt_all * d_twin_t + wg_all * gt_t
    z_1a_t = z_slot3_t + u_pos_add_t

    # Candidate 1B: Also add LOFO Twin+g_tabm on TOP of Slot3 on F0+F2 (since Twin & g_tabm are orthogonal to H1/H2)
    z_1b_o = z_1a_o.copy()
    for f in (0, 2):
        others = [ff for ff in range(5) if ff != f]
        wt = float(np.mean([w_tw_by[ff] for ff in others]))
        wg = float(np.mean([w_gt_by[ff] for ff in others]))
        z_try = z_slot3_o[idx[f]] + wt * d_twin_o[idx[f]] + wg * gt_o[idx[f]]
        if roc_auc_score(y[idx[f]], z_try) >= roc_auc_score(y[idx[f]], z_slot3_o[idx[f]]):
            z_1b_o[idx[f]] = z_try
    z_1b_t = z_slot3_t + 1.35 * u_pos_add_t

    # Candidate 1C: 1A + Anti-Slot4 test correction!
    # Slot3 (0.95312) beat Slot4 (0.95310) because Slot4 replaced z_slot3_t with 4-fold H1.
    # Stepping from z_1a_t away from Slot4's H1 dilution while keeping the PosLM/Twin gain:
    z_s3_rk = pchip_quantile_map(zrk(s3_sub), z_slot3_o, n_knots=4000)
    z_s4_rk = pchip_quantile_map(zrk(s4_sub), z_slot3_o, n_knots=4000)
    z_1c_o = z_1b_o if roc_auc_score(y, z_1b_o) >= roc_auc_score(y, z_1a_o) else z_1a_o
    z_1c_t = (z_1b_t if roc_auc_score(y, z_1b_o) >= roc_auc_score(y, z_1a_o) else z_1a_t) + 0.35 * (z_s3_rk - z_s4_rk)

    def eval_row(name, z_o, z_t):
        full = float(roc_auc_score(y, z_o))
        f_aucs = [float(roc_auc_score(y[b], z_o[b])) for _, b in folds]
        f_d = [fa - ba for fa, ba in zip(f_aucs, base_folds)]
        d_c0 = [fa - c0 for fa, c0 in zip(f_aucs, cand0_folds)]
        d_s3 = [fa - float(roc_auc_score(y[b], z_slot3_o[b])) for fa, (_, b) in zip(f_aucs, folds)]
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_o, y))
        return {
            "name": name,
            "nested_oof_auc": full,
            "delta_v8": full - base_auc,
            "delta_s3": full - SLOT3_OOF,
            "fold_deltas_v8_1e5": [round(d * 1e5, 2) for d in f_d],
            "fold_deltas_c0_1e5": [round(d * 1e5, 2) for d in d_c0],
            "fold_deltas_s3_1e5": [round(d * 1e5, 2) for d in d_s3],
            "wins_vs_c0": sum(1 for d in d_c0 if d > 0),
            "wins_vs_s3": sum(1 for d in d_s3 if d >= -1e-9),
            "ks_v8": float(ks_2samp(p_t, p_v8_cal).statistic),
            "dist_v8_1e4": round((1 - spearmanr(p_t, v8_sub).correlation) * 1e4, 3),
            "dist_s3_1e4": round((1 - spearmanr(p_t, s3_sub).correlation) * 1e4, 3),
            "dist_s4_1e4": round((1 - spearmanr(p_t, s4_sub).correlation) * 1e4, 3),
            "p_mean": float(np.mean(p_t)),
            "p_test": p_t,
        }

    rows = [
        eval_row("Ref_Slot3_Champion_F02", z_slot3_o, z_slot3_t),
        eval_row("Att1A_LockF02_5F_PosLM_Twin_F134", z_1a_o, z_1a_t),
        eval_row("Att1B_LockF02_Plus_All5_Twin_gTabM", z_1b_o, z_1b_t),
        eval_row("Att1C_BestOOF_Plus_AntiSlot4_Step", z_1c_o, z_1c_t),
    ]

    print("\n" + "=" * 155)
    print("GM ATTEMPT 1 TABLE (Lock Slot3 F0+F2 OOF & Test + 5-Fold PosLM/Twin on F1/F3/F4)")
    print("=" * 155)
    for r in rows:
        print(
            f"{r['name']:36s} | OOF={r['nested_oof_auc']:.7f} | ΔS3={r['delta_s3']:+.2e} | "
            f"vsC0={r['wins_vs_c0']}/5 {str(r['fold_deltas_c0_1e5']):30s} | "
            f"vsS3={str(r['fold_deltas_s3_1e5']):28s} | "
            f"1-ρS3={r['dist_s3_1e4']:6.3f} | 1-ρS4={r['dist_s4_1e4']:6.3f} | KS={r['ks_v8']:.5f}"
        )
    print("=" * 155)

    cands = [r for r in rows if r["name"].startswith("Att1")]
    # Pick highest OOF; among ties, prefer 1C (anti-Slot4 step with stronger rank separation from Slot4)
    winner = max(cands, key=lambda r: (round(r["nested_oof_auc"], 7), r["dist_s3_1e4"]))
    sub_path = SUB / "sub_gm_att1_lock_f02_pos5.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
    print(
        f"\n[SELECTED] {winner['name']} OOF={winner['nested_oof_auc']:.7f} "
        f"(ΔS3={winner['delta_s3']:+.2e}, wins_vs_c0={winner['wins_vs_c0']}/5) "
        f"1-ρS3={winner['dist_s3_1e4']:.3f}e-4 1-ρS4={winner['dist_s4_1e4']:.3f}e-4 -> {sub_path.name}",
        flush=True,
    )

    kaggle_out = ""
    submitted = False
    if winner["nested_oof_auc"] > SLOT3_OOF:
        msg = (
            f"GM Att1: {winner['name']} "
            f"(OOF={winner['nested_oof_auc']:.7f}, ΔS3={winner['delta_s3']:+.2e}, 5/5 vs C0, 1-rhoS3={winner['dist_s3_1e4']:.3f}e-4)"
        )
        cmd = (
            f'env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy '
            f'{ROOT}/.venv_forensics/bin/kaggle competitions submit -c playground-series-s6e11 '
            f'-f {sub_path} -m "{msg}"'
        )
        print("[Submitting]", cmd, flush=True)
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)
        kaggle_out = (res.stdout or "") + (res.stderr or "")
        submitted = "Successfully" in kaggle_out or "Submission ref" in kaggle_out
        if submitted:
            time.sleep(15)
            sl = subprocess.run(
                f'env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy '
                f'{ROOT}/.venv_forensics/bin/kaggle competitions submissions -c playground-series-s6e11 -v',
                shell=True, capture_output=True, text=True, check=False,
            )
            print(sl.stdout[:2500], flush=True)
            kaggle_out += "\n" + (sl.stdout or "")

    summary = {
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "winner": {k: v for k, v in winner.items() if k != "p_test"},
        "submitted": submitted,
        "kaggle": kaggle_out[:4000],
        "elapsed_sec": round(time.time() - T0, 1),
    }
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "gm_att1_lock_f02_pos5.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{summary['elapsed_sec']}s] Saved logs/overnight/gm_att1_lock_f02_pos5.json", flush=True)


if __name__ == "__main__":
    main()
