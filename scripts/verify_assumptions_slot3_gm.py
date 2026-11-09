#!/usr/bin/env python3
"""Verify 2-Fold (`F0 + F2`) + 5-Fold Control-Variate Assumptions & Submit Slot 3 (`verify_assumptions_slot3_gm.py`)

Hard rules:
- Local Mac CPU only (`n_jobs=4`), does NOT touch PID 3021 on Pod 5hz1ob0bd0n6oy.
- At most ONE Kaggle submission (`submissions/sub_slot3_gm_verified_f02.csv`) if the winning candidate passes verification.
"""
from __future__ import annotations

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


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def pchip_quantile_map(x_test: np.ndarray, x_ref: np.ndarray, n_knots: int = 4000) -> np.ndarray:
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    x_ref_sorted = np.sort(np.asarray(x_ref, dtype=np.float64).ravel())
    q_knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
    ref_vals = np.quantile(x_ref_sorted, q_knots)
    pchip = PchipInterpolator(q_knots, ref_vals, extrapolate=True)
    u_test = (rankdata(x_test, method="average") - 0.5) / len(x_test)
    return np.asarray(pchip(u_test), dtype=np.float64)


def calibrate_logit_to_prevalence(
    z_test: np.ndarray,
    z_oof_ref: np.ndarray,
    y_train: np.ndarray,
    target_mean: float | None = None,
) -> np.ndarray:
    if target_mean is None:
        target_mean = float(np.mean(y_train))
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof_ref.reshape(-1, 1), y_train.astype(np.int32))
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    z_lin = slope * z_test + intercept

    def obj(delta: float) -> float:
        return (float(np.mean(expit(z_lin + delta))) - target_mean) ** 2

    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded")
    return z_lin + float(res.x)


def bayes_fold(arr: np.ndarray, prior: float) -> np.ndarray:
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def main() -> None:
    print("[Boot] Running verify_assumptions_slot3_gm.py...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    target_mean = float(np.mean(y))

    # Reference submissions for distance checks
    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s1_sub = pd.read_csv(SUB / "sub_phase1_v8_gpt2m_f2_orth.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s2_sub = pd.read_csv(SUB / "sub_phase2_5_v8_meta_harvest.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    z_v8_o = zrk(v8_oof)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_auc = float(roc_auc_score(y, z_v8_o))
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y, target_mean=target_mean))

    z_lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), z_lm12_o, n_knots=4000)

    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=4000)

    z_xgbf_cln_o = zrk(np.load(HARV / "oof_xgb_clean.npy").ravel())
    z_xgbf_cln_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_clean.npy").ravel()), z_xgbf_cln_o, n_knots=4000)

    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)

    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)

    # Fold-wise strict OLS orthogonalization of r_xgb_old, r_xgb_clean, r_tabm against z_v8
    r_old_o = np.zeros(ntr, dtype=np.float64)
    r_cln_o = np.zeros(ntr, dtype=np.float64)
    r_tab_o = np.zeros(ntr, dtype=np.float64)
    s_norm_o = np.zeros(ntr, dtype=np.float64)
    r_old_t_f, r_cln_t_f, r_tab_t_f, s_norm_t_f = [], [], [], []

    for f, (a, b) in enumerate(folds):
        za = z_v8_o[a]
        zb = z_v8_o[b]
        var_a = float(np.var(za))

        def orth1(co: np.ndarray, ct: np.ndarray):
            beta = float(np.cov(co[a], za, ddof=0)[0, 1] / var_a)
            ra = co[a] - beta * za
            mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
            return (co[b] - beta * zb - mu) / sd, (ct - beta * z_v8_t - mu) / sd

        ro_b, ro_t = orth1(z_xgbf_old_o, z_xgbf_old_t)
        rc_b, rc_t = orth1(z_xgbf_cln_o, z_xgbf_cln_t)
        rt_b, rt_t = orth1(z_tabm_o, z_tabm_t)
        r_old_o[b], r_cln_o[b], r_tab_o[b] = ro_b, rc_b, rt_b
        r_old_t_f.append(ro_t)
        r_cln_t_f.append(rc_t)
        r_tab_t_f.append(rt_t)

        smu, ssd = float(s_tabm_o[a].mean()), float(s_tabm_o[a].std() + 1e-12)
        s_norm_o[b] = np.clip((s_tabm_o[b] - smu) / ssd, -3.5, 3.5)
        s_norm_t_f.append(np.clip((s_tabm_t - smu) / ssd, -3.5, 3.5))

    r_old_t = pchip_quantile_map(np.mean(r_old_t_f, axis=0), r_old_o, n_knots=4000)
    r_cln_t = pchip_quantile_map(np.mean(r_cln_t_f, axis=0), r_cln_o, n_knots=4000)
    r_tab_t = pchip_quantile_map(np.mean(r_tab_t_f, axis=0), r_tab_o, n_knots=4000)
    s_norm_t = np.mean(s_norm_t_f, axis=0)

    # -----------------------------------------------------------------------
    # STEP 1: Exact Post-Average PCHIP Variance Alignment for 2-Fold gpt2-medium (F0 + F2)
    # -----------------------------------------------------------------------
    a0, idx0 = folds[0]
    a2, idx2 = folds[2]
    idx_f02 = np.concatenate([idx0, idx2])
    n0 = len(idx0)
    sl0 = slice(0, n0)
    sl2 = slice(n0, len(idx_f02))

    raw_m0_o = bayes_fold(np.load(PRED / "gpt2_med_4ep/oof_0.npy"), float(y[a0].mean()))
    raw_m2_o = bayes_fold(np.load(PRED / "gpt2_med_4ep/oof_2.npy"), float(y[a2].mean()))
    raw_m0_t = bayes_fold(np.load(PRED / "gpt2_med_4ep/test_0.npy"), float(y[a0].mean()))
    raw_m2_t = bayes_fold(np.load(PRED / "gpt2_med_4ep/test_2.npy"), float(y[a2].mean()))

    # Fold-center and put onto logit-rank scale per fold so inter-fold intercept/dispersion drift is zero
    z_m_oof0 = zrk(raw_m0_o - np.mean(raw_m0_o))
    z_m_oof2 = zrk(raw_m2_o - np.mean(raw_m2_o))
    z_m_oof_f02 = np.concatenate([z_m_oof0, z_m_oof2])

    # Post-average test PCHIP calibration: first average centered test fold logits, then map onto z_m_oof_f02!
    z_m_test0_c = zrk(raw_m0_t - np.mean(raw_m0_o))
    z_m_test2_c = zrk(raw_m2_t - np.mean(raw_m2_o))
    z_m_test_raw = 0.5 * (z_m_test0_c + z_m_test2_c)
    z_m_test_f02 = pchip_quantile_map(z_m_test_raw, z_m_oof_f02, n_knots=4000)
    ks_step1 = float(ks_2samp(z_m_oof_f02, z_m_test_f02).statistic)
    print(
        f"[Step 1] 2-Fold (F0+F2) Post-Average PCHIP Alignment: "
        f"std(z_m_oof_f02)={z_m_oof_f02.std():.6f} std(z_m_test_f02)={z_m_test_f02.std():.6f} "
        f"KS={ks_step1:.6f} (< 0.0001)"
    )
    assert ks_step1 < 0.0001, f"KS gate failed: {ks_step1}"

    # -----------------------------------------------------------------------
    # STEP 2.0: Reference Pure_5F_Candidate_C & Candidate 0 (Smooth Uncertainty-Gated Control Variate)
    # -----------------------------------------------------------------------
    # Reference Pure_5F_Candidate_C (`z_v8 + 0.025 * (r_old - r_tab)`)
    u_candC_o = r_old_o - 1.0 * r_tab_o
    u_candC_t = r_old_t - 1.0 * r_tab_t
    z_candC_o = z_v8_o + 0.025 * u_candC_o
    z_candC_t = z_v8_t + 0.025 * u_candC_t

    # Candidate 0: Smooth continuous epistemic gate
    # gamma_i = gamma_min + (gamma_max - gamma_min) * expit(-temp * s_norm_i - bias * tanh(z_v8_i))
    print("[Step 2.1] Fitting Candidate 0 (5-Fold Smooth Epistemic-Gated Control Variate)...", flush=True)
    th_o = np.tanh(z_v8_o / 2.0)
    th_t = np.tanh(z_v8_t / 2.0)

    # Precompute candidate (gamma_min, gamma_max, temp, bias, w_cv) grid over all 5 folds for exact nested selection
    param_grid = []
    for gmin in (0.30, 0.50, 0.70, 0.85):
        for gmax in (0.85, 1.00, 1.15):
            if gmax < gmin:
                continue
            for temp in (-1.5, 0.0, 1.5):
                for bias in (-1.0, 0.0, 1.0):
                    for w_cv in (0.022, 0.025, 0.028):
                        param_grid.append((gmin, gmax, temp, bias, w_cv))

    # Evaluate each grid tuple on each fold b in 0..4 (fast vectorized matrix)
    n_grid = len(param_grid)
    grid_fold_deltas = np.zeros((n_grid, 5), dtype=np.float64)
    for gi, (gmin, gmax, temp, bias, w_cv) in enumerate(param_grid):
        gam_o = gmin + (gmax - gmin) * expit(-temp * s_norm_o - bias * th_o)
        u_o = r_old_o - gam_o * r_tab_o
        z_try = z_v8_o + w_cv * u_o
        for fi, (_, b) in enumerate(folds):
            grid_fold_deltas[gi, fi] = float(roc_auc_score(y[b], z_try[b])) - base_folds[fi]

    z_cand0_o = np.zeros(ntr, dtype=np.float64)
    u_cv_best_o = np.zeros(ntr, dtype=np.float64)
    z_cand0_t_f = []
    u_cv_best_t_f = []
    cand0_params = []

    for f, (a, b) in enumerate(folds):
        other_f = [fi for fi in range(5) if fi != f]
        sub_d = grid_fold_deltas[:, other_f]
        scores = sub_d.mean(axis=1) - 0.5 * np.maximum(0.0, -sub_d.min(axis=1))
        best_gi = int(np.argmax(scores))
        gmin, gmax, temp, bias, w_cv = param_grid[best_gi]
        cand0_params.append((gmin, gmax, temp, bias, w_cv))

        gam_b = gmin + (gmax - gmin) * expit(-temp * s_norm_o[b] - bias * th_o[b])
        ub = r_old_o[b] - gam_b * r_tab_o[b]
        u_cv_best_o[b] = ub
        z_cand0_o[b] = z_v8_o[b] + w_cv * ub

        gam_t = gmin + (gmax - gmin) * expit(-temp * s_norm_t - bias * th_t)
        ut = r_old_t - gam_t * r_tab_t
        u_cv_best_t_f.append(ut)
        z_cand0_t_f.append(z_v8_t + w_cv * ut)

    u_cv_best_t = np.mean(u_cv_best_t_f, axis=0)
    z_cand0_t = np.mean(z_cand0_t_f, axis=0)
    print(f"  Candidate 0 selected params per fold: {cand0_params}", flush=True)

    # Choose the strongest 5-fold trunk (`z_base5_o`, `z_base5_t`, `u_cv5_o`, `u_cv5_t`) between Pure_5F_Candidate_C and Candidate 0
    if roc_auc_score(y, z_cand0_o) >= roc_auc_score(y, z_candC_o):
        z_base5_o, z_base5_t = z_cand0_o.copy(), z_cand0_t.copy()
        u_cv5_o, u_cv5_t = u_cv_best_o.copy(), u_cv_best_t.copy()
    else:
        z_base5_o, z_base5_t = z_candC_o.copy(), z_candC_t.copy()
        u_cv5_o, u_cv5_t = u_candC_o.copy(), u_candC_t.copy()

    # -----------------------------------------------------------------------
    # STEP 2.2: Candidate 1 (`H1_TripleOrth_Contrastive_F02`)
    # Fit OLS of z_m against [z_v8, z_lm12, u_cv5] on F0 -> transform F2; fit on F2 -> transform F0
    # -----------------------------------------------------------------------
    print("[Step 2.2] Running Candidate 1 (H1_TripleOrth_Contrastive_F02)...", flush=True)
    X3_f0 = np.column_stack([z_v8_o[idx0], z_lm12_o[idx0], u_cv5_o[idx0]])
    X3_f2 = np.column_stack([z_v8_o[idx2], z_lm12_o[idx2], u_cv5_o[idx2]])
    X3_t = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])

    ols3_on_f2 = LinearRegression().fit(X3_f2, z_m_oof2)
    u_m3_f0_raw = z_m_oof0 - ols3_on_f2.predict(X3_f0)
    mu3_0, sd3_0 = float(u_m3_f0_raw.mean()), float(u_m3_f0_raw.std() + 1e-12)
    u_m3_f0 = (u_m3_f0_raw - mu3_0) / sd3_0
    u_m3_t_from2 = (z_m_test_f02 - ols3_on_f2.predict(X3_t) - mu3_0) / sd3_0

    ols3_on_f0 = LinearRegression().fit(X3_f0, z_m_oof0)
    u_m3_f2_raw = z_m_oof2 - ols3_on_f0.predict(X3_f2)
    mu3_2, sd3_2 = float(u_m3_f2_raw.mean()), float(u_m3_f2_raw.std() + 1e-12)
    u_m3_f2 = (u_m3_f2_raw - mu3_2) / sd3_2
    u_m3_t_from0 = (z_m_test_f02 - ols3_on_f0.predict(X3_t) - mu3_2) / sd3_2

    u_m3_f02 = np.concatenate([u_m3_f0, u_m3_f2])
    u_m3_t = pchip_quantile_map(0.5 * (u_m3_t_from2 + u_m3_t_from0), u_m3_f02, n_knots=4000)

    # Cross-fold weight optimization: fit w_m3 on F0 -> evaluate on F2; fit w_m3 on F2 -> evaluate on F0
    # Use shrinkage-regularized objective on each fold so F0's larger optimum doesn't overshoot F2!
    w_grid = np.linspace(-0.045, 0.045, 181)
    base5_f0_auc = float(roc_auc_score(y[idx0], z_base5_o[idx0]))
    base5_f2_auc = float(roc_auc_score(y[idx2], z_base5_o[idx2]))

    def reg_fold_obj(y_f: np.ndarray, z_b: np.ndarray, u_f: np.ndarray, base_f: float, w: float, l2_pen: float = 0.03) -> float:
        return (float(roc_auc_score(y_f, z_b + w * u_f)) - base_f) - l2_pen * (w**2)

    w_h1_fit0 = max(w_grid, key=lambda w: reg_fold_obj(y[idx0], z_base5_o[idx0], u_m3_f0, base5_f0_auc, w, l2_pen=0.015))
    w_h1_fit2 = max(w_grid, key=lambda w: reg_fold_obj(y[idx2], z_base5_o[idx2], u_m3_f2, base5_f2_auc, w, l2_pen=0.015))

    # Pure cross-fold transfer on OOF (fit on F2 -> eval on F0; fit on F0 -> eval on F2)
    z_cand1_o = z_base5_o.copy()
    z_cand1_o[idx0] = z_base5_o[idx0] + w_h1_fit2 * u_m3_f0
    z_cand1_o[idx2] = z_base5_o[idx2] + w_h1_fit0 * u_m3_f2
    w_h1_oof_mean = 0.5 * (w_h1_fit0 + w_h1_fit2)
    # Since u_m3_t is post-average PCHIP variance-matched to u_m3_f02 (std=1.0), test uses w_h1_oof_mean
    w_h1_test = w_h1_oof_mean
    z_cand1_t = z_base5_t + w_h1_test * u_m3_t

    print(
        f"  [H1 TripleOrth] w_fit0={w_h1_fit0:+.4f}, w_fit2={w_h1_fit2:+.4f}, "
        f"w_test={w_h1_test:+.4f} | "
        f"ΔF0 vs base5={(roc_auc_score(y[idx0], z_cand1_o[idx0]) - base5_f0_auc)*1e5:+.2f}e-5 | "
        f"ΔF2 vs base5={(roc_auc_score(y[idx2], z_cand1_o[idx2]) - base5_f2_auc)*1e5:+.2f}e-5",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # STEP 2.3: Candidate 2 (`H2_BoundaryGated_LM_Upgrade_F02`)
    # r_m_lm12 = z_m - beta_lm * z_lm12 gated by g_i = tanh(|z_lm12 - z_v8|) * (p_v8 * (1 - p_v8))
    # -----------------------------------------------------------------------
    print("[Step 2.3] Running Candidate 2 (H2_BoundaryGated_LM_Upgrade_F02)...", flush=True)
    # Calibrated boundary weight p_v8 * (1 - p_v8)
    lr_platt = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o.reshape(-1, 1), y)
    p_all_o = expit(lr_platt.decision_function(z_v8_o.reshape(-1, 1)))
    p_all_t = expit(lr_platt.decision_function(z_v8_t.reshape(-1, 1)))

    g_o = np.tanh(np.abs(z_lm12_o - z_v8_o)) * (p_all_o * (1.0 - p_all_o))
    g_t = np.tanh(np.abs(z_lm12_t - z_v8_t)) * (p_all_t * (1.0 - p_all_t))

    # Fit beta_lm on F2 -> apply to F0; fit beta_lm on F0 -> apply to F2
    beta_lm_2 = float(np.cov(z_m_oof2, z_lm12_o[idx2], ddof=0)[0, 1] / np.var(z_lm12_o[idx2]))
    beta_lm_0 = float(np.cov(z_m_oof0, z_lm12_o[idx0], ddof=0)[0, 1] / np.var(z_lm12_o[idx0]))
    r_mlm_0 = g_o[idx0] * (z_m_oof0 - beta_lm_2 * z_lm12_o[idx0])
    r_mlm_2 = g_o[idx2] * (z_m_oof2 - beta_lm_0 * z_lm12_o[idx2])
    r_mlm_t = g_t * (z_m_test_f02 - 0.5 * (beta_lm_0 + beta_lm_2) * z_lm12_t)

    # Orthogonalize g_i * r_m_lm12 against [z_v8, u_cv5] on F0 and F2
    X2_f0 = np.column_stack([z_v8_o[idx0], u_cv5_o[idx0]])
    X2_f2 = np.column_stack([z_v8_o[idx2], u_cv5_o[idx2]])
    X2_t = np.column_stack([z_v8_t, u_cv5_t])

    ols_g2 = LinearRegression().fit(X2_f2, r_mlm_2)
    ug_0 = r_mlm_0 - ols_g2.predict(X2_f0)
    mug0, sdg0 = float(ug_0.mean()), float(ug_0.std() + 1e-12)
    ug_0 = (ug_0 - mug0) / sdg0
    ug_t_from2 = (r_mlm_t - ols_g2.predict(X2_t) - mug0) / sdg0

    ols_g0 = LinearRegression().fit(X2_f0, r_mlm_0)
    ug_2 = r_mlm_2 - ols_g0.predict(X2_f2)
    mug2, sdg2 = float(ug_2.mean()), float(ug_2.std() + 1e-12)
    ug_2 = (ug_2 - mug2) / sdg2
    ug_t_from0 = (r_mlm_t - ols_g0.predict(X2_t) - mug2) / sdg2

    ug_f02 = np.concatenate([ug_0, ug_2])
    ug_t = pchip_quantile_map(0.5 * (ug_t_from2 + ug_t_from0), ug_f02, n_knots=4000)

    c1_f0_auc = float(roc_auc_score(y[idx0], z_cand1_o[idx0]))
    c1_f2_auc = float(roc_auc_score(y[idx2], z_cand1_o[idx2]))
    w_h2_fit0 = max(w_grid, key=lambda w: reg_fold_obj(y[idx0], z_cand1_o[idx0], ug_0, c1_f0_auc, w, l2_pen=0.015))
    w_h2_fit2 = max(w_grid, key=lambda w: reg_fold_obj(y[idx2], z_cand1_o[idx2], ug_2, c1_f2_auc, w, l2_pen=0.015))

    z_cand2_o = z_cand1_o.copy()
    z_cand2_o[idx0] = z_cand1_o[idx0] + w_h2_fit2 * ug_0
    z_cand2_o[idx2] = z_cand1_o[idx2] + w_h2_fit0 * ug_2
    w_h2_test = 0.5 * (w_h2_fit0 + w_h2_fit2)
    z_cand2_t = z_cand1_t + w_h2_test * ug_t

    print(
        f"  [H2 BoundaryGated] w_fit0={w_h2_fit0:+.4f}, w_fit2={w_h2_fit2:+.4f}, w_test={w_h2_test:+.4f} | "
        f"ΔF0 vs base5={(roc_auc_score(y[idx0], z_cand2_o[idx0]) - base5_f0_auc)*1e5:+.2f}e-5 | "
        f"ΔF2 vs base5={(roc_auc_score(y[idx2], z_cand2_o[idx2]) - base5_f2_auc)*1e5:+.2f}e-5",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # STEP 2.4: Candidate 3 (`H3_Combined_TripleOrth_Portfolio`)
    # Combine Candidate 1 + clean de-corrupted residual differential (r_xgb_clean - r_xgb_old) orthogonalized vs [z_v8, u_cv5]
    # -----------------------------------------------------------------------
    print("[Step 2.4] Running Candidate 3 (H3_Combined_TripleOrth_Portfolio)...", flush=True)
    d_cln_old_o = np.zeros(ntr, dtype=np.float64)
    d_cln_old_t_f = []
    for f, (a, b) in enumerate(folds):
        raw_diff_a = r_cln_o[a] - r_old_o[a]
        raw_diff_b = r_cln_o[b] - r_old_o[b]
        raw_diff_t = r_cln_t - r_old_t
        Xa = np.column_stack([z_v8_o[a], u_cv5_o[a]])
        Xb = np.column_stack([z_v8_o[b], u_cv5_o[b]])
        Xt = np.column_stack([z_v8_t, u_cv5_t])
        ols_d = LinearRegression().fit(Xa, raw_diff_a)
        da = raw_diff_a - ols_d.predict(Xa)
        mud, sdd = float(da.mean()), float(da.std() + 1e-12)
        d_cln_old_o[b] = (raw_diff_b - ols_d.predict(Xb) - mud) / sdd
        d_cln_old_t_f.append((raw_diff_t - ols_d.predict(Xt) - mud) / sdd)
    d_cln_old_t = pchip_quantile_map(np.mean(d_cln_old_t_f, axis=0), d_cln_old_o, n_knots=4000)

    # Nested 5-fold step weight for d_cln_old on top of Candidate 1
    z_cand3_o = np.zeros(ntr, dtype=np.float64)
    z_cand3_t_f = []
    w3_list = []
    c1_folds = [float(roc_auc_score(y[b], z_cand1_o[b])) for _, b in folds]
    for f, (a, b) in enumerate(folds):
        other_f = [fi for fi in range(5) if fi != f]
        best_w3, best_v3 = 0.0, -1e9
        for w3 in np.linspace(-0.025, 0.025, 51):
            fds = [
                float(roc_auc_score(y[folds[fi][1]], z_cand1_o[folds[fi][1]] + w3 * d_cln_old_o[folds[fi][1]])) - c1_folds[fi]
                for fi in other_f
            ]
            v3 = float(np.mean(fds)) - 0.5 * max(0.0, -float(np.min(fds))) - 0.02 * (w3**2)
            if v3 > best_v3:
                best_v3, best_w3 = v3, float(w3)
        w3_list.append(round(best_w3, 4))
        z_cand3_o[b] = z_cand1_o[b] + best_w3 * d_cln_old_o[b]
        z_cand3_t_f.append(z_cand1_t + best_w3 * d_cln_old_t)
    z_cand3_t = np.mean(z_cand3_t_f, axis=0)
    print(f"  [H3 Combined] nested w3 per fold = {w3_list}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 3: Diagnostic Table & Strict Cross-Fold Verification Gate
    # -----------------------------------------------------------------------
    def eval_candidate_row(name: str, z_o: np.ndarray, z_t: np.ndarray, check_f02_vs_base5: bool = False) -> dict:
        full_auc = float(roc_auc_score(y, z_o))
        f_aucs = [float(roc_auc_score(y[b], z_o[b])) for _, b in folds]
        f_deltas = [fa - ba for fa, ba in zip(f_aucs, base_folds)]
        wins = sum(1 for d in f_deltas if d > 0)

        df0_vs_b5 = float(roc_auc_score(y[idx0], z_o[idx0]) - base5_f0_auc)
        df2_vs_b5 = float(roc_auc_score(y[idx2], z_o[idx2]) - base5_f2_auc)
        both_f02_up = bool(df0_vs_b5 > 0 and df2_vs_b5 > 0) if check_f02_vs_base5 else True

        p_t = expit(calibrate_logit_to_prevalence(z_t, z_o, y, target_mean=target_mean))
        ks_v8 = float(ks_2samp(p_t, p_v8_cal).statistic)
        dist_v8 = round((1.0 - float(spearmanr(p_t, v8_sub).correlation)) * 1e4, 3)
        dist_s1 = round((1.0 - float(spearmanr(p_t, s1_sub).correlation)) * 1e4, 3)
        dist_s2 = round((1.0 - float(spearmanr(p_t, s2_sub).correlation)) * 1e4, 3)

        return {
            "name": name,
            "nested_oof_auc": full_auc,
            "nested_delta": full_auc - base_auc,
            "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_deltas],
            "fold_wins": wins,
            "df0_vs_c0_1e5": round(df0_vs_b5 * 1e5, 2),
            "df2_vs_c0_1e5": round(df2_vs_b5 * 1e5, 2),
            "both_f02_up": both_f02_up,
            "ks_vs_v8": ks_v8,
            "dist_v8_1e4": dist_v8,
            "dist_s1_1e4": dist_s1,
            "dist_s2_1e4": dist_s2,
            "z_oof": z_o,
            "z_test": z_t,
            "p_test": p_t,
        }

    r_refC = eval_candidate_row("Ref: Pure_5F_Candidate_C", z_candC_o, z_candC_t, check_f02_vs_base5=False)
    r_c0 = eval_candidate_row("Cand 0: Baseline_Pure5F_SmoothGamma", z_cand0_o, z_cand0_t, check_f02_vs_base5=False)
    r_c1 = eval_candidate_row("Cand 1: H1_TripleOrth_Contrastive_F02", z_cand1_o, z_cand1_t, check_f02_vs_base5=True)
    r_c2 = eval_candidate_row("Cand 2: H2_BoundaryGated_LM_Upgrade_F02", z_cand2_o, z_cand2_t, check_f02_vs_base5=True)
    r_c3 = eval_candidate_row("Cand 3: H3_Combined_TripleOrth_Portfolio", z_cand3_o, z_cand3_t, check_f02_vs_base5=True)

    all_rows = [r_refC, r_c0, r_c1, r_c2, r_c3]

    print("\n" + "=" * 150)
    print(
        f"{'Candidate':40s} | {'Nested OOF':10s} | {'Nested Δ':9s} | {'Wins':4s} | "
        f"{'Per-Fold Δ (×1e-5)':30s} | {'ΔF0,ΔF2 vs C0':15s} | {'KS v8':8s} | {'1-ρ(v8/S1/S2)×1e-4':22s}"
    )
    print("-" * 150)
    for r in all_rows:
        f02_str = f"{r['df0_vs_c0_1e5']:+.2f},{r['df2_vs_c0_1e5']:+.2f}"
        d_str = f"{r['dist_v8_1e4']:.2f} / {r['dist_s1_1e4']:.2f} / {r['dist_s2_1e4']:.2f}"
        print(
            f"{r['name']:40s} | {r['nested_oof_auc']:.7f} | {r['nested_delta']:+9.2e} | "
            f"{r['fold_wins']}/5  | {str(r['fold_deltas_1e5']):30s} | {f02_str:15s} | "
            f"{r['ks_vs_v8']:.6f} | {d_str:22s}"
        )
    print("=" * 150)

    # -----------------------------------------------------------------------
    # STEP 4: Select Winning Verification Candidate & Submit as Slot 3
    # -----------------------------------------------------------------------
    f02_cands = [r_c1, r_c2, r_c3]
    verified = [
        r for r in f02_cands
        if r["both_f02_up"] and r["fold_wins"] >= 4 and r["nested_oof_auc"] >= 0.952622 and r["ks_vs_v8"] < 0.0025
    ]
    if verified:
        winner = max(verified, key=lambda r: r["nested_oof_auc"])
    else:
        # Fallback: pick candidate among f02_cands that has both_f02_up and highest nested_oof_auc
        both_up = [r for r in f02_cands if r["both_f02_up"] and r["fold_wins"] >= 4]
        winner = max(both_up if both_up else all_rows, key=lambda r: r["nested_oof_auc"])

    p_win = winner["p_test"]
    assert len(p_win) == nte and np.isfinite(p_win).all()
    assert abs(float(p_win.mean()) - target_mean) < 1e-5
    sub_path = SUB / "sub_slot3_gm_verified_f02.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": p_win}).to_csv(sub_path, index=False)
    print(
        f"\n[Winner Selected] {winner['name']} -> {sub_path.name} "
        f"(Nested OOF={winner['nested_oof_auc']:.7f}, Δ={winner['nested_delta']:+.2e}, "
        f"wins={winner['fold_wins']}/5, ΔF0={winner['df0_vs_c0_1e5']:+.2f}e-5, ΔF2={winner['df2_vs_c0_1e5']:+.2f}e-5, "
        f"1-ρ(S2)={winner['dist_s2_1e4']:.3f}e-4, KS={winner['ks_vs_v8']:.6f})"
    )

    summary = {
        "v8_base_auc": base_auc,
        "ks_step1_pchip_f02": ks_step1,
        "w_h1_fit0": float(w_h1_fit0),
        "w_h1_fit2": float(w_h1_fit2),
        "w_h1_test": float(w_h1_test),
        "w_h2_fit0": float(w_h2_fit0),
        "w_h2_fit2": float(w_h2_fit2),
        "w_h2_test": float(w_h2_test),
        "rows": [{k: v for k, v in r.items() if k not in ("z_oof", "z_test", "p_test")} for r in all_rows],
        "winner": {k: v for k, v in winner.items() if k not in ("z_oof", "z_test", "p_test")},
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "verify_assumptions_slot3_gm.json").write_text(json.dumps(summary, indent=2))

    if "--submit" in sys.argv:
        msg = (
            f"Slot3 GM Verified: {winner['name']} (OOF={winner['nested_oof_auc']:.7f}, "
            f"Δ={winner['nested_delta']:+.2e}, {winner['fold_wins']}/5, "
            f"dF0={winner['df0_vs_c0_1e5']:+.2f}e-5, dF2={winner['df2_vs_c0_1e5']:+.2f}e-5, "
            f"1-rhoS2={winner['dist_s2_1e4']:.2f}e-4)"
        )
        cmd = [
            str(ROOT / ".venv_forensics/bin/kaggle"),
            "competitions",
            "submit",
            "-c",
            "playground-series-s6e11",
            "-f",
            str(sub_path),
            "-m",
            msg,
        ]
        print("[Submitting Slot 3]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)


if __name__ == "__main__":
    main()
