#!/usr/bin/env python3
"""Two-Stage 3-Fold (F0+F1+F2) Grandmaster Verification (`verify_slot4_gm_f012.py`).

Local Mac CPU only (n_jobs=4). Does NOT touch Pod PID 3021.
Uses Slot3-exact bayes_fold (logaddexp). Auto-submits Slot 4 only if strict gate passes.
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
ACTIVE = (0, 1, 2)
SHRINK = float(np.sqrt(3.0 / 5.0))
SLOT3_OOF = 0.9526269
GATE_OOF_MIN = 0.9526320  # Slot3 + 0.50e-5


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


def calibrate_logit_to_prevalence(z_test, z_oof_ref, y_train, target_mean=None):
    if target_mean is None:
        target_mean = float(np.mean(y_train))
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof_ref.reshape(-1, 1), y_train.astype(np.int32))
    z_lin = float(lr.coef_[0, 0]) * z_test + float(lr.intercept_[0])

    def obj(delta: float) -> float:
        return (float(np.mean(expit(z_lin + delta))) - target_mean) ** 2

    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded")
    return z_lin + float(res.x)


def bayes_fold(arr: np.ndarray, prior: float) -> np.ndarray:
    """Exact Slot3 K-order reduction (logaddexp)."""
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def bayes_std(arr: np.ndarray) -> np.ndarray:
    llr = arr[:, :, 1].astype(np.float64) - arr[:, :, 0].astype(np.float64)
    return llr.std(axis=1)


def align_disp(z: np.ndarray, ref: np.ndarray) -> np.ndarray:
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


def main() -> None:
    print("[Boot] verify_slot4_gm_f012.py (Slot3-exact bayes + F012 LOFO)...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    idx = {f: folds[f][1] for f in range(5)}
    a_tr = {f: folds[f][0] for f in range(5)}
    idx012 = np.concatenate([idx[f] for f in ACTIVE])

    # ---- refs ----
    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s3_sub = pd.read_csv(SUB / "sub_slot3_gm_verified_f02.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_v8_o = zrk(v8_oof)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_auc = float(roc_auc_score(y, z_v8_o))
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y, target_mean=0.016700))

    z_lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), z_lm12_o, n_knots=4000)
    z_g3_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_oof.npy").ravel())
    z_g3_t = pchip_quantile_map(zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_test.npy").ravel()), z_g3_o, n_knots=4000)
    z_masa_o = zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy").ravel())
    z_masa_t = pchip_quantile_map(
        zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy").ravel()),
        z_masa_o, n_knots=4000,
    )
    z_rev_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel())
    z_rev_t = pchip_quantile_map(zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy").ravel()), z_rev_o, n_knots=4000)

    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=4000)
    z_xgbf_cln_o = zrk(np.load(HARV / "oof_xgb_clean.npy").ravel())
    z_xgbf_cln_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_clean.npy").ravel()), z_xgbf_cln_o, n_knots=4000)
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)
    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)

    r_old_o, r_old_t = orth_vs_v8(z_xgbf_old_o, z_xgbf_old_t, z_v8_o, z_v8_t, folds)
    r_cln_o, r_cln_t = orth_vs_v8(z_xgbf_cln_o, z_xgbf_cln_t, z_v8_o, z_v8_t, folds)
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)

    # Cand0 Smooth-Gamma (exact Slot3 nested grid)
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
    grid_fold_deltas = np.zeros((len(param_grid), 5), dtype=np.float64)
    for gi, (gmin, gmax, temp, bias, w_cv) in enumerate(param_grid):
        gam = gmin + (gmax - gmin) * expit(-temp * s_norm_o - bias * th_o)
        z_try = z_v8_o + w_cv * (r_old_o - gam * r_tab_o)
        for fi, (_, b) in enumerate(folds):
            grid_fold_deltas[gi, fi] = float(roc_auc_score(y[b], z_try[b])) - base_folds[fi]

    z_cand0_o = np.zeros(ntr, dtype=np.float64)
    u_cv5_o = np.zeros(ntr, dtype=np.float64)
    z_cand0_t_f, u_cv5_t_f = [], []
    for f, (a, b) in enumerate(folds):
        other = [fi for fi in range(5) if fi != f]
        sub = grid_fold_deltas[:, other]
        scores = sub.mean(1) - 0.5 * np.maximum(0.0, -sub.min(1))
        gmin, gmax, temp, bias, w_cv = param_grid[int(np.argmax(scores))]
        gam_b = gmin + (gmax - gmin) * expit(-temp * s_norm_o[b] - bias * th_o[b])
        ub = r_old_o[b] - gam_b * r_tab_o[b]
        u_cv5_o[b] = ub
        z_cand0_o[b] = z_v8_o[b] + w_cv * ub
        gam_t = gmin + (gmax - gmin) * expit(-temp * s_norm_t - bias * th_t)
        ut = r_old_t - gam_t * r_tab_t
        u_cv5_t_f.append(ut)
        z_cand0_t_f.append(z_v8_t + w_cv * ut)
    z_cand0_t = np.mean(z_cand0_t_f, axis=0)
    u_cv5_t = np.mean(u_cv5_t_f, axis=0)
    # Slot3 trunk choice
    z_candC_o = z_v8_o + 0.025 * (r_old_o - r_tab_o)
    z_candC_t = z_v8_t + 0.025 * (r_old_t - r_tab_t)
    if roc_auc_score(y, z_cand0_o) >= roc_auc_score(y, z_candC_o):
        z_base5_o, z_base5_t, u_cv5_o, u_cv5_t = z_cand0_o, z_cand0_t, u_cv5_o, u_cv5_t
    else:
        z_base5_o, z_base5_t = z_candC_o, z_candC_t
        u_cv5_o, u_cv5_t = r_old_o - r_tab_o, r_old_t - r_tab_t
    cand0_auc = float(roc_auc_score(y, z_base5_o))
    cand0_folds = [float(roc_auc_score(y[b], z_base5_o[b])) for _, b in folds]
    print(f"[Cand0] Nested OOF={cand0_auc:.7f} (Δ vs v8={cand0_auc-base_auc:+.2e})", flush=True)

    lr_platt = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o.reshape(-1, 1), y)
    p_all_o = expit(lr_platt.decision_function(z_v8_o.reshape(-1, 1)))
    p_all_t = expit(lr_platt.decision_function(z_v8_t.reshape(-1, 1)))
    h_o = p_all_o * (1.0 - p_all_o)
    h_t = p_all_t * (1.0 - p_all_t)

    # ========================================================================
    # STEP 1: Artifacts + Slot3-exact bayes + lm12 dispersion align + PCHIP
    # ========================================================================
    print("\n[Step 1] Verify artifacts + Slot3-exact bayes_fold + lm12 align + PCHIP...", flush=True)
    raw_o, raw_t, s_raw_o, s_raw_t = {}, {}, {}, {}
    z_m_oof, z_m_tst, s_m_oof, s_m_tst = {}, {}, {}, {}
    z_m_zrk = {}  # Slot3-exact zrk(centered) for Ref reproduction
    for f in ACTIVE:
        op = PRED / f"gpt2_med_4ep/oof_{f}.npy"
        tp = PRED / f"gpt2_med_4ep/test_{f}.npy"
        oa, ta = np.load(op), np.load(tp)
        assert oa.shape == (len(idx[f]), 8, 2) and np.isfinite(oa).all(), oa.shape
        assert ta.shape == (nte, 4, 2) and np.isfinite(ta).all(), ta.shape
        prior = float(y[a_tr[f]].mean())
        raw_o[f] = bayes_fold(oa, prior)
        raw_t[f] = bayes_fold(ta, prior)
        s_raw_o[f] = bayes_std(oa)
        s_raw_t[f] = bayes_std(ta)
        # Slot3-exact representation (for Ref)
        z_m_zrk[f] = zrk(raw_o[f] - np.mean(raw_o[f]))
        # User-spec: align mean/std to lm12 on that fold / test
        z_m_oof[f] = align_disp(raw_o[f], z_lm12_o[idx[f]])
        z_m_tst[f] = align_disp(raw_t[f], z_lm12_t)
        s_m_oof[f] = (s_raw_o[f] - s_raw_o[f].mean()) / (s_raw_o[f].std() + 1e-12)
        s_m_tst[f] = (s_raw_t[f] - s_raw_t[f].mean()) / (s_raw_t[f].std() + 1e-12)
        print(
            f"  F{f}: oof{oa.shape} test{ta.shape} "
            f"align_auc={roc_auc_score(y[idx[f]], z_m_oof[f]):.6f} "
            f"zrk_auc={roc_auc_score(y[idx[f]], z_m_zrk[f]):.6f} "
            f"corr_lm12={np.corrcoef(z_m_oof[f], z_lm12_o[idx[f]])[0,1]:.5f}",
            flush=True,
        )

    z_m_oof_f012 = np.concatenate([z_m_oof[f] for f in ACTIVE])
    s_m_oof_f012 = np.concatenate([s_m_oof[f] for f in ACTIVE])
    z_m_test_avg = np.mean([z_m_tst[f] for f in ACTIVE], axis=0)
    s_m_test_avg = np.mean([s_m_tst[f] for f in ACTIVE], axis=0)
    z_m_test_f012 = pchip_quantile_map(z_m_test_avg, z_m_oof_f012, n_knots=4000)
    s_m_test_f012 = pchip_quantile_map(s_m_test_avg, s_m_oof_f012, n_knots=4000)
    ks_z = float(ks_2samp(z_m_oof_f012, z_m_test_f012).statistic)
    print(
        f"[Step 1] Post-avg PCHIP: std_oof={z_m_oof_f012.std():.6f} std_test={z_m_test_f012.std():.6f} "
        f"KS={ks_z:.6f} (target <0.0001)",
        flush=True,
    )
    if ks_z >= 0.0001:
        print(f"[WARN] KS={ks_z:.6f} >= 0.0001 — continuing", flush=True)

    # Slot3-exact F02 test map for Ref
    z_m_oof_f02_zrk = np.concatenate([z_m_zrk[0], z_m_zrk[2]])
    z_m_t0_zrk = zrk(raw_t[0] - np.mean(raw_o[0]))
    z_m_t2_zrk = zrk(raw_t[2] - np.mean(raw_o[2]))
    z_m_test_f02 = pchip_quantile_map(0.5 * (z_m_t0_zrk + z_m_t2_zrk), z_m_oof_f02_zrk, n_knots=4000)

    # ========================================================================
    # STEP 2: Single-fold diagnostics + LOFO candidates
    # ========================================================================
    print("\n[Step 2] Single-fold diagnostics + LOFO candidates...", flush=True)
    w_grid = np.linspace(-0.08, 0.08, 321)
    w_grid_s3 = np.linspace(-0.045, 0.045, 181)

    def opt_w(y_f, z_b, u_f, grid=w_grid, l2=0.015):
        base = float(roc_auc_score(y_f, z_b))
        return float(max(grid, key=lambda w: roc_auc_score(y_f, z_b + w * u_f) - base - l2 * (w**2)))

    def ols_u(z_m, X):
        ols = LinearRegression().fit(X, z_m)
        u = z_m - ols.predict(X)
        return (u - u.mean()) / (u.std() + 1e-12), ols

    print("\n[Single-Fold Diagnostic]", flush=True)
    for f in ACTIVE:
        X3 = np.column_stack([z_v8_o[idx[f]], z_lm12_o[idx[f]], u_cv5_o[idx[f]]])
        d_g3 = z_g3_o[idx[f]] - z_lm12_o[idx[f]]
        X4 = np.column_stack([z_v8_o[idx[f]], z_lm12_o[idx[f]], d_g3, u_cv5_o[idx[f]]])
        u3, _ = ols_u(z_m_oof[f], X3)
        u4, _ = ols_u(z_m_oof[f], X4)
        beta = float(np.cov(z_m_oof[f], z_lm12_o[idx[f]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[f]]))
        r_mlm = z_m_oof[f] - beta * z_lm12_o[idx[f]]
        g_b = np.tanh(np.abs(z_lm12_o[idx[f]] - z_v8_o[idx[f]])) * h_o[idx[f]]
        b_lm = g_b * r_mlm
        b_lm = (b_lm - b_lm.mean()) / (b_lm.std() + 1e-12)
        zb = z_base5_o[idx[f]]
        base = float(roc_auc_score(y[idx[f]], zb))
        for name, u in (("u_m3", u3), ("u_m4", u4), ("b_lm", b_lm)):
            w = opt_w(y[idx[f]], zb, u)
            d = float(roc_auc_score(y[idx[f]], zb + w * u) - base)
            print(f"  F{f} {name}: w={w:+.4f} Δ={d*1e5:+.2f}e-5", flush=True)
        # Also raw r_m_lm12 vs lm12
        w_r = opt_w(y[idx[f]], z_lm12_o[idx[f]], (r_mlm - r_mlm.mean()) / (r_mlm.std() + 1e-12))
        print(f"  F{f} r_m_lm12 vs lm12: w={w_r:+.4f}", flush=True)

    # ---- helpers: LOFO OLS residual ----
    def lofo_ols_u(z_by, X_by):
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

    def lofo_apply(u_by, z_base, l2=0.015, soft_sign=True):
        """LOFO: weight for hold = mean of per-fold opts on other folds."""
        fold_opt = {}
        for f in ACTIVE:
            uf = u_by[f]
            zb = z_base[idx[f]]
            fold_opt[f] = opt_w(y[idx[f]], zb, uf, l2=l2)
        z_out = z_base.copy()
        w_hold = {}
        for hold in ACTIVE:
            trf = [ff for ff in ACTIVE if ff != hold]
            w = float(np.mean([fold_opt[ff] for ff in trf]))
            if soft_sign and len(set(np.sign(fold_opt[ff]) for ff in trf)) > 1:
                w *= 0.25
            w_hold[hold] = w
            z_out[idx[hold]] = z_base[idx[hold]] + w * u_by[hold]
        return z_out, fold_opt, w_hold

    def predict_lofo_u_test(store, X_t, z_t, u_oof_by):
        us = []
        for hold in ACTIVE:
            ols, mu, sd = store[hold]
            us.append((z_t - ols.predict(X_t) - mu) / sd)
        return pchip_quantile_map(np.mean(us, axis=0), np.concatenate([u_oof_by[f] for f in ACTIVE]), n_knots=4000)

    # ========== Ref: Slot3 Champion F02 (exact) ==========
    # H1 on zrk z_m
    X3_0 = np.column_stack([z_v8_o[idx[0]], z_lm12_o[idx[0]], u_cv5_o[idx[0]]])
    X3_2 = np.column_stack([z_v8_o[idx[2]], z_lm12_o[idx[2]], u_cv5_o[idx[2]]])
    X3_t = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])

    def pairwise_dst_std(z_src, X_src, z_dst, X_dst):
        ols = LinearRegression().fit(X_src, z_src)
        u_raw = z_dst - ols.predict(X_dst)
        mu, sd = float(u_raw.mean()), float(u_raw.std() + 1e-12)
        return (u_raw - mu) / sd, (ols, mu, sd)

    u_h1_0, h1_s2 = pairwise_dst_std(z_m_zrk[2], X3_2, z_m_zrk[0], X3_0)
    u_h1_2, h1_s0 = pairwise_dst_std(z_m_zrk[0], X3_0, z_m_zrk[2], X3_2)
    w_h1_0 = opt_w(y[idx[0]], z_base5_o[idx[0]], u_h1_0, grid=w_grid_s3)
    w_h1_2 = opt_w(y[idx[2]], z_base5_o[idx[2]], u_h1_2, grid=w_grid_s3)
    z_s3_h1 = z_base5_o.copy()
    z_s3_h1[idx[0]] = z_base5_o[idx[0]] + w_h1_2 * u_h1_0
    z_s3_h1[idx[2]] = z_base5_o[idx[2]] + w_h1_0 * u_h1_2

    g_o = np.tanh(np.abs(z_lm12_o - z_v8_o)) * h_o
    g_t = np.tanh(np.abs(z_lm12_t - z_v8_t)) * h_t
    beta0 = float(np.cov(z_m_zrk[0], z_lm12_o[idx[0]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[0]]))
    beta2 = float(np.cov(z_m_zrk[2], z_lm12_o[idx[2]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[2]]))
    r_mlm0 = g_o[idx[0]] * (z_m_zrk[0] - beta2 * z_lm12_o[idx[0]])
    r_mlm2 = g_o[idx[2]] * (z_m_zrk[2] - beta0 * z_lm12_o[idx[2]])
    X2_0 = np.column_stack([z_v8_o[idx[0]], u_cv5_o[idx[0]]])
    X2_2 = np.column_stack([z_v8_o[idx[2]], u_cv5_o[idx[2]]])
    X2_t = np.column_stack([z_v8_t, u_cv5_t])
    ug0, h2_s2 = pairwise_dst_std(r_mlm2, X2_2, r_mlm0, X2_0)
    ug2, h2_s0 = pairwise_dst_std(r_mlm0, X2_0, r_mlm2, X2_2)
    w_h2_0 = opt_w(y[idx[0]], z_s3_h1[idx[0]], ug0, grid=w_grid_s3)
    w_h2_2 = opt_w(y[idx[2]], z_s3_h1[idx[2]], ug2, grid=w_grid_s3)
    z_slot3_o = z_s3_h1.copy()
    z_slot3_o[idx[0]] = z_s3_h1[idx[0]] + w_h2_2 * ug0
    z_slot3_o[idx[2]] = z_s3_h1[idx[2]] + w_h2_0 * ug2
    print(
        f"  Ref Slot3_F02: w_h1=({w_h1_0:+.4f},{w_h1_2:+.4f}) w_h2=({w_h2_0:+.4f},{w_h2_2:+.4f}) "
        f"OOF={roc_auc_score(y, z_slot3_o):.7f}",
        flush=True,
    )
    # Slot3 test
    ols_h1_2, mu_h1_2, sd_h1_2 = h1_s2
    ols_h1_0, mu_h1_0, sd_h1_0 = h1_s0
    u_h1_t = 0.5 * (
        (z_m_test_f02 - ols_h1_2.predict(X3_t) - mu_h1_2) / sd_h1_2
        + (z_m_test_f02 - ols_h1_0.predict(X3_t) - mu_h1_0) / sd_h1_0
    )
    u_h1_t = pchip_quantile_map(u_h1_t, np.concatenate([u_h1_0, u_h1_2]), n_knots=4000)
    z_slot3_t = z_base5_t + 0.5 * (w_h1_0 + w_h1_2) * u_h1_t
    feat_t_s3 = g_t * (z_m_test_f02 - 0.5 * (beta0 + beta2) * z_lm12_t)
    ols_h2_2, mu_h2_2, sd_h2_2 = h2_s2
    ols_h2_0, mu_h2_0, sd_h2_0 = h2_s0
    ug_t = 0.5 * (
        (feat_t_s3 - ols_h2_2.predict(X2_t) - mu_h2_2) / sd_h2_2
        + (feat_t_s3 - ols_h2_0.predict(X2_t) - mu_h2_0) / sd_h2_0
    )
    ug_t = pchip_quantile_map(ug_t, np.concatenate([ug0, ug2]), n_knots=4000)
    z_slot3_t = z_slot3_t + 0.5 * (w_h2_0 + w_h2_2) * ug_t

    # ========== Cand1: Slot3 H2 formula extended to F012 (LOFO) ==========
    # Build gated r_m_lm12 per fold with LOFO beta, then orth vs [v8, ucv5]
    r_mlm_by, g_bound_raw = {}, {}
    beta_store = {}
    for hold in ACTIVE:
        trf = [ff for ff in ACTIVE if ff != hold]
        ztr = np.concatenate([z_m_oof[ff] for ff in trf])
        lmtr = np.concatenate([z_lm12_o[idx[ff]] for ff in trf])
        beta = float(np.cov(ztr, lmtr, ddof=0)[0, 1] / np.var(lmtr))
        r = z_m_oof[hold] - beta * z_lm12_o[idx[hold]]
        rtr = ztr - beta * lmtr
        mu, sd = float(rtr.mean()), float(rtr.std() + 1e-12)
        r_mlm_by[hold] = (r - mu) / sd
        beta_store[hold] = (beta, mu, sd)
        g_bound_raw[hold] = (
            np.tanh(np.abs(z_lm12_o[idx[hold]] - z_v8_o[idx[hold]])) * h_o[idx[hold]] * r_mlm_by[hold]
        )

    def lofo_orth_v8_ucv(feat_by):
        out, store = {}, {}
        for hold in ACTIVE:
            trf = [ff for ff in ACTIVE if ff != hold]
            Xtr = np.vstack([np.column_stack([z_v8_o[idx[ff]], u_cv5_o[idx[ff]]]) for ff in trf])
            ytr = np.concatenate([feat_by[ff] for ff in trf])
            ols = LinearRegression().fit(Xtr, ytr)
            utr = ytr - ols.predict(Xtr)
            mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
            Xh = np.column_stack([z_v8_o[idx[hold]], u_cv5_o[idx[hold]]])
            uh = feat_by[hold] - ols.predict(Xh)
            out[hold] = (uh - mu) / sd
            store[hold] = (ols, mu, sd)
        return out, store

    g_bound_u, g_bound_store = lofo_orth_v8_ucv(g_bound_raw)
    z_cand1_o, fold_opt_c1, w_hold_c1 = lofo_apply(g_bound_u, z_base5_o)
    # Cand1 test
    beta_mean = float(np.mean([beta_store[h][0] for h in ACTIVE]))
    r_mlm_t = z_m_test_f012 - beta_mean * z_lm12_t
    feat_t_b = g_t * r_mlm_t  # abs gate
    feat_t_b = np.tanh(np.abs(z_lm12_t - z_v8_t)) * h_t * r_mlm_t
    us = []
    for hold in ACTIVE:
        ols, mu, sd = g_bound_store[hold]
        us.append((feat_t_b - ols.predict(X2_t) - mu) / sd)
    g_bound_t = pchip_quantile_map(np.mean(us, axis=0), np.concatenate([g_bound_u[f] for f in ACTIVE]), n_knots=4000)
    w1_test = SHRINK * float(np.mean(list(fold_opt_c1.values())))
    z_cand1_t = z_base5_t + w1_test * g_bound_t

    # ========== Cand2: QuadOrth [v8, lm12, g3-lm12, ucv5] + H2 g_bound ==========
    X4_by = {
        f: np.column_stack([
            z_v8_o[idx[f]], z_lm12_o[idx[f]],
            z_g3_o[idx[f]] - z_lm12_o[idx[f]], u_cv5_o[idx[f]],
        ])
        for f in ACTIVE
    }
    u_m4_lofo, ols4_store = lofo_ols_u(z_m_oof, X4_by)
    z_c2_h1, fold_opt_c2a, _ = lofo_apply(u_m4_lofo, z_base5_o)
    # Then H2 g_bound on top
    z_cand2_o, fold_opt_c2b, _ = lofo_apply(g_bound_u, z_c2_h1)
    X4_t = np.column_stack([z_v8_t, z_lm12_t, z_g3_t - z_lm12_t, u_cv5_t])
    u_m4_t = predict_lofo_u_test(ols4_store, X4_t, z_m_test_f012, u_m4_lofo)
    w2a = SHRINK * float(np.mean(list(fold_opt_c2a.values())))
    w2b = SHRINK * float(np.mean(list(fold_opt_c2b.values())))
    z_cand2_t = z_base5_t + w2a * u_m4_t + w2b * g_bound_t

    # ========== Cand3: RegimeAdaptive PosLM (non-neg / sign-consistent LOFO) ==========
    # z_lm_super diagnostic; gates on r_m_lm12
    g_dir_raw, g_cert_raw = {}, {}
    for f in ACTIVE:
        r = r_mlm_by[f]
        g_dir_raw[f] = np.tanh(z_lm12_o[idx[f]] - z_v8_o[idx[f]]) * r
        # expit(-z_norm(s_m)) * h * r
        sn = s_m_oof[f]
        g_cert_raw[f] = expit(-sn) * h_o[idx[f]] * r

    g_dir_u, g_dir_store = lofo_orth_v8_ucv(g_dir_raw)
    # orth g_dir vs [v8, lm12, ucv5] — user said against [z_v8, z_lm12, u_cv5]
    def lofo_orth_v8_lm_ucv(feat_by):
        out, store = {}, {}
        for hold in ACTIVE:
            trf = [ff for ff in ACTIVE if ff != hold]
            Xtr = np.vstack([
                np.column_stack([z_v8_o[idx[ff]], z_lm12_o[idx[ff]], u_cv5_o[idx[ff]]]) for ff in trf
            ])
            ytr = np.concatenate([feat_by[ff] for ff in trf])
            ols = LinearRegression().fit(Xtr, ytr)
            utr = ytr - ols.predict(Xtr)
            mu, sd = float(utr.mean()), float(utr.std() + 1e-12)
            Xh = np.column_stack([z_v8_o[idx[hold]], z_lm12_o[idx[hold]], u_cv5_o[idx[hold]]])
            out[hold] = ((feat_by[hold] - ols.predict(Xh)) - mu) / sd
            store[hold] = (ols, mu, sd)
        return out, store

    g_dir_u, g_dir_store = lofo_orth_v8_lm_ucv(g_dir_raw)
    g_cert_u, g_cert_store = lofo_orth_v8_lm_ucv(g_cert_raw)
    # Also re-orth g_bound vs full [v8,lm12,ucv5] for Cand3
    g_bound3_u, g_bound3_store = lofo_orth_v8_lm_ucv(g_bound_raw)

    def combine_nn_lofo(bases, z_base):
        """Greedy add bases with hard sign-consistency (same sign on all 3 folds) and all3↑."""
        z_cur = z_base.copy()
        used = []
        for name, u_by in bases:
            opts = []
            for f in ACTIVE:
                opts.append(opt_w(y[idx[f]], z_cur[idx[f]], u_by[f]))
            signs = [np.sign(w) for w in opts]
            if 0 in signs or len(set(signs)) > 1:
                # Prefer non-negative: if all positive-ish fail, try forcing w>=0 grid
                opts_pos = []
                for f in ACTIVE:
                    zb = z_cur[idx[f]]
                    base = float(roc_auc_score(y[idx[f]], zb))
                    grid_pos = np.linspace(0.0, 0.08, 161)
                    opts_pos.append(float(max(grid_pos, key=lambda w: roc_auc_score(y[idx[f]], zb + w * u_by[f]) - base - 0.015 * (w**2))))
                if any(w <= 1e-8 for w in opts_pos):
                    print(f"  Cand3 skip {name}: sign-inconsistent { [round(w,4) for w in opts] }", flush=True)
                    continue
                opts = opts_pos
            # LOFO apply
            z_try = z_cur.copy()
            for hold in ACTIVE:
                trf = [ff for ff in ACTIVE if ff != hold]
                # map fold index to opts position
                w = float(np.mean([opts[ACTIVE.index(ff)] for ff in trf]))
                z_try[idx[hold]] = z_cur[idx[hold]] + w * u_by[hold]
            ok = True
            deltas = []
            for f in ACTIVE:
                d = float(roc_auc_score(y[idx[f]], z_try[idx[f]])) - float(roc_auc_score(y[idx[f]], z_cur[idx[f]]))
                deltas.append(d)
                if float(roc_auc_score(y[idx[f]], z_try[idx[f]])) <= cand0_folds[f]:
                    ok = False
            if ok and all(d >= -1e-8 for d in deltas):
                z_cur = z_try
                used.append((name, [round(d * 1e5, 2) for d in deltas], [round(w, 4) for w in opts]))
                print(f"  Cand3 accept {name}: Δ={[round(d*1e5,2) for d in deltas]} w={[round(w,4) for w in opts]}", flush=True)
            else:
                print(f"  Cand3 reject {name}: Δ={[round(d*1e5,2) for d in deltas]} (need all3↑ vs Cand0)", flush=True)
        return z_cur, used

    # Also try path: Slot3 + F1-only PosLM (proven earlier to beat Slot3 OOF)
    z_cand3_o, used_c3 = combine_nn_lofo(
        [("g_bound", g_bound3_u), ("g_dir", g_dir_u), ("g_cert", g_cert_u)],
        z_base5_o,
    )
    # Path S3+F1only PosLM
    z_c3_f1 = z_slot3_o.copy()
    used_f1 = []
    for name, u_by in [("g_bound", g_bound3_u), ("g_dir", g_dir_u), ("g_cert", g_cert_u)]:
        # weights from F0+F2 on Cand0 base for sign stability
        w_cands = []
        for tf in (0, 2):
            w_cands.append(opt_w(y[idx[tf]], z_base5_o[idx[tf]], u_by[tf]))
        w1 = opt_w(y[idx[1]], z_base5_o[idx[1]], u_by[1])
        signs = [np.sign(w) for w in w_cands] + [np.sign(w1)]
        if len(set(s for s in signs if s != 0)) > 1:
            continue
        w_apply = float(np.mean(w_cands))
        z_try = z_c3_f1[idx[1]] + w_apply * u_by[1]
        d1 = float(roc_auc_score(y[idx[1]], z_try)) - cand0_folds[1]
        if d1 > 0:
            z_c3_f1[idx[1]] = z_try
            used_f1.append((name, w_apply, d1))
            print(f"  Cand3 F1only accept {name}: w={w_apply:+.4f} ΔF1={d1*1e5:+.2f}e-5", flush=True)

    def _all3(z):
        return all(float(roc_auc_score(y[idx[f]], z[idx[f]])) > cand0_folds[f] for f in ACTIVE)

    c3_paths = []
    if _all3(z_cand3_o):
        c3_paths.append(("C0+gates", z_cand3_o, used_c3))
    if _all3(z_c3_f1):
        c3_paths.append(("S3+F1only", z_c3_f1, used_f1))
    if not c3_paths:
        c3_paths.append(("C0+gates_soft", z_cand3_o, used_c3))
    best3 = max(c3_paths, key=lambda t: float(roc_auc_score(y, t[1])))
    z_cand3_o, used_c3, c3_path = best3[1], best3[2], best3[0]
    print(f"  Cand3 path={c3_path} OOF={roc_auc_score(y, z_cand3_o):.7f}", flush=True)

    # Cand3 test
    def feat_test_from_store(raw_t_feat, store, u_oof):
        us = []
        for hold in ACTIVE:
            ols, mu, sd = store[hold]
            # store OLS may be 2-col or 3-col
            n_feat = ols.coef_.shape[0]
            if n_feat == 2:
                Xt = X2_t
            else:
                Xt = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])
            us.append((raw_t_feat - ols.predict(Xt) - mu) / sd)
        return pchip_quantile_map(np.mean(us, axis=0), np.concatenate([u_oof[f] for f in ACTIVE]), n_knots=4000)

    g_dir_raw_t = np.tanh(z_lm12_t - z_v8_t) * r_mlm_t
    g_cert_raw_t = expit(-s_m_test_f012) * h_t * r_mlm_t
    g_bound3_t = feat_test_from_store(
        np.tanh(np.abs(z_lm12_t - z_v8_t)) * h_t * r_mlm_t, g_bound3_store, g_bound3_u
    )
    g_dir_t = feat_test_from_store(g_dir_raw_t, g_dir_store, g_dir_u)
    g_cert_t = feat_test_from_store(g_cert_raw_t, g_cert_store, g_cert_u)

    ut_map = {"g_bound": g_bound3_t, "g_dir": g_dir_t, "g_cert": g_cert_t}
    if c3_path.startswith("S3"):
        z_cand3_t = z_slot3_t.copy()
        for item in used_c3:
            name, w_apply = item[0], float(item[1])
            if name in ut_map:
                z_cand3_t = z_cand3_t + SHRINK * w_apply * ut_map[name]
    else:
        z_cand3_t = z_base5_t.copy()
        for item in used_c3:
            name, opts = item[0], item[2]
            if name in ut_map:
                z_cand3_t = z_cand3_t + SHRINK * float(np.mean(opts)) * ut_map[name]

    # ========================================================================
    # STEP 3: Table + Gate
    # ========================================================================
    print("\n[Step 3] Diagnostic table & Slot 4 gate...", flush=True)

    def eval_row(name, z_o, z_t):
        full = float(roc_auc_score(y, z_o))
        f_aucs = [float(roc_auc_score(y[b], z_o[b])) for _, b in folds]
        f_d = [fa - ba for fa, ba in zip(f_aucs, base_folds)]
        d_c0 = [float(roc_auc_score(y[idx[f]], z_o[idx[f]])) - cand0_folds[f] for f in ACTIVE]
        all3 = all(d > 0 for d in d_c0)
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_o, y, target_mean=0.016700))
        return {
            "name": name,
            "nested_oof_auc": full,
            "nested_delta_v8": full - base_auc,
            "delta_vs_slot3_oof": full - SLOT3_OOF,
            "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_d],
            "fold_wins": sum(1 for d in f_d if d > 0),
            "d_f012_vs_c0_1e5": [round(d * 1e5, 2) for d in d_c0],
            "all3_f012_up": all3,
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
        eval_row("Cand1_Slot3_Exact_F012", z_cand1_o, z_cand1_t),
        eval_row("Cand2_DeCorrupted_QuadOrth_H2_F012", z_cand2_o, z_cand2_t),
        eval_row("Cand3_RegimeAdaptive_LM_Upgrade_F012", z_cand3_o, z_cand3_t),
    ]

    print("\n" + "=" * 155)
    print("SLOT4 F012 GRANDMASTER TABLE")
    print("=" * 155)
    print(
        f"{'Candidate':42s} | {'NestedOOF':9s} | {'Δv8':9s} | {'ΔS3OOF':9s} | "
        f"{'Wins':4s} | {'PerFoldΔ×1e-5':28s} | {'ΔF012 vs C0':16s} | {'All3':5s} | "
        f"{'KS v8':7s} | {'1-ρS3×1e-4':10s}"
    )
    print("-" * 155)
    for r in rows:
        print(
            f"{r['name']:42s} | {r['nested_oof_auc']:.7f} | {r['nested_delta_v8']:+.2e} | "
            f"{r['delta_vs_slot3_oof']:+.2e} | {r['fold_wins']}/5 | {str(r['fold_deltas_1e5']):28s} | "
            f"{str(r['d_f012_vs_c0_1e5']):16s} | {str(r['all3_f012_up']):5s} | "
            f"{r['ks_vs_v8']:.5f} | {r['dist_s3_1e4']:10.3f}"
        )
    print("=" * 155)

    # Strict gate
    print("\n[Gate checks]", flush=True)
    qualified = []
    for r in rows:
        if r["name"] in ("v8_Champion", "Cand0_SmoothGamma_5F", "Ref_Slot3_Champion_F02"):
            continue
        a = r["nested_oof_auc"] >= GATE_OOF_MIN
        b = r["all3_f012_up"]
        c = r["dist_s3_1e4"] >= 0.35
        d = r["ks_vs_v8"] < 0.0025
        print(
            f"  {r['name']}: OOF>={GATE_OOF_MIN}={a} all3={b} "
            f"1-ρS3>={0.35}={c} KS<0.0025={d}",
            flush=True,
        )
        if a and b and c and d:
            qualified.append(r)

    sub_path = SUB / "sub_slot4_gm_verified_f012.csv"
    if qualified:
        winner = max(qualified, key=lambda r: r["nested_oof_auc"])
        pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
        print(
            f"\n[GATE PASS] {winner['name']} OOF={winner['nested_oof_auc']:.7f} "
            f"ΔS3={winner['delta_vs_slot3_oof']:+.2e} -> {sub_path.name} mean={winner['p_mean']:.6f}",
            flush=True,
        )
        msg = (
            f"Slot4 GM Verified F012: {winner['name']} "
            f"(OOF={winner['nested_oof_auc']:.7f}, ΔS3={winner['delta_vs_slot3_oof']:+.2e}, all3↑)"
        )
        cmd = [
            str(ROOT / ".venv_forensics/bin/kaggle"), "competitions", "submit",
            "-c", "playground-series-s6e11", "-f", str(sub_path), "-m", msg,
        ]
        print("[Submitting Slot 4]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)
        # live LB
        time.sleep(8)
        sub_list = subprocess.run(
            [str(ROOT / ".venv_forensics/bin/kaggle"), "competitions", "submissions",
             "-c", "playground-series-s6e11", "-v"],
            capture_output=True, text=True, check=False,
        )
        print(sub_list.stdout[:2000], flush=True)
        gate_status = "SUBMITTED"
    else:
        # stage best for inspection but HOLD
        pool = [r for r in rows if r["name"] not in ("v8_Champion", "Cand0_SmoothGamma_5F")]
        winner = max(pool, key=lambda r: (r["all3_f012_up"], r["nested_oof_auc"]))
        pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
        reasons = []
        for r in pool:
            if r["name"] == winner["name"]:
                if r["nested_oof_auc"] < GATE_OOF_MIN:
                    reasons.append(f"OOF {r['nested_oof_auc']:.7f} < {GATE_OOF_MIN} (need +0.50e-5 over Slot3)")
                if not r["all3_f012_up"]:
                    reasons.append(f"not all3↑ ΔF012={r['d_f012_vs_c0_1e5']}")
                if r["dist_s3_1e4"] < 0.35:
                    reasons.append(f"1-ρ(S3)={r['dist_s3_1e4']:.3f}e-4 < 0.35e-4")
                if r["ks_vs_v8"] >= 0.0025:
                    reasons.append(f"KS={r['ks_vs_v8']:.6f} >= 0.0025")
        print(
            f"\n[GATE HOLD] Best={winner['name']} OOF={winner['nested_oof_auc']:.7f} "
            f"all3={winner['all3_f012_up']} — NOT submitting. Reasons: {'; '.join(reasons) or 'no candidate cleared all gates'}",
            flush=True,
        )
        print("Preserving 4 remaining daily submissions. Wait for Fold3 oof_3.npy (~18:22 Local).", flush=True)
        gate_status = "HOLD"

    summary = {
        "cand0_auc": cand0_auc,
        "slot3_reconstruct_oof": float(roc_auc_score(y, z_slot3_o)),
        "ks_step1": ks_z,
        "c3_path": c3_path,
        "used_c3": str(used_c3),
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "winner": {k: v for k, v in winner.items() if k != "p_test"},
        "qualified_count": len(qualified),
        "gate_status": gate_status,
        "elapsed_sec": round(time.time() - T0, 1),
    }
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "verify_slot4_gm_f012.json").write_text(json.dumps(summary, indent=2))
    print(f"[{summary['elapsed_sec']}s] Saved {LOG / 'verify_slot4_gm_f012.json'}", flush=True)


if __name__ == "__main__":
    main()
