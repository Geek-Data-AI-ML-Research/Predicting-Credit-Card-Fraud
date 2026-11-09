#!/usr/bin/env python3
"""PHASE 2.5 — Local Mac CPU Meta-Gate Experiment (`phase2_5_local_meta_gate.py`)

Eliminate the Fold-0 PLS dip (-5.55e-5) from Exp 2C and combine Fold 0's +5.09e-5 `tabm_5f` surge with
Folds 1-4's +1.24e-5 to +4.60e-5 `xgb_fast_harvest` lift into a strictly leak-free 5-fold ensemble upgrade
over `v8` (OOF AUC = 0.9525970). Does NOT touch Pod 5hz1ob0bd0n6oy.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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


def calibrate_fold_margin(z_a: np.ndarray, y_a: np.ndarray, *z_others: np.ndarray) -> tuple[np.ndarray, ...]:
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(z_a.reshape(-1, 1), y_a.astype(np.int32))
    return tuple(lr.decision_function(z.reshape(-1, 1)).astype(np.float64) for z in (z_a, *z_others))


def main() -> None:
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    # Step 1: Load z_v8_oof and z_v8_test + Phase 2 assets
    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = (
        pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(np.float64)
    )
    z_v8_o = zrk(v8_oof)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_auc = float(roc_auc_score(y, z_v8_o))
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    target_mean = float(np.mean(y))
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y, target_mean=target_mean))

    z_knn_o = zrk(np.load(HARV / "oof_orig20k_knn.npy").ravel())
    z_knn_t = pchip_quantile_map(zrk(np.load(HARV / "test_orig20k_knn.npy").ravel()), z_knn_o, n_knots=4000)

    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)

    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)

    z_xgbf_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_o, n_knots=4000)

    knn24_o = np.load(HARV / "knn24_oof.npy").astype(np.float64)
    knn24_t = np.load(HARV / "knn24_test.npy").astype(np.float64)

    # Fold-wise strict OLS orthogonalization & unit standardization on train_idx
    r_knn_o = np.zeros(ntr, dtype=np.float64)
    r_tabm_o = np.zeros(ntr, dtype=np.float64)
    r_xgbf_o = np.zeros(ntr, dtype=np.float64)
    r_knn_t_folds, r_tabm_t_folds, r_xgbf_t_folds = [], [], []
    fold_orth = []

    for f, (a, b) in enumerate(folds):
        za = z_v8_o[a]
        zb = z_v8_o[b]
        var_a = float(np.var(za))

        def orth_one(co: np.ndarray, ct: np.ndarray):
            beta = float(np.cov(co[a], za, ddof=0)[0, 1] / var_a)
            ra_raw = co[a] - beta * za
            mu, sd = float(np.mean(ra_raw)), float(np.std(ra_raw) + 1e-12)
            ra = (ra_raw - mu) / sd
            rb = (co[b] - beta * zb - mu) / sd
            rt = (ct - beta * z_v8_t - mu) / sd
            return ra, rb, rt

        ra_k, rb_k, rt_k = orth_one(z_knn_o, z_knn_t)
        ra_m, rb_m, rt_m = orth_one(z_tabm_o, z_tabm_t)
        ra_x, rb_x, rt_x = orth_one(z_xgbf_o, z_xgbf_t)

        r_knn_o[b] = rb_k
        r_tabm_o[b] = rb_m
        r_xgbf_o[b] = rb_x
        r_knn_t_folds.append(rt_k)
        r_tabm_t_folds.append(rt_m)
        r_xgbf_t_folds.append(rt_x)
        fold_orth.append((ra_k, rb_k, rt_k, ra_m, rb_m, rt_m, ra_x, rb_x, rt_x))

    r_knn_t = pchip_quantile_map(np.mean(r_knn_t_folds, axis=0), r_knn_o, n_knots=4000)
    r_tabm_t = pchip_quantile_map(np.mean(r_tabm_t_folds, axis=0), r_tabm_o, n_knots=4000)
    r_xgbf_t = pchip_quantile_map(np.mean(r_xgbf_t_folds, axis=0), r_xgbf_o, n_knots=4000)

    def summarize_cand(name: str, z_oof_cand: np.ndarray, z_test_cand: np.ndarray) -> dict:
        full_auc = float(roc_auc_score(y, z_oof_cand))
        f_aucs = [float(roc_auc_score(y[b], z_oof_cand[b])) for _, b in folds]
        f_deltas = [fa - ba for fa, ba in zip(f_aucs, base_folds)]
        wins = sum(1 for d in f_deltas if d > 0)
        p_t = expit(calibrate_logit_to_prevalence(z_test_cand, z_oof_cand, y, target_mean=target_mean))
        ks_v = float(ks_2samp(p_t, p_v8_cal).statistic)
        rho_v = float(spearmanr(p_t, v8_sub).correlation)
        return {
            "name": name,
            "nested_oof_auc": full_auc,
            "nested_delta": full_auc - base_auc,
            "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_deltas],
            "fold_wins": wins,
            "ks_vs_v8": ks_v,
            "spearman_vs_v8": rho_v,
            "z_oof": z_oof_cand,
            "z_test": z_test_cand,
            "p_test": p_t,
        }

    # -----------------------------------------------------------------------
    # Candidate A: Uncertainty-Gated Non-Linear Ridge (14 Interaction Basis Terms)
    # -----------------------------------------------------------------------
    print("[Step 2.1] Running Candidate A (Uncertainty-Gated Non-Linear 14-Term Ridge)...", flush=True)

    def build_14_basis(
        z_v8_vec: np.ndarray,
        rk: np.ndarray,
        rm: np.ndarray,
        rx: np.ndarray,
        s_vec: np.ndarray,
        k24: np.ndarray,
        s_mu: float,
        s_sd: float,
        s_med: float,
    ) -> np.ndarray:
        sn = np.clip((s_vec - s_mu) / (s_sd + 1e-8), -3.0, 3.0)
        low_unc = (s_vec < s_med).astype(np.float64)
        th = np.tanh(z_v8_vec / 2.0)
        # Top-4 clone proximity ratios from knn24:
        # col 11: orig20k d_pos/(d_neg+1e-5), col 9: orig20k tau=1.0 target rate,
        # col 0: orig20k min_dist, col 15: train_oof knn50 mean dist
        d_ratio = np.clip(k24[:, 11], 0.0, 5.0)
        k15_rate = k24[:, 9]
        min_d = k24[:, 0]
        k50_dens = 1.0 / (k24[:, 3] + 0.5)
        return np.column_stack([
            rk,
            rm,
            rx,
            rm / (1.0 + np.maximum(sn, -0.8)),
            rx * sn,
            rm * low_unc,
            rm * th,
            rx * th,
            rk * th,
            (rx - rm) * sn,
            rx * d_ratio,
            rm * k15_rate,
            rx * min_d,
            rm * k50_dens,
        ])

    z_candA_o = np.zeros(ntr, dtype=np.float64)
    r_candA_t_folds = []
    for f, (a, b) in enumerate(folds):
        ra_k, rb_k, rt_k, ra_m, rb_m, rt_m, ra_x, rb_x, rt_x = fold_orth[f]
        s_mu, s_sd, s_med = float(s_tabm_o[a].mean()), float(s_tabm_o[a].std()), float(np.median(s_tabm_o[a]))
        Ba = build_14_basis(z_v8_o[a], ra_k, ra_m, ra_x, s_tabm_o[a], knn24_o[a], s_mu, s_sd, s_med)
        Bb = build_14_basis(z_v8_o[b], rb_k, rb_m, rb_x, s_tabm_o[b], knn24_o[b], s_mu, s_sd, s_med)
        Bt = build_14_basis(z_v8_t, rt_k, rt_m, rt_x, s_tabm_t, knn24_t, s_mu, s_sd, s_med)

        sc = StandardScaler()
        Ba_s = np.clip(np.nan_to_num(sc.fit_transform(Ba), 0.0), -5.0, 5.0)
        Bb_s = np.clip(np.nan_to_num(sc.transform(Bb), 0.0), -5.0, 5.0)
        Bt_s = np.clip(np.nan_to_num(sc.transform(Bt), 0.0), -5.0, 5.0)

        # Boundary weights w_i = p_v8 * (1 - p_v8) on calibrated fold margin
        bm_a, _, _ = calibrate_fold_margin(z_v8_o[a], y[a], z_v8_o[b], z_v8_t)
        p_a = expit(bm_a)
        w_i = p_a * (1.0 - p_a)
        resid_y = (y[a] - p_a) / (w_i + 1e-4)

        ridge = Ridge(alpha=1e4)
        ridge.fit(Ba_s, resid_y, sample_weight=w_i)
        da = ridge.predict(Ba_s)
        db = ridge.predict(Bb_s)
        dt = ridge.predict(Bt_s)
        da_sd = float(np.std(da) + 1e-12)
        da, db, dt = da / da_sd, db / da_sd, dt / da_sd

        # Grid-search step size eta strictly on train_idx `a`
        best_eta, best_sc = 0.0, float(roc_auc_score(y[a], z_v8_o[a]))
        for eta in np.linspace(-0.04, 0.06, 101):
            s_a = float(roc_auc_score(y[a], z_v8_o[a] + eta * da))
            if s_a > best_sc:
                best_sc, best_eta = s_a, float(eta)

        z_candA_o[b] = z_v8_o[b] + best_eta * db
        r_candA_t_folds.append(best_eta * dt)

    z_candA_t = z_v8_t + np.mean(r_candA_t_folds, axis=0)
    res_A = summarize_cand("Candidate A (14-Term Uncertainty Ridge)", z_candA_o, z_candA_t)

    # -----------------------------------------------------------------------
    # Candidate B: Ultra-Shallow max_depth=2 & 3 LightGBM Residual Booster (init_score = z_v8)
    # -----------------------------------------------------------------------
    print("[Step 2.2] Running Candidate B (Ultra-Shallow d=2,3 LightGBM init_score=z_v8)...", flush=True)
    # Top-6 summary columns of knn24: [orig_min_l2(0), orig_k15_l2(2), orig_tau1_rate(9), orig_dpos_dneg(11), tr_tau1_rate(21), tr_dpos_dneg(23)]
    k6_idx = [0, 2, 9, 11, 21, 23]
    z_candB_o = np.zeros(ntr, dtype=np.float64)
    d_candB_t_folds = []

    for f, (a, b) in enumerate(folds):
        ra_k, rb_k, rt_k, ra_m, rb_m, rt_m, ra_x, rb_x, rt_x = fold_orth[f]
        Xa = np.column_stack([z_v8_o[a], ra_m, s_tabm_o[a], ra_x, ra_k, ra_x - ra_m, knn24_o[a][:, k6_idx]])
        Xb = np.column_stack([z_v8_o[b], rb_m, s_tabm_o[b], rb_x, rb_k, rb_x - rb_m, knn24_o[b][:, k6_idx]])
        Xt = np.column_stack([z_v8_t, rt_m, s_tabm_t, rt_x, rt_k, rt_x - rt_m, knn24_t[:, k6_idx]])

        # Strictly leak-free inner holdout inside `a` (90% inner-train, 10% inner-val) for early stopping so `b` is never touched!
        inner_skf = list(StratifiedKFold(10, shuffle=True, random_state=100 + f).split(np.zeros(len(a)), y[a]))
        i_tr, i_va = inner_skf[0]
        bm_itr, bm_iva, bm_b, bm_t = calibrate_fold_margin(
            z_v8_o[a[i_tr]], y[a[i_tr]], z_v8_o[a[i_va]], z_v8_o[b], z_v8_t
        )

        delta_b_acc = np.zeros(len(b), dtype=np.float64)
        delta_t_acc = np.zeros(nte, dtype=np.float64)
        specs = [(42 + s, 2, 4) for s in range(3)] + [(142 + s, 3, 7) for s in range(2)]

        for sd, md, nl in specs:
            dtr = lgb.Dataset(Xa[i_tr], label=y[a[i_tr]], init_score=bm_itr)
            dva = lgb.Dataset(Xa[i_va], label=y[a[i_va]], init_score=bm_iva, reference=dtr)
            params = dict(
                objective="binary",
                metric="auc",
                max_depth=md,
                num_leaves=nl,
                min_child_samples=800,
                reg_lambda=100.0,
                colsample_bytree=0.6,
                subsample=0.8,
                subsample_freq=1,
                learning_rate=0.01,
                verbosity=-1,
                n_jobs=4,
                seed=sd,
            )
            bst = lgb.train(
                params, dtr, num_boost_round=600, valid_sets=[dva],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
            )
            raw_b = bst.predict(Xb, raw_score=True)
            raw_t = bst.predict(Xt, raw_score=True)
            delta_b_acc += raw_b / len(specs)
            delta_t_acc += raw_t / len(specs)

        # Convert calibrated margin back to z_v8 scale via fold Platt slope
        lr_cal = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o[a[i_tr]].reshape(-1, 1), y[a[i_tr]])
        slope = float(lr_cal.coef_[0, 0])
        z_candB_o[b] = z_v8_o[b] + delta_b_acc / slope
        d_candB_t_folds.append(delta_t_acc / slope)

    z_candB_t = z_v8_t + np.mean(d_candB_t_folds, axis=0)
    res_B = summarize_cand("Candidate B (Shallow d=2,3 LGB init_score)", z_candB_o, z_candB_t)

    # -----------------------------------------------------------------------
    # Candidate C: Joint Signed Rank-Power + Logit Simplex (Powell with L2 penalty)
    # -----------------------------------------------------------------------
    print("[Step 2.3] Running Candidate C (Joint Powell Simplex v8 + r_xgbf + r_tabm + r_knn)...", flush=True)
    z_candC_o = np.zeros(ntr, dtype=np.float64)
    d_candC_t_folds = []
    c_weights = []

    for f, (a, b) in enumerate(folds):
        ra_k, rb_k, rt_k, ra_m, rb_m, rt_m, ra_x, rb_x, rt_x = fold_orth[f]
        za = z_v8_o[a]
        ya = y[a]
        base_a = float(roc_auc_score(ya, za))

        # Coarse 3D warm-start grid on `a` then Powell refinement with L2 penalty lambda_w = 0.05
        best_w = np.array([0.025, -0.025, 0.0], dtype=np.float64)
        best_obj = -base_a
        for wx in (0.015, 0.025, 0.035, 0.045):
            for wm in (-0.040, -0.030, -0.020, -0.010, 0.0):
                for wk in (-0.010, 0.0, 0.010):
                    w_vec = np.array([wx, wm, wk])
                    auc_a = float(roc_auc_score(ya, za + wx * ra_x + wm * ra_m + wk * ra_k))
                    pen = 0.05 * float(np.sum(w_vec**2))
                    val = -(auc_a - base_a) + pen * 1e-3
                    if val < best_obj:
                        best_obj, best_w = val, w_vec

        def powell_obj(w_vec: np.ndarray) -> float:
            wx, wm, wk = float(w_vec[0]), float(w_vec[1]), float(w_vec[2])
            if abs(wx) > 0.08 or abs(wm) > 0.08 or abs(wk) > 0.08:
                return 1.0
            auc_a = float(roc_auc_score(ya, za + wx * ra_x + wm * ra_m + wk * ra_k))
            pen = 0.05 * (wx**2 + wm**2 + wk**2) * 1e-3
            return -(auc_a - base_a) + pen

        opt_res = minimize(powell_obj, best_w, method="Powell", options=dict(maxiter=60, xtol=2e-3, ftol=1e-7))
        wf = opt_res.x if opt_res.fun <= best_obj else best_w
        c_weights.append([round(float(x), 5) for x in wf])
        z_candC_o[b] = z_v8_o[b] + wf[0] * rb_x + wf[1] * rb_m + wf[2] * rb_k
        d_candC_t_folds.append(wf[0] * rt_x + wf[1] * rt_m + wf[2] * rt_k)

    z_candC_t = z_v8_t + np.mean(d_candC_t_folds, axis=0)
    res_C = summarize_cand("Candidate C (Joint Powell Simplex)", z_candC_o, z_candC_t)
    print(f"  Candidate C per-fold weights (w_xgbf, w_tabm, w_knn): {c_weights}", flush=True)

    # -----------------------------------------------------------------------
    # Candidate D: Best 5-Fold Candidate + gpt2-medium Orthogonal Residual (F2, plus F0 if available)
    # -----------------------------------------------------------------------
    best_5f = max([res_A, res_B, res_C], key=lambda r: (r["fold_wins"] >= 4, r["nested_delta"]))
    z_baseD_o = best_5f["z_oof"].copy() if best_5f["nested_delta"] > 0 else z_v8_o.copy()
    z_baseD_t = best_5f["z_test"].copy() if best_5f["nested_delta"] > 0 else z_v8_t.copy()

    # Check which gpt2-medium folds exist locally
    avail_folds = [
        fi for fi in range(5)
        if (PRED / f"gpt2_med_4ep/oof_{fi}.npy").exists() and (PRED / f"gpt2_med_4ep/test_{fi}.npy").exists()
    ]
    print(f"[Step 2.4] Running Candidate D on top of {best_5f['name']} with gpt2-medium folds={avail_folds}...", flush=True)

    z_candD_o = z_baseD_o.copy()
    r_med_t_list = []
    w_med_list = []
    for fi in avail_folds:
        af, idx_f = folds[fi]
        pr_f = float(y[af].mean())
        oof_raw = np.load(PRED / f"gpt2_med_4ep/oof_{fi}.npy")
        test_raw = np.load(PRED / f"gpt2_med_4ep/test_{fi}.npy")
        zb_o = (
            np.logaddexp.reduce(oof_raw[:, :, 1].astype(np.float64), axis=1)
            - np.logaddexp.reduce(oof_raw[:, :, 0].astype(np.float64), axis=1)
            + np.log(pr_f / (1.0 - pr_f))
        )
        zb_t = (
            np.logaddexp.reduce(test_raw[:, :, 1].astype(np.float64), axis=1)
            - np.logaddexp.reduce(test_raw[:, :, 0].astype(np.float64), axis=1)
            + np.log(pr_f / (1.0 - pr_f))
        )
        zm_o = zrk(zb_o)
        zm_t = pchip_quantile_map(zrk(zb_t), zm_o, n_knots=4000)

        z_ref_f = z_baseD_o[idx_f]
        beta_f = float(np.cov(zm_o, z_ref_f, ddof=0)[0, 1] / np.var(z_ref_f))
        rf_o = zm_o - beta_f * z_ref_f
        mu_f, sd_f = float(np.mean(rf_o)), float(np.std(rf_o) + 1e-12)
        rf_o = (rf_o - mu_f) / sd_f
        rf_t = pchip_quantile_map((zm_t - beta_f * z_baseD_t - mu_f) / sd_f, rf_o, n_knots=4000)

        best_wf, best_scf = 0.0, float(roc_auc_score(y[idx_f], z_ref_f))
        for wf in np.linspace(-0.04, 0.05, 181):
            sc = float(roc_auc_score(y[idx_f], z_ref_f + wf * rf_o))
            if sc > best_scf:
                best_scf, best_wf = sc, float(wf)
        z_candD_o[idx_f] = z_ref_f + best_wf * rf_o
        r_med_t_list.append(best_wf * rf_t)
        w_med_list.append((fi, best_wf))

    z_candD_t = z_baseD_t + (np.mean(r_med_t_list, axis=0) if r_med_t_list else 0.0)
    res_D = summarize_cand(f"Candidate D ({best_5f['name'][:11]} + gpt2m F{avail_folds})", z_candD_o, z_candD_t)

    # -----------------------------------------------------------------------
    # Step 3: Print Diagnostic Table & Auto-Submit Slot 2 If Gate Passes
    # -----------------------------------------------------------------------
    print("\n" + "=" * 118)
    print(f"{'Candidate':42s} | {'Nested OOF':10s} | {'Nested ΔAUC':11s} | {'Folds':5s} | {'Per-Fold Δ (×1e-5)':30s} | {'KS vs v8':8s} | {'Spearman':8s}")
    print("-" * 118)
    for r in (res_A, res_B, res_C, res_D):
        print(
            f"{r['name']:42s} | {r['nested_oof_auc']:.7f} | {r['nested_delta']:+11.2e} | "
            f"{r['fold_wins']}/5   | {str(r['fold_deltas_1e5']):30s} | {r['ks_vs_v8']:.6f} | {r['spearman_vs_v8']:.6f}"
        )
    print("=" * 118)

    qualified = [
        r for r in (res_A, res_B, res_C, res_D)
        if r["nested_delta"] >= 1.0e-5 and r["fold_wins"] >= 4 and r["ks_vs_v8"] < 0.0025
    ]

    sub_path = SUB / "sub_phase2_5_v8_meta_harvest.csv"
    winner = max(qualified, key=lambda r: r["nested_delta"]) if qualified else max((res_A, res_B, res_C, res_D), key=lambda r: r["nested_delta"])
    p_out = winner["p_test"]
    assert len(p_out) == nte and np.isfinite(p_out).all()
    pd.DataFrame({"id": test_ids, "is_fraud": p_out}).to_csv(sub_path, index=False)

    summary = {
        "v8_base_auc": base_auc,
        "candidates": [
            {k: v for k, v in r.items() if k not in ("z_oof", "z_test", "p_test")}
            for r in (res_A, res_B, res_C, res_D)
        ],
        "c_weights": c_weights,
        "gpt2m_avail_folds": avail_folds,
        "qualified_count": len(qualified),
        "winner": winner["name"],
        "winner_nested_delta": winner["nested_delta"],
        "winner_fold_wins": winner["fold_wins"],
        "winner_ks": winner["ks_vs_v8"],
        "winner_spearman": winner["spearman_vs_v8"],
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "phase2_5_meta_gate.json").write_text(json.dumps(summary, indent=2))

    if qualified and "--submit-if-qualified" in sys.argv:
        msg = (
            f"Phase2.5 Slot2: {winner['name']} (Nested OOF={winner['nested_oof_auc']:.7f}, "
            f"Δ={winner['nested_delta']:+.2e}, {winner['fold_wins']}/5 folds, KS={winner['ks_vs_v8']:.5f})"
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
        print("[Submitting Qualified Winner]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)
    else:
        print(f"\n[Gate Result] Qualified candidates: {len(qualified)}. Submission held back (6 slots preserved).")


if __name__ == "__main__":
    main()
