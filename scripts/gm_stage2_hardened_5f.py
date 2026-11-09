#!/usr/bin/env python3
"""DS-Hardened Pure 5-Fold Residual-Boosted & Control-Variate Pipeline (`gm_stage2_hardened_5f.py`)

Hard Grandmaster Rules:
1. ZERO PARTIAL-FOLD CONTAMINATION: Strictly exclude 1-fold/2-fold `gpt2-medium` from all features and blends.
2. ZERO VALIDATION PEEKING: Fixed, heavily regularized tree budget (`n_estimators=80`, `lr=0.005`) bagged across
   3 seeds on `train_idx` (and an internal 4-fold out-of-fold inside `train_idx` so Level-3 portfolio weights
   never see in-sample tree predictions).
3. ZERO TREE-ON-TREE STACKING OVERFIT: Do NOT feed `r_xgbf` into the new residual trees; train residual boosters
   purely on the 36 non-tree orthogonal signals (`TabM`, `knn24`, and the 4 complete 5-fold Sequence LMs), and
   combine with `d_xgbf_tabm = r_xgbf - r_tabm` at the ensemble stage.
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
import xgboost as xgb
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
N_JOBS = 4


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


def calibrate_fold_platt(z_a: np.ndarray, y_a: np.ndarray, *z_others: np.ndarray):
    """Fit 1D Platt scaling on fold train_idx `a` so mean(expit(bm_a)) == mean(y_a)."""
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(z_a.reshape(-1, 1), y_a.astype(np.int32))
    slope = float(lr.coef_[0, 0])
    outs = [lr.decision_function(z.reshape(-1, 1)).astype(np.float64) for z in (z_a, *z_others)]
    return slope, outs


def signed_geom_mean(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    prod = u * v
    return np.sign(prod) * np.sqrt(np.abs(prod))


def build_fold_features_36(
    a: np.ndarray,
    b: np.ndarray,
    y_a: np.ndarray,
    z_v8_o: np.ndarray,
    z_v8_t: np.ndarray,
    z_tabm_o: np.ndarray,
    z_tabm_t: np.ndarray,
    s_tabm_o: np.ndarray,
    s_tabm_t: np.ndarray,
    z_knn_o: np.ndarray,
    z_knn_t: np.ndarray,
    z_xgbf_o: np.ndarray,
    z_xgbf_t: np.ndarray,
    lms_o: list[np.ndarray],
    lms_t: list[np.ndarray],
    knn18_o: np.ndarray,
    knn18_t: np.ndarray,
):
    """Step 1: Construct the exact 36 non-tree features + control variate strictly from `train_idx` (`a`)."""
    za = z_v8_o[a]
    zb = z_v8_o[b]
    zt = z_v8_t
    var_a = float(np.var(za))

    def orth_unit(co: np.ndarray, ct: np.ndarray):
        beta = float(np.cov(co[a], za, ddof=0)[0, 1] / var_a)
        ra_raw = co[a] - beta * za
        mu = float(np.mean(ra_raw))
        sd = float(np.std(ra_raw) + 1e-12)
        return (
            (ra_raw - mu) / sd,
            (co[b] - beta * zb - mu) / sd,
            (ct - beta * zt - mu) / sd,
        )

    # 1. Anchor Regime Features (2 cols): z_v8, h_v8 = p_v8 * (1 - p_v8)
    slope, (bm_a, bm_b, bm_t) = calibrate_fold_platt(za, y_a, zb, zt)
    pa, pb, pt = expit(bm_a), expit(bm_b), expit(bm_t)
    ha, hb, ht = pa * (1.0 - pa), pb * (1.0 - pb), pt * (1.0 - pt)

    # 2. Neural TabM & Epistemic Uncertainty Signals (5 cols)
    ra_m, rb_m, rt_m = orth_unit(z_tabm_o, z_tabm_t)
    s_mu, s_sd = float(np.mean(s_tabm_o[a])), float(np.std(s_tabm_o[a]) + 1e-12)
    sa_raw = np.maximum(s_tabm_o[a], 0.0)
    sb_raw = np.maximum(s_tabm_o[b], 0.0)
    st_raw = np.maximum(s_tabm_t, 0.0)
    sa = (s_tabm_o[a] - s_mu) / s_sd
    sb = (s_tabm_o[b] - s_mu) / s_sd
    st = (s_tabm_t - s_mu) / s_sd

    tabm_a = np.column_stack([ra_m, sa, ra_m / (0.05 + sa_raw), ra_m * np.tanh(za / 2.0), sa * za])
    tabm_b = np.column_stack([rb_m, sb, rb_m / (0.05 + sb_raw), rb_m * np.tanh(zb / 2.0), sb * zb])
    tabm_t = np.column_stack([rt_m, st, rt_m / (0.05 + st_raw), rt_m * np.tanh(zt / 2.0), st * zt])

    # 3. Sequence LM Multi-View Consensus Signals (7 cols): r_lm12, r_masa, r_gpt2_3ep, r_gpt2_rev + mean, std, prec
    r_lms_a, r_lms_b, r_lms_t = [], [], []
    for lo, lt in zip(lms_o, lms_t):
        la, lb, lt_r = orth_unit(lo, lt)
        r_lms_a.append(la)
        r_lms_b.append(lb)
        r_lms_t.append(lt_r)
    L_a = np.column_stack(r_lms_a)
    L_b = np.column_stack(r_lms_b)
    L_t = np.column_stack(r_lms_t)

    lm_mean_a, lm_std_a = L_a.mean(axis=1), L_a.std(axis=1)
    lm_mean_b, lm_std_b = L_b.mean(axis=1), L_b.std(axis=1)
    lm_mean_t, lm_std_t = L_t.mean(axis=1), L_t.std(axis=1)

    lm_a = np.column_stack([L_a, lm_mean_a, lm_std_a, lm_mean_a / (0.05 + lm_std_a)])
    lm_b = np.column_stack([L_b, lm_mean_b, lm_std_b, lm_mean_b / (0.05 + lm_std_b)])
    lm_t = np.column_stack([L_t, lm_mean_t, lm_std_t, lm_mean_t / (0.05 + lm_std_t)])

    # 4. Geometric orig20k & OOF Clone Retrieval (r_knn + 18 cols = 19 cols)
    ra_k, rb_k, rt_k = orth_unit(z_knn_o, z_knn_t)
    sc_k = StandardScaler()
    k18_a = np.clip(np.nan_to_num(sc_k.fit_transform(knn18_o[a]), 0.0), -6.0, 6.0)
    k18_b = np.clip(np.nan_to_num(sc_k.transform(knn18_o[b]), 0.0), -6.0, 6.0)
    k18_t = np.clip(np.nan_to_num(sc_k.transform(knn18_t), 0.0), -6.0, 6.0)
    geo_a = np.column_stack([ra_k, k18_a])
    geo_b = np.column_stack([rb_k, k18_b])
    geo_t = np.column_stack([rt_k, k18_t])

    # 5. Cross-Paradigm Agreement Terms (3 cols)
    ag_a = np.column_stack([
        signed_geom_mean(lm_mean_a, ra_m),
        signed_geom_mean(lm_mean_a, ra_k),
        signed_geom_mean(ra_m, ra_k),
    ])
    ag_b = np.column_stack([
        signed_geom_mean(lm_mean_b, rb_m),
        signed_geom_mean(lm_mean_b, rb_k),
        signed_geom_mean(rb_m, rb_k),
    ])
    ag_t = np.column_stack([
        signed_geom_mean(lm_mean_t, rt_m),
        signed_geom_mean(lm_mean_t, rt_k),
        signed_geom_mean(rt_m, rt_k),
    ])

    X_a = np.column_stack([za, ha, tabm_a, lm_a, geo_a, ag_a]).astype(np.float32)
    X_b = np.column_stack([zb, hb, tabm_b, lm_b, geo_b, ag_b]).astype(np.float32)
    X_t = np.column_stack([zt, ht, tabm_t, lm_t, geo_t, ag_t]).astype(np.float32)
    assert X_a.shape[1] == 36, f"Expected 36 features, got {X_a.shape[1]}"

    # Also compute r_xgbf ONLY for the Stage-3 control variate d_xgbf_tabm = r_xgbf - r_tabm (NOT in X_a/X_b/X_t!)
    ra_x, rb_x, rt_x = orth_unit(z_xgbf_o, z_xgbf_t)
    d_xt_a = ra_x - ra_m
    d_xt_b = rb_x - rb_m
    d_xt_t = rt_x - rt_m

    aux = {
        "slope": slope,
        "bm_a": bm_a,
        "bm_b": bm_b,
        "bm_t": bm_t,
        "pa": pa,
        "pb": pb,
        "pt": pt,
        "ha": ha,
        "hb": hb,
        "ht": ht,
        "ra_m": ra_m,
        "rb_m": rb_m,
        "rt_m": rt_m,
        "lm_mean_a": lm_mean_a,
        "lm_mean_b": lm_mean_b,
        "lm_mean_t": lm_mean_t,
        "ra_k": ra_k,
        "rb_k": rb_k,
        "rt_k": rt_k,
        "sa": sa,
        "sb": sb,
        "st": st,
        "d_xt_a": d_xt_a,
        "d_xt_b": d_xt_b,
        "d_xt_t": d_xt_t,
        "agree_lm_tabm_a": ag_a[:, 0],
        "agree_lm_tabm_b": ag_b[:, 0],
        "agree_lm_tabm_t": ag_t[:, 0],
    }
    return X_a, X_b, X_t, aux


def fit_predict_hist_lgb_3seed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    bm_tr: np.ndarray,
    slope: float,
    X_evals: list[np.ndarray],
    seeds: tuple[int, ...] = (42, 142, 242),
) -> list[np.ndarray]:
    """Model 1: HistLGB_HessClipped (3-seed bag, max_bin=63, fixed 80 trees, zero val_idx peeking)."""
    outs = [np.zeros(len(Xe), dtype=np.float64) for Xe in X_evals]
    for sd in seeds:
        dtr = lgb.Dataset(X_tr, label=y_tr, init_score=bm_tr)
        params = dict(
            objective="binary",
            max_depth=3,
            num_leaves=6,
            min_child_samples=1000,
            min_sum_hessian_in_leaf=50.0,
            max_delta_step=0.15,
            colsample_bytree=0.5,
            subsample=0.7,
            subsample_freq=1,
            reg_alpha=25.0,
            reg_lambda=150.0,
            learning_rate=0.005,
            max_bin=63,
            n_jobs=N_JOBS,
            verbosity=-1,
            seed=sd,
        )
        bst = lgb.train(params, dtr, num_boost_round=80)
        for i, Xe in enumerate(X_evals):
            # raw_score=True returns pure tree residual margin (z_pred - bm); divide by Platt slope to map to z_v8 scale
            outs[i] += (bst.predict(Xe, raw_score=True) / slope) / len(seeds)
    return outs


def fit_predict_hist_xgb_3seed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    bm_tr: np.ndarray,
    slope: float,
    X_evals: list[np.ndarray],
    bm_evals: list[np.ndarray],
    seeds: tuple[int, ...] = (42, 142, 242),
) -> list[np.ndarray]:
    """Model 2: HistXGB_HessClipped (3-seed bag, tree_method='hist', max_bin=63, fixed 80 trees, zero val_idx peeking)."""
    outs = [np.zeros(len(Xe), dtype=np.float64) for Xe in X_evals]
    dtr = xgb.DMatrix(X_tr, label=y_tr, base_margin=bm_tr)
    dev_mats = [xgb.DMatrix(Xe, base_margin=bme) for Xe, bme in zip(X_evals, bm_evals)]
    for sd in seeds:
        params = dict(
            objective="binary:logistic",
            tree_method="hist",
            max_depth=3,
            min_child_weight=50.0,
            max_delta_step=0.15,
            colsample_bytree=0.5,
            subsample=0.7,
            reg_alpha=25.0,
            reg_lambda=150.0,
            learning_rate=0.005,
            max_bin=63,
            nthread=N_JOBS,
            verbosity=0,
            seed=sd,
        )
        bst = xgb.train(params, dtr, num_boost_round=80)
        for i, dm in enumerate(dev_mats):
            pred_margin = bst.predict(dm, output_margin=True)
            outs[i] += ((pred_margin - bm_evals[i]) / slope) / len(seeds)
    return outs


def build_spline_matrix(
    r_m: np.ndarray,
    r_lm: np.ndarray,
    r_k: np.ndarray,
    s_m: np.ndarray,
    d_xt: np.ndarray,
    sp_tfm: SplineTransformer | None = None,
    sc_tfm: StandardScaler | None = None,
    fit: bool = False,
):
    core4 = np.clip(np.column_stack([r_m, r_lm, r_k, s_m]), -4.0, 4.0)
    if fit:
        sp_tfm = SplineTransformer(n_knots=4, degree=2, include_bias=False)
        sp_part = sp_tfm.fit_transform(core4)
    else:
        sp_part = sp_tfm.transform(core4)
    inter = np.column_stack([
        r_m * r_lm,
        r_m * r_k,
        r_lm * r_k,
        r_m * s_m,
        r_lm * s_m,
        d_xt,
        d_xt * s_m,
    ])
    M = np.hstack([sp_part, inter])
    if fit:
        sc_tfm = StandardScaler()
        M_s = np.clip(np.nan_to_num(sc_tfm.fit_transform(M), 0.0), -5.0, 5.0)
    else:
        M_s = np.clip(np.nan_to_num(sc_tfm.transform(M), 0.0), -5.0, 5.0)
    return M_s, sp_tfm, sc_tfm


def fit_predict_spline_ridge(
    y_tr: np.ndarray,
    p_tr: np.ndarray,
    h_tr: np.ndarray,
    slope: float,
    core_tr: tuple,
    core_evals: list[tuple],
) -> list[np.ndarray]:
    """Model 3: Spline_Interaction_Ridge (n_knots=4, degree=2 + pairwise products + d_xgbf_tabm, Ridge(alpha=5e4))."""
    M_tr, sp_tfm, sc_tfm = build_spline_matrix(*core_tr, fit=True)
    w_tr = np.sqrt(h_tr)
    working_resid = (y_tr - p_tr) / (h_tr + 1e-4)
    ridge = Ridge(alpha=5e4)
    ridge.fit(M_tr, working_resid, sample_weight=w_tr)
    outs = []
    for ce in core_evals:
        Me, _, _ = build_spline_matrix(*ce, sp_tfm=sp_tfm, sc_tfm=sc_tfm, fit=False)
        outs.append(ridge.predict(Me) / slope)
    return outs


def main() -> None:
    print(f"[Boot] Starting gm_stage2_hardened_5f.py (n_jobs={N_JOBS})...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    target_mean = float(np.mean(y))

    # Load v8 Champion
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
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y, target_mean=target_mean))

    # Load 100% 5-fold populated Phase 2 & Sequence LM assets (ZERO 1-fold/2-fold gpt2-medium!)
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
    # 18 lowest-collinearity geometric distance/density columns from knn24:
    # (9 from orig20k retrieval + 9 from OOF train retrieval: min_l2, k5_l2, k50_l2, min_cos, k15_cos, tau0.5, tau1.0, tau2.0, d_pos/d_neg)
    k18_cols = [0, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22, 23]
    knn18_o = knn24_o[:, k18_cols]
    knn18_t = knn24_t[:, k18_cols]

    # 4 complete 5-fold Sequence LMs: lm12, masa, gpt2_3ep, gpt2_rev_fixed
    lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), lm12_o, n_knots=4000)

    masa_o = zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy").ravel())
    masa_t = pchip_quantile_map(
        zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy").ravel()),
        masa_o,
        n_knots=4000,
    )

    g3_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_oof.npy").ravel())
    g3_t = pchip_quantile_map(zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_test.npy").ravel()), g3_o, n_knots=4000)

    rev_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel())
    rev_t = pchip_quantile_map(zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy").ravel()), rev_o, n_knots=4000)

    lms_o = [lm12_o, masa_o, g3_o, rev_o]
    lms_t = [lm12_t, masa_t, g3_t, rev_t]

    # Arrays to hold out-of-fold predictions across the 5 folds
    delta_lgb_oof = np.zeros(ntr, dtype=np.float64)
    delta_xgb_oof = np.zeros(ntr, dtype=np.float64)
    delta_spl_oof = np.zeros(ntr, dtype=np.float64)
    d_xt_oof = np.zeros(ntr, dtype=np.float64)
    agree_oof = np.zeros(ntr, dtype=np.float64)

    delta_lgb_t_folds = []
    delta_xgb_t_folds = []
    delta_spl_t_folds = []
    d_xt_t_folds = []
    agree_t_folds = []

    z_minimax_oof = np.zeros(ntr, dtype=np.float64)
    z_minimax_t_folds = []
    minimax_weights = []

    # Step 1 & Step 2 & Step 3 per outer fold f in 0..4
    for f, (a, b) in enumerate(folds):
        tf0 = time.time()
        X_a, X_b, X_t, aux = build_fold_features_36(
            a, b, y[a],
            z_v8_o, z_v8_t,
            z_tabm_o, z_tabm_t,
            s_tabm_o, s_tabm_t,
            z_knn_o, z_knn_t,
            z_xgbf_o, z_xgbf_t,
            lms_o, lms_t,
            knn18_o, knn18_t,
        )
        slope = aux["slope"]
        bm_a, bm_b, bm_t = aux["bm_a"], aux["bm_b"], aux["bm_t"]

        # Fit the 3 residual correctors on full `train_idx` (`a`) -> predict on `val_idx` (`b`) and `test`
        dlgb_b, dlgb_t = fit_predict_hist_lgb_3seed(X_a, y[a], bm_a, slope, [X_b, X_t])
        dxgb_b, dxgb_t = fit_predict_hist_xgb_3seed(X_a, y[a], bm_a, slope, [X_b, X_t], [bm_b, bm_t])
        dspl_b, dspl_t = fit_predict_spline_ridge(
            y[a], aux["pa"], aux["ha"], slope,
            (aux["ra_m"], aux["lm_mean_a"], aux["ra_k"], aux["sa"], aux["d_xt_a"]),
            [
                (aux["rb_m"], aux["lm_mean_b"], aux["rb_k"], aux["sb"], aux["d_xt_b"]),
                (aux["rt_m"], aux["lm_mean_t"], aux["rt_k"], aux["st"], aux["d_xt_t"]),
            ],
        )

        delta_lgb_oof[b] = dlgb_b
        delta_xgb_oof[b] = dxgb_b
        delta_spl_oof[b] = dspl_b
        d_xt_oof[b] = aux["d_xt_b"]
        agree_oof[b] = aux["agree_lm_tabm_b"]

        delta_lgb_t_folds.append(dlgb_t)
        delta_xgb_t_folds.append(dxgb_t)
        delta_spl_t_folds.append(dspl_t)
        d_xt_t_folds.append(aux["d_xt_t"])
        agree_t_folds.append(aux["agree_lm_tabm_t"])

        # Step 3: To optimize portfolio weights `(w1, w2, w3, w4)` on `train_idx` (`a`) WITHOUT in-sample tree overfit,
        # compute out-of-fold basis predictions across the 4 internal training folds inside `a`!
        other_folds = [fi for fi in range(5) if fi != f]
        # Map global index -> position in `a`
        pos_in_a = np.full(ntr, -1, dtype=np.int32)
        pos_in_a[a] = np.arange(len(a), dtype=np.int32)
        inner_splits = []
        for fi_val in other_folds:
            val_glob = folds[fi_val][1]
            iv = pos_in_a[val_glob]
            mask = np.ones(len(a), dtype=bool)
            mask[iv] = False
            it = np.where(mask)[0]
            inner_splits.append((it, iv))

        dlgb_a_oof = np.zeros(len(a), dtype=np.float64)
        dxgb_a_oof = np.zeros(len(a), dtype=np.float64)
        dspl_a_oof = np.zeros(len(a), dtype=np.float64)

        for it, iv in inner_splits:
            dlgb_a_oof[iv] = fit_predict_hist_lgb_3seed(X_a[it], y[a[it]], bm_a[it], slope, [X_a[iv]])[0]
            dxgb_a_oof[iv] = fit_predict_hist_xgb_3seed(X_a[it], y[a[it]], bm_a[it], slope, [X_a[iv]], [bm_a[iv]])[0]
            dspl_a_oof[iv] = fit_predict_spline_ridge(
                y[a[it]], aux["pa"][it], aux["ha"][it], slope,
                (aux["ra_m"][it], aux["lm_mean_a"][it], aux["ra_k"][it], aux["sa"][it], aux["d_xt_a"][it]),
                [(aux["ra_m"][iv], aux["lm_mean_a"][iv], aux["ra_k"][iv], aux["sa"][iv], aux["d_xt_a"][iv])],
            )[0]

        # Standardize the 4 basis columns on `train_idx` (`a`) and apply identical scale to `b` and `t`
        B_a_raw = np.column_stack([
            aux["d_xt_a"],
            0.5 * (dlgb_a_oof + dxgb_a_oof),
            dspl_a_oof,
            aux["agree_lm_tabm_a"],
        ])
        B_b_raw = np.column_stack([
            aux["d_xt_b"],
            0.5 * (dlgb_b + dxgb_b),
            dspl_b,
            aux["agree_lm_tabm_b"],
        ])
        B_t_raw = np.column_stack([
            aux["d_xt_t"],
            0.5 * (dlgb_t + dxgb_t),
            dspl_t,
            aux["agree_lm_tabm_t"],
        ])

        b_mu = B_a_raw.mean(axis=0)
        b_sd = B_a_raw.std(axis=0) + 1e-12
        B_a = (B_a_raw - b_mu) / b_sd
        B_b = (B_b_raw - b_mu) / b_sd
        B_t = (B_t_raw - b_mu) / b_sd

        za_f = z_v8_o[a]
        ya_f = y[a]
        inner_base_aucs = [float(roc_auc_score(ya_f[iv], za_f[iv])) for _, iv in inner_splits]

        def minimax_neg_obj(w_vec: np.ndarray) -> float:
            if np.max(np.abs(w_vec)) > 0.08:
                return 1.0
            pred_a = za_f + B_a @ w_vec
            deltas = [
                float(roc_auc_score(ya_f[iv], pred_a[iv])) - base_iv
                for (_, iv), base_iv in zip(inner_splits, inner_base_aucs)
            ]
            mean_4f = float(np.mean(inner_base_aucs) + np.mean(deltas))
            min_delta = float(np.min(deltas))
            penalty_worst = 0.5 * max(0.0, -min_delta)
            penalty_l2 = 1e-4 * float(np.sum(w_vec**2))
            return -(mean_4f - penalty_worst - penalty_l2)

        # Warm-start grid over (w1: d_xgbf_tabm, w2: tree_resid, w3: spline, w4: agree)
        best_w = np.array([0.035, 0.0, 0.0, 0.0], dtype=np.float64)
        best_val = minimax_neg_obj(best_w)
        for w1 in (0.020, 0.030, 0.035, 0.045):
            for w2 in (-0.010, 0.0, 0.008, 0.015):
                for w3 in (0.0, 0.008, 0.015):
                    for w4 in (-0.008, 0.0, 0.008):
                        cand_w = np.array([w1, w2, w3, w4], dtype=np.float64)
                        v = minimax_neg_obj(cand_w)
                        if v < best_val:
                            best_val, best_w = v, cand_w

        opt_res = minimize(
            minimax_neg_obj, best_w, method="Powell", options=dict(maxiter=80, xtol=2e-3, ftol=1e-7)
        )
        wf = opt_res.x if opt_res.fun <= best_val else best_w
        minimax_weights.append([round(float(x), 5) for x in wf])
        z_minimax_oof[b] = z_v8_o[b] + B_b @ wf
        z_minimax_t_folds.append(z_v8_t + B_t @ wf)
        print(
            f"  [Fold {f}] done in {time.time() - tf0:.1f}s | minimax w={minimax_weights[-1]} | "
            f"fold ΔAUC={(roc_auc_score(y[b], z_minimax_oof[b]) - base_folds[f])*1e5:+.2f}e-5",
            flush=True,
        )

    # Assemble 5-fold averaged test predictions
    delta_lgb_t = np.mean(delta_lgb_t_folds, axis=0)
    delta_xgb_t = np.mean(delta_xgb_t_folds, axis=0)
    delta_spl_t = np.mean(delta_spl_t_folds, axis=0)
    d_xt_t = np.mean(d_xt_t_folds, axis=0)
    z_minimax_t = np.mean(z_minimax_t_folds, axis=0)

    # Pure 5-Fold Candidate C (`z_v8 + 0.025 * (r_xgbf - r_tabm)`, zero 1-fold test_2.npy)
    z_pureC_oof = z_v8_o + 0.025 * d_xt_oof
    z_pureC_t = z_v8_t + 0.025 * pchip_quantile_map(d_xt_t, d_xt_oof, n_knots=4000)

    # Solo & combined residual boosters at optimal nested step size
    def eval_1d_residual(name: str, d_oof: np.ndarray, d_t: np.ndarray) -> dict:
        z_o = np.zeros(ntr, dtype=np.float64)
        z_t_f = []
        ws = []
        for f, (a, b) in enumerate(folds):
            sd = float(np.std(d_oof[a]) + 1e-12)
            da = d_oof[a] / sd
            db = d_oof[b] / sd
            dt = d_t / sd
            best_eta, best_sc = 0.0, float(roc_auc_score(y[a], z_v8_o[a]))
            for eta in np.linspace(-0.04, 0.05, 91):
                sc = float(roc_auc_score(y[a], z_v8_o[a] + eta * da))
                if sc > best_sc:
                    best_sc, best_eta = sc, float(eta)
            ws.append(round(best_eta, 4))
            z_o[b] = z_v8_o[b] + best_eta * db
            z_t_f.append(z_v8_t + best_eta * dt)
        return score_row(name, z_o, np.mean(z_t_f, axis=0))

    def score_row(name: str, z_o: np.ndarray, z_t: np.ndarray) -> dict:
        full_auc = float(roc_auc_score(y, z_o))
        f_aucs = [float(roc_auc_score(y[b], z_o[b])) for _, b in folds]
        f_deltas = [fa - ba for fa, ba in zip(f_aucs, base_folds)]
        wins = sum(1 for d in f_deltas if d > 0)
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_o, y, target_mean=target_mean))
        ks_v = float(ks_2samp(p_t, p_v8_cal).statistic)
        rho_v = float(spearmanr(p_t, v8_sub).correlation)
        return {
            "name": name,
            "nested_oof_auc": full_auc,
            "nested_delta": full_auc - base_auc,
            "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_deltas],
            "min_fold_delta_1e5": round(min(f_deltas) * 1e5, 2),
            "fold_wins": wins,
            "ks_vs_v8": ks_v,
            "spearman_vs_v8": rho_v,
            "z_oof": z_o,
            "z_test": z_t,
            "p_test": p_t,
        }

    row_v8 = score_row("v8_Champion (Baseline)", z_v8_o, z_v8_t)
    row_pureC = score_row("Pure_5F_Candidate_C (0.025*d_xgbf_tabm)", z_pureC_oof, z_pureC_t)
    row_lgb = eval_1d_residual("HistLGB_HessClipped (36 non-tree feats)", delta_lgb_oof, delta_lgb_t)
    row_xgb = eval_1d_residual("HistXGB_HessClipped (36 non-tree feats)", delta_xgb_oof, delta_xgb_t)
    row_spl = eval_1d_residual("Spline_Interaction_Ridge", delta_spl_oof, delta_spl_t)
    row_tree_comb = eval_1d_residual(
        "HistLGB+HistXGB Combined", 0.5 * (delta_lgb_oof + delta_xgb_oof), 0.5 * (delta_lgb_t + delta_xgb_t)
    )
    row_minimax = score_row("GM_Minimax_Portfolio_5F", z_minimax_oof, z_minimax_t)

    rows = [row_v8, row_pureC, row_lgb, row_xgb, row_tree_comb, row_spl, row_minimax]

    print("\n" + "=" * 124)
    print(
        f"{'Model / Portfolio':42s} | {'Nested OOF':10s} | {'Nested ΔAUC':11s} | {'Folds':5s} | "
        f"{'Per-Fold Δ (×1e-5)':32s} | {'KS vs v8':8s} | {'Spearman':8s}"
    )
    print("-" * 124)
    for r in rows:
        print(
            f"{r['name']:42s} | {r['nested_oof_auc']:.7f} | {r['nested_delta']:+11.2e} | "
            f"{r['fold_wins']}/5   | {str(r['fold_deltas_1e5']):32s} | {r['ks_vs_v8']:.6f} | {r['spearman_vs_v8']:.6f}"
        )
    print("=" * 124)

    # Save both sub_pure5f_cand_c.csv and sub_gm_minimax_5f.csv locally
    p_pureC_path = SUB / "sub_pure5f_cand_c.csv"
    p_minimax_path = SUB / "sub_gm_minimax_5f.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": row_pureC["p_test"]}).to_csv(p_pureC_path, index=False)
    pd.DataFrame({"id": test_ids, "is_fraud": row_minimax["p_test"]}).to_csv(p_minimax_path, index=False)
    print(f"[Saved] {p_pureC_path.name} & {p_minimax_path.name}")

    # Strict Submission Gate:
    # Only auto-submit Slot 3 if GM_Minimax_Portfolio_5F beats Pure_5F_Candidate_C on Nested 5-Fold OOF
    # AND achieves 5/5 positive folds (or 4/5 with worst fold > -1.0e-5)
    beats_pureC = row_minimax["nested_oof_auc"] > row_pureC["nested_oof_auc"] + 1e-7
    folds_ok = row_minimax["fold_wins"] == 5 or (
        row_minimax["fold_wins"] >= 4 and row_minimax["min_fold_delta_1e5"] > -1.0
    )
    qualifies_slot3 = bool(beats_pureC and folds_ok and row_minimax["ks_vs_v8"] < 0.0025)

    report = {
        "v8_base_auc": base_auc,
        "minimax_weights": minimax_weights,
        "rows": [{k: v for k, v in r.items() if k not in ("z_oof", "z_test", "p_test")} for r in rows],
        "beats_pureC": beats_pureC,
        "folds_ok": folds_ok,
        "qualifies_slot3": qualifies_slot3,
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "gm_stage2_hardened_5f.json").write_text(json.dumps(report, indent=2))

    if qualifies_slot3 and "--submit-if-qualified" in sys.argv:
        msg = (
            f"GM_Minimax_Portfolio_5F (Nested OOF={row_minimax['nested_oof_auc']:.7f}, "
            f"Δ={row_minimax['nested_delta']:+.2e}, {row_minimax['fold_wins']}/5 folds, "
            f"minF={row_minimax['min_fold_delta_1e5']:+.2f}e-5, KS={row_minimax['ks_vs_v8']:.5f})"
        )
        cmd = [
            str(ROOT / ".venv_forensics/bin/kaggle"),
            "competitions",
            "submit",
            "-c",
            "playground-series-s6e11",
            "-f",
            str(p_minimax_path),
            "-m",
            msg,
        ]
        print("[Submitting Slot 3]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)
    else:
        print(
            f"\n[Submission Gate] qualifies_slot3={qualifies_slot3} "
            f"(beats_pureC={beats_pureC}, folds_ok={folds_ok}, min_fold_delta={row_minimax['min_fold_delta_1e5']:+.2f}e-5). "
            f"Holding Slot 3 (5 submissions preserved)."
        )


if __name__ == "__main__":
    main()
