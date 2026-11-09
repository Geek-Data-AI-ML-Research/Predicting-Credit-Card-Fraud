#!/usr/bin/env python3
"""De-Corrupt & Boost 5-Fold Pipeline (`de_corrupt_and_boost_5f.py`, `n_jobs=4`)

Fixes the `gpt2_3ep_oof.npy` Folds 0 & 1 corruption (where F0/F1 were copies of `lm12` while F2/F3/F4 were
3-epoch runs) by rebuilding `xgb_fast_harvest_clean` strictly on homogeneous 5-fold LMs (`lm12`, `masa`,
`gpt2_rev_fixed`), `TabM`, `knn24`, and top-30 `KMAT` features WITHOUT `PLSRegression(h16)` and WITHOUT `gpt2_3ep`.
Then evaluates:
  - Step 2: Control-variate damping factor `gamma` in `u(gamma) = r_xgb - gamma * r_tabm` (for old, clean, and 50/50 blend).
  - Step 3: Adding the 2-fold (`F0 + F2`) `[z_v8, z_lm12]` Double-Orthogonalized `gpt2-medium` residual (`oof_0 + oof_2`, `test_0 + test_2`).
  - Step 4: Full comparison table + strict submission gate (`5/5 positive folds` AND `nested ΔAUC >= +2.8e-5` AND `1 - rho >= 1.5e-4`).
"""
from __future__ import annotations

import hashlib
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
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, skew, spearmanr
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_kmat_and_ratios(tr: pd.DataFrame, te: pd.DataFrame):
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
        ("merchant_category", "auth_method"),
        ("merchant_category", "channel"),
        ("card_type", "auth_method"),
        ("is_foreign_transaction", "ip_country_mismatch"),
        ("used_vpn", "ip_country_mismatch"),
        ("is_new_merchant", "billing_shipping_mismatch"),
        ("cvv_retry_count", "auth_method"),
        ("prior_disputes", "is_ai_generated_scam_attempt"),
    ]:
        K[f"j_{c1}_{c2}"] = (full[c1].astype(str) + "_" + full[c2].astype(str)).values

    knames = list(K.keys())
    codes = [pd.factorize(pd.Series(K[n]))[0].astype(np.int32) for n in knames]
    KMAT = np.column_stack(codes)
    NCAT = np.array([c.max() + 1 for c in codes], dtype=np.int32)

    amt = np.log1p(np.maximum(full["amount_usd"].to_numpy(np.float64), 0.0))
    bal = np.log1p(np.maximum(full["account_balance_usd"].to_numpy(np.float64), 0.0))
    hrs = np.log1p(np.maximum(full["hours_since_last_txn"].to_numpy(np.float64), 0.0))
    dist = np.log1p(np.maximum(full["distance_from_home_km"].to_numpy(np.float64), 0.0))
    vel = full["velocity_score"].to_numpy(np.float64)
    mrisk = full["merchant_risk_score"].to_numpy(np.float64)
    tc24 = full["txn_count_last_24h"].to_numpy(np.float64)
    cvv = full["cvv_retry_count"].to_numpy(np.float64)
    pdisp = full["prior_disputes"].to_numpy(np.float64)
    cage = full["card_age_months"].to_numpy(np.float64)
    uage = full["customer_age"].to_numpy(np.float64)
    flags_sum = full[FLAGS].to_numpy(np.float64).sum(axis=1)

    RATIOS = np.column_stack([
        amt, bal, hrs, dist, vel, mrisk, tc24, cvv, pdisp, cage, uage, flags_sum,
        amt - bal,
        full["amount_usd"].to_numpy(np.float64) / (full["account_balance_usd"].to_numpy(np.float64) + 10.0),
        vel / (tc24 + 1.0),
        vel * mrisk,
        dist / (hrs + 0.5),
        cvv * mrisk,
        flags_sum * mrisk,
        cage / (uage * 12.0 + 1.0),
    ]).astype(np.float32)

    return KMAT, NCAT, RATIOS


def fold_te_top30(
    KMAT: np.ndarray,
    NCAT: np.ndarray,
    y_tr: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ntr: int,
    nte: int,
    seed: int = 42,
    smooth: float = 20.0,
):
    Ka = KMAT[a]
    Kb = KMAT[b]
    Kt = KMAT[ntr : ntr + nte]
    ya = y_tr[a].astype(np.float64)
    prior = float(ya.mean())
    n_keys = KMAT.shape[1]

    Ea = np.zeros((len(a), n_keys), dtype=np.float32)
    Eb = np.zeros((len(b), n_keys), dtype=np.float32)
    Et = np.zeros((nte, n_keys), dtype=np.float32)

    inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(a)), y_tr[a]))
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
    Ea = logit(np.clip(Ea, eps, 1 - eps)).astype(np.float32)
    Eb = logit(np.clip(Eb, eps, 1 - eps)).astype(np.float32)
    Et = logit(np.clip(Et, eps, 1 - eps)).astype(np.float32)

    ya_c = ya - prior
    ea_c = Ea - Ea.mean(axis=0)
    corrs = np.abs((ea_c * ya_c[:, None]).sum(axis=0) / (np.sqrt((ea_c**2).sum(axis=0)) + 1e-8))
    top30 = np.argsort(corrs)[::-1][:30]
    return Ea[:, top30], Eb[:, top30], Et[:, top30]


def build_clean_lm_block(ntr: int, nte: int, folds: list):
    """Build 100% homogeneous 5-fold LM block (`lm12`, `masa`, `gpt2_rev_fixed` + per-order rev stats — NO `gpt2_3ep`!)."""
    lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), lm12_o, n_knots=4000)

    masa_o = zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy").ravel())
    masa_t = pchip_quantile_map(
        zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy").ravel()),
        masa_o,
        n_knots=4000,
    )

    rev_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel())
    rev_t = pchip_quantile_map(zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy").ravel()), rev_o, n_knots=4000)

    rev_ord_o = np.zeros((ntr, 4), dtype=np.float32)
    rev_ord_t_folds = []
    for f, (_, b) in enumerate(folds):
        ro = np.load(PRED / f"gpt2_rev/oof_{f}.npy").astype(np.float64)
        rt = np.load(PRED / f"gpt2_rev/test_{f}.npy").astype(np.float64)
        do = ro[:, :, 1] - ro[:, :, 0]
        dt = rt[:, :, 1] - rt[:, :, 0]
        rev_ord_o[b, 0] = do.mean(axis=1)
        rev_ord_o[b, 1] = do.std(axis=1)
        rev_ord_o[b, 2] = do.min(axis=1)
        rev_ord_o[b, 3] = do.max(axis=1)
        tb = np.column_stack([dt.mean(axis=1), dt.std(axis=1), dt.min(axis=1), dt.max(axis=1)]).astype(np.float32)
        rev_ord_t_folds.append(tb)
    rev_ord_t = np.mean(rev_ord_t_folds, axis=0)

    M_o = np.column_stack([lm12_o, masa_o, rev_o])
    M_t = np.column_stack([lm12_t, masa_t, rev_t])

    def stats3(M: np.ndarray) -> np.ndarray:
        return np.column_stack([
            M,
            M.mean(axis=1),
            M.std(axis=1),
            M.min(axis=1),
            M.max(axis=1),
            M[:, 0] - M[:, 1],
            M[:, 0] - M[:, 2],
            M[:, 1] - M[:, 2],
        ]).astype(np.float32)

    return np.hstack([stats3(M_o), rev_ord_o]), np.hstack([stats3(M_t), rev_ord_t]), lm12_o, lm12_t


def bayes_fold(arr: np.ndarray, prior: float) -> np.ndarray:
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def main() -> None:
    print(f"[Boot] Starting de_corrupt_and_boost_5f.py (n_jobs={N_JOBS})...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    for df in (tr, te):
        df[FLAGS] = df[FLAGS].astype("int8")
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    target_mean = float(np.mean(y))

    # Verify Fold 0 & Fold 2 gpt2-medium files
    for fi in (0, 2):
        op = PRED / f"gpt2_med_4ep/oof_{fi}.npy"
        tp = PRED / f"gpt2_med_4ep/test_{fi}.npy"
        ao, at = np.load(op), np.load(tp)
        assert ao.shape == (len(folds[fi][1]), 8, 2) and np.isfinite(ao).all()
        assert at.shape == (nte, 4, 2) and np.isfinite(at).all()
        print(f"[Gate] Fold {fi}: oof_{fi}.npy sha256={sha256_file(op)[:12]}.. | test_{fi}.npy sha256={sha256_file(tp)[:12]}..")

    # Step 1: Build clean homogeneous 5-fold feature matrix X_clean and train oof/test_xgb_clean.npy
    clean_oof_path = HARV / "oof_xgb_clean.npy"
    clean_test_path = HARV / "test_xgb_clean.npy"

    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)
    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float32)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float32)

    z_knn_o = zrk(np.load(HARV / "oof_orig20k_knn.npy").ravel())
    z_knn_t = pchip_quantile_map(zrk(np.load(HARV / "test_orig20k_knn.npy").ravel()), z_knn_o, n_knots=4000)
    knn24_o = np.load(HARV / "knn24_oof.npy").astype(np.float32)
    knn24_t = np.load(HARV / "knn24_test.npy").astype(np.float32)

    lm_clean_o, lm_clean_t, z_lm12_o, z_lm12_t = build_clean_lm_block(ntr, nte, folds)
    KMAT, NCAT, RATIOS = build_kmat_and_ratios(tr, te)
    Ra_all = RATIOS[:ntr]
    Rt_all = RATIOS[ntr : ntr + nte]

    base_neural_o = np.column_stack([
        lm_clean_o,
        z_tabm_o.astype(np.float32),
        s_tabm_o,
        (z_tabm_o - z_lm12_o).astype(np.float32),
        z_knn_o.astype(np.float32),
        knn24_o,
    ]).astype(np.float32)
    base_neural_t = np.column_stack([
        lm_clean_t,
        z_tabm_t.astype(np.float32),
        s_tabm_t,
        (z_tabm_t - z_lm12_t).astype(np.float32),
        z_knn_t.astype(np.float32),
        knn24_t,
    ]).astype(np.float32)

    oof_xgb_clean = np.zeros(ntr, dtype=np.float64)
    test_xgb_clean_folds = []

    print("[Step 1] Training 5-fold 3-seed `xgb_fast_harvest_clean` (NO corrupted gpt2_3ep, NO PLS h16)...", flush=True)
    for f, (a, b) in enumerate(folds):
        tf0 = time.time()
        Ea30, Eb30, Et30 = fold_te_top30(KMAT, NCAT, y, a, b, ntr, nte, seed=42 + f)
        Xa = np.hstack([base_neural_o[a], Ea30, Ra_all[a]]).astype(np.float32)
        Xb = np.hstack([base_neural_o[b], Eb30, Ra_all[b]]).astype(np.float32)
        Xt = np.hstack([base_neural_t, Et30, Rt_all]).astype(np.float32)

        pred_b = np.zeros(len(b), dtype=np.float64)
        pred_t = np.zeros(nte, dtype=np.float64)
        seeds = (42 + f, 142 + f, 242 + f)
        for sd in seeds:
            model = lgb.LGBMClassifier(
                objective="binary",
                max_depth=4,
                num_leaves=12,
                min_child_samples=80,
                colsample_bytree=0.5,
                subsample=0.8,
                subsample_freq=1,
                reg_lambda=25.0,
                learning_rate=0.02,
                n_estimators=350,
                n_jobs=N_JOBS,
                random_state=sd,
                verbose=-1,
            )
            model.fit(Xa, y[a])
            pred_b += zrk(model.predict_proba(Xb)[:, 1]) / len(seeds)
            pred_t += zrk(model.predict_proba(Xt)[:, 1]) / len(seeds)

        oof_xgb_clean[b] = pred_b
        test_xgb_clean_folds.append(pred_t)
        print(
            f"  [Step 1] fold {f} xgb_clean OOF AUC = {roc_auc_score(y[b], pred_b):.6f} ({time.time() - tf0:.1f}s)",
            flush=True,
        )

    test_xgb_clean = np.mean(test_xgb_clean_folds, axis=0)
    np.save(clean_oof_path, oof_xgb_clean)
    np.save(clean_test_path, test_xgb_clean)

    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=4000)

    z_xgbf_cln_o = zrk(oof_xgb_clean)
    z_xgbf_cln_t = pchip_quantile_map(zrk(test_xgb_clean), z_xgbf_cln_o, n_knots=4000)

    print(
        f"[Step 1 Summary] Solo 5-Fold OOF AUC: old xgb_fast_harvest = {roc_auc_score(y, z_xgbf_old_o):.6f} | "
        f"clean xgb_fast_harvest_clean = {roc_auc_score(y, z_xgbf_cln_o):.6f} "
        f"(corr={np.corrcoef(z_xgbf_old_o, z_xgbf_cln_o)[0,1]:.5f})",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Step 2: Optimize Control-Variate Damping Factor `gamma` for 5/5 Folds
    # -----------------------------------------------------------------------
    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = (
        pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(np.float64)
    )
    p25_sub = (
        pd.read_csv(SUB / "sub_phase2_5_v8_meta_harvest.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(np.float64)
    )
    z_v8_o = zrk(v8_oof)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_auc = float(roc_auc_score(y, z_v8_o))
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y, target_mean=target_mean))

    # Compute fold-wise orthogonal residuals against:
    # (i) single [z_v8] and (ii) double [z_v8, z_lm12]
    def build_orth_residuals(co: np.ndarray, ct: np.ndarray, double_lm12: bool = False):
        ro = np.zeros(ntr, dtype=np.float64)
        rt_folds = []
        for f, (a, b) in enumerate(folds):
            if double_lm12:
                Xa = np.column_stack([z_v8_o[a], z_lm12_o[a]])
                Xb = np.column_stack([z_v8_o[b], z_lm12_o[b]])
                Xt = np.column_stack([z_v8_t, z_lm12_t])
            else:
                Xa = z_v8_o[a].reshape(-1, 1)
                Xb = z_v8_o[b].reshape(-1, 1)
                Xt = z_v8_t.reshape(-1, 1)
            ols = LinearRegression().fit(Xa, co[a])
            ra = co[a] - ols.predict(Xa)
            mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
            ro[b] = (co[b] - ols.predict(Xb) - mu) / sd
            rt_folds.append((ct - ols.predict(Xt) - mu) / sd)
        return ro, pchip_quantile_map(np.mean(rt_folds, axis=0), ro, n_knots=4000)

    r_old_1, r_old_1_t = build_orth_residuals(z_xgbf_old_o, z_xgbf_old_t, double_lm12=False)
    r_cln_1, r_cln_1_t = build_orth_residuals(z_xgbf_cln_o, z_xgbf_cln_t, double_lm12=False)
    r_tab_1, r_tab_1_t = build_orth_residuals(z_tabm_o, z_tabm_t, double_lm12=False)

    r_old_2, r_old_2_t = build_orth_residuals(z_xgbf_old_o, z_xgbf_old_t, double_lm12=True)
    r_cln_2, r_cln_2_t = build_orth_residuals(z_xgbf_cln_o, z_xgbf_cln_t, double_lm12=True)
    r_tab_2, r_tab_2_t = build_orth_residuals(z_tabm_o, z_tabm_t, double_lm12=True)

    def eval_u_series(name: str, u_o: np.ndarray, u_t: np.ndarray, w_fixed: float = 0.025) -> dict:
        # Both nested w (fit on 4 training folds) and fixed w=0.025
        z_o_fix = z_v8_o + w_fixed * u_o
        z_t_fix = z_v8_t + w_fixed * u_t

        # Nested w via minimax-penalized 1D grid on `train_idx`
        z_o_nst = np.zeros(ntr, dtype=np.float64)
        z_t_nst_f = []
        ws = []
        for f, (a, b) in enumerate(folds):
            other_f = [fi for fi in range(5) if fi != f]
            best_w, best_v = w_fixed, -1e9
            for w in np.linspace(0.010, 0.045, 36):
                f_ds = [
                    float(roc_auc_score(y[folds[fi][1]], z_v8_o[folds[fi][1]] + w * u_o[folds[fi][1]])) - base_folds[fi]
                    for fi in other_f
                ]
                val = float(np.mean(f_ds)) - 0.5 * max(0.0, -float(np.min(f_ds)))
                if val > best_v:
                    best_v, best_w = val, float(w)
            ws.append(round(best_w, 4))
            z_o_nst[b] = z_v8_o[b] + best_w * u_o[b]
            z_t_nst_f.append(z_v8_t + best_w * u_t)
        z_t_nst = np.mean(z_t_nst_f, axis=0)

        # Report using nested OOF
        auc_nst = float(roc_auc_score(y, z_o_nst))
        f_ds = [float(roc_auc_score(y[b], z_o_nst[b])) - bf for (_, b), bf in zip(folds, base_folds)]
        wins = sum(1 for d in f_ds if d > 0)
        p_t = expit(calibrate_logit_to_prevalence(z_t_nst, z_o_nst, y, target_mean=target_mean))
        ks_v8 = float(ks_2samp(p_t, p_v8_cal).statistic)
        rho_v8 = float(spearmanr(p_t, v8_sub).correlation)
        rho_s2 = float(spearmanr(p_t, p25_sub).correlation)
        return {
            "name": name,
            "nested_oof_auc": auc_nst,
            "nested_delta": auc_nst - base_auc,
            "fixed_delta": float(roc_auc_score(y, z_o_fix) - base_auc),
            "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_ds],
            "min_fold_1e5": round(min(f_ds) * 1e5, 2),
            "fold_wins": wins,
            "ws": ws,
            "ks_vs_v8": ks_v8,
            "rho_vs_v8": rho_v8,
            "dist_s2_1e4": round((1.0 - rho_s2) * 1e4, 3),
            "z_oof": z_o_nst,
            "z_test": z_t_nst,
            "p_test": p_t,
        }

    candidates = []
    # Solo clean vs solo old
    candidates.append(eval_u_series("1a. r_xgb_old solo (gamma=0)", r_old_1, r_old_1_t))
    candidates.append(eval_u_series("1b. r_xgb_clean solo (gamma=0, NO gpt2_3ep)", r_cln_1, r_cln_1_t))

    for gamma in (0.35, 0.50, 0.65, 0.75, 0.85, 1.00):
        # Single-orth vs z_v8
        candidates.append(
            eval_u_series(f"2a. Old [v8]: r_xgb_old - {gamma:.2f}*r_tabm", r_old_1 - gamma * r_tab_1, r_old_1_t - gamma * r_tab_1_t)
        )
        candidates.append(
            eval_u_series(f"2b. Clean [v8]: r_xgb_clean - {gamma:.2f}*r_tabm", r_cln_1 - gamma * r_tab_1, r_cln_1_t - gamma * r_tab_1_t)
        )
        candidates.append(
            eval_u_series(
                f"2c. Mix50 [v8]: 0.5*(old+clean) - {gamma:.2f}*r_tabm",
                0.5 * (r_old_1 + r_cln_1) - gamma * r_tab_1,
                0.5 * (r_old_1_t + r_cln_1_t) - gamma * r_tab_1_t,
            )
        )
        # Double-orth vs [z_v8, z_lm12]
        candidates.append(
            eval_u_series(
                f"2d. Mix50 [v8,lm12]: 0.5*(old+clean) - {gamma:.2f}*r_tabm",
                0.5 * (r_old_2 + r_cln_2) - gamma * r_tab_2,
                0.5 * (r_old_2_t + r_cln_2_t) - gamma * r_tab_2_t,
            )
        )

    # Pick top Step 2 candidate (prioritize 5/5 folds, then min_fold_1e5, then nested_delta)
    best_step2 = max(candidates, key=lambda r: (r["fold_wins"], r["nested_delta"]))

    # -----------------------------------------------------------------------
    # Step 3: Evaluate 2-Fold (F0 + F2) Double-Orthogonalized gpt2-medium (test_0 + test_2)
    # -----------------------------------------------------------------------
    a0, idx0 = folds[0]
    a2, idx2 = folds[2]
    z_m0_o = zrk(bayes_fold(np.load(PRED / "gpt2_med_4ep/oof_0.npy"), float(y[a0].mean())))
    z_m0_t = pchip_quantile_map(zrk(bayes_fold(np.load(PRED / "gpt2_med_4ep/test_0.npy"), float(y[a0].mean()))), z_m0_o, n_knots=4000)

    z_m2_o = zrk(bayes_fold(np.load(PRED / "gpt2_med_4ep/oof_2.npy"), float(y[a2].mean())))
    z_m2_t = pchip_quantile_map(zrk(bayes_fold(np.load(PRED / "gpt2_med_4ep/test_2.npy"), float(y[a2].mean()))), z_m2_o, n_knots=4000)

    # Cross-fitted Double-Orthogonalization against [z_v8, z_lm12] on F0 and F2
    X_cov0 = np.column_stack([z_v8_o[idx0], z_lm12_o[idx0]])
    X_cov2 = np.column_stack([z_v8_o[idx2], z_lm12_o[idx2]])
    X_cov_t = np.column_stack([z_v8_t, z_lm12_t])

    ols_on_2 = LinearRegression().fit(X_cov2, z_m2_o)
    u_m0 = z_m0_o - ols_on_2.predict(X_cov0)
    mu0, sd0 = float(u_m0.mean()), float(u_m0.std() + 1e-12)
    u_m0 = (u_m0 - mu0) / sd0
    u_m0_t = pchip_quantile_map((z_m0_t - ols_on_2.predict(X_cov_t) - mu0) / sd0, u_m0, n_knots=4000)

    ols_on_0 = LinearRegression().fit(X_cov0, z_m0_o)
    u_m2 = z_m2_o - ols_on_0.predict(X_cov2)
    mu2, sd2 = float(u_m2.mean()), float(u_m2.std() + 1e-12)
    u_m2 = (u_m2 - mu2) / sd2
    u_m2_t = pchip_quantile_map((z_m2_t - ols_on_0.predict(X_cov_t) - mu2) / sd2, u_m2, n_knots=4000)

    # Cross-fold out-of-fold weight selection: fit w_m on F2 -> apply to F0; fit w_m on F0 -> apply to F2
    z_s2_o = best_step2["z_oof"].copy()
    z_s2_t = best_step2["z_test"].copy()
    w_grid = np.linspace(-0.06, 0.04, 201)
    w_fit_on_f0 = max(w_grid, key=lambda w: roc_auc_score(y[idx0], z_s2_o[idx0] + w * u_m0))
    w_fit_on_f2 = max(w_grid, key=lambda w: roc_auc_score(y[idx2], z_s2_o[idx2] + w * u_m2))
    w_mean_02 = 0.5 * (w_fit_on_f0 + w_fit_on_f2)

    z_step3_o = z_s2_o.copy()
    z_step3_o[idx0] = z_s2_o[idx0] + w_fit_on_f2 * u_m0
    z_step3_o[idx2] = z_s2_o[idx2] + w_fit_on_f0 * u_m2
    u_m_test_f02 = 0.5 * (u_m0_t + u_m2_t)
    z_step3_t = z_s2_t + w_mean_02 * u_m_test_f02

    auc_s3 = float(roc_auc_score(y, z_step3_o))
    f_ds_s3 = [float(roc_auc_score(y[b], z_step3_o[b])) - bf for (_, b), bf in zip(folds, base_folds)]
    wins_s3 = sum(1 for d in f_ds_s3 if d > 0)
    p_t_s3 = expit(calibrate_logit_to_prevalence(z_step3_t, z_step3_o, y, target_mean=target_mean))
    step3_res = {
        "name": f"3. {best_step2['name'][:26]} + gpt2m_F02_DoubleOrth",
        "nested_oof_auc": auc_s3,
        "nested_delta": auc_s3 - base_auc,
        "fixed_delta": auc_s3 - base_auc,
        "fold_deltas_1e5": [round(d * 1e5, 2) for d in f_ds_s3],
        "min_fold_1e5": round(min(f_ds_s3) * 1e5, 2),
        "fold_wins": wins_s3,
        "ws": [round(w_fit_on_f0, 4), round(w_fit_on_f2, 4)],
        "ks_vs_v8": float(ks_2samp(p_t_s3, p_v8_cal).statistic),
        "rho_vs_v8": float(spearmanr(p_t_s3, v8_sub).correlation),
        "dist_s2_1e4": round((1.0 - float(spearmanr(p_t_s3, p25_sub).correlation)) * 1e4, 3),
        "z_oof": z_step3_o,
        "z_test": z_step3_t,
        "p_test": p_t_s3,
    }
    candidates.append(step3_res)

    # Sort & print top table
    print("\n" + "=" * 132)
    print(
        f"{'Candidate':46s} | {'Nested OOF':10s} | {'Nested Δ':9s} | {'Folds':5s} | "
        f"{'Per-Fold Δ (×1e-5)':32s} | {'KS v8':8s} | {'1-ρ(S2)':8s}"
    )
    print("-" * 132)
    for r in candidates:
        print(
            f"{r['name']:46s} | {r['nested_oof_auc']:.7f} | {r['nested_delta']:+9.2e} | "
            f"{r['fold_wins']}/5   | {str(r['fold_deltas_1e5']):32s} | {r['ks_vs_v8']:.6f} | {r['dist_s2_1e4']:6.3f}e-4"
        )
    print("=" * 132)

    # Check strict submission gate: 5/5 positive folds AND nested ΔAUC >= +2.8e-5 AND 1-rho(S2) >= 1.5e-4
    qualified = [
        r for r in candidates
        if r["fold_wins"] == 5 and r["nested_delta"] >= 2.8e-5 and r["dist_s2_1e4"] >= 1.5 and r["ks_vs_v8"] < 0.0025
    ]
    top_overall = max(candidates, key=lambda r: (r["fold_wins"], r["nested_delta"]))
    sub_path = SUB / "sub_decorrupt_boost_5f.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": top_overall["p_test"]}).to_csv(sub_path, index=False)

    summary = {
        "v8_base_auc": base_auc,
        "old_xgb_solo_auc": float(roc_auc_score(y, z_xgbf_old_o)),
        "clean_xgb_solo_auc": float(roc_auc_score(y, z_xgbf_cln_o)),
        "w_fit_on_f0": float(w_fit_on_f0),
        "w_fit_on_f2": float(w_fit_on_f2),
        "top_overall": {k: v for k, v in top_overall.items() if k not in ("z_oof", "z_test", "p_test")},
        "qualified_count": len(qualified),
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "de_corrupt_and_boost_5f.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n[Gate Result] Top candidate = {top_overall['name']} "
        f"(Nested OOF={top_overall['nested_oof_auc']:.7f}, Δ={top_overall['nested_delta']:+.2e}, "
        f"wins={top_overall['fold_wins']}/5, 1-ρ(S2)={top_overall['dist_s2_1e4']:.3f}e-4). "
        f"Qualified for auto-submit: {len(qualified)}."
    )


if __name__ == "__main__":
    main()
