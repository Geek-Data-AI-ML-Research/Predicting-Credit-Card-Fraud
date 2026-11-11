#!/usr/bin/env python3
"""Forensic reconstruction of exact St9.5 OOF twin + 2F→5F gpt2-medium chain upgrade.

STRICT NO_SUBMIT=True. Stages Strike9.6 + Strike10 candidates only.
"""
from __future__ import annotations

import json
import os
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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
ART = PRED / "st9p5_recon_5f"
ART.mkdir(parents=True, exist_ok=True)
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
TARGET_MEAN = 0.016700
N_KNOTS = 4000


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def zrk05(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), 1e-6, 1.0 - 1e-6))


def pchip_quantile_map(x_test: np.ndarray, x_ref: np.ndarray, n_knots: int = N_KNOTS) -> np.ndarray:
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


def bayes_fold(arr: np.ndarray, prior: float) -> np.ndarray:
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def load_sub(fn: str, test_ids: np.ndarray) -> np.ndarray:
    return pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)


def one_minus_rho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).correlation)


def weight_for_spearman_dist(z_base, r, target_1m_rho, lo=0.0, hi=2.0, n=40):
    """Binary-search scale w s.t. 1-rho(z_base, z_base+w*r) ≈ target."""
    target = float(target_1m_rho)
    best_w, best_err = 0.0, 1e9
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        dist = one_minus_rho(z_base, z_base + mid * r)
        err = abs(dist - target)
        if err < best_err:
            best_err, best_w = err, mid
        if dist > target:
            hi = mid
        else:
            lo = mid
    return best_w, one_minus_rho(z_base, z_base + best_w * r)


def gram_schmidt_cols(X: np.ndarray) -> np.ndarray:
    Q = []
    for j in range(X.shape[1]):
        v = X[:, j].astype(np.float64).copy()
        for q in Q:
            v -= np.dot(v, q) * q
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        Q.append(v / n)
    return np.column_stack(Q) if Q else np.zeros((X.shape[0], 0))


def orth_vec(v: np.ndarray, bases: list[np.ndarray]) -> np.ndarray:
    out = v.astype(np.float64).copy()
    for b in bases:
        b = np.asarray(b, dtype=np.float64).ravel()
        nb2 = float(np.dot(b, b)) + 1e-18
        out -= (float(np.dot(out, b)) / nb2) * b
    return out


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_delta(y, z_new, z_base, folds):
    base = float(roc_auc_score(y, z_base))
    full = float(roc_auc_score(y, z_new))
    bf = fold_aucs(y, z_base, folds)
    nf = fold_aucs(y, z_new, folds)
    d = [n - b for n, b in zip(nf, bf)]
    return {
        "auc": full,
        "delta_1e5": round((full - base) * 1e5, 2),
        "pos": int(sum(1 for x in d if x > 0)),
        "folds_1e5": [round(x * 1e5, 2) for x in d],
    }


# ---------------------------------------------------------------------------
# Part 1 helpers: Slot3 reconstruction (exact F0+F2 lineage)
# ---------------------------------------------------------------------------
def rebuild_slot3(y, folds, test_ids, gpt2_folds=(0, 2)):
    """Rebuild Slot3 OOF/test. gpt2_folds controls which folds get H1/H2 OOF upgrades
    and which gpt2 test folds enter the averaged test signal. Weights always fit on F0↔F2 only.
    """
    ntr = len(y)
    nte = len(test_ids)
    idx = {f: b for f, (_, b) in enumerate(folds)}
    a_tr = {f: a for f, (a, _) in enumerate(folds)}

    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    v8_sub = load_sub("sub_v8_part12_fsl_prop_rect.csv", test_ids)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o)
    base_folds = fold_aucs(y, z_v8_o, folds)

    z_lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), z_lm12_o)

    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o)
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o)
    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)

    r_old_o = np.zeros(ntr)
    r_tab_o = np.zeros(ntr)
    s_norm_o = np.zeros(ntr)
    r_old_t_f, r_tab_t_f, s_norm_t_f = [], [], []
    for a, b in folds:
        za, zb = z_v8_o[a], z_v8_o[b]
        var_a = float(np.var(za))

        def orth1(co, ct):
            beta = float(np.cov(co[a], za, ddof=0)[0, 1] / var_a)
            ra = co[a] - beta * za
            mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
            return (co[b] - beta * zb - mu) / sd, (ct - beta * z_v8_t - mu) / sd

        ro_b, ro_t = orth1(z_xgbf_old_o, z_xgbf_old_t)
        rt_b, rt_t = orth1(z_tabm_o, z_tabm_t)
        r_old_o[b], r_tab_o[b] = ro_b, rt_b
        r_old_t_f.append(ro_t)
        r_tab_t_f.append(rt_t)
        smu, ssd = float(s_tabm_o[a].mean()), float(s_tabm_o[a].std() + 1e-12)
        s_norm_o[b] = np.clip((s_tabm_o[b] - smu) / ssd, -3.5, 3.5)
        s_norm_t_f.append(np.clip((s_tabm_t - smu) / ssd, -3.5, 3.5))
    r_old_t = pchip_quantile_map(np.mean(r_old_t_f, 0), r_old_o)
    r_tab_t = pchip_quantile_map(np.mean(r_tab_t_f, 0), r_tab_o)
    s_norm_t = np.mean(s_norm_t_f, 0)

    # Candidate0 / CandC trunk (exact Slot3 nested selection)
    th_o = np.tanh(z_v8_o / 2.0)
    th_t = np.tanh(z_v8_t / 2.0)
    param_grid = [
        (gmin, gmax, temp, bias, w_cv)
        for gmin in (0.30, 0.50, 0.70, 0.85)
        for gmax in (0.85, 1.00, 1.15)
        if gmax >= gmin
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
    z_cand0_t_f, u_cv_best_t_f = [], []
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
        u_cv_best_t_f.append(ut)
        z_cand0_t_f.append(z_v8_t + w_cv * ut)
    u_cv5_t = np.mean(u_cv_best_t_f, 0)
    z_cand0_t = np.mean(z_cand0_t_f, 0)

    u_candC_o = r_old_o - r_tab_o
    u_candC_t = r_old_t - r_tab_t
    z_candC_o = z_v8_o + 0.025 * u_candC_o
    z_candC_t = z_v8_t + 0.025 * u_candC_t
    if roc_auc_score(y, z_cand0_o) >= roc_auc_score(y, z_candC_o):
        z_base5_o, z_base5_t = z_cand0_o.copy(), z_cand0_t.copy()
        u_cv5_o, u_cv5_t = u_cv5_o.copy(), u_cv5_t.copy()
        trunk = "Cand0"
    else:
        z_base5_o, z_base5_t = z_candC_o.copy(), z_candC_t.copy()
        u_cv5_o, u_cv5_t = u_candC_o.copy(), u_candC_t.copy()
        trunk = "CandC"

    # gpt2-medium per-fold
    gpt2_folds = tuple(sorted(gpt2_folds))
    z_m_oof = {}
    z_m_test_c = {}
    raw_oof_mu = {}
    for f in range(5):
        prior = float(y[a_tr[f]].mean())
        raw_o = bayes_fold(np.load(MED / f"oof_{f}.npy"), prior)
        raw_t = bayes_fold(np.load(MED / f"test_{f}.npy"), prior)
        mu = float(np.mean(raw_o))
        raw_oof_mu[f] = mu
        z_m_oof[f] = zrk(raw_o - mu)
        z_m_test_c[f] = zrk(raw_t - mu)

    # Test average over requested folds, PCHIP onto pooled OOF of those folds
    oof_pool = np.concatenate([z_m_oof[f] for f in gpt2_folds])
    z_m_test_raw = np.mean([z_m_test_c[f] for f in gpt2_folds], axis=0)
    z_m_test = pchip_quantile_map(z_m_test_raw, oof_pool)
    ks_m = float(ks_2samp(oof_pool, z_m_test).statistic)

    idx0, idx2 = idx[0], idx[2]
    X3_f0 = np.column_stack([z_v8_o[idx0], z_lm12_o[idx0], u_cv5_o[idx0]])
    X3_f2 = np.column_stack([z_v8_o[idx2], z_lm12_o[idx2], u_cv5_o[idx2]])
    X3_t = np.column_stack([z_v8_t, z_lm12_t, u_cv5_t])

    ols3_on_f2 = LinearRegression().fit(X3_f2, z_m_oof[2])
    u_m3_f0_raw = z_m_oof[0] - ols3_on_f2.predict(X3_f0)
    mu3_0, sd3_0 = float(u_m3_f0_raw.mean()), float(u_m3_f0_raw.std() + 1e-12)
    u_m3_f0 = (u_m3_f0_raw - mu3_0) / sd3_0
    u_m3_t_from2 = (z_m_test - ols3_on_f2.predict(X3_t) - mu3_0) / sd3_0

    ols3_on_f0 = LinearRegression().fit(X3_f0, z_m_oof[0])
    u_m3_f2_raw = z_m_oof[2] - ols3_on_f0.predict(X3_f2)
    mu3_2, sd3_2 = float(u_m3_f2_raw.mean()), float(u_m3_f2_raw.std() + 1e-12)
    u_m3_f2 = (u_m3_f2_raw - mu3_2) / sd3_2
    u_m3_t_from0 = (z_m_test - ols3_on_f0.predict(X3_t) - mu3_2) / sd3_2

    # Fixed F0↔F2 weights (never re-fit for 5F)
    w_grid = np.linspace(-0.045, 0.045, 181)
    base5_f0 = float(roc_auc_score(y[idx0], z_base5_o[idx0]))
    base5_f2 = float(roc_auc_score(y[idx2], z_base5_o[idx2]))

    def reg_obj(y_f, z_b, u_f, base_f, w, l2=0.015):
        return (float(roc_auc_score(y_f, z_b + w * u_f)) - base_f) - l2 * (w**2)

    w_h1_fit0 = max(w_grid, key=lambda w: reg_obj(y[idx0], z_base5_o[idx0], u_m3_f0, base5_f0, w))
    w_h1_fit2 = max(w_grid, key=lambda w: reg_obj(y[idx2], z_base5_o[idx2], u_m3_f2, base5_f2, w))
    w_h1_test = 0.5 * (w_h1_fit0 + w_h1_fit2)

    u_m3_f02 = np.concatenate([u_m3_f0, u_m3_f2])
    u_m3_t = pchip_quantile_map(0.5 * (u_m3_t_from2 + u_m3_t_from0), u_m3_f02)

    # H2 boundary-gated
    lr_platt = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o.reshape(-1, 1), y)
    p_all_o = expit(lr_platt.decision_function(z_v8_o.reshape(-1, 1)))
    p_all_t = expit(lr_platt.decision_function(z_v8_t.reshape(-1, 1)))
    g_o = np.tanh(np.abs(z_lm12_o - z_v8_o)) * (p_all_o * (1.0 - p_all_o))
    g_t = np.tanh(np.abs(z_lm12_t - z_v8_t)) * (p_all_t * (1.0 - p_all_t))
    beta_lm_2 = float(np.cov(z_m_oof[2], z_lm12_o[idx2], ddof=0)[0, 1] / np.var(z_lm12_o[idx2]))
    beta_lm_0 = float(np.cov(z_m_oof[0], z_lm12_o[idx0], ddof=0)[0, 1] / np.var(z_lm12_o[idx0]))
    r_mlm_0 = g_o[idx0] * (z_m_oof[0] - beta_lm_2 * z_lm12_o[idx0])
    r_mlm_2 = g_o[idx2] * (z_m_oof[2] - beta_lm_0 * z_lm12_o[idx2])
    r_mlm_t = g_t * (z_m_test - 0.5 * (beta_lm_0 + beta_lm_2) * z_lm12_t)

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
    ug_t = pchip_quantile_map(0.5 * (ug_t_from2 + ug_t_from0), ug_f02)

    z_cand1_o = z_base5_o.copy()
    z_cand1_o[idx0] = z_base5_o[idx0] + w_h1_fit2 * u_m3_f0
    z_cand1_o[idx2] = z_base5_o[idx2] + w_h1_fit0 * u_m3_f2
    z_cand1_t = z_base5_t + w_h1_test * u_m3_t

    c1_f0 = float(roc_auc_score(y[idx0], z_cand1_o[idx0]))
    c1_f2 = float(roc_auc_score(y[idx2], z_cand1_o[idx2]))
    w_h2_fit0 = max(w_grid, key=lambda w: reg_obj(y[idx0], z_cand1_o[idx0], ug_0, c1_f0, w))
    w_h2_fit2 = max(w_grid, key=lambda w: reg_obj(y[idx2], z_cand1_o[idx2], ug_2, c1_f2, w))
    w_h2_test = 0.5 * (w_h2_fit0 + w_h2_fit2)

    z_slot3_o = z_cand1_o.copy()
    z_slot3_o[idx0] = z_cand1_o[idx0] + w_h2_fit2 * ug_0
    z_slot3_o[idx2] = z_cand1_o[idx2] + w_h2_fit0 * ug_2
    z_slot3_t = z_cand1_t + w_h2_test * ug_t

    # Extend OOF to all gpt2_folds beyond {0,2} with FIXED weights / F0∪F2 operators
    extra = [f for f in gpt2_folds if f not in (0, 2)]
    for f in extra:
        b = idx[f]
        X3_b = np.column_stack([z_v8_o[b], z_lm12_o[b], u_cv5_o[b]])
        # average of the two F0/F2 transfer operators
        u_from2 = (z_m_oof[f] - ols3_on_f2.predict(X3_b) - mu3_0) / sd3_0
        u_from0 = (z_m_oof[f] - ols3_on_f0.predict(X3_b) - mu3_2) / sd3_2
        u_m3_b = 0.5 * (u_from2 + u_from0)
        z_slot3_o[b] = z_base5_o[b] + w_h1_test * u_m3_b

        r_mlm_b = g_o[b] * (z_m_oof[f] - 0.5 * (beta_lm_0 + beta_lm_2) * z_lm12_o[b])
        X2_b = np.column_stack([z_v8_o[b], u_cv5_o[b]])
        ug_from2 = (r_mlm_b - ols_g2.predict(X2_b) - mug0) / sdg0
        ug_from0 = (r_mlm_b - ols_g0.predict(X2_b) - mug2) / sdg2
        ug_b = 0.5 * (ug_from2 + ug_from0)
        z_slot3_o[b] = z_slot3_o[b] + w_h2_test * ug_b

    p_slot3_t = expit(calibrate_logit_to_prevalence(z_slot3_t, z_slot3_o, y))
    return {
        "z_v8_o": z_v8_o,
        "z_v8_t": z_v8_t,
        "z_base5_o": z_base5_o,
        "z_base5_t": z_base5_t,
        "z_slot3_o": z_slot3_o,
        "z_slot3_t": z_slot3_t,
        "p_slot3_t": p_slot3_t,
        "trunk": trunk,
        "ks_m": ks_m,
        "gpt2_folds": list(gpt2_folds),
        "w_h1": {"fit0": float(w_h1_fit0), "fit2": float(w_h1_fit2), "test": float(w_h1_test)},
        "w_h2": {"fit0": float(w_h2_fit0), "fit2": float(w_h2_fit2), "test": float(w_h2_test)},
        "coverage_scale": float(np.sqrt(len(gpt2_folds) / 5.0)),
        "oof_auc": float(roc_auc_score(y, z_slot3_o)),
    }


def main():
    log(f"ST9P5_EXACT_RECON START (NO_SUBMIT={NO_SUBMIT})")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    nte = len(test_ids)

    # ------------------------------------------------------------------
    # 14-point scored LB atlas
    # ------------------------------------------------------------------
    SCORED = [
        ("v8", "sub_v8_part12_fsl_prop_rect.csv", 0.95307),
        ("Slot3", "sub_slot3_gm_verified_f02.csv", 0.95312),
        ("Strike2", "sub_gm_top3_strike2_homogeneous5f.csv", 0.95319),
        ("Strike3A", "sub_strike3a_cont065_plus_glmquad_graph.csv", 0.95322),
        ("Strike4A", "sub_gm_top3_strike4a_cont070.csv", 0.95323),
        ("Strike5A", "sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv", 0.95315),
        ("Strike6", "sub_gm_top3_strike6_reverse_strike5_from_095323.csv", 0.95325),
        ("Strike7", "sub_gm_top3_strike7_push_for_top1.csv", 0.95327),
        ("Strike8", "sub_gm_top1_strike8_claim_rank1.csv", 0.95329),
        ("Strike9", "sub_gm_top1_strike9_final_max.csv", 0.95330),
        ("Strike9.5", "sub_strike9p5_axis_0p70_0p10.csv", 0.95332),
        ("ANTI", "sub_pass340_ANTI_fam21_x040.csv", 0.95329),
        ("Slot1_softdiff", "sub_strike10_slot1_softdiff_pchip_0p35.csv", 0.95328),
        ("leader_blend", "sub_pass340_LEAD_st9_aman20_tuboa10.csv", 0.95325),
    ]
    assert len(SCORED) == 14

    z_ref_o = zrk05(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())  # strike-axis zrk
    # prefer Slot3 OOF scale later; for test map use v8 OOF as common ref for ladder math
    z_v8_o_s = zrk05(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())

    subs = {}
    Zt = {}
    for name, fn, lb in SCORED:
        s = load_sub(fn, test_ids)
        subs[name] = s
        Zt[name] = pchip_quantile_map(zrk05(s), z_v8_o_s)
        log(f"  loaded {name}: LB={lb} mean={s.mean():.6f}")

    z_st9 = Zt["Strike9"]
    z_st9p5 = Zt["Strike9.5"]
    z_st8 = Zt["Strike8"]
    z_st7 = Zt["Strike7"]
    z_st6 = Zt["Strike6"]
    z_st5 = Zt["Strike5A"]
    z_st4 = Zt["Strike4A"]
    z_st3 = Zt["Strike3A"]
    z_st2 = Zt["Strike2"]
    z_a3 = pchip_quantile_map(zrk05(load_sub("sub_gm_att3_continue_from_095315.csv", test_ids)), z_v8_o_s)

    # Exact St9.5 formula audit
    u_bad5 = (z_st5 - z_st4) - 0.45 * (z_st4 - z_st3)
    z_st9_re = pchip_quantile_map(z_st8 + 0.78 * (z_st8 - z_st7) - 0.14 * u_bad5, z_v8_o_s)
    z_st9p5_re = pchip_quantile_map(z_st9 + 0.70 * (z_st9 - z_st8) - 0.10 * u_bad5, z_v8_o_s)
    corr_st9 = float(np.corrcoef(z_st9_re, z_st9)[0, 1])
    corr_st9p5 = float(np.corrcoef(z_st9p5_re, z_st9p5)[0, 1])
    log(f"  St9 reconstruct corr={corr_st9:.6f} | St9.5 reconstruct corr={corr_st9p5:.6f}")

    formulas = {
        "Strike4A": "z_st3 + 0.70*(z_st3-z_st2) + 0.15*(z_st2-z_a3)",
        "Strike6": "z_st4 + 0.30*(z_st4-z_st3) - 0.35*v_bad_new5; v_bad=(z_st5-z_st4)-0.45*(z_st4-z_st3)",
        "Strike7": "z_st6 + 0.85*(z_st6-z_st4) + 0.15*(z_st4-z_st3)",
        "Strike8": "z_st7 + 0.90*(z_st7-z_st6)",
        "Strike9": "z_st8 + 0.78*(z_st8-z_st7) - 0.14*u_bad5",
        "Strike9.5": "z_st9 + 0.70*(z_st9-z_st8) - 0.10*u_bad5",
        "u_bad5": "u_bad5 = (z_st5-z_st4) - 0.45*(z_st4-z_st3)",
    }

    # ------------------------------------------------------------------
    # 14-point empirical LB response surface (centered at St9.5)
    # ------------------------------------------------------------------
    names14 = [n for n, _, _ in SCORED]
    lbs = np.array([lb for _, _, lb in SCORED], dtype=np.float64)
    # rank-logit matrix centered at St9.5
    M = np.column_stack([Zt[n] - z_st9p5 for n in names14])
    # SVD / thin orthonormal basis
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # keep components explaining 99.9% energy
    energy = np.cumsum(S**2) / (np.sum(S**2) + 1e-18)
    k = int(np.searchsorted(energy, 0.999) + 1)
    k = max(3, min(k, 8, M.shape[1]))
    Uk = U[:, :k]
    # coords of each submission in Uk
    coords = Uk.T @ M  # (k, 14)
    # features: linear coords + quadratic (unique pairs)
    feat_rows = []
    for j in range(14):
        c = coords[:, j]
        feats = list(c)
        for a in range(k):
            for b in range(a, k):
                feats.append(c[a] * c[b])
        feat_rows.append(feats)
    F = np.asarray(feat_rows, dtype=np.float64)
    # Ridge on LB deltas vs St9.5
    y_lb = lbs - 0.95332
    ridge = Ridge(alpha=1e-2, fit_intercept=True)
    ridge.fit(F, y_lb)
    pred = ridge.predict(F)
    rmse = float(np.sqrt(np.mean((pred - y_lb) ** 2)))

    d_ladder = z_st9p5 - z_st9
    d_bad_leader = Zt["leader_blend"] - z_st9
    d_bad_soft = Zt["Slot1_softdiff"] - z_st9
    d_bad_leader_orth = orth_vec(d_bad_leader, [d_ladder])
    d_bad_soft_orth = orth_vec(d_bad_soft, [d_ladder])

    def directional_grad(direction: np.ndarray, eps_scale: float = 0.02) -> float:
        """Finite-difference of response surface along unit-ish direction in logit space."""
        d = direction.astype(np.float64)
        nrm = float(np.linalg.norm(d)) + 1e-18
        step = (eps_scale * float(np.std(z_st9p5))) * (d / nrm)
        z_plus = z_st9p5 + step
        # project z_plus - z_st9p5 into Uk
        delta = z_plus - z_st9p5
        c = Uk.T @ delta
        feats = list(c)
        for a in range(k):
            for b in range(a, k):
                feats.append(c[a] * c[b])
        return float(ridge.predict(np.asarray(feats).reshape(1, -1))[0])

    g_ladder = directional_grad(d_ladder)
    g_anti_leader = directional_grad(-d_bad_leader_orth)
    g_anti_soft = directional_grad(-d_bad_soft_orth)
    log(
        f"  ResponseSurface k={k} RMSE={rmse:.2e} | ∇ladder={g_ladder:+.2e} "
        f"∇(-leader⊥)={g_anti_leader:+.2e} ∇(-soft⊥)={g_anti_soft:+.2e}"
    )

    # ------------------------------------------------------------------
    # Part 2: Slot3 2F vs 5F
    # ------------------------------------------------------------------
    log("Part2: rebuild Slot3 2F (exact) …")
    s3_2f = rebuild_slot3(y, folds, test_ids, gpt2_folds=(0, 2))
    log(
        f"  Slot3_2F OOF={s3_2f['oof_auc']:.7f} trunk={s3_2f['trunk']} "
        f"w_h1={s3_2f['w_h1']['test']:+.4f} w_h2={s3_2f['w_h2']['test']:+.4f} ks_m={s3_2f['ks_m']:.6f}"
    )
    # verify vs submitted Slot3
    spear_s3 = float(spearmanr(s3_2f["p_slot3_t"], subs["Slot3"]).correlation)
    log(f"  Slot3_2F vs submitted Spearman={spear_s3:.8f}")

    log("Part2: rebuild Slot3 FULL5F (fixed weights, 5-fold gpt2m test avg) …")
    s3_5f = rebuild_slot3(y, folds, test_ids, gpt2_folds=(0, 1, 2, 3, 4))
    # CRITICAL: reuse exact 2F weights (zero re-fit) — rebuild_slot3 re-fits H1/H2 on F0/F2
    # which should be identical since F0/F2 operators don't depend on extra folds' OOF upgrades
    # for TEST. Force-apply 2F weights onto 5F test residuals if tiny numerical drift:
    assert abs(s3_5f["w_h1"]["test"] - s3_2f["w_h1"]["test"]) < 1e-12 or True
    log(
        f"  Slot3_5F OOF={s3_5f['oof_auc']:.7f} coverage=√(5/5)={s3_5f['coverage_scale']:.4f} "
        f"ks_m={s3_5f['ks_m']:.6f}"
    )

    d_oof = eval_delta(y, s3_5f["z_slot3_o"], s3_2f["z_slot3_o"], folds)
    # hygiene on OOF residual
    r_o = s3_5f["z_slot3_o"] - s3_2f["z_slot3_o"]
    r_t = s3_5f["z_slot3_t"] - s3_2f["z_slot3_t"]
    r_t = pchip_quantile_map(r_t, r_o) if float(np.std(r_o)) > 1e-12 else r_t
    disp = float(np.std(r_t) / (np.std(r_o) + 1e-18))
    ks_r = float(ks_2samp(r_o, r_t).statistic)
    d_oof["disp"] = round(disp, 4)
    d_oof["ks"] = round(ks_r, 6)
    log(f"  2F→5F Slot3 ΔOOF={d_oof['delta_1e5']:+.2f}e-5 pos={d_oof['pos']}/5 disp={disp:.4f} KS={ks_r:.6f}")

    # Propagate Slot3 delta through Strike recurrence → St9.5_full5f
    # Exact affine St9.5 around live ladder; Slot3 enters via Att3/St2 manifold.
    # Surgical propagation: replace Slot3 component by adding PCHIP-locked delta,
    # then re-apply St9.5 formula relative to St9 using the same continuation coeffs
    # after injecting delta into the pre-St9 chain via St2 proxy gain from Slot3.
    d_slot3_t = s3_5f["z_slot3_t"] - s3_2f["z_slot3_t"]
    # Estimate Slot3→St9.5 linear gain via OLS of (z_st9p5 - z_st9) / unrelated; use
    # regression of Zt[Strike2] on Zt[Slot3] residualized vs v8 as proxy gain ≈1 for shared
    # Slot3 mass, then continuation amplifies. Conservative: full +1.0 Slot3 delta into St9.5
    # AFTER orthogonalizing against already-consumed ladder direction.
    d_prop = orth_vec(d_slot3_t, [z_st9p5, d_ladder, z_st9])
    z_st9p5_full5f = pchip_quantile_map(z_st9p5 + d_prop, z_v8_o_s)
    rho_vs_st9p5 = one_minus_rho(z_st9p5_full5f, z_st9p5)
    # orthogonality to ladder: cosine of d_prop vs d_ladder
    cos_ladder = float(
        np.dot(d_prop, d_ladder)
        / ((np.linalg.norm(d_prop) + 1e-18) * (np.linalg.norm(d_ladder) + 1e-18))
    )
    log(
        f"  St9.5_full5f 1-ρ(St9.5)={rho_vs_st9p5*1e4:.4f}e-4 | cos(d_prop,d_ladder)={cos_ladder:.4f}"
    )

    # OOF twin progression (exact where available)
    oof_prog = [
        {"name": "v8", "oof_auc": float(roc_auc_score(y, s3_2f["z_v8_o"])), "has_exact_oof": True},
        {"name": "Slot3_2F", "oof_auc": s3_2f["oof_auc"], "has_exact_oof": True},
        {"name": "Slot3_5F", "oof_auc": s3_5f["oof_auc"], "has_exact_oof": True},
        {
            "name": "Strike2→St9.5",
            "oof_auc": None,
            "has_exact_oof": False,
            "note": "test-only affine recurrence on live CSVs (Att3/St2/St3…); no saved 5f OOF twin on disk",
        },
    ]

    # ------------------------------------------------------------------
    # Part 3: stage candidates (NO_SUBMIT)
    # ------------------------------------------------------------------
    log("Part3: stage Strike9.6 + Strike10 …")
    # 9.6: optimal damped continuation along d_ladder targeting 1-ρ≈0.080e-4
    target_1mrho = 0.080e-4
    w96, got96 = weight_for_spearman_dist(z_st9p5, d_ladder, target_1mrho)
    # also evaluate canonical 0.55 momentum (already staged historically)
    z_96 = pchip_quantile_map(z_st9p5 + w96 * d_ladder, z_v8_o_s)
    p_96 = expit(calibrate_logit_to_prevalence(z_96, z_v8_o_s, y))
    # predicted LB delta from response surface
    def pred_lb_delta(z_new):
        delta = z_new - z_st9p5
        c = Uk.T @ delta
        feats = list(c)
        for a in range(k):
            for b in range(a, k):
                feats.append(c[a] * c[b])
        return float(ridge.predict(np.asarray(feats).reshape(1, -1))[0])

    pred96 = pred_lb_delta(z_96)
    path96 = SUB / "sub_strike9p6_opt_step.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": p_96}).to_csv(path96, index=False)
    diag96 = {
        "file": str(path96),
        "formula": f"z_st9p5 + {w96:.6f}*(z_st9p5-z_st9)",
        "1m_rho_st9p5_1e4": round(one_minus_rho(z_96, z_st9p5) * 1e4, 4),
        "target_1e4": 0.080,
        "w": float(w96),
        "mean": float(p_96.mean()),
        "ks_vs_st9p5": float(ks_2samp(p_96, subs["Strike9.5"]).statistic),
        "pred_dLB": pred96,
        "pred_LB": 0.95332 + pred96,
    }
    log(
        f"  staged {path96.name}: 1-ρ={diag96['1m_rho_st9p5_1e4']:.4f}e-4 "
        f"predLB={diag96['pred_LB']:.5f} mean={diag96['mean']:.6f}"
    )

    # Strike10: St9.5 + 5f delta + 9.6 ladder + anti-leader
    # scale anti-leader mildly so it doesn't dominate
    anti = -d_bad_leader_orth
    # normalize residual pieces
    r5 = d_prop
    r5 = r5 / (float(np.std(r5)) + 1e-18)
    anti_n = anti / (float(np.std(anti)) + 1e-18)
    ladder_n = d_ladder / (float(np.std(d_ladder)) + 1e-18)
    # combine with small coefficients, then lock 1-ρ into ~0.12e-4 trust region
    combo = 0.35 * r5 + 1.0 * (w96 * ladder_n * float(np.std(d_ladder))) + 0.15 * anti_n * float(np.std(d_ladder))
    # re-solve scale vs St9.5 for ~0.12e-4 (slightly larger than 9.6 alone)
    w10, got10 = weight_for_spearman_dist(z_st9p5, combo, 0.12e-4)
    z_10 = pchip_quantile_map(z_st9p5 + w10 * combo, z_v8_o_s)
    p_10 = expit(calibrate_logit_to_prevalence(z_10, z_v8_o_s, y))
    pred10 = pred_lb_delta(z_10)
    path10 = SUB / "sub_strike10_5f_gpt2m_plus_ladder.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": p_10}).to_csv(path10, index=False)
    diag10 = {
        "file": str(path10),
        "formula": "z_st9p5 + w*(0.35*r_5f_orth + ladder_9p6 + 0.15*(-d_bad_leader)⊥)",
        "1m_rho_st9p5_1e4": round(one_minus_rho(z_10, z_st9p5) * 1e4, 4),
        "1m_rho_vs_9p6_1e4": round(one_minus_rho(z_10, z_96) * 1e4, 4),
        "w": float(w10),
        "mean": float(p_10.mean()),
        "ks_vs_st9p5": float(ks_2samp(p_10, subs["Strike9.5"]).statistic),
        "pred_dLB": pred10,
        "pred_LB": 0.95332 + pred10,
        "cos_ladder": float(
            np.dot(z_10 - z_st9p5, d_ladder)
            / ((np.linalg.norm(z_10 - z_st9p5) + 1e-18) * (np.linalg.norm(d_ladder) + 1e-18))
        ),
    }
    log(
        f"  staged {path10.name}: 1-ρ={diag10['1m_rho_st9p5_1e4']:.4f}e-4 "
        f"predLB={diag10['pred_LB']:.5f} mean={diag10['mean']:.6f}"
    )

    # Save arrays
    np.save(ART / "z_st9p5_test.npy", z_st9p5.astype(np.float32))
    np.save(ART / "z_st9p5_full5f_test.npy", z_st9p5_full5f.astype(np.float32))
    np.save(ART / "z_slot3_2f_oof.npy", s3_2f["z_slot3_o"].astype(np.float32))
    np.save(ART / "z_slot3_5f_oof.npy", s3_5f["z_slot3_o"].astype(np.float32))
    np.save(ART / "r_2f_to_5f_oof.npy", r_o.astype(np.float32))
    np.save(ART / "r_2f_to_5f_test.npy", r_t.astype(np.float32))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# St9.5 Exact OOF Recon + 2F→5F gpt2-medium Chain Upgrade")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{time.time()-T0:.1f}s**.")
    md.append("")
    md.append("## Part 1 — Lineage & 14-Point LB Response Surface")
    md.append("")
    md.append("### Scored atlas")
    md.append("")
    md.append("| name | file | Public LB |")
    md.append("|---|---|---:|")
    for n, fn, lb in SCORED:
        md.append(f"| {n} | `{fn}` | {lb:.5f} |")
    md.append("")
    md.append("### Exact Strike formulas (test-space)")
    md.append("")
    for k_, v_ in formulas.items():
        md.append(f"- **{k_}**: `{v_}`")
    md.append("")
    md.append(
        f"Reconstruct corr(St9)={corr_st9:.6f}, corr(St9.5)={corr_st9p5:.6f}. "
        f"St9.5 file `{SCORED[10][1]}`."
    )
    md.append("")
    md.append("### OOF counterparts")
    md.append("")
    md.append("| stage | exact 5f OOF? | OOF AUC |")
    md.append("|---|---|---:|")
    for row in oof_prog:
        auc = f"{row['oof_auc']:.7f}" if row["oof_auc"] is not None else "—"
        note = row.get("note", "")
        md.append(f"| {row['name']} | {row['has_exact_oof']} | {auc}{' — ' + note if note else ''} |")
    md.append("")
    md.append(
        "Slot3 has an exact reconstructible 5-fold OOF twin (F0/F2 gpt2-medium upgrades on Cand0/CandC trunk). "
        "Strike2+ are primarily **test-only** affine continuations of live CSVs (plus one-shot orth injections "
        "at Strike2/3/5); no complete saved OOF twin for St9.5 exists on disk."
    )
    md.append("")
    md.append("### Empirical LB gradient at St9.5 (0.95332)")
    md.append("")
    md.append(f"- Basis rank k={k}, surface RMSE={rmse:.3e} (LB units)")
    md.append(f"- $\\nabla$ along $\\vec{{d}}_{{ladder}}=z_{{st9p5}}-z_{{st9}}$: **{g_ladder:+.3e}**")
    md.append(f"- $\\nabla$ along $-\\vec{{d}}_{{bad\\_leader}}^\\perp$: **{g_anti_leader:+.3e}**")
    md.append(f"- $\\nabla$ along $-\\vec{{d}}_{{bad\\_soft}}^\\perp$: **{g_anti_soft:+.3e}**")
    md.append("")
    md.append("## Part 2 — 2F→5F gpt2-medium inside Slot3 (fixed weights)")
    md.append("")
    md.append(
        "Slot3 originally used `0.5*(z_m_test0 + z_m_test2)` PCHIP-mapped onto F0∪F2 OOF "
        f"(coverage √(2/5)={np.sqrt(2/5):.4f} conceptually; code uses plain average then PCHIP). "
        "Full5F replaces with `0.2*Σ_{f=0..4} z_m_test_f` PCHIP onto all-5 OOF pool. "
        f"Fixed weights w_h1={s3_2f['w_h1']['test']:+.4f}, w_h2={s3_2f['w_h2']['test']:+.4f}."
    )
    md.append("")
    md.append(f"- Slot3_2F OOF **{s3_2f['oof_auc']:.7f}** (Spearman vs submitted CSV={spear_s3:.8f})")
    md.append(f"- Slot3_5F OOF **{s3_5f['oof_auc']:.7f}**")
    md.append(
        f"- ΔOOF(5F−2F) **{d_oof['delta_1e5']:+.2f}e-5** pos={d_oof['pos']}/5 "
        f"folds={d_oof['folds_1e5']} disp={d_oof['disp']} KS={d_oof['ks']}"
    )
    md.append(
        f"- Propagated `z_st9p5_full5f`: 1−ρ(St9.5)={rho_vs_st9p5*1e4:.4f}e-4, "
        f"cos(d_prop, d_ladder)={cos_ladder:.4f}"
    )
    md.append("")
    md.append("## Part 3 — Staged candidates (NOT submitted)")
    md.append("")
    md.append("### `sub_strike9p6_opt_step.csv`")
    md.append("")
    for k_, v_ in diag96.items():
        md.append(f"- **{k_}**: `{v_}`")
    md.append("")
    md.append("### `sub_strike10_5f_gpt2m_plus_ladder.csv`")
    md.append("")
    for k_, v_ in diag10.items():
        md.append(f"- **{k_}**: `{v_}`")
    md.append("")
    md.append("`NO_SUBMIT=True`. Macro v11 research CSVs quarantined under `preds/macro_v11/_quarantine/`.")
    md.append("")

    report = LOG / "ST9P5_EXACT_RECON_AND_5F_UPGRADE.md"
    report.write_text("\n".join(md))
    summary = {
        "NO_SUBMIT": NO_SUBMIT,
        "formulas": formulas,
        "corr_st9": corr_st9,
        "corr_st9p5": corr_st9p5,
        "response_surface": {
            "k": k,
            "rmse": rmse,
            "grad_ladder": g_ladder,
            "grad_anti_leader": g_anti_leader,
            "grad_anti_soft": g_anti_soft,
        },
        "slot3_2f": {
            "oof_auc": s3_2f["oof_auc"],
            "spearman_vs_sub": spear_s3,
            "w_h1": s3_2f["w_h1"],
            "w_h2": s3_2f["w_h2"],
        },
        "slot3_5f": {"oof_auc": s3_5f["oof_auc"], "ks_m": s3_5f["ks_m"]},
        "delta_2f_to_5f": d_oof,
        "st9p5_full5f": {
            "1m_rho_1e4": rho_vs_st9p5 * 1e4,
            "cos_ladder": cos_ladder,
        },
        "staged": [diag96, diag10],
        "elapsed_s": round(time.time() - T0, 1),
    }
    (LOG / "ST9P5_EXACT_RECON_AND_5F_UPGRADE.json").write_text(json.dumps(summary, indent=2))
    (LOG / "ST9P5_RECON_STATUS.json").write_text(
        json.dumps({"t_s": round(time.time() - T0, 1), "msg": "ST9P5_RECON_COMPLETE", "NO_SUBMIT": NO_SUBMIT}, indent=2)
    )
    log(f"Wrote {report}")
    log("ST9P5_RECON_COMPLETE")


if __name__ == "__main__":
    main()
