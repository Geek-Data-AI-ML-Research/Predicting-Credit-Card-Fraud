#!/usr/bin/env python3
"""GM v8 — Exact-L7 Stage-B interaction boosters + Fold-0 macro peak (0.95365+ ladder).

STRICT NO_SUBMIT=True. Sequential float32 loads + del/gc; soft RSS target < 600 MB.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

VOL = Path(os.environ.get("GM_V8_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
WORK = Path(os.environ.get("GM_V8_ROOT", "/tmp/gm_deep_tech"))
DS = Path(
    os.environ.get(
        "GM_V8_DS",
        "/tmp/gm_v6_audit/external/nov11_095347_recon/datasets",
    )
)
PRED = VOL / "preds"
ART = WORK / "preds/gm_v8_real_l7"
ART_V7 = WORK / "preds/gm_deep_tech_v7"
SUB = WORK / "submissions"
VOL_SUB = VOL / "submissions"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
VOL_ART = VOL / "preds/gm_v8_real_l7"
for p in (ART, SUB, LOG):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT
EPS = 1e-6
RSS_SOFT_MB = 600
T0 = time.time()
ATH = 0.95354

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def rss_mb() -> float:
    # macOS: ru_maxrss is bytes → convert to MiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def rss_curr_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    # macOS fallback via ps
    try:
        import subprocess

        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return rss_mb()


def log(msg: str) -> None:
    print(
        f"[{time.time()-T0:6.1f}s|RSS={rss_curr_mb():.0f}/{rss_mb():.0f}MB] {msg}",
        flush=True,
    )
    if rss_curr_mb() > RSS_SOFT_MB:
        print(f"  WARN curr RSS {rss_curr_mb():.0f}MB > soft cap {RSS_SOFT_MB}", flush=True)


def purge(*_objs):
    gc.collect()


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).ravel()


def first_existing(*cands: Path) -> Path:
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(cands)


def zrk_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    z = logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS))
    return z.astype(np.float32)


def unitize_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def orth_against_f32(v: np.ndarray, bases) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = np.asarray(b, dtype=np.float64).ravel()
        b = b / (float(np.linalg.norm(b)) + 1e-18)
        out -= float(np.dot(out, b)) * b
    return out.astype(np.float32)


def fold_aucs(y, z, folds):
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def _jitter(x, scale=1e-11):
    x = np.asarray(x, dtype=np.float64)
    return x + scale * (rankdata(x, method="ordinal") / len(x) - 0.5)


def force_exact_law(oof_vec, test_vec):
    o = unitize_f32(oof_vec).astype(np.float64)
    t = np.asarray(test_vec, dtype=np.float64).ravel()
    o_s = np.sort(o)
    n_o, n_t = len(o_s), len(t)
    order = np.argsort(_jitter(t, 1e-7), kind="mergesort")
    pos = np.empty(n_t, dtype=np.int64)
    pos[order] = np.minimum(((np.arange(n_t) + 0.5) * n_o / n_t).astype(np.int64), n_o - 1)
    t_al = o_s[pos]
    t_al = t_al - float(t_al.mean())
    t_al = t_al * (float(o.std()) / (float(t_al.std()) + 1e-12))
    return o.astype(np.float32), t_al.astype(np.float32)


def align_method_b_smooth(oof_vec, test_vec, folds):
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = np.zeros_like(oof_vec)
    for a, b in folds:
        mu, sd = float(oof_vec[b].mean()), float(oof_vec[b].std() + 1e-12)
        oof_std[b] = (oof_vec[b] - mu) / sd
    oof_std = (oof_std - oof_std.mean()) / (oof_std.std() + 1e-12)
    test_std = (test_vec - test_vec.mean()) / (test_vec.std() + 1e-12)
    q = np.linspace(0.0, 1.0, 2001)
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std), q),
        np.quantile(_jitter(oof_std), q),
        extrapolate=True,
    )
    test_al = pchip(_jitter(test_std))
    test_al = (test_al - test_al.mean()) / (test_al.std() + 1e-12)
    if float(ks_2samp(oof_std, test_al).statistic) >= 0.0001:
        return force_exact_law(oof_std, test_std)
    return oof_std.astype(np.float32), test_al.astype(np.float32)


def align_f0_to_test(o_f0: np.ndarray, t_raw: np.ndarray):
    """Align test law to Fold-0 OOF vector (exact order-stat map)."""
    return force_exact_law(o_f0, t_raw)


def spearman_dist(a, b) -> float:
    return float(1.0 - spearmanr(a, b).correlation)


def weight_for_spearman_dist(z0, u, target, hi=120.0, n=56):
    z0 = np.asarray(z0, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-18)
    lo = 0.0
    best_w, best_err, got = 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got = err, mid, d
        if d > target:
            hi = mid
        else:
            lo = mid
    return float(best_w), float(got)


def pchip_to_st9(score, p_st9):
    score = np.asarray(score, dtype=np.float64).ravel()
    sorted_st9 = np.sort(np.asarray(p_st9, dtype=np.float64).ravel())
    n = len(score)
    pchip = PchipInterpolator(np.linspace(0, 1, n), sorted_st9, extrapolate=True)
    order = np.argsort(_jitter(score, 1e-7), kind="mergesort")
    out = np.empty(n)
    out[order] = pchip(np.linspace(0, 1, n))
    return np.clip(out, 1e-12, 1 - 1e-12)


def to_probs_0167(z, p_st9):
    p = pchip_to_st9(z, p_st9)
    zl = logit(np.clip(p, 1e-12, 1 - 1e-12))

    def obj(d):
        return (float(np.mean(expit(zl + d))) - 0.016700) ** 2

    d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
    return expit(zl + d).astype(np.float64)


def tree_logit_residual(p_model, z_margin):
    p = np.clip(np.asarray(p_model, dtype=np.float64).ravel(), 1e-6, 1 - 1e-6)
    zm = np.asarray(z_margin, dtype=np.float64).ravel()
    X = np.column_stack([np.ones(len(zm)), zm])
    coef, *_ = np.linalg.lstsq(X, logit(p), rcond=None)
    tree = logit(p) - (coef[0] + coef[1] * zm)
    return unitize_f32(tree)


def lofo_resid(y_vec, bases, folds):
    y_vec = np.asarray(y_vec, dtype=np.float64).ravel()
    B = np.column_stack([unitize_f32(b).astype(np.float64) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize_f32(resid)


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    za = np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1)
    ya = np.asarray(y_a, dtype=np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, dtype=np.float64).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def llr_from_lm(arr: np.ndarray) -> np.ndarray:
    """arr (..., 2). Negative median ⇒ log-probs; else probabilities."""
    arr = np.asarray(arr)
    if float(np.nanmedian(arr)) < 0:
        return arr[..., 1] - arr[..., 0]
    p0 = np.clip(arr[..., 0], 1e-12, 1.0)
    p1 = np.clip(arr[..., 1], 1e-12, 1.0)
    return np.log(p1) - np.log(p0)


def bayes_llr(llr_perm: np.ndarray) -> np.ndarray:
    logp1 = -np.logaddexp(0, -llr_perm)
    logp0 = -np.logaddexp(0, llr_perm)
    lp1 = logsumexp(logp1, axis=1) - np.log(llr_perm.shape[1])
    lp0 = logsumexp(logp0, axis=1) - np.log(llr_perm.shape[1])
    return lp1 - lp0


def poly2_cross(X: np.ndarray) -> np.ndarray:
    """2nd-order poly + pairwise products (no cubes). float32 out."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    cols = [X]
    cols.append(X * X)
    # pairwise (i<j) — keep memory modest: only if d is small
    if d <= 16:
        pair = []
        for i in range(d):
            for j in range(i + 1, d):
                pair.append(X[:, i] * X[:, j])
        if pair:
            cols.append(np.column_stack(pair).astype(np.float32))
    return np.hstack(cols).astype(np.float32)


def main():
    log("======== GM v8 REAL-L7 STAGE-B / 0.95365 LEAP (NO_SUBMIT) ========")
    log(f"soft RSS cap={RSS_SOFT_MB}MB OMP={os.environ.get('OMP_NUM_THREADS')}")

    # ---------- labels / folds ----------
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"]
    )["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    idx_f0 = np.where(m0)[0]
    ntr = len(y)
    nte = 299984
    n_f0 = int(m0.sum())
    assert n_f0 == 139992, n_f0
    y_f0 = y[idx_f0]
    log(f"ntr={ntr} n_f0={n_f0} fraud_f0={int(y_f0.sum())} rate={float(y_f0.mean()):.6f}")

    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize_f32(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    z_ref_t = unitize_f32(load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy"))
    z_ref_auc = float(roc_auc_score(y, z_ref.astype(np.float64)))
    base_folds = fold_aucs(y, z_ref, folds)
    base0 = float(roc_auc_score(y[m0], z_ref[m0].astype(np.float64)))
    log(f"z_ref_old AUC={z_ref_auc:.7f}")

    d1_o = unitize_f32(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    d1_t = unitize_f32(load_f32(PRED / "gm_v4_polish/d1_v4_test.npy"))

    # ---------- tree_logit (prefer v7.2 cache; else rebuild extratrees+monotone) ----------
    try:
        tree_o = load_f32(ART_V7 / "tree_logit_v72_oof.npy")
        tree_t = load_f32(ART_V7 / "tree_logit_v72_test.npy")
        tree_src = "v72_cache"
    except Exception:
        tree_src = "rebuild_extratrees+monotone"
        p_lgb_o = load_f32(PRED / "gm_v5_overnight/lgb_stageB_extratrees_bag6_oof.npy")
        p_lgb_t = load_f32(PRED / "gm_v5_overnight/lgb_stageB_extratrees_bag6_test.npy")
        p_xgb_o = load_f32(PRED / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_oof.npy")
        p_xgb_t = load_f32(PRED / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_test.npy")
        v5_o = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v5_harvest_oof.npy")))
        v5_t = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v5_harvest_test.npy")))
        z8_o = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v8_part12_rect_oof.npy")))
        z8_t = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v8_part12_rect_test.npy")))
        t_lgb_o = tree_logit_residual(p_lgb_o, v5_o)
        t_lgb_t = tree_logit_residual(p_lgb_t, v5_t)
        del p_lgb_o, p_lgb_t
        t_xgb_o = unitize_f32(
            0.5 * tree_logit_residual(p_xgb_o, v5_o).astype(np.float64)
            + 0.5 * tree_logit_residual(p_xgb_o, z8_o).astype(np.float64)
        )
        t_xgb_t = unitize_f32(
            0.5 * tree_logit_residual(p_xgb_t, v5_t).astype(np.float64)
            + 0.5 * tree_logit_residual(p_xgb_t, z8_t).astype(np.float64)
        )
        del p_xgb_o, p_xgb_t, v5_o, v5_t, z8_o, z8_t
        tree_o = unitize_f32(0.55 * t_lgb_o.astype(np.float64) + 0.45 * t_xgb_o.astype(np.float64))
        tree_t = unitize_f32(0.55 * t_lgb_t.astype(np.float64) + 0.45 * t_xgb_t.astype(np.float64))
        del t_lgb_o, t_lgb_t, t_xgb_o, t_xgb_t
        tree_o = lofo_resid(tree_o, [z_ref, d1_o], folds)
        tree_t = orth_against_f32(tree_t, [z_ref_t, d1_t])
        tree_o, tree_t = align_method_b_smooth(tree_o, tree_t, folds)
        purge()
    log(f"tree_logit source={tree_src}")

    # ---------- CLEAN u_stageB_plus (NO Block3!) ----------
    u_sb_o = unit_f32(d1_o)
    u_sb_t = unit_f32(load_f32(PRED / "gm_breakout/u_stageB_5f.npy"))
    if float(np.dot(u_sb_t.astype(np.float64), d1_t.astype(np.float64))) < 0:
        u_sb_t = (-u_sb_t.astype(np.float64)).astype(np.float32)

    u_plus_o = unit_f32(
        0.78 * u_sb_o.astype(np.float64) + 0.22 * unit_f32(tree_o).astype(np.float64)
    )
    u_plus_t = unit_f32(
        0.78 * u_sb_t.astype(np.float64) + 0.22 * unit_f32(tree_t).astype(np.float64)
    )
    u_plus_o, u_plus_t = align_method_b_smooth(u_plus_o, u_plus_t, folds)
    dks = float(ks_2samp(u_plus_o.astype(np.float64), u_plus_t.astype(np.float64)).statistic)
    log(f"u_stageB_plus CLEAN (no Block3) aligned KS={dks:.6f}")
    np.save(ART / "u_stageB_plus_clean_oof.npy", u_plus_o)
    np.save(ART / "u_stageB_plus_clean_test.npy", u_plus_t)

    # ---------- S7 proxy OOF ----------
    gpt_path = PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"
    gpt_o = unitize_f32(load_f32(gpt_path)) if gpt_path.exists() else np.zeros(ntr, np.float32)
    m1_o = unitize_f32(
        0.65 * u_sb_o.astype(np.float64)
        - 0.35 * (-unit_f32(gpt_o)).astype(np.float64)
        - 0.20 * unitize_f32(gpt_o).astype(np.float64)
    )
    m2_o = unitize_f32(0.65 * u_sb_o.astype(np.float64) + 0.35 * u_sb_o.astype(np.float64))
    s6_o = unitize_f32(0.70 * m1_o.astype(np.float64) + 0.30 * m2_o.astype(np.float64))
    w_s7 = 0.055
    del m1_o, m2_o
    purge()

    def oof_pack(z):
        fa = fold_aucs(y, z, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
        d0 = (float(roc_auc_score(y[m0], z[m0])) - base0) * 1e5
        # folds 1-4 mean lift vs base
        d14 = float(np.mean([d5[i] for i in range(5) if i != 0])) if False else float(
            np.mean([(fa[i] - base_folds[i]) * 1e5 for i in range(1, 5)])
        )
        # map fold index carefully: folds list order is StratifiedKFold order, not fold_id
        # For Stage-B proxy on non-F0: mean of all folds except the fold containing most m0 overlap
        return {
            "lift5": (float(roc_auc_score(y, z)) - z_ref_auc) * 1e5,
            "d5_mean": float(np.mean(d5)),
            "d0": d0,
            "d14": d14,
            "pos": int(sum(x > 0 for x in d5)),
            "d5": d5,
            "fa": fa,
        }

    # Identify which CV fold index aligns with fold_id==0 (goodpjw)
    f0_overlap = []
    for i, (_, b) in enumerate(folds):
        f0_overlap.append(float(np.mean(m0[b])))
    f0_cv_i = int(np.argmax(f0_overlap))
    log(f"fold_id==0 overlaps CV folds: {[round(x,3) for x in f0_overlap]} → use CV[{f0_cv_i}] for F0")

    z_s7_oof = z_ref.astype(np.float64) + w_s7 * unitize_f32(s6_o).astype(np.float64)
    s7 = oof_pack(z_s7_oof)
    # redefine d0 using goodpjw fold_id==0 (true L7 support)
    s7["d0"] = (float(roc_auc_score(y[m0], z_s7_oof[m0])) - base0) * 1e5
    s7_f14 = []
    for i, (_, b) in enumerate(folds):
        if i == f0_cv_i:
            continue
        s7_f14.append((float(roc_auc_score(y[b], z_s7_oof[b])) - base_folds[i]) * 1e5)
    s7["d14"] = float(np.mean(s7_f14))
    log(f"S7_proxy: 5F={s7['lift5']:+.2f}e-5 F0={s7['d0']:+.2f}e-5 F14={s7['d14']:+.2f}e-5 pos={s7['pos']}/5")

    # =====================================================================
    # PART 1 — macro Fold-0 sweep Ray A / B_clean / C_opt
    # =====================================================================
    log("======== PART1 Fold-0 macro sweep RayA / B_clean / C_opt ========")
    steps = [
        0.10e-4,
        0.20e-4,
        0.35e-4,
        0.50e-4,
        0.75e-4,
        1.00e-4,
        1.35e-4,
        1.75e-4,
        2.25e-4,
        2.50e-4,
    ]
    dw_per = 0.08  # weight per 1e-4 of 1-rho (same scale as v7.2)

    # Test-side rays for pred_lin / t / 1-rho(areeb)
    z_s7 = unitize_f32(
        zrk_f32(
            pd.read_csv(VOL_SUB / "sub_master_slot7_verified_s7A.csv", usecols=["is_fraud"])[
                "is_fraud"
            ].to_numpy(np.float64)
        )
    )
    m1 = unit_f32(load_f32(PRED / "gm_breakout/m1.npy"))
    m2 = unit_f32(load_f32(PRED / "gm_breakout/m2.npy"))
    u_ray_a = unit_f32(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    u_3lad = unit_f32(load_f32(PRED / "gm_breakout/u_3ladder_grad.npy"))
    u_gpt = unit_f32(load_f32(PRED / "gm_breakout/u_gpt2m_lm.npy"))
    z_s5 = unitize_f32(
        zrk_f32(
            pd.read_csv(VOL_SUB / "sub_master_s5_decontam_s2_winner_0p22e4.csv", usecols=["is_fraud"])[
                "is_fraud"
            ].to_numpy(np.float64)
        )
    )
    u_priv = unit_f32(orth_against_f32(z_s7.astype(np.float64) - z_s5.astype(np.float64), [u_sb_t, u_3lad, u_gpt]))
    del z_s5
    u_ray_b = unit_f32(
        0.80 * u_plus_t.astype(np.float64)
        + 0.20 * u_priv.astype(np.float64)
        - 0.10 * u_gpt.astype(np.float64)
    )
    z_areeb = unitize_f32(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    if spearman_dist(z_areeb, z_s7.astype(np.float64) + 5.0 * u_ray_b.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7
    ):
        u_ray_b = (-u_ray_b.astype(np.float64)).astype(np.float32)
    if spearman_dist(z_areeb, z_s7.astype(np.float64) + 5.0 * u_ray_a.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7
    ):
        u_ray_a = (-u_ray_a.astype(np.float64)).astype(np.float32)
    u_ray_c = unit_f32(0.60 * u_ray_a.astype(np.float64) + 0.40 * u_ray_b.astype(np.float64))

    def pred_lb(z):
        d = spearman_dist(z_areeb, z)
        t = 100.0 * np.sqrt(max(d, 0.0))
        return 0.95346910 + 6.99725804e-05 * t, t, d

    # OOF directions for A/B/C
    u_ray_a_o = unitize_f32(s6_o)  # coupled continuation
    u_ray_b_o = unitize_f32(
        0.80 * u_plus_o.astype(np.float64)
        + 0.20 * unitize_f32(d1_o).astype(np.float64)  # priv proxy on OOF ≈ stageB residual
        - 0.10 * unitize_f32(gpt_o).astype(np.float64)
    )
    u_ray_c_o = unitize_f32(0.60 * u_ray_a_o.astype(np.float64) + 0.40 * u_ray_b_o.astype(np.float64))

    rows = []
    peak_f0 = {"ray": None, "step": None, "d0": -1e9}

    for step in steps:
        dw = (step / 1e-4) * dw_per
        # Ray A: push s6 weight past S7
        zA = z_ref.astype(np.float64) + (w_s7 + dw) * u_ray_a_o.astype(np.float64)
        # Ray B: freeze -3ladder at S7, advance clean stageB_plus
        zB = z_s7_oof + dw * u_ray_b_o.astype(np.float64)
        # Ray C: 0.60 A + 0.40 B in delta space from S7
        zC = z_s7_oof + dw * u_ray_c_o.astype(np.float64)

        def vs_s7(z):
            d0 = (float(roc_auc_score(y[m0], z[m0])) - base0) * 1e5 - s7["d0"]
            d14 = []
            for i, (_, b) in enumerate(folds):
                if i == f0_cv_i:
                    continue
                d14.append((float(roc_auc_score(y[b], z[b])) - base_folds[i]) * 1e5)
            d14m = float(np.mean(d14)) - s7["d14"]
            return d0, d14m

        # test geometry for each ray
        def test_geom(u):
            w, got = weight_for_spearman_dist(z_s7, u, step)
            z = unitize_f32(z_s7.astype(np.float64) + w * unit_f32(u).astype(np.float64))
            plb, t_est, d_areeb = pred_lb(z)
            return got, t_est, plb, d_areeb

        for name, z, u in [
            ("A", zA, u_ray_a),
            ("B_clean", zB, u_ray_b),
            ("C_opt", zC, u_ray_c),
        ]:
            d0, d14 = vs_s7(z)
            got, t_est, plb, d_areeb = test_geom(u)
            row = {
                "ray": name,
                "step": step,
                "got_s7": got,
                "d_areeb": d_areeb,
                "t": t_est,
                "pred_lin": plb,
                "d0_vsS7": d0,
                "d14_vsS7": d14,
                "d0_abs": d0 + s7["d0"],
            }
            rows.append(row)
            if d0 > peak_f0["d0"]:
                peak_f0 = {"ray": name, "step": step, "d0": d0, "d0_abs": row["d0_abs"], "pred": plb, "t": t_est}
            log(
                f"  {name:7s} step={step:.2e} t≈{t_est:.3f} pred={plb:.6f} "
                f"F0vsS7={d0:+.2f}e-5 F14vsS7={d14:+.2f}e-5 absF0={row['d0_abs']:+.2f}e-5"
            )
        purge()

    log(
        f"PEAK Fold-0: ray={peak_f0['ray']} step={peak_f0['step']:.2e} "
        f"ΔF0={peak_f0['d0']:+.2f}e-5 abs={peak_f0.get('d0_abs', float('nan')):+.2f}e-5 "
        f"pred={peak_f0.get('pred', float('nan')):.6f}"
    )
    (ART / "part1_macro_sweep.json").write_text(json.dumps({"s7": s7, "rows": rows, "peak_f0": peak_f0}, indent=2, default=float))

    # =====================================================================
    # PART 2 — Exact-L7 Stage-B SubCV on Fold-0
    # =====================================================================
    log("======== PART2 Exact-L7 StageB Fold0_SubCV_5F ========")
    lm7_o = np.load(
        first_existing(
            DS / "goodpjw2008__s6e11-compact-lm-seed7" / "lm7_oof_f0.npy",
            Path("/tmp/gm_v6_audit/external/nov11_095347_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f0.npy"),
        )
    )
    lm7_t = np.load(
        first_existing(
            DS / "goodpjw2008__s6e11-compact-lm-seed7" / "lm7_test_f0.npy",
            Path("/tmp/gm_v6_audit/external/nov11_095347_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_test_f0.npy"),
        )
    )
    assert lm7_o.shape == (139992, 10, 2), lm7_o.shape
    assert lm7_t.shape == (299984, 4, 2), lm7_t.shape
    llr_o = llr_from_lm(lm7_o).astype(np.float32)
    llr_t = llr_from_lm(lm7_t).astype(np.float32)
    del lm7_o, lm7_t
    purge()

    z_l7_bayes_f0 = bayes_llr(llr_o.astype(np.float64)).astype(np.float32)
    z_l7_bayes_t = bayes_llr(llr_t.astype(np.float64)).astype(np.float32)
    z_l7_mean_f0 = llr_o.mean(1).astype(np.float32)
    z_l7_mean_t = llr_t.mean(1).astype(np.float32)
    z_l7_std_f0 = llr_o.std(1).astype(np.float32)
    z_l7_std_t = llr_t.std(1).astype(np.float32)
    z_l7_min_f0 = llr_o.min(1).astype(np.float32)
    z_l7_min_t = llr_t.min(1).astype(np.float32)
    z_l7_max_f0 = llr_o.max(1).astype(np.float32)
    z_l7_max_t = llr_t.max(1).astype(np.float32)
    del llr_o, llr_t
    purge()

    # align each L7 stat test → Fold0 law
    def al_pair(fo, te, tag):
        o_al, t_al = align_f0_to_test(fo, te)
        ks = float(ks_2samp(o_al.astype(np.float64), t_al.astype(np.float64)).statistic)
        disp = float(np.std(t_al) / (np.std(o_al) + 1e-18))
        log(f"  align {tag}: disp={disp:.4f} KS={ks:.6f}")
        return o_al, t_al

    z_l7_bayes_f0, z_l7_bayes_t = al_pair(z_l7_bayes_f0, z_l7_bayes_t, "l7_bayes")
    z_l7_mean_f0, z_l7_mean_t = al_pair(z_l7_mean_f0, z_l7_mean_t, "l7_mean")
    z_l7_std_f0, z_l7_std_t = al_pair(z_l7_std_f0, z_l7_std_t, "l7_std")
    z_l7_min_f0, z_l7_min_t = al_pair(z_l7_min_f0, z_l7_min_t, "l7_min")
    z_l7_max_f0, z_l7_max_t = al_pair(z_l7_max_f0, z_l7_max_t, "l7_max")

    # companions (full train → slice f0; test full)
    mega_o = zrk_f32(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"))
    mega_t = zrk_f32(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_test.npy"))
    mega_o, mega_t = align_method_b_smooth(mega_o, mega_t, folds)
    mega_f0 = mega_o[idx_f0]

    gpt_o_full = unitize_f32(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    gpt_t = unitize_f32(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_test.npy"))
    gpt_f0 = gpt_o_full[idx_f0]

    w12_o = unitize_f32(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_oof.npy"))
    w12_t = unitize_f32(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_test.npy"))
    w12_f0 = w12_o[idx_f0]

    tabm_o = unitize_f32(zrk_f32(load_f32(PRED / "fast_gpu_harvest/oof_tabm_5f.npy")))
    tabm_t = unitize_f32(zrk_f32(load_f32(PRED / "fast_gpu_harvest/test_tabm_5f.npy")))
    tabm_f0 = tabm_o[idx_f0]

    p42_o = unitize_f32(zrk_f32(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy")))
    p42_t = unitize_f32(zrk_f32(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_test.npy")))
    p43_o = unitize_f32(zrk_f32(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy")))
    p43_t = unitize_f32(zrk_f32(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_test.npy")))
    p45_o = unitize_f32(zrk_f32(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy")))
    p45_t = unitize_f32(zrk_f32(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_test.npy")))

    bank = VOL / "external/banks/goodpjw2008_s6e11-oof-bank"
    mlp_o = unitize_f32(zrk_f32(load_f32(bank / "oof_realmlp_dig.npy")))
    mlp_t = unitize_f32(zrk_f32(load_f32(bank / "test_realmlp_dig.npy")))
    gb_o = unitize_f32(zrk_f32(load_f32(bank / "oof_gb_bag5.npy")))
    gb_t = unitize_f32(zrk_f32(load_f32(bank / "test_gb_bag5.npy")))
    mlp_f0, gb_f0 = mlp_o[idx_f0], gb_o[idx_f0]
    p42_f0, p43_f0, p45_f0 = p42_o[idx_f0], p43_o[idx_f0], p45_o[idx_f0]

    # KMAT TE top35 + rare/disp from X_v5 via mmap (fold0 + test only)
    meta = json.loads((PRED / "gm_v5_overnight/X_v5_meta.json").read_text())
    top_keys = list(meta.get("top15_keys", list(range(15))))
    # extend to 35: top15 + remaining first cols
    te_idx = []
    for k in top_keys + list(range(68)):
        if k not in te_idx:
            te_idx.append(k)
        if len(te_idx) >= 35:
            break
    # extras: z_lm_x_rare, z_lm_x_disp after 68+15=83 → look up by name
    extra_names = meta.get("extra_names", [])
    # layout: [0:68 KMAT] [68:83 kmat5seed] [83:  extras]
    base_extra = 68 + int(meta.get("n_kmat5seed", 15))
    rare_i = base_extra + extra_names.index("z_lm_x_rare") if "z_lm_x_rare" in extra_names else None
    disp_i = base_extra + extra_names.index("z_lm_x_disp") if "z_lm_x_disp" in extra_names else None
    cols = te_idx[:]
    if rare_i is not None:
        cols.append(rare_i)
    if disp_i is not None:
        cols.append(disp_i)
    log(f"X_v5 mmap cols={len(cols)} (TE35+rare/disp)")

    Xo = np.load(PRED / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
    Xt = np.load(PRED / "gm_v5_overnight/X_v5_test.npy", mmap_mode="r")
    X_f0_te = np.asarray(Xo[idx_f0][:, cols], dtype=np.float32)
    X_te_te = np.asarray(Xt[:, cols], dtype=np.float32)
    del Xo, Xt
    purge()

    # Cross-LM features on Fold0 / Test
    def unit_local(v):
        return unitize_f32(v)

    zl7b_f0 = unit_local(z_l7_bayes_f0)
    zl7b_t = unit_local(z_l7_bayes_t)
    zl7m_f0 = unit_local(z_l7_mean_f0)
    zl7m_t = unit_local(z_l7_mean_t)
    zl7s_f0 = unit_local(z_l7_std_f0)
    zl7s_t = unit_local(z_l7_std_t)

    d_l7_mega_f0 = unit_local(zl7b_f0.astype(np.float64) - mega_f0.astype(np.float64))
    d_l7_mega_t = unit_local(zl7b_t.astype(np.float64) - mega_t.astype(np.float64))
    d_l7_gpt_f0 = unit_local(zl7b_f0.astype(np.float64) - gpt_f0.astype(np.float64))
    d_l7_gpt_t = unit_local(zl7b_t.astype(np.float64) - gpt_t.astype(np.float64))
    d_l7_w12_f0 = unit_local(zl7b_f0.astype(np.float64) - w12_f0.astype(np.float64))
    d_l7_w12_t = unit_local(zl7b_t.astype(np.float64) - w12_t.astype(np.float64))
    prod_l7_mega_f0 = unit_local(zl7b_f0.astype(np.float64) * mega_f0.astype(np.float64))
    prod_l7_mega_t = unit_local(zl7b_t.astype(np.float64) * mega_t.astype(np.float64))

    # boundary-weighted certainty
    zref_f0 = z_ref[idx_f0].astype(np.float64)
    # map z_ref to p via rank→expit for bnd weight
    p_ref_f0 = expit(zref_f0)
    p_ref_t = expit(z_ref_t.astype(np.float64))
    std_pos_f0 = np.clip(zl7s_f0.astype(np.float64) - zl7s_f0.astype(np.float64).min() + 1e-3, 1e-3, None)
    std_pos_t = np.clip(zl7s_t.astype(np.float64) - zl7s_t.astype(np.float64).min() + 1e-3, 1e-3, None)
    bnd_f0 = unit_local(p_ref_f0 * (1.0 - p_ref_f0) * zl7b_f0.astype(np.float64) / (1.0 + std_pos_f0))
    bnd_t = unit_local(p_ref_t * (1.0 - p_ref_t) * zl7b_t.astype(np.float64) / (1.0 + std_pos_t))

    # Build feature matrices (float32)
    feat_core_f0 = np.column_stack(
        [
            zl7b_f0,
            zl7m_f0,
            zl7s_f0,
            unit_local(z_l7_min_f0),
            unit_local(z_l7_max_f0),
            mega_f0,
            gpt_f0,
            w12_f0,
            tabm_f0,
            p42_f0,
            p43_f0,
            p45_f0,
            mlp_f0,
            gb_f0,
            d_l7_mega_f0,
            d_l7_gpt_f0,
            d_l7_w12_f0,
            prod_l7_mega_f0,
            bnd_f0,
        ]
    ).astype(np.float32)
    feat_core_t = np.column_stack(
        [
            zl7b_t,
            zl7m_t,
            zl7s_t,
            unit_local(z_l7_min_t),
            unit_local(z_l7_max_t),
            mega_t,
            gpt_t,
            w12_t,
            tabm_t,
            p42_t,
            p43_t,
            p45_t,
            mlp_t,
            gb_t,
            d_l7_mega_t,
            d_l7_gpt_t,
            d_l7_w12_t,
            prod_l7_mega_t,
            bnd_t,
        ]
    ).astype(np.float32)

    X_f0 = np.hstack([feat_core_f0, X_f0_te]).astype(np.float32)
    X_te = np.hstack([feat_core_t, X_te_te]).astype(np.float32)
    del feat_core_f0, feat_core_t, X_f0_te, X_te_te
    purge()
    log(f"X_f0={X_f0.shape} X_te={X_te.shape} RSS={rss_curr_mb():.0f}MB")

    # base_margin from z_ref calibrated to prevalence on Fold0
    zref_raw_f0 = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")[idx_f0]
    zref_raw_t = load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy")
    bm_f0, bm_t = calibrate_logit_to_prevalence(zref_raw_f0, y_f0, zref_raw_t)
    del zref_raw_f0, zref_raw_t
    purge()

    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_f0), y_f0))

    # ---- Model 1: GLM-Quad Ridge ----
    log("  train glm_quad_real_l7 …")
    glm_cols = [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]  # bayes,mean,std,mega,gpt,w12,tabm,L42,L43,mlp,gb
    # also include rare if present as last-1 of TE block — use last 2 of X as rare/disp if appended
    Xg_f0 = poly2_cross(X_f0[:, glm_cols])
    Xg_t = poly2_cross(X_te[:, glm_cols])
    glm_oof = np.zeros(n_f0, dtype=np.float64)
    glm_te_acc = []
    for si, (a, b) in enumerate(subfolds):
        model = Ridge(alpha=1500.0, fit_intercept=True)
        model.fit(Xg_f0[a], y_f0[a].astype(np.float64))
        glm_oof[b] = model.predict(Xg_f0[b])
        glm_te_acc.append(model.predict(Xg_t))
        del model
        log(f"    glm SubF{si}")
    glm_te = np.mean(glm_te_acc, 0)
    del Xg_f0, Xg_t, glm_te_acc
    purge()
    np.save(ART / "glm_quad_real_l7_oof_f0.npy", glm_oof.astype(np.float32))
    np.save(ART / "glm_quad_real_l7_test.npy", glm_te.astype(np.float32))

    # ---- Model 2: LGB ExtraTrees bag5 ----
    log("  train lgb_stageB_real_l7_bag5 …")
    import lightgbm as lgb

    lgb_oof = np.zeros(n_f0, dtype=np.float64)
    lgb_te = np.zeros(nte, dtype=np.float64)
    lgb_seeds = [11, 22, 33, 44, 55]
    for seed in lgb_seeds:
        oof_s = np.zeros(n_f0, dtype=np.float64)
        te_s = np.zeros(nte, dtype=np.float64)
        for si, (a, b) in enumerate(subfolds):
            a_tr, a_va = train_test_split(a, test_size=0.10, random_state=seed + si, stratify=y_f0[a])
            dtr = lgb.Dataset(X_f0[a_tr], label=y_f0[a_tr], free_raw_data=False)
            dva = lgb.Dataset(X_f0[a_va], label=y_f0[a_va], reference=dtr, free_raw_data=False)
            # inject base_margin via init_score
            dtr.set_init_score(bm_f0[a_tr])
            dva.set_init_score(bm_f0[a_va])
            params = dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                extra_trees=True,
                num_leaves=6,
                max_depth=3,
                learning_rate=0.02,
                feature_fraction=0.35,
                bagging_fraction=0.8,
                bagging_freq=1,
                min_child_samples=200,
                lambda_l2=40.0,
                verbosity=-1,
                seed=seed + si,
                num_threads=2,
            )
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=1200,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
            )
            oof_s[b] = booster.predict(X_f0[b], raw_score=True)
            te_s += booster.predict(X_te, raw_score=True) / 5.0
            del booster, dtr, dva
            purge()
        lgb_oof += oof_s / len(lgb_seeds)
        lgb_te += te_s / len(lgb_seeds)
        log(f"    lgb seed={seed} done RSS={rss_curr_mb():.0f}MB")
    np.save(ART / "lgb_stageB_real_l7_bag5_oof_f0.npy", lgb_oof.astype(np.float32))
    np.save(ART / "lgb_stageB_real_l7_bag5_test.npy", lgb_te.astype(np.float32))

    # ---- Model 3: XGB bag5 ----
    log("  train xgb_stageB_real_l7_bag5 …")
    import xgboost as xgb

    xgb_oof = np.zeros(n_f0, dtype=np.float64)
    xgb_te = np.zeros(nte, dtype=np.float64)
    xgb_seeds = [7, 17, 27, 37, 47]
    for seed in xgb_seeds:
        oof_s = np.zeros(n_f0, dtype=np.float64)
        te_s = np.zeros(nte, dtype=np.float64)
        for si, (a, b) in enumerate(subfolds):
            depth = 2 if (seed + si) % 2 == 0 else 3
            a_tr, a_va = train_test_split(a, test_size=0.10, random_state=seed + si, stratify=y_f0[a])
            dtr = xgb.DMatrix(X_f0[a_tr], label=y_f0[a_tr], base_margin=bm_f0[a_tr])
            dva = xgb.DMatrix(X_f0[a_va], label=y_f0[a_va], base_margin=bm_f0[a_va])
            dte = xgb.DMatrix(X_te, base_margin=bm_t)
            dbo = xgb.DMatrix(X_f0[b], base_margin=bm_f0[b])
            params = dict(
                objective="binary:logistic",
                eval_metric="auc",
                max_depth=depth,
                eta=0.02,
                colsample_bynode=0.35,
                subsample=0.8,
                min_child_weight=180,
                reg_lambda=35.0,
                tree_method="hist",
                seed=seed + si,
                nthread=2,
            )
            booster = xgb.train(
                params,
                dtr,
                num_boost_round=1200,
                evals=[(dva, "va")],
                early_stopping_rounds=60,
                verbose_eval=False,
            )
            # raw margin-ish: use predict with output_margin
            oof_s[b] = booster.predict(dbo, output_margin=True)
            te_s += booster.predict(dte, output_margin=True) / 5.0
            del booster, dtr, dva, dte, dbo
            purge()
        xgb_oof += oof_s / len(xgb_seeds)
        xgb_te += te_s / len(xgb_seeds)
        log(f"    xgb seed={seed} done RSS={rss_curr_mb():.0f}MB")
    np.save(ART / "xgb_stageB_real_l7_bag5_oof_f0.npy", xgb_oof.astype(np.float32))
    np.save(ART / "xgb_stageB_real_l7_bag5_test.npy", xgb_te.astype(np.float32))

    # ---- Model 4: CatBoost bag3 ----
    log("  train cat_stageB_real_l7_bag3 …")
    from catboost import CatBoostClassifier, Pool

    cat_oof = np.zeros(n_f0, dtype=np.float64)
    cat_te = np.zeros(nte, dtype=np.float64)
    cat_seeds = [3, 13, 23]
    for seed in cat_seeds:
        oof_s = np.zeros(n_f0, dtype=np.float64)
        te_s = np.zeros(nte, dtype=np.float64)
        for si, (a, b) in enumerate(subfolds):
            a_tr, a_va = train_test_split(a, test_size=0.10, random_state=seed + si, stratify=y_f0[a])
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                depth=4,
                l2_leaf_reg=35,
                learning_rate=0.025,
                iterations=900,
                random_seed=seed + si,
                thread_count=2,
                verbose=False,
                early_stopping_rounds=60,
                allow_writing_files=False,
            )
            try:
                tr_pool = Pool(X_f0[a_tr], y_f0[a_tr], baseline=bm_f0[a_tr])
                va_pool = Pool(X_f0[a_va], y_f0[a_va], baseline=bm_f0[a_va])
                model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
                del tr_pool, va_pool
            except Exception:
                model.fit(X_f0[a_tr], y_f0[a_tr], eval_set=(X_f0[a_va], y_f0[a_va]), use_best_model=True)
            oof_s[b] = model.predict(X_f0[b], prediction_type="RawFormulaVal")
            te_s += model.predict(X_te, prediction_type="RawFormulaVal") / 5.0
            del model
            purge()
        cat_oof += oof_s / len(cat_seeds)
        cat_te += te_s / len(cat_seeds)
        log(f"    cat seed={seed} done RSS={rss_curr_mb():.0f}MB")
    np.save(ART / "cat_stageB_real_l7_bag3_oof_f0.npy", cat_oof.astype(np.float32))
    np.save(ART / "cat_stageB_real_l7_bag3_test.npy", cat_te.astype(np.float32))

    # Combined ensemble on Fold0 / Test
    ens_f0 = unitize_f32(
        0.20 * unitize_f32(zrk_f32(glm_oof)).astype(np.float64)
        + 0.30 * unitize_f32(lgb_oof).astype(np.float64)
        + 0.30 * unitize_f32(xgb_oof).astype(np.float64)
        + 0.20 * unitize_f32(cat_oof).astype(np.float64)
    )
    ens_t = unitize_f32(
        0.20 * unitize_f32(zrk_f32(glm_te)).astype(np.float64)
        + 0.30 * unitize_f32(lgb_te).astype(np.float64)
        + 0.30 * unitize_f32(xgb_te).astype(np.float64)
        + 0.20 * unitize_f32(cat_te).astype(np.float64)
    )

    # LOFO orth residual vs [z_ref, δ1, l7_bayes, mega, mlp, gb] on Fold0
    bases_f0 = [
        z_ref[idx_f0],
        d1_o[idx_f0],
        zl7b_f0,
        mega_f0,
        mlp_f0,
        gb_f0,
    ]
    u_real_f0 = lofo_resid(ens_f0, bases_f0, subfolds)
    # test: project against same bases aligned
    u_real_t = orth_against_f32(
        ens_t,
        [z_ref_t, d1_t, zl7b_t, mega_t, mlp_t, gb_t],
    )
    # align test to Fold0 residual law
    u_real_f0, u_real_t = force_exact_law(u_real_f0, u_real_t)
    ks_u = float(ks_2samp(u_real_f0.astype(np.float64), u_real_t.astype(np.float64)).statistic)
    disp_u = float(np.std(u_real_t) / (np.std(u_real_f0) + 1e-18))
    log(f"u_real_l7_stageB aligned disp={disp_u:.4f} KS={ks_u:.6f}")
    np.save(ART / "u_real_l7_stageB_oof_f0.npy", u_real_f0)
    np.save(ART / "u_real_l7_stageB_test.npy", u_real_t)

    # Verify 5/5 SubCV lift on S7_proxy[idx_f0]
    z_s7_f0 = z_s7_oof[idx_f0]
    base_sub = [float(roc_auc_score(y_f0[b], z_s7_f0[b])) for _, b in subfolds]
    deltas = []
    for w_try in [0.02, 0.03, 0.04, 0.05, 0.06]:
        z = z_s7_f0 + w_try * u_real_f0.astype(np.float64)
        dsub = [(float(roc_auc_score(y_f0[b], z[b])) - base_sub[i]) * 1e5 for i, (_, b) in enumerate(subfolds)]
        pos = sum(x > 0 for x in dsub)
        mean_d = float(np.mean(dsub))
        deltas.append({"w": w_try, "mean": mean_d, "pos": pos, "dsub": dsub})
        log(f"  u_real @w={w_try}: meanΔ={mean_d:+.2f}e-5 pos={pos}/5 d={[round(x,2) for x in dsub]}")
    best_u = max(deltas, key=lambda r: (r["pos"], r["mean"]))
    log(f"BEST u_real_l7: w={best_u['w']} pos={best_u['pos']}/5 mean={best_u['mean']:+.2f}e-5")

    # free heavy matrices before Part3
    del X_f0, X_te, ens_f0, ens_t, glm_oof, lgb_oof, xgb_oof, cat_oof, glm_te, lgb_te, xgb_te, cat_te
    purge()

    # =====================================================================
    # PART 3 — 5 NOSUBMIT slots
    # =====================================================================
    log("======== PART3 stage 5 NOSUBMIT breakout slots ========")
    # combine Ray C_opt + u_real_l7 on test
    u_combo = unit_f32(
        0.70 * u_ray_c.astype(np.float64) + 0.30 * unit_f32(u_real_t).astype(np.float64)
    )
    if spearman_dist(z_areeb, z_s7.astype(np.float64) + 5.0 * u_combo.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7
    ):
        u_combo = (-u_combo.astype(np.float64)).astype(np.float32)

    # adaptive reserve: peak_f0 step along C_opt+realL7, or 1.75e-4 fallback
    adapt_step = float(peak_f0["step"]) if peak_f0["step"] is not None else 1.75e-4
    if adapt_step < 0.70e-4:
        adapt_step = 1.75e-4

    st9_csv = first_existing(
        VOL_SUB / "sub_gm_top1_strike9_final_max.csv",
        Path("/tmp/gm_v6_audit/external/nov11_095347_recon/datasets/moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv"),
    )
    st9_df = pd.read_csv(st9_csv, usecols=["id", "is_fraud"])
    ids = st9_df["id"].to_numpy()
    p_st9 = st9_df["is_fraud"].to_numpy(np.float64)
    del st9_df
    purge()

    specs = [
        (1, "clean_rayC_plus_realL7_0p15e4", u_combo, 0.15e-4, "sub_gm_v8_slot1_clean_rayC_plus_realL7_0p15e4_NOSUBMIT.csv"),
        (2, "realL7_stageB_med_0p35e4", u_combo, 0.35e-4, "sub_gm_v8_slot2_realL7_stageB_med_0p35e4_NOSUBMIT.csv"),
        (3, "fold0_peak_macro_0p70e4", u_combo, 0.70e-4, "sub_gm_v8_slot3_fold0_peak_macro_0p70e4_NOSUBMIT.csv"),
        (4, "apex_breakout_1p20e4", u_combo, 1.20e-4, "sub_gm_v8_slot4_apex_breakout_1p20e4_NOSUBMIT.csv"),
        (5, "adaptive_095365_reserve", u_combo, adapt_step, "sub_gm_v8_slot5_adaptive_095365_reserve_NOSUBMIT.csv"),
    ]

    cands = []
    z_list = []
    for slot, name, u, target, fname in specs:
        w, got = weight_for_spearman_dist(z_s7, u, target)
        z = unitize_f32(z_s7.astype(np.float64) + w * unit_f32(u).astype(np.float64))
        p = to_probs_0167(z, p_st9)
        assert len(p) == 299984 and np.isfinite(p).all()
        plb, t_est, d_areeb = pred_lb(z)
        out = SUB / fname
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(out, index=False)
        try:
            pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(VOL_SUB / fname, index=False)
        except Exception as e:
            log(f"  vol sync warn {fname}: {e}")
        row = {
            "slot": slot,
            "name": name,
            "file": fname,
            "target": target,
            "got": got,
            "mean": float(p.mean()),
            "pred_lb": plb,
            "t_est": t_est,
            "d_areeb": d_areeb,
            "w": w,
        }
        cands.append(row)
        z_list.append(z.copy())
        log(f"  Slot{slot} {name}: got={got:.2e} mean={p.mean():.6f} t≈{t_est:.3f} predLB={plb:.6f}")
        del p
        purge()

    log("======== pairwise uniqueness ========")
    uniq_ok = True
    pair_rows = []
    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            d = spearman_dist(z_list[i], z_list[j])
            ok = d > 0.02e-4
            uniq_ok = uniq_ok and ok
            pair_rows.append((i + 1, j + 1, d, ok))
            log(f"  slot{i+1} vs slot{j+1}: 1-ρ={d:.3e} {'OK' if ok else 'DUP!'}")

    # ---------- Report ----------
    md = []
    md.append("# GM v8 — Exact-L7 Stage-B & 0.95365 Leap Engine")
    md.append("")
    md.append(f"**NO_SUBMIT=True**. ATH **`{ATH}`**. Soft RSS cap `{RSS_SOFT_MB}` MB.")
    md.append("")
    md.append(f"- tree_logit: `{tree_src}`")
    md.append("- `u_stageB_plus = unit(0.78·u_stageB + 0.22·tree_logit)` — **Block3 zeroed** (v7.2 −9e-5 bug fixed)")
    md.append(f"- peak Fold-0 OOF: ray=`{peak_f0['ray']}` step=`{peak_f0['step']}` ΔF0=`{peak_f0['d0']:+.2f}e-5`")
    md.append(f"- u_real_l7 best SubCV: w=`{best_u['w']}` pos=`{best_u['pos']}/5` meanΔ=`{best_u['mean']:+.2f}e-5` d={best_u['dsub']}")
    md.append(f"- observed RSS curr/max: `{rss_curr_mb():.0f}` / `{rss_mb():.0f}` MB")
    md.append("")
    md.append("## Part 1 — Complete-formula Fold-0 macro sweep")
    md.append("")
    md.append(f"S7_proxy: 5F=`{s7['lift5']:+.2f}e-5` F0=`{s7['d0']:+.2f}e-5` F14=`{s7['d14']:+.2f}e-5`")
    md.append("")
    md.append("| ray | 1−ρ(S7) | 1−ρ(areeb) | t | pred_lin | F0 Δ vsS7 | F14 Δ vsS7 | abs F0 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['ray']} | {r['step']:.2e} | {r['d_areeb']:.2e} | {r['t']:.3f} | {r['pred_lin']:.6f} | "
            f"{r['d0_vsS7']:+.2f} | {r['d14_vsS7']:+.2f} | {r['d0_abs']:+.2f} |"
        )
    md.append("")
    md.append(
        f"**t_F0\\*** ≈ ray `{peak_f0['ray']}` @ `1−ρ(S7)={peak_f0['step']}` "
        f"(pred_lin `{peak_f0.get('pred', float('nan')):.6f}`, t≈`{peak_f0.get('t', float('nan')):.3f}`)."
    )
    md.append("")
    md.append("## Part 2 — Exact-L7 Stage-B Fold0_SubCV_5F")
    md.append("")
    md.append(f"u_real_l7_stageB: disp=`{disp_u:.4f}` KS=`{ks_u:.6f}`")
    md.append("")
    md.append("| w | mean Δe-5 | pos | SubF deltas |")
    md.append("|---:|---:|---:|---|")
    for r in deltas:
        md.append(f"| {r['w']} | {r['mean']:+.2f} | {r['pos']}/5 | {[round(x,2) for x in r['dsub']]} |")
    md.append("")
    md.append("## Part 3 — 5 NOSUBMIT slots (Ray C_opt + u_real_l7)")
    md.append("")
    md.append(f"Pairwise uniqueness `1−ρ > 0.02e-4`: **{'PASS' if uniq_ok else 'FAIL'}**")
    md.append("")
    md.append("| slot | file | 1−ρ(S7) | t≈ | pred LB | mean |")
    md.append("|---:|---|---:|---:|---:|---:|")
    for c in cands:
        md.append(
            f"| {c['slot']} | `{c['file']}` | {c['target']:.2e} | {c['t_est']:.3f} | "
            f"{c['pred_lb']:.6f} | {c['mean']:.6f} |"
        )
    md.append("")
    md.append("**NO_SUBMIT=True** — fire Slot1 tomorrow as probe; escalate 2→3→4 only while LB climbs; Slot5 adaptive reserve.")
    report = "\n".join(md)
    rp = LOG / "GM_V8_REAL_L7_095365_ENGINE.md"
    rp.write_text(report)
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / "GM_V8_REAL_L7_095365_ENGINE.md").write_text(report)
        VOL_ART.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log(f"vol report sync warn: {e}")

    (ART / "v8_summary.json").write_text(
        json.dumps(
            {
                "NO_SUBMIT": True,
                "tree_src": tree_src,
                "s7": s7,
                "peak_f0": peak_f0,
                "best_u_real": best_u,
                "cands": cands,
                "uniq_ok": uniq_ok,
                "rss_curr_mb": rss_curr_mb(),
                "rss_max_mb": rss_mb(),
            },
            indent=2,
            default=float,
        )
    )
    # sync script to volume
    try:
        (VOL / "scripts").mkdir(parents=True, exist_ok=True)
        src = Path(__file__).resolve()
        (VOL / "scripts" / src.name).write_text(src.read_text())
    except Exception as e:
        log(f"script sync warn: {e}")

    log(f"REPORT → {rp}")
    log("======== GM v8 DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
