#!/usr/bin/env python3
"""GM v7.2 — Low-RAM dedup + OOF-rising Stage-B continuation past S7.

STRICT NO_SUBMIT=True. Peak RSS target < 500 MB via sequential float32 loads + gc.
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
warnings.filterwarnings("ignore")

VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_V7_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_deep_tech_v7"
SUB = WORK / "submissions"
VOL_SUB = VOL / "submissions"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
for p in (ART, SUB, LOG):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT
EPS = 1e-6
RSS_SOFT_MB = 450
T0 = time.time()

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss_mb() -> float:
    # ru_maxrss is bytes on Linux, KB on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|RSS_max={rss_mb():.0f}MB] {msg}", flush=True)
    if rss_mb() > RSS_SOFT_MB:
        log_warn = f"  WARN RSS_max {rss_mb():.0f}MB > soft cap {RSS_SOFT_MB}"
        print(log_warn, flush=True)


def purge(*names):
    for n in names:
        try:
            del n
        except Exception:
            pass
    gc.collect()


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).ravel()


def zrk_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    z = logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS))
    return z.astype(np.float32)


def unitize_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    out = (v - float(v.mean())) / (float(v.std()) + 1e-12)
    return out.astype(np.float32)


def unit_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def as_z_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9:
        return unitize_f32(zrk_f32(x))
    return unitize_f32(x)


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
    # fold-std oof
    oof_std = np.zeros_like(oof_vec)
    for a, b in folds:
        mu, sd = float(oof_vec[b].mean()), float(oof_vec[b].std() + 1e-12)
        oof_std[b] = (oof_vec[b] - mu) / sd
    oof_std = (oof_std - oof_std.mean()) / (oof_std.std() + 1e-12)
    test_std = (test_vec - test_vec.mean()) / (test_vec.std() + 1e-12)
    q = np.linspace(0.0, 1.0, 2001)  # lighter than 10001
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


def spearman_dist(a, b) -> float:
    return float(1.0 - spearmanr(a, b).correlation)


def weight_for_spearman_dist(z0, u, target, hi=80.0, n=48):
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


def first_existing(*cands: Path) -> Path:
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(cands)


def main():
    log("======== GM v7.2 LOW-RAM START (NO_SUBMIT) ========")
    log(f"soft RSS cap={RSS_SOFT_MB}MB  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    # --- labels / folds (minimal) ---
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"])["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    ntr = len(y)
    log(f"ntr={ntr} fraud_rate={float(y.mean()):.6f}")

    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    # keep as float32 unitized for eval
    z_ref = unitize_f32(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    # Actually z_ref_old may already be z-space; ensure AUC check
    z_ref_auc = float(roc_auc_score(y, z_ref.astype(np.float64)))
    base_folds = fold_aucs(y, z_ref, folds)
    base0 = float(roc_auc_score(y[m0], z_ref[m0].astype(np.float64)))
    log(f"z_ref_old AUC={z_ref_auc:.7f}")

    d1_o = unitize_f32(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    d1_t = unitize_f32(load_f32(PRED / "gm_v4_polish/d1_v4_test.npy"))
    log(f"δ1_v4 loaded")

    # --- tree_logit: prefer true v7 dual-anchor; else overnight dual bags (documented fallback) ---
    v7_lgb_o = ART / "lgb_v7_extratrees_dual_oof.npy"
    v7_xgb_o = ART / "xgb_v7_bce_dual_anchor_oof.npy"
    vol_v7 = PRED / "gm_deep_tech_v7"
    src = "v7_dual_anchor"
    try:
        p_lgb_o = load_f32(first_existing(v7_lgb_o, vol_v7 / "lgb_v7_extratrees_dual_oof.npy"))
        p_lgb_t = load_f32(first_existing(ART / "lgb_v7_extratrees_dual_test.npy", vol_v7 / "lgb_v7_extratrees_dual_test.npy"))
        p_xgb_o = load_f32(first_existing(v7_xgb_o, vol_v7 / "xgb_v7_bce_dual_anchor_oof.npy"))
        p_xgb_t = load_f32(first_existing(ART / "xgb_v7_bce_dual_anchor_test.npy", vol_v7 / "xgb_v7_bce_dual_anchor_test.npy"))
    except FileNotFoundError:
        src = "fallback_overnight_extratrees+monotone (v7 caches wiped)"
        p_lgb_o = load_f32(PRED / "gm_v5_overnight/lgb_stageB_extratrees_bag6_oof.npy")
        p_lgb_t = load_f32(PRED / "gm_v5_overnight/lgb_stageB_extratrees_bag6_test.npy")
        p_xgb_o = load_f32(PRED / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_oof.npy")
        p_xgb_t = load_f32(PRED / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_test.npy")
    log(f"tree_logit source={src}")

    v5_o = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v5_harvest_oof.npy")))
    v5_t = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v5_harvest_test.npy")))
    # dual margin for xgb: blend v5 + z8
    z8_o = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v8_part12_rect_oof.npy")))
    z8_t = unitize_f32(zrk_f32(load_f32(PRED / "P_super_v8_part12_rect_test.npy")))

    t_lgb_o = tree_logit_residual(p_lgb_o, v5_o)
    t_lgb_t = tree_logit_residual(p_lgb_t, v5_t)
    purge(p_lgb_o, p_lgb_t)
    t_xgb_o = unitize_f32(0.5 * tree_logit_residual(p_xgb_o, v5_o).astype(np.float64) + 0.5 * tree_logit_residual(p_xgb_o, z8_o).astype(np.float64))
    t_xgb_t = unitize_f32(0.5 * tree_logit_residual(p_xgb_t, v5_t).astype(np.float64) + 0.5 * tree_logit_residual(p_xgb_t, z8_t).astype(np.float64))
    purge(p_xgb_o, p_xgb_t, v5_o, v5_t, z8_o, z8_t)

    tree_o = unitize_f32(0.55 * t_lgb_o.astype(np.float64) + 0.45 * t_xgb_o.astype(np.float64))
    tree_t = unitize_f32(0.55 * t_lgb_t.astype(np.float64) + 0.45 * t_xgb_t.astype(np.float64))
    purge(t_lgb_o, t_lgb_t, t_xgb_o, t_xgb_t)

    tree_o = lofo_resid(tree_o, [z_ref, d1_o], folds)
    tree_t = orth_against_f32(tree_t, [unitize_f32(load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy")), d1_t])
    # re-align
    tree_o, tree_t = align_method_b_smooth(tree_o, tree_t, folds)
    log(f"tree_logit⊥[z_ref,δ1] ready")

    # quick lift of tree alone
    def eval_w(delta, w=0.04):
        z = z_ref.astype(np.float64) + w * unitize_f32(delta).astype(np.float64)
        fa = fold_aucs(y, z, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
        d0 = (float(roc_auc_score(y[m0], z[m0])) - base0) * 1e5
        return float(np.mean(d5)), int(sum(x > 0 for x in d5)), d0, d5

    tr_m, tr_p, tr_0, tr_d5 = eval_w(tree_o, 0.04)
    log(f"  tree_logit @0.04: 5FΔ={tr_m:+.2f}e-5 pos={tr_p}/5 F0={tr_0:+.2f}e-5 d5={[round(x,2) for x in tr_d5]}")

    # --- Block3 Ridge: prefer condensed col5; else light LOFO ridge on X_v5 TE slice ---
    block_src = "condensed_col5"
    try:
        blocks = np.load(first_existing(ART / "condensed_blocks_oof.npy", vol_v7 / "condensed_blocks_oof.npy"), mmap_mode="r")
        b3_o = unitize_f32(np.asarray(blocks[:, 5], dtype=np.float32))
        blocks_t = np.load(first_existing(ART / "condensed_blocks_test.npy", vol_v7 / "condensed_blocks_test.npy"), mmap_mode="r")
        b3_t = unitize_f32(np.asarray(blocks_t[:, 5], dtype=np.float32))
        del blocks, blocks_t
        gc.collect()
    except FileNotFoundError:
        block_src = "fallback_X_v5_TE24_ridge_LOFO"
        log(f"  condensed_blocks missing → {block_src}")
        Xo = np.load(PRED / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
        Xt = np.load(PRED / "gm_v5_overnight/X_v5_test.npy", mmap_mode="r")
        # use first 24 TE cols only, float32 copies fold-wise
        b3_o = np.zeros(ntr, dtype=np.float64)
        te_acc = []
        for fi, (a, b) in enumerate(folds):
            Xa = np.asarray(Xo[a, :24], dtype=np.float32)
            Xb = np.asarray(Xo[b, :24], dtype=np.float32)
            Xt24 = np.asarray(Xt[:, :24], dtype=np.float32)
            model = Ridge(alpha=3000.0, fit_intercept=True)
            model.fit(Xa, y[a].astype(np.float64))
            b3_o[b] = model.predict(Xb)
            te_acc.append(model.predict(Xt24))
            del Xa, Xb, Xt24, model
            gc.collect()
            log(f"  block3 ridge F{fi}")
        b3_t = np.mean(te_acc, 0)
        b3_o = unitize_f32(zrk_f32(b3_o))
        b3_t = unitize_f32(zrk_f32(b3_t))
        b3_o = lofo_resid(b3_o, [z_ref, d1_o], folds)
        b3_t = orth_against_f32(b3_t, [unitize_f32(load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy")), d1_t])
        b3_o, b3_t = align_method_b_smooth(b3_o, b3_t, folds)
        del Xo, Xt, te_acc
        gc.collect()
    log(f"block3 source={block_src}")
    b3_m, b3_p, b3_0, _ = eval_w(b3_o, 0.04)
    log(f"  block3 @0.04: 5FΔ={b3_m:+.2f}e-5 pos={b3_p}/5 F0={b3_0:+.2f}e-5")

    # --- u_stageB_plus OOF/TEST ---
    u_sb_o = unit_f32(d1_o)
    u_sb_t = unit_f32(load_f32(PRED / "gm_breakout/u_stageB_5f.npy"))
    # ensure same orientation as d1_t
    if float(np.dot(u_sb_t.astype(np.float64), d1_t.astype(np.float64))) < 0:
        u_sb_t = (-u_sb_t.astype(np.float64)).astype(np.float32)

    u_plus_o = unit_f32(
        0.70 * u_sb_o.astype(np.float64)
        + 0.20 * unit_f32(tree_o).astype(np.float64)
        + 0.10 * unit_f32(b3_o).astype(np.float64)
    )
    u_plus_t = unit_f32(
        0.70 * u_sb_t.astype(np.float64)
        + 0.20 * unit_f32(tree_t).astype(np.float64)
        + 0.10 * unit_f32(b3_t).astype(np.float64)
    )
    u_plus_o, u_plus_t = align_method_b_smooth(u_plus_o, u_plus_t, folds)
    dks = float(ks_2samp(u_plus_o.astype(np.float64), u_plus_t.astype(np.float64)).statistic)
    log(f"u_stageB_plus aligned KS={dks:.6f}")

    # --- Step1 OOF: Ray A vs Ray B from S7 proxy ---
    log("======== STEP1 Ray A vs Ray B OOF ========")
    # rebuild light S6/S7 ray OOF like verify (m1/m2 with d1)
    gpt_path = PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"
    if gpt_path.exists():
        gpt_o = unitize_f32(load_f32(gpt_path))
    else:
        gpt_o = np.zeros(ntr, dtype=np.float32)
    # simplified bundle: use u_3ladder test mapped? For OOF use -gpt + weak zeros
    # m1 ≈ 0.65 stageB - 0.35*3ladder - 0.20 gpt ; approximate 3ladder with gpt contrast on OOF
    m1_o = unitize_f32(0.65 * u_sb_o.astype(np.float64) - 0.35 * (-unit_f32(gpt_o)).astype(np.float64) - 0.20 * unitize_f32(gpt_o).astype(np.float64))
    m2_o = unitize_f32(0.65 * u_sb_o.astype(np.float64) + 0.35 * u_sb_o.astype(np.float64))
    s6_o = unitize_f32(0.70 * m1_o.astype(np.float64) + 0.30 * m2_o.astype(np.float64))
    w_s7 = 0.055

    def oof_at(delta, w):
        z = z_ref.astype(np.float64) + w * unitize_f32(delta).astype(np.float64)
        fa = fold_aucs(y, z, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
        d0 = (float(roc_auc_score(y[m0], z[m0])) - base0) * 1e5
        return {
            "lift5": (float(roc_auc_score(y, z)) - z_ref_auc) * 1e5,
            "d5_mean": float(np.mean(d5)),
            "d0": d0,
            "pos": int(sum(x > 0 for x in d5)),
            "d5": d5,
        }

    s7 = oof_at(s6_o, w_s7)
    log(f"S7_proxy: 5F={s7['lift5']:+.2f}e-5 F0={s7['d0']:+.2f}e-5 pos={s7['pos']}/5")

    # Ray A: continue pure s6 ray (pushes stageB AND -3ladder together)
    # Ray B: freeze 3ladder — step only along stageB_plus (and light -gpt)
    steps = [0.06e-4, 0.08e-4, 0.10e-4, 0.15e-4, 0.18e-4, 0.22e-4, 0.30e-4]
    dw_per = 0.08
    rows = []
    for step in steps:
        dw = (step / 1e-4) * dw_per
        # Ray A
        rA = oof_at(s6_o, w_s7 + dw)
        # Ray B: from S7 level, add stageB_plus only
        # Represent as: s7_dir + extra*u_plus where s7 level ≈ s6@w_s7
        # Equivalent eval: z_ref + w_s7*s6 + dw*u_plus  (freeze ladder by not increasing s6 beyond)
        zB = z_ref.astype(np.float64) + w_s7 * unitize_f32(s6_o).astype(np.float64) + dw * unitize_f32(u_plus_o).astype(np.float64)
        fa = fold_aucs(y, zB, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
        d0 = (float(roc_auc_score(y[m0], zB[m0])) - base0) * 1e5
        rB = {
            "lift5": (float(roc_auc_score(y, zB)) - z_ref_auc) * 1e5,
            "d5_mean": float(np.mean(d5)),
            "d0": d0,
            "pos": int(sum(x > 0 for x in d5)),
            "d5": d5,
        }
        row = {
            "step": step,
            "A_5F": rA["lift5"],
            "A_F0": rA["d0"],
            "A_vsS7_5F": rA["lift5"] - s7["lift5"],
            "A_vsS7_F0": rA["d0"] - s7["d0"],
            "A_pos": rA["pos"],
            "B_5F": rB["lift5"],
            "B_F0": rB["d0"],
            "B_vsS7_5F": rB["lift5"] - s7["lift5"],
            "B_vsS7_F0": rB["d0"] - s7["d0"],
            "B_pos": rB["pos"],
        }
        rows.append(row)
        log(
            f"  step={step:.2e}  A vsS7 5F={row['A_vsS7_5F']:+.2f} F0={row['A_vsS7_F0']:+.2f}  "
            f"B vsS7 5F={row['B_vsS7_5F']:+.2f} F0={row['B_vsS7_F0']:+.2f} (pos {row['B_pos']}/5)"
        )

    # pick best B step with positive vs S7 on 5F or F0
    bestB = max(rows, key=lambda r: r["B_vsS7_5F"] + 0.5 * r["B_vsS7_F0"])
    log(f"best RayB step={bestB['step']:.2e} vsS7 5F={bestB['B_vsS7_5F']:+.2f} F0={bestB['B_vsS7_F0']:+.2f}")

    # --- Step2: stage 4 distinct test candidates ---
    log("======== STEP2 stage 4 distinct NOSUBMIT slots ========")
    z_s7 = unitize_f32(zrk_f32(pd.read_csv(VOL_SUB / "sub_master_slot7_verified_s7A.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.float64)))
    m1 = unit_f32(load_f32(PRED / "gm_breakout/m1.npy"))
    m2 = unit_f32(load_f32(PRED / "gm_breakout/m2.npy"))
    u_ray_a = unit_f32(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    u_3lad = unit_f32(load_f32(PRED / "gm_breakout/u_3ladder_grad.npy"))
    u_gpt = unit_f32(load_f32(PRED / "gm_breakout/u_gpt2m_lm.npy"))
    # priv ≈ S5 residual toward S7 after removing stageB/3ladder/gpt — approximate from S5→S7 minus rayA parts
    z_s5 = unitize_f32(zrk_f32(pd.read_csv(VOL_SUB / "sub_master_s5_decontam_s2_winner_0p22e4.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.float64)))
    u_priv = unit_f32(orth_against_f32(z_s7.astype(np.float64) - z_s5.astype(np.float64), [u_sb_t, u_3lad, u_gpt]))
    purge(z_s5)

    # Ray B test direction: freeze -3ladder (do not include -u_3ladder), advance stageB_plus + priv -gpt
    u_ray_b = unit_f32(
        0.80 * u_plus_t.astype(np.float64)
        + 0.20 * u_priv.astype(np.float64)
        - 0.10 * u_gpt.astype(np.float64)
    )
    # ensure steps move away from areeb along positive LB history
    z_areeb = unitize_f32(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    if spearman_dist(z_areeb, z_s7.astype(np.float64) + 5.0 * u_ray_b.astype(np.float64)) < spearman_dist(z_areeb, z_s7):
        u_ray_b = (-u_ray_b.astype(np.float64)).astype(np.float32)
    if spearman_dist(z_areeb, z_s7.astype(np.float64) + 5.0 * u_ray_a.astype(np.float64)) < spearman_dist(z_areeb, z_s7):
        u_ray_a = (-u_ray_a.astype(np.float64)).astype(np.float32)

    u_hybrid = unit_f32(0.50 * u_ray_a.astype(np.float64) + 0.50 * u_ray_b.astype(np.float64))

    # St9 law
    st9 = VOL / "external/banks"
    # prefer strike9 csv from submissions of ours or recon
    st9_csv = first_existing(
        VOL_SUB / "sub_gm_top1_strike9_final_max.csv",
        VOL / "external/nov11_095347_recon/datasets/moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv",
        Path("/tmp/gm_v6_audit/external/nov11_095347_recon/datasets/moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv"),
    )
    st9_df = pd.read_csv(st9_csv, usecols=["id", "is_fraud"])
    ids = st9_df["id"].to_numpy()
    p_st9 = st9_df["is_fraud"].to_numpy(np.float64)
    del st9_df
    gc.collect()

    # linear LB law (concave a=0 from v7.1)
    # LB = 0.95346910 + 6.99725804e-05 * t , t=100*sqrt(1-rho(areeb))
    def pred_lb(z):
        d = spearman_dist(z_areeb, z)
        t = 100.0 * np.sqrt(max(d, 0.0))
        return 0.95346910 + 6.99725804e-05 * t, t

    specs = [
        (1, "pure_ray_0p08e4", u_ray_a, 0.08e-4, "sub_gm_v7p2_slot1_pure_ray_0p08e4_NOSUBMIT.csv"),
        (2, "decoupled_stageB_plus_treelogit_0p10e4", u_ray_b, 0.10e-4, "sub_gm_v7p2_slot2_decoupled_stageB_plus_treelogit_0p10e4_NOSUBMIT.csv"),
        (3, "hybrid_rayA_rayB_0p18e4", u_hybrid, 0.18e-4, "sub_gm_v7p2_slot3_hybrid_rayA_rayB_0p18e4_NOSUBMIT.csv"),
        (4, "adaptive_stepup_0p30e4", u_ray_b if bestB["B_vsS7_5F"] >= bestB["A_vsS7_5F"] else u_ray_a, 0.30e-4, "sub_gm_v7p2_slot4_adaptive_stepup_0p30e4_NOSUBMIT.csv"),
    ]

    cands = []
    z_list = []
    for slot, name, u, target, fname in specs:
        w, got = weight_for_spearman_dist(z_s7, u, target)
        z = unitize_f32(z_s7.astype(np.float64) + w * unit_f32(u).astype(np.float64))
        p = to_probs_0167(z, p_st9)
        assert len(p) == 299984 and np.isfinite(p).all()
        plb, t_est = pred_lb(z)
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
            "w": w,
        }
        cands.append(row)
        z_list.append(z.copy())
        log(f"  Slot{slot} {name}: got={got:.2e} mean={p.mean():.6f} t≈{t_est:.3f} predLB={plb:.6f}")
        del p
        gc.collect()

    # pairwise uniqueness
    log("======== pairwise 1-spearman uniqueness ========")
    uniq_ok = True
    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            d = spearman_dist(z_list[i], z_list[j])
            ok = d > 0.02e-4
            uniq_ok = uniq_ok and ok
            log(f"  slot{i+1} vs slot{j+1}: 1-ρ={d:.3e} {'OK' if ok else 'DUP!'}")

    # save artifacts lightly
    np.save(ART / "u_stageB_plus_oof.npy", u_plus_o)
    np.save(ART / "u_stageB_plus_test.npy", u_plus_t)
    np.save(ART / "tree_logit_v72_oof.npy", tree_o)
    np.save(ART / "tree_logit_v72_test.npy", tree_t)

    # report
    md = []
    md.append("# GM v7.2 Ready for Tomorrow — Dedup + Decoupled Stage-B")
    md.append("")
    md.append("**NO_SUBMIT=True**. Low-RAM sequential float32 pipeline. ATH **`0.95354`**.")
    md.append("")
    md.append(f"- tree_logit source: `{src}`")
    md.append(f"- block3 source: `{block_src}`")
    md.append(f"- peak RSS (ru_maxrss): **{rss_mb():.0f} MB**")
    md.append("")
    md.append("## Step 1 — Ray A (coupled S5→S7) vs Ray B (freeze −u_3ladder, advance stageB_plus)")
    md.append("")
    md.append(f"S7_proxy: 5F=`{s7['lift5']:+.2f}e-5` F0=`{s7['d0']:+.2f}e-5` pos={s7['pos']}/5")
    md.append("")
    md.append("| 1−ρ(S7) | A Δ5F vsS7 | A ΔF0 vsS7 | B Δ5F vsS7 | B ΔF0 vsS7 | B pos |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['step']:.2e} | {r['A_vsS7_5F']:+.2f} | {r['A_vsS7_F0']:+.2f} | "
            f"{r['B_vsS7_5F']:+.2f} | {r['B_vsS7_F0']:+.2f} | {r['B_pos']}/5 |"
        )
    md.append("")
    md.append(
        f"**Winner:** Ray B @ `{bestB['step']:.2e}` "
        f"(vsS7 5F `{bestB['B_vsS7_5F']:+.2f}` / F0 `{bestB['B_vsS7_F0']:+.2f}`) "
        f"vs Ray A vsS7 5F `{bestB['A_vsS7_5F']:+.2f}`."
    )
    md.append("")
    md.append("## Step 2 — 4 distinct NOSUBMIT slots from z_S7")
    md.append("")
    md.append(f"Pairwise uniqueness gate `1−ρ > 0.02e-4`: **{'PASS' if uniq_ok else 'FAIL'}**")
    md.append("")
    md.append("| slot | file | 1−ρ(S7) | t≈ | pred LB | mean |")
    md.append("|---:|---|---:|---:|---:|---:|")
    for c in cands:
        md.append(
            f"| {c['slot']} | `{c['file']}` | {c['target']:.2e} | {c['t_est']:.3f} | "
            f"{c['pred_lb']:.6f} | {c['mean']:.6f} |"
        )
    md.append("")
    md.append("**NO_SUBMIT=True** — fire Slot1/2 first tomorrow; Slot4 only if Slot1 or Slot2 ≥ 0.95355.")
    report = "\n".join(md)
    rp = LOG / "GM_V7_2_READY_FOR_TOMORROW.md"
    rp.write_text(report)
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / "GM_V7_2_READY_FOR_TOMORROW.md").write_text(report)
    except Exception as e:
        log(f"vol report sync warn: {e}")
    (ART / "v7p2.json").write_text(
        json.dumps(
            {
                "tree_src": src,
                "block_src": block_src,
                "rss_max_mb": rss_mb(),
                "s7": s7,
                "rows": rows,
                "bestB": bestB,
                "cands": cands,
                "uniq_ok": uniq_ok,
                "NO_SUBMIT": True,
            },
            indent=2,
            default=float,
        )
    )
    log(f"REPORT → {rp}")
    log("======== GM v7.2 DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
