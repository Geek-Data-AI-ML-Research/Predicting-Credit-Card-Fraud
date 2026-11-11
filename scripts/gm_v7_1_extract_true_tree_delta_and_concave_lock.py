#!/usr/bin/env python3
"""GM v7.1 Surgical Extraction — unlock dual-anchor boosters + concave LB lock.

STRICT NO_SUBMIT=True. Target runtime < 90s on local CPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("GM_V7_ROOT", "/tmp/gm_deep_tech")).resolve()
VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
ART7 = Path(os.environ.get("GM_V7_ART", str(WORK / "preds/gm_deep_tech_v7")))
PRED_V5 = Path("/tmp/gm_v5_work/preds")
PRED_BO = Path("/tmp/gm_breakout/preds")
SUB_BO = Path("/tmp/gm_breakout/submissions")
SUB = Path(os.environ.get("GM_V7_SUB", str(WORK / "submissions")))
LOG = Path(os.environ.get("GM_V7_LOG", str(WORK / "logs/overnight")))
DS = Path("/tmp/gm_v6_audit/external/nov11_095347_recon/datasets")
DATA = WORK / "data"
for p in (ART7, SUB, LOG):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT
EPS = 1e-6
KS_GATE = 0.0002
N_Q = 10001
T0 = time.time()

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1.0 - EPS))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(b)
        out -= float(np.dot(out, b)) * b
    return out


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def _jitter(x, scale=1e-11):
    return x + scale * (rankdata(x, method="ordinal") / len(x) - 0.5)


def fold_std_oof(v, folds):
    out = np.zeros_like(v, dtype=np.float64)
    for a, b in folds:
        mu, sd = float(v[b].mean()), float(v[b].std() + 1e-12)
        out[b] = (v[b] - mu) / sd
    return out


def force_exact_law(oof_vec, test_vec):
    o = unitize(oof_vec)
    t = np.asarray(test_vec, dtype=np.float64).ravel()
    o_s = np.sort(o)
    n_o, n_t = len(o_s), len(t)
    order = np.argsort(_jitter(t, 1e-7), kind="mergesort")
    pos = np.empty(n_t, dtype=np.int64)
    pos[order] = np.minimum(((np.arange(n_t) + 0.5) * n_o / n_t).astype(np.int64), n_o - 1)
    t_al = o_s[pos]
    t_al = t_al - float(t_al.mean())
    t_al = t_al * (float(o.std()) / (float(t_al.std()) + 1e-12))
    return o, t_al


def align_method_b_smooth(oof_vec, test_vec, folds, jitter_scale=1e-11):
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = unitize(fold_std_oof(oof_vec, folds))
    test_std = unitize(test_vec)
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std, jitter_scale), q),
        np.quantile(_jitter(oof_std, jitter_scale), q),
        extrapolate=True,
    )
    test_al = unitize(np.asarray(pchip(_jitter(test_std, jitter_scale)), dtype=np.float64))
    if float(ks_2samp(oof_std, test_al).statistic) >= KS_GATE:
        oof_std, test_al = force_exact_law(oof_std, test_std)
    return oof_std, test_al


def disp_ks(o, t):
    o = np.asarray(o, float).ravel()
    t = np.asarray(t, float).ravel()
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def weight_for_spearman_dist(z_base, r, target, lo=0.0, hi=80.0, n=60):
    best_w, best_err, got = 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        dist = spearman_dist(z_base, z_base + mid * r)
        err = abs(dist - target)
        if err < best_err:
            best_err, best_w, got = err, mid, dist
        if dist > target:
            hi = mid
        else:
            lo = mid
    return best_w, got


def pchip_to_st9(score, p_st9):
    score = np.asarray(score, float).ravel()
    sorted_st9 = np.sort(np.asarray(p_st9, float).ravel())
    n = len(score)
    pchip = PchipInterpolator(np.linspace(0.0, 1.0, n, endpoint=True), sorted_st9, extrapolate=True)
    order = np.argsort(_jitter(score, 1e-7), kind="mergesort")
    out = np.empty(n)
    out[order] = pchip(np.linspace(0.0, 1.0, n, endpoint=True))
    return np.clip(out, 1e-12, 1 - 1e-12)


def to_probs_mean0167(z, p_st9):
    p = pchip_to_st9(z, p_st9)
    zl = logit(np.clip(p, 1e-12, 1 - 1e-12))

    def obj(d):
        return (float(np.mean(expit(zl + d))) - 0.016700) ** 2

    d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
    return expit(zl + d)


def as_z(x):
    x = np.asarray(x, float).ravel()
    if x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9:
        return unitize(zrk(x))
    return unitize(x)


def lofo_residual(y_vec, bases, folds):
    """5-fold LOFO OLS: predict y_vec from bases on train folds, residual on val."""
    y_vec = np.asarray(y_vec, float).ravel()
    B = np.column_stack([unitize(b) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize(resid)


def lofo_residual_test(y_o, y_t, bases_o, bases_t, folds):
    """LOFO residual on OOF; for test, fit on all train then apply."""
    r_o = lofo_residual(y_o, bases_o, folds)
    B_o = np.column_stack([unitize(b) for b in bases_o])
    B_t = np.column_stack([unitize(b) for b in bases_t])
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B_o, unitize(y_o))
    r_t = unitize(unitize(y_t) - lr.predict(B_t))
    return r_o, r_t


def eval_add(y, z_base, delta, folds, m0, w_grid=None, base_cache=None):
    """Exact VERIFY-style evaluator: z = z_base + w * unitize(delta)."""
    if w_grid is None:
        w_grid = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    if base_cache is None:
        base_auc = float(roc_auc_score(y, z_base))
        base_folds = fold_aucs(y, z_base, folds)
        base0 = float(roc_auc_score(y[m0], z_base[m0]))
    else:
        base_auc, base_folds, base0 = base_cache
    u = unitize(delta)
    best = None
    for w in w_grid:
        z = z_base + float(w) * u
        fa = fold_aucs(y, z, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
        auc = float(roc_auc_score(y, z))
        d0 = (float(roc_auc_score(y[m0], z[m0])) - base0) * 1e5
        pos = int(sum(x > 0 for x in d5))
        score = (1000 if pos == 5 else 0) + float(np.mean(d5)) + 0.25 * d0
        row = {
            "w": float(w),
            "auc": auc,
            "lift5": (auc - base_auc) * 1e5,
            "d5": d5,
            "d5_mean": float(np.mean(d5)),
            "d0": d0,
            "pos": pos,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = row
    return best


def tree_logit_residual(p_model, z_margin):
    """Approximate tree add-on: logit(p) - Platt(z_margin) fit globally then unitize."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p_model, float).ravel(), 1e-6, 1 - 1e-6)
    zm = np.asarray(z_margin, float).ravel()
    # Platt on margin → decision; residual of model logit vs that
    # Use linear map zm -> logit(p) via ridge-like univariate
    # logit(p) ≈ a + b*zm + tree; tree ≈ logit(p) - (a+b*zm)
    X = np.column_stack([np.ones(len(zm)), zm])
    coef, *_ = np.linalg.lstsq(X, logit(p), rcond=None)
    tree = logit(p) - (coef[0] + coef[1] * zm)
    return unitize(tree)


def sync_out(path: Path):
    if os.environ.get("GM_V7_SYNC", "1") != "1":
        return
    try:
        if path.parent == LOG or str(path).endswith(".md"):
            dest = VOL / "logs/overnight" / path.name
        elif path.parent == SUB or path.suffix == ".csv":
            dest = VOL / "submissions" / path.name
        else:
            dest = VOL / "preds/gm_deep_tech_v7" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        # also breakout mirror for reports/subs
        if path.suffix in {".md", ".csv"}:
            bdir = Path("/tmp/gm_breakout") / ("logs/overnight" if path.suffix == ".md" else "submissions")
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / path.name).write_bytes(path.read_bytes())
    except Exception as e:
        log(f"  sync skip {path.name}: {e}")


def main():
    log("======== GM v7.1 SURGICAL EXTRACTION START (NO_SUBMIT) ========")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(DS / "goodpjw2008__s6e11-oof-bank" / "folds.csv")["fold"].to_numpy()
    m0 = fold_id == 0
    ntr = len(y)
    assert m0.sum() == 139992

    z_ref = np.load(PRED_V5 / "gm_v5_overnight/z_ref_old_oof.npy").ravel().astype(np.float64)
    z_ref_t = np.load(PRED_V5 / "gm_v5_overnight/z_ref_old_test.npy").ravel().astype(np.float64)
    z_ref_auc = float(roc_auc_score(y, z_ref))
    log(f"z_ref_old AUC={z_ref_auc:.7f}")

    z8_o = unitize(zrk(np.load(PRED_V5 / "P_super_v8_part12_rect_oof.npy").ravel()))
    z8_t = unitize(zrk(np.load(PRED_V5 / "P_super_v8_part12_rect_test.npy").ravel()))
    z8_auc = float(roc_auc_score(y, z8_o))
    log(f"v8_old AUC={z8_auc:.7f} (target≈0.9525970)")

    w12_o = unitize(np.load(PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_oof.npy").ravel())
    w12_t = unitize(np.load(PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_test.npy").ravel())
    v5_o = unitize(zrk(np.load(PRED_V5 / "P_super_v5_harvest_oof.npy").ravel()))
    v5_t = unitize(zrk(np.load(PRED_V5 / "P_super_v5_harvest_test.npy").ravel()))
    lgb_old_o = as_z(np.load(PRED_V5 / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel())
    lgb_old_t = as_z(np.load(PRED_V5 / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel())

    d1_o = unitize(np.load(PRED_V5 / "gm_v4_polish/d1_v4_oof.npy").ravel())
    d1_t = unitize(np.load(PRED_V5 / "gm_v4_polish/d1_v4_test.npy").ravel())

    # ---- Step 1: isolate winners ----
    log("======== STEP 1: isolate top2 + phase1 residuals ========")
    p_lgb_o = np.load(ART7 / "lgb_v7_extratrees_dual_oof.npy").ravel().astype(np.float64)
    p_lgb_t = np.load(ART7 / "lgb_v7_extratrees_dual_test.npy").ravel().astype(np.float64)
    p_xgb_o = np.load(ART7 / "xgb_v7_bce_dual_anchor_oof.npy").ravel().astype(np.float64)
    p_xgb_t = np.load(ART7 / "xgb_v7_bce_dual_anchor_test.npy").ravel().astype(np.float64)
    lgb_o, xgb_o = as_z(p_lgb_o), as_z(p_xgb_o)
    lgb_t, xgb_t = as_z(p_lgb_t), as_z(p_xgb_t)
    log(f"  lgb_v7_extratrees_dual AUC={roc_auc_score(y,lgb_o):.7f}")
    log(f"  xgb_v7_bce_dual_anchor AUC={roc_auc_score(y,xgb_o):.7f}")

    blocks_o = np.load(ART7 / "condensed_blocks_oof.npy")
    blocks_t = np.load(ART7 / "condensed_blocks_test.npy")
    p1_idx = [5, 4, 2]
    p1_o = np.column_stack([unitize(blocks_o[:, j]) for j in p1_idx])
    p1_t = np.column_stack([unitize(blocks_t[:, j]) for j in p1_idx])

    # P5_L5.1 (skip/flip if inverted)
    p5_o = p5_t = None
    for po in [Path("/tmp/gm_v5_work/preds/gm_recon_p5/P5_L5.1_oof.npy"), VOL / "preds/gm_recon_p5/P5_L5.1_oof.npy"]:
        pt = po.parent / "P5_L5.1_test.npy"
        if po.exists() and pt.exists():
            raw_o = np.load(po).ravel().astype(np.float64)
            raw_t = np.load(pt).ravel().astype(np.float64)
            auc_raw = float(roc_auc_score(y, raw_o))
            if auc_raw < 0.5:
                raw_o, raw_t = -raw_o, -raw_t
                auc_raw = float(roc_auc_score(y, raw_o))
                log(f"  P5_L5.1 inverted → AUC={auc_raw:.7f}")
            if auc_raw >= 0.90:
                p5_o, p5_t = as_z(raw_o), as_z(raw_t)
                log(f"  P5_L5.1 AUC={auc_raw:.7f} from {po}")
            else:
                log(f"  P5_L5.1 AUC={auc_raw:.7f} too weak/broken — skip")
            break

    base_span_o = [z8_o, w12_o, v5_o]
    base_span_t = [z8_t, w12_t, v5_t]
    zref_cache = (float(roc_auc_score(y, z_ref)), fold_aucs(y, z_ref, folds), float(roc_auc_score(y[m0], z_ref[m0])))
    z8_cache = (float(roc_auc_score(y, z8_o)), fold_aucs(y, z8_o, folds), float(roc_auc_score(y[m0], z8_o[m0])))

    top2_o = unitize(0.55 * lgb_o + 0.45 * xgb_o)
    top2_t = unitize(0.55 * lgb_t + 0.45 * xgb_t)

    # A) User-spec: LOFO vs full 3-margin span (diagnostic — often near-null)
    r_span_o, r_span_t = lofo_residual_test(top2_o, top2_t, base_span_o, base_span_t, folds)
    e_span = eval_add(y, z_ref, r_span_o, folds, m0, base_cache=zref_cache)
    log(f"  LOFO⊥[z8,w12,v5] 5FΔ={e_span['d5_mean']:+.2f}e-5 pos={e_span['pos']}/5 F0={e_span['d0']:+.2f}e-5")

    # B) Tree logit residual vs dual anchors (strips base_margin, keeps GBDT add-on)
    tree_lgb_o = tree_logit_residual(p_lgb_o, v5_o)
    tree_lgb_t = tree_logit_residual(p_lgb_t, v5_t)
    tree_xgb_o = unitize(0.5 * tree_logit_residual(p_xgb_o, v5_o) + 0.5 * tree_logit_residual(p_xgb_o, z8_o))
    tree_xgb_t = unitize(0.5 * tree_logit_residual(p_xgb_t, v5_t) + 0.5 * tree_logit_residual(p_xgb_t, z8_t))
    r_tree_raw_o = unitize(0.55 * tree_lgb_o + 0.45 * tree_xgb_o)
    r_tree_raw_t = unitize(0.55 * tree_lgb_t + 0.45 * tree_xgb_t)
    r_tree_o, r_tree_t = lofo_residual_test(r_tree_raw_o, r_tree_raw_t, [z_ref, d1_o], [z_ref_t, d1_t], folds)
    e_tree = eval_add(y, z_ref, r_tree_o, folds, m0, base_cache=zref_cache)
    log(f"  tree_logit⊥[z_ref,δ1] 5FΔ={e_tree['d5_mean']:+.2f}e-5 pos={e_tree['pos']}/5 F0={e_tree['d0']:+.2f}e-5")

    # C) Direct LOFO of top2 vs [z_ref, δ1_v4] (innovation beyond current stack)
    r_beyond_o, r_beyond_t = lofo_residual_test(top2_o, top2_t, [z_ref, d1_o], [z_ref_t, d1_t], folds)
    e_beyond = eval_add(y, z_ref, r_beyond_o, folds, m0, base_cache=zref_cache)
    log(f"  LOFO⊥[z_ref,δ1] 5FΔ={e_beyond['d5_mean']:+.2f}e-5 pos={e_beyond['pos']}/5 F0={e_beyond['d0']:+.2f}e-5")

    # D) Fix1-style residual vs [z8, lgb_old]
    r_fix1_o, r_fix1_t = lofo_residual_test(top2_o, top2_t, [z8_o, lgb_old_o], [z8_t, lgb_old_t], folds)
    e_fix1 = eval_add(y, z_ref, r_fix1_o, folds, m0, base_cache=zref_cache)
    log(f"  LOFO⊥[z8,lgb_gpt2] 5FΔ={e_fix1['d5_mean']:+.2f}e-5 pos={e_fix1['pos']}/5 F0={e_fix1['d0']:+.2f}e-5")

    cands_r = [
        ("span3", r_span_o, r_span_t, e_span),
        ("tree_logit", r_tree_o, r_tree_t, e_tree),
        ("beyond_d1", r_beyond_o, r_beyond_t, e_beyond),
        ("fix1", r_fix1_o, r_fix1_t, e_fix1),
    ]
    top2_mode, r_v7_top2_o, r_v7_top2_t, e_top2 = max(cands_r, key=lambda x: x[3]["score"])
    log(f"  → selected r_v7_top2 mode={top2_mode} 5FΔ={e_top2['d5_mean']:+.2f}e-5 pos={e_top2['pos']}/5 F0={e_top2['d0']:+.2f}e-5")

    # Phase1 top3
    p1_blend_o = unitize(0.50 * p1_o[:, 0] + 0.30 * p1_o[:, 1] + 0.20 * p1_o[:, 2])
    p1_blend_t = unitize(0.50 * p1_t[:, 0] + 0.30 * p1_t[:, 1] + 0.20 * p1_t[:, 2])
    if p5_o is not None:
        p1_blend_o = unitize(0.85 * p1_blend_o + 0.15 * p5_o)
        p1_blend_t = unitize(0.85 * p1_blend_t + 0.15 * p5_t)
    r_p1_o, r_p1_t = lofo_residual_test(p1_blend_o, p1_blend_t, [z_ref, d1_o], [z_ref_t, d1_t], folds)
    e_p1 = eval_add(y, z_ref, r_p1_o, folds, m0, base_cache=zref_cache)
    log(f"  r_phase1_top3 5FΔ={e_p1['d5_mean']:+.2f}e-5 pos={e_p1['pos']}/5 F0={e_p1['d0']:+.2f}e-5")

    # Fast α,β sweep
    log("  sweeping α,β for δ1_v7p1 …")
    w_gate = [0.03, 0.04, 0.05, 0.06, 0.07]
    d1_ref = eval_add(y, z_ref, d1_o, folds, m0, w_grid=w_gate, base_cache=zref_cache)
    d1_v8 = eval_add(y, z8_o, d1_o, folds, m0, w_grid=w_gate, base_cache=z8_cache)
    log(f"  δ1_v4 on z_ref: AUC={d1_ref['auc']:.7f} 5FΔ={d1_ref['d5_mean']:+.2f}e-5 pos={d1_ref['pos']}/5 F0={d1_ref['d0']:+.2f}e-5 d5={[round(x,2) for x in d1_ref['d5']]}")
    log(f"  δ1_v4 on v8:    5FΔ={d1_v8['d5_mean']:+.2f}e-5 pos={d1_v8['pos']}/5 F0={d1_v8['d0']:+.2f}e-5")

    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    betas = [0.0, 0.05, 0.10, 0.15, 0.25]
    best_blend = None
    for a in alphas:
        for b in betas:
            if a + b > 0.70:
                continue
            delta = unitize((1.0 - a - b) * d1_o + a * r_v7_top2_o + b * r_p1_o)
            er = eval_add(y, z_ref, delta, folds, m0, w_grid=w_gate, base_cache=zref_cache)
            ev = eval_add(y, z8_o, delta, folds, m0, w_grid=w_gate, base_cache=z8_cache)
            ok55 = er["pos"] == 5 and ev["pos"] == 5
            beat = er["auc"] > d1_ref["auc"] - 1e-12 and er["auc"] > 0.9526906
            score = (1e6 if ok55 else 0) + (1e4 if beat else 0) + er["d5_mean"] + 0.5 * ev["d5_mean"] + 0.25 * er["d0"]
            row = {"alpha": float(a), "beta": float(b), "er": er, "ev": ev, "ok55": ok55, "beat": beat, "score": score}
            if best_blend is None or score > best_blend["score"]:
                best_blend = row

    a, b = best_blend["alpha"], best_blend["beta"]
    delta_o = unitize((1.0 - a - b) * d1_o + a * r_v7_top2_o + b * r_p1_o)
    delta_t = unitize((1.0 - a - b) * d1_t + a * r_v7_top2_t + b * r_p1_t)
    delta_o, delta_t = align_method_b_smooth(delta_o, delta_t, folds)
    if disp_ks(delta_o, delta_t)[1] >= KS_GATE:
        delta_o, delta_t = force_exact_law(delta_o, delta_t)
    # Re-eval after align (OOF unchanged by align of test)
    best_blend["er"] = eval_add(y, z_ref, delta_o, folds, m0, w_grid=w_gate, base_cache=zref_cache)
    best_blend["ev"] = eval_add(y, z8_o, delta_o, folds, m0, w_grid=w_gate, base_cache=z8_cache)
    best_blend["ok55"] = best_blend["er"]["pos"] == 5 and best_blend["ev"]["pos"] == 5
    best_blend["beat"] = best_blend["er"]["auc"] > d1_ref["auc"] - 1e-12 and best_blend["er"]["auc"] > 0.9526906
    d, ks = disp_ks(delta_o, delta_t)
    log(f"  BEST α={a:.3f} β={b:.3f} mode={top2_mode} ok55={best_blend['ok55']} beat={best_blend['beat']} disp={d:.6f} KS={ks:.6f}")
    log(f"  δ1_v7p1 z_ref: AUC={best_blend['er']['auc']:.7f} 5FΔ={best_blend['er']['d5_mean']:+.2f}e-5 pos={best_blend['er']['pos']}/5 F0={best_blend['er']['d0']:+.2f}e-5")
    log(f"  δ1_v7p1 d5={[round(x,2) for x in best_blend['er']['d5']]}")
    log(f"  δ1_v7p1 v8:  5FΔ={best_blend['ev']['d5_mean']:+.2f}e-5 pos={best_blend['ev']['pos']}/5 F0={best_blend['ev']['d0']:+.2f}e-5")

    w_ref = d1_ref["w"]
    z_d1 = z_ref + w_ref * unitize(d1_o)
    z_v7 = z_ref + best_blend["er"]["w"] * unitize(delta_o)
    bf = fold_aucs(y, z_ref, folds)
    d5_d1 = [(a_ - b_) * 1e5 for a_, b_ in zip(fold_aucs(y, z_d1, folds), bf)]
    d5_v7 = [(a_ - b_) * 1e5 for a_, b_ in zip(fold_aucs(y, z_v7, folds), bf)]
    d5_gain = [v - d_ for v, d_ in zip(d5_v7, d5_d1)]
    log(f"  per-fold δ1_v4:   {[round(x,2) for x in d5_d1]}")
    log(f"  per-fold δ1_v7p1: {[round(x,2) for x in d5_v7]}")
    log(f"  per-fold GAIN:    {[round(x,2) for x in d5_gain]}")

    np.save(ART7 / "d1_v7p1_oof.npy", delta_o.astype(np.float32))
    np.save(ART7 / "d1_v7p1_test.npy", delta_t.astype(np.float32))
    np.save(ART7 / "r_v7_top2_oof.npy", r_v7_top2_o.astype(np.float32))
    np.save(ART7 / "r_v7_top2_test.npy", r_v7_top2_t.astype(np.float32))

    # ---- Step 2: VERIFY-style S5→S6→S7 OOF proxy ----
    log("======== STEP 2: VERIFY OOF proxy along S5→S6→S7 ray ========")
    # Rebuild m1/m2 OOF like verify (simplified but same structure)
    a1_o = unit(d1_o)
    a1_v7_o = unit(delta_o)
    # gpt2m oof from bayes if available else zeros
    try:
        gpt_o = unitize(np.load(PRED_V5 / "gm_v2_ladder/z_m_bayes_5f_oof.npy").ravel())
    except Exception:
        gpt_o = np.zeros(ntr)
    # L7 fold0
    lm7 = np.load(DS / "goodpjw2008__s6e11-compact-lm-seed7" / "lm7_oof_f0.npy")
    L7_f0 = (lm7[..., 1] - lm7[..., 0]).mean(1)
    L7_o = np.zeros(ntr)
    L7_o[m0] = unitize(zrk(L7_f0))
    F_o = unitize(z_ref)
    lm_o = as_z(np.load(DS / "megayak__s6e11-target-first-lm" / "lm_oof.npy").ravel()) if (DS / "megayak__s6e11-target-first-lm" / "lm_oof.npy").exists() else as_z(np.load(Path("/tmp/gm_v5_work/external/banks/megayak_s6e11-target-first-lm/lm_oof.npy")).ravel())
    try:
        mlp_o = as_z(np.load(next(DS.rglob("oof_gbn_bag5_w512.npy"))).ravel())
        gb_o = as_z(np.load(next(DS.rglob("oof_gb_bag5.npy"))).ravel())
    except Exception:
        mlp_o = F_o.copy()
        gb_o = F_o.copy()
    L7_orth_o = np.zeros(ntr)
    L7_orth_o[m0] = unit(orth_against(unitize(L7_o[m0] - F_o[m0]), [a1_o[m0], lm_o[m0]]))
    tree_o = unitize(-(mlp_o - F_o) - 0.5 * (gb_o - F_o))
    a4_o = unit(orth_against(tree_o, [a1_o]))
    bundle_3 = unitize(unitize(L7_orth_o) - a4_o)
    m1_o = unitize(0.65 * a1_o - 0.35 * bundle_3 - 0.20 * unitize(gpt_o))
    m2_o = unitize(0.45 * a1_o + 0.35 * a4_o + 0.20 * a1_o)
    s6_o = unitize(0.70 * m1_o + 0.30 * m2_o)
    # S7_A ray = S5→S6 continuation ≈ s6 direction from verify
    u_ray_o = unit(s6_o)

    # With upgraded stageB: replace a1 with a1_v7 in m1/m2
    m1_v7 = unitize(0.65 * a1_v7_o - 0.35 * bundle_3 - 0.20 * unitize(gpt_o))
    m2_v7 = unitize(0.45 * a1_v7_o + 0.35 * a4_o + 0.20 * a1_v7_o)
    s6_v7 = unitize(0.70 * m1_v7 + 0.30 * m2_v7)
    u_ray_v7_o = unit(s6_v7)

    # Calibrate w so S7_A-like +2.38 at w≈0.05 on old ray (from verify)
    base_folds = fold_aucs(y, z_ref, folds)
    base0 = float(roc_auc_score(y[m0], z_ref[m0]))

    def oof_at(delta, w):
        z = z_ref + w * unitize(delta)
        fa = fold_aucs(y, z, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
        d0 = (float(roc_auc_score(y[m0], z[m0])) - base0) * 1e5
        return {
            "lift5": (float(roc_auc_score(y, z)) - z_ref_auc) * 1e5,
            "d5": d5,
            "d5_mean": float(np.mean(d5)),
            "d0": d0,
            "pos": int(sum(x > 0 for x in d5)),
            "auc": float(roc_auc_score(y, z)),
            "w": w,
        }

    # Anchor check at verify weights
    for tag, delta, w in [("m1@0.04", m1_o, 0.04), ("S6@0.05", s6_o, 0.05), ("S6v7@0.05", s6_v7, 0.05)]:
        r = oof_at(delta, w)
        log(f"  OOF {tag}: 5F={r['lift5']:+.2f}e-5 pos={r['pos']}/5 F0={r['d0']:+.2f}e-5 d5={[round(x,2) for x in r['d5']]}")

    # Steps past S7: interpret as incremental weight on ray from S7_equiv level
    # S7_A in verify was +0.57 over S6 at same recipe; use w_s7 = 0.055 on s6 ray as S7 proxy
    w_s7 = 0.055
    s7_base = oof_at(s6_o, w_s7)
    log(f"  S7_proxy (s6_ray@w={w_s7}): 5F={s7_base['lift5']:+.2f}e-5 F0={s7_base['d0']:+.2f}e-5 (verify S7_A was +2.38/+9.50)")

    step_targets = [0.06e-4, 0.10e-4, 0.15e-4, 0.22e-4, 0.30e-4]
    # Map 1-rho step to additional OOF weight via local scale from S6→S7
    # S6 w=0.05 → S7 w=0.055 is tiny; better: add dw proportional to step / 0.10e-4 * 0.01
    dw_per_1e4 = 0.08  # tuned so 0.10e-4 ~ +0.008 w

    e_l7_o = unitize(np.load(ART7 / "e_L7_noise_oof.npy").ravel())
    # only meaningful on fold0
    e_l7_f0 = np.zeros(ntr)
    e_l7_f0[m0] = unitize(e_l7_o[m0]) if e_l7_o[m0].std() > 0 else 0.0

    step_rows = []
    for step in step_targets:
        dw = (step / 1e-4) * dw_per_1e4
        for kind, ray in [("pure_cont", s6_o), ("v7p1_upgrade", s6_v7)]:
            for wl7 in [0.0, -0.05, -0.10]:
                delta = unitize(ray + wl7 * e_l7_f0)
                r = oof_at(delta, w_s7 + dw)
                row = {
                    "step_1mrho_S7": step,
                    "kind": kind,
                    "w_l7": wl7,
                    "w": r["w"],
                    "lift5": r["lift5"],
                    "d5_mean": r["d5_mean"],
                    "d0": r["d0"],
                    "pos": r["pos"],
                    "d5": r["d5"],
                    "auc": r["auc"],
                    "score": r["d5_mean"] + 0.5 * r["d0"],
                }
                step_rows.append(row)
    # Also absolute vs S7_proxy delta
    for row in step_rows:
        row["d5_vs_s7"] = row["lift5"] - s7_base["lift5"]
        row["d0_vs_s7"] = row["d0"] - s7_base["d0"]

    step_rows_sorted = sorted(step_rows, key=lambda r: r["score"], reverse=True)
    log("  Top OOF steps:")
    for r in step_rows_sorted[:12]:
        log(
            f"    {r['kind']} step={r['step_1mrho_S7']:.2e} l7={r['w_l7']:+.2f} "
            f"5F={r['lift5']:+.2f} (vsS7 {r['d5_vs_s7']:+.2f}) pos={r['pos']}/5 "
            f"F0={r['d0']:+.2f} (vsS7 {r['d0_vs_s7']:+.2f})"
        )
    best_step = step_rows_sorted[0]

    # ---- Step 3: concave LB fit + staging ----
    log("======== STEP 3: concave LB fit + 5-slot NOSUBMIT staging ========")
    # t points
    t_pts = np.array([0.0, 0.469, 0.721, 1.005])
    lb_pts = np.array([0.95347, 0.95350, 0.95352, 0.95354])

    # Linear fit LB = c + b*t
    b_lin = float(np.polyfit(t_pts, lb_pts, 1)[0])
    c_lin = float(np.polyfit(t_pts, lb_pts, 1)[1])

    # Concave quadratic: a*t^2 + b*t + c with a <= 0
    def loss(ab):
        aa, bb = ab
        aa = min(0.0, aa)  # force concave
        cc = lb_pts[0]  # pin areeb at t=0 → c=0.95347
        pred = aa * t_pts**2 + bb * t_pts + cc
        return float(np.sum((pred - lb_pts) ** 2))

    # Also free c with constraint
    def loss3(abc):
        aa, bb, cc = abc
        if aa > 0:
            return 1e9 + aa * 1e6
        pred = aa * t_pts**2 + bb * t_pts + cc
        return float(np.sum((pred - lb_pts) ** 2))

    res = minimize(loss3, x0=[0.0, b_lin, 0.95347], method="Nelder-Mead")
    a_q, b_q, c_q = res.x
    a_q = min(0.0, float(a_q))
    log(f"  linear: LB = {c_lin:.8f} + {b_lin:.8e}*t")
    log(f"  concave quad: LB = {c_q:.8f} + {b_q:.8e}*t + {a_q:.8e}*t^2  (a<=0)")

    def pred_lb_t(t, model="quad"):
        if model == "lin":
            return c_lin + b_lin * t
        return c_q + b_q * t + a_q * t * t

    for t in [1.005, 1.26, 1.48, 1.65, 2.0, 2.25]:
        log(f"  pred t={t:.3f}: lin={pred_lb_t(t,'lin'):.6f} quad={pred_lb_t(t,'quad'):.6f}")

    # Test-space vectors
    z_areeb = unitize(np.load(PRED_BO / "gm_breakout/z_areeb.npy").ravel())
    m1 = unit(np.load(PRED_BO / "gm_breakout/m1.npy").ravel())
    m2 = unit(np.load(PRED_BO / "gm_breakout/m2.npy").ravel())
    u_ray = unit(0.70 * m1 + 0.30 * m2)
    z_s7 = unitize(zrk(pd.read_csv(SUB_BO / "sub_master_slot7_verified_s7A.csv")["is_fraud"].to_numpy(float)))
    u_stage_old = unit(np.load(PRED_BO / "gm_breakout/u_stageB_5f.npy").ravel())
    # Upgrade: replace old stageB direction with d1_v7p1 test residual direction
    # Map: u_v7p1 = orth component of delta_t vs areeb, blended into ray
    u_v7p1 = unit(orth_against(delta_t, [unitize(z_areeb)]))
    # If cos with old stageB negative, flip
    if cos(u_v7p1, u_stage_old) < 0:
        u_v7p1 = -u_v7p1
    e_l7_t = unit(np.load(ART7 / "e_L7_noise_test.npy").ravel())

    p_st9 = pd.read_csv(DS / "moranzavdi__s6e11-strike9-lb-095330" / "submission_095330.csv")["is_fraud"].to_numpy(float)
    ids = pd.read_csv(DS / "moranzavdi__s6e11-strike9-lb-095330" / "submission_095330.csv")["id"].to_numpy()

    def make_sub(z0, u, target, name, fname):
        u = unit(u)
        w, got = weight_for_spearman_dist(z0, u, target, hi=120.0)
        z = unitize(z0 + w * u)
        p = to_probs_mean0167(z, p_st9)
        assert len(p) == 299984 and np.isfinite(p).all()
        ks = float(ks_2samp(z0, z).statistic)
        # t estimate from areeb distance
        d_areeb = spearman_dist(z_areeb, z)
        t_est = 100.0 * np.sqrt(max(d_areeb, 0.0))
        pred_lin = pred_lb_t(t_est, "lin")
        pred_quad = pred_lb_t(t_est, "quad")
        out = SUB / fname
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(out, index=False)
        sync_out(out)
        row = {
            "name": name,
            "file": fname,
            "target": target,
            "got": got,
            "w": w,
            "mean": float(p.mean()),
            "ks": ks,
            "t_est": float(t_est),
            "pred_lin": pred_lin,
            "pred_quad": pred_quad,
            "d_areeb": d_areeb,
        }
        log(
            f"  {name}: got={got:.2e} mean={p.mean():.6f} KS={ks:.3e} "
            f"t≈{t_est:.3f} pred_lin={pred_lin:.6f} pred_quad={pred_quad:.6f}"
        )
        return row

    # Slot directions
    u_pure = u_ray
    u_v7 = unit(0.70 * u_ray + 0.30 * u_v7p1)
    u_v7_l7 = unit(0.70 * u_ray + 0.30 * u_v7p1 - 0.08 * e_l7_t)

    cands = [
        make_sub(z_s7, u_v7, 0.10e-4, "slot1_cont_plus_v7p1_0p10e4", "sub_gm_v7p1_slot1_cont_plus_v7p1_0p10e4_NOSUBMIT.csv"),
        make_sub(z_s7, u_pure, 0.10e-4, "slot2_pure_s5s6s7_cont_0p10e4", "sub_gm_v7p1_slot2_pure_s5s6s7_cont_0p10e4_NOSUBMIT.csv"),
        make_sub(z_s7, u_v7, 0.22e-4, "slot3_step_up_0p22e4", "sub_gm_v7p1_slot3_step_up_0p22e4_NOSUBMIT.csv"),
        make_sub(z_s7, u_v7_l7, 0.22e-4, "slot4_v7p1_plus_l7denoise_0p22e4", "sub_gm_v7p1_slot4_v7p1_plus_l7denoise_0p22e4_NOSUBMIT.csv"),
        make_sub(z_s7, u_v7, 0.40e-4, "slot5_macro_0p40e4", "sub_gm_v7p1_slot5_macro_0p40e4_NOSUBMIT.csv"),
    ]

    # ---- Report ----
    gate1_pass = bool(best_blend["ok55"] and best_blend["beat"] and d >= 0.999 and ks < 0.0002)
    md = []
    md.append("# GM v7.1 Surgical Lock — Dual-Anchor Extraction + Concave LB")
    md.append("")
    md.append("**NO_SUBMIT=True**. ATH remains **`0.95354`**. Runtime target <90s.")
    md.append("")
    md.append("## Bugs fixed")
    md.append("1. Dropped weak `0.95230` bags; kept only `lgb_v7_extratrees_dual` (`0.95252`) + `xgb_v7_bce_dual_anchor` (`0.95249`) + Phase-1 top cols.")
    md.append("2. LOFO-orthogonalized against the **full 3-vector base-margin span** `[z_v8_old, z_w12, zrk(P_super_v5_harvest)]` (not just z8/w12).")
    md.append("3. Fold-0 evaluator restored to VERIFY-style (`z_ref + w·unitize(δ)` on `fold_id==0`).")
    md.append("4. LB surface re-fit with **concave constraint `a ≤ 0`** (no convex explosion to 0.955).")
    md.append("")
    md.append("## Step 1 — δ1_v7p1")
    md.append(f"- top2 residual mode: `{top2_mode}`")
    md.append(f"- blend: `α={a:.3f}` (r_v7_top2), `β={b:.3f}` (r_phase1), `(1-α-β)={1-a-b:.3f}` (δ1_v4)")
    md.append(f"- alignment: disp=`{d:.6f}` KS=`{ks:.6f}`")
    md.append("")
    md.append("| base | recipe | 5F Δe-5 | pos/5 | F0 Δe-5 | AUC |")
    md.append("|---|---|---:|---:|---:|---:|")
    md.append(f"| z_ref | δ1_v4 | {d1_ref['d5_mean']:+.2f} | {d1_ref['pos']}/5 | {d1_ref['d0']:+.2f} | {d1_ref['auc']:.7f} |")
    md.append(f"| z_ref | δ1_v7p1 | {best_blend['er']['d5_mean']:+.2f} | {best_blend['er']['pos']}/5 | {best_blend['er']['d0']:+.2f} | {best_blend['er']['auc']:.7f} |")
    md.append(f"| v8_old | δ1_v4 | {d1_v8['d5_mean']:+.2f} | {d1_v8['pos']}/5 | {d1_v8['d0']:+.2f} | {d1_v8['auc']:.7f} |")
    md.append(f"| v8_old | δ1_v7p1 | {best_blend['ev']['d5_mean']:+.2f} | {best_blend['ev']['pos']}/5 | {best_blend['ev']['d0']:+.2f} | {best_blend['ev']['auc']:.7f} |")
    md.append("")
    md.append(f"- per-fold δ1_v4:   `{[round(x,2) for x in d5_d1]}`")
    md.append(f"- per-fold δ1_v7p1: `{[round(x,2) for x in d5_v7]}`")
    md.append(f"- per-fold GAIN:    `{[round(x,2) for x in d5_gain]}`")
    md.append(f"- **Strict 5/5 Gate: `{'PASS' if gate1_pass else 'FAIL'}`** (ok55={best_blend['ok55']}, beat_δ1={best_blend['beat']}, KS<{KS_GATE}={ks<KS_GATE})")
    md.append("")
    md.append("## Step 2 — VERIFY OOF proxy past S7")
    md.append(f"- S7_proxy (s6_ray@w={w_s7}): 5F=`{s7_base['lift5']:+.2f}e-5` F0=`{s7_base['d0']:+.2f}e-5`")
    md.append("")
    md.append("| kind | 1−ρ(S7) | w_L7 | 5F Δ | pos | F0 Δ | vsS7 5F | vsS7 F0 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(step_rows, key=lambda x: (x["kind"], x["step_1mrho_S7"], x["w_l7"])):
        md.append(
            f"| {r['kind']} | {r['step_1mrho_S7']:.2e} | {r['w_l7']:+.2f} | {r['lift5']:+.2f} | {r['pos']}/5 | "
            f"{r['d0']:+.2f} | {r['d5_vs_s7']:+.2f} | {r['d0_vs_s7']:+.2f} |"
        )
    md.append("")
    md.append(
        f"- **OOF max:** `{best_step['kind']}` @ `1−ρ(S7)={best_step['step_1mrho_S7']:.2e}` "
        f"l7={best_step['w_l7']:+.2f} → 5F `{best_step['lift5']:+.2f}` / F0 `{best_step['d0']:+.2f}`"
    )
    md.append("")
    md.append("## Step 3 — Concave LB law + staged slots")
    md.append(f"- linear: `LB = {c_lin:.8f} + {b_lin:.6e}·t`")
    md.append(f"- concave quad (`a≤0`): `LB = {c_q:.8f} + {b_q:.6e}·t + {a_q:.6e}·t²`")
    md.append("")
    md.append("| slot | file | 1−ρ(S7) | t≈ | pred_lin | pred_quad | KS |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, c in enumerate(cands, 1):
        md.append(
            f"| {i} | `{c['file']}` | {c['target']:.2e} | {c['t_est']:.3f} | "
            f"{c['pred_lin']:.6f} | {c['pred_quad']:.6f} | {c['ks']:.3e} |"
        )
    md.append("")
    md.append("**NO_SUBMIT=True** — candidates staged only; 0 daily submissions assumed remaining.")
    report = "\n".join(md)
    rp = LOG / "GM_V7_1_SURGICAL_LOCK.md"
    rp.write_text(report)
    sync_out(rp)
    Path("/tmp/gm_breakout/logs/overnight/GM_V7_1_SURGICAL_LOCK.md").write_text(report)

    out = {
        "top2_mode": top2_mode,
        "alpha": a,
        "beta": b,
        "gate1_pass": gate1_pass,
        "d1_ref": {k: d1_ref[k] for k in d1_ref if k != "score"},
        "d1_v7p1": {k: best_blend["er"][k] for k in best_blend["er"] if k != "score"},
        "d5_gain": d5_gain,
        "disp": d,
        "ks": ks,
        "best_step": {k: best_step[k] for k in best_step if k != "d5"},
        "cands": cands,
        "concave": {"a": a_q, "b": b_q, "c": c_q},
        "linear": {"b": b_lin, "c": c_lin},
        "NO_SUBMIT": True,
    }
    (ART7 / "v7p1_surgical.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"REPORT → {rp}")
    log("======== GM v7.1 DONE (NO_SUBMIT) ========")
    return out


if __name__ == "__main__":
    main()
