#!/usr/bin/env python3
"""GM v4.1 — Fast binary cache + Gram-Schmidt orthonormal lock (STRICT NO_SUBMIT).

Fixes Fix-3 multicollinearity (d_ladder / e1 / e2) via exact orthonormal geometry,
caches all hot vectors for <5s reloads, verifies joint 5/5 δ1_v4+δ3_v4, restages
tomorrow's 5-slot ladder from Today_S5.

Artifacts → preds/gm_v4_1_fast_cache.npz, preds/gm_v4_1/
Report    → logs/overnight/GM_V4_1_ORTHONORMAL_LOCK.md
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
PRED = ROOT / "preds"
ART4 = PRED / "gm_v4_polish"
ART2 = PRED / "gm_v2_ladder"
ART3 = PRED / "gm_v3_research"
ART = PRED / "gm_v4_1"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
P12 = ROOT / "external/nov08_part12"
CACHE = PRED / "gm_v4_1_fast_cache.npz"
ART.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
KS_GATE = 0.0002
N_Q = 10001

LB_NAMES = [
    ("v8", 0.95307),
    ("Slot3_old", 0.95312),
    ("St2", 0.95319),
    ("St3A", 0.95322),
    ("St4A", 0.95323),
    ("St5A", 0.95315),
    ("St6", 0.95325),
    ("St7", 0.95327),
    ("St8", 0.95329),
    ("St9", 0.95330),
    ("leader", 0.95325),
    ("soft", 0.95328),
    ("ANTI", 0.95329),
    ("St9.5", 0.95332),
    ("Today_S1", 0.95333),
    ("Today_S2", 0.95333),
    ("Today_S3", 0.95329),
    ("Today_S4", 0.95334),
    ("Today_S5", 0.95335),
]


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
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(b)
        out -= float(np.dot(out, b)) * b
    return out


def orth_unit(v, bases=None):
    """Mean-0 / std-1, then L2-unit after orth vs prior basis columns."""
    v = unitize(v)
    if bases:
        v = orth_against(v, bases)
        v = unitize(v)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


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
    oof_std = fold_std_oof(oof_vec, folds)
    oof_std = unitize(oof_std)
    test_std = unitize(test_vec)
    oof_j = _jitter(oof_std, jitter_scale)
    test_j = _jitter(test_std, jitter_scale)
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(np.quantile(test_j, q), np.quantile(oof_j, q), extrapolate=True)
    test_al = np.asarray(pchip(test_j), dtype=np.float64)
    test_al = unitize(test_al)
    if float(ks_2samp(oof_std, test_al).statistic) >= KS_GATE:
        oof_std, test_al = force_exact_law(oof_std, test_std)
    return oof_std, test_al


def disp_ks(o, t):
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def finalize_vs_bad(ro, rt, B_bad, folds, tag):
    rt_p = orth_against(rt, [B_bad[:, j] for j in range(B_bad.shape[1])])
    o_al, t_al = align_method_b_smooth(ro, rt_p, folds)
    d, ks = disp_ks(o_al, t_al)
    if ks >= KS_GATE:
        o_al, t_al = force_exact_law(o_al, t_al)
        d, ks = disp_ks(o_al, t_al)
    # re-strip any numerical leakage into B_bad, then re-force
    t_al = orth_against(t_al, [B_bad[:, j] for j in range(B_bad.shape[1])])
    o_al, t_al = force_exact_law(o_al, t_al)
    d, ks = disp_ks(o_al, t_al)
    max_cos = max(abs(cos(t_al, B_bad[:, j])) for j in range(B_bad.shape[1])) if B_bad.size else 0.0
    log(f"  {tag}: disp={d:.6f} KS={ks:.6f} max|cos(B_bad)|={max_cos:.2e}")
    return o_al, t_al, d, ks, max_cos


def weight_for_spearman_dist(z_base, r, target, lo=0.0, hi=80.0, n=56):
    best_w, best_err, got = 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        dist = 1.0 - float(spearmanr(z_base, z_base + mid * r).correlation)
        err = abs(dist - target)
        if err < best_err:
            best_err, best_w, got = err, mid, dist
        if dist > target:
            hi = mid
        else:
            lo = mid
    return best_w, got


def to_probs(z, y):
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    q = np.linspace(0, 1, 10001)
    u = (rankdata(z, method="average") - 0.5) / len(z)
    pchip = PchipInterpolator(q, np.quantile(z_v8_o, q), extrapolate=True)
    zm = np.asarray(pchip(u), float)
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_v8_o.reshape(-1, 1), y.astype(np.int32))
    zlin = float(lr.coef_[0, 0]) * zm + float(lr.intercept_[0])

    def obj(d):
        return (float(np.mean(expit(zlin + d))) - 0.016700) ** 2

    d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
    return expit(zlin + d)


# ============================================================================
# Step 1 — Build / load fast cache
# ============================================================================
def build_or_load_cache(y, folds, test_ids):
    if CACHE.exists() and os.environ.get("REBUILD_CACHE", "0") != "1":
        log(f"Loading existing cache {CACHE}")
        return dict(np.load(CACHE, allow_pickle=True))

    log("=== STEP1: building gm_v4_1_fast_cache.npz ===")
    # LB bank
    lb_cache = ART4 / "_cache_lb_z.npz"
    if lb_cache.exists():
        Zraw = np.load(lb_cache)
        Z = {k: Zraw[k].astype(np.float64) for k, _ in LB_NAMES}
        log(f"  LB z from {lb_cache}")
    else:
        raise SystemExit("Missing preds/gm_v4_polish/_cache_lb_z.npz — run v4 polish first")

    lb_scores = {k: float(s) for k, s in LB_NAMES}
    payload = {}
    for k, _ in LB_NAMES:
        payload[f"z_{k}"] = Z[k].astype(np.float32)
        payload[f"lb_{k}"] = np.array(lb_scores[k], dtype=np.float64)

    # v4 winners
    payload["d1_v4_oof"] = np.load(ART4 / "d1_v4_oof.npy").astype(np.float64)
    payload["d1_v4_test"] = np.load(ART4 / "d1_v4_test.npy").astype(np.float64)
    payload["d3_v4_lin_oof"] = np.load(ART4 / "d3_v4_linear_oof.npy").astype(np.float64)
    payload["d3_v4_lin_test"] = np.load(ART4 / "d3_v4_linear_test.npy").astype(np.float64)
    payload["d3_v4_bnd_oof"] = np.load(ART4 / "d3_v4_boundary_oof.npy").astype(np.float64)
    payload["d3_v4_bnd_test"] = np.load(ART4 / "d3_v4_boundary_test.npy").astype(np.float64)

    # zjoint / glm_graph — use Method-B aligned zrk of lab preds (unitized)
    z_zj_o = zrk(np.load(PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy").ravel())
    z_zj_t = zrk(np.load(PRED / "part11/lab1_zjoint_graph_lgb4_test.npy").ravel())
    z_gg_o = zrk(np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel())
    z_gg_t = zrk(np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel())
    _, u_zj_t = align_method_b_smooth(z_zj_o, z_zj_t, folds)
    _, u_gg_t = align_method_b_smooth(z_gg_o, z_gg_t, folds)
    payload["u_zjoint_oof"] = unitize(z_zj_o).astype(np.float64)
    payload["u_zjoint_test"] = unitize(u_zj_t).astype(np.float64)
    payload["u_glm_graph_oof"] = unitize(z_gg_o).astype(np.float64)
    payload["u_glm_graph_test"] = unitize(u_gg_t).astype(np.float64)

    # anchors
    if (ART2 / "z_ref_clean_oof.npy").exists():
        z_ref_o = zrk(np.load(ART2 / "z_ref_clean_oof.npy").ravel())
    else:
        z_ref_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    payload["z_ref_o"] = z_ref_o.astype(np.float64)
    payload["z_v8_old_oof"] = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel()).astype(np.float64)
    payload["z_w12_oof"] = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel()).astype(np.float64)
    payload["y"] = y.astype(np.int8)
    # folds as (5, 2) object of index arrays — store as concatenated with offsets
    fold_tr, fold_va, offs_tr, offs_va = [], [], [0], [0]
    for a, b in folds:
        fold_tr.append(a.astype(np.int32))
        fold_va.append(b.astype(np.int32))
        offs_tr.append(offs_tr[-1] + len(a))
        offs_va.append(offs_va[-1] + len(b))
    payload["fold_train_idx"] = np.concatenate(fold_tr)
    payload["fold_val_idx"] = np.concatenate(fold_va)
    payload["fold_train_off"] = np.asarray(offs_tr, dtype=np.int32)
    payload["fold_val_off"] = np.asarray(offs_va, dtype=np.int32)
    payload["test_ids"] = test_ids.astype(np.int64)
    payload["lb_name_order"] = np.array([k for k, _ in LB_NAMES], dtype=object)

    np.savez_compressed(CACHE, **payload)
    log(f"  wrote {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")
    return payload


def folds_from_cache(C):
    offs_tr = C["fold_train_off"]
    offs_va = C["fold_val_off"]
    tr = C["fold_train_idx"]
    va = C["fold_val_idx"]
    folds = []
    for i in range(5):
        folds.append((tr[offs_tr[i] : offs_tr[i + 1]], va[offs_va[i] : offs_va[i + 1]]))
    return folds


def Z_from_cache(C):
    Z = {}
    for k, s in LB_NAMES:
        Z[k] = {"z": np.asarray(C[f"z_{k}"], dtype=np.float64), "lb": float(np.asarray(C[f"lb_{k}"]))}
    return Z


# ============================================================================
# Step 2 — Gram-Schmidt orthonormal basis
# ============================================================================
def build_orth_basis(Z):
    log("=== STEP2: Gram-Schmidt orthonormal basis ===")
    z95 = Z["St9.5"]["z"]
    z_s1 = Z["Today_S1"]["z"]
    z_s3 = Z["Today_S3"]["z"]
    z_s4 = Z["Today_S4"]["z"]
    z_s5 = Z["Today_S5"]["z"]
    e1 = Z["St4A"]["z"] - Z["St3A"]["z"]
    e2 = (Z["St5A"]["z"] - Z["St4A"]["z"]) - 0.45 * (Z["St4A"]["z"] - Z["St3A"]["z"])
    d_bad_leader = Z["leader"]["z"] - Z["St9"]["z"]
    d_bad_soft = Z["soft"]["z"] - Z["St9"]["z"]

    q1 = orth_unit(z_s1 - z95)
    q2 = orth_unit(e1, [q1])
    q_bad_tree = orth_unit(e2, [q1, q2])
    q_bad_S3 = orth_unit(z_s3 - z_s1, [q1, q2, q_bad_tree])
    q_bad_ldr = orth_unit(d_bad_leader, [q1, q2, q_bad_tree, q_bad_S3])
    q_bad_soft = orth_unit(d_bad_soft, [q1, q2, q_bad_tree, q_bad_S3, q_bad_ldr])
    q_S5_win = orth_unit(z_s5 - z_s1, [q1, q2, q_bad_tree, q_bad_S3, q_bad_ldr, q_bad_soft])

    names = ["q1_ladder", "q2_e1", "q_bad_tree", "q_bad_S3", "q_bad_ldr", "q_bad_soft", "q_S5_win"]
    Q = np.column_stack([q1, q2, q_bad_tree, q_bad_S3, q_bad_ldr, q_bad_soft, q_S5_win])

    # Verify orthonormality
    G = Q.T @ Q
    off = G - np.eye(7)
    log(f"  Gram off-diag max|⟨qi,qj⟩|={float(np.max(np.abs(off))):.2e}  diag norms={np.round(np.diag(G),6).tolist()}")
    for j, nm in enumerate(names):
        log(f"  {nm}: mean={Q[:,j].mean():+.2e} std={Q[:,j].std():.6f} ||q||={np.linalg.norm(Q[:,j]):.6f}")

    def coords(z, anchor=z95):
        delta = unitize(z - anchor)  # express in same scale family; use raw LS on Q
        # Exact coords via inner products (Q orthonormal in L2)
        c = Q.T @ (z - anchor)
        return {names[i]: float(c[i]) for i in range(7)}

    # Also report coords of delta from S1 for S3/S4/S5 readability
    table = {}
    for key in ["Today_S1", "Today_S3", "Today_S4", "Today_S5"]:
        table[key] = {
            "lb": Z[key]["lb"],
            "vs_S95": coords(Z[key]["z"], z95),
            "vs_S1": coords(Z[key]["z"], z_s1),
        }
        c = table[key]["vs_S1"]
        log(
            f"  {key} LB={Z[key]['lb']:.5f} vs_S1: "
            + ", ".join(f"{k}={v:+.4f}" for k, v in c.items())
        )

    poison = {
        "Today_S4_c_bad_S3_vs_S1": table["Today_S4"]["vs_S1"]["q_bad_S3"],
        "Today_S5_c_bad_S3_vs_S1": table["Today_S5"]["vs_S1"]["q_bad_S3"],
        "Today_S4_c_bad_tree_vs_S1": table["Today_S4"]["vs_S1"]["q_bad_tree"],
        "Today_S5_c_bad_tree_vs_S1": table["Today_S5"]["vs_S1"]["q_bad_tree"],
        "Today_S5_c_S5_win_vs_S1": table["Today_S5"]["vs_S1"]["q_S5_win"],
    }
    log(
        f"  POISON CHECK: S4 c_bad_S3={poison['Today_S4_c_bad_S3_vs_S1']:+.4f} "
        f"S5 c_bad_S3={poison['Today_S5_c_bad_S3_vs_S1']:+.4f} "
        f"S5 c_S5_win={poison['Today_S5_c_S5_win_vs_S1']:+.4f}"
    )

    B_bad = np.column_stack([q_bad_tree, q_bad_S3, q_bad_ldr, q_bad_soft])
    return {
        "names": names,
        "Q": Q,
        "q1": q1,
        "q2": q2,
        "q_bad_tree": q_bad_tree,
        "q_bad_S3": q_bad_S3,
        "q_bad_ldr": q_bad_ldr,
        "q_bad_soft": q_bad_soft,
        "q_S5_win": q_S5_win,
        "B_bad": B_bad,
        "table": table,
        "poison": poison,
        "gram_off_max": float(np.max(np.abs(off))),
    }


# ============================================================================
# Step 3 — Joint 5/5 OOF synergy
# ============================================================================
def joint_oof_synergy(C, folds, B_bad, report):
    log("=== STEP3: joint 5/5 OOF synergy on z_ref_o ===")
    y = np.asarray(C["y"], dtype=np.int32)
    z_ref = np.asarray(C["z_ref_o"], dtype=np.float64)
    base_folds = fold_aucs(y, z_ref, folds)
    base_auc = float(roc_auc_score(y, z_ref))
    log(f"  z_ref_o AUC={base_auc:.7f} folds={[round(x,7) for x in base_folds]}")

    d1_o = np.asarray(C["d1_v4_oof"], dtype=np.float64)
    d1_t = np.asarray(C["d1_v4_test"], dtype=np.float64)
    d3l_o = np.asarray(C["d3_v4_lin_oof"], dtype=np.float64)
    d3l_t = np.asarray(C["d3_v4_lin_test"], dtype=np.float64)
    d3b_o = np.asarray(C["d3_v4_bnd_oof"], dtype=np.float64)
    d3b_t = np.asarray(C["d3_v4_bnd_test"], dtype=np.float64)
    uj_o = np.asarray(C["u_zjoint_oof"], dtype=np.float64)
    uj_t = np.asarray(C["u_zjoint_test"], dtype=np.float64)
    ug_o = np.asarray(C["u_glm_graph_oof"], dtype=np.float64)
    ug_t = np.asarray(C["u_glm_graph_test"], dtype=np.float64)

    clean = {}
    for tag, ro, rt in [
        ("d1_v4", d1_o, d1_t),
        ("d3_lin", d3l_o, d3l_t),
        ("d3_bnd", d3b_o, d3b_t),
        ("u_zjoint", uj_o, uj_t),
        ("u_glm_graph", ug_o, ug_t),
    ]:
        o, t, d, ks, mc = finalize_vs_bad(ro, rt, B_bad, folds, tag)
        clean[tag] = (o, t, d, ks, mc)

    # Nested weight search for combo on z_ref
    w_grid = np.linspace(-0.05, 0.25, 31)
    keys = ["d1_v4", "d3_lin", "d3_bnd"]
    R_o = np.column_stack([clean[k][0] for k in keys])
    R_t = np.column_stack([clean[k][1] for k in keys])

    nested = np.zeros(len(y))
    fold_w = []
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        best_w = np.zeros(3)
        best_s = -1e9
        # coordinate ascent 2 passes
        w = np.zeros(3)
        for _ in range(2):
            for j in range(3):
                bw, bs = w[j], -1e9
                for cw in w_grid:
                    wt = w.copy()
                    wt[j] = cw
                    lifts = [
                        float(roc_auc_score(y[folds[oi][1]], z_ref[folds[oi][1]] + R_o[folds[oi][1]] @ wt))
                        - base_folds[oi]
                        for oi in other
                    ]
                    s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                    if s > bs:
                        bs, bw = s, float(cw)
                w[j] = bw
        # score
        lifts = [
            float(roc_auc_score(y[folds[oi][1]], z_ref[folds[oi][1]] + R_o[folds[oi][1]] @ w)) - base_folds[oi]
            for oi in other
        ]
        s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
        if s > best_s:
            best_s, best_w = s, w.copy()
        nested[b] = z_ref[b] + R_o[b] @ best_w
        fold_w.append(best_w)
        log(f"  fold{f} w={dict(zip(keys, np.round(best_w,4)))} val_lift_others={[round(x*1e5,2) for x in lifts]}")

    mean_w = np.mean(fold_w, 0)
    nf = fold_aucs(y, nested, folds)
    deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
    pos = int(sum(x > 0 for x in deltas))
    auc = float(roc_auc_score(y, nested))
    log(f"  δ_v4_combo mean_w={dict(zip(keys, np.round(mean_w,4)))}")
    log(f"  joint Δe5={[round(x,2) for x in deltas]} pos={pos}/5 AUC={auc:.7f} (vs ref {base_auc:.7f})")

    # Build test combo + extras
    d_combo_o = R_o @ mean_w
    d_combo_t = R_t @ mean_w
    # add small good U components for later staging (not in nested OOF weights, but available)
    d_combo_o, d_combo_t, d, ks, mc = finalize_vs_bad(d_combo_o, d_combo_t, B_bad, folds, "d_v4_combo")
    np.save(ART / "d_v4_combo_oof.npy", d_combo_o)
    np.save(ART / "d_v4_combo_test.npy", d_combo_t)
    for tag, (_, t, _, _, _) in clean.items():
        np.save(ART / f"{tag}_clean_test.npy", t)

    out = {
        "base_auc": base_auc,
        "base_folds": base_folds,
        "mean_w": {k: float(w) for k, w in zip(keys, mean_w)},
        "delta_e5": deltas,
        "pos": pos,
        "auc": auc,
        "hygiene": {k: {"disp": v[2], "ks": v[3], "max_cos_bad": v[4]} for k, v in clean.items()},
        "combo_disp_ks_cos": [d, ks, mc],
    }
    report["step3"] = out
    return d_combo_o, d_combo_t, clean, mean_w, out


# ============================================================================
# Step 4 — Stage 5 candidates
# ============================================================================
def stage_candidates(y, test_ids, Z, basis, d_combo_t, clean, report):
    log("=== STEP4: stage 5 NOSUBMIT candidates from Today_S5 ===")
    z_s5 = Z["Today_S5"]["z"]
    q1 = basis["q1"]
    q_S5_win = basis["q_S5_win"]
    q_bad_S3 = basis["q_bad_S3"]
    q_bad_tree = basis["q_bad_tree"]
    B_bad = basis["B_bad"]
    poison = basis["poison"]

    # Candidate 1: subtract inherited +c_bad_S3 (and residual bad_tree) from S5, advance q_S5_win + 0.5 q1
    c_s3 = float(poison["Today_S5_c_bad_S3_vs_S1"])
    c_tree = float(poison["Today_S5_c_bad_tree_vs_S1"])
    # Geometric depoison: remove positive loadings on bad axes from S5 relative to S1 span
    # Using S5-S1 projection coeffs onto bad axes
    delta_s5_s1 = Z["Today_S5"]["z"] - Z["Today_S1"]["z"]
    c_bad_s3_raw = float(np.dot(delta_s5_s1, q_bad_S3))
    c_bad_tree_raw = float(np.dot(delta_s5_s1, q_bad_tree))
    purge = -max(0.0, c_bad_s3_raw) * q_bad_S3 - max(0.0, c_bad_tree_raw) * q_bad_tree
    advance = unitize(q_S5_win + 0.5 * q1)
    axis1 = unitize(purge + 0.35 * advance)
    # ensure no positive bad loading in the step
    axis1 = orth_against(axis1, [q_bad_S3, q_bad_tree, basis["q_bad_ldr"], basis["q_bad_soft"]])
    axis1 = unitize(axis1)
    w1, got1 = weight_for_spearman_dist(z_s5, axis1, 0.22e-4)
    z1 = z_s5 + w1 * axis1

    # Candidate 2: depoisoned S5 + δ_v4_combo @ 0.40e-4
    d_combo = unitize(orth_against(d_combo_t, [B_bad[:, j] for j in range(B_bad.shape[1])]))
    axis2 = unitize(0.40 * axis1 + 0.60 * d_combo)
    axis2 = orth_against(axis2, [B_bad[:, j] for j in range(B_bad.shape[1])])
    axis2 = unitize(axis2)
    w2, got2 = weight_for_spearman_dist(z_s5, axis2, 0.40e-4)
    z2 = z_s5 + w2 * axis2

    # Candidate 3: + good U components @ 0.75e-4
    u_good = unitize(0.5 * clean["u_zjoint"][1] + 0.5 * clean["u_glm_graph"][1])
    u_good = orth_against(u_good, [B_bad[:, j] for j in range(B_bad.shape[1])])
    axis3 = unitize(0.35 * axis1 + 0.45 * d_combo + 0.20 * unitize(u_good))
    axis3 = orth_against(axis3, [B_bad[:, j] for j in range(B_bad.shape[1])])
    axis3 = unitize(axis3)
    w3, got3 = weight_for_spearman_dist(z_s5, axis3, 0.75e-4)
    z3 = z_s5 + w3 * axis3

    # Candidate 4: large 1.30e-4
    axis4 = unitize(0.30 * axis1 + 0.50 * d_combo + 0.20 * unitize(u_good) + 0.15 * q1)
    axis4 = orth_against(axis4, [B_bad[:, j] for j in range(B_bad.shape[1])])
    axis4 = unitize(axis4)
    w4, got4 = weight_for_spearman_dist(z_s5, axis4, 1.30e-4)
    z4 = z_s5 + w4 * axis4

    # Candidate 5: adaptive reserve template @ 2.00e-4
    axis5 = unitize(0.25 * axis1 + 0.45 * d_combo + 0.15 * unitize(u_good) + 0.15 * q1 + 0.10 * basis["q2"])
    axis5 = orth_against(axis5, [B_bad[:, j] for j in range(B_bad.shape[1])])
    axis5 = unitize(axis5)
    w5, got5 = weight_for_spearman_dist(z_s5, axis5, 2.00e-4)
    z5 = z_s5 + w5 * axis5

    staged = []
    specs = [
        ("slot1_depoison_S5", z1, axis1, 0.22e-4, got1, w1),
        ("slot2_v4combo_micro_0p40e4", z2, axis2, 0.40e-4, got2, w2),
        ("slot3_v4combo_med_0p75e4", z3, axis3, 0.75e-4, got3, w3),
        ("slot4_v4combo_large_1p30e4", z4, axis4, 1.30e-4, got4, w4),
        ("slot5_adaptive_reserve", z5, axis5, 2.00e-4, got5, w5),
    ]
    for tag, z, axis, target, got, w in specs:
        delta = z - z_s5
        tel = {
            "tag": tag,
            "target_1mrho": target,
            "got_1mrho": float(1.0 - spearmanr(z, z_s5).correlation),
            "w": float(w),
            "cos_q1": cos(delta, q1),
            "cos_q2": cos(delta, basis["q2"]),
            "cos_bad_tree": cos(delta, q_bad_tree),
            "cos_bad_S3": cos(delta, q_bad_S3),
            "cos_S5_win": cos(delta, q_S5_win),
            "max_cos_B_bad": max(abs(cos(delta, B_bad[:, j])) for j in range(B_bad.shape[1])),
        }
        p = to_probs(z, y)
        fn = f"sub_gm_v4p1_{tag}_NOSUBMIT.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(SUB / fn, index=False)
        np.save(ART / f"{tag}_z.npy", z)
        np.save(ART / f"{tag}_p.npy", p)
        tel["file"] = fn
        tel["mean_p"] = float(p.mean())
        # hygiene of axis vs a forced twin
        o_syn = unitize(np.concatenate([axis, axis])[: len(y)]) if False else unitize(
            np.interp(np.linspace(0, 1, len(y)), np.linspace(0, 1, len(axis)), np.sort(axis))
        )
        o_al, t_al = force_exact_law(o_syn, axis)
        d, ks = disp_ks(o_al, t_al)
        tel["disp"] = d
        tel["ks"] = ks
        log(
            f"  staged {tag}: 1−ρ={tel['got_1mrho']*1e4:.3f}e-4 "
            f"cos(bad_S3)={tel['cos_bad_S3']:+.4f} cos(bad_tree)={tel['cos_bad_tree']:+.4f} "
            f"max|cos B_bad|={tel['max_cos_B_bad']:.2e} KS={ks:.6f} → {fn}"
        )
        staged.append(tel)

    report["step4"] = {
        "poison_coeffs_used": {
            "c_bad_S3_raw": c_bad_s3_raw,
            "c_bad_tree_raw": c_bad_tree_raw,
        },
        "staged": staged,
    }
    return staged


def write_report(report, basis):
    md = []
    md.append("# GM v4.1 — Fast Cache + Orthonormal Lock")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{report.get('elapsed_s', '?')}s**.")
    md.append("")
    md.append("## Step 1 — Fast binary cache")
    md.append("")
    md.append(f"- Cache: `{CACHE}` ({report.get('cache_mb', '?')} MB)")
    md.append("- Contents: 19 LB rank-logit vectors + scores, δ1_v4 / δ3_v4_lin / δ3_v4_bnd OOF+test, u_zjoint, u_glm_graph, z_ref_o, z_v8_old_oof, z_w12_oof, y, folds")
    md.append("")
    md.append("## Step 2 — Gram-Schmidt orthonormal basis (vs St9.5 / S1)")
    md.append("")
    md.append(f"- max|⟨qi,qj⟩| off-diag = `{basis['gram_off_max']:.2e}` (target 0)")
    md.append("")
    md.append("| sub | LB | c_q1 | c_q2 | c_bad_tree | c_bad_S3 | c_bad_ldr | c_bad_soft | c_S5_win |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ["Today_S1", "Today_S3", "Today_S4", "Today_S5"]:
        row = basis["table"][key]
        c = row["vs_S1"]
        md.append(
            f"| {key} | {row['lb']:.5f} | {c['q1_ladder']:+.4f} | {c['q2_e1']:+.4f} | "
            f"{c['q_bad_tree']:+.4f} | {c['q_bad_S3']:+.4f} | {c['q_bad_ldr']:+.4f} | "
            f"{c['q_bad_soft']:+.4f} | {c['q_S5_win']:+.4f} |"
        )
    md.append("")
    p = basis["poison"]
    md.append(
        f"**Poison check:** Today_S4 `c_bad_S3={p['Today_S4_c_bad_S3_vs_S1']:+.4f}`, "
        f"Today_S5 `c_bad_S3={p['Today_S5_c_bad_S3_vs_S1']:+.4f}`, "
        f"Today_S5 `c_S5_win={p['Today_S5_c_S5_win_vs_S1']:+.4f}`."
    )
    md.append("")
    md.append("## Step 3 — Joint 5/5 OOF synergy on `z_ref_o`")
    md.append("")
    s3 = report.get("step3", {})
    md.append(f"- `z_ref_o` AUC={s3.get('base_auc')}")
    md.append(f"- mean weights: `{s3.get('mean_w')}`")
    md.append(f"- **δ_v4_combo** Δe5=`{s3.get('delta_e5')}` pos={s3.get('pos')}/5 AUC={s3.get('auc')}")
    md.append(f"- hygiene: `{s3.get('hygiene')}`")
    md.append(f"- combo disp/KS/maxcos: `{s3.get('combo_disp_ks_cos')}`")
    md.append("")
    md.append("## Step 4 — Staged candidates (`NO_SUBMIT`)")
    md.append("")
    md.append("| slot | 1−ρ(S5) | cos(q1) | cos(bad_S3) | cos(bad_tree) | max\\|cos B_bad\\| | KS | file |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for s in report.get("step4", {}).get("staged", []):
        md.append(
            f"| {s['tag']} | {s['got_1mrho']*1e4:.3f}e-4 | {s['cos_q1']:+.3f} | "
            f"{s['cos_bad_S3']:+.4f} | {s['cos_bad_tree']:+.4f} | {s['max_cos_B_bad']:.2e} | "
            f"{s['ks']:.6f} | `{s['file']}` |"
        )
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    (LOG / "GM_V4_1_ORTHONORMAL_LOCK.md").write_text("\n".join(md))
    (LOG / "GM_V4_1_ORTHONORMAL_LOCK.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"Wrote {LOG / 'GM_V4_1_ORTHONORMAL_LOCK.md'}")


def main():
    assert NO_SUBMIT, "STRICT NO_SUBMIT"
    log("GM_V4_1_ORTHONORMAL_LOCK START")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    C = build_or_load_cache(y, folds, test_ids)
    # prefer folds from cache for exact reproducibility
    if "fold_val_off" in C:
        folds = folds_from_cache(C)
        y = np.asarray(C["y"], dtype=np.int32)
        test_ids = np.asarray(C["test_ids"])
    Z = Z_from_cache(C)

    report = {"NO_SUBMIT": True, "cache_mb": round(CACHE.stat().st_size / 1e6, 2)}
    basis = build_orth_basis(Z)
    report["step2"] = {
        "gram_off_max": basis["gram_off_max"],
        "table": basis["table"],
        "poison": basis["poison"],
    }
    np.save(ART / "Q_orth_basis.npy", basis["Q"])
    np.save(ART / "B_bad_orth.npy", basis["B_bad"])

    d_combo_o, d_combo_t, clean, mean_w, syn = joint_oof_synergy(C, folds, basis["B_bad"], report)
    stage_candidates(y, test_ids, Z, basis, d_combo_t, clean, report)

    report["elapsed_s"] = round(time.time() - T0, 1)
    write_report(report, basis)
    log(f"GM_V4_1_ORTHONORMAL_LOCK_COMPLETE elapsed={report['elapsed_s']}s")

    # Print the coordinate table prominently for the user
    print("\n===== ORTHONORMAL COORDINATE TABLE (vs Today_S1) =====", flush=True)
    print(f"{'sub':12s} {'LB':>8s} {'q1':>9s} {'q2':>9s} {'bad_tree':>10s} {'bad_S3':>9s} {'bad_ldr':>9s} {'bad_soft':>9s} {'S5_win':>9s}", flush=True)
    for key in ["Today_S1", "Today_S3", "Today_S4", "Today_S5"]:
        c = basis["table"][key]["vs_S1"]
        print(
            f"{key:12s} {basis['table'][key]['lb']:8.5f} "
            f"{c['q1_ladder']:+9.4f} {c['q2_e1']:+9.4f} {c['q_bad_tree']:+10.4f} "
            f"{c['q_bad_S3']:+9.4f} {c['q_bad_ldr']:+9.4f} {c['q_bad_soft']:+9.4f} {c['q_S5_win']:+9.4f}",
            flush=True,
        )
    print(
        f"\nPoison: S4 c_bad_S3={basis['poison']['Today_S4_c_bad_S3_vs_S1']:+.4f}  "
        f"S5 c_bad_S3={basis['poison']['Today_S5_c_bad_S3_vs_S1']:+.4f}  "
        f"S5 c_S5_win={basis['poison']['Today_S5_c_S5_win_vs_S1']:+.4f}",
        flush=True,
    )
    print(f"Joint δ_v4_combo: pos={syn['pos']}/5 Δe5={[round(x,2) for x in syn['delta_e5']]}", flush=True)


if __name__ == "__main__":
    main()
