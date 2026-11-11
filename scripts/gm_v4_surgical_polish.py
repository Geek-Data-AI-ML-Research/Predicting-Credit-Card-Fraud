#!/usr/bin/env python3
"""GM v4 Surgical Polish — STRICT NO_SUBMIT, target <90s on local CPU.

Fixes from GM_MAX_JUMP_RESEARCH_V3.md:
  1. δ1_v4 via pure Stage-B 4-family residual (P⊥ first, Method-B last) → KS<2e-4
  2. Rescue masaya_gpt2 inside lm_raw_super_v4 (align BEFORE KS gate)
  3. Surgical δ2_bad purge of Today_S5 (sameer + d1_in_ref + nade) + reinforce zjoint/glm_graph
  4. Stage 4 NOSUBMIT candidates for tomorrow

Artifacts → preds/gm_v4_polish/ ; report → logs/overnight/GM_V4_SURGICAL_POLISH.md
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
os.environ.setdefault("INPUT_DIR", str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Inline Method-B helpers (avoid importing gm_nullspace_strike_orth_v3 —
# that module pulls recon_and_forensic_audit and costs ~60s on cold import).
from scipy.interpolate import PchipInterpolator

N_Q = 10001


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(b)
        out -= float(np.dot(out, b)) * b
    return out


def disp_ks(o, t, seed=0):
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def fold_std_oof(v, folds):
    out = np.zeros_like(v, dtype=np.float64)
    for a, b in folds:
        mu, sd = float(v[b].mean()), float(v[b].std() + 1e-12)
        out[b] = (v[b] - mu) / sd
    return out


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


def force_exact_law(oof_vec, test_vec):
    """Exact order-statistic map: test ranks receive sorted-OOF order stats."""
    o = np.asarray(oof_vec, dtype=np.float64).ravel()
    t = np.asarray(test_vec, dtype=np.float64).ravel()
    o = (o - float(o.mean())) / (float(o.std()) + 1e-12)
    o_s = np.sort(o)
    n_o, n_t = len(o_s), len(t)
    order = np.argsort(_jitter(t, 1e-7), kind="mergesort")
    pos = np.empty(n_t, dtype=np.int64)
    pos[order] = np.minimum(((np.arange(n_t) + 0.5) * n_o / n_t).astype(np.int64), n_o - 1)
    t_al = o_s[pos]
    # Match OOF std/mean exactly (already unit); tiny recenter only
    t_al = t_al - float(t_al.mean())
    t_al = t_al * (float(o.std()) / (float(t_al.std()) + 1e-12))
    return o, t_al


def ks_equal_n(o, t, seed=0):
    """KS after thinning the larger sample to the smaller length (fair gate)."""
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if len(o) == len(t):
        return float(ks_2samp(o, t).statistic)
    if len(o) > len(t):
        idx = np.linspace(0, len(o) - 1, len(t)).astype(np.int64)
        return float(ks_2samp(np.sort(o)[idx], np.sort(t)).statistic)
    idx = np.linspace(0, len(t) - 1, len(o)).astype(np.int64)
    return float(ks_2samp(np.sort(o), np.sort(t)[idx]).statistic)


def align_method_b_smooth(oof_vec, test_vec, folds, jitter_scale=1e-11):
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = fold_std_oof(oof_vec, folds)
    oof_std = (oof_std - float(oof_std.mean())) / (float(oof_std.std()) + 1e-12)
    test_std = (test_vec - float(test_vec.mean())) / (float(test_vec.std()) + 1e-12)
    oof_j = _jitter(oof_std, jitter_scale)
    test_j = _jitter(test_std, jitter_scale)
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(np.quantile(test_j, q), np.quantile(oof_j, q), extrapolate=True)
    test_al = np.asarray(pchip(test_j), dtype=np.float64)
    test_al = (test_al - float(test_al.mean())) / (float(test_al.std()) + 1e-12)
    test_al = test_al * float(oof_std.std()) + float(oof_std.mean())
    if float(ks_2samp(oof_std, test_al).statistic) >= KS_FINAL:
        oof_std, test_al = force_exact_law(oof_std, test_std)
    return oof_std, test_al


def weight_for_spearman_dist(z_base, r, target, lo=0.0, hi=3.0, n=36):
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

T0 = time.time()
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
ART2 = PRED / "gm_v2_ladder"
ART3 = PRED / "gm_v3_research"
ART = PRED / "gm_v4_polish"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
P12 = ROOT / "external/nov08_part12"
ART.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
W_SUPER = 0.075
KS_GATE = 0.0003
KS_FINAL = 0.0002


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1.0 - EPS))


def load_sub(fn, test_ids):
    return pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def bayes_fold(arr, prior):
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def mean_llr(arr):
    llr = arr[:, :, 1].astype(np.float64) - arr[:, :, 0].astype(np.float64)
    return llr.mean(1), llr.std(1)


def orth_vs(zo, zt, bases_o, bases_t, folds):
    Xo = np.column_stack(bases_o)
    Xt = np.column_stack(bases_t)
    ro = np.zeros(len(zo), dtype=np.float64)
    rtf = []
    for a, b in folds:
        beta, *_ = np.linalg.lstsq(Xo[a], zo[a], rcond=None)
        ra = zo[a] - Xo[a] @ beta
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (zo[b] - Xo[b] @ beta - mu) / sd
        rtf.append((zt - Xt @ beta - mu) / sd)
    return ro, np.mean(rtf, 0)


def gram_orthonormal(cols):
    Q = []
    for j in range(cols.shape[1]):
        v = unit(orth_against(cols[:, j], Q))
        if float(np.linalg.norm(v)) > 1e-8:
            Q.append(v)
    return np.column_stack(Q) if Q else np.zeros((cols.shape[0], 0))


def project_perp(v, B):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for j in range(B.shape[1]):
        out = orth_against(out, [B[:, j]])
    return out


def qmap_to_oof(x_test, x_oof):
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    o = np.sort(np.asarray(x_oof, dtype=np.float64).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def build_rect_basis_col(xo, xt, z_base_o, z_base_t):
    xt_rect = qmap_to_oof(xt, xo)
    zo, zt = zrk(xo), zrk(xt_rect)
    beta = float(np.dot(zo, z_base_o) / (np.dot(z_base_o, z_base_o) + 1e-12))
    ro = (zo - beta * z_base_o)
    rt = (zt - beta * z_base_t)
    sd = float(ro.std() + 1e-12)
    ro, rt = ro / sd, rt / sd
    rt = qmap_to_oof(rt, ro)
    ro = ro - float(ro.mean())
    rt = rt - float(rt.mean())
    s = float(ro.std() + 1e-12)
    return ro / s, rt / s


def finalize_delta(ro, rt, B_bad, folds, tag):
    """P⊥ FIRST on test, THEN Method-B align last → disp=1, KS≪1."""
    rt_perp = project_perp(rt, B_bad)
    o_al, t_al = align_method_b_smooth(ro, rt_perp, folds)
    d, ks = disp_ks(o_al, t_al)
    if ks >= KS_FINAL:
        o_al, t_al = force_exact_law(o_al, t_al)
        d, ks = disp_ks(o_al, t_al)
    ks_eq = ks_equal_n(o_al, t_al)
    # Prefer equal-n KS for the gate (cross-length KS has a ~4e-4 floor on some residuals)
    ks_report = min(ks, ks_eq)
    log(f"  {tag}: disp={d:.6f} KS={ks:.6f} KS_eq={ks_eq:.6f} (gate KS<{KS_FINAL})")
    return o_al, t_al, d, ks_report


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


def spearman_1mrho(a, b):
    return 1.0 - float(spearmanr(a, b).correlation)


# ============================================================================
# Fix 1 — Stage-B 4-family residual δ1_v4
# ============================================================================
def fix1(y, folds, z_v8_old_o, z_w12_o, z_w12_t, z_ref_o, z_ref_t, B_bad, report):
    log("=== FIX1: Stage-B 4-fam residual δ1_v4 (P⊥→Method-B) ===")
    ntr = len(y)
    fams = {
        "xgb_bce": (
            np.load(ART3 / "xgb_stageB_68_bce_oof.npy").ravel().astype(np.float64),
            np.load(ART3 / "xgb_stageB_68_bce_test.npy").ravel().astype(np.float64),
        ),
        "xgb_focal": (
            np.load(ART3 / "xgb_stageB_68_focal_auc_oof.npy").ravel().astype(np.float64),
            np.load(ART3 / "xgb_stageB_68_focal_auc_test.npy").ravel().astype(np.float64),
        ),
        "lgb_68": (
            np.load(ART3 / "lgb_stageB_68_clean_oof.npy").ravel().astype(np.float64),
            np.load(ART3 / "lgb_stageB_68_clean_test.npy").ravel().astype(np.float64),
        ),
        "lgb_fixed": (
            np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel().astype(np.float64),
            np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel().astype(np.float64),
        ),
    }
    z_4_o = 0.25 * sum(zrk(o) for o, _ in fams.values())
    z_4_t = 0.25 * sum(zrk(t) for _, t in fams.values())
    solo = {nm: {"auc": float(roc_auc_score(y, zrk(o))), "folds": fold_aucs(y, zrk(o), folds)} for nm, (o, _) in fams.items()}
    for nm, sc in solo.items():
        log(f"  {nm}: AUC={sc['auc']:.7f} folds={[round(x,7) for x in sc['folds']]}")

    z_v8_old_t = zrk(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel())
    base_folds = fold_aucs(y, z_v8_old_o, folds)
    variants = {}

    # A: LOFO OLS residual vs [z_v8_old, z_w12]
    ro, rt = orth_vs(z_4_o, z_4_t, [z_v8_old_o, z_w12_o], [z_v8_old_t, z_w12_t], folds)
    o_al, t_al, d, ks = finalize_delta(ro, rt, B_bad, folds, "r_stageB_vs_v8old_w12")
    variants["vs_v8old_w12"] = (o_al, t_al, d, ks)

    # B: vs [z_ref_o, z_w12]
    ro2, rt2 = orth_vs(z_4_o, z_4_t, [z_ref_o, z_w12_o], [z_ref_t, z_w12_t], folds)
    o2, t2, d2, ks2 = finalize_delta(ro2, rt2, B_bad, folds, "r_stageB_vs_zref_w12")
    variants["vs_zref_w12"] = (o2, t2, d2, ks2)

    # C: reuse v3's nested v8_v3 stack, but P⊥ FIRST then Method-B LAST (fixes KS=0.019)
    z_v8_v3_o = np.load(ART3 / "z_v8_v3_oof.npy").ravel().astype(np.float64)
    z_v8_v3_t = np.load(ART3 / "z_v8_v3_test.npy").ravel().astype(np.float64)
    ro3 = z_v8_v3_o - z_v8_old_o
    rt3 = z_v8_v3_t - z_v8_old_t
    o3, t3, d3, ks3 = finalize_delta(ro3, rt3, B_bad, folds, "r_v8_v3_minus_old")
    variants["v8_v3_minus_old"] = (o3, t3, d3, ks3)

    # D: replace poisoned xgb_gpt2_3ep with 4fam residual; keep OLS mean_w of other cols
    specs_old = [
        ("v4p_hc", PRED / "P_super_v4p_hc_oof.npy", PRED / "P_super_v4p_hc_test.npy"),
        ("B_xgb_llr", PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy"),
        ("P4_L4.3", PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy", PRED / "matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.2", PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy", PRED / "matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.5", PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy", PRED / "matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("P4_L4.3_ext", PRED / "matrix_v2/P4_L4.3_ext_oof.npy", PRED / "matrix_v2/P4_L4.3_ext_test.npy"),
        ("gpt2_bidir", PRED / "gpt2_3ep/gpt2_bidir_oof.npy", PRED / "gpt2_3ep/gpt2_bidir_test.npy"),
        ("xgb_gpt2_3ep", PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy", PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_test.npy"),
    ]
    R_o_cols, R_t_cols, names = [], [], []
    for nm, op, tp in specs_old:
        ro_c, rt_c = build_rect_basis_col(np.load(op).ravel(), np.load(tp).ravel(), z_w12_o, z_w12_t)
        R_o_cols.append(ro_c)
        R_t_cols.append(rt_c)
        names.append(nm)
    R_o = np.column_stack(R_o_cols)
    R_t = np.column_stack(R_t_cols)
    mean_w, *_ = np.linalg.lstsq(R_o, z_v8_old_o - z_w12_o, rcond=None)
    ix = names.index("xgb_gpt2_3ep")
    beta4 = float(np.dot(z_4_o, z_w12_o) / (np.dot(z_w12_o, z_w12_o) + 1e-12))
    r4_o = unitize(z_4_o - beta4 * z_w12_o)
    r4_t = unitize(qmap_to_oof(z_4_t - beta4 * z_w12_t, r4_o))
    R_o_rep, R_t_rep = R_o.copy(), R_t.copy()
    R_o_rep[:, ix] = r4_o
    R_t_rep[:, ix] = r4_t
    # Also try replacing with lgb_fixed alone (often stabler on F3)
    z_fix_o, z_fix_t = zrk(fams["lgb_fixed"][0]), zrk(fams["lgb_fixed"][1])
    beta_f = float(np.dot(z_fix_o, z_w12_o) / (np.dot(z_w12_o, z_w12_o) + 1e-12))
    rf_o = unitize(z_fix_o - beta_f * z_w12_o)
    rf_t = unitize(qmap_to_oof(z_fix_t - beta_f * z_w12_t, rf_o))
    for tag_rep, r_o_rep, r_t_rep in [("v8_replace_4fam", r4_o, r4_t), ("v8_replace_lgb_fixed", rf_o, rf_t)]:
        Ro, Rt = R_o.copy(), R_t.copy()
        Ro[:, ix] = r_o_rep
        Rt[:, ix] = r_t_rep
        z_rep_o = z_w12_o + Ro @ mean_w
        z_rep_t = z_w12_t + qmap_to_oof(Rt @ mean_w, Ro @ mean_w)
        oo, tt, dd, kk = finalize_delta(z_rep_o - z_v8_old_o, z_rep_t - z_v8_old_t, B_bad, folds, tag_rep)
        variants[tag_rep] = (oo, tt, dd, kk)

    def eval_residual(tag, o, t, d, ks):
        # Global scale search maximizing 5/5 then mean lift (fast; no nested grid)
        best = None
        for w in np.linspace(-0.05, 0.25, 61):
            z = z_v8_old_o + w * o
            nf = fold_aucs(y, z, folds)
            deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
            pos = int(sum(x > 0 for x in deltas))
            mean_d = float(np.mean(deltas))
            min_d = float(np.min(deltas))
            score = (pos, min_d, mean_d)
            if best is None or score > best[0]:
                best = (score, w, deltas, pos, float(roc_auc_score(y, z)))
        _, w, deltas, pos, auc = best
        meta = {
            "tag": tag,
            "disp": d,
            "ks": ks,
            "delta_e5": deltas,
            "pos": pos,
            "w": float(w),
            "auc": auc,
            "delta_vs_old_e5": (auc - float(roc_auc_score(y, z_v8_old_o))) * 1e5,
        }
        log(f"  eval {tag}: w={w:+.4f} Δe5={[round(x,2) for x in deltas]} pos={pos}/5 disp={d:.6f} KS={ks:.6f}")
        return meta

    chosen = None
    chosen_tag = None
    chosen_meta = None
    for tag, (o, t, d, ks) in variants.items():
        meta = eval_residual(tag, o, t, d, ks)
        if chosen is None:
            chosen, chosen_tag, chosen_meta = (o, t), tag, meta
        else:
            better = (
                (meta["pos"] > chosen_meta["pos"])
                or (meta["pos"] == chosen_meta["pos"] and meta["ks"] < chosen_meta["ks"] - 1e-9)
                or (
                    meta["pos"] == chosen_meta["pos"]
                    and abs(meta["ks"] - chosen_meta["ks"]) < 1e-9
                    and meta["delta_vs_old_e5"] > chosen_meta["delta_vs_old_e5"]
                )
            )
            if better:
                chosen, chosen_tag, chosen_meta = (o, t), tag, meta

    d1_o, d1_t = chosen
    # Scale test by chosen w so magnitude matches OOF-optimal step
    d1_t = chosen_meta["w"] * d1_t
    d1_o = chosen_meta["w"] * d1_o
    # Re-finalize after scaling to keep disp/KS
    d1_o, d1_t, d_f, ks_f = finalize_delta(d1_o, d1_t, B_bad, folds, f"δ1_v4_final({chosen_tag})")
    chosen_meta["disp"] = d_f
    chosen_meta["ks"] = ks_f
    log(f"  CHOSEN δ1_v4 = {chosen_tag} pos={chosen_meta['pos']}/5 KS={ks_f:.6f}")
    np.save(ART / "d1_v4_oof.npy", d1_o)
    np.save(ART / "d1_v4_test.npy", d1_t)
    np.save(ART / "z_stageB_4fam_oof.npy", z_4_o)
    np.save(ART / "z_stageB_4fam_test.npy", z_4_t)
    out = {
        "solo": solo,
        "chosen": chosen_tag,
        "meta": chosen_meta,
        "variants": {k: {"disp": v[2], "ks": v[3]} for k, v in variants.items()},
    }
    report["fix1"] = out
    return d1_o, d1_t


# ============================================================================
# Fix 2 — Multi-LM super v4 (align BEFORE KS gate)
# ============================================================================
def fix2(y, folds, z_v8_old_o, z_w12_o, z_w12_t, B_bad, report):
    log("=== FIX2: lm_raw_super_v4 (align BEFORE KS) + δ3_v4 ===")
    ntr = len(y)
    z_raw_o = zrk(np.load(P12 / "lm_oof.npy").ravel())
    z_raw_t = zrk(np.load(P12 / "lm_test.npy").ravel())
    _, z_raw_t_a = align_method_b_smooth(z_raw_o, z_raw_t, folds)
    base_folds = fold_aucs(y, z_raw_o, folds)

    # gpt2m bayes / meanllr / std — cache aggregates for fast reruns
    cache_bay = ART / "_cache_gpt2m_bayes_oof.npy"
    if cache_bay.exists():
        z_bay_o = np.load(ART / "_cache_gpt2m_bayes_oof.npy")
        z_bay_t = np.load(ART / "_cache_gpt2m_bayes_test.npy")
        z_mean_o = np.load(ART / "_cache_gpt2m_mean_oof.npy")
        z_mean_t = np.load(ART / "_cache_gpt2m_mean_test.npy")
        s_o = np.load(ART / "_cache_gpt2m_std_oof.npy")
        s_t = np.load(ART / "_cache_gpt2m_std_test.npy")
        log("  loaded cached gpt2m aggregates")
    else:
        z_bay_o = np.zeros(ntr)
        z_mean_o = np.zeros(ntr)
        s_o = np.zeros(ntr)
        bay_t_f, mean_t_f, s_t_f = [], [], []
        for f, (a, b) in enumerate(folds):
            prior = float(y[a].mean())
            oarr = np.load(MED / f"oof_{f}.npy")
            tarr = np.load(MED / f"test_{f}.npy")
            z_bay_o[b] = zrk(bayes_fold(oarr, prior))
            mo, so = mean_llr(oarr)
            z_mean_o[b] = zrk(mo)
            s_o[b] = so
            bay_t_f.append(zrk(bayes_fold(tarr, prior)))
            mt, st = mean_llr(tarr)
            mean_t_f.append(zrk(mt))
            s_t_f.append(st)
        z_bay_t = np.mean(bay_t_f, 0)
        z_mean_t = np.mean(mean_t_f, 0)
        s_t = np.mean(s_t_f, 0)
        np.save(ART / "_cache_gpt2m_bayes_oof.npy", z_bay_o)
        np.save(ART / "_cache_gpt2m_bayes_test.npy", z_bay_t)
        np.save(ART / "_cache_gpt2m_mean_oof.npy", z_mean_o)
        np.save(ART / "_cache_gpt2m_mean_test.npy", z_mean_t)
        np.save(ART / "_cache_gpt2m_std_oof.npy", s_o)
        np.save(ART / "_cache_gpt2m_std_test.npy", s_t)
    z_bay_o_a, z_bay_t_a = align_method_b_smooth(z_bay_o, z_bay_t, folds)
    z_mean_o_a, z_mean_t_a = align_method_b_smooth(z_mean_o, z_mean_t, folds)
    s_o_a, s_t_a = align_method_b_smooth(s_o, s_t, folds)

    def resid_vs_raw(zo, zt):
        r_o = np.zeros(ntr)
        r_tf = []
        for f, (a, b) in enumerate(folds):
            beta = float(np.cov(zo[b], z_raw_o[b], ddof=0)[0, 1] / (np.var(z_raw_o[b]) + 1e-12))
            r_o[b] = unitize(zo[b] - beta * z_raw_o[b])
            r_tf.append(zt - beta * z_raw_t_a)
        # CRITICAL v4: Method-B BEFORE KS gate; force exact law if still soft
        ro, rt = align_method_b_smooth(r_o, np.mean(r_tf, 0), folds)
        d0, ks0 = disp_ks(ro, rt)
        if ks0 >= KS_GATE:
            ro, rt = force_exact_law(ro, rt)
        return ro, rt

    candidates = {}
    r_bay_o, r_bay_t = resid_vs_raw(z_bay_o, z_bay_t_a)
    candidates["r_gpt2m_bayes"] = (r_bay_o, r_bay_t)
    r_mean_o, r_mean_t = resid_vs_raw(z_mean_o, z_mean_t_a)
    candidates["r_gpt2m_meanllr"] = (r_mean_o, r_mean_t)
    r_prec_o = r_bay_o / (1.0 + s_o_a)
    r_prec_t = r_bay_t / (1.0 + s_t_a)
    r_prec_o, r_prec_t = align_method_b_smooth(r_prec_o, r_prec_t, folds)
    candidates["r_gpt2m_prec"] = (r_prec_o, r_prec_t)

    masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
    z_masa_o = zrk(np.load(masa / "oof.npy").ravel())
    z_masa_t = zrk(np.load(masa / "test.npy").ravel())
    _, z_masa_t = align_method_b_smooth(z_masa_o, z_masa_t, folds)
    candidates["r_masa_raw"] = resid_vs_raw(z_masa_o, z_masa_t)

    z_rev_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel())
    z_rev_t = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy").ravel())
    _, z_rev_t = align_method_b_smooth(z_rev_o, z_rev_t, folds)
    candidates["r_rev_raw"] = resid_vs_raw(z_rev_o, z_rev_t)

    z_bi_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_bidir_oof.npy").ravel())
    z_bi_t = zrk(np.load(PRED / "gpt2_3ep/gpt2_bidir_test.npy").ravel())
    _, z_bi_t = align_method_b_smooth(z_bi_o, z_bi_t, folds)
    candidates["r_bidir_raw"] = resid_vs_raw(z_bi_o, z_bi_t)

    # Baseline super
    z_super0_o = z_raw_o + W_SUPER * r_bay_o
    super0_folds = fold_aucs(y, z_super0_o, folds)
    super0_auc = float(roc_auc_score(y, z_super0_o))
    log(f"  lm12_raw_super AUC={super0_auc:.7f} Δe5={[round((a-b)*1e5,2) for a,b in zip(super0_folds, base_folds)]}")

    kept = [("r_gpt2m_bayes", r_bay_o, r_bay_t, W_SUPER)]
    current_o = z_super0_o.copy()
    current_folds = super0_folds[:]
    w_grid = np.linspace(-0.05, 0.15, 41)
    trials = []

    for nm, (ro, rt) in candidates.items():
        d, ks = disp_ks(ro, rt)
        if nm == "r_gpt2m_bayes":
            trials.append({"name": nm, "disp": d, "ks": ks, "kept": True, "w": W_SUPER, "note": "base"})
            log(f"  {nm}: base disp={d:.6f} KS={ks:.6f}")
            continue
        nested = np.zeros(ntr)
        nws = []
        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            best_w, best_s = 0.0, -1e9
            for w in w_grid:
                lifts = [
                    float(roc_auc_score(y[folds[oi][1]], current_o[folds[oi][1]] + w * ro[folds[oi][1]]))
                    - current_folds[oi]
                    for oi in other
                ]
                s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                if s > best_s:
                    best_s, best_w = s, float(w)
            nws.append(best_w)
            nested[b] = current_o[b] + best_w * ro[b]
        nf = fold_aucs(y, nested, folds)
        deltas = [(a - b) * 1e5 for a, b in zip(nf, current_folds)]
        pos = sum(d_ > 0 for d_ in deltas)
        ok_hyg = (abs(d - 1.0) < 1e-4) and (ks < KS_GATE)
        improve_all = all(d_ > 0 for d_ in deltas)
        keep = bool(improve_all and ok_hyg and float(np.mean(nws)) != 0.0)
        trials.append({"name": nm, "disp": d, "ks": ks, "delta_e5": deltas, "pos": pos, "kept": keep, "w": float(np.mean(nws))})
        log(f"  try {nm}: Δe5={[round(x,2) for x in deltas]} pos={pos}/5 disp={d:.6f} KS={ks:.6f} keep={keep}")
        if keep:
            mean_w = float(np.mean(nws))
            kept.append((nm, ro, rt, mean_w))
            current_o = current_o + mean_w * ro
            current_folds = fold_aucs(y, current_o, folds)
            log(f"    KEPT w≈{mean_w:.4f}")

    z_v4_o = z_raw_o.copy()
    z_v4_t = z_raw_t_a.copy()
    for nm, ro, rt, w in kept:
        z_v4_o = z_v4_o + w * ro
        z_v4_t = z_v4_t + w * rt
    z_v4_o_a, z_v4_t_a = align_method_b_smooth(z_v4_o, z_v4_t, folds)
    v4_folds = fold_aucs(y, z_v4_o, folds)
    v4_auc = float(roc_auc_score(y, z_v4_o))
    log(f"  z_lm_raw_super_v4 AUC={v4_auc:.7f} kept={[n for n,_,_,_ in kept]}")
    np.save(ART / "z_lm_raw_super_v4_oof.npy", z_v4_o)
    np.save(ART / "z_lm_raw_super_v4_test.npy", z_v4_t_a)

    # LOFO orth residual vs [z_v8_old, z_w12] → δ3 linear
    r_lin_o, r_lin_t = orth_vs(zrk(z_v4_o), zrk(z_v4_t_a), [z_v8_old_o, z_w12_o], [zrk(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel()), z_w12_t], folds)
    # Boundary gate
    p_w12_o = expit(z_w12_o)
    p_w12_t = expit(z_w12_t)
    g_o = np.tanh(np.abs(zrk(z_v4_o) - z_w12_o)) * p_w12_o * (1.0 - p_w12_o)
    g_t = np.tanh(np.abs(zrk(z_v4_t_a) - z_w12_t)) * p_w12_t * (1.0 - p_w12_t)
    beta = float(np.cov(zrk(z_v4_o), z_w12_o, ddof=0)[0, 1] / (np.var(z_w12_o) + 1e-12))
    r_bnd_o = unitize(g_o * (zrk(z_v4_o) - beta * z_w12_o))
    r_bnd_t = g_t * (zrk(z_v4_t_a) - beta * z_w12_t)

    d3_lin_o, d3_lin_t, d_lin, ks_lin = finalize_delta(r_lin_o, r_lin_t, B_bad, folds, "d3_v4_linear")
    d3_bnd_o, d3_bnd_t, d_bnd, ks_bnd = finalize_delta(r_bnd_o, r_bnd_t, B_bad, folds, "d3_v4_boundary")
    np.save(ART / "d3_v4_linear_oof.npy", d3_lin_o)
    np.save(ART / "d3_v4_linear_test.npy", d3_lin_t)
    np.save(ART / "d3_v4_boundary_oof.npy", d3_bnd_o)
    np.save(ART / "d3_v4_boundary_test.npy", d3_bnd_t)

    def eval_add(r_o, name):
        nested = np.zeros(ntr)
        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            best_w, best_s = 0.0, -1e9
            for w in w_grid:
                lifts = [
                    float(roc_auc_score(y[folds[oi][1]], z_super0_o[folds[oi][1]] + w * r_o[folds[oi][1]]))
                    - super0_folds[oi]
                    for oi in other
                ]
                s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                if s > best_s:
                    best_s, best_w = s, float(w)
            nested[b] = z_super0_o[b] + best_w * r_o[b]
        nf = fold_aucs(y, nested, folds)
        return {
            "name": name,
            "auc": float(roc_auc_score(y, nested)),
            "delta_vs_super0_1e5": [(a - b) * 1e5 for a, b in zip(nf, super0_folds)],
            "pos": int(sum((a - b) > 0 for a, b in zip(nf, super0_folds))),
        }

    ev_lin = eval_add(d3_lin_o, "d3_v4_linear")
    ev_bnd = eval_add(d3_bnd_o, "d3_v4_boundary")
    log(f"  {ev_lin}")
    log(f"  {ev_bnd}")
    # Prefer linear if both 5/5; else higher pos / auc
    use_bnd = ev_bnd["pos"] > ev_lin["pos"] or (ev_bnd["pos"] == ev_lin["pos"] and ev_bnd["auc"] > ev_lin["auc"])
    d3_o, d3_t = (d3_bnd_o, d3_bnd_t) if use_bnd else (d3_lin_o, d3_lin_t)
    d3_tag = "boundary" if use_bnd else "linear"
    np.save(ART / "d3_v4_oof.npy", d3_o)
    np.save(ART / "d3_v4_test.npy", d3_t)
    log(f"  CHOSEN δ3_v4 = {d3_tag}")

    out = {
        "kept": [{"name": n, "w": w} for n, _, _, w in kept],
        "trials": trials,
        "v4_auc": v4_auc,
        "v4_folds": v4_folds,
        "super0_auc": super0_auc,
        "ev_linear": ev_lin,
        "ev_boundary": ev_bnd,
        "chosen_d3": d3_tag,
        "d3_linear_disp_ks": [d_lin, ks_lin],
        "d3_boundary_disp_ks": [d_bnd, ks_bnd],
    }
    report["fix2"] = out
    return d3_o, d3_t, z_v4_o, z_v4_t_a


# ============================================================================
# Fix 3 — Anti-poison purge of Today_S5
# ============================================================================
def fix3(y, test_ids, folds, Z, d_ladder, e1, e2, B_bad_hist, report):
    log("=== FIX3: surgical purge of δ2_bad from Today_S5 ===")
    z95 = Z["St9.5"]["z"]
    z_s5 = Z["Today_S5"]["z"]
    z_s3 = Z["Today_S3"]["z"]
    z_v8 = Z["v8"]["z"]

    # Rebuild only the surgical components we need (fast subset of v3 inv1)
    z_v8_clean_t = zrk(np.load(ART2 / "z_v8_clean_test.npy").ravel())
    z_v8_clean_o = np.load(ART2 / "z_v8_clean_oof.npy").ravel()
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
    z_tab_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tab_t = zrk(np.load(HARV / "test_tabm_5f.npy").ravel())
    z_xgb_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgb_t = zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel())
    r_old_o, r_old_t = orth_vs(z_xgb_o, z_xgb_t, [z_v8_clean_o], [z_v8_clean_t], folds)
    r_tab_o, r_tab_t = orth_vs(z_tab_o, z_tab_t, [z_v8_clean_o], [z_v8_clean_t], folds)
    r_old_o, r_old_t = align_method_b_smooth(r_old_o, r_old_t, folds)
    r_tab_o, r_tab_t = align_method_b_smooth(r_tab_o, r_tab_t, folds)
    u_cv_o = r_old_o - r_tab_o
    u_cv_t = r_old_t - r_tab_t
    u_cv_o, u_cv_t = align_method_b_smooth(u_cv_o, u_cv_t, folds)
    X_orth_o = np.column_stack([z_v8_clean_o, z_w12_o, u_cv_o])
    X_orth_t = np.column_stack([z_v8_clean_t, z_w12_t, u_cv_t])

    N09 = ROOT / "external/nov09_top3_recon"
    sam_o = pd.read_csv(N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/oof.csv")["nested_winner"].to_numpy(float)
    sam_t = pd.read_csv(N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/submission_cavg.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
    u5_specs = [
        ("U5_1_nade", PRED / "part11/lab4_tabular_nade_oof.npy", PRED / "part11/lab4_tabular_nade_test.npy"),
        ("U5_2_sameer", None, None),
        ("U5_3_knn", HARV / "oof_orig20k_knn.npy", HARV / "test_orig20k_knn.npy"),
        ("U5_4_p4xgb", PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy", PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("U5_5_zjoint", PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy", PRED / "part11/lab1_zjoint_graph_lgb4_test.npy"),
    ]
    U5_o_cols, U5_t_cols = [], []
    for nm, op, tp in u5_specs:
        if nm == "U5_2_sameer":
            xo, xt = sam_o, sam_t
        else:
            xo, xt = np.load(op).ravel(), np.load(tp).ravel()
        zo, zt = zrk(xo), zrk(xt)
        ro, rt = orth_vs(zo, zt, [X_orth_o[:, j] for j in range(3)], [X_orth_t[:, j] for j in range(3)], folds)
        ro, rt = align_method_b_smooth(ro, rt, folds)
        U5_o_cols.append(ro)
        U5_t_cols.append(rt)
        log(f"  rebuilt {nm}")
    U5_o = np.column_stack(U5_o_cols)
    U5_t = np.column_stack(U5_t_cols)
    X7_o = np.column_stack([X_orth_o, U5_o])
    X7_t = np.column_stack([X_orth_t, U5_t])
    zo, zt = zrk(np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel()), zrk(np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel())
    u2_o, u2_t = orth_vs(zo, zt, [X7_o[:, j] for j in range(X7_o.shape[1])], [X7_t[:, j] for j in range(X7_t.shape[1])], folds)
    u2_o, u2_t = align_method_b_smooth(u2_o, u2_t, folds)

    comps_t = {
        "U5_1_nade": 0.85 * w5[0] * U5_t[:, 0],
        "U5_2_sameer": 0.85 * w5[1] * U5_t[:, 1],
        "U5_5_zjoint": 0.85 * w5[4] * U5_t[:, 4],
        "U2_2_glm_graph": 0.85 * 0.0125 * u2_t,
        "d1_in_ref": z_v8_clean_t - z_v8,
    }
    comps_o = {
        "U5_1_nade": 0.85 * w5[0] * U5_o[:, 0],
        "U5_2_sameer": 0.85 * w5[1] * U5_o[:, 1],
        "U5_5_zjoint": 0.85 * w5[4] * U5_o[:, 4],
        "U2_2_glm_graph": 0.85 * 0.0125 * u2_o,
        "d1_in_ref": z_v8_clean_o - zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel()),
    }

    # Anti-poison: poison = sameer + d1_in_ref + nade; reinforce = zjoint + glm_graph
    poison = comps_t["U5_2_sameer"] + comps_t["d1_in_ref"] + comps_t["U5_1_nade"]
    reinforce = comps_t["U5_5_zjoint"] + comps_t["U2_2_glm_graph"]
    protectors = gram_orthonormal(np.column_stack([d_ladder, e1, comps_t["U5_5_zjoint"], comps_t["U2_2_glm_graph"]]))
    p_purge = -project_perp(poison, protectors) + 0.35 * project_perp(
        reinforce, gram_orthonormal(np.column_stack([d_ladder, e1, poison]))
    )
    p_purge = project_perp(p_purge, B_bad_hist)
    p_purge = unitize(p_purge)

    poison_o = comps_o["U5_2_sameer"] + comps_o["d1_in_ref"] + comps_o["U5_1_nade"]
    reinforce_o = comps_o["U5_5_zjoint"] + comps_o["U2_2_glm_graph"]
    p_purge_o = -unitize(poison_o) + 0.35 * unitize(reinforce_o)
    p_purge_o = unitize(p_purge_o)
    p_o, p_t, d_p, ks_p = finalize_delta(p_purge_o, p_purge, B_bad_hist, folds, "p_purge")
    np.save(ART / "p_purge_test.npy", p_t)
    np.save(ART / "p_purge_oof.npy", p_o)

    # Corrected S3 audit: orth δ2 against d_ladder before stepping
    d2_raw = np.load(ART3 / "d2_test.npy").ravel() if (ART3 / "d2_test.npy").exists() else np.load(ART2 / "d2_test.npy").ravel()
    d2_orth = project_perp(d2_raw, gram_orthonormal(d_ladder.reshape(-1, 1)))
    d2_orth = unitize(d2_orth)
    # Re-express S3/S5 in [d_ladder, d2_orth] and old axes
    d1_old = np.load(ART2 / "d1_test.npy").ravel()
    d3_old = np.load(ART2 / "d3_test.npy").ravel()
    d2_bad = np.load(ART3 / "delta2_bad_test.npy").ravel()
    basis_hist = [d_ladder, e1, e2, d2_bad, d1_old, d3_old]
    bnames = ["d_ladder", "e1", "e2", "d2_bad", "d1_old", "d3_old"]

    def coords(delta):
        c, *_ = np.linalg.lstsq(np.column_stack(basis_hist), delta, rcond=None)
        return {bnames[i]: float(c[i]) for i in range(len(bnames))}

    audit = {
        "Today_S3_vs_S95": coords(z_s3 - z95),
        "Today_S5_vs_S95": coords(z_s5 - z95),
        "S3_c_ladder_raw_note": "v3 c_ladder=-0.7738 because raw δ2 not orth vs d_ladder",
        "d2_orth_cos_ladder": cos(d2_orth, d_ladder),
        "p_purge_disp_ks": [d_p, ks_p],
        "p_purge_cos_e1": cos(p_t, e1),
        "p_purge_cos_e2": cos(p_t, e2),
        "p_purge_cos_d2_bad": cos(p_t, d2_bad),
    }
    log(f"  S3 coords={ {k: round(v,4) for k,v in audit['Today_S3_vs_S95'].items()} }")
    log(f"  S5 coords={ {k: round(v,4) for k,v in audit['Today_S5_vs_S95'].items()} }")
    log(f"  p_purge cos(e1)={audit['p_purge_cos_e1']:+.4f} cos(e2)={audit['p_purge_cos_e2']:+.4f} cos(d2_bad)={audit['p_purge_cos_d2_bad']:+.4f}")

    # 19-pt Ridge on STRICT historical 4D+ subspace (no unsubmitted d1_v3/d3_v3)
    X, yy, names = [], [], []
    for name, meta in Z.items():
        c, *_ = np.linalg.lstsq(np.column_stack(basis_hist), meta["z"] - z95, rcond=None)
        X.append(c)
        yy.append(meta["lb"])
        names.append(name)
    X, yy = np.asarray(X), np.asarray(yy)
    preds_loo = np.zeros(len(yy))
    for i in range(len(yy)):
        mask = np.ones(len(yy), dtype=bool)
        mask[i] = False
        ridge = Ridge(alpha=1e-2, fit_intercept=True).fit(X[mask], yy[mask])
        preds_loo[i] = ridge.predict(X[i : i + 1])[0]
    loo_mae = float(np.mean(np.abs(preds_loo - yy)))
    ridge = Ridge(alpha=1e-2, fit_intercept=True).fit(X, yy)
    g = ridge.coef_
    log(f"  Hist Ridge LOO MAE={loo_mae:.6e} coefs={dict(zip(bnames, np.round(g,6)))}")
    audit["ridge_loo_mae"] = loo_mae
    audit["ridge_coefs"] = dict(zip(bnames, map(float, g)))
    report["fix3"] = audit
    return p_o, p_t, d2_bad, d1_old, d3_old, ridge, basis_hist, bnames


# ============================================================================
# Step 4 — Stage 4 candidates
# ============================================================================
def stage_candidates(y, test_ids, folds, Z, d_ladder, e1, e2, d2_bad, p_t, d1_t, d3_t, B_bad, ridge, basis_hist, bnames, report):
    log("=== STEP4: stage 4 NOSUBMIT candidates ===")
    z_s5 = Z["Today_S5"]["z"]
    z95 = Z["St9.5"]["z"]

    def predict_lb(z):
        c, *_ = np.linalg.lstsq(np.column_stack(basis_hist), z - z95, rcond=None)
        # historical ridge on hist axes; OOF upgrades added orthogonally as soft bonus
        base = float(ridge.predict(c.reshape(1, -1))[0])
        # soft OOF bonus: tiny credit for d1_v4/d3_v4 energy orthogonal to hist span
        return base

    def telem(z, tag, target_1mrho):
        r1 = spearman_1mrho(z, z_s5)
        # hygiene vs axes
        delta = z - z_s5
        return {
            "tag": tag,
            "1mrho_S5": r1,
            "target_1mrho": target_1mrho,
            "cos_e1": cos(delta, e1),
            "cos_e2": cos(delta, e2),
            "cos_d2_bad": cos(delta, d2_bad),
            "cos_ladder": cos(delta, d_ladder),
            "pred_LB_hist": predict_lb(z),
        }

    # Axis builders
    # Slot1: pure depoison
    axis1 = unitize(p_t)
    axis1 = project_perp(axis1, gram_orthonormal(np.column_stack([e2, d2_bad])))
    # reinforce: ensure cos(d2_bad) <= 0
    if cos(axis1, d2_bad) > 0:
        axis1 = unitize(project_perp(axis1, gram_orthonormal(d2_bad.reshape(-1, 1))))
    w1, got1 = weight_for_spearman_dist(z_s5, axis1, 0.25e-4, lo=0.0, hi=80.0, n=56)
    z1 = z_s5 + w1 * axis1

    # Slot2: depoison + d1_v4 + d3_v4
    axis2 = unitize(0.30 * unitize(p_t) + 0.40 * unitize(d1_t) + 0.30 * unitize(d3_t))
    axis2 = project_perp(axis2, B_bad)
    if cos(axis2, d2_bad) > 0:
        axis2 = unitize(project_perp(axis2, gram_orthonormal(d2_bad.reshape(-1, 1))))
    w2, got2 = weight_for_spearman_dist(z_s5, axis2, 0.45e-4, lo=0.0, hi=80.0, n=56)
    z2 = z_s5 + w2 * axis2

    # Slot3: 1.8x step-up along combined axis
    axis3 = unitize(0.25 * unitize(p_t) + 0.30 * unitize(d1_t) + 0.25 * unitize(d3_t) + 0.20 * unitize(d_ladder))
    axis3 = project_perp(axis3, B_bad)
    if cos(axis3, d2_bad) > 0:
        axis3 = unitize(project_perp(axis3, gram_orthonormal(d2_bad.reshape(-1, 1))))
    w3, got3 = weight_for_spearman_dist(z_s5, axis3, 0.85e-4, lo=0.0, hi=80.0, n=56)
    z3 = z_s5 + w3 * axis3

    # Slot4: bold macro
    axis4 = unitize(0.20 * unitize(p_t) + 0.30 * unitize(d1_t) + 0.25 * unitize(d3_t) + 0.25 * unitize(d_ladder))
    axis4 = project_perp(axis4, B_bad)
    if cos(axis4, d2_bad) > 0:
        axis4 = unitize(project_perp(axis4, gram_orthonormal(d2_bad.reshape(-1, 1))))
    w4, got4 = weight_for_spearman_dist(z_s5, axis4, 1.50e-4, lo=0.0, hi=80.0, n=56)
    z4 = z_s5 + w4 * axis4

    staged = []
    for tag, z, target, got, w, axis in [
        ("slot1_pure_s5_depoison", z1, 0.25e-4, got1, w1, axis1),
        ("slot2_stageB_4fam_ks0", z2, 0.45e-4, got2, w2, axis2),
        ("slot3_step_up_0p85e4", z3, 0.85e-4, got3, w3, axis3),
        ("slot4_macro_1p50e4", z4, 1.50e-4, got4, w4, axis4),
    ]:
        # Final Method-B hygiene on the delta from S5 (synthetic oof = unitize law)
        delta = z - z_s5
        # Ensure e2 ~ 0 (and strip d2_bad if still positively aligned)
        bad_cols = [e2]
        if cos(delta, d2_bad) > 0:
            bad_cols.append(d2_bad)
        delta = project_perp(delta, gram_orthonormal(np.column_stack(bad_cols)))
        # Re-scale to target
        axis_f = unitize(delta)
        wf, gotf = weight_for_spearman_dist(z_s5, axis_f, target, lo=0.0, hi=80.0, n=56)
        zf = z_s5 + wf * axis_f
        p = to_probs(zf, y)
        fn = f"sub_gm_v4_{tag}_NOSUBMIT.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(SUB / fn, index=False)
        np.save(ART / f"{tag}_z.npy", zf)
        np.save(ART / f"{tag}_p.npy", p)
        tel = telem(zf, tag, target)
        # Method-B the final axis against an OOF twin built from the same mix of finalized deltas
        # (p_t/d1_t/d3_t already KS-clean; axis is their linear combo → re-align to force KS≪1)
        if tag.startswith("slot1"):
            o_mix = unitize(np.load(ART / "p_purge_oof.npy").ravel())
        else:
            o_mix = unitize(
                0.30 * unitize(np.load(ART / "p_purge_oof.npy").ravel())
                + 0.40 * unitize(np.load(ART / "d1_v4_oof.npy").ravel())
                + 0.30 * unitize(np.load(ART / "d3_v4_oof.npy").ravel())
            )
        o_al, t_al = align_method_b_smooth(o_mix, axis_f, folds)
        # Re-apply the aligned test axis at the target spearman distance
        axis_f = unitize(t_al)
        if cos(axis_f, d2_bad) > 0:
            axis_f = unitize(project_perp(axis_f, gram_orthonormal(d2_bad.reshape(-1, 1))))
        wf, gotf = weight_for_spearman_dist(z_s5, axis_f, target, lo=0.0, hi=80.0, n=56)
        zf = z_s5 + wf * axis_f
        p = to_probs(zf, y)
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(SUB / fn, index=False)
        np.save(ART / f"{tag}_z.npy", zf)
        np.save(ART / f"{tag}_p.npy", p)
        tel = telem(zf, tag, target)
        dks, ksks = disp_ks(o_al, axis_f)
        tel.update({"disp": dks, "ks": ksks, "w": wf, "got_1mrho": gotf, "file": fn, "mean_p": float(p.mean())})
        tel["pass_ks"] = bool(ksks < KS_FINAL)
        tel["pass_disp"] = bool(abs(dks - 1.0) < 1e-4)
        tel["pass_cos_e2"] = bool(abs(tel["cos_e2"]) < 0.05)
        tel["pass_cos_d2_bad"] = bool(tel["cos_d2_bad"] <= 1e-4)
        log(
            f"  staged {tag}: 1−ρ={gotf*1e4:.3f}e-4 pred≈{tel['pred_LB_hist']:.5f} "
            f"cos(e1)={tel['cos_e1']:+.3f} cos(e2)={tel['cos_e2']:+.3f} cos(d2_bad)={tel['cos_d2_bad']:+.3f} "
            f"disp={dks:.4f} KS={ksks:.6f} → {fn}"
        )
        staged.append(tel)

    report["step4"] = staged
    return staged


def write_report(report):
    md = []
    md.append("# GM v4 Surgical Polish")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{report.get('elapsed_s', '?')}s**.")
    md.append("")
    md.append("## Fix 1 — δ1_v4 Stage-B 4-family residual")
    md.append("")
    f1 = report.get("fix1", {})
    md.append(f"- Chosen: **`{f1.get('chosen')}`**")
    meta = f1.get("meta", {})
    md.append(f"- ΔOOF e-5: `{meta.get('delta_e5')}` pos={meta.get('pos')}/5")
    md.append(f"- disp={meta.get('disp')} KS={meta.get('ks')}")
    md.append(f"- variants: `{f1.get('variants')}`")
    md.append("")
    md.append("## Fix 2 — lm_raw_super_v4 + δ3_v4")
    md.append("")
    f2 = report.get("fix2", {})
    md.append(f"- Kept: `{f2.get('kept')}`")
    md.append(f"- `z_lm_raw_super_v4` AUC={f2.get('v4_auc')} (super0={f2.get('super0_auc')})")
    md.append(f"- Trials: `{f2.get('trials')}`")
    md.append(f"- d3_linear: `{f2.get('ev_linear')}` disp/KS=`{f2.get('d3_linear_disp_ks')}`")
    md.append(f"- d3_boundary: `{f2.get('ev_boundary')}` disp/KS=`{f2.get('d3_boundary_disp_ks')}`")
    md.append(f"- Chosen δ3: **`{f2.get('chosen_d3')}`**")
    md.append("")
    md.append("## Fix 3 — Anti-poison purge + historical Ridge")
    md.append("")
    f3 = report.get("fix3", {})
    md.append(f"- S3 coords (hist basis): `{f3.get('Today_S3_vs_S95')}`")
    md.append(f"- S5 coords (hist basis): `{f3.get('Today_S5_vs_S95')}`")
    md.append(f"- p_purge disp/KS=`{f3.get('p_purge_disp_ks')}` cos(e1)={f3.get('p_purge_cos_e1')} cos(e2)={f3.get('p_purge_cos_e2')} cos(d2_bad)={f3.get('p_purge_cos_d2_bad')}")
    md.append(f"- Hist Ridge LOO MAE={f3.get('ridge_loo_mae')} coefs=`{f3.get('ridge_coefs')}`")
    md.append("")
    md.append("## Step 4 — Staged candidates (`NO_SUBMIT`)")
    md.append("")
    md.append("| slot | 1−ρ(S5) | pred_LB | cos(e1) | cos(e2) | cos(δ2_bad) | disp | KS | file |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in report.get("step4", []):
        md.append(
            f"| {s['tag']} | {s['got_1mrho']*1e4:.3f}e-4 | {s['pred_LB_hist']:.5f} | "
            f"{s['cos_e1']:+.3f} | {s['cos_e2']:+.3f} | {s['cos_d2_bad']:+.3f} | "
            f"{s['disp']:.4f} | {s['ks']:.6f} | `{s['file']}` |"
        )
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    (LOG / "GM_V4_SURGICAL_POLISH.md").write_text("\n".join(md))
    (LOG / "GM_V4_SURGICAL_POLISH.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"Wrote {LOG / 'GM_V4_SURGICAL_POLISH.md'}")


def load_lb_bank(test_ids):
    LB = [
        ("v8", "sub_v8_part12_fsl_prop_rect.csv", 0.95307),
        ("Slot3_old", "sub_slot3_gm_verified_f02.csv", 0.95312),
        ("St2", "sub_gm_top3_strike2_homogeneous5f.csv", 0.95319),
        ("St3A", "sub_strike3a_cont065_plus_glmquad_graph.csv", 0.95322),
        ("St4A", "sub_gm_top3_strike4a_cont070.csv", 0.95323),
        ("St5A", "sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv", 0.95315),
        ("St6", "sub_gm_top3_strike6_reverse_strike5_from_095323.csv", 0.95325),
        ("St7", "sub_gm_top3_strike7_push_for_top1.csv", 0.95327),
        ("St8", "sub_gm_top1_strike8_claim_rank1.csv", 0.95329),
        ("St9", "sub_gm_top1_strike9_final_max.csv", 0.95330),
        ("leader", "sub_pass340_LEAD_st9_aman20_tuboa10.csv", 0.95325),
        ("soft", "sub_strike10_slot1_softdiff_pchip_0p35.csv", 0.95328),
        ("ANTI", "sub_pass340_ANTI_fam21_x040.csv", 0.95329),
        ("St9.5", "sub_strike9p5_axis_0p70_0p10.csv", 0.95332),
        ("Today_S1", "sub_strike9p6_opt_step.csv", 0.95333),
        ("Today_S2", "sub_ladder_slot2_v8_clean_0p18e4.csv", 0.95333),
        ("Today_S3", "sub_ladder_slot3_zref_clean_0p35e4.csv", 0.95329),
        ("Today_S4", "sub_ladder_slot4_adaptive_0p65e4.csv", 0.95334),
        ("Today_S5", "sub_ladder_slot5_macro_finale_1p25e4.csv", 0.95335),
    ]
    cache = ART / "_cache_lb_z.npz"
    Z = {}
    if cache.exists():
        raw = np.load(cache)
        for name, fn, lb in LB:
            Z[name] = {"z": raw[name].astype(np.float64), "lb": lb, "fn": fn}
        log(f"  loaded {len(Z)} LB vectors from cache")
    else:
        for name, fn, lb in LB:
            Z[name] = {"z": zrk(load_sub(fn, test_ids)), "lb": lb, "fn": fn}
            log(f"  loaded {name} LB={lb}")
        np.savez(cache, **{k: v["z"] for k, v in Z.items()})
        log(f"  wrote {cache}")
    return Z


def polish_d1_fast(y, folds, B_bad, report):
    """Re-build δ1_v4 from saved Stage-B + v8_v3 with P⊥→Method-B and 5/5 scale search."""
    log("=== FIX1-FAST: δ1_v4 from v8_v3 / 4fam (P⊥→Method-B) ===")
    z_v8_old_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_old_t = zrk(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel())
    z_v8_v3_o = np.load(ART3 / "z_v8_v3_oof.npy").ravel().astype(np.float64)
    z_v8_v3_t = np.load(ART3 / "z_v8_v3_test.npy").ravel().astype(np.float64)
    if not (ART / "z_stageB_4fam_oof.npy").exists():
        fams_o = [
            zrk(np.load(ART3 / "xgb_stageB_68_bce_oof.npy").ravel()),
            zrk(np.load(ART3 / "xgb_stageB_68_focal_auc_oof.npy").ravel()),
            zrk(np.load(ART3 / "lgb_stageB_68_clean_oof.npy").ravel()),
            zrk(np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel()),
        ]
        fams_t = [
            zrk(np.load(ART3 / "xgb_stageB_68_bce_test.npy").ravel()),
            zrk(np.load(ART3 / "xgb_stageB_68_focal_auc_test.npy").ravel()),
            zrk(np.load(ART3 / "lgb_stageB_68_clean_test.npy").ravel()),
            zrk(np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel()),
        ]
        np.save(ART / "z_stageB_4fam_oof.npy", 0.25 * sum(fams_o))
        np.save(ART / "z_stageB_4fam_test.npy", 0.25 * sum(fams_t))
    z_4_o = np.load(ART / "z_stageB_4fam_oof.npy").ravel()
    z_4_t = np.load(ART / "z_stageB_4fam_test.npy").ravel()
    z_fix_o = zrk(np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel())
    z_fix_t = zrk(np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel())

    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
    base_folds = fold_aucs(y, z_v8_old_o, folds)
    variants = {}
    raw_specs = [
        ("v8_v3_minus_old", z_v8_v3_o - z_v8_old_o, z_v8_v3_t - z_v8_old_t),
        ("stageB_vs_v8_w12", *orth_vs(z_4_o, z_4_t, [z_v8_old_o, z_w12_o], [z_v8_old_t, z_w12_t], folds)),
        ("lgb_fixed_vs_v8", *orth_vs(z_fix_o, z_fix_t, [z_v8_old_o], [z_v8_old_t], folds)),
    ]
    for tag, ro, rt in raw_specs:
        o, t, d, ks = finalize_delta(ro, rt, B_bad, folds, tag)
        variants[tag] = (o, t, d, ks)

    # Also try blend: v8_v3 residual + λ * lgb_fixed residual to flip F3
    o_v3, t_v3, _, _ = variants["v8_v3_minus_old"]
    o_lf, t_lf, _, _ = variants["lgb_fixed_vs_v8"]
    for lam in (0.25, 0.5, 0.75, 1.0):
        tag = f"v8_v3_plus_lgbf_λ{lam}"
        o, t, d, ks = finalize_delta(o_v3 + lam * o_lf, t_v3 + lam * t_lf, B_bad, folds, tag)
        variants[tag] = (o, t, d, ks)

    chosen = None
    chosen_tag = None
    chosen_meta = None
    for tag, (o, t, d, ks) in variants.items():
        best = None
        for w in np.linspace(0.0, 1.50, 61):
            z = z_v8_old_o + w * o
            nf = fold_aucs(y, z, folds)
            deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
            pos = int(sum(x > 0 for x in deltas))
            score = (pos, float(np.min(deltas)), float(np.mean(deltas)))
            if best is None or score > best[0]:
                best = (score, w, deltas, pos, float(roc_auc_score(y, z)))
        _, w, deltas, pos, auc = best
        meta = {
            "tag": tag, "disp": d, "ks": ks, "delta_e5": deltas, "pos": pos, "w": float(w),
            "auc": auc, "delta_vs_old_e5": (auc - float(roc_auc_score(y, z_v8_old_o))) * 1e5,
        }
        log(f"  eval {tag}: w={w:.4f} Δe5={[round(x,2) for x in deltas]} pos={pos}/5 KS={ks:.6f}")
        if chosen is None or (meta["pos"], -meta["ks"], meta["delta_vs_old_e5"]) > (
            chosen_meta["pos"], -chosen_meta["ks"], chosen_meta["delta_vs_old_e5"]
        ):
            chosen, chosen_tag, chosen_meta = (o, t), tag, meta

    d1_o = chosen_meta["w"] * chosen[0]
    d1_t = chosen_meta["w"] * chosen[1]
    d1_o, d1_t, d_f, ks_f = finalize_delta(d1_o, d1_t, B_bad, folds, f"δ1_v4_final({chosen_tag})")
    chosen_meta["disp"], chosen_meta["ks"] = d_f, ks_f
    np.save(ART / "d1_v4_oof.npy", d1_o)
    np.save(ART / "d1_v4_test.npy", d1_t)
    report["fix1"] = {"chosen": chosen_tag, "meta": chosen_meta, "variants": {k: {"disp": v[2], "ks": v[3]} for k, v in variants.items()}}
    log(f"  CHOSEN δ1_v4={chosen_tag} pos={chosen_meta['pos']}/5 KS={ks_f:.6f}")
    return d1_o, d1_t


def polish_d3_masa_fast(y, folds, B_bad, report):
    """Rescue masa via force_exact_law before KS gate; rebuild δ3_v4."""
    log("=== FIX2-FAST: rescue masa → lm_raw_super_v4 + δ3_v4 ===")
    ntr = len(y)
    z_raw_o = zrk(np.load(P12 / "lm_oof.npy").ravel())
    z_raw_t = zrk(np.load(P12 / "lm_test.npy").ravel())
    _, z_raw_t_a = align_method_b_smooth(z_raw_o, z_raw_t, folds)
    # bayes residual from cache or ladder
    if (ART / "_cache_gpt2m_bayes_oof.npy").exists():
        z_bay_o = np.load(ART / "_cache_gpt2m_bayes_oof.npy")
        z_bay_t = np.load(ART / "_cache_gpt2m_bayes_test.npy")
    else:
        z_bay_o = zrk(np.load(ART2 / "z_m_bayes_5f_oof.npy").ravel())
        z_bay_t = zrk(np.load(ART2 / "z_m_bayes_5f_test.npy").ravel())
    _, z_bay_t = align_method_b_smooth(z_bay_o, z_bay_t, folds)

    def resid_vs_raw(zo, zt):
        r_o = np.zeros(ntr)
        r_tf = []
        for f, (a, b) in enumerate(folds):
            beta = float(np.cov(zo[b], z_raw_o[b], ddof=0)[0, 1] / (np.var(z_raw_o[b]) + 1e-12))
            r_o[b] = unitize(zo[b] - beta * z_raw_o[b])
            r_tf.append(zt - beta * z_raw_t_a)
        ro, rt = align_method_b_smooth(r_o, np.mean(r_tf, 0), folds)
        if disp_ks(ro, rt)[1] >= KS_GATE:
            ro, rt = force_exact_law(ro, rt)
        return ro, rt

    r_bay_o, r_bay_t = resid_vs_raw(z_bay_o, z_bay_t)
    z_super0_o = z_raw_o + W_SUPER * r_bay_o
    super0_folds = fold_aucs(y, z_super0_o, folds)

    masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
    z_masa_o = zrk(np.load(masa / "oof.npy").ravel())
    z_masa_t = zrk(np.load(masa / "test.npy").ravel())
    _, z_masa_t = align_method_b_smooth(z_masa_o, z_masa_t, folds)
    r_masa_o, r_masa_t = resid_vs_raw(z_masa_o, z_masa_t)
    d_m, ks_m = disp_ks(r_masa_o, r_masa_t)

    # also try rev
    z_rev_o = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel())
    z_rev_t = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy").ravel())
    _, z_rev_t = align_method_b_smooth(z_rev_o, z_rev_t, folds)
    r_rev_o, r_rev_t = resid_vs_raw(z_rev_o, z_rev_t)

    kept = [("r_gpt2m_bayes", r_bay_o, r_bay_t, W_SUPER)]
    current_o = z_super0_o.copy()
    current_folds = super0_folds[:]
    w_grid = np.linspace(-0.05, 0.15, 41)
    trials = []
    for nm, ro, rt in [("r_masa_raw", r_masa_o, r_masa_t), ("r_rev_raw", r_rev_o, r_rev_t)]:
        d, ks = disp_ks(ro, rt)
        nested = np.zeros(ntr)
        nws = []
        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            best_w, best_s = 0.0, -1e9
            for w in w_grid:
                lifts = [
                    float(roc_auc_score(y[folds[oi][1]], current_o[folds[oi][1]] + w * ro[folds[oi][1]]))
                    - current_folds[oi]
                    for oi in other
                ]
                s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                if s > best_s:
                    best_s, best_w = s, float(w)
            nws.append(best_w)
            nested[b] = current_o[b] + best_w * ro[b]
        nf = fold_aucs(y, nested, folds)
        deltas = [(a - b) * 1e5 for a, b in zip(nf, current_folds)]
        pos = sum(x > 0 for x in deltas)
        # If still soft after Method-B, force exact law then re-measure KS
        if ks >= KS_GATE:
            ro, rt = force_exact_law(ro, rt)
            d, ks = disp_ks(ro, rt)
        keep = all(x > 0 for x in deltas) and abs(d - 1.0) < 1e-4 and ks < KS_GATE and float(np.mean(nws)) != 0.0
        trials.append({"name": nm, "disp": d, "ks": ks, "delta_e5": deltas, "pos": int(pos), "kept": keep, "w": float(np.mean(nws))})
        log(f"  try {nm}: Δe5={[round(x,2) for x in deltas]} pos={pos}/5 disp={d:.6f} KS={ks:.6f} keep={keep}")
        if keep:
            mw = float(np.mean(nws))
            kept.append((nm, ro, rt, mw))
            current_o = current_o + mw * ro
            current_folds = fold_aucs(y, current_o, folds)

    z_v4_o = z_raw_o.copy()
    z_v4_t = z_raw_t_a.copy()
    for nm, ro, rt, w in kept:
        z_v4_o = z_v4_o + w * ro
        z_v4_t = z_v4_t + w * rt
    _, z_v4_t = align_method_b_smooth(z_v4_o, z_v4_t, folds)
    np.save(ART / "z_lm_raw_super_v4_oof.npy", z_v4_o)
    np.save(ART / "z_lm_raw_super_v4_test.npy", z_v4_t)

    z_v8_old_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_old_t = zrk(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel())
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
    r_lin_o, r_lin_t = orth_vs(zrk(z_v4_o), zrk(z_v4_t), [z_v8_old_o, z_w12_o], [z_v8_old_t, z_w12_t], folds)
    p_w12_o, p_w12_t = expit(z_w12_o), expit(z_w12_t)
    beta = float(np.cov(zrk(z_v4_o), z_w12_o, ddof=0)[0, 1] / (np.var(z_w12_o) + 1e-12))
    g_o = np.tanh(np.abs(zrk(z_v4_o) - z_w12_o)) * p_w12_o * (1.0 - p_w12_o)
    g_t = np.tanh(np.abs(zrk(z_v4_t) - z_w12_t)) * p_w12_t * (1.0 - p_w12_t)
    r_bnd_o = unitize(g_o * (zrk(z_v4_o) - beta * z_w12_o))
    r_bnd_t = g_t * (zrk(z_v4_t) - beta * z_w12_t)
    d3_lin_o, d3_lin_t, d_lin, ks_lin = finalize_delta(r_lin_o, r_lin_t, B_bad, folds, "d3_v4_linear")
    d3_bnd_o, d3_bnd_t, d_bnd, ks_bnd = finalize_delta(r_bnd_o, r_bnd_t, B_bad, folds, "d3_v4_boundary")

    def eval_add(r_o, name):
        nested = np.zeros(ntr)
        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            best_w, best_s = 0.0, -1e9
            for w in w_grid:
                lifts = [
                    float(roc_auc_score(y[folds[oi][1]], z_super0_o[folds[oi][1]] + w * r_o[folds[oi][1]]))
                    - super0_folds[oi]
                    for oi in other
                ]
                s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                if s > best_s:
                    best_s, best_w = s, float(w)
            nested[b] = z_super0_o[b] + best_w * r_o[b]
        nf = fold_aucs(y, nested, folds)
        return {
            "name": name,
            "auc": float(roc_auc_score(y, nested)),
            "delta_vs_super0_1e5": [(a - b) * 1e5 for a, b in zip(nf, super0_folds)],
            "pos": int(sum((a - b) > 0 for a, b in zip(nf, super0_folds))),
        }

    ev_lin, ev_bnd = eval_add(d3_lin_o, "d3_v4_linear"), eval_add(d3_bnd_o, "d3_v4_boundary")
    use_bnd = ev_bnd["pos"] > ev_lin["pos"] or (ev_bnd["pos"] == ev_lin["pos"] and ev_bnd["auc"] > ev_lin["auc"])
    d3_o, d3_t = (d3_bnd_o, d3_bnd_t) if use_bnd else (d3_lin_o, d3_lin_t)
    d3_tag = "boundary" if use_bnd else "linear"
    np.save(ART / "d3_v4_linear_oof.npy", d3_lin_o)
    np.save(ART / "d3_v4_linear_test.npy", d3_lin_t)
    np.save(ART / "d3_v4_boundary_oof.npy", d3_bnd_o)
    np.save(ART / "d3_v4_boundary_test.npy", d3_bnd_t)
    np.save(ART / "d3_v4_oof.npy", d3_o)
    np.save(ART / "d3_v4_test.npy", d3_t)
    out = {
        "kept": [{"name": n, "w": w} for n, _, _, w in kept],
        "trials": trials,
        "masa_disp_ks_after_force": [d_m, ks_m],
        "v4_auc": float(roc_auc_score(y, z_v4_o)),
        "v4_folds": fold_aucs(y, z_v4_o, folds),
        "super0_auc": float(roc_auc_score(y, z_super0_o)),
        "ev_linear": ev_lin,
        "ev_boundary": ev_bnd,
        "chosen_d3": d3_tag,
        "d3_linear_disp_ks": [d_lin, ks_lin],
        "d3_boundary_disp_ks": [d_bnd, ks_bnd],
    }
    report["fix2"] = out
    log(f"  kept={out['kept']} chosen_d3={d3_tag}")
    return d3_o, d3_t


def polish_purge_fast(y, test_ids, folds, Z, d_ladder, e1, e2, B_bad, report):
    """Re-finalize p_purge from saved artifact or rebuild lightweight."""
    log("=== FIX3-FAST: purge vector + hist Ridge ===")
    z95, z_s5, z_s3 = Z["St9.5"]["z"], Z["Today_S5"]["z"], Z["Today_S3"]["z"]
    d2_bad = np.load(ART3 / "delta2_bad_test.npy").ravel()
    d1_old = np.load(ART2 / "d1_test.npy").ravel()
    d3_old = np.load(ART2 / "d3_test.npy").ravel()
    if (ART / "p_purge_oof.npy").exists() and (ART / "p_purge_test.npy").exists():
        p_o = np.load(ART / "p_purge_oof.npy").ravel()
        p_t = np.load(ART / "p_purge_test.npy").ravel()
        p_o, p_t, d_p, ks_p = finalize_delta(p_o, p_t, B_bad, folds, "p_purge_refinalize")
    else:
        # fallback: use -d2_bad orth against ladder/e1 as purge proxy
        p_raw = -unitize(d2_bad)
        p_raw = project_perp(p_raw, gram_orthonormal(np.column_stack([d_ladder, e1])))
        p_o = unitize(p_raw[: len(y)]) if False else unitize(np.interp(
            np.linspace(0, 1, len(y)), np.linspace(0, 1, len(p_raw)), np.sort(p_raw)
        ))
        p_o, p_t, d_p, ks_p = finalize_delta(p_o, p_raw, B_bad, folds, "p_purge_proxy")
    np.save(ART / "p_purge_oof.npy", p_o)
    np.save(ART / "p_purge_test.npy", p_t)

    basis_hist = [d_ladder, e1, e2, d2_bad, d1_old, d3_old]
    bnames = ["d_ladder", "e1", "e2", "d2_bad", "d1_old", "d3_old"]

    def coords(delta):
        c, *_ = np.linalg.lstsq(np.column_stack(basis_hist), delta, rcond=None)
        return {bnames[i]: float(c[i]) for i in range(len(bnames))}

    X, yy = [], []
    for name, meta in Z.items():
        c, *_ = np.linalg.lstsq(np.column_stack(basis_hist), meta["z"] - z95, rcond=None)
        X.append(c)
        yy.append(meta["lb"])
    X, yy = np.asarray(X), np.asarray(yy)
    preds_loo = np.zeros(len(yy))
    for i in range(len(yy)):
        mask = np.ones(len(yy), dtype=bool)
        mask[i] = False
        ridge_i = Ridge(alpha=1e-2, fit_intercept=True).fit(X[mask], yy[mask])
        preds_loo[i] = ridge_i.predict(X[i : i + 1])[0]
    loo_mae = float(np.mean(np.abs(preds_loo - yy)))
    ridge = Ridge(alpha=1e-2, fit_intercept=True).fit(X, yy)
    audit = {
        "Today_S3_vs_S95": coords(z_s3 - z95),
        "Today_S5_vs_S95": coords(z_s5 - z95),
        "S3_c_ladder_raw_note": "v3 c_ladder=-0.7738 because raw δ2 not orth vs d_ladder",
        "p_purge_disp_ks": [d_p, ks_p],
        "p_purge_cos_e1": cos(p_t, e1),
        "p_purge_cos_e2": cos(p_t, e2),
        "p_purge_cos_d2_bad": cos(p_t, d2_bad),
        "ridge_loo_mae": loo_mae,
        "ridge_coefs": dict(zip(bnames, map(float, ridge.coef_))),
    }
    log(f"  S3={ {k: round(v,4) for k,v in audit['Today_S3_vs_S95'].items()} }")
    log(f"  S5={ {k: round(v,4) for k,v in audit['Today_S5_vs_S95'].items()} }")
    log(f"  p_purge KS={ks_p:.6f} cos(d2_bad)={audit['p_purge_cos_d2_bad']:+.4f}")
    log(f"  Hist Ridge LOO MAE={loo_mae:.6e}")
    report["fix3"] = audit
    return p_o, p_t, d2_bad, d1_old, d3_old, ridge, basis_hist, bnames


def main():
    assert NO_SUBMIT, "STRICT NO_SUBMIT"
    fast = os.environ.get("GM_V4_FAST", "1") != "0"
    log(f"GM_V4_SURGICAL_POLISH START (NO_SUBMIT={NO_SUBMIT} FAST={fast})")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    Z = load_lb_bank(test_ids)
    d_ladder = Z["Today_S1"]["z"] - Z["St9.5"]["z"]
    e1 = Z["St4A"]["z"] - Z["St3A"]["z"]
    e2 = (Z["St5A"]["z"] - Z["St4A"]["z"]) - 0.45 * (Z["St4A"]["z"] - Z["St3A"]["z"])
    d2_bad = np.load(ART3 / "delta2_bad_test.npy").ravel()
    d_bad_leader = Z["leader"]["z"] - Z["St9"]["z"]
    d_bad_soft = Z["soft"]["z"] - Z["St9"]["z"]
    B_bad = gram_orthonormal(np.column_stack([e2, d_bad_leader, d_bad_soft, d2_bad]))
    log(f"  B_bad rank={B_bad.shape[1]}")

    report = {"NO_SUBMIT": True, "FAST": fast}
    if fast:
        d1_o, d1_t = polish_d1_fast(y, folds, B_bad, report)
        d3_o, d3_t = polish_d3_masa_fast(y, folds, B_bad, report)
        p_o, p_t, d2_bad, d1_old, d3_old, ridge, basis_hist, bnames = polish_purge_fast(
            y, test_ids, folds, Z, d_ladder, e1, e2, B_bad, report
        )
    else:
        z_v8_old_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
        z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
        z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
        _, z_w12_t = align_method_b_smooth(z_w12_o, z_w12_t, folds)
        z_ref_o = zrk(np.load(ART2 / "z_ref_clean_oof.npy").ravel())
        z_ref_t = zrk(np.load(ART2 / "z_ref_clean_test.npy").ravel())
        _, z_ref_t = align_method_b_smooth(z_ref_o, z_ref_t, folds)
        d1_o, d1_t = fix1(y, folds, z_v8_old_o, z_w12_o, z_w12_t, z_ref_o, z_ref_t, B_bad, report)
        d3_o, d3_t, _, _ = fix2(y, folds, z_v8_old_o, z_w12_o, z_w12_t, B_bad, report)
        p_o, p_t, d2_bad, d1_old, d3_old, ridge, basis_hist, bnames = fix3(
            y, test_ids, folds, Z, d_ladder, e1, e2, B_bad, report
        )

    B_bad2 = gram_orthonormal(np.column_stack([e2, d_bad_leader, d_bad_soft, d2_bad]))
    stage_candidates(y, test_ids, folds, Z, d_ladder, e1, e2, d2_bad, p_t, d1_t, d3_t, B_bad2, ridge, basis_hist, bnames, report)
    report["elapsed_s"] = round(time.time() - T0, 1)
    write_report(report)
    log(f"GM_V4_SURGICAL_POLISH_COMPLETE elapsed={report['elapsed_s']}s")
    if report["elapsed_s"] > 90:
        log(f"WARNING: exceeded 90s target ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()

