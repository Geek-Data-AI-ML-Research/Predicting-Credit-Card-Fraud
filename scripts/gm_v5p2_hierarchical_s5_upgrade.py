#!/usr/bin/env python3
"""GM v5.2 — Hierarchical 9-bag Stage-B + masaya/δ3 upgrade on untouched Ladder S5.

Adaptive 2-slot probe protocol (Probe A → branch → Probe B). Uses cached .npy/.npz only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("GM_V5P2_ROOT", Path(__file__).resolve().parents[1])).resolve()
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
ART41 = PRED / "gm_v4_1"
ART3 = PRED / "gm_v3_research"
ART2 = PRED / "gm_v2_ladder"
ART5 = PRED / "gm_v5_overnight"
P12 = ROOT / "external/nov08_part12"
CACHE41 = PRED / "gm_v4_1_fast_cache.npz"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
ART = PRED / "gm_v5p2"
ART.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

EPS = 1e-6
KS_GATE = 0.0002
KS_STRICT = 0.0001
N_Q = 10001
KAG = str(ROOT / ".venv_forensics/bin/kaggle")
COMP = "playground-series-s6e11"
DO_SUBMIT = os.environ.get("GM_V5P2_SUBMIT", "1") != "0"


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
    n = float(np.linalg.norm(v))
    return v / (n + 1e-18)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(np.asarray(b, dtype=np.float64).ravel())
        out -= float(np.dot(out, b)) * b
    return out


def gram_orthonormal(X):
    X = np.asarray(X, dtype=np.float64)
    cols = []
    for j in range(X.shape[1]):
        v = unitize(X[:, j])
        for c in cols:
            v = v - float(np.dot(v, c)) * c
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        cols.append(v / n)
    return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


def fold_std_oof(v, folds):
    out = np.zeros_like(v, dtype=np.float64)
    for _, b in folds:
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
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


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


def load_ot(oof_path, test_path):
    return np.load(oof_path).ravel().astype(np.float64), np.load(test_path).ravel().astype(np.float64)


def ab_zrk_align(oof_raw, test_raw, folds):
    zo, zt = zrk(oof_raw), zrk(test_raw)
    return align_method_b_smooth(zo, zt, folds)


def lofo_orth(zo, zt, bases_o, bases_t, folds):
    """Per-fold OLS residual of zo vs bases on train→apply to val/test; then Method-B."""
    Xo = np.column_stack([unitize(b) for b in bases_o])
    Xt = np.column_stack([unitize(b) for b in bases_t])
    ro = np.zeros(len(zo), dtype=np.float64)
    rt_folds = []
    for a, b in folds:
        beta, *_ = np.linalg.lstsq(Xo[a], zo[a], rcond=None)
        ra = zo[a] - Xo[a] @ beta
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (zo[b] - Xo[b] @ beta - mu) / sd
        rt_folds.append((zt - Xt @ beta - mu) / sd)
    rt = np.mean(rt_folds, 0)
    return align_method_b_smooth(ro, rt, folds)


def finalize_delta(ro, rt, B_bad, folds, tag=None, ks_gate=KS_GATE, silent=False):
    bases = [B_bad[:, j] for j in range(B_bad.shape[1])] if B_bad is not None and B_bad.size else []
    rt_p = orth_against(rt, bases) if bases else np.asarray(rt, dtype=np.float64).ravel()
    o_al, t_al = align_method_b_smooth(ro, rt_p, folds)
    d, ks = disp_ks(o_al, t_al)
    if ks >= ks_gate:
        o_al, t_al = force_exact_law(o_al, t_al)
        d, ks = disp_ks(o_al, t_al)
    if bases:
        t_al = orth_against(t_al, bases)
        o_al, t_al = force_exact_law(o_al, t_al)
        d, ks = disp_ks(o_al, t_al)
    mc = max((abs(cos(t_al, b)) for b in bases), default=0.0)
    if not silent and tag:
        log(f"  {tag}: disp={d:.6f} KS={ks:.6f} max|cos|={mc:.2e}")
    return o_al, t_al, d, ks


def map_probs_via_s5(z, p_s5):
    """Exact order-statistic map onto Ladder S5 probability curve."""
    z = np.asarray(z, dtype=np.float64).ravel()
    p_s5 = np.asarray(p_s5, dtype=np.float64).ravel()
    assert len(z) == len(p_s5)
    order = np.argsort(_jitter(z, 1e-7), kind="mergesort")
    p = np.empty_like(p_s5)
    p[order] = np.sort(p_s5)
    return p


def eval_add(base_o, delta_o, y, folds, tag, w_grid=None):
    base_folds = fold_aucs(y, base_o, folds)
    base_auc = float(roc_auc_score(y, base_o))
    if w_grid is None:
        w_grid = np.linspace(0.0, 0.35, 71)
    best = None
    for w in w_grid:
        z = base_o + w * delta_o
        nf = fold_aucs(y, z, folds)
        deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
        pos = int(sum(x > 0 for x in deltas))
        mean_d = float(np.mean(deltas))
        min_d = float(np.min(deltas))
        auc = float(roc_auc_score(y, z))
        score = (pos, min_d, mean_d, auc)
        if best is None or score > best[0]:
            best = (score, float(w), deltas, pos, auc)
    _, w, deltas, pos, auc = best
    log(
        f"  eval {tag}: w={w:+.4f} AUC={auc:.7f} (base={base_auc:.7f}) "
        f"Δe5={[round(x, 2) for x in deltas]} pos={pos}/5"
    )
    return {"w": w, "deltas": deltas, "pos": pos, "auc": auc, "base_auc": base_auc}


def kaggle_submit(path: Path, msg: str) -> str:
    cmd = [KAG, "competitions", "submit", "-c", COMP, "-f", str(path), "-m", msg]
    log(f"SUBMIT {path.name}: {msg}")
    subprocess.check_call(cmd)
    return path.name


def kaggle_poll(filename: str, timeout_s: int = 900, every: int = 20) -> float:
    import csv
    from io import StringIO

    t_end = time.time() + timeout_s
    while time.time() < t_end:
        out = subprocess.check_output(
            [KAG, "competitions", "submissions", "-c", COMP, "-v"], text=True
        )
        rows = list(csv.DictReader(StringIO(out)))
        for row in rows:
            if row.get("fileName") != filename:
                continue
            status = (row.get("status") or "").strip()
            score = (row.get("publicScore") or "").strip()
            log(f"  poll {filename}: status={status} score={score}")
            st_u = status.upper()
            if "COMPLETE" in st_u and score:
                return float(score)
            if any(x in st_u for x in ("ERROR", "INVALID")):
                raise RuntimeError(f"submission failed: {row}")
            break
        time.sleep(every)
    raise TimeoutError(f"no score for {filename} within {timeout_s}s")


def main():
    log("GM_V5P2_HIERARCHICAL_S5_UPGRADE START")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    # ---- load refs / cached deltas ----
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_t = zrk(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel())
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
    z_ref_o = np.load(ART5 / "z_ref_old_oof.npy").ravel().astype(np.float64)
    if (ART5 / "z_ref_old_test.npy").exists():
        z_ref_t = np.load(ART5 / "z_ref_old_test.npy").ravel().astype(np.float64)
    else:
        z_ref_t = z_v8_t.copy()
    _zrc = np.load(ART2 / "z_ref_clean_oof.npy").ravel().astype(np.float64)
    z_ref_clean_o = zrk(_zrc) if (_zrc.min() >= -1e-9 and _zrc.max() <= 1.0 + 1e-9) else unitize(_zrc)
    if float(roc_auc_score(y, z_ref_clean_o)) < 0.90:
        z_ref_clean_o = zrk(_zrc)

    d1_v4_o, d1_v4_t = load_ot(ART4 / "d1_v4_oof.npy", ART4 / "d1_v4_test.npy")
    d3l_o, d3l_t = load_ot(ART4 / "d3_v4_linear_oof.npy", ART4 / "d3_v4_linear_test.npy")
    d3b_o, d3b_t = load_ot(ART4 / "d3_v4_boundary_oof.npy", ART4 / "d3_v4_boundary_test.npy")
    B_bad = np.load(ART41 / "B_bad_orth.npy")
    Q = np.load(ART41 / "Q_orth_basis.npy")
    u_zj_t = unit(np.load(ART41 / "u_zjoint_clean_test.npy").ravel())
    u_gg_t = unit(np.load(ART41 / "u_glm_graph_clean_test.npy").ravel())

    log(
        f"  z_ref_old AUC={roc_auc_score(y,z_ref_o):.7f}  "
        f"z_v8 AUC={roc_auc_score(y,z_v8_o):.7f}  "
        f"z_ref_clean AUC={roc_auc_score(y,z_ref_clean_o):.7f}"
    )

    # ---- Step 1: 4 pillars ----
    log("======== STEP1: 4-pillar Stage-B pool ========")
    reuse_d1 = os.environ.get("GM_V5P2_REUSE_D1", "0") == "1" and (ART / "d1_v5p2_oof.npy").exists()
    if reuse_d1:
        log("  REUSE cached d1_v5p2")
        d1_o = np.load(ART / "d1_v5p2_oof.npy").ravel().astype(np.float64)
        d1_t = np.load(ART / "d1_v5p2_test.npy").ravel().astype(np.float64)
        d1_disp, d1_ks = disp_ks(d1_o, d1_t)
        ev = eval_add(z_v8_o, unitize(d1_o), y, folds, "δ1_v5p2@v8(reuse)")
        pack = {
            "key": ("reuse",),
            "pos": ev["pos"],
            "deltas": ev["deltas"],
            "auc_v8": ev["auc"],
            "w": ev["w"],
            "z_ref_lift_e5": (
                float(roc_auc_score(y, z_ref_o + ev["w"] * unitize(d1_o))) - float(roc_auc_score(y, z_ref_o))
            )
            * 1e5,
            "ks": d1_ks,
        }
        log(
            f"  CHOSEN δ1_v5p2 key=reuse pos={pack['pos']}/5 "
            f"Δe5={[round(x,2) for x in pack['deltas']]} KS={d1_ks:.6f}"
        )
    else:
        fams = {
            "xgb_bce": load_ot(ART3 / "xgb_stageB_68_bce_oof.npy", ART3 / "xgb_stageB_68_bce_test.npy"),
            "xgb_focal": load_ot(ART3 / "xgb_stageB_68_focal_auc_oof.npy", ART3 / "xgb_stageB_68_focal_auc_test.npy"),
            "xgb_mono": load_ot(ART5 / "xgb_stageB_monotone_lm_bag6_oof.npy", ART5 / "xgb_stageB_monotone_lm_bag6_test.npy"),
            "xgb_tail": load_ot(ART5 / "xgb_stageB_tail_specialist_bag6_oof.npy", ART5 / "xgb_stageB_tail_specialist_bag6_test.npy"),
            "lgb_et": load_ot(ART5 / "lgb_stageB_extratrees_bag6_oof.npy", ART5 / "lgb_stageB_extratrees_bag6_test.npy"),
            "lgb_clean": load_ot(ART3 / "lgb_stageB_68_clean_oof.npy", ART3 / "lgb_stageB_68_clean_test.npy"),
            "lgb_col": load_ot(ART5 / "lgb_stageB_colsub_deep_bag6_oof.npy", ART5 / "lgb_stageB_colsub_deep_bag6_test.npy"),
            "cat": load_ot(ART5 / "cat_stageB_bag9_oof.npy", ART5 / "cat_stageB_bag9_test.npy"),
            "lgb_fixed": load_ot(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy", PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy"),
        }
        aligned = {}
        for k, (o, t) in fams.items():
            ao, at = ab_zrk_align(o, t, folds)
            aligned[k] = (ao, at)
            log(f"  aligned {k}: OOF AUC={roc_auc_score(y,ao):.7f}")

        def mix(weights: dict):
            oo = np.zeros(len(y), dtype=np.float64)
            tt = np.zeros(aligned["cat"][1].shape[0], dtype=np.float64)
            s = 0.0
            for k, w in weights.items():
                oo += w * aligned[k][0]
                tt += w * aligned[k][1]
                s += w
            return unitize(oo / s), unitize(tt / s)

        px_o, px_t = mix({"xgb_bce": 0.30, "xgb_focal": 0.25, "xgb_mono": 0.25, "xgb_tail": 0.20})
        pl_o, pl_t = mix({"lgb_et": 0.40, "lgb_clean": 0.35, "lgb_col": 0.25})
        pc_o, pc_t = aligned["cat"]
        pf_o, pf_t = aligned["lgb_fixed"]
        for tag, oo in [("Pillar_XGB_24", px_o), ("Pillar_LGB_18", pl_o), ("Pillar_CAT_9", pc_o), ("Pillar_LGB_Fixed", pf_o)]:
            log(f"  {tag} AUC={roc_auc_score(y,oo):.7f}")

        bases_o, bases_t = [z_v8_o, z_w12_o], [z_v8_t, z_w12_t]
        rx_o, rx_t = lofo_orth(px_o, px_t, bases_o, bases_t, folds)
        rl_o, rl_t = lofo_orth(pl_o, pl_t, bases_o, bases_t, folds)
        rc_o, rc_t = lofo_orth(pc_o, pc_t, bases_o, bases_t, folds)
        rf_o, rf_t = lofo_orth(pf_o, pf_t, bases_o, bases_t, folds)
        pxn_o, pxn_t = mix({"xgb_mono": 0.55, "xgb_tail": 0.45})
        pln_o, pln_t = mix({"lgb_et": 0.60, "lgb_col": 0.40})
        rxn_o, rxn_t = lofo_orth(pxn_o, pxn_t, bases_o, bases_t, folds)
        rln_o, rln_t = lofo_orth(pln_o, pln_t, bases_o, bases_t, folds)

        d1_v4_o_u, d1_v4_t_u = unitize(d1_v4_o), unitize(d1_v4_t)
        grid = [0.0, 0.25, 0.50, 0.85]
        candidates = []
        for ax in grid:
            for al in grid:
                for ac in grid:
                    if ax + al + ac > 1.25:
                        continue
                    ro = unitize(d1_v4_o_u + ax * rxn_o + al * rln_o + ac * rc_o)
                    rt = unitize(d1_v4_t_u + ax * rxn_t + al * rln_t + ac * rc_t)
                    candidates.append(((ax, al, ac), ro, rt))
        for ax, al, ac, af in [
            (0.40, 0.35, 0.25, 0.40),
            (0.55, 0.25, 0.20, 0.55),
            (0.25, 0.55, 0.20, 0.35),
            (0.0, 0.0, 0.0, 0.0),
        ]:
            ro = unitize(d1_v4_o_u + ax * rx_o + al * rl_o + ac * rc_o + af * rf_o)
            rt = unitize(d1_v4_t_u + ax * rx_t + al * rl_t + ac * rc_t + af * rf_t)
            candidates.append(((ax, al, ac, af), ro, rt))

        log(f"  grid candidates={len(candidates)}")
        base_folds_v8 = fold_aucs(y, z_v8_o, folds)
        base_auc_ref = float(roc_auc_score(y, z_ref_o))
        w_grid_fast = np.linspace(0.0, 0.28, 29)
        best = None
        for key, ro, rt in candidates:
            oo, tt = unitize(ro), unitize(rt)
            local_best = None
            for w in w_grid_fast:
                z = z_v8_o + w * oo
                nf = fold_aucs(y, z, folds)
                deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds_v8)]
                pos = int(sum(x > 0 for x in deltas))
                score = (pos, float(np.min(deltas)), float(np.mean(deltas)), float(roc_auc_score(y, z)))
                if local_best is None or score > local_best[0]:
                    local_best = (score, float(w), deltas)
            (pos, min_d, mean_d, auc), w, deltas = local_best
            z_ref_lift = float(roc_auc_score(y, z_ref_o + w * oo)) - base_auc_ref
            if pos < 5 or z_ref_lift <= 0:
                continue
            pack = {
                "key": key,
                "pos": pos,
                "min_d": min_d,
                "mean_d": mean_d,
                "auc_v8": auc,
                "w": w,
                "deltas": deltas,
                "z_ref_lift_e5": z_ref_lift * 1e5,
                "ks": 0.0,
                "o": oo,
                "t": tt,
            }
            rank = (pos, pack["z_ref_lift_e5"], mean_d)
            if best is None or rank > best[0]:
                best = (rank, pack)

        if best is None:
            log("  WARN: no 5/5+z_ref candidate — falling back to δ1_v4")
            oo, tt, d, ks = finalize_delta(d1_v4_o, d1_v4_t, B_bad, folds, "δ1_v4_fallback")
            ev = eval_add(z_v8_o, oo, y, folds, "δ1_v4_fallback")
            pack = {
                "key": ("fallback",),
                "pos": ev["pos"],
                "deltas": ev["deltas"],
                "auc_v8": ev["auc"],
                "w": ev["w"],
                "z_ref_lift_e5": (float(roc_auc_score(y, z_ref_o + ev["w"] * oo)) - float(roc_auc_score(y, z_ref_o))) * 1e5,
                "ks": ks,
                "o": oo,
                "t": tt,
            }
        else:
            pack = best[1]
            oo, tt, d, ks = finalize_delta(pack["o"], pack["t"], B_bad, folds, "δ1_v5p2")
            pack["o"], pack["t"], pack["ks"], pack["disp"] = oo, tt, ks, d
            ev = eval_add(z_v8_o, oo, y, folds, "δ1_v5p2@v8")
            pack["w"], pack["deltas"], pack["pos"], pack["auc_v8"] = ev["w"], ev["deltas"], ev["pos"], ev["auc"]
            pack["z_ref_lift_e5"] = (
                float(roc_auc_score(y, z_ref_o + pack["w"] * oo)) - float(roc_auc_score(y, z_ref_o))
            ) * 1e5
            pack["ev_ref"] = eval_add(z_ref_o, oo, y, folds, "δ1_v5p2@z_ref_old")

        d1_o = pack["w"] * pack["o"]
        d1_t = pack["w"] * pack["t"]
        d1_o, d1_t, d1_disp, d1_ks = finalize_delta(d1_o, d1_t, B_bad, folds, "δ1_v5p2_scaled")
        if d1_ks >= 0.0002:
            d1_o, d1_t = force_exact_law(d1_o, d1_t)
            d1_t = orth_against(d1_t, [B_bad[:, j] for j in range(B_bad.shape[1])])
            d1_o, d1_t = force_exact_law(d1_o, d1_t)
            d1_disp, d1_ks = disp_ks(d1_o, d1_t)
            log(f"  δ1_v5p2 re-forced KS={d1_ks:.6f} disp={d1_disp:.6f}")
        log(
            f"  CHOSEN δ1_v5p2 key={pack['key']} pos={pack['pos']}/5 "
            f"Δe5={[round(x,2) for x in pack['deltas']]} z_ref_lift={pack['z_ref_lift_e5']:+.2f}e-5 KS={d1_ks:.6f}"
        )
        np.save(ART / "d1_v5p2_oof.npy", d1_o)
        np.save(ART / "d1_v5p2_test.npy", d1_t)

    # ---- Step 2: combo with δ3_v4 ----
    log("======== STEP2: δ_v5p2_combo + novelty over S5 ========")
    d3l_o_u, d3l_t_u = unitize(d3l_o), unitize(d3l_t)
    d3b_o_u, d3b_t_u = unitize(d3b_o), unitize(d3b_t)
    d1_o_u, d1_t_u = unitize(d1_o), unitize(d1_t)

    combo_best = None
    base_folds_ref = fold_aucs(y, z_ref_o, folds)
    base_folds_clean = fold_aucs(y, z_ref_clean_o, folds)
    # Tight weight set around proven δ1+δ3 mix (v4 used NN; here low-DOF nonneg)
    weight_sets = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.5, 0.0),
        (1.0, 0.0, 0.5),
        (1.0, 0.5, 0.35),
        (0.75, 0.75, 0.25),
        (0.50, 1.0, 0.50),
        (1.25, 0.75, 0.25),
        (1.0, 1.0, 0.0),
        (0.85, 0.60, 0.40),
    ]
    s_grid = np.linspace(0.0, 0.35, 36)
    for w1, w3l, w3b in weight_sets:
        ro = unitize(w1 * d1_o_u + w3l * d3l_o_u + w3b * d3b_o_u)
        local = None
        for s in s_grid:
            z = z_ref_o + s * ro
            nf = fold_aucs(y, z, folds)
            deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds_ref)]
            pos = int(sum(x > 0 for x in deltas))
            score = (pos, float(np.min(deltas)), float(np.mean(deltas)), float(roc_auc_score(y, z)))
            if local is None or score > local[0]:
                local = (score, float(s), deltas)
        (pos, min_d, mean_d, auc), s, deltas = local
        zc = z_ref_clean_o + s * ro
        nf_c = fold_aucs(y, zc, folds)
        deltas_c = [(a - b) * 1e5 for a, b in zip(nf_c, base_folds_clean)]
        pos_c = int(sum(x > 0 for x in deltas_c))
        rank = (pos, pos_c, mean_d, float(np.mean(deltas_c)), auc)
        if combo_best is None or rank > combo_best[0]:
            combo_best = (
                rank,
                {
                    "w1": w1,
                    "w3l": w3l,
                    "w3b": w3b,
                    "s": s,
                    "pos_old": pos,
                    "pos_clean": pos_c,
                    "deltas_old": deltas,
                    "deltas_clean": deltas_c,
                    "auc_old": auc,
                    "auc_clean": float(roc_auc_score(y, zc)),
                    "ro": ro,
                },
            )

    cb = combo_best[1]
    log(
        f"  combo w=({cb['w1']},{cb['w3l']},{cb['w3b']}) s={cb['s']:.4f} "
        f"old pos={cb['pos_old']}/5 Δe5={[round(x,2) for x in cb['deltas_old']]} "
        f"clean pos={cb['pos_clean']}/5 Δe5={[round(x,2) for x in cb['deltas_clean']]}"
    )
    combo_o = cb["s"] * cb["ro"]
    combo_t = cb["s"] * unitize(cb["w1"] * d1_t_u + cb["w3l"] * d3l_t_u + cb["w3b"] * d3b_t_u)
    combo_o, combo_t, c_disp, c_ks = finalize_delta(combo_o, combo_t, B_bad, folds, "δ_v5p2_combo")
    np.save(ART / "d_v5p2_combo_oof.npy", combo_o)
    np.save(ART / "d_v5p2_combo_test.npy", combo_t)

    # Untouched ladder z / today's drop
    p_s5 = pd.read_csv(SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv")["is_fraud"].to_numpy(np.float64)
    p_s4 = pd.read_csv(SUB / "sub_ladder_slot4_adaptive_0p65e4.csv")["is_fraud"].to_numpy(np.float64)
    p_s1 = pd.read_csv(SUB / "sub_strike9p6_opt_step.csv")["is_fraud"].to_numpy(np.float64)
    p_drop = pd.read_csv(SUB / "sub_gm_v5_overnight_slot5.csv")["is_fraud"].to_numpy(np.float64)
    z_s5 = zrk(p_s5)
    z_s4 = zrk(p_s4)
    z_s1 = zrk(p_s1)
    z_drop = zrk(p_drop)

    e2 = Q[:, 2]  # q_bad_tree ≡ Strike5A tree dilution direction in GS basis
    d_bad_soft = Q[:, 5]
    drop_dir = unitize(z_drop - z_s5)
    B_strict = gram_orthonormal(np.column_stack([e2, d_bad_soft, drop_dir]))
    log(f"  B_strict_bad cols={B_strict.shape[1]}")

    combo_t_perp = orth_against(combo_t, [B_strict[:, j] for j in range(B_strict.shape[1])])
    combo_o2, combo_t2, _, ks2 = finalize_delta(
        combo_o, combo_t_perp, None, folds, "δ_v5p2_combo⊥strict", ks_gate=KS_STRICT
    )
    # re-strip strict bad after align
    combo_t2 = orth_against(combo_t2, [B_strict[:, j] for j in range(B_strict.shape[1])])
    combo_o2, combo_t2 = force_exact_law(combo_o2, combo_t2)
    d2, ks2 = disp_ks(combo_o2, combo_t2)
    log(f"  combo⊥strict final disp={d2:.6f} KS={ks2:.6f} cos(drop)={cos(combo_t2, drop_dir):+.4f}")

    # Novel = combo minus OLS onto [δ1_old, δ3_old] then ⊥ strict
    d1_old_t = unitize(d1_v4_t)
    d3_old_t = unitize(0.5 * unitize(d3l_t) + 0.5 * unitize(d3b_t))
    X = np.column_stack([d1_old_t, d3_old_t])
    beta, *_ = np.linalg.lstsq(X, combo_t2, rcond=None)
    novel = combo_t2 - X @ beta
    novel = orth_against(novel, [B_strict[:, j] for j in range(B_strict.shape[1])])
    novel = unitize(novel)
    log(f"  δ_novel OLS β={beta.tolist()} ||novel||={np.linalg.norm(novel):.4f} cos(drop)={cos(novel, drop_dir):+.4f}")
    np.save(ART / "delta_novel_test.npy", novel)
    np.save(ART / "d_v5p2_combo_perp_test.npy", combo_t2)

    # Good reinforce: (u_zjoint + u_glm_graph) ⊥ strict
    u_good = unit(orth_against(unitize(u_zj_t + u_gg_t), [B_strict[:, j] for j in range(B_strict.shape[1])]))

    # ---- Step 3: Probe A ----
    log("======== STEP3: Probe A from untouched S5 ========")
    u_A = unitize(0.60 * unitize(combo_t2) + 0.25 * novel + 0.15 * u_good)
    u_A = orth_against(u_A, [B_strict[:, j] for j in range(B_strict.shape[1])])
    # hard-zero any residual drop component
    u_A = orth_against(u_A, [drop_dir])
    u_A = unit(u_A)
    assert abs(cos(u_A, drop_dir)) < 1e-8, cos(u_A, drop_dir)

    target_A = 0.25e-4
    wA, gotA = weight_for_spearman_dist(z_s5, u_A, target_A)
    z_A = z_s5 + wA * u_A
    # keep orthogonal to drop dir numerically
    deltaA = z_A - z_s5
    c_drop = float(np.dot(deltaA, unit(drop_dir)))
    z_A = z_A - c_drop * unit(drop_dir)
    wA2, gotA = weight_for_spearman_dist(z_s5, unit(z_A - z_s5), target_A)
    z_A = z_s5 + wA2 * unit(z_A - z_s5)
    assert abs(cos(z_A - z_s5, drop_dir)) < 1e-6

    p_A = map_probs_via_s5(z_A, p_s5)
    ks_s5 = float(ks_2samp(p_A, p_s5).statistic)
    log(
        f"  ProbeA: 1−ρ={1 - spearmanr(z_s5, z_A).correlation:.3e} (target {target_A:.3e}) "
        f"mean={p_A.mean():.6f} KS(S5)={ks_s5:.6f} cos(drop)={cos(z_A - z_s5, drop_dir):+.2e} "
        f"nan={np.isnan(p_A).sum()} inf={np.isinf(p_A).sum()}"
    )
    assert len(p_A) == 299984
    assert np.isnan(p_A).sum() == 0 and np.isinf(p_A).sum() == 0
    assert abs(p_A.mean() - float(p_s5.mean())) < 1e-12
    assert ks_s5 < 0.0001
    log(f"  ProbeA mean_p={p_A.mean():.6f} (S5 mean={p_s5.mean():.6f})")

    fn_A = "sub_gm_v5p2_probeA_unpurged_s5_plus_v5combo_0p25e4.csv"
    path_A = SUB / fn_A
    pd.DataFrame({"id": test_ids, "is_fraud": p_A}).to_csv(path_A, index=False)
    np.save(ART / "probeA_z.npy", z_A)
    np.save(ART / "probeA_u.npy", u_A)

    report = {
        "d1_key": list(pack["key"]) if isinstance(pack["key"], tuple) else pack["key"],
        "d1_pos": pack["pos"],
        "d1_deltas": pack["deltas"],
        "d1_ks": float(d1_ks),
        "combo": {
            "w1": cb["w1"],
            "w3l": cb["w3l"],
            "w3b": cb["w3b"],
            "s": cb["s"],
            "pos_old": cb["pos_old"],
            "pos_clean": cb["pos_clean"],
            "deltas_old": cb["deltas_old"],
            "deltas_clean": cb["deltas_clean"],
            "auc_old": cb["auc_old"],
            "auc_clean": cb["auc_clean"],
            "ks": float(c_ks),
        },
        "probeA": {
            "file": fn_A,
            "1mrho": float(1 - spearmanr(z_s5, z_A).correlation),
            "mean_p": float(p_A.mean()),
            "ks_s5": ks_s5,
            "cos_drop": float(cos(z_A - z_s5, drop_dir)),
        },
    }

    lb_A = None
    if DO_SUBMIT:
        kaggle_submit(path_A, "GM V5.2 ProbeA: untouched S5 + v5p2_combo/novel/zjoint @ 0.25e-4 (unpurged, ⊥drop)")
        lb_A = kaggle_poll(fn_A)
        report["probeA"]["lb"] = lb_A
        log(f"  LB_ProbeA = {lb_A:.5f}")
    else:
        log("  SKIP submit (GM_V5P2_SUBMIT=0)")

    # ---- Adaptive Probe B ----
    log("======== STEP3b: Adaptive Probe B ========")
    cont = unit(z_s5 - z_s4)  # proven S4→S5 ascent
    cont = orth_against(cont, [drop_dir])
    cont = unit(cont)

    branch = None
    if lb_A is None:
        branch = "nosubmit_dry"
        # dry-run: build Branch2 shape
        u_B = unit(0.50 * u_A + 0.50 * cont)
        target_B = 0.50e-4
        z_new = z_s5
    elif lb_A >= 0.95336:
        branch = 1
        z_new = z_A
        u_B = unit(u_A + 0.20 * cont)
        target_B = 0.60e-4  # 2.2x of 0.25e-4 vs S5… user said 1-rho(z_S5)=0.60e-4
        # measure vs S5
        base_for_step = z_s5
    elif round(float(lb_A), 5) == 0.95335:
        branch = 2
        u_B = unit(0.50 * u_A + 0.50 * cont)
        target_B = 0.50e-4
    else:
        branch = 3
        u_B = unit(cont + 0.20 * (-u_A))
        target_B = 0.22e-4

    if branch == 1:
        # step from S5 along winning direction to 0.60e-4 from S5
        base_for_step = z_s5
        # direction from S5 toward breakout axis
        u_B = unit(u_A + 0.20 * cont)
        u_B = orth_against(u_B, [drop_dir])
        u_B = unit(u_B)
        wB, gotB = weight_for_spearman_dist(base_for_step, u_B, target_B)
        z_B = base_for_step + wB * u_B
    else:
        u_B = orth_against(u_B, [drop_dir])
        u_B = unit(u_B)
        wB, gotB = weight_for_spearman_dist(z_s5, u_B, target_B)
        z_B = z_s5 + wB * u_B

    # strip drop
    z_B = z_B - float(np.dot(z_B - z_s5, unit(drop_dir))) * unit(drop_dir)
    # re-calibrate distance to S5
    u_B2 = unit(z_B - z_s5)
    wB, gotB = weight_for_spearman_dist(z_s5, u_B2, target_B)
    z_B = z_s5 + wB * u_B2

    p_B = map_probs_via_s5(z_B, p_s5)
    ks_s5_b = float(ks_2samp(p_B, p_s5).statistic)
    fn_B = f"sub_gm_v5p2_probeB_branch{branch}_{gotB*1e4:.2f}e4.csv".replace(" ", "")
    if branch == "nosubmit_dry":
        fn_B = "sub_gm_v5p2_probeB_dryrun.csv"
    path_B = SUB / fn_B
    pd.DataFrame({"id": test_ids, "is_fraud": p_B}).to_csv(path_B, index=False)
    np.save(ART / "probeB_z.npy", z_B)
    log(
        f"  ProbeB branch={branch}: 1−ρ(S5)={1 - spearmanr(z_s5, z_B).correlation:.3e} "
        f"target={target_B:.3e} KS(S5)={ks_s5_b:.6f} mean={p_B.mean():.6f}"
    )

    lb_B = None
    if DO_SUBMIT and branch != "nosubmit_dry":
        msg_B = {
            1: f"GM V5.2 ProbeB Branch1 breakout: S5+2.2x u_A@0.60e-4 after A={lb_A:.5f}",
            2: f"GM V5.2 ProbeB Branch2 crest: 0.5 u_A + 0.5 (S5-S4) @0.50e-4 after A={lb_A:.5f}",
            3: f"GM V5.2 ProbeB Branch3 reverse: (S5-S4)-0.2 u_A @0.22e-4 after A={lb_A:.5f}",
        }[branch]
        kaggle_submit(path_B, msg_B)
        lb_B = kaggle_poll(fn_B)
        report["probeB"] = {
            "branch": branch,
            "file": fn_B,
            "lb": lb_B,
            "1mrho": float(1 - spearmanr(z_s5, z_B).correlation),
            "target": target_B,
        }
        log(f"  LB_ProbeB = {lb_B:.5f}")
    else:
        report["probeB"] = {"branch": branch, "file": fn_B, "lb": None}

    # ---- Report ----
    best_lb = 0.95335
    lines = [
        "# GM V5.2 Adaptive Probes — Untouched Ladder S5 Upgrade",
        "",
        f"Elapsed compute+submit **{time.time()-T0:.1f}s**. Champion held: Ladder S5 **0.95335**.",
        "",
        "## Step 1 — δ1_v5p2 (4-pillar / NEW-overnight blend into δ1_v4)",
        "",
        f"- Chosen key=`{pack['key']}`",
        f"- On `v8_old`: pos **{pack['pos']}/5** Δe5=`{[round(x,2) for x in pack['deltas']]}` AUC=`{pack['auc_v8']:.7f}`",
        f"- z_ref_old lift ≈ `{pack['z_ref_lift_e5']:+.2f}e-5`; KS=`{d1_ks:.6f}`",
        f"- MLP dropped (0.95171).",
        "",
        "## Step 2 — δ_v5p2_combo + novelty",
        "",
        f"- Combo weights `(w1,w3_lin,w3_bnd)=({cb['w1']},{cb['w3l']},{cb['w3b']})` scale=`{cb['s']:.4f}`",
        f"- On `z_ref_old` ({cb['auc_old']:.7f}): pos **{cb['pos_old']}/5** Δe5=`{[round(x,2) for x in cb['deltas_old']]}`",
        f"- On `z_ref_clean` ({cb['auc_clean']:.7f}): pos **{cb['pos_clean']}/5** Δe5=`{[round(x,2) for x in cb['deltas_clean']]}`",
        f"- `B_strict_bad = orth_span(e2, d_bad_soft, z_v5_drop−z_S5)` cols={B_strict.shape[1]}",
        f"- combo⊥strict KS=`{ks2:.6f}` cos(drop)=`{cos(combo_t2, drop_dir):+.4f}`",
        f"- δ_novel OLS β on [δ1_v4, δ3_old]=`{[round(float(x),4) for x in beta]}`",
        "",
        "## Step 3 — Adaptive probes",
        "",
        f"| probe | file | 1−ρ(S5) | LB | Δbest |",
        f"|---|---|---:|---:|---:|",
        f"| A | `{fn_A}` | {report['probeA']['1mrho']:.3e} | {lb_A if lb_A is not None else '—'} | "
        f"{((lb_A-best_lb)*1e5 if lb_A is not None else float('nan')):+.1f}e-5 |",
        f"| B (branch {branch}) | `{fn_B}` | {float(1-spearmanr(z_s5,z_B).correlation):.3e} | "
        f"{lb_B if lb_B is not None else '—'} | "
        f"{((lb_B-best_lb)*1e5 if lb_B is not None else float('nan')):+.1f}e-5 |",
        "",
        f"**Branch rule used:** `{branch}` "
        + (
            f"(LB_A={lb_A:.5f})"
            if lb_A is not None
            else "(dry-run)"
        ),
        "",
        "Untouched S5 preserved (no bad_S3/ldr purge). Stop after Probe B — 7 slots reserved if 2 submitted.",
        "",
    ]
    report_path = LOG / "GM_V5P2_ADAPTIVE_PROBES.md"
    report_path.write_text("\n".join(lines))
    (ART / "REPORT.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"Wrote {report_path}")
    log(f"GM_V5P2_COMPLETE elapsed={time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
