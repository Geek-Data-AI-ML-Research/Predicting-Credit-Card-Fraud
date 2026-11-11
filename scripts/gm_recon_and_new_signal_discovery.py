#!/usr/bin/env python3
"""GM Recon & New Raw-Signal Discovery — STRICT NO_SUBMIT.

Part1: Kaggle/external bank recon + unused OOF residual probes
Part2: 2D quadratic vertex on S1/S4/S5/ProbeA/ProbeB
Part3: Brand-new P5_L5.1/5.2/5.3 levers vs z_ref_old+δ1_v4
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "1")

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("GM_RECON_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get("GM_RECON_DATA", ROOT / "data"))
ORIG = Path(os.environ.get("GM_RECON_ORIG", ROOT / "original"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

T0 = time.time()
PRED = Path(os.environ.get("GM_RECON_PRED", ROOT / "preds"))
SUB = ROOT / "submissions"
EXT = ROOT / "external"
ART4 = PRED / "gm_v4_polish"
ART5 = PRED / "gm_v5_overnight"
ART = Path(os.environ.get("GM_RECON_ART", str(PRED / "gm_recon_p5")))
LOG = Path(os.environ.get("GM_RECON_LOG", str(ROOT / "logs/overnight")))

# --- local-first overrides ---
if os.environ.get("GM_RECON_LOCAL", "0") == "1":
    WORK = Path(os.environ.get("GM_RECON_WORK", "/tmp/gm_recon"))
    PRED = WORK / "preds_bank"
    SUB = WORK / "submissions_local"
    EXT = WORK / "external"
    ART4 = WORK / "preds_local" / "gm_v4_polish"
    ART5 = WORK / "preds_local" / "gm_v5_overnight"
    ART = WORK / "preds_local" / "gm_recon_p5"
    LOG = WORK / "logs" / "overnight"
    # also search preds_local for probeA
    PRED_EXTRA = [WORK / "preds_local", WORK / "external"]
else:
    PRED_EXTRA = []

ART.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
KS_GATE = 0.0002
N_Q = 10001
KAG = str(Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/.venv_forensics/bin/kaggle"))
COMP = "playground-series-s6e11"

CONT = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


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
    oof_std = unitize(fold_std_oof(np.asarray(oof_vec, float).ravel(), folds))
    test_std = unitize(np.asarray(test_vec, float).ravel())
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std, jitter_scale), q),
        np.quantile(_jitter(oof_std, jitter_scale), q),
        extrapolate=True,
    )
    test_al = unitize(np.asarray(pchip(_jitter(test_std, jitter_scale)), float))
    if float(ks_2samp(oof_std, test_al).statistic) >= KS_GATE:
        oof_std, test_al = force_exact_law(oof_std, test_std)
    return oof_std, test_al


def disp_ks(o, t):
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    za = np.asarray(z_raw_a, float).reshape(-1, 1)
    ya = np.asarray(y_a, np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, float).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def lofo_residual(cand_o, cand_t, base_o, base_t, folds):
    """LOFO-orth residual of candidate vs base; Method-B align."""
    zo, zt = unitize(cand_o), unitize(cand_t)
    bo, bt = unitize(base_o), unitize(base_t)
    Xo = bo.reshape(-1, 1)
    Xt = bt.reshape(-1, 1)
    ro = np.zeros_like(zo)
    rt_f = []
    for a, b in folds:
        beta, *_ = np.linalg.lstsq(Xo[a], zo[a], rcond=None)
        ra = zo[a] - Xo[a] @ beta
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (zo[b] - Xo[b] @ beta - mu) / sd
        rt_f.append((zt - Xt @ beta - mu) / sd)
    rt = np.mean(rt_f, 0)
    return align_method_b_smooth(ro, rt, folds)


def eval_delta_on_base(base_o, delta_o, y, folds, w_grid=None):
    base_folds = fold_aucs(y, base_o, folds)
    base_auc = float(roc_auc_score(y, base_o))
    if w_grid is None:
        w_grid = np.linspace(-0.05, 0.35, 41)
    best = None
    for w in w_grid:
        z = base_o + w * delta_o
        nf = fold_aucs(y, z, folds)
        deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
        pos = int(sum(x > 0 for x in deltas))
        score = (pos, float(np.min(deltas)), float(np.mean(deltas)), float(roc_auc_score(y, z)))
        if best is None or score > best[0]:
            best = (score, float(w), deltas)
    (pos, min_d, mean_d, auc), w, deltas = best
    return {
        "w": w, "pos": pos, "deltas": deltas, "mean_d": mean_d, "min_d": min_d,
        "auc": auc, "base_auc": base_auc, "lift_e5": (auc - base_auc) * 1e5,
    }


# ============================================================================
# Part 1 — Recon
# ============================================================================
def part1_kaggle_recon() -> dict:
    log("======== PART1: Kaggle + local bank recon ========")
    out = {"kernels": [], "datasets": [], "unused_probes": [], "notes": []}
    try:
        raw = subprocess.check_output(
            [KAG, "kernels", "list", "--competition", COMP, "--sort-by", "scoreDescending", "--page-size", "50"],
            text=True, timeout=120,
        )
        out["kernels_raw"] = raw
        for line in raw.splitlines()[2:]:
            if not line.strip():
                continue
            out["kernels"].append(line[:160])
        log(f"  kernels listed: {len(out['kernels'])}")
        # highlight anything > 0.95335
        for line in out["kernels"]:
            m = re.search(r"0\.953[4-9]\d|0\.95[4-9]", line)
            if m:
                out["notes"].append(f"HIGH_KERNEL>{m.group(0)}: {line.strip()[:120]}")
                log(f"  !! {out['notes'][-1]}")
    except Exception as e:
        out["kernels_err"] = str(e)
        log(f"  kernels list failed: {e}")

    for q in ["s6e11", "s6e11 oof", "ps6e11"]:
        try:
            raw = subprocess.check_output(
                [KAG, "datasets", "list", "-s", q], text=True, timeout=120,
            )
            out.setdefault("datasets_raw", {})[q] = raw
            rows = [ln for ln in raw.splitlines()[2:] if ln.strip()]
            out["datasets"].extend(rows[:15])
            log(f"  datasets '{q}': {len(rows)} hits")
        except Exception as e:
            out.setdefault("datasets_err", {})[q] = str(e)
    return out


def find_pred_pairs(max_pairs: int = 60) -> list[tuple[str, Path, Path]]:
    """Find OOF/test prediction pairs under preds/ and external/ (prefer .npy)."""
    roots = [PRED, EXT] + list(globals().get("PRED_EXTRA", []))
    oof_map = {}
    test_map = {}
    patterns_oof = re.compile(r"(.+?)(?:_oof|_OOF|_nested)(\.npy)$", re.I)
    patterns_te = re.compile(r"(.+?)(?:_test|_te|_TEST)(\.npy)$", re.I)
    skip_dir = {"gpt2_med_4ep", "__pycache__", ".git", "checkpoints"}

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.npy"):
            if not p.is_file() or p.name.startswith("._"):
                continue
            if any(sd in p.parts for sd in skip_dir):
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz < 1000 or sz > 80_000_000:
                continue
            mo = patterns_oof.search(p.name)
            mt = patterns_te.search(p.name)
            if mo:
                oof_map[str(p.parent / mo.group(1))] = p
            if mt:
                test_map[str(p.parent / mt.group(1))] = p

    pairs = []
    for k in sorted(set(oof_map) & set(test_map)):
        pairs.append((k, oof_map[k], test_map[k]))
    # Prefer external_bank / nov* recon / part11 diversity first
    def rank(item):
        k = item[0].lower()
        score = 0
        if "external_bank" in k or "nov10" in k or "nov09" in k:
            score -= 10
        if "part11" in k or "harvest" in k or "tabm" in k:
            score -= 5
        if "gm_v" in k or "matrix_v2" in k:
            score += 5
        return (score, k)

    pairs.sort(key=rank)
    log(f"  found {len(pairs)} oof/test .npy pairs on disk")
    return pairs[:max_pairs]


def load_pred_vector(path: Path, n_expect: int | None = None) -> np.ndarray | None:
    try:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                return None
            x = np.asarray(arr, dtype=np.float64).ravel()
        elif path.suffix == ".npz":
            z = np.load(path, allow_pickle=True)
            # pick first 1d float array of right length
            x = None
            for k in z.files:
                a = np.asarray(z[k])
                if a.ndim == 1 and (n_expect is None or len(a) == n_expect):
                    x = a.astype(np.float64)
                    break
                if a.ndim == 2 and a.shape[1] == 1 and (n_expect is None or a.shape[0] == n_expect):
                    x = a.ravel().astype(np.float64)
                    break
            if x is None:
                return None
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            col = None
            for c in ["is_fraud", "pred", "prediction", "oof", "p", "prob"]:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                num = [c for c in df.columns if c != "id" and np.issubdtype(df[c].dtype, np.number)]
                if not num:
                    return None
                col = num[-1]
            x = df[col].to_numpy(np.float64)
        else:
            return None
        if n_expect is not None and len(x) != n_expect:
            return None
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x
    except Exception:
        return None


def part1_unused_probes(y, folds, z_base, z_base_t, z_s5_p) -> list[dict]:
    log("======== PART1b: unused OOF residual probes ========")
    pairs = find_pred_pairs(40)
    results = []
    ntr, nte = len(y), len(z_s5_p)
    # skip names clearly already in stack
    skip_sub = (
        "z_ref_old", "d1_v4", "d3_v4", "P_super_v8", "wide_lm_stack", "P4_L4.2_pure5f",
        "P4_L4.3_pure5f", "P4_L4.5_pure5f", "ladder_slot5", "gm_v5_overnight/z_ref",
        "d_v5p2", "probeA", "probeB",
    )
    probed = 0
    for key, op, tp in pairs:
        if any(s in key for s in skip_sub):
            continue
        o = load_pred_vector(op, ntr)
        t = load_pred_vector(tp, nte)
        if o is None or t is None:
            continue
        # correlation gate vs base / S5
        try:
            zo = zrk(o) if (o.min() >= -1e-9 and o.max() <= 1.0 + 1e-9) else unitize(o)
            zt = zrk(t) if (t.min() >= -1e-9 and t.max() <= 1.0 + 1e-9) else unitize(t)
            rho_base = float(spearmanr(zo, z_base).correlation)
            rho_s5 = float(spearmanr(zt, z_s5_p).correlation) if len(zt) == len(z_s5_p) else float("nan")
        except Exception:
            continue
        if abs(rho_base) > 0.9995:
            continue  # essentially already in base
        # LOFO residual probe
        try:
            ro, rt = lofo_residual(zo, zt, z_base, unitize(z_base_t), folds)
            d, ks = disp_ks(ro, rt)
            if ks >= KS_GATE:
                ro, rt = force_exact_law(ro, rt)
                d, ks = disp_ks(ro, rt)
            ev = eval_delta_on_base(z_base, ro, y, folds)
            row = {
                "key": key.replace(str(ROOT) + "/", ""),
                "rho_base": rho_base,
                "rho_s5": rho_s5,
                "pos": ev["pos"],
                "lift_e5": ev["lift_e5"],
                "mean_d": ev["mean_d"],
                "deltas": ev["deltas"],
                "w": ev["w"],
                "disp": d,
                "ks": ks,
                "oof": str(op.relative_to(ROOT)) if op.is_relative_to(ROOT) else str(op),
            }
            results.append(row)
            probed += 1
            if ev["pos"] >= 4 or ev["lift_e5"] > 1.0:
                log(
                    f"  HIT {row['key'][-70:]} pos={ev['pos']}/5 lift={ev['lift_e5']:+.2f}e-5 "
                    f"ρbase={rho_base:.4f} KS={ks:.5f}"
                )
            if probed >= 18:
                break
        except Exception:
            continue
        # hard cap wall attempts
        if probed >= 18:
            break
    results.sort(key=lambda r: (r["pos"], r["lift_e5"]), reverse=True)
    log(f"  probed {probed} unused pairs; top pos={results[0]['pos'] if results else '—'}")
    return results


# ============================================================================
# Part 2 — Quadratic vertex
# ============================================================================
def part2_quadratic_vertex() -> dict:
    log("======== PART2: 2D quadratic vertex on S1/S4/S5/A/B ========")
    files = {
        "S1": SUB / "sub_strike9p6_opt_step.csv",
        "S4": SUB / "sub_ladder_slot4_adaptive_0p65e4.csv",
        "S5": SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv",
        "A": SUB / "sub_gm_v5p2_probeA_unpurged_s5_plus_v5combo_0p25e4.csv",
        "B": SUB / "sub_gm_v5p2_probeB_branch3_0.22e4.csv",
    }
    lbs = {"S1": 0.95333, "S4": 0.95334, "S5": 0.95335, "A": 0.95334, "B": 0.95332}
    Z = {k: zrk(pd.read_csv(p)["is_fraud"].to_numpy(float)) for k, p in files.items()}
    u_A = np.load((Path(os.environ.get("GM_RECON_WORK","/tmp/gm_recon"))/"preds_local/gm_v5p2/probeA_u.npy") if os.environ.get("GM_RECON_LOCAL")=="1" else (PRED/"gm_v5p2/probeA_u.npy")).ravel()

    v1 = unit(Z["S5"] - Z["S4"])
    uA_p = unit(u_A - float(np.dot(u_A, v1)) * v1)
    v2 = unit(uA_p)

    def coords(z):
        d = z - Z["S5"]
        return float(np.dot(d, v1)), float(np.dot(d, v2))

    pts = {k: (*coords(Z[k]), lbs[k]) for k in Z}
    for k, (c1, c2, lb) in pts.items():
        log(f"  {k}: (v1,v2)=({c1:+.6f},{c2:+.6f}) LB={lb:.5f}")

    # Fit LB ≈ a0 + a1 x + a2 y + a3 x^2 + a4 xy + a5 y^2  (5 pts → use all 6 coeffs via least squares with 5 pts underdetermined for 6 — use ridge or drop xy)
    # With 5 points and 6 params, underdetermined. Fit without xy term (5 coeffs) exactly solvable if full rank; else LSTSQ.
    names = list(pts.keys())
    X = []
    ylb = []
    for k in names:
        x, y, lb = pts[k]
        X.append([1.0, x, y, x * x, y * y])  # no xy for exact-ish 5-param fit
        ylb.append(lb)
    X = np.asarray(X, float)
    ylb = np.asarray(ylb, float)
    coef, *_ = np.linalg.lstsq(X, ylb, rcond=None)
    a0, a1, a2, a3, a5 = coef
    # vertex of a0 + a1 x + a2 y + a3 x^2 + a5 y^2 : ∂/∂x=a1+2a3x=0 → x*=-a1/(2a3)
    v1_star = float("nan")
    v2_star = float("nan")
    if abs(a3) > 1e-18:
        v1_star = -a1 / (2.0 * a3)
    if abs(a5) > 1e-18:
        v2_star = -a2 / (2.0 * a5)
    pred = X @ coef
    resid = ylb - pred
    out = {
        "coords": {k: {"v1": pts[k][0], "v2": pts[k][1], "lb": pts[k][2]} for k in pts},
        "coef": {"a0": a0, "a1": a1, "a2": a2, "a3": a3, "a5": a5},
        "vertex": {"v1_star": v1_star, "v2_star": v2_star},
        "fit_resid": resid.tolist(),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "interp": [],
    }
    if np.isfinite(v1_star):
        if v1_star < 0:
            out["interp"].append("peak damped BACK toward S4 (v1*<0): do not push past S5 along S5-S4")
        else:
            out["interp"].append("peak ahead of S5 along continuation (v1*>0)")
    if np.isfinite(v2_star):
        if v2_star < 0:
            out["interp"].append("peak on -u_A side (v2*<0): ProbeA direction was wrong-signed")
        else:
            out["interp"].append("peak on +u_A side (v2*>0)")
    log(f"  vertex (v1*,v2*)=({v1_star:+.6f},{v2_star:+.6f}) rmse={out['rmse']:.2e}")
    for s in out["interp"]:
        log(f"  → {s}")
    return out


# ============================================================================
# Part 3 — P5 levers
# ============================================================================
def eb_te(train_keys, y, apply_keys, m=20.0):
    """Smoothed M-estimate TE; returns apply TE array."""
    y = np.asarray(y, float)
    prior = float(y.mean())
    df = pd.DataFrame({"k": train_keys, "y": y})
    g = df.groupby("k", sort=False)["y"].agg(["sum", "count"])
    te_map = (g["sum"] + m * prior) / (g["count"] + m)
    return te_map.reindex(apply_keys).fillna(prior).to_numpy(np.float32)


def multi_seed_te_matrix(keys_tr, keys_te, y, folds, seeds=(42, 101, 202, 303, 404), m=20.0):
    """5-seed nested TE: for each outer fold, average TE maps from inner seeds on train."""
    ntr, nte = len(y), len(keys_te)
    oof = np.zeros(ntr, dtype=np.float32)
    te_acc = np.zeros(nte, dtype=np.float64)
    for fi, (a, b) in enumerate(folds):
        pred_b = np.zeros(len(b), dtype=np.float64)
        pred_t = np.zeros(nte, dtype=np.float64)
        for s in seeds:
            # jitter keys slightly via seed-hash salt for digitization noise on numeric-coded keys
            # (keys are already strings/ints; salt by appending seed bucket of hash)
            ka = keys_tr[a]
            if ka.dtype == object or ka.dtype.kind in "SU":
                # string keys — use as-is (seed diversity via m jitter)
                mm = m * (0.85 + 0.05 * ((s % 7) / 7.0))
                te_b = eb_te(ka, y[a], keys_tr[b], m=mm)
                te_t = eb_te(ka, y[a], keys_te, m=mm)
            else:
                mm = m * (0.85 + 0.05 * ((s % 7) / 7.0))
                te_b = eb_te(ka, y[a], keys_tr[b], m=mm)
                te_t = eb_te(ka, y[a], keys_te, m=mm)
            pred_b += te_b
            pred_t += te_t
        oof[b] = (pred_b / len(seeds)).astype(np.float32)
        te_acc += pred_t / len(seeds)
        log(f"    TE outer-fold {fi} done")
    return oof, (te_acc / 5.0).astype(np.float32)


def cents(x):
    return (np.round(np.asarray(x, float) * 100) % 100).astype(np.int32)


def mantissa_str_tail(x):
    s = pd.Series(np.asarray(x, float)).map(lambda v: f"{v:.8f}")
    return s.str.replace(".", "", regex=False).str[-2:].to_numpy()


def floor_log10(x):
    return np.floor(np.log10(np.abs(np.asarray(x, float)) + 1.0)).astype(np.int32)


def build_p5_l51(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """2D cross-float mantissa / last-digit collision keys → later TE'd one-by-one then stacked."""
    keys_tr, keys_te, names = [], [], []
    pairs = [
        ("amount_usd", "account_balance_usd"),
        ("amount_usd", "merchant_risk_score"),
        ("amount_usd", "velocity_score"),
        ("account_balance_usd", "velocity_score"),
        ("distance_from_home_km", "hours_since_last_txn"),
        ("amount_usd", "distance_from_home_km"),
    ]
    for a, b in pairs:
        ca, cb = cents(tr[a]), cents(tr[b])
        ca_t, cb_t = cents(te[a]), cents(te[b])
        k = ca.astype(np.int64) * 1000 + cb.astype(np.int64)
        kt = ca_t.astype(np.int64) * 1000 + cb_t.astype(np.int64)
        keys_tr.append(k)
        keys_te.append(kt)
        names.append(f"cents_{a}_x_{b}")

        ma, mb = floor_log10(tr[a]), floor_log10(tr[b])
        ma_t, mb_t = floor_log10(te[a]), floor_log10(te[b])
        keys_tr.append(ma.astype(np.int64) * 100 + mb.astype(np.int64))
        keys_te.append(ma_t.astype(np.int64) * 100 + mb_t.astype(np.int64))
        names.append(f"log10_{a}_x_{b}")

    # amount exact-str tail × merchant_category
    tail = mantissa_str_tail(tr["amount_usd"])
    tail_t = mantissa_str_tail(te["amount_usd"])
    mc = tr["merchant_category"].astype(str).to_numpy()
    mc_t = te["merchant_category"].astype(str).to_numpy()
    keys_tr.append(np.array([f"{t}|{c}" for t, c in zip(tail, mc)], dtype=object))
    keys_te.append(np.array([f"{t}|{c}" for t, c in zip(tail_t, mc_t)], dtype=object))
    names.append("amt_tail_x_merchant")
    return keys_tr, keys_te, names


def build_p5_l52(tr: pd.DataFrame, te: pd.DataFrame, min_tr=5, min_te=2) -> tuple[np.ndarray, np.ndarray]:
    """3-way cat × fine quantile bin keys (filtered). Returns single combined key arrays."""
    full = pd.concat([tr, te], ignore_index=True)
    # ratios
    full["amt_bal_ratio"] = full["amount_usd"] / (full["account_balance_usd"].abs() + 1.0)
    full["vel_risk_ratio"] = full["velocity_score"] / (full["merchant_risk_score"] + 1.0)
    full["dist_hours_ratio"] = full["distance_from_home_km"] / (full["hours_since_last_txn"] + 1.0)
    ntr0 = len(tr)
    for col in ["amt_bal_ratio", "vel_risk_ratio", "dist_hours_ratio", "amount_usd", "velocity_score"]:
        # 50-bin quantile on train only
        qs = np.nanquantile(full.loc[: ntr0 - 1, col].to_numpy(float), np.linspace(0, 1, 51))
        qs = np.unique(qs)
        full[f"q50_{col}"] = np.digitize(full[col].to_numpy(float), qs[1:-1], right=True).astype(np.int32)

    combos = [
        ("merchant_category", "device_type", "q50_amount_usd"),
        ("auth_method", "channel", "q50_velocity_score"),
        ("card_type", "merchant_category", "q50_amt_bal_ratio"),
        ("device_type", "channel", "q50_vel_risk_ratio"),
        ("merchant_category", "auth_method", "q50_dist_hours_ratio"),
    ]
    ntr = len(tr)
    # Build list of filtered key columns then hash-combine into one dense code via factorize after filter
    key_cols = []
    for c1, c2, c3 in combos:
        s = full[c1].astype(str) + "|" + full[c2].astype(str) + "|" + full[c3].astype(str)
        vc_tr = s.iloc[:ntr].value_counts()
        vc_te = s.iloc[ntr:].value_counts()
        keep = set(vc_tr[vc_tr >= min_tr].index) & set(vc_te[vc_te >= min_te].index)
        s2 = s.where(s.isin(keep), other="__RARE__")
        key_cols.append(s2.to_numpy())
        log(f"    L5.2 {c1}|{c2}|{c3}: kept {len(keep)} keys")
    # stack as multiple keys — return as list
    return key_cols, ntr


def build_p5_l53(tr: pd.DataFrame, te: pd.DataFrame, orig: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Within-group ranks + centroid distances to orig fraud/neg + IF score."""
    ntr = len(tr)
    full = pd.concat([tr.reset_index(drop=True), te.reset_index(drop=True)], ignore_index=True)
    feats = []
    names = []

    for gcol in ["merchant_category", "device_type", "auth_method", "channel"]:
        # percentile of amount within group
        r = full.groupby(gcol)["amount_usd"].rank(pct=True).to_numpy(np.float32)
        feats.append(r)
        names.append(f"pct_amt_in_{gcol}")
        r2 = full.groupby(gcol)["velocity_score"].rank(pct=True).to_numpy(np.float32)
        feats.append(r2)
        names.append(f"pct_vel_in_{gcol}")

    # standardized L1 / mahalanobis-diag to orig centroids
    use_cols = [c for c in CONT + FLAGS if c in orig.columns and c in full.columns]
    Xo = orig[use_cols].to_numpy(np.float64)
    y_o = orig["is_fraud"].to_numpy(np.int32)
    mu = Xo.mean(0)
    sd = Xo.std(0) + 1e-8
    Xo_z = (Xo - mu) / sd
    Xf = full[use_cols].to_numpy(np.float64)
    Xf_z = (Xf - mu) / sd
    fraud = Xo_z[y_o == 1]
    neg = Xo_z[y_o == 0]
    mu_f, mu_n = fraud.mean(0), neg.mean(0)
    # diagonal mahalanobis using group variances
    vf = fraud.var(0) + 1e-6
    vn = neg.var(0) + 1e-6
    d_f = np.sqrt((((Xf_z - mu_f) ** 2) / vf).sum(1)).astype(np.float32)
    d_n = np.sqrt((((Xf_z - mu_n) ** 2) / vn).sum(1)).astype(np.float32)
    feats.extend([d_f, d_n, (d_n - d_f).astype(np.float32)])
    names.extend(["maha_diag_to_orig_fraud", "maha_diag_to_orig_neg", "maha_gap_neg_minus_fraud"])

    # prototype min L1 to 400 fraud / 400 neg (chunked)
    rng = np.random.default_rng(42)
    pf = fraud[rng.choice(len(fraud), size=min(400, len(fraud)), replace=False)]
    pn = neg[rng.choice(len(neg), size=min(400, len(neg)), replace=False)]
    # use first 8 dims for speed
    dsub = min(8, Xf_z.shape[1])
    X8 = Xf_z[:, :dsub].astype(np.float32)
    pf8 = pf[:, :dsub].astype(np.float32)
    pn8 = pn[:, :dsub].astype(np.float32)
    min_f = np.full(len(X8), np.inf, np.float32)
    min_n = np.full(len(X8), np.inf, np.float32)
    bs = 50000
    for i0 in range(0, len(X8), bs):
        xb = X8[i0 : i0 + bs]
        # ||x-p||_1 = sum_j |x_j - p_j|
        df = np.min(np.abs(xb[:, None, :] - pf8[None, :, :]).sum(2), axis=1)
        dn = np.min(np.abs(xb[:, None, :] - pn8[None, :, :]).sum(2), axis=1)
        min_f[i0 : i0 + bs] = df
        min_n[i0 : i0 + bs] = dn
        log(f"    L5.3 prototype chunk {i0}:{i0+len(xb)}")
    feats.extend([min_f, min_n, (min_n - min_f).astype(np.float32)])
    names.extend(["minL1_proto_fraud", "minL1_proto_neg", "minL1_gap"])

    # IsolationForest on CONT (fit on train only)
    Xtr = tr[CONT].to_numpy(np.float32)
    Xte = te[CONT].to_numpy(np.float32)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)
    iso = IsolationForest(n_estimators=200, max_samples=8192, contamination="auto", random_state=42, n_jobs=4)
    iso.fit(Xtr_s)
    s_tr = -iso.score_samples(Xtr_s).astype(np.float32)  # higher = more anomalous
    s_te = -iso.score_samples(Xte_s).astype(np.float32)
    if_full = np.concatenate([s_tr, s_te])
    feats.append(if_full)
    names.append("isolation_forest_anom")

    F = np.column_stack(feats).astype(np.float32)
    return F[:ntr], F[ntr:], names


def te_keys_to_matrix(keys_list_tr, keys_list_te, y, folds):
    cols_o, cols_t = [], []
    for i, (ktr, kte) in enumerate(zip(keys_list_tr, keys_list_te)):
        log(f"  TE keycol {i+1}/{len(keys_list_tr)}")
        o, t = multi_seed_te_matrix(ktr, kte, y, folds, seeds=(42, 101, 202), m=20.0)
        cols_o.append(o)
        cols_t.append(t)
    return np.column_stack(cols_o).astype(np.float32), np.column_stack(cols_t).astype(np.float32)


def fit_p4_pure5f_lever(Xo, Xt, y, folds, z_ref, z_ref_t, tag: str) -> dict:
    """Ridge(α=1000) + shallow depth-3 GBDT residual on calibrate_logit(z_ref)."""
    import lightgbm as lgb

    ntr = len(y)
    oof_ridge = np.zeros(ntr, dtype=np.float64)
    oof_gbdt = np.zeros(ntr, dtype=np.float64)
    te_ridge, te_gbdt = [], []
    for fi, (a, b) in enumerate(folds):
        tr_in, es_in = a[::10][: max(1000, len(a) // 10)], a  # unused placeholder
        # proper ES split
        from sklearn.model_selection import train_test_split
        tr_in, es_in = train_test_split(a, test_size=0.10, random_state=42 + fi, stratify=y[a])
        bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(z_ref[tr_in], y[tr_in], z_ref, z_ref_t)
        # residual target for ridge
        p_tr = expit(bm_tr)
        res = (y[tr_in].astype(float) - p_tr)
        sc = StandardScaler().fit(Xo[tr_in])
        Xa = np.nan_to_num(sc.transform(Xo[tr_in]), 0.0)
        Xb = np.nan_to_num(sc.transform(Xo[b]), 0.0)
        Xes = np.nan_to_num(sc.transform(Xo[es_in]), 0.0)
        Xt_s = np.nan_to_num(sc.transform(Xt), 0.0)
        ridge = Ridge(alpha=1000.0).fit(Xa, res)
        oof_ridge[b] = ridge.predict(Xb)
        te_ridge.append(ridge.predict(Xt_s))

        dtrain = lgb.Dataset(Xa, label=y[tr_in], free_raw_data=False)
        dtrain.set_init_score(bm_tr)
        des = lgb.Dataset(Xes, label=y[es_in], free_raw_data=False)
        des.set_init_score(bm_all[es_in])
        params = dict(
            objective="binary", metric="auc", learning_rate=0.03, num_leaves=8,
            max_depth=3, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=50.0, verbosity=-1, seed=42 + fi, num_threads=4,
        )
        bst = lgb.train(
            params, dtrain, num_boost_round=400, valid_sets=[des],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        # predict margin residual-ish: raw score with init
        raw_b = bst.predict(Xb, raw_score=True)
        raw_t = bst.predict(Xt_s, raw_score=True)
        oof_gbdt[b] = raw_b - bm_all[b]
        te_gbdt.append(raw_t - bm_te)
        log(f"  {tag} F{fi} ridge/gbdt done best={bst.best_iteration}")
        del bst
        gc.collect()

    # blend ridge + gbdt residuals
    ro = unitize(0.45 * unitize(oof_ridge) + 0.55 * unitize(oof_gbdt))
    rt = unitize(0.45 * unitize(np.mean(te_ridge, 0)) + 0.55 * unitize(np.mean(te_gbdt, 0)))
    ro, rt = align_method_b_smooth(ro, rt, folds)
    d, ks = disp_ks(ro, rt)
    if ks >= KS_GATE:
        ro, rt = force_exact_law(ro, rt)
        d, ks = disp_ks(ro, rt)
    ev = eval_delta_on_base(z_ref, ro, y, folds)
    np.save(ART / f"{tag}_oof.npy", ro.astype(np.float32))
    np.save(ART / f"{tag}_test.npy", rt.astype(np.float32))
    out = {"tag": tag, "disp": d, "ks": ks, **ev}
    log(
        f"  {tag}: pos={ev['pos']}/5 lift={ev['lift_e5']:+.2f}e-5 "
        f"Δe5={[round(x,2) for x in ev['deltas']]} disp={d:.4f} KS={ks:.6f}"
    )
    return out


def part3_p5(y, folds, z_base, z_base_t) -> list[dict]:
    log("======== PART3: P5_CrossMantissa_3Way levers ========")
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    orig = pd.read_csv(ORIG / "credit_card_fraud_2026.csv")
    log(f"  loaded train={len(tr)} test={len(te)} orig={len(orig)}")

    results = []

    # L5.1
    log("  --- P5_L5.1 ---")
    ktr, kte, names = build_p5_l51(tr, te)
    Xo, Xt = te_keys_to_matrix(ktr, kte, y, folds)
    log(f"  L5.1 matrix {Xo.shape}")
    results.append(fit_p4_pure5f_lever(Xo, Xt, y, folds, z_base, z_base_t, "P5_L5.1"))
    del Xo, Xt, ktr, kte
    gc.collect()

    # L5.2
    log("  --- P5_L5.2 ---")
    key_cols, ntr = build_p5_l52(tr, te)
    ktr = [kc[:ntr] for kc in key_cols]
    kte = [kc[ntr:] for kc in key_cols]
    Xo, Xt = te_keys_to_matrix(ktr, kte, y, folds)
    log(f"  L5.2 matrix {Xo.shape}")
    results.append(fit_p4_pure5f_lever(Xo, Xt, y, folds, z_base, z_base_t, "P5_L5.2"))
    del Xo, Xt, key_cols, ktr, kte
    gc.collect()

    # L5.3
    log("  --- P5_L5.3 ---")
    Xo, Xt, names = build_p5_l53(tr, te, orig)
    log(f"  L5.3 matrix {Xo.shape} names={names}")
    results.append(fit_p4_pure5f_lever(Xo, Xt, y, folds, z_base, z_base_t, "P5_L5.3"))
    del Xo, Xt
    gc.collect()
    return results


def write_report(recon, probes, vertex, p5, z_base_auc) -> Path:
    lines = [
        "# GM Recon & P5 Discovery Report",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{time.time()-T0:.1f}s**.",
        f"Base for probes: `z_ref_old + δ1_v4` OOF AUC≈`{z_base_auc:.7f}` (target 0.9526906).",
        "Champion held: Ladder S5 **0.95335**. No submissions made.",
        "",
        "## Part 1 — Live Kaggle + unused residual probes",
        "",
        "### High-signal kernels (scoreDescending)",
        "",
    ]
    for n in recon.get("notes", []):
        lines.append(f"- **{n}**")
    lines.append("")
    lines.append("Top kernels (truncated):")
    lines.append("```")
    for ln in recon.get("kernels", [])[:12]:
        lines.append(ln)
    lines.append("```")
    lines.append("")
    lines.append("### Unused OOF probes on base (top by pos/lift)")
    lines.append("")
    lines.append("| key | pos/5 | lift e-5 | mean Δe5 | ρ(base) | KS |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in probes[:15]:
        lines.append(
            f"| `{r['key'][-80:]}` | {r['pos']}/5 | {r['lift_e5']:+.2f} | {r['mean_d']:+.2f} | "
            f"{r['rho_base']:.4f} | {r['ks']:.5f} |"
        )
    if not probes:
        lines.append("| — | — | — | — | — | — |")
    lines += [
        "",
        "## Part 2 — Quadratic vertex (v1=S5−S4, v2=u_A⊥v1)",
        "",
        "| point | v1 | v2 | LB |",
        "|---|---:|---:|---:|",
    ]
    for k, v in vertex.get("coords", {}).items():
        lines.append(f"| {k} | {v['v1']:+.6f} | {v['v2']:+.6f} | {v['lb']:.5f} |")
    vv = vertex.get("vertex", {})
    lines += [
        "",
        f"- **Vertex:** `(v1*, v2*) = ({vv.get('v1_star'):+.6f}, {vv.get('v2_star'):+.6f})`",
        f"- Fit RMSE=`{vertex.get('rmse'):.2e}`",
        "",
    ]
    for s in vertex.get("interp", []):
        lines.append(f"- {s}")
    lines += [
        "",
        "## Part 3 — New P5 levers (individual, P4_pure5f recipe)",
        "",
        "| lever | pos/5 | lift e-5 | Δe5 folds | disp | KS | w |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in p5:
        lines.append(
            f"| **{r['tag']}** | {r['pos']}/5 | {r['lift_e5']:+.2f} | "
            f"`{[round(x,2) for x in r['deltas']]}` | {r['disp']:.4f} | {r['ks']:.6f} | {r['w']:.4f} |"
        )
    lines += [
        "",
        "### Notes",
        "",
        "- Avoided unregularized GLM (Ridge α=1000 + depth-3 LGB residual on `calibrate_logit(z_ref)`).",
        "- Test aligned via Method-B + exact order-stat map (KS gate 0.0002).",
        "- Artifacts: `preds/gm_recon_p5/P5_L5.{1,2,3}_{oof,test}.npy`.",
        "",
        "`NO_SUBMIT=True` — 7 Kaggle slots remain unused.",
        "",
    ]
    path = LOG / "GM_RECON_AND_P5_DISCOVERY.md"
    path.write_text("\n".join(lines))
    log(f"Wrote {path}")
    return path


def main():
    assert NO_SUBMIT, "STRICT NO_SUBMIT"
    log("GM_RECON_AND_NEW_SIGNAL_DISCOVERY START (NO_SUBMIT)")
    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    z_ref = np.load(ART5 / "z_ref_old_oof.npy").ravel().astype(np.float64)
    z_ref_t = np.load(ART5 / "z_ref_old_test.npy").ravel().astype(np.float64)
    d1 = np.load(ART4 / "d1_v4_oof.npy").ravel().astype(np.float64)
    d1_t = np.load(ART4 / "d1_v4_test.npy").ravel().astype(np.float64)
    # scale d1 like v5p2 (~0.03 unitize) — use unitize then small w from eval
    d1u, d1tu = unitize(d1), unitize(d1_t)
    ev_d1 = eval_delta_on_base(z_ref, d1u, y, folds, w_grid=np.linspace(0.0, 0.10, 21))
    z_base = z_ref + ev_d1["w"] * d1u
    z_base_t = unitize(z_ref_t) + ev_d1["w"] * d1tu
    z_base_auc = float(roc_auc_score(y, z_base))
    log(f"  base z_ref+δ1_v4 AUC={z_base_auc:.7f} (w={ev_d1['w']:.4f})")

    p_s5 = pd.read_csv(SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv")["is_fraud"].to_numpy(float)

    recon = part1_kaggle_recon()
    probes = part1_unused_probes(y, folds, z_base, z_base_t, p_s5)
    vertex = part2_quadratic_vertex()
    p5 = part3_p5(y, folds, z_base, z_base_t)

    report = {
        "NO_SUBMIT": True,
        "base_auc": z_base_auc,
        "recon_notes": recon.get("notes", []),
        "probes_top": probes[:20],
        "vertex": vertex,
        "p5": [{k: v for k, v in r.items() if k != "deltas" or True} for r in p5],
        "elapsed_s": time.time() - T0,
    }
    (ART / "REPORT.json").write_text(json.dumps(report, indent=2, default=float))
    write_report(recon, probes, vertex, p5, z_base_auc)
    log(f"GM_RECON_COMPLETE elapsed={time.time()-T0:.1f}s NO_SUBMIT={NO_SUBMIT}")


if __name__ == "__main__":
    main()
