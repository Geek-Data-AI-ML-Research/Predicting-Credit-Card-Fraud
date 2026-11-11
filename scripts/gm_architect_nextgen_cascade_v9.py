#!/usr/bin/env python3
"""GM Architect Nextgen Cascade v9 — disk rescue + P4_L4.6 hier EB + newton-cholesky regime.

STRICT NO_SUBMIT=True. No submission CSVs. No Playbook-3 ensemble residuals.
Soft RSS < 600 MB; wall target < 10 min on Mac CPU.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path(os.environ.get("GM_V9_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
WORK = Path(os.environ.get("GM_V9_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_architect_v9"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
for p in (ART, LOG):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT
EPS = 1e-6
RSS_SOFT = 600.0
T0 = time.time()
INNER_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]  # 8-seed hier EB
INNER_SPLITS = 3  # lean inner CV (speed); still multi-seed bagging

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss_curr() -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def rss_max() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|RSS={rss_curr():.0f}/{rss_max():.0f}MB] {msg}", flush=True)
    if rss_curr() > RSS_SOFT:
        print(f"  WARN RSS {rss_curr():.0f}MB > soft {RSS_SOFT}", flush=True)


def purge(*_):
    gc.collect()


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).ravel()


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def _jitter(x, scale=1e-11):
    x = np.asarray(x, dtype=np.float64)
    return x + scale * (rankdata(x, method="ordinal") / len(x) - 0.5)


def force_exact_law(oof_vec, test_vec):
    o = unitize(oof_vec).astype(np.float64)
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


def disp_ks(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.std(b) / (np.std(a) + 1e-18)), float(ks_2samp(a, b).statistic)


def lofo_resid(y_vec, bases, folds):
    y_vec = np.asarray(y_vec, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize(resid)


def fold_aucs(y, z, folds):
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_delta(y, z_base, z_new, folds, m0=None):
    bf = fold_aucs(y, z_base, folds)
    nf = fold_aucs(y, z_new, folds)
    d5 = [(a - b) * 1e5 for a, b in zip(nf, bf)]
    out = {"d5_mean": float(np.mean(d5)), "pos": int(sum(x > 0 for x in d5)), "d5": [float(x) for x in d5]}
    if m0 is not None:
        out["d0"] = (float(roc_auc_score(y[m0], z_new[m0])) - float(roc_auc_score(y[m0], z_base[m0]))) * 1e5
    return out


def subcv_delta(y_f0, z_base, delta, subfolds, ws=(0.02, 0.03, 0.04, 0.05)):
    base = [float(roc_auc_score(y_f0[b], z_base[b])) for _, b in subfolds]
    best = None
    for w in ws:
        z = z_base + w * unitize(delta).astype(np.float64)
        d = [(float(roc_auc_score(y_f0[b], z[b])) - base[i]) * 1e5 for i, (_, b) in enumerate(subfolds)]
        row = {"w": w, "mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    return best


def regime_expand(r, r_rare, r_disp, r_bnd):
    r = np.asarray(r, dtype=np.float64).ravel()
    return np.column_stack(
        [
            r,
            r * r_rare,
            r * r_disp,
            r * r_bnd,
            np.maximum(0.0, r - 1.5),
        ]
    ).astype(np.float32)


def fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds, C=0.02, ridge_alpha=5000.0):
    """LOFO newton-cholesky / Ridge on 5 regime features; return OOF score."""
    X = regime_expand(r, r_rare, r_disp, r_bnd)
    oof = np.zeros(len(y), dtype=np.float64)
    for a, b in folds:
        try:
            clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=200, fit_intercept=True)
            clf.fit(X[a], y[a])
            oof[b] = clf.decision_function(X[b])
        except Exception:
            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X[a], y[a].astype(np.float64))
            oof[b] = ridge.predict(X[b])
    return unitize(oof)


def first_existing(*cands):
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(cands)


def build_rayA(y, folds, m0, idx_f0, z_ref, d1):
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    u_sb = unit(d1)
    m1 = unitize(0.65 * u_sb.astype(np.float64) - 0.35 * (-unit(gpt)).astype(np.float64) - 0.20 * gpt.astype(np.float64))
    m2 = unitize(1.0 * u_sb.astype(np.float64))
    s6 = unitize(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    w_s7, dw = 0.055, (0.75e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_ray = z_ref.astype(np.float64) + (w_s7 + dw) * s6.astype(np.float64)
    base0 = float(roc_auc_score(y[m0], z_ref[m0]))
    log(f"  S7_f0 Δ={(float(roc_auc_score(y[m0], z_s7[m0]))-base0)*1e5:+.2f}e-5  RayA Δ={(float(roc_auc_score(y[m0], z_ray[m0]))-base0)*1e5:+.2f}e-5")
    return z_s7[idx_f0], z_ray[idx_f0]


# ---------------------------------------------------------------------------
# Pillar 1 — disk rescue
# ---------------------------------------------------------------------------
def pillar1(y, folds, m0, idx_f0, z_ref, d1, bases_orth, z_base, rayA_f0, r_rare, r_disp, r_bnd):
    log("======== PILLAR 1: Disk asset rescue + PCHIP + regime ========")
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))

    candidates = [
        ("P4_L4.3_ext", PRED / "matrix_v2/P4_L4.3_ext_oof.npy", PRED / "matrix_v2/P4_L4.3_ext_test.npy"),
        ("v10_ridge_resid_orth_LM", PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy", PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("v10_cb_d4_bag3_orth_LM", PRED / "v10_transductive_bulsah/v10_cb_d4_bag3_orth_LM_oof.npy", PRED / "v10_transductive_bulsah/v10_cb_d4_bag3_orth_LM_test.npy"),
        ("v10_lgb_d3_bag5_orth_LM", PRED / "v10_transductive_bulsah/v10_lgb_d3_bag5_orth_LM_oof.npy", PRED / "v10_transductive_bulsah/v10_lgb_d3_bag5_orth_LM_test.npy"),
        ("v10_combo3_orth_LM", PRED / "v10_transductive_bulsah/v10_combo3_orth_LM_oof.npy", PRED / "v10_transductive_bulsah/v10_combo3_orth_LM_test.npy"),
        ("P_super_v5_harvest", PRED / "P_super_v5_harvest_oof.npy", PRED / "P_super_v5_harvest_test.npy"),
        ("z_v5_harvest", PRED / "gm_v5_overnight/z_v5_harvest_oof.npy", PRED / "gm_v5_overnight/z_v5_harvest_test.npy"),
        ("P_super_v5p5_clean", PRED / "P_super_v5p5_clean_oof.npy", PRED / "P_super_v5p5_clean_test.npy"),
        ("paradigm_B_xgb_llr", PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy"),
        ("P4_L4.2_pure5f", PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy", PRED / "matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3_pure5f", PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy", PRED / "matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.5_pure5f", PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy", PRED / "matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("tree_logit_v72", WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy", WORK / "preds/gm_deep_tech_v7/tree_logit_v72_test.npy"),
        ("d1_v5_super", PRED / "gm_v5_overnight/d1_v5_super_oof.npy", PRED / "gm_v5_overnight/d1_v5_super_test.npy"),
        ("own_telgb", PRED / "own_telgb_oof.npy", PRED / "own_telgb_test.npy"),
    ]
    # optional wide_stack
    for root in (PRED, WORK / "preds", Path("/tmp/gm_v5_work/preds")):
        for name in ("wide_stack_nested_oof.npy", "wide_lm_stack_nested.npy"):
            po = root / name
            pt = root / name.replace("oof", "test").replace(".npy", "_test.npy")
            if po.exists():
                candidates.append(("wide_stack_nested", po, pt if pt.exists() else po))

    rows = []
    rescued = {}
    for name, po, pt in candidates:
        if not Path(po).exists():
            log(f"  SKIP missing {name}")
            continue
        try:
            o_raw = load_f32(po)
            if Path(pt).exists() and Path(pt) != Path(po):
                t_raw = load_f32(pt)
            else:
                t_raw = o_raw[:299984] if len(o_raw) > 299984 else o_raw  # fallback unused
            if len(o_raw) != len(y):
                log(f"  SKIP bad shape {name} {o_raw.shape}")
                continue
            # as_z
            if float(o_raw.min()) >= -1e-6 and float(o_raw.max()) <= 1.0 + 1e-6:
                o_z, t_z = zrk(o_raw), zrk(t_raw) if len(t_raw) == 299984 else zrk(o_raw)
            else:
                o_z, t_z = unitize(o_raw), unitize(t_raw) if len(t_raw) == 299984 else unitize(o_raw)
            if len(t_raw) == 299984:
                o_al, t_al = align_method_b_smooth(o_z, t_z, folds)
            else:
                o_al, t_al = unitize(o_z), unitize(o_z)
            disp, ks = disp_ks(o_al, t_al) if len(t_raw) == 299984 else (1.0, 0.0)
            # LOFO orth
            r = lofo_resid(o_al, bases_orth, folds)
            # 1D scalar lift
            best1 = None
            for w in [0.02, 0.03, 0.04, 0.05, 0.06]:
                ev = eval_delta(y, z_base, z_base + w * r.astype(np.float64), folds, m0)
                ev["w"] = w
                if best1 is None or (ev["pos"], ev["d5_mean"]) > (best1["pos"], best1["d5_mean"]):
                    best1 = ev
            # regime 4-param
            r_reg = fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds)
            # orth regime residual vs bases again lightly
            r_reg = lofo_resid(r_reg, bases_orth + [r], folds)
            bestR = None
            for w in [0.02, 0.03, 0.04, 0.05, 0.06]:
                ev = eval_delta(y, z_base, z_base + w * r_reg.astype(np.float64), folds, m0)
                ev["w"] = w
                if bestR is None or (ev["pos"], ev["d5_mean"]) > (bestR["pos"], bestR["d5_mean"]):
                    bestR = ev
            # Gate B Fold0 SubCV vs RayA
            b1 = subcv_delta(y_f0, rayA_f0, r[idx_f0], subfolds)
            bR = subcv_delta(y_f0, rayA_f0, r_reg[idx_f0], subfolds)
            row = {
                "name": name,
                "disp": disp,
                "ks": ks,
                "1d": best1,
                "regime": bestR,
                "f0_1d": b1,
                "f0_regime": bR,
            }
            rows.append(row)
            rescued[name] = {"r": r, "r_reg": r_reg, "t": t_al}
            log(
                f"  {name}: KS={ks:.5f} 1D={best1['d5_mean']:+.2f}e-5@{best1['pos']}/5 "
                f"REG={bestR['d5_mean']:+.2f}e-5@{bestR['pos']}/5 "
                f"F0reg={bR['mean']:+.2f}@{bR['pos']}/5"
            )
            del o_raw, t_raw, o_z, t_z, o_al
            purge()
        except Exception as e:
            log(f"  FAIL {name}: {e}")
    # persist top by regime 5F
    rows_sorted = sorted(rows, key=lambda r: (r["regime"]["pos"], r["regime"]["d5_mean"]), reverse=True)
    for i, row in enumerate(rows_sorted[:5]):
        np.save(ART / f"rescued_{i}_{row['name']}_oof.npy", rescued[row["name"]]["r_reg"])
        if len(rescued[row["name"]]["t"]) == 299984:
            np.save(ART / f"rescued_{i}_{row['name']}_test.npy", rescued[row["name"]]["t"])
    return rows_sorted, rescued


# ---------------------------------------------------------------------------
# Pillar 2 — P4_L4.6 hierarchical EB
# ---------------------------------------------------------------------------
def _as_int_codes(v):
    v = np.asarray(v)
    if v.dtype.kind in "iu":
        return v.astype(np.int64)
    if v.dtype.kind == "f":
        return v.astype(np.int64)
    codes, _ = pd.factorize(pd.Series(v), sort=False)
    return codes.astype(np.int64)


def _factorize_pair(a_vals, b_vals):
    """Joint factorize two columns into single int32 code (no python zip)."""
    a = _as_int_codes(a_vals)
    b = _as_int_codes(b_vals)
    # unique-ish hash then factorize to dense ids
    key = a * 1_000_003 + b + 17
    codes, _ = pd.factorize(key, sort=False)
    return codes.astype(np.int32)


def _te_excess_1d(codes, y, train_idx, val_idx, m=20.0):
    """Smoothed TE log-odds from train_idx → predict val_idx."""
    c_tr = codes[train_idx]
    y_tr = y[train_idx].astype(np.float64)
    mx = int(codes.max()) + 1
    cnt = np.bincount(c_tr, minlength=mx).astype(np.float64)
    sm = np.bincount(c_tr, weights=y_tr, minlength=mx)
    prior = float(y_tr.mean())
    p = (sm + m * prior) / (cnt + m)
    llr = logit(np.clip(p, 1e-4, 1 - 1e-4))
    return llr[np.clip(codes[val_idx], 0, mx - 1)]


def _hier2d_excess(code_ab, code_a, code_b, y, train_idx, val_idx, m=10.0):
    """Δ_hier = sum(y - p_1D) / (n_ab + m) on train; apply to val."""
    # 1D marginals on train
    eta_a = _te_excess_1d(code_a, y, train_idx, train_idx, m=20.0)
    eta_b = _te_excess_1d(code_b, y, train_idx, train_idx, m=20.0)
    # on train rows
    eta0 = float(logit(np.clip(y[train_idx].mean(), 1e-4, 1 - 1e-4)))
    p1d = expit(eta_a + eta_b - eta0)
    resid = y[train_idx].astype(np.float64) - p1d
    cab = code_ab[train_idx]
    mx = int(code_ab.max()) + 1
    cnt = np.bincount(cab, minlength=mx).astype(np.float64)
    sm = np.bincount(cab, weights=resid, minlength=mx)
    delta = sm / (cnt + m)
    return delta[np.clip(code_ab[val_idx], 0, mx - 1)]


def pillar2(y, folds, m0, idx_f0, z_ref, d1, z_base, rayA_f0, r_rare, r_disp, r_bnd, bases_orth):
    log("======== PILLAR 2: P4_L4.6 Hierarchical EB 5F ========")
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))
    ntr, nte = len(y), 299984

    usecols = [
        "amount_usd",
        "hours_since_last_txn",
        "distance_from_home_km",
        "account_balance_usd",
        "velocity_score",
        "merchant_risk_score",
        "merchant_category",
        "card_type",
        "channel",
        "device_type",
        "auth_method",
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    log("  loading train/test feature cols …")
    tr = pd.read_csv(VOL / "data/train.csv", usecols=usecols)
    te = pd.read_csv(VOL / "data/test.csv", usecols=usecols)
    full = pd.concat([tr, te], ignore_index=True)
    del tr, te
    purge()

    FLAGS = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    flag_mask = np.zeros(len(full), dtype=np.int32)
    for i, c in enumerate(FLAGS):
        flag_mask |= (full[c].astype(np.int32).to_numpy() & 1) << i

    def qbin(x, q):
        # quantile bins on full to keep train/test aligned
        qs = np.quantile(x, np.linspace(0, 1, q + 1))
        qs = np.unique(qs)
        return np.digitize(x, qs[1:-1], right=True).astype(np.int32)

    def step10(x):
        return np.round(x * 10).astype(np.int32)

    vel = full["velocity_score"].to_numpy(np.float64)
    risk = full["merchant_risk_score"].to_numpy(np.float64)
    hrs = full["hours_since_last_txn"].to_numpy(np.float64)
    dist = full["distance_from_home_km"].to_numpy(np.float64)
    amt = full["amount_usd"].to_numpy(np.float64)
    bal = full["account_balance_usd"].to_numpy(np.float64)
    cat = full["merchant_category"].astype(str).to_numpy()
    card = full["card_type"].astype(str).to_numpy()
    chan = full["channel"].astype(str).to_numpy()

    # --- Sub-block 2A keys (lean ~18) ---
    keys_A = []
    labels_A = []
    for name, arr, qb in [
        ("vel_q100", vel, 100),
        ("vel_q50", vel, 50),
        ("risk_q100", risk, 100),
        ("hrs_q50", hrs, 50),
        ("dist_q50", dist, 50),
        ("vel_s10", step10(vel), None),
        ("risk_s10", step10(risk), None),
        ("hrs_s10", step10(hrs), None),
    ]:
        if qb is not None:
            b = qbin(arr, qb)
        else:
            b = arr.astype(np.int32)
        # cross flag_mask
        codes = _factorize_pair(b, flag_mask)
        keys_A.append(codes)
        labels_A.append(f"{name}_x_flag")
        # cross channel
        codes2 = _factorize_pair(b, chan)
        keys_A.append(codes2)
        labels_A.append(f"{name}_x_chan")
        if len(keys_A) >= 18:
            break
    log(f"  BlockA keys={len(keys_A)}")

    # --- Sub-block 2B bulsah (~16) ---
    keys_B = []
    labels_B = []

    def cents(x):
        return (np.round(np.abs(x) * 100).astype(np.int64) % 100).astype(np.int32)

    def lead2(x):
        s = pd.Series(np.floor(np.abs(x)).astype(np.int64).astype(str))
        return (s.str[:2].fillna("0")).values

    for name, arr in [("amt", amt), ("bal", bal), ("risk", risk), ("vel", vel)]:
        c = cents(arr)
        keys_B.append(_factorize_pair(c, cat))
        labels_B.append(f"{name}_cents_x_cat")
        keys_B.append(_factorize_pair(c, flag_mask))
        labels_B.append(f"{name}_cents_x_flag")
        keys_B.append(_factorize_pair(c, card))
        labels_B.append(f"{name}_cents_x_card")
        if len(keys_B) >= 16:
            break
    # pairwise cents amt x vel
    keys_B.append(_factorize_pair(cents(amt), cents(vel)))
    labels_B.append("amt_cents_x_vel_cents")
    log(f"  BlockB keys={len(keys_B)}")

    # --- Sub-block 2C top pairs (~20) ---
    # build candidate 1D codes
    cat_c, _ = pd.factorize(pd.Series(cat))
    card_c, _ = pd.factorize(pd.Series(card))
    chan_c, _ = pd.factorize(pd.Series(chan))
    vel_c = qbin(vel, 50)
    risk_c = qbin(risk, 50)
    amt_c = qbin(amt, 50)
    hrs_c = qbin(hrs, 50)
    dist_c = qbin(dist, 50)
    bal_c = qbin(bal, 50)
    singles = {
        "cat": cat_c.astype(np.int32),
        "card": card_c.astype(np.int32),
        "chan": chan_c.astype(np.int32),
        "flag": flag_mask,
        "vel": vel_c,
        "risk": risk_c,
        "amt": amt_c,
        "hrs": hrs_c,
        "dist": dist_c,
        "bal": bal_c,
    }
    pair_names = [
        ("cat", "vel"),
        ("cat", "risk"),
        ("cat", "amt"),
        ("cat", "flag"),
        ("card", "vel"),
        ("card", "risk"),
        ("card", "amt"),
        ("chan", "vel"),
        ("chan", "risk"),
        ("chan", "flag"),
        ("vel", "risk"),
        ("vel", "hrs"),
        ("vel", "dist"),
        ("risk", "amt"),
        ("risk", "hrs"),
        ("amt", "bal"),
        ("amt", "flag"),
        ("hrs", "dist"),
        ("flag", "vel"),
        ("flag", "risk"),
    ]
    keys_C = []
    singles_C = []  # (code_ab, code_a, code_b)
    for a, b in pair_names:
        cab = _factorize_pair(singles[a], singles[b])
        keys_C.append(cab)
        singles_C.append((cab, singles[a], singles[b]))
    log(f"  BlockC pairs={len(keys_C)}")

    del full, cat, card, chan, vel, risk, hrs, dist, amt, bal
    purge()

    def filter_support(codes, ntr, nte, min_tr=5, min_te=2):
        """Return mask of valid codes based on train/test counts."""
        mx = int(codes.max()) + 1
        ctr = np.bincount(codes[:ntr], minlength=mx)
        cte = np.bincount(codes[ntr:], minlength=mx)
        valid = (ctr >= min_tr) & (cte >= min_te)
        # remap invalid to 0 bucket
        out = codes.copy()
        out[~valid[codes]] = 0
        return out

    keys_A = [filter_support(c, ntr, nte) for c in keys_A]
    keys_B = [filter_support(c, ntr, nte) for c in keys_B]
    keys_C = [filter_support(c, ntr, nte) for c in keys_C]
    singles_C = [(filter_support(cab, ntr, nte), ca, cb) for (cab, ca, cb) in singles_C]

    def condense_block(block_keys, kind="te"):
        """5-fold OOF condensation of a block via 8-seed inner TE → Ridge/newton-cholesky."""
        oof = np.zeros(ntr, dtype=np.float64)
        te_acc = np.zeros(nte, dtype=np.float64)
        for fi, (a, b) in enumerate(folds):
            # build feature matrix on a∪b∪test via multi-seed inner TE
            feats_tr = []  # for all ntr? only need a for fit, b for oof — build on a+b rows
            feats_te = []
            # For speed: average 8 seeds of stacked key TEs (mean LLR across keys & seeds)
            # Then expand to per-key means → Ridge
            X_ab = np.zeros((len(a) + len(b), len(block_keys)), dtype=np.float32)
            X_te = np.zeros((nte, len(block_keys)), dtype=np.float32)
            idx_ab = np.concatenate([a, b])
            for ki, codes in enumerate(block_keys):
                acc_ab = np.zeros(len(idx_ab), dtype=np.float64)
                acc_te = np.zeros(nte, dtype=np.float64)
                for seed in INNER_SEEDS:
                    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
                    te_a = np.zeros(len(a), dtype=np.float64)
                    for ia, ib in inner.split(np.zeros(len(a)), y[a]):
                        tr_i = a[ia]
                        if kind == "hier":
                            cab, ca, cb = singles_C[ki]
                            d10 = _hier2d_excess(cab, ca, cb, y, tr_i, a[ib], m=10.0)
                            d50 = _hier2d_excess(cab, ca, cb, y, tr_i, a[ib], m=50.0)
                            te_a[ib] = 0.5 * (d10 + d50)
                        else:
                            te_a[ib] = _te_excess_1d(codes, y, tr_i, a[ib], m=20.0)
                    if kind == "hier":
                        cab, ca, cb = singles_C[ki]
                        pred_b = 0.5 * (
                            _hier2d_excess(cab, ca, cb, y, a, b, m=10.0)
                            + _hier2d_excess(cab, ca, cb, y, a, b, m=50.0)
                        )
                        eta_a = _te_excess_1d(ca, y, a, a, m=20.0)
                        eta_b = _te_excess_1d(cb, y, a, a, m=20.0)
                        eta0 = float(logit(np.clip(y[a].mean(), 1e-4, 1 - 1e-4)))
                        p1d = expit(eta_a + eta_b - eta0)
                        resid = y[a].astype(np.float64) - p1d
                        mx = int(cab.max()) + 1
                        cnt = np.bincount(cab[a], minlength=mx).astype(np.float64)
                        sm = np.bincount(cab[a], weights=resid, minlength=mx)
                        dlt = 0.5 * (sm / (cnt + 10.0) + sm / (cnt + 50.0))
                        pred_te = dlt[np.clip(cab[ntr:], 0, mx - 1)]
                    else:
                        mx = int(codes.max()) + 1
                        cnt = np.bincount(codes[a], minlength=mx).astype(np.float64)
                        sm = np.bincount(codes[a], weights=y[a].astype(np.float64), minlength=mx)
                        prior = float(y[a].mean())
                        p = (sm + 20.0 * prior) / (cnt + 20.0)
                        llr = logit(np.clip(p, 1e-4, 1 - 1e-4))
                        pred_b = llr[np.clip(codes[b], 0, mx - 1)]
                        pred_te = llr[np.clip(codes[ntr:], 0, mx - 1)]
                    acc_ab[: len(a)] += te_a
                    acc_ab[len(a) :] += pred_b
                    acc_te += pred_te
                acc_ab /= len(INNER_SEEDS)
                acc_te /= len(INNER_SEEDS)
                X_ab[:, ki] = acc_ab.astype(np.float32)
                X_te[:, ki] = acc_te.astype(np.float32)
            # condense with Ridge + newton-cholesky on outer-train a
            # Stage-1 offset: z_ref on a
            Xa = X_ab[: len(a)]
            Xb = X_ab[len(a) :]
            # add z_ref as offset col
            Xa2 = np.column_stack([Xa, z_ref[a].astype(np.float32)])
            Xb2 = np.column_stack([Xb, z_ref[b].astype(np.float32)])
            Xt2 = np.column_stack([X_te, np.zeros(nte, dtype=np.float32)])  # no z_ref test in OOF path; use 0 and rely on TE
            # better: don't include z_ref in features — condense pure block then orth later
            ridge = Ridge(alpha=1500.0)
            ridge.fit(Xa, y[a].astype(np.float64))
            pred_b = ridge.predict(Xb)
            pred_te = ridge.predict(X_te)
            try:
                clf = LogisticRegression(C=0.04, solver="newton-cholesky", max_iter=200)
                clf.fit(Xa, y[a])
                pred_b = 0.5 * (pred_b + clf.decision_function(Xb))
                pred_te = 0.5 * (pred_te + clf.decision_function(X_te))
            except Exception:
                pass
            oof[b] = pred_b
            te_acc += pred_te / 5.0
            log(f"    block fold{fi} done RSS={rss_curr():.0f}MB")
            del X_ab, X_te, Xa, Xb, Xa2, Xb2, Xt2
            purge()
        return unitize(zrk(oof) if oof.min() >= 0 and oof.max() <= 1.5 else oof), unitize(
            zrk(te_acc) if te_acc.min() >= 0 and te_acc.max() <= 1.5 else te_acc
        )

    # For time budget: use 4 seeds if elapsed already high — keep 8 as requested but blocks are lean
    log("  condensing Block A (stepgrid) …")
    A_o, A_t = condense_block(keys_A, kind="te")
    A_o, A_t = align_method_b_smooth(A_o, A_t, folds)
    dA, kA = disp_ks(A_o, A_t)
    log(f"  L4.6A disp={dA:.4f} KS={kA:.6f}")

    log("  condensing Block B (bulsah) …")
    B_o, B_t = condense_block(keys_B, kind="te")
    B_o, B_t = align_method_b_smooth(B_o, B_t, folds)
    dB, kB = disp_ks(B_o, B_t)
    log(f"  L4.6B disp={dB:.4f} KS={kB:.6f}")

    log("  condensing Block C (hier2d) …")
    # temporarily point singles_C for hier path
    C_o, C_t = condense_block(keys_C, kind="hier")
    C_o, C_t = align_method_b_smooth(C_o, C_t, folds)
    dC, kC = disp_ks(C_o, C_t)
    log(f"  L4.6C disp={dC:.4f} KS={kC:.6f}")

    np.save(ART / "P4_L4.6A_stepgrid_5f_oof.npy", A_o)
    np.save(ART / "P4_L4.6A_stepgrid_5f_test.npy", A_t)
    np.save(ART / "P4_L4.6B_bulsah_5f_oof.npy", B_o)
    np.save(ART / "P4_L4.6B_bulsah_5f_test.npy", B_t)
    np.save(ART / "P4_L4.6C_hier2d_5f_oof.npy", C_o)
    np.save(ART / "P4_L4.6C_hier2d_5f_test.npy", C_t)

    results = []
    for name, o, t, d, k in [
        ("P4_L4.6A_stepgrid", A_o, A_t, dA, kA),
        ("P4_L4.6B_bulsah", B_o, B_t, dB, kB),
        ("P4_L4.6C_hier2d", C_o, C_t, dC, kC),
    ]:
        r = lofo_resid(o, bases_orth, folds)
        best1 = None
        for w in [0.02, 0.03, 0.04, 0.05, 0.06]:
            ev = eval_delta(y, z_base, z_base + w * r.astype(np.float64), folds, m0)
            ev["w"] = w
            if best1 is None or (ev["pos"], ev["d5_mean"]) > (best1["pos"], best1["d5_mean"]):
                best1 = ev
        r_reg = fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds)
        r_reg = lofo_resid(r_reg, bases_orth + [r], folds)
        bestR = None
        for w in [0.02, 0.03, 0.04, 0.05, 0.06]:
            ev = eval_delta(y, z_base, z_base + w * r_reg.astype(np.float64), folds, m0)
            ev["w"] = w
            if bestR is None or (ev["pos"], ev["d5_mean"]) > (bestR["pos"], bestR["d5_mean"]):
                bestR = ev
        bR = subcv_delta(y_f0, rayA_f0, r_reg[idx_f0], subfolds)
        b1 = subcv_delta(y_f0, rayA_f0, r[idx_f0], subfolds)
        results.append(
            {"name": name, "disp": d, "ks": k, "1d": best1, "regime": bestR, "f0_1d": b1, "f0_regime": bR, "r": r, "r_reg": r_reg}
        )
        log(
            f"  {name}: 1D={best1['d5_mean']:+.2f}@{best1['pos']}/5 REG={bestR['d5_mean']:+.2f}@{bestR['pos']}/5 "
            f"F0={bR['mean']:+.2f}@{bR['pos']}/5"
        )
    del keys_A, keys_B, keys_C, singles_C, singles
    purge()
    return results


# ---------------------------------------------------------------------------
# Pillar 3 — regime fusion across innovations
# ---------------------------------------------------------------------------
def pillar3(y, folds, m0, idx_f0, z_ref, d1, z_base, rayA_f0, r_rare, r_disp, r_bnd, bases_orth, p1_rows, p1_rescued, p2_results):
    log("======== PILLAR 3: newton-cholesky regime cascade ========")
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))

    innov = {}
    innov["delta1_v4"] = unitize(d1)
    innov["tree_logit_v7"] = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"):
        if tag in p1_rescued:
            innov[tag] = p1_rescued[tag]["r"]
        else:
            po = PRED / "matrix_v2" / f"{tag}_oof.npy"
            if po.exists():
                innov[tag] = lofo_resid(unitize(zrk(load_f32(po))), bases_orth, folds)
    for row in p2_results:
        innov[row["name"]] = row["r"]
    # top rescued
    for row in p1_rows[:4]:
        innov[f"rescued_{row['name']}"] = p1_rescued[row["name"]]["r"]
    # LM orths
    mega = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"))
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    innov["megayak_orth"] = lofo_resid(mega, bases_orth, folds)
    innov["gpt2m_5f_orth"] = lofo_resid(gpt, bases_orth, folds)
    # L7 f0 orth — place into full length
    l7p = ART.parent / "ds_first_principles/L7_EB_10hat_oof_f0.npy"
    if not l7p.exists():
        l7p = WORK / "preds/ds_first_principles/L7_EB_10hat_oof_f0.npy"
    if l7p.exists():
        l7 = np.zeros(len(y), dtype=np.float32)
        l7[idx_f0] = load_f32(l7p)
        innov["L7_f0_orth"] = lofo_resid(l7, bases_orth, folds)

    rows = []
    for name, r in innov.items():
        r_reg = fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds)
        r_reg = lofo_resid(r_reg, [z_ref, d1], folds)  # light orth to base
        best = None
        for w in [0.02, 0.03, 0.04, 0.05, 0.06]:
            ev = eval_delta(y, z_base, z_base + w * r_reg.astype(np.float64), folds, m0)
            ev["w"] = w
            if best is None or (ev["pos"], ev["d5_mean"]) > (best["pos"], best["d5_mean"]):
                best = ev
        bR = subcv_delta(y_f0, rayA_f0, r_reg[idx_f0], subfolds)
        rows.append({"name": name, "regime": best, "f0_regime": bR})
        log(f"  {name}: REG={best['d5_mean']:+.2f}e-5@{best['pos']}/5 F0={bR['mean']:+.2f}@{bR['pos']}/5")
        purge()
    rows.sort(key=lambda r: (r["regime"]["pos"], r["regime"]["d5_mean"]), reverse=True)
    return rows


def write_report(p1, p2, p3):
    md = []
    md.append("# GM Architect Nextgen Cascade v9")
    md.append("")
    md.append(f"**NO_SUBMIT=True**. Elapsed `{time.time()-T0:.1f}s`. Peak RSS ≈ `{rss_max():.0f}` MB (soft 600).")
    md.append("")
    md.append("## Pillar 1 — Rescued disk assets (PCHIP + LOFO orth + regime)")
    md.append("")
    md.append("| asset | disp | KS | 1D Δ5F | 1D pos | REG Δ5F | REG pos | F0 REG vs RayA | F0 pos |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in p1:
        md.append(
            f"| {r['name']} | {r['disp']:.4f} | {r['ks']:.6f} | {r['1d']['d5_mean']:+.2f} | {r['1d']['pos']}/5 | "
            f"{r['regime']['d5_mean']:+.2f} | {r['regime']['pos']}/5 | {r['f0_regime']['mean']:+.2f} | {r['f0_regime']['pos']}/5 |"
        )
    md.append("")
    md.append("## Pillar 2 — P4_L4.6 Hierarchical EB")
    md.append("")
    md.append("| block | disp | KS | 1D Δ5F | 1D pos | REG Δ5F | REG pos | F0 REG vs RayA | F0 pos |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in p2:
        md.append(
            f"| {r['name']} | {r['disp']:.4f} | {r['ks']:.6f} | {r['1d']['d5_mean']:+.2f} | {r['1d']['pos']}/5 | "
            f"{r['regime']['d5_mean']:+.2f} | {r['regime']['pos']}/5 | {r['f0_regime']['mean']:+.2f} | {r['f0_regime']['pos']}/5 |"
        )
    md.append("")
    md.append("## Pillar 3 — Newton-Cholesky regime fusion")
    md.append("")
    md.append("| innovation | REG Δ5F | REG pos | F0 REG vs RayA | F0 pos |")
    md.append("|---|---:|---:|---:|---:|")
    for r in p3:
        md.append(
            f"| {r['name']} | {r['regime']['d5_mean']:+.2f} | {r['regime']['pos']}/5 | "
            f"{r['f0_regime']['mean']:+.2f} | {r['f0_regime']['pos']}/5 |"
        )
    md.append("")
    md.append("**NO_SUBMIT=True** — research only; no CSVs staged. Playbook-3 ensemble residuals not used.")
    text = "\n".join(md) + "\n"
    rp = LOG / "GM_ARCHITECT_V9_REPORT.md"
    rp.write_text(text)
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / "GM_ARCHITECT_V9_REPORT.md").write_text(text)
    except Exception as e:
        log(f"vol sync warn: {e}")
    log(f"REPORT → {rp}")
    return rp


def main():
    log("======== GM ARCHITECT V9 CASCADE (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"])["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    idx_f0 = np.where(m0)[0]
    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    d1 = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    w12 = unitize(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_oof.npy"))
    mega = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"))
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    bases_orth = [z_ref, d1, w12, mega, gpt]
    z_base = z_ref.astype(np.float64) + 0.04 * d1.astype(np.float64)
    log(f"z_base AUC={roc_auc_score(y, z_base):.7f}")
    s7_f0, rayA_f0 = build_rayA(y, folds, m0, idx_f0, z_ref, d1)

    # regime indicators from X_v5 extras + disagreement
    meta = json.loads((PRED / "gm_v5_overnight/X_v5_meta.json").read_text())
    extra = meta.get("extra_names", [])
    base_extra = 68 + int(meta.get("n_kmat5seed", 15))
    Xo = np.load(PRED / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
    if "z_lm_x_rare" in extra:
        r_rare = unitize(np.asarray(Xo[:, base_extra + extra.index("z_lm_x_rare")], dtype=np.float32))
    else:
        r_rare = unitize(np.asarray(Xo[:, 0], dtype=np.float32))
    if "z_lm_x_disp" in extra:
        r_disp = unitize(np.asarray(Xo[:, base_extra + extra.index("z_lm_x_disp")], dtype=np.float32))
    else:
        r_disp = unitize(np.abs(gpt.astype(np.float64) - unitize(d1).astype(np.float64)))
    del Xo
    purge()
    # also disagreement LM vs tree
    tree = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    r_disp = unitize(0.5 * r_disp.astype(np.float64) + 0.5 * np.abs(gpt.astype(np.float64) - tree.astype(np.float64)))
    p_base = expit(z_ref.astype(np.float64))
    r_bnd = unitize(np.sqrt(np.clip(p_base * (1 - p_base), 1e-8, None)))
    # standardize regimes
    r_rare, r_disp, r_bnd = unitize(r_rare), unitize(r_disp), unitize(r_bnd)
    log(f"regimes ready RSS={rss_curr():.0f}MB")

    p1_rows, p1_rescued = pillar1(y, folds, m0, idx_f0, z_ref, d1, bases_orth, z_base, rayA_f0, r_rare, r_disp, r_bnd)
    p2_results = pillar2(y, folds, m0, idx_f0, z_ref, d1, z_base, rayA_f0, r_rare, r_disp, r_bnd, bases_orth)
    p3_rows = pillar3(y, folds, m0, idx_f0, z_ref, d1, z_base, rayA_f0, r_rare, r_disp, r_bnd, bases_orth, p1_rows, p1_rescued, p2_results)

    # strip heavy arrays before json
    p2_lite = [{k: v for k, v in r.items() if k not in ("r", "r_reg")} for r in p2_results]
    (ART / "v9_summary.json").write_text(
        json.dumps({"p1": p1_rows, "p2": p2_lite, "p3": p3_rows, "rss_max": rss_max(), "elapsed": time.time() - T0}, indent=2, default=float)
    )
    write_report(p1_rows, p2_lite, p3_rows)
    try:
        (VOL / "scripts").mkdir(parents=True, exist_ok=True)
        src = Path(__file__).resolve()
        (VOL / "scripts" / "gm_architect_nextgen_cascade_v9.py").write_text(src.read_text())
    except Exception as e:
        log(f"script sync warn: {e}")
    log("======== V9 DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
