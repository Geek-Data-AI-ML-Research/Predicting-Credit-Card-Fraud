#!/usr/bin/env python3
"""GM Max-Jump Local Research Suite v3 — STRICT NO_SUBMIT.

Four investigations to maximize jump above LB 0.95335.
Artifacts → preds/gm_v3_research/ ; report → logs/overnight/GM_MAX_JUMP_RESEARCH_V3.md
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
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from gm_nullspace_strike_orth_v3 import (
    align_method_b_smooth,
    cos,
    disp_ks,
    orth_against,
    unit,
)

T0 = time.time()
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
MV2 = PRED / "matrix_v2"
ART2 = PRED / "gm_v2_ladder"
ART = PRED / "gm_v3_research"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
P12 = ROOT / "external/nov08_part12"
ART.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0") == "1"
EPS = 1e-6
W_SUPER = 0.075


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


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    za = np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1)
    ya = np.asarray(y_a, dtype=np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, dtype=np.float64).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def qmap_to_oof(x_test, x_oof):
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    o = np.sort(np.asarray(x_oof, dtype=np.float64).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def rk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)


def orth_vs(zo, zt, bases_o, bases_t, folds):
    Xo = np.column_stack(bases_o)
    Xt = np.column_stack(bases_t)
    ro = np.zeros(len(zo))
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


def coords_in_basis(delta, basis_cols):
    """Least-squares coords of delta in columns of basis_cols."""
    B = np.column_stack([unitize(c) for c in basis_cols])
    # also try raw scale LS
    coef, *_ = np.linalg.lstsq(np.column_stack(basis_cols), delta, rcond=None)
    recon = np.column_stack(basis_cols) @ coef
    resid_n = float(np.linalg.norm(delta - recon) / (np.linalg.norm(delta) + 1e-12))
    return coef, resid_n


def build_kmat_te(K_tr, K_te, y, folds, n_keys):
    ntr = len(y)
    K_oof = np.zeros((ntr, n_keys), dtype=np.float32)
    K_te_folds = []
    for a, b in folds:
        prior = float(y[a].mean())
        te_b, te_te = [], []
        for j in range(n_keys):
            ka = K_tr[a, j]
            if np.issubdtype(ka.dtype, np.floating) or len(np.unique(ka)) > 500:
                edges = np.unique(np.quantile(ka, np.linspace(0, 1, 32)))
                ba = np.digitize(ka, edges)
                bb = np.digitize(K_tr[b, j], edges)
                bt = np.digitize(K_te[:, j], edges)
            else:
                ba, bb, bt = ka, K_tr[b, j], K_te[:, j]
            tmp = pd.DataFrame({"k": ba, "y": y[a]})
            gstat = tmp.groupby("k")["y"].agg(["sum", "count"])
            m = 20.0
            te_map = ((gstat["sum"] + prior * m) / (gstat["count"] + m)).to_dict()
            te_b.append(np.array([te_map.get(k, prior) for k in bb], dtype=np.float32))
            te_te.append(np.array([te_map.get(k, prior) for k in bt], dtype=np.float32))
        K_oof[b] = np.column_stack(te_b)
        K_te_folds.append(np.column_stack(te_te))
    return K_oof, np.mean(K_te_folds, 0).astype(np.float32)


def build_rect_basis(specs, z_base_o, z_base_t):
    R_o, R_t, metas = [], [], []
    for name, xo, xt in specs:
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
        ro, rt = ro / s, rt / s
        metas.append({"name": name, "beta": beta, "ks": float(ks_2samp(ro, rt).statistic)})
        R_o.append(ro)
        R_t.append(rt)
    return np.column_stack(R_o), np.column_stack(R_t), metas


def nested_coord_ascent(y, z_base, R_o, folds, names):
    M = len(names)
    grid = (-0.04, -0.03, -0.022, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.022, 0.03, 0.04)
    oof_z = np.zeros(len(y))
    fold_w = []
    for fi, (a, b) in enumerate(folds):
        w = np.zeros(M)
        best = float(roc_auc_score(y[a], z_base[a]))
        for _ in range(3):
            for j in range(M):
                bw = w[j]
                for cw in grid:
                    wt = w.copy()
                    wt[j] = cw
                    sc = float(roc_auc_score(y[a], z_base[a] + R_o[a] @ wt))
                    if sc > best + 1e-9:
                        best, bw = sc, cw
                w[j] = bw
        oof_z[b] = z_base[b] + R_o[b] @ w
        fold_w.append(w)
    return oof_z, np.mean(fold_w, 0), fold_w


# ============================================================================
# Investigation 1
# ============================================================================
def inv1(y, test_ids, folds, z_v8_old_o, report):
    log("=== INV1: coordinate audit + δ2 surgical split ===")
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
    Z = {}
    for name, fn, lb in LB:
        p = SUB / fn
        if not p.exists():
            log(f"  MISSING {fn}")
            continue
        Z[name] = {"z": zrk(load_sub(fn, test_ids)), "p": load_sub(fn, test_ids), "lb": lb, "fn": fn}
        log(f"  loaded {name} LB={lb} file={fn}")

    z95 = Z["St9.5"]["z"]
    z96 = Z["Today_S1"]["z"]
    d_ladder = z96 - z95
    # Prefer saved Method-B deltas if present; else reconstruct from clean artifacts
    if (ART2 / "d1_test.npy").exists():
        d1 = np.load(ART2 / "d1_test.npy").ravel()
        d2 = np.load(ART2 / "d2_test.npy").ravel()
        d3 = np.load(ART2 / "d3_test.npy").ravel()
    else:
        d1 = unitize(zrk(np.load(ART2 / "z_v8_clean_test.npy").ravel()) - zrk(Z["v8"]["z"]))
        d2 = unitize(zrk(np.load(ART2 / "z_ref_clean_test.npy").ravel()) - zrk(Z["v8"]["z"]))
        d3 = unitize(zrk(np.load(ART2 / "z_lm12_raw_super_test.npy").ravel()) - zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel()))

    basis = [d_ladder, d1, d2, d3]
    bnames = ["c_ladder", "c_d1", "c_d2", "c_d3"]
    coef_table = {}
    for key in ["Today_S3", "Today_S4", "Today_S5", "Today_S1", "Today_S2"]:
        if key not in Z:
            continue
        delta = Z[key]["z"] - z95
        coef, resid = coords_in_basis(delta, basis)
        coef_table[key] = {
            "lb": Z[key]["lb"],
            **{bnames[i]: float(coef[i]) for i in range(4)},
            "resid_rel": float(resid),
            "cos_d2": float(cos(delta, d2)),
            "energy_d2": float(abs(coef[2]) * np.linalg.norm(d2) / (np.linalg.norm(delta) + 1e-12)),
        }
        log(
            f"  {key} LB={Z[key]['lb']}: "
            + ", ".join(f"{bnames[i]}={coef[i]:+.4f}" for i in range(4))
            + f" resid={resid:.3f} cos(δ2)={coef_table[key]['cos_d2']:+.3f}"
        )

    # Empirical 19-pt LB gradient in span of step axes
    # Fit Ridge: LB ~ coords of (z - z_v8) in [e1, d_ladder, d1, d2, d3]
    e1 = Z["St4A"]["z"] - Z["St3A"]["z"]
    e2 = (Z["St5A"]["z"] - Z["St4A"]["z"]) - 0.45 * (Z["St4A"]["z"] - Z["St3A"]["z"])
    e3 = e2.copy()
    g_axes = [e1, d_ladder, d1, d2, d3]
    Xg, yg = [], []
    for name, meta in Z.items():
        c, *_ = np.linalg.lstsq(np.column_stack(g_axes), meta["z"] - Z["v8"]["z"], rcond=None)
        Xg.append(c)
        yg.append(meta["lb"])
    Xg, yg = np.asarray(Xg), np.asarray(yg)
    ridge_g = Ridge(alpha=1e-2, fit_intercept=True).fit(Xg, yg)
    g_lb = ridge_g.coef_  # gradient in axis coords
    g_vec = np.column_stack(g_axes) @ g_lb
    log(f"  empirical LB ridge coefs [e1,ladder,d1,d2,d3]={np.round(g_lb,4).tolist()}")

    # Reconstruct δ2 components on TEST (aligned)
    log("  rebuilding δ2 subcomponents…")
    z_v8_old_t = Z["v8"]["z"]
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
    specs5 = [
        ("U5_1_nade", PRED / "part11/lab4_tabular_nade_oof.npy", PRED / "part11/lab4_tabular_nade_test.npy"),
        ("U5_2_sameer", None, None),
        ("U5_3_knn", HARV / "oof_orig20k_knn.npy", HARV / "test_orig20k_knn.npy"),
        ("U5_4_p4xgb", MV2 / "harvest_p4_cuda_xgb_oof.npy", MV2 / "harvest_p4_cuda_xgb_test.npy"),
        ("U5_5_zjoint", PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy", PRED / "part11/lab1_zjoint_graph_lgb4_test.npy"),
    ]
    U5_o, U5_t, U5_names = [], [], []
    for i, (nm, op, tp) in enumerate(specs5):
        if nm == "U5_2_sameer":
            xo, xt = sam_o, sam_t
        else:
            xo, xt = np.load(op).ravel(), np.load(tp).ravel()
        zo, zt = zrk(xo), zrk(xt)
        ro, rt = orth_vs(zo, zt, [X_orth_o[:, j] for j in range(3)], [X_orth_t[:, j] for j in range(3)], folds)
        ro, rt = align_method_b_smooth(ro, rt, folds)
        U5_o.append(ro)
        U5_t.append(rt)
        U5_names.append(nm)
    U5_o, U5_t = np.column_stack(U5_o), np.column_stack(U5_t)
    w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
    X7_o = np.column_stack([X_orth_o, U5_o])
    X7_t = np.column_stack([X_orth_t, U5_t])
    specs2 = [
        ("U2_1_glm_quad", PRED / "ladder/glm_quad_oof.npy", PRED / "ladder/glm_quad_test.npy"),
        ("U2_2_glm_graph", PRED / "part11/lab1_glm_plus_graph_oof.npy", PRED / "part11/lab1_glm_plus_graph_test.npy"),
    ]
    U2_o, U2_t, U2_names = [], [], []
    for nm, op, tp in specs2:
        zo, zt = zrk(np.load(op).ravel()), zrk(np.load(tp).ravel())
        ro, rt = orth_vs(zo, zt, [X7_o[:, j] for j in range(X7_o.shape[1])], [X7_t[:, j] for j in range(X7_t.shape[1])], folds)
        ro, rt = align_method_b_smooth(ro, rt, folds)
        U2_o.append(ro)
        U2_t.append(rt)
        U2_names.append(nm)
    U2_o, U2_t = np.column_stack(U2_o), np.column_stack(U2_t)
    w2 = np.array([0.0150, 0.0125])

    comps = {
        "u_cv": (0.025 * u_cv_t, 0.025 * u_cv_o),
    }
    for i, nm in enumerate(U5_names):
        comps[nm] = (0.85 * w5[i] * U5_t[:, i], 0.85 * w5[i] * U5_o[:, i])
    for i, nm in enumerate(U2_names):
        comps[nm] = (0.85 * w2[i] * U2_t[:, i], 0.85 * w2[i] * U2_o[:, i])

    # Also include v8_clean-v8_old as δ1_in_δ2
    d1_in = z_v8_clean_t - z_v8_old_t
    comps["d1_in_ref"] = (d1_in, z_v8_clean_o - z_v8_old_o)

    cos_rows = []
    good_t = np.zeros(len(test_ids))
    bad_t = np.zeros(len(test_ids))
    good_names, bad_names = [], []
    for nm, (ct, co) in comps.items():
        row = {
            "name": nm,
            "cos_ladder": cos(ct, d_ladder),
            "cos_e1": cos(ct, e1),
            "cos_e2": cos(ct, e2),
            "cos_g_lb": cos(ct, g_vec),
            "norm": float(np.linalg.norm(ct)),
        }
        # good if aligned with +e1 or +g_lb; bad if aligned with e2 or strongly against g_lb
        score = row["cos_e1"] + 0.5 * row["cos_g_lb"] - 0.75 * abs(min(0.0, row["cos_e2"])) - 0.5 * abs(min(0.0, row["cos_g_lb"]))
        # tree/knn heuristic
        if nm in ("U5_3_knn", "U5_4_p4xgb", "u_cv") or row["cos_e2"] > 0.05 or row["cos_g_lb"] < -0.02:
            bad_t = bad_t + ct
            bad_names.append(nm)
            row["bucket"] = "bad"
        elif nm in ("U2_1_glm_quad", "U2_2_glm_graph", "U5_1_nade") or row["cos_e1"] > 0.02 or row["cos_g_lb"] > 0.02:
            good_t = good_t + ct
            good_names.append(nm)
            row["bucket"] = "good"
        else:
            # default by score
            if score >= 0:
                good_t = good_t + ct
                good_names.append(nm)
                row["bucket"] = "good"
            else:
                bad_t = bad_t + ct
                bad_names.append(nm)
                row["bucket"] = "bad"
        cos_rows.append(row)
        log(f"  {nm}: cos(e1)={row['cos_e1']:+.3f} cos(e2)={row['cos_e2']:+.3f} cos(g)={row['cos_g_lb']:+.3f} → {row['bucket']}")

    d2_good = unitize(good_t) if np.linalg.norm(good_t) > 1e-12 else good_t
    d2_bad = unitize(bad_t) if np.linalg.norm(bad_t) > 1e-12 else bad_t
    np.save(ART / "delta2_good_test.npy", d2_good)
    np.save(ART / "delta2_bad_test.npy", d2_bad)
    np.save(ART / "d_ladder_test.npy", d_ladder)
    np.save(ART / "d1_test.npy", d1)
    np.save(ART / "d2_test.npy", d2)
    np.save(ART / "d3_test.npy", d3)

    # How much +c_d2 in Slot5?
    s5 = coef_table.get("Today_S5", {})
    poison = {
        "Today_S5_c_d2": s5.get("c_d2"),
        "Today_S5_cos_d2": s5.get("cos_d2"),
        "Today_S3_c_d2": coef_table.get("Today_S3", {}).get("c_d2"),
        "interpretation": (
            "Today_S3 was pure δ2 step and dropped −4 ticks. "
            "Today_S4 Case-B used 0.35*δ2; Today_S5 continued from S4 — residual +c_d2 is inherited poison."
        ),
    }
    out = {
        "coef_table": coef_table,
        "delta2_components": cos_rows,
        "delta2_good_names": good_names,
        "delta2_bad_names": bad_names,
        "poison": poison,
        "g_lb_coefs": g_lb.tolist(),
    }
    report["inv1"] = out
    (ART / "inv1_coord_audit.json").write_text(json.dumps(out, indent=2))
    return Z, d_ladder, d1, d2, d3, e1, e2, d2_good, d2_bad, g_vec, coef_table


# ============================================================================
# Investigation 2
# ============================================================================
def inv2(y, folds, test_ids, B_bad, report):
    log("=== INV2: multi-LM super + boundary gate ===")
    ntr = len(y)
    z_raw_o = zrk(np.load(P12 / "lm_oof.npy").ravel())
    z_raw_t = zrk(np.load(P12 / "lm_test.npy").ravel())
    _, z_raw_t_a = align_method_b_smooth(z_raw_o, z_raw_t, folds)
    base_folds = fold_aucs(y, z_raw_o, folds)
    base_auc = float(roc_auc_score(y, z_raw_o))

    # gpt2m bayes + meanllr + std
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
        ro, rt = align_method_b_smooth(r_o, np.mean(r_tf, 0), folds)
        return ro, rt

    candidates = {}
    # bayes residual (today's)
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
    log(f"  baseline lm12_raw_super AUC={super0_auc:.7f} Δe5={[round((a-b)*1e5,2) for a,b in zip(super0_folds, base_folds)]}")

    # Greedy LOFO: try adding each residual with nested weight
    kept = [("r_gpt2m_bayes", r_bay_o, r_bay_t, W_SUPER)]
    kept_names = ["r_gpt2m_bayes"]
    current_o = z_super0_o.copy()
    current_folds = super0_folds[:]
    w_grid = np.linspace(-0.05, 0.15, 41)

    for nm, (ro, rt) in candidates.items():
        if nm == "r_gpt2m_bayes":
            d, ks = disp_ks(ro, rt)
            log(f"  {nm}: already in base disp={d:.6f} KS={ks:.6f}")
            continue
        d, ks = disp_ks(ro, rt)
        # nested LOFO weight
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
        pos = sum(d_ > 1e-4 for d_ in deltas)  # tiny numerical
        ok_hyg = (abs(d - 1.0) < 1e-4) and (ks < 0.0003)
        improve_all = all(d_ > 0 for d_ in deltas)
        log(f"  try {nm}: Δe5={[round(x,2) for x in deltas]} pos={pos}/5 disp={d:.6f} KS={ks:.6f} keep={improve_all and ok_hyg}")
        if improve_all and ok_hyg and float(np.mean(nws)) != 0.0:
            mean_w = float(np.mean(nws))
            kept.append((nm, ro, rt, mean_w))
            kept_names.append(nm)
            current_o = current_o + mean_w * ro
            current_folds = fold_aucs(y, current_o, folds)
            log(f"    KEPT w≈{mean_w:.4f}")

    # Build v3 oof/test
    z_v3_o = z_raw_o.copy()
    z_v3_t = z_raw_t_a.copy()
    for nm, ro, rt, w in kept:
        z_v3_o = z_v3_o + w * ro
        z_v3_t = z_v3_t + w * rt
    z_v3_o_a, z_v3_t_a = align_method_b_smooth(z_v3_o, z_v3_t, folds)
    v3_folds = fold_aucs(y, z_v3_o, folds)
    v3_auc = float(roc_auc_score(y, z_v3_o))
    log(f"  z_lm_raw_super_v3 AUC={v3_auc:.7f} Δvs_raw={[round((a-b)*1e5,2) for a,b in zip(v3_folds, base_folds)]}")
    np.save(ART / "z_lm_raw_super_v3_oof.npy", z_v3_o)
    np.save(ART / "z_lm_raw_super_v3_test.npy", z_v3_t_a)

    # Boundary gate vs w12
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
    _, z_w12_t_a = align_method_b_smooth(z_w12_o, z_w12_t, folds)
    p_w12_o = expit(z_w12_o)
    p_w12_t = expit(z_w12_t_a)
    g_o = np.tanh(np.abs(zrk(z_v3_o) - z_w12_o)) * p_w12_o * (1.0 - p_w12_o)
    g_t = np.tanh(np.abs(zrk(z_v3_t_a) - z_w12_t_a)) * p_w12_t * (1.0 - p_w12_t)
    # residual of v3 vs w12, gated
    beta = float(np.cov(zrk(z_v3_o), z_w12_o, ddof=0)[0, 1] / (np.var(z_w12_o) + 1e-12))
    r_lin_o = unitize(zrk(z_v3_o) - beta * z_w12_o)
    r_lin_t = zrk(z_v3_t_a) - beta * z_w12_t_a
    r_lin_o, r_lin_t = align_method_b_smooth(r_lin_o, r_lin_t, folds)
    r_bnd_o = unitize(g_o * (zrk(z_v3_o) - beta * z_w12_o))
    r_bnd_t = g_t * (zrk(z_v3_t_a) - beta * z_w12_t_a)
    r_bnd_o, r_bnd_t = align_method_b_smooth(r_bnd_o, r_bnd_t, folds)

    # Project through P_perp
    d3_lin_t = project_perp(r_lin_t, B_bad)
    d3_bnd_t = project_perp(r_bnd_t, B_bad)
    d3_lin_o, d3_lin_t = align_method_b_smooth(r_lin_o, d3_lin_t, folds)
    d3_bnd_o, d3_bnd_t = align_method_b_smooth(r_bnd_o, d3_bnd_t, folds)
    for tag, o, t in [("d3_linear", d3_lin_o, d3_lin_t), ("d3_boundary", d3_bnd_o, d3_bnd_t)]:
        d, ks = disp_ks(o, t)
        log(f"  {tag}: disp={d:.6f} KS={ks:.6f}")
        np.save(ART / f"{tag}_oof.npy", o)
        np.save(ART / f"{tag}_test.npy", t)

    # Nested OOF eval of adding boundary to raw
    def eval_add(r_o, name):
        nested = np.zeros(ntr)
        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            best_w, best_s = 0.0, -1e9
            for w in w_grid:
                lifts = [
                    float(roc_auc_score(y[folds[oi][1]], z_raw_o[folds[oi][1]] + W_SUPER * r_bay_o[folds[oi][1]] + w * r_o[folds[oi][1]]))
                    - super0_folds[oi]
                    for oi in other
                ]
                s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                if s > best_s:
                    best_s, best_w = s, float(w)
            nested[b] = z_raw_o[b] + W_SUPER * r_bay_o[b] + best_w * r_o[b]
        nf = fold_aucs(y, nested, folds)
        return {
            "name": name,
            "auc": float(roc_auc_score(y, nested)),
            "delta_vs_super0_1e5": [(a - b) * 1e5 for a, b in zip(nf, super0_folds)],
            "pos": int(sum((a - b) > 0 for a, b in zip(nf, super0_folds))),
        }

    ev_lin = eval_add(d3_lin_o, "d3_linear")
    ev_bnd = eval_add(d3_bnd_o, "d3_boundary")
    log(f"  {ev_lin}")
    log(f"  {ev_bnd}")

    out = {
        "kept": [{"name": n, "w": w} for n, _, _, w in kept],
        "v3_auc": v3_auc,
        "v3_folds": v3_folds,
        "super0_auc": super0_auc,
        "ev_linear": ev_lin,
        "ev_boundary": ev_bnd,
    }
    report["inv2"] = out
    return z_v3_o, z_v3_t_a, d3_lin_o, d3_lin_t, d3_bnd_o, d3_bnd_t, z_bay_o_a, z_bay_t_a


# ============================================================================
# Investigation 3
# ============================================================================
def inv3(y, folds, test_ids, z_v3_o, z_v3_t, z_bay_o, z_bay_t, z_v8_old_o, report):
    log("=== INV3: 68-KMAT 3-family Stage-B + δ1_v3 hygiene ===")
    ntr = len(y)
    # Diagnose KS of today's d1
    z_v8_old_t = zrk(load_sub("sub_v8_part12_fsl_prop_rect.csv", test_ids))
    z_v8_clean_o = np.load(ART2 / "z_v8_clean_oof.npy").ravel()
    z_v8_clean_t = np.load(ART2 / "z_v8_clean_test.npy").ravel()
    d1_raw_o = zrk(z_v8_clean_o) - z_v8_old_o
    d1_raw_t = zrk(z_v8_clean_t) - z_v8_old_t
    d1_a_o, d1_a_t = align_method_b_smooth(d1_raw_o, d1_raw_t, folds)
    d_ks = disp_ks(d1_raw_o, d1_raw_t)
    d_ks_a = disp_ks(d1_a_o, d1_a_t)
    # Check same p12_fsl backbone: both v8 old and clean use Slot-2 formula; clean test is zrk(rank)
    log(f"  raw δ1 disp/KS={d_ks} after Method-B={d_ks_a}")

    if SKIP_TRAIN and (ART / "xgb_stageB_68_bce_oof.npy").exists():
        log("  SKIP_TRAIN=1 — loading existing Stage-B families")
        fams = {}
        for tag in ["xgb_stageB_68_bce", "xgb_stageB_68_focal_auc", "lgb_stageB_68_clean"]:
            fams[tag] = (np.load(ART / f"{tag}_oof.npy"), np.load(ART / f"{tag}_test.npy"))
    else:
        log("  importing gboost for KMAT/BASE…")
        import gboost as gb

        log(f"  KMAT={gb.KMAT.shape} BASE={gb.BASE.shape}")
        K = gb.KMAT
        K_tr, K_te = K[:ntr], K[ntr:]
        n_keys = 68  # exact first 68
        K_oof, K_te_m = build_kmat_te(K_tr[:, :n_keys], K_te[:, :n_keys], y, folds, n_keys)

        def ab(xo, xt):
            return align_method_b_smooth(zrk(xo) if xo.ndim == 1 else xo, zrk(xt) if xt.ndim == 1 else xt, folds)

        z_v4_o, z_v4_t = ab(np.load(PRED / "P_super_v4p_hc_oof.npy").ravel(), np.load(PRED / "P_super_v4p_hc_test.npy").ravel())
        z_rev_o, z_rev_t = ab(
            np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel(),
            np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy").ravel(),
        )
        l43_o, l43_t = ab(np.load(MV2 / "P4_L4.3_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.3_pure5f_test.npy").ravel())
        l42_o, l42_t = ab(np.load(MV2 / "P4_L4.2_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.2_pure5f_test.npy").ravel())
        l45_o, l45_t = ab(np.load(MV2 / "P4_L4.5_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.5_pure5f_test.npy").ravel())
        z_tab_o, z_tab_t = ab(np.load(HARV / "oof_tabm_5f.npy").ravel(), np.load(HARV / "test_tabm_5f.npy").ravel())
        zs_o, zs_t = align_method_b_smooth(zrk(z_v3_o), zrk(z_v3_t), folds)
        zm_o, zm_t = z_bay_o, z_bay_t

        r_rare = K_oof[:, 0].astype(np.float64)
        r_disp = K_oof[:, 1].astype(np.float64)
        r_rare_t = K_te_m[:, 0].astype(np.float64)
        r_disp_t = K_te_m[:, 1].astype(np.float64)
        extra_o = np.column_stack(
            [zs_o, zm_o, z_rev_o, zs_o - z_v4_o, zs_o * r_rare, zs_o * r_disp, l43_o, l42_o, l45_o, z_tab_o]
        ).astype(np.float32)
        extra_t = np.column_stack(
            [zs_t, zm_t, z_rev_t, zs_t - z_v4_t, zs_t * r_rare_t, zs_t * r_disp_t, l43_t, l42_t, l45_t, z_tab_t]
        ).astype(np.float32)
        for j in range(extra_o.shape[1]):
            eo, et = align_method_b_smooth(extra_o[:, j], extra_t[:, j], folds)
            extra_o[:, j] = eo.astype(np.float32)
            extra_t[:, j] = et.astype(np.float32)

        X_o = np.hstack([K_oof, extra_o, gb.BASE[:ntr].astype(np.float32)])
        X_t = np.hstack([K_te_m, extra_t, gb.BASE[ntr:].astype(np.float32)])
        log(f"  X_stageB_68 shape o={X_o.shape} t={X_t.shape}")

        v5_o = zrk(np.load(PRED / "P_super_v5_harvest_oof.npy").ravel())
        v5_t = zrk(np.load(PRED / "P_super_v5_harvest_test.npy").ravel())
        p_v5 = expit(v5_o)
        # overlap-band focal weights
        w_focal = 0.25 + np.power(p_v5 * (1.0 - p_v5), 0.45)
        w_focal = w_focal.astype(np.float64)

        def train_xgb_bag(tag, use_focal=False):
            specs = [(42 + i, 2) for i in range(3)] + [(100 + i, 3) for i in range(3)]
            oof_bag = np.zeros(ntr)
            te_bag = []
            for bi, (seed, depth) in enumerate(specs):
                oof_s = np.zeros(ntr)
                te_s = []
                for fi, (a, b) in enumerate(folds):
                    tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                    bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
                    sw_tr = w_focal[tr_in] if use_focal else None
                    dtr = xgb.DMatrix(X_o[tr_in], label=y[tr_in], base_margin=bm_tr, weight=sw_tr)
                    des = xgb.DMatrix(X_o[es_in], label=y[es_in], base_margin=bm_all[es_in])
                    dva = xgb.DMatrix(X_o[b], label=y[b], base_margin=bm_all[b])
                    dte = xgb.DMatrix(X_t, base_margin=bm_te)
                    params = dict(
                        max_depth=depth, eta=0.025, subsample=0.8, colsample_bynode=0.35,
                        reg_lambda=30.0, min_child_weight=200, objective="binary:logistic",
                        eval_metric="auc", tree_method="hist", max_bin=256, nthread=4, seed=seed,
                    )
                    booster = xgb.train(
                        params, dtr, num_boost_round=400, evals=[(des, "es")],
                        early_stopping_rounds=40, verbose_eval=False,
                    )
                    oof_s[b] = booster.predict(dva)
                    te_s.append(booster.predict(dte))
                    log(f"  {tag} bag{bi} d{depth} F{fi} best_iter={booster.best_iteration}")
                oof_bag += oof_s
                te_bag.append(np.mean(te_s, 0))
            return oof_bag / len(specs), np.mean(te_bag, 0)

        def train_lgb_bag(tag):
            specs = [(42 + i, 4 if i % 2 == 0 else 7) for i in range(6)]
            # actually 3x leaves=4 + 3x leaves=7
            specs = [(42 + i, 4) for i in range(3)] + [(100 + i, 7) for i in range(3)]
            oof_bag = np.zeros(ntr)
            te_bag = []
            for bi, (seed, leaves) in enumerate(specs):
                oof_s = np.zeros(ntr)
                te_s = []
                for fi, (a, b) in enumerate(folds):
                    tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                    bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
                    dtr = lgb.Dataset(X_o[tr_in], label=y[tr_in], init_score=bm_tr)
                    des = lgb.Dataset(X_o[es_in], label=y[es_in], init_score=bm_all[es_in], reference=dtr)
                    params = dict(
                        objective="binary", metric="auc", learning_rate=0.025, num_leaves=leaves,
                        feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1,
                        lambda_l2=30.0, min_child_samples=250, verbosity=-1, seed=seed, num_threads=4,
                    )
                    booster = lgb.train(
                        params, dtr, num_boost_round=400, valid_sets=[des],
                        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
                    )
                    # predict with init score
                    oof_s[b] = expit(booster.predict(X_o[b], raw_score=True) + bm_all[b])
                    te_s.append(expit(booster.predict(X_t, raw_score=True) + bm_te))
                    log(f"  {tag} bag{bi} L{leaves} F{fi} best_iter={booster.best_iteration}")
                oof_bag += oof_s
                te_bag.append(np.mean(te_s, 0))
            return oof_bag / len(specs), np.mean(te_bag, 0)

        fams = {}
        o, t = train_xgb_bag("xgb_bce", use_focal=False)
        fams["xgb_stageB_68_bce"] = (o, t)
        o, t = train_xgb_bag("xgb_focal", use_focal=True)
        fams["xgb_stageB_68_focal_auc"] = (o, t)
        o, t = train_lgb_bag("lgb_clean")
        fams["lgb_stageB_68_clean"] = (o, t)
        for tag, (o, t) in fams.items():
            np.save(ART / f"{tag}_oof.npy", o)
            np.save(ART / f"{tag}_test.npy", t)

    fam_scores = {}
    for tag, (o, t) in fams.items():
        zo = zrk(o)
        fam_scores[tag] = {"auc": float(roc_auc_score(y, zo)), "folds": fold_aucs(y, zo, folds)}
        log(f"  {tag}: AUC={fam_scores[tag]['auc']:.7f} folds={[round(x,7) for x in fam_scores[tag]['folds']]}")

    # Rebuild v8 stack with new families + lgb_fixed
    w12_o = np.load(P12 / "wide_lm_stack_nested.npy").ravel()
    w12_t = np.load(P12 / "wide_lm_stack_test.npy").ravel()
    p9_t = np.load(P12 / "part9_test.npy").ravel()
    z_w12_o, z_w12_t = zrk(w12_o), zrk(w12_t)
    specs = [
        ("v4p_hc", np.load(PRED / "P_super_v4p_hc_oof.npy").ravel(), np.load(PRED / "P_super_v4p_hc_test.npy").ravel()),
        ("B_xgb_llr", np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel(), np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel()),
        ("P4_L4.3", np.load(MV2 / "P4_L4.3_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.3_pure5f_test.npy").ravel()),
        ("P4_L4.2", np.load(MV2 / "P4_L4.2_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.2_pure5f_test.npy").ravel()),
        ("P4_L4.5", np.load(MV2 / "P4_L4.5_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.5_pure5f_test.npy").ravel()),
        ("P4_L4.3_ext", np.load(MV2 / "P4_L4.3_ext_oof.npy").ravel(), np.load(MV2 / "P4_L4.3_ext_test.npy").ravel()),
        ("gpt2_bidir", np.load(PRED / "gpt2_3ep/gpt2_bidir_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/gpt2_bidir_test.npy").ravel()),
        ("lgb_gpt2_3ep_fixed", np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel()),
    ]
    for tag, (o, t) in fams.items():
        specs.append((tag, o, t))
    R_o, R_t, metas = build_rect_basis(specs, z_w12_o, z_w12_t)
    names = [m["name"] for m in metas]
    oof_v8v3, mean_w, _ = nested_coord_ascent(y, z_w12_o, R_o, folds, names)
    v8v3_auc = float(roc_auc_score(y, oof_v8v3))
    v8v3_folds = fold_aucs(y, oof_v8v3, folds)
    v8_old_folds = fold_aucs(y, z_v8_old_o, folds)
    d_v8 = [(a - b) * 1e5 for a, b in zip(v8v3_folds, v8_old_folds)]
    log(f"  v8_v3 AUC={v8v3_auc:.7f} vs old {float(roc_auc_score(y,z_v8_old_o)):.7f} Δe5={[round(x,2) for x in d_v8]} pos={sum(x>0 for x in d_v8)}/5")

    dz_oof = R_o @ mean_w
    dz_test = qmap_to_oof(R_t @ mean_w, dz_oof)
    z_p12_fsl = zrk(0.68 * zrk(w12_t) + 0.32 * zrk(p9_t))
    beta_fsl = float(np.dot(dz_test, z_p12_fsl) / (np.dot(z_p12_fsl, z_p12_fsl) + 1e-12))
    dz_orth = qmap_to_oof(dz_test - beta_fsl * z_p12_fsl, dz_oof)
    z_v8_v3_t = zrk(rk(z_p12_fsl + dz_orth))
    np.save(ART / "z_v8_v3_oof.npy", oof_v8v3)
    np.save(ART / "z_v8_v3_test.npy", z_v8_v3_t)

    d1_v3_o = oof_v8v3 - z_v8_old_o
    d1_v3_t = z_v8_v3_t - z_v8_old_t
    d1_v3_o, d1_v3_t = align_method_b_smooth(d1_v3_o, d1_v3_t, folds)
    d, ks = disp_ks(d1_v3_o, d1_v3_t)
    log(f"  δ1_v3 Method-B disp={d:.6f} KS={ks:.6f}")
    np.save(ART / "d1_v3_oof.npy", d1_v3_o)
    np.save(ART / "d1_v3_test.npy", d1_v3_t)

    out = {
        "d1_today_raw_disp_ks": d_ks,
        "d1_today_aligned_disp_ks": d_ks_a,
        "families": fam_scores,
        "v8_v3": {"auc": v8v3_auc, "folds": v8v3_folds, "delta_vs_old_1e5": d_v8, "pos": int(sum(x > 0 for x in d_v8))},
        "d1_v3_disp_ks": [d, ks],
        "mean_w": {n: float(w) for n, w in zip(names, mean_w)},
    }
    report["inv3"] = out
    return d1_v3_o, d1_v3_t, oof_v8v3, z_v8_v3_t


# ============================================================================
# Investigation 4
# ============================================================================
def inv4(y, test_ids, folds, Z, d_ladder, d1, d2, d3, e1, e2, d2_good, d2_bad, d1_v3_t, d3_lin_t, d3_bnd_t, report):
    log("=== INV4: 19-pt Ridge surface + stage candidates (NO_SUBMIT) ===")
    z95 = Z["St9.5"]["z"]
    z_s5 = Z["Today_S5"]["z"]
    z_v8 = Z["v8"]["z"]
    z_lead = Z["leader"]["z"]
    z_soft = Z["soft"]["z"]
    z_st9 = Z["St9"]["z"]
    d_bad_leader = z_lead - z_st9
    d_bad_soft = z_soft - z_st9
    B_bad = gram_orthonormal(np.column_stack([e2, e2, d_bad_leader, d_bad_soft, d2_bad]))
    log(f"  B_bad expanded rank={B_bad.shape[1]}")

    # Prefer boundary if it was positive else linear
    d3_v3 = d3_bnd_t if report.get("inv2", {}).get("ev_boundary", {}).get("pos", 0) >= 3 else d3_lin_t
    d1_v3 = project_perp(d1_v3_t, B_bad)
    d3_v3 = project_perp(d3_v3, B_bad)
    d2_g = project_perp(d2_good, B_bad)
    for tag, v in [("d1_v3", d1_v3), ("d3_v3", d3_v3), ("d2_good", d2_g)]:
        # re-std
        v = unitize(v)
        if tag == "d1_v3":
            d1_v3 = v
        elif tag == "d3_v3":
            d3_v3 = v
        else:
            d2_g = v

    # LOO Ridge on 19 points in coords of [ladder, d1, d2, d3, d1_v3, d3_v3, d2_g]
    axes = [d_ladder, d1, d2, d3, d1_v3, d3_v3, d2_g]
    anames = ["ladder", "d1", "d2", "d3", "d1_v3", "d3_v3", "d2_good"]
    X, yy, names = [], [], []
    for name, meta in Z.items():
        c, *_ = np.linalg.lstsq(np.column_stack(axes), meta["z"] - z95, rcond=None)
        X.append(c)
        yy.append(meta["lb"])
        names.append(name)
    X, yy = np.asarray(X), np.asarray(yy)
    # LOO CV
    preds_loo = np.zeros(len(yy))
    for i in range(len(yy)):
        mask = np.ones(len(yy), dtype=bool)
        mask[i] = False
        ridge = Ridge(alpha=1e-2, fit_intercept=True).fit(X[mask], yy[mask])
        preds_loo[i] = ridge.predict(X[i : i + 1])[0]
    loo_mae = float(np.mean(np.abs(preds_loo - yy)))
    ridge = Ridge(alpha=1e-2, fit_intercept=True).fit(X, yy)
    g = ridge.coef_
    log(f"  Ridge LOO MAE={loo_mae:.6f} coefs={dict(zip(anames, np.round(g,4)))}")

    def predict_path(delta_from_s5):
        c, *_ = np.linalg.lstsq(np.column_stack(axes), (z_s5 + delta_from_s5) - z95, rcond=None)
        return float(ridge.predict(c.reshape(1, -1))[0])

    # Candidate 1: purge δ2_bad from Slot5, advance ladder+d1_v3+d3_v3
    # Estimate residual δ2_bad contamination in Slot5 via projection
    c_s5, *_ = np.linalg.lstsq(np.column_stack([d_ladder, d1, d2, d3]), z_s5 - z95, rcond=None)
    poison_scale = float(c_s5[2])  # c_d2
    # subtract d2_bad component overlapping d2
    purge = -abs(poison_scale) * 0.35 * d2_bad  # conservative fraction of Case-B 0.35 weight
    step1 = unitize(d_ladder) * 0.15 + d1_v3 * 0.40 + d3_v3 * 0.45
    step1 = project_perp(step1, B_bad)
    step1 = unitize(step1)
    # scale to ~0.5e-4 spearman from s5
    from gm_nullspace_strike_orth_v3 import weight_for_spearman_dist

    w1, got1 = weight_for_spearman_dist(z_s5, step1 + purge, 0.50e-4, lo=0.0, hi=50.0, n=48)
    z_c1 = z_s5 + w1 * unitize(step1 + purge)
    pred_c1 = predict_path(z_c1 - z_s5)

    # Candidate 2: full combo
    step2 = unitize(d_ladder) * 0.10 + d1_v3 * 0.40 + d3_v3 * 0.35 + d2_g * 0.15
    step2 = project_perp(step2, B_bad)
    step2 = unitize(step2)
    w2, got2 = weight_for_spearman_dist(z_s5, step2, 0.80e-4, lo=0.0, hi=50.0, n=48)
    z_c2 = z_s5 + w2 * step2
    pred_c2 = predict_path(z_c2 - z_s5)

    # Save staged (NO_SUBMIT) probability maps for tomorrow
    def to_probs(z):
        # map onto v8 oof law
        z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
        q = np.linspace(0, 1, 10001)
        from scipy.interpolate import PchipInterpolator

        u = (rankdata(z, method="average") - 0.5) / len(z)
        pchip = PchipInterpolator(q, np.quantile(z_v8_o, q), extrapolate=True)
        zm = np.asarray(pchip(u), float)
        lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
        lr.fit(z_v8_o.reshape(-1, 1), y.astype(np.int32))
        zlin = float(lr.coef_[0, 0]) * zm + float(lr.intercept_[0])
        from scipy.optimize import minimize_scalar

        def obj(d):
            return (float(np.mean(expit(zlin + d))) - 0.016700) ** 2

        d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
        return expit(zlin + d)

    for tag, z, pred, got in [
        ("cand1_purge_slot3_poison", z_c1, pred_c1, got1),
        ("cand2_stageB_multilm", z_c2, pred_c2, got2),
    ]:
        p = to_probs(z)
        np.save(ART / f"{tag}_z.npy", z)
        np.save(ART / f"{tag}_p.npy", p)
        # stage CSV but do NOT submit
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(SUB / f"sub_gm_v3_{tag}_STAGED_NOSUBMIT.csv", index=False)
        log(f"  staged {tag}: pred_LB={pred:.5f} 1mrho={got*1e4:.3f}e-4 mean={p.mean():.6f}")

    out = {
        "loo_mae": loo_mae,
        "ridge_coefs": dict(zip(anames, map(float, g))),
        "cand1": {"pred_LB": pred_c1, "1mrho": got1, "w": w1, "poison_c_d2": poison_scale},
        "cand2": {"pred_LB": pred_c2, "1mrho": got2, "w": w2},
        "NO_SUBMIT": True,
    }
    report["inv4"] = out
    return out


def write_report(report):
    md = []
    md.append("# GM Max-Jump Research Suite v3")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{report.get('elapsed_s', '?')}s**.")
    md.append("")
    md.append("## Investigation 1 — Coordinate audit & δ2 poison")
    md.append("")
    inv1 = report.get("inv1", {})
    md.append("| sub | LB | c_ladder | c_δ1 | c_δ2 | c_δ3 | cos(δ2) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, row in inv1.get("coef_table", {}).items():
        md.append(
            f"| {k} | {row['lb']:.5f} | {row['c_ladder']:+.4f} | {row['c_d1']:+.4f} | "
            f"{row['c_d2']:+.4f} | {row['c_d3']:+.4f} | {row['cos_d2']:+.3f} |"
        )
    md.append("")
    poison = inv1.get("poison", {})
    md.append(
        f"**Slot5 inherited δ2 coeff `c_δ2={poison.get('Today_S5_c_d2')}`** "
        f"(Slot3 pure-δ2 drop had `c_δ2={poison.get('Today_S3_c_d2')}`)."
    )
    md.append("")
    md.append("### δ2 component cosines")
    md.append("")
    md.append("| component | cos(e1) | cos(e2) | cos(g_LB) | bucket |")
    md.append("|---|---:|---:|---:|---|")
    for r in inv1.get("delta2_components", []):
        md.append(
            f"| {r['name']} | {r['cos_e1']:+.3f} | {r['cos_e2']:+.3f} | {r['cos_g_lb']:+.3f} | {r['bucket']} |"
        )
    md.append("")
    md.append(f"- δ2_good: `{inv1.get('delta2_good_names')}`")
    md.append(f"- δ2_bad: `{inv1.get('delta2_bad_names')}`")
    md.append("")
    md.append("## Investigation 2 — Multi-LM super + boundary")
    md.append("")
    inv2 = report.get("inv2", {})
    md.append(f"- Kept residuals: `{inv2.get('kept')}`")
    md.append(f"- `z_lm_raw_super_v3` AUC={inv2.get('v3_auc')} (base super={inv2.get('super0_auc')})")
    md.append(f"- d3_linear eval: `{inv2.get('ev_linear')}`")
    md.append(f"- d3_boundary eval: `{inv2.get('ev_boundary')}`")
    md.append("")
    md.append("## Investigation 3 — 68-KMAT 3-family Stage-B / δ1_v3")
    md.append("")
    inv3 = report.get("inv3", {})
    md.append(f"- Today δ1 raw disp/KS: `{inv3.get('d1_today_raw_disp_ks')}` → aligned `{inv3.get('d1_today_aligned_disp_ks')}`")
    for tag, sc in inv3.get("families", {}).items():
        md.append(f"- **{tag}**: AUC={sc['auc']:.7f} folds={sc['folds']}")
    v8 = inv3.get("v8_v3", {})
    md.append(f"- **v8_v3**: AUC={v8.get('auc')} Δvs_old_e5={v8.get('delta_vs_old_1e5')} pos={v8.get('pos')}/5")
    md.append(f"- δ1_v3 Method-B disp/KS: `{inv3.get('d1_v3_disp_ks')}`")
    md.append("")
    md.append("## Investigation 4 — Ridge surface & staged candidates (`NO_SUBMIT`)")
    md.append("")
    inv4 = report.get("inv4", {})
    md.append(f"- Ridge LOO MAE={inv4.get('loo_mae')}")
    md.append(f"- Ridge coefs: `{inv4.get('ridge_coefs')}`")
    md.append(f"- **Cand1 Purge_Slot3_Poison**: pred_LB={inv4.get('cand1',{}).get('pred_LB')} 1−ρ={inv4.get('cand1',{}).get('1mrho')}")
    md.append(f"- **Cand2 StageB+MultiLM**: pred_LB={inv4.get('cand2',{}).get('pred_LB')} 1−ρ={inv4.get('cand2',{}).get('1mrho')}")
    md.append("")
    md.append("Staged CSVs (not submitted): `submissions/sub_gm_v3_*_STAGED_NOSUBMIT.csv`")
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    (LOG / "GM_MAX_JUMP_RESEARCH_V3.md").write_text("\n".join(md))
    log(f"Wrote {LOG / 'GM_MAX_JUMP_RESEARCH_V3.md'}")


def main():
    assert NO_SUBMIT, "This suite is STRICT NO_SUBMIT"
    log(f"GM_MAX_JUMP_RESEARCH_V3 START (NO_SUBMIT={NO_SUBMIT} SKIP_TRAIN={SKIP_TRAIN})")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    z_v8_old_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())

    report = {"NO_SUBMIT": True}
    Z, d_ladder, d1, d2, d3, e1, e2, d2_good, d2_bad, g_vec, coef_table = inv1(
        y, test_ids, folds, z_v8_old_o, report
    )

    # Build B_bad early for inv2
    z_st3 = Z["St3A"]["z"]
    z_st4 = Z["St4A"]["z"]
    z_st5 = Z["St5A"]["z"]
    z_st9 = Z["St9"]["z"]
    B_bad = gram_orthonormal(
        np.column_stack(
            [
                e2,
                Z["leader"]["z"] - z_st9,
                Z["soft"]["z"] - z_st9,
                d2_bad,
            ]
        )
    )

    z_v3_o, z_v3_t, d3_lin_o, d3_lin_t, d3_bnd_o, d3_bnd_t, z_bay_o, z_bay_t = inv2(
        y, folds, test_ids, B_bad, report
    )
    d1_v3_o, d1_v3_t, oof_v8v3, z_v8_v3_t = inv3(
        y, folds, test_ids, z_v3_o, z_v3_t, z_bay_o, z_bay_t, z_v8_old_o, report
    )
    inv4(
        y, test_ids, folds, Z, d_ladder, d1, d2, d3, e1, e2, d2_good, d2_bad,
        d1_v3_t, d3_lin_t, d3_bnd_t, report,
    )
    report["elapsed_s"] = round(time.time() - T0, 1)
    (LOG / "GM_MAX_JUMP_RESEARCH_V3.json").write_text(json.dumps(report, indent=2, default=float))
    write_report(report)
    log("GM_MAX_JUMP_RESEARCH_V3_COMPLETE")


if __name__ == "__main__":
    main()
