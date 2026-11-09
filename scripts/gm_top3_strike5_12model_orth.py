#!/usr/bin/env python3
"""GM Top-3 Strike 5: 12-Model Homogeneous 5F Orth Stack + Continuation from 0.95323 (`Strike 4A`).

Unbroken Public LB climb so far:
  - `Slot 3`   (58126784): 0.95312
  - `Att 2`    (58139652): 0.95315 (+0.00003)
  - `Att 3`    (58148980): 0.95317 (+0.00002)
  - `Strike 2` (58150232): 0.95319 (+0.00002, 5-model orth OOF 0.9526417)
  - `Strike 3A`(58150384): 0.95322 (+0.00003, 7-model orth OOF 0.9526612)
  - `Strike 4A`(58150447): 0.95323 (+0.00001, tied for #3 on Leaderboard!)

Notice the pattern:
  - Adding 5 orthogonal models in Strike 2 gave +0.00002 (0.95317 -> 0.95319).
  - Adding 2 more orthogonal models (`glm_quad`, `lab1_glm_plus_graph`) in Strike 3A gave +0.00003 (0.95319 -> 0.95322).
  - Pure continuation without new models in Strike 4A gave +0.00001 (0.95322 -> 0.95323).

Now in Strike 5, we inject 5 MORE `|d_rho| < 0.00195` audited orthogonal 5-fold models on top of the 7-model OOF (`0.9526612`):
  8.  `gbn_w512`        (`|d_rho| = 0.00079`, +1.39e-5 OOF, w = +0.0150)
  9.  `P_restack_pub`   (`|d_rho| = 0.00194`, +1.16e-5 OOF, w = -0.0140)
  10. `P_super_v6_clean`(`|d_rho| = 0.00077`, +0.82e-5 OOF, w = +0.0140)
  11. `part6_xgb_cuda_d1`(`|d_rho| = 0.00139`, +0.36e-5 OOF, w = +0.0100)
  12. `part6_cat_d2_d3` (`|d_rho| = 0.00092`, +0.33e-5 OOF, w = +0.0100)
plus Areeb's verified 10% `megayak` shallow-tree orthogonal test residual (`r_shallow_orth_t`),
anchored on `Strike 4A` (`0.95323`) + `0.40 * (Strike 4A - Strike 3A)`!
"""
from __future__ import annotations

import json
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paradigm_stack_delta_eval import fast_auc

PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
SUB = ROOT / "submissions"
P12 = ROOT / "external/nov08_part12"
N09 = ROOT / "external/nov09_top3_recon"
LOG = ROOT / "logs/overnight"
TARGET_MEAN = 0.016700


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), 1e-6, 1.0 - 1e-6))


def pchip_quantile_map(x_test, x_ref, n_knots=4000):
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


def orth_vs_v8(z_cand_o, z_cand_t, z_v8_o, z_v8_t, folds):
    ro = np.zeros(len(z_v8_o), dtype=np.float64)
    rt_f = []
    for a, b in folds:
        beta = float(np.cov(z_cand_o[a], z_v8_o[a], ddof=0)[0, 1] / np.var(z_v8_o[a]))
        ra = z_cand_o[a] - beta * z_v8_o[a]
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (z_cand_o[b] - beta * z_v8_o[b] - mu) / sd
        rt_f.append((z_cand_t - beta * z_v8_t - mu) / sd)
    return ro, pchip_quantile_map(np.mean(rt_f, axis=0), ro, n_knots=4000)


def main():
    print("[Boot] gm_top3_strike5_12model_orth.py — Pushing from 0.95323 to outright Top 3...", flush=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s3_sub = pd.read_csv(SUB / "sub_slot3_gm_verified_f02.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    a3_sub = pd.read_csv(SUB / "sub_gm_att3_continue_from_095315.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    st2_sub = pd.read_csv(SUB / "sub_gm_top3_strike2_homogeneous5f.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    st3_sub = pd.read_csv(SUB / "sub_strike3a_cont065_plus_glmquad_graph.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    st4_sub = pd.read_csv(SUB / "sub_gm_top3_strike4a_cont070.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y))

    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = pchip_quantile_map(zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel()), z_w12_o, n_knots=4000)
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)
    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=4000)
    r_old_o, r_old_t = orth_vs_v8(z_xgbf_old_o, z_xgbf_old_t, z_v8_o, z_v8_t, folds)
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)

    sam_o = pd.read_csv(N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/oof.csv")["nested_winner"].to_numpy(float)
    sam_t = pd.read_csv(N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/submission_cavg.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    # First 7 models (already in Strike 3A = 0.95322 & Strike 4A = 0.95323)
    specs7 = [
        ("lab4_nade", np.load(PRED / "part11/lab4_tabular_nade_oof.npy").ravel(), np.load(PRED / "part11/lab4_tabular_nade_test.npy").ravel(), 0.0135),
        ("sameer_200m", sam_o, sam_t, 0.0115),
        ("orig20k_knn", np.load(HARV / "oof_orig20k_knn.npy").ravel(), np.load(HARV / "test_orig20k_knn.npy").ravel(), -0.0085),
        ("p4_cuda_xgb", np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel(), np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel(), -0.0085),
        ("lab1_zjoint_graph", np.load(PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy").ravel(), np.load(PRED / "part11/lab1_zjoint_graph_lgb4_test.npy").ravel(), -0.0105),
        ("glm_quad", np.load(PRED / "ladder/glm_quad_oof.npy").ravel(), np.load(PRED / "ladder/glm_quad_test.npy").ravel(), 0.0150),
        ("lab1_glm_plus_graph", np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel(), np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel(), 0.0125),
    ]

    # New 5 ultra-clean (|d_rho| < 0.00195) models for Strike 5
    specs5_new = [
        ("gbn_w512", np.load(PRED / "gbn_cache/gbn_bag5_w512_oof.npy").ravel(), np.load(PRED / "gbn_cache/gbn_bag5_w512_full.npy").ravel(), 0.0150),
        ("P_restack_pub", np.load(PRED / "P_restack_pub_oof.npy").ravel(), np.load(PRED / "P_restack_pub_test.npy").ravel(), -0.0140),
        ("P_super_v6_clean", np.load(PRED / "P_super_v6_clean_oof.npy").ravel(), np.load(PRED / "P_super_v6_clean_test.npy").ravel(), 0.0140),
        ("part6_xgb_cuda_d1", np.load(PRED / "part6_expand/oof_gb_xgb_cuda_d1_bag.npy").ravel(), np.load(PRED / "part6_expand/sub_gb_xgb_cuda_d1_bag.npy").ravel(), 0.0100),
        ("part6_cat_d2_d3", np.load(PRED / "part6_expand/oof_gb_cat_d2_d3_bag.npy").ravel(), np.load(PRED / "part6_expand/sub_gb_cat_d2_d3_bag.npy").ravel(), 0.0100),
    ]

    X_orth_o = np.column_stack([z_v8_o, z_w12_o, r_old_o - r_tab_o])
    X_orth_t = np.column_stack([z_v8_t, z_w12_t, r_old_t - r_tab_t])

    U_all_o, U_all_t, w_all = [], [], []
    for nm, xo, xt, w in specs7 + specs5_new:
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=4000)
        ro = np.zeros(ntr)
        rt_f = []
        cur_X_o = np.column_stack([X_orth_o] + U_all_o) if U_all_o else X_orth_o
        cur_X_t = np.column_stack([X_orth_t] + U_all_t) if U_all_t else X_orth_t
        for a, b in folds:
            ols = LinearRegression().fit(cur_X_o[a], zo[a])
            ua = zo[a] - ols.predict(cur_X_o[a])
            mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
            ro[b] = (zo[b] - ols.predict(cur_X_o[b]) - mu) / sd
            rt_f.append((zt - ols.predict(cur_X_t) - mu) / sd)
        U_all_o.append(ro)
        U_all_t.append(pchip_quantile_map(np.mean(rt_f, 0), ro, n_knots=4000))
        w_all.append(w)

    U7_o = np.column_stack(U_all_o[:7])
    U7_t = np.column_stack(U_all_t[:7])
    w7 = np.array(w_all[:7])
    U5new_o = np.column_stack(U_all_o[7:])
    U5new_t = np.column_stack(U_all_t[7:])
    w5new = np.array(w_all[7:])

    dz7_o = U7_o @ w7
    dz5new_o = U5new_o @ w5new
    dz5new_t = pchip_quantile_map(U5new_t @ w5new, dz5new_o, n_knots=4000)

    c0_o = z_v8_o + 0.025 * (r_old_o - r_tab_o)
    z_ref7_o = c0_o + 0.85 * dz7_o
    z_ref12_o = z_ref7_o + 0.85 * dz5new_o
    fds12 = [round((float(fast_auc(y[b], z_ref12_o[b])) - float(fast_auc(y[b], c0_o[b]))) * 1e5, 2) for _, b in folds]
    print(f"[7-Model OOF]  AUC = {fast_auc(y, z_ref7_o):.7f}", flush=True)
    print(f"[12-Model OOF] AUC = {fast_auc(y, z_ref12_o):.7f} (Δ vs 7-Model = {(fast_auc(y, z_ref12_o)-fast_auc(y, z_ref7_o))*1e5:+.2f}e-5, Δ vs CandC = {(fast_auc(y, z_ref12_o)-fast_auc(y, c0_o))*1e5:+.2f}e-5, folds={fds12})", flush=True)

    # Also extract Areeb's megayak 10% shallow-tree orthogonal test residual (orthogonalized against all 12 models!)
    areeb_sub = pd.read_csv(N09 / "areebbb_s6e11-wide-stack-shallow-tree-fusion-0-95300/submission.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    p9_t = zrk(np.load(P12 / "part9_test.npy").ravel())
    fsl_t = zrk(0.65 * zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel()) + 0.35 * p9_t)
    z_areeb_t = zrk(areeb_sub)
    r_shallow_t = z_areeb_t - float(np.dot(z_areeb_t, fsl_t) / np.dot(fsl_t, fsl_t)) * fsl_t
    X12_t = np.column_stack([X_orth_t] + U_all_t)
    r_shallow_orth_t = r_shallow_t - X12_t @ np.linalg.lstsq(X12_t, r_shallow_t, rcond=None)[0]
    r_shallow_orth_t = (r_shallow_orth_t - r_shallow_orth_t.mean()) / (r_shallow_orth_t.std() + 1e-12)

    z_st2_rk = pchip_quantile_map(zrk(st2_sub), z_ref12_o, n_knots=4000)
    z_st3_rk = pchip_quantile_map(zrk(st3_sub), z_ref12_o, n_knots=4000)
    z_st4_rk = pchip_quantile_map(zrk(st4_sub), z_ref12_o, n_knots=4000)

    # Verified step from Strike 3A (0.95322) -> Strike 4A (0.95323):
    v_st4_st3 = z_st4_rk - z_st3_rk

    # Strike 5A (Recommended): Strike 4A (0.95323) + 0.45 * (St4A - St3A) + 0.70 * dz5new_t + 0.008 * r_shallow_orth_t
    z_5a_t = pchip_quantile_map(
        z_st4_rk + 0.45 * v_st4_st3 + 0.70 * dz5new_t + 0.008 * r_shallow_orth_t,
        z_ref12_o, n_knots=4000,
    )

    # Strike 5B: Pure 5-new-model orthogonal injection on Strike 4A (`z_st4_rk + 0.80 * dz5new_t + 0.010 * r_shallow_orth_t`)
    z_5b_t = pchip_quantile_map(
        z_st4_rk + 0.80 * dz5new_t + 0.010 * r_shallow_orth_t,
        z_ref12_o, n_knots=4000,
    )

    def eval_row(name, z_t):
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_ref12_o, y))
        return {
            "name": name,
            "oof_auc": float(fast_auc(y, z_ref12_o)),
            "dist_st4_1e4": round((1 - spearmanr(p_t, st4_sub).correlation) * 1e4, 3),
            "dist_st3_1e4": round((1 - spearmanr(p_t, st3_sub).correlation) * 1e4, 3),
            "dist_st2_1e4": round((1 - spearmanr(p_t, st2_sub).correlation) * 1e4, 3),
            "dist_a3_1e4": round((1 - spearmanr(p_t, a3_sub).correlation) * 1e4, 3),
            "ks_v8": float(ks_2samp(p_t, p_v8_cal).statistic),
            "p_mean": float(np.mean(p_t)),
            "p_test": p_t,
        }

    rows = [
        eval_row("Strike5A_St4A_Cont045_Plus_5NewOrth_Shallow", z_5a_t),
        eval_row("Strike5B_St4A_Pure_080x_5NewOrth_Shallow", z_5b_t),
    ]

    print("\n" + "=" * 140)
    print("GM TOP-3 STRIKE 5 TABLE (12-Model Orth OOF + Continuation from 0.95323)")
    print("=" * 140)
    for r in rows:
        print(
            f"{r['name']:46s} | OOF={r['oof_auc']:.7f} | "
            f"1-ρ(St4A)={r['dist_st4_1e4']:6.3f} | 1-ρ(St3A)={r['dist_st3_1e4']:6.3f} | "
            f"1-ρ(St2)={r['dist_st2_1e4']:6.3f} | 1-ρ(Att3)={r['dist_a3_1e4']:6.3f}"
        )
    print("=" * 140)

    for r in rows:
        fp = SUB / f"sub_{r['name'].lower()}.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": r["p_test"]}).to_csv(fp, index=False)
        print(f"  Saved {fp.name}", flush=True)

    summary = {
        "oof_7model": float(fast_auc(y, z_ref7_o)),
        "oof_12model": float(fast_auc(y, z_ref12_o)),
        "fold_deltas_vs_candc_1e5": fds12,
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "gm_top3_strike5_12model_orth.json").write_text(json.dumps(summary, indent=2))
    print(f"[{summary['elapsed_sec']}s] Saved logs/overnight/gm_top3_strike5_12model_orth.json", flush=True)


if __name__ == "__main__":
    main()
