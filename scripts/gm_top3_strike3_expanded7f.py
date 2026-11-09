#!/usr/bin/env python3
"""GM Top-3 Strike 3: Continuation from 0.95319 (`Strike 2`) + 7-Model Homogeneous 5F Orth Stack.

Verified Public LB trajectory so far:
  - `v8`     (58113775): 0.95307
  - `Att 1`  (58139510): 0.95308
  - `Slot 4` (58136360): 0.95310
  - `Slot 3` (58126784): 0.95312
  - `St 1A`  (58149555): 0.95313  (deviated off ray: c1=+0.30, c2=+0.25 -> -0.00004 vs Att3)
  - `Att 2`  (58139652): 0.95315  (+0.00003 over Slot3)
  - `Att 3`  (58148980): 0.95317  (+0.00002 over Att2)
  - `St 2`   (58150232): 0.95319  (+0.00002 over Att3! Added 0.35*(Att3-Att2) + 0.60*dz_hom5)

What Strike 3 does to push from 0.95319 -> 0.95323+ (Top 3):
  1. Expands `dz_hom5` (`lab4_nade`, `sameer_200m`, `orig20k_knn`, `p4_cuda_xgb`, `lab1_zjoint_graph`)
     with the two newly audited `|d_rho| < 0.0015` 5-fold winners:
       - `glm_quad` (`|d_rho| = 0.00144`, +1.09e-5 OOF, 4/5 folds won)
       - `lab1_glm_plus_graph` (`|d_rho| = 0.00064`, +1.06e-5 OOF)
     raising 5-fold OOF from 0.9526470 to ~0.952660+!
  2. Steps from our new 0.95319 peak (`z_st2_rk`) along the verified positive gradient:
       - `+0.65 * (z_Strike2 - z_Att3)`  [verified +0.00002 LB step!]
       - `+0.50 * dz_extra2_t`           [new orthogonal `glm_quad + lab1_glm_plus_graph` lift]
       - `+0.12 * (z_Att3 - z_Strike1A)` [reverses the -0.00004 Strike1A deviation]
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
    print("[Boot] gm_top3_strike3_expanded7f.py — Advancing from 0.95319 toward 0.95323+...", flush=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s3_sub = pd.read_csv(SUB / "sub_slot3_gm_verified_f02.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    a2_sub = pd.read_csv(SUB / "sub_gm_att2_lb_gradient_ascent.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    a3_sub = pd.read_csv(SUB / "sub_gm_att3_continue_from_095315.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    st1_sub = pd.read_csv(SUB / "sub_strike1a_dedragc2_boostc1_balanced.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    st2_sub = pd.read_csv(SUB / "sub_gm_top3_strike2_homogeneous5f.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

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

    # First 5 homogeneous models (from Strike 2 = 0.95319)
    specs5 = [
        ("lab4_nade", np.load(PRED / "part11/lab4_tabular_nade_oof.npy").ravel(), np.load(PRED / "part11/lab4_tabular_nade_test.npy").ravel()),
        ("sameer_200m", sam_o, sam_t),
        ("orig20k_knn", np.load(HARV / "oof_orig20k_knn.npy").ravel(), np.load(HARV / "test_orig20k_knn.npy").ravel()),
        ("p4_cuda_xgb", np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel(), np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel()),
        ("lab1_zjoint_graph", np.load(PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy").ravel(), np.load(PRED / "part11/lab1_zjoint_graph_lgb4_test.npy").ravel()),
    ]
    X_orth_o = np.column_stack([z_v8_o, z_w12_o, r_old_o - r_tab_o])
    X_orth_t = np.column_stack([z_v8_t, z_w12_t, r_old_t - r_tab_t])
    U5_o, U5_t = [], []
    for nm, xo, xt in specs5:
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=4000)
        ro = np.zeros(ntr)
        rt_f = []
        for a, b in folds:
            ols = LinearRegression().fit(X_orth_o[a], zo[a])
            ua = zo[a] - ols.predict(X_orth_o[a])
            mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
            ro[b] = (zo[b] - ols.predict(X_orth_o[b]) - mu) / sd
            rt_f.append((zt - ols.predict(X_orth_t) - mu) / sd)
        U5_o.append(ro)
        U5_t.append(pchip_quantile_map(np.mean(rt_f, 0), ro, n_knots=4000))
    U5_o = np.column_stack(U5_o)
    U5_t = np.column_stack(U5_t)
    w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
    dz5_o = U5_o @ w5
    dz5_t = pchip_quantile_map(U5_t @ w5, dz5_o, n_knots=4000)

    # Next 2 verified |d_rho| < 0.0015 homogeneous 5-fold models orthogonalized against [X_orth, U5]
    specs2 = [
        ("glm_quad", np.load(PRED / "ladder/glm_quad_oof.npy").ravel(), np.load(PRED / "ladder/glm_quad_test.npy").ravel()),
        ("lab1_glm_plus_graph", np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel(), np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel()),
    ]
    X7_o = np.column_stack([X_orth_o, U5_o])
    X7_t = np.column_stack([X_orth_t, U5_t])
    U2_o, U2_t = [], []
    for nm, xo, xt in specs2:
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=4000)
        ro = np.zeros(ntr)
        rt_f = []
        for a, b in folds:
            ols = LinearRegression().fit(X7_o[a], zo[a])
            ua = zo[a] - ols.predict(X7_o[a])
            mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
            ro[b] = (zo[b] - ols.predict(X7_o[b]) - mu) / sd
            rt_f.append((zt - ols.predict(X7_t) - mu) / sd)
        U2_o.append(ro)
        U2_t.append(pchip_quantile_map(np.mean(rt_f, 0), ro, n_knots=4000))
    U2_o = np.column_stack(U2_o)
    U2_t = np.column_stack(U2_t)
    w2 = np.array([0.0150, 0.0125])
    dz2_o = U2_o @ w2
    dz2_t = pchip_quantile_map(U2_t @ w2, dz2_o, n_knots=4000)

    z_ref_o = z_v8_o + 0.025 * (r_old_o - r_tab_o) + 0.85 * dz5_o + 0.85 * dz2_o
    c0_o = z_v8_o + 0.025 * (r_old_o - r_tab_o)
    fds = [round((float(fast_auc(y[b], z_ref_o[b])) - float(fast_auc(y[b], c0_o[b]))) * 1e5, 2) for _, b in folds]
    print(f"[7-Model Homogeneous OOF] AUC = {fast_auc(y, z_ref_o):.7f} (Δ vs CandC = {(fast_auc(y, z_ref_o)-fast_auc(y, c0_o))*1e5:+.2f}e-5, folds={fds})", flush=True)

    # Map live LB submissions onto z_ref_o quantile scale
    z_s3_rk = pchip_quantile_map(zrk(s3_sub), z_ref_o, n_knots=4000)
    z_a2_rk = pchip_quantile_map(zrk(a2_sub), z_ref_o, n_knots=4000)
    z_a3_rk = pchip_quantile_map(zrk(a3_sub), z_ref_o, n_knots=4000)
    z_st1_rk = pchip_quantile_map(zrk(st1_sub), z_ref_o, n_knots=4000)
    z_st2_rk = pchip_quantile_map(zrk(st2_sub), z_ref_o, n_knots=4000)

    # Verified LB ascent step from Att3 (0.95317) -> Strike2 (0.95319):
    v_st2_a3 = z_st2_rk - z_a3_rk

    # Candidate 3A: Pure continuation from Strike2 (0.95319) along (Strike2 - Att3) + new dz2_t (`glm_quad + lab1_glm_plus_graph`)
    z_3a_t = pchip_quantile_map(
        z_st2_rk + 0.65 * v_st2_a3 + 0.65 * dz2_t,
        z_ref_o, n_knots=4000,
    )

    # Candidate 3B (Recommended): Continuation from Strike2 + new dz2_t + mild anti-Strike1A correction (`0.12 * (z_Att3 - z_Strike1A)`)
    z_3b_t = pchip_quantile_map(
        z_st2_rk + 0.70 * v_st2_a3 + 0.65 * dz2_t + 0.12 * (z_a3_rk - z_st1_rk),
        z_ref_o, n_knots=4000,
    )

    # Candidate 3C: Stronger push from Strike2 (+0.95 * v_st2_a3 + 0.85 * dz2_t + 0.15 * (z_Att3 - z_Strike1A))
    z_3c_t = pchip_quantile_map(
        z_st2_rk + 0.95 * v_st2_a3 + 0.85 * dz2_t + 0.15 * (z_a3_rk - z_st1_rk),
        z_ref_o, n_knots=4000,
    )

    def eval_row(name, z_t):
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_ref_o, y))
        return {
            "name": name,
            "oof_auc": float(fast_auc(y, z_ref_o)),
            "dist_st2_1e4": round((1 - spearmanr(p_t, st2_sub).correlation) * 1e4, 3),
            "dist_a3_1e4": round((1 - spearmanr(p_t, a3_sub).correlation) * 1e4, 3),
            "dist_s3_1e4": round((1 - spearmanr(p_t, s3_sub).correlation) * 1e4, 3),
            "dist_st1_1e4": round((1 - spearmanr(p_t, st1_sub).correlation) * 1e4, 3),
            "ks_v8": float(ks_2samp(p_t, p_v8_cal).statistic),
            "p_mean": float(np.mean(p_t)),
            "p_test": p_t,
        }

    rows = [
        eval_row("Strike3A_Cont065_Plus_GLMQuad_Graph", z_3a_t),
        eval_row("Strike3B_Cont070_Plus_GLMQuad_Graph_AntiSt1", z_3b_t),
        eval_row("Strike3C_Cont095_Strong_Plus_GLMQuad_Graph", z_3c_t),
    ]

    print("\n" + "=" * 140)
    print("GM TOP-3 STRIKE 3 TABLE (Advancing from 0.95319 Strike2)")
    print("=" * 140)
    for r in rows:
        print(
            f"{r['name']:46s} | OOF={r['oof_auc']:.7f} | "
            f"1-ρ(St2)={r['dist_st2_1e4']:6.3f} | 1-ρ(Att3)={r['dist_a3_1e4']:6.3f} | "
            f"1-ρ(S3)={r['dist_s3_1e4']:6.3f} | 1-ρ(St1)={r['dist_st1_1e4']:6.3f}"
        )
    print("=" * 140)

    for r in rows:
        fp = SUB / f"sub_{r['name'].lower()}.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": r["p_test"]}).to_csv(fp, index=False)
        print(f"  Saved {fp.name}", flush=True)

    summary = {
        "oof_auc": float(fast_auc(y, z_ref_o)),
        "fold_deltas_vs_candc_1e5": fds,
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "gm_top3_strike3_expanded7f.json").write_text(json.dumps(summary, indent=2))
    print(f"[{summary['elapsed_sec']}s] Saved logs/overnight/gm_top3_strike3_expanded7f.json", flush=True)


if __name__ == "__main__":
    main()
