#!/usr/bin/env python3
"""Day-2 Strike 10 prep: genuine orthogonal-family leap from Strike9 (LB 0.95330).

Why these three (nested 5F OOF on top of the exact 7-model stack = 0.9526612):
  softpseudo_clean_diff : ΔOOF = +8.05e-5, 5/5 folds positive  << primary leap
  idf_bayes_p3_m20      : ΔOOF = +1.59e-5, 3/5 folds positive
  lab1_graph_glm        : ΔOOF = +0.55e-5, 4/5 folds positive

Produces ready-to-submit CSVs (do NOT submit until daily quota resets).

Trust-region note: pure continuation wins lived at 1-ρ ≈ 0.11–0.14e-4 / step_rms≈0.009.
New-family Strike2 lived at ≈0.51e-4. Full soft@0.04 is ≈2.7e-4 — too far for Slot1.
We therefore ship a scale ladder (0.25 / 0.40 / 0.55 / 1.00) and lead with mid-scale.
"""
from __future__ import annotations

import json
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
P12 = ROOT / "external/nov08_part12"
HARV = PRED / "fast_gpu_harvest"
N09 = ROOT / "external/nov09_top3_recon"
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


def fast_auc(y, s):
    return float(roc_auc_score(y, s))


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


def fold_orth_pair(zo, zt, X_o, X_t, folds):
    """Nested OLS residual of zo against X_o; map zt residual onto OOF residual scale."""
    ntr = len(zo)
    ro = np.zeros(ntr, dtype=np.float64)
    rt_f = []
    for a, b in folds:
        ols = LinearRegression().fit(X_o[a], zo[a])
        ua = zo[a] - ols.predict(X_o[a])
        mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
        ro[b] = (zo[b] - ols.predict(X_o[b]) - mu) / sd
        rt_f.append((zt - ols.predict(X_t) - mu) / sd)
    rt = pchip_quantile_map(np.mean(rt_f, axis=0), ro, n_knots=4000)
    return ro, rt


def main():
    print("[Boot] gm_day2_strike10_orth_leap.py — SoftPseudo + IDF + Graph leap from St9 (0.95330)...", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    # Live LB anchors (quantile-mapped to v8 OOF)
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)

    def lz_v8(fn):
        s = pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
        return pchip_quantile_map(zrk(s), z_v8_o, n_knots=4000), s

    z_st8, st8_sub = lz_v8("sub_gm_top1_strike8_claim_rank1.csv")
    z_st9, st9_sub = lz_v8("sub_gm_top1_strike9_final_max.csv")

    # Rebuild exact 7-model homogeneous OOF / test orth basis (Strike3 protocol)
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
        ro, rt = fold_orth_pair(zo, zt, X_orth_o, X_orth_t, folds)
        U5_o.append(ro)
        U5_t.append(rt)
        print(f"  [U5] {nm}: residual ready", flush=True)
    U5_o = np.column_stack(U5_o)
    U5_t = np.column_stack(U5_t)
    w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
    dz5_o = U5_o @ w5
    dz5_t = pchip_quantile_map(U5_t @ w5, dz5_o, n_knots=4000)

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
        ro, rt = fold_orth_pair(zo, zt, X7_o, X7_t, folds)
        U2_o.append(ro)
        U2_t.append(rt)
        print(f"  [U2] {nm}: residual ready", flush=True)
    U2_o = np.column_stack(U2_o)
    U2_t = np.column_stack(U2_t)
    w2 = np.array([0.0150, 0.0125])
    dz2_o = U2_o @ w2
    dz2_t = pchip_quantile_map(U2_t @ w2, dz2_o, n_knots=4000)

    z_ref_o = z_v8_o + 0.025 * (r_old_o - r_tab_o) + 0.85 * dz5_o + 0.85 * dz2_o
    z_ref_t = pchip_quantile_map(
        z_v8_t + 0.025 * (r_old_t - r_tab_t) + 0.85 * dz5_t + 0.85 * dz2_t,
        z_ref_o, n_knots=4000,
    )
    base_auc = float(fast_auc(y, z_ref_o))
    print(f"[7M base] OOF = {base_auc:.7f}", flush=True)

    # New Day-2 orthogonal family (project out full 7M + CandC control)
    X9_o = np.column_stack([X7_o, U2_o])
    X9_t = np.column_stack([X7_t, U2_t])
    new_specs = [
        ("softpseudo_diff", np.load(PRED / "glm_softpseudo_clean_diff_oof.npy").ravel(), np.load(PRED / "glm_softpseudo_clean_diff_test.npy").ravel()),
        ("idf_bayes_p3", np.load(PRED / "part11/lab1_idf_bayes_p3_m20_oof.npy").ravel(), np.load(PRED / "part11/lab1_idf_bayes_p3_m20_test.npy").ravel()),
        ("graph_glm", np.load(PRED / "part11/lab1_graph_glm_oof.npy").ravel(), np.load(PRED / "part11/lab1_graph_glm_test.npy").ravel()),
    ]
    U3_o, U3_t = [], []
    for nm, xo, xt in new_specs:
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=4000)
        ro, rt = fold_orth_pair(zo, zt, X9_o, X9_t, folds)
        U3_o.append(ro)
        U3_t.append(rt)
        d_solo = (float(fast_auc(y, z_ref_o + 0.025 * ro)) - base_auc) * 1e5
        fds = [
            round((float(fast_auc(y[b], (z_ref_o + 0.025 * ro)[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5, 2)
            for _, b in folds
        ]
        print(f"  [U3] {nm}: solo@0.025 ΔOOF={d_solo:+.2f}e-5 folds={fds}", flush=True)
    U3_o = np.column_stack(U3_o)
    U3_t = np.column_stack(U3_t)

    # Grid-search trio weights on nested OOF (small, positive-biased)
    print("[Grid] searching (w_soft, w_idf, w_graph)...", flush=True)
    best = None
    grid_soft = [0.010, 0.015, 0.020, 0.025, 0.030]
    grid_idf = [0.000, 0.008, 0.012, 0.016, 0.020]
    grid_graph = [0.000, 0.006, 0.010, 0.014]
    for ws in grid_soft:
        for wi in grid_idf:
            for wg in grid_graph:
                w = np.array([ws, wi, wg], dtype=np.float64)
                z_try = z_ref_o + U3_o @ w
                d = (float(fast_auc(y, z_try)) - base_auc) * 1e5
                fds = [
                    (float(fast_auc(y[b], z_try[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5
                    for _, b in folds
                ]
                pos = sum(1 for v in fds if v > 0)
                # Prefer: high ΔOOF, then more positive folds, then lower L1 weight
                key = (d, pos, -float(np.abs(w).sum()))
                if best is None or key > best[0]:
                    best = (key, w, d, pos, [round(v, 2) for v in fds])
    w3, d3, pos3, fds3 = best[1], best[2], best[3], best[4]
    print(f"[Grid winner] w={w3.tolist()} | ΔOOF={d3:+.2f}e-5 | pos={pos3}/5 | folds={fds3}", flush=True)

    # Softpseudo-only best weight (for Slot B / E)
    best_ws, best_ds, best_poss, best_fdss = 0.025, -1e9, 0, []
    for ws in np.linspace(0.008, 0.040, 33):
        z_try = z_ref_o + ws * U3_o[:, 0]
        d = (float(fast_auc(y, z_try)) - base_auc) * 1e5
        fds = [
            (float(fast_auc(y[b], z_try[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5
            for _, b in folds
        ]
        pos = sum(1 for v in fds if v > 0)
        if (d, pos) > (best_ds, best_poss):
            best_ws, best_ds, best_poss, best_fdss = ws, d, pos, [round(v, 2) for v in fds]
    print(f"[Soft-only] w={best_ws:+.4f} | ΔOOF={best_ds:+.2f}e-5 | pos={best_poss}/5 | folds={best_fdss}", flush=True)

    # Soft+Graph (drop idf which was only 3/5)
    best_sg, best_dsg, best_possg, best_fdssg = None, -1e9, 0, []
    for ws in [0.015, 0.020, 0.025, 0.030]:
        for wg in [0.000, 0.008, 0.012, 0.016]:
            w = np.array([ws, 0.0, wg])
            z_try = z_ref_o + U3_o @ w
            d = (float(fast_auc(y, z_try)) - base_auc) * 1e5
            fds = [
                (float(fast_auc(y[b], z_try[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5
                for _, b in folds
            ]
            pos = sum(1 for v in fds if v > 0)
            if (d, pos) > (best_dsg, best_possg):
                best_sg, best_dsg, best_possg, best_fdssg = w, d, pos, [round(v, 2) for v in fds]
    print(f"[Soft+Graph] w={best_sg.tolist()} | ΔOOF={best_dsg:+.2f}e-5 | pos={best_possg}/5 | folds={best_fdssg}", flush=True)

    dz3_full_o = U3_o @ w3
    dz3_full_t = pchip_quantile_map(U3_t @ w3, dz3_full_o, n_knots=4000)
    dz_soft_o = best_ws * U3_o[:, 0]
    dz_soft_t = pchip_quantile_map(best_ws * U3_t[:, 0], dz_soft_o, n_knots=4000)
    dz_sg_o = U3_o @ best_sg
    dz_sg_t = pchip_quantile_map(U3_t @ best_sg, dz_sg_o, n_knots=4000)

    new_oof_full = float(fast_auc(y, z_ref_o + dz3_full_o))
    new_oof_soft = float(fast_auc(y, z_ref_o + dz_soft_o))
    new_oof_sg = float(fast_auc(y, z_ref_o + dz_sg_o))
    print(
        f"[Final OOF] full={new_oof_full:.7f} (Δ={(new_oof_full-base_auc)*1e5:+.2f}e-5) | "
        f"soft={new_oof_soft:.7f} (Δ={(new_oof_soft-base_auc)*1e5:+.2f}e-5) | "
        f"soft+graph={new_oof_sg:.7f} (Δ={(new_oof_sg-base_auc)*1e5:+.2f}e-5)",
        flush=True,
    )

    # Map St9 into z_ref_o space, then inject Day-2 residuals
    z_st9_ref = pchip_quantile_map(z_st9, z_ref_o, n_knots=4000)
    z_st8_ref = pchip_quantile_map(z_st8, z_ref_o, n_knots=4000)
    v_cont = z_st9_ref - z_st8_ref

    def make_sub(name, z_t, note):
        z_q = pchip_quantile_map(z_t, z_ref_o, n_knots=4000)
        p = expit(calibrate_logit_to_prevalence(z_q, z_ref_o, y))
        assert len(p) == nte and np.isfinite(p).all()
        dist_st9 = (1 - float(spearmanr(p, st9_sub).correlation)) * 1e4
        dist_st8 = (1 - float(spearmanr(p, st8_sub).correlation)) * 1e4
        step_rms = float(np.sqrt(np.mean((z_q - z_st9_ref) ** 2)))
        out = SUB / name
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(out, index=False)
        row = {
            "file": name,
            "note": note,
            "dist_st9_1e4": round(dist_st9, 3),
            "dist_st8_1e4": round(dist_st8, 3),
            "step_rms": round(step_rms, 5),
            "p_mean": float(p.mean()),
            "ks_st9": float(ks_2samp(p, st9_sub).statistic),
        }
        print(
            f"  Saved {name}: 1-ρ(St9)={dist_st9:.3f}e-4 | 1-ρ(St8)={dist_st8:.3f}e-4 | "
            f"step_rms={step_rms:.5f} | mean={p.mean():.6f} | {note}",
            flush=True,
        )
        return row

    rows = []
    # Softpseudo is the dominant axis (grid zeroed graph; soft-only beat the trio).
    # Scale ladder keeps Slot1 inside a Strike2-like trust region.
    scales = [
        ("A", 0.40, "040", "lead probe — target 1-ρ(St9)≈0.4–0.7e-4"),
        ("B", 0.55, "055", "step-up if A gains"),
        ("C", 0.25, "025", "ultra-safe if A overshoots"),
        ("D", 1.00, "100", "full soft@OOF-opt (aggressive)"),
    ]
    for tag, sc, sc_tag, why in scales:
        rows.append(make_sub(
            f"sub_day2_slot{tag}_softpseudo_x{sc_tag}.csv",
            z_st9_ref + sc * dz_soft_t,
            f"St9 + {sc:.2f}*soft@{best_ws:.3f} (OOFΔ_soft={best_ds:+.2f}e-5 @1.0) | {why}",
        ))

    # Slot E — soft mid-scale + mild continuation (for 2-point LB gradient after A)
    rows.append(make_sub(
        "sub_day2_slotE_soft_x0p40_plus_cont.csv",
        z_st9_ref + 0.40 * dz_soft_t + 0.35 * v_cont,
        f"St9 + 0.40*soft@{best_ws:.3f} + 0.35*(St9-St8)",
    ))
    # Slot F — soft+idf at mid scale (grid liked idf with soft; graph→0)
    rows.append(make_sub(
        "sub_day2_slotF_soft_idf_x0p40.csv",
        z_st9_ref + 0.40 * dz3_full_t,
        f"St9 + 0.40*trio(w={w3.tolist()}, Δ={d3:+.2f}e-5 @1.0)",
    ))
    # Slot G — soft+graph mid scale
    rows.append(make_sub(
        "sub_day2_slotG_soft_graph_x0p40.csv",
        z_st9_ref + 0.40 * dz_sg_t,
        f"St9 + 0.40*soft+graph(w={best_sg.tolist()}, Δ={best_dsg:+.2f}e-5 @1.0)",
    ))
    # Slot H — pure continuation backup
    rows.append(make_sub(
        "sub_day2_slotH_st9_cont055.csv",
        z_st9_ref + 0.55 * v_cont,
        "St9 + 0.55*(St9-St8) continuation-only backup",
    ))

    # Persist residuals for tomorrow LB-gradient follow-ups
    np.save(PRED / "day2_dz3_full_oof.npy", dz3_full_o.astype(np.float32))
    np.save(PRED / "day2_dz3_full_test.npy", dz3_full_t.astype(np.float32))
    np.save(PRED / "day2_dz_soft_oof.npy", dz_soft_o.astype(np.float32))
    np.save(PRED / "day2_dz_soft_test.npy", dz_soft_t.astype(np.float32))
    np.save(PRED / "day2_U3_oof.npy", U3_o.astype(np.float32))
    np.save(PRED / "day2_U3_test.npy", U3_t.astype(np.float32))

    submit_cmds = {
        "slotA": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotA_softpseudo_x040.csv -m "Day2 SlotA: St9(0.95330)+0.40*SoftPseudoDiff orth"',
        "slotB": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotB_softpseudo_x055.csv -m "Day2 SlotB: St9+0.55*SoftPseudoDiff"',
        "slotC": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotC_softpseudo_x025.csv -m "Day2 SlotC: St9+0.25*SoftPseudoDiff safe"',
        "slotD": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotD_softpseudo_x100.csv -m "Day2 SlotD: St9+1.0*SoftPseudoDiff full"',
        "slotE": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotE_soft_x0p40_plus_cont.csv -m "Day2 SlotE: St9+0.40*Soft+0.35*(St9-St8)"',
        "slotF": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotF_soft_idf_x0p40.csv -m "Day2 SlotF: St9+0.40*(Soft+IDF)"',
        "slotG": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotG_soft_graph_x0p40.csv -m "Day2 SlotG: St9+0.40*(Soft+Graph)"',
        "slotH": 'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotH_st9_cont055.csv -m "Day2 SlotH: St9+0.55*(St9-St8) cont backup"',
    }
    plan = {
        "base_7m_oof": base_auc,
        "new_oof_full": new_oof_full,
        "new_oof_soft": new_oof_soft,
        "new_oof_soft_graph": new_oof_sg,
        "w3_full": w3.tolist(),
        "w3_delta_1e5": round(d3, 2),
        "w3_pos_folds": pos3,
        "w3_fold_deltas_1e5": fds3,
        "w_soft": float(best_ws),
        "w_soft_delta_1e5": round(best_ds, 2),
        "w_soft_pos_folds": best_poss,
        "w_soft_fold_deltas_1e5": best_fdss,
        "w_soft_graph": best_sg.tolist(),
        "w_soft_graph_delta_1e5": round(best_dsg, 2),
        "recommended_order": ["slotA", "slotB", "slotE", "slotF", "slotC", "slotD", "slotG", "slotH"],
        "rationale": (
            "Softpseudo_clean_diff is the only 5/5-fold family with huge nested OOF "
            f"(+{best_ds:.2f}e-5 at w={best_ws:.3f}). Full injection is ~2.7e-4 from St9 — "
            "too far vs proven 0.11–0.51e-4 window — so lead with 0.40x (SlotA), step to "
            "0.55x (SlotB) if A gains, then soft+cont (SlotE) for LB gradient. "
            "SlotF tests Soft+IDF mid-scale. SlotC/H are safe fallbacks; SlotD is full push."
        ),
        "submit_cmds": submit_cmds,
        "rows": rows,
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "gm_day2_strike10_orth_leap.json").write_text(json.dumps(plan, indent=2))
    (LOG / "DAY2_SUBMIT_PLAN.md").write_text(
        "\n".join([
            "# Day-2 Submit Plan (Strike 10 SoftPseudo Orth Leap)",
            "",
            f"Base 7M OOF: `{base_auc:.7f}` → Soft-only OOF: `{new_oof_soft:.7f}` "
            f"(Δ `{(new_oof_soft-base_auc)*1e5:+.2f}e-5`, pos `{best_poss}/5`).",
            f"Soft+IDF grid OOF: `{new_oof_full:.7f}` (Δ `{(new_oof_full-base_auc)*1e5:+.2f}e-5`).",
            "",
            "## Recommended order",
            "1. **Slot A** — `sub_day2_slotA_softpseudo_x040.csv` (0.40× softpseudo — lead)",
            "2. **Slot B** — `sub_day2_slotB_softpseudo_x055.csv` (0.55× if A gains)",
            "3. **Slot E** — `sub_day2_slotE_soft_x0p40_plus_cont.csv` (soft + cont gradient)",
            "4. **Slot F** — `sub_day2_slotF_soft_idf_x0p40.csv` (soft+IDF mid)",
            "5. Hold remaining for LB-gradient follow-up; use C/H if A regresses; D only if mid-scale wins big.",
            "",
            "## Submit commands (quota reset)",
            "```bash",
            submit_cmds["slotA"],
            submit_cmds["slotB"],
            submit_cmds["slotE"],
            "```",
            "",
            f"Generated in {plan['elapsed_sec']}s. Do **not** submit until daily quota resets.",
            "",
        ])
    )
    print(f"[{plan['elapsed_sec']}s] Saved logs/overnight/gm_day2_strike10_orth_leap.json + DAY2_SUBMIT_PLAN.md", flush=True)
    print("READY for tomorrow. Do NOT submit today (0 remaining).", flush=True)


if __name__ == "__main__":
    main()
