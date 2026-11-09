#!/usr/bin/env python3
"""Day-2 Retake MaxLeap: reclaim #1 from 0.95332 (need 0.95334+).

Uses the research-locked SoftPseudo family pack
  soft_diff=0.045 + idf_p3=0.0225 + gb_soft_bag3=0.025 + soft_clean035=0.010
which delivered nested ΔOOF = +19.67e-5 (5/5) on the 7M stack.

Aggressive scales (past Slot I/J/K) + soft@0.07 pure + family+continuation:
  sub_retake_R1_softfamily_x080.csv   # main retake probe
  sub_retake_R2_softfamily_x100.csv   # full family
  sub_retake_R3_softdiff_w07_x070.csv
  sub_retake_R4_softfamily_x070_cont.csv
  sub_retake_R5_softfamily_x090_cont.csv
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
W_SOFT, W_IDF, W_GB, W_C035 = 0.045, 0.0225, 0.025, 0.010


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

    return z_lin + float(minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded").x)


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
    ro = np.zeros(len(zo), dtype=np.float64)
    rt_f = []
    for a, b in folds:
        ols = LinearRegression().fit(X_o[a], zo[a])
        ua = zo[a] - ols.predict(X_o[a])
        mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
        ro[b] = (zo[b] - ols.predict(X_o[b]) - mu) / sd
        rt_f.append((zt - ols.predict(X_t) - mu) / sd)
    return ro, pchip_quantile_map(np.mean(rt_f, axis=0), ro, n_knots=4000)


def main():
    print("[Boot] gm_day2_retake_maxleap.py — reclaim #1 from 0.95332 (target 0.95334+)...", flush=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)

    def lz(fn):
        s = pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
        return pchip_quantile_map(zrk(s), z_v8_o, n_knots=4000), s

    z_st8, st8_sub = lz("sub_gm_top1_strike8_claim_rank1.csv")
    z_st9, st9_sub = lz("sub_gm_top1_strike9_final_max.csv")

    # Prefer cached family residuals from 10b if present
    fam_o_path, fam_t_path = PRED / "day2_dz_softfamily_oof.npy", PRED / "day2_dz_softfamily_test.npy"
    soft_o_path, soft_t_path = PRED / "day2_dz_soft_oof.npy", PRED / "day2_dz_soft_test.npy"
    u3_o_path, u3_t_path = PRED / "day2_U3_oof.npy", PRED / "day2_U3_test.npy"

    if fam_o_path.exists() and fam_t_path.exists() and soft_o_path.exists() and u3_o_path.exists():
        print("[Cache] reusing day2_dz_softfamily + day2_U3 / soft residuals", flush=True)
        dz_fam_o = np.load(fam_o_path).astype(np.float64).ravel()
        dz_fam_t = np.load(fam_t_path).astype(np.float64).ravel()
        # soft unit = day2_dz_soft / 0.04 (locked earlier)
        u_soft_o = np.load(soft_o_path).astype(np.float64).ravel() / 0.04
        u_soft_t = np.load(soft_t_path).astype(np.float64).ravel() / 0.04
        # Rebuild z_ref_o cheaply for OOF audit via v8 + fam? Need exact base.
        # Use fam OOF delta vs reconstructing 7M — load from json if available
        base_auc = 0.9526611592872501
        fam_auc = float(fast_auc(y, zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel()) * 0 + dz_fam_o))
        # Better: reconstruct z_ref from components is heavy; use saved JSON numbers + relative AUC on z_v8 + scaled fam is wrong.
        # Full rebuild below for correctness when cache missing; with cache still rebuild z_ref lightly.
        rebuild = True
    else:
        rebuild = True

    if rebuild:
        print("[Rebuild] 7M + SoftFamily residuals...", flush=True)
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
            (np.load(PRED / "part11/lab4_tabular_nade_oof.npy").ravel(), np.load(PRED / "part11/lab4_tabular_nade_test.npy").ravel()),
            (sam_o, sam_t),
            (np.load(HARV / "oof_orig20k_knn.npy").ravel(), np.load(HARV / "test_orig20k_knn.npy").ravel()),
            (np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel(), np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel()),
            (np.load(PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy").ravel(), np.load(PRED / "part11/lab1_zjoint_graph_lgb4_test.npy").ravel()),
        ]
        X_orth_o = np.column_stack([z_v8_o, z_w12_o, r_old_o - r_tab_o])
        X_orth_t = np.column_stack([z_v8_t, z_w12_t, r_old_t - r_tab_t])
        U5_o, U5_t = [], []
        for xo, xt in specs5:
            zo = zrk(xo)
            zt = pchip_quantile_map(zrk(xt), zo, n_knots=4000)
            ro, rt = fold_orth_pair(zo, zt, X_orth_o, X_orth_t, folds)
            U5_o.append(ro); U5_t.append(rt)
        U5_o, U5_t = np.column_stack(U5_o), np.column_stack(U5_t)
        w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
        dz5_o = U5_o @ w5
        dz5_t = pchip_quantile_map(U5_t @ w5, dz5_o, n_knots=4000)

        specs2 = [
            (np.load(PRED / "ladder/glm_quad_oof.npy").ravel(), np.load(PRED / "ladder/glm_quad_test.npy").ravel()),
            (np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel(), np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel()),
        ]
        X7_o = np.column_stack([X_orth_o, U5_o])
        X7_t = np.column_stack([X_orth_t, U5_t])
        U2_o, U2_t = [], []
        for xo, xt in specs2:
            zo = zrk(xo)
            zt = pchip_quantile_map(zrk(xt), zo, n_knots=4000)
            ro, rt = fold_orth_pair(zo, zt, X7_o, X7_t, folds)
            U2_o.append(ro); U2_t.append(rt)
        U2_o, U2_t = np.column_stack(U2_o), np.column_stack(U2_t)
        w2 = np.array([0.0150, 0.0125])
        dz2_o = U2_o @ w2
        dz2_t = pchip_quantile_map(U2_t @ w2, dz2_o, n_knots=4000)

        z_ref_o = z_v8_o + 0.025 * (r_old_o - r_tab_o) + 0.85 * dz5_o + 0.85 * dz2_o
        base_auc = float(fast_auc(y, z_ref_o))
        X9_o = np.column_stack([X7_o, U2_o])
        X9_t = np.column_stack([X7_t, U2_t])

        fam = [
            (PRED / "glm_softpseudo_clean_diff_oof.npy", PRED / "glm_softpseudo_clean_diff_test.npy", W_SOFT),
            (PRED / "part11/lab1_idf_bayes_p3_m20_oof.npy", PRED / "part11/lab1_idf_bayes_p3_m20_test.npy", W_IDF),
            (PRED / "gb_softpseudo_clean_bag3_oof.npy", PRED / "gb_softpseudo_clean_bag3_test.npy", W_GB),
            (PRED / "glm_softpseudo_clean_w0.35_oof.npy", PRED / "glm_softpseudo_clean_w0.35_test.npy", W_C035),
        ]
        U_o, U_t, ww = [], [], []
        for op, tp, w in fam:
            zo = zrk(np.load(op).ravel())
            zt = pchip_quantile_map(zrk(np.load(tp).ravel()), zo, n_knots=4000)
            ro, rt = fold_orth_pair(zo, zt, X9_o, X9_t, folds)
            U_o.append(ro); U_t.append(rt); ww.append(w)
        U_o, U_t = np.column_stack(U_o), np.column_stack(U_t)
        ww = np.asarray(ww, dtype=np.float64)
        dz_fam_o = U_o @ ww
        dz_fam_t = pchip_quantile_map(U_t @ ww, dz_fam_o, n_knots=4000)
        u_soft_o, u_soft_t = U_o[:, 0], U_t[:, 0]

    fam_auc = float(fast_auc(y, z_ref_o + dz_fam_o))
    s07_auc = float(fast_auc(y, z_ref_o + 0.07 * u_soft_o))
    fds_fam = [
        round((float(fast_auc(y[b], (z_ref_o + dz_fam_o)[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5, 2)
        for _, b in folds
    ]
    print(f"[FAM] OOF={fam_auc:.7f} Δ={(fam_auc-base_auc)*1e5:+.2f}e-5 folds={fds_fam}", flush=True)
    print(f"[S07] OOF={s07_auc:.7f} Δ={(s07_auc-base_auc)*1e5:+.2f}e-5", flush=True)

    z_st9_ref = pchip_quantile_map(z_st9, z_ref_o, n_knots=4000)
    z_st8_ref = pchip_quantile_map(z_st8, z_ref_o, n_knots=4000)
    v_cont = z_st9_ref - z_st8_ref
    dz_s07_t = pchip_quantile_map(0.07 * u_soft_t, 0.07 * u_soft_o, n_knots=4000)

    def make_sub(name, z_t, note):
        z_q = pchip_quantile_map(z_t, z_ref_o, n_knots=4000)
        p = expit(calibrate_logit_to_prevalence(z_q, z_ref_o, y))
        assert len(p) == nte and np.isfinite(p).all()
        dist = (1 - float(spearmanr(p, st9_sub).correlation)) * 1e4
        rms = float(np.sqrt(np.mean((z_q - z_st9_ref) ** 2)))
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(SUB / name, index=False)
        row = {"file": name, "note": note, "dist_st9_1e4": round(dist, 3), "step_rms": round(rms, 5), "p_mean": float(p.mean())}
        print(f"  Saved {name}: 1-ρ(St9)={dist:.3f}e-4 rms={rms:.5f} | {note}", flush=True)
        return row

    rows = [
        make_sub("sub_retake_R1_softfamily_x080.csv", z_st9_ref + 0.80 * dz_fam_t,
                 f"RETAKE lead: St9+0.80*FAM ΔOOF@1={(fam_auc-base_auc)*1e5:+.2f}e-5"),
        make_sub("sub_retake_R2_softfamily_x100.csv", z_st9_ref + 1.00 * dz_fam_t,
                 f"RETAKE full: St9+1.00*FAM ΔOOF@1={(fam_auc-base_auc)*1e5:+.2f}e-5"),
        make_sub("sub_retake_R3_softdiff_w07_x070.csv", z_st9_ref + 0.70 * dz_s07_t,
                 f"RETAKE soft@0.07 x0.70 ΔOOF@1={(s07_auc-base_auc)*1e5:+.2f}e-5"),
        make_sub("sub_retake_R4_softfamily_x070_cont.csv", z_st9_ref + 0.70 * dz_fam_t + 0.40 * v_cont,
                 "RETAKE St9+0.70*FAM+0.40*(St9-St8)"),
        make_sub("sub_retake_R5_softfamily_x090_cont.csv", z_st9_ref + 0.90 * dz_fam_t + 0.50 * v_cont,
                 "RETAKE St9+0.90*FAM+0.50*(St9-St8) max push"),
    ]

    plan = {
        "target": "beat 0.95332 -> 0.95334+",
        "base_7m_oof": base_auc,
        "fam_oof": fam_auc,
        "fam_delta_1e5": round((fam_auc - base_auc) * 1e5, 2),
        "fam_folds_1e5": fds_fam,
        "s07_delta_1e5": round((s07_auc - base_auc) * 1e5, 2),
        "submit_order": ["R1", "R4", "R2", "R5", "R3"],
        "rows": rows,
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "gm_day2_retake_maxleap.json").write_text(json.dumps(plan, indent=2))
    (LOG / "RETAKE_SUBMIT_PLAN.md").write_text(
        "\n".join([
            "# Retake Plan — beat 0.95332 (need 0.95334+)",
            "",
            f"SoftFamily nested ΔOOF `{(fam_auc-base_auc)*1e5:+.2f}e-5` (5/5). Soft@0.07 Δ `{(s07_auc-base_auc)*1e5:+.2f}e-5`.",
            "",
            "## Submit order (when quota available)",
            "1. **R1** `sub_retake_R1_softfamily_x080.csv` — primary retake",
            "2. **R4** `sub_retake_R4_softfamily_x070_cont.csv` — family + cont",
            "3. **R2** full family if R1 gains but < 0.95334",
            "4. **R5** max push / **R3** soft@0.07 alternate",
            "",
            "```bash",
            'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_retake_R1_softfamily_x080.csv -m "Retake R1: St9+0.80*SoftFamily vs 0.95332"',
            'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_retake_R4_softfamily_x070_cont.csv -m "Retake R4: St9+0.70*SoftFamily+0.40cont"',
            "```",
            "",
        ])
    )
    print(f"[{plan['elapsed_sec']}s] Retake CSVs ready. Target 0.95334+.", flush=True)


if __name__ == "__main__":
    main()
