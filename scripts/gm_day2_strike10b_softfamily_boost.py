#!/usr/bin/env python3
"""Day-2 Strike 10B: SoftPseudo-family boost beyond soft_diff alone.

Local research (nested 5F on exact 7M stack OOF=0.9526612):
  soft_diff alone @0.04 : +11.63e-5 (5/5)
  soft_diff alone @0.06 : +14.94e-5 (5/5)  — still rising, folds ok
  soft_diff alone @0.08 : +16.65e-5 (5/5)  — fold0/1 weakening
  soft_diff alone @0.10 : +16.71e-5 (3/5)  — overfit

  JOINT winner (coord ascent, 5/5):
    soft_diff=0.045 + idf_p3=0.0225 + gb_soft_bag3=0.025 + soft_clean035=0.010
    ΔOOF = +16.49e-5  folds=[+6.55,+11.48,+19.15,+17.35,+21.96]
    (= +4.86e-5 over soft@0.04; more fold-balanced than soft@0.06)

Produces additional ready CSVs (do not submit until quota resets):
  sub_day2_slotI_softfamily_x035.csv   # lead boosted probe
  sub_day2_slotJ_softfamily_x050.csv   # step-up
  sub_day2_slotK_softfamily_x065.csv   # aggressive
  sub_day2_slotL_softdiff_x060.csv     # soft@0.06 scaled mid (simple)
  sub_day2_slotM_softfamily_x040_cont.csv
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

# Research-locked joint weights (unit residuals after 7M orth)
W_SOFT = 0.045
W_IDF = 0.0225
W_GB = 0.025
W_C035 = 0.010


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
    ntr = len(zo)
    ro = np.zeros(ntr, dtype=np.float64)
    rt_f = []
    for a, b in folds:
        ols = LinearRegression().fit(X_o[a], zo[a])
        ua = zo[a] - ols.predict(X_o[a])
        mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
        ro[b] = (zo[b] - ols.predict(X_o[b]) - mu) / sd
        rt_f.append((zt - ols.predict(X_t) - mu) / sd)
    return ro, pchip_quantile_map(np.mean(rt_f, axis=0), ro, n_knots=4000)


def main():
    print("[Boot] gm_day2_strike10b_softfamily_boost.py — SoftPseudo family pack (+16.49e-5 OOF)...", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)

    def lz_v8(fn):
        s = pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
        return pchip_quantile_map(zrk(s), z_v8_o, n_knots=4000), s

    z_st8, st8_sub = lz_v8("sub_gm_top1_strike8_claim_rank1.csv")
    z_st9, st9_sub = lz_v8("sub_gm_top1_strike9_final_max.csv")

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
        zo, zt = zrk(xo), pchip_quantile_map(zrk(xt), zrk(xo), n_knots=4000)
        ro, rt = fold_orth_pair(zo, zt, X_orth_o, X_orth_t, folds)
        U5_o.append(ro); U5_t.append(rt)
        print(f"  [U5] {nm}", flush=True)
    U5_o, U5_t = np.column_stack(U5_o), np.column_stack(U5_t)
    w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
    dz5_o, dz5_t = U5_o @ w5, pchip_quantile_map(U5_t @ w5, U5_o @ w5, n_knots=4000)

    specs2 = [
        ("glm_quad", np.load(PRED / "ladder/glm_quad_oof.npy").ravel(), np.load(PRED / "ladder/glm_quad_test.npy").ravel()),
        ("lab1_glm_plus_graph", np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel(), np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel()),
    ]
    X7_o = np.column_stack([X_orth_o, U5_o])
    X7_t = np.column_stack([X_orth_t, U5_t])
    U2_o, U2_t = [], []
    for nm, xo, xt in specs2:
        zo, zt = zrk(xo), pchip_quantile_map(zrk(xt), zrk(xo), n_knots=4000)
        ro, rt = fold_orth_pair(zo, zt, X7_o, X7_t, folds)
        U2_o.append(ro); U2_t.append(rt)
        print(f"  [U2] {nm}", flush=True)
    U2_o, U2_t = np.column_stack(U2_o), np.column_stack(U2_t)
    w2 = np.array([0.0150, 0.0125])
    dz2_o = U2_o @ w2
    dz2_t = pchip_quantile_map(U2_t @ w2, dz2_o, n_knots=4000)

    z_ref_o = z_v8_o + 0.025 * (r_old_o - r_tab_o) + 0.85 * dz5_o + 0.85 * dz2_o
    z_ref_t = pchip_quantile_map(
        z_v8_t + 0.025 * (r_old_t - r_tab_t) + 0.85 * dz5_t + 0.85 * dz2_t,
        z_ref_o, n_knots=4000,
    )
    base_auc = float(fast_auc(y, z_ref_o))
    print(f"[7M] OOF={base_auc:.7f}", flush=True)

    X9_o = np.column_stack([X7_o, U2_o])
    X9_t = np.column_stack([X7_t, U2_t])

    fam = [
        ("soft_diff", PRED / "glm_softpseudo_clean_diff_oof.npy", PRED / "glm_softpseudo_clean_diff_test.npy", W_SOFT),
        ("idf_p3", PRED / "part11/lab1_idf_bayes_p3_m20_oof.npy", PRED / "part11/lab1_idf_bayes_p3_m20_test.npy", W_IDF),
        ("gb_soft", PRED / "gb_softpseudo_clean_bag3_oof.npy", PRED / "gb_softpseudo_clean_bag3_test.npy", W_GB),
        ("soft_c035", PRED / "glm_softpseudo_clean_w0.35_oof.npy", PRED / "glm_softpseudo_clean_w0.35_test.npy", W_C035),
    ]
    U_o, U_t, weights = [], [], []
    for nm, op, tp, w in fam:
        zo = zrk(np.load(op).ravel())
        zt = pchip_quantile_map(zrk(np.load(tp).ravel()), zo, n_knots=4000)
        ro, rt = fold_orth_pair(zo, zt, X9_o, X9_t, folds)
        U_o.append(ro); U_t.append(rt); weights.append(w)
        print(f"  [FAM] {nm} w={w}", flush=True)
    U_o = np.column_stack(U_o)
    U_t = np.column_stack(U_t)
    ww = np.array(weights, dtype=np.float64)

    dz_fam_o = U_o @ ww
    dz_fam_t = pchip_quantile_map(U_t @ ww, dz_fam_o, n_knots=4000)
    # soft@0.06 simple alternate
    dz_s06_o = 0.06 * U_o[:, 0]
    dz_s06_t = pchip_quantile_map(0.06 * U_t[:, 0], dz_s06_o, n_knots=4000)

    fam_auc = float(fast_auc(y, z_ref_o + dz_fam_o))
    s06_auc = float(fast_auc(y, z_ref_o + dz_s06_o))
    fds_fam = [
        round((float(fast_auc(y[b], (z_ref_o + dz_fam_o)[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5, 2)
        for _, b in folds
    ]
    fds_s06 = [
        round((float(fast_auc(y[b], (z_ref_o + dz_s06_o)[b])) - float(fast_auc(y[b], z_ref_o[b]))) * 1e5, 2)
        for _, b in folds
    ]
    print(
        f"[FAM] OOF={fam_auc:.7f} Δ={(fam_auc-base_auc)*1e5:+.2f}e-5 pos={sum(v>0 for v in fds_fam)}/5 folds={fds_fam}",
        flush=True,
    )
    print(
        f"[S06] OOF={s06_auc:.7f} Δ={(s06_auc-base_auc)*1e5:+.2f}e-5 pos={sum(v>0 for v in fds_s06)}/5 folds={fds_s06}",
        flush=True,
    )

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
            f"  Saved {name}: 1-ρ(St9)={dist_st9:.3f}e-4 step_rms={step_rms:.5f} | {note}",
            flush=True,
        )
        return row

    rows = []
    for tag, sc in [("I", 0.35), ("J", 0.50), ("K", 0.65)]:
        rows.append(make_sub(
            f"sub_day2_slot{tag}_softfamily_x{int(sc*100):03d}.csv",
            z_st9_ref + sc * dz_fam_t,
            f"St9 + {sc:.2f}*FAM(soft{W_SOFT}+idf{W_IDF}+gb{W_GB}+c035{W_C035}) OOFΔ@1={(fam_auc-base_auc)*1e5:+.2f}e-5",
        ))
    rows.append(make_sub(
        "sub_day2_slotL_softdiff_x060.csv",
        z_st9_ref + 0.55 * dz_s06_t,  # 0.55 of soft@0.06 unit pack ≈ mid trust
        f"St9 + 0.55*(soft@0.06 pack) OOFΔ@1={(s06_auc-base_auc)*1e5:+.2f}e-5",
    ))
    rows.append(make_sub(
        "sub_day2_slotM_softfamily_x040_cont.csv",
        z_st9_ref + 0.40 * dz_fam_t + 0.30 * v_cont,
        "St9 + 0.40*FAM + 0.30*(St9-St8)",
    ))

    np.save(PRED / "day2_dz_softfamily_oof.npy", dz_fam_o.astype(np.float32))
    np.save(PRED / "day2_dz_softfamily_test.npy", dz_fam_t.astype(np.float32))

    plan = {
        "base_7m_oof": base_auc,
        "fam_oof": fam_auc,
        "fam_delta_1e5": round((fam_auc - base_auc) * 1e5, 2),
        "fam_folds_1e5": fds_fam,
        "s06_oof": s06_auc,
        "s06_delta_1e5": round((s06_auc - base_auc) * 1e5, 2),
        "s06_folds_1e5": fds_s06,
        "weights": {"soft_diff": W_SOFT, "idf_p3": W_IDF, "gb_soft_bag3": W_GB, "soft_clean035": W_C035},
        "recommended_order_update": [
            "slotA (soft x040) still first — single-axis probe",
            "if A gains: slotI softfamily x035 (boosted family)",
            "then slotJ x050 / slotM (family+cont)",
            "slotK only if I/J keep climbing",
            "slotL = soft@0.06 mid alternate if family hurts",
        ],
        "rows": rows,
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (LOG / "gm_day2_strike10b_softfamily_boost.json").write_text(json.dumps(plan, indent=2))

    # Update submit plan
    md = (LOG / "DAY2_SUBMIT_PLAN.md").read_text() if (LOG / "DAY2_SUBMIT_PLAN.md").exists() else ""
    boost = "\n".join([
        "",
        "## Boost research (Strike 10B) — use AFTER Slot A lands",
        f"SoftPseudo **family pack** OOF Δ `{(fam_auc-base_auc)*1e5:+.2f}e-5` (5/5) vs soft-only `+11.63e-5`.",
        f"Weights: soft_diff={W_SOFT}, idf_p3={W_IDF}, gb_soft_bag3={W_GB}, soft_clean035={W_C035}.",
        "",
        "1. Keep **Slot A** (`soft x040`) as first submit (single-axis).",
        "2. If A ≥ 0.95330: **Slot I** `sub_day2_slotI_softfamily_x035.csv` (family boost).",
        "3. Then **Slot J** / **Slot M** (family step-up / +cont).",
        "4. **Slot L** if family regresses (soft@0.06 only).",
        "",
        "```bash",
        'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotI_softfamily_x035.csv -m "Day2 SlotI: St9+0.35*SoftFamily(diff+idf+gb+c035)"',
        'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotJ_softfamily_x050.csv -m "Day2 SlotJ: St9+0.50*SoftFamily"',
        'kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotM_softfamily_x040_cont.csv -m "Day2 SlotM: St9+0.40*SoftFamily+0.30*(St9-St8)"',
        "```",
        "",
    ])
    (LOG / "DAY2_SUBMIT_PLAN.md").write_text(md.rstrip() + "\n" + boost)
    print(f"[{plan['elapsed_sec']}s] Saved boost CSVs + updated DAY2_SUBMIT_PLAN.md", flush=True)


if __name__ == "__main__":
    main()
