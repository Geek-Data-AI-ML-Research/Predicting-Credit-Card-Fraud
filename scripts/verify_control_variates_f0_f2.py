#!/usr/bin/env python3
"""30-Second Diagnostic Script: `verify_control_variates_f0_f2.py` (Local CPU only — does NOT touch PID 3021)

1. Verify the Paired LM Control Variate Law on Fold 0 and Fold 2 (`idx0` and `idx2`):
   - Compare single-orthogonalization against `z_v8` vs Double-Orthogonalization (Gram-Schmidt against
     `z_v8`, `z_lm12`, `z_gpt2_3ep`, and `z_gpt2_rev`) on Fold 0, Fold 2, and pooled (F0 + F2) with
     cross-fold transfer (`train on F0 -> eval on F2` and `train on F2 -> eval on F0`).
2. Test Double-Orthogonalization on 5-Fold `z_xgbf` (`u_xgbf = z_xgbf - OLS(z_v8, z_tabm, z_knn, z_lm12)`
   and subsets) across all 5 folds (`[F0, F1, F2, F3, F4]`).
3. Compare `sub_pure5f_cand_c.csv`, `sub_phase2_5_v8_meta_harvest.csv`, and `sub_v8_part12_fsl_prop_rect.csv`
   (KS and Spearman rank correlation + rank displacement).
"""
from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def pchip_quantile_map(x_test: np.ndarray, x_ref: np.ndarray, n_knots: int = 4000) -> np.ndarray:
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    x_ref_sorted = np.sort(np.asarray(x_ref, dtype=np.float64).ravel())
    q_knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
    ref_vals = np.quantile(x_ref_sorted, q_knots)
    pchip = PchipInterpolator(q_knots, ref_vals, extrapolate=True)
    u_test = (rankdata(x_test, method="average") - 0.5) / len(x_test)
    return np.asarray(pchip(u_test), dtype=np.float64)


def bayes_fold(oof_raw: np.ndarray, prior: float) -> np.ndarray:
    return (
        np.logaddexp.reduce(oof_raw[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(oof_raw[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def main() -> None:
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr = len(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    z_v8 = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_lm12 = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_masa = zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy").ravel())
    z_g3 = zrk(np.load(PRED / "gpt2_3ep/gpt2_3ep_oof.npy").ravel())
    z_rev = zrk(np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy").ravel())

    # -----------------------------------------------------------------------
    # PART 1: Verify Paired LM Control Variate Law on Fold 0 and Fold 2
    # -----------------------------------------------------------------------
    print("=" * 110)
    print("PART 1: Paired LM Control Variate & Gram-Schmidt Double-Orthogonalization on Fold 0 & Fold 2")
    print("=" * 110)

    a0, idx0 = folds[0]
    a2, idx2 = folds[2]
    oof0_raw = np.load(PRED / "gpt2_med_4ep/oof_0.npy")
    oof2_raw = np.load(PRED / "gpt2_med_4ep/oof_2.npy")
    z_m0 = zrk(bayes_fold(oof0_raw, float(y[a0].mean())))
    z_m2 = zrk(bayes_fold(oof2_raw, float(y[a2].mean())))

    # Stack F0 and F2 together (279,984 rows) to test cross-fold OLS + weight transfer (fit on F0 -> eval on F2, and vice versa)
    idx02 = np.concatenate([idx0, idx2])
    n0 = len(idx0)
    s0 = slice(0, n0)
    s2 = slice(n0, len(idx02))

    # Standardize within each fold so F0 and F2 share identical marginal scale
    z_m_02 = np.concatenate([(z_m0 - z_m0.mean()) / z_m0.std(), (z_m2 - z_m2.mean()) / z_m2.std()])
    y_02 = y[idx02]
    v8_02 = z_v8[idx02]
    lm12_02 = z_lm12[idx02]
    g3_02 = z_g3[idx02]
    masa_02 = z_masa[idx02]
    rev_02 = z_rev[idx02]

    base_f0 = float(roc_auc_score(y[idx0], v8_02[s0]))
    base_f2 = float(roc_auc_score(y[idx2], v8_02[s2]))
    base_02 = float(roc_auc_score(y_02, v8_02))
    print(f"Baseline v8 AUCs: Fold 0 = {base_f0:.6f} | Fold 2 = {base_f2:.6f} | Pooled(F0+F2) = {base_02:.6f}\n")

    basis_specs = [
        ("1. Single-Orth vs [z_v8]", np.column_stack([v8_02])),
        ("2. Double-Orth vs [z_v8, z_lm12]", np.column_stack([v8_02, lm12_02])),
        ("3. Double-Orth vs [z_v8, z_lm12, z_gpt2_3ep]", np.column_stack([v8_02, lm12_02, g3_02])),
        ("4. Full-LM-Orth vs [z_v8, lm12, g3, masa, rev]", np.column_stack([v8_02, lm12_02, g3_02, masa_02, rev_02])),
    ]

    w_grid = np.linspace(-0.08, 0.08, 321)
    for label, X_cov in basis_specs:
        # (a) Joint OLS on F0+F2, plus cross-fold OLS (fit on F2 -> apply to F0, fit on F0 -> apply to F2)
        ols_all = LinearRegression().fit(X_cov, z_m_02)
        u_all = z_m_02 - ols_all.predict(X_cov)
        u_all = (u_all - u_all.mean()) / u_all.std()

        # Cross-fitted residual u_cf (OLS fit on F2 predicts F0; OLS fit on F0 predicts F2)
        u_cf = np.empty_like(z_m_02)
        ols_f2 = LinearRegression().fit(X_cov[s2], z_m_02[s2])
        u0 = z_m_02[s0] - ols_f2.predict(X_cov[s0])
        u_cf[s0] = (u0 - u0.mean()) / u0.std()

        ols_f0 = LinearRegression().fit(X_cov[s0], z_m_02[s0])
        u2 = z_m_02[s2] - ols_f0.predict(X_cov[s2])
        u_cf[s2] = (u2 - u2.mean()) / u2.std()

        best_w0 = max(w_grid, key=lambda w: roc_auc_score(y_02[s0], v8_02[s0] + w * u_cf[s0]))
        d0_opt = float(roc_auc_score(y_02[s0], v8_02[s0] + best_w0 * u_cf[s0]) - base_f0)

        best_w2 = max(w_grid, key=lambda w: roc_auc_score(y_02[s2], v8_02[s2] + w * u_cf[s2]))
        d2_opt = float(roc_auc_score(y_02[s2], v8_02[s2] + best_w2 * u_cf[s2]) - base_f2)

        best_w02 = max(w_grid, key=lambda w: roc_auc_score(y_02, v8_02 + w * u_cf))
        d02_opt = float(roc_auc_score(y_02, v8_02 + best_w02 * u_cf) - base_02)
        d0_at_w02 = float(roc_auc_score(y_02[s0], v8_02[s0] + best_w02 * u_cf[s0]) - base_f0)
        d2_at_w02 = float(roc_auc_score(y_02[s2], v8_02[s2] + best_w02 * u_cf[s2]) - base_f2)

        # Out-of-fold transfer: apply best_w0 to F2 and best_w2 to F0
        pred_oof_transfer = np.empty_like(v8_02)
        pred_oof_transfer[s0] = v8_02[s0] + best_w2 * u_cf[s0]
        pred_oof_transfer[s2] = v8_02[s2] + best_w0 * u_cf[s2]
        d_transfer = float(roc_auc_score(y_02, pred_oof_transfer) - base_02)

        print(f"{label}")
        print(f"  OLS coefs = {[round(float(c), 4) for c in ols_all.coef_]} | R^2 = {ols_all.score(X_cov, z_m_02):.5f}")
        print(
            f"  F0 opt: w={best_w0:+.4f} (Δ={d0_opt*1e5:+.2f}e-5) | "
            f"F2 opt: w={best_w2:+.4f} (Δ={d2_opt*1e5:+.2f}e-5) | "
            f"Joint F0+F2 opt: w={best_w02:+.4f} (Δ_pool={d02_opt*1e5:+.2f}e-5, F0={d0_at_w02*1e5:+.2f}e-5, F2={d2_at_w02*1e5:+.2f}e-5) | "
            f"Cross-Fold Transfer Δ={d_transfer*1e5:+.2f}e-5"
        )

    # -----------------------------------------------------------------------
    # PART 2: Double-Orthogonalization of z_xgbf Across All 5 Folds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("PART 2: 5-Fold Double-Orthogonalization of z_xgbf (Exp 2C) vs [z_v8, z_tabm, z_knn, z_lm12, ...]")
    print("=" * 110)

    z_knn = zrk(np.load(HARV / "oof_orig20k_knn.npy").ravel())
    z_tabm = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_xgbf = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())

    base_5f = float(roc_auc_score(y, z_v8))
    base_folds = [float(roc_auc_score(y[b], z_v8[b])) for _, b in folds]

    xgbf_specs = [
        ("Ref: Pure_5F_Candidate_C (0.025*(r_xgbf - r_tabm))", "cand_c", None),
        ("A. u_xgbf = z_xgbf - OLS(z_v8, z_tabm)", "ols", [z_v8, z_tabm]),
        ("B. u_xgbf = z_xgbf - OLS(z_v8, z_tabm, z_knn)", "ols", [z_v8, z_tabm, z_knn]),
        ("C. u_xgbf = z_xgbf - OLS(z_v8, z_tabm, z_knn, z_lm12)", "ols", [z_v8, z_tabm, z_knn, z_lm12]),
        ("D. u_xgbf = z_xgbf - OLS(z_v8, z_tabm, z_knn, lm12, g3, rev, masa)", "ols", [z_v8, z_tabm, z_knn, z_lm12, z_g3, z_rev, z_masa]),
    ]

    for name, kind, cov_list in xgbf_specs:
        u_oof = np.zeros(ntr, dtype=np.float64)
        nested_oof = np.zeros(ntr, dtype=np.float64)
        ws = []
        for f, (a, b) in enumerate(folds):
            if kind == "cand_c":
                za = z_v8[a]
                var_a = float(np.var(za))
                bx = float(np.cov(z_xgbf[a], za, ddof=0)[0, 1] / var_a)
                bm = float(np.cov(z_tabm[a], za, ddof=0)[0, 1] / var_a)
                rx_a = z_xgbf[a] - bx * za
                rm_a = z_tabm[a] - bm * za
                rx_b = (z_xgbf[b] - bx * z_v8[b] - rx_a.mean()) / rx_a.std()
                rm_b = (z_tabm[b] - bm * z_v8[b] - rm_a.mean()) / rm_a.std()
                rx_a = (rx_a - rx_a.mean()) / rx_a.std()
                rm_a = (rm_a - rm_a.mean()) / rm_a.std()
                ua = rx_a - rm_a
                ub = rx_b - rm_b
                u_oof[b] = ub
                wf = 0.025
                ws.append(wf)
                nested_oof[b] = z_v8[b] + wf * ub
            else:
                Xa = np.column_stack([c[a] for c in cov_list])
                Xb = np.column_stack([c[b] for c in cov_list])
                ols = LinearRegression().fit(Xa, z_xgbf[a])
                ua_raw = z_xgbf[a] - ols.predict(Xa)
                mu, sd = float(ua_raw.mean()), float(ua_raw.std() + 1e-12)
                ua = (ua_raw - mu) / sd
                ub = (z_xgbf[b] - ols.predict(Xb) - mu) / sd
                u_oof[b] = ub
                res = minimize_scalar(
                    lambda w: -float(roc_auc_score(y[a], z_v8[a] + w * ua)),
                    bounds=(-0.06, 0.08),
                    method="bounded",
                )
                wf = float(res.x)
                ws.append(round(wf, 4))
                nested_oof[b] = z_v8[b] + wf * ub

        nested_auc = float(roc_auc_score(y, nested_oof))
        f_deltas = [float(roc_auc_score(y[b], nested_oof[b])) - bf for (_, b), bf in zip(folds, base_folds)]
        wins = sum(1 for d in f_deltas if d > 0)

        # Also evaluate at fixed w = mean(ws)
        mean_w = float(np.mean(ws))
        fixed_oof = z_v8 + mean_w * u_oof
        fixed_auc = float(roc_auc_score(y, fixed_oof))
        fixed_f_deltas = [float(roc_auc_score(y[b], fixed_oof[b])) - bf for (_, b), bf in zip(folds, base_folds)]
        fixed_wins = sum(1 for d in fixed_f_deltas if d > 0)

        print(f"{name}")
        print(
            f"  Nested OOF = {nested_auc:.7f} (Δ={nested_auc - base_5f:+.2e}) | wins={wins}/5 | "
            f"Per-Fold Δ (×1e-5) = {[round(d*1e5, 2) for d in f_deltas]} | ws={ws}"
        )
        if kind != "cand_c":
            print(
                f"  Fixed w={mean_w:+.4f} OOF = {fixed_auc:.7f} (Δ={fixed_auc - base_5f:+.2e}) | wins={fixed_wins}/5 | "
                f"Per-Fold Δ (×1e-5) = {[round(d*1e5, 2) for d in fixed_f_deltas]}"
            )

    # -----------------------------------------------------------------------
    # PART 3: Compare sub_pure5f_cand_c.csv vs sub_phase2_5_v8_meta_harvest.csv vs v8
    # -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("PART 3: Submission Comparison (sub_pure5f_cand_c vs sub_phase2_5_v8_meta_harvest vs v8)")
    print("=" * 110)

    s_v8 = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s_p25 = pd.read_csv(SUB / "sub_phase2_5_v8_meta_harvest.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s_pureC = pd.read_csv(SUB / "sub_pure5f_cand_c.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    pairs = [
        ("sub_pure5f_cand_c vs sub_phase2_5_v8_meta_harvest (Slot 2)", s_pureC, s_p25),
        ("sub_pure5f_cand_c vs v8_champion (0.95307)", s_pureC, s_v8),
        ("sub_phase2_5_v8_meta_harvest (Slot 2) vs v8_champion (0.95307)", s_p25, s_v8),
    ]
    for title, a, b in pairs:
        ks = float(ks_2samp(a, b).statistic)
        rho = float(spearmanr(a, b).correlation)
        ra = rankdata(a) / len(a)
        rb = rankdata(b) / len(b)
        mean_abs_rk = float(np.mean(np.abs(ra - rb)))
        p99_abs_rk = float(np.quantile(np.abs(ra - rb), 0.99))
        print(
            f"{title:62s} | KS={ks:.6f} | Spearman={rho:.7f} | "
            f"1-rho={(1-rho)*1e4:.3f}e-4 | mean|Δrk|={mean_abs_rk:.5f} | p99|Δrk|={p99_abs_rk:.5f}"
        )


if __name__ == "__main__":
    main()
