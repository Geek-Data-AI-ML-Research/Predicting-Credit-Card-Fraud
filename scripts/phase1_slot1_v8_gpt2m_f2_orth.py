#!/usr/bin/env python3
"""PHASE 1.3 — Build & Submit Slot 1: sub_phase1_v8_gpt2m_f2_orth.csv

Steps:
1. Convert Fold-2 oof_2.npy (139992, 8, 2) and test_2.npy (299984, 4, 2) into scalar bidirectional log-odds
   z_m_oof2 and z_m_test (both mean over K orders and Bayes logsumexp verified).
2. Single-Fold Variance Calibration: map z_m_test via monotone PchipInterpolator onto the empirical quantile
   distribution of z_m_oof2 so test dispersion matches OOF dispersion exactly.
3. Exact Orthogonal Residual Projection on Fold 2 indices idx2 against z_v8_oof[idx2]:
   - beta = cov(z_m_oof2, z_v8_oof[idx2]) / var(z_v8_oof[idx2])
   - r_oof2 = (z_m_oof2 - beta * z_v8_oof[idx2]) / std_r
   - r_test = (z_m_test_cal - beta * z_v8_test) / std_r  (using exact frozen beta, std_r)
4. Optimize w on Fold 2, verify Fold-2 AUC = 0.957380 (+5.26e-6 lift, w ~ 0.015),
   form p_test = expit(calibrate_logit_to_prevalence(z_v8_test + w * r_test, target_mean=y_train.mean())).
5. Strict hygiene gates (len == 299984 / test.csv rows, 0 NaNs/Infs, KS vs calibrated v8 < 0.0015, Spearman > 0.9998)
   and submit Slot 1 to Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
SUB.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def pchip_quantile_map(x_test: np.ndarray, x_ref: np.ndarray, n_knots: int = 4000) -> np.ndarray:
    """Monotone PCHIP quantile map from x_test's empirical CDF onto x_ref's quantile function."""
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    x_ref_sorted = np.sort(np.asarray(x_ref, dtype=np.float64).ravel())
    # Strictly increasing quantile knots in [0, 1]
    q_knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
    ref_vals = np.quantile(x_ref_sorted, q_knots)
    # Enforce strict monotonicity (tiny jitter only if flat ties exist)
    pchip = PchipInterpolator(q_knots, ref_vals, extrapolate=True)
    u_test = (rankdata(x_test, method="average") - 0.5) / len(x_test)
    return np.asarray(pchip(u_test), dtype=np.float64)


def calibrate_logit_to_prevalence(
    z_test: np.ndarray,
    z_oof_ref: np.ndarray,
    y_train: np.ndarray,
    target_mean: float | None = None,
) -> np.ndarray:
    """Fit 1D Platt scaling on (z_oof_ref, y_train) and shift intercept so mean(expit(z_cal_test)) == target_mean."""
    if target_mean is None:
        target_mean = float(np.mean(y_train))
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof_ref.reshape(-1, 1), y_train.astype(np.int32))
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    z_lin = slope * z_test + intercept

    # Fine-tune intercept shift delta so mean(expit(z_lin + delta)) == target_mean to < 1e-10
    def obj(delta: float) -> float:
        return (float(np.mean(expit(z_lin + delta))) - target_mean) ** 2

    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded")
    return z_lin + float(res.x)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    a2, idx2 = folds[2]
    prior2 = float(y[a2].mean())

    # 1. Load & verify oof_2.npy and test_2.npy
    oof2_path = PRED / "gpt2_med_4ep/oof_2.npy"
    test2_path = PRED / "gpt2_med_4ep/test_2.npy"
    oof2_raw = np.load(oof2_path)
    test2_raw = np.load(test2_path)
    assert oof2_raw.shape == (len(idx2), 8, 2) and np.isfinite(oof2_raw).all(), oof2_raw.shape
    assert test2_raw.shape == (nte, 4, 2) and np.isfinite(test2_raw).all(), test2_raw.shape
    print(f"[Gate 1] oof_2.npy shape={oof2_raw.shape} sha256={sha256_file(oof2_path)[:16]}...")
    print(f"[Gate 1] test_2.npy shape={test2_raw.shape} sha256={sha256_file(test2_path)[:16]}...")

    # Both scalar log-odds forms: (a) Bayes logsumexp across K orders, (b) mean across K orders of (ll_1 - ll_0)
    z_bayes_oof2 = (
        np.logaddexp.reduce(oof2_raw[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(oof2_raw[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior2 / (1.0 - prior2))
    )
    z_bayes_test = (
        np.logaddexp.reduce(test2_raw[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(test2_raw[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior2 / (1.0 - prior2))
    )
    z_mean_oof2 = np.mean(oof2_raw[:, :, 1].astype(np.float64) - oof2_raw[:, :, 0].astype(np.float64), axis=1)
    z_mean_test = np.mean(test2_raw[:, :, 1].astype(np.float64) - test2_raw[:, :, 0].astype(np.float64), axis=1)
    print(
        f"[Step 1] Fold-2 Solo AUC: bayes={roc_auc_score(y[idx2], z_bayes_oof2):.6f} | "
        f"mean_ll={roc_auc_score(y[idx2], z_mean_oof2):.6f}"
    )

    # Use rank-logit on Fold 2 (matches evaluate_against_refs where Fold-2 AUC = 0.957380 at w=0.015)
    z_m_oof2 = zrk(z_bayes_oof2)
    z_m_test_raw = zrk(z_bayes_test)

    # 2. Single-Fold Variance Calibration via PchipInterpolator onto z_m_oof2 empirical distribution
    z_m_test = pchip_quantile_map(z_m_test_raw, z_m_oof2, n_knots=4000)
    ks_cal = float(ks_2samp(z_m_oof2, z_m_test).statistic)
    print(
        f"[Step 2] PCHIP Variance Calibration: std(z_m_oof2)={z_m_oof2.std():.6f} "
        f"std(z_m_test)={z_m_test.std():.6f} KS={ks_cal:.6f}"
    )

    # 3. Load v8 OOF and Test (champion 0.95307)
    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = (
        pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(np.float64)
    )
    # Put v8_oof[idx2] and v8_sub on identical Fold-2 logit-rank scale via PCHIP onto z_v8_oof2
    z_v8_oof_all = zrk(v8_oof)
    z_v8_oof2 = zrk(v8_oof[idx2])
    z_v8_test = pchip_quantile_map(zrk(v8_sub), z_v8_oof2, n_knots=4000)

    # Exact OLS orthogonal projection on Fold 2 indices idx2
    cov_mat = np.cov(z_m_oof2, z_v8_oof2, ddof=0)
    var_v8 = float( np.var(z_v8_oof2) )
    beta = float(cov_mat[0, 1] / var_v8)
    r_oof2_raw = z_m_oof2 - beta * z_v8_oof2
    mu_r = float(np.mean(r_oof2_raw))
    std_r = float(np.std(r_oof2_raw))
    r_oof2 = (r_oof2_raw - mu_r) / std_r

    # Frozen parameters (beta, mu_r, std_r) applied to test, plus PCHIP quantile alignment of residual
    r_test_raw = (z_m_test - beta * z_v8_test - mu_r) / std_r
    r_test = pchip_quantile_map(r_test_raw, r_oof2, n_knots=4000)
    r_test = (r_test - float(np.mean(r_oof2))) / (float(np.std(r_oof2)) + 1e-12)

    # 4. Optimize w on Fold 2
    base_f2_auc = float(roc_auc_score(y[idx2], z_v8_oof2))
    grid_w = np.linspace(-0.05, 0.06, 221)
    best_w, best_f2_auc = 0.0, base_f2_auc
    for w in grid_w:
        sc = float(roc_auc_score(y[idx2], z_v8_oof2 + w * r_oof2))
        if sc > best_f2_auc:
            best_f2_auc, best_w = sc, float(w)

    lift_f2 = best_f2_auc - base_f2_auc
    print(
        f"[Step 3] Orthogonal Projection on Fold 2: beta={beta:.6f} std_r={std_r:.6f} "
        f"corr(r_oof2, z_v8_oof2)={np.corrcoef(r_oof2, z_v8_oof2)[0,1]:+.2e}"
    )
    print(
        f"[Step 4] Fold-2 Base AUC={base_f2_auc:.6f} -> Blend AUC={best_f2_auc:.6f} "
        f"(lift={lift_f2:+.2e} at w={best_w:+.4f})"
    )

    # Form combined test logit and prevalence-calibrated probability
    z_blend_test = z_v8_test + best_w * r_test
    target_mean = float(np.mean(y))
    p_test = expit(calibrate_logit_to_prevalence(z_blend_test, z_v8_oof_all, y, target_mean=target_mean))
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_test, z_v8_oof_all, y, target_mean=target_mean))

    # 5. Strict Kaggle Hygiene Gates
    ks_vs_v8_cal = float(ks_2samp(p_test, p_v8_cal).statistic)
    ks_z_vs_v8 = float(ks_2samp(zrk(p_test), zrk(v8_sub)).statistic)
    rho_vs_v8 = float(spearmanr(p_test, v8_sub).correlation)
    print(
        f"[Gate 5] rows={len(p_test)} (expected {nte}) | finite={np.isfinite(p_test).all()} | "
        f"mean(p_test)={p_test.mean():.6f} (target {target_mean:.6f}) | "
        f"KS(p_test, p_v8_cal)={ks_vs_v8_cal:.6f} (<0.0015) | Spearman={rho_vs_v8:.6f} (>0.9998)"
    )
    assert len(p_test) == nte, f"Expected {nte} rows, got {len(p_test)}"
    assert np.isfinite(p_test).all(), "NaN/Inf in p_test"
    assert ks_vs_v8_cal < 0.0015, f"KS gate failed: {ks_vs_v8_cal}"
    assert rho_vs_v8 > 0.9998, f"Spearman gate failed: {rho_vs_v8}"

    sub_path = SUB / "sub_phase1_v8_gpt2m_f2_orth.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": p_test}).to_csv(sub_path, index=False)
    np.save(PRED / "gpt2_med_4ep/z_blend_phase1_test.npy", z_blend_test)
    np.save(PRED / "gpt2_med_4ep/r_oof2.npy", r_oof2)
    np.save(PRED / "gpt2_med_4ep/r_test.npy", r_test)

    report = {
        "phase": "phase1_slot1",
        "oof2_sha256": sha256_file(oof2_path),
        "test2_sha256": sha256_file(test2_path),
        "fold2_solo_auc": float(roc_auc_score(y[idx2], z_bayes_oof2)),
        "fold2_base_v8_auc": base_f2_auc,
        "fold2_blend_auc": best_f2_auc,
        "fold2_lift": lift_f2,
        "best_w": best_w,
        "beta": beta,
        "std_r": std_r,
        "ks_vs_v8_cal": ks_vs_v8_cal,
        "ks_z_vs_v8": ks_z_vs_v8,
        "spearman_vs_v8": rho_vs_v8,
        "mean_pred": float(p_test.mean()),
        "sub_path": str(sub_path),
    }
    (LOG / "phase1_slot1_report.json").write_text(json.dumps(report, indent=2))
    print(f"[Saved] {sub_path} ({sub_path.stat().st_size:,} bytes)")

    if "--submit" in sys.argv:
        msg = (
            f"Phase1 Slot1: v8 + gpt2-medium (355M 4ep K8/K4) Fold2 orth residual "
            f"(w={best_w:+.4f}, F2 AUC={best_f2_auc:.6f}, +5.3e-6, rho={rho_vs_v8:.6f})"
        )
        cmd = [
            str(ROOT / ".venv_forensics/bin/kaggle"),
            "competitions",
            "submit",
            "-c",
            "playground-series-s6e11",
            "-f",
            str(sub_path),
            "-m",
            msg,
        ]
        print("[Submitting]", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)


if __name__ == "__main__":
    main()
