#!/usr/bin/env python3
"""PHASE 2.4 — Local Sync, Sequential Orthogonal Residual Audit, & Slot 2 Submission

1. Load `preds/fast_gpu_harvest/{oof,test}_orig20k_knn.npy`, `{oof,test}_tabm_5f.npy`, `{oof,test}_xgb_fast_harvest.npy`.
2. Run sequential Greedy Orthogonal Residual Selection on top of `v8`:
   - At each step, orthogonalize the candidate logit against the running ensemble logit (`r = z_cand - beta * z_curr`).
   - Optimize step weight via Nelder-Mead / golden-section on 5-fold CV (both nested out-of-fold and full OOF reported).
   - Gate: require BOTH full 5-fold OOF lift >= +1.0e-5 AND positive lift on at least 4 of 5 folds.
3. Re-apply the Fold-2 `gpt2-medium` orthogonal residual on top of the updated 5-fold stack.
4. Run prevalence calibration + KS gate (`KS < 0.003`), print the full ablation table, and if net 5-fold CV gain over `v8` is >= +1.5e-5, submit Slot 2 (`sub_phase2_v8_fast_harvest.csv`) to Kaggle.
"""
from __future__ import annotations

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


def calibrate_logit_to_prevalence(
    z_test: np.ndarray,
    z_oof_ref: np.ndarray,
    y_train: np.ndarray,
    target_mean: float | None = None,
) -> np.ndarray:
    if target_mean is None:
        target_mean = float(np.mean(y_train))
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof_ref.reshape(-1, 1), y_train.astype(np.int32))
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    z_lin = slope * z_test + intercept

    def obj(delta: float) -> float:
        return (float(np.mean(expit(z_lin + delta))) - target_mean) ** 2

    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded")
    return z_lin + float(res.x)


def eval_candidate_orth(
    y: np.ndarray,
    folds: list,
    z_curr_oof: np.ndarray,
    z_curr_test: np.ndarray,
    cand_oof: np.ndarray,
    cand_test: np.ndarray,
):
    """Orthogonalize candidate against running ensemble, optimize step weight via golden-section on 5-fold CV."""
    zc_o = zrk(cand_oof)
    zc_t = pchip_quantile_map(zrk(cand_test), zc_o, n_knots=4000)

    beta = float(np.cov(zc_o, z_curr_oof, ddof=0)[0, 1] / np.var(z_curr_oof))
    r_o_raw = zc_o - beta * z_curr_oof
    mu_r, std_r = float(np.mean(r_o_raw)), float(np.std(r_o_raw))
    r_o = (r_o_raw - mu_r) / std_r
    r_t_raw = (zc_t - beta * z_curr_test - mu_r) / std_r
    r_t = pchip_quantile_map(r_t_raw, r_o, n_knots=4000)
    r_t = (r_t - float(np.mean(r_o))) / (float(np.std(r_o)) + 1e-12)

    base_auc = float(roc_auc_score(y, z_curr_oof))
    base_folds = [float(roc_auc_score(y[b], z_curr_oof[b])) for _, b in folds]

    # Nested 5-fold golden-section search on each fold's training split `a`
    nested_oof = np.zeros_like(z_curr_oof)
    fold_ws = []
    for a, b in folds:
        res_f = minimize_scalar(
            lambda w: -float(roc_auc_score(y[a], z_curr_oof[a] + w * r_o[a])),
            bounds=(-0.08, 0.08),
            method="bounded",
        )
        wf = float(res_f.x)
        fold_ws.append(wf)
        nested_oof[b] = z_curr_oof[b] + wf * r_o[b]

    mean_w = float(np.mean(fold_ws))
    # Also evaluate using mean_w across all 5 folds and nested_oof
    pooled_meanw_oof = z_curr_oof + mean_w * r_o
    auc_nested = float(roc_auc_score(y, nested_oof))
    auc_meanw = float(roc_auc_score(y, pooled_meanw_oof))

    # Pick conservative weight (nested vs mean_w)
    new_folds = [float(roc_auc_score(y[b], pooled_meanw_oof[b])) for _, b in folds]
    fold_deltas = [nf - bf for nf, bf in zip(new_folds, base_folds)]
    fold_wins = sum(1 for d in fold_deltas if d > 0)

    return {
        "solo_auc": float(roc_auc_score(y, zc_o)),
        "corr_with_curr": float(np.corrcoef(zc_o, z_curr_oof)[0, 1]),
        "beta": beta,
        "std_r": std_r,
        "fold_ws": [round(w, 5) for w in fold_ws],
        "mean_w": mean_w,
        "base_auc": base_auc,
        "auc_nested": auc_nested,
        "delta_nested": auc_nested - base_auc,
        "auc_meanw": auc_meanw,
        "delta_meanw": auc_meanw - base_auc,
        "fold_deltas": fold_deltas,
        "fold_wins": fold_wins,
        "r_o": r_o,
        "r_t": r_t,
    }


def main() -> None:
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy(np.int64)
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    v8_oof = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_sub = (
        pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(np.float64)
    )
    z_v8_o = zrk(v8_oof)
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    v8_base_auc = float(roc_auc_score(y, z_v8_o))
    print(f"[Baseline] v8 5-fold OOF AUC = {v8_base_auc:.7f}")

    cands = {
        "orig20k_knn": (
            np.load(HARV / "oof_orig20k_knn.npy").ravel(),
            np.load(HARV / "test_orig20k_knn.npy").ravel(),
        ),
        "tabm_5f": (
            np.load(HARV / "oof_tabm_5f.npy").ravel(),
            np.load(HARV / "test_tabm_5f.npy").ravel(),
        ),
        "xgb_fast_harvest": (
            np.load(HARV / "oof_xgb_fast_harvest.npy").ravel(),
            np.load(HARV / "test_xgb_fast_harvest.npy").ravel(),
        ),
    }

    z_curr_o = z_v8_o.copy()
    z_curr_t = z_v8_t.copy()
    remaining = list(cands.keys())
    ablation = []
    accepted = []

    step = 0
    while remaining:
        step += 1
        step_evals = {}
        print(f"\n--- Greedy Step {step} (running OOF AUC = {roc_auc_score(y, z_curr_o):.7f}) ---")
        for name in remaining:
            co, ct = cands[name]
            ev = eval_candidate_orth(y, folds, z_curr_o, z_curr_t, co, ct)
            step_evals[name] = ev
            print(
                f"  cand={name:18s} solo={ev['solo_auc']:.6f} corr={ev['corr_with_curr']:.5f} "
                f"w={ev['mean_w']:+.4f} Δ_meanw={ev['delta_meanw']:+.2e} Δ_nested={ev['delta_nested']:+.2e} "
                f"wins={ev['fold_wins']}/5 folds={[round(d*1e5, 2) for d in ev['fold_deltas']]}e-5"
            )

        best_name = max(remaining, key=lambda n: min(step_evals[n]["delta_meanw"], step_evals[n]["delta_nested"]))
        best_ev = step_evals[best_name]
        lift = min(best_ev["delta_meanw"], best_ev["delta_nested"])

        ablation.append({
            "step": step,
            "best_candidate": best_name,
            "solo_auc": best_ev["solo_auc"],
            "corr": best_ev["corr_with_curr"],
            "mean_w": best_ev["mean_w"],
            "delta_meanw": best_ev["delta_meanw"],
            "delta_nested": best_ev["delta_nested"],
            "fold_wins": best_ev["fold_wins"],
            "fold_deltas": best_ev["fold_deltas"],
            "passed_gate": bool(lift >= 1.0e-5 and best_ev["fold_wins"] >= 4),
        })

        if lift >= 1.0e-5 and best_ev["fold_wins"] >= 4:
            w = best_ev["mean_w"]
            z_curr_o = z_curr_o + w * best_ev["r_o"]
            z_curr_t = z_curr_t + w * best_ev["r_t"]
            accepted.append((best_name, w))
            remaining.remove(best_name)
            print(f"  -> ACCEPTED {best_name} (w={w:+.4f}, new OOF AUC={roc_auc_score(y, z_curr_o):.7f})")
        else:
            print(f"  -> STOPPED: best candidate {best_name} did not pass gate (lift>=+1.0e-5 & >=4/5 folds)")
            break

    net_5f_gain = float(roc_auc_score(y, z_curr_o) - v8_base_auc)
    print(f"\n[Summary] Accepted={accepted} | Net 5-fold CV gain over v8 = {net_5f_gain:+.2e}")

    # Re-apply Fold-2 gpt2-medium orthogonal residual on top of updated 5-fold stack
    a2, idx2 = folds[2]
    prior2 = float(y[a2].mean())
    oof2_raw = np.load(PRED / "gpt2_med_4ep/oof_2.npy")
    test2_raw = np.load(PRED / "gpt2_med_4ep/test_2.npy")
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
    z_m_oof2 = zrk(z_bayes_oof2)
    z_m_test = pchip_quantile_map(zrk(z_bayes_test), z_m_oof2, n_knots=4000)

    z_curr_oof2 = zrk(z_curr_o[idx2])
    z_curr_test_f2 = pchip_quantile_map(zrk(z_curr_t), z_curr_oof2, n_knots=4000)
    beta_m = float(np.cov(z_m_oof2, z_curr_oof2, ddof=0)[0, 1] / np.var(z_curr_oof2))
    r_m_o2 = z_m_oof2 - beta_m * z_curr_oof2
    mu_m, std_m = float(np.mean(r_m_o2)), float(np.std(r_m_o2))
    r_m_o2 = (r_m_o2 - mu_m) / std_m
    r_m_t = pchip_quantile_map((z_m_test - beta_m * z_curr_test_f2 - mu_m) / std_m, r_m_o2, n_knots=4000)

    best_wm, best_f2 = 0.0, float(roc_auc_score(y[idx2], z_curr_oof2))
    for wm in np.linspace(-0.04, 0.05, 181):
        sc = float(roc_auc_score(y[idx2], z_curr_oof2 + wm * r_m_o2))
        if sc > best_f2:
            best_f2, best_wm = sc, float(wm)

    z_final_test = z_curr_test_f2 + best_wm * r_m_t
    target_mean = float(np.mean(y))
    p_test = expit(calibrate_logit_to_prevalence(z_final_test, z_curr_o, y, target_mean=target_mean))
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y, target_mean=target_mean))
    ks_val = float(ks_2samp(p_test, p_v8_cal).statistic)
    rho_val = float(spearmanr(p_test, v8_sub).correlation)

    sub_path = SUB / "sub_phase2_v8_fast_harvest.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": p_test}).to_csv(sub_path, index=False)

    report = {
        "v8_base_auc": v8_base_auc,
        "accepted": accepted,
        "net_5f_gain": net_5f_gain,
        "ablation": ablation,
        "gpt2m_f2_w": best_wm,
        "gpt2m_f2_auc": best_f2,
        "ks_vs_v8_cal": ks_val,
        "spearman_vs_v8": rho_val,
        "qualifies_for_slot2_submit": bool(net_5f_gain >= 1.5e-5 and ks_val < 0.003),
    }
    (LOG / "phase2_slot2_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if "--submit-if-qualified" in sys.argv and report["qualifies_for_slot2_submit"]:
        msg = f"Phase2 Slot2: v8 + fast GPU harvest {accepted} (+{net_5f_gain:.2e} 5f CV) + gpt2m F2 orth"
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
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, res.stderr, flush=True)


if __name__ == "__main__":
    main()
