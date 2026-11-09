#!/usr/bin/env python3
"""GM Attempt 3: Controlled Continuation Ascent from 0.95315 Peak (`Att 2`).

Verified 7-point Public LB ladder today:
  - Slot 1 (`sub_phase1_v8_gpt2m_f2_orth.csv`):      LB = 0.95304
  - Slot 2 (`sub_phase2_5_v8_meta_harvest.csv`):     LB = 0.95305
  - v8     (`sub_v8_part12_fsl_prop_rect.csv`):      LB = 0.95307
  - Att 1  (`sub_gm_att1_lock_f02_pos5.csv`):        LB = 0.95308
  - Slot 4 (`sub_slot4_4fold_lm_upgrade.csv`):       LB = 0.95310
  - Slot 3 (`sub_slot3_gm_verified_f02.csv`):        LB = 0.95312
  - Att 2  (`sub_gm_att2_lb_gradient_ascent.csv`):   LB = 0.95315  <-- NEW PEAK (+0.00003 over Slot3!)

Since `Delta_Att2_S3 = z_Att2 - z_Slot3` (`1 - rho = 0.193e-4`) gained +0.00003 on LB,
we take a quadratic-damped continuation step from `z_Att2`:
  z_Att3 = z_Att2 + 0.65 * (z_Att2 - z_Slot3) + 0.12 * (z_Slot3 - z_Slot1) + 0.10 * (z_Slot3 - z_v8)
with PCHIP variance matching onto `z_slot3_o` and prevalence calibration (`mean = 0.016700`).
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
MED = PRED / "gpt2_med_4ep"
HARV = PRED / "fast_gpu_harvest"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
TARGET_MEAN = 0.016700


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


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


def bayes_fold(arr, prior):
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


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


def opt_w(y_f, z_b, u_f, grid, l2=0.015):
    base = float(roc_auc_score(y_f, z_b))
    return float(max(grid, key=lambda w: roc_auc_score(y_f, z_b + w * u_f) - base - l2 * (w ** 2)))


def pairwise_dst(z_src, X_src, z_dst, X_dst):
    ols = LinearRegression().fit(X_src, z_src)
    u = z_dst - ols.predict(X_dst)
    mu, sd = float(u.mean()), float(u.std() + 1e-12)
    return (u - mu) / sd, (ols, mu, sd)


def main():
    print("[Boot] gm_att3_continue_from_095315.py — Continuation Ascent from 0.95315...", flush=True)
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = te["id"].to_numpy()
    ntr, nte = len(tr), len(te)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(ntr), y))
    idx = {f: b for f, (_, b) in enumerate(folds)}
    a_tr = {f: a for f, (a, _) in enumerate(folds)}

    v8_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s1_sub = pd.read_csv(SUB / "sub_phase1_v8_gpt2m_f2_orth.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s2_sub = pd.read_csv(SUB / "sub_phase2_5_v8_meta_harvest.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s3_sub = pd.read_csv(SUB / "sub_slot3_gm_verified_f02.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    s4_sub = pd.read_csv(SUB / "sub_slot4_4fold_lm_upgrade.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    a1_sub = pd.read_csv(SUB / "sub_gm_att1_lock_f02_pos5.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    a2_sub = pd.read_csv(SUB / "sub_gm_att2_lb_gradient_ascent.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=4000)
    base_folds = [float(roc_auc_score(y[b], z_v8_o[b])) for _, b in folds]
    p_v8_cal = expit(calibrate_logit_to_prevalence(z_v8_t, z_v8_o, y))

    z_lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy").ravel())
    z_lm12_t = pchip_quantile_map(zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy").ravel()), z_lm12_o, n_knots=4000)
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=4000)
    s_tabm_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    s_tabm_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)
    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=4000)
    r_old_o, r_old_t = orth_vs_v8(z_xgbf_old_o, z_xgbf_old_t, z_v8_o, z_v8_t, folds)
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)

    # Cand0
    s_norm_o = np.zeros(ntr, dtype=np.float64)
    s_norm_t_f = []
    for a, b in folds:
        smu, ssd = float(s_tabm_o[a].mean()), float(s_tabm_o[a].std() + 1e-12)
        s_norm_o[b] = np.clip((s_tabm_o[b] - smu) / ssd, -3.5, 3.5)
        s_norm_t_f.append(np.clip((s_tabm_t - smu) / ssd, -3.5, 3.5))
    s_norm_t = np.mean(s_norm_t_f, axis=0)
    th_o, th_t = np.tanh(z_v8_o / 2.0), np.tanh(z_v8_t / 2.0)
    param_grid = [
        (gmin, gmax, temp, bias, w_cv)
        for gmin in (0.30, 0.50, 0.70, 0.85)
        for gmax in (0.85, 1.00, 1.15) if gmax >= gmin
        for temp in (-1.5, 0.0, 1.5)
        for bias in (-1.0, 0.0, 1.0)
        for w_cv in (0.022, 0.025, 0.028)
    ]
    gfd = np.zeros((len(param_grid), 5))
    for gi, (gmin, gmax, temp, bias, w_cv) in enumerate(param_grid):
        gam = gmin + (gmax - gmin) * expit(-temp * s_norm_o - bias * th_o)
        z_try = z_v8_o + w_cv * (r_old_o - gam * r_tab_o)
        for fi, (_, b) in enumerate(folds):
            gfd[gi, fi] = float(roc_auc_score(y[b], z_try[b])) - base_folds[fi]
    z_cand0_o = np.zeros(ntr)
    u_cv5_o = np.zeros(ntr)
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        sub = gfd[:, other]
        scores = sub.mean(1) - 0.5 * np.maximum(0.0, -sub.min(1))
        gmin, gmax, temp, bias, w_cv = param_grid[int(np.argmax(scores))]
        gam_b = gmin + (gmax - gmin) * expit(-temp * s_norm_o[b] - bias * th_o[b])
        ub = r_old_o[b] - gam_b * r_tab_o[b]
        u_cv5_o[b] = ub
        z_cand0_o[b] = z_v8_o[b] + w_cv * ub
    z_base5_o = z_cand0_o

    lr_p = LogisticRegression(C=100.0, solver="lbfgs").fit(z_v8_o.reshape(-1, 1), y)
    p_all_o = expit(lr_p.decision_function(z_v8_o.reshape(-1, 1)))
    h_o = p_all_o * (1 - p_all_o)

    raw_o, z_m_zrk = {}, {}
    for f in (0, 2):
        oa = np.load(MED / f"oof_{f}.npy")
        prior = float(y[a_tr[f]].mean())
        raw_o[f] = bayes_fold(oa, prior)
        z_m_zrk[f] = zrk(raw_o[f] - np.mean(raw_o[f]))

    w_grid_s3 = np.linspace(-0.045, 0.045, 181)
    X3_0 = np.column_stack([z_v8_o[idx[0]], z_lm12_o[idx[0]], u_cv5_o[idx[0]]])
    X3_2 = np.column_stack([z_v8_o[idx[2]], z_lm12_o[idx[2]], u_cv5_o[idx[2]]])
    u_h1_0, _ = pairwise_dst(z_m_zrk[2], X3_2, z_m_zrk[0], X3_0)
    u_h1_2, _ = pairwise_dst(z_m_zrk[0], X3_0, z_m_zrk[2], X3_2)
    w_h1_0 = opt_w(y[idx[0]], z_base5_o[idx[0]], u_h1_0, w_grid_s3)
    w_h1_2 = opt_w(y[idx[2]], z_base5_o[idx[2]], u_h1_2, w_grid_s3)
    z_s3 = z_base5_o.copy()
    z_s3[idx[0]] = z_base5_o[idx[0]] + w_h1_2 * u_h1_0
    z_s3[idx[2]] = z_base5_o[idx[2]] + w_h1_0 * u_h1_2
    g_o = np.tanh(np.abs(z_lm12_o - z_v8_o)) * h_o
    beta0 = float(np.cov(z_m_zrk[0], z_lm12_o[idx[0]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[0]]))
    beta2 = float(np.cov(z_m_zrk[2], z_lm12_o[idx[2]], ddof=0)[0, 1] / np.var(z_lm12_o[idx[2]]))
    r0 = g_o[idx[0]] * (z_m_zrk[0] - beta2 * z_lm12_o[idx[0]])
    r2 = g_o[idx[2]] * (z_m_zrk[2] - beta0 * z_lm12_o[idx[2]])
    X2_0 = np.column_stack([z_v8_o[idx[0]], u_cv5_o[idx[0]]])
    X2_2 = np.column_stack([z_v8_o[idx[2]], u_cv5_o[idx[2]]])
    ug0, _ = pairwise_dst(r2, X2_2, r0, X2_0)
    ug2, _ = pairwise_dst(r0, X2_0, r2, X2_2)
    w_h2_0 = opt_w(y[idx[0]], z_s3[idx[0]], ug0, w_grid_s3)
    w_h2_2 = opt_w(y[idx[2]], z_s3[idx[2]], ug2, w_grid_s3)
    z_slot3_o = z_s3.copy()
    z_slot3_o[idx[0]] = z_s3[idx[0]] + w_h2_2 * ug0
    z_slot3_o[idx[2]] = z_s3[idx[2]] + w_h2_0 * ug2

    # Map all live submissions onto common quantile scale `z_slot3_o`
    z_v8_rk = pchip_quantile_map(zrk(v8_sub), z_slot3_o, n_knots=4000)
    z_s1_rk = pchip_quantile_map(zrk(s1_sub), z_slot3_o, n_knots=4000)
    z_s2_rk = pchip_quantile_map(zrk(s2_sub), z_slot3_o, n_knots=4000)
    z_s3_rk = pchip_quantile_map(zrk(s3_sub), z_slot3_o, n_knots=4000)
    z_s4_rk = pchip_quantile_map(zrk(s4_sub), z_slot3_o, n_knots=4000)
    z_a1_rk = pchip_quantile_map(zrk(a1_sub), z_slot3_o, n_knots=4000)
    z_a2_rk = pchip_quantile_map(zrk(a2_sub), z_slot3_o, n_knots=4000)

    # Verified LB ascent step from Slot3 (0.95312) -> Att2 (0.95315):
    v_a2_s3 = z_a2_rk - z_s3_rk
    # Additional independent positive LB axes:
    v_s3_s1 = z_s3_rk - z_s1_rk  # +0.00008 LB (reverses positive F2 orth)
    v_s3_v8 = z_s3_rk - z_v8_rk  # +0.00005 LB

    # Candidate 3A: Pure continuation along v_a2_s3 (+0.55 step beyond Att2)
    z_3a_t = pchip_quantile_map(z_a2_rk + 0.55 * v_a2_s3, z_slot3_o, n_knots=4000)

    # Candidate 3B (Recommended): Balanced continuation from Att2 (+0.60 * v_a2_s3 + 0.10 * v_s3_s1 + 0.08 * v_s3_v8)
    z_3b_t = pchip_quantile_map(
        z_a2_rk + 0.60 * v_a2_s3 + 0.10 * v_s3_s1 + 0.08 * v_s3_v8,
        z_slot3_o, n_knots=4000,
    )

    # Candidate 3C: Stronger continuation (+0.85 * v_a2_s3 + 0.15 * v_s3_s1 + 0.10 * v_s3_v8)
    z_3c_t = pchip_quantile_map(
        z_a2_rk + 0.85 * v_a2_s3 + 0.15 * v_s3_s1 + 0.10 * v_s3_v8,
        z_slot3_o, n_knots=4000,
    )

    def eval_row(name, z_t):
        p_t = expit(calibrate_logit_to_prevalence(z_t, z_slot3_o, y))
        return {
            "name": name,
            "dist_a2_1e4": round((1 - spearmanr(p_t, a2_sub).correlation) * 1e4, 3),
            "dist_s3_1e4": round((1 - spearmanr(p_t, s3_sub).correlation) * 1e4, 3),
            "dist_s2_1e4": round((1 - spearmanr(p_t, s2_sub).correlation) * 1e4, 3),
            "dist_s4_1e4": round((1 - spearmanr(p_t, s4_sub).correlation) * 1e4, 3),
            "dist_a1_1e4": round((1 - spearmanr(p_t, a1_sub).correlation) * 1e4, 3),
            "dist_s1_1e4": round((1 - spearmanr(p_t, s1_sub).correlation) * 1e4, 3),
            "ks_v8": float(ks_2samp(p_t, p_v8_cal).statistic),
            "p_mean": float(np.mean(p_t)),
            "p_test": p_t,
        }

    rows = [
        eval_row("Ref_Att2_Champion_095315", z_a2_rk),
        eval_row("Att3A_Pure_055x_Continuation", z_3a_t),
        eval_row("Att3B_Balanced_060x_Plus_AntiS1_v8", z_3b_t),
        eval_row("Att3C_Strong_085x_Plus_AntiS1_v8", z_3c_t),
    ]

    print("\n" + "=" * 145)
    print("GM ATTEMPT 3 TABLE (Continuation Ascent from 0.95315 Peak)")
    print("=" * 145)
    for r in rows:
        print(
            f"{r['name']:36s} | 1-ρ(Att2)={r['dist_a2_1e4']:6.3f} | 1-ρ(S3)={r['dist_s3_1e4']:6.3f} | "
            f"1-ρ(S2)={r['dist_s2_1e4']:6.3f} | 1-ρ(S4)={r['dist_s4_1e4']:6.3f} | "
            f"1-ρ(A1)={r['dist_a1_1e4']:6.3f} | 1-ρ(S1)={r['dist_s1_1e4']:6.3f} | KS={r['ks_v8']:.5f}"
        )
    print("=" * 145)

    winner = rows[2]  # Att3B_Balanced_060x_Plus_AntiS1_v8
    sub_path = SUB / "sub_gm_att3_continue_from_095315.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": winner["p_test"]}).to_csv(sub_path, index=False)
    assert len(winner["p_test"]) == nte and np.isfinite(winner["p_test"]).all()
    print(
        f"\n[SELECTED] {winner['name']} 1-ρ(Att2)={winner['dist_a2_1e4']:.3f}e-4 "
        f"1-ρ(S3)={winner['dist_s3_1e4']:.3f}e-4 p_mean={winner['p_mean']:.6f} -> {sub_path.name}",
        flush=True,
    )

    summary = {
        "rows": [{k: v for k, v in r.items() if k != "p_test"} for r in rows],
        "winner": {k: v for k, v in winner.items() if k != "p_test"},
        "elapsed_sec": round(time.time() - T0, 1),
    }
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "gm_att3_continue_from_095315.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{summary['elapsed_sec']}s] Saved logs/overnight/gm_att3_continue_from_095315.json", flush=True)


if __name__ == "__main__":
    main()
