#!/usr/bin/env python3
"""
3-Minute Forensic Audit on Existing Disk Artifacts (Zero New Heavy Training — Uses Idle A40).
Step 1: Full Sub-Module Breakdown across preds/paradigm_A/, B/, C/, and preds/matrix_v2/
Step 2: Raw Feature-Matrix Ridge / L4-Regime Residual Signal (Zero KMAT dilution)
Step 3: Non-Linear Meta-Stacking Over the ~90-Model Bank Already on Disk (3 Meta-Tests)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, skew, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("INPUT_DIR", str(ROOT))

from paradigm_stack_delta_eval import fast_auc, rk01, test1_orthogonal_alpha_probe, to_z_ranklogit  # noqa: E402

BASELINE = 0.9523071
EPS = 1e-6


def fast_spearman(x: np.ndarray, y_rk: np.ndarray) -> float:
    """Fast Spearman rho when y_rk is already rank01(y)."""
    rx = rk01(x)
    rx = rx - rx.mean()
    ry = y_rk - y_rk.mean()
    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))) + 1e-12
    return float(np.dot(rx, ry) / denom)


def gen_L4_3_stepgrid_oof(tr_df: pd.DataFrame, te_df: pd.DataFrame, y: np.ndarray, folds: list) -> tuple[np.ndarray, np.ndarray]:
    """Fast 5-fold OOF step-grid smoothed log-odds for continuous features (P4_L4.3)."""
    NUMS = [
        "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    ]
    n_tr, n_te = len(tr_df), len(te_df)
    F_oof = np.zeros((n_tr, len(NUMS) * 2), dtype=np.float32)
    F_te_folds = np.zeros((n_te, len(NUMS) * 2), dtype=np.float32)

    X_tr_raw = tr_df[NUMS].to_numpy(np.float64)
    X_te_raw = te_df[NUMS].to_numpy(np.float64)

    for fi, (a, b) in enumerate(folds):
        ya = y[a].astype(np.float64)
        p0 = float(ya.mean())
        for j in range(len(NUMS)):
            va, vb, vt = X_tr_raw[a, j], X_tr_raw[b, j], X_te_raw[:, j]
            qs = np.unique(np.quantile(va, np.linspace(0, 1, 65)))
            if len(qs) < 3:
                continue
            ba = np.clip(np.searchsorted(qs, va, side="right") - 1, 0, len(qs) - 2)
            bb = np.clip(np.searchsorted(qs, vb, side="right") - 1, 0, len(qs) - 2)
            bt = np.clip(np.searchsorted(qs, vt, side="right") - 1, 0, len(qs) - 2)
            K = len(qs) - 1
            cnt = np.bincount(ba, minlength=K).astype(np.float64)
            pos = np.bincount(ba, weights=ya, minlength=K).astype(np.float64)
            rate = (pos + 20.0 * p0) / (cnt + 20.0)
            pad = np.pad(rate, 2, mode="edge")
            ker = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smooth = np.convolve(pad, ker, mode="valid")
            lo = logit(np.clip(smooth, 1e-5, 1 - 1e-5))
            jump = np.abs(rate - smooth)
            F_oof[b, 2 * j] = lo[bb]
            F_oof[b, 2 * j + 1] = jump[bb]
            F_te_folds[:, 2 * j] += lo[bt] / len(folds)
            F_te_folds[:, 2 * j + 1] += jump[bt] / len(folds)
    return F_oof, F_te_folds


def main():
    t0 = time.time()
    print("=" * 100)
    print("3-MINUTE FORENSIC AUDIT ON EXISTING DISK ARTIFACTS (A40 CUDA + CPU VECTORIZED)")
    print("=" * 100, flush=True)

    bank = np.load(ROOT / "preds/v4p_bank_cache.npz", allow_pickle=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    tr_df = pd.read_csv(ROOT / "data/train.csv")
    te_df = pd.read_csv(ROOT / "data/test.csv")
    n, nt = len(y), len(te_df)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    a0, b0 = folds[0]
    y_b0 = y[b0]

    v4p_hc_oof = bank["v4p_hc_oof"].astype(np.float64)
    v4p_hc_test = bank["v4p_hc_test"].astype(np.float64)
    v4p_rk_oof = rk01(v4p_hc_oof)
    v4p_rk_test = rk01(v4p_hc_test)
    z_base_oof = to_z_ranklogit(v4p_hc_oof)
    z_base_test = to_z_ranklogit(v4p_hc_test)
    base_auc = float(fast_auc(y, v4p_hc_oof))
    base_f0_auc = float(fast_auc(y_b0, v4p_hc_oof[b0]))

    # -------------------------------------------------------------------------
    # STEP 1: FULL SUB-MODULE BREAKDOWN
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Scanning all artifacts in preds/paradigm_A/, B/, C/, and preds/matrix_v2/ ...", flush=True)

    # Pair map for OOF + Test files
    scan_items = [
        # Paradigm A
        ("paradigm_A/A1_chimera_train.npy", "paradigm_A/A1_chimera_test.npy"),
        ("paradigm_A/A2_clonegraph_oof.npy", "paradigm_A/A2_clonegraph_test.npy"),
        ("paradigm_A/f_matrix_oof.npy", "paradigm_A/f_matrix_test.npy"),
        ("paradigm_A/tabr_oof.npy", "paradigm_A/tabr_test.npy"),
        ("paradigm_A/xgb_oof.npy", "paradigm_A/xgb_test.npy"),
        ("paradigm_A/combo_oof.npy", "paradigm_A/combo_test.npy"),
        ("paradigm_A/stack_oof.npy", None),
        ("paradigm_A/stack_oof_corrected.npy", None),
        ("paradigm_A/checkpoint_P4_L4.3_fold0.npz", None),
        # Paradigm B
        ("paradigm_B/tokens72_train.npy", "paradigm_B/tokens72_test.npy"),
        ("paradigm_B/llr72_oof.npy", "paradigm_B/llr72_test.npy"),
        ("paradigm_B/h64_oof.npy", "paradigm_B/h64_test.npy"),
        ("paradigm_B/llm_solo_oof.npy", "paradigm_B/llm_solo_test.npy"),
        ("paradigm_B/xgb_llr_oof.npy", "paradigm_B/xgb_llr_test.npy"),
        ("paradigm_B/combo_oof.npy", "paradigm_B/combo_test.npy"),
        ("paradigm_B/stack_oof_corrected.npy", None),
        # Paradigm C
        ("paradigm_C/copula_dae_oof.npy", "paradigm_C/copula_dae_test.npy"),
        ("paradigm_C/tabm_k32_oof.npy", "paradigm_C/tabm_k32_test.npy"),
        ("paradigm_C/stealth_oof.npy", "paradigm_C/stealth_test.npy"),
        ("paradigm_C/combo_oof.npy", "paradigm_C/combo_test.npy"),
        ("paradigm_C/stack_oof_corrected.npy", None),
        # matrix_v2 fold0 checkpoints
        ("matrix_v2/P3_L3.1_fold0.npz", None),
        ("matrix_v2/P3_L3.2_fold0.npz", None),
        ("matrix_v2/P3_L3.3_fold0.npz", None),
        ("matrix_v2/P3_L3.4_fold0.npz", None),
        ("matrix_v2/P3_L3.5_fold0.npz", None),
        ("matrix_v2/P4_L4.2_fold0.npz", None),
        ("matrix_v2/P4_L4.3_fold0.npz", None),
        ("matrix_v2/P4_L4.4_fold0.npz", None),
        ("matrix_v2/P4_L4.5_fold0.npz", None),
    ]

    step1_rows = []
    for fo_rel, ft_rel in scan_items:
        po = ROOT / "preds" / fo_rel
        if not po.exists():
            continue
        pt = (ROOT / "preds" / ft_rel) if ft_rel and (ROOT / "preds" / ft_rel).exists() else None

        if po.suffix == ".npz":
            z = np.load(po, allow_pickle=True)
            df0 = np.asarray(z["delta_fold0"], dtype=np.float64).ravel()
            dte = np.asarray(z["delta_test_probe"], dtype=np.float64).ravel()
            # Fold0 checkpoint: score on b0 vs v4p_hc_oof[b0]
            # Note: delta_fold0 is the residual logit offset from z_champ; evaluate both solo and ortho probe on b0
            cand_b0 = bank["z_champ_oof"][b0].astype(np.float64) + df0
            cand_te = bank["z_champ_test"].astype(np.float64) + dte
            solo_auc = float(fast_auc(y_b0, cand_b0))
            rho_oof = fast_spearman(cand_b0, rk01(v4p_hc_oof[b0]))
            rho_te = fast_spearman(cand_te, v4p_rk_test)
            drho = abs(rho_oof - rho_te)
            ks_raw = float(ks_2samp(df0, dte).statistic)
            # Ortho probe on Fold 0 (split b0 50/50 stratified)
            skf2 = list(StratifiedKFold(2, shuffle=True, random_state=42).split(np.zeros(len(y_b0)), y_b0))
            t1_f0 = test1_orthogonal_alpha_probe(y_b0, v4p_hc_oof[b0], cand_b0, skf2)
            step1_rows.append({
                "file": fo_rel,
                "test_file": "in_npz(299984,)",
                "shape": f"f0:{df0.shape}",
                "solo_auc": f"{solo_auc:.6f} (F0)",
                "rho_v4p": f"{rho_oof:+.5f}",
                "ortho_delta": f"{t1_f0['delta_auc']:+.3e} (F0)",
                "abs_drho": f"{drho:.5f}",
                "ks_stat": f"{ks_raw:.4f}",
            })
            continue

        arr_o = np.load(po)
        arr_t = np.load(pt) if pt is not None else None
        shp_str = f"{arr_o.shape}" + (f" / {arr_t.shape}" if arr_t is not None else "")

        if arr_o.ndim == 1:
            vec_o = np.nan_to_num(arr_o.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            solo_auc = float(fast_auc(y, vec_o))
            rho_oof = fast_spearman(vec_o, v4p_rk_oof)
            t1 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, vec_o, folds)
            if arr_t is not None:
                vec_t = np.nan_to_num(arr_t.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                rho_te = fast_spearman(vec_t, v4p_rk_test)
                drho_str = f"{abs(rho_oof - rho_te):.5f}"
                ks_str = f"{float(ks_2samp(vec_o, vec_t).statistic):.4f}"
            else:
                drho_str = "N/A (no test)"
                ks_str = "N/A"
            step1_rows.append({
                "file": fo_rel,
                "test_file": ft_rel or "-",
                "shape": shp_str,
                "solo_auc": f"{solo_auc:.6f}",
                "rho_v4p": f"{rho_oof:+.5f}",
                "ortho_delta": f"{t1['delta_auc']:+.3e}",
                "abs_drho": drho_str,
                "ks_stat": ks_str,
            })
        else:
            # 2D Feature Matrix: evaluate 5-fold Ridge linear probe -> 1D OOF + Test score
            Xo_mat = np.nan_to_num(arr_o.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            Xt_mat = np.nan_to_num(arr_t.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) if arr_t is not None else None
            probe_o = np.zeros(n, dtype=np.float64)
            probe_t = np.zeros(nt, dtype=np.float64) if Xt_mat is not None else None
            for a, b in folds:
                mu = Xo_mat[a].mean(0)
                sd = Xo_mat[a].std(0) + 1e-6
                Xa = (Xo_mat[a] - mu) / sd
                Xb = (Xo_mat[b] - mu) / sd
                # Fast closed-form ridge on GPU
                Xa_t = torch.from_numpy(Xa).cuda()
                ya_t = torch.from_numpy(y[a].astype(np.float32)).cuda()
                d_f = Xa.shape[1]
                A_mat = Xa_t.T @ Xa_t + 100.0 * torch.eye(d_f, device="cuda")
                rhs = Xa_t.T @ (ya_t - ya_t.mean())
                w = torch.linalg.solve(A_mat, rhs).cpu().numpy()
                probe_o[b] = Xb @ w
                if Xt_mat is not None:
                    probe_t += ((Xt_mat - mu) / sd @ w) / len(folds)
            solo_auc = float(fast_auc(y, probe_o))
            rho_oof = fast_spearman(probe_o, v4p_rk_oof)
            t1 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, probe_o, folds)
            if probe_t is not None:
                rho_te = fast_spearman(probe_t, v4p_rk_test)
                drho_str = f"{abs(rho_oof - rho_te):.5f}"
                ks_str = f"{float(ks_2samp(probe_o, probe_t).statistic):.4f}"
            else:
                drho_str = "N/A"
                ks_str = "N/A"
            step1_rows.append({
                "file": fo_rel,
                "test_file": ft_rel or "-",
                "shape": shp_str,
                "solo_auc": f"{solo_auc:.6f} (Ridge)",
                "rho_v4p": f"{rho_oof:+.5f}",
                "ortho_delta": f"{t1['delta_auc']:+.3e}",
                "abs_drho": drho_str,
                "ks_stat": ks_str,
            })

    print("\n### STEP 1 TABLE: Full Sub-Module Breakdown")
    print(f"{'File Name (OOF / Pair)':42s} | {'Shape':27s} | {'Solo 5F OOF AUC':17s} | {'ρ vs v4p_hc':11s} | {'Test-1 Ortho ΔAUC':17s} | {'|Δρ| OOF-Test':13s} | {'KS Stat':8s}")
    print("-" * 150)
    for r in step1_rows:
        print(f"{r['file']:42s} | {r['shape']:27s} | {r['solo_auc']:17s} | {r['rho_v4p']:11s} | {r['ortho_delta']:17s} | {r['abs_drho']:13s} | {r['ks_stat']:8s}")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STEP 2: RAW FEATURE-MATRIX RIDGE / L4-REGIME RESIDUAL SIGNAL
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Building F_all_new = [F_A_oof(42), llr72_oof(72), h64_oof(64), copula_dae_oof(57), P4_L4.3_stepgrid(26)] ...", flush=True)
    t_s2 = time.time()
    F_A_o = np.nan_to_num(np.load(ROOT / "preds/paradigm_A/f_matrix_oof.npy").astype(np.float32))
    llr72_o = np.nan_to_num(np.load(ROOT / "preds/paradigm_B/llr72_oof.npy").astype(np.float32))
    h64_o = np.nan_to_num(np.load(ROOT / "preds/paradigm_B/h64_oof.npy").astype(np.float32))
    cop_o = np.nan_to_num(np.load(ROOT / "preds/paradigm_C/copula_dae_oof.npy").astype(np.float32))
    step_o, step_t = gen_L4_3_stepgrid_oof(tr_df, te_df, y, folds)

    F_all_new = np.hstack([F_A_o, llr72_o, h64_o, cop_o, step_o]).astype(np.float32)
    print(f"[STEP 2] F_all_new shape = {F_all_new.shape} (built in {time.time()-t_s2:.1f}s)", flush=True)

    # Extract exact r_rare and r_disp from build_psuper_v4p_bank.py
    import gboost as gb
    kcnt_tr = gb.KCNT[:n].astype(np.float32)
    n_float = min(6, kcnt_tr.shape[1])

    def stdize_vec(v):
        v = np.asarray(v, dtype=np.float32)
        return (v - float(v.mean())) / (float(v.std()) + 1e-12)

    r_rare = stdize_vec((kcnt_tr[:, :n_float] <= 3).mean(1))
    names_v4p = [str(x) for x in bank["names_v4p"]]
    ni = {nm: i for i, nm in enumerate(names_v4p)}
    O_v4p = bank["O_v4p"].astype(np.float64)
    T_v4p = bank["T_v4p"].astype(np.float64)
    mod5 = ["gb_bag5", "gpt2", "glm_quad", "ebm_i25_bag4", "gbn_bag5_w512"]

    def to_rlogit_col(col):
        c = np.asarray(col, dtype=np.float64)
        if c.min() >= 0.0 and c.max() <= 1.0:
            return logit(np.clip(c, EPS, 1.0 - EPS))
        return c

    mats_tr = np.column_stack([to_rlogit_col(O_v4p[:, ni[k]]) for k in mod5])
    r_disp = stdize_vec(mats_tr.std(1))

    # Build expanded matrix [F_all_new, F_all_new * r_rare, F_all_new * r_disp] -> 261 * 3 = 783 cols
    # Standardize F_all_new per fold to avoid scale issues before multiplying by regime indicators
    print("[STEP 2] Fitting 5-fold CUDA IRLS / Newton-Cholesky Logistic & Ridge with init_score = logit(rk(v4p_hc_oof)) ...", flush=True)

    # Test multiple regularization strengths C / l2_lambda on GPU IRLS (exact Newton-Cholesky on CUDA)
    s2_results = []
    for mode_name, use_interactions in [("F_all_new only (261 cols)", False), ("F_all_new + *r_rare + *r_disp (783 cols)", True)]:
        for l2_pen in [1e2, 1e3, 1e4, 5e4, 2e5]:
            oof_pred = np.zeros(n, dtype=np.float64)
            fold_wins = 0
            for a, b in folds:
                mu = F_all_new[a].mean(0)
                sd = F_all_new[a].std(0) + 1e-5
                Fa = np.clip((F_all_new[a] - mu) / sd, -8.0, 8.0)
                Fb = np.clip((F_all_new[b] - mu) / sd, -8.0, 8.0)
                if use_interactions:
                    Xa = np.hstack([Fa, Fa * r_rare[a, None], Fa * r_disp[a, None]])
                    Xb = np.hstack([Fb, Fb * r_rare[b, None], Fb * r_disp[b, None]])
                else:
                    Xa, Xb = Fa, Fb

                # Add intercept
                Xa_g = torch.from_numpy(np.column_stack([Xa, np.ones(len(a), dtype=np.float32)])).cuda()
                Xb_g = torch.from_numpy(np.column_stack([Xb, np.ones(len(b), dtype=np.float32)])).cuda()
                ya_g = torch.from_numpy(y[a].astype(np.float32)).cuda()
                z0_a = torch.from_numpy(z_base_oof[a].astype(np.float32)).cuda()
                z0_b = torch.from_numpy(z_base_oof[b].astype(np.float32)).cuda()

                d_col = Xa_g.shape[1]
                w = torch.zeros(d_col, device="cuda", dtype=torch.float32)
                reg_eye = torch.eye(d_col, device="cuda", dtype=torch.float32) * float(l2_pen)
                reg_eye[-1, -1] = 1.0  # weak reg on intercept

                # 4 Newton-Cholesky IRLS steps around init_score = z0_a
                for _ in range(4):
                    eta = z0_a + Xa_g @ w
                    p = torch.sigmoid(eta)
                    W = torch.clamp(p * (1.0 - p), min=1e-5)
                    grad = Xa_g.T @ (p - ya_g) + reg_eye @ w
                    H = (Xa_g.T * W) @ Xa_g + reg_eye
                    step = torch.linalg.solve(H, grad)
                    w = w - step
                    if float(torch.max(torch.abs(step))) < 1e-5:
                        break

                pred_b = (z0_b + Xb_g @ w).cpu().numpy()
                oof_pred[b] = pred_b
                if fast_auc(y[b], pred_b) > fast_auc(y[b], z_base_oof[b]):
                    fold_wins += 1

            auc_s2 = float(fast_auc(y, oof_pred))
            delta_s2 = auc_s2 - base_auc
            # Also run Test-1 ortho alpha probe on the learned residual (oof_pred - z_base_oof)
            resid_only = oof_pred - z_base_oof
            t1_s2 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, resid_only, folds)
            s2_results.append({
                "spec": mode_name,
                "l2_pen": l2_pen,
                "direct_oof_auc": auc_s2,
                "direct_delta": delta_s2,
                "fold_wins": f"{fold_wins}/5",
                "ortho_alpha_auc": t1_s2["stack_oof_auc"],
                "ortho_alpha_delta": t1_s2["delta_auc"],
                "mean_alpha": t1_s2["mean_alpha"],
            })

    print("\n### STEP 2 TABLE: Raw Feature-Matrix Newton-Cholesky / Ridge Residual Signal (Zero KMAT)")
    print(f"{'Feature Specification':42s} | {'L2 Penalty':10s} | {'Direct OOF AUC':14s} | {'Direct ΔAUC':13s} | {'Wins':5s} | {'Nested α-Probe ΔAUC':19s} | {'Mean α':8s}")
    print("-" * 128)
    for r in s2_results:
        print(f"{r['spec']:42s} | {r['l2_pen']:10.0e} | {r['direct_oof_auc']:14.7f} | {r['direct_delta']:+13.3e} | {r['fold_wins']:5s} | {r['ortho_alpha_delta']:+19.3e} | {r['mean_alpha']:+8.4f}")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STEP 3: NON-LINEAR META-STACKING OVER ~90-MODEL BANK (O_v4p + A/B/C OOFs)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Building 90-Model Logit Bank: O_v4p (81) + v4p/hc/champ (3) + A/B/C OOFs (6) = 90 models ...", flush=True)
    # Standardize all 90 models onto rank-logit scale z_m = logit(clip(rk(m), 1e-6, 1-1e-6))
    extra_oofs = [
        ("v4p_oof", bank["v4p_oof"]),
        ("z_champ_oof", bank["z_champ_oof"]),
        ("A_xgb", np.load(ROOT / "preds/paradigm_A/xgb_oof.npy")),
        ("A_tabr", np.load(ROOT / "preds/paradigm_A/tabr_oof.npy")),
        ("B_xgb_llr", np.load(ROOT / "preds/paradigm_B/xgb_llr_oof.npy")),
        ("B_llm_solo", np.load(ROOT / "preds/paradigm_B/llm_solo_oof.npy")),
        ("C_tabm_k32", np.load(ROOT / "preds/paradigm_C/tabm_k32_oof.npy")),
        ("C_stealth", np.load(ROOT / "preds/paradigm_C/stealth_oof.npy")),
        ("P_super_v4_oof", np.load(ROOT / "preds/P_super_v4_oof.npy")),
    ]
    Z_list = []
    model_names_90 = []
    for j in range(O_v4p.shape[1]):
        Z_list.append(to_rlogit_col(O_v4p[:, j]).astype(np.float32))
        model_names_90.append(names_v4p[j])
    for nm, vec in extra_oofs:
        Z_list.append(to_z_ranklogit(vec).astype(np.float32))
        model_names_90.append(nm)

    Z90 = np.column_stack(Z_list).astype(np.float32)
    # Center each column and compute row-wise dispersion stats across the 90 models on GPU
    Z90_g = torch.from_numpy(Z90).cuda()
    Z90_std_g = (Z90_g - Z90_g.mean(0, keepdim=True)) / (Z90_g.std(0, keepdim=True) + 1e-6)
    row_mean = Z90_std_g.mean(1, keepdim=True)
    row_std = Z90_std_g.std(1)
    row_med = torch.median(Z90_std_g, dim=1).values
    row_max_med = Z90_std_g.max(1).values - row_med
    row_min_med = Z90_std_g.min(1).values - row_med
    centered = Z90_std_g - row_mean
    row_skew = (centered ** 3).mean(1) / (row_std ** 3 + 1e-6)
    row_kurt = (centered ** 4).mean(1) / (row_std ** 4 + 1e-6)

    disp_feats = torch.stack([row_std, row_max_med, row_min_med, row_skew, row_kurt], dim=1).cpu().numpy()
    kcnt_feats = np.column_stack([
        np.log1p(kcnt_tr).mean(1),
        np.log1p(kcnt_tr).std(1),
        np.log1p(kcnt_tr[:, :n_float]).mean(1),
        r_rare,
        r_disp,
    ]).astype(np.float32)

    X_meta1 = np.hstack([Z90, disp_feats, kcnt_feats]).astype(np.float32)
    print(f"[STEP 3.1] Meta-Test 1 matrix shape = {X_meta1.shape}. Running 5-fold CUDA XGBoost (max_depth=2, min_child_weight=500, colsample=0.3) ...", flush=True)

    meta1_results = []
    for n_rounds, eta_val in [(80, 0.02), (150, 0.03), (250, 0.03)]:
        oof_m1 = np.zeros(n, dtype=np.float64)
        wins = 0
        for a, b in folds:
            dtr = xgb.DMatrix(X_meta1[a], label=y[a], base_margin=z_base_oof[a].astype(np.float32))
            dva = xgb.DMatrix(X_meta1[b], label=y[b], base_margin=z_base_oof[b].astype(np.float32))
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 2,
                "max_leaves": 3,
                "eta": eta_val,
                "min_child_weight": 500,
                "subsample": 0.7,
                "colsample_bytree": 0.3,
                "reg_lambda": 50.0,
                "reg_alpha": 10.0,
                "tree_method": "hist",
                "device": "cuda:0",
                "seed": 42,
            }
            bst = xgb.train(params, dtr, num_boost_round=n_rounds, verbose_eval=False)
            pb = bst.predict(dva, output_margin=True)
            oof_m1[b] = pb
            if fast_auc(y[b], pb) > fast_auc(y[b], z_base_oof[b]):
                wins += 1
        auc_m1 = float(fast_auc(y, oof_m1))
        t1_m1 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, oof_m1 - z_base_oof, folds)
        meta1_results.append({
            "config": f"Depth-2 Leaves-3 XGB (rounds={n_rounds}, eta={eta_val})",
            "direct_auc": auc_m1,
            "direct_delta": auc_m1 - base_auc,
            "wins": f"{wins}/5",
            "nested_alpha_delta": t1_m1["delta_auc"],
            "mean_alpha": t1_m1["mean_alpha"],
        })

    # Meta-Test 2: Pairwise Cross-Model Product / Degree-2 Polynomial Kernel Ridge across 9 orthogonal families
    print("[STEP 3.2] Meta-Test 2: Top-200 pairwise products (z_m1 - z_bar)*(z_m2 - z_bar) across orthogonal model families ...", flush=True)
    # Pick 21 diverse orthogonal representatives across the 9 families -> 21*20/2 = 210 pairwise products
    rep_Order = [
        "gb_bag5", "gpt2", "glm_quad", "ebm_i25_bag4", "gbn_bag5_w512",
        "lgb_diglead_l3_bag", "glm_ohk2_bag", "lgb_dig_s100_bag5", "lgb_rawdig_l4",
        "masaya_stack_main", "dhan_18_blend_lm50", "dhan_18_blend_pooled2", "quey_gen", "quey_cnn",
        "v4p_oof", "z_champ_oof", "A_xgb", "A_tabr", "B_xgb_llr", "B_llm_solo", "C_tabm_k32",
    ]
    name_to_idx = {nm: i for i, nm in enumerate(model_names_90)}
    rep_idx = [name_to_idx[k] for k in rep_Order if k in name_to_idx]
    Z_rep = Z90_std_g[:, rep_idx]  # (N, 21)
    zbar_rep = Z_rep.mean(1, keepdim=True)
    D_rep = Z_rep - zbar_rep  # (N, 21)
    # All 210 upper-triangle products
    iu, ju = np.triu_indices(D_rep.shape[1], k=1)
    P_pairs = (D_rep[:, iu] * D_rep[:, ju]).cpu().numpy().astype(np.float32)  # (N, 210)

    meta2_results = []
    for l2_pen in [1e3, 1e4, 5e4, 2e5]:
        oof_m2 = np.zeros(n, dtype=np.float64)
        wins = 0
        for a, b in folds:
            # Select Top-200 by |corr| with residual (y - sigmoid(z_base)) inside fold a
            res_y = y[a] - expit(z_base_oof[a])
            Pa = P_pairs[a]
            corrs = np.abs((Pa.T @ (res_y - res_y.mean())) / (Pa.std(0) * len(a) + 1e-6))
            top200 = np.argsort(-corrs)[:200]
            mu = Pa[:, top200].mean(0)
            sd = Pa[:, top200].std(0) + 1e-5
            Xa = np.clip((Pa[:, top200] - mu) / sd, -8.0, 8.0)
            Xb = np.clip((P_pairs[b][:, top200] - mu) / sd, -8.0, 8.0)

            Xa_g = torch.from_numpy(np.column_stack([Xa, np.ones(len(a), dtype=np.float32)])).cuda()
            Xb_g = torch.from_numpy(np.column_stack([Xb, np.ones(len(b), dtype=np.float32)])).cuda()
            ya_g = torch.from_numpy(y[a].astype(np.float32)).cuda()
            z0_a = torch.from_numpy(z_base_oof[a].astype(np.float32)).cuda()
            z0_b = torch.from_numpy(z_base_oof[b].astype(np.float32)).cuda()

            d_col = Xa_g.shape[1]
            w = torch.zeros(d_col, device="cuda", dtype=torch.float32)
            reg_eye = torch.eye(d_col, device="cuda", dtype=torch.float32) * float(l2_pen)
            reg_eye[-1, -1] = 1.0
            for _ in range(4):
                eta = z0_a + Xa_g @ w
                p = torch.sigmoid(eta)
                W = torch.clamp(p * (1.0 - p), min=1e-5)
                grad = Xa_g.T @ (p - ya_g) + reg_eye @ w
                H = (Xa_g.T * W) @ Xa_g + reg_eye
                step = torch.linalg.solve(H, grad)
                w = w - step
            pred_b = (z0_b + Xb_g @ w).cpu().numpy()
            oof_m2[b] = pred_b
            if fast_auc(y[b], pred_b) > fast_auc(y[b], z_base_oof[b]):
                wins += 1
        auc_m2 = float(fast_auc(y, oof_m2))
        t1_m2 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, oof_m2 - z_base_oof, folds)
        meta2_results.append({
            "config": f"Top-200 Cross-Family Degree-2 Kernel IRLS (L2={l2_pen:.0e})",
            "direct_auc": auc_m2,
            "direct_delta": auc_m2 - base_auc,
            "wins": f"{wins}/5",
            "nested_alpha_delta": t1_m2["delta_auc"],
            "mean_alpha": t1_m2["mean_alpha"],
        })

    # Meta-Test 3: 64-Stratum Post-Calibration on v4p_hc across (flag_bitmask x KCNT_quartile)
    print("[STEP 3.3] Meta-Test 3: 64-Stratum Post-Calibration across (flag_bitmask x KCNT_quartile) ...", flush=True)
    FLAGS = [
        "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
    ]
    flag_mat = tr_df[FLAGS].to_numpy(np.int32)
    # Top-4 binary flags (16 bitmasks) x 4 KCNT quartiles = 64 strata; also test all 6 flags (64 bitmasks) x 4 quartiles = 256 strata
    bit16 = flag_mat[:, 0] + 2 * flag_mat[:, 1] + 4 * flag_mat[:, 2] + 8 * flag_mat[:, 5]
    bit64 = sum((flag_mat[:, j] << j) for j in range(6))
    kcnt_score = np.log1p(kcnt_tr).mean(1)
    q_bins = np.searchsorted(np.quantile(kcnt_score, [0.25, 0.5, 0.75]), kcnt_score)
    strata_64 = bit16 * 4 + q_bins
    strata_256 = bit64 * 4 + q_bins

    meta3_results = []
    for s_name, s_vec in [("64-Stratum (16 flag-mask x 4 KCNT-Q)", strata_64), ("64-Stratum (6-bit flag-mask)", bit64), ("256-Stratum (64 flag-mask x 4 KCNT-Q)", strata_256)]:
        for shrink in [50.0, 200.0, 1000.0]:
            oof_off = np.copy(z_base_oof)
            oof_platt = np.copy(z_base_oof)
            wins_off, wins_platt = 0, 0
            for a, b in folds:
                sa, sb = s_vec[a], s_vec[b]
                ya = y[a].astype(np.float64)
                za, zb = z_base_oof[a], z_base_oof[b]
                pa = expit(za)
                w_a = pa * (1.0 - pa)
                # Newton step for per-stratum additive logit offset: delta_s = sum(y_i - p_i) / (sum(w_i) + shrink)
                max_s = int(s_vec.max()) + 1
                num = np.bincount(sa, weights=(ya - pa), minlength=max_s)
                den = np.bincount(sa, weights=w_a, minlength=max_s) + shrink
                delta_s = num / den
                oof_off[b] = zb + delta_s[sb]
                if fast_auc(y[b], oof_off[b]) > fast_auc(y[b], zb):
                    wins_off += 1

                # Stratum temperature + offset (2-param shrunk toward [1, 0])
                temp_s = np.ones(max_s, dtype=np.float64)
                bias_s = np.zeros(max_s, dtype=np.float64)
                for sid in range(max_s):
                    idx_s = np.where(sa == sid)[0]
                    if len(idx_s) < 30:
                        continue
                    zs = za[idx_s]
                    ys = ya[idx_s]
                    ps = pa[idx_s]
                    ws = w_a[idx_s]
                    z_c = zs - zs.mean()
                    # 1-step regularized Newton for (delta_temp, delta_bias)
                    g0 = float(np.sum(ys - ps))
                    g1 = float(np.sum((ys - ps) * z_c))
                    h00 = float(np.sum(ws)) + shrink
                    h11 = float(np.sum(ws * z_c * z_c)) + shrink * 5.0
                    bias_s[sid] = g0 / h00
                    temp_s[sid] = 1.0 + g1 / h11
                z_mean_s = np.bincount(sa, weights=za, minlength=max_s) / np.maximum(np.bincount(sa, minlength=max_s), 1)
                oof_platt[b] = z_mean_s[sb] + temp_s[sb] * (zb - z_mean_s[sb]) + bias_s[sb]
                if fast_auc(y[b], oof_platt[b]) > fast_auc(y[b], zb):
                    wins_platt += 1

            auc_off = float(fast_auc(y, oof_off))
            auc_pl = float(fast_auc(y, oof_platt))
            meta3_results.append({
                "config": f"{s_name} Offset-only (shrink={shrink:.0f})",
                "direct_auc": auc_off,
                "direct_delta": auc_off - base_auc,
                "wins": f"{wins_off}/5",
            })
            meta3_results.append({
                "config": f"{s_name} Temp+Offset (shrink={shrink:.0f})",
                "direct_auc": auc_pl,
                "direct_delta": auc_pl - base_auc,
                "wins": f"{wins_platt}/5",
            })

    print("\n### STEP 3 TABLE: Non-Linear Meta-Stacking Over 90-Model Bank (Nested 5-Fold CV)")
    print(f"{'Meta-Learner Specification':58s} | {'Direct OOF AUC':14s} | {'Direct ΔAUC':13s} | {'Wins':5s} | {'Nested α-Probe ΔAUC':19s}")
    print("-" * 120)
    for r in meta1_results + meta2_results:
        print(f"{r['config']:58s} | {r['direct_auc']:14.7f} | {r['direct_delta']:+13.3e} | {r['wins']:5s} | {r['nested_alpha_delta']:+19.3e}")
    for r in meta3_results:
        print(f"{r['config']:58s} | {r['direct_auc']:14.7f} | {r['direct_delta']:+13.3e} | {r['wins']:5s} | {'(calibrated directly)':19s}")

    out_json = {
        "baseline_auc": base_auc,
        "elapsed_sec": round(time.time() - t0, 2),
        "step1": step1_rows,
        "step2": s2_results,
        "step3_meta1": meta1_results,
        "step3_meta2": meta2_results,
        "step3_meta3": meta3_results,
    }
    out_p = ROOT / "logs/overnight/forensic_audit_3min.json"
    out_p.write_text(json.dumps(out_json, indent=2))
    print(f"\n[DONE] Forensic audit complete in {time.time()-t0:.1f}s -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
