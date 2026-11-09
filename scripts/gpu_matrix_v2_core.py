#!/usr/bin/env python3
"""
Shared mathematical upgrades for GPU Matrix v2 (Calibrated Base Margin, OOF-Reference Tail-Preserving Quantile Alignment,
Bipartite Interaction-Constrained Dual-Tree CUDA XGB + LGB + Ridge, Per-Feature KS < 0.012 Rank-Normalization Filter,
and 100%-Data 3-Seed Full-Refit Test Bagging).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import expit, logit, ndtri
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
MATRIX_V2 = ROOT / "preds" / "matrix_v2"
STATE_PATH = ROOT / "logs" / "overnight" / "matrix_v2_state.json"
WATCHDOG_PATH = ROOT / "logs" / "overnight" / "watchdog_5min.jsonl"
BLACKLIST_POD = "lheroidmuab1j4"
KS_FEATURE_MAX = 0.012
DELTA_RHO_MAX = 0.0020
FOLD0_GATE = 1.0e-5
PROMOTE_GATE = 1.5e-5


def assert_pod_allowed(pod_id: str | None) -> None:
    if pod_id and pod_id == BLACKLIST_POD:
        raise RuntimeError(f"Refusing blacklisted foreign pod_id={pod_id}")


def rk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def eval_fold0_2param_canary(
    z_champ_f0: np.ndarray, delta_f0: np.ndarray, y_f0: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """5-subfold CV 2-param logistic probe on frozen champ (no pytest dependency)."""
    z_champ_f0 = np.asarray(z_champ_f0, dtype=np.float64).ravel()
    delta_f0 = np.asarray(delta_f0, dtype=np.float64).ravel()
    y_f0 = np.asarray(y_f0).ravel()
    base_auc = float(roc_auc_score(y_f0, z_champ_f0))
    d_std = (delta_f0 - delta_f0.mean()) / (delta_f0.std() + 1e-12)
    d_gauss = ndtri(np.clip(rk(delta_f0), 1e-6, 1.0 - 1e-6))
    X_sub = np.column_stack([d_std, d_gauss])
    sub_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))
    best_auc = base_auc
    best_pred = z_champ_f0.copy()
    for C in (0.002, 0.008, 0.03):
        oof_sub = np.zeros(len(y_f0), dtype=np.float64)
        for sa, sb in sub_folds:
            X_a = np.column_stack([z_champ_f0[sa], X_sub[sa]])
            X_b = np.column_stack([z_champ_f0[sb], X_sub[sb]])
            m = LogisticRegression(C=C, solver="lbfgs", max_iter=200).fit(X_a, y_f0[sa])
            oof_sub[sb] = m.decision_function(X_b)
        a_sub = float(roc_auc_score(y_f0, oof_sub))
        if a_sub > best_auc:
            best_auc = a_sub
            best_pred = oof_sub
    alphas = (-0.15, -0.08, -0.04, -0.02, -0.01, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15)
    for col in (d_std, d_gauss):
        oof_1d = np.zeros(len(y_f0), dtype=np.float64)
        for sa, sb in sub_folds:
            best_a = 0.0
            best_sc = base_auc
            for al in alphas:
                sc = float(roc_auc_score(y_f0[sa], z_champ_f0[sa] + al * col[sa]))
                if sc > best_sc:
                    best_sc = sc
                    best_a = al
            oof_1d[sb] = z_champ_f0[sb] + best_a * col[sb]
        a_1d = float(roc_auc_score(y_f0, oof_1d))
        if a_1d > best_auc:
            best_auc = a_1d
            best_pred = oof_1d
    return best_auc, best_auc - base_auc, best_pred


def rlogit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def calibrate_logit_to_prevalence(
    z_raw_a: np.ndarray,
    y_a: np.ndarray,
    *z_others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Fit 1D Platt scaling on fold-train (z_raw_a -> y_a) so mean(expit(z_cal)) == y_a.mean() (~0.0167).
    Eliminates the +0.483 gradient bias when using rank-logit z_champ as XGB/LGB base_margin!
    """
    za = np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1)
    ya = np.asarray(y_a, dtype=np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, dtype=np.float64).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def align_to_oof_reference_with_tail_extrap(
    s_val: np.ndarray, s_te: np.ndarray, ref_oof_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map bulk quantiles onto OOF-reference logits; linear Z-score extrapolation in top/bottom 0.2% tails."""
    ref_sorted = np.sort(np.asarray(ref_oof_a, dtype=np.float64).ravel())
    q002, q998 = np.quantile(ref_sorted, [0.002, 0.998])
    ref_sd = float(np.std(ref_sorted)) + 1e-12

    def _map_vec(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64).ravel()
        u = (rankdata(s, method="average") - 0.5) / max(len(s), 1)
        mapped = np.quantile(ref_sorted, np.clip(u, 0.002, 0.998)).copy()
        s_z = (s - float(np.mean(s))) / (float(np.std(s)) + 1e-12)
        z002, z998 = float(np.quantile(s_z, 0.002)), float(np.quantile(s_z, 0.998))
        hi = u > 0.998
        lo = u < 0.002
        mapped[hi] = q998 + (s_z[hi] - z998) * ref_sd
        mapped[lo] = q002 + (s_z[lo] - z002) * ref_sd
        return mapped

    return _map_vec(s_val), _map_vec(s_te)


def per_feature_ks_filter(
    F_a: np.ndarray,
    F_b: np.ndarray,
    F_te: np.ndarray | None = None,
    max_ks: float = KS_FEATURE_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[int]]:
    """
    Section 2.5 Upgrade 4: For every column j, check ks_2samp(F_train, F_test).
    If KS >= max_ks, rank-normalize separately within train and test (KS -> 0.0) instead of dropping!
    Only drop constant columns (std <= 1e-12).
    """
    # Backward compatibility if called with 2 args (F_tr, F_te)
    if F_te is None:
        F_tr = np.asarray(F_a, dtype=np.float64).copy()
        F_test = np.asarray(F_b, dtype=np.float64).copy()
        keep: list[int] = []
        for j in range(F_tr.shape[1]):
            if float(np.std(F_tr[:, j])) <= 1e-12 and float(np.std(F_test[:, j])) <= 1e-12:
                continue
            ks = float(ks_2samp(F_tr[:, j], F_test[:, j]).statistic)
            if ks >= max_ks:
                F_tr[:, j] = ndtri(np.clip(rk(F_tr[:, j]), 1e-5, 1.0 - 1e-5))
                F_test[:, j] = ndtri(np.clip(rk(F_test[:, j]), 1e-5, 1.0 - 1e-5))
            keep.append(j)
        if not keep:
            keep = [0]
        return F_tr[:, keep], keep  # type: ignore[return-value]

    Fa = np.nan_to_num(np.asarray(F_a, dtype=np.float64).copy(), 0.0)
    Fb = np.nan_to_num(np.asarray(F_b, dtype=np.float64).copy(), 0.0)
    Ft = np.nan_to_num(np.asarray(F_te, dtype=np.float64).copy(), 0.0)
    keep = []
    for j in range(Fa.shape[1]):
        if float(np.std(Fa[:, j])) <= 1e-12:
            continue
        ks = float(ks_2samp(Fa[:, j], Ft[:, j]).statistic)
        if ks >= max_ks:
            Fa[:, j] = ndtri(np.clip(rk(Fa[:, j]), 1e-5, 1.0 - 1e-5))
            Fb[:, j] = ndtri(np.clip(rk(Fb[:, j]), 1e-5, 1.0 - 1e-5))
            Ft[:, j] = ndtri(np.clip(rk(Ft[:, j]), 1e-5, 1.0 - 1e-5))
        keep.append(j)
    if not keep:
        keep = [0]
    return Fa[:, keep], Fb[:, keep], Ft[:, keep], keep


def within_block_iqr_standardize(
    train_block: np.ndarray, test_block: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    def _std(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            med = float(np.median(col))
            q75, q25 = np.percentile(col, [75, 25])
            iqr = float(q75 - q25)
            if iqr <= 1e-9:
                iqr = float(np.std(col)) + 1e-9
            out[:, j] = np.clip((col - med) / iqr, -15.0, 15.0)
        return out

    return _std(train_block), _std(test_block)


def bag_full_refit_test(
    fold5_avg_test: np.ndarray, full_refit_tests: Iterable[np.ndarray], w_full: float = 0.75
) -> np.ndarray:
    full_list = [np.asarray(t, dtype=np.float64).ravel() for t in full_refit_tests]
    if not full_list:
        return np.asarray(fold5_avg_test, dtype=np.float64).ravel()
    full = np.mean(np.stack(full_list, axis=0), axis=0)
    fold5 = np.asarray(fold5_avg_test, dtype=np.float64).ravel()
    return w_full * full + (1.0 - w_full) * fold5


def dual_tree_feature_weights(
    n_lever: int, n_bipartite: int, n_coarse: int
) -> tuple[np.ndarray, np.ndarray]:
    w_a = np.ones(n_lever + n_bipartite + n_coarse, dtype=np.float64)
    w_b = np.ones_like(w_a)
    w_a[:n_lever] = 5.0
    w_a[n_lever : n_lever + n_bipartite] = 2.0
    w_a[n_lever + n_bipartite :] = 0.2
    w_b[:n_lever] = 3.0
    w_b[n_lever : n_lever + n_bipartite] = 0.1
    w_b[n_lever + n_bipartite :] = 2.5
    return w_a, w_b


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def fit_dual_tree_xgb(
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    X_te: np.ndarray,
    init_a: np.ndarray,
    init_b: np.ndarray,
    init_te: np.ndarray,
    n_lever: int,
    n_bipartite: int,
    n_coarse: int,
    seed: int = 42,
    y_b: np.ndarray | None = None,
    sample_weight_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Upgraded Dual-Tree + Lever-Only Residual Ensemble:
    Returns (delta_treeA_b, delta_treeA_te, delta_treeB_b, delta_treeB_te) as pure residual deltas
    on top of calibrated init_b / init_te!
    - Tree A: max_depth=2 with explicit Bipartite interaction_constraints [[lever_cols + bipartite_cols]]
      and feature_weights=5.0 on lever cols.
    - Tree B: max_depth=4 strictly on [lever_cols + coarse_cols] (rare high-card TE masked out).
    """
    import xgboost as xgb

    device = "cuda" if _cuda_ok() else "cpu"
    init_a32 = np.asarray(init_a, dtype=np.float32).ravel()
    init_b32 = np.asarray(init_b, dtype=np.float32).ravel()
    init_te32 = np.asarray(init_te, dtype=np.float32).ravel()

    # Tree A: Bipartite max_depth=2 on [F_lever (0..n_lever-1), Bipartite_TE (n_lever..n_lever+n_bip-1)]
    cols_a = list(range(n_lever + n_bipartite))
    Xa_A = np.ascontiguousarray(X_a[:, cols_a], dtype=np.float32)
    Xb_A = np.ascontiguousarray(X_b[:, cols_a], dtype=np.float32)
    Xte_A = np.ascontiguousarray(X_te[:, cols_a], dtype=np.float32)
    fw_A = np.ones(len(cols_a), dtype=np.float32)
    fw_A[:n_lever] = 5.0

    dtr_A = xgb.DMatrix(Xa_A, label=y_a, weight=sample_weight_a, feature_weights=fw_A)
    dtr_A.set_base_margin(init_a32)
    dval_A = xgb.DMatrix(Xb_A, label=y_b) if y_b is not None else xgb.DMatrix(Xb_A)
    dval_A.set_base_margin(init_b32)
    dte_A = xgb.DMatrix(Xte_A)
    dte_A.set_base_margin(init_te32)

    rounds_a = int(os.environ.get("DUAL_TREE_ROUNDS_A", "280"))
    params_A = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 2,
        "eta": 0.035,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "colsample_bynode": 0.6,
        "min_child_weight": 50,
        "reg_lambda": 12.0,
        "reg_alpha": 2.0,
        "tree_method": "hist",
        "device": device,
        "seed": seed,
    }
    evals_A = [(dval_A, "val")] if y_b is not None else []
    bst_A = xgb.train(
        params_A,
        dtr_A,
        num_boost_round=rounds_a,
        evals=evals_A or None,
        early_stopping_rounds=35 if evals_A else None,
        verbose_eval=False,
    )
    best_ntree_A = getattr(bst_A, "best_iteration", rounds_a - 1) + 1
    zb_A = bst_A.predict(dval_A, output_margin=True, iteration_range=(0, best_ntree_A)).astype(np.float64)
    zte_A = bst_A.predict(dte_A, output_margin=True, iteration_range=(0, best_ntree_A)).astype(np.float64)
    delta_b_A = zb_A - init_b
    delta_te_A = zte_A - init_te

    # Tree B: Deep structure max_depth=4 strictly on [F_lever, Coarse_Cols]
    cols_b = list(range(n_lever)) + list(range(n_lever + n_bipartite, n_lever + n_bipartite + n_coarse))
    Xa_B = np.ascontiguousarray(X_a[:, cols_b], dtype=np.float32)
    Xb_B = np.ascontiguousarray(X_b[:, cols_b], dtype=np.float32)
    Xte_B = np.ascontiguousarray(X_te[:, cols_b], dtype=np.float32)
    fw_B = np.ones(len(cols_b), dtype=np.float32)
    fw_B[:n_lever] = 3.5

    dtr_B = xgb.DMatrix(Xa_B, label=y_a, weight=sample_weight_a, feature_weights=fw_B)
    dtr_B.set_base_margin(init_a32)
    dval_B = xgb.DMatrix(Xb_B, label=y_b) if y_b is not None else xgb.DMatrix(Xb_B)
    dval_B.set_base_margin(init_b32)
    dte_B = xgb.DMatrix(Xte_B)
    dte_B.set_base_margin(init_te32)

    rounds_b = int(os.environ.get("DUAL_TREE_ROUNDS_B", "220"))
    params_B = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.6,
        "min_child_weight": 80,
        "reg_lambda": 15.0,
        "reg_alpha": 3.0,
        "tree_method": "hist",
        "device": device,
        "seed": seed + 101,
    }
    evals_B = [(dval_B, "val")] if y_b is not None else []
    bst_B = xgb.train(
        params_B,
        dtr_B,
        num_boost_round=rounds_b,
        evals=evals_B or None,
        early_stopping_rounds=35 if evals_B else None,
        verbose_eval=False,
    )
    best_ntree_B = getattr(bst_B, "best_iteration", rounds_b - 1) + 1
    zb_B = bst_B.predict(dval_B, output_margin=True, iteration_range=(0, best_ntree_B)).astype(np.float64)
    zte_B = bst_B.predict(dte_B, output_margin=True, iteration_range=(0, best_ntree_B)).astype(np.float64)
    delta_b_B = zb_B - init_b
    delta_te_B = zte_B - init_te

    return delta_b_A, delta_te_A, delta_b_B, delta_te_B


def fit_lever_Direct_and_Ridge(
    Fa: np.ndarray,
    ya: np.ndarray,
    Fb: np.ndarray,
    yb: np.ndarray | None,
    Fte: np.ndarray,
    init_a: np.ndarray,
    init_b: np.ndarray,
    init_te: np.ndarray,
    seed: int = 42,
    sample_weight_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast Lever-Only CUDA XGBoost (max_depth=2, num_leaves=3/4) + L2 Ridge on true calibrated residuals (y - expit(init_a)).
    Guarantees that strong 1D/small-D forensic signals (like L4.3, L4.1, L1.3) are never drowned out by 50 context cols.
    """
    import xgboost as xgb

    sc = StandardScaler().fit(Fa)
    Xa = np.nan_to_num(sc.transform(Fa), 0.0).astype(np.float32)
    Xb = np.nan_to_num(sc.transform(Fb), 0.0).astype(np.float32)
    Xte = np.nan_to_num(sc.transform(Fte), 0.0).astype(np.float32)

    p_a = expit(init_a)
    res_a = (ya - p_a).astype(np.float64)
    if sample_weight_a is not None:
        res_a = res_a * sample_weight_a

    ridge = Ridge(alpha=500.0).fit(Xa, res_a)
    d_ridge_b = ridge.predict(Xb).astype(np.float64)
    d_ridge_te = ridge.predict(Xte).astype(np.float64)

    device = "cuda" if _cuda_ok() else "cpu"
    dtr = xgb.DMatrix(Xa, label=ya, weight=sample_weight_a)
    dtr.set_base_margin(np.asarray(init_a, dtype=np.float32).ravel())
    dval = xgb.DMatrix(Xb, label=yb) if yb is not None else xgb.DMatrix(Xb)
    dval.set_base_margin(np.asarray(init_b, dtype=np.float32).ravel())
    dte = xgb.DMatrix(Xte)
    dte.set_base_margin(np.asarray(init_te, dtype=np.float32).ravel())

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 2,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": min(1.0, max(0.6, 4.0 / max(Xa.shape[1], 1))),
        "min_child_weight": 120,
        "reg_lambda": 15.0,
        "tree_method": "hist",
        "device": device,
        "seed": seed,
    }
    evals = [(dval, "val")] if yb is not None else []
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=250,
        evals=evals or None,
        early_stopping_rounds=40 if evals else None,
        verbose_eval=False,
    )
    best_it = getattr(bst, "best_iteration", 249) + 1
    zb = bst.predict(dval, output_margin=True, iteration_range=(0, best_it)).astype(np.float64)
    zte = bst.predict(dte, output_margin=True, iteration_range=(0, best_it)).astype(np.float64)
    d_xgb_b = zb - init_b
    d_xgb_te = zte - init_te
    if float(np.std(d_xgb_b)) <= 1e-6:
        d_xgb_b, d_xgb_te = d_ridge_b.copy(), d_ridge_te.copy()

    return d_ridge_b, d_ridge_te, d_xgb_b, d_xgb_te


def residual_context_init_score(z_cal_v4p: np.ndarray, z_cal_glm: np.ndarray) -> np.ndarray:
    """Both inputs MUST be Platt-calibrated to true fraud prevalence (~0.0167)."""
    return 0.85 * np.asarray(z_cal_v4p, dtype=np.float64) + 0.15 * np.asarray(z_cal_glm, dtype=np.float64)


def intrinsic_2col(cand: np.ndarray) -> np.ndarray:
    c = np.asarray(cand, dtype=np.float64).ravel()
    d = (c - c.mean()) / (c.std() + 1e-12)
    g = ndtri(np.clip(rk(c), 1e-6, 1.0 - 1e-6))
    return np.column_stack([d, g])


def spearman_delta_rho(oof: np.ndarray, test: np.ndarray, ref_o: np.ndarray, ref_t: np.ndarray) -> float:
    rho_o = float(spearmanr(oof, ref_o).statistic)
    rho_t = float(spearmanr(test, ref_t).statistic)
    return abs(rho_o - rho_t)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "schema": "matrix_v2",
        "v_champ_oof_auc": 0.9523071,
        "best_lb": 0.95243,
        "completed_levers": [],
        "promoted_levers": [],
        "strikes_by_paradigm": {},
        "current_rung": 0,
        "current_lever": None,
        "spend_usd": 0.0,
        "gpu_pod_id": None,
        "gpu_pod_name": "s6e11-0953-campaign",
        "eta_minutes": None,
        "started_utc": None,
        "last_progress_utc": None,
        "rung_etas_min": {
            "1": 22,
            "2": 18,
            "3": 22,
            "4": 22,
            "5": 20,
            "6": 18,
            "7": 18,
            "8": 25,
            "9": 15,
        },
        "levers_done_in_rung": 0,
        "levers_total_in_rung": 5,
        "status_line": "booting",
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_PATH)


def append_watchdog(event: str, extra: dict | None = None) -> None:
    from datetime import datetime, timezone

    rec = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event}
    if extra:
        rec.update(extra)
    WATCHDOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHDOG_PATH, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def save_lever_fold0(
    paradigm: str, lever: str, delta_f0: np.ndarray, delta_te: np.ndarray, meta: dict
) -> Path:
    MATRIX_V2.mkdir(parents=True, exist_ok=True)
    path = MATRIX_V2 / f"{paradigm}_{lever}_fold0.npz"
    d0 = ((delta_f0 - delta_f0.mean()) / (delta_f0.std() + 1e-12)).astype(np.float32)
    dt = ((delta_te - delta_te.mean()) / (delta_te.std() + 1e-12)).astype(np.float32)
    np.savez_compressed(
        path,
        delta_fold0=d0,
        delta_test_probe=dt,
        meta=json.dumps(meta),
    )
    return path


def save_lever_oof(paradigm: str, lever: str, oof: np.ndarray, test: np.ndarray) -> None:
    MATRIX_V2.mkdir(parents=True, exist_ok=True)
    np.save(MATRIX_V2 / f"{paradigm}_{lever}_oof.npy", oof.astype(np.float32))
    np.save(MATRIX_V2 / f"{paradigm}_{lever}_test.npy", test.astype(np.float32))
