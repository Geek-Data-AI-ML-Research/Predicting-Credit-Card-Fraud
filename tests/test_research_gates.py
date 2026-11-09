"""Mandatory pytest anti-hallucination and anti-drift verification harness (Section 0d)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def rk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / len(x)


def rlogit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def eval_fold0_2param_canary(z_champ_f0: np.ndarray, delta_f0: np.ndarray, y_f0: np.ndarray) -> tuple[float, float, np.ndarray]:
    """5-subfold cross-validated 2-parameter logistic probe inside Fold 0 on frozen z_champ_f0."""
    base_auc = float(roc_auc_score(y_f0, z_champ_f0))
    d_std = (delta_f0 - delta_f0.mean()) / (delta_f0.std() + 1e-12)
    d_gauss = ndtri(np.clip(rk(delta_f0), 1e-6, 1.0 - 1e-6))
    X_sub = np.column_stack([d_std, d_gauss])

    sub_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))
    best_auc = base_auc
    best_pred = z_champ_f0.copy()

    # Check both small ridge-shrunk logistic offset and direct step grid
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

    # Also check 1-parameter cross-validated step along d_std or d_gauss
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


def test_step0_backbone_and_splits():
    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    assert len(y) == 699960
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    sizes = [len(b) for _, b in folds]
    assert sizes == [139992, 139992, 139992, 139992, 139992]

    v4p_oof = np.load(ROOT / "preds" / "P_super_v4p_oof.npy").ravel()
    v4p_test = np.load(ROOT / "preds" / "P_super_v4p_test.npy").ravel()
    assert np.isfinite(v4p_oof).all() and np.isfinite(v4p_test).all()
    assert v4p_oof.shape == (699960,) and v4p_test.shape == (299984,)

    a0, b0 = folds[0]
    assert roc_auc_score(y, v4p_oof) == pytest.approx(0.9522827, abs=1e-6)
    assert roc_auc_score(y[b0], v4p_oof[b0]) == pytest.approx(0.9513610, abs=1e-6)

    from build_psuper_v4p_bank import build_and_cache_v4p_bank

    bank = build_and_cache_v4p_bank(force=False)
    assert np.max(np.abs(bank["v4p_oof"] - v4p_oof)) < 1e-6
    assert roc_auc_score(y, bank["v4p_hc_oof"]) >= 0.952305

    state_p = ROOT / "logs" / "overnight" / "matrix_8x5_state.json"
    if state_p.exists():
        st = json.loads(state_p.read_text())
        assert st.get("gpu_pod_id") != "lheroidmuab1j4"
        assert float(st.get("spend_usd", 0.0)) <= 20.00


def verify_lever_fold0_disk(paradigm_id: str, lever_id: str, matrix_dir: str = "matrix_v2") -> dict:
    """Programmatic Fold-0 canary gate verification called by pytest and orchestrator."""
    npz_path = ROOT / "preds" / matrix_dir / f"{paradigm_id}_{lever_id}_fold0.npz"
    if not npz_path.exists():
        npz_path = ROOT / "preds" / "matrix_8x5" / f"{paradigm_id}_{lever_id}_fold0.npz"
    assert npz_path.exists(), f"Missing {npz_path}"
    d = np.load(npz_path)
    delta_f0 = d["delta_fold0"].astype(np.float64).ravel()
    delta_te = d["delta_test_probe"].astype(np.float64).ravel()
    assert delta_f0.shape == (139992,) and delta_te.shape == (299984,)
    assert np.isfinite(delta_f0).all() and np.isfinite(delta_te).all()
    assert float(np.std(delta_f0)) > 1e-5, "Constant delta_fold0"

    ks_stat = float(ks_2samp(delta_f0, delta_te).statistic)
    # Per-feature KS filter target is 0.012; residual canary allows up to ~0.05 on raw scale
    assert abs(float(delta_f0.mean() - delta_te.mean())) < 0.05, "Mean drift between Fold0 and Test"
    assert ks_stat < 0.05, f"KS drift too high: {ks_stat}"

    bank = np.load(ROOT / "preds" / "v4p_bank_cache.npz")
    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    _, b0 = folds[0]
    z_champ_f0 = bank["z_champ_oof"][b0].astype(np.float64)
    auc_f0, delta_auc, _ = eval_fold0_2param_canary(z_champ_f0, delta_f0, y[b0])
    return {
        "paradigm_id": paradigm_id,
        "lever_id": lever_id,
        "fold0_canary_auc": auc_f0,
        "delta_fold0": delta_auc,
        "ks_stat": ks_stat,
        "passed_fold0": bool(delta_auc >= 1.0e-5),
    }


def verify_lever_5fold_disk(paradigm_id: str, lever_id: str, z_champ_oof: np.ndarray, z_champ_test: np.ndarray) -> dict:
    """Programmatic 5-fold promotion gate verification with anti-leakage and paired bootstrap."""
    oof_p = ROOT / "preds" / "matrix_8x5" / f"{paradigm_id}_{lever_id}_oof.npy"
    te_p = ROOT / "preds" / "matrix_8x5" / f"{paradigm_id}_{lever_id}_test.npy"
    assert oof_p.exists() and te_p.exists()
    oof = np.load(oof_p).astype(np.float64).ravel()
    test = np.load(te_p).astype(np.float64).ravel()
    assert oof.shape == (699960,) and test.shape == (299984,)
    assert np.isfinite(oof).all() and np.isfinite(test).all()

    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    v4p_oof = np.load(ROOT / "preds" / "P_super_v4p_oof.npy").ravel()
    v4p_test = np.load(ROOT / "preds" / "P_super_v4p_test.npy").ravel()

    # Standardize oof per fold if raw residual has cross-fold mean shift
    fold_means = [float(np.mean(oof[b])) for _, b in folds]
    fold_stds = [float(np.std(oof[b])) for _, b in folds]
    for k, (_, b) in enumerate(folds):
        assert fold_stds[k] > 1e-5, f"Fold {k} std is zero"

    rho_o = float(spearmanr(oof, v4p_oof).statistic)
    rho_t = float(spearmanr(test, v4p_test).statistic)
    delta_rho = abs(rho_o - rho_t)

    # Out-of-fold 2-feature addition on top of z_champ_oof
    base_auc = float(roc_auc_score(y, z_champ_oof))
    d_o = (oof - oof.mean()) / (oof.std() + 1e-12)
    d_t = (test - oof.mean()) / (oof.std() + 1e-12)
    g_o = ndtri(np.clip(rk(oof), 1e-6, 1.0 - 1e-6))
    g_t = ndtri(np.clip(rk(test), 1e-6, 1.0 - 1e-6))

    best_auc = base_auc
    best_oof = z_champ_oof.copy()
    best_test = z_champ_test.copy()

    alphas = (-0.12, -0.06, -0.03, -0.015, -0.008, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12)
    for col_o, col_t in ((d_o, d_t), (g_o, g_t)):
        cand_o = np.zeros(len(y), dtype=np.float64)
        cand_t_folds = []
        for a, b in folds:
            best_al = 0.0
            best_sc = float(roc_auc_score(y[a], z_champ_oof[a]))
            for al in alphas:
                sc = float(roc_auc_score(y[a], z_champ_oof[a] + al * col_o[a]))
                if sc > best_sc:
                    best_sc = sc
                    best_al = al
            cand_o[b] = z_champ_oof[b] + best_al * col_o[b]
            cand_t_folds.append(z_champ_test + best_al * col_t)
        a_cand = float(roc_auc_score(y, cand_o))
        if a_cand > best_auc:
            best_auc = a_cand
            best_oof = cand_o
            best_test = np.mean(cand_t_folds, axis=0)

    delta_oof = best_auc - base_auc
    rng = np.random.default_rng(42)
    boot_deltas = []
    for _ in range(250):
        idx = np.concatenate([b[rng.integers(0, len(b), len(b))] for _, b in folds])
        boot_deltas.append(float(roc_auc_score(y[idx], best_oof[idx]) - roc_auc_score(y[idx], z_champ_oof[idx])))
    ci90_low = float(np.quantile(boot_deltas, 0.05))
    ci90_high = float(np.quantile(boot_deltas, 0.95))

    return {
        "paradigm_id": paradigm_id,
        "lever_id": lever_id,
        "base_auc": base_auc,
        "new_auc": best_auc,
        "delta_oof": delta_oof,
        "ci90": [ci90_low, ci90_high],
        "delta_rho": delta_rho,
        "promoted": bool(delta_oof >= 1.0e-5 and ci90_low > 0.0 and delta_rho < 0.0025),
        "best_oof": best_oof,
        "best_test": best_test,
    }
