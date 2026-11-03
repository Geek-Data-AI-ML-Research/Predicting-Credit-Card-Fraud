#!/usr/bin/env python3
"""Stage 6.3: Self-trained shallow CatBoost + EBM (or additive fallback) on te_smooth50 view.

Pipeline: inside-fold TE m=50 + frequency encoding + quantized bin TEs for top risk scores.
Evaluate vs te_xgb_smooth50 (0.949447). NO kaggle submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from materialize_self_test_preds import build_frames, te_apply  # noqa: E402
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import (  # noqa: E402
    CATS,
    FLAGS,
    NUMS,
    SEED,
    TARGET,
    locked_folds,
    oof_target_encode,
)

warnings.filterwarnings("ignore")

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "stage6")
N_JOBS = max(1, (os.cpu_count() or 4) - 1)
CTRL_OOF = 0.949447
BIN_COLS = ["merchant_risk_score", "velocity_score", "amount_usd", "distance_from_home_km"]


def metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def freq_encode(series_fit: pd.Series, series_apply: pd.Series) -> np.ndarray:
    vc = series_fit.value_counts()
    return series_apply.map(vc).fillna(0).astype(np.float32).values


def quantized_bin_te(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_apply: np.ndarray,
    n_bins: int = 32,
    smooth: float = 50.0,
) -> np.ndarray:
    # quantile bins on fit
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x_fit, qs))
    if len(edges) < 3:
        return np.full(len(x_apply), float(y_fit.mean()), np.float32)
    b_fit = np.clip(np.digitize(x_fit, edges[1:-1], right=True), 0, len(edges) - 2)
    b_ap = np.clip(np.digitize(x_apply, edges[1:-1], right=True), 0, len(edges) - 2)
    prior = float(y_fit.mean())
    nmax = int(b_fit.max()) + 1
    s = np.bincount(b_fit, weights=y_fit, minlength=nmax)
    n = np.bincount(b_fit, minlength=nmax)
    return ((s[b_ap] + prior * smooth) / (n[b_ap] + smooth)).astype(np.float32)


def build_xy_smooth50(
    train,
    dig_tr,
    dig_te,
    dig_num,
    codes_tr,
    codes_te,
    te_names,
    y,
    tr_idx,
    va_idx,
    test_df,
    *,
    with_test: bool,
):
    smooth = 50.0
    te_tr, te_va = oof_target_encode(codes_tr, y, tr_idx, va_idx, smooth=smooth, seed=SEED)
    te_cols = [f"te_{c}" for c in te_names]
    # frequency encodings on cats
    freq_tr, freq_va = [], []
    freq_te = [] if with_test else None
    for c in CATS:
        freq_tr.append(freq_encode(train.iloc[tr_idx][c].astype(str), train.iloc[tr_idx][c].astype(str)))
        freq_va.append(freq_encode(train.iloc[tr_idx][c].astype(str), train.iloc[va_idx][c].astype(str)))
        if with_test:
            freq_te.append(freq_encode(train.iloc[tr_idx][c].astype(str), test_df[c].astype(str)))
    freq_tr = np.column_stack(freq_tr)
    freq_va = np.column_stack(freq_va)
    freq_names = [f"freq_{c}" for c in CATS]
    # quantized bin TE
    bin_tr, bin_va = [], []
    bin_te = [] if with_test else None
    for c in BIN_COLS:
        bin_tr.append(
            quantized_bin_te(
                train.iloc[tr_idx][c].values.astype(float),
                y[tr_idx],
                train.iloc[tr_idx][c].values.astype(float),
                smooth=smooth,
            )
        )
        bin_va.append(
            quantized_bin_te(
                train.iloc[tr_idx][c].values.astype(float),
                y[tr_idx],
                train.iloc[va_idx][c].values.astype(float),
                smooth=smooth,
            )
        )
        if with_test:
            bin_te.append(
                quantized_bin_te(
                    train.iloc[tr_idx][c].values.astype(float),
                    y[tr_idx],
                    test_df[c].values.astype(float),
                    smooth=smooth,
                )
            )
    bin_tr = np.column_stack(bin_tr)
    bin_va = np.column_stack(bin_va)
    bin_names = [f"binte_{c}" for c in BIN_COLS]

    num_tr = train.iloc[tr_idx][NUMS + FLAGS].reset_index(drop=True)
    num_va = train.iloc[va_idx][NUMS + FLAGS].reset_index(drop=True)
    d_tr = dig_tr.iloc[tr_idx][dig_num].reset_index(drop=True)
    d_va = dig_tr.iloc[va_idx][dig_num].reset_index(drop=True)
    def _as_float(*parts):
        X = pd.concat(parts, axis=1)
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X.astype(np.float32)

    X_tr = _as_float(
        num_tr,
        d_tr,
        pd.DataFrame(te_tr, columns=te_cols),
        pd.DataFrame(freq_tr, columns=freq_names),
        pd.DataFrame(bin_tr, columns=bin_names),
    )
    X_va = _as_float(
        num_va,
        d_va,
        pd.DataFrame(te_va, columns=te_cols),
        pd.DataFrame(freq_va, columns=freq_names),
        pd.DataFrame(bin_va, columns=bin_names),
    )
    X_te = None
    if with_test:
        te_te = te_apply(codes_tr[tr_idx], y[tr_idx], codes_te, smooth)
        num_te = test_df[NUMS + FLAGS].reset_index(drop=True)
        X_te = _as_float(
            num_te,
            dig_te[dig_num].reset_index(drop=True),
            pd.DataFrame(te_te, columns=te_cols),
            pd.DataFrame(np.column_stack(freq_te), columns=freq_names),
            pd.DataFrame(np.column_stack(bin_te), columns=bin_names),
        )
    return X_tr, X_va, X_te


def try_catboost():
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier
    except Exception:
        return None


def try_ebm():
    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier
    except Exception:
        return None


def run_arm(name, folds, y, fit_predict, control_folds, control_oof):
    oof = np.zeros(len(y), np.float64)
    test_sum = None
    recs = []
    t0 = time.time()
    print(f"\n=== Arm {name} ===", flush=True)
    for f, (tr, va) in enumerate(folds):
        tr_p, va_p, te_p, it = fit_predict(tr, va, f)
        oof[va] = va_p
        if te_p is not None:
            test_sum = te_p if test_sum is None else test_sum + te_p
        tr_m, va_m = metrics(y[tr], tr_p), metrics(y[va], va_p)
        gap = tr_m["roc_auc"] - va_m["roc_auc"]
        recs.append(
            {
                "fold": f,
                "valid_roc_auc": va_m["roc_auc"],
                "train_roc_auc": tr_m["roc_auc"],
                "train_valid_gap": float(gap),
                "valid_pr_auc": va_m["pr_auc"],
                "valid_brier": va_m["brier"],
                "best_iteration": it,
            }
        )
        print(
            f"  fold{f}: auc={va_m['roc_auc']:.6f} gap={gap:.6f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    oof_m = metrics(y, oof)
    folds_v = [r["valid_roc_auc"] for r in recs]
    gate = promotion_gate(control_folds, folds_v, control_oof, oof_m["roc_auc"])
    return {
        "name": name,
        "oof_roc_auc": oof_m["roc_auc"],
        "oof_pr_auc": oof_m["pr_auc"],
        "oof_brier": oof_m["brier"],
        "train_valid_gap_mean": float(np.mean([r["train_valid_gap"] for r in recs])),
        "folds": recs,
        "gate_vs_te_xgb_smooth50": gate,
        "oof": oof,
        "test": None if test_sum is None else test_sum / len(folds),
    }


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = locked_folds(y)
    # control fold vector from stage4 log if present
    ctrl_folds = None
    s4 = os.path.join(LOGS, "stage4_levers_result.json")
    if os.path.exists(s4):
        doc = json.load(open(s4))
        for a in doc.get("new_arms", doc.get("arms", [])):
            if a.get("name") == "te_xgb_smooth50":
                ctrl_folds = [f["valid_roc_auc"] for f in a["folds"]]
                ctrl_oof = a["oof_roc_auc"]
                break
    if ctrl_folds is None:
        ctrl_oof = CTRL_OOF
        # approximate from saved oof
        o = np.load(os.path.join(ROOT, "preds/stage4/te_xgb_smooth50_oof.npy"))
        ctrl_folds = [float(roc_auc_score(y[va], o[va])) for _, va in folds]
        ctrl_oof = float(roc_auc_score(y, o))

    print("building frames...", flush=True)
    dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = build_frames(train, test)
    os.makedirs(PRED, exist_ok=True)

    def make_fit(kind):
        CatBoostClassifier = try_catboost()
        EBM = try_ebm()

        def fit_predict(tr, va, f):
            X_tr, X_va, X_te = build_xy_smooth50(
                train,
                dig_tr,
                dig_te,
                dig_num,
                codes_tr,
                codes_te,
                te_names,
                y,
                tr,
                va,
                test,
                with_test=True,
            )
            if kind == "catboost" and CatBoostClassifier is not None:
                cb_kw = dict(
                    depth=3,
                    l2_leaf_reg=15,
                    learning_rate=0.04,
                    iterations=4000,
                    eval_metric="AUC",
                    random_seed=SEED + f,
                    od_type="Iter",
                    od_wait=200,
                    verbose=False,
                    thread_count=N_JOBS,
                )
                # Prefer GPU when available (Runpod / CUDA hosts)
                try:
                    import subprocess

                    if subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                        cb_kw["task_type"] = "GPU"
                        cb_kw["devices"] = "0"
                except Exception:
                    pass
                m = CatBoostClassifier(**cb_kw)
                m.fit(X_tr, y[tr], eval_set=(X_va, y[va]), use_best_model=True)
                tr_p = m.predict_proba(X_tr)[:, 1]
                va_p = m.predict_proba(X_va)[:, 1]
                te_p = m.predict_proba(X_te)[:, 1]
                it = int(m.get_best_iteration() or m.tree_count_)
                return tr_p, va_p, te_p, it
            if kind == "ebm" and EBM is not None:
                m = EBM(
                    max_bins=512,
                    interactions=15,
                    outer_bags=4,
                    random_state=SEED + f,
                    n_jobs=N_JOBS,
                )
                m.fit(X_tr, y[tr])
                tr_p = m.predict_proba(X_tr)[:, 1]
                va_p = m.predict_proba(X_va)[:, 1]
                te_p = m.predict_proba(X_te)[:, 1]
                return tr_p, va_p, te_p, -1
            # fallback: stacked additive stump boosters
            if kind in ("catboost", "additive_fallback", "ebm"):
                # depth-1 + depth-2 XGB/LGBM blend
                p1 = dict(
                    learning_rate=0.04,
                    max_depth=1,
                    min_child_weight=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=5.0,
                    tree_method="hist",
                    eval_metric="auc",
                    n_jobs=N_JOBS,
                )
                p2 = dict(
                    learning_rate=0.03,
                    num_leaves=3,
                    max_depth=2,
                    min_child_samples=80,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.85,
                    reg_lambda=5.0,
                    n_jobs=N_JOBS,
                    verbose=-1,
                )
                m1 = xgb.XGBClassifier(**p1, n_estimators=4000, random_state=SEED + f, early_stopping_rounds=200)
                m1.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
                m2 = lgb.LGBMClassifier(**p2, n_estimators=4000, random_state=SEED + f)
                m2.fit(
                    X_tr,
                    y[tr],
                    eval_set=[(X_va, y[va])],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(200, verbose=False)],
                )
                tr_p = 0.5 * (m1.predict_proba(X_tr)[:, 1] + m2.predict_proba(X_tr)[:, 1])
                va_p = 0.5 * (m1.predict_proba(X_va)[:, 1] + m2.predict_proba(X_va)[:, 1])
                te_p = 0.5 * (m1.predict_proba(X_te)[:, 1] + m2.predict_proba(X_te)[:, 1])
                return tr_p, va_p, te_p, int(m1.best_iteration + 1)
            raise RuntimeError(kind)

        return fit_predict

    arms = []
    # Arm A: CatBoost (or fallback)
    a = run_arm(
        "te_smooth50_catboost_d3" if try_catboost() else "te_smooth50_additive_fallback_A",
        folds,
        y,
        make_fit("catboost"),
        ctrl_folds,
        ctrl_oof,
    )
    np.save(os.path.join(PRED, f"{a['name']}_oof.npy"), a["oof"].astype(np.float32))
    if a["test"] is not None:
        np.save(os.path.join(PRED, f"{a['name']}_full.npy"), a["test"].astype(np.float32))
    arms.append({k: v for k, v in a.items() if k not in ("oof", "test")})

    # Arm B: EBM or additive fallback B
    bname = "te_smooth50_ebm" if try_ebm() else "te_smooth50_additive_stumps"
    # if catboost missing we already ran fallback once — still run stump blend labeled B
    b = run_arm(bname, folds, y, make_fit("ebm" if try_ebm() else "additive_fallback"), ctrl_folds, ctrl_oof)
    np.save(os.path.join(PRED, f"{b['name']}_oof.npy"), b["oof"].astype(np.float32))
    if b["test"] is not None:
        np.save(os.path.join(PRED, f"{b['name']}_full.npy"), b["test"].astype(np.float32))
    arms.append({k: v for k, v in b.items() if k not in ("oof", "test")})

    doc = {
        "stage": "6.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "control": {"name": "te_xgb_smooth50", "oof": ctrl_oof, "folds": ctrl_folds},
        "catboost_available": try_catboost() is not None,
        "ebm_available": try_ebm() is not None,
        "arms": arms,
        "promoted": [a["name"] for a in arms if a["gate_vs_te_xgb_smooth50"]["promote"]],
    }
    with open(os.path.join(LOGS, "stage6_3_additive_result.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps({"promoted": doc["promoted"], "oofs": [(a["name"], a["oof_roc_auc"]) for a in arms]}, indent=2))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
