#!/usr/bin/env python3
"""Stage 6.3b: Ultra-fine hist CUDA XGBoost on te_smooth50 (+ multi-scale TE).

NO kaggle submit. Gate vs te_xgb_smooth50 0.949447.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from te_smooth50_features import build_codes, build_fold_matrix  # noqa: E402
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import SEED, TARGET, locked_folds  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "stage6")
CTRL_OOF = 0.949447
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


def load_control(y, folds):
    s4 = os.path.join(LOGS, "stage4_levers_result.json")
    if os.path.exists(s4):
        doc = json.load(open(s4))
        for a in doc.get("new_arms", doc.get("arms", [])):
            if a.get("name") == "te_xgb_smooth50":
                return a["oof_roc_auc"], [f["valid_roc_auc"] for f in a["folds"]]
    p = os.path.join(ROOT, "preds", "stage4", "te_xgb_smooth50_oof.npy")
    o = np.load(p)
    return float(roc_auc_score(y, o)), [float(roc_auc_score(y[va], o[va])) for _, va in folds]


def fit_xgb(X_tr, y_tr, X_va, y_va, params, seed):
    kwargs = dict(params)
    n_est = int(kwargs.pop("n_estimators", 6000))
    m = xgb.XGBClassifier(
        **kwargs,
        n_estimators=n_est,
        random_state=seed,
        early_stopping_rounds=200,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def run_arm(name, params, build_xy, train, test, y, folds, ctrl_oof, ctrl_folds):
    oof = np.zeros(len(y), np.float64)
    test_sum = np.zeros(len(test), np.float64)
    fold_recs = []
    t0 = time.time()
    print(f"\n=== Arm {name} ===", flush=True)
    for f, (tr, va) in enumerate(folds):
        X_tr, X_va, X_te = build_xy(tr, va)
        cols = X_tr.columns
        X_va = X_va.reindex(columns=cols, fill_value=0)
        X_te = X_te.reindex(columns=cols, fill_value=0)
        m = fit_xgb(X_tr, y[tr], X_va, y[va], dict(params), SEED + f)
        tr_p = m.predict_proba(X_tr)[:, 1]
        va_p = m.predict_proba(X_va)[:, 1]
        te_p = m.predict_proba(X_te)[:, 1]
        oof[va] = va_p
        test_sum += te_p
        fa = float(roc_auc_score(y[va], va_p))
        gap = float(roc_auc_score(y[tr], tr_p) - fa)
        fold_recs.append(
            {
                "fold": f,
                "valid_roc_auc": fa,
                "train_valid_gap": gap,
                "best_iteration": int(m.best_iteration + 1),
            }
        )
        print(
            f"  fold{f}: auc={fa:.6f} gap={gap:.6f} it={m.best_iteration+1} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    oof_auc = float(roc_auc_score(y, oof))
    folds_v = [r["valid_roc_auc"] for r in fold_recs]
    gate = promotion_gate(ctrl_folds, folds_v, ctrl_oof, oof_auc)
    te = test_sum / len(folds)
    os.makedirs(PRED, exist_ok=True)
    np.save(os.path.join(PRED, f"{name}_oof.npy"), oof.astype(np.float32))
    np.save(os.path.join(PRED, f"{name}_full.npy"), te.astype(np.float32))
    return {
        "name": name,
        "oof_roc_auc": oof_auc,
        "folds": fold_recs,
        "gate_vs_te_xgb_smooth50": gate,
        "promote": gate["promote"],
        "params": params,
    }


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = locked_folds(y)
    ctrl_oof, ctrl_folds = load_control(y, folds)
    print(f"control={ctrl_oof:.6f} building codes...", flush=True)
    dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = build_codes(train, test)

    base_xgb = dict(
        max_depth=2,
        learning_rate=0.03,
        n_estimators=6000,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=30.0,
        reg_alpha=5.0,
        tree_method="hist",
        device="cuda",
        max_bin=4096,
        eval_metric="auc",
    )

    def xy50(tr, va):
        return build_fold_matrix(
            train,
            test,
            dig_tr,
            dig_te,
            dig_num,
            codes_tr,
            codes_te,
            te_names,
            y,
            tr,
            va,
            smooth=50.0,
            with_ratios=True,
        )

    def xy_multi(tr, va):
        return build_fold_matrix(
            train,
            test,
            dig_tr,
            dig_te,
            dig_num,
            codes_tr,
            codes_te,
            te_names,
            y,
            tr,
            va,
            smooth=100.0,
            extra_smooth=20.0,
            with_ratios=True,
        )

    arms = []
    arms.append(
        run_arm(
            "te_xgb_smooth50_bin4096",
            dict(base_xgb),
            xy50,
            train,
            test,
            y,
            folds,
            ctrl_oof,
            ctrl_folds,
        )
    )
    arms.append(
        run_arm(
            "te_xgb_smooth100_depth2",
            dict(base_xgb),
            xy_multi,
            train,
            test,
            y,
            folds,
            ctrl_oof,
            ctrl_folds,
        )
    )

    doc = {
        "stage": "6.3b",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "device": "cuda",
        "control": {"name": "te_xgb_smooth50", "oof": ctrl_oof, "folds": ctrl_folds},
        "arms": arms,
        "promoted": [a["name"] for a in arms if a["promote"]],
        "policy": "NO_KAGGLE_SUBMIT",
    }
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, "stage6_3b_xgb_cuda_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(json.dumps({"promoted": doc["promoted"], "oofs": [(a["name"], a["oof_roc_auc"]) for a in arms]}, indent=2))
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
