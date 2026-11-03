#!/usr/bin/env python3
"""Generate leak-free OOF+test preds for promoted self models (smooth50, leaves3)."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from validation_stage3_baselines import (  # noqa: E402
    CATS,
    FLAGS,
    NUMS,
    SEED,
    TARGET,
    digit_frame,
    locked_folds,
    oof_target_encode,
    te_matrix,
)

DATA = os.path.join(ROOT, "data")
PRED4 = os.path.join(ROOT, "preds", "stage4")
PRED5 = os.path.join(ROOT, "preds", "stage5")
LOGS = os.path.join(ROOT, "logs")
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


def te_apply(codes_fit: np.ndarray, y_fit: np.ndarray, codes_apply: np.ndarray, smooth: float) -> np.ndarray:
    """TE from fit rows onto a separate apply codes matrix (same col cardinality)."""
    prior = float(y_fit.mean())
    n_cols = codes_fit.shape[1]
    out = np.empty((len(codes_apply), n_cols), np.float32)
    for j in range(n_cols):
        kf = codes_fit[:, j]
        ka = codes_apply[:, j]
        nmax = int(max(int(kf.max()), int(ka.max())) + 1)
        s = np.bincount(kf, weights=y_fit, minlength=nmax)
        n = np.bincount(kf, minlength=nmax)
        out[:, j] = (s[ka] + prior * smooth) / (n[ka] + smooth)
    return out


def build_frames(train, test):
    dig_tr = digit_frame(train)
    dig_te = digit_frame(test)
    dig_num = [c for c in dig_tr.columns if dig_tr[c].dtype != object]
    dig_cat = [c for c in dig_tr.columns if dig_tr[c].dtype == object]
    te_tr_src = pd.concat([train[CATS].astype(str), dig_tr[dig_cat].astype(str)], axis=1)
    te_te_src = pd.concat([test[CATS].astype(str), dig_te[dig_cat].astype(str)], axis=1)
    te_names = list(te_tr_src.columns)
    codes_tr = np.column_stack(
        [
            pd.Categorical(te_tr_src[c], categories=sorted(pd.concat([te_tr_src[c], te_te_src[c]]).unique())).codes.astype(
                np.int32
            )
            for c in te_names
        ]
    )
    codes_te = np.column_stack(
        [
            pd.Categorical(te_te_src[c], categories=sorted(pd.concat([te_tr_src[c], te_te_src[c]]).unique())).codes.astype(
                np.int32
            )
            for c in te_names
        ]
    )
    return dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te


def _as_float_frame(*parts: pd.DataFrame) -> pd.DataFrame:
    X = pd.concat(parts, axis=1)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(np.float32)


def xy_fold(train, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names, y, tr_idx, va_idx, smooth):
    te_tr, te_va = oof_target_encode(codes_tr, y, tr_idx, va_idx, smooth=smooth, seed=SEED)
    te_te = te_apply(codes_tr[tr_idx], y[tr_idx], codes_te, smooth)
    te_cols = [f"te_{c}" for c in te_names]
    num_tr = train.iloc[tr_idx][NUMS + FLAGS].reset_index(drop=True)
    num_va = train.iloc[va_idx][NUMS + FLAGS].reset_index(drop=True)
    d_tr = dig_tr.iloc[tr_idx][dig_num].reset_index(drop=True)
    d_va = dig_tr.iloc[va_idx][dig_num].reset_index(drop=True)
    X_tr = _as_float_frame(num_tr, d_tr, pd.DataFrame(te_tr, columns=te_cols))
    X_va = _as_float_frame(num_va, d_va, pd.DataFrame(te_va, columns=te_cols))
    return X_tr, X_va, te_te, te_cols


def run_xgb_smooth50(train, test, y, folds, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names):
    oof = np.zeros(len(y), np.float64)
    test_sum = np.zeros(len(test), np.float64)
    params = dict(
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=N_JOBS,
    )
    fold_aucs = []
    t0 = time.time()
    for f, (tr, va) in enumerate(folds):
        X_tr, X_va, te_te, te_cols = xy_fold(
            train, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names, y, tr, va, 50.0
        )
        num_te = test[NUMS + FLAGS].reset_index(drop=True)
        d_te = dig_te[dig_num].reset_index(drop=True)
        X_te = _as_float_frame(num_te, d_te, pd.DataFrame(te_te, columns=te_cols))
        m = xgb.XGBClassifier(**params, n_estimators=4000, random_state=SEED + f, early_stopping_rounds=200)
        m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
        oof[va] = m.predict_proba(X_va)[:, 1]
        test_sum += m.predict_proba(X_te)[:, 1]
        fa = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(fa)
        print(f"  te_xgb_smooth50 fold{f} auc={fa:.6f} it={m.best_iteration+1} ({time.time()-t0:.0f}s)", flush=True)
    return oof, test_sum / len(folds), fold_aucs


def run_lgbm_leaves3(train, test, y, folds, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names):
    oof = np.zeros(len(y), np.float64)
    test_sum = np.zeros(len(test), np.float64)
    params = dict(
        learning_rate=0.03,
        num_leaves=3,
        max_depth=2,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    fold_aucs = []
    t0 = time.time()
    for f, (tr, va) in enumerate(folds):
        X_tr, X_va, te_te, te_cols = xy_fold(
            train, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names, y, tr, va, 20.0
        )
        # Stage5 used smooth=20 default for leaves sweep
        num_te = test[NUMS + FLAGS].reset_index(drop=True)
        d_te = dig_te[dig_num].reset_index(drop=True)
        X_te = _as_float_frame(num_te, d_te, pd.DataFrame(te_te, columns=te_cols))
        m = lgb.LGBMClassifier(**params, n_estimators=3000, random_state=SEED + f)
        m.fit(
            X_tr,
            y[tr],
            eval_set=[(X_va, y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        oof[va] = m.predict_proba(X_va)[:, 1]
        test_sum += m.predict_proba(X_te)[:, 1]
        fa = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(fa)
        print(
            f"  te_lgbm_leaves3 fold{f} auc={fa:.6f} it={m.best_iteration_} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    return oof, test_sum / len(folds), fold_aucs


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = locked_folds(y)
    print("building digit/TE frames...", flush=True)
    dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = build_frames(train, test)
    os.makedirs(PRED4, exist_ok=True)
    os.makedirs(PRED5, exist_ok=True)

    print("\n=== te_xgb_smooth50 (test materialize) ===", flush=True)
    o1, t1, f1 = run_xgb_smooth50(
        train, test, y, folds, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names
    )
    np.save(os.path.join(PRED4, "te_xgb_smooth50_oof.npy"), o1.astype(np.float32))
    np.save(os.path.join(PRED4, "te_xgb_smooth50_full.npy"), t1.astype(np.float32))
    print(f"  OOF {roc_auc_score(y, o1):.6f}", flush=True)

    print("\n=== te_lgbm_leaves3 (test materialize) ===", flush=True)
    o2, t2, f2 = run_lgbm_leaves3(
        train, test, y, folds, dig_tr, dig_te, dig_num, codes_tr, codes_te, te_names
    )
    np.save(os.path.join(PRED5, "te_lgbm_leaves3_oof.npy"), o2.astype(np.float32))
    np.save(os.path.join(PRED5, "te_lgbm_leaves3_full.npy"), t2.astype(np.float32))
    print(f"  OOF {roc_auc_score(y, o2):.6f}", flush=True)

    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "te_xgb_smooth50": {
            "oof": float(roc_auc_score(y, o1)),
            "fold_aucs": f1,
            "test_mean": float(t1.mean()),
        },
        "te_lgbm_leaves3": {
            "oof": float(roc_auc_score(y, o2)),
            "fold_aucs": f2,
            "test_mean": float(t2.mean()),
        },
    }
    with open(os.path.join(LOGS, "materialize_self_test_preds.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
