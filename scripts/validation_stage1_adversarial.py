#!/usr/bin/env python3
"""Stage 1: Adversarial validation + chronology audit (Handbook 5.2 → S6E11).

1.1 Chronology audit on id / day_of_week / time_of_day_hour
1.2 Adversarial validation train↔test (target Adv AUC ≈ 0.50)
1.3 Decision lock: Adv≈0.50 + no time order → StratifiedKFold(5, shuffle=True, rs=42)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
TARGET = "is_fraud"
ID_COL = "id"
CHRONO_COLS = ("id", "day_of_week", "time_of_day_hour")


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def chronology_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("\n[Stage 1.1] Chronology audit")
    y = train[TARGET].astype(int).to_numpy()
    report: dict = {"columns": {}, "time_order_suspected": False, "notes": []}

    # id range / overlap — sequential blocks would suggest prequential CV
    if ID_COL in train.columns and ID_COL in test.columns:
        tr_ids = train[ID_COL]
        te_ids = test[ID_COL]
        overlap = int(len(set(tr_ids) & set(te_ids)))
        tr_sorted = tr_ids.is_monotonic_increasing
        # Fraud rate by id quintile on train
        q = pd.qcut(tr_ids, 5, labels=False, duplicates="drop")
        fraud_by_q = (
            pd.DataFrame({"q": q, "y": y}).groupby("q")["y"].mean().to_dict()
        )
        id_info = {
            "train_min": int(tr_ids.min()),
            "train_max": int(tr_ids.max()),
            "test_min": int(te_ids.min()),
            "test_max": int(te_ids.max()),
            "overlap_n": overlap,
            "train_monotonic_increasing": bool(tr_sorted),
            "train_fraud_rate_by_id_quintile": {str(k): float(v) for k, v in fraud_by_q.items()},
            "univariate_fraud_auc": _safe_auc(y, tr_ids.to_numpy()),
        }
        # Contiguous non-overlap is common on Kaggle (train written then test) and is
        # NOT alone evidence of a temporal DGP split — need fraud drift or Adv lift.
        contiguous = (tr_ids.max() < te_ids.min()) or (te_ids.max() < tr_ids.min())
        id_info["contiguous_non_overlapping_ranges"] = bool(contiguous)
        rates = list(fraud_by_q.values()) if fraud_by_q else []
        fraud_drift_pp = (max(rates) - min(rates)) * 100 if rates else 0.0
        id_info["fraud_rate_quintile_range_pp"] = float(fraud_drift_pp)
        if contiguous:
            report["notes"].append(
                "id ranges contiguous non-overlapping (often a Kaggle write artifact; "
                "not sufficient alone for prequential CV)"
            )
        if fraud_drift_pp > 5.0:
            report["time_order_suspected"] = True
            report["notes"].append(
                f"fraud rate varies {fraud_drift_pp:.2f}pp across id quintiles (>5pp)"
            )
        report["columns"]["id"] = id_info
        print(
            f"  id: train[{id_info['train_min']}, {id_info['train_max']}] "
            f"test[{id_info['test_min']}, {id_info['test_max']}] "
            f"overlap={overlap} contiguous_blocks={contiguous} "
            f"fraud_AUC={id_info['univariate_fraud_auc']}"
        )
        print(f"       fraud_rate_by_id_quintile={id_info['train_fraud_rate_by_id_quintile']}")

    for col in ("day_of_week", "time_of_day_hour"):
        if col not in train.columns:
            continue
        tr = train[col]
        te = test[col]
        tr_dist = tr.value_counts(normalize=True).sort_index()
        te_dist = te.value_counts(normalize=True).sort_index()
        # L1 distance between categorical/hourly distributions
        idx = sorted(set(tr_dist.index) | set(te_dist.index))
        l1 = float(
            sum(abs(tr_dist.get(i, 0.0) - te_dist.get(i, 0.0)) for i in idx)
        )
        fraud_by_val = (
            pd.DataFrame({"v": tr, "y": y}).groupby("v")["y"].mean().to_dict()
        )
        info = {
            "train_min": float(tr.min()),
            "train_max": float(tr.max()),
            "test_min": float(te.min()),
            "test_max": float(te.max()),
            "train_mean": float(tr.mean()),
            "test_mean": float(te.mean()),
            "dist_l1_train_vs_test": l1,
            "univariate_fraud_auc": _safe_auc(y, tr.to_numpy()),
            "train_fraud_rate_by_value": {str(k): float(v) for k, v in fraud_by_val.items()},
        }
        # Large train↔test dist shift on time features ⇒ possible chronology mismatch
        if l1 > 0.15:
            report["time_order_suspected"] = True
            report["notes"].append(f"{col}: train↔test L1 dist={l1:.3f} (>0.15)")
        report["columns"][col] = info
        print(
            f"  {col}: train[{info['train_min']}, {info['train_max']}] "
            f"test[{info['test_min']}, {info['test_max']}] "
            f"L1(train,test)={l1:.4f} fraud_AUC={info['univariate_fraud_auc']}"
        )

    print(f"  time_order_suspected={report['time_order_suspected']}")
    if report["notes"]:
        for n in report["notes"]:
            print(f"  note: {n}")
    return report


def run_adversarial_validation(df_train, df_test, features):
    print("\n[Stage 1.2] Adversarial validation (train↔test)")
    adv_df = pd.concat(
        [
            df_train[features].assign(is_test=0),
            df_test[features].assign(is_test=1),
        ],
        ignore_index=True,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(adv_df))
    importances = np.zeros(len(features))
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(adv_df, adv_df["is_test"]), 1):
        X_tr = adv_df.iloc[tr_idx][features]
        y_tr = adv_df.iloc[tr_idx]["is_test"]
        X_va = adv_df.iloc[va_idx][features]
        y_va = adv_df.iloc[va_idx]["is_test"]
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        importances += model.feature_importances_ / 5
        fold_auc = float(roc_auc_score(y_va, oof[va_idx]))
        fold_aucs.append(fold_auc)
        print(f"  fold {fold}: Adv AUC={fold_auc:.5f}")

    adv_auc = float(roc_auc_score(adv_df["is_test"], oof))
    top_shift = (
        pd.Series(importances, index=features).sort_values(ascending=False).head(10)
    )
    print(f"  OOF Adversarial AUC: {adv_auc:.5f}  (folds mean±sd "
          f"{np.mean(fold_aucs):.5f}±{np.std(fold_aucs):.5f})")
    print("  Top distinguishing features:")
    for name, imp in top_shift.items():
        print(f"    {name}: {imp:.1f}")
    return adv_auc, fold_aucs, top_shift


def decision_lock(adv_auc: float, chrono: dict) -> dict:
    print("\n[Stage 1.3] Decision lock")
    iid_ok = adv_auc < 0.52
    no_time = not chrono.get("time_order_suspected", False)
    if iid_ok and no_time:
        cv = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
        rationale = (
            f"Adv AUC={adv_auc:.5f} ≈ 0.50 and chronology audit found no time-order "
            "structure → lock shuffled stratified 5-fold."
        )
        locked = "stratified_kfold_5_rs42"
    else:
        cv = "prequential / time-block CV (design blocks from id or time features)"
        rationale = (
            f"Adv AUC={adv_auc:.5f} and/or time_order_suspected="
            f"{chrono.get('time_order_suspected')} → do NOT use shuffled K-Fold; "
            "design prequential blocks."
        )
        locked = "prequential_blocks"
    print(f"  locked_cv={locked}")
    print(f"  contract: {cv}")
    print(f"  rationale: {rationale}")
    return {
        "locked_cv": locked,
        "cv_contract": cv,
        "adv_auc": adv_auc,
        "adv_iid_ok": iid_ok,
        "time_order_suspected": not no_time,
        "rationale": rationale,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default=os.path.join(LOGS, "stage1_adversarial_result.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    print(f"Loaded train={len(train):,} test={len(test):,}")

    features = [c for c in test.columns if c != ID_COL]
    cat_cols = train[features].select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    for col in cat_cols:
        cats = sorted(pd.concat([train[col].astype(str), test[col].astype(str)]).unique())
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col] = pd.Categorical(test[col].astype(str), categories=cats)

    chrono = chronology_audit(train, test)
    # Adversarial features: include chrono cols (except pure id as label leak of row identity —
    # id is unique per row across concat so would trivially separate; exclude id).
    adv_features = [c for c in features if c != ID_COL]
    adv_auc, fold_aucs, top_shift = run_adversarial_validation(train, test, adv_features)
    decision = decision_lock(adv_auc, chrono)

    result = {
        "stage": 1,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "chronology": chrono,
        "adversarial": {
            "oof_auc": adv_auc,
            "fold_aucs": fold_aucs,
            "top_features": {k: float(v) for k, v in top_shift.items()},
        },
        "decision": decision,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
