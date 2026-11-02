#!/usr/bin/env python3
"""Stage 2: Master CV harness + promotion gate (Handbook 5.2 → S6E11).

Locked CV (Stage 1.3): StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

2.1 Per-fold: valid/train ROC-AUC, valid PR-AUC, valid Brier; OOF mean ± SD
2.2 Promote FE/HP only if OOF ROC-AUC improves AND ≥4/5 folds improve
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
TARGET = "is_fraud"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
MIN_FOLD_WINS = 4  # Stage 2.2: ≥4/5 folds must improve


def locked_skf() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def evaluate_handbook_cv(
    df_train: pd.DataFrame,
    features: list[str],
    model_params: dict,
    *,
    label: str = "model",
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    skf = locked_skf()
    oof_preds = np.zeros(len(df_train))
    fold_records = []
    y_all = df_train[TARGET].astype(int)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_train, y_all)):
        X_tr = df_train.iloc[tr_idx][features]
        y_tr = y_all.iloc[tr_idx]
        X_va = df_train.iloc[va_idx][features]
        y_va = y_all.iloc[va_idx]

        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        va_preds = model.predict_proba(X_va)[:, 1]
        tr_preds = model.predict_proba(X_tr)[:, 1]
        oof_preds[va_idx] = va_preds
        gap = float(roc_auc_score(y_tr, tr_preds) - roc_auc_score(y_va, va_preds))
        fold_records.append(
            {
                "fold": fold,
                "valid_roc_auc": float(roc_auc_score(y_va, va_preds)),
                "train_roc_auc": float(roc_auc_score(y_tr, tr_preds)),
                "train_valid_gap": gap,
                "valid_pr_auc": float(average_precision_score(y_va, va_preds)),
                "valid_brier": float(brier_score_loss(y_va, va_preds)),
            }
        )
        print(
            f"  [{label}] fold {fold}: valid_auc="
            f"{fold_records[-1]['valid_roc_auc']:.6f} "
            f"gap={gap:.6f} pr={fold_records[-1]['valid_pr_auc']:.6f} "
            f"brier={fold_records[-1]['valid_brier']:.6f}"
        )

    res_df = pd.DataFrame(fold_records)
    oof_roc = float(roc_auc_score(y_all, oof_preds))
    oof_pr = float(average_precision_score(y_all, oof_preds))
    oof_brier = float(brier_score_loss(y_all, oof_preds))
    summary = {
        "label": label,
        "params": {k: v for k, v in model_params.items() if k != "verbose"},
        "oof_roc_auc": oof_roc,
        "oof_pr_auc": oof_pr,
        "oof_brier": oof_brier,
        "fold_valid_roc_auc_mean": float(res_df["valid_roc_auc"].mean()),
        "fold_valid_roc_auc_std": float(res_df["valid_roc_auc"].std()),
        "train_valid_gap_mean": float(res_df["train_valid_gap"].mean()),
        "folds": fold_records,
        "cv_contract": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    }
    print(f"\n--- CV Summary [{label}] ---")
    print(
        f"OOF ROC-AUC : {oof_roc:.6f} "
        f"(Fold Mean: {summary['fold_valid_roc_auc_mean']:.6f} "
        f"± {summary['fold_valid_roc_auc_std']:.6f})"
    )
    print(f"Train−Valid Gap (mean): {summary['train_valid_gap_mean']:.6f}")
    print(f"OOF PR-AUC  : {oof_pr:.6f} | OOF Brier: {oof_brier:.6f}")
    print(res_df.round(6).to_string(index=False))
    return oof_preds, res_df, summary


def promotion_gate(
    baseline_folds: list[float],
    challenger_folds: list[float],
    baseline_oof: float,
    challenger_oof: float,
    *,
    min_fold_wins: int = MIN_FOLD_WINS,
    eps: float = 1e-12,
) -> dict:
    """Stage 2.2: promote iff OOF improves AND ≥min_fold_wins folds improve."""
    assert len(baseline_folds) == len(challenger_folds) == N_SPLITS
    deltas = [c - b for b, c in zip(baseline_folds, challenger_folds)]
    wins = sum(1 for d in deltas if d > eps)
    oof_delta = challenger_oof - baseline_oof
    oof_ok = oof_delta > eps
    folds_ok = wins >= min_fold_wins
    promote = bool(oof_ok and folds_ok)
    decision = {
        "promote": promote,
        "oof_delta": float(oof_delta),
        "oof_improved": oof_ok,
        "fold_wins": wins,
        "fold_deltas": [float(d) for d in deltas],
        "min_fold_wins_required": min_fold_wins,
        "folds_gate_passed": folds_ok,
        "rule": (
            f"Promote only if OOF ROC-AUC improves AND ≥{min_fold_wins}/{N_SPLITS} "
            "folds improve (reject fold-variance chasing)."
        ),
        "reason": (
            "PASS: OOF↑ and fold majority"
            if promote
            else (
                "REJECT: OOF did not improve"
                if not oof_ok
                else f"REJECT: only {wins}/{N_SPLITS} folds improved (need ≥{min_fold_wins})"
            )
        ),
    }
    print("\n--- Stage 2.2 Promotion gate ---")
    print(f"  OOF Δ = {oof_delta:+.6f}  (need > 0) → {'OK' if oof_ok else 'FAIL'}")
    print(
        f"  Fold wins = {wins}/{N_SPLITS}  (need ≥{min_fold_wins}) → "
        f"{'OK' if folds_ok else 'FAIL'}"
    )
    print(f"  fold Δ: {[f'{d:+.6f}' for d in deltas]}")
    print(f"  DECISION: {'PROMOTE' if promote else 'REJECT'} — {decision['reason']}")
    return decision


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> list[str]:
    cat_cols = train[features].select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    for col in cat_cols:
        cats = sorted(pd.concat([train[col].astype(str), test[col].astype(str)]).unique())
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col] = pd.Categorical(test[col].astype(str), categories=cats)
    return cat_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default=os.path.join(LOGS, "stage2_handbook_cv_result.json"))
    ap.add_argument("--skip-challenger", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    features = [c for c in test.columns if c != ID_COL]
    encode_categoricals(train, test, features)
    print(
        f"Loaded train={len(train):,} test={len(test):,} n_features={len(features)}"
    )
    print(f"CV contract: StratifiedKFold({N_SPLITS}, shuffle=True, random_state={SEED})")

    # Control baseline — shallow LGBM under locked folds (Stage 3.1 precursor)
    baseline_params = dict(
        n_estimators=2000,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    print("\n[Stage 2.1] Handbook CV logger — baseline (depth=4, leaves=15)")
    _, base_df, baseline = evaluate_handbook_cv(
        train, features, baseline_params, label="baseline_d4_l15"
    )

    gate = None
    challenger = None
    if not args.skip_challenger:
        # Challenger: deeper trees — often raises train AUC / gap without honest OOF lift
        challenger_params = dict(baseline_params)
        challenger_params.update(max_depth=8, num_leaves=63)
        print("\n[Stage 2.1] Challenger (depth=8, leaves=63) — exercises promotion gate")
        _, chal_df, challenger = evaluate_handbook_cv(
            train, features, challenger_params, label="challenger_d8_l63"
        )
        gate = promotion_gate(
            baseline_folds=base_df["valid_roc_auc"].tolist(),
            challenger_folds=chal_df["valid_roc_auc"].tolist(),
            baseline_oof=baseline["oof_roc_auc"],
            challenger_oof=challenger["oof_roc_auc"],
        )

    result = {
        "stage": 2,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "cv_contract": baseline["cv_contract"],
        "promotion_rule": (
            f"OOF ROC-AUC improves AND ≥{MIN_FOLD_WINS}/{N_SPLITS} folds improve"
        ),
        "baseline": baseline,
        "challenger": challenger,
        "promotion_gate": gate,
        "control_fold_vector": baseline["folds"],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
