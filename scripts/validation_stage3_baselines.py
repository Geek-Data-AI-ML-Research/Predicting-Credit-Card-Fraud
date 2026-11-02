#!/usr/bin/env python3
"""Stage 3: Baseline arms under locked StratifiedKFold(5, shuffle=True, rs=42).

3.1 Shallow LightGBM / XGB — raw / digit / TE views as separate arms
3.2 Record fold vector + Train−Valid gap as control for later levers
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "stage3")

TARGET = "is_fraud"
ID_COL = "id"
N_SPLITS, SEED = 5, 42
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUMS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
DEC = {
    "amount_usd": 2,
    "account_balance_usd": 2,
    "hours_since_last_txn": 2,
    "distance_from_home_km": 2,
    "velocity_score": 1,
    "merchant_risk_score": 1,
}


def locked_folds(y: np.ndarray):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def digit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Digit-key style features from float columns (public shallow-digit pattern)."""
    out = {}
    for c, k in DEC.items():
        s = df[c].map(lambda v, kk=k: f"{v:.{kk}f}")
        parts = s.str.split(".", n=1, expand=True)
        ip, fp = parts[0], parts[1]
        L = ip.str.len().astype(str)
        out[f"int_{c}"] = ip
        out[f"frac_{c}"] = fp
        out[f"ten_{c}"] = np.floor(df[c] / 10).astype("int64").astype(str)
        out[f"dec_{c}"] = np.floor(df[c] * 10 + 1e-6).astype("int64").astype(str)
        out[f"l1_{c}"] = L + "|" + ip.str.slice(0, 1)
        out[f"l2_{c}"] = L + "|" + ip.str.slice(0, 2)
        # numeric digit crumbs
        scale = 10 ** k
        kk = np.round(df[c].values * scale).astype(np.int64)
        out[f"cnt_{c}"] = df[c].map(df[c].value_counts()).astype(np.float32)
        out[f"fracn_{c}"] = (kk % scale).astype(np.int16)
        out[f"d1_{c}"] = ((kk // 10 ** max(k - 1, 0)) % 10).astype(np.int8)
    return pd.DataFrame(out)


def te_matrix(
    codes: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    *,
    smooth: float = 20.0,
) -> np.ndarray:
    prior = float(y[fit_idx].mean())
    n_cols = codes.shape[1]
    out = np.empty((len(apply_idx), n_cols), np.float32)
    yf = y[fit_idx]
    for j in range(n_cols):
        kf = codes[fit_idx, j]
        ka = codes[apply_idx, j]
        nmax = int(max(kf.max(), ka.max()) + 1)
        s = np.bincount(kf, weights=yf, minlength=nmax)
        n = np.bincount(kf, minlength=nmax)
        out[:, j] = (s[ka] + prior * smooth) / (n[ka] + smooth)
    return out


def oof_target_encode(
    codes: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    *,
    smooth: float = 20.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner 5-fold TE on train_idx; TE valid from full train_idx."""
    prior = float(y[train_idx].mean())
    inner = np.empty((len(train_idx), codes.shape[1]), np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for a, b in skf.split(train_idx, y[train_idx]):
        inner[b] = te_matrix(codes, y, train_idx[a], train_idx[b], smooth=smooth)
    valid = te_matrix(codes, y, train_idx, valid_idx, smooth=smooth)
    _ = prior
    return inner, valid


def metrics(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def run_arm(
    name: str,
    kind: str,
    folds,
    y: np.ndarray,
    build_xy,
    *,
    model_params: dict,
    n_estimators: int = 4000,
) -> dict:
    oof = np.zeros(len(y), dtype=np.float64)
    fold_recs = []
    t0 = time.time()
    print(f"\n=== Arm {name} ({kind}) ===", flush=True)

    for f, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = build_xy(tr_idx, va_idx)
        y_tr, y_va = y[tr_idx], y[va_idx]

        if kind == "lgb":
            model = lgb.LGBMClassifier(
                **model_params, n_estimators=n_estimators, random_state=SEED + f
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            tr_p = model.predict_proba(X_tr)[:, 1]
            va_p = model.predict_proba(X_va)[:, 1]
            best_iter = int(model.best_iteration_ or n_estimators)
        else:
            model = xgb.XGBClassifier(
                **model_params,
                n_estimators=n_estimators,
                random_state=SEED + f,
                early_stopping_rounds=200,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            tr_p = model.predict_proba(X_tr)[:, 1]
            va_p = model.predict_proba(X_va)[:, 1]
            best_iter = int(model.best_iteration + 1)

        oof[va_idx] = va_p
        tr_m, va_m = metrics(y_tr, tr_p), metrics(y_va, va_p)
        gap = tr_m["roc_auc"] - va_m["roc_auc"]
        rec = {
            "fold": f,
            "valid_roc_auc": va_m["roc_auc"],
            "train_roc_auc": tr_m["roc_auc"],
            "train_valid_gap": float(gap),
            "valid_pr_auc": va_m["pr_auc"],
            "valid_brier": va_m["brier"],
            "best_iteration": best_iter,
        }
        fold_recs.append(rec)
        print(
            f"  fold {f}: valid_auc={va_m['roc_auc']:.6f} gap={gap:.6f} "
            f"pr={va_m['pr_auc']:.6f} trees={best_iter} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    oof_m = metrics(y, oof)
    folds_auc = [r["valid_roc_auc"] for r in fold_recs]
    summary = {
        "name": name,
        "kind": kind,
        "params": {k: v for k, v in model_params.items() if k not in ("verbose",)},
        "oof_roc_auc": oof_m["roc_auc"],
        "oof_pr_auc": oof_m["pr_auc"],
        "oof_brier": oof_m["brier"],
        "fold_valid_roc_auc_mean": float(np.mean(folds_auc)),
        "fold_valid_roc_auc_std": float(np.std(folds_auc, ddof=1)),
        "train_valid_gap_mean": float(np.mean([r["train_valid_gap"] for r in fold_recs])),
        "folds": fold_recs,
        "elapsed_sec": float(time.time() - t0),
    }
    print(
        f"==> {name}: OOF AUC={summary['oof_roc_auc']:.6f} "
        f"({summary['fold_valid_roc_auc_mean']:.6f}±{summary['fold_valid_roc_auc_std']:.6f}) "
        f"gap={summary['train_valid_gap_mean']:.6f}",
        flush=True,
    )
    np.save(os.path.join(PRED, f"{name}_oof.npy"), oof.astype(np.float32))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default=os.path.join(LOGS, "stage3_baselines_result.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="all", help="comma list or all")
    args = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.smoke:
        # stratified sample for dry-run
        pos = train[train[TARGET] == 1]
        neg = train[train[TARGET] == 0].sample(n=min(20000, len(train)), random_state=SEED)
        train = pd.concat([pos.sample(n=min(2000, len(pos)), random_state=SEED), neg]).sample(
            frac=1, random_state=SEED
        )
        test = test.iloc[:5000].copy()

    y = train[TARGET].astype(int).to_numpy()
    for c in CATS:
        cats = sorted(pd.concat([train[c].astype(str), test[c].astype(str)]).unique())
        train[c] = pd.Categorical(train[c].astype(str), categories=cats)

    folds = locked_folds(y)
    print(
        f"train={len(train):,} fraud={y.mean():.4%} | "
        f"CV=StratifiedKFold({N_SPLITS}, shuffle=True, rs={SEED}) | cpus={N_JOBS}",
        flush=True,
    )

    # --- feature views ---
    raw_cols = NUMS + FLAGS + CATS
    dig = digit_frame(train)
    dig_num_cols = [
        c
        for c in dig.columns
        if pd.api.types.is_numeric_dtype(dig[c])
    ]
    dig_cat_cols = [c for c in dig.columns if c not in dig_num_cols]
    for c in dig_cat_cols:
        dig[c] = pd.Categorical(dig[c].astype(str))

    # codes for TE: cats + digit string keys
    te_source = pd.concat([train[CATS].astype(str), dig[dig_cat_cols].astype(str)], axis=1)
    te_names = list(te_source.columns)
    codes = np.column_stack(
        [pd.factorize(te_source[c], sort=True)[0].astype(np.int32) for c in te_names]
    )

    LGB_SHALLOW = dict(
        learning_rate=0.03,
        num_leaves=4,
        max_depth=4,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    XGB_SHALLOW = dict(
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=N_JOBS,
        enable_categorical=True,
        max_cat_to_onehot=1,
    )

    def xy_raw(tr_idx, va_idx):
        cols = raw_cols
        return train.iloc[tr_idx][cols], train.iloc[va_idx][cols]

    def xy_digit(tr_idx, va_idx):
        base_tr = train.iloc[tr_idx][raw_cols].reset_index(drop=True)
        base_va = train.iloc[va_idx][raw_cols].reset_index(drop=True)
        d_tr = dig.iloc[tr_idx].reset_index(drop=True)
        d_va = dig.iloc[va_idx].reset_index(drop=True)
        return pd.concat([base_tr, d_tr], axis=1), pd.concat([base_va, d_va], axis=1)

    def xy_te(tr_idx, va_idx):
        # numeric + flags + OOF TE of cats/digit-keys
        te_tr, te_va = oof_target_encode(codes, y, tr_idx, va_idx, smooth=20.0, seed=SEED)
        te_cols = [f"te_{c}" for c in te_names]
        num_tr = train.iloc[tr_idx][NUMS + FLAGS].reset_index(drop=True)
        num_va = train.iloc[va_idx][NUMS + FLAGS].reset_index(drop=True)
        # add numeric digit crumbs only
        dnum_tr = dig.iloc[tr_idx][dig_num_cols].reset_index(drop=True)
        dnum_va = dig.iloc[va_idx][dig_num_cols].reset_index(drop=True)
        X_tr = pd.concat(
            [num_tr, dnum_tr, pd.DataFrame(te_tr, columns=te_cols)], axis=1
        )
        X_va = pd.concat(
            [num_va, dnum_va, pd.DataFrame(te_va, columns=te_cols)], axis=1
        )
        return X_tr, X_va

    arm_specs = {
        "raw_lgbm": ("lgb", xy_raw, LGB_SHALLOW),
        "digit_lgbm": ("lgb", xy_digit, LGB_SHALLOW),
        "te_lgbm": ("lgb", xy_te, LGB_SHALLOW),
        "raw_xgb": ("xgb", xy_raw, XGB_SHALLOW),
        "digit_xgb": ("xgb", xy_digit, XGB_SHALLOW),
        "te_xgb": ("xgb", xy_te, XGB_SHALLOW),
    }
    want = (
        list(arm_specs.keys())
        if args.arms.strip() == "all"
        else [a.strip() for a in args.arms.split(",") if a.strip()]
    )

    results = []
    for name in want:
        kind, builder, params = arm_specs[name]
        results.append(
            run_arm(name, kind, folds, y, builder, model_params=dict(params))
        )

    # control table sorted by OOF AUC
    ranked = sorted(results, key=lambda r: r["oof_roc_auc"], reverse=True)
    control = {
        "best_arm": ranked[0]["name"] if ranked else None,
        "best_oof_roc_auc": ranked[0]["oof_roc_auc"] if ranked else None,
        "arms_ranked": [
            {
                "name": r["name"],
                "oof_roc_auc": r["oof_roc_auc"],
                "fold_mean": r["fold_valid_roc_auc_mean"],
                "fold_std": r["fold_valid_roc_auc_std"],
                "gap_mean": r["train_valid_gap_mean"],
                "fold_vector": [f["valid_roc_auc"] for f in r["folds"]],
            }
            for r in ranked
        ],
    }
    print("\n=== Stage 3.2 control table ===")
    for row in control["arms_ranked"]:
        print(
            f"  {row['name']:12s} OOF={row['oof_roc_auc']:.6f} "
            f"gap={row['gap_mean']:.6f} folds={[round(x, 5) for x in row['fold_vector']]}"
        )

    out = {
        "stage": 3,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "cv_contract": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "promotion_rule": "OOF ROC-AUC↑ AND ≥4/5 folds↑ (Stage 2.2)",
        "n_train": int(len(train)),
        "fraud_rate": float(y.mean()),
        "arms": results,
        "control": control,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"OOF preds in {PRED}/")


if __name__ == "__main__":
    main()
