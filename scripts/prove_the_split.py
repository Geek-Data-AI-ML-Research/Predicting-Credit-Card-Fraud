#!/usr/bin/env python3
"""S6E11 core rule — prove how train was separated from test, then lock CV.

Handbook 5.2: validation must mirror the train/test separation process.
Outputs logs/core_rule_split_proof.json with a binary verdict + locked CV.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
TARGET = "is_fraud"
ID_COL = "id"
CHRONO = ("id", "day_of_week", "time_of_day_hour")


def feature_shift_table(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    for c in cols:
        if c not in train.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[c]):
            a, b = train[c].astype(float).dropna(), test[c].astype(float).dropna()
            ks = ks_2samp(a, b)
            rows.append(
                {
                    "feature": c,
                    "kind": "numeric",
                    "train_mean": float(a.mean()),
                    "test_mean": float(b.mean()),
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                }
            )
        else:
            tr = train[c].astype(str).value_counts(normalize=True)
            te = test[c].astype(str).value_counts(normalize=True)
            idx = sorted(set(tr.index) | set(te.index))
            l1 = float(sum(abs(tr.get(i, 0.0) - te.get(i, 0.0)) for i in idx))
            rows.append({"feature": c, "kind": "categorical", "dist_l1": l1})
    return rows


def adversarial_auc(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict:
    adv = pd.concat(
        [train[features].assign(is_test=0), test[features].assign(is_test=1)],
        ignore_index=True,
    )
    cat_cols = adv[features].select_dtypes(include=["object", "string", "category", "bool"]).columns
    for c in cat_cols:
        cats = sorted(adv[c].astype(str).unique())
        adv[c] = pd.Categorical(adv[c].astype(str), categories=cats)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(adv))
    fold_aucs = []
    for tr, va in skf.split(adv, adv["is_test"]):
        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1
        )
        m.fit(adv.iloc[tr][features], adv.iloc[tr]["is_test"])
        oof[va] = m.predict_proba(adv.iloc[va][features])[:, 1]
        fold_aucs.append(float(roc_auc_score(adv.iloc[va]["is_test"], oof[va])))
    return {
        "oof_auc": float(roc_auc_score(adv["is_test"], oof)),
        "fold_aucs": fold_aucs,
        "fold_mean": float(np.mean(fold_aucs)),
        "fold_std": float(np.std(fold_aucs)),
    }


def chronology(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    y = train[TARGET].astype(int).to_numpy()
    out: dict = {"time_order_suspected": False, "notes": [], "columns": {}}
    tr_ids, te_ids = train[ID_COL], test[ID_COL]
    contiguous = bool((tr_ids.max() < te_ids.min()) or (te_ids.max() < tr_ids.min()))
    q = pd.qcut(tr_ids, 5, labels=False, duplicates="drop")
    rates = pd.DataFrame({"q": q, "y": y}).groupby("q")["y"].mean()
    drift_pp = float((rates.max() - rates.min()) * 100)
    out["columns"]["id"] = {
        "train_range": [int(tr_ids.min()), int(tr_ids.max())],
        "test_range": [int(te_ids.min()), int(te_ids.max())],
        "overlap_n": int(len(set(tr_ids) & set(te_ids))),
        "contiguous_non_overlapping": contiguous,
        "fraud_auc": float(roc_auc_score(y, tr_ids)),
        "fraud_rate_quintile_range_pp": drift_pp,
        "interpretation": (
            "Contiguous ids are a Kaggle file-write artifact (train then test), "
            "not evidence of a temporal DGP split — fraud AUC≈0.50 and flat quintiles."
            if contiguous and drift_pp < 5
            else "Possible time structure — investigate further."
        ),
    }
    if contiguous and drift_pp < 5:
        out["notes"].append("id contiguous but not time-ordered DGP (artifact)")
    elif contiguous and drift_pp >= 5:
        out["time_order_suspected"] = True
        out["notes"].append(f"id contiguous AND fraud drift {drift_pp:.2f}pp")

    for col in ("day_of_week", "time_of_day_hour"):
        tr, te = train[col], test[col]
        tr_d = tr.value_counts(normalize=True).sort_index()
        te_d = te.value_counts(normalize=True).sort_index()
        idx = sorted(set(tr_d.index) | set(te_d.index))
        l1 = float(sum(abs(tr_d.get(i, 0.0) - te_d.get(i, 0.0)) for i in idx))
        out["columns"][col] = {
            "l1_train_test": l1,
            "fraud_auc": float(roc_auc_score(y, tr.to_numpy())),
        }
        if l1 > 0.15:
            out["time_order_suspected"] = True
            out["notes"].append(f"{col} L1={l1:.3f} (>0.15)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default=os.path.join(LOGS, "core_rule_split_proof.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    features = [c for c in test.columns if c != ID_COL]

    print("=== S6E11 core rule: prove the split ===")
    print(f"train={len(train):,} test={len(test):,}")

    chrono = chronology(train, test)
    print("\n[Chronology]")
    for k, v in chrono["columns"].items():
        print(f"  {k}: {v}")
    print(f"  time_order_suspected={chrono['time_order_suspected']}")

    # exclude id from adversarial (unique → trivial)
    adv_feats = [c for c in features if c != ID_COL]
    print("\n[Adversarial train↔test]")
    adv = adversarial_auc(train, test, adv_feats)
    print(f"  OOF Adv AUC={adv['oof_auc']:.5f} (folds {adv['fold_mean']:.5f}±{adv['fold_std']:.5f})")

    print("\n[Feature shift (KS / L1)]")
    shifts = feature_shift_table(train, test, features)
    # highlight largest KS
    num = sorted([r for r in shifts if r["kind"] == "numeric"], key=lambda r: -r["ks_stat"])
    for r in num[:8]:
        print(f"  {r['feature']}: KS={r['ks_stat']:.4f} p={r['ks_pvalue']:.3g}")

    iid_ok = adv["oof_auc"] < 0.52
    no_time = not chrono["time_order_suspected"]
    proven_iid = bool(iid_ok and no_time)
    if proven_iid:
        locked = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
        split_type = "i.i.d. / random-like synthetic split (not temporal)"
        rationale = (
            f"Adv AUC={adv['oof_auc']:.5f}≈0.50 and chronology shows no real time order "
            "(contiguous ids = write artifact). Mirror with shuffled stratified K-Fold."
        )
    else:
        locked = "prequential / time-block CV"
        split_type = "time-structured or shifted"
        rationale = "Elevated Adv AUC and/or time structure → do not use shuffled K-Fold."

    print("\n[Verdict]")
    print(f"  split_type={split_type}")
    print(f"  proven_iid={proven_iid}")
    print(f"  locked_cv={locked}")
    print(f"  rationale={rationale}")

    proof = {
        "title": "S6E11 core rule — prove the split",
        "handbook_ref": "Fraud Detection Handbook Ch 5.2",
        "core_rule": "Validation split must mirror how test was separated from train.",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "chronology": chrono,
        "adversarial": adv,
        "feature_shifts": shifts,
        "verdict": {
            "proven_iid": proven_iid,
            "split_type": split_type,
            "locked_cv": locked,
            "rationale": rationale,
            "adv_auc_threshold": 0.52,
        },
    }
    with open(args.out, "w") as f:
        json.dump(proof, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
