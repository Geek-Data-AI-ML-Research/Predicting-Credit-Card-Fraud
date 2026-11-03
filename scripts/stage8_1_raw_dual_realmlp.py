#!/usr/bin/env python3
"""Stage 8.1: Raw + dual-categorical RealMLP (NO precomputed float TE).

Fold-0 canary: >0.9500 continue; <0.9495 EARLY_ABORT.
NO kaggle submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED8 = os.path.join(ROOT, "preds", "stage8")
SUBS = os.path.join(ROOT, "submissions")
SEED = 42
CTRL = 0.949447
CANARY_GO = 0.9500
CANARY_ABORT = 0.9495

RAW_NUM = [
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
RAW_CAT = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]
RAW_BIN = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
# numeric cols to dual-encode as category strings
DUAL_ROUND = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
    "time_of_day_hour",
    "txn_count_last_24h",
]


def build_frame(train: pd.DataFrame, test: pd.DataFrame, with_orig: bool):
    frames = []
    for df, is_test in ((train, False), (test, True)):
        out = pd.DataFrame(index=df.index)
        for c in RAW_NUM + RAW_BIN:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        for c in RAW_CAT:
            out[c] = df[c].astype(str).astype("category")
        for c in DUAL_ROUND:
            out[f"{c}_cat"] = df[c].round(2).astype(str).astype("category")
            # frequency on combined later
            out[f"{c}_freq"] = 0.0
        out["is_orig"] = 0.0
        out["is_fraud"] = df[TARGET].values if not is_test else -1
        out["_split"] = "test" if is_test else "train"
        frames.append(out)
    full = pd.concat(frames, ignore_index=True)
    # freqs on train+test combined (transductive OK for counts)
    for c in DUAL_ROUND:
        vc = full[f"{c}_cat"].value_counts()
        full[f"{c}_freq"] = full[f"{c}_cat"].map(vc).astype(np.float64) / len(full)

    if with_orig:
        op = os.path.join(DATA, "original.csv")
        if os.path.exists(op):
            orig = pd.read_csv(op)
            # align columns if present
            o = pd.DataFrame(index=orig.index)
            ok = True
            for c in RAW_NUM + RAW_BIN:
                if c not in orig.columns:
                    ok = False
                    break
                o[c] = pd.to_numeric(orig[c], errors="coerce").astype(np.float64)
            if ok and TARGET in orig.columns:
                for c in RAW_CAT:
                    o[c] = orig[c].astype(str).astype("category") if c in orig.columns else "NA"
                for c in DUAL_ROUND:
                    o[f"{c}_cat"] = orig[c].round(2).astype(str).astype("category")
                    o[f"{c}_freq"] = 0.0
                o["is_orig"] = 1.0
                o["is_fraud"] = orig[TARGET].astype(int).values
                o["_split"] = "orig"
                full = pd.concat([full, o], ignore_index=True)
                for c in DUAL_ROUND:
                    vc = full[f"{c}_cat"].value_counts()
                    full[f"{c}_freq"] = full[f"{c}_cat"].map(vc).astype(np.float64) / len(full)
                print(f"appended original n={len(o)}", flush=True)
            else:
                print("original.csv missing aligned cols — skip", flush=True)
        else:
            print("no original.csv — skip", flush=True)

    # re-category after concat
    for c in list(full.columns):
        if full[c].dtype.name == "category" or c.endswith("_cat") or c in RAW_CAT:
            full[c] = full[c].astype(str).astype("category")
    return full


def fit_predict_fold(X_tr, y_tr, X_va, X_te, seed):
    from pytabkit import RealMLP_TD_Classifier

    model = RealMLP_TD_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=0)
    model.fit(X_tr, y_tr)
    # predict_proba
    pva = model.predict_proba(X_va)[:, 1]
    pte = model.predict_proba(X_te)[:, 1]
    return pva, pte


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    ntr = len(train)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))

    full = build_frame(train, test, with_orig=True)
    feat_cols = [c for c in full.columns if c not in ("is_fraud", "_split")]
    train_full = full[full["_split"] == "train"].reset_index(drop=True)
    test_full = full[full["_split"] == "test"].reset_index(drop=True)
    orig_full = full[full["_split"] == "orig"].reset_index(drop=True)
    assert len(train_full) == ntr

    oof = np.zeros(ntr, np.float64)
    test_sum = np.zeros(len(test), np.float64)
    fold_aucs = []
    status = "RUNNING"

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        X_tr = train_full.iloc[tr][feat_cols].copy()
        y_tr = y[tr]
        if len(orig_full):
            X_tr = pd.concat([X_tr, orig_full[feat_cols]], axis=0)
            y_tr = np.concatenate([y_tr, orig_full["is_fraud"].astype(int).values])
        X_va = train_full.iloc[va][feat_cols].copy()
        X_te = test_full[feat_cols].copy()
        # align categories
        for c in feat_cols:
            if str(X_tr[c].dtype) == "category" or c.endswith("_cat") or c in RAW_CAT:
                cats = pd.Categorical(pd.concat([X_tr[c].astype(str), X_va[c].astype(str), X_te[c].astype(str)])).categories
                X_tr[c] = pd.Categorical(X_tr[c].astype(str), categories=cats)
                X_va[c] = pd.Categorical(X_va[c].astype(str), categories=cats)
                X_te[c] = pd.Categorical(X_te[c].astype(str), categories=cats)

        t1 = time.time()
        pva, pte = fit_predict_fold(X_tr, y_tr, X_va, X_te, SEED + f)
        fa = float(roc_auc_score(y[va], pva))
        oof[va] = pva
        test_sum += pte
        fold_aucs.append(fa)
        print(f"  fold{f}: auc={fa:.6f} ({time.time()-t1:.0f}s)", flush=True)

        if f == 0:
            if fa < CANARY_ABORT:
                status = f"EARLY_ABORT_FOLD0_{fa:.4f}"
                print(f"CANARY FAIL {status}", flush=True)
                break
            if fa < CANARY_GO:
                print(f"CANARY soft: fold0={fa:.6f} < {CANARY_GO} — continuing cautiously", flush=True)

    os.makedirs(PRED8, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)

    doc = {
        "stage": "8.1_raw_dual_realmlp",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "status": status,
        "fold_aucs": fold_aucs,
        "control": CTRL,
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    if status.startswith("EARLY_ABORT"):
        doc["oof_auc"] = None
        doc["promote"] = False
    elif len(fold_aucs) == 5:
        te = test_sum / 5.0
        oof_auc = float(roc_auc_score(y, oof))
        # ctrl folds
        p_ctrl = os.path.join(ROOT, "preds", "stage4", "te_xgb_smooth50_oof.npy")
        if os.path.exists(p_ctrl):
            co = np.load(p_ctrl)
            ctrl_oof = float(roc_auc_score(y, co))
            ctrl_folds = [float(roc_auc_score(y[va], co[va])) for _, va in folds]
        else:
            ctrl_oof, ctrl_folds = CTRL, [CTRL] * 5
        gate = promotion_gate(ctrl_folds, fold_aucs, ctrl_oof, oof_auc)
        doc.update({"oof_auc": oof_auc, "gate": gate, "promote": gate["promote"]})
        np.save(os.path.join(PRED8, "raw_dual_realmlp_oof.npy"), oof.astype(np.float32))
        np.save(os.path.join(PRED8, "raw_dual_realmlp_full.npy"), te.astype(np.float32))
        print(f"OOF={oof_auc:.6f} {gate['reason']}", flush=True)
        if gate["promote"]:
            sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
            from scipy.stats import rankdata as rd

            sub["is_fraud"] = rd(te) / len(te)
            path = os.path.join(SUBS, f"stage8_1_raw_dual_realmlp_{oof_auc:.5f}.csv")
            sub.to_csv(path, index=False)
            doc["submission"] = path
    else:
        doc["oof_auc"] = None
        doc["promote"] = False

    path = os.path.join(LOGS, "stage8_1_raw_dual_realmlp_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(f"wrote {path} status={status} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
