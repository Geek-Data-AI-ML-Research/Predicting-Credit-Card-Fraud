#!/usr/bin/env python3
"""Stage 8.2: CUDA residual XGB with base_margin = best ensemble logit.

NO kaggle submit. Gate vs campaign best / nested control.
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
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from te_smooth50_features import build_codes, build_fold_matrix  # noqa: E402
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED7 = os.path.join(ROOT, "preds", "stage7")
PRED8 = os.path.join(ROOT, "preds", "stage8")
SUBS = os.path.join(ROOT, "submissions")
SEED = 42
EPS = 1e-6


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def load_margins(y):
    """Prefer Platt-calibrated campaign_best logits from stage7_align_handoff.

    base_margin MUST be log-odds centered near logit(0.017)≈-4.05, NOT 0.0.
    """
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    # 1) Preferred: align_handoff already wrote calibrated logits
    op = os.path.join(PRED7, "campaign_best_oof_logit.npy")
    tp = os.path.join(PRED7, "campaign_best_test_logit.npy")
    meta_p = os.path.join(LOGS, "stage7_align_handoff.json")
    if os.path.exists(op) and os.path.exists(tp):
        Lo, Lt = np.load(op).astype(np.float64), np.load(tp).astype(np.float64)
        p = expit(Lo)
        mean_p, mean_l = float(p.mean()), float(Lo.mean())
        src = "campaign_best"
        if os.path.exists(meta_p):
            src = json.load(open(meta_p)).get("handoff", {}).get("source", src)
        print(
            f"  margin check: mean_p={mean_p:.5f} mean_logit={mean_l:.3f} "
            f"(want p~0.017 logit~-4.05)",
            flush=True,
        )
        if abs(mean_p - 0.5) < 0.05:
            raise RuntimeError(
                f"base_margin looks uncalibrated (mean_p={mean_p:.3f}≈0.5). "
                "Re-run stage7_align_handoff.py before Stage 8.2."
            )
        return Lo, Lt, float(roc_auc_score(y, p)), f"campaign_best:{src}"

    # 2) Fallback: nested Platt on best available score vector
    for base in ("campaign_best", "best_7_2", "best_7_1", "nested_probit_hillclimb"):
        op = os.path.join(PRED7, f"{base}_oof.npy")
        tp = os.path.join(PRED7, f"{base}_test.npy")
        if not (os.path.exists(op) and os.path.exists(tp)):
            continue
        o, t = np.load(op).astype(np.float64), np.load(tp).astype(np.float64)
        p_oof = np.zeros(len(y))
        p_te = np.zeros(len(t))
        for tr, va in folds:
            lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
            lr.fit(o[tr].reshape(-1, 1), y[tr])
            p_oof[va] = lr.predict_proba(o[va].reshape(-1, 1))[:, 1]
            p_te += lr.predict_proba(t.reshape(-1, 1))[:, 1]
        p_te /= 5.0
        Lo, Lt = to_logit(p_oof), to_logit(p_te)
        print(
            f"  fallback Platt on {base}: mean_p={p_oof.mean():.5f} mean_logit={Lo.mean():.3f}",
            flush=True,
        )
        return Lo, Lt, float(roc_auc_score(y, p_oof)), f"platt:{base}"
    raise FileNotFoundError("No Stage 7 best logits found — run align_handoff first")


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_vel"] = out["merchant_risk_score"] * out["velocity_score"]
    out["ip_dist"] = out["ip_country_mismatch"] * out["distance_from_home_km"]
    out["cvv_risk"] = out["cvv_retry_count"] * out["merchant_risk_score"]
    out["hrs_vel"] = out["hours_since_last_txn"] / (out["velocity_score"] + 1e-3)
    out["amt_bal"] = out["amount_usd"] / (out["account_balance_usd"].abs() + 1.0)
    return out


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    L_oof, L_test, base_auc, base_name = load_margins(y)
    print(f"base_margin from {base_name} auc={base_auc:.6f}", flush=True)

    train_i = add_interactions(train)
    test_i = add_interactions(test)
    dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = build_codes(train, test)

    oof = np.zeros(len(y), np.float64)
    test_sum = np.zeros(len(test), np.float64)
    fold_aucs = []
    rng = np.random.default_rng(SEED)

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        X_tr, X_va, X_te = build_fold_matrix(
            train_i,
            test_i,
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
        cols = X_tr.columns
        X_va = X_va.reindex(columns=cols, fill_value=0)
        X_te = X_te.reindex(columns=cols, fill_value=0)

        # base_margin regularization on train
        bm_tr = 0.95 * L_oof[tr] + rng.normal(0, 0.10, size=len(tr))
        bm_va = L_oof[va]
        bm_te = L_test

        dtr = xgb.DMatrix(X_tr, label=y[tr], base_margin=bm_tr)
        dva = xgb.DMatrix(X_va, label=y[va], base_margin=bm_va)
        dte = xgb.DMatrix(X_te, base_margin=bm_te)
        params = dict(
            max_depth=2,
            eta=0.005,
            subsample=0.7,
            colsample_bytree=0.3,
            reg_lambda=100.0,
            reg_alpha=20.0,
            tree_method="hist",
            device="cuda",
            max_bin=1024,
            eval_metric="auc",
            objective="binary:logistic",
        )
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=3000,
            evals=[(dva, "va")],
            early_stopping_rounds=150,
            verbose_eval=False,
        )
        pva = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        pte = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
        oof[va] = pva
        test_sum += pte
        fa = float(roc_auc_score(y[va], pva))
        base_fa = float(roc_auc_score(y[va], 1 / (1 + np.exp(-L_oof[va]))))
        fold_aucs.append(fa)
        print(
            f"  residual va={fa:.6f} base={base_fa:.6f} Δ={fa-base_fa:+.6f} it={booster.best_iteration+1}",
            flush=True,
        )

    te = test_sum / 5.0
    oof_auc = float(roc_auc_score(y, oof))
    base_folds = [float(roc_auc_score(y[va], 1 / (1 + np.exp(-L_oof[va])))) for _, va in folds]
    gate = promotion_gate(base_folds, fold_aucs, base_auc, oof_auc)
    print(f"OOF={oof_auc:.6f} base={base_auc:.6f} {gate['reason']}", flush=True)

    os.makedirs(PRED8, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    np.save(os.path.join(PRED8, "residual_xgb_oof.npy"), oof.astype(np.float32))
    np.save(os.path.join(PRED8, "residual_xgb_full.npy"), te.astype(np.float32))

    doc = {
        "stage": "8.2_residual_xgb",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "base": base_name,
        "base_auc": base_auc,
        "oof_auc": oof_auc,
        "fold_aucs": fold_aucs,
        "gate_vs_base": gate,
        "promote": gate["promote"],
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    if gate["promote"]:
        sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
        sub["is_fraud"] = rankdata(te) / len(te)
        path = os.path.join(SUBS, f"stage8_2_residual_xgb_{oof_auc:.5f}.csv")
        sub.to_csv(path, index=False)
        doc["submission"] = path
        print(f"WROTE {path}", flush=True)
    path = os.path.join(LOGS, "stage8_2_residual_xgb_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
