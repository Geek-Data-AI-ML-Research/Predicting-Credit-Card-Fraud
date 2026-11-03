#!/usr/bin/env python3
"""Stage 8.3: Leak-free soft pseudo-label / distillation on 300k test (CUDA).

NO kaggle submit. Nested: PL only appended to tr_idx, never va_idx.
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
from scipy.special import expit, logit
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
CTRL = 0.949447  # vs te_xgb_smooth50 for base arms; also report vs campaign
CANARY_ABORT = 0.9500  # Fold-0 gate: te_smooth50 saturated → abort to Stage 8.1 / 9


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def load_fold_test_preds(ntr, nte, y, folds):
    """Prefer align_handoff campaign_best (Platt probs) — never a losing raw arm."""
    # calibrated probabilities from handoff
    op = os.path.join(PRED7, "campaign_best_oof_p.npy")
    tp = os.path.join(PRED7, "campaign_best_test_p.npy")
    if os.path.exists(op) and os.path.exists(tp):
        return np.load(op).astype(np.float64), np.load(tp).astype(np.float64), "campaign_best_p"
    for base in ("campaign_best", "best_7_2", "best_7_1", "nested_probit_hillclimb"):
        tp = os.path.join(PRED7, f"{base}_test.npy")
        opath = os.path.join(PRED7, f"{base}_oof.npy")
        if os.path.exists(tp) and os.path.exists(opath):
            return np.load(opath).astype(np.float64), np.load(tp).astype(np.float64), base
    z = np.load(os.path.join(PRED7, "meta_bank.npz"), allow_pickle=True)
    O, T = z["O"].astype(np.float64), z["T"].astype(np.float64)
    return O.mean(1), T.mean(1), "bank_mean"


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    p_oof, p_test, src = load_fold_test_preds(len(y), len(test), y, folds)
    print(f"pseudo source={src} oof_auc={roc_auc_score(y, p_oof):.6f}", flush=True)

    dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = build_codes(train, test)

    arms = {
        "oof_xgb_pseudo": {"oof": np.zeros(len(y)), "test": np.zeros(len(test)), "folds": []},
        "oof_xgb_distill": {"oof": np.zeros(len(y)), "test": np.zeros(len(test)), "folds": []},
    }

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        X_tr, X_va, X_te = build_fold_matrix(
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
        cols = X_tr.columns
        X_va = X_va.reindex(columns=cols, fill_value=0)
        X_te = X_te.reindex(columns=cols, fill_value=0)

        # fold-specific test preds: use global p_test (ensemble is already OOF-safe for train;
        # test is transductive — same across folds is OK for distillation target)
        p_te = p_test

        # ----- Arm A: high-confidence hard PL -----
        hi = np.where(p_te > 0.75)[0]
        lo = np.where(p_te < 0.0015)[0]
        pl_idx = np.concatenate([hi, lo]) if len(hi) + len(lo) else np.array([], dtype=int)
        print(f"  PL hard n={len(pl_idx)} hi={len(hi)} lo={len(lo)}", flush=True)

        X_tr_a = X_tr
        y_tr_a = y[tr]
        w_tr_a = np.ones(len(tr), np.float32)
        if len(pl_idx):
            X_pl = X_te.iloc[pl_idx]
            y_pl = np.concatenate([np.ones(len(hi)), np.zeros(len(lo))]).astype(int)
            w_pl = np.full(len(pl_idx), 0.5, np.float32)
            X_tr_a = pd.concat([X_tr, X_pl], axis=0)
            y_tr_a = np.concatenate([y_tr_a, y_pl])
            w_tr_a = np.concatenate([w_tr_a, w_pl])

        m_a = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=4000,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=30.0,
            reg_alpha=5.0,
            tree_method="hist",
            device="cuda",
            max_bin=256,
            eval_metric="auc",
            early_stopping_rounds=150,
            random_state=SEED + f,
        )
        m_a.fit(
            X_tr_a,
            y_tr_a,
            sample_weight=w_tr_a,
            eval_set=[(X_va, y[va])],
            verbose=False,
        )
        pva = m_a.predict_proba(X_va)[:, 1]
        pte = m_a.predict_proba(X_te)[:, 1]
        arms["oof_xgb_pseudo"]["oof"][va] = pva
        arms["oof_xgb_pseudo"]["test"] += pte
        fa_a = float(roc_auc_score(y[va], pva))
        arms["oof_xgb_pseudo"]["folds"].append(fa_a)
        print(f"  ArmA pseudo va={fa_a:.6f}", flush=True)

        # ----- Arm B: soft distillation -----
        soft_tr = 0.7 * y[tr].astype(np.float64) + 0.3 * p_oof[tr]
        X_tr_b = pd.concat([X_tr, X_te], axis=0)
        y_tr_b = np.concatenate([soft_tr, p_te])
        w_tr_b = np.concatenate([np.ones(len(tr)), np.full(len(test), 0.5)])
        m_b = xgb.XGBRegressor(
            objective="reg:logistic",
            max_depth=3,
            learning_rate=0.02,
            n_estimators=3000,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=30.0,
            tree_method="hist",
            device="cuda",
            max_bin=256,
            early_stopping_rounds=150,
            random_state=SEED + f,
        )
        m_b.fit(
            X_tr_b,
            y_tr_b,
            sample_weight=w_tr_b,
            eval_set=[(X_va, y[va].astype(np.float64))],
            verbose=False,
        )
        pva_b = np.clip(m_b.predict(X_va), 0, 1)
        pte_b = np.clip(m_b.predict(X_te), 0, 1)
        arms["oof_xgb_distill"]["oof"][va] = pva_b
        arms["oof_xgb_distill"]["test"] += pte_b
        fa_b = float(roc_auc_score(y[va], pva_b))
        arms["oof_xgb_distill"]["folds"].append(fa_b)
        print(f"  ArmB distill va={fa_b:.6f}", flush=True)

        # Fold-0 canary: te_smooth50 saturated → abort (do not burn 2×5 folds)
        if f == 0 and max(fa_a, fa_b) < CANARY_ABORT:
            status = f"EARLY_ABORT_FOLD0_max_{max(fa_a, fa_b):.4f}"
            print(
                f"CANARY FAIL {status} (<{CANARY_ABORT}) — te_smooth50 saturated; "
                "advancing to Stage 8.1 / Stage 9",
                flush=True,
            )
            os.makedirs(LOGS, exist_ok=True)
            doc = {
                "stage": "8.3_pseudolabel",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": round(time.time() - t0, 1),
                "status": status,
                "source": src,
                "fold0": {"ArmA": fa_a, "ArmB": fa_b},
                "promote": False,
                "reason": "Fold0 max AUC below 0.9500; te_smooth50 residual capacity exhausted",
                "policy": "NO_KAGGLE_SUBMIT",
                "keep_pod_warm": True,
                "next": ["stage8_1_raw_dual_realmlp", "stage9_1_full73_power_hillclimb", "stage9_2_orig20k_orthogonal_residual"],
            }
            path = os.path.join(LOGS, "stage8_3_pseudolabel_result.json")
            json.dump(doc, open(path, "w"), indent=2)
            print(f"wrote {path}", flush=True)
            return

    # control folds from te_xgb if present
    ctrl_oof, ctrl_folds = CTRL, None
    p_ctrl = os.path.join(ROOT, "preds", "stage4", "te_xgb_smooth50_oof.npy")
    if os.path.exists(p_ctrl):
        co = np.load(p_ctrl)
        ctrl_oof = float(roc_auc_score(y, co))
        ctrl_folds = [float(roc_auc_score(y[va], co[va])) for _, va in folds]
    else:
        ctrl_folds = [CTRL] * 5

    os.makedirs(PRED8, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    results = {}
    for name, s in arms.items():
        s["test"] /= 5.0
        # only full OOF if all folds filled
        if len(s["folds"]) < 5:
            continue
        s["oof_auc"] = float(roc_auc_score(y, s["oof"]))
        gate = promotion_gate(ctrl_folds, s["folds"], ctrl_oof, s["oof_auc"])
        results[name] = {
            "oof_auc": s["oof_auc"],
            "fold_aucs": s["folds"],
            "gate_vs_te_xgb": gate,
            "promote": gate["promote"],
        }
        np.save(os.path.join(PRED8, f"{name}.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(PRED8, f"{name}_full.npy"), s["test"].astype(np.float32))
        os.makedirs(os.path.join(ROOT, "preds", "stage6"), exist_ok=True)
        np.save(os.path.join(ROOT, "preds", "stage6", f"{name}_oof.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(ROOT, "preds", "stage6", f"{name}_full.npy"), s["test"].astype(np.float32))
        print(f"{name}: OOF={s['oof_auc']:.6f} {gate['reason']}", flush=True)
        if gate["promote"]:
            sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
            sub["is_fraud"] = rankdata(s["test"]) / len(test)
            path = os.path.join(SUBS, f"stage8_3_{name}_{s['oof_auc']:.5f}.csv")
            sub.to_csv(path, index=False)
            results[name]["submission"] = path

    doc = {
        "stage": "8.3_pseudolabel",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "status": "COMPLETE",
        "source": src,
        "arms": results,
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    path = os.path.join(LOGS, "stage8_3_pseudolabel_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
