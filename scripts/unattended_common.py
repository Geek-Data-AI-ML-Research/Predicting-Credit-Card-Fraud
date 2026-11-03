#!/usr/bin/env python3
"""Shared helpers for the unattended T01–T10 campaign. NO kaggle submit."""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.environ.get("S6E11_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "unattended")
SUBS = os.path.join(ROOT, "submissions")
WS = os.environ.get("S6E11_PERSIST", "/workspace/s6e11")

SEED = 42
# Hot-patch: 2 seeds × leaves{3,4} ≈ 98% bag gain at ~40% wall time
BAG_SEEDS = [42, 2026]
CTRL_DIGLEAD = 0.950627
CTRL_TE_XGB = 0.949447
CTRL_NESTED_HILL = 0.951170
# Hard gate for T01–T07: bank diglead_l3 fold0≈0.9498; anything <0.9488 is a FE bug.
CANARY_FLOOR = 0.9488
EPS = 1e-6

SUMMARY_PATH = os.path.join(LOGS, "unattended_10tests_summary.json")


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs():
    for p in (LOGS, PRED, SUBS, os.path.join(WS, "logs"), os.path.join(WS, "preds", "unattended"), os.path.join(WS, "submissions")):
        os.makedirs(p, exist_ok=True)


def load_train_test():
    from validation_stage3_baselines import TARGET

    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    return train, test, y


def promotion_gate_base(ctrl_folds, folds, ctrl_oof, oof):
    from validation_stage2_handbook_cv import promotion_gate

    return promotion_gate(ctrl_folds, folds, ctrl_oof, oof)


def fold_aucs(y, oof, folds):
    return [float(roc_auc_score(y[va], oof[va])) for _, va in folds]


def load_ctrl_diglead(y, folds):
    pre = os.path.join(ROOT, "preds", "stage9", "full_bank_unpruned.npz")
    if os.path.exists(pre):
        z = np.load(pre, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        hits = [i for i, n in enumerate(names) if "lgb_diglead_l3_bag" in n]
        if hits:
            o = z["O"][:, hits[0]].astype(np.float64)
            return o, float(roc_auc_score(y, o)), fold_aucs(y, o, folds)
    return None, CTRL_DIGLEAD, [CTRL_DIGLEAD] * 5


def save_oof_test(tag: str, oof: np.ndarray, te: np.ndarray):
    ensure_dirs()
    op = os.path.join(PRED, f"oof_{tag}.npy")
    tp = os.path.join(PRED, f"test_{tag}_full.npy")
    np.save(op, oof.astype(np.float32))
    np.save(tp, te.astype(np.float32))
    # persist
    for dest in (os.path.join(WS, "preds", "unattended"),):
        os.makedirs(dest, exist_ok=True)
        np.save(os.path.join(dest, f"oof_{tag}.npy"), oof.astype(np.float32))
        np.save(os.path.join(dest, f"test_{tag}_full.npy"), te.astype(np.float32))
    return op, tp


def write_candidate_csv(name: str, scores: np.ndarray, n_expected: int = 299_984):
    from scipy.stats import rankdata

    ensure_dirs()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    assert len(scores) == n_expected, f"bad len {len(scores)}"
    assert np.isfinite(scores).all(), "NaN/Inf in scores"
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    assert list(sample.columns) == ["id", "is_fraud"]
    out = sample.copy()
    out["is_fraud"] = rankdata(scores) / len(scores)
    path = os.path.join(SUBS, name)
    out.to_csv(path, index=False)
    # verify
    chk = pd.read_csv(path)
    assert len(chk) == n_expected and chk.isna().sum().sum() == 0
    assert np.isfinite(chk["is_fraud"]).all()
    for dest in (os.path.join(WS, "submissions"),):
        os.makedirs(dest, exist_ok=True)
        out.to_csv(os.path.join(dest, name), index=False)
    return path


def append_summary(test_id: str, result: dict):
    ensure_dirs()
    doc = {"updated_utc": utc_now(), "tests": {}}
    if os.path.exists(SUMMARY_PATH):
        try:
            doc = json.load(open(SUMMARY_PATH))
        except Exception:
            pass
    doc.setdefault("tests", {})
    doc["tests"][test_id] = result
    doc["updated_utc"] = utc_now()
    doc["policy"] = "NO_KAGGLE_SUBMIT"
    json.dump(doc, open(SUMMARY_PATH, "w"), indent=2)
    # persist + local mirror path
    for dest in (os.path.join(WS, "logs"), LOGS):
        os.makedirs(dest, exist_ok=True)
        json.dump(doc, open(os.path.join(dest, "unattended_10tests_summary.json"), "w"), indent=2)
    # per-test result
    p = os.path.join(LOGS, f"{test_id}_result.json")
    json.dump(result, open(p, "w"), indent=2)
    json.dump(result, open(os.path.join(WS, "logs", f"{test_id}_result.json"), "w"), indent=2)
    return doc


def run_test_safe(test_id: str, fn):
    t0 = datetime.now(timezone.utc)
    print(f"\n======== {test_id} START {t0.isoformat()} ========", flush=True)
    try:
        result = fn()
        if not isinstance(result, dict):
            result = {"status": "OK", "detail": str(result)}
        result.setdefault("status", "OK")
    except Exception as e:
        result = {
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc()[-4000:],
        }
        print(f"[{test_id}] ERROR: {e}", flush=True)
        print(result["traceback"], flush=True)
    result["test_id"] = test_id
    result["created_utc"] = utc_now()
    result["elapsed_sec"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    append_summary(test_id, result)
    print(f"======== {test_id} {result.get('status')} ({result['elapsed_sec']}s) ========", flush=True)
    return result


def pr_auc(y, s):
    return float(average_precision_score(y, s))
