"""Part 11 shared gates / IO. Policy: NO_SUBMIT."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

PRED11 = ROOT / "preds" / "part11"
OUT = ROOT / "logs" / "overnight"
REPORT = OUT / "part11_paradigm_shift.json"
PRED11.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "is_fraud"
BASELINE_STACK = 0.9521425
BASELINE_GLM = 0.950474
BASELINE_GB_GPT2_BAG5 = 0.951677
POLICY = "NO_SUBMIT"
T0 = time.time()

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
FLOATS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]
FEATS = NUMS + FLAGS + CATS


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "part11_paradigm_shift.log", "a") as fh:
        fh.write(msg + "\n")


def rk(v):
    v = np.asarray(v, float).ravel()
    return rankdata(v) / len(v)


def rlogit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def folds_y(y, seed=42):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def load_y_ids():
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    te_ids = pd.read_csv(ROOT / "data" / "sample_submission.csv")["id"].to_numpy()
    return y, te_ids


def nested_cv(P, y, folds, C=0.01, tol=1e-8, max_iter=200):
    from part7_clean_superstack import nested_cv as _n

    return _n(P, y, folds, C=C, tol=tol, max_iter=max_iter)


def best_nested(O, y, folds, Cs=(0.005, 0.006, 0.008, 0.01, 0.012, 0.015)):
    best = (-1.0, None, None, None)
    for C in Cs:
        nest, a, fa = nested_cv(O, y, folds, C=C)
        if a > best[0]:
            best = (a, C, fa, nest)
    return best


def stack_gate(O_base, new_cols_o, y, folds, label="new"):
    """Net nested CV lift when appending new rank/rlogit-ready columns to base stack."""
    base_a, base_C, base_fa, _ = best_nested(O_base, y, folds)
    if new_cols_o.ndim == 1:
        new_cols_o = new_cols_o.reshape(-1, 1)
    # stacker expects [0,1]-ish ranks; convert continuous scores via rk
    cols = []
    for j in range(new_cols_o.shape[1]):
        cols.append(rk(new_cols_o[:, j]))
    O2 = np.column_stack([O_base] + cols)
    a2, C2, fa2, nest2 = best_nested(O2, y, folds)
    delta = a2 - base_a
    log(
        f"  GATE {label}: base={base_a:.6f} C={base_C} -> +{new_cols_o.shape[1]}cols "
        f"OOF={a2:.6f} Δ={delta:+.6f} C={C2}  keep={delta > 1e-9}"
    )
    return {
        "label": label,
        "base_oof": base_a,
        "base_C": base_C,
        "with_oof": a2,
        "with_C": C2,
        "delta": delta,
        "keep": bool(delta > 1e-9),
        "base_folds": list(map(float, base_fa)),
        "with_folds": list(map(float, fa2)),
        "beats_9521425": bool(a2 > BASELINE_STACK + 1e-9),
    }


def load_champ_stack(y, folds):
    try:
        from part9_joint_boost_and_basis import build_8b_champ

        names, O, T = build_8b_champ(y, folds)
        a, C, fa, nest = best_nested(O, y, folds)
        log(f"  8B champ nested OOF={a:.6f} C={C} (ref P_super={BASELINE_STACK})")
        return names, O, T, {"oof": a, "C": C, "folds": list(map(float, fa)), "source": "8b"}
    except Exception as e:
        log(f"  8B champ unavailable ({e}); falling back to P_super_champ ranks")
        po = np.load(ROOT / "preds" / "P_super_champ_oof.npy").ravel()
        pt = np.load(ROOT / "preds" / "P_super_champ_test.npy").ravel()
        O = rk(po).reshape(-1, 1)
        T = rk(pt).reshape(-1, 1)
        a, C, fa, nest = best_nested(O, y, folds)
        log(f"  P_super nested OOF={a:.6f} C={C}")
        return ["P_super_champ"], O, T, {"oof": a, "C": C, "folds": list(map(float, fa)), "source": "p_super_fallback"}


def save_report(rep):
    rep["policy"] = POLICY
    rep["runtime_s"] = time.time() - T0
    tmp = REPORT.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rep, indent=2, default=float))
    tmp.replace(REPORT)


def load_report():
    if REPORT.exists():
        try:
            return json.loads(REPORT.read_text())
        except Exception:
            pass
    return {"policy": POLICY, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
