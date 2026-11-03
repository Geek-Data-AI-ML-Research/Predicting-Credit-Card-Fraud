#!/usr/bin/env python3
"""90s-class Fold-0 gate for Part-5 diglead_l3 (must be >= 0.9494 before T01–T10).

Reproduces bank lgb_diglead_l3 fold0 (~0.9498) with:
  - Part-5 LEAD_TE + drop_keys
  - smooth=20, num_leaves=3, reg_lambda=1.0, single seed=42
  - NO orig20k append

NO kaggle submit.
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("S6E11_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diglead_features import build_base_frames, build_fold_diglead_matrix  # noqa: E402
from unattended_common import locked_folds  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

GATE = 0.9494
CTRL_FOLD0 = 0.9498  # bank diglead_l3 / diglead_l3_bag fold0


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = train[TARGET].astype(int).values
    folds = locked_folds(y)
    tr, va = folds[0]
    print("building Part-5 diglead frames...", flush=True)
    XLtr, XLte, lead_te = build_base_frames(train, test)
    print(f"  XLtr={XLtr.shape} lead_te={len(lead_te)}", flush=True)
    Xa, Xb, Xt = build_fold_diglead_matrix(
        XLtr,
        XLte,
        lead_te,
        y,
        tr,
        va,
        smooth=20.0,
        teseed=0,
        return_frame=True,
    )
    print(f"  after drop_keys: X_tr={Xa.shape} X_va={Xb.shape} (no raw key_*)", flush=True)
    assert not any(str(c).startswith("key_") for c in Xa.columns), "raw key_* leaked into fit matrix"
    m = lgb.LGBMClassifier(
        n_estimators=20000,
        learning_rate=0.03,
        num_leaves=3,
        max_depth=2,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        max_bin=255,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )
    m.fit(
        Xa,
        y[tr],
        eval_set=[(Xb, y[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(300, verbose=False)],
    )
    p = m.predict_proba(Xb)[:, 1]
    auc = float(roc_auc_score(y[va], p))
    elapsed = time.time() - t0
    print(
        f"VERIFY fold0 auc={auc:.6f}  trees={m.best_iteration_}  "
        f"Δ_vs_bank_fold0={auc - CTRL_FOLD0:+.6f}  ({elapsed:.0f}s)",
        flush=True,
    )
    if auc < GATE:
        print(f"FAIL: fold0 {auc:.6f} < gate {GATE} — do NOT rebuild cache / restart T01–T10", flush=True)
        sys.exit(2)
    print(f"PASS: fold0 {auc:.6f} >= {GATE} — safe to rebuild cache + run suite", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
