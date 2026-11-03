#!/usr/bin/env python3
"""Pre-build 5-fold Part-5 diglead matrices to DIGLEAD_CACHE (float32 npy).

Requires CACHE_SCHEMA match. Wipe stale caches first. NO kaggle submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.environ.get("S6E11_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CACHE = os.environ.get("DIGLEAD_CACHE", "/workspace/cache_diglead_folds")
if not os.path.isdir("/workspace"):
    CACHE = os.path.join(ROOT, "preds", "cache_diglead_folds")


def main():
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    from diglead_features import CACHE_SCHEMA
    from diglead_matrix_cache import DEFAULT_SMOOTH, WITH_ORIG, DigleadView
    import pandas as pd
    from validation_stage3_baselines import TARGET

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = train[TARGET].astype(int).values
    print(f"[prebuild] start cache→{CACHE} schema={CACHE_SCHEMA}", flush=True)
    # Force rebuild: remove old npy so fold_matrix won't hit stale files mid-write
    for f in range(5):
        for kind in ("X_tr", "X_va", "X_te", "tr_idx", "va_idx"):
            p = os.path.join(CACHE, f"fold{f}_{kind}.npy")
            if os.path.exists(p):
                os.remove(p)
    view = DigleadView(train, test, y)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": CACHE_SCHEMA,
        "smooth": DEFAULT_SMOOTH,
        "with_orig": WITH_ORIG,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "folds": {},
    }
    for f in range(5):
        print(f"[prebuild] fold{f}...", flush=True)
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f, use_cache=False)
        np.save(os.path.join(CACHE, f"fold{f}_tr_idx.npy"), tr.astype(np.int32))
        np.save(os.path.join(CACHE, f"fold{f}_va_idx.npy"), va.astype(np.int32))
        np.save(os.path.join(CACHE, f"fold{f}_X_tr.npy"), Xtr.astype(np.float32))
        np.save(os.path.join(CACHE, f"fold{f}_X_va.npy"), Xva.astype(np.float32))
        np.save(os.path.join(CACHE, f"fold{f}_X_te.npy"), Xte.astype(np.float32))
        meta["folds"][str(f)] = {
            "X_tr": list(Xtr.shape),
            "X_va": list(Xva.shape),
            "X_te": list(Xte.shape),
        }
        print(f"  saved fold{f} X_tr={Xtr.shape} X_va={Xva.shape} X_te={Xte.shape}", flush=True)
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(CACHE, "meta.json"), "w"), indent=2)
    alt = os.path.join(ROOT, "preds", "cache_diglead_folds")
    if os.path.abspath(alt) != os.path.abspath(CACHE):
        os.makedirs(alt, exist_ok=True)
        os.system(f"cp -a {CACHE}/* {alt}/")
    print(f"[prebuild] DONE in {meta['elapsed_sec']}s → {CACHE}", flush=True)


if __name__ == "__main__":
    main()
