"""Collect public + self OOF/test prediction banks for S6E11 stacking."""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

GENERIC = {
    "oof",
    "preds",
    "artifacts",
    "output",
    "outputs",
    "working",
    "artifacts_extra",
    "fold_predictions",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
}


def owner_of(dn: str) -> str:
    b = os.path.basename(dn)
    return os.path.basename(os.path.dirname(dn)) if b in GENERIC else b


def locked_folds(y: np.ndarray):
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


def collect_oof_bank(
    root: str,
    n_train: int,
    n_test: int,
    *,
    require_test: bool = False,
    min_auc: Optional[float] = None,
    y: Optional[np.ndarray] = None,
) -> dict[str, tuple[np.ndarray, Optional[np.ndarray]]]:
    """Return name -> (oof, test_or_None). Prefer *_full.npy / *_test.npy for test."""
    paths: dict[str, tuple[np.ndarray, Optional[np.ndarray]]] = {}

    def put(name: str, o: np.ndarray, t: Optional[np.ndarray]):
        if len(o) != n_train:
            return
        if t is not None and len(t) != n_test:
            t = None
        if require_test and t is None:
            return
        if name in paths and paths[name][1] is not None and t is None:
            return
        paths[name] = (o.astype(np.float64), None if t is None else t.astype(np.float64))

    for f in sorted(glob.glob(f"{root}/**/*.npy", recursive=True)):
        if "/._" in f or os.path.basename(f).startswith("._"):
            continue
        bn, dn = os.path.basename(f), os.path.dirname(f)
        if bn.startswith("oof_"):
            base = bn[4:-4]
            tpath = os.path.join(dn, "test_" + base + ".npy")
            if not os.path.exists(tpath):
                tpath = os.path.join(dn, "full_" + base + ".npy")
            o = np.load(f)
            t = np.load(tpath) if os.path.exists(tpath) else None
            put(owner_of(dn)[:14] + ":" + base, o, t)
        elif bn.endswith("_oof.npy"):
            base = bn[:-8]
            # skip stack meta OOFs that aren't base models unless wanted
            tpath = os.path.join(dn, base + "_test.npy")
            if not os.path.exists(tpath):
                tpath = os.path.join(dn, base + "_full.npy")
            o = np.load(f)
            t = np.load(tpath) if os.path.exists(tpath) else None
            put(owner_of(dn)[:14] + ":" + base, o, t)

    for f in sorted(glob.glob(f"{root}/**/oof.npy", recursive=True)):
        if "/._" in f:
            continue
        dn = os.path.dirname(f)
        tpath = os.path.join(dn, "test.npy")
        if not os.path.exists(tpath):
            continue
        n = "dir:" + os.path.relpath(dn, root).replace(os.sep, "/")[:60]
        put(n, np.load(f), np.load(tpath))

    for f in sorted(glob.glob(f"{root}/**/*.npz", recursive=True)):
        if "/._" in f or os.path.basename(f).startswith("._"):
            continue
        try:
            z = np.load(f, allow_pickle=False)
        except Exception:
            continue
        keys = list(z.keys())
        for k in keys:
            if "oof" not in k.lower() or z[k].ndim != 1 or len(z[k]) != n_train:
                continue
            test_arr = None
            for alt in (
                k.replace("oof", "test"),
                k.replace("oof", "full"),
                k.replace("oof", "tst"),
                "test",
                "full",
                "stack_test",
            ):
                if alt in z and getattr(z[alt], "ndim", 0) == 1 and len(z[alt]) == n_test:
                    test_arr = z[alt]
                    break
            if test_arr is None:
                continue
            n = owner_of(os.path.dirname(f))[:14] + ":" + os.path.basename(f)[:-4] + ":" + k
            put(n, z[k], test_arr)
        if "seed_oof" in z and "fold_test" in z:
            o = z["seed_oof"].mean(1) if z["seed_oof"].ndim == 2 else z["seed_oof"]
            t = z["fold_test"].mean(0) if z["fold_test"].ndim == 2 else z["fold_test"]
            n = owner_of(os.path.dirname(f))[:14] + ":" + os.path.basename(f)[:-4] + ":seedmean"
            put(n, o, t)
        if "oof" in z and "fold_test" in z and z["oof"].ndim == 1 and len(z["oof"]) == n_train:
            t = z["fold_test"].mean(0) if z["fold_test"].ndim == 2 else z["fold_test"]
            n = owner_of(os.path.dirname(f))[:14] + ":" + os.path.basename(f)[:-4] + ":foldmean"
            put(n, z["oof"], t)

    fold_root = os.path.join(root, "artifacts_extra", "s6e11-cross-fitted-artifacts", "fold_predictions")
    if os.path.isdir(fold_root):
        models: dict[str, list[str]] = {}
        for f in sorted(glob.glob(fold_root + "/*_fold*.npz")):
            base = os.path.basename(f).rsplit("_fold", 1)[0]
            models.setdefault(base, []).append(f)
        for base, files in models.items():
            oof = np.zeros(n_train, np.float64)
            test_sum = np.zeros(n_test, np.float64)
            ok = True
            for f in files:
                z = np.load(f)
                if not {"valid_index", "valid", "test"} <= set(z.keys()):
                    ok = False
                    break
                oof[z["valid_index"]] = z["valid"]
                test_sum += z["test"]
            if ok and len(files) == 5:
                put("crossfit:" + base, oof, test_sum / 5.0)

    # Prefer explicit self aliases for promoted models
    alias = {
        "self:te_xgb_smooth50": os.path.join(root, "preds", "stage4", "te_xgb_smooth50_oof.npy"),
        "self:te_lgbm_leaves3": os.path.join(root, "preds", "stage5", "te_lgbm_leaves3_oof.npy"),
    }
    for name, opath in alias.items():
        if not os.path.exists(opath):
            continue
        tpath = opath.replace("_oof.npy", "_full.npy")
        if not os.path.exists(tpath):
            tpath = opath.replace("_oof.npy", "_test.npy")
        t = np.load(tpath) if os.path.exists(tpath) else None
        put(name, np.load(opath), t)

    if min_auc is not None and y is not None:
        paths = {
            n: vt
            for n, vt in paths.items()
            if roc_auc_score(y, vt[0]) >= min_auc
        }
    return paths


def fold_aucs(y: np.ndarray, oof: np.ndarray, folds) -> list[float]:
    return [float(roc_auc_score(y[va], oof[va])) for _, va in folds]


def load_y(root: str) -> tuple[np.ndarray, int, int]:
    y = pd.read_csv(os.path.join(root, "data", "train.csv"), usecols=["is_fraud"])[
        "is_fraud"
    ].astype(int).values
    nte = len(pd.read_csv(os.path.join(root, "data", "sample_submission.csv")))
    return y, len(y), nte
