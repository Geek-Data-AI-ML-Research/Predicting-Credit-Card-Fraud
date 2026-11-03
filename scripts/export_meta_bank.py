#!/usr/bin/env python3
"""Export 48-pruned OOF bank (+ self/promoted) for Stage 7.2 GPU meta-stacker."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from oof_bank_loader import collect_oof_bank, load_y  # noqa: E402

PRED7 = os.path.join(ROOT, "preds", "stage7")
CORR = 0.995


def main():
    y, ntr, nte = load_y(ROOT)
    bank = collect_oof_bank(ROOT, ntr, nte, require_test=True, y=y)
    # force self
    for alias, op, tp in [
        ("self:te_xgb_smooth50", "preds/stage4/te_xgb_smooth50_oof.npy", "preds/stage4/te_xgb_smooth50_full.npy"),
        ("self:te_lgbm_leaves3", "preds/stage5/te_lgbm_leaves3_oof.npy", "preds/stage5/te_lgbm_leaves3_full.npy"),
    ]:
        op, tp = os.path.join(ROOT, op), os.path.join(ROOT, tp)
        if os.path.exists(op) and os.path.exists(tp):
            bank[alias] = (np.load(op).astype(np.float64), np.load(tp).astype(np.float64))
    # add any stage6 promotions
    for root, _, files in os.walk(os.path.join(ROOT, "preds", "stage6")):
        for f in files:
            if f.endswith("_oof.npy"):
                base = f[:-8]
                tp = os.path.join(root, base + "_full.npy")
                op = os.path.join(root, f)
                if os.path.exists(tp):
                    bank[f"stage6:{base}"] = (np.load(op).astype(np.float64), np.load(tp).astype(np.float64))

    names = sorted(bank, key=lambda n: -roc_auc_score(y, bank[n][0]))
    names = [n for n in names if roc_auc_score(y, bank[n][0]) >= 0.94]
    O = np.column_stack([bank[n][0] for n in names])
    T = np.column_stack([bank[n][1] for n in names])
    # prune
    from scipy.special import logit

    Z = logit(np.clip(O, 1e-6, 1 - 1e-6))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    corr = np.corrcoef(Z.T)
    order = np.argsort([-roc_auc_score(y, O[:, j]) for j in range(O.shape[1])])
    keep = []
    for j in order:
        if any(corr[j, k] > CORR for k in keep):
            continue
        keep.append(int(j))
    names_p = [names[j] for j in keep]
    Op, Tp = O[:, keep], T[:, keep]
    os.makedirs(PRED7, exist_ok=True)
    path = os.path.join(PRED7, "meta_bank.npz")
    np.savez_compressed(path, names=np.array(names_p, dtype=object), O=Op, T=Tp, y=y)
    meta = {
        "n_members": len(names_p),
        "names": names_p,
        "top_aucs": {n: float(roc_auc_score(y, bank[n][0])) for n in names_p[:20]},
    }
    json.dump(meta, open(os.path.join(PRED7, "meta_bank_meta.json"), "w"), indent=2)
    print(f"wrote {path} m={len(names_p)}")


if __name__ == "__main__":
    main()
