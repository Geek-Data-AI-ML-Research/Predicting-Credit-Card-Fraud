#!/usr/bin/env python3
"""Stage 6.1: Audit pooled public + self OOF bank (logit corr, prune, diversity)."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from oof_bank_loader import collect_oof_bank, fold_aucs, load_y, locked_folds  # noqa: E402

LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(LOGS, "stage6_1_oof_audit.json")
EPS = 1e-6
CORR_THRESH = 0.995


def to_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, EPS, 1.0 - EPS))


def main() -> None:
    t0 = time.time()
    y, ntr, nte = load_y(ROOT)
    folds = locked_folds(y)
    bank = collect_oof_bank(ROOT, ntr, nte, require_test=False, y=y)
    print(f"collected {len(bank)} OOF members (train={ntr} test={nte})", flush=True)

    required = ["self:te_xgb_smooth50", "self:te_lgbm_leaves3"]
    # also accept owner-prefixed variants
    names = list(bank)
    for req in required:
        if req not in bank:
            # fuzzy
            key = req.split(":", 1)[1]
            hits = [n for n in names if n.endswith(":" + key) or n.endswith(key)]
            if hits:
                # alias best
                best = max(hits, key=lambda n: roc_auc_score(y, bank[n][0]))
                bank[req] = bank[best]
                print(f"  aliased {req} <- {best}", flush=True)
            else:
                print(f"  WARNING missing {req}", flush=True)

    rows = []
    for n, (o, t) in bank.items():
        auc = float(roc_auc_score(y, o))
        fa = fold_aucs(y, o, folds)
        rows.append(
            {
                "name": n,
                "oof_auc": auc,
                "fold_aucs": fa,
                "has_test": t is not None,
                "mean": float(np.mean(o)),
                "std": float(np.std(o)),
            }
        )
    rows.sort(key=lambda r: -r["oof_auc"])
    print("\n=== Top 20 by standalone OOF ===", flush=True)
    print(f"{'rank':>4} {'oof':>10} {'folds':>28} {'test':>5} name", flush=True)
    for i, r in enumerate(rows[:20], 1):
        fs = ",".join(f"{a:.4f}" for a in r["fold_aucs"])
        print(
            f"{i:4d} {r['oof_auc']:.6f} [{fs}] {'Y' if r['has_test'] else 'N':>5} {r['name']}",
            flush=True,
        )

    # Logit-space pairwise correlation on full train
    order = [r["name"] for r in rows]
    Z = np.column_stack([to_logit(bank[n][0]) for n in order])
    # standardize columns
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    corr = np.corrcoef(Z.T)
    # redundancy prune: pairs > 0.995 keep higher OOF
    auc_map = {r["name"]: r["oof_auc"] for r in rows}
    keep = []
    dropped = []
    for n in order:
        red = False
        for k in keep:
            i, j = order.index(n), order.index(k)
            if corr[i, j] > CORR_THRESH:
                red = True
                dropped.append(
                    {
                        "dropped": n,
                        "kept": k,
                        "logit_corr": float(corr[i, j]),
                        "dropped_auc": auc_map[n],
                        "kept_auc": auc_map[k],
                    }
                )
                break
        if not red:
            keep.append(n)

    top_name = order[0]
    top_i = 0
    diversity = []
    for i, n in enumerate(order):
        if n == top_name:
            max_corr = 1.0
        else:
            # max corr with any kept model that has higher OOF, else with top
            max_corr = float(corr[i, top_i])
        diversity.append(
            {
                "name": n,
                "oof_auc": auc_map[n],
                "max_corr_with_top": max_corr,
                "diversity": float(1.0 - max_corr),
                "diversity_adj_score": float(auc_map[n] + 0.002 * (1.0 - max_corr)),
            }
        )
    # Top 15 by diversity-adjusted residual correlation with best model
    # Sort by diversity (high) among models with oof reasonably close, or by diversity_adj
    by_div = sorted(
        [d for d in diversity if d["name"] != top_name],
        key=lambda d: (-d["diversity"], -d["oof_auc"]),
    )[:15]

    print(f"\n=== Prune corr>{CORR_THRESH}: kept {len(keep)} / {len(order)} ===", flush=True)
    print(f"dropped {len(dropped)} redundant pairs (first 10):", flush=True)
    for d in dropped[:10]:
        print(
            f"  drop {d['dropped'][:40]:40s} keep {d['kept'][:40]:40s} "
            f"corr={d['logit_corr']:.5f}",
            flush=True,
        )

    print("\n=== Top 15 by diversity vs best model ===", flush=True)
    print(f"best={top_name} oof={auc_map[top_name]:.6f}", flush=True)
    print(f"{'div':>8} {'oof':>10} {'corr_top':>9} name", flush=True)
    for d in by_div:
        print(
            f"{d['diversity']:8.4f} {d['oof_auc']:.6f} {d['max_corr_with_top']:9.4f} {d['name']}",
            flush=True,
        )

    # compact corr of top-20 names for JSON (full matrix too large / redundant)
    top20 = order[:20]
    idx20 = [order.index(n) for n in top20]
    corr20 = corr[np.ix_(idx20, idx20)].tolist()

    self_status = {
        req: {
            "present": req in bank,
            "oof_auc": float(roc_auc_score(y, bank[req][0])) if req in bank else None,
            "has_test": bool(bank[req][1] is not None) if req in bank else False,
            "fold_aucs": fold_aucs(y, bank[req][0], folds) if req in bank else None,
        }
        for req in required
    }

    doc = {
        "stage": "6.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_members": len(order),
        "n_kept_after_prune": len(keep),
        "corr_threshold": CORR_THRESH,
        "self_promoted": self_status,
        "top20_standalone": rows[:20],
        "top15_diversity_vs_best": by_div,
        "best_model": {"name": top_name, "oof_auc": auc_map[top_name]},
        "kept_after_prune": keep,
        "dropped_redundant": dropped,
        "corr_logit_top20_names": top20,
        "corr_logit_top20": corr20,
        "all_members": [
            {
                "name": r["name"],
                "oof_auc": r["oof_auc"],
                "fold_aucs": r["fold_aucs"],
                "has_test": r["has_test"],
                "diversity_vs_best": next(
                    (d["diversity"] for d in diversity if d["name"] == r["name"]), None
                ),
            }
            for r in rows
        ],
    }
    os.makedirs(LOGS, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {OUT} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
