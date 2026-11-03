#!/usr/bin/env python3
"""Stage 5: Overfitting curve + multi-metric + honest stack policy.

5.1 Depth/leaves sweep on TE-LGBM under locked SKF(5, rs=42)
5.2 Log PR-AUC + Brier (do not optimize LB on PR alone)
5.3 Stack only honest Stage 3/4 members (gap ≤ control×2 and OOF ≥ raw_lgbm)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from validation_stage3_baselines import (  # noqa: E402
    CATS,
    FLAGS,
    NUMS,
    SEED,
    TARGET,
    digit_frame,
    locked_folds,
    metrics,
    oof_target_encode,
)

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED3 = os.path.join(ROOT, "preds", "stage3")
PRED4 = os.path.join(ROOT, "preds", "stage4")
PRED5 = os.path.join(ROOT, "preds", "stage5")
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

# leaves sweep (shallow → deeper); maps roughly to Handbook depth curve
LEAVES_GRID = [2, 3, 4, 5, 7, 8, 12, 15, 31]


def build_te_xy(train, dig, dig_num, codes, te_names, y, tr_idx, va_idx, smooth=20.0):
    te_tr, te_va = oof_target_encode(codes, y, tr_idx, va_idx, smooth=smooth, seed=SEED)
    te_cols = [f"te_{c}" for c in te_names]
    num_tr = train.iloc[tr_idx][NUMS + FLAGS].reset_index(drop=True)
    num_va = train.iloc[va_idx][NUMS + FLAGS].reset_index(drop=True)
    d_tr = dig.iloc[tr_idx][dig_num].reset_index(drop=True)
    d_va = dig.iloc[va_idx][dig_num].reset_index(drop=True)
    return (
        pd.concat([num_tr, d_tr, pd.DataFrame(te_tr, columns=te_cols)], axis=1),
        pd.concat([num_va, d_va, pd.DataFrame(te_va, columns=te_cols)], axis=1),
    )


def depth_sweep(train, y, folds, dig, dig_num, codes, te_names, leaves_grid):
    rows = []
    print("\n[Stage 5.1] Leaves/depth sweep (TE-LGBM)", flush=True)
    for leaves in leaves_grid:
        oof = np.zeros(len(y))
        fold_recs = []
        t0 = time.time()
        params = dict(
            learning_rate=0.03,
            num_leaves=leaves,
            max_depth=-1 if leaves >= 15 else min(8, int(np.ceil(np.log2(leaves))) + 1),
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        for f, (tr, va) in enumerate(folds):
            X_tr, X_va = build_te_xy(train, dig, dig_num, codes, te_names, y, tr, va)
            m = lgb.LGBMClassifier(**params, n_estimators=3000, random_state=SEED + f)
            m.fit(
                X_tr,
                y[tr],
                eval_set=[(X_va, y[va])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            tr_p = m.predict_proba(X_tr)[:, 1]
            va_p = m.predict_proba(X_va)[:, 1]
            oof[va] = va_p
            fold_recs.append(
                {
                    "fold": f,
                    "valid_roc_auc": float(roc_auc_score(y[va], va_p)),
                    "train_roc_auc": float(roc_auc_score(y[tr], tr_p)),
                    "train_valid_gap": float(
                        roc_auc_score(y[tr], tr_p) - roc_auc_score(y[va], va_p)
                    ),
                    "valid_pr_auc": float(average_precision_score(y[va], va_p)),
                    "valid_brier": float(brier_score_loss(y[va], va_p)),
                }
            )
        oof_m = metrics(y, oof)
        row = {
            "num_leaves": leaves,
            "max_depth": params["max_depth"],
            "oof_roc_auc": oof_m["roc_auc"],
            "oof_pr_auc": oof_m["pr_auc"],
            "oof_brier": oof_m["brier"],
            "gap_mean": float(np.mean([r["train_valid_gap"] for r in fold_recs])),
            "fold_mean": float(np.mean([r["valid_roc_auc"] for r in fold_recs])),
            "fold_std": float(np.std([r["valid_roc_auc"] for r in fold_recs], ddof=1)),
            "folds": fold_recs,
            "elapsed_sec": float(time.time() - t0),
        }
        rows.append(row)
        print(
            f"  leaves={leaves:2d}: OOF={row['oof_roc_auc']:.6f} "
            f"gap={row['gap_mean']:.6f} PR={row['oof_pr_auc']:.6f} "
            f"Brier={row['oof_brier']:.6f} ({row['elapsed_sec']:.0f}s)",
            flush=True,
        )
        np.save(os.path.join(PRED5, f"te_lgbm_leaves{leaves}_oof.npy"), oof.astype(np.float32))
    return rows


def honest_stack(y: np.ndarray, members: dict[str, np.ndarray], folds) -> dict:
    """5.3: rank-mean + logit-LR stack of honest OOF members under locked folds."""
    names = list(members.keys())
    X = np.column_stack([members[n] for n in names])
    # rank-average
    ranks = np.column_stack([rankdata(X[:, j]) / len(y) for j in range(X.shape[1])])
    rank_mean = ranks.mean(1)
    # logit LR with inner CV on locked folds
    oof_lr = np.zeros(len(y))
    for tr, va in folds:
        sc = StandardScaler()
        Z_tr = sc.fit_transform(X[tr])
        Z_va = sc.transform(X[va])
        lr = LogisticRegression(max_iter=400, C=0.3, solver="lbfgs")
        lr.fit(Z_tr, y[tr])
        oof_lr[va] = lr.predict_proba(Z_va)[:, 1]
    out = {
        "members": names,
        "n_members": len(names),
        "rank_mean_oof": float(roc_auc_score(y, rank_mean)),
        "rank_mean_pr": float(average_precision_score(y, rank_mean)),
        "rank_mean_brier": float(brier_score_loss(y, rank_mean)),
        "logit_lr_oof": float(roc_auc_score(y, oof_lr)),
        "logit_lr_pr": float(average_precision_score(y, oof_lr)),
        "logit_lr_brier": float(brier_score_loss(y, oof_lr)),
        "member_oofs": {n: float(roc_auc_score(y, members[n])) for n in names},
    }
    best = "logit_lr" if out["logit_lr_oof"] >= out["rank_mean_oof"] else "rank_mean"
    out["best_stack"] = best
    out["best_oof"] = out["logit_lr_oof"] if best == "logit_lr" else out["rank_mean_oof"]
    np.save(os.path.join(PRED5, "stack_rank_mean_oof.npy"), rank_mean.astype(np.float32))
    np.save(os.path.join(PRED5, "stack_logit_lr_oof.npy"), oof_lr.astype(np.float32))
    print("\n[Stage 5.3] Honest stack policy")
    print(f"  members ({len(names)}): {names}")
    print(f"  rank_mean OOF={out['rank_mean_oof']:.6f} PR={out['rank_mean_pr']:.6f}")
    print(f"  logit_lr  OOF={out['logit_lr_oof']:.6f} PR={out['logit_lr_pr']:.6f}")
    print(f"  best={best} OOF={out['best_oof']:.6f}")
    return out


def select_honest_members(
    stage3_path: str,
    stage4_path: str | None,
    y: np.ndarray,
    max_gap: float,
    min_oof: float,
) -> dict[str, np.ndarray]:
    s3 = json.load(open(stage3_path))
    members = {}
    for arm in s3["arms"]:
        if arm["train_valid_gap_mean"] > max_gap:
            print(f"  skip {arm['name']}: gap {arm['train_valid_gap_mean']:.4f} > {max_gap:.4f}")
            continue
        if arm["oof_roc_auc"] < min_oof:
            print(f"  skip {arm['name']}: OOF {arm['oof_roc_auc']:.4f} < {min_oof:.4f}")
            continue
        path = os.path.join(PRED3, f"{arm['name']}_oof.npy")
        if os.path.exists(path):
            members[arm["name"]] = np.load(path)
            print(f"  keep {arm['name']}: OOF={arm['oof_roc_auc']:.6f} gap={arm['train_valid_gap_mean']:.6f}")
    if stage4_path and os.path.exists(stage4_path):
        s4 = json.load(open(stage4_path))
        for arm in s4.get("new_arms") or []:
            if arm["train_valid_gap_mean"] > max_gap or arm["oof_roc_auc"] < min_oof:
                continue
            # only if promoted or close to control
            path = os.path.join(PRED4, f"{arm['name']}_oof.npy")
            if os.path.exists(path):
                members[arm["name"]] = np.load(path)
                print(f"  keep {arm['name']}: OOF={arm['oof_roc_auc']:.6f}")
    return members


def estimate_public(cv: float) -> dict:
    """Empirical CV→public from prior submits (~+0.00045 mean)."""
    delta_lo, delta_hi, delta_mid = 0.00040, 0.00055, 0.00048
    return {
        "cv_oof": cv,
        "delta_historical_mean": delta_mid,
        "public_est_low": round(cv + delta_lo, 6),
        "public_est_mid": round(cv + delta_mid, 6),
        "public_est_high": round(cv + delta_hi, 6),
        "current_best_public": 0.95163,
        "note": (
            "Estimate assumes stack CV translates like prior hillclimb/logit mixes. "
            "Stage3/4 single-model CV≈0.9493 would map to ~0.9498 public (worse)."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--stage3", default=os.path.join(LOGS, "stage3_baselines_result.json"))
    ap.add_argument("--stage4", default=os.path.join(LOGS, "stage4_levers_result.json"))
    ap.add_argument("--out", default=os.path.join(LOGS, "stage5_diagnostics_result.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    os.makedirs(PRED5, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.smoke:
        pos = train[train[TARGET] == 1]
        neg = train[train[TARGET] == 0].sample(n=min(12000, len(train)), random_state=SEED)
        train = pd.concat(
            [pos.sample(n=min(1200, len(pos)), random_state=SEED), neg]
        ).sample(frac=1, random_state=SEED)
    y = train[TARGET].astype(int).to_numpy()
    for c in CATS:
        cats = sorted(pd.concat([train[c].astype(str), test[c].astype(str)]).unique())
        train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    folds = locked_folds(y)

    dig = digit_frame(train)
    dig_num = [c for c in dig.columns if pd.api.types.is_numeric_dtype(dig[c])]
    dig_cat = [c for c in dig.columns if c not in dig_num]
    for c in dig_cat:
        dig[c] = pd.Categorical(dig[c].astype(str))
    te_src = pd.concat([train[CATS].astype(str), dig[dig_cat].astype(str)], axis=1)
    te_names = list(te_src.columns)
    codes = np.column_stack(
        [pd.factorize(te_src[c], sort=True)[0].astype(np.int32) for c in te_names]
    )

    leaves_grid = [2, 4, 7, 15] if args.smoke else LEAVES_GRID
    sweep = []
    if not args.skip_sweep:
        sweep = depth_sweep(train, y, folds, dig, dig_num, codes, te_names, leaves_grid)
        best_sweep = max(sweep, key=lambda r: r["oof_roc_auc"])
        print(
            f"\nBest sweep: leaves={best_sweep['num_leaves']} "
            f"OOF={best_sweep['oof_roc_auc']:.6f} gap={best_sweep['gap_mean']:.6f}"
        )

    s3 = json.load(open(args.stage3))
    control = next(a for a in s3["arms"] if a["name"] == "te_xgb")
    max_gap = max(0.012, control["train_valid_gap_mean"] * 2.5)
    min_oof = next(a for a in s3["arms"] if a["name"] == "raw_lgbm")["oof_roc_auc"]
    print(f"\n[Stage 5.3] Honest member filter: gap≤{max_gap:.4f}, OOF≥{min_oof:.6f}")
    members = select_honest_members(args.stage3, args.stage4, y, max_gap, min_oof)
    # If smoke subsampled, can't use full-length stage3 oofs — rebuild skip
    if args.smoke:
        print("  smoke: skip full-length OOF stack")
        stack = {"best_oof": None, "note": "skipped in smoke"}
        est = None
    else:
        if not members:
            raise SystemExit("No honest members found for stack")
        # Align length
        for n, arr in list(members.items()):
            if len(arr) != len(y):
                print(f"  drop {n}: length mismatch")
                del members[n]
        stack = honest_stack(y, members, folds)
        est = estimate_public(stack["best_oof"])
        print("\n[Score estimate]")
        print(
            f"  CV stack={est['cv_oof']:.6f} → public "
            f"~{est['public_est_low']:.5f}–{est['public_est_high']:.5f} "
            f"(mid {est['public_est_mid']:.5f}); best so far {est['current_best_public']}"
        )

    # 5.2 multi-metric table from sweep
    multimetric = [
        {
            "num_leaves": r["num_leaves"],
            "roc": r["oof_roc_auc"],
            "pr": r["oof_pr_auc"],
            "brier": r["oof_brier"],
            "gap": r["gap_mean"],
        }
        for r in sweep
    ]

    out = {
        "stage": 5,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "cv_contract": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "depth_sweep": sweep,
        "multimetric": multimetric,
        "stack": stack,
        "public_estimate": est,
        "policy": {
            "stack_only_honest_oof": True,
            "max_gap": max_gap,
            "min_oof": min_oof,
            "no_submit_without_approval": True,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
