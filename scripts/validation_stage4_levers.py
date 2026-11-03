#!/usr/bin/env python3
"""Stage 4: One-variable levers vs Stage 3 control (te_xgb), Stage 2.2 gate.

4.1 Propose digit / TE / raw / GLM / leaves / TE-smooth as single levers
4.2 Promote only if OOF↑ AND ≥4/5 folds↑ vs control fold vector
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
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import (  # noqa: E402
    CATS,
    DEC,
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
N_JOBS = max(1, (os.cpu_count() or 4) - 1)
CONTROL_NAME = "te_xgb"


def load_stage3_control(path: str) -> dict:
    doc = json.load(open(path))
    control = next(a for a in doc["arms"] if a["name"] == CONTROL_NAME)
    return {
        "name": CONTROL_NAME,
        "oof": control["oof_roc_auc"],
        "folds": [f["valid_roc_auc"] for f in control["folds"]],
        "gap": control["train_valid_gap_mean"],
        "arms": doc["arms"],
    }


def gate_vs_control(control: dict, name: str, oof: float, folds: list[float]) -> dict:
    g = promotion_gate(control["folds"], folds, control["oof"], oof)
    g["lever"] = name
    g["challenger_oof"] = oof
    g["challenger_folds"] = folds
    return g


def run_lgb(X_tr, y_tr, X_va, y_va, params, n_estimators=4000, seed=SEED):
    m = lgb.LGBMClassifier(**params, n_estimators=n_estimators, random_state=seed)
    m.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1], int(m.best_iteration_ or n_estimators)


def run_xgb(X_tr, y_tr, X_va, y_va, params, n_estimators=4000, seed=SEED):
    m = xgb.XGBClassifier(
        **params, n_estimators=n_estimators, random_state=seed, early_stopping_rounds=200
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1], int(m.best_iteration + 1)


def cv_arm(name, folds, y, build_xy, fit_fn) -> dict:
    oof = np.zeros(len(y))
    recs = []
    t0 = time.time()
    print(f"\n=== Lever {name} ===", flush=True)
    for f, (tr, va) in enumerate(folds):
        X_tr, X_va = build_xy(tr, va)
        tr_p, va_p, it = fit_fn(X_tr, y[tr], X_va, y[va], seed=SEED + f)
        oof[va] = va_p
        tr_m, va_m = metrics(y[tr], tr_p), metrics(y[va], va_p)
        gap = tr_m["roc_auc"] - va_m["roc_auc"]
        recs.append(
            {
                "fold": f,
                "valid_roc_auc": va_m["roc_auc"],
                "train_roc_auc": tr_m["roc_auc"],
                "train_valid_gap": float(gap),
                "valid_pr_auc": va_m["pr_auc"],
                "valid_brier": va_m["brier"],
                "best_iteration": it,
            }
        )
        print(
            f"  fold {f}: auc={va_m['roc_auc']:.6f} gap={gap:.6f} "
            f"pr={va_m['pr_auc']:.6f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    oof_m = metrics(y, oof)
    summary = {
        "name": name,
        "oof_roc_auc": oof_m["roc_auc"],
        "oof_pr_auc": oof_m["pr_auc"],
        "oof_brier": oof_m["brier"],
        "fold_valid_roc_auc_mean": float(np.mean([r["valid_roc_auc"] for r in recs])),
        "fold_valid_roc_auc_std": float(np.std([r["valid_roc_auc"] for r in recs], ddof=1)),
        "train_valid_gap_mean": float(np.mean([r["train_valid_gap"] for r in recs])),
        "folds": recs,
        "elapsed_sec": float(time.time() - t0),
    }
    print(
        f"==> {name}: OOF={summary['oof_roc_auc']:.6f} gap={summary['train_valid_gap_mean']:.6f}",
        flush=True,
    )
    np.save(os.path.join(PRED4, f"{name}_oof.npy"), oof.astype(np.float32))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument(
        "--stage3",
        default=os.path.join(LOGS, "stage3_baselines_result.json"),
    )
    ap.add_argument("--out", default=os.path.join(LOGS, "stage4_levers_result.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-new", action="store_true", help="only gate stage3 arms")
    args = ap.parse_args()
    os.makedirs(PRED4, exist_ok=True)

    control = load_stage3_control(args.stage3)
    print(
        f"Control {control['name']}: OOF={control['oof']:.6f} "
        f"folds={[round(x, 5) for x in control['folds']]}"
    )

    # --- 4.1/4.2: gate existing Stage 3 arms (one view each) ---
    gated = []
    for arm in control["arms"]:
        if arm["name"] == CONTROL_NAME:
            continue
        folds = [f["valid_roc_auc"] for f in arm["folds"]]
        g = gate_vs_control(control, f"stage3:{arm['name']}", arm["oof_roc_auc"], folds)
        g["variable"] = f"view={arm['name']}"
        g["gap"] = arm["train_valid_gap_mean"]
        gated.append(g)

    new_arms = []
    if not args.skip_new:
        train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        if args.smoke:
            pos = train[train[TARGET] == 1]
            neg = train[train[TARGET] == 0].sample(n=min(15000, len(train)), random_state=SEED)
            train = pd.concat(
                [pos.sample(n=min(1500, len(pos)), random_state=SEED), neg]
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

        def xy_te(tr_idx, va_idx, smooth=20.0):
            te_tr, te_va = oof_target_encode(
                codes, y, tr_idx, va_idx, smooth=smooth, seed=SEED
            )
            te_cols = [f"te_{c}" for c in te_names]
            num_tr = train.iloc[tr_idx][NUMS + FLAGS].reset_index(drop=True)
            num_va = train.iloc[va_idx][NUMS + FLAGS].reset_index(drop=True)
            d_tr = dig.iloc[tr_idx][dig_num].reset_index(drop=True)
            d_va = dig.iloc[va_idx][dig_num].reset_index(drop=True)
            return (
                pd.concat([num_tr, d_tr, pd.DataFrame(te_tr, columns=te_cols)], axis=1),
                pd.concat([num_va, d_va, pd.DataFrame(te_va, columns=te_cols)], axis=1),
            )

        LGB = dict(
            learning_rate=0.03,
            num_leaves=4,
            max_depth=4,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        XGB = dict(
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="auc",
            n_jobs=N_JOBS,
        )

        # Lever A: num_leaves 4 → 7 on TE-LGBM
        def fit_l7(X_tr, y_tr, X_va, y_va, seed=SEED):
            p = dict(LGB)
            p["num_leaves"] = 7
            return run_lgb(X_tr, y_tr, X_va, y_va, p, seed=seed)

        arm = cv_arm("te_lgbm_leaves7", folds, y, lambda a, b: xy_te(a, b, 20.0), fit_l7)
        arm["variable"] = "num_leaves:4→7 (TE-LGBM)"
        new_arms.append(arm)
        gated.append(
            gate_vs_control(
                control,
                arm["name"],
                arm["oof_roc_auc"],
                [f["valid_roc_auc"] for f in arm["folds"]],
            )
            | {"variable": arm["variable"], "gap": arm["train_valid_gap_mean"]}
        )

        # Lever B: TE smooth 20 → 50
        def fit_te50(X_tr, y_tr, X_va, y_va, seed=SEED):
            return run_xgb(X_tr, y_tr, X_va, y_va, dict(XGB), seed=seed)

        arm = cv_arm(
            "te_xgb_smooth50",
            folds,
            y,
            lambda a, b: xy_te(a, b, 50.0),
            fit_te50,
        )
        arm["variable"] = "TE smooth:20→50 (TE-XGB)"
        new_arms.append(arm)
        gated.append(
            gate_vs_control(
                control,
                arm["name"],
                arm["oof_roc_auc"],
                [f["valid_roc_auc"] for f in arm["folds"]],
            )
            | {"variable": arm["variable"], "gap": arm["train_valid_gap_mean"]}
        )

        # Lever C: XGB depth 2 → 3 on TE
        def fit_d3(X_tr, y_tr, X_va, y_va, seed=SEED):
            p = dict(XGB)
            p["max_depth"] = 3
            return run_xgb(X_tr, y_tr, X_va, y_va, p, seed=seed)

        arm = cv_arm("te_xgb_depth3", folds, y, lambda a, b: xy_te(a, b, 20.0), fit_d3)
        arm["variable"] = "max_depth:2→3 (TE-XGB)"
        new_arms.append(arm)
        gated.append(
            gate_vs_control(
                control,
                arm["name"],
                arm["oof_roc_auc"],
                [f["valid_roc_auc"] for f in arm["folds"]],
            )
            | {"variable": arm["variable"], "gap": arm["train_valid_gap_mean"]}
        )

        # Lever D: GLM on TE features (new model family)
        def fit_glm(X_tr, y_tr, X_va, y_va, seed=SEED):
            sc = StandardScaler()
            Z_tr = sc.fit_transform(X_tr.astype(np.float64))
            Z_va = sc.transform(X_va.astype(np.float64))
            m = LogisticRegression(
                max_iter=500, C=0.5, solver="lbfgs", random_state=seed, n_jobs=N_JOBS
            )
            m.fit(Z_tr, y_tr)
            return m.predict_proba(Z_tr)[:, 1], m.predict_proba(Z_va)[:, 1], 1

        arm = cv_arm("te_glm", folds, y, lambda a, b: xy_te(a, b, 20.0), fit_glm)
        arm["variable"] = "model_family: GLM on TE view"
        new_arms.append(arm)
        gated.append(
            gate_vs_control(
                control,
                arm["name"],
                arm["oof_roc_auc"],
                [f["valid_roc_auc"] for f in arm["folds"]],
            )
            | {"variable": arm["variable"], "gap": arm["train_valid_gap_mean"]}
        )

    promoted = [g for g in gated if g.get("promote")]
    rejected = [g for g in gated if not g.get("promote")]
    print("\n=== Stage 4.2 gate summary ===")
    for g in gated:
        mark = "PROMOTE" if g["promote"] else "REJECT "
        print(
            f"  {mark} {g['lever']}: OOFΔ={g['oof_delta']:+.6f} "
            f"wins={g['fold_wins']}/5 | {g.get('variable', '')}"
        )

    out = {
        "stage": 4,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "cv_contract": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "control": {
            "name": control["name"],
            "oof_roc_auc": control["oof"],
            "folds": control["folds"],
            "gap": control["gap"],
        },
        "promotion_rule": "OOF↑ AND ≥4/5 folds↑ vs control",
        "gates": gated,
        "promoted": [g["lever"] for g in promoted],
        "rejected": [g["lever"] for g in rejected],
        "new_arms": new_arms,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"Promoted: {out['promoted'] or '(none)'}")


if __name__ == "__main__":
    main()
