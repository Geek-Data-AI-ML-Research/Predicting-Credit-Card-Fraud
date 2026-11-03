#!/usr/bin/env python3
"""Stage 10: Upgrade diglead_l3_bag discovery with smooth=50 TE + orig20k feats.

Arms:
  10A lgb_diglead_l3_smooth50_orig_bag  — LGBM leaves 3/4 × 5 seeds
  10B xgb_cuda_diglead_l3_smooth50_orig_bag — CUDA XGB max_leaves 3/4 × 5 seeds
  10C cat_cuda_diglead_d2_smooth50_orig — CUDA CatBoost depth=2

Gate each vs lgb_diglead_l3_bag (0.95063). NO kaggle submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diglead_features import (  # noqa: E402
    build_diglead_codes,
    build_fold_diglead_matrix,
    load_orig20k_cdist,
)
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED9 = os.path.join(ROOT, "preds", "stage9")
PRED10 = os.path.join(ROOT, "preds", "stage10")
SUBS = os.path.join(ROOT, "submissions")
SEED = 42
BAG_SEEDS = [42, 100, 2026, 777, 999]
CTRL_DIGLEAD = 0.95063
EPS = 1e-6

# Stage 9.2 embedding cols (must match stage9_2)
CONT = [
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
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
FLAG_W = {
    "ip_country_mismatch": 2.0,
    "billing_shipping_mismatch": 1.5,
    "is_ai_generated_scam_attempt": 2.5,
    "used_vpn": 1.2,
    "is_foreign_transaction": 1.0,
    "is_new_merchant": 1.0,
}
GROUP_KEYS = [
    "auth_method",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
Z_COLS = ["velocity_score", "merchant_risk_score", "distance_from_home_km", "hours_since_last_txn"]


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def embed_frame(df, scaler=None, fit=False):
    Xc = df[CONT].astype(np.float64).values
    Xf = df[FLAGS].astype(np.float64).values
    fw = np.array([FLAG_W[c] for c in FLAGS], dtype=np.float64)
    Xf = Xf * fw
    if fit:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(Xc)
    else:
        Xc = scaler.transform(Xc)
    return np.hstack([Xc, Xf]).astype(np.float32), scaler


def ensure_cdist(train, test, device="cuda"):
    F_tr, F_te = load_orig20k_cdist(ROOT)
    if F_tr is not None:
        print(f"loaded orig20k cdist F_tr={F_tr.shape} F_te={F_te.shape}", flush=True)
        return F_tr, F_te
    print("computing orig20k cdist (stage9 cache miss)...", flush=True)
    import torch
    from stage9_2_orig20k_orthogonal_residual import cdist_features

    orig_path = os.path.join(DATA, "original.csv")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(ROOT, "original", "credit_card_fraud_2026.csv")
    orig = pd.read_csv(orig_path)
    y_orig = orig[TARGET].astype(int).values
    X_tr, scaler = embed_frame(train, fit=True)
    X_te, _ = embed_frame(test, scaler=scaler, fit=False)
    X_or, _ = embed_frame(orig, scaler=scaler, fit=False)
    F_tr = cdist_features(X_tr, X_or, y_orig, device, batch=2048)
    F_te = cdist_features(X_te, X_or, y_orig, device, batch=2048)
    os.makedirs(PRED9, exist_ok=True)
    np.save(os.path.join(PRED9, "orig20k_cdist_train.npy"), F_tr)
    np.save(os.path.join(PRED9, "orig20k_cdist_test.npy"), F_te)
    return F_tr, F_te


def group_z_fold(train, test, tr_idx):
    """Nested group z on train + test using fold-train stats (matches stage9.2)."""
    from stage9_2_orig20k_orthogonal_residual import group_zscores

    z_all = group_zscores(train, tr_idx)
    train_fold = train.iloc[tr_idx]
    keys_tr = train_fold[GROUP_KEYS].astype(str).agg("|".join, axis=1)
    g = train_fold.groupby(keys_tr.values)[Z_COLS]
    mu = g.mean()
    sd = g.std().fillna(1.0) + 1e-6
    cnt = g.size()
    keep = cnt >= 30
    mu, sd = mu[keep], sd[keep]
    gmu = train_fold[Z_COLS].mean().values
    gsd = train_fold[Z_COLS].std().values + 1e-6
    keys_te = test[GROUP_KEYS].astype(str).agg("|".join, axis=1).values
    raw_te = test[Z_COLS].astype(np.float64).values
    z_te = np.zeros((len(test), len(Z_COLS) + 1), np.float64)
    uniq, inv = np.unique(keys_te, return_inverse=True)
    mu_map = {k: mu.loc[k].values for k in mu.index}
    sd_map = {k: sd.loc[k].values for k in sd.index}
    for i, k in enumerate(uniq):
        mask = inv == i
        muv = mu_map.get(k, gmu)
        sdv = sd_map.get(k, gsd)
        zz = (raw_te[mask] - muv) / sdv
        z_te[mask, :-1] = zz
        z_te[mask, -1] = np.sqrt((zz ** 2).sum(1))
    return z_all.astype(np.float32), z_te.astype(np.float32)


def fold_orig_block(F_tr, F_te, z_tr_all, z_te, tr, va):
    otr = np.hstack([F_tr[tr], z_tr_all[tr]])
    ova = np.hstack([F_tr[va], z_tr_all[va]])
    ote = np.hstack([F_te, z_te])
    return (
        np.nan_to_num(otr).astype(np.float32),
        np.nan_to_num(ova).astype(np.float32),
        np.nan_to_num(ote).astype(np.float32),
    )


def train_lgb_bag(X_tr, y_tr, X_va, y_va, X_te, seeds, leaves_list):
    """50/50 blend of leaves configs × seed bag average."""
    va_sum = np.zeros(len(X_va), np.float64)
    te_sum = np.zeros(len(X_te), np.float64)
    n = 0
    for leaves in leaves_list:
        for s in seeds:
            m = lgb.LGBMClassifier(
                learning_rate=0.03,
                num_leaves=leaves,
                max_depth=3 if leaves >= 4 else 2,
                min_child_samples=100,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                reg_lambda=25.0,
                n_estimators=5000,
                n_jobs=-1,
                verbose=-1,
                random_state=s,
            )
            m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            va_sum += m.predict_proba(X_va)[:, 1]
            te_sum += m.predict_proba(X_te)[:, 1]
            n += 1
    return va_sum / n, te_sum / n


def train_xgb_bag(X_tr, y_tr, X_va, y_va, X_te, seeds, leaves_list):
    va_sum = np.zeros(len(X_va), np.float64)
    te_sum = np.zeros(len(X_te), np.float64)
    n = 0
    for leaves in leaves_list:
        for s in seeds:
            params = dict(
                learning_rate=0.03,
                max_depth=0,
                grow_policy="lossguide",
                max_leaves=leaves,
                max_bin=256,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.45,
                reg_lambda=30.0,
                tree_method="hist",
                device="cuda",
                eval_metric="auc",
                objective="binary:logistic",
                seed=s,
            )
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)
            dte = xgb.DMatrix(X_te)
            booster = xgb.train(
                params,
                dtr,
                num_boost_round=5000,
                evals=[(dva, "va")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            it = booster.best_iteration + 1
            va_sum += booster.predict(dva, iteration_range=(0, it))
            te_sum += booster.predict(dte, iteration_range=(0, it))
            n += 1
    return va_sum / n, te_sum / n


def train_cat_d2(X_tr, y_tr, X_va, y_va, X_te):
    from catboost import CatBoostClassifier, Pool

    model = CatBoostClassifier(
        iterations=6000,
        depth=2,
        learning_rate=0.04,
        l2_leaf_reg=25,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=200,
    )
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True)
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]


def load_ctrl_diglead_oof(y):
    """Best-effort OOF for gate baseline from bank npz."""
    pre = os.path.join(PRED9, "full_bank_unpruned.npz")
    if not os.path.exists(pre):
        return None, CTRL_DIGLEAD
    z = np.load(pre, allow_pickle=True)
    names = list(z["names"])
    O = z["O"]
    hits = [i for i, n in enumerate(names) if "lgb_diglead_l3_bag" in str(n)]
    if not hits:
        return None, CTRL_DIGLEAD
    o = O[:, hits[0]].astype(np.float64)
    return o, float(roc_auc_score(y, o))


def main():
    t0 = time.time()
    os.makedirs(PRED10, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = locked_folds(y)
    print(f"train={len(train)} test={len(test)} fraud={y.mean():.4%}", flush=True)

    print("building diglead codes...", flush=True)
    dl_tr, dl_te, dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = build_diglead_codes(
        train, test
    )
    print(f"  diglead keys={dl_tr.shape[1]} dig_num={len(dig_num)} te_cols={len(te_names)}", flush=True)

    F_tr, F_te = ensure_cdist(train, test, device="cuda")
    ctrl_oof, ctrl_auc = load_ctrl_diglead_oof(y)
    print(f"gate baseline lgb_diglead_l3_bag OOF={ctrl_auc:.6f}", flush=True)
    ctrl_folds = (
        [float(roc_auc_score(y[va], ctrl_oof[va])) for _, va in folds]
        if ctrl_oof is not None
        else [ctrl_auc] * 5
    )

    arms = {
        "lgb_diglead_l3_smooth50_orig_bag": {
            "oof": np.zeros(len(y)),
            "test": np.zeros(len(test)),
            "folds": [],
            "kind": "lgb",
        },
        "xgb_cuda_diglead_l3_smooth50_orig_bag": {
            "oof": np.zeros(len(y)),
            "test": np.zeros(len(test)),
            "folds": [],
            "kind": "xgb",
        },
        "cat_cuda_diglead_d2_smooth50_orig": {
            "oof": np.zeros(len(y)),
            "test": np.zeros(len(test)),
            "folds": [],
            "kind": "cat",
        },
    }

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        z_tr_all, z_te = group_z_fold(train, test, tr)
        otr, ova, ote = fold_orig_block(F_tr, F_te, z_tr_all, z_te, tr, va)
        # pass full-train orig by reconstructing: cdist full + z full
        orig_full = np.hstack([F_tr, z_tr_all]).astype(np.float32)
        orig_full = np.nan_to_num(orig_full)

        X_tr, X_va, X_te = build_fold_diglead_matrix(
            train,
            test,
            dl_tr,
            dl_te,
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
            orig_tr=orig_full,
            orig_te=ote,
            orig_va=ova,
        )
        print(f"  X_tr={X_tr.shape} X_va={X_va.shape} X_te={X_te.shape}", flush=True)
        Xtr, Xva, Xte = X_tr.values, X_va.values, X_te.values

        # 10A
        pva, pte = train_lgb_bag(Xtr, y[tr], Xva, y[va], Xte, BAG_SEEDS, [3, 4])
        arms["lgb_diglead_l3_smooth50_orig_bag"]["oof"][va] = pva
        arms["lgb_diglead_l3_smooth50_orig_bag"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        arms["lgb_diglead_l3_smooth50_orig_bag"]["folds"].append(fa)
        print(f"  10A lgb_bag va={fa:.6f}", flush=True)

        # 10B
        pva, pte = train_xgb_bag(Xtr, y[tr], Xva, y[va], Xte, BAG_SEEDS, [3, 4])
        arms["xgb_cuda_diglead_l3_smooth50_orig_bag"]["oof"][va] = pva
        arms["xgb_cuda_diglead_l3_smooth50_orig_bag"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        arms["xgb_cuda_diglead_l3_smooth50_orig_bag"]["folds"].append(fa)
        print(f"  10B xgb_bag va={fa:.6f}", flush=True)

        # 10C
        pva, pte = train_cat_d2(Xtr, y[tr], Xva, y[va], Xte)
        arms["cat_cuda_diglead_d2_smooth50_orig"]["oof"][va] = pva
        arms["cat_cuda_diglead_d2_smooth50_orig"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        arms["cat_cuda_diglead_d2_smooth50_orig"]["folds"].append(fa)
        print(f"  10C cat_d2 va={fa:.6f}", flush=True)

    results = {}
    for name, s in arms.items():
        s["test"] /= 5.0
        s["oof_auc"] = float(roc_auc_score(y, s["oof"]))
        gate = promotion_gate(ctrl_folds, s["folds"], ctrl_auc, s["oof_auc"])
        # also soft gate vs absolute 0.95063
        beat_abs = s["oof_auc"] > CTRL_DIGLEAD
        results[name] = {
            "oof_auc": s["oof_auc"],
            "fold_aucs": s["folds"],
            "delta_vs_diglead_l3_bag": s["oof_auc"] - ctrl_auc,
            "beat_0.95063": beat_abs,
            "gate_vs_diglead": gate,
            "inject": True,  # always inject into hillclimb; promote flag separate
            "promote_base": gate["promote"] or beat_abs,
        }
        np.save(os.path.join(PRED10, f"{name}_oof.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(PRED10, f"{name}_test.npy"), s["test"].astype(np.float32))
        print(
            f"{name}: OOF={s['oof_auc']:.6f} Δ={s['oof_auc']-ctrl_auc:+.6f} "
            f"beat_abs={beat_abs} {gate['reason']}",
            flush=True,
        )
        sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
        from scipy.stats import rankdata

        sub["is_fraud"] = rankdata(s["test"]) / len(test)
        path = os.path.join(SUBS, f"stage10_{name}_{s['oof_auc']:.5f}.csv")
        sub.to_csv(path, index=False)
        results[name]["submission"] = path

    doc = {
        "stage": "10_diglead_superbase",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "ctrl_diglead_l3_bag": ctrl_auc,
        "bag_seeds": BAG_SEEDS,
        "smooth": 50.0,
        "orig20k": True,
        "arms": results,
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
        "next": "stage10_reinject_hillclimb",
    }
    out = os.path.join(LOGS, "stage10_diglead_superbase_result.json")
    json.dump(doc, open(out, "w"), indent=2)
    print(f"wrote {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
