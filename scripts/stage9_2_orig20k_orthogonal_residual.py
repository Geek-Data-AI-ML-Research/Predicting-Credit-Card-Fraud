#!/usr/bin/env python3
"""Stage 9.2: GPU torch.cdist Original-20k seed features + orthogonal residual on nested_hill.

ONLY new external columns — zero raw / te_smooth50. NO kaggle submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED7 = os.path.join(ROOT, "preds", "stage7")
PRED9 = os.path.join(ROOT, "preds", "stage9")
SUBS = os.path.join(ROOT, "submissions")
SEED = 42
EPS = 1e-6
CTRL = 0.951140

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
# MI-ish weights for risk flags in distance (higher = more important)
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


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def load_margins(y):
    op = os.path.join(PRED7, "campaign_best_oof_logit.npy")
    tp = os.path.join(PRED7, "campaign_best_test_logit.npy")
    if os.path.exists(op) and os.path.exists(tp):
        Lo, Lt = np.load(op).astype(np.float64), np.load(tp).astype(np.float64)
        p = expit(Lo)
        return Lo, Lt, float(roc_auc_score(y, p)), "campaign_best"
    op = os.path.join(PRED7, "ctrl_nested_hill_oof.npy")
    if os.path.exists(op):
        o = np.load(op).astype(np.float64)
        Lo = to_logit((rankdata(o) / len(o)))
        # no test — use rank of o mean proxy later
        return Lo, None, float(roc_auc_score(y, o)), "ctrl_nested_hill"
    raise FileNotFoundError("missing campaign_best logits")


def embed_frame(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = False):
    Xc = df[CONT].astype(np.float64).values
    Xf = df[FLAGS].astype(np.float64).values
    # weight flags
    fw = np.array([FLAG_W[c] for c in FLAGS], dtype=np.float64)
    Xf = Xf * fw
    if fit:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(Xc)
    else:
        Xc = scaler.transform(Xc)
    X = np.hstack([Xc, Xf]).astype(np.float32)
    return X, scaler


@torch.no_grad()
def cdist_features(X_q: np.ndarray, X_orig: np.ndarray, y_orig: np.ndarray, device: str, batch: int = 4096):
    """Build 12 external distance/kernel features vs orig 20k."""
    xo = torch.tensor(X_orig, device=device)
    yo = torch.tensor(y_orig.astype(np.float32), device=device)
    fraud_idx = torch.where(yo > 0.5)[0]
    legit_idx = torch.where(yo < 0.5)[0]
    xo_f, xo_l = xo[fraud_idx], xo[legit_idx]
    n = X_q.shape[0]
    feats = np.zeros((n, 12), np.float32)
    taus = (0.5, 1.0, 2.0, 4.0)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        xq = torch.tensor(X_q[start:end], device=device)
        # distances to fraud
        df = torch.cdist(xq, xo_f)  # (b, n_fraud)
        dl = torch.cdist(xq, xo_l)
        min_f, _ = df.min(dim=1)
        min_l, _ = dl.min(dim=1)
        top5_f = torch.topk(df, k=min(5, df.shape[1]), largest=False).values.mean(1)
        top20_f = torch.topk(df, k=min(20, df.shape[1]), largest=False).values.mean(1)
        ratio = min_f / (min_l + 1e-6)
        # kernel soft fraud prob using all orig via fraud distances + labels on full set
        d_all = torch.cdist(xq, xo)
        row = []
        for t in taus:
            k = torch.exp(-(d_all ** 2) / (2 * t * t))
            p = (k * yo).sum(1) / (k.sum(1) + 1e-12)
            row.append(p)
        exact = (d_all < 0.25).float().sum(1)
        block = torch.stack(
            [min_f, top5_f, top20_f, min_l, ratio, row[0], row[1], row[2], row[3], exact],
            dim=1,
        )
        # pad to 12 with top5 legit mean + fraud-legit gap
        top5_l = torch.topk(dl, k=min(5, dl.shape[1]), largest=False).values.mean(1)
        gap = top5_l - top5_f
        block = torch.cat([block, top5_l[:, None], gap[:, None]], dim=1)
        feats[start:end] = block.cpu().numpy()
        if start % (batch * 10) == 0:
            print(f"  cdist {end}/{n}", flush=True)
    return feats


def group_zscores(df: pd.DataFrame, train_idx: np.ndarray):
    """OOF-safe vectorized group z: fit stats on train_idx, apply to all rows."""
    keys = df[GROUP_KEYS].astype(str).agg("|".join, axis=1)
    vals = df[Z_COLS].astype(np.float64)
    tr = df.iloc[train_idx]
    tr_keys = keys.iloc[train_idx]
    # group means/stds
    g = tr.groupby(tr_keys.values)[Z_COLS]
    mu = g.mean()
    sd = g.std().fillna(1.0) + 1e-6
    cnt = g.size()
    # drop tiny groups
    keep = cnt >= 30
    mu, sd = mu[keep], sd[keep]
    gmu = vals.iloc[train_idx].mean().values
    gsd = vals.iloc[train_idx].std().values + 1e-6
    out = np.zeros((len(df), len(Z_COLS)), np.float64)
    # map each key
    key_arr = keys.values
    unique, inv = np.unique(key_arr, return_inverse=True)
    mu_map = {k: mu.loc[k].values for k in mu.index}
    sd_map = {k: sd.loc[k].values for k in sd.index}
    raw = vals.values
    for i, k in enumerate(unique):
        mask = inv == i
        muv = mu_map.get(k, gmu)
        sdv = sd_map.get(k, gsd)
        out[mask] = (raw[mask] - muv) / sdv
    mahal = np.sqrt((out ** 2).sum(1, keepdims=True))
    return np.hstack([out, mahal]).astype(np.float32)


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    orig_path = os.path.join(DATA, "original.csv")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(ROOT, "original", "credit_card_fraud_2026.csv")
    orig = pd.read_csv(orig_path)
    y = train[TARGET].astype(int).values
    y_orig = orig[TARGET].astype(int).values
    print(f"orig n={len(orig)} fraud={int(y_orig.sum())}", flush=True)

    Lo, Lt, base_auc, base_name = load_margins(y)
    if Lt is None:
        Lt = np.full(len(test), Lo.mean())
    print(f"base_margin {base_name} auc={base_auc:.6f} mean_logit={Lo.mean():.3f}", flush=True)

    # embeddings
    X_tr, scaler = embed_frame(train, fit=True)
    X_te, _ = embed_frame(test, scaler=scaler, fit=False)
    X_or, _ = embed_frame(orig, scaler=scaler, fit=False)

    print("computing GPU cdist features train...", flush=True)
    F_tr = cdist_features(X_tr, X_or, y_orig, device, batch=2048)
    print("computing GPU cdist features test...", flush=True)
    F_te = cdist_features(X_te, X_or, y_orig, device, batch=2048)
    os.makedirs(PRED9, exist_ok=True)
    np.save(os.path.join(PRED9, "orig20k_cdist_train.npy"), F_tr)
    np.save(os.path.join(PRED9, "orig20k_cdist_test.npy"), F_te)
    print(f"saved orig20k cdist → {PRED9}/orig20k_cdist_*.npy", flush=True)

    # group z — fit on full train (nested inside folds below for residual)
    # We'll recompute nested in fold loop using train_mask=tr

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    arms = {
        "ortho_logit_offset": {"oof": np.zeros(len(y)), "test": np.zeros(len(test)), "folds": []},
        "ortho_xgb_residual": {"oof": np.zeros(len(y)), "test": np.zeros(len(test)), "folds": []},
    }

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        z_tr_all = group_zscores(train, tr)
        # test z with fold-train group stats
        train_fold = train.iloc[tr]
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
        z_te = z_te.astype(np.float32)

        Xtr = np.hstack([F_tr[tr], z_tr_all[tr]]).astype(np.float32)
        Xva = np.hstack([F_tr[va], z_tr_all[va]]).astype(np.float32)
        Xte = np.hstack([F_te, z_te]).astype(np.float32)
        Xtr = np.nan_to_num(Xtr)
        Xva = np.nan_to_num(Xva)
        Xte = np.nan_to_num(Xte)

        # Arm 1: logistic offset on base_margin
        # model: logit = base + X @ beta  via LR on features predicting residual in prob space
        # Fit LR on X predicting y with sample: use offset by stacking base as feature
        Xtr_off = np.column_stack([Lo[tr], Xtr])
        Xva_off = np.column_stack([Lo[va], Xva])
        Xte_off = np.column_stack([Lt, Xte])
        lr = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        lr.fit(Xtr_off, y[tr])
        pva = lr.predict_proba(Xva_off)[:, 1]
        pte = lr.predict_proba(Xte_off)[:, 1]
        arms["ortho_logit_offset"]["oof"][va] = pva
        arms["ortho_logit_offset"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        base_fa = float(roc_auc_score(y[va], expit(Lo[va])))
        arms["ortho_logit_offset"]["folds"].append(fa)
        print(f"  ortho_logit va={fa:.6f} base={base_fa:.6f} Δ={fa-base_fa:+.6f}", flush=True)

        # Arm 2: shallow CUDA XGB with base_margin
        dtr = xgb.DMatrix(Xtr, label=y[tr], base_margin=Lo[tr])
        dva = xgb.DMatrix(Xva, label=y[va], base_margin=Lo[va])
        dte = xgb.DMatrix(Xte, base_margin=Lt)
        params = dict(
            max_depth=2,
            eta=0.005,
            subsample=0.7,
            colsample_bytree=0.5,
            reg_lambda=200.0,
            reg_alpha=20.0,
            tree_method="hist",
            device="cuda",
            max_bin=256,
            eval_metric="auc",
            objective="binary:logistic",
        )
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=2000,
            evals=[(dva, "va")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        pva = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        pte = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
        arms["ortho_xgb_residual"]["oof"][va] = pva
        arms["ortho_xgb_residual"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        arms["ortho_xgb_residual"]["folds"].append(fa)
        print(
            f"  ortho_xgb va={fa:.6f} base={base_fa:.6f} Δ={fa-base_fa:+.6f} it={booster.best_iteration+1}",
            flush=True,
        )

    base_folds = [float(roc_auc_score(y[va], expit(Lo[va]))) for _, va in folds]
    os.makedirs(PRED9, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    results = {}
    for name, s in arms.items():
        s["test"] /= 5.0
        s["oof_auc"] = float(roc_auc_score(y, s["oof"]))
        gate = promotion_gate(base_folds, s["folds"], base_auc, s["oof_auc"])
        results[name] = {
            "oof_auc": s["oof_auc"],
            "fold_aucs": s["folds"],
            "gate_vs_base": gate,
            "promote": gate["promote"],
            "delta": s["oof_auc"] - base_auc,
        }
        np.save(os.path.join(PRED9, f"{name}_oof.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(PRED9, f"{name}_test.npy"), s["test"].astype(np.float32))
        print(f"{name}: OOF={s['oof_auc']:.6f} Δ={s['oof_auc']-base_auc:+.6f} {gate['reason']}", flush=True)
        if gate["promote"]:
            sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
            sub["is_fraud"] = rankdata(s["test"]) / len(test)
            path = os.path.join(SUBS, f"stage9_2_{name}_{s['oof_auc']:.5f}.csv")
            sub.to_csv(path, index=False)
            results[name]["submission"] = path

    doc = {
        "stage": "9.2_orig20k_orthogonal_residual",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "base": {"name": base_name, "oof": base_auc, "folds": base_folds},
        "n_orig": int(len(orig)),
        "n_orig_fraud": int(y_orig.sum()),
        "n_external_features": int(F_tr.shape[1] + 5),
        "arms": results,
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    path = os.path.join(LOGS, "stage9_2_orig20k_orthogonal_residual_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    agg = os.path.join(LOGS, "stage9_results.json")
    prev = json.load(open(agg)) if os.path.exists(agg) else {}
    prev["stage9_2"] = doc
    prev["updated_utc"] = datetime.now(timezone.utc).isoformat()
    json.dump(prev, open(agg, "w"), indent=2)
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
