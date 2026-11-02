#!/usr/bin/env python3
"""S6E11 credit-card fraud ensemble.

Self-trained digit-key + target-encoding members (adapted from public
0.95154 notebook patterns) stacked with public RealMLP/EBM/LGB artifacts.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ORIG = os.path.join(ROOT, "original", "credit_card_fraud_2026.csv")
ART = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "submissions")
PRED = os.path.join(ROOT, "preds")

SEED, N_FOLDS, TARGET = 42, 5, "is_fraud"
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

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
FEATS = NUMS + FLAGS + CATS
DEC = {
    "amount_usd": 2,
    "account_balance_usd": 2,
    "hours_since_last_txn": 2,
    "distance_from_home_km": 2,
    "velocity_score": 1,
    "merchant_risk_score": 1,
}
FLOATS = list(DEC)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def key_frame(D: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c, k in DEC.items():
        s = D[c].map(lambda v, kk=k: f"{v:.{kk}f}")
        ip, fp = s.str.split(".").str[0], s.str.split(".").str[1]
        L = ip.str.len().astype(str)
        out[f"int_{c}"], out[f"frac_{c}"] = ip, fp
        out[f"ten_{c}"] = np.floor(D[c] / 10).astype("int64").astype(str)
        out[f"dec_{c}"] = np.floor(D[c] * 10 + 1e-6).astype("int64").astype(str)
        out[f"l1_{c}"], out[f"l2_{c}"] = L + "|" + ip.str[:1], L + "|" + ip.str[:2]
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-self", action="store_true", help="only stack public artifacts")
    ap.add_argument("--skip-glm", action="store_true")
    ap.add_argument("--skip-xgb", action="store_true")
    ap.add_argument("--max-trees", type=int, default=30000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PRED, exist_ok=True)

    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    og = pd.read_csv(ORIG).drop(columns="transaction_id")
    if args.smoke:
        tr, te = tr.iloc[:40000].copy(), te.iloc[:10000].copy()

    for d in (tr, te, og):
        d[FLAGS] = d[FLAGS].astype("int8")
    y = tr[TARGET].astype(int).values
    for c in CATS:
        cats = sorted(pd.concat([tr[c], te[c], og[c]]).unique())
        for d in (tr, te):
            d[c] = pd.Categorical(d[c], categories=cats)
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    NTR = len(tr)
    FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    OOF: dict[str, np.ndarray] = {}
    FULL: dict[str, np.ndarray] = {}

    print(
        f"train {tr.shape} | test {te.shape} | fraud {y.mean():.4%} | "
        f"cpus {N_JOBS} | smoke={args.smoke}",
        flush=True,
    )

    # ---------- public artifacts ----------
    for f in sorted(glob.glob(os.path.join(ART, "*_oof.npy"))):
        name = os.path.basename(f).replace("_oof.npy", "")
        oof = np.load(f)
        if args.smoke or len(oof) != NTR:
            continue
        test_path = next(
            (
                p
                for p in (
                    os.path.join(ART, f"{name}_full.npy"),
                    os.path.join(ART, f"{name}_test.npy"),
                )
                if os.path.exists(p)
            ),
            None,
        )
        if test_path is None:
            continue
        test = np.load(test_path)
        if len(test) != len(te):
            continue
        OOF[f"art_{name}"] = oof.astype(np.float64)
        FULL[f"art_{name}"] = test.astype(np.float64)
        print(f"artifact {name}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)

    if args.skip_self:
        _stack_and_save(y, te, OOF, FULL, "artifact_only")
        return

    # ---------- digit keys + encodings ----------
    KF = key_frame(full)
    KEYS = list(KF.columns)
    ENC_COLS = FEATS + KEYS
    CODES = np.column_stack(
        [
            pd.factorize(full[c].astype(str) if c in CATS else full[c])[0]
            for c in FEATS
        ]
        + [pd.factorize(KF[k])[0] for k in KEYS]
    ).astype(np.int32)
    NCODE = CODES.max(0) + 1

    NUM = full.copy()
    for c in FLOATS:
        vc = pd.concat([full[c], og[c]]).value_counts()
        NUM[f"cnt_{c}"] = full[c].map(vc).values.astype(np.float32)
        k = np.round(full[c].values * 10 ** DEC[c]).astype(np.int64)
        NUM[f"frac_{c}"] = (k % 10 ** DEC[c]).astype(np.int16)
        NUM[f"d1_{c}"] = ((k // 10 ** (DEC[c] - 1)) % 10).astype(np.int8)
        if DEC[c] == 2:
            NUM[f"d2_{c}"] = (k % 10).astype(np.int8)
    print(f"{len(KEYS)} keys, {NUM.shape[1]} numeric columns", flush=True)

    TE_NAMES = ["te_" + c for c in ENC_COLS]
    TEST = np.arange(NTR, NTR + len(te))
    _TE: dict = {}

    def te_block(rows_fit, rows_apply, smooth, prior):
        out = np.empty((len(rows_apply), len(ENC_COLS)), np.float32)
        yf = y[rows_fit]
        for j in range(len(ENC_COLS)):
            kf, ka = CODES[rows_fit, j], CODES[rows_apply, j]
            s = np.bincount(kf, weights=yf, minlength=NCODE[j])
            n = np.bincount(kf, minlength=NCODE[j])
            out[:, j] = (s[ka] + prior * smooth) / (n[ka] + smooth)
        return out

    def encode(train_idx, apply_idx, smooth=20.0, seed=0):
        prior = y[train_idx].mean()
        inner = np.empty((len(train_idx), len(ENC_COLS)), np.float32)
        for i, j in StratifiedKFold(5, shuffle=True, random_state=seed).split(
            train_idx, y[train_idx]
        ):
            inner[j] = te_block(train_idx[i], train_idx[j], smooth, prior)
        return inner, te_block(train_idx, apply_idx, smooth, prior)

    def encoded(f, smooth=20.0, seed=0):
        key = (f, smooth, seed)
        if key not in _TE:
            a, b = (np.arange(NTR), None) if f == "full" else FOLDS[f]
            ea, eo = encode(
                a, TEST if b is None else np.concatenate([b, TEST]), smooth, seed
            )
            _TE[key] = (ea, None, eo) if b is None else (ea, eo[: len(b)], eo[len(b) :])
        return _TE[key]

    def clear_te():
        _TE.clear()
        gc.collect()

    LR = 0.03
    LGB = dict(
        learning_rate=LR,
        num_leaves=4,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    XGB = dict(
        learning_rate=LR,
        max_depth=2,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=N_JOBS,
        enable_categorical=True,
        max_cat_to_onehot=1,
    )
    NUM_TR, NUM_TE = NUM.iloc[:NTR].reset_index(drop=True), NUM.iloc[NTR:].reset_index(
        drop=True
    )

    def frame(rows, enc, use_te):
        D = NUM_TR.iloc[rows] if rows is not None else NUM_TE
        D = D.reset_index(drop=True)
        return pd.concat([D, pd.DataFrame(enc, columns=TE_NAMES)], axis=1) if use_te else D

    def fit(kind, n, seed, Xa, ya, Xb=None, yb=None, **over):
        if kind == "lgb":
            m = lgb.LGBMClassifier(**{**LGB, **over}, n_estimators=n, random_state=seed)
            if Xb is None:
                return m.fit(Xa, ya), n
            m.fit(
                Xa,
                ya,
                eval_set=[(Xb, yb)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(300, verbose=False)],
            )
            return m, m.best_iteration_
        m = xgb.XGBClassifier(
            **{**XGB, **over},
            n_estimators=n,
            random_state=seed,
            **({} if Xb is None else dict(early_stopping_rounds=300)),
        )
        if Xb is None:
            return m.fit(Xa, ya, verbose=False), n
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m, m.best_iteration + 1

    def run_tree(name, kind, smooth=20.0, seed=SEED, teseed=0, use_te=True, **over):
        oof, its, t0 = np.zeros(NTR), [], time.time()
        for f, (a, b) in enumerate(FOLDS):
            ea, eb, _ = encoded(f, smooth, teseed) if use_te else (None, None, None)
            m, it = fit(
                kind,
                args.max_trees,
                seed,
                frame(a, ea, use_te),
                y[a],
                frame(b, eb, use_te),
                y[b],
                **over,
            )
            oof[b] = m.predict_proba(frame(b, eb, use_te))[:, 1]
            its.append(it)
            print(
                f"{name} fold {f}: AUC {roc_auc_score(y[b], oof[b]):.5f}, "
                f"{it} trees ({time.time() - t0:.0f}s)",
                flush=True,
            )
        n = int(round(np.mean(its) * 1.25))
        ea, _, et = encoded("full", smooth, teseed) if use_te else (None, None, None)
        m, _ = fit(kind, n, seed, frame(np.arange(NTR), ea, use_te), y, **over)
        OOF[name], FULL[name] = oof, m.predict_proba(frame(None, et, use_te))[:, 1]
        print(
            f"==> {name}: CV AUC {roc_auc_score(y, oof):.5f} | "
            f"refit {n} trees ({time.time() - t0:.0f}s)\n",
            flush=True,
        )
        np.save(os.path.join(PRED, f"{name}_oof.npy"), oof.astype(np.float32))
        np.save(os.path.join(PRED, f"{name}_full.npy"), FULL[name].astype(np.float32))

    def bag(name, parts):
        OOF[name] = np.mean([OOF.pop(p) for p in parts], axis=0)
        FULL[name] = np.mean([FULL.pop(p) for p in parts], axis=0)
        print(f"==> {name} ({len(parts)} seeds): CV AUC {roc_auc_score(y, OOF[name]):.5f}\n", flush=True)

    # core members
    run_tree("lgb_k_0", "lgb")
    run_tree("lgb_k_l7", "lgb", num_leaves=7)
    if not args.skip_xgb:
        run_tree("xgb_k", "xgb")

    if not args.skip_glm:
        num = full[NUMS].astype(float).copy()
        for c in ("amount_usd", "account_balance_usd", "hours_since_last_txn"):
            num[c] = np.log1p(num[c])
        Z = StandardScaler().fit_transform(num)
        ONEHOT = sparse.hstack(
            [
                sparse.csr_matrix(Z),
                sparse.csr_matrix(full[FLAGS].values.astype(float)),
                OneHotEncoder(handle_unknown="ignore").fit_transform(full[CATS].astype(str)),
                sparse.csr_matrix(
                    SplineTransformer(n_knots=8, degree=3, knots="quantile").fit_transform(Z)
                ),
                OneHotEncoder(handle_unknown="ignore", min_frequency=5).fit_transform(full[NUMS]),
                OneHotEncoder(handle_unknown="ignore", min_frequency=5).fit_transform(KF),
            ]
        ).tocsr()
        OH_TR, OH_TE = ONEHOT[:NTR], ONEHOT[NTR:]

        def glm_design(Ms, Es):
            sc = StandardScaler().fit(logit(Es[0]))
            return [
                sparse.hstack([M, sparse.csr_matrix(sc.transform(logit(E)))]).tocsr()
                for M, E in zip(Ms, Es)
            ]

        glm = lambda: LogisticRegression(C=0.01, max_iter=4000, tol=1e-6)
        oof, t0 = np.zeros(NTR), time.time()
        for f, (a, b) in enumerate(FOLDS):
            ea, eb, _ = encoded(f)
            Da, Db = glm_design([OH_TR[a], OH_TR[b]], [ea, eb])
            oof[b] = glm().fit(Da, y[a]).predict_proba(Db)[:, 1]
            print(f"glm_joint fold {f}: AUC {roc_auc_score(y[b], oof[b]):.5f} ({time.time() - t0:.0f}s)", flush=True)
        ea, _, et = encoded("full")
        Da, Dt = glm_design([OH_TR, OH_TE], [ea, et])
        OOF["glm_joint"], FULL["glm_joint"] = oof, glm().fit(Da, y).predict_proba(Dt)[:, 1]
        print(f"==> glm_joint {Da.shape}: CV AUC {roc_auc_score(y, oof):.5f} ({time.time() - t0:.0f}s)\n", flush=True)
        np.save(os.path.join(PRED, "glm_joint_oof.npy"), oof.astype(np.float32))
        np.save(os.path.join(PRED, "glm_joint_full.npy"), FULL["glm_joint"].astype(np.float32))

    clear_te()
    run_tree("lgb_k_1", "lgb", seed=1, teseed=1)
    clear_te()
    run_tree("lgb_k_2", "lgb", seed=2, teseed=2)
    clear_te()
    bag("lgb_k", ["lgb_k_0", "lgb_k_1", "lgb_k_2"])
    run_tree("lgb_k_s100_0", "lgb", smooth=100.0)
    clear_te()
    run_tree("lgb_k_s100_1", "lgb", smooth=100.0, seed=1, teseed=1)
    clear_te()
    bag("lgb_k_s100", ["lgb_k_s100_0", "lgb_k_s100_1"])
    run_tree("lgb_raw", "lgb", use_te=False)

    _stack_and_save(y, te, OOF, FULL, "self_plus_artifacts")


def _stack_and_save(y, te, OOF, FULL, tag: str) -> None:
    NAMES = list(OOF)
    Lo = np.column_stack([logit(OOF[n]) for n in NAMES])
    Lf = np.column_stack([logit(FULL[n]) for n in NAMES])
    st = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Lo, y):
        st[b] = LogisticRegression(C=0.01, max_iter=3000).fit(Lo[a], y[a]).decision_function(Lo[b])
    stack = LogisticRegression(C=0.01, max_iter=3000).fit(Lo, y)
    print("\n=== stack summary ===", flush=True)
    for n, w in zip(NAMES, stack.coef_[0]):
        print(f"  {n:30s} auc={roc_auc_score(y, OOF[n]):.6f}  w={w:+.4f}", flush=True)
    cv = roc_auc_score(y, st)
    print(f"STACK CV AUC {cv:.6f}", flush=True)

    pred = expit(stack.decision_function(Lf))
    sub = pd.DataFrame({"id": te["id"], TARGET: pred})
    path = os.path.join(OUT, f"{tag}_{cv:.5f}.csv")
    sub.to_csv(path, index=False)
    latest = os.path.join(OUT, "latest.csv")
    sub.to_csv(latest, index=False)
    print(f"wrote {path}", flush=True)
    print(f"wrote {latest}", flush=True)


if __name__ == "__main__":
    main()
