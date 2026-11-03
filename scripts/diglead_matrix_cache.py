#!/usr/bin/env python3
"""Build/cache Part-5 diglead fold matrices for T01–T07.

Schema: diglead_l3_part5_v2 (TE keys + drop_keys + optional orig20k/group_z).
Old 142-col caches are rejected via CACHE_SCHEMA mismatch.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("S6E11_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diglead_features import (  # noqa: E402
    CACHE_SCHEMA,
    build_base_frames,
    build_fold_diglead_matrix,
    load_orig20k_cdist,
)
from unattended_common import ensure_dirs, locked_folds  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

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

# Default: Part-5 diglead smooth. Override with DIGLEAD_SMOOTH=50 for Stage-10 upgrade.
DEFAULT_SMOOTH = float(os.environ.get("DIGLEAD_SMOOTH", "20"))
# Append orig20k+group_z unless DIGLEAD_NO_ORIG=1
WITH_ORIG = os.environ.get("DIGLEAD_NO_ORIG", "0") != "1"


def embed_frame(df, scaler=None, fit=False):
    Xc = df[CONT].astype(np.float64).values
    Xf = df[FLAGS].astype(np.float64).values * np.array([FLAG_W[c] for c in FLAGS])
    if fit:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(Xc)
    else:
        Xc = scaler.transform(Xc)
    return np.hstack([Xc, Xf]).astype(np.float32), scaler


def ensure_cdist(train, test):
    F_tr, F_te = load_orig20k_cdist(ROOT)
    if F_tr is not None:
        return F_tr, F_te
    import torch
    from stage9_2_orig20k_orthogonal_residual import cdist_features

    data = os.path.join(ROOT, "data")
    orig_path = os.path.join(data, "original.csv")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(ROOT, "original", "credit_card_fraud_2026.csv")
    orig = pd.read_csv(orig_path)
    y_orig = orig[TARGET].astype(int).values
    X_tr, scaler = embed_frame(train, fit=True)
    X_te, _ = embed_frame(test, scaler=scaler, fit=False)
    X_or, _ = embed_frame(orig, scaler=scaler, fit=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"computing cdist on {device}...", flush=True)
    F_tr = cdist_features(X_tr, X_or, y_orig, device, batch=2048)
    F_te = cdist_features(X_te, X_or, y_orig, device, batch=2048)
    pred9 = os.path.join(ROOT, "preds", "stage9")
    os.makedirs(pred9, exist_ok=True)
    np.save(os.path.join(pred9, "orig20k_cdist_train.npy"), F_tr)
    np.save(os.path.join(pred9, "orig20k_cdist_test.npy"), F_te)
    return F_tr, F_te


def group_z_fold(train, test, tr_idx):
    from stage9_2_orig20k_orthogonal_residual import group_zscores

    z_all = group_zscores(train, tr_idx).astype(np.float32)
    train_fold = train.iloc[tr_idx]
    keys_tr = train_fold[GROUP_KEYS].astype(str).agg("|".join, axis=1)
    g = train_fold.groupby(keys_tr.values)[Z_COLS]
    mu, sd = g.mean(), g.std().fillna(1.0) + 1e-6
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
        zz = (raw_te[mask] - mu_map.get(k, gmu)) / sd_map.get(k, gsd)
        z_te[mask, :-1] = zz
        z_te[mask, -1] = np.sqrt((zz ** 2).sum(1))
    return z_all, z_te.astype(np.float32)


def _cache_dirs():
    return [
        os.environ.get("DIGLEAD_CACHE", "/workspace/cache_diglead_folds"),
        os.path.join(ROOT, "preds", "cache_diglead_folds"),
    ]


def _cache_ok(cache: str) -> bool:
    meta = os.path.join(cache, "meta.json")
    if not os.path.exists(meta):
        return False
    try:
        doc = json.load(open(meta))
    except Exception:
        return False
    return (
        doc.get("schema") == CACHE_SCHEMA
        and abs(float(doc.get("smooth", -1)) - DEFAULT_SMOOTH) < 1e-9
        and bool(doc.get("with_orig")) == WITH_ORIG
    )


class DigleadView:
    def __init__(self, train, test, y):
        self.train, self.test, self.y = train, test, y
        print(
            f"building diglead Part-5 frames (smooth={DEFAULT_SMOOTH}, orig={WITH_ORIG})...",
            flush=True,
        )
        self.XLtr, self.XLte, self.lead_te = build_base_frames(train, test)
        # compat aliases used by T04 interaction arm
        self.dl_tr = self.XLtr[[c for c in self.XLtr.columns if c.startswith(("cents_", "key_"))]]
        self.dl_te = self.XLte[[c for c in self.XLte.columns if c.startswith(("cents_", "key_"))]]
        self.F_tr = self.F_te = None
        if WITH_ORIG:
            self.F_tr, self.F_te = ensure_cdist(train, test)
        self.folds = locked_folds(y)
        self.col_names = None
        print(
            f"  XLtr={self.XLtr.shape} lead_te={len(self.lead_te)} cols",
            flush=True,
        )

    def fold_matrix(self, f, *, use_cache: bool = True):
        if use_cache:
            for cache in _cache_dirs():
                xt = os.path.join(cache, f"fold{f}_X_tr.npy")
                if os.path.exists(xt) and _cache_ok(cache):
                    tr = np.load(os.path.join(cache, f"fold{f}_tr_idx.npy"))
                    va = np.load(os.path.join(cache, f"fold{f}_va_idx.npy"))
                    Xtr = np.load(xt, mmap_mode="r").astype(np.float32, copy=True)
                    Xva = np.load(os.path.join(cache, f"fold{f}_X_va.npy"), mmap_mode="r").astype(
                        np.float32, copy=True
                    )
                    Xte = np.load(os.path.join(cache, f"fold{f}_X_te.npy"), mmap_mode="r").astype(
                        np.float32, copy=True
                    )
                    print(f"  [cache hit] fold{f} {cache} X_tr={Xtr.shape}", flush=True)
                    return tr, va, Xtr, Xva, Xte
                if os.path.exists(xt) and not _cache_ok(cache):
                    print(f"  [cache skip] stale schema in {cache}", flush=True)

        tr, va = self.folds[f]
        otr = ova = ote = None
        if WITH_ORIG and self.F_tr is not None:
            z_tr_all, z_te = group_z_fold(self.train, self.test, tr)
            otr = np.nan_to_num(np.hstack([self.F_tr[tr], z_tr_all[tr]])).astype(np.float32)
            ova = np.nan_to_num(np.hstack([self.F_tr[va], z_tr_all[va]])).astype(np.float32)
            ote = np.nan_to_num(np.hstack([self.F_te, z_te])).astype(np.float32)
        X_tr, X_va, X_te = build_fold_diglead_matrix(
            self.XLtr,
            self.XLte,
            self.lead_te,
            self.y,
            tr,
            va,
            smooth=DEFAULT_SMOOTH,
            teseed=0,
            orig_tr=otr,
            orig_va=ova,
            orig_te=ote,
        )
        return tr, va, X_tr, X_va, X_te


_VIEW = None


def get_view():
    global _VIEW
    if _VIEW is None:
        ensure_dirs()
        train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
        test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        y = train[TARGET].astype(int).values
        _VIEW = DigleadView(train, test, y)
    return _VIEW
