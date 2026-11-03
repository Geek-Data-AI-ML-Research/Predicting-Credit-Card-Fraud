#!/usr/bin/env python3
"""Part-5 / lgb_diglead_l3_bag feature protocol (CV 0.950627).

Critical vs the broken 142-col cache:
  - Target-encode FEATS + cents + key_* (incl. leading digits).
  - DROP raw key_* columns before the tree fit (`drop_keys`).
  - Keep raw FEATS + cents_* + te_* only.
  - Default TE smooth=20 (bank diglead); smooth=50 is an optional upgrade.

Passing raw diglead integer codes into LightGBM with colsample_bytree=0.5
collapses fold0 to ~0.945–0.946 (Stage-3 digit_lgbm territory).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from validation_stage3_baselines import CATS, FLAGS, NUMS, SEED

FEATS = NUMS + FLAGS + CATS
FLOATS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]

# Cache schema version — bump when matrix layout changes (invalidates old npy).
CACHE_SCHEMA = "diglead_l3_part5_v2"


def digit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Part-5 digit_frame: cents + key_int/ten/dec (vectorized)."""
    out = {}
    for c in FLOATS:
        x = df[c].to_numpy(dtype=np.float64)
        out[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int16)
        out[f"key_int_{c}"] = np.floor(x).astype(np.int64)
        out[f"key_ten_{c}"] = np.floor(x / 10).astype(np.int64)
        out[f"key_dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
    return pd.DataFrame(out, index=df.index)


def lead_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Leading 1/2 digits packed with integer length (Part-5 lead_frame)."""
    out = {}
    for c in FLOATS:
        i = np.floor(df[c].to_numpy(dtype=np.float64)).astype(np.int64)
        s = pd.Series(i.astype(str), index=df.index)
        L = s.str.len().astype(np.int64)
        out[f"key_l1_{c}"] = (L * 10 + s.str[0].astype(int)).to_numpy()
        out[f"key_l2_{c}"] = (L * 100 + s.str[:2].astype(int)).to_numpy()
    return pd.DataFrame(out, index=df.index)


def drop_keys(D: pd.DataFrame) -> pd.DataFrame:
    """Remove raw high-cardinality key_* columns after TE (Part-5)."""
    return D[[c for c in D.columns if not str(c).startswith("key_")]]


def add_te(Xa, ya, others, cols, smooth=20.0, inner=5, seed=0):
    """Part-5 nested TE: append te_<col> for each col; leave raw cols in place."""
    prior = float(ya.mean())
    Xa = Xa.copy()
    others = [o.copy() for o in others]

    def key(X, c):
        if str(X[c].dtype) == "category":
            return X[c].astype(str).values
        return X[c].values

    def rate(k, t):
        s = pd.DataFrame({"k": k, "y": t}).groupby("k")["y"].agg(["sum", "count"])
        return (s["sum"] + prior * smooth) / (s["count"] + smooth)

    for c in cols:
        ka = key(Xa, c)
        enc = np.zeros(len(Xa), dtype=np.float32)
        for i, j in StratifiedKFold(inner, shuffle=True, random_state=seed).split(ka, ya):
            enc[j] = pd.Series(ka[j]).map(rate(ka[i], ya[i])).fillna(prior).values
        Xa[f"te_{c}"] = enc
        r = rate(ka, ya)
        for o in others:
            o[f"te_{c}"] = pd.Series(key(o, c)).map(r).fillna(prior).values.astype(np.float32)
    return Xa, others


def build_base_frames(train: pd.DataFrame, test: pd.DataFrame):
    """Xtr/Xte = FEATS (+ cats as Categorical) + digit keys + leading keys."""
    allv = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
    full = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
    for c in CATS:
        full[c] = pd.Categorical(full[c], categories=sorted(allv[c].astype(str).unique()))
    Xtr = full.iloc[: len(train)].reset_index(drop=True)
    Xte = full.iloc[len(train) :].reset_index(drop=True)

    DG = digit_frame(pd.concat([train[FLOATS], test[FLOATS]], ignore_index=True))
    LD = lead_frame(pd.concat([train[FLOATS], test[FLOATS]], ignore_index=True))
    DGtr, DGte = DG.iloc[: len(train)].reset_index(drop=True), DG.iloc[len(train) :].reset_index(drop=True)
    LDtr, LDte = LD.iloc[: len(train)].reset_index(drop=True), LD.iloc[len(train) :].reset_index(drop=True)

    XLtr = pd.concat([Xtr, DGtr, LDtr], axis=1)
    XLte = pd.concat([Xte, DGte, LDte], axis=1)
    CENTS = [c for c in DGtr.columns if c.startswith("cents_")]
    KEYS = [c for c in DGtr.columns if c.startswith("key_")]
    DEC = [c for c in KEYS if c.startswith("key_dec_")]
    LEAD_TE = FEATS + CENTS + [c for c in KEYS if c not in DEC] + list(LDtr.columns) + DEC
    return XLtr, XLte, LEAD_TE


def _as_float32(D: pd.DataFrame) -> np.ndarray:
    """Numeric matrix for .npy cache / non-pandas learners. Cats → codes."""
    out = D.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category" or out[c].dtype == object:
            out[c] = out[c].astype("category").cat.codes.astype(np.float32)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).values


def build_fold_diglead_matrix(
    XLtr: pd.DataFrame,
    XLte: pd.DataFrame,
    lead_te: list[str],
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    smooth: float = 20.0,
    teseed: int = 0,
    orig_tr: np.ndarray | None = None,
    orig_va: np.ndarray | None = None,
    orig_te: np.ndarray | None = None,
    return_frame: bool = False,
):
    """Leak-free Part-5 diglead fold matrices (+ optional orig20k/group_z floats)."""
    Xa = XLtr.iloc[tr_idx].reset_index(drop=True)
    Xb = XLtr.iloc[va_idx].reset_index(drop=True)
    Xt = XLte.reset_index(drop=True)
    Xa, (Xb, Xt) = add_te(Xa, y[tr_idx], [Xb, Xt], lead_te, smooth=smooth, seed=teseed)
    Xa, Xb, Xt = drop_keys(Xa), drop_keys(Xb), drop_keys(Xt)

    if orig_tr is not None and orig_te is not None:
        n_o = orig_tr.shape[1]
        oc = [f"orig_{i}" for i in range(n_o)]
        ova = orig_va if orig_va is not None else orig_tr
        Xa = pd.concat([Xa, pd.DataFrame(orig_tr, columns=oc)], axis=1)
        Xb = pd.concat([Xb, pd.DataFrame(ova, columns=oc)], axis=1)
        Xt = pd.concat([Xt, pd.DataFrame(orig_te, columns=oc)], axis=1)

    if return_frame:
        return Xa, Xb, Xt
    return _as_float32(Xa), _as_float32(Xb), _as_float32(Xt)


# ---- backward-compat stubs used by older call sites ----
def diglead_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Deprecated alias — returns Part-5 digit+lead keys (includes key_ prefix)."""
    return pd.concat([digit_frame(frame), lead_frame(frame)], axis=1)


def build_diglead_codes(train: pd.DataFrame, test: pd.DataFrame):
    """Build Part-5 XLtr/XLte + LEAD_TE list (replaces old codes path)."""
    XLtr, XLte, lead_te = build_base_frames(train, test)
    return XLtr, XLte, lead_te


def load_orig20k_cdist(root: str):
    pred9 = os.path.join(root, "preds", "stage9")
    tr = os.path.join(pred9, "orig20k_cdist_train.npy")
    te = os.path.join(pred9, "orig20k_cdist_test.npy")
    if os.path.exists(tr) and os.path.exists(te):
        return np.load(tr), np.load(te)
    return None, None
