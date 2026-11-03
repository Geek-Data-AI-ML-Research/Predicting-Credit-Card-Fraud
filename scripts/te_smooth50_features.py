"""Shared leak-free TE smooth=50 (+ freq) feature builder for GPU stages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_stage3_baselines import (
    CATS,
    FLAGS,
    NUMS,
    SEED,
    digit_frame,
    oof_target_encode,
)


def te_apply(codes_fit, y_fit, codes_apply, smooth: float) -> np.ndarray:
    prior = float(y_fit.mean())
    n_cols = codes_fit.shape[1]
    out = np.empty((len(codes_apply), n_cols), np.float32)
    for j in range(n_cols):
        kf = codes_fit[:, j]
        ka = codes_apply[:, j]
        nmax = int(max(int(kf.max()), int(ka.max())) + 1)
        s = np.bincount(kf, weights=y_fit, minlength=nmax)
        n = np.bincount(kf, minlength=nmax)
        out[:, j] = (s[ka] + prior * smooth) / (n[ka] + smooth)
    return out


def build_codes(train: pd.DataFrame, test: pd.DataFrame):
    dig_tr = digit_frame(train)
    dig_te = digit_frame(test)
    dig_num = [c for c in dig_tr.columns if dig_tr[c].dtype != object]
    dig_cat = [c for c in dig_tr.columns if dig_tr[c].dtype == object]
    te_tr_src = pd.concat([train[CATS].astype(str), dig_tr[dig_cat].astype(str)], axis=1)
    te_te_src = pd.concat([test[CATS].astype(str), dig_te[dig_cat].astype(str)], axis=1)
    te_names = list(te_tr_src.columns)
    codes_tr = np.column_stack(
        [
            pd.Categorical(
                te_tr_src[c],
                categories=sorted(pd.concat([te_tr_src[c], te_te_src[c]]).unique()),
            ).codes.astype(np.int32)
            for c in te_names
        ]
    )
    codes_te = np.column_stack(
        [
            pd.Categorical(
                te_te_src[c],
                categories=sorted(pd.concat([te_tr_src[c], te_te_src[c]]).unique()),
            ).codes.astype(np.int32)
            for c in te_names
        ]
    )
    return dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te


def _as_float(*parts: pd.DataFrame) -> pd.DataFrame:
    X = pd.concat(parts, axis=1)
    # drop duplicate colnames keeping first
    X = X.loc[:, ~X.columns.duplicated()]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # pytabkit RealMLP/TabM reject NaNs in continuous columns
    return X.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def freq_cols(train, tr_idx, apply_df, cats=CATS):
    cols = {}
    for c in cats:
        vc = train.iloc[tr_idx][c].astype(str).value_counts()
        cols[f"freq_{c}"] = apply_df[c].astype(str).map(vc).fillna(0).astype(np.float32).values
    return pd.DataFrame(cols)


def risk_ratios(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ratio_vel_hours": (
                df["velocity_score"].astype(float) / (df["hours_since_last_txn"].astype(float) + 0.1)
            ).astype(np.float32),
            "risk_x_cvv": (
                df["merchant_risk_score"].astype(float) * (df["cvv_retry_count"].astype(float) + 1.0)
            ).astype(np.float32),
        }
    )


def build_fold_matrix(
    train,
    test,
    dig_tr,
    dig_te,
    dig_num,
    codes_tr,
    codes_te,
    te_names,
    y,
    tr_idx,
    va_idx,
    *,
    smooth: float = 50.0,
    extra_smooth: float | None = None,
    with_ratios: bool = False,
):
    """Return X_tr, X_va, X_te float frames (leak-free TE from tr_idx)."""
    te_tr, te_va = oof_target_encode(codes_tr, y, tr_idx, va_idx, smooth=smooth, seed=SEED)
    te_te = te_apply(codes_tr[tr_idx], y[tr_idx], codes_te, smooth)
    te_cols = [f"te{int(smooth)}_{c}" for c in te_names]

    parts_tr = [
        train.iloc[tr_idx][NUMS + FLAGS].reset_index(drop=True),
        dig_tr.iloc[tr_idx][dig_num].reset_index(drop=True),
        pd.DataFrame(te_tr, columns=te_cols),
        freq_cols(train, tr_idx, train.iloc[tr_idx]),
    ]
    parts_va = [
        train.iloc[va_idx][NUMS + FLAGS].reset_index(drop=True),
        dig_tr.iloc[va_idx][dig_num].reset_index(drop=True),
        pd.DataFrame(te_va, columns=te_cols),
        freq_cols(train, tr_idx, train.iloc[va_idx]),
    ]
    parts_te = [
        test[NUMS + FLAGS].reset_index(drop=True),
        dig_te[dig_num].reset_index(drop=True),
        pd.DataFrame(te_te, columns=te_cols),
        freq_cols(train, tr_idx, test),
    ]
    if extra_smooth is not None:
        te2_tr, te2_va = oof_target_encode(
            codes_tr, y, tr_idx, va_idx, smooth=extra_smooth, seed=SEED
        )
        te2_te = te_apply(codes_tr[tr_idx], y[tr_idx], codes_te, extra_smooth)
        c2 = [f"te{int(extra_smooth)}_{c}" for c in te_names]
        parts_tr.append(pd.DataFrame(te2_tr, columns=c2))
        parts_va.append(pd.DataFrame(te2_va, columns=c2))
        parts_te.append(pd.DataFrame(te2_te, columns=c2))
    if with_ratios:
        parts_tr.append(risk_ratios(train.iloc[tr_idx]).reset_index(drop=True))
        parts_va.append(risk_ratios(train.iloc[va_idx]).reset_index(drop=True))
        parts_te.append(risk_ratios(test).reset_index(drop=True))
    return _as_float(*parts_tr), _as_float(*parts_va), _as_float(*parts_te)
