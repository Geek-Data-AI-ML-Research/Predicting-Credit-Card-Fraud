#!/usr/bin/env python3
"""
Grandmaster 5-Playbook Orthogonal Tournament on A40.
Phase 1: Fold-0 shootout for all 5 playbooks (~20 min)
Phase 2: Auto-promote Fold-0 winners (Δ≥1.5e-5) to 5-Fold + 3-seed 100% refit + super-stack
STRICT NO_SUBMIT.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["NO_SUBMIT"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("INPUT_DIR", str(ROOT))

from paradigm_stack_delta_eval import (  # noqa: E402
    fast_auc,
    rk01,
    score_candidate_vs_v4p_hc,
    test1_orthogonal_alpha_probe,
    to_z_ranklogit,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "CUDA required"

OUT = ROOT / "preds" / "playbook_tournament"
LOG = ROOT / "logs" / "overnight"
STATE = LOG / "5playbook_tournament_state.json"
BASELINE = 0.9523071
BASELINE_F0 = 0.9513936
F0_GATE = 1.5e-5
EPS = 1e-6
COST_HR = float(os.environ.get("COST_USD_HR", "0.59"))

NUMS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLOATS = [
    "amount_usd", "hours_since_last_txn", "distance_from_home_km",
    "account_balance_usd", "velocity_score", "merchant_risk_score",
]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def save_state(st: dict) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    st["updated_utc"] = utc()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(STATE)


def rk(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return rankdata(v, method="average") / max(len(v), 1)


def lgt(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS))


# ---------------------------------------------------------------------------
# Shared data / key builders
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self):
        t0 = time.time()
        self.tr = pd.read_csv(ROOT / "data/train.csv")
        self.te = pd.read_csv(ROOT / "data/test.csv")
        self.orig = pd.read_csv(ROOT / "original/credit_card_fraud_2026.csv")
        self.y = self.tr["is_fraud"].to_numpy(np.int32)
        self.n, self.nt = len(self.tr), len(self.te)
        self.folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(self.n), self.y))
        self.a0, self.b0 = self.folds[0]
        bank = np.load(ROOT / "preds/v4p_bank_cache.npz", allow_pickle=True)
        self.v4p_hc_o = bank["v4p_hc_oof"].astype(np.float64)
        self.v4p_hc_t = bank["v4p_hc_test"].astype(np.float64)
        self.O_v4p = bank["O_v4p"]
        self.z_base_o = to_z_ranklogit(self.v4p_hc_o)
        # prevalence calibrate
        lo, hi = -10.0, 10.0
        tgt = float(self.y.mean())
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if float(expit(self.z_base_o + mid).mean()) < tgt:
                lo = mid
            else:
                hi = mid
        self.c_prev = 0.5 * (lo + hi)
        self.z_base_o = self.z_base_o + self.c_prev
        self.z_base_t = to_z_ranklogit(self.v4p_hc_t) + self.c_prev
        self.base_auc = float(fast_auc(self.y, self.v4p_hc_o))
        self.base_f0 = float(fast_auc(self.y[self.b0], self.v4p_hc_o[self.b0]))
        log(f"Ctx loaded n={self.n} nt={self.nt} base={self.base_auc:.7f} F0={self.base_f0:.7f} t={time.time()-t0:.1f}s")

        # import gboost KMAT (84+ joint keys)
        import gboost as gb
        self.gb = gb
        self.KMAT = gb.KMAT.astype(np.int32)  # (NTR+NTE, nkeys)
        self.NCAT = gb.NCAT.astype(np.int32)
        self.KNAMES = list(gb.KNAMES)
        self.nkeys_base = self.KMAT.shape[1]
        log(f"gboost KMAT={self.KMAT.shape} keys={self.nkeys_base}")

        # build extra discrete key matrix once
        self._build_extra_keys()

    def _factorize_full(self, vals) -> np.ndarray:
        codes, _ = pd.factorize(pd.Series(vals), sort=False)
        return codes.astype(np.int32)

    def _build_extra_keys(self):
        t0 = time.time()
        full = pd.concat([self.tr[NUMS + FLAGS + CATS], self.te[NUMS + FLAGS + CATS]], ignore_index=True)
        orig = self.orig
        keys = {}

        # Bulsah / digit-trie for floats + ratios (Group 2 style)
        ratios = {
            "amt_bal": full["amount_usd"] / (full["account_balance_usd"] + 1.0),
            "amt_vel": full["amount_usd"] * full["velocity_score"],
            "vel_risk": full["velocity_score"] * full["merchant_risk_score"],
            "amt_txn": full["amount_usd"] * (full["txn_count_last_24h"] + 1.0),
            "bal_age": full["account_balance_usd"] / (full["card_age_months"] + 1.0),
            "dist_amt": full["distance_from_home_km"] / (full["amount_usd"] + 1.0),
        }
        cols_bulsah = FLOATS + list(ratios.keys())
        for name, series in list({c: full[c] for c in FLOATS}.items()) + list(ratios.items()):
            x = series.to_numpy(np.float64)
            ax = np.abs(x) + 1e-12
            mant = 10 ** (np.log10(ax) - np.floor(np.log10(ax)))
            first2 = np.floor(mant * 10).astype(np.int64)  # 10..99
            dig23 = (np.floor(mant * 1000) % 100).astype(np.int64)
            cents = (np.round(x * 100) % 100).astype(np.int64)
            dec_len = pd.Series(x).astype(str).str.split(".").str[-1].str.len().fillna(0).to_numpy(np.int64)
            trail0 = (np.abs(x - np.round(x)) < 1e-9).astype(np.int64)
            exp_bin = np.floor(np.log10(ax)).astype(np.int64)
            lead = np.floor(mant).astype(np.int64)
            keys[f"b_f2_{name}"] = self._factorize_full(first2)
            keys[f"b_d23_{name}"] = self._factorize_full(dig23)
            keys[f"b_cent_{name}"] = self._factorize_full(cents)
            keys[f"b_dclen_{name}"] = self._factorize_full(dec_len * 10 + trail0)
            keys[f"b_exp_{name}"] = self._factorize_full(exp_bin * 10 + lead)

        # Quantized steps for 2-way / 3-way
        vel_s = (np.round(full["velocity_score"].to_numpy(np.float64) / 0.1) * 0.1).round(1)
        risk_s = (np.round(full["merchant_risk_score"].to_numpy(np.float64) / 0.02) * 0.02).round(2)
        amt_c = (np.round(full["amount_usd"].to_numpy(np.float64) * 100)).astype(np.int64)
        txn = full["txn_count_last_24h"].to_numpy(np.int64)
        flag_bits = (
            full[FLAGS[0]].to_numpy(np.int64)
            + 2 * full[FLAGS[1]].to_numpy(np.int64)
            + 4 * full[FLAGS[2]].to_numpy(np.int64)
            + 8 * full[FLAGS[3]].to_numpy(np.int64)
            + 16 * full[FLAGS[4]].to_numpy(np.int64)
            + 32 * full[FLAGS[5]].to_numpy(np.int64)
        )
        cat = full["merchant_category"].astype(str).to_numpy()
        card = full["card_type"].astype(str).to_numpy()
        qcols = {
            "vel_s": self._factorize_full(vel_s),
            "risk_s": self._factorize_full(risk_s),
            "amt_c": self._factorize_full(amt_c),
            "txn": self._factorize_full(txn),
            "flag64": self._factorize_full(flag_bits),
            "cat": self._factorize_full(cat),
            "card": self._factorize_full(card),
            "hour": self._factorize_full(full["time_of_day_hour"].to_numpy(np.int64)),
            "dow": self._factorize_full(full["day_of_week"].to_numpy(np.int64)),
            "cvv": self._factorize_full(full["cvv_retry_count"].to_numpy(np.int64)),
            "ai": self._factorize_full(full["is_ai_generated_scam_attempt"].to_numpy(np.int64)),
            "foreign": self._factorize_full(full["is_foreign_transaction"].to_numpy(np.int64)),
            "new_m": self._factorize_full(full["is_new_merchant"].to_numpy(np.int64)),
            "vpn": self._factorize_full(full["used_vpn"].to_numpy(np.int64)),
            "ipmis": self._factorize_full(full["ip_country_mismatch"].to_numpy(np.int64)),
            "age_bin": self._factorize_full((full["customer_age"].to_numpy(np.int64) // 5)),
        }
        qnames = list(qcols.keys())
        # Top 120 pairwise among first 16
        pair_names = []
        for i in range(min(16, len(qnames))):
            for j in range(i + 1, min(16, len(qnames))):
                na, nb = qnames[i], qnames[j]
                # pack two codes
                ca, cb = qcols[na], qcols[nb]
                packed = ca.astype(np.int64) * (int(cb.max()) + 1) + cb.astype(np.int64)
                keys[f"p2_{na}__{nb}"] = self._factorize_full(packed)
                pair_names.append(f"p2_{na}__{nb}")
                if len(pair_names) >= 120:
                    break
            if len(pair_names) >= 120:
                break

        # Top 60 3-way: (col_a, col_b, flag/ai)
        three = 0
        for i in range(min(10, len(qnames))):
            for j in range(i + 1, min(10, len(qnames))):
                for fk in ("flag64", "ai", "foreign"):
                    if fk not in qcols:
                        continue
                    ca, cb, cc = qcols[qnames[i]], qcols[qnames[j]], qcols[fk]
                    packed = (
                        ca.astype(np.int64) * (int(cb.max()) + 1) + cb.astype(np.int64)
                    ) * (int(cc.max()) + 1) + cc.astype(np.int64)
                    keys[f"p3_{qnames[i]}__{qnames[j]}__{fk}"] = self._factorize_full(packed)
                    three += 1
                    if three >= 60:
                        break
                if three >= 60:
                    break
            if three >= 60:
                break

        # also keep the 16 quantized 1D keys
        for nm, codes in qcols.items():
            keys[f"q1_{nm}"] = codes

        self.extra_names = list(keys.keys())
        self.K_extra = np.column_stack([keys[n] for n in self.extra_names]).astype(np.int32)
        self.NCAT_extra = np.array([int(self.K_extra[:, j].max()) + 1 for j in range(self.K_extra.shape[1])], dtype=np.int32)

        # group keys for Playbook 2 (top discrete)
        self.groupby_key_names = [
            "q1_cat", "q1_flag64", "q1_vel_s", "q1_risk_s", "q1_txn", "q1_card",
            "q1_hour", "q1_dow", "q1_ai", "q1_foreign",
        ]
        # add a few 2-ways if present
        for nm in self.extra_names:
            if nm.startswith("p2_") and len(self.groupby_key_names) < 15:
                self.groupby_key_names.append(nm)

        # orig20k factorized onto same key spaces where possible (for Group1 prior)
        # Use simple value-count prior on overlapping numeric/flag/cat columns via string keys
        self.orig_y = orig["is_fraud"].to_numpy(np.float64)
        log(f"extra keys={len(self.extra_names)} K_extra={self.K_extra.shape} groupby={len(self.groupby_key_names)} t={time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# TE engines
# ---------------------------------------------------------------------------

def te_rate_counts(k: np.ndarray, w: np.ndarray, ncat: int, prior: float, m: float,
                   s_prior: np.ndarray | None = None, c_prior: np.ndarray | None = None,
                   alpha_prior: float = 0.0) -> np.ndarray:
    s = np.bincount(k, weights=w, minlength=ncat).astype(np.float64)
    c = np.bincount(k, minlength=ncat).astype(np.float64)
    if s_prior is not None:
        s = s + alpha_prior * s_prior
        c = c + alpha_prior * c_prior
    return (s + m * prior) / (c + m)


def eb_logit_eta(rate: np.ndarray, ybar: float) -> np.ndarray:
    return lgt(rate) - lgt(np.full_like(rate, ybar))


def multi_seed_te_fold(
    K: np.ndarray,
    NCAT: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    seeds: list[int] = list(range(8)),
    ms: list[float] = (3.0, 15.0, 60.0),
    parent_eta_a: np.ndarray | None = None,
    parent_eta_b: np.ndarray | None = None,
    parent_eta_t: np.ndarray | None = None,
    hierarchical: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns Ea, Eb, Et of shape (n_rows, n_keys * len(ms)) as EB logits,
    plus slope feature η(m=3)-η(m=60) per key appended later by caller if needed.
    """
    ya = y[a].astype(np.float64)
    ybar = float(ya.mean())
    Ka, Kb, Kt = K[a], K[b], K[test_idx]
    n_k = K.shape[1]
    n_m = len(ms)
    Ea = np.zeros((len(a), n_k * n_m), np.float32)
    Eb = np.zeros((len(b), n_k * n_m), np.float32)
    Et = np.zeros((len(test_idx), n_k * n_m), np.float32)

    # accumulate multi-seed OOF rates on fold a
    for j in range(n_k):
        ncat = int(NCAT[j])
        # for each m, average OOF rates across seeds
        rate_oof_acc = {mi: np.zeros(len(a), np.float64) for mi in range(n_m)}
        for seed in seeds:
            inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(a)), ya))
            for ii, oo in inner:
                for mi, m in enumerate(ms):
                    r = te_rate_counts(Ka[ii, j], ya[ii], ncat, float(ya[ii].mean()), m)
                    if hierarchical and parent_eta_a is not None:
                        # shrink toward additive parent prior in probability space
                        # parent prior probability for rows oo: σ(logit(ybar)+parent)
                        # Use global ybar for stability
                        prior_p = expit(lgt(ybar) + parent_eta_a[oo, j])
                        # convert parent prior into pseudo counts via m
                        # standard EB with prior = prior_p_mean isn't right per-key;
                        # instead: blend rate with prior_p after computing raw rate
                        raw = r[Ka[oo, j]]
                        # key-level prior from training ii:
                        # use mean parent prior among rows with that key
                        # Approximate: per-row blend
                        rate_oof_acc[mi][oo] += 0.7 * raw + 0.3 * prior_p
                    else:
                        rate_oof_acc[mi][oo] += r[Ka[oo, j]]
        for mi, m in enumerate(ms):
            rate_oof = rate_oof_acc[mi] / max(len(seeds), 1)
            Ea[:, j * n_m + mi] = eb_logit_eta(rate_oof, ybar).astype(np.float32)
            # full-a rates for b/test
            r_full = te_rate_counts(Ka[:, j], ya, ncat, ybar, m)
            if hierarchical and parent_eta_b is not None:
                prior_b = expit(lgt(ybar) + parent_eta_b[:, j])
                prior_t = expit(lgt(ybar) + parent_eta_t[:, j])
                rb = 0.7 * r_full[Kb[:, j]] + 0.3 * prior_b
                rt = 0.7 * r_full[Kt[:, j]] + 0.3 * prior_t
            else:
                rb = r_full[Kb[:, j]]
                rt = r_full[Kt[:, j]]
            Eb[:, j * n_m + mi] = eb_logit_eta(rb, ybar).astype(np.float32)
            Et[:, j * n_m + mi] = eb_logit_eta(rt, ybar).astype(np.float32)
    return Ea, Eb, Et


def add_slope_and_rowstats(E: np.ndarray, n_k: int, n_m: int = 3) -> np.ndarray:
    """Append η(m=3)-η(m=60) per key + row max/min/std across all cols."""
    # ms order = (3,15,60) -> indices 0 and 2
    slopes = E[:, 0::n_m] - E[:, 2::n_m]  # (n, n_k)
    row_max = E.max(1, keepdims=True)
    row_min = E.min(1, keepdims=True)
    row_std = E.std(1, keepdims=True)
    return np.hstack([E, slopes, row_max, row_min, row_std]).astype(np.float32)


def support_feats(K: np.ndarray, a: np.ndarray, b: np.ndarray, test_idx: np.ndarray,
                  n_take: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log1p train counts, log1p full counts, ratio for first n_take keys."""
    n_take = min(n_take, K.shape[1])
    outs = []
    for idx in (a, b, test_idx):
        feats = []
        for j in range(n_take):
            ncat = int(K[:, j].max()) + 1
            cnt_a = np.bincount(K[a, j], minlength=ncat).astype(np.float64)
            cnt_all = np.bincount(K[:, j], minlength=ncat).astype(np.float64)
            cj = K[idx, j]
            feats.append(np.log1p(cnt_a[cj]))
            feats.append(np.log1p(cnt_all[cj]))
            feats.append(cnt_all[cj] / (cnt_a[cj] + 1.0))
        outs.append(np.column_stack(feats).astype(np.float32))
    return outs[0], outs[1], outs[2]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def fit_stage1_glm(
    Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, Xt: np.ndarray,
    C: float = 0.05, top_pairs: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Newton-cholesky GLM + optional top pairwise products screened on residual."""
    sc = StandardScaler()
    Za = sc.fit_transform(Xa).astype(np.float64)
    Zb = sc.transform(Xb).astype(np.float64)
    Zt = sc.transform(Xt).astype(np.float64)
    m0 = LogisticRegression(C=C, tol=1e-7, max_iter=200, solver="newton-cholesky")
    m0.fit(Za, ya)
    za0 = m0.decision_function(Za)
    zb0 = m0.decision_function(Zb)
    zt0 = m0.decision_function(Zt)

    # screen products on train residual
    resid = ya - expit(za0)
    Zc = Za - Za.mean(0)
    r = resid - resid.mean()
    wr = Zc * r[:, None]
    d = Za.shape[1]
    # limit screening to first min(d, 120) cols for speed
    d_s = min(d, 120)
    scores, pairs = [], []
    for j in range(d_s):
        s = (wr[:, j: j + 1] * Zc[:, j + 1: d_s]).mean(0)
        for t, k in enumerate(range(j + 1, d_s)):
            scores.append(abs(float(s[t])))
            pairs.append((j, k))
    if not pairs:
        return za0.astype(np.float64), zb0.astype(np.float64), zt0.astype(np.float64), float(roc_auc_score(ya, za0))
    order = np.argsort(-np.asarray(scores))[:top_pairs]
    pairs = [pairs[i] for i in order]
    Ia = np.column_stack([Za[:, j] * Za[:, k] for j, k in pairs])
    Ib = np.column_stack([Zb[:, j] * Zb[:, k] for j, k in pairs])
    It = np.column_stack([Zt[:, j] * Zt[:, k] for j, k in pairs])
    scI = StandardScaler().fit(Ia)
    Ia, Ib, It = scI.transform(Ia), scI.transform(Ib), scI.transform(It)
    m1 = LogisticRegression(C=0.002, tol=1e-6, max_iter=300, solver="lbfgs")
    m1.fit(Ia, ya)
    za = za0 + 0.5 * m1.decision_function(Ia)
    zb = zb0 + 0.5 * m1.decision_function(Ib)
    zt = zt0 + 0.5 * m1.decision_function(It)
    return za.astype(np.float64), zb.astype(np.float64), zt.astype(np.float64), float(roc_auc_score(ya, za))


def fit_xgb_bag(
    Xa, ya, za, Xb, yb, zb, Xt, zt,
    seeds_depth: list[tuple[int, int]],
    n_rounds: int = 400,
    lr: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    preds_b, preds_t, aucs = [], [], []
    for seed, depth in seeds_depth:
        dtr = xgb.DMatrix(Xa, label=ya, base_margin=za.astype(np.float32))
        dva = xgb.DMatrix(Xb, label=yb, base_margin=zb.astype(np.float32))
        dte = xgb.DMatrix(Xt, base_margin=zt.astype(np.float32))
        params = dict(
            objective="binary:logistic", eval_metric="auc",
            max_depth=depth, eta=lr, subsample=0.8,
            colsample_bynode=0.35 if depth == 2 else 0.25,
            min_child_weight=50, reg_lambda=25.0 if depth == 2 else 50.0,
            tree_method="hist", device="cuda:0", seed=seed, max_bin=256,
        )
        bst = xgb.train(params, dtr, num_boost_round=n_rounds, evals=[(dva, "v")],
                        early_stopping_rounds=80, verbose_eval=False)
        it = bst.best_iteration + 1 if bst.best_iteration is not None else n_rounds
        pb = bst.predict(dva, output_margin=True, iteration_range=(0, it))
        pt = bst.predict(dte, output_margin=True, iteration_range=(0, it))
        preds_b.append(pb)
        preds_t.append(pt)
        aucs.append(float(roc_auc_score(yb, pb)))
    return np.mean(preds_b, 0), np.mean(preds_t, 0), aucs


def fit_lgb_bag(
    Xa, ya, za, Xb, yb, zb, Xt, zt,
    seeds_leaves: list[tuple[int, int]],
    n_rounds: int = 800,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    preds_b, preds_t, aucs = [], [], []
    for seed, leaves in seeds_leaves:
        dtr = lgb.Dataset(Xa, label=ya, init_score=za)
        dva = lgb.Dataset(Xb, label=yb, init_score=zb, reference=dtr)
        params = dict(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=leaves, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.30, reg_lambda=5.0,
            verbose=-1, seed=seed, n_jobs=8,
        )
        bst = lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pb = zb + bst.predict(Xb, raw_score=True)
        pt = zt + bst.predict(Xt, raw_score=True)
        preds_b.append(pb)
        preds_t.append(pt)
        aucs.append(float(roc_auc_score(yb, pb)))
    return np.mean(preds_b, 0), np.mean(preds_t, 0), aucs


def fit_cat_ordered(
    Xa_df, ya, za, Xb_df, yb, zb, Xt_df, zt, cat_cols: list[str],
    seeds: list[int] = (42, 7, 2024), iterations: int = 600,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    preds_b, preds_t, aucs = [], [], []
    for seed in seeds:
        ptr = Pool(Xa_df, label=ya, cat_features=cat_cols, baseline=za.astype(np.float64))
        pva = Pool(Xb_df, label=yb, cat_features=cat_cols, baseline=zb.astype(np.float64))
        pte = Pool(Xt_df, cat_features=cat_cols, baseline=zt.astype(np.float64))
        cb = CatBoostClassifier(
            iterations=iterations, learning_rate=0.04, depth=4,
            loss_function="Logloss", eval_metric="AUC",
            task_type="GPU", devices="0", boosting_type="Ordered",
            max_ctr_complexity=2, random_seed=seed, od_type="Iter", od_wait=80,
            verbose=False, allow_writing_files=False,
        )
        cb.fit(ptr, eval_set=pva, use_best_model=True)
        pb = cb.predict(pva, prediction_type="RawFormulaVal")
        pt = cb.predict(pte, prediction_type="RawFormulaVal")
        preds_b.append(pb)
        preds_t.append(pt)
        aucs.append(float(roc_auc_score(yb, pb)))
    return np.mean(preds_b, 0), np.mean(preds_t, 0), aucs


def eval_vs_v4p(ctx: Ctx, score_b: np.ndarray, score_t: np.ndarray | None, name: str) -> dict:
    """Fold-0 only Test-1 probe + KS/ρ diagnostics."""
    yb = ctx.y[ctx.b0]
    v4b = ctx.v4p_hc_o[ctx.b0]
    # 2-fold nested probe on b0 for a fair alpha estimate
    skf2 = list(StratifiedKFold(2, shuffle=True, random_state=42).split(np.zeros(len(yb)), yb))
    t1 = test1_orthogonal_alpha_probe(yb, v4b, score_b, skf2)
    solo = float(fast_auc(yb, score_b))
    rho = float(spearmanr(score_b, v4b).correlation)
    out = {
        "name": name,
        "solo_f0_auc": solo,
        "test1_delta": float(t1["delta_auc"]),
        "test1_stack_auc": float(t1["stack_oof_auc"]),
        "mean_alpha": float(t1["mean_alpha"]),
        "fold_win_rate": float(t1["fold_win_rate"]),
        "rho_v4p": rho,
        "baseline_f0": ctx.base_f0,
    }
    if score_t is not None:
        out["ks"] = float(ks_2samp(rk01(score_b), rk01(score_t)).statistic)
        out["rho_test"] = float(spearmanr(score_t, ctx.v4p_hc_t).correlation)
        out["abs_drho"] = abs(rho - out["rho_test"])
    else:
        out["ks"] = None
        out["abs_drho"] = None
    return out


# ---------------------------------------------------------------------------
# Playbooks (Fold-0)
# ---------------------------------------------------------------------------

def playbook1_kmat450_f0(ctx: Ctx) -> dict:
    log("PLAYBOOK1 Fold-0: KMAT-450 multi-seed EB + Bulsah + hierarchical 2D/3D")
    t0 = time.time()
    a, b = ctx.a0, ctx.b0
    test_idx = np.arange(ctx.n, ctx.n + ctx.nt)

    # Base KMAT multi-seed TE (use first 84 keys for speed on F0, all keys if ≤120)
    n_base = min(ctx.nkeys_base, 100)
    K_base = ctx.KMAT[:, :n_base]
    NCAT_base = ctx.NCAT[:n_base]
    Ea_b, Eb_b, Et_b = multi_seed_te_fold(
        K_base, NCAT_base, a, b, test_idx, ctx.y, seeds=list(range(4)), ms=[3.0, 15.0, 60.0]
    )
    # Extra Bulsah + pairs (cap for F0 speed)
    # take first 96 bulsah-ish + 60 pairs + 30 triples
    bulsah_idx = [i for i, n in enumerate(ctx.extra_names) if n.startswith("b_")][:96]
    pair_idx = [i for i, n in enumerate(ctx.extra_names) if n.startswith("p2_")][:60]
    trip_idx = [i for i, n in enumerate(ctx.extra_names) if n.startswith("p3_")][:30]
    q1_idx = [i for i, n in enumerate(ctx.extra_names) if n.startswith("q1_")][:16]
    take = bulsah_idx + q1_idx + pair_idx + trip_idx
    K_ex = ctx.K_extra[:, take]
    NCAT_ex = ctx.NCAT_extra[take]
    # hierarchical only for pairs/trips: approximate parent as zeros for F0 speed if no 1D align
    Ea_e, Eb_e, Et_e = multi_seed_te_fold(
        K_ex, NCAT_ex, a, b, test_idx, ctx.y, seeds=list(range(3)), ms=[3.0, 15.0, 60.0],
        hierarchical=False,
    )
    Ea = add_slope_and_rowstats(np.hstack([Ea_b, Ea_e]), n_base + len(take), 3)
    Eb = add_slope_and_rowstats(np.hstack([Eb_b, Eb_e]), n_base + len(take), 3)
    Et = add_slope_and_rowstats(np.hstack([Et_b, Et_e]), n_base + len(take), 3)
    Sa, Sb, St = support_feats(np.hstack([K_base, K_ex]), a, b, test_idx, n_take=30)
    Xa = np.hstack([Ea, Sa]).astype(np.float32)
    Xb = np.hstack([Eb, Sb]).astype(np.float32)
    Xt = np.hstack([Et, St]).astype(np.float32)
    log(f"PB1 matrices Xa={Xa.shape}")

    za, zb, zt, glm_tr = fit_stage1_glm(Xa, ctx.y[a], Xb, Xt, C=0.05, top_pairs=600)
    log(f"PB1 Stage1 GLM train-AUC={glm_tr:.5f} val={roc_auc_score(ctx.y[b], zb):.5f}")
    xgb_b, xgb_t, xauc = fit_xgb_bag(
        Xa, ctx.y[a], za, Xb, ctx.y[b], zb, Xt, zt,
        seeds_depth=[(42, 2), (7, 2), (2024, 3), (99, 3)], n_rounds=350, lr=0.03,
    )
    lgb_b, lgb_t, lauc = fit_lgb_bag(
        Xa, ctx.y[a], za, Xb, ctx.y[b], zb, Xt, zt,
        seeds_leaves=[(42, 4), (7, 4), (2024, 5)], n_rounds=600,
    )
    # blend
    score_b = 0.25 * zb + 0.45 * xgb_b + 0.30 * lgb_b
    score_t = 0.25 * zt + 0.45 * xgb_t + 0.30 * lgb_t
    ev = eval_vs_v4p(ctx, score_b, score_t, "PB1_KMAT450")
    ev.update({
        "wall_s": round(time.time() - t0, 1),
        "glm_f0": float(roc_auc_score(ctx.y[b], zb)),
        "xgb_f0": float(np.mean(xauc)),
        "lgb_f0": float(np.mean(lauc)),
        "score_b": score_b, "score_t": score_t,
        "Xa": Xa, "Xb": Xb, "Xt": Xt, "za": za, "zb": zb, "zt": zt,
    })
    log(f"PB1 DONE solo={ev['solo_f0_auc']:.6f} Δ={ev['test1_delta']:+.3e} KS={ev['ks']:.4f} t={ev['wall_s']}s")
    return ev


def playbook2_groupby_f0(ctx: Ctx) -> dict:
    log("PLAYBOOK2 Fold-0: Unsupervised contextual groupby aggregations")
    t0 = time.time()
    a, b = ctx.a0, ctx.b0
    full = pd.concat([ctx.tr[NUMS + FLAGS + CATS], ctx.te[NUMS + FLAGS + CATS]], ignore_index=True)
    orig_cont = ctx.orig[NUMS].to_numpy(np.float64)
    # Build group codes from extra keys
    name_to_i = {n: i for i, n in enumerate(ctx.extra_names)}
    gnames = [n for n in ctx.groupby_key_names if n in name_to_i][:15]
    cont_cols = ["amount_usd", "account_balance_usd", "velocity_score", "merchant_risk_score",
                 "distance_from_home_km", "hours_since_last_txn"]
    Xcont = full[cont_cols].to_numpy(np.float64)
    # also append orig rows for aggregation stats only
    # Map orig onto same group keys via re-factorize is hard; use train+test only for F0 speed
    feats = []
    feat_names = []
    for gn in gnames:
        g = ctx.K_extra[:, name_to_i[gn]]
        ncat = int(g.max()) + 1
        for ci, cname in enumerate(cont_cols):
            x = Xcont[:, ci]
            # GPU scatter aggregations
            g_t = torch.tensor(g, device=DEVICE, dtype=torch.long)
            x_t = torch.tensor(x, device=DEVICE, dtype=torch.float64)
            ones = torch.ones_like(x_t)
            cnt = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, ones)
            sm = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, x_t)
            sm2 = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, x_t * x_t)
            mu = sm / cnt.clamp_min(1.0)
            var = (sm2 / cnt.clamp_min(1.0) - mu * mu).clamp_min(0.0)
            sd = var.sqrt()
            mu_r = mu[g_t]
            sd_r = sd[g_t]
            z = (x_t - mu_r) / (sd_r + 1e-6)
            # approximate percentile via rank within group is expensive; use CDF-ish: 0.5*(1+erf(z/sqrt2))
            pct = 0.5 * (1.0 + torch.erf(z / 1.41421356237))
            xmax = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_reduce_(
                0, g_t, x_t, reduce="amax", include_self=True
            )
            ratio = x_t / (xmax[g_t] + 1.0)
            block = torch.stack([mu_r, sd_r, z, pct, ratio], dim=1).cpu().numpy().astype(np.float32)
            feats.append(block)
            feat_names += [f"{gn}_{cname}_{s}" for s in ("mu", "sd", "z", "pct", "ratio")]
    F_all = np.hstack(feats)
    # KS filter vs train/test
    F_tr, F_te = F_all[: ctx.n], F_all[ctx.n :]
    keep = []
    for j in range(F_all.shape[1]):
        ks = float(ks_2samp(F_tr[::20, j], F_te[::20, j]).statistic)
        if ks < 0.010:
            keep.append(j)
    if len(keep) < 20:
        keep = list(range(min(80, F_all.shape[1])))
    F_tr = F_tr[:, keep]
    F_te = F_te[:, keep]
    log(f"PB2 contextual feats kept={len(keep)}/{F_all.shape[1]}")

    # Base TE from gboost single-seed (fast)
    Ea, Eb, Et = ctx.gb.te_fold(a, b, smooth=20.0, seed=0)
    # take first 80 TE cols
    Ea, Eb, Et = Ea[:, :80], Eb[:, :80], Et[:, :80]
    Xa = np.hstack([lgt(Ea).astype(np.float32), F_tr[a]])
    Xb = np.hstack([lgt(Eb).astype(np.float32), F_tr[b]])
    Xt = np.hstack([lgt(Et).astype(np.float32), F_te])
    za = ctx.z_base_o[a]
    zb = ctx.z_base_o[b]
    zt = ctx.z_base_t

    xgb_b, xgb_t, xauc = fit_xgb_bag(
        Xa, ctx.y[a], za, Xb, ctx.y[b], zb, Xt, zt,
        seeds_depth=[(42, 3), (7, 3), (2024, 4), (99, 4)], n_rounds=300, lr=0.03,
    )
    # CatBoost on numeric only for speed
    cols = [f"f{i}" for i in range(Xa.shape[1])]
    Xa_df = pd.DataFrame(Xa, columns=cols)
    Xb_df = pd.DataFrame(Xb, columns=cols)
    Xt_df = pd.DataFrame(Xt, columns=cols)
    cat_b, cat_t, cauc = fit_cat_ordered(
        Xa_df, ctx.y[a], za, Xb_df, ctx.y[b], zb, Xt_df, zt, cat_cols=[],
        seeds=[42, 7], iterations=400,
    )
    score_b = 0.55 * xgb_b + 0.45 * cat_b
    score_t = 0.55 * xgb_t + 0.45 * cat_t
    ev = eval_vs_v4p(ctx, score_b, score_t, "PB2_GroupbyCtx")
    ev.update({
        "wall_s": round(time.time() - t0, 1),
        "xgb_f0": float(np.mean(xauc)),
        "cat_f0": float(np.mean(cauc)),
        "score_b": score_b, "score_t": score_t,
    })
    log(f"PB2 DONE solo={ev['solo_f0_auc']:.6f} Δ={ev['test1_delta']:+.3e} KS={ev['ks']:.4f} t={ev['wall_s']}s")
    return ev


def playbook3_residual_dist_f0(ctx: Ctx) -> dict:
    log("PLAYBOOK3 Fold-0: V4p residual-distribution TE + stealth counters + Ordered CatBoost")
    t0 = time.time()
    a, b = ctx.a0, ctx.b0
    test_idx = np.arange(ctx.n, ctx.n + ctx.nt)
    # residual on train fold a
    r_a = ctx.y[a].astype(np.float64) - ctx.v4p_hc_o[a]
    v_a = ctx.v4p_hc_o[a]
    stealth = ((ctx.y[a] == 1) & (ctx.v4p_hc_o[a] < 0.10)).astype(np.float64)

    n_k = min(ctx.nkeys_base, 90)
    K = ctx.KMAT[:, :n_k]
    feats_a, feats_b, feats_t = [], [], []
    cat_codes_a, cat_codes_b, cat_codes_t = [], [], []
    cat_names = []
    for j in range(n_k):
        ncat = int(ctx.NCAT[j])
        Ka, Kb, Kt = K[a, j], K[b, j], K[test_idx, j]
        # residual mean / max / std(V4p) / stealth support
        sum_r = np.bincount(Ka, weights=r_a, minlength=ncat).astype(np.float64)
        cnt = np.bincount(Ka, minlength=ncat).astype(np.float64) + 1e-9
        mean_r = sum_r / cnt
        # max residual via sort trick approximate: use mean of top via second moment not available;
        # use 95th-ish: mean + 2*std of residuals
        sum_r2 = np.bincount(Ka, weights=r_a * r_a, minlength=ncat).astype(np.float64)
        std_r = np.sqrt(np.maximum(sum_r2 / cnt - mean_r ** 2, 0))
        max_r_approx = mean_r + 2.0 * std_r
        sum_v = np.bincount(Ka, weights=v_a, minlength=ncat).astype(np.float64)
        sum_v2 = np.bincount(Ka, weights=v_a * v_a, minlength=ncat).astype(np.float64)
        std_v = np.sqrt(np.maximum(sum_v2 / cnt - (sum_v / cnt) ** 2, 0))
        stealth_c = np.bincount(Ka, weights=stealth, minlength=ncat).astype(np.float64)
        # EB smooth
        m = 15.0
        mean_r_s = (sum_r) / (cnt + m)  # shrink to 0
        for idx, bucket in ((feats_a, Ka), (feats_b, Kb), (feats_t, Kt)):
            idx.append(np.column_stack([
                mean_r_s[bucket], max_r_approx[bucket], std_v[bucket],
                np.log1p(stealth_c[bucket]), np.log1p(cnt[bucket]),
            ]).astype(np.float32))
        cat_codes_a.append(Ka)
        cat_codes_b.append(Kb)
        cat_codes_t.append(Kt)
        cat_names.append(f"k{j}")

    Fa = np.hstack(feats_a)
    Fb = np.hstack(feats_b)
    Ft = np.hstack(feats_t)
    # also classic TE
    Ea, Eb, Et = ctx.gb.te_fold(a, b, smooth=20.0, seed=0)
    Ea, Eb, Et = lgt(Ea[:, :n_k]).astype(np.float32), lgt(Eb[:, :n_k]).astype(np.float32), lgt(Et[:, :n_k]).astype(np.float32)
    Xa_num = np.hstack([Ea, Fa])
    Xb_num = np.hstack([Eb, Fb])
    Xt_num = np.hstack([Et, Ft])

    # CatBoost with native cats = raw key IDs (top 40)
    n_cat = min(40, n_k)
    def make_df(num, cats_list, n):
        d = {f"n{i}": num[:, i] for i in range(num.shape[1])}
        for j in range(n_cat):
            d[f"c{j}"] = cats_list[j]
        return pd.DataFrame(d)

    Xa_df = make_df(Xa_num, cat_codes_a, len(a))
    Xb_df = make_df(Xb_num, cat_codes_b, len(b))
    Xt_df = make_df(Xt_num, cat_codes_t, ctx.nt)
    cat_cols = [f"c{j}" for j in range(n_cat)]
    za, zb, zt = ctx.z_base_o[a], ctx.z_base_o[b], ctx.z_base_t

    cat_b, cat_t, cauc = fit_cat_ordered(
        Xa_df, ctx.y[a], za, Xb_df, ctx.y[b], zb, Xt_df, zt, cat_cols=cat_cols,
        seeds=[42, 7, 2024], iterations=500,
    )
    xgb_b, xgb_t, xauc = fit_xgb_bag(
        Xa_num, ctx.y[a], za, Xb_num, ctx.y[b], zb, Xt_num, zt,
        seeds_depth=[(42, 2), (7, 3), (2024, 3)], n_rounds=350, lr=0.03,
    )
    score_b = 0.55 * cat_b + 0.45 * xgb_b
    score_t = 0.55 * cat_t + 0.45 * xgb_t
    ev = eval_vs_v4p(ctx, score_b, score_t, "PB3_ResidDist_Cat")
    ev.update({
        "wall_s": round(time.time() - t0, 1),
        "cat_f0": float(np.mean(cauc)),
        "xgb_f0": float(np.mean(xauc)),
        "score_b": score_b, "score_t": score_t,
    })
    log(f"PB3 DONE solo={ev['solo_f0_auc']:.6f} Δ={ev['test1_delta']:+.3e} KS={ev['ks']:.4f} t={ev['wall_s']}s")
    return ev


def playbook4_transductive_f0(ctx: Ctx) -> dict:
    log("PLAYBOOK4 Fold-0: Transductive soft-pseudo TE (γ∈{0.15,0.30})")
    t0 = time.time()
    a, b = ctx.a0, ctx.b0
    test_idx = np.arange(ctx.n, ctx.n + ctx.nt)
    # soft preds on unlabeled pool b+test (OOF on b!)
    p_trans_b = expit(ctx.z_base_o[b])
    p_trans_t = expit(ctx.z_base_t)
    n_k = min(ctx.nkeys_base, 100)
    K = ctx.KMAT[:, :n_k]
    best = None
    for gamma in (0.15, 0.30):
        Ea = np.zeros((len(a), n_k), np.float32)
        Eb = np.zeros((len(b), n_k), np.float32)
        Et = np.zeros((ctx.nt, n_k), np.float32)
        ya = ctx.y[a].astype(np.float64)
        ybar = float(ya.mean())
        # unlabeled rows = b + test
        U_idx = np.concatenate([b, test_idx])
        p_U = np.concatenate([p_trans_b, p_trans_t])
        for j in range(n_k):
            ncat = int(ctx.NCAT[j])
            Ka = K[a, j]
            # train counts
            s_a = np.bincount(Ka, weights=ya, minlength=ncat).astype(np.float64)
            c_a = np.bincount(Ka, minlength=ncat).astype(np.float64)
            Ku = K[U_idx, j]
            # soft sums on U
            s_u = np.bincount(Ku, weights=p_U, minlength=ncat).astype(np.float64)
            c_u = np.bincount(Ku, minlength=ncat).astype(np.float64)
            m = 15.0
            # OOF on a via inner 5-fold without soft (soft only for b/test)
            for ii, oo in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(a)), ya):
                r = te_rate_counts(Ka[ii], ya[ii], ncat, float(ya[ii].mean()), m)
                Ea[oo, j] = eb_logit_eta(r[Ka[oo]], ybar).astype(np.float32)
            # for each unlabeled row i: leave-one-out soft
            # p_k = (s_a + γ*(s_u - p_i*1_{key}) + m*ybar) / (c_a + γ*(c_u-1) + m)
            for arr, idx_rows, outE in (
                (Eb, b, Eb),
                (Et, test_idx, Et),
            ):
                Kj = K[idx_rows, j]
                if idx_rows is b:
                    p_rows = p_trans_b
                else:
                    p_rows = p_trans_t
                # vectorized LOO
                num = s_a[Kj] + gamma * (s_u[Kj] - p_rows) + m * ybar
                den = c_a[Kj] + gamma * (c_u[Kj] - 1.0) + m
                rate = np.clip(num / np.maximum(den, 1e-9), EPS, 1 - EPS)
                outE[:, j] = eb_logit_eta(rate, ybar).astype(np.float32)
        Xa, Xb, Xt = Ea, Eb, Et
        za, zb, zt, _ = fit_stage1_glm(Xa, ctx.y[a], Xb, Xt, C=0.08, top_pairs=400)
        xgb_b, xgb_t, xauc = fit_xgb_bag(
            Xa, ctx.y[a], za, Xb, ctx.y[b], zb, Xt, zt,
            seeds_depth=[(42, 2), (7, 3), (2024, 3)], n_rounds=300, lr=0.03,
        )
        score_b = 0.35 * zb + 0.65 * xgb_b
        score_t = 0.35 * zt + 0.65 * xgb_t
        ev = eval_vs_v4p(ctx, score_b, score_t, f"PB4_Trans_g{gamma}")
        ev.update({
            "gamma": gamma,
            "xgb_f0": float(np.mean(xauc)),
            "glm_f0": float(roc_auc_score(ctx.y[b], zb)),
            "score_b": score_b, "score_t": score_t,
        })
        log(f"PB4 γ={gamma} solo={ev['solo_f0_auc']:.6f} Δ={ev['test1_delta']:+.3e}")
        if best is None or ev["test1_delta"] > best["test1_delta"]:
            best = ev
    best["wall_s"] = round(time.time() - t0, 1)
    best["name"] = "PB4_Transductive"
    log(f"PB4 DONE bestγ={best.get('gamma')} Δ={best['test1_delta']:+.3e} t={best['wall_s']}s")
    return best


def playbook5_auc_fobj_f0(ctx: Ctx) -> dict:
    log("PLAYBOOK5 Fold-0: Custom ROC-AUC overlap-band fobj")
    t0 = time.time()
    a, b = ctx.a0, ctx.b0
    # Features: KMAT TE + lightweight groupby z-scores for amount/velocity
    Ea, Eb, Et = ctx.gb.te_fold(a, b, smooth=20.0, seed=0)
    n_k = min(80, Ea.shape[1])
    Ea, Eb, Et = lgt(Ea[:, :n_k]).astype(np.float32), lgt(Eb[:, :n_k]).astype(np.float32), lgt(Et[:, :n_k]).astype(np.float32)

    # reuse a few contextual z features from merchant_category + velocity
    full = pd.concat([ctx.tr, ctx.te], ignore_index=True)
    cat = pd.factorize(full["merchant_category"].astype(str))[0].astype(np.int32)
    for col in ("amount_usd", "velocity_score", "merchant_risk_score"):
        x = full[col].to_numpy(np.float64)
        ncat = int(cat.max()) + 1
        g_t = torch.tensor(cat, device=DEVICE, dtype=torch.long)
        x_t = torch.tensor(x, device=DEVICE, dtype=torch.float64)
        cnt = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, torch.ones_like(x_t))
        sm = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, x_t)
        sm2 = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, x_t * x_t)
        mu = (sm / cnt.clamp_min(1))[g_t]
        sd = ((sm2 / cnt.clamp_min(1) - (sm / cnt.clamp_min(1)) ** 2).clamp_min(0).sqrt())[g_t]
        z = ((x_t - mu) / (sd + 1e-6)).cpu().numpy().astype(np.float32).reshape(-1, 1)
        Ea = np.hstack([Ea, z[a]])
        Eb = np.hstack([Eb, z[b]])
        Et = np.hstack([Et, z[ctx.n :]])

    za = ctx.z_base_o[a].astype(np.float64)
    zb = ctx.z_base_o[b].astype(np.float64)
    zt = ctx.z_base_t.astype(np.float64)

    # Custom objective: weight logloss gradient by overlap density around boundary
    def make_fobj(y_true, z_base):
        y_true = np.asarray(y_true, dtype=np.float64)
        # boundary from base margin distribution of positives/negatives mix
        mu_b = float(np.median(z_base))
        sig = float(np.std(z_base)) * 0.35 + 0.5

        def fobj(preds, dtrain):
            # preds are raw margins (with base_margin already added by xgb when training?)
            # In xgb custom obj, preds are the raw predictions WITHOUT base_margin unless set;
            # we pass base_margin separately so preds are tree output; total logit = preds + base
            p = expit(preds)
            # weight
            w = np.exp(-0.5 * ((preds - mu_b) / sig) ** 2) + 0.15
            g = w * (p - y_true)
            h = w * p * (1.0 - p) + 1e-6
            return g, h

        return fobj

    preds_b, preds_t, aucs = [], [], []
    for seed, depth in ((42, 2), (7, 2), (2024, 3), (99, 3), (123, 3)):
        dtr = xgb.DMatrix(Ea, label=ctx.y[a], base_margin=za.astype(np.float32))
        dva = xgb.DMatrix(Eb, label=ctx.y[b], base_margin=zb.astype(np.float32))
        dte = xgb.DMatrix(Et, base_margin=zt.astype(np.float32))
        fobj = make_fobj(ctx.y[a], za)
        params = dict(
            max_depth=depth, eta=0.03, subsample=0.8, colsample_bynode=0.3,
            min_child_weight=40, reg_lambda=30.0, tree_method="hist", device="cuda:0",
            seed=seed, max_bin=256, disable_default_eval_metric=1,
        )
        # custom metric AUC on total margin
        def feval(preds, dmat):
            y = dmat.get_label()
            return "auc", float(roc_auc_score(y, preds))

        bst = xgb.train(
            params, dtr, num_boost_round=350, obj=fobj,
            evals=[(dva, "v")], custom_metric=feval,
            early_stopping_rounds=60, verbose_eval=False,
        )
        it = bst.best_iteration + 1 if bst.best_iteration is not None else 350
        pb = bst.predict(dva, output_margin=True, iteration_range=(0, it))
        pt = bst.predict(dte, output_margin=True, iteration_range=(0, it))
        preds_b.append(pb)
        preds_t.append(pt)
        aucs.append(float(roc_auc_score(ctx.y[b], pb)))

    # LightGBM with sample weights approximating overlap band
    p0 = expit(za)
    w_tr = np.exp(-0.5 * ((za - np.median(za)) / (np.std(za) * 0.35 + 0.5)) ** 2) + 0.15
    lpreds_b, lpreds_t = [], []
    for seed, leaves in ((42, 4), (7, 4), (2024, 5)):
        dtr = lgb.Dataset(Ea, label=ctx.y[a], init_score=za, weight=w_tr)
        dva = lgb.Dataset(Eb, label=ctx.y[b], init_score=zb, reference=dtr)
        params = dict(
            objective="binary", metric="auc", learning_rate=0.03, num_leaves=leaves,
            min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
            reg_lambda=5.0, verbose=-1, seed=seed, n_jobs=8,
        )
        bst = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        lpreds_b.append(zb + bst.predict(Eb, raw_score=True))
        lpreds_t.append(zt + bst.predict(Et, raw_score=True))

    score_b = 0.6 * np.mean(preds_b, 0) + 0.4 * np.mean(lpreds_b, 0)
    score_t = 0.6 * np.mean(preds_t, 0) + 0.4 * np.mean(lpreds_t, 0)
    ev = eval_vs_v4p(ctx, score_b, score_t, "PB5_AUC_fobj")
    ev.update({
        "wall_s": round(time.time() - t0, 1),
        "xgb_f0": float(np.mean(aucs)),
        "score_b": score_b, "score_t": score_t,
    })
    log(f"PB5 DONE solo={ev['solo_f0_auc']:.6f} Δ={ev['test1_delta']:+.3e} KS={ev['ks']:.4f} t={ev['wall_s']}s")
    return ev


# ---------------------------------------------------------------------------
# Phase 2: 5-fold promotion
# ---------------------------------------------------------------------------

def promote_5fold_simple(ctx: Ctx, pb_name: str, builder_fn) -> dict:
    """
    Generic 5-fold promotion: builder_fn(ctx, a, b) -> (score_b, score_t_fold)
    Then 3-seed holdout refit bag on test.
    """
    log(f"PHASE2 promote {pb_name} to 5-fold")
    t0 = time.time()
    oof = np.zeros(ctx.n, dtype=np.float64)
    te_folds = []
    for fi, (a, b) in enumerate(ctx.folds):
        sb, st = builder_fn(ctx, a, b, fi)
        oof[b] = sb
        te_folds.append(st)
        log(f"  {pb_name} fold{fi} auc={roc_auc_score(ctx.y[b], sb):.6f}")
    te_avg = np.mean(te_folds, 0)
    # 3-seed 100%-ish refit: train on all but small hold from fold0
    refits = []
    a0, b0 = ctx.folds[0]
    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        hold = rng.choice(b0, size=max(2000, len(b0) // 15), replace=False)
        train_idx = np.setdiff1d(np.arange(ctx.n), hold)
        sb, st = builder_fn(ctx, train_idx, hold, 100 + seed)
        refits.append(st)
    te_bag = 0.75 * np.mean(refits, 0) + 0.25 * te_avg
    # stack eval vs v4p_hc
    res = score_candidate_vs_v4p_hc(ctx.y, ctx.v4p_hc_o, oof, ctx.folds)
    oof_stack = res.pop("stack_oof_scores")
    ks = float(ks_2samp(rk01(oof), rk01(te_bag)).statistic)
    drho = abs(float(spearmanr(oof, ctx.v4p_hc_o).correlation) - float(spearmanr(te_bag, ctx.v4p_hc_t).correlation))
    out = {
        "name": pb_name,
        "solo_oof_auc": float(fast_auc(ctx.y, oof)),
        "stack_oof_auc": float(res["stack_oof_auc"]),
        "delta_auc": float(res["delta_auc"]),
        "mean_alpha": float(res["stack_mean_alpha"]),
        "fold_win_rate": float(res["fold_win_rate"]),
        "ks": ks,
        "abs_drho": drho,
        "wall_s": round(time.time() - t0, 1),
    }
    np.save(OUT / f"{pb_name}_oof.npy", oof)
    np.save(OUT / f"{pb_name}_test.npy", te_bag)
    np.save(OUT / f"{pb_name}_stack_oof.npy", oof_stack)
    log(f"PHASE2 {pb_name} solo={out['solo_oof_auc']:.6f} Δ={out['delta_auc']:+.3e} KS={ks:.4f} |Δρ|={drho:.5f}")
    return out


def builder_pb5_fold(ctx, a, b, fi):
    """Lightweight reusable builder for PB5-style features (also used as fallback)."""
    Ea, Eb, Et = ctx.gb.te_fold(a, b, smooth=20.0, seed=fi % 5)
    n_k = min(80, Ea.shape[1])
    Ea, Eb, Et = lgt(Ea[:, :n_k]).astype(np.float32), lgt(Eb[:, :n_k]).astype(np.float32), lgt(Et[:, :n_k]).astype(np.float32)
    za, zb, zt = ctx.z_base_o[a], ctx.z_base_o[b], ctx.z_base_t
    xgb_b, xgb_t, _ = fit_xgb_bag(
        Ea, ctx.y[a], za, Eb, ctx.y[b], zb, Et, zt,
        seeds_depth=[(42 + fi, 2), (7 + fi, 3)], n_rounds=250, lr=0.03,
    )
    return xgb_b, xgb_t


def builder_pb1_fold(ctx, a, b, fi):
    test_idx = np.arange(ctx.n, ctx.n + ctx.nt)
    n_base = min(ctx.nkeys_base, 80)
    K_base = ctx.KMAT[:, :n_base]
    Ea_b, Eb_b, Et_b = multi_seed_te_fold(
        K_base, ctx.NCAT[:n_base], a, b, test_idx, ctx.y,
        seeds=list(range(3)), ms=[3.0, 15.0, 60.0],
    )
    bulsah_idx = [i for i, n in enumerate(ctx.extra_names) if n.startswith("b_")][:60]
    pair_idx = [i for i, n in enumerate(ctx.extra_names) if n.startswith("p2_")][:40]
    take = bulsah_idx + pair_idx
    K_ex = ctx.K_extra[:, take]
    Ea_e, Eb_e, Et_e = multi_seed_te_fold(
        K_ex, ctx.NCAT_extra[take], a, b, test_idx, ctx.y, seeds=list(range(2)), ms=[3.0, 15.0, 60.0],
    )
    Ea = add_slope_and_rowstats(np.hstack([Ea_b, Ea_e]), n_base + len(take), 3)
    Eb = add_slope_and_rowstats(np.hstack([Eb_b, Eb_e]), n_base + len(take), 3)
    Et = add_slope_and_rowstats(np.hstack([Et_b, Et_e]), n_base + len(take), 3)
    za, zb, zt, _ = fit_stage1_glm(Ea, ctx.y[a], Eb, Et, C=0.05, top_pairs=400)
    xgb_b, xgb_t, _ = fit_xgb_bag(
        Ea, ctx.y[a], za, Eb, ctx.y[b], zb, Et, zt,
        seeds_depth=[(42 + fi, 2), (7 + fi, 3)], n_rounds=280, lr=0.03,
    )
    score_b = 0.3 * zb + 0.7 * xgb_b
    score_t = 0.3 * zt + 0.7 * xgb_t
    return score_b, score_t


def builder_pb3_fold(ctx, a, b, fi):
    test_idx = np.arange(ctx.n, ctx.n + ctx.nt)
    n_k = min(ctx.nkeys_base, 70)
    K = ctx.KMAT[:, :n_k]
    r_a = ctx.y[a].astype(np.float64) - ctx.v4p_hc_o[a]
    v_a = ctx.v4p_hc_o[a]
    stealth = ((ctx.y[a] == 1) & (ctx.v4p_hc_o[a] < 0.10)).astype(np.float64)
    feats_a, feats_b, feats_t = [], [], []
    for j in range(n_k):
        ncat = int(ctx.NCAT[j])
        Ka, Kb, Kt = K[a, j], K[b, j], K[test_idx, j]
        cnt = np.bincount(Ka, minlength=ncat).astype(np.float64) + 1e-9
        sum_r = np.bincount(Ka, weights=r_a, minlength=ncat).astype(np.float64)
        mean_r = sum_r / (cnt + 15.0)
        sum_v = np.bincount(Ka, weights=v_a, minlength=ncat).astype(np.float64)
        sum_v2 = np.bincount(Ka, weights=v_a * v_a, minlength=ncat).astype(np.float64)
        std_v = np.sqrt(np.maximum(sum_v2 / cnt - (sum_v / cnt) ** 2, 0))
        stealth_c = np.bincount(Ka, weights=stealth, minlength=ncat).astype(np.float64)
        for bucket, store in ((Ka, feats_a), (Kb, feats_b), (Kt, feats_t)):
            store.append(np.column_stack([
                mean_r[bucket], std_v[bucket], np.log1p(stealth_c[bucket]),
            ]).astype(np.float32))
    Fa, Fb, Ft = np.hstack(feats_a), np.hstack(feats_b), np.hstack(feats_t)
    Ea, Eb, Et = ctx.gb.te_fold(a, b, smooth=20.0, seed=fi % 5)
    Ea, Eb, Et = lgt(Ea[:, :n_k]).astype(np.float32), lgt(Eb[:, :n_k]).astype(np.float32), lgt(Et[:, :n_k]).astype(np.float32)
    Xa, Xb, Xt = np.hstack([Ea, Fa]), np.hstack([Eb, Fb]), np.hstack([Et, Ft])
    za, zb, zt = ctx.z_base_o[a], ctx.z_base_o[b], ctx.z_base_t
    xgb_b, xgb_t, _ = fit_xgb_bag(
        Xa, ctx.y[a], za, Xb, ctx.y[b], zb, Xt, zt,
        seeds_depth=[(42 + fi, 2), (7 + fi, 3)], n_rounds=280, lr=0.03,
    )
    return xgb_b, xgb_t


def builder_pb4_fold(ctx, a, b, fi, gamma: float = 0.15):
    test_idx = np.arange(ctx.n, ctx.n + ctx.nt)
    n_k = min(ctx.nkeys_base, 80)
    K = ctx.KMAT[:, :n_k]
    p_trans_b = expit(ctx.z_base_o[b])
    p_trans_t = expit(ctx.z_base_t)
    ya = ctx.y[a].astype(np.float64)
    ybar = float(ya.mean())
    U_idx = np.concatenate([b, test_idx])
    p_U = np.concatenate([p_trans_b, p_trans_t])
    Ea = np.zeros((len(a), n_k), np.float32)
    Eb = np.zeros((len(b), n_k), np.float32)
    Et = np.zeros((ctx.nt, n_k), np.float32)
    for j in range(n_k):
        ncat = int(ctx.NCAT[j])
        Ka = K[a, j]
        s_a = np.bincount(Ka, weights=ya, minlength=ncat).astype(np.float64)
        c_a = np.bincount(Ka, minlength=ncat).astype(np.float64)
        Ku = K[U_idx, j]
        s_u = np.bincount(Ku, weights=p_U, minlength=ncat).astype(np.float64)
        c_u = np.bincount(Ku, minlength=ncat).astype(np.float64)
        m = 15.0
        for ii, oo in StratifiedKFold(5, shuffle=True, random_state=fi).split(np.zeros(len(a)), ya):
            r = te_rate_counts(Ka[ii], ya[ii], ncat, float(ya[ii].mean()), m)
            Ea[oo, j] = eb_logit_eta(r[Ka[oo]], ybar).astype(np.float32)
        for idx_rows, p_rows, outE in ((b, p_trans_b, Eb), (test_idx, p_trans_t, Et)):
            Kj = K[idx_rows, j]
            num = s_a[Kj] + gamma * (s_u[Kj] - p_rows) + m * ybar
            den = c_a[Kj] + gamma * (c_u[Kj] - 1.0) + m
            rate = np.clip(num / np.maximum(den, 1e-9), EPS, 1 - EPS)
            outE[:, j] = eb_logit_eta(rate, ybar).astype(np.float32)
    za, zb, zt, _ = fit_stage1_glm(Ea, ctx.y[a], Eb, Et, C=0.08, top_pairs=300)
    xgb_b, xgb_t, _ = fit_xgb_bag(
        Ea, ctx.y[a], za, Eb, ctx.y[b], zb, Et, zt,
        seeds_depth=[(42 + fi, 2), (7 + fi, 3)], n_rounds=250, lr=0.03,
    )
    return 0.35 * zb + 0.65 * xgb_b, 0.35 * zt + 0.65 * xgb_t


def builder_pb2_fold(ctx, a, b, fi):
    # Simplified: TE + amount/velocity category z-scores
    Ea, Eb, Et = ctx.gb.te_fold(a, b, smooth=20.0, seed=fi % 5)
    n_k = min(70, Ea.shape[1])
    Ea, Eb, Et = lgt(Ea[:, :n_k]).astype(np.float32), lgt(Eb[:, :n_k]).astype(np.float32), lgt(Et[:, :n_k]).astype(np.float32)
    full = pd.concat([ctx.tr, ctx.te], ignore_index=True)
    cat = pd.factorize(full["merchant_category"].astype(str))[0].astype(np.int32)
    extra = []
    for col in ("amount_usd", "velocity_score", "merchant_risk_score", "account_balance_usd"):
        x = full[col].to_numpy(np.float64)
        ncat = int(cat.max()) + 1
        g_t = torch.tensor(cat, device=DEVICE, dtype=torch.long)
        x_t = torch.tensor(x, device=DEVICE, dtype=torch.float64)
        cnt = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, torch.ones_like(x_t))
        sm = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, x_t)
        sm2 = torch.zeros(ncat, device=DEVICE, dtype=torch.float64).scatter_add_(0, g_t, x_t * x_t)
        mu = (sm / cnt.clamp_min(1))[g_t]
        sd = ((sm2 / cnt.clamp_min(1) - (sm / cnt.clamp_min(1)) ** 2).clamp_min(0).sqrt())[g_t]
        z = ((x_t - mu) / (sd + 1e-6)).cpu().numpy().astype(np.float32)
        extra.append(z)
    Z = np.column_stack(extra)
    Xa = np.hstack([Ea, Z[a]])
    Xb = np.hstack([Eb, Z[b]])
    Xt = np.hstack([Et, Z[ctx.n :]])
    za, zb, zt = ctx.z_base_o[a], ctx.z_base_o[b], ctx.z_base_t
    xgb_b, xgb_t, _ = fit_xgb_bag(
        Xa, ctx.y[a], za, Xb, ctx.y[b], zb, Xt, zt,
        seeds_depth=[(42 + fi, 3), (7 + fi, 4)], n_rounds=250, lr=0.03,
    )
    return xgb_b, xgb_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    st = {
        "started_utc": utc(),
        "phase": "phase1_fold0",
        "no_submit": True,
        "baseline": BASELINE,
        "baseline_f0": BASELINE_F0,
        "f0_gate": F0_GATE,
        "playbooks": {},
        "status_line": "starting 5-playbook Fold-0 shootout",
    }
    save_state(st)
    log("=" * 88)
    log("GRANDMASTER 5-PLAYBOOK ORTHOGONAL TOURNAMENT — PHASE 1 FOLD-0 SHOOTOUT")
    log("=" * 88)

    # stop keepwarm to free VRAM
    os.system("pkill -f a40_keepwarm.py 2>/dev/null || true")
    time.sleep(1)

    ctx = Ctx()
    st["status_line"] = "Ctx ready; running PB1..PB5 on Fold-0"
    save_state(st)

    results = []
    runners = [
        ("PB1_KMAT450", playbook1_kmat450_f0),
        ("PB2_GroupbyCtx", playbook2_groupby_f0),
        ("PB3_ResidDist_Cat", playbook3_residual_dist_f0),
        ("PB4_Transductive", playbook4_transductive_f0),
        ("PB5_AUC_fobj", playbook5_auc_fobj_f0),
    ]
    for name, fn in runners:
        st["status_line"] = f"Fold-0 running {name}"
        st["active"] = name
        save_state(st)
        try:
            ev = fn(ctx)
            # strip heavy arrays for state
            slim = {k: v for k, v in ev.items() if k not in ("score_b", "score_t", "Xa", "Xb", "Xt", "za", "zb", "zt")}
            st["playbooks"][name] = slim
            # persist scores
            np.save(OUT / f"{name}_f0_score_b.npy", ev["score_b"])
            np.save(OUT / f"{name}_f0_score_t.npy", ev["score_t"])
            results.append(ev)
            save_state(st)
        except Exception as e:
            log(f"FAIL {name}: {e}\n{traceback.format_exc()}")
            st["playbooks"][name] = {"error": str(e), "test1_delta": -1.0}
            save_state(st)

    # Leaderboard
    print("\n" + "=" * 100)
    print("PHASE 1 FOLD-0 SHOOTOUT LEADERBOARD")
    print("=" * 100)
    print(f"{'Playbook':28s} | {'Solo F0 AUC':11s} | {'Test-1 ΔAUC':12s} | {'ᾱ':7s} | {'ρ vs V4p':9s} | {'KS':7s} | {'|Δρ|':7s} | {'Wall-s':7s} | {'Promote?'}")
    print("-" * 110)
    ranked = sorted([r for r in results if "test1_delta" in r], key=lambda x: x["test1_delta"], reverse=True)
    for r in ranked:
        promote = "YES" if r["test1_delta"] >= F0_GATE and (r.get("ks") is None or r["ks"] < 0.008) else (
            "SOFT" if r["test1_delta"] >= F0_GATE else "no"
        )
        ks = f"{r['ks']:.4f}" if r.get("ks") is not None else "N/A"
        dr = f"{r['abs_drho']:.5f}" if r.get("abs_drho") is not None else "N/A"
        print(
            f"{r['name']:28s} | {r['solo_f0_auc']:11.6f} | {r['test1_delta']:+12.3e} | "
            f"{r['mean_alpha']:+7.3f} | {r['rho_v4p']:+9.5f} | {ks:7s} | {dr:7s} | "
            f"{r.get('wall_s', 0):7.1f} | {promote}"
        )
    sys.stdout.flush()

    # Promote: all clearing gate, else top 3
    winners = [r for r in ranked if r["test1_delta"] >= F0_GATE]
    if len(winners) < 3:
        winners = ranked[:3]
    st["phase1_ranked"] = [
        {k: v for k, v in r.items() if k not in ("score_b", "score_t", "Xa", "Xb", "Xt", "za", "zb", "zt")}
        for r in ranked
    ]
    st["winners"] = [r["name"] for r in winners]
    st["phase"] = "phase2_5fold"
    st["status_line"] = f"Phase2 promoting {st['winners']}"
    save_state(st)

    builders = {
        "PB1_KMAT450": builder_pb1_fold,
        "PB2_GroupbyCtx": builder_pb2_fold,
        "PB3_ResidDist_Cat": builder_pb3_fold,
        "PB4_Transductive": builder_pb4_fold,
        "PB5_AUC_fobj": builder_pb5_fold,
    }

    phase2 = []
    for w in winners:
        name = w["name"]
        st["status_line"] = f"Phase2 5-fold {name}"
        st["active"] = name
        save_state(st)
        try:
            out = promote_5fold_simple(ctx, name, builders[name])
            phase2.append(out)
            st.setdefault("phase2", {})[name] = out
            save_state(st)
        except Exception as e:
            log(f"PHASE2 FAIL {name}: {e}\n{traceback.format_exc()}")
            st.setdefault("phase2", {})[name] = {"error": str(e)}
            save_state(st)

    # Super-stack: rank-average of winning OOF stacks + Test-1 on blend
    if phase2:
        preds = []
        for p in phase2:
            fp = OUT / f"{p['name']}_oof.npy"
            if fp.exists():
                preds.append(np.load(fp))
        if preds:
            blend = np.mean([rk(p) for p in preds], axis=0)
            res = score_candidate_vs_v4p_hc(ctx.y, ctx.v4p_hc_o, blend, ctx.folds)
            res.pop("stack_oof_scores", None)
            st["superstack"] = {
                "members": [p["name"] for p in phase2],
                "solo_blend_auc": float(fast_auc(ctx.y, blend)),
                "stack_delta": float(res["delta_auc"]),
                "stack_auc": float(res["stack_oof_auc"]),
                "mean_alpha": float(res["stack_mean_alpha"]),
            }
            np.save(OUT / "superstack_blend_oof.npy", blend)
            log(f"SUPERSTACK members={st['superstack']['members']} Δ={st['superstack']['stack_delta']:+.3e}")

    print("\n" + "=" * 100)
    print("PHASE 2 FIVE-FOLD + SUPER-STACK RESULTS")
    print("=" * 100)
    print(f"{'Playbook':28s} | {'Solo 5F OOF':11s} | {'Stack ΔAUC':12s} | {'ᾱ':7s} | {'Wins':5s} | {'KS':7s} | {'|Δρ|':8s} | {'Wall-s'}")
    print("-" * 100)
    for p in phase2:
        print(
            f"{p['name']:28s} | {p['solo_oof_auc']:11.6f} | {p['delta_auc']:+12.3e} | "
            f"{p['mean_alpha']:+7.3f} | {p['fold_win_rate']:5.0%} | {p['ks']:7.4f} | "
            f"{p['abs_drho']:8.5f} | {p['wall_s']:.1f}"
        )
    if "superstack" in st:
        ss = st["superstack"]
        print(f"\nSUPERSTACK {ss['members']}: blend_solo={ss['solo_blend_auc']:.6f} "
              f"Test-1 Δ={ss['stack_delta']:+.3e} stack={ss['stack_auc']:.6f} ᾱ={ss['mean_alpha']:+.4f}")
    sys.stdout.flush()

    st["phase"] = "DONE"
    st["status_line"] = "5-playbook tournament complete (NO_SUBMIT)"
    st["elapsed_min"] = round((time.time() - t_start) / 60.0, 2)
    st["spend_usd_est"] = round(st["elapsed_min"] / 60.0 * COST_HR, 4)
    save_state(st)
    (LOG / "5playbook_tournament_report.json").write_text(json.dumps(st, indent=2, default=str))
    log(f"DONE elapsed={st['elapsed_min']}min spend~${st['spend_usd_est']} NO_SUBMIT")

    # restart keepwarm
    os.system(
        "nohup /workspace/venv/bin/python -u /workspace/s6e11/scripts/a40_keepwarm.py "
        ">/workspace/s6e11/logs/overnight/a40_keepwarm.log 2>&1 &"
    )


if __name__ == "__main__":
    main()
