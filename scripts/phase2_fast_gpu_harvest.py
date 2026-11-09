#!/usr/bin/env python3
"""PHASE 2 — 55-Minute Leak-Free 5-Fold Fast GPU Harvest on Pod 5hz1ob0bd0n6oy (A40)

Runs three strictly 5-fold leak-free model families on StratifiedKFold(5, shuffle=True, random_state=42):
  2A. RankGauss + Ridge-Weighted GPU orig20k & OOF k-NN Retrieval (~10 min) -> oof/test_orig20k_knn.npy
  2B. 5-Fold TabM (Piecewise-Linear Quantile Embeddings n_bins=48 + k=32 Adapter Mini-Ensemble + SWA, ~25 min) -> oof/test_tabm_5f.npy
  2C. Stage-A Orthogonal h16 Manifold (PLSRegression(16)) & Shallow Interaction GBDT (~20 min) -> oof/test_xgb_fast_harvest.npy
"""
from __future__ import annotations

import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit, logit
from scipy.stats import rankdata, skew
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(os.environ.get("S6_ROOT", "/workspace/s6e11"))
OUT = ROOT / "preds/fast_gpu_harvest"
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
NUMS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
FEATS = NUMS + FLAGS + CATS
FLOATS = [
    "amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
    "velocity_score", "merchant_risk_score",
]


def log(*a) -> None:
    s = f"[{time.time() - T0:6.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "harvest.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def build_key_codes_and_ratios(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame):
    """Build integer key codes for train+test+orig20k (up to 108 keys) + continuous interaction ratios."""
    full = pd.concat([tr[FEATS], te[FEATS], og[FEATS]], ignore_index=True)
    K = {}
    for c in FEATS:
        K[f"v_{c}"] = full[c].astype(str).values if c in CATS else full[c].values
    for c in FLOATS:
        x = full[c].values.astype(np.float64)
        K[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
        K[f"int_{c}"] = np.floor(x).astype(np.int64)
        K[f"ten_{c}"] = np.floor(x / 10).astype(np.int64)
        K[f"dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
        s = pd.Series(np.floor(x).astype(np.int64)).astype(str)
        K[f"l1_{c}"] = (s.str.len() * 10 + s.str[0].astype(int)).values
        K[f"l2_{c}"] = (s.str.len() * 100 + s.str[:2].astype(int)).values

    # GPT-2 token keys on floats
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        for c in FLOATS:
            str_vals = full[c].astype(str)
            uniq = str_vals.unique()
            mp = {v: tok.encode(" " + v) for v in uniq}
            enc = str_vals.map(mp)
            K[f"t1_{c}"] = enc.map(lambda t: t[0] if len(t) > 0 else -1).values
            K[f"t2_{c}"] = enc.map(lambda t: t[1] if len(t) > 1 else -1).values
            K[f"tl_{c}"] = enc.map(lambda t: t[-1] if len(t) > 0 else -1).values
            K[f"tn_{c}"] = enc.map(len).values
    except Exception as e:
        log(f"token keys fallback: {e}")

    # 8 high-leverage joint keys (merchant_category x risk decile, foreign x ip_mismatch, etc.)
    for c1, c2 in [
        ("merchant_category", "auth_method"),
        ("merchant_category", "channel"),
        ("card_type", "auth_method"),
        ("is_foreign_transaction", "ip_country_mismatch"),
        ("used_vpn", "ip_country_mismatch"),
        ("is_new_merchant", "billing_shipping_mismatch"),
        ("cvv_retry_count", "auth_method"),
        ("prior_disputes", "is_ai_generated_scam_attempt"),
    ]:
        K[f"j_{c1}_{c2}"] = (full[c1].astype(str) + "_" + full[c2].astype(str)).values

    knames = list(K.keys())[:108]
    codes = [pd.factorize(pd.Series(K[n]))[0].astype(np.int32) for n in knames]
    KMAT = np.column_stack(codes)
    NCAT = np.array([c.max() + 1 for c in codes], dtype=np.int32)

    # Continuous interaction ratios + log transforms
    amt = np.log1p(np.maximum(full["amount_usd"].to_numpy(np.float64), 0.0))
    bal = np.log1p(np.maximum(full["account_balance_usd"].to_numpy(np.float64), 0.0))
    hrs = np.log1p(np.maximum(full["hours_since_last_txn"].to_numpy(np.float64), 0.0))
    dist = np.log1p(np.maximum(full["distance_from_home_km"].to_numpy(np.float64), 0.0))
    vel = full["velocity_score"].to_numpy(np.float64)
    mrisk = full["merchant_risk_score"].to_numpy(np.float64)
    tc24 = full["txn_count_last_24h"].to_numpy(np.float64)
    cvv = full["cvv_retry_count"].to_numpy(np.float64)
    pdisp = full["prior_disputes"].to_numpy(np.float64)
    cage = full["card_age_months"].to_numpy(np.float64)
    uage = full["customer_age"].to_numpy(np.float64)
    flags_sum = full[FLAGS].to_numpy(np.float64).sum(axis=1)

    RATIOS = np.column_stack([
        amt, bal, hrs, dist, vel, mrisk, tc24, cvv, pdisp, cage, uage, flags_sum,
        amt - bal,
        full["amount_usd"].to_numpy(np.float64) / (full["account_balance_usd"].to_numpy(np.float64) + 10.0),
        vel / (tc24 + 1.0),
        vel * mrisk,
        dist / (hrs + 0.5),
        cvv * mrisk,
        flags_sum * mrisk,
        cage / (uage * 12.0 + 1.0),
    ]).astype(np.float32)

    log(f"Built KMAT={KMAT.shape} (keys={len(knames)}) + RATIOS={RATIOS.shape}")
    return KMAT, NCAT, RATIOS


def fold_te_108(
    KMAT: np.ndarray,
    NCAT: np.ndarray,
    y_tr: np.ndarray,
    y_og: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ntr: int,
    nte: int,
    seed: int = 42,
    smooth: float = 20.0,
):
    """Strictly leak-free inner-5-fold OOF TE on train_idx `a`, applied to `b`, `test`, and `orig20k`."""
    Ka = KMAT[a]
    Kb = KMAT[b]
    Kt = KMAT[ntr : ntr + nte]
    Ko = KMAT[ntr + nte :]
    ya = y_tr[a].astype(np.float64)
    prior = float(ya.mean())
    n_keys = KMAT.shape[1]

    Ea = np.zeros((len(a), n_keys), dtype=np.float32)
    Eb = np.zeros((len(b), n_keys), dtype=np.float32)
    Et = np.zeros((nte, n_keys), dtype=np.float32)
    Eo = np.zeros((len(Ko), n_keys), dtype=np.float32)

    inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(a)), y_tr[a]))
    for j in range(n_keys):
        nc = int(NCAT[j])
        for i_tr, i_va in inner:
            s = np.bincount(Ka[i_tr, j], weights=ya[i_tr], minlength=nc)
            n = np.bincount(Ka[i_tr, j], minlength=nc)
            r_in = (s + prior * smooth) / (n + smooth)
            Ea[i_va, j] = r_in[Ka[i_va, j]]
        s_all = np.bincount(Ka[:, j], weights=ya, minlength=nc)
        n_all = np.bincount(Ka[:, j], minlength=nc)
        r_all = (s_all + prior * smooth) / (n_all + smooth)
        Eb[:, j] = r_all[Kb[:, j]]
        Et[:, j] = r_all[Kt[:, j]]
        Eo[:, j] = r_all[Ko[:, j]]

    eps = 1e-5
    return (
        logit(np.clip(Ea, eps, 1 - eps)).astype(np.float32),
        logit(np.clip(Eb, eps, 1 - eps)).astype(np.float32),
        logit(np.clip(Et, eps, 1 - eps)).astype(np.float32),
        logit(np.clip(Eo, eps, 1 - eps)).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Experiment 2A: RankGauss + Ridge-Weighted GPU orig20k & OOF k-NN Retrieval
# ---------------------------------------------------------------------------
@torch.no_grad()
def gpu_knn_features_12(
    Q: np.ndarray,
    R: np.ndarray,
    y_ref: np.ndarray,
    chunk_q: int = 4096,
    chunk_r: int = 65536,
) -> np.ndarray:
    """Compute 12 exact retrieval features per query row in Q against reference bank (R, y_ref) on CUDA:
      0..3: min L2 distance to top-1, mean top-5, mean top-15, mean top-50
      4..7: 1 - max cosine similarity to top-1, mean top-5, mean top-15, mean top-50
      8..10: inverse-distance kernel-smoothed fraud rate for tau in [0.5, 1.0, 2.0] (over top-50 L2 neighbors)
      11: d_pos / (d_neg + 1e-5) (distance to nearest is_fraud==1 vs nearest is_fraud==0)
    """
    Rt = torch.as_tensor(R, dtype=torch.float32, device=DEV)
    R_norm = F.normalize(Rt, p=2, dim=1)
    y_t = torch.as_tensor(y_ref, dtype=torch.float32, device=DEV)
    pos_mask = y_t > 0.5
    neg_mask = ~pos_mask
    R_pos = Rt[pos_mask]
    R_neg = Rt[neg_mask]

    nq = len(Q)
    out = np.zeros((nq, 12), dtype=np.float32)
    K_MAX = 50

    for qs in range(0, nq, chunk_q):
        qe = min(qs + chunk_q, nq)
        Qt = torch.as_tensor(Q[qs:qe], dtype=torch.float32, device=DEV)
        Q_norm = F.normalize(Qt, p=2, dim=1)
        b = qe - qs

        # Maintain running top-50 smallest L2 distances & their labels, and top-50 largest cosine sims
        best_l2 = torch.full((b, K_MAX), 1e9, dtype=torch.float32, device=DEV)
        best_l2_y = torch.zeros((b, K_MAX), dtype=torch.float32, device=DEV)
        best_cos = torch.full((b, K_MAX), -1e9, dtype=torch.float32, device=DEV)

        for rs in range(0, len(Rt), chunk_r):
            re = min(rs + chunk_r, len(Rt))
            Rc = Rt[rs:re]
            yc = y_t[rs:re]
            d_l2 = torch.cdist(Qt, Rc, p=2.0)
            k_here = min(K_MAX, re - rs)
            vals_l2, idx_l2 = torch.topk(d_l2, k=k_here, dim=1, largest=False)
            y_here = yc[idx_l2]

            comb_l2 = torch.cat([best_l2, vals_l2], dim=1)
            comb_y = torch.cat([best_l2_y, y_here], dim=1)
            top_l2, top_idx = torch.topk(comb_l2, k=K_MAX, dim=1, largest=False)
            best_l2 = top_l2
            best_l2_y = torch.gather(comb_y, 1, top_idx)

            sim_c = Q_norm @ R_norm[rs:re].T
            vals_c, _ = torch.topk(sim_c, k=k_here, dim=1, largest=True)
            comb_c = torch.cat([best_cos, vals_c], dim=1)
            best_cos, _ = torch.topk(comb_c, k=K_MAX, dim=1, largest=True)

        # Nearest pos and nearest neg L2 distance
        d_pos = torch.full((b,), 1e9, dtype=torch.float32, device=DEV)
        for rs in range(0, len(R_pos), chunk_r):
            d_pos = torch.minimum(d_pos, torch.cdist(Qt, R_pos[rs : rs + chunk_r], p=2.0).min(dim=1).values)
        d_neg = torch.full((b,), 1e9, dtype=torch.float32, device=DEV)
        for rs in range(0, len(R_neg), chunk_r):
            d_neg = torch.minimum(d_neg, torch.cdist(Qt, R_neg[rs : rs + chunk_r], p=2.0).min(dim=1).values)

        f = torch.zeros((b, 12), dtype=torch.float32, device=DEV)
        f[:, 0] = best_l2[:, 0]
        f[:, 1] = best_l2[:, :5].mean(dim=1)
        f[:, 2] = best_l2[:, :15].mean(dim=1)
        f[:, 3] = best_l2[:, :50].mean(dim=1)

        c_dist = 1.0 - best_cos
        f[:, 4] = c_dist[:, 0]
        f[:, 5] = c_dist[:, :5].mean(dim=1)
        f[:, 6] = c_dist[:, :15].mean(dim=1)
        f[:, 7] = c_dist[:, :50].mean(dim=1)

        # Kernel-smoothed fraud target rate over top-50 L2 neighbors
        d_scale = best_l2 / (best_l2[:, [14]] + 1e-5)
        for ti, tau in enumerate((0.5, 1.0, 2.0)):
            w = torch.exp(-d_scale / tau)
            f[:, 8 + ti] = (w * best_l2_y).sum(dim=1) / (w.sum(dim=1) + 1e-8)

        f[:, 11] = d_pos / (d_neg + 1e-5)
        out[qs:qe] = f.cpu().numpy()

    return out


def run_exp_2a(
    KMAT: np.ndarray,
    NCAT: np.ndarray,
    RATIOS: np.ndarray,
    y: np.ndarray,
    y_og: np.ndarray,
    folds: list,
    ntr: int,
    nte: int,
):
    log("=== Experiment 2A: RankGauss + Ridge-Weighted GPU orig20k & OOF k-NN Retrieval ===")
    oof_24 = np.zeros((ntr, 24), dtype=np.float32)
    test_24_folds = []

    Ra_all = RATIOS[:ntr]
    Rt_all = RATIOS[ntr : ntr + nte]
    Ro_all = RATIOS[ntr + nte :]

    for f, (a, b) in enumerate(folds):
        tf0 = time.time()
        Ea, Eb, Et, Eo = fold_te_108(KMAT, NCAT, y, y_og, a, b, ntr, nte, seed=42 + f)
        Xa_raw = np.hstack([Ea, Ra_all[a]])
        Xb_raw = np.hstack([Eb, Ra_all[b]])
        Xt_raw = np.hstack([Et, Rt_all])
        Xo_raw = np.hstack([Eo, Ro_all])

        # Fit QuantileTransformer strictly on train_idx + orig20k features
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=2000, random_state=42 + f, subsample=100000)
        qt.fit(np.vstack([Xa_raw, Xo_raw]))
        Za = qt.transform(Xa_raw).astype(np.float32)
        Zb = qt.transform(Xb_raw).astype(np.float32)
        Zt = qt.transform(Xt_raw).astype(np.float32)
        Zo = qt.transform(Xo_raw).astype(np.float32)

        # Fit L2 logistic regression on train_idx (Za, y[a]) to obtain w_ridge
        lr = LogisticRegression(C=0.05, solver="lbfgs", max_iter=200)
        lr.fit(Za, y[a])
        w_scale = np.sqrt(np.abs(lr.coef_[0]) + 1e-4).astype(np.float32)
        w_scale = w_scale / (np.mean(w_scale) + 1e-8)

        Za_w = Za * w_scale
        Zb_w = Zb * w_scale
        Zt_w = Zt * w_scale
        Zo_w = Zo * w_scale

        # (a) Against orig20k reference rows
        fb_orig = gpu_knn_features_12(Zb_w, Zo_w, y_og)
        ft_orig = gpu_knn_features_12(Zt_w, Zo_w, y_og)

        # (b) Against train_idx rows (use all positive rows in `a` + 120k stratified negative subsample for fast exact cdist)
        rng = np.random.RandomState(100 + f)
        pos_a = np.where(y[a] == 1)[0]
        neg_a = np.where(y[a] == 0)[0]
        sub_neg = rng.choice(neg_a, size=min(120000, len(neg_a)), replace=False)
        ref_idx = np.concatenate([pos_a, sub_neg])
        fb_tr = gpu_knn_features_12(Zb_w, Za_w[ref_idx], y[a][ref_idx])
        ft_tr = gpu_knn_features_12(Zt_w, Za_w[ref_idx], y[a][ref_idx])

        oof_24[b] = np.hstack([fb_orig, fb_tr])
        test_24_folds.append(np.hstack([ft_orig, ft_tr]))
        log(f"  [2A] fold {f} retrieval 24-feat built in {time.time() - tf0:.1f}s")

    test_24 = np.mean(test_24_folds, axis=0).astype(np.float32)
    np.save(OUT / "knn24_oof.npy", oof_24)
    np.save(OUT / "knn24_test.npy", test_24)

    # Fit 5-fold L2 Logistic + shallow GBDT wrapper (depth=4) on the 24-feature retrieval block
    import xgboost as xgb
    try:
        from catboost import CatBoostClassifier
        has_cb = True
    except Exception:
        has_cb = False

    oof_knn = np.zeros(ntr, dtype=np.float64)
    test_knn_folds = []
    for f, (a, b) in enumerate(folds):
        sc = StandardScaler()
        Xa_s = np.nan_to_num(sc.fit_transform(oof_24[a]), 0.0)
        Xb_s = np.nan_to_num(sc.transform(oof_24[b]), 0.0)
        Xt_s = np.nan_to_num(sc.transform(test_24), 0.0)

        lr_w = LogisticRegression(C=0.1, solver="lbfgs", max_iter=300).fit(Xa_s, y[a])
        p_lr_b = zrk(lr_w.decision_function(Xb_s))
        p_lr_t = zrk(lr_w.decision_function(Xt_s))

        if has_cb:
            cb = CatBoostClassifier(
                iterations=800, depth=4, learning_rate=0.03, l2_leaf_reg=15.0,
                eval_metric="AUC", random_seed=42 + f, verbose=False,
            )
            cb.fit(Xa_s, y[a], eval_set=(Xb_s, y[b]), early_stopping_rounds=60, verbose=False)
            p_tree_b = zrk(cb.predict_proba(Xb_s)[:, 1])
            p_tree_t = zrk(cb.predict_proba(Xt_s)[:, 1])
        else:
            dtr = xgb.DMatrix(Xa_s, label=y[a])
            dva = xgb.DMatrix(Xb_s, label=y[b])
            dte = xgb.DMatrix(Xt_s)
            bst = xgb.train(
                dict(max_depth=4, eta=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=15.0,
                     min_child_weight=50, objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda", seed=42 + f),
                dtr, num_boost_round=800, evals=[(dva, "v")], early_stopping_rounds=60, verbose_eval=False,
            )
            p_tree_b = zrk(bst.predict(dva))
            p_tree_t = zrk(bst.predict(dte))

        oof_knn[b] = 0.5 * p_lr_b + 0.5 * p_tree_b
        test_knn_folds.append(0.5 * p_lr_t + 0.5 * p_tree_t)

    test_knn = np.mean(test_knn_folds, axis=0)
    np.save(OUT / "oof_orig20k_knn.npy", oof_knn.astype(np.float64))
    np.save(OUT / "test_orig20k_knn.npy", test_knn.astype(np.float64))
    log(f"=== [2A DONE] oof_orig20k_knn Solo OOF AUC = {roc_auc_score(y, oof_knn):.6f} ===")
    return oof_24, test_24, oof_knn, test_knn


# ---------------------------------------------------------------------------
# Experiment 2B: 5-Fold TabM (PLE Quantile Embeddings n_bins=48 + k=32 Adapters + SWA)
# ---------------------------------------------------------------------------
def compute_ple_bins(Xa: np.ndarray, n_bins: int = 48) -> torch.Tensor:
    """Compute strictly fold-train quantile edges [F, n_bins + 1] on CUDA."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(Xa, qs, axis=0).T.astype(np.float32)  # [F, n_bins + 1]
    widths = np.maximum(edges[:, 1:] - edges[:, :-1], 1e-5)
    return torch.as_tensor(edges[:, :-1], device=DEV), torch.as_tensor(widths, device=DEV)


class TabM32(nn.Module):
    """Piecewise-Linear Quantile Embeddings (n_bins=48) -> 3 residual blocks (d=256, drop=0.15) with k=32 rank-1 adapters."""

    def __init__(self, n_features: int, n_bins: int = 48, d_model: int = 256, k_heads: int = 32, dropout: float = 0.15):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.d_model = d_model
        self.k = k_heads

        # Shared linear projection from flattened PLE [n_features * n_bins] -> d_model, modulated by k=32 input adapters
        in_dim = n_features * n_bins
        self.in_proj = nn.Linear(in_dim, d_model)
        self.in_r = nn.Parameter(torch.ones(k_heads, d_model) + 0.05 * torch.randn(k_heads, d_model))
        self.in_norm = nn.LayerNorm(d_model)

        # 3 Residual blocks with rank-1 k=32 modulation
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.fc1 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc2 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.r_scale = nn.ParameterList([
            nn.Parameter(torch.ones(k_heads, d_model) + 0.05 * torch.randn(k_heads, d_model)) for _ in range(3)
        ])
        self.s_bias = nn.ParameterList([nn.Parameter(torch.zeros(k_heads, d_model)) for _ in range(3)])
        self.drop = nn.Dropout(dropout)

        # k=32 independent output heads: [k, d_model] + [k]
        self.head_w = nn.Parameter(torch.randn(k_heads, d_model) * (1.0 / np.sqrt(d_model)))
        self.head_b = nn.Parameter(torch.zeros(k_heads))

    def forward(self, x_raw: torch.Tensor, edges_left: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        # x_raw: [B, F] -> PLE [B, F, n_bins] in [0, 1]
        ple = torch.clamp((x_raw.unsqueeze(-1) - edges_left.unsqueeze(0)) / widths.unsqueeze(0), 0.0, 1.0)
        flat = ple.reshape(x_raw.shape[0], -1)  # [B, F * n_bins]
        h0 = self.in_norm(F.silu(self.in_proj(flat)))  # [B, D]
        h = h0.unsqueeze(1) * self.in_r.unsqueeze(0)  # [B, K, D]

        for i in range(3):
            u = self.norms[i](h)
            u = F.silu(self.fc1[i](u))
            u = self.drop(u)
            u = self.fc2[i](u)
            u = u * self.r_scale[i].unsqueeze(0) + self.s_bias[i].unsqueeze(0)
            h = h + self.drop(u)

        # Output [B, K] logits
        logits = torch.einsum("bkd,kd->bk", h, self.head_w) + self.head_b.unsqueeze(0)
        return logits


def pairwise_ranking_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Margin/logistic pairwise ranking loss across all k=32 heads."""
    pos_mask = y > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return logits.sum() * 0.0
    pos_l = logits[pos_mask]  # [P, K]
    neg_l = logits[neg_mask]  # [N, K]
    # Sample up to 256 negatives for fast pairwise difference [P, N_sub, K]
    if len(neg_l) > 256:
        idx = torch.randperm(len(neg_l), device=logits.device)[:256]
        neg_l = neg_l[idx]
    diff = pos_l.unsqueeze(1) - neg_l.unsqueeze(0)  # [P, N_sub, K]
    return F.softplus(-diff).mean()


def run_exp_2b(
    KMAT: np.ndarray,
    NCAT: np.ndarray,
    RATIOS: np.ndarray,
    lm_feats_oof: np.ndarray,
    lm_feats_test: np.ndarray,
    y: np.ndarray,
    y_og: np.ndarray,
    folds: list,
    ntr: int,
    nte: int,
):
    log("=== Experiment 2B: 5-Fold TabM (PLE n_bins=48 + k=32 Adapter Mini-Ensemble + SWA) ===")
    oof_tabm_mean = np.zeros(ntr, dtype=np.float64)
    oof_tabm_std = np.zeros(ntr, dtype=np.float64)
    test_tabm_mean_folds = []
    test_tabm_std_folds = []

    Ra_all = RATIOS[:ntr]
    Rt_all = RATIOS[ntr : ntr + nte]

    epochs = 18
    batch_size = 2048

    for f, (a, b) in enumerate(folds):
        tf0 = time.time()
        torch.manual_seed(1000 + f)
        Ea, Eb, Et, _ = fold_te_108(KMAT, NCAT, y, y_og, a, b, ntr, nte, seed=42 + f)
        Xa = np.hstack([Ea, Ra_all[a], lm_feats_oof[a]]).astype(np.float32)
        Xb = np.hstack([Eb, Ra_all[b], lm_feats_oof[b]]).astype(np.float32)
        Xt = np.hstack([Et, Rt_all, lm_feats_test]).astype(np.float32)

        edges_left, widths = compute_ple_bins(Xa, n_bins=48)
        model = TabM32(n_features=Xa.shape[1], n_bins=48, d_model=256, k_heads=32, dropout=0.15).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)

        Xa_t = torch.as_tensor(Xa, dtype=torch.float32, device=DEV)
        ya_t = torch.as_tensor(y[a], dtype=torch.float32, device=DEV)

        swa_states = []
        n_a = len(a)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_a, device=DEV)
            for s in range(0, n_a, batch_size):
                idx = perm[s : s + batch_size]
                xb = Xa_t[idx]
                yb = ya_t[idx]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(xb, edges_left, widths)  # [B, 32]
                    bce = F.binary_cross_entropy_with_logits(lg.float(), yb.unsqueeze(1).expand_as(lg))
                    rnk = pairwise_ranking_loss(lg.float(), yb)
                    loss = 0.85 * bce + 0.15 * rnk
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if ep >= epochs - 4:
                swa_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        # Load SWA average of final 4 epochs
        avg_state = {k: torch.mean(torch.stack([st[k].float() for st in swa_states], dim=0), dim=0) for k in swa_states[0]}
        model.load_state_dict(avg_state)
        model.eval()

        def predict_heads(X_np: np.ndarray):
            preds_m, preds_s = [], []
            with torch.no_grad():
                for s in range(0, len(X_np), 4096):
                    xt = torch.as_tensor(X_np[s : s + 4096], dtype=torch.float32, device=DEV)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(xt, edges_left, widths).float()
                    preds_m.append(lg.mean(dim=1).cpu().numpy())
                    preds_s.append(lg.std(dim=1).cpu().numpy())
            return np.concatenate(preds_m), np.concatenate(preds_s)

        bm, bs = predict_heads(Xb)
        tm, ts = predict_heads(Xt)
        oof_tabm_mean[b] = bm
        oof_tabm_std[b] = bs
        test_tabm_mean_folds.append(tm)
        test_tabm_std_folds.append(ts)
        log(f"  [2B] fold {f} TabM SWA OOF AUC = {roc_auc_score(y[b], bm):.6f} ({time.time() - tf0:.1f}s)")
        del Xa_t, ya_t, model
        torch.cuda.empty_cache()

    test_tabm_mean = np.mean(test_tabm_mean_folds, axis=0)
    test_tabm_std = np.mean(test_tabm_std_folds, axis=0)
    np.save(OUT / "oof_tabm_5f.npy", oof_tabm_mean.astype(np.float64))
    np.save(OUT / "test_tabm_5f.npy", test_tabm_mean.astype(np.float64))
    np.save(OUT / "oof_tabm_5f_std.npy", oof_tabm_std.astype(np.float64))
    np.save(OUT / "test_tabm_5f_std.npy", test_tabm_std.astype(np.float64))
    log(f"=== [2B DONE] oof_tabm_5f Pooled OOF AUC = {roc_auc_score(y, oof_tabm_mean):.6f} ===")
    return oof_tabm_mean, test_tabm_mean, oof_tabm_std, test_tabm_std


# ---------------------------------------------------------------------------
# Experiment 2C: Stage-A Orthogonal h16 Manifold (PLS=16) & Shallow Interaction GBDT
# ---------------------------------------------------------------------------
def build_lm_dispersion_block(ntr: int, nte: int, folds: list, y: np.ndarray):
    """Build strictly 5-fold complete LM block (lm12, masa, gpt2_3ep, gpt2_rev + per-order dispersion where available)."""
    lm12_o = zrk(np.load(ROOT / "external/nov08_part12/lm_oof.npy"))
    lm12_t = zrk(np.load(ROOT / "external/nov08_part12/lm_test.npy"))
    masa_o = zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy"))
    masa_t = zrk(np.load(ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy"))
    g3_o = zrk(np.load(ROOT / "preds/gpt2_3ep/gpt2_3ep_oof.npy"))
    g3_t = zrk(np.load(ROOT / "preds/gpt2_3ep/gpt2_3ep_test.npy"))
    rev_o = zrk(np.load(ROOT / "preds/gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy"))
    rev_t = zrk(np.load(ROOT / "preds/gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"))

    # Per-order stats across K=6 orders of gpt2_rev (exists for all 5 folds)
    rev_ord_o = np.zeros((ntr, 10), dtype=np.float32)
    rev_ord_t_folds = []
    for f, (_, b) in enumerate(folds):
        ro = np.load(ROOT / f"preds/gpt2_rev/oof_{f}.npy").astype(np.float64)
        rt = np.load(ROOT / f"preds/gpt2_rev/test_{f}.npy").astype(np.float64)
        do = ro[:, :, 1] - ro[:, :, 0]  # [len(b), 6]
        dt = rt[:, :, 1] - rt[:, :, 0]  # [nte, 4]
        rev_ord_o[b, :6] = do
        rev_ord_o[b, 6] = do.mean(axis=1)
        rev_ord_o[b, 7] = do.std(axis=1)
        rev_ord_o[b, 8] = do.min(axis=1)
        rev_ord_o[b, 9] = do.max(axis=1)
        t_block = np.zeros((nte, 10), dtype=np.float32)
        t_block[:, :4] = dt
        t_block[:, 4:6] = dt[:, :2]
        t_block[:, 6] = dt.mean(axis=1)
        t_block[:, 7] = dt.std(axis=1)
        t_block[:, 8] = dt.min(axis=1)
        t_block[:, 9] = dt.max(axis=1)
        rev_ord_t_folds.append(t_block)
    rev_ord_t = np.mean(rev_ord_t_folds, axis=0)

    # Cross-LM mean, std, min, max, skew, and pairwise differences across the 4 complete 5-fold LMs
    stack_o = np.column_stack([lm12_o, masa_o, g3_o, rev_o])
    stack_t = np.column_stack([lm12_t, masa_t, g3_t, rev_t])

    def cross_stats(M: np.ndarray) -> np.ndarray:
        return np.column_stack([
            M,
            M.mean(axis=1),
            M.std(axis=1),
            M.min(axis=1),
            M.max(axis=1),
            skew(M, axis=1),
            M[:, 0] - M[:, 1],
            M[:, 0] - M[:, 2],
            M[:, 0] - M[:, 3],
            M[:, 1] - M[:, 2],
            M[:, 2] - M[:, 3],
        ]).astype(np.float32)

    lm_o = np.hstack([cross_stats(stack_o), rev_ord_o]).astype(np.float32)
    lm_t = np.hstack([cross_stats(stack_t), rev_ord_t]).astype(np.float32)
    return lm_o, lm_t


def run_exp_2c(
    KMAT: np.ndarray,
    NCAT: np.ndarray,
    RATIOS: np.ndarray,
    lm_o: np.ndarray,
    lm_t: np.ndarray,
    oof_24: np.ndarray,
    test_24: np.ndarray,
    oof_tabm_m: np.ndarray,
    test_tabm_m: np.ndarray,
    oof_tabm_s: np.ndarray,
    test_tabm_s: np.ndarray,
    y: np.ndarray,
    y_og: np.ndarray,
    folds: list,
    ntr: int,
    nte: int,
):
    log("=== Experiment 2C: Stage-A Orthogonal h16 Manifold (PLS=16) & Shallow Interaction GBDT ===")
    import xgboost as xgb

    # Neural + LM + kNN block for PLS(16)
    neural_o = np.column_stack([lm_o, zrk(oof_tabm_m), oof_tabm_s, oof_24]).astype(np.float32)
    neural_t = np.column_stack([lm_t, zrk(test_tabm_m), test_tabm_s, test_24]).astype(np.float32)

    oof_xgb = np.zeros(ntr, dtype=np.float64)
    test_xgb_folds = []

    Ra_all = RATIOS[:ntr]
    Rt_all = RATIOS[ntr : ntr + nte]

    for f, (a, b) in enumerate(folds):
        tf0 = time.time()
        Ea, Eb, Et, _ = fold_te_108(KMAT, NCAT, y, y_og, a, b, ntr, nte, seed=42 + f)
        # Select top-40 KMAT features by univariate correlation with y[a] inside fold f
        ya_c = y[a] - y[a].mean()
        ea_c = Ea - Ea.mean(axis=0)
        corrs = np.abs((ea_c * ya_c[:, None]).sum(axis=0) / (np.sqrt((ea_c**2).sum(axis=0)) + 1e-8))
        top40 = np.argsort(corrs)[::-1][:40]

        # Fit PLSRegression(n_components=16) strictly on fold train `a`
        sc = StandardScaler()
        Na_s = np.nan_to_num(sc.fit_transform(neural_o[a]), 0.0)
        Nb_s = np.nan_to_num(sc.transform(neural_o[b]), 0.0)
        Nt_s = np.nan_to_num(sc.transform(neural_t), 0.0)

        pls = PLSRegression(n_components=16, max_iter=500)
        pls.fit(Na_s, y[a])
        h16_a = pls.transform(Na_s).astype(np.float32)
        h16_b = pls.transform(Nb_s).astype(np.float32)
        h16_t = pls.transform(Nt_s).astype(np.float32)

        Xa = np.hstack([neural_o[a], h16_a, Ea[:, top40], Ra_all[a]]).astype(np.float32)
        Xb = np.hstack([neural_o[b], h16_b, Eb[:, top40], Ra_all[b]]).astype(np.float32)
        Xt = np.hstack([neural_t, h16_t, Et[:, top40], Rt_all]).astype(np.float32)

        # Bag over 3 seeds for stability
        pred_b = np.zeros(len(b), dtype=np.float64)
        pred_t = np.zeros(nte, dtype=np.float64)
        seeds = (42 + f, 142 + f, 242 + f)
        dtr = xgb.DMatrix(Xa, label=y[a])
        dva = xgb.DMatrix(Xb, label=y[b])
        dte = xgb.DMatrix(Xt)

        for sd in seeds:
            params = dict(
                max_depth=4,
                min_child_weight=80,
                colsample_bytree=0.5,
                subsample=0.8,
                reg_lambda=25.0,
                learning_rate=0.02,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device="cuda",
                seed=sd,
            )
            bst = xgb.train(
                params, dtr, num_boost_round=2500, evals=[(dva, "v")],
                early_stopping_rounds=100, verbose_eval=False,
            )
            pred_b += zrk(bst.predict(dva)) / len(seeds)
            pred_t += zrk(bst.predict(dte)) / len(seeds)

        oof_xgb[b] = pred_b
        test_xgb_folds.append(pred_t)
        log(f"  [2C] fold {f} xgb_fast_harvest OOF AUC = {roc_auc_score(y[b], pred_b):.6f} ({time.time() - tf0:.1f}s)")

    test_xgb = np.mean(test_xgb_folds, axis=0)
    np.save(OUT / "oof_xgb_fast_harvest.npy", oof_xgb.astype(np.float64))
    np.save(OUT / "test_xgb_fast_harvest.npy", test_xgb.astype(np.float64))
    log(f"=== [2C DONE] oof_xgb_fast_harvest Pooled OOF AUC = {roc_auc_score(y, oof_xgb):.6f} ===")
    return oof_xgb, test_xgb


def main() -> None:
    log(f"Starting Phase 2 Fast GPU Harvest on {DEV} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    og = pd.read_csv(ROOT / "original/credit_card_fraud_2026.csv")
    for df in (tr, te, og):
        df[FLAGS] = df[FLAGS].astype("int8")

    y = tr["is_fraud"].to_numpy(np.int32)
    y_og = og["is_fraud"].to_numpy(np.int32)
    ntr, nte = len(tr), len(te)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    KMAT, NCAT, RATIOS = build_key_codes_and_ratios(tr, te, og)
    lm_o, lm_t = build_lm_dispersion_block(ntr, nte, folds, y)

    # 2A
    oof_24, test_24, oof_knn, test_knn = run_exp_2a(KMAT, NCAT, RATIOS, y, y_og, folds, ntr, nte)
    gc.collect()
    torch.cuda.empty_cache()

    # 2B
    oof_tabm_m, test_tabm_m, oof_tabm_s, test_tabm_s = run_exp_2b(
        KMAT, NCAT, RATIOS, lm_o, lm_t, y, y_og, folds, ntr, nte
    )
    gc.collect()
    torch.cuda.empty_cache()

    # 2C
    oof_xgb, test_xgb = run_exp_2c(
        KMAT, NCAT, RATIOS, lm_o, lm_t, oof_24, test_24, oof_tabm_m, test_tabm_m, oof_tabm_s, test_tabm_s,
        y, y_og, folds, ntr, nte
    )

    summary = {
        "oof_orig20k_knn_auc": float(roc_auc_score(y, oof_knn)),
        "oof_tabm_5f_auc": float(roc_auc_score(y, oof_tabm_m)),
        "oof_xgb_fast_harvest_auc": float(roc_auc_score(y, oof_xgb)),
        "elapsed_sec": round(time.time() - T0, 1),
    }
    (OUT / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
    log("PHASE 2 FAST GPU HARVEST COMPLETE:", json.dumps(summary))


if __name__ == "__main__":
    main()
