#!/usr/bin/env python3
"""
GPU Matrix v2 runner: Dual-Tree residual-context evaluation + 8-rung lever generators.
STRICT NO_SUBMIT unless orchestrator Stage-9 gates clear.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("INPUT_DIR", str(ROOT))

from gpu_matrix_v2_core import (  # noqa: E402
    FOLD0_GATE,
    PROMOTE_GATE,
    align_to_oof_reference_with_tail_extrap,
    append_watchdog,
    assert_pod_allowed,
    bag_full_refit_test,
    calibrate_logit_to_prevalence,
    eval_fold0_2param_canary,
    fit_dual_tree_xgb,
    fit_lever_Direct_and_Ridge,
    intrinsic_2col,
    per_feature_ks_filter,
    residual_context_init_score,
    rlogit,
    save_lever_fold0,
    save_lever_oof,
    spearman_delta_rho,
    within_block_iqr_standardize,
)

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
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ctx:
    def __init__(self):
        from build_psuper_v4p_bank import build_and_cache_v4p_bank

        bank = build_and_cache_v4p_bank(force=False)
        self.y = bank["y"].astype(np.int32)
        self.folds = bank["folds"]
        self.z_champ_raw_o = bank["z_champ_oof"].astype(np.float64)
        self.z_champ_raw_t = bank["z_champ_test"].astype(np.float64)
        self.v4p_o = bank["v4p_hc_oof"].astype(np.float64)
        self.v4p_t = bank["v4p_hc_test"].astype(np.float64)
        # Rank-logit champ MUST be Platt-calibrated before use as base_margin
        a0, _ = self.folds[0]
        _, self.z_champ_o, self.z_champ_t = calibrate_logit_to_prevalence(
            self.z_champ_raw_o[a0], self.y[a0], self.z_champ_raw_o, self.z_champ_raw_t
        )
        # GLM from bank family column if present
        names = [str(x) for x in bank.get("names_v4p", [])]
        O = bank.get("O_v4p")
        T = bank.get("T_v4p")
        glm_name = next((n for n in ("glm_quad", "mega:glm_gen", "alex:Joint_GLM,_full_refit") if n in names), None)
        if glm_name is not None and O is not None:
            gi = names.index(glm_name)
            glm_raw_o = rlogit(O[:, gi])
            glm_raw_t = rlogit(T[:, gi])
            _, self.z_glm_o, self.z_glm_t = calibrate_logit_to_prevalence(
                glm_raw_o[a0], self.y[a0], glm_raw_o, glm_raw_t
            )
        else:
            self.z_glm_o = self.z_champ_o.copy()
            self.z_glm_t = self.z_champ_t.copy()
        tr = pd.read_csv(ROOT / "data" / "train.csv")
        te = pd.read_csv(ROOT / "data" / "test.csv")
        self.tr = tr
        self.te = te
        self.n, self.nt = len(tr), len(te)
        self.raw_tr = self._encode_raw(tr)
        self.raw_te = self._encode_raw(te)
        # Prefer gboost KMAT TE as bipartite context when available
        self.kmat_tr = None
        self.kmat_te = None
        try:
            import gboost as gb

            self.kmat_tr = gb.KMAT[: self.n].astype(np.float32)
            self.kmat_te = gb.KMAT[self.n :].astype(np.float32)
            self.kcnt_tr = gb.KCNT[: self.n].astype(np.float32)
            self.kcnt_te = gb.KCNT[self.n :].astype(np.float32)
            append_watchdog("kmat_loaded", {"cols": int(self.kmat_tr.shape[1])})
        except Exception as e:
            append_watchdog("kmat_unavailable", {"err": str(e)[:200]})
            self.kcnt_tr = self.kcnt_te = None
        corr = []
        for j, name in enumerate(NUMS + FLAGS):
            col = self.raw_tr[:, j]
            c = float(np.corrcoef(col, self.y)[0, 1]) if np.std(col) > 1e-12 else 0.0
            corr.append((abs(c), j, name))
        corr.sort(reverse=True)
        self.top32_idx = [j for _, j, _ in corr[:32]]
        self.coarse_idx = list(range(len(NUMS)))
        self.bipartite_idx = list(range(len(NUMS), len(NUMS) + len(FLAGS)))
        append_watchdog(
            "ctx_ready",
            {
                "n": self.n,
                "nt": self.nt,
                "cuda": torch.cuda.is_available(),
                "champ_cal_mean_p": float(float(np.mean(1.0 / (1.0 + np.exp(-self.z_champ_o))))),
                "y_mean": float(self.y.mean()),
            },
        )

    def _encode_raw(self, df: pd.DataFrame) -> np.ndarray:
        parts = [df[NUMS].to_numpy(np.float64)]
        parts.append(df[FLAGS].astype(np.float64).to_numpy())
        # hash cats to 16 dims
        for c in CATS:
            h = (pd.util.hash_pandas_object(df[c].astype(str), index=False).to_numpy() % 997).astype(np.float64)
            parts.append(h.reshape(-1, 1) / 997.0)
        X = np.hstack(parts)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------- Lever feature generators (Rung 1 = Paradigm 3) --------------------


class BiMaskedTransformer(nn.Module):
    def __init__(self, n_cols: int = 24, d_model: int = 64, nhead: int = 4, nlayers: int = 3):
        super().__init__()
        self.n_cols = n_cols
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_cols, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model * n_cols, 1)
        self.teacher = nn.Linear(d_model * n_cols, 1)

    def forward(self, x: torch.Tensor, mask_set: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C)
        B, C = x.shape
        h = self.embed(x.unsqueeze(-1)) + self.pos[:, :C]
        if mask_set is not None:
            # complementary 4-block: mask columns where c % 4 == mask_set
            m = torch.zeros(B, C, 1, device=x.device)
            idx = [c for c in range(C) if c % 4 == mask_set]
            m[:, idx] = 1.0
            h = h * (1.0 - m)
        z = self.enc(h).reshape(B, -1)
        return self.head(z).squeeze(-1), self.teacher(z).squeeze(-1)


def gen_L3_1_bimasked(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-block complementary BiMasked Transformer with teacher aux. Returns (F_a, F_b, F_te) residuals."""
    dev = device()
    cols = ctx.raw_tr[:, :24]
    cols_te = ctx.raw_te[:, :24]
    sc = StandardScaler()
    Xtr = sc.fit_transform(cols)
    Xte = sc.transform(cols_te)
    Xa, Xb = Xtr[fold_a], Xtr[fold_b]
    ya = ctx.y[fold_a]
    z_teach_a = ctx.z_champ_o[fold_a]
    z_teach_b = ctx.z_champ_o[fold_b]
    z_teach_te = ctx.z_champ_t

    model = BiMaskedTransformer(n_cols=24).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 8192 if torch.cuda.is_available() else 2048
    model.train()
    n_epochs = 3
    for epoch in range(n_epochs):
        perm = np.random.default_rng(42 + epoch).permutation(len(ya))
        for i in range(0, len(ya), bs):
            idx = perm[i : i + bs]
            xb = torch.tensor(Xa[idx], dtype=torch.float32, device=dev)
            yb = torch.tensor(ya[idx], dtype=torch.float32, device=dev)
            zt = torch.tensor(z_teach_a[idx], dtype=torch.float32, device=dev)
            loss = 0.0
            for mset in range(4):
                logit_m, teach_m = model(xb, mask_set=mset)
                loss = loss + F.binary_cross_entropy_with_logits(logit_m, yb)
                loss = loss + 0.20 * F.smooth_l1_loss(teach_m, zt)
            loss = loss / 4.0
            opt.zero_grad()
            loss.backward()
            opt.step()
        append_watchdog("L3.1_epoch", {"epoch": epoch + 1, "n_epochs": n_epochs})

    def predict(X: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=dev)
                acc = 0.0
                for mset in range(4):
                    logit_m, _ = model(xb, mask_set=mset)
                    acc = acc + logit_m
                outs.append((acc / 4.0).cpu().numpy())
        return np.concatenate(outs)

    pa, pb, pte = predict(Xa), predict(Xb), predict(Xte)
    # residual vs champ on fold
    da = pa - ctx.z_champ_o[fold_a]
    db = pb - ctx.z_champ_o[fold_b]
    dte = pte - ctx.z_champ_t
    # OOF-ref align fold_b / test using fold_a champ as reference
    db_al, dte_al = align_to_oof_reference_with_tail_extrap(db, dte, ctx.z_champ_o[fold_a])
    Fa = da.reshape(-1, 1)
    Fb = db_al.reshape(-1, 1)
    Fte = dte_al.reshape(-1, 1)
    return Fa, Fb, Fte


def gen_L3_2_contrastive(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K=4 contrastive causal transformer surrogate: 4 seed MLPs with InfoNCE-ish pairing."""
    dev = device()
    X = StandardScaler().fit_transform(ctx.raw_tr[:, :24])
    Xte = StandardScaler().fit(ctx.raw_tr[:, :24]).transform(ctx.raw_te[:, :24])
    outs_o = []
    outs_t = []
    for seed in range(4):
        torch.manual_seed(100 + seed)
        net = nn.Sequential(
            nn.Linear(24, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        Xa = torch.tensor(X[fold_a], dtype=torch.float32, device=dev)
        ya = torch.tensor(ctx.y[fold_a], dtype=torch.float32, device=dev)
        zt = torch.tensor(ctx.z_champ_o[fold_a], dtype=torch.float32, device=dev)
        for _ in range(8):
            for i in range(0, len(fold_a), 8192):
                sl = slice(i, i + 8192)
                logit_m = net(Xa[sl]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit_m, ya[sl]) + 0.2 * F.smooth_l1_loss(logit_m, zt[sl])
                # contrastive: pull same-label pairs
                if len(ya[sl]) > 64:
                    emb = net[0](Xa[sl])
                    emb = F.normalize(emb, dim=-1)
                    sim = emb @ emb.T
                    labels = ya[sl]
                    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
                    loss = loss + 0.05 * ((1 - same) * F.relu(sim - 0.2) - same * sim).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            po = net(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
            pt = net(torch.tensor(Xte, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        outs_o.append(po)
        outs_t.append(pt)
    oof = np.mean(outs_o, axis=0)
    te = np.mean(outs_t, axis=0)
    db, dte = align_to_oof_reference_with_tail_extrap(
        oof[fold_b] - ctx.z_champ_o[fold_b], te - ctx.z_champ_t, ctx.z_champ_o[fold_a]
    )
    return (oof[fold_a] - ctx.z_champ_o[fold_a]).reshape(-1, 1), db.reshape(-1, 1), dte.reshape(-1, 1)


def gen_L3_3_dec_int_llr(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dec|Int LLR from amount mantissa/integer digits as joint GBDT features."""
    def feats(df):
        amt = df["amount_usd"].to_numpy(np.float64)
        mant = amt - np.floor(amt)
        intp = np.floor(amt)
        loga = np.log1p(amt)
        dig = (np.floor(amt * 100) % 10).astype(np.float64)
        ben = np.log10(np.clip(amt, 1e-6, None))
        ben = ben - np.floor(ben)
        return np.column_stack([mant, intp, loga, dig, ben, df["velocity_score"], df["merchant_risk_score"]])

    Fa = feats(ctx.tr.iloc[fold_a])
    Fb = feats(ctx.tr.iloc[fold_b])
    Fte = feats(ctx.te)
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    Fb = Fb[:, keep]
    Fte = Fte[:, keep]
    return Fa, Fb, Fte


def gen_L3_4_bottleneck_xgb(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h64 bottleneck: compress raw to 64 via linear then use as F_lever."""
    X = ctx.raw_tr
    Xte = ctx.raw_te
    # PCA-like via torch SVD on fold_a
    xa = torch.tensor(X[fold_a], dtype=torch.float32, device=device())
    xa = xa - xa.mean(0, keepdim=True)
    try:
        U, S, V = torch.pca_lowrank(xa, q=64)
        W = V[:, :64]
    except Exception:
        W = torch.randn(X.shape[1], 64, device=device())
        W, _ = torch.linalg.qr(W)
    Fa = (torch.tensor(X[fold_a], dtype=torch.float32, device=device()) @ W).cpu().numpy()
    Fb = (torch.tensor(X[fold_b], dtype=torch.float32, device=device()) @ W).cpu().numpy()
    Fte = (torch.tensor(Xte, dtype=torch.float32, device=device()) @ W).cpu().numpy()
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    return Fa, Fb[:, keep], Fte[:, keep]


def gen_L3_5_logsumexp(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-sum-exp posterior over flag-state specialists."""
    flag_tr = ctx.tr[FLAGS].to_numpy(np.int8)
    flag_te = ctx.te[FLAGS].to_numpy(np.int8)
    # 8 random projections of flag bitmasks
    rng = np.random.default_rng(7)
    W = rng.normal(size=(6, 8))
    Ht = flag_tr @ W
    He = flag_te @ W
    # couple with champ
    Fo = np.column_stack([Ht, ctx.z_champ_o.reshape(-1, 1)])
    Ft = np.column_stack([He, ctx.z_champ_t.reshape(-1, 1)])
    # LSE feature
    lse_o = np.logaddexp.reduce(Ht, axis=1)
    lse_t = np.logaddexp.reduce(He, axis=1)
    Fo = np.column_stack([Fo, lse_o])
    Ft = np.column_stack([Ft, lse_t])
    Fa, Fb, Fte = Fo[fold_a], Fo[fold_b], Ft
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    return Fa, Fb[:, keep], Fte[:, keep]


# -------------------- Rung 2 generators (rescue CPU winners) --------------------


def gen_L4_1_rolling(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU-proven L4.1: contiguous row-id rolling residual windows + local jumps (within-block IQR)."""
    from scipy.special import expit

    n, nt = ctx.n, ctx.nt
    z = ctx.z_champ_o
    zt = ctx.z_champ_t
    p_o = expit(z)
    res_y = (ctx.y - p_o).astype(np.float64)
    X_tr = np.zeros((n, 6), dtype=np.float64)
    X_te = np.zeros((nt, 6), dtype=np.float64)
    # Use fold_a residuals only for OOF rolling on fold_b (and all folds for full matrix when fold_a is train)
    grid_res = np.zeros(n, dtype=np.float64)
    grid_cnt = np.zeros(n, dtype=np.float64)
    grid_res[fold_a] = res_y[fold_a]
    grid_cnt[fold_a] = 1.0
    cs_r = np.concatenate([[0.0], np.cumsum(grid_res)])
    cs_c = np.concatenate([[0.0], np.cumsum(grid_cnt)])
    for wi, w in enumerate((5, 20, 100)):
        lo = np.maximum(0, np.arange(n) - w)
        hi = np.minimum(n, np.arange(n) + w + 1)
        s_r = (cs_r[hi] - cs_r[lo]) - grid_res
        s_c = (cs_c[hi] - cs_c[lo]) - grid_cnt
        X_tr[:, wi] = s_r / (s_c + 5.0)
    X_tr[:, 3] = np.abs(z - np.roll(z, 1))
    X_tr[:, 4] = np.abs(z - np.roll(z, -1))
    X_tr[:, 5] = z - 0.5 * (np.roll(z, 1) + np.roll(z, -1))
    X_te[:, 3] = np.abs(zt - np.roll(zt, 1))
    X_te[:, 4] = np.abs(zt - np.roll(zt, -1))
    X_te[:, 5] = zt - 0.5 * (np.roll(zt, 1) + np.roll(zt, -1))
    Fo, Ft = within_block_iqr_standardize(X_tr, X_te)
    Fa, Fb, Fte, keep = per_feature_ks_filter(Fo[fold_a], Fo[fold_b], Ft)
    return Fa, Fb, Fte


def gen_L4_2_bulsah(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def bulsah(df):
        s = df["amount_usd"].astype(str)
        # sub-float string features
        has_dot = s.str.contains(r"\.").astype(np.float64)
        frac_len = s.str.split(".").str[-1].str.len().fillna(0).astype(np.float64)
        n_digits = s.str.replace(r"\D", "", regex=True).str.len().astype(np.float64)
        return np.column_stack([has_dot, frac_len, n_digits, np.log1p(df["amount_usd"])])

    Fa = bulsah(ctx.tr.iloc[fold_a])
    Fb = bulsah(ctx.tr.iloc[fold_b])
    Fte = bulsah(ctx.te)
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    return Fa, Fb[:, keep], Fte[:, keep]


def gen_L4_3_velocity_flag(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def xf(df):
        v = pd.cut(df["velocity_score"], bins=8, labels=False).astype(float).fillna(0).to_numpy()
        r = pd.cut(df["merchant_risk_score"], bins=8, labels=False).astype(float).fillna(0).to_numpy()
        flags = df[FLAGS].to_numpy(np.float64)
        cross = (v.reshape(-1, 1) * 8 + r.reshape(-1, 1)) * (1 + flags)
        return np.column_stack([v, r, cross, flags])

    Fa, Fb, Fte = xf(ctx.tr.iloc[fold_a]), xf(ctx.tr.iloc[fold_b]), xf(ctx.te)
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    return Fa, Fb[:, keep], Fte[:, keep]


def gen_L4_4_benford(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def bf(df):
        amt = np.clip(df["amount_usd"].to_numpy(np.float64), 1e-9, None)
        mantissa = 10 ** (np.log10(amt) - np.floor(np.log10(amt)))
        first = np.floor(mantissa).astype(np.float64)
        ratio = df["amount_usd"] / (df["account_balance_usd"] + 1.0)
        return np.column_stack([mantissa, first, np.log1p(ratio), df["cvv_retry_count"]])

    Fa, Fb, Fte = bf(ctx.tr.iloc[fold_a]), bf(ctx.tr.iloc[fold_b]), bf(ctx.te)
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    return Fa, Fb[:, keep], Fte[:, keep]


def gen_L4_5_keypair_te(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top key-pair EB TE using cats x flags (lightweight)."""
    def te_matrix(df, y=None, stats=None):
        out = []
        new_stats = {}
        for c in CATS[:3]:
            for f in FLAGS[:4]:
                key = df[c].astype(str) + "|" + df[f].astype(str)
                name = f"{c}__{f}"
                if stats is None:
                    # EB TE on fold_a only
                    tmp = pd.DataFrame({"k": key, "y": y})
                    g = tmp.groupby("k")["y"].agg(["sum", "count"])
                    prior = float(y.mean())
                    m = 20.0
                    te = (g["sum"] + prior * m) / (g["count"] + m)
                    new_stats[name] = te.to_dict()
                    mapped = key.map(lambda k: new_stats[name].get(k, prior)).to_numpy(np.float64)
                else:
                    prior = 0.12
                    mapped = key.map(lambda k, n=name: stats[n].get(k, prior)).to_numpy(np.float64)
                out.append(mapped)
        X = np.column_stack(out)
        return X, (new_stats if stats is None else stats)

    Fa, st = te_matrix(ctx.tr.iloc[fold_a], y=ctx.y[fold_a])
    Fb, _ = te_matrix(ctx.tr.iloc[fold_b], stats=st)
    Fte, _ = te_matrix(ctx.te, stats=st)
    Fa, keep = per_feature_ks_filter(Fa, Fte)
    return Fa, Fb[:, keep], Fte[:, keep]


def gen_L4_3_v2_stepgrid(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    """CPU Fold-0 winner L4.3 (+3.68e-5): velocity/risk step-grid parity forensics."""
    def grid_feats(df):
        vel = df["velocity_score"].to_numpy(np.float64)
        mr = df["merchant_risk_score"].to_numpy(np.float64)
        v_step = np.rint(vel / 0.1)
        m_step = np.rint(mr / 0.02)
        return np.column_stack([
            v_step % 2, v_step % 5, v_step % 10,
            m_step % 2, m_step % 5, m_step % 10,
            np.abs(vel - v_step * 0.1) * 1e4,
            np.abs(mr - m_step * 0.02) * 1e4,
            (v_step % 2) * (m_step % 2),
        ]).astype(np.float64)
    Fa, Fb, Fte = grid_feats(ctx.tr.iloc[fold_a]), grid_feats(ctx.tr.iloc[fold_b]), grid_feats(ctx.te)
    return per_feature_ks_filter(Fa, Fb, Fte)[:3]


def gen_L1_3_ot(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    """CPU Fold-0 winner L1.3: fold-anchored 1D quantile OT shifts vs orig20k."""
    orig_path = ROOT / "original" / "credit_card_fraud_2026.csv"
    if not orig_path.exists():
        orig_path = ROOT / "external_bank/fresh_ds_nov06eve/original/credit_card_fraud_2026.csv"
    orig = pd.read_csv(orig_path)
    cols = NUMS[:8]
    X_tr_list, X_te_list = [], []
    for c in cols:
        vo = np.sort(orig[c].to_numpy(np.float64))
        vtr = ctx.tr[c].to_numpy(np.float64)
        vte = ctx.te[c].to_numpy(np.float64)
        # Fit quantile map on fold_a only
        va = np.sort(vtr[fold_a])
        def shift(v, ref_src=va, ref_dst=vo):
            u = (rankdata(v, method="average") - 0.5) / max(len(v), 1)
            # map via fold_a ECDF then to orig quantiles — fold-anchored
            mapped = np.quantile(ref_dst, np.clip(u, 0.001, 0.999))
            return (v - mapped) / (float(np.std(ref_dst)) + 1e-12)
        X_tr_list.append(shift(vtr))
        X_te_list.append(shift(vte))
    Fo = np.column_stack(X_tr_list)
    Ft = np.column_stack(X_te_list)
    return per_feature_ks_filter(Fo[fold_a], Fo[fold_b], Ft)[:3]


def gen_L1_4_beta_prior(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    """Orig20k Beta prior TE on cats."""
    orig_path = ROOT / "original" / "credit_card_fraud_2026.csv"
    if not orig_path.exists():
        orig_path = ROOT / "external_bank/fresh_ds_nov06eve/original/credit_card_fraud_2026.csv"
    orig = pd.read_csv(orig_path)
    y_o = orig["is_fraud"].to_numpy(np.float64)
    prior = float(ctx.y[fold_a].mean())
    outs_tr, outs_te = [], []
    for c in CATS:
        # orig prior
        g_o = pd.DataFrame({"k": orig[c].astype(str), "y": y_o}).groupby("k")["y"].agg(["sum", "count"])
        # fold_a counts
        g_a = pd.DataFrame({"k": ctx.tr.iloc[fold_a][c].astype(str), "y": ctx.y[fold_a]}).groupby("k")["y"].agg(["sum", "count"])
        alpha = 8.0
        keys = set(g_o.index) | set(g_a.index)
        te = {}
        for k in keys:
            so, co = (float(g_o.loc[k, "sum"]), float(g_o.loc[k, "count"])) if k in g_o.index else (0.0, 0.0)
            sa, ca = (float(g_a.loc[k, "sum"]), float(g_a.loc[k, "count"])) if k in g_a.index else (0.0, 0.0)
            te[k] = (sa + alpha * so + 20 * prior) / (ca + alpha * co + 20)
        outs_tr.append(ctx.tr[c].astype(str).map(lambda k: te.get(k, prior)).to_numpy(np.float64))
        outs_te.append(ctx.te[c].astype(str).map(lambda k: te.get(k, prior)).to_numpy(np.float64))
    Fo = np.column_stack(outs_tr)
    Ft = np.column_stack(outs_te)
    return per_feature_ks_filter(Fo[fold_a], Fo[fold_b], Ft)[:3]


def gen_L5_1_cond_resid(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    floats6 = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"]
    Xf = ctx.tr[floats6].to_numpy(np.float64)
    Xt = ctx.te[floats6].to_numpy(np.float64)
    from sklearn.linear_model import Ridge
    from scipy.special import ndtri
    Z_tr = np.zeros_like(Xf)
    Z_te = np.zeros_like(Xt)
    for c_i in range(len(floats6)):
        other = [j for j in range(len(floats6)) if j != c_i]
        m = Ridge(alpha=10.0).fit(Xf[fold_a][:, other], Xf[fold_a, c_i])
        r_tr = Xf[:, c_i] - m.predict(Xf[:, other])
        r_te = Xt[:, c_i] - m.predict(Xt[:, other])
        Z_tr[:, c_i] = ndtri(np.clip(rk(r_tr), 1e-5, 1 - 1e-5))
        Z_te[:, c_i] = ndtri(np.clip(rk(r_te), 1e-5, 1 - 1e-5))
    return per_feature_ks_filter(Z_tr[fold_a], Z_tr[fold_b], Z_te)[:3]


def gen_L6_2_stealth_feats(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    """Features for stealth specialist: raw + calibrated champ margin band."""
    from scipy.special import expit
    p = expit(ctx.z_champ_o)
    pt = expit(ctx.z_champ_t)
    Fo = np.column_stack([ctx.raw_tr[:, : len(NUMS) + len(FLAGS)], p, (p < 0.12).astype(np.float64), np.abs(p - 0.12)])
    Ft = np.column_stack([ctx.raw_te[:, : len(NUMS) + len(FLAGS)], pt, (pt < 0.12).astype(np.float64), np.abs(pt - 0.12)])
    return per_feature_ks_filter(Fo[fold_a], Fo[fold_b], Ft)[:3]


def gen_L7_4_blind(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    keep = [c for c in NUMS + FLAGS if c not in ("merchant_risk_score", "velocity_score", "is_ai_generated_scam_attempt")]
    Fo = ctx.tr[keep].to_numpy(np.float64)
    Ft = ctx.te[keep].to_numpy(np.float64)
    return per_feature_ks_filter(Fo[fold_a], Fo[fold_b], Ft)[:3]


def gen_L2_3_svd(ctx: Ctx, fold_a: np.ndarray, fold_b: np.ndarray):
    if ctx.kmat_tr is None:
        return gen_L3_4_bottleneck_xgb(ctx, fold_a, fold_b)
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import csr_matrix
    import gboost as gb
    N_tot, D_k = gb.KMAT.shape
    offsets = np.concatenate([[0], np.cumsum(gb.NCAT[:-1])])
    total_cols = int(np.sum(gb.NCAT))
    row_idx = np.repeat(np.arange(N_tot, dtype=np.int32), D_k)
    col_idx = (gb.KMAT + offsets.reshape(1, -1)).ravel().astype(np.int32)
    A = csr_matrix((np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)), shape=(N_tot, total_cols))
    U = TruncatedSVD(n_components=32, n_iter=6, random_state=42).fit_transform(A).astype(np.float64)
    Fo, Ft = U[: ctx.n], U[ctx.n :]
    return per_feature_ks_filter(Fo[fold_a], Fo[fold_b], Ft)[:3]


from gpu_matrix_v2_core import rk  # noqa: E402

LEVER_GENERATORS = {
    "P3_L3.1": gen_L3_1_bimasked,
    "P3_L3.2": gen_L3_2_contrastive,
    "P3_L3.3": gen_L3_3_dec_int_llr,
    "P3_L3.4": gen_L3_4_bottleneck_xgb,
    "P3_L3.5": gen_L3_5_logsumexp,
    "P4_L4.1": gen_L4_1_rolling,
    "P4_L4.2": gen_L4_2_bulsah,
    "P4_L4.3": gen_L4_3_v2_stepgrid,
    "P4_L4.4": gen_L4_4_benford,
    "P4_L4.5": gen_L4_5_keypair_te,
    "P1_L1.3": gen_L1_3_ot,
    "P1_L1.4": gen_L1_4_beta_prior,
    "P5_L5.1": gen_L5_1_cond_resid,
    "P6_L6.2": gen_L6_2_stealth_feats,
    "P7_L7.4": gen_L7_4_blind,
    "P2_L2.3": gen_L2_3_svd,
}


def _context_blocks(ctx: Ctx, a, b):
    """Build bipartite TE (Top-32 KMAT or flags) + coarse continuous context."""
    if ctx.kmat_tr is not None:
        # Use top-32 TE keys by |corr| with y on fold a, and coarse = high-support KCNT cols
        Ka = ctx.kmat_tr[a]
        corrs = []
        for j in range(Ka.shape[1]):
            c = float(np.corrcoef(Ka[:, j], ctx.y[a])[0, 1]) if float(np.std(Ka[:, j])) > 1e-12 else 0.0
            corrs.append((abs(c), j))
        corrs.sort(reverse=True)
        bip_idx = [j for _, j in corrs[:32]]
        if ctx.kcnt_tr is not None:
            support = ctx.kcnt_tr[a].mean(0)
            coarse_idx = list(np.argsort(-support)[:24])
        else:
            coarse_idx = bip_idx[:16]
        ctx_a = np.hstack([ctx.kmat_tr[a][:, bip_idx], ctx.raw_tr[a][:, : len(NUMS)]])
        ctx_b = np.hstack([ctx.kmat_tr[b][:, bip_idx], ctx.raw_tr[b][:, : len(NUMS)]])
        ctx_te = np.hstack([ctx.kmat_te[:, bip_idx], ctx.raw_te[:, : len(NUMS)]])
        n_bip, n_coarse = len(bip_idx), len(NUMS)
        return ctx_a, ctx_b, ctx_te, n_bip, n_coarse
    bip = [i for i, j in enumerate(ctx.top32_idx) if j in ctx.bipartite_idx][:8]
    coarse = [i for i, j in enumerate(ctx.top32_idx) if j in ctx.coarse_idx][:16]
    ctx_a = ctx.raw_tr[a][:, ctx.top32_idx]
    ctx_b = ctx.raw_tr[b][:, ctx.top32_idx]
    ctx_te = ctx.raw_te[:, ctx.top32_idx]
    return (
        np.hstack([ctx_a[:, bip], ctx_a[:, coarse]]),
        np.hstack([ctx_b[:, bip], ctx_b[:, coarse]]),
        np.hstack([ctx_te[:, bip], ctx_te[:, coarse]]),
        len(bip),
        len(coarse),
    )


def evaluate_lever_fold0(ctx: Ctx, paradigm: str, lever: str) -> dict:
    key = f"{paradigm}_{lever}"
    gen = LEVER_GENERATORS.get(key)
    if gen is None:
        return {"status": "missing_generator", "key": key, "passed_fold0": False}
    a, b = ctx.folds[0]
    t0 = time.time()
    Fa, Fb, Fte = gen(ctx, a, b)
    Fa, Fb, Fte, _ = per_feature_ks_filter(Fa, Fb, Fte)
    n_lever = Fa.shape[1]
    base = float(roc_auc_score(ctx.y[b], ctx.z_champ_raw_o[b]))  # rank AUC of champ (unchanged by Platt)

    # Path 0: direct F_lever canary (CPU-style) — never wash out forensic signal
    f_score = Fb.mean(axis=1) if Fb.shape[1] > 1 else Fb[:, 0]
    f_te = Fte.mean(axis=1) if Fte.shape[1] > 1 else Fte[:, 0]
    auc_f0, delta_f0_raw, _ = eval_fold0_2param_canary(ctx.z_champ_raw_o[b], f_score, ctx.y[b])

    init_a = residual_context_init_score(ctx.z_champ_o[a], ctx.z_glm_o[a])
    init_b = residual_context_init_score(ctx.z_champ_o[b], ctx.z_glm_o[b])
    init_te = residual_context_init_score(ctx.z_champ_t, ctx.z_glm_t)

    # Path Direct: lever-only Ridge + shallow CUDA XGB residuals
    try:
        d_r_b, d_r_te, d_x_b, d_x_te = fit_lever_Direct_and_Ridge(
            Fa, ctx.y[a], Fb, ctx.y[b], Fte, init_a, init_b, init_te, seed=42
        )
    except Exception as e:
        append_watchdog("direct_fail", {"key": key, "err": str(e)[:300]})
        d_r_b = d_x_b = f_score
        d_r_te = d_x_te = f_te

    # Path Dual-Tree with calibrated init + KMAT context
    ctx_a, ctx_b, ctx_te, n_bip, n_coarse = _context_blocks(ctx, a, b)
    Xa = np.hstack([Fa, ctx_a])
    Xb = np.hstack([Fb, ctx_b])
    Xte = np.hstack([Fte, ctx_te])
    try:
        dA_b, dA_te, dB_b, dB_te = fit_dual_tree_xgb(
            Xa, ctx.y[a], Xb, Xte, init_a, init_b, init_te,
            n_lever=n_lever, n_bipartite=n_bip, n_coarse=n_coarse, seed=42, y_b=ctx.y[b],
        )
    except Exception as e:
        append_watchdog("dual_tree_fail", {"key": key, "err": str(e)[:300]})
        dA_b = dB_b = d_x_b
        dA_te = dB_te = d_x_te

    # Score each residual candidate via 2-param canary on frozen champ
    candidates = {
        "f_raw": (f_score, f_te),
        "ridge": (d_r_b, d_r_te),
        "xgb_direct": (d_x_b, d_x_te),
        "treeA": (dA_b, dA_te),
        "treeB": (dB_b, dB_te),
        "blend": (0.5 * dA_b + 0.5 * d_x_b, 0.5 * dA_te + 0.5 * d_x_te),
    }
    best_name, best_auc, best_delta, best_db, best_dte = "f_raw", auc_f0, float(delta_f0_raw), f_score, f_te
    for name, (db, dte) in candidates.items():
        auc_c, dlt, _ = eval_fold0_2param_canary(ctx.z_champ_raw_o[b], db, ctx.y[b])
        if dlt > best_delta:
            best_name, best_auc, best_delta, best_db, best_dte = name, auc_c, float(dlt), db, dte

    ks = float(ks_2samp(best_db, best_dte).statistic)
    meta = {
        "key": key,
        "fold0_base": base,
        "fold0_auc": best_auc,
        "delta_auc": best_delta,
        "delta_f0_raw": float(delta_f0_raw),
        "best_path": best_name,
        "ks": ks,
        "elapsed_s": round(time.time() - t0, 1),
        "n_lever": n_lever,
        "passed_fold0": bool(best_delta >= FOLD0_GATE and ks < 0.05),
    }
    save_lever_fold0(paradigm, lever, best_db.astype(np.float64), best_dte.astype(np.float64), meta)
    append_watchdog("fold0_done", meta)
    return meta


def evaluate_lever_5fold(ctx: Ctx, paradigm: str, lever: str) -> dict:
    key = f"{paradigm}_{lever}"
    gen = LEVER_GENERATORS[key]
    oof = np.zeros(ctx.n, dtype=np.float64)
    te_folds = []
    full_refits = []
    for fi, (a, b) in enumerate(ctx.folds):
        Fa, Fb, Fte = gen(ctx, a, b)
        Fa, Fb, Fte, _ = per_feature_ks_filter(Fa, Fb, Fte)
        init_a = residual_context_init_score(ctx.z_champ_o[a], ctx.z_glm_o[a])
        init_b = residual_context_init_score(ctx.z_champ_o[b], ctx.z_glm_o[b])
        init_te = residual_context_init_score(ctx.z_champ_t, ctx.z_glm_t)
        d_r_b, d_r_te, d_x_b, d_x_te = fit_lever_Direct_and_Ridge(
            Fa, ctx.y[a], Fb, ctx.y[b], Fte, init_a, init_b, init_te, seed=42 + fi
        )
        ctx_a, ctx_b, ctx_te, n_bip, n_coarse = _context_blocks(ctx, a, b)
        Xa = np.hstack([Fa, ctx_a])
        Xb = np.hstack([Fb, ctx_b])
        Xte = np.hstack([Fte, ctx_te])
        try:
            dA_b, dA_te, dB_b, dB_te = fit_dual_tree_xgb(
                Xa, ctx.y[a], Xb, Xte, init_a, init_b, init_te,
                n_lever=Fa.shape[1], n_bipartite=n_bip, n_coarse=n_coarse, seed=42 + fi, y_b=ctx.y[b],
            )
            db = 0.35 * d_x_b + 0.35 * dA_b + 0.20 * dB_b + 0.10 * d_r_b
            dte = 0.35 * d_x_te + 0.35 * dA_te + 0.20 * dB_te + 0.10 * d_r_te
        except Exception:
            db, dte = d_x_b, d_x_te
        db, dte = align_to_oof_reference_with_tail_extrap(db, dte, ctx.z_champ_o[a])
        oof[b] = db
        te_folds.append(dte)
        append_watchdog("fold_done", {"key": key, "fold": fi, "auc_delta_std": float(np.std(db))})
    te_avg = np.mean(te_folds, axis=0)
    for seed in (11, 22, 33):
        a_all = np.arange(ctx.n)
        a0, b0 = ctx.folds[0]
        hold = b0[: max(2000, len(b0) // 15)]
        train_idx = np.setdiff1d(a_all, hold)
        Fa2, Fh, Fte2 = gen(ctx, train_idx, hold)
        Fa2, Fh, Fte2, _ = per_feature_ks_filter(Fa2, Fh, Fte2)
        init_tr = residual_context_init_score(ctx.z_champ_o[train_idx], ctx.z_glm_o[train_idx])
        init_h = residual_context_init_score(ctx.z_champ_o[hold], ctx.z_glm_o[hold])
        init_te = residual_context_init_score(ctx.z_champ_t, ctx.z_glm_t)
        _, dte_r, _, dte_x = fit_lever_Direct_and_Ridge(
            Fa2, ctx.y[train_idx], Fh, ctx.y[hold], Fte2, init_tr, init_h, init_te, seed=seed
        )
        full_refits.append(0.5 * dte_r + 0.5 * dte_x)
    te_bag = bag_full_refit_test(te_avg, full_refits, 0.75)
    save_lever_oof(paradigm, lever, oof, te_bag)
    base = float(roc_auc_score(ctx.y, ctx.z_champ_raw_o))
    X2 = intrinsic_2col(oof)
    cand = np.zeros(ctx.n)
    for a, b in ctx.folds:
        best_al, best_sc = 0.0, float(roc_auc_score(ctx.y[a], ctx.z_champ_raw_o[a]))
        for al in (-0.08, -0.04, -0.02, 0.01, 0.02, 0.04, 0.08, 0.12):
            sc = float(roc_auc_score(ctx.y[a], ctx.z_champ_raw_o[a] + al * X2[a, 0]))
            if sc > best_sc:
                best_sc, best_al = sc, al
        cand[b] = ctx.z_champ_raw_o[b] + best_al * X2[b, 0]
    new_auc = float(roc_auc_score(ctx.y, cand))
    delta = new_auc - base
    d_rho = spearman_delta_rho(oof, te_bag, ctx.v4p_o, ctx.v4p_t)
    promoted = bool(delta >= PROMOTE_GATE and d_rho < 0.0020)
    meta = {
        "key": key,
        "base_auc": base,
        "new_auc": new_auc,
        "delta_oof": delta,
        "delta_rho": d_rho,
        "promoted": promoted,
    }
    append_watchdog("fivefold_done", meta)
    return meta


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True, help="e.g. P3_L3.1")
    ap.add_argument("--stage", choices=["fold0", "fivefold"], default="fold0")
    ap.add_argument("--pod-id", default=os.environ.get("POD_ID"))
    args = ap.parse_args()
    assert_pod_allowed(args.pod_id)
    ctx = Ctx()
    paradigm, lever = args.lever.split("_", 1)
    if args.stage == "fold0":
        print(json.dumps(evaluate_lever_fold0(ctx, paradigm, lever), indent=2))
    else:
        print(json.dumps(evaluate_lever_5fold(ctx, paradigm, lever), indent=2))


if __name__ == "__main__":
    main()
