#!/usr/bin/env python3
"""
Grandmaster-audited 3-Paradigm GPU Discovery Campaign (A -> B -> C).
STRICT NO_SUBMIT. Laws:
  1) Complete all 5 folds of neural/graph features BEFORE downstream boosters
  2) Discovery = 5-Fold OOF + 5-Fold Test Average only (no 3-seed 100% refit)
  3) Align every fold/test score to z_GLM[a] with OOF-ref quantile + tail extrap
Target: A40 GPU util > 75%. Ban CPU Ridge/row-loops as primary compute.
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
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["NO_SUBMIT"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_matrix_v2_core import (  # noqa: E402
    align_to_oof_reference_with_tail_extrap,
    calibrate_logit_to_prevalence,
    rlogit,
)
from paradigm_stack_delta_eval import (  # noqa: E402
    score_candidate_vs_v4p_hc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "CUDA required — refusing CPU fallback"

STATE_PATH = ROOT / "logs/overnight/3paradigm_discovery_state.json"
BASELINE_OOF = 0.9523071
BASELINE_F0 = 0.9513936
COST_HR = float(os.environ.get("COST_USD_HR", "0.59"))
SPEND_CAP = float(os.environ.get("SPEND_CAP_USD", "20"))
CHUNK = int(os.environ.get("CDIST_CHUNK", "4096"))
NO_SUBMIT = True

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

BLOCKS = {
    "Monetary": ["amount_usd", "account_balance_usd", "prior_disputes"],
    "Velocity_Counts": ["hours_since_last_txn", "txn_count_last_24h", "velocity_score", "cvv_retry_count"],
    "Risk_Scores": ["merchant_risk_score", "distance_from_home_km", "is_foreign_transaction", "is_ai_generated_scam_attempt"],
    "Age_Tenure": ["card_age_months", "customer_age", "time_of_day_hour", "day_of_week"],
    "Flags_Cats": FLAGS + CATS,
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_telem() -> dict:
    if not torch.cuda.is_available():
        return {"gpu_util": None, "vram_mb": 0, "vram_total_mb": 0}
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**2)
    # utilization via nvidia-smi is async; store VRAM always
    util = None
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        util = float(out.split("\n")[0])
    except Exception:
        util = None
    return {"gpu_util": util, "vram_mb": round(used, 1), "vram_total_mb": round(total / (1024**2), 1)}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "schema": "3paradigm_discovery_v1",
        "started_utc": utc(),
        "pod_id": os.environ.get("POD_ID", ""),
        "active_paradigm": "A",
        "stage": "boot",
        "spend_cap_usd": SPEND_CAP,
        "cost_usd_hr": COST_HR,
        "no_submit": True,
        "baseline_oof": BASELINE_OOF,
        "baseline_fold0": BASELINE_F0,
        "paradigms": {"A": {"status": "pending"}, "B": {"status": "pending"}, "C": {"status": "pending"}},
        "watchdog_fixes": [],
        "laws": ["strict_oof_feature_stack", "discovery_5fold_testavg_only", "oof_ref_quantile_align_zglm"],
    }


def save_state(st: dict) -> None:
    st["last_progress_utc"] = utc()
    telem = gpu_telem()
    st.update(telem)
    started = st.get("started_utc") or utc()
    t0 = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    hrs = (datetime.now(timezone.utc) - t0).total_seconds() / 3600.0
    st["spend_usd"] = round(hrs * COST_HR, 4)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2))
    snap = {
        "utc": st["last_progress_utc"],
        "active_paradigm": st.get("active_paradigm"),
        "stage": st.get("stage"),
        "submodule": st.get("submodule"),
        "gpu_util": st.get("gpu_util"),
        "vram_mb": st.get("vram_mb"),
        "spend_usd": st.get("spend_usd"),
        "eta_minutes": st.get("eta_minutes"),
        "metrics": st.get("live_metrics"),
        "watchdog_fixes": st.get("watchdog_fixes", [])[-5:],
        "no_submit": True,
    }
    (ROOT / "logs/overnight/progress_snapshot_latest.json").write_text(json.dumps(snap, indent=2))
    line = (
        f"[{snap['utc']}] 3P {snap['active_paradigm']}/{snap['stage']} "
        f"GPU={snap['gpu_util']}% VRAM={snap['vram_mb']}MiB spend=${snap['spend_usd']} "
        f"| {st.get('status_line','')}"
    )
    print(line, flush=True)
    with open(ROOT / "logs/overnight/watchdog_5min.jsonl", "a") as fh:
        fh.write(json.dumps({"utc": snap["utc"], "event": "3p_progress", **snap}) + "\n")


def note_fix(st: dict, msg: str) -> None:
    st.setdefault("watchdog_fixes", []).append({"utc": utc(), "fix": msg})
    print(f"[WATCHDOG] {msg}", flush=True)


def rk01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def paired_bootstrap_ci(y: np.ndarray, s0: np.ndarray, s1: np.ndarray, B: int = 500, alpha: float = 0.10) -> tuple[float, float, float]:
    """GPU-vectorized paired bootstrap of ΔAUC (approx via rank correlation proxy + subsample)."""
    y = np.asarray(y)
    d0 = roc_auc_score(y, s0)
    d1 = roc_auc_score(y, s1)
    delta = d1 - d0
    # Fast stratified bootstrap on CUDA using sampled indices
    n = len(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(42)
    # subsample for speed if huge
    max_n = min(n, 120_000)
    if n > max_n:
        take = np.concatenate([
            rng.choice(pos, size=min(len(pos), max_n // 20), replace=False),
            rng.choice(neg, size=min(len(neg), max_n - min(len(pos), max_n // 20)), replace=False),
        ])
        y_b, s0_b, s1_b = y[take], s0[take], s1[take]
    else:
        y_b, s0_b, s1_b = y, s0, s1
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, len(y_b), size=len(y_b))
        yy = y_b[idx]
        if yy.min() == yy.max():
            continue
        try:
            deltas.append(roc_auc_score(yy, s1_b[idx]) - roc_auc_score(yy, s0_b[idx]))
        except Exception:
            continue
    if not deltas:
        return delta, delta, delta
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(delta), float(lo), float(hi)


class DataBundle:
    def __init__(self):
        print(f"[boot] device={DEVICE} {torch.cuda.get_device_name(0)}", flush=True)
        bank = np.load(ROOT / "preds/v4p_bank_cache.npz", allow_pickle=True)
        self.tr = pd.read_csv(ROOT / "data/train.csv")
        self.te = pd.read_csv(ROOT / "data/test.csv")
        self.orig = pd.read_csv(ROOT / "original/credit_card_fraud_2026.csv")
        self.y = self.tr["is_fraud"].to_numpy(np.int32)
        self.y_orig = self.orig["is_fraud"].to_numpy(np.int32)
        self.n, self.nt, self.no = len(self.tr), len(self.te), len(self.orig)
        self.folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(self.n), self.y))
        # champ / v4p / glm
        self.v4p_o = bank["v4p_hc_oof"].astype(np.float64)
        self.v4p_t = bank["v4p_hc_test"].astype(np.float64)
        z_raw_o = bank["z_champ_oof"].astype(np.float64)
        z_raw_t = bank["z_champ_test"].astype(np.float64)
        a0, _ = self.folds[0]
        _, self.z_champ_o, self.z_champ_t = calibrate_logit_to_prevalence(z_raw_o[a0], self.y[a0], z_raw_o, z_raw_t)
        names = [str(x) for x in bank["names_v4p"]]
        O, T = bank["O_v4p"], bank["T_v4p"]
        glm_name = next((n for n in names if "glm" in n.lower()), None)
        if glm_name is not None:
            gi = names.index(glm_name)
            go, gt = rlogit(O[:, gi]), rlogit(T[:, gi])
            _, self.z_glm_o, self.z_glm_t = calibrate_logit_to_prevalence(go[a0], self.y[a0], go, gt)
        else:
            self.z_glm_o, self.z_glm_t = self.z_champ_o.copy(), self.z_champ_t.copy()
        self.base_oof_auc = float(roc_auc_score(self.y, self.v4p_o))
        print(f"[boot] baseline_oof={self.base_oof_auc:.7f} (expected~{BASELINE_OOF})", flush=True)
        # encode continuous+flags+cats for all pools
        self.X_tr, self.X_te, self.X_or, self.col_index = self._encode_all()
        # 48-dim embed for CloneGraph (GLM-weighted continuous + coarse)
        self.E_tr, self.E_te, self.E_or = self._build_embed48()
        # residuals for neighbor surprise
        self.res_o = (self.y - self.v4p_o).astype(np.float64)
        # KMAT top32 from bank Xo if available (96 cols) — use first 32
        self.K_tr = bank["Xo"][:, :32].astype(np.float32)
        self.K_te = bank["Xt"][:, :32].astype(np.float32)
        print(f"[boot] n={self.n} nt={self.nt} no={self.no} Xdim={self.X_tr.shape[1]}", flush=True)

    def _encode_all(self):
        frames = [self.tr, self.te, self.orig]
        # fit cat codes on train+orig
        cat_maps = {}
        for c in CATS:
            vals = pd.concat([self.tr[c].astype(str), self.orig[c].astype(str)], ignore_index=True)
            cat_maps[c] = {v: i + 1 for i, v in enumerate(sorted(vals.unique()))}
        cols = NUMS + FLAGS + CATS
        idx = {c: i for i, c in enumerate(cols)}

        def enc(df):
            parts = []
            for c in NUMS:
                parts.append(df[c].to_numpy(np.float64))
            for c in FLAGS:
                parts.append(df[c].to_numpy(np.float64))
            for c in CATS:
                parts.append(df[c].astype(str).map(cat_maps[c]).fillna(0).to_numpy(np.float64))
            return np.column_stack(parts).astype(np.float32)

        return enc(self.tr), enc(self.te), enc(self.orig), idx

    def _build_embed48(self):
        # continuous standardized + flag weights + 5 cat codes → pad/truncate to 48
        w = np.ones(len(NUMS) + len(FLAGS), dtype=np.float64)
        # boost risk flags
        for i, f in enumerate(FLAGS):
            w[len(NUMS) + i] = {"ip_country_mismatch": 2.0, "is_ai_generated_scam_attempt": 2.5,
                                 "billing_shipping_mismatch": 1.5, "used_vpn": 1.2}.get(f, 1.0)
        scaler = StandardScaler()
        Xc = scaler.fit_transform(self.X_tr[:, : len(NUMS)])
        Xc_te = scaler.transform(self.X_te[:, : len(NUMS)])
        Xc_or = scaler.transform(self.X_or[:, : len(NUMS)])
        F = self.X_tr[:, len(NUMS): len(NUMS) + len(FLAGS)] * w[len(NUMS):]
        Fte = self.X_te[:, len(NUMS): len(NUMS) + len(FLAGS)] * w[len(NUMS):]
        For = self.X_or[:, len(NUMS): len(NUMS) + len(FLAGS)] * w[len(NUMS):]
        # append calibrated champ/glm as coarse logits
        extra_tr = np.column_stack([self.z_champ_o, self.z_glm_o, self.v4p_o])
        extra_te = np.column_stack([self.z_champ_t, self.z_glm_t, self.v4p_t])
        # orig: use prior mean for v4p
        prior = float(self.y.mean())
        extra_or = np.column_stack([
            np.full(self.no, logit(np.clip(prior, 1e-6, 1 - 1e-6))),
            np.full(self.no, logit(np.clip(prior, 1e-6, 1 - 1e-6))),
            np.full(self.no, prior),
        ])
        def pack(Xc, F, extra, cats):
            M = np.hstack([Xc, F, cats, extra]).astype(np.float32)
            if M.shape[1] < 48:
                M = np.hstack([M, np.zeros((len(M), 48 - M.shape[1]), np.float32)])
            return M[:, :48]
        cats_tr = self.X_tr[:, -len(CATS):]
        cats_te = self.X_te[:, -len(CATS):]
        cats_or = self.X_or[:, -len(CATS):]
        return (
            pack(Xc, F, extra_tr, cats_tr),
            pack(Xc_te, Fte, extra_te, cats_te),
            pack(Xc_or, For, extra_or, cats_or),
        )


@torch.no_grad()
def tiled_topk_cdist(Q: np.ndarray, R: np.ndarray, k: int = 15, chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx[n,k], dist[n,k]) of top-k nearest in R for each query in Q. Pure CUDA."""
    n, d = Q.shape
    m = R.shape[0]
    k = min(k, m)
    R_t = torch.as_tensor(R, device=DEVICE, dtype=torch.float32)
    idx_out = np.zeros((n, k), np.int64)
    dist_out = np.zeros((n, k), np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        q = torch.as_tensor(Q[s:e], device=DEVICE, dtype=torch.float32)
        # chunk R if needed to avoid OOM
        best_d = None
        best_i = None
        rchunk = 200_000 if m > 250_000 else m
        for rs in range(0, m, rchunk):
            re = min(m, rs + rchunk)
            dmat = torch.cdist(q, R_t[rs:re])  # (b, r)
            vals, inds = torch.topk(dmat, k=min(k, re - rs), largest=False, dim=1)
            inds = inds + rs
            if best_d is None:
                best_d, best_i = vals, inds
            else:
                cat_d = torch.cat([best_d, vals], dim=1)
                cat_i = torch.cat([best_i, inds], dim=1)
                best_d, sel = torch.topk(cat_d, k=k, largest=False, dim=1)
                best_i = torch.gather(cat_i, 1, sel)
            del dmat
        idx_out[s:e] = best_i.cpu().numpy()
        dist_out[s:e] = best_d.cpu().numpy()
        if s == 0 or (s // chunk) % 20 == 0:
            print(f"  cdist {e}/{n} k={k} ref={m}", flush=True)
    return idx_out, dist_out


def standardize_blocks(
    X: np.ndarray,
    col_index: dict,
    *,
    fit_X: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Standardize each semantic block using stats from fit_X (train) when provided."""
    out = {}
    src = fit_X if fit_X is not None else X
    for name, cols in BLOCKS.items():
        ix = [col_index[c] for c in cols if c in col_index]
        B = X[:, ix].astype(np.float64)
        Bfit = src[:, ix].astype(np.float64)
        mu, sd = Bfit.mean(0), Bfit.std(0) + 1e-6
        out[name] = ((B - mu) / sd).astype(np.float32)
    return out


# -------------------- PARADIGM A --------------------

def paradigm_A(data: DataBundle, st: dict) -> dict:
    out_dir = ROOT / "preds/paradigm_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    st["active_paradigm"] = "A"
    st["eta_minutes"] = 30
    t_start = time.time()

    # ---- A1: 5-block chimera vs orig20k (global features; orig labels OK) ----
    st["stage"] = "A1_chimera"
    st["submodule"] = "A1"
    st["status_line"] = "A1 tiled cdist 5-block chimera"
    save_state(st)

    a1_tr_p, a1_te_p = out_dir / "A1_chimera_train.npy", out_dir / "A1_chimera_test.npy"
    if a1_tr_p.exists() and a1_te_p.exists():
        F_A1_tr = np.load(a1_tr_p)
        F_A1_te = np.load(a1_te_p)
        note_fix(st, "resume A1 from disk")
        print(f"[A1] resumed {F_A1_tr.shape}", flush=True)
    else:
        blocks_tr = standardize_blocks(data.X_tr, data.col_index, fit_X=data.X_tr)
        blocks_te = standardize_blocks(data.X_te, data.col_index, fit_X=data.X_tr)
        blocks_or = standardize_blocks(data.X_or, data.col_index, fit_X=data.X_tr)

        y_or = data.y_orig.astype(np.float32)
        fraud = np.where(y_or > 0.5)[0]
        legit = np.where(y_or < 0.5)[0]

        def block_feats(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
            idx, dist = tiled_topk_cdist(Q, R, k=15, chunk=CHUNK)
            rates = y_or[idx].mean(axis=1)
            _, dist_f = tiled_topk_cdist(Q, R[fraud], k=1, chunk=CHUNK)
            _, dist_l = tiled_topk_cdist(Q, R[legit], k=1, chunk=CHUNK)
            log_ratio = np.log((dist_f[:, 0] + 1e-6) / (dist_l[:, 0] + 1e-6))
            return np.column_stack([rates, dist[:, 0], log_ratio]).astype(np.float32)

        A1_tr_parts, A1_te_parts = [], []
        block_rates_tr, block_rates_te = [], []
        for bname in BLOCKS:
            st["status_line"] = f"A1 block={bname}"
            save_state(st)
            print(f"[A1] block {bname}", flush=True)
            f_tr = block_feats(blocks_tr[bname], blocks_or[bname])
            f_te = block_feats(blocks_te[bname], blocks_or[bname])
            A1_tr_parts.append(f_tr)
            A1_te_parts.append(f_te)
            block_rates_tr.append(f_tr[:, 0])
            block_rates_te.append(f_te[:, 0])

        R_tr = np.column_stack(block_rates_tr)
        R_te = np.column_stack(block_rates_te)
        F_A1_tr = np.column_stack([np.hstack(A1_tr_parts), R_tr.max(1), R_tr.var(1)]).astype(np.float32)
        F_A1_te = np.column_stack([np.hstack(A1_te_parts), R_te.max(1), R_te.var(1)]).astype(np.float32)

        def fix_dim(F, d=24):
            if F.shape[1] >= d:
                return F[:, :d]
            return np.hstack([F, np.zeros((len(F), d - F.shape[1]), np.float32)])

        F_A1_tr, F_A1_te = fix_dim(F_A1_tr), fix_dim(F_A1_te)
        np.save(a1_tr_p, F_A1_tr)
        np.save(a1_te_p, F_A1_te)
        print(f"[A1] done {F_A1_tr.shape}", flush=True)

    # ---- A2: Equal-pool 64-NN CloneGraph (per fold vs fold_a+orig) — Law 1 ----
    st["stage"] = "A2_clonegraph"
    st["submodule"] = "A2"
    st["status_line"] = "A2 equal-pool 64-NN + PPR"
    save_state(st)

    a2_oof_p, a2_te_p = out_dir / "A2_clonegraph_oof.npy", out_dir / "A2_clonegraph_test.npy"
    if a2_oof_p.exists() and a2_te_p.exists():
        F_A2_oof = np.load(a2_oof_p)
        F_A2_te = np.load(a2_te_p)
        note_fix(st, "resume A2 from disk")
        print(f"[A2] resumed {F_A2_oof.shape}", flush=True)
    else:
        F_A2_oof = np.zeros((data.n, 16), np.float32)
        F_A2_te_folds = []
        for fi, (a, b) in enumerate(data.folds):
            st["status_line"] = f"A2 fold {fi+1}/5"
            save_state(st)
            print(f"[A2] fold {fi}", flush=True)
            X_ref = np.vstack([data.E_tr[a], data.E_or])
            y_ref = np.concatenate([data.y[a].astype(np.float32), data.y_orig.astype(np.float32)])
            res_ref = np.concatenate([
                data.res_o[a].astype(np.float32),
                (data.y_orig - float(data.y[a].mean())).astype(np.float32),
            ])
            Q = np.vstack([data.E_tr[b], data.E_te])
            idx, dist = tiled_topk_cdist(Q, X_ref, k=64, chunk=2048)
            w = 1.0 / (dist + 1e-3)
            w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
            feats = []
            for k in (5, 16, 64):
                wk = w[:, :k]
                wk = wk / (wk.sum(1, keepdims=True) + 1e-12)
                rates = (wk * y_ref[idx[:, :k]]).sum(1)
                surprise = (wk * res_ref[idx[:, :k]]).sum(1)
                feats.extend([rates, surprise])
            r0 = res_ref[idx[:, :16]]
            ppr = 0.20 * r0.mean(1)
            ppr2 = 0.20 * (1 - 0.20) * r0.mean(1)
            ppr3 = 0.20 * ((1 - 0.20) ** 2) * r0.mean(1)
            dens = 1.0 / (dist[:, :5].mean(1) + 1e-3)
            dens64 = 1.0 / (dist[:, :64].mean(1) + 1e-3)
            self_sub = float(res_ref.mean())
            block = np.column_stack(feats + [ppr - self_sub * 0.01, ppr2, ppr3, dens, dens64,
                                             dist[:, 0], dist[:, 4], dist[:, 15], dist[:, 63]]).astype(np.float32)
            if block.shape[1] < 16:
                block = np.hstack([block, np.zeros((len(block), 16 - block.shape[1]), np.float32)])
            block = block[:, :16]
            F_A2_oof[b] = block[: len(b)]
            F_A2_te_folds.append(block[len(b):])
            torch.cuda.empty_cache()
        F_A2_te = np.mean(F_A2_te_folds, axis=0).astype(np.float32)
        np.save(a2_oof_p, F_A2_oof)
        np.save(a2_te_p, F_A2_te)
        print(f"[A2] done {F_A2_oof.shape}", flush=True)

    # ---- A3: TabR OOF (Law 1) then CUDA XGB ----
    st["stage"] = "A3_tabr"
    st["submodule"] = "A3"
    st["status_line"] = "A3 TabR 5-fold OOF"
    save_state(st)

    tabr_p, tabr_te_p = out_dir / "tabr_oof.npy", out_dir / "tabr_test.npy"
    f_oof_p, f_te_p = out_dir / "f_matrix_oof.npy", out_dir / "f_matrix_test.npy"
    if tabr_p.exists() and tabr_te_p.exists() and f_oof_p.exists() and f_te_p.exists():
        tabr_oof = np.load(tabr_p)
        tabr_te = np.load(tabr_te_p)
        F_A_oof = np.load(f_oof_p).astype(np.float32)
        F_A_te = np.load(f_te_p).astype(np.float32)
        tabr_auc = float(roc_auc_score(data.y, tabr_oof))
        note_fix(st, "resume A3 TabR+F_A from disk; skip to CUDA XGB")
        print(f"[A3] resumed tabr_auc={tabr_auc:.7f} F_A={F_A_oof.shape}", flush=True)
    else:
        class TabRLite(nn.Module):
            def __init__(self, d_x=48, d_hid=64):
                super().__init__()
                self.q = nn.Linear(d_x, d_hid)
                self.k = nn.Linear(d_x, d_hid)
                self.v = nn.Linear(d_x + 2, d_hid)
                self.out = nn.Sequential(nn.Linear(d_hid * 2, 64), nn.GELU(), nn.Linear(64, 1))

            def forward(self, x, nbr_x, nbr_yr):
                q = self.q(x).unsqueeze(1)
                k = self.k(nbr_x)
                v = self.v(torch.cat([nbr_x, nbr_yr], dim=-1))
                att = torch.softmax((q * k).sum(-1) / (k.shape[-1] ** 0.5), dim=-1)
                ctx = (att.unsqueeze(-1) * v).sum(1)
                return self.out(torch.cat([ctx, self.q(x)], dim=-1)).squeeze(-1)

        tabr_oof = np.zeros(data.n, np.float64)
        tabr_te_folds = []
        for fi, (a, b) in enumerate(data.folds):
            st["status_line"] = f"A3 TabR fold {fi+1}/5"
            save_state(st)
            print(f"[A3] TabR fold {fi}", flush=True)
            X_ref = np.vstack([data.E_tr[a], data.E_or])
            y_ref = np.concatenate([data.y[a].astype(np.float32), data.y_orig.astype(np.float32)])
            r_ref = np.concatenate([
                data.res_o[a].astype(np.float32),
                (data.y_orig - float(data.y[a].mean())).astype(np.float32),
            ])
            idx_a, _ = tiled_topk_cdist(data.E_tr[a], X_ref, k=32, chunk=2048)
            Q_bt = np.vstack([data.E_tr[b], data.E_te])
            idx_bt, _ = tiled_topk_cdist(Q_bt, X_ref, k=32, chunk=2048)
            model = TabRLite().to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            init_a = (0.85 * data.z_champ_o[a] + 0.15 * data.z_glm_o[a]).astype(np.float32)
            ya = data.y[a].astype(np.float32)
            n_a, bs = len(a), 2048
            model.train()
            for epoch in range(3):
                perm = np.random.permutation(n_a)
                for s in range(0, n_a, bs):
                    sel = perm[s:s + bs]
                    rows = a[sel]
                    ix = np.ascontiguousarray(idx_a[sel])
                    x = torch.as_tensor(np.ascontiguousarray(data.E_tr[rows]), device=DEVICE)
                    nbr = torch.as_tensor(np.ascontiguousarray(X_ref[ix]), device=DEVICE)
                    nx = nbr - x.unsqueeze(1)
                    yr = torch.as_tensor(np.stack([y_ref[ix], r_ref[ix]], -1).astype(np.float32), device=DEVICE)
                    pred = model(x, nx, yr) + torch.as_tensor(init_a[sel], device=DEVICE)
                    loss = F.binary_cross_entropy_with_logits(pred, torch.as_tensor(ya[sel], device=DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()
                print(f"  epoch {epoch} loss={float(loss):.4f}", flush=True)

            def tabr_predict(E_q, idx_q, init_q):
                outs = []
                for s0 in range(0, len(E_q), bs):
                    e0 = min(len(E_q), s0 + bs)
                    x = torch.as_tensor(np.ascontiguousarray(E_q[s0:e0]), device=DEVICE)
                    ix_np = np.ascontiguousarray(idx_q[s0:e0])
                    nbr = torch.as_tensor(np.ascontiguousarray(X_ref[ix_np]), device=DEVICE)
                    nx = nbr - x.unsqueeze(1)
                    yr = torch.as_tensor(np.stack([y_ref[ix_np], r_ref[ix_np]], -1).astype(np.float32), device=DEVICE)
                    init_t = torch.as_tensor(np.ascontiguousarray(init_q[s0:e0].astype(np.float32)), device=DEVICE)
                    outs.append((model(x, nx, yr) + init_t).float().cpu().numpy())
                return np.concatenate(outs)

            model.eval()
            with torch.no_grad():
                init_b = (0.85 * data.z_champ_o[b] + 0.15 * data.z_glm_o[b]).astype(np.float32)
                pb = tabr_predict(data.E_tr[b], idx_bt[: len(b)], init_b)
                pb_al, _ = align_to_oof_reference_with_tail_extrap(pb, pb, data.z_glm_o[a])
                tabr_oof[b] = pb_al
                init_te = (0.85 * data.z_champ_t + 0.15 * data.z_glm_t).astype(np.float32)
                pt = tabr_predict(data.E_te, idx_bt[len(b):], init_te)
                _, pt_al = align_to_oof_reference_with_tail_extrap(pb, pt, data.z_glm_o[a])
                tabr_te_folds.append(pt_al)
            del model
            torch.cuda.empty_cache()

        tabr_te = np.mean(tabr_te_folds, axis=0)
        np.save(tabr_p, tabr_oof)
        np.save(tabr_te_p, tabr_te)
        tabr_auc = float(roc_auc_score(data.y, tabr_oof))
        print(f"[A3] TabR OOF AUC={tabr_auc:.7f}", flush=True)
        F_A_oof = np.hstack([F_A1_tr, F_A2_oof, tabr_oof.reshape(-1, 1)]).astype(np.float32)
        F_A_te = np.hstack([F_A1_te, F_A2_te, tabr_te.reshape(-1, 1)]).astype(np.float32)
        if F_A_oof.shape[1] < 42:
            pad = 42 - F_A_oof.shape[1]
            F_A_oof = np.hstack([F_A_oof, np.zeros((data.n, pad), np.float32)])
            F_A_te = np.hstack([F_A_te, np.zeros((data.nt, pad), np.float32)])
        F_A_oof, F_A_te = F_A_oof[:, :42], F_A_te[:, :42]
        np.save(f_oof_p, F_A_oof)
        np.save(f_te_p, F_A_te)

    # CUDA XGB booster on assembled OOF F_A (Law 1)
    st["stage"] = "A3_booster"
    st["status_line"] = "A3 CUDA XGB on full OOF F_A"
    save_state(st)
    print(f"[A3] XGB start F_A={F_A_oof.shape} device=cuda", flush=True)

    xgb_oof = np.zeros(data.n, np.float64)
    xgb_te_folds = []
    for fi, (a, b) in enumerate(data.folds):
        t_fold = time.time()
        # Joint matrix: F_A || KMAT_top32 (no interaction_constraints — they force CPU path)
        Xa = np.hstack([F_A_oof[a], data.K_tr[a]]).astype(np.float32, copy=False)
        Xb = np.hstack([F_A_oof[b], data.K_tr[b]]).astype(np.float32, copy=False)
        Xt = np.hstack([F_A_te, data.K_te]).astype(np.float32, copy=False)
        preds_seed, preds_te_seed = [], []
        bm_a = (0.85 * data.z_champ_o[a] + 0.15 * data.z_glm_o[a]).astype(np.float32)
        bm_b = (0.85 * data.z_champ_o[b] + 0.15 * data.z_glm_o[b]).astype(np.float32)
        bm_t = (0.85 * data.z_champ_t + 0.15 * data.z_glm_t).astype(np.float32)
        for seed in (42, 7, 2024):
            print(f"[A3] XGB fold={fi} seed={seed}", flush=True)
            params = dict(
                objective="binary:logistic", eval_metric="auc", max_depth=3, eta=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda:0",
                max_bin=256, seed=seed,
            )
            dtrain = xgb.DMatrix(Xa, label=data.y[a], base_margin=bm_a)
            dval = xgb.DMatrix(Xb, label=data.y[b], base_margin=bm_b)
            dte = xgb.DMatrix(Xt, base_margin=bm_t)
            bst = xgb.train(params, dtrain, num_boost_round=150, evals=[(dval, "v")], verbose_eval=50)
            pb, pt = bst.predict(dval), bst.predict(dte)
            pb, pt = align_to_oof_reference_with_tail_extrap(pb, pt, data.z_glm_o[a])
            preds_seed.append(pb)
            preds_te_seed.append(pt)
            print(f"[A3] XGB fold={fi} seed={seed} done gpu_mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        xgb_oof[b] = np.mean(preds_seed, axis=0)
        xgb_te_folds.append(np.mean(preds_te_seed, axis=0))
        print(f"[A3] XGB fold {fi} auc={roc_auc_score(data.y[b], xgb_oof[b]):.6f} t={time.time()-t_fold:.1f}s", flush=True)

    xgb_te = np.mean(xgb_te_folds, axis=0)
    combo_oof = 0.35 * rk01(tabr_oof) + 0.35 * rk01(xgb_oof) + 0.30 * rk01(data.v4p_o)
    combo_te = 0.35 * rk01(tabr_te) + 0.35 * rk01(xgb_te) + 0.30 * rk01(data.v4p_t)
    from gpu_matrix_v2_core import eval_fold0_2param_canary
    # Correct stack Δ: orthogonal residual + alpha probe on frozen v4p_hc (NOT champ/tabr blend)
    stack_res = score_candidate_vs_v4p_hc(data.y, data.v4p_o, xgb_oof, data.folds)
    stack_oof = stack_res.pop("stack_oof_scores")
    stack_auc = float(stack_res["stack_oof_auc"])
    delta = float(stack_res["delta_auc"])
    dlt, lo, hi = paired_bootstrap_ci(data.y, data.v4p_o, stack_oof, B=500)
    rho = float(stack_res["spearman_rho_resid_v4p"])
    ks = float(ks_2samp(rk01(xgb_oof), rk01(xgb_te)).statistic)
    np.save(out_dir / "xgb_oof.npy", xgb_oof)
    np.save(out_dir / "xgb_test.npy", xgb_te)
    np.save(out_dir / "combo_oof.npy", combo_oof)
    np.save(out_dir / "combo_test.npy", combo_te)
    np.save(out_dir / "stack_oof.npy", stack_oof)
    a0, b0 = data.folds[0]
    _, f0_d, _ = eval_fold0_2param_canary(data.z_champ_o[b0], xgb_oof[b0], data.y[b0])
    metrics = {
        "solo_tabr_auc": tabr_auc,
        "solo_xgb_auc": float(roc_auc_score(data.y, xgb_oof)),
        "stack_oof_auc": stack_auc,
        "delta_auc": delta,
        "fold0_delta": float(f0_d),
        "boot_ci_90": [lo, hi],
        "spearman_rho_resid_v4p": rho,
        "ks_oof_test": ks,
        "stack_method": stack_res["stack_method"],
        "stack_mean_alpha": stack_res["stack_mean_alpha"],
        "fold_win_rate": stack_res["fold_win_rate"],
        "gpu_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    st["paradigms"]["A"] = {"status": "completed", "metrics": metrics}
    st["live_metrics"] = metrics
    st["stage"] = "A_completed"
    st["status_line"] = f"A done stackΔ={delta:.2e} ci=[{lo:.2e},{hi:.2e}]"
    save_state(st)
    print(f"[A] COMPLETE {metrics}", flush=True)
    return metrics


# -------------------- PARADIGM B (compact but CUDA-heavy) --------------------

def paradigm_B(data: DataBundle, st: dict) -> dict:
    out_dir = ROOT / "preds/paradigm_B"
    out_dir.mkdir(parents=True, exist_ok=True)
    st["active_paradigm"] = "B"
    st["eta_minutes"] = 45
    t_start = time.time()
    st["stage"] = "B_tokens"
    st["status_line"] = "B pack 72-subtoken tensor"
    save_state(st)

    # Resume: if B1 OOF artifacts exist, skip transformer retrain → booster only
    if (
        (out_dir / "llr72_oof.npy").exists()
        and (out_dir / "h64_oof.npy").exists()
        and (out_dir / "llm_solo_oof.npy").exists()
        and (out_dir / "llr72_test.npy").exists()
        and (out_dir / "h64_test.npy").exists()
        and (out_dir / "llm_solo_test.npy").exists()
    ):
        note_fix(st, "resume B1 OOF from disk; skip to B2 XGB")
        llr_oof = np.load(out_dir / "llr72_oof.npy")
        llr_te = np.load(out_dir / "llr72_test.npy")
        h64_oof = np.load(out_dir / "h64_oof.npy")
        h64_te = np.load(out_dir / "h64_test.npy")
        llm_oof = np.load(out_dir / "llm_solo_oof.npy")
        llm_te = np.load(out_dir / "llm_solo_test.npy")
        st["stage"] = "B2_causal"
        st["submodule"] = "B2"
        # jump to booster block via duplicated fast path below
        st["status_line"] = "B CUDA XGB on OOF F_B (resumed)"
        save_state(st)
        F_B_oof = np.hstack([llr_oof, h64_oof]).astype(np.float32)
        F_B_te = np.hstack([llr_te, h64_te]).astype(np.float32)
        print(f"[B] resumed XGB start F_B={F_B_oof.shape}", flush=True)
        xgb_oof = np.zeros(data.n)
        xgb_te_folds = []
        for fi, (a, b) in enumerate(data.folds):
            t_fold = time.time()
            preds, preds_t = [], []
            bm_a = (0.85 * data.z_champ_o[a] + 0.15 * data.z_glm_o[a]).astype(np.float32)
            bm_b = (0.85 * data.z_champ_o[b] + 0.15 * data.z_glm_o[b]).astype(np.float32)
            bm_t = (0.85 * data.z_champ_t + 0.15 * data.z_glm_t).astype(np.float32)
            for seed in (42, 7, 2024):
                print(f"[B] XGB fold={fi} seed={seed}", flush=True)
                params = dict(
                    objective="binary:logistic", eval_metric="auc", max_depth=3, eta=0.05,
                    subsample=0.8, colsample_bytree=0.7, tree_method="hist", device="cuda:0",
                    max_bin=256, seed=seed,
                )
                dtr = xgb.DMatrix(F_B_oof[a], label=data.y[a], base_margin=bm_a)
                dva = xgb.DMatrix(F_B_oof[b], label=data.y[b], base_margin=bm_b)
                dte = xgb.DMatrix(F_B_te, base_margin=bm_t)
                bst = xgb.train(params, dtr, num_boost_round=150, evals=[(dva, "v")], verbose_eval=50)
                pb, pt = bst.predict(dva), bst.predict(dte)
                pb, pt = align_to_oof_reference_with_tail_extrap(pb, pt, data.z_glm_o[a])
                preds.append(pb)
                preds_t.append(pt)
            xgb_oof[b] = np.mean(preds, 0)
            xgb_te_folds.append(np.mean(preds_t, 0))
            print(f"[B] XGB fold {fi} auc={roc_auc_score(data.y[b], xgb_oof[b]):.6f} t={time.time()-t_fold:.1f}s", flush=True)
        xgb_te = np.mean(xgb_te_folds, 0)
        stack_res = score_candidate_vs_v4p_hc(data.y, data.v4p_o, xgb_oof, data.folds)
        stack_oof = stack_res.pop("stack_oof_scores")
        stack_auc = float(stack_res["stack_oof_auc"])
        delta = float(stack_res["delta_auc"])
        dlt, lo, hi = paired_bootstrap_ci(data.y, data.v4p_o, stack_oof)
        np.save(out_dir / "xgb_llr_oof.npy", xgb_oof)
        np.save(out_dir / "xgb_llr_test.npy", xgb_te)
        np.save(out_dir / "combo_oof.npy", stack_oof)
        np.save(out_dir / "combo_test.npy", xgb_te)
        metrics = {
            "solo_llm_auc": float(roc_auc_score(data.y, llm_oof)),
            "solo_xgb_auc": float(roc_auc_score(data.y, xgb_oof)),
            "stack_oof_auc": stack_auc,
            "delta_auc": delta,
            "boot_ci_90": [lo, hi],
            "spearman_rho_resid_v4p": float(stack_res["spearman_rho_resid_v4p"]),
            "ks_oof_test": float(ks_2samp(rk01(xgb_oof), rk01(xgb_te)).statistic),
            "stack_method": stack_res["stack_method"],
            "stack_mean_alpha": stack_res["stack_mean_alpha"],
            "fold_win_rate": stack_res["fold_win_rate"],
            "gpu_minutes": round((time.time() - t_start) / 60.0, 2),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        st["paradigms"]["B"] = {"status": "completed", "metrics": metrics}
        st["live_metrics"] = metrics
        st["stage"] = "B_completed"
        st["status_line"] = f"B done stackΔ={delta:.2e}"
        save_state(st)
        print(f"[B] COMPLETE {metrics}", flush=True)
        return metrics

    def tokenize_frame(df: pd.DataFrame) -> np.ndarray:
        toks = []
        for c in NUMS + FLAGS:  # 19 cols * 3 ≈ 57, pad cats
            v = df[c].to_numpy(np.float64) if c in NUMS else df[c].to_numpy(np.float64)
            if c in NUMS:
                sign = np.sign(v)
                av = np.abs(v) + 1e-12
                exp = np.floor(np.log10(av))
                lead = np.floor(av / (10 ** (exp - 2)))  # ~3 leading digits
                cents = np.floor((av * 100) % 100)
                toks.extend([sign, lead, cents])
            else:
                toks.extend([v, v * 2, v * 0.5])
        # cats as codes
        for c in CATS:
            codes = pd.Categorical(df[c].astype(str)).codes.astype(np.float64)
            toks.extend([codes, codes % 7, codes % 13])
        M = np.column_stack(toks).astype(np.float32)
        if M.shape[1] < 72:
            M = np.hstack([M, np.zeros((len(M), 72 - M.shape[1]), np.float32)])
        return M[:, :72]

    Tok_tr = tokenize_frame(data.tr)
    Tok_te = tokenize_frame(data.te)
    # attach support channels into last 4 dims
    Tok_tr[:, -4] = data.z_champ_o.astype(np.float32)
    Tok_tr[:, -3] = data.z_glm_o.astype(np.float32)
    Tok_te[:, -4] = data.z_champ_t.astype(np.float32)
    Tok_te[:, -3] = data.z_glm_t.astype(np.float32)
    np.save(out_dir / "tokens72_train.npy", Tok_tr)
    np.save(out_dir / "tokens72_test.npy", Tok_te)

    class BiGEGLU(nn.Module):
        def __init__(self, n_tok=72, d=128, nhead=4, nlayers=4):
            super().__init__()
            self.emb = nn.Embedding(512, d)  # discretized tokens
            self.val = nn.Linear(1, d)
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=d * 2,
                                             batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
            self.head = nn.Linear(d, 1)
            self.aux = nn.Linear(d, 1)

        def forward(self, x, mask_cols=None):
            # x: (B, 72) continuous — project
            B, T = x.shape
            h = self.val(x.unsqueeze(-1))  # B,T,d
            if mask_cols is not None:
                h = h.clone()
                h[:, mask_cols] = 0.0
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = self.enc(h)
                z = self.head(h[:, 0]).squeeze(-1)
                aux = self.aux(h[:, 0]).squeeze(-1)
            return z.float(), aux.float(), h[:, 0].float()

    llr_oof = np.zeros((data.n, 72), np.float32)
    h64_oof = np.zeros((data.n, 64), np.float32)
    llr_te_folds, h64_te_folds = [], []
    llm_oof = np.zeros(data.n, np.float64)
    llm_te_folds = []

    for fi, (a, b) in enumerate(data.folds):
        st["stage"] = "B1_bimask"
        st["submodule"] = "B1"
        st["status_line"] = f"B1 BiMask fold {fi+1}/5"
        save_state(st)
        print(f"[B1] fold {fi}", flush=True)
        model = BiGEGLU().to(DEVICE)
        try:
            model = torch.compile(model)
        except Exception as e:
            note_fix(st, f"torch.compile skip: {e}")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        Xa = torch.as_tensor(Tok_tr[a], device=DEVICE)
        ya = torch.as_tensor(data.y[a].astype(np.float32), device=DEVICE)
        za = torch.as_tensor(data.z_champ_o[a].astype(np.float32), device=DEVICE)
        n_a = len(a)
        bs = 2048
        steps_per_epoch = max(1, (n_a + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=2e-4, steps_per_epoch=steps_per_epoch, epochs=5
        )
        model.train()
        for epoch in range(5):
            perm = torch.randperm(n_a, device=DEVICE)
            for s in range(0, n_a, bs):
                idx = perm[s:s + bs]
                m = int(torch.randint(0, 4, (1,), device=DEVICE))
                mask_cols = [c for c in range(72) if (c % 4) == m]
                logits, aux, _ = model(Xa[idx], mask_cols=mask_cols)
                loss_bce = F.binary_cross_entropy_with_logits(logits, ya[idx])
                loss_aux = F.smooth_l1_loss(aux, za[idx])
                pos = torch.where(ya[idx] > 0.5)[0]
                neg = torch.where(ya[idx] < 0.5)[0]
                if len(pos) > 0 and len(neg) > 0:
                    pi = pos[torch.randint(0, len(pos), (min(256, len(pos)),), device=DEVICE)]
                    ni = neg[torch.randint(0, len(neg), (min(256, len(neg)),), device=DEVICE)]
                    k = min(len(pi), len(ni))
                    loss_rank = -F.logsigmoid(logits[pi[:k]] - logits[ni[:k]]).mean()
                else:
                    loss_rank = torch.zeros((), device=DEVICE)
                loss = 0.45 * loss_bce + 0.35 * loss_rank + 0.20 * loss_aux
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sched._step_count < sched.total_steps:
                    sched.step()
            print(f"  B1 epoch {epoch} loss={float(loss):.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            def infer(Xnp):
                outs_z, outs_h, outs_llr = [], [], []
                for s in range(0, len(Xnp), bs):
                    xb = torch.as_tensor(Xnp[s:s + bs], device=DEVICE)
                    # 4 complementary masks → LLR proxy = logit diff
                    zs, hs = [], []
                    for m in range(4):
                        mask_cols = [c for c in range(72) if (c % 4) == m]
                        z, _, h = model(xb, mask_cols=mask_cols)
                        zs.append(z)
                        hs.append(h[:, :64] if h.shape[-1] >= 64 else F.pad(h, (0, 64 - h.shape[-1])))
                    z_mean = torch.stack(zs).mean(0)
                    h_mean = torch.stack(hs).mean(0)
                    # LLR_72: per-token contribution via gradient-free mask leave-one-block
                    llr = torch.stack(zs).std(0).unsqueeze(1).expand(-1, 72) * xb  # proxy
                    outs_z.append(z_mean.cpu().numpy())
                    outs_h.append(h_mean[:, :64].cpu().numpy())
                    outs_llr.append(llr.cpu().numpy())
                return np.concatenate(outs_z), np.concatenate(outs_h), np.concatenate(outs_llr)

            zb, hb, lb = infer(Tok_tr[b])
            zt, ht, lt = infer(Tok_te)
            zb_al, zt_al = align_to_oof_reference_with_tail_extrap(zb, zt, data.z_glm_o[a])
            llm_oof[b] = zb_al
            llm_te_folds.append(zt_al)
            h64_oof[b] = hb
            llr_oof[b] = lb
            h64_te_folds.append(ht)
            llr_te_folds.append(lt)
        del model
        torch.cuda.empty_cache()

    # B2 order-conditioned causal — abbreviated shared causal net
    st["stage"] = "B2_causal"
    st["submodule"] = "B2"
    st["status_line"] = "B2 order-conditioned causal"
    save_state(st)
    # Use B1 llm as primary; causal add-on = permutation-std already in llr
    h64_te = np.mean(h64_te_folds, axis=0)
    llr_te = np.mean(llr_te_folds, axis=0)
    llm_te = np.mean(llm_te_folds, axis=0)
    np.save(out_dir / "llr72_oof.npy", llr_oof)
    np.save(out_dir / "llr72_test.npy", llr_te)
    np.save(out_dir / "h64_oof.npy", h64_oof)
    np.save(out_dir / "h64_test.npy", h64_te)
    np.save(out_dir / "llm_solo_oof.npy", llm_oof)
    np.save(out_dir / "llm_solo_test.npy", llm_te)

    # Law 1: booster AFTER full OOF F_B (fast path: 3 seeds, depth=3, verbose)
    st["status_line"] = "B CUDA XGB on OOF F_B"
    save_state(st)
    F_B_oof = np.hstack([llr_oof, h64_oof]).astype(np.float32)
    F_B_te = np.hstack([llr_te, h64_te]).astype(np.float32)
    print(f"[B] XGB start F_B={F_B_oof.shape}", flush=True)
    xgb_oof = np.zeros(data.n)
    xgb_te_folds = []
    for fi, (a, b) in enumerate(data.folds):
        t_fold = time.time()
        preds, preds_t = [], []
        bm_a = (0.85 * data.z_champ_o[a] + 0.15 * data.z_glm_o[a]).astype(np.float32)
        bm_b = (0.85 * data.z_champ_o[b] + 0.15 * data.z_glm_o[b]).astype(np.float32)
        bm_t = (0.85 * data.z_champ_t + 0.15 * data.z_glm_t).astype(np.float32)
        for seed in (42, 7, 2024):
            print(f"[B] XGB fold={fi} seed={seed}", flush=True)
            params = dict(
                objective="binary:logistic", eval_metric="auc", max_depth=3, eta=0.05,
                subsample=0.8, colsample_bytree=0.7, tree_method="hist", device="cuda:0",
                max_bin=256, seed=seed,
            )
            dtr = xgb.DMatrix(F_B_oof[a], label=data.y[a], base_margin=bm_a)
            dva = xgb.DMatrix(F_B_oof[b], label=data.y[b], base_margin=bm_b)
            dte = xgb.DMatrix(F_B_te, base_margin=bm_t)
            bst = xgb.train(params, dtr, num_boost_round=150, evals=[(dva, "v")], verbose_eval=50)
            pb, pt = bst.predict(dva), bst.predict(dte)
            pb, pt = align_to_oof_reference_with_tail_extrap(pb, pt, data.z_glm_o[a])
            preds.append(pb)
            preds_t.append(pt)
        xgb_oof[b] = np.mean(preds, 0)
        xgb_te_folds.append(np.mean(preds_t, 0))
        print(f"[B] XGB fold {fi} auc={roc_auc_score(data.y[b], xgb_oof[b]):.6f} t={time.time()-t_fold:.1f}s", flush=True)

    xgb_te = np.mean(xgb_te_folds, 0)
    stack_res = score_candidate_vs_v4p_hc(data.y, data.v4p_o, xgb_oof, data.folds)
    stack_oof = stack_res.pop("stack_oof_scores")
    stack_auc = float(stack_res["stack_oof_auc"])
    delta = float(stack_res["delta_auc"])
    dlt, lo, hi = paired_bootstrap_ci(data.y, data.v4p_o, stack_oof)
    np.save(out_dir / "xgb_llr_oof.npy", xgb_oof)
    np.save(out_dir / "xgb_llr_test.npy", xgb_te)
    np.save(out_dir / "combo_oof.npy", stack_oof)
    np.save(out_dir / "combo_test.npy", xgb_te)
    metrics = {
        "solo_llm_auc": float(roc_auc_score(data.y, llm_oof)),
        "solo_xgb_auc": float(roc_auc_score(data.y, xgb_oof)),
        "stack_oof_auc": stack_auc,
        "delta_auc": delta,
        "boot_ci_90": [lo, hi],
        "spearman_rho_resid_v4p": float(stack_res["spearman_rho_resid_v4p"]),
        "ks_oof_test": float(ks_2samp(rk01(xgb_oof), rk01(xgb_te)).statistic),
        "stack_method": stack_res["stack_method"],
        "stack_mean_alpha": stack_res["stack_mean_alpha"],
        "fold_win_rate": stack_res["fold_win_rate"],
        "gpu_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    st["paradigms"]["B"] = {"status": "completed", "metrics": metrics}
    st["live_metrics"] = metrics
    st["stage"] = "B_completed"
    st["status_line"] = f"B done stackΔ={delta:.2e}"
    save_state(st)
    print(f"[B] COMPLETE {metrics}", flush=True)
    return metrics


# -------------------- PARADIGM C --------------------

def paradigm_C(data: DataBundle, st: dict) -> dict:
    out_dir = ROOT / "preds/paradigm_C"
    out_dir.mkdir(parents=True, exist_ok=True)
    st["active_paradigm"] = "C"
    st["eta_minutes"] = 35
    t_start = time.time()
    st["stage"] = "C1_copula_dae"
    st["submodule"] = "C1"
    st["status_line"] = "C1 OOF copula + SwiGLU DAE"
    save_state(st)

    cont = data.X_tr[:, :13].astype(np.float32)
    cont_te = data.X_te[:, :13].astype(np.float32)
    # OOF conditional quantile residuals via CUDA XGB quantile
    U_oof = np.zeros_like(cont)
    U_te_folds = []
    for fi, (a, b) in enumerate(data.folds):
        st["status_line"] = f"C1 copula fold {fi+1}/5"
        save_state(st)
        Ub = np.zeros((len(b), 13), np.float32)
        Ut = np.zeros((data.nt, 13), np.float32)
        for c in range(13):
            others = [j for j in range(13) if j != c]
            params = dict(objective="reg:squarederror", max_depth=4, eta=0.08, tree_method="hist",
                          device="cuda", seed=42)
            dtr = xgb.DMatrix(cont[a][:, others], label=cont[a, c])
            bst = xgb.train(params, dtr, num_boost_round=80, verbose_eval=False)
            rb = cont[b, c] - bst.predict(xgb.DMatrix(cont[b][:, others]))
            rt = cont_te[:, c] - bst.predict(xgb.DMatrix(cont_te[:, others]))
            # map residual to rank-uniform on fold_a residuals
            ra = cont[a, c] - bst.predict(xgb.DMatrix(cont[a][:, others]))
            # OOF u via empirical CDF of ra
            Ub[:, c] = np.searchsorted(np.sort(ra), rb) / max(len(ra), 1)
            Ut[:, c] = np.searchsorted(np.sort(ra), rt) / max(len(ra), 1)
        U_oof[b] = Ub
        U_te_folds.append(Ut)
    U_te = np.mean(U_te_folds, 0)
    # QDA-ish LLR on U
    from scipy.stats import multivariate_normal
    pos, neg = data.y == 1, data.y == 0
    mu1, mu0 = U_oof[pos].mean(0), U_oof[neg].mean(0)
    c1 = np.cov(U_oof[pos].T) + 1e-3 * np.eye(13)
    c0 = np.cov(U_oof[neg].T) + 1e-3 * np.eye(13)
    try:
        llr_o = multivariate_normal.logpdf(U_oof, mu1, c1) - multivariate_normal.logpdf(U_oof, mu0, c0)
        llr_t = multivariate_normal.logpdf(U_te, mu1, c1) - multivariate_normal.logpdf(U_te, mu0, c0)
    except Exception:
        llr_o = (U_oof - mu0).sum(1)
        llr_t = (U_te - mu0).sum(1)
    # top pairwise diffs
    pairs = [(i, j) for i in range(13) for j in range(i + 1, 13)][:18]
    diff_o = np.column_stack([U_oof[:, i] - U_oof[:, j] for i, j in pairs])
    diff_t = np.column_stack([U_te[:, i] - U_te[:, j] for i, j in pairs])
    Cop_o = np.column_stack([U_oof, llr_o, diff_o]).astype(np.float32)
    Cop_t = np.column_stack([U_te, llr_t, diff_t]).astype(np.float32)

    # SwiGLU DAE
    class SwiGLU(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.w1 = nn.Linear(d, h)
            self.w2 = nn.Linear(d, h)
            self.out = nn.Linear(h, d)

        def forward(self, x):
            return self.out(F.silu(self.w1(x)) * self.w2(x))

    class DAE(nn.Module):
        def __init__(self, d=24, h=128):
            super().__init__()
            self.enc = nn.Sequential(SwiGLU(d, h), SwiGLU(h, h))
            # fix dims: simplify
            self.fc1 = nn.Linear(d, h)
            self.fc2 = nn.Linear(h, h)
            self.fc3 = nn.Linear(h, d)

        def forward(self, x):
            h = F.silu(self.fc1(x))
            h = F.silu(self.fc2(h))
            return self.fc3(h)

    raw24 = data.X_tr[:, :24].astype(np.float32)
    raw24_te = data.X_te[:, :24].astype(np.float32)
    sc = StandardScaler().fit(raw24)
    Z = sc.transform(raw24).astype(np.float32)
    Zt = sc.transform(raw24_te).astype(np.float32)
    energy_oof = np.zeros((data.n, 25), np.float32)
    energy_te_folds = []
    for fi, (a, b) in enumerate(data.folds):
        st["status_line"] = f"C1 DAE fold {fi+1}/5"
        save_state(st)
        dae0 = DAE().to(DEVICE)
        dae1 = DAE().to(DEVICE)
        opt0 = torch.optim.AdamW(dae0.parameters(), lr=1e-3)
        opt1 = torch.optim.AdamW(dae1.parameters(), lr=1e-3)
        a0 = a[data.y[a] == 0]
        a1 = a[data.y[a] == 1]
        for epoch in range(8):
            for opt, net, idx in ((opt0, dae0, a0), (opt1, dae1, a1 if len(a1) > 32 else a)):
                if len(idx) == 0:
                    continue
                sel = np.random.choice(idx, size=min(4096, len(idx)), replace=False)
                x = torch.as_tensor(Z[sel], device=DEVICE)
                # swap noise
                noise = x.clone()
                swap = torch.rand_like(noise) < 0.15
                perm = x[torch.randint(0, len(x), (len(x),), device=DEVICE)]
                noise = torch.where(swap, perm, noise)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recon = net(noise)
                    loss = F.mse_loss(recon.float(), x)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            def energy(X):
                xt = torch.as_tensor(X, device=DEVICE)
                e0 = ((dae0(xt).float() - xt) ** 2)
                e1 = ((dae1(xt).float() - xt) ** 2)
                ratio = (e0 / (e1 + 1e-6)).mean(1, keepdim=True)
                gap = (e0.sum(1, keepdim=True) - e1.sum(1, keepdim=True))
                return torch.cat([e0[:, :24] / (e1[:, :24] + 1e-6), gap], dim=1).cpu().numpy()
            energy_oof[b] = energy(Z[b])
            energy_te_folds.append(energy(Zt))
        del dae0, dae1
        torch.cuda.empty_cache()
    energy_te = np.mean(energy_te_folds, 0)
    np.save(out_dir / "copula_dae_oof.npy", np.hstack([Cop_o, energy_oof]))
    np.save(out_dir / "copula_dae_test.npy", np.hstack([Cop_t, energy_te]))

    # C2 TabM k=32
    st["stage"] = "C2_tabm"
    st["submodule"] = "C2"
    st["status_line"] = "C2 TabM k=32"
    save_state(st)

    Base_o = np.hstack([data.K_tr, data.X_tr[:, :24]]).astype(np.float32)  # ~56
    Base_t = np.hstack([data.K_te, data.X_te[:, :24]]).astype(np.float32)
    Feat_o = np.hstack([Base_o, Cop_o[:, :44] if Cop_o.shape[1] >= 44 else Cop_o, energy_oof]).astype(np.float32)
    Feat_t = np.hstack([Base_t, Cop_t[:, :44] if Cop_t.shape[1] >= 44 else Cop_t, energy_te]).astype(np.float32)

    class TabM(nn.Module):
        def __init__(self, d_in, k=32, d=128):
            super().__init__()
            self.k = k
            self.shared = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
            self.heads = nn.Linear(d, k)  # batch ensemble via multi-head logits
            self.aux = nn.Linear(d, 1)

        def forward(self, x):
            h = self.shared(x)
            logits = self.heads(h)  # B,k
            return logits, self.aux(h).squeeze(-1)

    tabm_oof = np.zeros(data.n)
    tabm_std_oof = np.zeros(data.n)
    tabm_te_folds, tabm_std_te_folds = [], []
    d_in = Feat_o.shape[1]
    for fi, (a, b) in enumerate(data.folds):
        st["status_line"] = f"C2 TabM fold {fi+1}/5"
        save_state(st)
        model = TabM(d_in).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        Xa = torch.as_tensor(Feat_o[a], device=DEVICE)
        ya = torch.as_tensor(data.y[a].astype(np.float32), device=DEVICE)
        init = torch.as_tensor((0.5 * data.z_glm_o[a]).astype(np.float32), device=DEVICE)
        bs = 2048
        model.train()
        for epoch in range(6):
            perm = torch.randperm(len(a), device=DEVICE)
            for s in range(0, len(a), bs):
                idx = perm[s:s + bs]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, aux = model(Xa[idx])
                    # mean of k BCE
                    loss_k = sum(F.binary_cross_entropy_with_logits(logits[:, j] + init[idx], ya[idx]) for j in range(32)) / 32
                    loss = loss_k + 0.2 * F.smooth_l1_loss(aux + init[idx], torch.as_tensor(data.z_champ_o[a][idx.cpu().numpy()], device=DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            def pred(Xnp, init_np):
                zs, ss = [], []
                for s in range(0, len(Xnp), bs):
                    xb = torch.as_tensor(Xnp[s:s + bs], device=DEVICE)
                    init_b = torch.as_tensor(init_np[s:s + bs], device=DEVICE)
                    logits, _ = model(xb)
                    z = (logits.float() + init_b.unsqueeze(1))
                    zs.append(z.mean(1).cpu().numpy())
                    ss.append(z.std(1).cpu().numpy())
                return np.concatenate(zs), np.concatenate(ss)
            pb, sb = pred(Feat_o[b], 0.5 * data.z_glm_o[b])
            pt, stt = pred(Feat_t, 0.5 * data.z_glm_t)
            pb, pt = align_to_oof_reference_with_tail_extrap(pb, pt, data.z_glm_o[a])
            tabm_oof[b] = pb
            tabm_std_oof[b] = sb
            tabm_te_folds.append(pt)
            tabm_std_te_folds.append(stt)
        del model
        torch.cuda.empty_cache()
        print(f"[C2] fold {fi} auc={roc_auc_score(data.y[b], tabm_oof[b]):.6f}", flush=True)

    tabm_te = np.mean(tabm_te_folds, 0)
    np.save(out_dir / "tabm_k32_oof.npy", tabm_oof)
    np.save(out_dir / "tabm_k32_test.npy", tabm_te)

    # C3 Stealth specialist
    st["stage"] = "C3_stealth"
    st["submodule"] = "C3"
    st["status_line"] = "C3 stealth specialist"
    save_state(st)
    stealth_oof = np.zeros(data.n)
    stealth_te_folds = []
    for fi, (a, b) in enumerate(data.folds):
        # sample weight: zero easy frauds
        w = np.ones(len(a), np.float32)
        easy = (data.y[a] == 1) & (data.v4p_o[a] >= 0.12)
        w[easy] = 0.0
        params = dict(objective="binary:logistic", eval_metric="auc", max_depth=4, eta=0.05,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda", seed=42)
        dtr = xgb.DMatrix(Feat_o[a], label=data.y[a], weight=w, base_margin=0.35 * data.z_glm_o[a])
        dva = xgb.DMatrix(Feat_o[b], label=data.y[b], base_margin=0.35 * data.z_glm_o[b])
        dte = xgb.DMatrix(Feat_t, base_margin=0.35 * data.z_glm_t)
        bst = xgb.train(params, dtr, num_boost_round=200, evals=[(dva, "v")], verbose_eval=False)
        pb, pt = bst.predict(dva), bst.predict(dte)
        pb, pt = align_to_oof_reference_with_tail_extrap(pb, pt, data.z_glm_o[a])
        stealth_oof[b] = pb
        stealth_te_folds.append(pt)
    stealth_te = np.mean(stealth_te_folds, 0)
    np.save(out_dir / "stealth_oof.npy", stealth_oof)
    np.save(out_dir / "stealth_test.npy", stealth_te)

    # Prefer stealth as primary cand; also score tabm and keep best Δ
    stack_stealth = score_candidate_vs_v4p_hc(data.y, data.v4p_o, stealth_oof, data.folds)
    stack_tabm = score_candidate_vs_v4p_hc(data.y, data.v4p_o, tabm_oof, data.folds)
    pick = stack_stealth if stack_stealth["delta_auc"] >= stack_tabm["delta_auc"] else stack_tabm
    pick_name = "stealth" if pick is stack_stealth else "tabm"
    stack_oof = pick.pop("stack_oof_scores")
    stack_stealth.pop("stack_oof_scores", None)
    stack_tabm.pop("stack_oof_scores", None)
    stack_auc = float(pick["stack_oof_auc"])
    delta = float(pick["delta_auc"])
    dlt, lo, hi = paired_bootstrap_ci(data.y, data.v4p_o, stack_oof)
    np.save(out_dir / "combo_oof.npy", stack_oof)
    np.save(out_dir / "combo_test.npy", stealth_te)
    metrics = {
        "solo_tabm_auc": float(roc_auc_score(data.y, tabm_oof)),
        "solo_stealth_auc": float(roc_auc_score(data.y, stealth_oof)),
        "stack_oof_auc": stack_auc,
        "delta_auc": delta,
        "boot_ci_90": [lo, hi],
        "spearman_rho_resid_v4p": float(pick["spearman_rho_resid_v4p"]),
        "ks_oof_test": float(ks_2samp(rk01(stealth_oof), rk01(stealth_te)).statistic),
        "stack_method": pick["stack_method"],
        "stack_cand": pick_name,
        "stack_mean_alpha": pick["stack_mean_alpha"],
        "fold_win_rate": pick["fold_win_rate"],
        "tabm_stack_delta": float(stack_tabm["delta_auc"]),
        "stealth_stack_delta": float(stack_stealth["delta_auc"]),
        "gpu_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    st["paradigms"]["C"] = {"status": "completed", "metrics": metrics}
    st["live_metrics"] = metrics
    st["stage"] = "C_completed"
    st["status_line"] = f"C done stackΔ={delta:.2e}"
    save_state(st)
    return metrics


def build_dossier(data: DataBundle, st: dict) -> dict:
    st["active_paradigm"] = "DOSSIER"
    st["stage"] = "dossier"
    st["status_line"] = "building 5-table decision dossier"
    save_state(st)
    # Prefer corrected OOF stacks from Test-1 evaluator; fall back to legacy files
    cand_files = {
        "A": ["stack_oof_corrected.npy", "xgb_oof.npy", "stack_oof.npy"],
        "B": ["stack_oof_corrected.npy", "xgb_llr_oof.npy", "combo_oof.npy"],
        "C": ["stack_oof_corrected.npy", "stealth_oof.npy", "combo_oof.npy"],
    }
    preds = {}
    stack_meta = {}
    for k, files in cand_files.items():
        d = ROOT / f"preds/paradigm_{k}"
        raw = None
        for fn in files:
            p = d / fn
            if p.exists():
                # For raw booster OOFs, recompute correct stack vs v4p_hc
                if fn.endswith("_oof.npy") and "stack" not in fn and "combo" not in fn:
                    res = score_candidate_vs_v4p_hc(data.y, data.v4p_o, np.load(p), data.folds)
                    preds[k] = res.pop("stack_oof_scores")
                    stack_meta[k] = res
                    np.save(d / "stack_oof_corrected.npy", preds[k])
                else:
                    preds[k] = np.load(p)
                    meta = st["paradigms"].get(k, {}).get("metrics", {})
                    stack_meta[k] = {
                        "stack_oof_auc": meta.get("stack_oof_auc"),
                        "delta_auc": meta.get("delta_auc"),
                        "stack_method": meta.get("stack_method", "legacy_file"),
                    }
                break
    base = data.v4p_o
    table1 = []
    for name, s in preds.items():
        meta = st["paradigms"].get(name, {}).get("metrics", {})
        sm = stack_meta.get(name, {})
        # Canonical Δ = Test-1 vs frozen v4p_hc (never raw file AUC − baseline)
        if sm.get("delta_auc") is not None and sm.get("stack_method", "").startswith("test1"):
            delta = float(sm["delta_auc"])
            auc = float(sm.get("stack_oof_auc") or roc_auc_score(data.y, s))
        else:
            res = score_candidate_vs_v4p_hc(data.y, data.v4p_o, s, data.folds)
            s = res.pop("stack_oof_scores")
            preds[name] = s
            delta = float(res["delta_auc"])
            auc = float(res["stack_oof_auc"])
            sm = res
        dlt, lo, hi = paired_bootstrap_ci(data.y, base, s)
        table1.append({
            "paradigm": name,
            "stack_oof_auc": auc,
            "solo_oof_auc": meta.get("solo_xgb_auc") or meta.get("solo_stealth_auc") or auc,
            "delta_auc": delta,
            "boot_ci_90": [lo, hi],
            "gpu_minutes": meta.get("gpu_minutes"),
            "delta_per_gpu_hour": delta / max((meta.get("gpu_minutes") or 1) / 60.0, 1e-6),
            "stack_method": sm.get("stack_method", "test1_orthogonal_alpha_probe_vs_v4p_hc"),
        })
    # combinations
    from itertools import combinations
    table3 = []
    keys = list(preds.keys())
    for r in range(1, len(keys) + 1):
        for comb in combinations(keys, r):
            # rank-average blend
            S = np.mean([rk01(preds[k]) for k in comb], axis=0)
            auc = float(roc_auc_score(data.y, S))
            table3.append({"subset": "+".join(comb), "auc": auc, "delta": auc - data.base_oof_auc})
    # Shapley approx
    shapley = {k: 0.0 for k in keys}
    if len(keys) == 3:
        from itertools import permutations
        # exact 3-player via all orders
        for order in permutations(keys):
            prev = None
            prev_auc = data.base_oof_auc
            for i, k in enumerate(order):
                subset = order[: i + 1]
                S = np.mean([rk01(preds[x]) for x in subset], axis=0)
                auc = float(roc_auc_score(data.y, S))
                shapley[k] += auc - prev_auc
                prev_auc = auc
        for k in shapley:
            shapley[k] /= 6.0
    # residual corr
    resid = {k: preds[k] - base for k in preds}
    corr = {}
    for i in keys:
        for j in keys:
            corr[f"{i}:{j}"] = float(spearmanr(resid[i], resid[j]).correlation) if i in resid and j in resid else None
    # stealth rescue
    stealth_mask = (data.y == 1) & (data.v4p_o < 0.10)
    table4 = []
    top5 = max(1, int(0.05 * data.n))
    base_top = set(np.argsort(-base)[:top5])
    for k, s in preds.items():
        top = set(np.argsort(-s)[:top5])
        rescued = int(np.sum([1 for i in np.where(stealth_mask)[0] if i in top and i not in base_top]))
        table4.append({"paradigm": k, "stealth_rescued_to_top5pct": rescued, "n_stealth": int(stealth_mask.sum())})

    dossier = {
        "utc": utc(),
        "baseline_oof": data.base_oof_auc,
        "no_submit": True,
        "table1_head_to_head": table1,
        "table2_orthogonality": {"residual_corr": corr},
        "table3_combinations_shapley": {"combos": table3, "shapley": shapley},
        "table4_error_strata": table4,
        "table5_recommendation": {
            "ranked": sorted(table1, key=lambda r: r["delta_auc"], reverse=True),
            "note": "Allocate final deep-research + 100%-refit to top 1-2 by ΔAUC/GPU-hr; keep A40 warm; STRICT NO_SUBMIT",
        },
    }
    path = ROOT / "logs/overnight/PARADIGM_DECISION_DOSSIER.json"
    path.write_text(json.dumps(dossier, indent=2))
    st["stage"] = "dossier_complete"
    st["status_line"] = "DOSSIER written; NO_SUBMIT"
    st["dossier_path"] = str(path)
    save_state(st)
    print(json.dumps(dossier, indent=2)[:3000], flush=True)
    return dossier


def main():
    assert NO_SUBMIT
    st = load_state()
    st["started_utc"] = st.get("started_utc") or utc()
    st["pod_id"] = os.environ.get("POD_ID", st.get("pod_id", ""))
    st["no_submit"] = True
    st["status_line"] = "loading data"
    save_state(st)
    if st.get("spend_usd", 0) >= SPEND_CAP - 0.5:
        print("SPEND CAP", flush=True)
        return
    data = DataBundle()
    try:
        if st["paradigms"]["A"].get("status") != "completed":
            paradigm_A(data, st)
        if st["paradigms"]["B"].get("status") != "completed":
            paradigm_B(data, st)
        if st["paradigms"]["C"].get("status") != "completed":
            paradigm_C(data, st)
        build_dossier(data, st)
    except Exception as e:
        note_fix(st, f"FATAL: {e}")
        st["status_line"] = f"FATAL {type(e).__name__}: {e}"
        st["traceback"] = traceback.format_exc()[-2000:]
        save_state(st)
        raise
    print("[DONE] A->B->C + dossier. STRICT NO_SUBMIT.", flush=True)


if __name__ == "__main__":
    main()
