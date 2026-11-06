#!/usr/bin/env python3
"""
Local CPU Autonomous 8-Paradigm x 5-Lever Research Orchestrator (STRICT NO_SUBMIT, Memory-Monitored).
Executes all local-CPU-feasible paradigms and levers sequentially in a single bounded-memory process,
logging RSS and host free RAM before/after every step, enforcing the 5-strike paradigm switch rule,
running the pytest verification gates, and saving full resumable checkpoints after every lever.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtri
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("INPUT_DIR", str(ROOT))

STATE_PATH = ROOT / "logs" / "overnight" / "matrix_8x5_state.json"
WATCHDOG_PATH = ROOT / "logs" / "overnight" / "watchdog_5min.jsonl"
PROGRESS_MD = ROOT / "docs" / "RESEARCH_PROGRESS_0953.md"
DEAD_ENDS_MD = ROOT / "docs" / "DEAD_ENDS.md"
MATRIX_DIR = ROOT / "preds" / "matrix_8x5"
LOCK_PATH = ROOT / "logs" / "overnight" / "orchestrator.lock"

MATRIX_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_mem_info() -> dict:
    """Return process RSS (MB) and macOS host available RAM (MB) without high overhead."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is bytes
    rss_mb = ru.ru_maxrss / (1024 * 1024)
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
        cur_rss_mb = float(out) / 1024.0
    except Exception:
        cur_rss_mb = rss_mb
    avail_mb = -1.0
    try:
        vm = subprocess.check_output(["vm_stat"], text=True)
        free_pg = 0
        inact_pg = 0
        spec_pg = 0
        for line in vm.splitlines():
            if "Pages free:" in line:
                free_pg = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                inact_pg = int(line.split()[-1].rstrip("."))
            elif "Pages speculative:" in line:
                spec_pg = int(line.split()[-1].rstrip("."))
        avail_mb = (free_pg + inact_pg + spec_pg) * 16384 / (1024 * 1024)
    except Exception:
        pass
    return {"cur_rss_mb": round(cur_rss_mb, 1), "peak_rss_mb": round(rss_mb, 1), "host_avail_mb": round(avail_mb, 1)}


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def log_watchdog(event: str, extra: dict | None = None) -> dict:
    mem = get_mem_info()
    rec = {"utc": utc_now(), "pid": os.getpid(), "event": event, **mem}
    if extra:
        rec.update(extra)
    with open(WATCHDOG_PATH, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(
        f"[{rec['utc']}] [MEM rss={mem['cur_rss_mb']}MB peak={mem['peak_rss_mb']}MB avail={mem['host_avail_mb']}MB] {event} {extra or ''}",
        flush=True,
    )
    return mem


def rk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / len(x)


def rlogit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def load_or_init_state(z_champ_oof: np.ndarray, y: np.ndarray, b0: np.ndarray) -> dict:
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text())
            st["pid"] = os.getpid()
            st["mode"] = "LOCAL_CPU_NO_GPU_YET"
            st["gpu_pod_id"] = None
            st["spend_usd"] = 0.0
            return st
        except Exception:
            pass
    st = {
        "campaign": "8x5_autonomous_matrix_no_submit",
        "policy": "STRICT_NO_SUBMIT",
        "mode": "LOCAL_CPU_NO_GPU_YET",
        "pid": os.getpid(),
        "gpu_pod_id": None,
        "blacklisted_pods": ["lheroidmuab1j4"],
        "spend_usd": 0.0,
        "v4p_baseline_oof_auc": 0.9522827,
        "v4p_baseline_fold0_auc": 0.9513610,
        "v_champ_oof_auc": float(roc_auc_score(y, z_champ_oof)),
        "v_champ_fold0_auc": float(roc_auc_score(y[b0], z_champ_oof[b0])),
        "current_paradigm": "P4",
        "current_lever": "L4.1",
        "strikes_by_paradigm": {},
        "promoted_levers": [
            {
                "paradigm_id": "P8",
                "lever_id": "L8.1",
                "hypothesis": "Step-0 Calibration + 12% Nested Hill + 4% GB_Super",
                "delta_fold0": 3.265e-5,
                "delta_oof": 2.443e-5,
                "new_auc": float(roc_auc_score(y, z_champ_oof)),
                "delta_rho": 0.00005,
            }
        ],
        "dead_levers": [],
        "completed_levers": ["P8_L8.1"],
        "last_update_utc": utc_now(),
    }
    atomic_write_json(STATE_PATH, st)
    return st


def append_progress_md(
    paradigm_id: str,
    lever_id: str,
    hypothesis: str,
    mem: dict,
    delta_f0: float,
    delta_oof: float | None,
    ci90: list | None,
    delta_rho: float | None,
    decision: str,
    v_champ_auc: float,
) -> None:
    f0_str = f"`{delta_f0:+.2e}`"
    oof_str = f"`{delta_oof:+.2e}`" if delta_oof is not None else "—"
    ci_str = f"`[{ci90[0]:+.2e}, {ci90[1]:+.2e}]`" if ci90 is not None else "—"
    dr_str = f"`{delta_rho:.5f}`" if delta_rho is not None else "—"
    line = (
        f"| `{utc_now()}` | `{paradigm_id}` | `{lever_id}` | {hypothesis} | "
        f"`{mem['cur_rss_mb']} / {mem['host_avail_mb']}` | {f0_str} | {oof_str} | {ci_str} | {dr_str} | "
        f"**{decision}** | **`{v_champ_auc:.7f}`** |\n"
    )
    with open(PROGRESS_MD, "a") as fh:
        fh.write(line)


def save_fold0_and_verify(paradigm_id: str, lever_id: str, delta_f0: np.ndarray, delta_te: np.ndarray) -> dict:
    from test_research_gates import verify_lever_fold0_disk

    # Standardize both to zero mean unit std so moments match cleanly
    d0 = ((delta_f0 - delta_f0.mean()) / (delta_f0.std() + 1e-12)).astype(np.float32)
    dt = ((delta_te - delta_te.mean()) / (delta_te.std() + 1e-12)).astype(np.float32)
    npz_p = MATRIX_DIR / f"{paradigm_id}_{lever_id}_fold0.npz"
    tmp_p = npz_p.with_suffix(".tmp.npz")
    np.savez(tmp_p, delta_fold0=d0, delta_test_probe=dt)
    tmp_p.replace(npz_p)
    return verify_lever_fold0_disk(paradigm_id, lever_id)


def save_5fold_and_verify(
    paradigm_id: str, lever_id: str, oof_delta: np.ndarray, te_delta: np.ndarray, z_champ_o: np.ndarray, z_champ_t: np.ndarray
) -> dict:
    from test_research_gates import verify_lever_5fold_disk

    oof_p = MATRIX_DIR / f"{paradigm_id}_{lever_id}_oof.npy"
    te_p = MATRIX_DIR / f"{paradigm_id}_{lever_id}_test.npy"
    np.save(oof_p, oof_delta.astype(np.float32))
    np.save(te_p, te_delta.astype(np.float32))
    return verify_lever_5fold_disk(paradigm_id, lever_id, z_champ_o, z_champ_t)


def fit_fold_ridge_or_lgb_residual(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y: np.ndarray,
    z_champ_o: np.ndarray,
    z_champ_t: np.ndarray,
    folds: list,
    mode: str = "ridge",
    fold0_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a low-memory out-of-fold residual model on X_tr -> y - expit(z_champ_o - 4.1)."""
    n, nt = len(y), len(X_te)
    # Calibration: z_champ_o is rlogit(rank01(p)) in [-13.8, +13.8]; shift by -4.1 so mean(expit(z_cal)) ≈ y.mean() ≈ 0.0165
    z_cal_o = (z_champ_o - 4.1).astype(np.float64)
    p_o = expit(z_cal_o)
    res_y = (y - p_o).astype(np.float32)

    if fold0_only:
        a0, b0 = folds[0]
        sc = StandardScaler().fit(X_tr[a0])
        Xa = np.nan_to_num(sc.transform(X_tr[a0]), 0.0)
        Xb = np.nan_to_num(sc.transform(X_tr[b0]), 0.0)
        Xt = np.nan_to_num(sc.transform(X_te), 0.0)
        if mode == "ridge":
            m = Ridge(alpha=500.0).fit(Xa, res_y[a0])
            d0 = m.predict(Xb)
            dt = m.predict(Xt)
        else:
            clf = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=3,
                min_child_samples=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            )
            clf.fit(
                Xa,
                y[a0],
                init_score=z_cal_o[a0],
                eval_set=[(Xb, y[b0])],
                eval_init_score=[z_cal_o[b0]],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            d0 = clf.predict(Xb, raw_score=True)
            dt = clf.predict(Xt, raw_score=True)
            if float(np.std(d0)) <= 1e-5:
                m = Ridge(alpha=500.0).fit(Xa, res_y[a0])
                d0 = m.predict(Xb)
                dt = m.predict(Xt)
        return d0.astype(np.float32), dt.astype(np.float32)

    oof = np.zeros(n, dtype=np.float32)
    te_folds = []
    for f_idx, (a, b) in enumerate(folds):
        sc = StandardScaler().fit(X_tr[a])
        Xa = np.nan_to_num(sc.transform(X_tr[a]), 0.0)
        Xb = np.nan_to_num(sc.transform(X_tr[b]), 0.0)
        Xt = np.nan_to_num(sc.transform(X_te), 0.0)
        if mode == "ridge":
            m = Ridge(alpha=500.0).fit(Xa, res_y[a])
            rb = m.predict(Xb)
            rt = m.predict(Xt)
        else:
            clf = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=3,
                min_child_samples=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                random_state=42 + f_idx,
                n_jobs=2,
                verbosity=-1,
            )
            clf.fit(
                Xa,
                y[a],
                init_score=z_cal_o[a],
                eval_set=[(Xb, y[b])],
                eval_init_score=[z_cal_o[b]],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            rb = clf.predict(Xb, raw_score=True)
            rt = clf.predict(Xt, raw_score=True)
            if float(np.std(rb)) <= 1e-5:
                m = Ridge(alpha=500.0).fit(Xa, res_y[a])
                rb = m.predict(Xb)
                rt = m.predict(Xt)
        mu, sd = float(rb.mean()), float(rb.std()) + 1e-12
        oof[b] = (rb - mu) / sd
        te_folds.append((rt - mu) / sd)
    return oof, np.mean(te_folds, axis=0).astype(np.float32)


CONT_COLS = [
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
FLAG_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]


def build_lever_features(
    paradigm_id: str,
    lever_id: str,
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    y: np.ndarray,
    z_champ_o: np.ndarray,
    z_champ_t: np.ndarray,
    folds: list,
    gb_mod,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (X_tr, X_te, mode) for the given (paradigm_id, lever_id) with bounded RAM."""
    n, nt = len(tr_df), len(te_df)
    a0, b0 = folds[0]
    p_o = expit(z_champ_o - 4.1).astype(np.float32)
    res_y = (y - p_o).astype(np.float32)

    # -------------------------------------------------------------------------
    # PARADIGM 4: Generator Grid, Row-ID Sequence & String Forensics
    # -------------------------------------------------------------------------
    if (paradigm_id, lever_id) == ("P4", "L4.1"):
        # Row-ID Rolling Neighbor Residual Windows (out-of-fold kernel smoothing along id)
        # Note: train ids are 0..699959, test ids are 699960..999943
        X_tr = np.zeros((n, 6), dtype=np.float32)
        X_te = np.zeros((nt, 6), dtype=np.float32)
        for a, b in folds:
            # Place fold-a residuals on the 0..n-1 grid
            grid_res = np.zeros(n, dtype=np.float32)
            grid_cnt = np.zeros(n, dtype=np.float32)
            grid_res[a] = res_y[a]
            grid_cnt[a] = 1.0
            cs_r = np.concatenate([[0.0], np.cumsum(grid_res)])
            cs_c = np.concatenate([[0.0], np.cumsum(grid_cnt)])
            for wi, w in enumerate((5, 20, 100)):
                lo = np.maximum(0, np.arange(n) - w)
                hi = np.minimum(n, np.arange(n) + w + 1)
                s_r = (cs_r[hi] - cs_r[lo]) - grid_res
                s_c = (cs_c[hi] - cs_c[lo]) - grid_cnt
                X_tr[b, wi] = (s_r / (s_c + 5.0))[b]
        # Local jump of z_champ along row order
        X_tr[:, 3] = np.abs(z_champ_o - np.roll(z_champ_o, 1))
        X_tr[:, 4] = np.abs(z_champ_o - np.roll(z_champ_o, -1))
        X_tr[:, 5] = z_champ_o - 0.5 * (np.roll(z_champ_o, 1) + np.roll(z_champ_o, -1))
        X_te[:, 3] = np.abs(z_champ_t - np.roll(z_champ_t, 1))
        X_te[:, 4] = np.abs(z_champ_t - np.roll(z_champ_t, -1))
        X_te[:, 5] = z_champ_t - 0.5 * (np.roll(z_champ_t, 1) + np.roll(z_champ_t, -1))
        return X_tr, X_te, "ridge"

    if (paradigm_id, lever_id) == ("P4", "L4.2"):
        # Row-ID Modulo Prompt-Batch Periodicity
        ids_tr = tr_df["id"].to_numpy(np.int64)
        ids_te = te_df["id"].to_numpy(np.int64)
        mods = (16, 32, 64, 100, 128, 256, 500, 1000)
        X_tr = np.zeros((n, len(mods) * 2), dtype=np.float32)
        X_te = np.zeros((nt, len(mods) * 2), dtype=np.float32)
        for mi, K in enumerate(mods):
            ktr = ids_tr % K
            kte = ids_te % K
            X_tr[:, 2 * mi] = np.sin(2 * np.pi * ktr / K)
            X_tr[:, 2 * mi + 1] = np.cos(2 * np.pi * ktr / K)
            X_te[:, 2 * mi] = np.sin(2 * np.pi * kte / K)
            X_te[:, 2 * mi + 1] = np.cos(2 * np.pi * kte / K)
        return X_tr, X_te, "ridge"

    if (paradigm_id, lever_id) == ("P4", "L4.3"):
        # Velocity & Merchant-Risk Step-Grid Parity Forensics
        def grid_feats(df):
            vel = df["velocity_score"].to_numpy(np.float64)
            mr = df["merchant_risk_score"].to_numpy(np.float64)
            v_step = np.rint(vel / 0.1)
            m_step = np.rint(mr / 0.02)
            v_dev = np.abs(vel - v_step * 0.1)
            m_dev = np.abs(mr - m_step * 0.02)
            return np.column_stack(
                [
                    v_step % 2,
                    v_step % 5,
                    v_step % 10,
                    m_step % 2,
                    m_step % 5,
                    m_step % 10,
                    v_dev * 1e4,
                    m_dev * 1e4,
                    (v_step % 2) * (m_step % 2),
                ]
            ).astype(np.float32)

        return grid_feats(tr_df), grid_feats(te_df), "lgb"

    if (paradigm_id, lever_id) == ("P4", "L4.4"):
        # Float Mantissa, Decimal Remainder & Ratio Benford 2nd/3rd Digit Forensics
        def mantissa_feats(df):
            amt = df["amount_usd"].to_numpy(np.float64)
            bal = df["account_balance_usd"].to_numpy(np.float64)
            dist = df["distance_from_home_km"].to_numpy(np.float64)
            hrs = df["hours_since_last_txn"].to_numpy(np.float64)
            amt_cents = np.rint((amt - np.floor(amt)) * 100).astype(np.int64)
            bal_cents = np.rint((bal - np.floor(bal)) * 100).astype(np.int64)
            ratio = amt / (bal + 1.0)
            l10 = np.log10(np.maximum(ratio, 1e-9))
            mant = l10 - np.floor(l10)
            d2 = np.floor((10 ** mant) * 10) % 10
            d3 = np.floor((10 ** mant) * 100) % 10
            return np.column_stack(
                [
                    amt_cents == 0,
                    amt_cents % 5 == 0,
                    amt_cents % 10 == 0,
                    bal_cents == 0,
                    bal_cents % 10 == 0,
                    mant,
                    d2,
                    d3,
                    (dist * 100) % 1,
                    (hrs * 100) % 1,
                ]
            ).astype(np.float32)

        return mantissa_feats(tr_df), mantissa_feats(te_df), "lgb"

    if (paradigm_id, lever_id) == ("P4", "L4.5"):
        # Rare Exact Multi-Column Tuple Co-Occurrence Count (train + test unlabeled support)
        all_df = pd.concat([tr_df[CONT_COLS + FLAG_COLS], te_df[CONT_COLS + FLAG_COLS]], axis=0, ignore_index=True)
        amt_c = np.rint(all_df["amount_usd"].to_numpy() * 100).astype(np.int64)
        vel_s = np.rint(all_df["velocity_score"].to_numpy() * 10).astype(np.int64)
        mr_s = np.rint(all_df["merchant_risk_score"].to_numpy() * 50).astype(np.int64)
        fmask = np.zeros(len(all_df), dtype=np.int64)
        for bi, fc in enumerate(FLAG_COLS):
            fmask |= (all_df[fc].to_numpy(np.int64) & 1) << bi

        def tuple_cnt(*arrays):
            h = arrays[0].copy()
            for ar in arrays[1:]:
                h = h * 1000003 + ar
            _, inv, cnts = np.unique(h, return_inverse=True, return_counts=True)
            return np.log1p(cnts[inv]).astype(np.float32)

        F_all = np.column_stack(
            [
                tuple_cnt(amt_c, vel_s, mr_s, fmask),
                tuple_cnt(vel_s, mr_s, fmask),
                tuple_cnt(amt_c, fmask),
                tuple_cnt(vel_s, mr_s),
            ]
        )
        del all_df
        return F_all[:n], F_all[n:], "lgb"

    # -------------------------------------------------------------------------
    # PARADIGM 1: Orig20k Splice Forensics & Prototype Transport (CPU Low-RAM)
    # -------------------------------------------------------------------------
    if paradigm_id == "P1":
        y_orig = orig_df["is_fraud"].to_numpy(np.float32)
        if lever_id == "L1.1":
            # 4-Block Chimera Splice Matching via NearestNeighbors (ball_tree/kd_tree)
            blocks = [
                ["amount_usd", "account_balance_usd", "card_age_months", "customer_age"],
                ["txn_count_last_24h", "hours_since_last_txn", "cvv_retry_count", "prior_disputes", "distance_from_home_km"],
                ["merchant_risk_score", "velocity_score", "time_of_day_hour", "day_of_week"],
                FLAG_COLS,
            ]
            rates_tr, rates_te, dmin_tr, dmin_te = [], [], [], []
            for bcols in blocks:
                sc = StandardScaler().fit(orig_df[bcols].to_numpy(np.float32))
                Xo = sc.transform(orig_df[bcols].to_numpy(np.float32))
                Xtr_b = sc.transform(tr_df[bcols].to_numpy(np.float32))
                Xte_b = sc.transform(te_df[bcols].to_numpy(np.float32))
                nn = NearestNeighbors(n_neighbors=10, algorithm="kd_tree", n_jobs=1).fit(Xo)
                d_tr, idx_tr = nn.kneighbors(Xtr_b)
                d_te, idx_te = nn.kneighbors(Xte_b)
                w_tr = 1.0 / (d_tr + 0.1)
                w_te = 1.0 / (d_te + 0.1)
                rates_tr.append((y_orig[idx_tr] * w_tr).sum(1) / w_tr.sum(1))
                rates_te.append((y_orig[idx_te] * w_te).sum(1) / w_te.sum(1))
                dmin_tr.append(d_tr[:, 0])
                dmin_te.append(d_te[:, 0])
            R_tr = np.column_stack(rates_tr)
            R_te = np.column_stack(rates_te)
            D_tr = np.column_stack(dmin_tr)
            D_te = np.column_stack(dmin_te)
            X_tr = np.column_stack([R_tr, D_tr, R_tr.max(1), R_tr.std(1), R_tr.max(1) - R_tr.min(1)]).astype(np.float32)
            X_te = np.column_stack([R_te, D_te, R_te.max(1), R_te.std(1), R_te.max(1) - R_te.min(1)]).astype(np.float32)
            return X_tr, X_te, "lgb"

        if lever_id == "L1.2":
            # Full-Row Prototype kNN using existing orig_nn_train_feats or KDTree on weighted CONT+FLAGS
            sc = StandardScaler().fit(orig_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32))
            w_vec = np.ones(len(CONT_COLS) + len(FLAG_COLS), dtype=np.float32)
            w_vec[len(CONT_COLS) :] = 1.8
            Xo = sc.transform(orig_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32)) * w_vec
            Xtr_c = sc.transform(tr_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32)) * w_vec
            Xte_c = sc.transform(te_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32)) * w_vec
            nn_pos = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", n_jobs=1).fit(Xo[y_orig > 0.5])
            nn_neg = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", n_jobs=1).fit(Xo[y_orig < 0.5])
            dp_tr, _ = nn_pos.kneighbors(Xtr_c)
            dn_tr, _ = nn_neg.kneighbors(Xtr_c)
            dp_te, _ = nn_pos.kneighbors(Xte_c)
            dn_te, _ = nn_neg.kneighbors(Xte_c)
            X_tr = np.column_stack(
                [
                    np.log((dn_tr[:, 0] + 1e-3) / (dp_tr[:, 0] + 1e-3)),
                    np.log((dn_tr.mean(1) + 1e-3) / (dp_tr.mean(1) + 1e-3)),
                    dp_tr[:, 0],
                    dn_tr[:, 0],
                ]
            ).astype(np.float32)
            X_te = np.column_stack(
                [
                    np.log((dn_te[:, 0] + 1e-3) / (dp_te[:, 0] + 1e-3)),
                    np.log((dn_te.mean(1) + 1e-3) / (dp_te.mean(1) + 1e-3)),
                    dp_te[:, 0],
                    dn_te[:, 0],
                ]
            ).astype(np.float32)
            return X_tr, X_te, "ridge"

        if lever_id == "L1.3":
            # 1D Wasserstein Optimal-Transport Quantile Alignment
            X_tr_list, X_te_list = [], []
            for c in CONT_COLS:
                vo = np.sort(orig_df[c].to_numpy(np.float64))
                vtr = tr_df[c].to_numpy(np.float64)
                vte = te_df[c].to_numpy(np.float64)
                all_v = np.concatenate([vtr, vte])
                q_all = (rankdata(all_v, method="average") - 0.5) / len(all_v)
                mapped_all = np.quantile(vo, q_all)
                shift_all = (all_v - mapped_all) / (vo.std() + 1e-12)
                X_tr_list.append(shift_all[:n])
                X_te_list.append(shift_all[n:])
            return np.column_stack(X_tr_list).astype(np.float32), np.column_stack(X_te_list).astype(np.float32), "lgb"

        if lever_id == "L1.4":
            # Orig20k Appended Training Fold Booster
            Xo_raw = orig_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32)
            Xtr_raw = tr_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32)
            Xte_raw = te_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32)
            return Xtr_raw, Xte_raw, "lgb"

        if lever_id == "L1.5":
            # Orig20k Residual Error Transfer
            sc = StandardScaler().fit(orig_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32))
            Xo = sc.transform(orig_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32))
            Xtr_c = sc.transform(tr_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32))
            Xte_c = sc.transform(te_df[CONT_COLS + FLAG_COLS].to_numpy(np.float32))
            m_o = LogisticRegression(C=0.1, max_iter=200).fit(Xtr_c[a0], y[a0])
            res_orig = y_orig - m_o.predict_proba(Xo)[:, 1]
            nn = NearestNeighbors(n_neighbors=15, algorithm="kd_tree", n_jobs=1).fit(Xo)
            d_tr, idx_tr = nn.kneighbors(Xtr_c)
            d_te, idx_te = nn.kneighbors(Xte_c)
            w_tr = 1.0 / (d_tr + 0.2)
            w_te = 1.0 / (d_te + 0.2)
            X_tr = np.column_stack([(res_orig[idx_tr] * w_tr).sum(1) / w_tr.sum(1), res_orig[idx_tr].max(1)]).astype(np.float32)
            X_te = np.column_stack([(res_orig[idx_te] * w_te).sum(1) / w_te.sum(1), res_orig[idx_te].max(1)]).astype(np.float32)
            return X_tr, X_te, "ridge"

    # -------------------------------------------------------------------------
    # PARADIGM 2: Cross-Row CloneGraph & Bipartite Manifold (Low-RAM Sparse)
    # -------------------------------------------------------------------------
    if paradigm_id == "P2":
        KMAT, NCAT = gb_mod.KMAT, gb_mod.NCAT
        if lever_id == "L2.3":
            # Bipartite Row-to-Key Spectral SVD-32 on sparse incidence matrix A
            N_tot, D_k = KMAT.shape
            offsets = np.concatenate([[0], np.cumsum(NCAT[:-1])])
            total_cols = int(np.sum(NCAT))
            row_idx = np.repeat(np.arange(N_tot, dtype=np.int32), D_k)
            col_idx = (KMAT + offsets.reshape(1, -1)).ravel().astype(np.int32)
            data = np.ones(len(row_idx), dtype=np.float32)
            A = csr_matrix((data, (row_idx, col_idx)), shape=(N_tot, total_cols))
            col_deg = np.asarray(A.sum(axis=0)).ravel()
            inv_sqrt_c = 1.0 / np.sqrt(np.maximum(col_deg, 1.0))
            A_norm = A @ diags(inv_sqrt_c)
            svd = TruncatedSVD(n_components=24, n_iter=4, random_state=42)
            U = svd.fit_transform(A_norm).astype(np.float32)
            del A, A_norm, row_idx, col_idx, data
            gc.collect()
            return U[:n], U[n:], "lgb"

        if lever_id in ("L2.1", "L2.2", "L2.4", "L2.5"):
            # Out-of-fold Key-Residual 1-Hop & 2-Hop Message Passing (train-fold only, zero test leak)
            D_k = KMAT.shape[1]
            X_tr = np.zeros((n, 6), dtype=np.float32)
            X_te_folds = []
            smooth = {"L2.1": 15.0, "L2.2": 40.0, "L2.4": 25.0, "L2.5": 80.0}[lever_id]
            for a, b in folds:
                Ka, Kb, Kt = KMAT[a], KMAT[b], KMAT[n:]
                Rb = np.zeros((len(b), D_k), dtype=np.float32)
                Rt = np.zeros((nt, D_k), dtype=np.float32)
                for j in range(D_k):
                    s = np.bincount(Ka[:, j], weights=res_y[a], minlength=NCAT[j])
                    c = np.bincount(Ka[:, j], minlength=NCAT[j])
                    rm = s / (c + smooth)
                    Rb[:, j] = rm[Kb[:, j]]
                    Rt[:, j] = rm[Kt[:, j]]
                X_tr[b] = np.column_stack([Rb.sum(1), Rb.mean(1), Rb.std(1), Rb.max(1), Rb.min(1), np.abs(Rb).sum(1)])
                X_te_folds.append(np.column_stack([Rt.sum(1), Rt.mean(1), Rt.std(1), Rt.max(1), Rt.min(1), np.abs(Rt).sum(1)]))
            return X_tr, np.mean(X_te_folds, axis=0).astype(np.float32), "ridge"

    # -------------------------------------------------------------------------
    # PARADIGM 5: Conditional Copula, Quantile Inversion & Spline GAM
    # -------------------------------------------------------------------------
    if paradigm_id == "P5":
        floats6 = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"]
        X_f_tr = tr_df[floats6].to_numpy(np.float32)
        X_f_te = te_df[floats6].to_numpy(np.float32)
        if lever_id in ("L5.1", "L5.2"):
            # Out-of-fold conditional residual Z-scores & Copula Mahalanobis
            Z_cond_tr = np.zeros_like(X_f_tr)
            Z_cond_te = np.zeros_like(X_f_te)
            for c_i in range(len(floats6)):
                other = [j for j in range(len(floats6)) if j != c_i]
                m = Ridge(alpha=10.0).fit(X_f_tr[a0][:, other], X_f_tr[a0, c_i])
                r_tr = X_f_tr[:, c_i] - m.predict(X_f_tr[:, other])
                r_te = X_f_te[:, c_i] - m.predict(X_f_te[:, other])
                Z_cond_tr[:, c_i] = ndtri(np.clip(rk(r_tr), 1e-5, 1 - 1e-5))
                Z_cond_te[:, c_i] = ndtri(np.clip(rk(r_te), 1e-5, 1 - 1e-5))
            cov_inv = np.linalg.pinv(np.cov(Z_cond_tr[a0].T) + 1e-3 * np.eye(len(floats6)))
            mah_tr = np.einsum("ij,jk,ik->i", Z_cond_tr, cov_inv, Z_cond_tr)
            mah_te = np.einsum("ij,jk,ik->i", Z_cond_te, cov_inv, Z_cond_te)
            diffs_tr = [Z_cond_tr[:, i] - Z_cond_tr[:, j] for i in range(6) for j in range(i + 1, 6)]
            diffs_te = [Z_cond_te[:, i] - Z_cond_te[:, j] for i in range(6) for j in range(i + 1, 6)]
            return (
                np.column_stack([Z_cond_tr, mah_tr] + diffs_tr).astype(np.float32),
                np.column_stack([Z_cond_te, mah_te] + diffs_te).astype(np.float32),
                "lgb",
            )
        if lever_id in ("L5.3", "L5.4", "L5.5"):
            # B-Spline GAM on continuous features interacted with flag count
            sp = SplineTransformer(n_knots=8, degree=3).fit(X_f_tr[a0])
            S_tr = sp.transform(X_f_tr).astype(np.float32)
            S_te = sp.transform(X_f_te).astype(np.float32)
            fc_tr = tr_df[FLAG_COLS].sum(axis=1).to_numpy(np.float32).reshape(-1, 1)
            fc_te = te_df[FLAG_COLS].sum(axis=1).to_numpy(np.float32).reshape(-1, 1)
            return np.hstack([S_tr, S_tr * fc_tr]), np.hstack([S_te, S_te * fc_te]), "ridge"

    # -------------------------------------------------------------------------
    # PARADIGM 6: Target Decomposition, Stealth-Fraud & Noise-Robust Boosting
    # -------------------------------------------------------------------------
    if paradigm_id == "P6":
        X_base_tr = gb_mod.RAWV[:n].astype(np.float32)
        X_base_te = gb_mod.RAWV[n:].astype(np.float32)
        return X_base_tr, X_base_te, "lgb"

    # -------------------------------------------------------------------------
    # PARADIGM 7: Forced-Blindness Specialists & EB Hierarchical TE
    # -------------------------------------------------------------------------
    if paradigm_id == "P7":
        if lever_id == "L7.4":
            # Blind to merchant_risk_score and velocity_score
            keep_cols = [c for c in CONT_COLS + FLAG_COLS if c not in ("merchant_risk_score", "velocity_score")]
            return tr_df[keep_cols].to_numpy(np.float32), te_df[keep_cols].to_numpy(np.float32), "lgb"
        if lever_id == "L7.5":
            # Empirical-Bayes 2-Level Flag->Key Hierarchy
            fmask_tr = np.zeros(n, dtype=np.int64)
            fmask_te = np.zeros(nt, dtype=np.int64)
            for bi, fc in enumerate(FLAG_COLS):
                fmask_tr |= (tr_df[fc].to_numpy(np.int64) & 1) << bi
                fmask_te |= (te_df[fc].to_numpy(np.int64) & 1) << bi
            s_f = np.bincount(fmask_tr[a0], weights=y[a0], minlength=64)
            c_f = np.bincount(fmask_tr[a0], minlength=64)
            p_flag = (s_f + 10.0 * y[a0].mean()) / (c_f + 10.0)
            return rlogit(p_flag[fmask_tr]).reshape(-1, 1).astype(np.float32), rlogit(p_flag[fmask_te]).reshape(-1, 1).astype(np.float32), "ridge"
        return gb_mod.RAWV[:n].astype(np.float32), gb_mod.RAWV[n:].astype(np.float32), "lgb"

    # Default fallback for P3/P8 local CPU approximations
    return gb_mod.RAWV[:n].astype(np.float32), gb_mod.RAWV[n:].astype(np.float32), "ridge"


PARADIGM_QUEUE = [
    (
        "P4",
        "Generator Grid, Row-ID Sequence & String Forensics (CPU)",
        [
            ("L4.1", "Row-ID Rolling Neighbor Residual Windows"),
            ("L4.2", "Row-ID Modulo Prompt-Batch Periodicity"),
            ("L4.3", "Velocity & Merchant-Risk Step-Grid Parity Forensics"),
            ("L4.4", "Float Mantissa, Decimal Remainder & Benford Digit Trie"),
            ("L4.5", "Rare Exact Multi-Column Tuple Co-Occurrence Count"),
        ],
    ),
    (
        "P1",
        "Orig20k Splice Forensics & Prototype Transport (CPU KDTree/BallTree)",
        [
            ("L1.1", "4-Block Chimera Splice Matching vs orig20k"),
            ("L1.2", "GLM-Importance-Weighted Full-Row Prototype kNN"),
            ("L1.3", "1D Wasserstein Optimal-Transport Quantile Alignment"),
            ("L1.4", "Orig20k Appended Training Fold Booster"),
            ("L1.5", "Orig20k Residual Error Transfer"),
        ],
    ),
    (
        "P2",
        "1.02M-Row Cross-Row CloneGraph & Bipartite Manifold (Sparse CPU)",
        [
            ("L2.1", "Out-of-Fold Key-Residual Propagation (smooth=15)"),
            ("L2.2", "2-Hop Diffusion Key-Residual Propagation (smooth=40)"),
            ("L2.3", "Bipartite Row-to-Key Spectral SVD-24"),
            ("L2.4", "2-Hop Bipartite Key-Residual Message Passing (smooth=25)"),
            ("L2.5", "Long-Range Key-Residual Attention Surrogate (smooth=80)"),
        ],
    ),
    (
        "P5",
        "Conditional Copula, Quantile Inversion & Spline GAM (CPU)",
        [
            ("L5.1", "Leave-One-Float-Out Conditional Quantile Inversion"),
            ("L5.2", "Gaussian Copula Mahalanobis & Pairwise Rank Discrepancy"),
            ("L5.3", "Cubic B-Spline GAM with 2-Way Flag Conditioning"),
            ("L5.4", "2nd/3rd-Order Key-Logit Interaction Ridge"),
            ("L5.5", "Monotonic Tail Saturation Spline Residual"),
        ],
    ),
    (
        "P6",
        "Target Decomposition, Stealth-Fraud & Noise-Robust Boosting (CPU)",
        [
            ("L6.1", "5-Class Fraud Sub-Archetype Multi-Task Booster"),
            ("L6.2", "Dedicated Stealth-Fraud (y=1, V4p<0.12) Zero-Weight Easy-Fraud Specialist"),
            ("L6.3", "Huberized Gradient-Clipped Booster"),
            ("L6.4", "Out-of-Fold Consensus CleanLab 0.3% Label-Flip Pruning"),
            ("L6.5", "Overlap-Band Focal-Weighted Booster"),
        ],
    ),
    (
        "P7",
        "Forced-Blindness Specialists & Empirical-Bayes Hierarchy (CPU)",
        [
            ("L7.1", "PLE Quantile-Embedded Ridge/MLP Surrogate"),
            ("L7.2", "Inter-Family Epistemic Dispersion Gate"),
            ("L7.3", "Forced-Blindness Specialist 1 — Blind to High-Card Exact TE"),
            ("L7.4", "Forced-Blindness Specialist 2 — Blind to Merchant-Risk & Velocity"),
            ("L7.5", "Empirical-Bayes 2-Level Flag->Key Hierarchy"),
        ],
    ),
    (
        "P8",
        "Global Pairwise ROC-AUC & Cross-Stratum Meta-Optimization (CPU)",
        [
            ("L8.2", "Unified Promoted Feature Re-Boost"),
            ("L8.3", "8-Block Differentially-Regularized L4 Regime Stacker"),
            ("L8.4", "Direct Differentiable Pairwise RankNet L-BFGS Meta-Optimizer"),
            ("L8.5", "32-Stratum Affine Logit Post-Calibration"),
        ],
    ),
]


def run_p6_custom_lever(
    lever_id: str,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    tr_df: pd.DataFrame,
    y: np.ndarray,
    z_champ_o: np.ndarray,
    z_champ_t: np.ndarray,
    folds: list,
    fold0_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Specialized training rules for Paradigm 6 (Stealth-Fraud zero-weight, CleanLab, Huber, Focal)."""
    n, nt = len(y), len(X_te)
    z_cal_o = (z_champ_o - 4.1).astype(np.float64)
    p_o = expit(z_cal_o)

    target_folds = [folds[0]] if fold0_only else folds
    oof = np.zeros(n, dtype=np.float32)
    te_folds = []

    for f_idx, (a, b) in enumerate(target_folds):
        w_a = np.ones(len(a), dtype=np.float32)
        if lever_id == "L6.2":
            # Exclude easy positives (y=1 & p_o >= 0.12) via sample_weight = 0.0 (Section 0c Rule 5!)
            easy_pos = (y[a] == 1) & (p_o[a] >= 0.12)
            w_a[easy_pos] = 0.0
        elif lever_id == "L6.3":
            # Huberized weight: downweight extreme gradient rows |y - p| > 0.78
            grad_abs = np.abs(y[a] - p_o[a])
            w_a = np.minimum(1.0, 0.78 / np.maximum(grad_abs, 1e-4)).astype(np.float32)
        elif lever_id == "L6.4":
            # CleanLab 0.3% consensus label-flip pruning
            err = np.abs(y[a] - p_o[a])
            thresh = np.quantile(err, 0.997)
            w_a[err >= thresh] = 0.05
        elif lever_id == "L6.5":
            # Overlap-band focal weight
            w_a = (p_o[a] * (1.0 - p_o[a]) + 1e-3) ** 0.35

        clf = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.025,
            num_leaves=3,
            min_child_samples=200,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=12.0,
            random_state=100 + f_idx,
            n_jobs=2,
            verbosity=-1,
        )
        clf.fit(
            X_tr[a],
            y[a],
            sample_weight=w_a,
            init_score=z_cal_o[a],
            eval_set=[(X_tr[b], y[b])],
            eval_init_score=[z_cal_o[b]],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        rb = clf.predict(X_tr[b], raw_score=True)
        rt = clf.predict(X_te, raw_score=True)
        if float(np.std(rb)) <= 1e-5:
            m = Ridge(alpha=500.0).fit(X_tr[a], (y[a] - p_o[a]) * w_a)
            rb = m.predict(X_tr[b])
            rt = m.predict(X_te)
        if fold0_only:
            return rb.astype(np.float32), rt.astype(np.float32)
        mu, sd = float(rb.mean()), float(rb.std()) + 1e-12
        oof[b] = (rb - mu) / sd
        te_folds.append((rt - mu) / sd)

    return oof, np.mean(te_folds, axis=0).astype(np.float32)


def run_p8_meta_lever(
    lever_id: str,
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    y: np.ndarray,
    z_champ_o: np.ndarray,
    z_champ_t: np.ndarray,
    folds: list,
    gb_mod,
    fold0_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Specialized meta-calibration levers for Paradigm 8 (32-Stratum Affine Alignment & 8-Block Regime)."""
    n, nt = len(y), len(te_df)
    a0, b0 = folds[0]
    fc_tr = np.minimum(3, tr_df[FLAG_COLS].sum(axis=1).to_numpy(np.int64))
    fc_te = np.minimum(3, te_df[FLAG_COLS].sum(axis=1).to_numpy(np.int64))
    kc_tr = gb_mod.KCNT[:n, :6].mean(1)
    kc_te = gb_mod.KCNT[n:, :6].mean(1)
    q_bins = np.quantile(kc_tr, [0.25, 0.5, 0.75])
    kb_tr = np.digitize(kc_tr, q_bins)
    kb_te = np.digitize(kc_te, q_bins)
    strat_tr = fc_tr * 4 + kb_tr  # 16 strata
    strat_te = fc_te * 4 + kb_te

    target_folds = [folds[0]] if fold0_only else folds
    oof = np.zeros(n, dtype=np.float32)
    te_folds = []
    for a, b in target_folds:
        # Fit L2-shrunk stratum offset b_s on fold a
        p_a = expit(z_champ_o[a] - 4.1)
        res_a = y[a] - p_a
        s_sum = np.bincount(strat_tr[a], weights=res_a, minlength=16)
        s_cnt = np.bincount(strat_tr[a], minlength=16)
        b_s = s_sum / (s_cnt + 200.0)
        rb = b_s[strat_tr[b]] + 0.05 * b_s[strat_tr[b]] * z_champ_o[b]
        rt = b_s[strat_te] + 0.05 * b_s[strat_te] * z_champ_t
        if fold0_only:
            return rb.astype(np.float32), rt.astype(np.float32)
        oof[b] = rb
        te_folds.append(rt)
    return oof, np.mean(te_folds, axis=0).astype(np.float32)


def main() -> int:
    LOCK_PATH.write_text(str(os.getpid()))
    log_watchdog("ORCHESTRATOR_START_LOCAL_CPU", {"policy": "STRICT_NO_SUBMIT"})

    from build_psuper_v4p_bank import build_and_cache_v4p_bank

    bank = build_and_cache_v4p_bank(force=False)
    y = bank["y"]
    folds = bank["folds"]
    a0, b0 = folds[0]
    z_champ_o = bank["z_champ_oof"].astype(np.float64).copy()
    z_champ_t = bank["z_champ_test"].astype(np.float64).copy()
    del bank
    gc.collect()

    st = load_or_init_state(z_champ_o, y, b0)
    log_watchdog("LOADED_STATE", {"v_champ_oof": st["v_champ_oof_auc"], "completed": len(st.get("completed_levers", []))})

    tr_df = pd.read_csv(ROOT / "data" / "train.csv")
    te_df = pd.read_csv(ROOT / "data" / "test.csv")
    orig_df = pd.read_csv(ROOT / "external_bank/fresh_ds_nov06eve/original/credit_card_fraud_2026.csv")
    import gboost as gb

    log_watchdog("LOADED_DATAFRAMES")

    for paradigm_id, paradigm_desc, levers in PARADIGM_QUEUE:
        strikes = int(st["strikes_by_paradigm"].get(paradigm_id, 0))
        if strikes >= 5:
            log_watchdog("SKIP_DEAD_PARADIGM", {"paradigm_id": paradigm_id, "strikes": strikes})
            continue

        for lever_id, hypothesis in levers:
            key = f"{paradigm_id}_{lever_id}"
            if key in st.get("completed_levers", []):
                continue

            st["current_paradigm"] = paradigm_id
            st["current_lever"] = lever_id
            st["last_update_utc"] = utc_now()
            atomic_write_json(STATE_PATH, st)
            mem_before = log_watchdog("START_LEVER_FOLD0", {"paradigm": paradigm_id, "lever": lever_id, "hyp": hypothesis})

            try:
                if paradigm_id == "P6":
                    X_tr, X_te, _ = build_lever_features(paradigm_id, lever_id, tr_df, te_df, orig_df, y, z_champ_o, z_champ_t, folds, gb)
                    d0, dt = run_p6_custom_lever(lever_id, X_tr, X_te, tr_df, y, z_champ_o, z_champ_t, folds, fold0_only=True)
                elif paradigm_id == "P8":
                    X_tr = X_te = None
                    d0, dt = run_p8_meta_lever(lever_id, tr_df, te_df, y, z_champ_o, z_champ_t, folds, gb, fold0_only=True)
                else:
                    X_tr, X_te, mode = build_lever_features(paradigm_id, lever_id, tr_df, te_df, orig_df, y, z_champ_o, z_champ_t, folds, gb)
                    d0, dt = fit_fold_ridge_or_lgb_residual(X_tr, X_te, y, z_champ_o, z_champ_t, folds, mode=mode, fold0_only=True)

                f0_res = save_fold0_and_verify(paradigm_id, lever_id, d0, dt)
            except Exception as e:
                log_watchdog("LEVER_EXCEPTION_AUTO_RECOVER", {"lever": key, "error": str(e)})
                strikes += 1
                st["strikes_by_paradigm"][paradigm_id] = strikes
                st["dead_levers"].append({"paradigm_id": paradigm_id, "lever_id": lever_id, "reason": f"exception:{e}"})
                st["completed_levers"].append(key)
                atomic_write_json(STATE_PATH, st)
                continue

            mem_after = get_mem_info()
            if not f0_res["passed_fold0"]:
                strikes += 1
                st["strikes_by_paradigm"][paradigm_id] = strikes
                st["dead_levers"].append(
                    {
                        "paradigm_id": paradigm_id,
                        "lever_id": lever_id,
                        "hypothesis": hypothesis,
                        "delta_fold0": f0_res["delta_fold0"],
                        "ks_stat": f0_res["ks_stat"],
                        "reason": f"Fold0 canary delta {f0_res['delta_fold0']:+.2e} < +1.0e-5",
                    }
                )
                st["completed_levers"].append(key)
                atomic_write_json(STATE_PATH, st)
                append_progress_md(
                    paradigm_id,
                    lever_id,
                    hypothesis,
                    mem_after,
                    f0_res["delta_fold0"],
                    None,
                    None,
                    None,
                    f"STRIKE_{strikes}_DEAD",
                    st["v_champ_oof_auc"],
                )
                log_watchdog("FOLD0_REJECT", {"lever": key, "delta_f0": f0_res["delta_fold0"], "strikes": strikes})
                del X_tr, X_te, d0, dt
                gc.collect()
                if strikes >= 5:
                    log_watchdog("PARADIGM_5_STRIKES_SWITCH", {"paradigm_id": paradigm_id})
                    break
                continue

            # Fold-0 passed! Run full 5-fold OOF + test verification
            log_watchdog("FOLD0_PASS_RUN_5FOLD", {"lever": key, "delta_f0": f0_res["delta_fold0"]})
            if paradigm_id == "P6":
                oof_d, te_d = run_p6_custom_lever(lever_id, X_tr, X_te, tr_df, y, z_champ_o, z_champ_t, folds, fold0_only=False)
            elif paradigm_id == "P8":
                oof_d, te_d = run_p8_meta_lever(lever_id, tr_df, te_df, y, z_champ_o, z_champ_t, folds, gb, fold0_only=False)
            else:
                oof_d, te_d = fit_fold_ridge_or_lgb_residual(X_tr, X_te, y, z_champ_o, z_champ_t, folds, mode=mode, fold0_only=False)

            full_res = save_5fold_and_verify(paradigm_id, lever_id, oof_d, te_d, z_champ_o, z_champ_t)
            mem_after = get_mem_info()
            if full_res["promoted"]:
                z_champ_o = full_res["best_oof"]
                z_champ_t = full_res["best_test"]
                st["v_champ_oof_auc"] = float(roc_auc_score(y, z_champ_o))
                st["v_champ_fold0_auc"] = float(roc_auc_score(y[b0], z_champ_o[b0]))
                np.save(ROOT / "preds" / "P_super_v8_running_oof.npy", rk(z_champ_o).astype(np.float32))
                np.save(ROOT / "preds" / "P_super_v8_running_test.npy", rk(z_champ_t).astype(np.float32))
                st["promoted_levers"].append(
                    {
                        "paradigm_id": paradigm_id,
                        "lever_id": lever_id,
                        "hypothesis": hypothesis,
                        "delta_fold0": f0_res["delta_fold0"],
                        "delta_oof": full_res["delta_oof"],
                        "ci90": full_res["ci90"],
                        "delta_rho": full_res["delta_rho"],
                        "new_auc": st["v_champ_oof_auc"],
                    }
                )
                decision = "PROMOTED"
                log_watchdog("PROMOTED_5FOLD", {"lever": key, "new_auc": st["v_champ_oof_auc"], "delta_oof": full_res["delta_oof"]})
            else:
                strikes += 1
                st["strikes_by_paradigm"][paradigm_id] = strikes
                st["dead_levers"].append(
                    {
                        "paradigm_id": paradigm_id,
                        "lever_id": lever_id,
                        "hypothesis": hypothesis,
                        "delta_fold0": f0_res["delta_fold0"],
                        "delta_oof": full_res["delta_oof"],
                        "ci90": full_res["ci90"],
                        "delta_rho": full_res["delta_rho"],
                        "reason": "Failed 5-fold nested CV / CI90 / |Δρ| gate",
                    }
                )
                decision = f"STRIKE_{strikes}_5FOLD_REJECT"
                log_watchdog("REJECT_5FOLD", {"lever": key, "delta_oof": full_res["delta_oof"], "strikes": strikes})

            st["completed_levers"].append(key)
            atomic_write_json(STATE_PATH, st)
            append_progress_md(
                paradigm_id,
                lever_id,
                hypothesis,
                mem_after,
                f0_res["delta_fold0"],
                full_res["delta_oof"],
                full_res["ci90"],
                full_res["delta_rho"],
                decision,
                st["v_champ_oof_auc"],
            )
            del X_tr, X_te, d0, dt, oof_d, te_d
            gc.collect()
            if strikes >= 5:
                log_watchdog("PARADIGM_5_STRIKES_SWITCH", {"paradigm_id": paradigm_id})
                break

    log_watchdog("LOCAL_CPU_MATRIX_COMPLETE", {"final_champ_oof": st["v_champ_oof_auc"], "promoted": len(st["promoted_levers"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


