#!/usr/bin/env python3
"""DS First-Principles AUC Research — H1 perm Bayes / H2 WMW / H3 generator shift.

STRICT NO_SUBMIT=True. No notebook scraping. No submission CSVs.
Soft RSS target < 500 MB via sequential float32 + del/gc.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path(os.environ.get("GM_DS_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
WORK = Path(os.environ.get("GM_DS_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/ds_first_principles"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
DS_L7 = Path(
    "/tmp/gm_v6_audit/external/nov11_095347_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
)
for p in (ART, LOG):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT
EPS = 1e-6
RSS_SOFT = 500.0
T0 = time.time()

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss_curr_mb() -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def rss_max_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|RSS={rss_curr_mb():.0f}/{rss_max_mb():.0f}MB] {msg}", flush=True)
    if rss_curr_mb() > RSS_SOFT:
        print(f"  WARN curr RSS {rss_curr_mb():.0f}MB > soft cap {RSS_SOFT}", flush=True)


def purge():
    gc.collect()


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).ravel()


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def llr_from_lm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if float(np.nanmedian(arr)) < 0:
        return (arr[..., 1] - arr[..., 0]).astype(np.float32)
    p0 = np.clip(arr[..., 0], 1e-12, 1.0)
    p1 = np.clip(arr[..., 1], 1e-12, 1.0)
    return (np.log(p1) - np.log(p0)).astype(np.float32)


def bayes_llr(llr_perm: np.ndarray) -> np.ndarray:
    llr_perm = np.asarray(llr_perm, dtype=np.float64)
    logp1 = -np.logaddexp(0, -llr_perm)
    logp0 = -np.logaddexp(0, llr_perm)
    lp1 = logsumexp(logp1, axis=1) - np.log(llr_perm.shape[1])
    lp0 = logsumexp(logp0, axis=1) - np.log(llr_perm.shape[1])
    return (lp1 - lp0).astype(np.float32)


def eb_margin(llr4: np.ndarray, kappa: float) -> np.ndarray:
    """z_EB = mean_w(LLR) / sqrt(1 + kappa * var_perm). Equal weights."""
    mu = llr4.mean(1)
    var = llr4.var(1)
    return (mu / np.sqrt(1.0 + kappa * var + 1e-12)).astype(np.float32)


def fold_aucs(y, z, folds):
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_delta(y, z_base, z_new, folds, m0=None):
    base_f = fold_aucs(y, z_base, folds)
    new_f = fold_aucs(y, z_new, folds)
    d5 = [(a - b) * 1e5 for a, b in zip(new_f, base_f)]
    out = {
        "d5_mean": float(np.mean(d5)),
        "pos": int(sum(x > 0 for x in d5)),
        "d5": [float(x) for x in d5],
        "auc_base": float(roc_auc_score(y, z_base)),
        "auc_new": float(roc_auc_score(y, z_new)),
    }
    if m0 is not None:
        b0 = float(roc_auc_score(y[m0], z_base[m0]))
        n0 = float(roc_auc_score(y[m0], z_new[m0]))
        out["d0"] = (n0 - b0) * 1e5
    return out


def subcv_delta(y_f0, z_base_f0, delta_f0, subfolds, ws=(0.02, 0.03, 0.04, 0.05)):
    base = [float(roc_auc_score(y_f0[b], z_base_f0[b])) for _, b in subfolds]
    best = None
    rows = []
    for w in ws:
        z = z_base_f0 + w * unitize(delta_f0).astype(np.float64)
        d = [(float(roc_auc_score(y_f0[b], z[b])) - base[i]) * 1e5 for i, (_, b) in enumerate(subfolds)]
        row = {"w": w, "mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}
        rows.append(row)
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    return best, rows


def disp_ks(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.std(b) / (np.std(a) + 1e-18)), float(ks_2samp(a, b).statistic)


def first_existing(*cands):
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(cands)


# ---------------------------------------------------------------------------
# Hypothesis 1
# ---------------------------------------------------------------------------
def hypothesis1(y, folds, fold_id, m0, idx_f0, z_ref, d1, s7_f0, rayA_f0):
    log("======== H1: K=4 vs K=10 perm truncation + Bayes reconstruction ========")
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))
    n_f0 = len(y_f0)

    lm7_o = np.load(DS_L7 / "lm7_oof_f0.npy")
    lm7_t = np.load(DS_L7 / "lm7_test_f0.npy")
    llr10 = llr_from_lm(lm7_o)  # (139992,10)
    llr4t = llr_from_lm(lm7_t)  # (299984,4)
    del lm7_o, lm7_t
    purge()

    # --- Diagnostic: solo AUC per order ---
    solo = []
    for k in range(10):
        auc = float(roc_auc_score(y_f0, llr10[:, k]))
        solo.append(auc)
        log(f"  L7 Fold0 solo perm k={k}: AUC={auc:.7f}")
    auc_mean4 = float(roc_auc_score(y_f0, llr10[:, :4].mean(1)))
    auc_mean10 = float(roc_auc_score(y_f0, llr10.mean(1)))
    auc4b = float(roc_auc_score(y_f0, bayes_llr(llr10[:, :4])))
    auc10b = float(roc_auc_score(y_f0, bayes_llr(llr10)))
    d10v4 = (auc_mean10 - auc_mean4) * 1e5
    d10bv4b = (auc10b - auc4b) * 1e5
    log(f"  mean4 AUC={auc_mean4:.7f}  mean10 AUC={auc_mean10:.7f}  Δ10vs4={d10v4:+.2f}e-5")
    log(f"  bayes4 AUC={auc4b:.7f} bayes10 AUC={auc10b:.7f} Δ={d10bv4b:+.2f}e-5")

    # Correlation: which OOF orders match test order marginals (via sorted-law Spearman on random subsample)
    rng = np.random.default_rng(42)
    # Compare each test order's sorted law to each OOF order's sorted law via quantile corr
    q = np.linspace(0.01, 0.99, 199)
    te_q = np.stack([np.quantile(llr4t[:, k], q) for k in range(4)], 0)
    oof_q = np.stack([np.quantile(llr10[:, k], q) for k in range(10)], 0)
    match = []
    for tk in range(4):
        cors = [float(np.corrcoef(te_q[tk], oof_q[ok])[0, 1]) for ok in range(10)]
        best = int(np.argmax(np.abs(cors)))
        match.append({"test_k": tk, "best_oof_k": best, "corr": cors[best], "all_corr": cors})
        log(f"  test_order{tk} ≈ oof_order{best} (corr={cors[best]:+.4f})")

    # κ sweep for EB on 4-order
    best_kappa, best_auc_k = 0.0, auc_mean4
    for kappa in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        z = eb_margin(llr10[:, :4].astype(np.float64), kappa)
        a = float(roc_auc_score(y_f0, z))
        if a > best_auc_k:
            best_auc_k, best_kappa = a, kappa
    log(f"  best κ-EB on 4-order: κ={best_kappa} AUC={best_auc_k:.7f} (Δvs mean4={(best_auc_k-auc_mean4)*1e5:+.2f}e-5)")

    # --- 4→10 reconstruction via Ridge+Huber SubCV ---
    # Target: bayes_llr of full 10 orders
    target10 = unitize(bayes_llr(llr10))
    # Features from 4 orders + stats + anchors
    mega = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"))[idx_f0]
    gpt_bayes = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))[idx_f0]
    w12 = unitize(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_oof.npy"))[idx_f0]

    X4 = np.column_stack(
        [
            llr10[:, 0],
            llr10[:, 1],
            llr10[:, 2],
            llr10[:, 3],
            llr10[:, :4].std(1),
            llr10[:, :4].min(1),
            llr10[:, :4].max(1),
            llr10[:, :4].mean(1),
            eb_margin(llr10[:, :4].astype(np.float64), best_kappa),
            mega,
            gpt_bayes,
            w12,
        ]
    ).astype(np.float32)
    # standardize fold-local later inside CV
    del mega, gpt_bayes, w12
    purge()

    hat10 = np.zeros(n_f0, dtype=np.float64)
    for si, (a, b) in enumerate(subfolds):
        mu = X4[a].mean(0)
        sd = X4[a].std(0) + 1e-8
        Xa = (X4[a] - mu) / sd
        Xb = (X4[b] - mu) / sd
        # Ridge primary
        ridge = Ridge(alpha=50.0, fit_intercept=True)
        ridge.fit(Xa, target10[a].astype(np.float64))
        pred_r = ridge.predict(Xb)
        # Huber residual polish
        resid = target10[a].astype(np.float64) - ridge.predict(Xa)
        try:
            hub = HuberRegressor(alpha=1.0, max_iter=200)
            hub.fit(Xa, resid)
            pred = pred_r + hub.predict(Xb)
        except Exception:
            pred = pred_r
        hat10[b] = pred
        log(f"  L7 4→10 recon SubF{si}")
        del ridge, Xa, Xb
        purge()

    L7_EB_10hat = unitize(hat10)
    # solo AUC of reconstruction
    auc_hat = float(roc_auc_score(y_f0, L7_EB_10hat))
    auc_naive4 = float(roc_auc_score(y_f0, unitize(llr10[:, :4].mean(1))))
    log(f"  L7_EB_10hat solo AUC={auc_hat:.7f} vs naive4={auc_naive4:.7f} Δ={(auc_hat-auc_naive4)*1e5:+.2f}e-5")

    # vs Ray A peak on Fold0 SubCV
    # residual of hat vs naive4, then add to rayA
    u_l7 = unitize(L7_EB_10hat.astype(np.float64) - unitize(llr10[:, :4].mean(1)).astype(np.float64))
    best_vs_s7, rows_s7 = subcv_delta(y_f0, s7_f0, u_l7, subfolds)
    best_vs_rayA, rows_rayA = subcv_delta(y_f0, rayA_f0, u_l7, subfolds)
    log(f"  L7_EB_10hat vs S7_f0: pos={best_vs_s7['pos']}/5 mean={best_vs_s7['mean']:+.2f}e-5 @w={best_vs_s7['w']}")
    log(f"  L7_EB_10hat vs RayA_peak: pos={best_vs_rayA['pos']}/5 mean={best_vs_rayA['mean']:+.2f}e-5 @w={best_vs_rayA['w']}")

    # Save Fold0 hat; build test hat with full-fit on Fold0
    mu = X4.mean(0)
    sd = X4.std(0) + 1e-8
    ridge = Ridge(alpha=50.0)
    ridge.fit((X4 - mu) / sd, target10.astype(np.float64))
    # test features
    mega_t = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_test.npy"))
    gpt_t = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_test.npy"))
    w12_t = unitize(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_test.npy"))
    Xt = np.column_stack(
        [
            llr4t[:, 0],
            llr4t[:, 1],
            llr4t[:, 2],
            llr4t[:, 3],
            llr4t.std(1),
            llr4t.min(1),
            llr4t.max(1),
            llr4t.mean(1),
            eb_margin(llr4t.astype(np.float64), best_kappa),
            mega_t,
            gpt_t,
            w12_t,
        ]
    ).astype(np.float32)
    hat_t = ridge.predict((Xt - mu) / sd).astype(np.float32)
    np.save(ART / "L7_EB_10hat_oof_f0.npy", L7_EB_10hat)
    np.save(ART / "L7_EB_10hat_test.npy", unitize(hat_t))
    disp, ks = disp_ks(L7_EB_10hat, unitize(hat_t))
    log(f"  L7_EB_10hat test align proxy disp={disp:.4f} KS={ks:.6f}")
    del X4, Xt, llr4t, mega_t, gpt_t, w12_t, hat_t, ridge
    purge()

    # --- gpt2_med K=8 OOF vs K=4 test reconstruction across 5 folds ---
    log("  --- gpt2_med 4→8 reconstruction (5-fold) ---")
    gpt_dir = PRED / "gpt2_med_4ep"
    gpt8_oof = np.zeros(len(y), dtype=np.float32)
    gpt4_oof = np.zeros(len(y), dtype=np.float32)
    gpt_hat = np.zeros(len(y), dtype=np.float32)
    gpt_diag = []
    for fi, (a, b) in enumerate(folds):
        o = np.load(gpt_dir / f"oof_{fi}.npy")  # (n_fold, 8, 2)
        llr8 = llr_from_lm(o)
        del o
        assert llr8.shape[0] == len(b)
        auc8 = float(roc_auc_score(y[b], llr8.mean(1)))
        auc4 = float(roc_auc_score(y[b], llr8[:, :4].mean(1)))
        gpt_diag.append({"fold": fi, "auc8": auc8, "auc4": auc4, "d84": (auc8 - auc4) * 1e5})
        log(f"  gpt2 F{fi}: mean8={auc8:.7f} mean4={auc4:.7f} Δ8vs4={(auc8-auc4)*1e5:+.2f}e-5")
        gpt8_oof[b] = bayes_llr(llr8)
        gpt4_oof[b] = bayes_llr(llr8[:, :4])
        # SubCV inside this fold's rows to predict 8-bayes from 4-order feats
        yb = y[b]
        inner = list(StratifiedKFold(5, shuffle=True, random_state=42 + fi).split(np.zeros(len(yb)), yb))
        Xg = np.column_stack(
            [
                llr8[:, 0],
                llr8[:, 1],
                llr8[:, 2],
                llr8[:, 3],
                llr8[:, :4].std(1),
                llr8[:, :4].min(1),
                llr8[:, :4].max(1),
                llr8[:, :4].mean(1),
            ]
        ).astype(np.float32)
        tgt = unitize(bayes_llr(llr8))
        hat_b = np.zeros(len(yb), dtype=np.float64)
        for ia, ib in inner:
            mu = Xg[ia].mean(0)
            sd = Xg[ia].std(0) + 1e-8
            model = Ridge(alpha=30.0)
            model.fit((Xg[ia] - mu) / sd, tgt[ia].astype(np.float64))
            hat_b[ib] = model.predict((Xg[ib] - mu) / sd)
            del model
        gpt_hat[b] = hat_b.astype(np.float32)
        del llr8, Xg, tgt, hat_b
        purge()

    # 5-fold lift of hat vs naive4 on z_ref+d1
    z_base = z_ref.astype(np.float64) + 0.04 * unitize(d1).astype(np.float64)
    u_gpt = unitize(unitize(gpt_hat).astype(np.float64) - unitize(gpt4_oof).astype(np.float64))
    # sweep small weights
    best_gpt = None
    for w in [0.01, 0.02, 0.03, 0.04, 0.05]:
        ev = eval_delta(y, z_base, z_base + w * u_gpt.astype(np.float64), folds, m0)
        ev["w"] = w
        if best_gpt is None or (ev["pos"], ev["d5_mean"]) > (best_gpt["pos"], best_gpt["d5_mean"]):
            best_gpt = ev
        log(f"  gpt_EB_8hat @w={w}: 5F={ev['d5_mean']:+.2f}e-5 pos={ev['pos']}/5 F0={ev.get('d0', float('nan')):+.2f}e-5")
    np.save(ART / "gpt2m_EB_8hat_oof.npy", unitize(gpt_hat))
    d84_mean = float(np.mean([r["d84"] for r in gpt_diag]))
    log(f"  gpt2 mean Δ8vs4 across folds={d84_mean:+.2f}e-5")

    # Fold0 SubCV of gpt hat residual on RayA
    u_gpt_f0 = u_gpt[idx_f0]
    best_gpt_rayA, _ = subcv_delta(y_f0, rayA_f0, u_gpt_f0, subfolds)
    log(f"  gpt_EB_8hat vs RayA_f0: pos={best_gpt_rayA['pos']}/5 mean={best_gpt_rayA['mean']:+.2f}e-5")

    del llr10, gpt8_oof, gpt4_oof, gpt_hat, u_gpt
    purge()

    return {
        "solo_perm_auc": solo,
        "auc_mean4": auc_mean4,
        "auc_mean10": auc_mean10,
        "d10_vs_4_e5": d10v4,
        "auc_bayes4": auc4b,
        "auc_bayes10": auc10b,
        "d_bayes10_vs_4_e5": d10bv4b,
        "order_match": match,
        "best_kappa": best_kappa,
        "auc_kappa": best_auc_k,
        "auc_L7_EB_10hat": auc_hat,
        "auc_naive4": auc_naive4,
        "L7_vs_S7": best_vs_s7,
        "L7_vs_RayA": best_vs_rayA,
        "L7_disp_ks": [disp, ks],
        "gpt_diag": gpt_diag,
        "gpt_d84_mean_e5": d84_mean,
        "gpt_best_5f": best_gpt,
        "gpt_vs_RayA": best_gpt_rayA,
    }


# ---------------------------------------------------------------------------
# Hypothesis 2 — WMW pairwise AUC
# ---------------------------------------------------------------------------
def wmw_loss_and_grad(theta, Phi_pos, Phi_neg, gamma, lam):
    """L = mean log(1+exp(-γ (s_i - s_j))) + λ||θ||^2 ; exact over provided anchors."""
    sp = Phi_pos @ theta  # (P,)
    sn = Phi_neg @ theta  # (N,)
    # pairwise diffs via broadcasting in chunks
    P = len(sp)
    N = len(sn)
    loss = 0.0
    g = np.zeros_like(theta)
    # chunk negatives
    bs = 512
    inv = 1.0 / (P * N)
    for j0 in range(0, N, bs):
        sn_b = sn[j0 : j0 + bs]
        Phi_nb = Phi_neg[j0 : j0 + bs]
        # (P, nb)
        diff = sp[:, None] - sn_b[None, :]
        # softplus(-γ diff) = log(1+exp(-γ diff))
        x = -gamma * diff
        # stable softplus
        soft = np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
        loss += float(soft.sum())
        # d softplus(x)/dθ : σ(x) * dx/dθ ; x=-γ(sp-sn); dx/dθ = -γ (Phi_p - Phi_n)
        sig = expit(x)  # (P, nb)
        # grad_θ = inv * sum_{i,j} σ * (-γ) * (Phi_i - Phi_j)
        # = inv*(-γ) * [ sum_j σ_ij * Phi_i  -  sum_i σ_ij * Phi_j ]
        coef_pos = sig.sum(1)  # (P,)
        coef_neg = sig.sum(0)  # (nb,)
        g += (-gamma) * (Phi_pos.T @ coef_pos - Phi_nb.T @ coef_neg)
    loss = loss * inv + lam * float(np.dot(theta, theta))
    g = g * inv + 2.0 * lam * theta
    return loss, g


def hypothesis2(y, folds, fold_id, m0, idx_f0, z_ref, d1, s7_f0, rayA_f0, h1):
    log("======== H2: Smooth pairwise WMW-AUC optimization (22-D) ========")
    ntr = len(y)
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))
    bank = VOL / "external/banks/goodpjw2008_s6e11-oof-bank"

    # Load basis columns sequentially into Phi float32
    cols = []
    names = []

    def add(name, v):
        cols.append(unitize(v))
        names.append(name)

    add("u_stageB", load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    add("u_tree_logit", load_f32(first_existing(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy")))
    # L7 EB: place Fold0 hat into full OOF (0 elsewhere)
    l7_full = np.zeros(ntr, dtype=np.float32)
    l7_full[idx_f0] = load_f32(ART / "L7_EB_10hat_oof_f0.npy")
    add("u_L7_EB_10hat", l7_full)
    add("u_gpt2m_EB_8hat", load_f32(ART / "gpt2m_EB_8hat_oof.npy"))
    add("u_zjoint", load_f32(PRED / "gm_v2_ladder/z_v8_clean_oof.npy"))
    add("u_glm_graph", zrk(load_f32(bank / "oof_glm_ohk.npy")))
    add("P4_L4.2", zrk(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy")))
    add("P4_L4.3", zrk(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy")))
    add("P4_L4.5", zrk(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy")))
    add("u_tabm", zrk(load_f32(PRED / "fast_gpu_harvest/oof_tabm_5f.npy")))
    # diffs vs z_ref
    zref_u = unitize(z_ref)
    add("u_MLP_diff", unitize(zrk(load_f32(bank / "oof_realmlp_dig.npy")).astype(np.float64) - zref_u.astype(np.float64)))
    add("u_GB_diff", unitize(zrk(load_f32(bank / "oof_gb_bag5.npy")).astype(np.float64) - zref_u.astype(np.float64)))
    add("u_realmlp_diff", unitize(zrk(load_f32(bank / "oof_realmlp_te.npy")).astype(np.float64) - zref_u.astype(np.float64)))
    add("u_ebm_diff", unitize(zrk(load_f32(bank / "oof_ebm_dig.npy")).astype(np.float64) - zref_u.astype(np.float64)))
    Phi_lin = np.column_stack(cols).astype(np.float32)  # (ntr, 14)
    assert Phi_lin.shape[1] == 14
    del cols
    purge()
    log(f"  linear basis 14 ready RSS={rss_curr_mb():.0f}MB")

    # Gate covariates g_k in [-1,1]
    # g_sup: from X_v5 rare/disp extras via mmap
    meta = json.loads((PRED / "gm_v5_overnight/X_v5_meta.json").read_text())
    extra_names = meta.get("extra_names", [])
    base_extra = 68 + int(meta.get("n_kmat5seed", 15))
    Xo = np.load(PRED / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
    if "z_lm_x_rare" in extra_names:
        rare = np.asarray(Xo[:, base_extra + extra_names.index("z_lm_x_rare")], dtype=np.float32)
    else:
        rare = np.asarray(Xo[:, 0], dtype=np.float32)
    if "z_lm_x_disp" in extra_names:
        disp = np.asarray(Xo[:, base_extra + extra_names.index("z_lm_x_disp")], dtype=np.float32)
    else:
        disp = rare
    del Xo
    purge()
    g_sup = np.tanh(unitize(rare).astype(np.float64)).astype(np.float32)
    # g_lm_cert: -zrk(perm std) — use gpt hat uncertainty proxy = |gpt_EB - z_m_bayes|
    gpt_bayes = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    gpt_eb = unitize(load_f32(ART / "gpt2m_EB_8hat_oof.npy"))
    g_lm_cert = np.tanh(-np.abs(gpt_eb.astype(np.float64) - gpt_bayes.astype(np.float64))).astype(np.float32)
    # g_disagree: |LM - Tree|
    tree = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    g_disagree = np.tanh(np.abs(gpt_bayes.astype(np.float64) - tree.astype(np.float64))).astype(np.float32)
    # g_tail
    g_tail = np.tanh(zref_u.astype(np.float64) - 1.65).astype(np.float32)
    del rare, disp, gpt_bayes, gpt_eb, tree
    purge()

    # 8 gated cols: stageB, tree, L7, gpt × {g_sup, g_lm_cert, g_disagree, g_tail} — pick 8
    # use: stageB*g_sup, stageB*g_tail, tree*g_disagree, tree*g_lm_cert,
    #      L7*g_sup, L7*g_lm_cert, gpt*g_disagree, gpt*g_tail
    gated = np.column_stack(
        [
            Phi_lin[:, 0] * g_sup,
            Phi_lin[:, 0] * g_tail,
            Phi_lin[:, 1] * g_disagree,
            Phi_lin[:, 1] * g_lm_cert,
            Phi_lin[:, 2] * g_sup,
            Phi_lin[:, 2] * g_lm_cert,
            Phi_lin[:, 3] * g_disagree,
            Phi_lin[:, 3] * g_tail,
        ]
    ).astype(np.float32)
    for j in range(gated.shape[1]):
        gated[:, j] = unitize(gated[:, j])
    Phi = np.hstack([Phi_lin, gated]).astype(np.float32)
    assert Phi.shape[1] == 22
    del Phi_lin, gated, g_sup, g_lm_cert, g_disagree, g_tail
    purge()
    log(f"  Phi 22 ready shape={Phi.shape} RSS={rss_curr_mb():.0f}MB")

    z_base = z_ref.astype(np.float64) + 0.04 * unitize(d1).astype(np.float64)  # ≈ 0.9526953 regime
    base_auc = float(roc_auc_score(y, z_base))
    log(f"  z_ref+0.04*δ1 AUC={base_auc:.7f}")

    gamma = 2.0
    lam_grid = [1e-3, 1e-2, 1e-1, 1.0]
    oof_score = np.zeros(ntr, dtype=np.float64)
    fold_rows = []

    rng = np.random.default_rng(42)
    for fi, (a, b) in enumerate(folds):
        best_lam, best_va, best_th = None, -1e9, None
        # inner split for lam
        a_tr, a_va = a[::2], a[1::2]  # cheap 50/50 stride split
        # if class imbalance broken, fallback
        if y[a_tr].sum() < 50 or y[a_va].sum() < 50:
            inner = StratifiedKFold(2, shuffle=True, random_state=42)
            (a_tr, a_va), _ = list(inner.split(np.zeros(len(a)), y[a]))
            a_tr, a_va = a[a_tr], a[a_va]
        for lam in lam_grid:
            pos_idx = a_tr[y[a_tr] == 1]
            neg_pool = a_tr[y[a_tr] == 0]
            # 4000 stratified neg anchors by score quantile of z_base
            zb = z_base[neg_pool]
            qs = np.quantile(zb, np.linspace(0, 1, 21))
            anchors = []
            per = max(1, 4000 // 20)
            for q0, q1 in zip(qs[:-1], qs[1:]):
                m = neg_pool[(zb >= q0) & (zb <= q1)] if q1 > q0 else neg_pool[(zb >= q0)]
                if len(m) == 0:
                    continue
                take = min(per, len(m))
                anchors.append(rng.choice(m, size=take, replace=False))
            neg_idx = np.concatenate(anchors)[:4000]
            Phi_pos = Phi[pos_idx].astype(np.float64)
            Phi_neg = Phi[neg_idx].astype(np.float64)

            def obj(th):
                return wmw_loss_and_grad(th, Phi_pos, Phi_neg, gamma, lam)

            th0 = np.zeros(22, dtype=np.float64)
            res = minimize(lambda th: obj(th)[0], th0, jac=lambda th: obj(th)[1], method="L-BFGS-B", options={"maxiter": 80})
            # validate on a_va by AUC of z_base + score
            s = Phi[a_va] @ res.x
            va = float(roc_auc_score(y[a_va], z_base[a_va] + 0.05 * unitize(s).astype(np.float64)))
            if va > best_va:
                best_va, best_lam, best_th = va, lam, res.x.copy()
            del Phi_pos, Phi_neg
            purge()
        # refit on full train fold a with best lam
        pos_idx = a[y[a] == 1]
        neg_pool = a[y[a] == 0]
        zb = z_base[neg_pool]
        qs = np.quantile(zb, np.linspace(0, 1, 21))
        anchors = []
        per = max(1, 4000 // 20)
        for q0, q1 in zip(qs[:-1], qs[1:]):
            m = neg_pool[(zb >= q0) & (zb <= q1)] if q1 > q0 else neg_pool[(zb >= q0)]
            if len(m) == 0:
                continue
            take = min(per, len(m))
            anchors.append(rng.choice(m, size=take, replace=False))
        neg_idx = np.concatenate(anchors)[:4000]
        Phi_pos = Phi[pos_idx].astype(np.float64)
        Phi_neg = Phi[neg_idx].astype(np.float64)

        def obj2(th):
            return wmw_loss_and_grad(th, Phi_pos, Phi_neg, gamma, best_lam)

        res = minimize(lambda th: obj2(th)[0], best_th, jac=lambda th: obj2(th)[1], method="L-BFGS-B", options={"maxiter": 100})
        oof_score[b] = Phi[b] @ res.x
        fold_rows.append({"fold": fi, "lam": best_lam, "va": best_va})
        log(f"  WMW F{fi}: λ={best_lam} vaAUC={best_va:.7f} RSS={rss_curr_mb():.0f}MB")
        del Phi_pos, Phi_neg
        purge()

    u_wmw = unitize(oof_score)
    np.save(ART / "u_wmw_auc_oof.npy", u_wmw)
    best5 = None
    for w in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        ev = eval_delta(y, z_base, z_base + w * u_wmw.astype(np.float64), folds, m0)
        ev["w"] = w
        if best5 is None or (ev["pos"], ev["d5_mean"]) > (best5["pos"], best5["d5_mean"]):
            best5 = ev
        log(f"  WMW @w={w}: 5F={ev['d5_mean']:+.2f}e-5 pos={ev['pos']}/5 F0={ev.get('d0', float('nan')):+.2f}e-5")

    # Fold0 SubCV on RayA peak — re-optimize θ on Fold0 only for honesty
    log("  WMW Fold0 SubCV …")
    Phi_f0 = Phi[idx_f0]
    oof_f0 = np.zeros(len(y_f0), dtype=np.float64)
    for si, (a, b) in enumerate(subfolds):
        best_lam, best_va, best_th = 0.1, -1e9, np.zeros(22)
        for lam in lam_grid:
            pos_idx = a[y_f0[a] == 1]
            neg_pool = a[y_f0[a] == 0]
            if len(pos_idx) < 20 or len(neg_pool) < 100:
                continue
            take = min(3000, len(neg_pool))
            neg_idx = rng.choice(neg_pool, size=take, replace=False)
            Phi_pos = Phi_f0[pos_idx].astype(np.float64)
            Phi_neg = Phi_f0[neg_idx].astype(np.float64)

            def obj(th, Pp=Phi_pos, Pn=Phi_neg, lm=lam):
                return wmw_loss_and_grad(th, Pp, Pn, gamma, lm)

            res = minimize(lambda th: obj(th)[0], np.zeros(22), jac=lambda th: obj(th)[1], method="L-BFGS-B", options={"maxiter": 60})
            s = Phi_f0[a[::3]] @ res.x  # rough va
            # use hold-in stride as weak va
            va = -res.fun
            if va > best_va:
                best_va, best_lam, best_th = va, lam, res.x.copy()
            del Phi_pos, Phi_neg
        # refit
        pos_idx = a[y_f0[a] == 1]
        neg_pool = a[y_f0[a] == 0]
        neg_idx = rng.choice(neg_pool, size=min(3000, len(neg_pool)), replace=False)
        Phi_pos = Phi_f0[pos_idx].astype(np.float64)
        Phi_neg = Phi_f0[neg_idx].astype(np.float64)

        def objf(th):
            return wmw_loss_and_grad(th, Phi_pos, Phi_neg, gamma, best_lam)

        res = minimize(lambda th: objf(th)[0], best_th, jac=lambda th: objf(th)[1], method="L-BFGS-B", options={"maxiter": 80})
        oof_f0[b] = Phi_f0[b] @ res.x
        log(f"    WMW SubF{si} λ={best_lam}")
        del Phi_pos, Phi_neg
        purge()

    best_s7, _ = subcv_delta(y_f0, s7_f0, oof_f0, subfolds)
    best_ray, _ = subcv_delta(y_f0, rayA_f0, oof_f0, subfolds)
    log(f"  WMW vs S7_f0: pos={best_s7['pos']}/5 mean={best_s7['mean']:+.2f}e-5")
    log(f"  WMW vs RayA_f0: pos={best_ray['pos']}/5 mean={best_ray['mean']:+.2f}e-5")

    del Phi, Phi_f0, oof_score
    purge()
    return {
        "base_auc": base_auc,
        "fold_rows": fold_rows,
        "best5": best5,
        "vs_S7_f0": best_s7,
        "vs_RayA_f0": best_ray,
        "basis_names": names + [f"gate{i}" for i in range(8)],
    }


# ---------------------------------------------------------------------------
# Hypothesis 3 — generator shift
# ---------------------------------------------------------------------------
def build_kmat_codes(df: pd.DataFrame) -> np.ndarray:
    """Compact ~60–68 discrete keys matching gboost key_table (no GPT tokens)."""
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
    FLOATS = [
        "amount_usd",
        "hours_since_last_txn",
        "distance_from_home_km",
        "account_balance_usd",
        "velocity_score",
        "merchant_risk_score",
    ]
    cols = []
    # categoricals + flags + raw nums as strings/values for factorize later jointly
    for c in NUMS + FLAGS + CATS:
        if c in CATS:
            cols.append(df[c].astype(str).values)
        else:
            cols.append(df[c].values)
    for c in FLOATS:
        x = df[c].values.astype(np.float64)
        cols.append((np.round(x * 100) % 100).astype(np.int64))
        cols.append(np.floor(x).astype(np.int64))
        cols.append(np.floor(x / 10).astype(np.int64))
        cols.append(np.floor(x * 10 + 1e-6).astype(np.int64))
        i = pd.Series(np.floor(x).astype(np.int64)).astype(str)
        cols.append((i.str.len().astype(int) * 10 + i.str[0].astype(int)).values)
        cols.append((i.str.len().astype(int) * 100 + i.str[:2].astype(int)).values)
    # return list of raw columns; factorize jointly outside
    return cols


def hypothesis3(y, folds, fold_id, m0, idx_f0, z_ref, d1, s7_f0, rayA_f0):
    log("======== H3: Synthetic generator conditional drift (orig20k) ========")
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))

    usecols = [
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
        "merchant_category",
        "card_type",
        "auth_method",
        "channel",
        "device_type",
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
        "is_fraud",
    ]
    log("  loading train+orig columns …")
    tr = pd.read_csv(VOL / "data/train.csv", usecols=usecols)
    orig = pd.read_csv(VOL / "original/credit_card_fraud_2026.csv", usecols=usecols)
    y_orig = orig["is_fraud"].to_numpy(np.int8)
    log(f"  orig n={len(orig)} fraud={int(y_orig.sum())}")

    tr_cols = build_kmat_codes(tr)
    or_cols = build_kmat_codes(orig)
    n_keys = len(tr_cols)
    n_use = min(68, n_keys)
    log(f"  built {n_keys} raw key cols; using first {n_use}")

    ntr = len(y)
    # Stream factorize + EB rates fold-wise to limit RAM: store only condensed OOF later
    # First factorize each key on train∪orig
    codes_tr = []
    codes_or = []
    for j in range(n_use):
        vals = np.concatenate([np.asarray(tr_cols[j]), np.asarray(or_cols[j])])
        codes, _ = pd.factorize(vals, sort=False)
        codes_tr.append(codes[:ntr].astype(np.int32))
        codes_or.append(codes[ntr:].astype(np.int32))
        del vals, codes
        if (j + 1) % 10 == 0:
            log(f"  factorized {j+1}/{n_use} RSS={rss_curr_mb():.0f}MB")
            purge()
    del tr_cols, or_cols, tr, orig
    purge()

    K_tr = np.column_stack(codes_tr)  # (ntr, 68)
    K_or = np.column_stack(codes_or)  # (20k, 68)
    del codes_tr, codes_or
    purge()
    log(f"  K_tr={K_tr.shape} K_or={K_or.shape} RSS={rss_curr_mb():.0f}MB")

    # Build 136 generator-shift features OOF
    # For memory: accumulate shift matrix in float32 (ntr, 136) ~ 699960*136*4 ≈ 380MB — TOO BIG
    # Instead: condense per-fold with Ridge immediately, or process in blocks of keys.
    # Strategy: for each fold, build (n_val, 136) only for validation rows, predict ridge LOFO.
    z_base = z_ref.astype(np.float64) + 0.04 * unitize(d1).astype(np.float64)
    # Platt margin for anchor
    from sklearn.linear_model import LogisticRegression

    gen_oof = np.zeros(ntr, dtype=np.float64)
    lgb_oof = np.zeros(ntr, dtype=np.float64)

    def eb_rate(codes, yv, alpha=10.0):
        # codes 1d int, return dict key->p and key->count
        # vectorized
        mx = int(codes.max()) + 1
        cnt = np.bincount(codes, minlength=mx).astype(np.float64)
        sm = np.bincount(codes, weights=yv.astype(np.float64), minlength=mx)
        prior = float(yv.mean())
        p = (sm + alpha * prior) / (cnt + alpha)
        return p, cnt

    try:
        import lightgbm as lgb
        have_lgb = True
    except Exception:
        have_lgb = False

    for fi, (a, b) in enumerate(folds):
        feats_a = []
        feats_b = []
        for j in range(n_use):
            p_syn, c_syn = eb_rate(K_tr[a, j], y[a])
            # orig rates for keys with count>=3
            p_or, c_or = eb_rate(K_or[:, j], y_orig)
            # map
            ca = K_tr[a, j]
            cb = K_tr[b, j]
            # clip codes
            ca_c = np.clip(ca, 0, len(p_syn) - 1)
            cb_c = np.clip(cb, 0, len(p_syn) - 1)
            # orig may have different cardinality — align by code id from joint factorize (same space)
            mx = max(len(p_syn), len(p_or))
            if len(p_syn) < mx:
                p_syn = np.pad(p_syn, (0, mx - len(p_syn)), constant_values=float(y[a].mean()))
                c_syn = np.pad(c_syn, (0, mx - len(c_syn)))
            if len(p_or) < mx:
                p_or = np.pad(p_or, (0, mx - len(p_or)), constant_values=float(y_orig.mean()))
                c_or = np.pad(c_or, (0, mx - len(c_or)))
            # mask low orig support
            valid = c_or >= 3
            p_or_safe = np.where(valid, p_or, float(y_orig.mean()))
            dlogit_a = logit(np.clip(p_syn[ca_c], 1e-4, 1 - 1e-4)) - logit(np.clip(p_or_safe[ca_c], 1e-4, 1 - 1e-4))
            dlogit_b = logit(np.clip(p_syn[cb_c], 1e-4, 1 - 1e-4)) - logit(np.clip(p_or_safe[cb_c], 1e-4, 1 - 1e-4))
            # where orig support low, zero the shift
            dlogit_a = np.where(valid[ca_c], dlogit_a, 0.0)
            dlogit_b = np.where(valid[cb_c], dlogit_b, 0.0)
            sratio_a = np.log((c_syn[ca_c] + 0.5) / (c_or[ca_c] + 0.5))
            sratio_b = np.log((c_syn[cb_c] + 0.5) / (c_or[cb_c] + 0.5))
            feats_a.append(dlogit_a.astype(np.float32))
            feats_a.append(sratio_a.astype(np.float32))
            feats_b.append(dlogit_b.astype(np.float32))
            feats_b.append(sratio_b.astype(np.float32))
        Xa = np.column_stack(feats_a).astype(np.float32)
        Xb = np.column_stack(feats_b).astype(np.float32)
        del feats_a, feats_b
        # anchor Platt(z_ref)
        lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
        lr.fit(z_ref[a].reshape(-1, 1), y[a])
        bm_a = lr.decision_function(z_ref[a].reshape(-1, 1))
        bm_b = lr.decision_function(z_ref[b].reshape(-1, 1))
        # Ridge on residual target
        # target = y - platt_prob as soft; use y with base_margin style: predict residual in probability space
        ridge = Ridge(alpha=5000.0)
        # fit on features to predict y, then use decision as score; include bm as col
        Xa2 = np.column_stack([Xa, bm_a.astype(np.float32)])
        Xb2 = np.column_stack([Xb, bm_b.astype(np.float32)])
        ridge.fit(Xa2, y[a].astype(np.float64))
        gen_oof[b] = ridge.predict(Xb2)
        if have_lgb:
            dtr = lgb.Dataset(Xa, label=y[a], free_raw_data=True)
            dtr.set_init_score(bm_a)
            params = dict(
                objective="binary",
                metric="auc",
                num_leaves=6,
                max_depth=3,
                learning_rate=0.03,
                min_child_samples=500,
                feature_fraction=0.5,
                verbosity=-1,
                seed=42 + fi,
                num_threads=2,
            )
            # small holdout
            cut = int(0.9 * len(a))
            dtr = lgb.Dataset(Xa[:cut], label=y[a][:cut])
            dtr.set_init_score(bm_a[:cut])
            dva = lgb.Dataset(Xa[cut:], label=y[a][cut:], reference=dtr)
            dva.set_init_score(bm_a[cut:])
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=400,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
            )
            lgb_oof[b] = booster.predict(Xb, raw_score=True)
            del booster, dtr, dva
        log(f"  gen-shift F{fi}: X={Xa.shape} RSS={rss_curr_mb():.0f}MB")
        del Xa, Xb, Xa2, Xb2, ridge
        purge()

    del K_tr, K_or
    purge()

    u_ridge = unitize(zrk(gen_oof) if gen_oof.min() >= 0 and gen_oof.max() <= 1.5 else gen_oof)
    best_r = None
    for w in [0.01, 0.02, 0.03, 0.04, 0.05]:
        ev = eval_delta(y, z_base, z_base + w * u_ridge.astype(np.float64), folds, m0)
        ev["w"] = w
        if best_r is None or (ev["pos"], ev["d5_mean"]) > (best_r["pos"], best_r["d5_mean"]):
            best_r = ev
        log(f"  genRidge @w={w}: 5F={ev['d5_mean']:+.2f}e-5 pos={ev['pos']}/5 F0={ev.get('d0', float('nan')):+.2f}e-5")

    best_l = None
    if have_lgb and np.any(lgb_oof != 0):
        u_lgb = unitize(lgb_oof)
        for w in [0.01, 0.02, 0.03, 0.04, 0.05]:
            ev = eval_delta(y, z_base, z_base + w * u_lgb.astype(np.float64), folds, m0)
            ev["w"] = w
            if best_l is None or (ev["pos"], ev["d5_mean"]) > (best_l["pos"], best_l["d5_mean"]):
                best_l = ev
            log(f"  genLGB @w={w}: 5F={ev['d5_mean']:+.2f}e-5 pos={ev['pos']}/5 F0={ev.get('d0', float('nan')):+.2f}e-5")
        np.save(ART / "u_gen_shift_lgb_oof.npy", u_lgb)
    np.save(ART / "u_gen_shift_ridge_oof.npy", u_ridge)

    # Fold0 SubCV vs RayA
    best_ray_r, _ = subcv_delta(y_f0, rayA_f0, u_ridge[idx_f0], subfolds)
    best_s7_r, _ = subcv_delta(y_f0, s7_f0, u_ridge[idx_f0], subfolds)
    log(f"  genRidge vs RayA_f0: pos={best_ray_r['pos']}/5 mean={best_ray_r['mean']:+.2f}e-5")
    out = {
        "n_keys": n_use,
        "best_ridge_5f": best_r,
        "best_lgb_5f": best_l,
        "vs_S7_f0": best_s7_r,
        "vs_RayA_f0": best_ray_r,
    }
    if best_l is not None:
        best_ray_l, _ = subcv_delta(y_f0, rayA_f0, unitize(lgb_oof)[idx_f0], subfolds)
        out["vs_RayA_f0_lgb"] = best_ray_l
        log(f"  genLGB vs RayA_f0: pos={best_ray_l['pos']}/5 mean={best_ray_l['mean']:+.2f}e-5")
    return out


def build_rayA_f0(y, folds, m0, idx_f0, z_ref, d1):
    """Reconstruct S7_proxy and RayA peak (@0.75e-4) on Fold0 rows — mirrors v8."""
    u_sb = unit(d1)
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    m1 = unitize(0.65 * u_sb.astype(np.float64) - 0.35 * (-unit(gpt)).astype(np.float64) - 0.20 * gpt.astype(np.float64))
    m2 = unitize(0.65 * u_sb.astype(np.float64) + 0.35 * u_sb.astype(np.float64))
    s6 = unitize(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    w_s7 = 0.055
    dw = (0.75e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_rayA = z_ref.astype(np.float64) + (w_s7 + dw) * s6.astype(np.float64)
    s7_f0 = z_s7[idx_f0]
    rayA_f0 = z_rayA[idx_f0]
    base0 = float(roc_auc_score(y[m0], z_ref[m0]))
    d0_s7 = (float(roc_auc_score(y[m0], z_s7[m0])) - base0) * 1e5
    d0_ray = (float(roc_auc_score(y[m0], z_rayA[m0])) - base0) * 1e5
    log(f"  S7_f0 Δ={d0_s7:+.2f}e-5  RayA@0.75e-4 Δ={d0_ray:+.2f}e-5 (abs peak target ~+18.54)")
    del m1, m2, s6, gpt
    purge()
    return s7_f0.astype(np.float64), rayA_f0.astype(np.float64), {"d0_s7": d0_s7, "d0_rayA": d0_ray}


def write_report(h1, h2, h3, ray_meta, rss_peak):
    def fmt_row(name, d5, pos, f0, f0pos, disp="—", ks="—"):
        d5s = f"{d5:+.2f}" if d5 is not None else "—"
        f0s = f"{f0:+.2f}" if f0 is not None else "—"
        poss = f"{pos}/5" if pos is not None else "—"
        f0ps = f"{f0pos}/5" if f0pos is not None else "—"
        return f"| {name} | {d5s} | {poss} | {f0s} | {f0ps} | {disp} | {ks} |"

    md = []
    md.append("# DS First-Principles AUC Research")
    md.append("")
    md.append("**NO_SUBMIT=True**. No notebook scraping. No submission CSVs.")
    md.append(f"Peak process RSS ≈ **{rss_peak:.0f} MB** (soft cap 500).")
    md.append("")
    md.append(f"Ray-A Fold0 reference: S7 Δ=`{ray_meta['d0_s7']:+.2f}e-5`, RayA@0.75e-4 Δ=`{ray_meta['d0_rayA']:+.2f}e-5`.")
    md.append("")
    md.append("## Summary comparison")
    md.append("")
    md.append("| Hypothesis | 5-Fold ΔOOF e-5 | pos/5 | Fold-0 SubCV ΔOOF e-5 (vs RayA) | SubCV pos/5 | disp | KS |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")

    # H1 L7
    md.append(
        fmt_row(
            "H1 L7_EB_10hat (vs RayA SubCV)",
            None,
            None,
            h1["L7_vs_RayA"]["mean"],
            h1["L7_vs_RayA"]["pos"],
            f"{h1['L7_disp_ks'][0]:.4f}",
            f"{h1['L7_disp_ks'][1]:.6f}",
        )
    )
    g5 = h1["gpt_best_5f"]
    md.append(
        fmt_row(
            "H1 gpt2m_EB_8hat",
            g5["d5_mean"],
            g5["pos"],
            h1["gpt_vs_RayA"]["mean"],
            h1["gpt_vs_RayA"]["pos"],
        )
    )
    md.append(
        fmt_row(
            "H2 WMW-AUC 22-D",
            h2["best5"]["d5_mean"],
            h2["best5"]["pos"],
            h2["vs_RayA_f0"]["mean"],
            h2["vs_RayA_f0"]["pos"],
        )
    )
    md.append(
        fmt_row(
            "H3 gen-shift Ridge",
            h3["best_ridge_5f"]["d5_mean"],
            h3["best_ridge_5f"]["pos"],
            h3["vs_RayA_f0"]["mean"],
            h3["vs_RayA_f0"]["pos"],
        )
    )
    if h3.get("best_lgb_5f"):
        md.append(
            fmt_row(
                "H3 gen-shift LGB",
                h3["best_lgb_5f"]["d5_mean"],
                h3["best_lgb_5f"]["pos"],
                h3.get("vs_RayA_f0_lgb", {}).get("mean"),
                h3.get("vs_RayA_f0_lgb", {}).get("pos"),
            )
        )

    md.append("")
    md.append("## Hypothesis 1 — Permutation truncation & Bayes reconstruction")
    md.append("")
    md.append(f"- Solo per-order Fold0 AUCs: `{[round(a,7) for a in h1['solo_perm_auc']]}`")
    md.append(f"- mean4 AUC=`{h1['auc_mean4']:.7f}` mean10 AUC=`{h1['auc_mean10']:.7f}` **Δ10vs4=`{h1['d10_vs_4_e5']:+.2f}e-5`**")
    md.append(f"- bayes4=`{h1['auc_bayes4']:.7f}` bayes10=`{h1['auc_bayes10']:.7f}` Δ=`{h1['d_bayes10_vs_4_e5']:+.2f}e-5`")
    md.append(f"- best κ-EB on 4-order: κ=`{h1['best_kappa']}` AUC=`{h1['auc_kappa']:.7f}`")
    md.append(f"- L7_EB_10hat solo=`{h1['auc_L7_EB_10hat']:.7f}` vs naive4=`{h1['auc_naive4']:.7f}`")
    md.append(f"- vs S7 SubCV: `{h1['L7_vs_S7']}`")
    md.append(f"- vs RayA SubCV: `{h1['L7_vs_RayA']}`")
    md.append(f"- gpt2 mean Δ8vs4=`{h1['gpt_d84_mean_e5']:+.2f}e-5` folds=`{h1['gpt_diag']}`")
    md.append(f"- gpt best 5F: `{h1['gpt_best_5f']}`")
    md.append("")
    md.append("## Hypothesis 2 — Direct WMW pairwise AUC")
    md.append("")
    md.append(f"- base z_ref+0.04·δ1 AUC=`{h2['base_auc']:.7f}`")
    md.append(f"- best 5F: `{h2['best5']}`")
    md.append(f"- vs S7_f0: `{h2['vs_S7_f0']}`")
    md.append(f"- vs RayA_f0: `{h2['vs_RayA_f0']}`")
    md.append("")
    md.append("## Hypothesis 3 — Generator conditional shift")
    md.append("")
    md.append(f"- keys used=`{h3['n_keys']}`")
    md.append(f"- Ridge 5F: `{h3['best_ridge_5f']}`")
    md.append(f"- LGB 5F: `{h3.get('best_lgb_5f')}`")
    md.append(f"- vs RayA_f0 Ridge: `{h3['vs_RayA_f0']}`")
    md.append("")
    md.append("**NO_SUBMIT=True** — research only; no CSVs staged.")
    text = "\n".join(md) + "\n"
    rp = LOG / "DS_FIRST_PRINCIPLES_RESEARCH.md"
    rp.write_text(text)
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / "DS_FIRST_PRINCIPLES_RESEARCH.md").write_text(text)
    except Exception as e:
        log(f"vol sync warn: {e}")
    log(f"REPORT → {rp}")
    return rp


def main():
    log("======== DS FIRST-PRINCIPLES AUC RESEARCH (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"]
    )["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    idx_f0 = np.where(m0)[0]
    assert int(m0.sum()) == 139992
    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    d1 = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    log(f"ntr={len(y)} z_ref AUC={roc_auc_score(y, z_ref):.7f}")

    s7_f0, rayA_f0, ray_meta = build_rayA_f0(y, folds, m0, idx_f0, z_ref, d1)

    h1 = hypothesis1(y, folds, fold_id, m0, idx_f0, z_ref, d1, s7_f0, rayA_f0)
    h2 = hypothesis2(y, folds, fold_id, m0, idx_f0, z_ref, d1, s7_f0, rayA_f0, h1)
    h3 = hypothesis3(y, folds, fold_id, m0, idx_f0, z_ref, d1, s7_f0, rayA_f0)

    (ART / "research_summary.json").write_text(
        json.dumps({"h1": h1, "h2": h2, "h3": h3, "ray_meta": ray_meta, "rss_max": rss_max_mb()}, indent=2, default=float)
    )
    write_report(h1, h2, h3, ray_meta, max(rss_curr_mb(), rss_max_mb()))
    # sync script
    try:
        (VOL / "scripts").mkdir(parents=True, exist_ok=True)
        src = Path(__file__).resolve()
        (VOL / "scripts" / "ds_first_principles_auc_research.py").write_text(src.read_text())
    except Exception as e:
        log(f"script sync warn: {e}")
    log("======== DS RESEARCH DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
