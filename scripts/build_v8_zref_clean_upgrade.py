#!/usr/bin/env python3
"""Leak-free xgb_lm_super_clean_5f + v8/z_ref clean upgrade (Mac CPU, float32, n_jobs=4)."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from gm_nullspace_strike_orth_v3 import align_method_b_smooth, disp_ks
from recon_and_forensic_audit import build_z_ref

T0 = time.time()
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
MV2 = PRED / "matrix_v2"
P12 = ROOT / "external/nov08_part12"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
ART = PRED / "gm_v2_ladder"
ART.mkdir(parents=True, exist_ok=True)
(PRED / "gpt2_3ep").mkdir(parents=True, exist_ok=True)
W_SUPER = 0.075
EPS = 1e-6


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1.0 - EPS))


def rk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)


def qmap_to_oof(x_test, x_oof):
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    o = np.sort(np.asarray(x_oof, dtype=np.float64).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def bayes_fold(arr, prior):
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    za = np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1)
    ya = np.asarray(y_a, dtype=np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, dtype=np.float64).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def build_kmat_te(K_tr, K_te, y, folds, n_keys):
    ntr = len(y)
    K_oof = np.zeros((ntr, n_keys), dtype=np.float32)
    K_te_folds = []
    for a, b in folds:
        prior = float(y[a].mean())
        te_b, te_te = [], []
        for j in range(n_keys):
            ka = K_tr[a, j]
            if np.issubdtype(ka.dtype, np.floating) or len(np.unique(ka)) > 500:
                edges = np.unique(np.quantile(ka, np.linspace(0, 1, 32)))
                ba = np.digitize(ka, edges)
                bb = np.digitize(K_tr[b, j], edges)
                bt = np.digitize(K_te[:, j], edges)
            else:
                ba, bb, bt = ka, K_tr[b, j], K_te[:, j]
            tmp = pd.DataFrame({"k": ba, "y": y[a]})
            gstat = tmp.groupby("k")["y"].agg(["sum", "count"])
            m = 20.0
            te_map = ((gstat["sum"] + prior * m) / (gstat["count"] + m)).to_dict()
            te_b.append(np.array([te_map.get(k, prior) for k in bb], dtype=np.float32))
            te_te.append(np.array([te_map.get(k, prior) for k in bt], dtype=np.float32))
        K_oof[b] = np.column_stack(te_b)
        K_te_folds.append(np.column_stack(te_te))
    return K_oof, np.mean(K_te_folds, axis=0).astype(np.float32)


def build_rect_basis(specs, z_base_o, z_base_t):
    R_o, R_t, metas = [], [], []
    for name, xo, xt in specs:
        xt_rect = qmap_to_oof(xt, xo)
        zo, zt = zrk(xo), zrk(xt_rect)
        beta = float(np.dot(zo, z_base_o) / (np.dot(z_base_o, z_base_o) + 1e-12))
        ro = (zo - beta * z_base_o)
        rt = (zt - beta * z_base_t)
        sd = float(ro.std() + 1e-12)
        ro, rt = ro / sd, rt / sd
        rt = qmap_to_oof(rt, ro)
        ro = ro - float(ro.mean())
        rt = rt - float(rt.mean())
        s = float(ro.std() + 1e-12)
        ro, rt = ro / s, rt / s
        metas.append({"name": name, "beta": beta, "ks_resid": float(ks_2samp(ro, rt).statistic)})
        R_o.append(ro)
        R_t.append(rt)
        log(f"  rect {name}: β={beta:+.4f} KS={metas[-1]['ks_resid']:.6f}")
    return np.column_stack(R_o), np.column_stack(R_t), metas


def nested_coord_ascent(y, z_base, R_o, folds, names):
    M = len(names)
    grid = (-0.04, -0.03, -0.022, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.022, 0.03, 0.04)
    oof_z = np.zeros(len(y), dtype=np.float64)
    fold_w, fold_d = [], []
    for fi, (a, b) in enumerate(folds):
        w = np.zeros(M, dtype=np.float64)
        best = float(roc_auc_score(y[a], z_base[a]))
        for _ in range(3):
            for j in range(M):
                bw = w[j]
                for cw in grid:
                    wt = w.copy()
                    wt[j] = cw
                    sc = float(roc_auc_score(y[a], z_base[a] + R_o[a] @ wt))
                    if sc > best + 1e-9:
                        best, bw = sc, cw
                w[j] = bw
        oof_z[b] = z_base[b] + R_o[b] @ w
        d = float(roc_auc_score(y[b], oof_z[b]) - roc_auc_score(y[b], z_base[b]))
        fold_w.append(w)
        fold_d.append(d)
        log(f"  Fold {fi}: Δ={d*1e5:+.2f}e-5")
    return oof_z, np.mean(fold_w, 0), fold_w, fold_d


def orth_vs(zo, zt, bases_o, bases_t, folds):
    Xo = np.column_stack(bases_o)
    Xt = np.column_stack(bases_t)
    ro = np.zeros(len(zo))
    rtf = []
    for a, b in folds:
        beta, *_ = np.linalg.lstsq(Xo[a], zo[a], rcond=None)
        ra = zo[a] - Xo[a] @ beta
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (zo[b] - Xo[b] @ beta - mu) / sd
        rtf.append((zt - Xt @ beta - mu) / sd)
    return ro, np.mean(rtf, 0)


def main():
    log("V8_ZREF_CLEAN_UPGRADE START")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    ntr = len(y)

    # ---- 1. lm12_raw_super ----
    log("=== Stage2.1 z_lm12_raw_super ===")
    z_lm_raw_o = zrk(np.load(P12 / "lm_oof.npy").ravel())
    z_lm_raw_t = zrk(np.load(P12 / "lm_test.npy").ravel())
    _, z_lm_raw_t_a = align_method_b_smooth(z_lm_raw_o, z_lm_raw_t, folds)

    z_m_o = np.zeros(ntr)
    z_m_t_folds = []
    for f, (a, b) in enumerate(folds):
        prior = float(y[a].mean())
        z_m_o[b] = zrk(bayes_fold(np.load(MED / f"oof_{f}.npy"), prior))
        z_m_t_folds.append(zrk(bayes_fold(np.load(MED / f"test_{f}.npy"), prior)))
    z_m_t_raw = np.mean(z_m_t_folds, 0)
    z_m_o_a, z_m_t_a = align_method_b_smooth(z_m_o, z_m_t_raw, folds)
    d_m, ks_m = disp_ks(z_m_o_a, z_m_t_a)
    log(f"  z_m Method-B disp={d_m:.6f} KS={ks_m:.6f}")

    r_o = np.zeros(ntr)
    r_t_folds, betas = [], []
    for f, (a, b) in enumerate(folds):
        zl, zm = z_lm_raw_o[b], z_m_o[b]
        beta = float(np.cov(zm, zl, ddof=0)[0, 1] / (np.var(zl) + 1e-12))
        betas.append(beta)
        r_o[b] = unitize(zm - beta * zl)
        r_t_folds.append(z_m_t_a - beta * z_lm_raw_t_a)
    r_o_a, r_t_a = align_method_b_smooth(r_o, np.mean(r_t_folds, 0), folds)
    d_r, ks_r = disp_ks(r_o_a, r_t_a)
    log(f"  r_m Method-B disp={d_r:.6f} KS={ks_r:.6f}")

    z_super_o = z_lm_raw_o + W_SUPER * r_o
    z_super_t = z_lm_raw_t_a + W_SUPER * r_t_a
    z_super_o_a, z_super_t_a = align_method_b_smooth(z_super_o, z_super_t, folds)
    raw_f = fold_aucs(y, z_lm_raw_o, folds)
    super_f = fold_aucs(y, z_super_o, folds)
    d_super = [(a - b) * 1e5 for a, b in zip(super_f, raw_f)]
    log(f"  super Δe5={['%+.2f' % d for d in d_super]} pos={sum(d > 0 for d in d_super)}/5")
    np.save(ART / "z_lm12_raw_super_oof.npy", z_super_o)
    np.save(ART / "z_lm12_raw_super_test.npy", z_super_t_a)
    np.save(ART / "z_m_bayes_5f_oof.npy", z_m_o)
    np.save(ART / "z_m_bayes_5f_test.npy", z_m_t_a)

    # ---- 2. xgb_lm_super_clean_5f ----
    log("=== Stage2.2 xgb_lm_super_clean_5f ===")
    os.environ["INPUT_DIR"] = str(ROOT)
    os.environ.setdefault("OUT_DIR", str(ROOT / "cache" / "gboost"))
    import gboost as gb

    log(f"  gboost KMAT={gb.KMAT.shape} BASE={gb.BASE.shape}")
    K = gb.KMAT
    K_tr, K_te = K[:ntr], K[ntr:]
    n_keys = min(108, K_tr.shape[1])
    K_oof, K_te_m = build_kmat_te(K_tr, K_te, y, folds, n_keys)

    def ab(o_path, t_path):
        o = zrk(np.load(o_path).ravel())
        t = zrk(np.load(t_path).ravel())
        return align_method_b_smooth(o, t, folds)

    z_v4_o, z_v4_t = ab(PRED / "P_super_v4p_hc_oof.npy", PRED / "P_super_v4p_hc_test.npy")
    z_rev_o, z_rev_t = ab(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy")
    l43_o, l43_t = ab(MV2 / "P4_L4.3_pure5f_oof.npy", MV2 / "P4_L4.3_pure5f_test.npy")
    l42_o, l42_t = ab(MV2 / "P4_L4.2_pure5f_oof.npy", MV2 / "P4_L4.2_pure5f_test.npy")
    l45_o, l45_t = ab(MV2 / "P4_L4.5_pure5f_oof.npy", MV2 / "P4_L4.5_pure5f_test.npy")
    z_tab_o, z_tab_t = ab(HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy")

    r_rare = K_oof[:, 0].astype(np.float64)
    r_disp = K_oof[:, min(1, n_keys - 1)].astype(np.float64)
    r_rare_t = K_te_m[:, 0].astype(np.float64)
    r_disp_t = K_te_m[:, min(1, n_keys - 1)].astype(np.float64)
    zs_o, zs_t = z_super_o_a, z_super_t_a
    zm_o, zm_t = z_m_o_a, z_m_t_a

    extra_o = np.column_stack(
        [zs_o, zm_o, z_rev_o, zs_o - z_v4_o, zs_o * r_rare, zs_o * r_disp, l43_o, l42_o, l45_o, z_tab_o]
    ).astype(np.float32)
    extra_t = np.column_stack(
        [zs_t, zm_t, z_rev_t, zs_t - z_v4_t, zs_t * r_rare_t, zs_t * r_disp_t, l43_t, l42_t, l45_t, z_tab_t]
    ).astype(np.float32)
    for j in range(extra_o.shape[1]):
        eo, et = align_method_b_smooth(extra_o[:, j], extra_t[:, j], folds)
        extra_o[:, j] = eo.astype(np.float32)
        extra_t[:, j] = et.astype(np.float32)

    X_o = np.hstack([K_oof, extra_o, gb.BASE[:ntr].astype(np.float32)])
    X_t = np.hstack([K_te_m, extra_t, gb.BASE[ntr:].astype(np.float32)])
    log(f"  X_fusion o={X_o.shape} t={X_t.shape}")

    v5_o = zrk(np.load(PRED / "P_super_v5_harvest_oof.npy").ravel())
    v5_t = zrk(np.load(PRED / "P_super_v5_harvest_test.npy").ravel())
    specs = [(42 + i, 2) for i in range(3)] + [(100 + i, 3) for i in range(3)]
    oof_bag = np.zeros(ntr)
    te_bag = []
    for bi, (seed, depth) in enumerate(specs):
        oof_s = np.zeros(ntr)
        te_s = []
        for fi, (a, b) in enumerate(folds):
            tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
            bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
            dtr = xgb.DMatrix(X_o[tr_in], label=y[tr_in], base_margin=bm_tr)
            des = xgb.DMatrix(X_o[es_in], label=y[es_in], base_margin=bm_all[es_in])
            dva = xgb.DMatrix(X_o[b], label=y[b], base_margin=bm_all[b])
            dte = xgb.DMatrix(X_t, base_margin=bm_te)
            params = dict(
                max_depth=depth, eta=0.025, subsample=0.8, colsample_bynode=0.35,
                reg_lambda=30.0, min_child_weight=200, objective="binary:logistic",
                eval_metric="auc", tree_method="hist", max_bin=256, nthread=4, seed=seed,
            )
            booster = xgb.train(
                params, dtr, num_boost_round=400, evals=[(des, "es")],
                early_stopping_rounds=40, verbose_eval=False,
            )
            oof_s[b] = booster.predict(dva)
            te_s.append(booster.predict(dte))
            log(f"  bag{bi} d{depth} F{fi} best_iter={booster.best_iteration}")
        oof_bag += oof_s
        te_bag.append(np.mean(te_s, 0))
    oof_xgb = oof_bag / len(specs)
    te_xgb = np.mean(te_bag, 0)
    xgb_folds = fold_aucs(y, zrk(oof_xgb), folds)
    xgb_auc = float(roc_auc_score(y, zrk(oof_xgb)))
    log(f"  xgb_clean AUC={xgb_auc:.7f} folds={[f'{x:.7f}' for x in xgb_folds]}")
    np.save(PRED / "gpt2_3ep/xgb_lm_super_clean_5f_oof.npy", oof_xgb)
    np.save(PRED / "gpt2_3ep/xgb_lm_super_clean_5f_test.npy", te_xgb)

    # ---- 3. z_v8_clean ----
    log("=== Stage2.3 z_v8_clean ===")
    w12_o = np.load(P12 / "wide_lm_stack_nested.npy").ravel().astype(np.float64)
    w12_t = np.load(P12 / "wide_lm_stack_test.npy").ravel().astype(np.float64)
    p9_t = np.load(P12 / "part9_test.npy").ravel().astype(np.float64)
    z_w12_o, z_w12_t = zrk(w12_o), zrk(w12_t)
    beta_c01 = float(np.cov(z_m_o, z_w12_o, ddof=0)[0, 1] / (np.var(z_w12_o) + 1e-12))
    c01_o, c01_t = align_method_b_smooth(z_m_o - beta_c01 * z_w12_o, z_m_t_a - beta_c01 * z_w12_t, folds)

    specs_r = [
        ("v4p_hc", np.load(PRED / "P_super_v4p_hc_oof.npy").ravel(), np.load(PRED / "P_super_v4p_hc_test.npy").ravel()),
        ("B_xgb_llr", np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel(), np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel()),
        ("P4_L4.3", np.load(MV2 / "P4_L4.3_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.3_pure5f_test.npy").ravel()),
        ("P4_L4.2", np.load(MV2 / "P4_L4.2_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.2_pure5f_test.npy").ravel()),
        ("P4_L4.5", np.load(MV2 / "P4_L4.5_pure5f_oof.npy").ravel(), np.load(MV2 / "P4_L4.5_pure5f_test.npy").ravel()),
        ("P4_L4.3_ext", np.load(MV2 / "P4_L4.3_ext_oof.npy").ravel(), np.load(MV2 / "P4_L4.3_ext_test.npy").ravel()),
        ("gpt2_bidir", np.load(PRED / "gpt2_3ep/gpt2_bidir_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/gpt2_bidir_test.npy").ravel()),
        ("xgb_gpt2_3ep", np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_test.npy").ravel()),
        ("xgb_lm_super_clean", oof_xgb, te_xgb),
        ("lgb_gpt2_3ep_fixed", np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel()),
        ("c01_gpt2m_lm12", c01_o, c01_t),
    ]
    R_o, R_t, metas = build_rect_basis(specs_r, z_w12_o, z_w12_t)
    names = [m["name"] for m in metas]
    oof_v8c_z, mean_w, _, fold_d = nested_coord_ascent(y, z_w12_o, R_o, folds, names)
    v8c_auc = float(roc_auc_score(y, oof_v8c_z))
    v8c_folds = fold_aucs(y, oof_v8c_z, folds)
    v8_old = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    v8_old_auc = float(roc_auc_score(y, v8_old))
    v8_old_folds = fold_aucs(y, v8_old, folds)
    d_v8 = [(a - b) * 1e5 for a, b in zip(v8c_folds, v8_old_folds)]
    log(f"  v8 {v8_old_auc:.7f}->{v8c_auc:.7f} Δ={((v8c_auc-v8_old_auc)*1e5):+.2f}e-5 pos={sum(d>0 for d in d_v8)}/5")

    dz_oof = R_o @ mean_w
    dz_test = qmap_to_oof(R_t @ mean_w, dz_oof)
    z_p12_fsl = zrk(0.68 * zrk(w12_t) + 0.32 * zrk(p9_t))
    beta_fsl = float(np.dot(dz_test, z_p12_fsl) / (np.dot(z_p12_fsl, z_p12_fsl) + 1e-12))
    dz_orth_fsl = qmap_to_oof(dz_test - beta_fsl * z_p12_fsl, dz_oof)
    z_v8_clean_t_rank = rk(z_p12_fsl + 1.0 * dz_orth_fsl)
    z_v8_clean_t = zrk(z_v8_clean_t_rank)
    np.save(ART / "z_v8_clean_oof.npy", oof_v8c_z)
    np.save(ART / "z_v8_clean_test.npy", z_v8_clean_t)
    np.save(ART / "z_v8_clean_test_rank.npy", z_v8_clean_t_rank)

    # ---- 4. z_ref_clean ----
    log("=== Stage2.4 z_ref_clean ===")
    z_v8_o, z_v8_t = oof_v8c_z, z_v8_clean_t
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = zrk(np.load(HARV / "test_tabm_5f.npy").ravel())
    z_xgbf_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_t = zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel())
    r_old_o, r_old_t = orth_vs(z_xgbf_o, z_xgbf_t, [z_v8_o], [z_v8_t], folds)
    r_tab_o, r_tab_t = orth_vs(z_tabm_o, z_tabm_t, [z_v8_o], [z_v8_t], folds)
    r_old_o, r_old_t = align_method_b_smooth(r_old_o, r_old_t, folds)
    r_tab_o, r_tab_t = align_method_b_smooth(r_tab_o, r_tab_t, folds)
    X_orth_o = np.column_stack([z_v8_o, z_w12_o, r_old_o - r_tab_o])
    X_orth_t = np.column_stack([z_v8_t, z_w12_t, r_old_t - r_tab_t])

    N09 = ROOT / "external/nov09_top3_recon"
    sam_o = pd.read_csv(N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/oof.csv")["nested_winner"].to_numpy(float)
    sam_t = pd.read_csv(N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/submission_cavg.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    specs5 = [
        (np.load(PRED / "part11/lab4_tabular_nade_oof.npy").ravel(), np.load(PRED / "part11/lab4_tabular_nade_test.npy").ravel()),
        (sam_o, sam_t),
        (np.load(HARV / "oof_orig20k_knn.npy").ravel(), np.load(HARV / "test_orig20k_knn.npy").ravel()),
        (np.load(MV2 / "harvest_p4_cuda_xgb_oof.npy").ravel(), np.load(MV2 / "harvest_p4_cuda_xgb_test.npy").ravel()),
        (np.load(PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy").ravel(), np.load(PRED / "part11/lab1_zjoint_graph_lgb4_test.npy").ravel()),
    ]
    U5_o, U5_t = [], []
    for xo, xt in specs5:
        zo, zt = zrk(xo), zrk(xt)
        ro, rt = orth_vs(zo, zt, [X_orth_o[:, j] for j in range(3)], [X_orth_t[:, j] for j in range(3)], folds)
        ro, rt = align_method_b_smooth(ro, rt, folds)
        U5_o.append(ro)
        U5_t.append(rt)
    U5_o, U5_t = np.column_stack(U5_o), np.column_stack(U5_t)
    w5 = np.array([0.0135, 0.0115, -0.0085, -0.0085, -0.0105])
    X7_o = np.column_stack([X_orth_o, U5_o])
    X7_t = np.column_stack([X_orth_t, U5_t])
    specs2 = [
        (np.load(PRED / "ladder/glm_quad_oof.npy").ravel(), np.load(PRED / "ladder/glm_quad_test.npy").ravel()),
        (np.load(PRED / "part11/lab1_glm_plus_graph_oof.npy").ravel(), np.load(PRED / "part11/lab1_glm_plus_graph_test.npy").ravel()),
    ]
    U2_o, U2_t = [], []
    for xo, xt in specs2:
        zo, zt = zrk(xo), zrk(xt)
        ro, rt = orth_vs(zo, zt, [X7_o[:, j] for j in range(X7_o.shape[1])], [X7_t[:, j] for j in range(X7_t.shape[1])], folds)
        ro, rt = align_method_b_smooth(ro, rt, folds)
        U2_o.append(ro)
        U2_t.append(rt)
    U2_o, U2_t = np.column_stack(U2_o), np.column_stack(U2_t)
    w2 = np.array([0.0150, 0.0125])
    z_ref_clean_o = z_v8_o + 0.025 * (r_old_o - r_tab_o) + 0.85 * (U5_o @ w5) + 0.85 * (U2_o @ w2)
    z_ref_clean_t = z_v8_t + 0.025 * (r_old_t - r_tab_t) + 0.85 * (U5_t @ w5) + 0.85 * (U2_t @ w2)
    _, z_ref_clean_t_a = align_method_b_smooth(z_ref_clean_o, z_ref_clean_t, folds)
    z_ref_c_auc = float(roc_auc_score(y, z_ref_clean_o))
    z_ref_c_folds = fold_aucs(y, z_ref_clean_o, folds)
    z_ref_old, _, _, z_ref_old_auc = build_z_ref(y, test_ids, folds)
    z_ref_old_folds = fold_aucs(y, z_ref_old, folds)
    d_ref = [(a - b) * 1e5 for a, b in zip(z_ref_c_folds, z_ref_old_folds)]
    log(f"  z_ref {z_ref_old_auc:.7f}->{z_ref_c_auc:.7f} Δ={((z_ref_c_auc-z_ref_old_auc)*1e5):+.2f}e-5 pos={sum(d>0 for d in d_ref)}/5")
    np.save(ART / "z_ref_clean_oof.npy", z_ref_clean_o)
    np.save(ART / "z_ref_clean_test.npy", z_ref_clean_t_a)

    summary = {
        "lm12_raw_super": {"delta_1e5": d_super, "folds": super_f, "pos": int(sum(d > 0 for d in d_super))},
        "xgb_lm_super_clean_5f": {"auc": xgb_auc, "folds": xgb_folds, "X_shape": list(X_o.shape)},
        "v8": {"old_auc": v8_old_auc, "clean_auc": v8c_auc, "delta_1e5": d_v8, "pos": int(sum(d > 0 for d in d_v8)), "clean_folds": v8c_folds, "old_folds": v8_old_folds},
        "z_ref": {"old_auc": z_ref_old_auc, "clean_auc": z_ref_c_auc, "delta_1e5": d_ref, "pos": int(sum(d > 0 for d in d_ref)), "clean_folds": z_ref_c_folds, "old_folds": z_ref_old_folds},
        "elapsed_s": round(time.time() - T0, 1),
    }
    (LOG / "V8_ZREF_CLEAN_UPGRADE.json").write_text(json.dumps(summary, indent=2))
    log("V8_ZREF_CLEAN_UPGRADE_COMPLETE")
    return summary


if __name__ == "__main__":
    main()
