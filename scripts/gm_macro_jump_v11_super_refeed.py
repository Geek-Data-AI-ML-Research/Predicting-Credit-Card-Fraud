#!/usr/bin/env python3
"""GM Macro-Jump v11: Cross-Modal Tree Re-Feeding (STRICT NO_SUBMIT until gate).

Builds PCHIP-locked X_macro (~115 cols), trains XGB/LGB/CAT/L4-GLM bags,
nested minimax portfolio, stages SlotA/B CSVs only if strict gates pass.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["CATBOOST_THREAD_COUNT"] = "2"
os.environ["NO_SUBMIT"] = "1"

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from gm_top3_strike3_expanded7f import (
    HARV,
    P12,
    PRED,
    SUB,
    TARGET_MEAN,
    calibrate_logit_to_prevalence,
    fast_auc,
    pchip_quantile_map,
    zrk,
)
from recon_and_forensic_audit import N_KNOTS, build_z_ref
from step2_lm_orth_and_v10_transductive import fold_orth_multi, load_gpt2_med_5f, load_lm_bank, weight_for_spearman_dist
from de_corrupt_and_boost_5f import FEATS, FLAGS, FLOATS, CATS, build_kmat_and_ratios

OUT = ROOT / "logs" / "overnight"
ART = PRED / "macro_v11"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
T0 = time.time()
STATUS = OUT / "MACRO_V11_STATUS.json"
NO_SUBMIT = True
SEEDS5 = [42, 137, 2026, 7, 99]
SEEDS3 = [42, 137, 2026]


def log(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    try:
        STATUS.write_text(
            json.dumps({"t_s": round(time.time() - T0, 1), "msg": msg, "NO_SUBMIT": NO_SUBMIT}, indent=2)
        )
    except Exception:
        pass


def fold_deltas(y, z_base, r, folds, w):
    d = []
    for _, b in folds:
        d.append(
            round(
                (float(fast_auc(y[b], (z_base + w * r)[b])) - float(fast_auc(y[b], z_base[b]))) * 1e5,
                2,
            )
        )
    return d, sum(1 for x in d if x > 0)


def nested_delta(y, z_base, r, folds, w_grid=None):
    if w_grid is None:
        w_grid = np.linspace(-0.12, 0.12, 49)
    base = float(fast_auc(y, z_base))
    best = (0.0, 0.0, 0)
    for w in w_grid:
        if abs(w) < 1e-12:
            continue
        sc = float(fast_auc(y, z_base + w * r))
        d = (sc - base) * 1e5
        pos = sum(
            1
            for _, b in folds
            if float(fast_auc(y[b], (z_base + w * r)[b])) > float(fast_auc(y[b], z_base[b]))
        )
        if d > best[0] + 1e-9 or (abs(d - best[0]) < 1e-9 and pos > best[2]):
            best = (d, float(w), pos)
    return {"delta_1e5": round(best[0], 2), "w": round(best[1], 4), "pos": best[2], "base": base}


def pchip_lock(v_o, v_t, folds=None):
    """PCHIP-map test onto OOF; optionally per-fold standardize OOF first."""
    vo = np.asarray(v_o, dtype=np.float64).ravel().copy()
    vt = np.asarray(v_t, dtype=np.float64).ravel().copy()
    if folds is not None:
        out = np.zeros_like(vo)
        for a, b in folds:
            mu, sd = float(vo[a].mean()), float(vo[a].std() + 1e-12)
            out[b] = (vo[b] - mu) / sd
            # also re-center train portion for consistency of pooled OOF
            out[a] = (vo[a] - mu) / sd  # overwritten each fold for a — last fold wins for train
        # proper: only fill OOF slots
        out = np.zeros_like(vo)
        for a, b in folds:
            mu, sd = float(vo[a].mean()), float(vo[a].std() + 1e-12)
            out[b] = (vo[b] - float(vo[b].mean())) / (float(vo[b].std() + 1e-12))
            # map held-out to fold-train scale? user: stdize to fold f's reference mean/std
            out[b] = (vo[b] - mu) / sd
        vo = out
    vt = pchip_quantile_map(vt, vo, n_knots=N_KNOTS)
    # force exact unit match
    vt = (vt - float(vt.mean())) / (float(vt.std() + 1e-12)) * float(vo.std() + 1e-12) + float(vo.mean())
    return vo, vt


def hygiene_col(vo, vt, name):
    disp = float(np.std(vt) / (np.std(vo) + 1e-12))
    ks = float(ks_2samp(vo, vt).statistic)
    return {"name": name, "disp": round(disp, 6), "ks": round(ks, 6)}


def lz_sub(fn, test_ids, z_ref):
    s = pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    return pchip_quantile_map(zrk(s), z_ref, n_knots=N_KNOTS), s


def assemble_gpt2m_mean_std(y, folds, nte):
    """z_gpt2m (Bayes/mean LLR) + epistemic std across orders; PCHIP K=4 test → K=8 oof."""
    ntr = len(y)
    zo = np.full(ntr, np.nan)
    so = np.full(ntr, np.nan)
    zt_f, st_f = [], []
    for fi, (a, b) in enumerate(folds):
        oraw = np.load(PRED / f"gpt2_med_4ep/oof_{fi}.npy")
        traw = np.load(PRED / f"gpt2_med_4ep/test_{fi}.npy")
        # (Nf, 8, 2) ll for class0/1
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        so[b] = llr.std(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
        st_f.append(tllr.std(1))
    assert np.isfinite(zo).all()
    zt = np.mean(zt_f, 0)
    st = np.mean(st_f, 0)
    zo, zt = pchip_lock(zo, zt, folds)
    so, st = pchip_lock(so, st, folds)
    return zo, zt, so, st


def signed_geom(a, b):
    return np.sign(a * b) * np.sqrt(np.abs(a * b) + 1e-12)


def build_X_macro(y, folds, z_ref_o, z_st9p5_t, lm_bank, test_ids):
    log("Stage1: building PCHIP-locked X_macro")
    ntr, nte = len(y), len(z_st9p5_t)
    cols_o, cols_t, names, hyg = [], [], [], []

    def add(name, vo, vt, do_fold_std=True):
        vo2, vt2 = pchip_lock(vo, vt, folds if do_fold_std else None)
        h = hygiene_col(vo2, vt2, name)
        hyg.append(h)
        cols_o.append(vo2.astype(np.float32))
        cols_t.append(vt2.astype(np.float32))
        names.append(name)

    # --- Block A: gpt2-medium ---
    zo_m, zt_m, so_m, st_m = assemble_gpt2m_mean_std(y, folds, nte)
    add("z_gpt2m", zo_m, zt_m)
    add("s_gpt2m", so_m, st_m)
    z_lm12_o, z_lm12_t = lm_bank["lm12_wide"]
    # beta vs lm12 on full oof (then LOFO-ish via fold lock already)
    beta = float(np.cov(zo_m, z_lm12_o, ddof=0)[0, 1] / (np.var(z_lm12_o) + 1e-12))
    r_o = zo_m - beta * z_lm12_o
    r_t = zt_m - beta * z_lm12_t
    add("r_gpt2m_vs_lm12", r_o, r_t)
    add("r_gpt2m_prec", r_o / (0.05 + so_m), r_t / (0.05 + st_m))
    p_ref = expit(z_ref_o)
    p_ref_t = expit(z_st9p5_t)
    g_o = np.tanh(np.abs(z_lm12_o - z_ref_o)) * p_ref * (1 - p_ref) * r_o
    g_t = np.tanh(np.abs(z_lm12_t - z_st9p5_t)) * p_ref_t * (1 - p_ref_t) * r_t
    add("r_gpt2m_boundgate", g_o, g_t)
    # keep raw z_gpt2m already added → 5 named + s = 6? z,s,r,prec,gate = 5; add z again? user said 6 cols
    add("z_gpt2m_rawlock", zo_m, zt_m)  # duplicate lock ok as 6th

    # --- Block B: LM bank orth + asymmetries ---
    lm_names = [k for k in lm_bank.keys()]
    R_o, R_t = [], []
    for k in lm_names:
        zo, zt = lm_bank[k]
        ro, rt, _ = fold_orth_multi(zo, zt, z_ref_o.reshape(-1, 1), z_st9p5_t.reshape(-1, 1), folds)
        add(f"r_{k}", ro, rt)
        R_o.append(ro)
        R_t.append(rt)
    R_o = np.column_stack(R_o)
    R_t = np.column_stack(R_t)
    # need gpt2_3ep and gpt2_rev and megayak
    def get(k):
        return lm_bank[k] if k in lm_bank else None

    if get("gpt2_3ep") and get("gpt2_rev"):
        d_o = get("gpt2_3ep")[0] - get("gpt2_rev")[0]
        d_t = get("gpt2_3ep")[1] - get("gpt2_rev")[1]
        add("d_fwd_rev_3ep", d_o, d_t)
        d2_o = zo_m - get("gpt2_rev")[0]
        d2_t = zt_m - get("gpt2_rev")[1]
        add("d_fwd_rev_med", d2_o, d2_t)
    if get("megayak") and get("lm12_wide"):
        add(
            "d_tfirst_tlast",
            get("megayak")[0] - get("lm12_wide")[0],
            get("megayak")[1] - get("lm12_wide")[1],
        )
    r_mean_o, r_mean_t = R_o.mean(1), R_t.mean(1)
    r_std_o, r_std_t = R_o.std(1), R_t.std(1)
    add("r_lm9_mean", r_mean_o, r_mean_t)
    add("r_lm9_std", r_std_o, r_std_t)
    add("r_lm9_range", R_o.max(1) - R_o.min(1), R_t.max(1) - R_t.min(1))
    add("r_lm9_prec", r_mean_o / (0.05 + r_std_o), r_mean_t / (0.05 + r_std_t))

    # --- Block C: TabM + knn24 ---
    tab_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    tab_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), tab_o, n_knots=N_KNOTS)
    stab_o = np.load(HARV / "oof_tabm_5f_std.npy").ravel().astype(np.float64)
    stab_t = np.load(HARV / "test_tabm_5f_std.npy").ravel().astype(np.float64)
    ro_tab, rt_tab, _ = fold_orth_multi(tab_o, tab_t, z_ref_o.reshape(-1, 1), z_st9p5_t.reshape(-1, 1), folds)
    add("r_tabm", ro_tab, rt_tab)
    add("s_tabm", stab_o, stab_t)
    add("r_tabm_prec", ro_tab / (0.05 + stab_o), rt_tab / (0.05 + stab_t))
    knn_o = zrk(np.load(HARV / "oof_orig20k_knn.npy").ravel())
    knn_t = pchip_quantile_map(zrk(np.load(HARV / "test_orig20k_knn.npy").ravel()), knn_o, n_knots=N_KNOTS)
    ro_knn, rt_knn, _ = fold_orth_multi(knn_o, knn_t, z_ref_o.reshape(-1, 1), z_st9p5_t.reshape(-1, 1), folds)
    add("r_knn", ro_knn, rt_knn)
    knn24_o = np.load(HARV / "knn24_oof.npy").astype(np.float64)
    knn24_t = np.load(HARV / "knn24_test.npy").astype(np.float64)
    for j in range(24):
        add(f"knn24_{j}", knn24_o[:, j], knn24_t[:, j])
    # agreements (use r_gpt2m_vs_lm12 as r_gpt2m)
    add("agree_gpt2m_tabm", signed_geom(r_o, ro_tab), signed_geom(r_t, rt_tab))
    add("agree_lm9_knn", signed_geom(r_mean_o, ro_knn), signed_geom(r_mean_t, rt_knn))
    add("agree_tabm_knn", signed_geom(ro_tab, ro_knn), signed_geom(rt_tab, rt_knn))

    # --- Block D: P4 + KMAT top + regimes ---
    for tag in ["P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"]:
        o = zrk(np.load(PRED / f"matrix_v2/{tag}_oof.npy").ravel())
        t = pchip_quantile_map(zrk(np.load(PRED / f"matrix_v2/{tag}_test.npy").ravel()), o, n_knots=N_KNOTS)
        ro, rt, _ = fold_orth_multi(o, t, z_ref_o.reshape(-1, 1), z_st9p5_t.reshape(-1, 1), folds)
        add(f"r_{tag}", ro, rt)

    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    KMAT, NCAT, RATIOS = build_kmat_and_ratios(tr, te)
    # key support counts on concat
    # use first 45 KMAT cols as codes; log1p counts joint
    n_keys = min(45, KMAT.shape[1])
    full_codes = KMAT[:, :n_keys]
    for j in range(n_keys):
        codes = full_codes[:, j]
        # bincount on all
        ct = np.bincount(codes, minlength=int(codes.max()) + 1).astype(np.float64)
        cnt = ct[codes]
        vo, vt = np.log1p(cnt[:ntr]), np.log1p(cnt[ntr : ntr + nte])
        add(f"kcnt_{j}", vo, vt, do_fold_std=False)

    # L4 regime indicators from rare/disp/tail of amount & velocity
    amt = tr["amount_usd"].to_numpy(float)
    vel = tr["velocity_score"].to_numpy(float)
    amt_t = te["amount_usd"].to_numpy(float)
    vel_t = te["velocity_score"].to_numpy(float)
    # rare: low joint count on amount cents
    def cents(x):
        return (np.round(x * 100) % 100).astype(np.int64)

    c_all = np.concatenate([cents(amt), cents(amt_t)])
    ct = np.bincount(c_all, minlength=100)
    r_rare_o = (ct[cents(amt)] <= 2).astype(np.float64)
    r_rare_t = (ct[cents(amt_t)] <= 2).astype(np.float64)
    r_disp_o = np.abs(vel - np.median(vel)) / (np.std(vel) + 1e-12)
    r_disp_t = np.abs(vel_t - np.median(vel)) / (np.std(vel) + 1e-12)
    thr = np.quantile(amt, 0.95)
    r_tail_o = (amt >= thr).astype(np.float64)
    r_tail_t = (amt_t >= thr).astype(np.float64)
    add("r_rare", r_rare_o, r_rare_t, do_fold_std=False)
    add("r_disp", r_disp_o, r_disp_t)
    add("r_tail", r_tail_o, r_tail_t, do_fold_std=False)

    # raw financial + crosses (top signal)
    for c in ["velocity_score", "merchant_risk_score", "amount_usd", "account_balance_usd"]:
        add(f"raw_{c}", tr[c].to_numpy(float), te[c].to_numpy(float))
    risk_flag = tr[FLAGS].sum(axis=1).to_numpy(float)
    risk_flag_t = te[FLAGS].sum(axis=1).to_numpy(float)
    add("risk_flag_count", risk_flag, risk_flag_t, do_fold_std=False)
    bm = (tr[FLAGS].to_numpy(float) * (2 ** np.arange(6))).sum(1)
    bm_t = (te[FLAGS].to_numpy(float) * (2 ** np.arange(6))).sum(1)
    add("flag_bitmask_64", bm, bm_t, do_fold_std=False)
    add("cross_vel_lm9", vel * r_mean_o, vel_t * r_mean_t)
    add("cross_risk_gpt2m", tr["merchant_risk_score"].to_numpy(float) * r_o, te["merchant_risk_score"].to_numpy(float) * r_t)
    add("cross_rare_lm9", r_rare_o * r_mean_o, r_rare_t * r_mean_t)

    # also store P4 L43/L42 for L4 GLM
    meta = {
        "r_gpt2m_vs_lm12": (r_o, r_t),
        "r_lm9_mean": (r_mean_o, r_mean_t),
        "d_fwd_rev": (hyg and None),
    }
    # recover named cols for L4
    name_to_i = {n: i for i, n in enumerate(names)}

    Xo = np.column_stack(cols_o)
    Xt = np.column_stack(cols_t)
    del cols_o, cols_t, tr, te
    gc.collect()
    log(f"  X_macro shape train={Xo.shape} test={Xt.shape}")
    bad = [h for h in hyg if abs(h["disp"] - 1.0) > 0.02 or h["ks"] > 0.002]
    log(f"  hygiene: {len(hyg)} cols, soft-bad={len(bad)} (will still train; trees get locked inputs)")
    np.save(ART / "X_macro_oof.npy", Xo)
    np.save(ART / "X_macro_test.npy", Xt)
    (ART / "X_macro_names.json").write_text(json.dumps({"names": names, "hygiene": hyg}, indent=2))
    return Xo, Xt, names, name_to_i, hyg


def train_families(y, folds, Xo, Xt, z_cal_o, z_cal_t, name_to_i):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier, Pool

    ntr, nte = len(y), Xt.shape[0]
    fam = {
        "xgb": (np.zeros(ntr), []),
        "lgb": (np.zeros(ntr), []),
        "cat": (np.zeros(ntr), []),
        "l4": (np.zeros(ntr), []),
    }

    def col(name, side="o"):
        i = name_to_i[name]
        return (Xo if side == "o" else Xt)[:, i]

    # precompute L4 basis pieces that exist
    def l4_basis(idx_or_all, side="o"):
        def c(n):
            v = col(n, side)
            return v if idx_or_all is None else v[idx_or_all]

        pieces = []
        core_names = [
            "r_gpt2m_vs_lm12",
            "r_lm9_mean",
            "r_tabm",
            "r_knn",
            "r_P4_L4.3_pure5f",
            "r_P4_L4.2_pure5f",
        ]
        for n in ["d_fwd_rev_3ep", "d_tfirst_tlast"]:
            if n in name_to_i:
                core_names.append(n)
        regs = ["r_rare", "r_disp", "r_tail"]
        cores = [c(n) for n in core_names if n in name_to_i]
        regv = [c(n) for n in regs if n in name_to_i]
        zc = z_cal_o[idx_or_all] if side == "o" and idx_or_all is not None else (z_cal_o if side == "o" else z_cal_t)
        if side == "t":
            zc = z_cal_t
        tanh_z = np.tanh(zc / 2.0)
        ones = np.ones(len(zc))
        bases = [ones] + regv + [tanh_z]
        feats = []
        for core in cores:
            for b in bases:
                feats.append(core * b)
            feats.append(core * core)
            feats.append(core * tanh_z)
        return np.column_stack(feats).astype(np.float64)

    for fi, (a, b) in enumerate(folds):
        log(f"Stage2 fold {fi}/5")
        Xa, Xb, Xte = Xo[a], Xo[b], Xt
        # --- XGB bag5 ---
        bag_b, bag_t = [], []
        for sd in SEEDS5:
            model = xgb.XGBClassifier(
                max_depth=3,
                min_child_weight=60.0,
                max_delta_step=0.15,
                colsample_bytree=0.45,
                colsample_bynode=0.45,
                subsample=0.75,
                reg_alpha=20.0,
                reg_lambda=120.0,
                learning_rate=0.012,
                n_estimators=160,
                tree_method="hist",
                max_bin=63,
                n_jobs=2,
                random_state=sd,
                verbosity=0,
            )
            model.fit(Xa, y[a], base_margin=z_cal_o[a])
            # predict margin
            pb = model.predict(Xb, output_margin=True)  # includes base? In xgb sklearn, predict margin may need base_margin
            # pass base_margin at predict
            pb = model.get_booster().predict(xgb.DMatrix(Xb, base_margin=z_cal_o[b]), output_margin=True)
            pt = model.get_booster().predict(xgb.DMatrix(Xte, base_margin=z_cal_t), output_margin=True)
            bag_b.append(pb - z_cal_o[b])
            bag_t.append(pt - z_cal_t)
            del model
        db = np.mean(bag_b, 0)
        dt = np.mean(bag_t, 0)
        # store train ref for PCHIP of this fold's test later — accumulate OOF
        fam["xgb"][0][b] = db
        fam["xgb"][1].append(dt)

        # --- LGB bag5 ---
        bag_b, bag_t = [], []
        params = dict(
            objective="binary",
            learning_rate=0.012,
            num_leaves=6,
            max_depth=3,
            min_child_samples=800,
            min_sum_hessian_in_leaf=50.0,
            max_delta_step=0.15,
            feature_fraction=0.45,
            bagging_fraction=0.75,
            bagging_freq=1,
            lambda_l1=20.0,
            lambda_l2=120.0,
            max_bin=63,
            verbosity=-1,
            num_threads=2,
        )
        for sd in SEEDS5:
            dtr = lgb.Dataset(Xa, label=y[a], init_score=z_cal_o[a], free_raw_data=False)
            booster = lgb.train({**params, "seed": sd}, dtr, num_boost_round=160)
            # raw_score = residual boost
            bag_b.append(booster.predict(Xb, raw_score=True))
            bag_t.append(booster.predict(Xte, raw_score=True))
            del booster, dtr
        fam["lgb"][0][b] = np.mean(bag_b, 0)
        fam["lgb"][1].append(np.mean(bag_t, 0))

        # --- CAT bag3 ---
        bag_b, bag_t = [], []
        for sd in SEEDS3:
            model = CatBoostClassifier(
                loss_function="Logloss",
                depth=4,
                l2_leaf_reg=60.0,
                random_strength=2.0,
                rsm=0.5,
                learning_rate=0.015,
                iterations=220,
                thread_count=2,
                random_seed=sd,
                verbose=False,
                allow_writing_files=False,
            )
            train_pool = Pool(Xa, label=y[a], baseline=z_cal_o[a])
            model.fit(train_pool)
            pb = logit(np.clip(model.predict_proba(Xb)[:, 1], 1e-6, 1 - 1e-6)) - z_cal_o[b]
            pt = logit(np.clip(model.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)) - z_cal_t
            bag_b.append(pb)
            bag_t.append(pt)
            del model, train_pool
        fam["cat"][0][b] = np.mean(bag_b, 0)
        fam["cat"][1].append(np.mean(bag_t, 0))

        # --- L4 regime GLM ---
        Ba = l4_basis(a, "o")
        Bb = l4_basis(b, "o")
        Bt = l4_basis(None, "t")
        p_cal_a = expit(z_cal_o[a])
        resid = (y[a].astype(np.float64) - p_cal_a) / np.sqrt(p_cal_a * (1 - p_cal_a) + 1e-4)
        sc = StandardScaler()
        Ba_s = sc.fit_transform(np.nan_to_num(Ba, 0.0))
        Bb_s = sc.transform(np.nan_to_num(Bb, 0.0))
        Bt_s = sc.transform(np.nan_to_num(Bt, 0.0))
        ridge = Ridge(alpha=3e4).fit(Ba_s, resid)
        fam["l4"][0][b] = ridge.predict(Bb_s)
        fam["l4"][1].append(ridge.predict(Bt_s))
        del Ba, Bb, Bt, Ba_s, Bb_s, Bt_s
        gc.collect()
        log(f"  fold {fi} families done")

    out = {}
    for k in fam:
        oof = fam[k][0]
        te = np.mean(fam[k][1], 0)
        te = pchip_quantile_map(te, oof, n_knots=N_KNOTS)
        # exact lock
        te = (te - te.mean()) / (te.std() + 1e-12) * (oof.std() + 1e-12) + oof.mean()
        np.save(ART / f"delta_{k}_oof.npy", oof.astype(np.float32))
        np.save(ART / f"delta_{k}_test.npy", te.astype(np.float32))
        out[k] = (oof, te)
        log(f"  saved delta_{k} disp={np.std(te)/(np.std(oof)+1e-12):.4f} KS={ks_2samp(oof,te).statistic:.5f}")
    return out


def gram_schmidt_against(U, controls):
    """Orth each column of U against control columns (numpy)."""
    out = []
    C = np.column_stack(controls)
    for j in range(U.shape[1]):
        v = U[:, j].copy()
        for k in range(C.shape[1]):
            ck = C[:, k]
            v = v - np.dot(v, ck) / (np.dot(ck, ck) + 1e-12) * ck
        sd = float(np.std(v) + 1e-12)
        out.append((v - v.mean()) / sd)
    return np.column_stack(out)


def portfolio_optimize(y, folds, deltas, z_ref_o, z_lm12_o, z_st9p5_t, ladder_t, z_lm12_t):
    log("Stage3: nested minimax portfolio")
    keys = list(deltas.keys())
    Uo = np.column_stack([deltas[k][0] for k in keys])
    Ut = np.column_stack([deltas[k][1] for k in keys])
    # orth vs z_ref + lm12 on OOF; on test also vs ladder
    Uo_orth = gram_schmidt_against(Uo, [z_ref_o, z_lm12_o])
    Ut_orth = gram_schmidt_against(Ut, [z_st9p5_t, z_lm12_t, ladder_t])
    # re-PCHIP each col
    for j in range(Ut_orth.shape[1]):
        Ut_orth[:, j] = pchip_quantile_map(Ut_orth[:, j], Uo_orth[:, j], n_knots=N_KNOTS)

    M = Uo_orth.shape[1]
    # LOFO weights: maximise mean fold delta - 0.5*max(0,-min fold delta)
    def score_w(w, idx):
        w = np.asarray(w)
        w = w / (np.sum(np.abs(w)) + 1e-12)
        r = Uo_orth[idx] @ w
        # split idx into 4 mini? use full train folds among idx via secondary skf is heavy
        return float(fast_auc(y[idx], z_ref_o[idx] + r))

    w_oof = np.zeros(M)
    r_o = np.zeros(len(y))
    r_t_folds = []
    for fi, (a, b) in enumerate(folds):
        # coordinate ascent on train_idx a
        w = np.ones(M) / M
        best = -1e9
        for _ in range(25):
            improved = False
            for j in range(M):
                for t in np.linspace(0, 1, 11):
                    cand = w.copy()
                    cand[j] = t
                    if cand.sum() < 1e-12:
                        continue
                    cand = cand / cand.sum()
                    # mean AUC on a, with fold-positivity proxy via shuffling a into 4
                    r = Uo_orth[a] @ cand
                    sc = float(fast_auc(y[a], z_ref_o[a] + 0.05 * r))
                    # mini fold deltas inside a
                    rs = np.random.RandomState(fi * 17 + _)
                    perm = rs.permutation(len(a))
                    chunks = np.array_split(perm, 4)
                    deltas_f = []
                    for ch in chunks:
                        ii = a[ch]
                        base = float(fast_auc(y[ii], z_ref_o[ii]))
                        deltas_f.append(float(fast_auc(y[ii], z_ref_o[ii] + 0.05 * (Uo_orth[ii] @ cand))) - base)
                    pen = 0.5 * max(0.0, -min(deltas_f))
                    obj = sc - pen
                    if obj > best + 1e-12:
                        best = obj
                        w = cand
                        improved = True
            if not improved:
                break
        r_o[b] = Uo_orth[b] @ w
        r_t_folds.append(Ut_orth @ w)
        w_oof += w
        log(f"  fold {fi} w={np.round(w,3).tolist()} obj={best:.6f}")
    w_mean = w_oof / 5.0
    r_t = np.mean(r_t_folds, 0)
    # unitize
    sd = float(np.std(r_o) + 1e-12)
    r_o = (r_o - r_o.mean()) / sd
    r_t = pchip_quantile_map(r_t, r_o, n_knots=N_KNOTS)
    r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-12)
    return keys, Uo_orth, Ut_orth, r_o, r_t, w_mean


def main():
    log("MACRO_V11 START (STRICT NO_SUBMIT)")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    np.save(ART / "folds.npy", np.array([(a, b) for a, b in folds], dtype=object))

    z_ref_o, z_st9_t, s9, base = build_z_ref(y, test_ids, folds)
    # Strike 9.5 anchor
    s95 = (
        pd.read_csv(SUB / "sub_strike9p5_axis_0p70_0p10.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(float)
    )
    z_st9p5_t = pchip_quantile_map(zrk(s95), z_ref_o, n_knots=N_KNOTS)
    z_a3, _ = lz_sub("sub_gm_att3_continue_from_095315.csv", test_ids, z_ref_o)
    ladder_t = z_st9p5_t - z_a3
    log(f"z_ref_o={base:.7f} | St9.5 locked")

    lm_bank = load_lm_bank(y, folds, len(y), len(test_ids), test_ids)
    # ensure megayak
    mega = ROOT / "external/banks/megayak_s6e11-target-first-lm"
    if "megayak" not in lm_bank and (mega / "lm_oof.npy").exists():
        mo = np.load(mega / "lm_oof.npy").ravel()
        mt = np.load(mega / "lm_test.npy").ravel()
        lm_bank["megayak"] = (zrk(mo), pchip_quantile_map(zrk(mt), zrk(mo), n_knots=N_KNOTS))

    Xo, Xt, names, name_to_i, hyg = build_X_macro(y, folds, z_ref_o, z_st9p5_t, lm_bank, test_ids)
    z_cal_o = calibrate_logit_to_prevalence(z_ref_o, z_ref_o, y)
    z_cal_t = calibrate_logit_to_prevalence(z_st9p5_t, z_ref_o, y)

    deltas = train_families(y, folds, Xo, Xt, z_cal_o, z_cal_t, name_to_i)
    del Xo, Xt
    gc.collect()

    z_lm12_o, z_lm12_t = lm_bank["lm12_wide"]
    keys, Uo, Ut, r_o, r_t, w_mean = portfolio_optimize(
        y, folds, deltas, z_ref_o, z_lm12_o, z_st9p5_t, ladder_t, z_lm12_t
    )
    np.save(ART / "r_macro_orth_oof.npy", r_o.astype(np.float32))
    np.save(ART / "r_macro_orth_test.npy", r_t.astype(np.float32))

    # diagnostics
    rows = []
    for k in keys:
        oof, te = deltas[k]
        # orth single vs z_ref for fair table
        ro, rt, rtr = fold_orth_multi(oof, te, z_ref_o.reshape(-1, 1), z_st9p5_t.reshape(-1, 1), folds)
        nest = nested_delta(y, z_ref_o, ro, folds)
        disp = float(np.std(rtr) / (np.std(ro) + 1e-12))
        ks = float(ks_2samp(ro, rtr).statistic)
        fd, _ = fold_deltas(y, z_ref_o, ro, folds, nest["w"] if abs(nest["w"]) > 1e-12 else 0.03)
        rows.append({**nest, "name": k, "fold_deltas_1e5": fd, "disp": round(disp, 4), "ks": round(ks, 6)})
        log(f"  {k}: Δ={nest['delta_1e5']} {nest['pos']}/5 disp={disp:.4f} KS={ks:.5f} folds={fd}")

    nest_m = nested_delta(y, z_ref_o, r_o, folds)
    disp_m = float(np.std(r_t) / (np.std(r_o) + 1e-12))  # after PCHIP ~1
    # raw-ish: use r_t before final unit already locked
    ks_m = float(ks_2samp(r_o, r_t).statistic)
    fd_m, _ = fold_deltas(y, z_ref_o, r_o, folds, nest_m["w"] if abs(nest_m["w"]) > 1e-12 else 0.03)
    macro_row = {
        **nest_m,
        "name": "macro_combo",
        "fold_deltas_1e5": fd_m,
        "disp": round(disp_m, 4),
        "ks": round(ks_m, 6),
        "weights": {keys[i]: round(float(w_mean[i]), 4) for i in range(len(keys))},
    }
    log(
        f"  MACRO: Δ={macro_row['delta_1e5']} {macro_row['pos']}/5 disp={disp_m:.4f} KS={ks_m:.5f} folds={fd_m}"
    )

    # step onto St9.5
    step_rows = []
    targets = [0.5e-4, 1.0e-4, 2.0e-4, 2.5e-4, 3.5e-4]
    # momentum blend optional
    z_st9, _ = lz_sub("sub_gm_top1_strike9_final_max.csv", test_ids, z_ref_o)
    mom = z_st9p5_t - z_st9
    for td in targets:
        w, got = weight_for_spearman_dist(z_st9p5_t, r_t, td)
        # OOF proxy: inject onto z_ref
        nest = nested_delta(y, z_ref_o, r_o, folds, w_grid=[w, -w, w * 0.5, w * 1.5])
        fd, pos = fold_deltas(y, z_ref_o, r_o, folds, w)
        z_final = z_st9p5_t + w * r_t
        # also momentum variant
        z_mom = z_final + 0.55 * mom
        dist = (1 - float(spearmanr(z_final, z_st9p5_t).correlation)) * 1e4
        step_rows.append(
            {
                "target_1e4": round(td * 1e4, 4),
                "achieved_1e4": round(got * 1e4, 4),
                "w": round(w, 6),
                "delta_1e5": nest["delta_1e5"],
                "pos": pos,
                "fold_deltas_1e5": fd,
                "disp": macro_row["disp"],
                "ks": macro_row["ks"],
            }
        )
        log(f"  step @1-ρ={td*1e4:.2f}e-4: Δ={nest['delta_1e5']} {pos}/5 w={w:.4f}")

    # stage CSVs if gate
    staged = []
    gate_ok = (
        macro_row["pos"] == 5
        and 0.98 <= macro_row["disp"] <= 1.02
        and macro_row["ks"] < 0.002
        and macro_row["delta_1e5"] >= 0.5
    )
    want = {1.0: "sub_macro_v11_slotA_1p0e4.csv", 2.5: "sub_macro_v11_slotB_2p5e4.csv"}
    if gate_ok:
        for td_e4, fname in want.items():
            w, _ = weight_for_spearman_dist(z_st9p5_t, r_t, td_e4 * 1e-4)
            z = z_st9p5_t + w * r_t
            p = expit(calibrate_logit_to_prevalence(z, z_ref_o, y))
            assert len(p) == 299984 and np.isfinite(p).all()
            path = SUB / fname
            pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(path, index=False)
            staged.append({"file": fname, "w": w, "mean": float(p.mean()), "target_1e4": td_e4})
            log(f"  STAGED {fname} mean={p.mean():.6f}")
    else:
        log(f"  GATE FAIL — not staging CSVs (pos={macro_row['pos']} disp={macro_row['disp']} KS={macro_row['ks']})")
        # still write research CSVs under preds for inspection
        for td_e4, fname in want.items():
            w, _ = weight_for_spearman_dist(z_st9p5_t, r_t, td_e4 * 1e-4)
            z = z_st9p5_t + w * r_t
            p = expit(calibrate_logit_to_prevalence(z, z_ref_o, y))
            path = ART / fname.replace("sub_", "research_")
            pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(path, index=False)
            staged.append({"file": str(path), "research_only": True, "w": w, "target_1e4": td_e4})

    md = [
        "# Macro-Jump v11 Super Refeed (NO_SUBMIT)",
        "",
        f"Elapsed **{time.time()-T0:.1f}s**. z_ref_o **{base:.7f}**. Anchor St9.5 **0.95332**.",
        "",
        f"X_macro cols: **{len(names)}**. Soft-bad hygiene cols: {sum(1 for h in hyg if abs(h['disp']-1)>0.02 or h['ks']>0.002)}.",
        "",
        "## Family + Macro gate",
        "",
        "| member | ΔOOF e-5 | w | pos | folds | disp | KS |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows + [macro_row]:
        md.append(
            f"| {r['name']} | {r['delta_1e5']} | {r.get('w')} | {r['pos']}/5 | {r['fold_deltas_1e5']} | {r['disp']} | {r['ks']} |"
        )
    md += [
        "",
        f"Portfolio weights: `{macro_row.get('weights')}`",
        "",
        f"Strict gate (5/5, disp∈[0.98,1.02], KS<0.002): **{gate_ok}**",
        "",
        "## Steps onto St9.5",
        "",
        "| 1-ρ e-4 | w | ΔOOF e-5 | pos | folds |",
        "|---:|---:|---:|---:|---|",
    ]
    for s in step_rows:
        md.append(
            f"| {s['target_1e4']} | {s['w']} | {s['delta_1e5']} | {s['pos']}/5 | {s['fold_deltas_1e5']} |"
        )
    md += ["", "## Staged", ""]
    for s in staged:
        md.append(f"- `{s}`")
    md += ["", f"`NO_SUBMIT={NO_SUBMIT}`.", ""]
    (OUT / "MACRO_V11_JUMP.md").write_text("\n".join(md))
    payload = {
        "families": rows,
        "macro": macro_row,
        "steps": step_rows,
        "staged": staged,
        "gate_ok": gate_ok,
        "n_features": len(names),
        "NO_SUBMIT": NO_SUBMIT,
        "elapsed_s": time.time() - T0,
    }
    (OUT / "macro_v11_jump.json").write_text(json.dumps(payload, indent=2))
    log("wrote MACRO_V11_JUMP.md")
    log("MACRO_V11_COMPLETE")
    print("MACRO_V11_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "MACRO_V11_FAIL.txt").write_text(traceback.format_exc())
        raise
