#!/usr/bin/env python3
"""Step 3: De-bias v10_lgb (drop z_anchor), pure-FE retrain, Strike-10 stage (NO_SUBMIT).

Artifacts:
  preds/v10_transductive_bulsah/X_all_f32.dat (+ pure masks)
  preds/v10_transductive_bulsah/v10_*_pure_fe_*.npy
  submissions/sub_strike10A_*.csv / sub_strike10B_*.csv  (staged only)
  logs/overnight/STEP3_STRIKE10.md / step3_strike10.json
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
os.environ["NO_SUBMIT"] = "1"

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from gm_top3_strike3_expanded7f import (
    PRED,
    SUB,
    TARGET_MEAN,
    calibrate_logit_to_prevalence,
    fast_auc,
    pchip_quantile_map,
    zrk,
)
from recon_and_forensic_audit import N_KNOTS, build_z_ref, nested_delta, resid_diagnostics
from step2_lm_orth_and_v10_transductive import (
    ART,
    build_v10_features,
    ecdf_align,
    fold_orth_multi,
    free_gb,
    load_lm_bank,
    log as _s2log,
    m_estimate_te,
    nested_at_w,
    rss_gb,
    weight_for_spearman_dist,
)

OUT = ROOT / "logs" / "overnight"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
STATUS = OUT / "STEP3_STATUS.json"
NO_SUBMIT = True

STEP_GRID_AB = [0.20e-4, 0.40e-4, 0.60e-4, 1.00e-4]
STEP_GRID_B_CLAMP = [0.20e-4, 0.40e-4, 0.60e-4]
STEP_GRID_C = [0.30e-4, 0.50e-4, 0.80e-4]
STEP_D = 0.30e-4


def log(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    try:
        STATUS.write_text(
            json.dumps(
                {
                    "t_s": round(time.time() - T0, 1),
                    "msg": msg,
                    "rss_gb": rss_gb(),
                    "free_gb": free_gb(),
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "NO_SUBMIT": NO_SUBMIT,
                },
                indent=2,
            )
        )
    except Exception:
        pass


def fold_deltas(y, z_base, r, folds, w):
    d = []
    for _, b in folds:
        base = float(fast_auc(y[b], z_base[b]))
        sc = float(fast_auc(y[b], (z_base + w * r)[b]))
        d.append(round((sc - base) * 1e5, 2))
    pos = sum(1 for x in d if x > 0)
    return d, pos


def unitize(ro, rt_raw):
    sd = float(np.std(ro) + 1e-12)
    ro = (ro - float(np.mean(ro))) / sd
    rt = pchip_quantile_map(rt_raw, ro, n_knots=N_KNOTS)
    rt = (rt - float(np.mean(ro))) / (float(np.std(ro)) + 1e-12)
    return ro, rt, rt_raw


def combine_lofo(members_o, members_t, members_tr, y, z_ref_o, folds, w_grid=None):
    """Nonneg weights (sum~1) maximizing OOF AUC of z_ref + U@w; return unit residual."""
    if w_grid is None:
        w_grid = np.linspace(0.0, 1.0, 11)
    Uo = np.column_stack(members_o)
    Ut = np.column_stack(members_t)
    Utr = np.column_stack(members_tr)
    M = Uo.shape[1]
    w = np.ones(M) / M
    best = (-1e9, w.copy())
    base = float(fast_auc(y, z_ref_o))
    for _ in range(40):
        improved = False
        for j in range(M):
            for t in w_grid:
                cand = w.copy()
                cand[j] = t
                if cand.sum() < 1e-12:
                    continue
                cand = cand / cand.sum()
                z = z_ref_o + Uo @ cand
                sc = float(fast_auc(y, z))
                d = (sc - base) * 1e5
                if d > best[0] + 1e-9:
                    best = (d, cand.copy())
                    w = cand
                    improved = True
        if not improved:
            break
    w = best[1]
    ro = Uo @ w
    rt_raw = Utr @ w
    sd = float(np.std(ro) + 1e-12)
    ro = (ro - float(np.mean(ro))) / sd
    rt_raw = (rt_raw - float(np.mean(ro)) * 0) / sd  # scale by same sd; mean handled in PCHIP
    rt = pchip_quantile_map(rt_raw, ro, n_knots=N_KNOTS)
    return ro, rt, rt_raw, w, best[0]


def ensure_feature_cache(y, folds, z_ref_o, z_st9_t):
    """Rebuild 481-feat matrix if missing; return X_tr, X_te, names, te_key_mats."""
    meta_path = ART / "feature_names.json"
    mmap_tr = ART / "X_tr_f32.dat"
    mmap_te = ART / "X_te_f32.dat"
    names = json.loads(meta_path.read_text())["feature_names"]
    ntr, nte, p = len(y), 299984, len(names)
    if mmap_tr.exists() and mmap_te.exists() and mmap_tr.stat().st_size == ntr * p * 4:
        log(f"loading cached feature memmap p={p}")
        X_tr = np.memmap(mmap_tr, dtype=np.float32, mode="r", shape=(ntr, p))
        X_te = np.memmap(mmap_te, dtype=np.float32, mode="r", shape=(nte, p))
        # rebuild te_key_mats only (cheap)
        tr = pd.read_csv(ROOT / "data/train.csv")
        te = pd.read_csv(ROOT / "data/test.csv")
        from step2_lm_orth_and_v10_transductive import bitmask64

        te_key_mats = {
            "mc_bm": (
                (tr["merchant_category"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
                (te["merchant_category"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
            ),
            "ct_bm": (
                (tr["card_type"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
                (te["card_type"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
            ),
            "am_bm": (
                (tr["auth_method"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
                (te["auth_method"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
            ),
            "dt_bm": (
                (tr["device_type"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
                (te["device_type"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
            ),
        }
        del tr, te
        gc.collect()
        return X_tr, X_te, names, te_key_mats

    log("feature cache miss — rebuilding 481-feat matrix (will cache)")
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    orig = pd.read_csv(ROOT / "original/credit_card_fraud_2026.csv")
    X_tr, X_te, te_key_mats, names = build_v10_features(tr, te, orig, z_ref_o, z_st9_t, y)
    del tr, te, orig
    gc.collect()
    # persist
    fp_tr = np.memmap(mmap_tr, dtype=np.float32, mode="w+", shape=X_tr.shape)
    fp_tr[:] = X_tr
    fp_tr.flush()
    fp_te = np.memmap(mmap_te, dtype=np.float32, mode="w+", shape=X_te.shape)
    fp_te[:] = X_te
    fp_te.flush()
    del fp_tr, fp_te
    log(f"cached X_tr {X_tr.shape} X_te {X_te.shape}")
    return X_tr, X_te, names, te_key_mats


def diagnose_zanch(X_tr, X_te, names, y, folds, z_cal_o):
    """Top-20 LGB split gains on fold0 + train/test zanch moments."""
    import lightgbm as lgb

    zanch_idx = [i for i, n in enumerate(names) if n.startswith("zanch_")]
    log(f"PART1 diagnose: {len(zanch_idx)}/{len(names)} z_anchor features")
    # train vs test moment shift for zanch
    shifts = []
    for i in zanch_idx:
        mu_o, mu_t = float(np.mean(X_tr[:, i])), float(np.mean(X_te[:, i]))
        sd_o, sd_t = float(np.std(X_tr[:, i]) + 1e-12), float(np.std(X_te[:, i]) + 1e-12)
        shifts.append(
            {
                "feat": names[i],
                "mean_tr": round(mu_o, 6),
                "mean_te": round(mu_t, 6),
                "std_tr": round(sd_o, 6),
                "std_te": round(sd_t, 6),
                "std_ratio": round(sd_t / sd_o, 4),
                "mean_diff": round(mu_t - mu_o, 6),
            }
        )
    shifts = sorted(shifts, key=lambda r: -abs(r["std_ratio"] - 1.0))
    # fold0 LGB with ALL feats for gain importance
    a, b = folds[0]
    dtr = lgb.Dataset(np.asarray(X_tr[a]), label=y[a], init_score=z_cal_o[a], free_raw_data=True)
    params = dict(
        objective="binary",
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        feature_fraction=0.25,
        bagging_fraction=0.85,
        bagging_freq=1,
        min_child_samples=150,
        lambda_l2=30.0,
        verbosity=-1,
        seed=42,
        num_threads=2,
    )
    booster = lgb.train(params, dtr, num_boost_round=350)
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    order = np.argsort(-gain)[:20]
    top20 = []
    for rank, j in enumerate(order, 1):
        top20.append(
            {
                "rank": rank,
                "feat": names[int(j)],
                "gain": float(gain[j]),
                "split": int(split[j]),
                "is_zanch": names[int(j)].startswith("zanch_"),
            }
        )
    n_zanch_top = sum(1 for r in top20 if r["is_zanch"])
    log(f"  top20 gain: {n_zanch_top}/20 are zanch_*; #1={top20[0]['feat']}")
    del booster, dtr
    gc.collect()
    return {
        "n_zanch": len(zanch_idx),
        "zanch_top20_count": n_zanch_top,
        "top20_gain": top20,
        "zanch_worst_std_shift": shifts[:15],
    }


def train_pure_fe(X_tr, X_te, names, te_key_mats, y, folds, z_ref_o, z_st9_t):
    """Retrain ridge + lgb on non-zanch features only."""
    import lightgbm as lgb

    keep = np.array([not n.startswith("zanch_") for n in names], dtype=bool)
    pure_names = [n for n, k in zip(names, keep) if k]
    log(f"PART1 pure-FE: keeping {keep.sum()}/{len(names)} features (dropped zanch)")
    (ART / "pure_fe_feature_names.json").write_text(json.dumps(pure_names, indent=2))

    z_cal_o = calibrate_logit_to_prevalence(z_ref_o, z_ref_o, y)
    z_cal_t = calibrate_logit_to_prevalence(z_st9_t, z_ref_o, y)
    ntr, nte = len(y), X_te.shape[0]
    ridge_o = np.zeros(ntr)
    lgb_o = np.zeros(ntr)
    ridge_te, lgb_te = [], []
    lgb_seeds = [42, 137, 2026, 7, 99]
    te_ms = [10.0, 30.0]
    te_seeds = list(range(8))

    Xp_tr = np.asarray(X_tr[:, keep], dtype=np.float32)
    Xp_te = np.asarray(X_te[:, keep], dtype=np.float32)
    # free full matrix refs if memmap — keep as is

    for fi, (a, b) in enumerate(folds):
        log(f"  pure fold {fi}/5 rss={rss_gb():.2f} free={free_gb():.2f}")
        prior = float(y[a].mean())
        te_a, te_b, te_t = [], [], []
        for kn, (ktr, kte) in te_key_mats.items():
            for m in te_ms:
                bag_a, bag_b, bag_t = [], [], []
                for sd in te_seeds:
                    rs = np.random.RandomState(1000 * fi + 17 * sd + int(m))
                    idx = rs.choice(len(a), size=int(0.9 * len(a)), replace=False)
                    aa = a[idx]
                    _, va, _ = m_estimate_te(ktr[aa], y[aa], ktr[a], kte, m, prior)
                    _, vb, vt = m_estimate_te(ktr[aa], y[aa], ktr[b], kte, m, prior)
                    bag_a.append(va)
                    bag_b.append(vb)
                    bag_t.append(vt)
                te_a.append(np.mean(bag_a, 0))
                te_b.append(np.mean(bag_b, 0))
                te_t.append(np.mean(bag_t, 0))
        Xa = np.column_stack([Xp_tr[a]] + te_a).astype(np.float32)
        Xb = np.column_stack([Xp_tr[b]] + te_b).astype(np.float32)
        Xt = np.column_stack([Xp_te] + te_t).astype(np.float32)
        del te_a, te_b, te_t
        gc.collect()

        resid_a = y[a].astype(np.float64) - expit(z_cal_o[a])
        sc = StandardScaler()
        Xa_s = sc.fit_transform(np.nan_to_num(Xa, 0.0)).astype(np.float32)
        Xb_s = sc.transform(np.nan_to_num(Xb, 0.0)).astype(np.float32)
        Xt_s = sc.transform(np.nan_to_num(Xt, 0.0)).astype(np.float32)
        ridge = Ridge(alpha=500.0).fit(Xa_s, resid_a)
        pa, pb, pt = ridge.predict(Xa_s), ridge.predict(Xb_s), ridge.predict(Xt_s)
        pb_al, pt_al = ecdf_align(pa, pb, pt)
        ridge_o[b] = pb_al
        ridge_te.append(pt_al)

        params = dict(
            objective="binary",
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            feature_fraction=0.25,
            bagging_fraction=0.85,
            bagging_freq=1,
            min_child_samples=150,
            lambda_l2=30.0,
            verbosity=-1,
            num_threads=2,
        )
        la, lb, lt = [], [], []
        for sd in lgb_seeds:
            dtr = lgb.Dataset(Xa, label=y[a], init_score=z_cal_o[a], free_raw_data=False)
            booster = lgb.train({**params, "seed": sd}, dtr, num_boost_round=350)
            la.append(booster.predict(Xa, raw_score=True))
            lb.append(booster.predict(Xb, raw_score=True))
            lt.append(booster.predict(Xt, raw_score=True))
            del booster, dtr
        pa, pb, pt = np.mean(la, 0), np.mean(lb, 0), np.mean(lt, 0)
        pb_al, pt_al = ecdf_align(pa, pb, pt)
        lgb_o[b] = pb_al
        lgb_te.append(pt_al)
        del Xa, Xb, Xt, Xa_s, Xb_s, Xt_s, la, lb, lt
        gc.collect()
        log(f"  pure fold {fi} done")

    ridge_t = np.mean(ridge_te, 0)
    lgb_t = np.mean(lgb_te, 0)
    np.save(ART / "v10_ridge_pure_fe_oof.npy", ridge_o.astype(np.float32))
    np.save(ART / "v10_ridge_pure_fe_test.npy", ridge_t.astype(np.float32))
    np.save(ART / "v10_lgb_pure_fe_oof.npy", lgb_o.astype(np.float32))
    np.save(ART / "v10_lgb_pure_fe_test.npy", lgb_t.astype(np.float32))
    del Xp_tr, Xp_te
    gc.collect()
    return {"v10_ridge_pure_fe": (ridge_o, ridge_t), "v10_lgb_pure_fe": (lgb_o, lgb_t)}


def audit_orth(name, oof, test, y, z_ref_o, z_st9_t, s9, folds, lm_bank):
    names = list(lm_bank.keys())
    Xo = np.column_stack([z_ref_o] + [lm_bank[n][0] for n in names])
    Xt = np.column_stack([z_st9_t] + [lm_bank[n][1] for n in names])
    ro, rt, rtr = fold_orth_multi(oof.astype(np.float64), test.astype(np.float64), Xo, Xt, folds)
    nest = nested_delta(y, z_ref_o, ro, folds)
    diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
    w = nest["w"]
    rho = (1.0 - float(spearmanr(z_st9_t + w * rt, z_st9_t).correlation)) * 1e4
    row = {
        "name": name,
        **nest,
        "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
        "ks_raw": diag["ks_raw"],
        "spearman_dist_st9_test_1e4": round(rho, 3),
        "gate_pass": (
            nest["delta_1e5"] >= 0.5
            and nest["pos"] >= 4
            and 0.85 <= diag["dispersion_ratio_raw"] <= 1.15
            and diag["ks_raw"] < 0.015
        ),
    }
    np.save(ART / f"{name}_orth_LM_oof.npy", ro.astype(np.float32))
    np.save(ART / f"{name}_orth_LM_test.npy", rt.astype(np.float32))
    np.save(ART / f"{name}_orth_LM_test_raw.npy", rtr.astype(np.float32))
    return row, ro, rt, rtr


def eval_portfolio(label, ro, rt, rtr, y, z_ref_o, z_st9_t, s9, s_p2, folds, targets):
    rows = []
    diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
    for td in targets:
        w, got = weight_for_spearman_dist(z_st9_t, rt, td)
        nest = nested_at_w(y, z_ref_o, ro, folds, w)
        fd, pos = fold_deltas(y, z_ref_o, ro, folds, w)
        z_final = z_st9_t + w * rt
        rho_st9 = 1.0 - float(spearmanr(z_final, z_st9_t).correlation)
        ks_st9 = float(ks_2samp(zrk(expit(z_final)), zrk(s9)).statistic)
        rho_p2 = 1.0 - float(spearmanr(z_final, zrk(s_p2)).correlation)
        ks_p2 = float(ks_2samp(zrk(expit(z_final)), zrk(s_p2)).statistic)
        row = {
            "portfolio": label,
            "target_1e4": round(td * 1e4, 4),
            "achieved_1e4": round(got * 1e4, 4),
            "w": round(w, 6),
            "delta_1e5": nest["delta_1e5"],
            "fold_deltas_1e5": fd,
            "pos": pos,
            "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
            "ks_raw": diag["ks_raw"],
            "test_1m_rho_st9": round(rho_st9, 8),
            "test_ks_vs_st9_ranks": round(ks_st9, 6),
            "test_1m_rho_p2": round(rho_p2, 8),
            "test_ks_vs_p2_ranks": round(ks_p2, 6),
        }
        # strict stage gate for strike10 candidates
        row["stage_gate"] = (
            pos == 5
            and 0.92 <= diag["dispersion_ratio_raw"] <= 1.06
            and diag["ks_raw"] < 0.007
            and 0.30e-4 <= got <= 0.60e-4 + 1e-9
        )
        rows.append(row)
        log(
            f"  {label} @1-ρ={td*1e4:.2f}e-4: Δ={nest['delta_1e5']} {pos}/5 "
            f"folds={fd} disp={diag['dispersion_ratio_raw']} KS={diag['ks_raw']} "
            f"stage={row['stage_gate']}"
        )
    return rows


def write_submission(path, test_ids, z_test, z_ref_o, y):
    z_cal = calibrate_logit_to_prevalence(z_test, z_ref_o, y, target_mean=TARGET_MEAN)
    p = expit(z_cal)
    assert len(p) == 299984
    assert np.isfinite(p).all()
    assert not np.isnan(p).any()
    df = pd.DataFrame({"id": test_ids, "is_fraud": p})
    assert len(df) == 299984
    df.to_csv(path, index=False)
    prev = float(p.mean())
    log(f"  staged {path.name} n={len(df)} mean={prev:.6f} (target {TARGET_MEAN})")
    return {"path": str(path), "n": len(df), "mean": prev, "min": float(p.min()), "max": float(p.max())}


def main():
    log("STEP3 START (NO_SUBMIT until verification table; CSVs staged only)")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    log("anchors + LM bank...")
    z_ref_o, z_st9_t, s9, base_auc = build_z_ref(y, test_ids, folds)
    log(f"z_ref_o AUC={base_auc:.7f}")
    lm_bank = load_lm_bank(y, folds, len(y), len(test_ids), test_ids)
    s_p2 = (
        pd.read_csv(SUB / "sub_pass340_P2_fam21_x085.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(float)
    )

    X_tr, X_te, names, te_key_mats = ensure_feature_cache(y, folds, z_ref_o, z_st9_t)
    z_cal_o = calibrate_logit_to_prevalence(z_ref_o, z_ref_o, y)

    diag = diagnose_zanch(X_tr, X_te, names, y, folds, z_cal_o)

    pure = train_pure_fe(X_tr, X_te, names, te_key_mats, y, folds, z_ref_o, z_st9_t)
    # can drop heavy mats
    del X_tr, X_te
    gc.collect()

    pure_audit = {}
    pure_res = {}
    for name, (oof, te) in pure.items():
        row, ro, rt, rtr = audit_orth(name, oof, te, y, z_ref_o, z_st9_t, s9, folds, lm_bank)
        pure_audit[name] = row
        pure_res[name] = (ro, rt, rtr)
        log(
            f"  {name}_orth_LM: Δ={row['delta_1e5']}e-5 {row['pos']}/5 "
            f"disp={row['dispersion_ratio_raw']} KS={row['ks_raw']} 1-ρ={row['spearman_dist_st9_test_1e4']}e-4"
        )

    # Load / recompute LM-orth residuals (prefer saved; else re-orth from raw preds)
    def load_or_reorth(orth_stem, raw_oof_stem=None):
        ro = np.load(ART / f"{orth_stem}_oof.npy").astype(np.float64)
        rt = np.load(ART / f"{orth_stem}_test.npy").astype(np.float64)
        raw_p = ART / f"{orth_stem}_test_raw.npy"
        if raw_p.exists():
            return ro, rt, np.load(raw_p).astype(np.float64)
        if raw_oof_stem and (ART / f"{raw_oof_stem}_oof.npy").exists():
            oof = np.load(ART / f"{raw_oof_stem}_oof.npy").astype(np.float64)
            te = np.load(ART / f"{raw_oof_stem}_test.npy").astype(np.float64)
            names = list(lm_bank.keys())
            Xo = np.column_stack([z_ref_o] + [lm_bank[n][0] for n in names])
            Xt = np.column_stack([z_st9_t] + [lm_bank[n][1] for n in names])
            ro2, rt2, rtr = fold_orth_multi(oof, te, Xo, Xt, folds)
            np.save(ART / f"{orth_stem}_test_raw.npy", rtr.astype(np.float32))
            return ro2, rt2, rtr
        # last resort: treat PCHIP test as raw (disp≈1)
        log(f"  WARN no raw residual for {orth_stem} — using PCHIP test as raw")
        return ro, rt, rt.copy()

    r_ridge, t_ridge, tr_ridge = load_or_reorth("v10_ridge_resid_orth_LM", "v10_ridge_resid")
    r_cb, t_cb, tr_cb = load_or_reorth("v10_cb_d4_bag3_orth_LM", "v10_cb_d4_bag3")
    r_lgb_old, t_lgb_old, tr_lgb_old = load_or_reorth("v10_lgb_d3_bag5_orth_LM", "v10_lgb_d3_bag5")
    r_soft, t_soft, tr_soft = load_or_reorth("soft_diff_orth_LM")
    # combo3 ≈ mean of three member orth residuals (match step2); keep saved oof/test, rebuild raw
    r_combo = np.load(ART / "v10_combo3_orth_LM_oof.npy").astype(np.float64)
    t_combo = np.load(ART / "v10_combo3_orth_LM_test.npy").astype(np.float64)
    tr_combo = (tr_ridge + tr_cb + tr_lgb_old) / 3.0
    # re-align oof unit scale of combo raw mean for disp diagnostics
    _sd = float(np.std((r_ridge + r_cb + r_lgb_old) / 3.0) + 1e-12)
    tr_combo = tr_combo  # raw pre-PCHIP average

    # Portfolio A
    members_o = [r_ridge, r_cb]
    members_t = [t_ridge, t_cb]
    members_tr = [tr_ridge, tr_cb]
    lgb_ok = False
    if "v10_lgb_pure_fe" in pure_res:
        row = pure_audit["v10_lgb_pure_fe"]
        if row["ks_raw"] < 0.009 and 0.90 <= row["dispersion_ratio_raw"] <= 1.10:
            lgb_ok = True
            members_o.append(pure_res["v10_lgb_pure_fe"][0])
            members_t.append(pure_res["v10_lgb_pure_fe"][1])
            members_tr.append(pure_res["v10_lgb_pure_fe"][2])
            log("  Portfolio A: including v10_lgb_pure_fe_orth_LM (KS/disp gate OK)")
        else:
            log(
                f"  Portfolio A: EXCLUDE lgb_pure (KS={row['ks_raw']} disp={row['dispersion_ratio_raw']})"
            )
    ro_a, rt_a, rtr_a, w_a, d_a = combine_lofo(
        members_o, members_t, members_tr, y, z_ref_o, folds
    )
    log(f"  Portfolio A weights={np.round(w_a,4).tolist()} freeΔ≈{d_a:.2f}e-5")

    # Portfolio C blend: equal-risk 0.5/0.5 then re-unitize
    ro_c = 0.5 * ro_a + 0.5 * r_soft
    rtr_c = 0.5 * rtr_a + 0.5 * tr_soft
    sd = float(np.std(ro_c) + 1e-12)
    ro_c = (ro_c - float(np.mean(ro_c))) / sd
    rtr_c = rtr_c / sd
    rt_c = pchip_quantile_map(rtr_c, ro_c, n_knots=N_KNOTS)

    # Corr matrix among key residuals
    corr_names = ["v10_ridge_orth_LM", "v10_cb_orth_LM", "v10_lgb_pure_fe_orth_LM", "soft_diff_orth_LM"]
    corr_vecs = [r_ridge, r_cb]
    if "v10_lgb_pure_fe" in pure_res:
        corr_vecs.append(pure_res["v10_lgb_pure_fe"][0])
    else:
        corr_vecs.append(np.zeros_like(r_ridge))
    corr_vecs.append(r_soft)
    C = np.corrcoef(np.column_stack(corr_vecs).T)
    corr = {
        corr_names[i]: {corr_names[j]: round(float(C[i, j]), 4) for j in range(4)}
        for i in range(4)
    }
    log(f"corr matrix: {corr}")

    log("PART2/3: step-grid portfolio audit")
    all_rows = []
    all_rows += eval_portfolio("A_V10_PassOnly", ro_a, rt_a, rtr_a, y, z_ref_o, z_st9_t, s9, s_p2, folds, STEP_GRID_AB)
    all_rows += eval_portfolio(
        "B_V10_Combo3_Clamped", r_combo, t_combo, tr_combo, y, z_ref_o, z_st9_t, s9, s_p2, folds, STEP_GRID_B_CLAMP
    )
    all_rows += eval_portfolio(
        "C_V10_PassOnly_plus_soft", ro_c, rt_c, rtr_c, y, z_ref_o, z_st9_t, s9, s_p2, folds, STEP_GRID_C
    )
    all_rows += eval_portfolio(
        "D_soft_diff_orth_LM_micro", r_soft, t_soft, tr_soft, y, z_ref_o, z_st9_t, s9, s_p2, folds, [STEP_D]
    )

    # Stage top 2 gated candidates
    gated = [r for r in all_rows if r["stage_gate"]]
    # prefer higher Δ then closer to 0.4e-4
    gated = sorted(gated, key=lambda r: (-r["delta_1e5"], abs(r["achieved_1e4"] - 0.4)))
    staged = []
    # map portfolio residual for writing
    port_res = {
        "A_V10_PassOnly": (ro_a, rt_a),
        "B_V10_Combo3_Clamped": (r_combo, t_combo),
        "C_V10_PassOnly_plus_soft": (ro_c, rt_c),
        "D_soft_diff_orth_LM_micro": (r_soft, t_soft),
    }
    out_names = [
        SUB / "sub_strike10A_v10_clean_micro.csv",
        SUB / "sub_strike10B_v10_softdiff_micro.csv",
    ]
    for i, r in enumerate(gated[:2]):
        ro, rt = port_res[r["portfolio"]]
        w, _ = weight_for_spearman_dist(z_st9_t, rt, r["target_1e4"] * 1e-4)
        # use achieved w from row
        w = r["w"]
        z_test = z_st9_t + w * rt
        meta = write_submission(out_names[i], test_ids, z_test, z_ref_o, y)
        meta.update({k: r[k] for k in ("portfolio", "target_1e4", "delta_1e5", "pos", "w")})
        staged.append(meta)
        # also save residual inject
        np.save(ART / f"strike10_cand{i+1}_r_oof.npy", ro.astype(np.float32))
        np.save(ART / f"strike10_cand{i+1}_r_test.npy", (w * rt).astype(np.float32))

    if len(gated) < 2:
        log(f"WARN only {len(gated)} candidates passed strict stage gate — staging best available under soft gate")
        soft = [
            r
            for r in all_rows
            if r["pos"] >= 4
            and 0.90 <= r["dispersion_ratio_raw"] <= 1.10
            and r["ks_raw"] < 0.010
            and 0.20e-4 <= r["achieved_1e4"] * 1e-4 <= 1.00e-4
        ]
        soft = sorted(soft, key=lambda r: (-r["delta_1e5"], abs(r["achieved_1e4"] - 0.4)))
        for i, r in enumerate(soft[:2]):
            if i < len(staged):
                continue
            if len(staged) >= 2:
                break
            ro, rt = port_res[r["portfolio"]]
            w = r["w"]
            meta = write_submission(out_names[len(staged)], test_ids, z_st9_t + w * rt, z_ref_o, y)
            meta.update({k: r[k] for k in ("portfolio", "target_1e4", "delta_1e5", "pos", "w")})
            meta["soft_gate"] = True
            staged.append(meta)

    # Markdown report
    md = [
        "# Step 3 — De-bias v10 + Strike-10 Stage (NO_SUBMIT)",
        "",
        f"Elapsed **{time.time()-T0:.1f}s**. z_ref_o **{base_auc:.7f}**. `NO_SUBMIT={NO_SUBMIT}`.",
        "",
        "## Part 1 — z_anchor diagnosis",
        "",
        f"- zanch features: **{diag['n_zanch']}**/481",
        f"- Top-20 LGB gain: **{diag['zanch_top20_count']}/20** are `zanch_*`",
        "",
        "| rank | feat | gain | split | zanch? |",
        "|---:|---|---:|---:|:---:|",
    ]
    for r in diag["top20_gain"]:
        md.append(f"| {r['rank']} | `{r['feat']}` | {r['gain']:.1f} | {r['split']} | {r['is_zanch']} |")
    md += ["", "### Worst zanch std_te/std_tr shifts", "", "| feat | std_ratio | mean_diff |", "|---|---:|---:|"]
    for r in diag["zanch_worst_std_shift"][:10]:
        md.append(f"| `{r['feat']}` | {r['std_ratio']} | {r['mean_diff']} |")
    md += ["", "## Part 1 — pure-FE orth audit", "", "| member | ΔOOF e-5 | w | pos | disp | KS | 1-ρ e-4 | gate |", "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for name, row in pure_audit.items():
        md.append(
            f"| {name}_orth_LM | {row['delta_1e5']} | {row['w']} | {row['pos']}/5 | "
            f"{row['dispersion_ratio_raw']} | {row['ks_raw']} | {row['spearman_dist_st9_test_1e4']} | "
            f"{'PASS' if row['gate_pass'] else 'fail'} |"
        )
    md += [
        "",
        f"Portfolio A includes lgb_pure: **{lgb_ok}**. A weights: `{np.round(w_a,4).tolist()}`",
        "",
        "## Residual correlation (OOF)",
        "",
        "```",
        json.dumps(corr, indent=2),
        "```",
        "",
        "## Part 2/3 — Strike-10 step-grid",
        "",
        "| portfolio | 1-ρ e-4 | w | ΔOOF e-5 | folds e-5 | pos | disp | KS | stage |",
        "|---|---:|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for r in all_rows:
        md.append(
            f"| {r['portfolio']} | {r['target_1e4']} | {r['w']} | {r['delta_1e5']} | "
            f"{r['fold_deltas_1e5']} | {r['pos']}/5 | {r['dispersion_ratio_raw']} | {r['ks_raw']} | "
            f"{'PASS' if r['stage_gate'] else 'fail'} |"
        )
    md += ["", "## Staged CSVs (NOT submitted)", ""]
    if not staged:
        md.append("_No candidate passed stage gate — no CSV written._")
    for s in staged:
        md.append(f"- `{s['path']}` portfolio={s.get('portfolio')} Δ={s.get('delta_1e5')}e-5 @1-ρ={s.get('target_1e4')}e-4 mean={s.get('mean')}")
    md += ["", "**Await confirmation before 1-slot live LB probe.**", ""]
    (OUT / "STEP3_STRIKE10.md").write_text("\n".join(md))
    payload = {
        "diagnosis": diag,
        "pure_audit": pure_audit,
        "portfolio_A_weights": w_a.tolist(),
        "lgb_pure_in_A": lgb_ok,
        "corr": corr,
        "grid": all_rows,
        "staged": staged,
        "NO_SUBMIT": NO_SUBMIT,
        "elapsed_s": time.time() - T0,
        "base_auc": base_auc,
    }
    (OUT / "step3_strike10.json").write_text(json.dumps(payload, indent=2))
    log("wrote STEP3_STRIKE10.md")
    log(f"staged={len(staged)} gated={len(gated)} NO_SUBMIT={NO_SUBMIT}")
    log("STEP3_COMPLETE — await confirmation before submit")
    print("STEP3_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "STEP3_FAIL.txt").write_text(traceback.format_exc())
        raise
