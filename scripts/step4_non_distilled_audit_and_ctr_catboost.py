#!/usr/bin/env python3
"""Step 4: Non-distilled audit (Strike6→9, LM H2_5F) + native CTR CatBoost.

STRICT NO_SUBMIT. SoftPseudo/zanch closed.
Outputs: preds/cb_native_ctr4_5f/ + logs/overnight/STEP4_NON_DISTILLED.md
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
from scipy.special import expit, logit
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

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
    fold_orth_multi,
    load_gpt2_med_5f,
    load_lm_bank,
    nested_at_w,
    weight_for_spearman_dist,
)

OUT = ROOT / "logs" / "overnight"
ART = PRED / "cb_native_ctr4_5f"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
T0 = time.time()
STATUS = OUT / "STEP4_STATUS.json"
NO_SUBMIT = True
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
RAW_COLS = [
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
] + FLAGS  # 24 raw


def log(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    try:
        STATUS.write_text(
            json.dumps(
                {
                    "t_s": round(time.time() - T0, 1),
                    "msg": msg,
                    "NO_SUBMIT": NO_SUBMIT,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
    except Exception:
        pass


def fold_deltas(y, z_base, r, folds, w):
    d = []
    for _, b in folds:
        d.append(
            round(
                (float(fast_auc(y[b], (z_base + w * r)[b])) - float(fast_auc(y[b], z_base[b])))
                * 1e5,
                2,
            )
        )
    return d, sum(1 for x in d if x > 0)


def lz_sub(fn, test_ids, z_ref):
    s = pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    return pchip_quantile_map(zrk(s), z_ref, n_knots=N_KNOTS), s


# -------------------- Part 1: Strike 6→9 + LM --------------------
def part1_strike_axis(y, test_ids, z_ref_o, z_st9_t, s9, folds):
    log("PART1: Strike 6→9 axis audit")
    # formulas from gm_top1_strike4_6_7_8_repro + strike9_final_max
    z_a3, _ = lz_sub("sub_gm_att3_continue_from_095315.csv", test_ids, z_ref_o)
    z_st2, _ = lz_sub("sub_gm_top3_strike2_homogeneous5f.csv", test_ids, z_ref_o)
    z_st3, _ = lz_sub("sub_strike3a_cont065_plus_glmquad_graph.csv", test_ids, z_ref_o)
    z_st4, _ = lz_sub("sub_gm_top3_strike4a_cont070.csv", test_ids, z_ref_o)
    z_st5, _ = lz_sub("sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv", test_ids, z_ref_o)
    z_st6, s6 = lz_sub("sub_gm_top3_strike6_reverse_strike5_from_095323.csv", test_ids, z_ref_o)
    z_st7, s7 = lz_sub("sub_gm_top3_strike7_push_for_top1.csv", test_ids, z_ref_o)
    z_st8, s8 = lz_sub("sub_gm_top1_strike8_claim_rank1.csv", test_ids, z_ref_o)
    z_st9, s9b = lz_sub("sub_gm_top1_strike9_final_max.csv", test_ids, z_ref_o)

    u_bad5 = (z_st5 - z_st4) - 0.45 * (z_st4 - z_st3)
    formulas = {
        "Strike4A": "z_st3 + 0.70*(z_st3-z_st2) + 0.15*(z_st2-z_a3)  → LB 0.95323",
        "Strike6": "z_st4 + 0.30*(z_st4-z_st3) - 0.35*v_bad_new5  → LB 0.95325",
        "Strike7": "z_st6 + 0.85*(z_st6-z_st4) + 0.15*(z_st4-z_st3)  → LB 0.95327",
        "Strike8": "z_st7 + 0.90*(z_st7-z_st6)  → LB 0.95329",
        "Strike9": "z_st8 + 0.78*(z_st8-z_st7) - 0.14*u_bad5  → LB 0.95330",
        "u_bad5": "u_bad5 = (z_st5-z_st4) - 0.45*(z_st4-z_st3)",
    }
    # verify St9 reconstruction
    z_9_re = pchip_quantile_map(z_st8 + 0.78 * (z_st8 - z_st7) - 0.14 * u_bad5, z_ref_o, n_knots=N_KNOTS)
    corr_st9 = float(np.corrcoef(z_9_re, z_st9)[0, 1])
    log(f"  St9 reconstruct corr vs saved={corr_st9:.6f}")

    # Strike 9.5 / 10_axis: continue along (St9-St8) with small bad5 damp, clamp 1-rho
    v_cont = z_st9 - z_st8
    cont_rows = []
    for a, bneg in [(0.55, 0.08), (0.70, 0.10), (0.85, 0.12), (1.00, 0.14)]:
        z_try = pchip_quantile_map(z_st9 + a * v_cont - bneg * u_bad5, z_ref_o, n_knots=N_KNOTS)
        p_try = expit(calibrate_logit_to_prevalence(z_try, z_ref_o, y))
        dist = (1.0 - float(spearmanr(p_try, s9).correlation)) * 1e4
        cont_rows.append(
            {
                "a_cont": a,
                "b_neg_bad5": bneg,
                "formula": f"z_st9 + {a}*(z_st9-z_st8) - {bneg}*u_bad5",
                "1m_rho_st9_e4": round(dist, 4),
                "step_rms": round(float(np.sqrt(np.mean((z_try - z_st9) ** 2))), 5),
            }
        )
        log(f"  St9.5 cand a={a} b={bneg}: 1-ρ(St9)={dist:.3f}e-4 rms={cont_rows[-1]['step_rms']}")

    # pairwise step distances on LB ladder
    ladder = [("St6", s6, 0.95325), ("St7", s7, 0.95327), ("St8", s8, 0.95329), ("St9", s9, 0.95330)]
    ladder_rho = {}
    for i in range(len(ladder) - 1):
        n0, s0, lb0 = ladder[i]
        n1, s1, lb1 = ladder[i + 1]
        d = (1.0 - float(spearmanr(s0, s1).correlation)) * 1e4
        ladder_rho[f"{n0}->{n1}"] = {"1m_rho_e4": round(d, 4), "dLB": round(lb1 - lb0, 5)}

    return {
        "formulas": formulas,
        "st9_reconstruct_corr": round(corr_st9, 6),
        "continuation_cands": cont_rows,
        "ladder_steps": ladder_rho,
        "z_st8": z_st8,
        "z_st9": z_st9,
        "u_bad5": u_bad5,
        "v_cont": v_cont,
    }


def part1_lm_h2(y, folds, z_ref_o, z_st9_t, s9, lm_bank):
    log("PART1: LM artifacts + H2_5F boundary-gated / disagreement gate")
    nte = len(z_st9_t)
    # inventory
    inv = {}
    for fi in range(5):
        op = PRED / f"gpt2_med_4ep/oof_{fi}.npy"
        tp = PRED / f"gpt2_med_4ep/test_{fi}.npy"
        inv[f"oof_{fi}"] = {"exists": op.exists(), "shape": list(np.load(op, mmap_mode="r").shape) if op.exists() else None}
        inv[f"test_{fi}"] = {"exists": tp.exists(), "shape": list(np.load(tp, mmap_mode="r").shape) if tp.exists() else None}
    gmo, gmt = load_gpt2_med_5f(y, folds, nte)
    inv["gpt2_med_5f_assembled"] = gmo is not None
    if gmo is not None:
        inv["gpt2_med_5f_oof_auc"] = round(float(fast_auc(y, gmo)), 7)
        log(f"  gpt2_med_5f assembled AUC={inv['gpt2_med_5f_oof_auc']}")

    # megayak / FSL
    mega = ROOT / "external/banks/megayak_s6e11-target-first-lm"
    if (mega / "lm_oof.npy").exists():
        mo = np.load(mega / "lm_oof.npy").ravel()
        mt = np.load(mega / "lm_test.npy").ravel()
        if len(mo) == len(y) and len(mt) == nte:
            lm_bank["megayak"] = (
                zrk(mo),
                pchip_quantile_map(zrk(mt), zrk(mo), n_knots=N_KNOTS),
            )
            log(f"  megayak LM AUC={fast_auc(y, zrk(mo)):.7f}")
    # FSL often in v8 submission path — check preds
    fsl_paths = list(PRED.glob("*fsl*")) + list((ROOT / "external").glob("**/*fsl*oof*"))
    inv["fsl_hits"] = [str(p) for p in fsl_paths[:10]]
    inv["lm_bank_keys"] = list(lm_bank.keys())

    results = {"inventory": inv}
    if gmo is None:
        log("  H2_5F SKIP — gpt2_med_5f incomplete")
        return results, lm_bank

    zo = zrk(gmo)
    zt = pchip_quantile_map(zrk(gmt), zo, n_knots=N_KNOTS)

    # Double-orth vs [z_ref, lm12]
    ctrl_names = []
    Xo_cols = [z_ref_o]
    Xt_cols = [z_st9_t]
    for k in ("lm12_wide", "lm12_raw"):
        if k in lm_bank:
            Xo_cols.append(lm_bank[k][0])
            Xt_cols.append(lm_bank[k][1])
            ctrl_names.append(k)
    Xo = np.column_stack(Xo_cols)
    Xt = np.column_stack(Xt_cols)
    ro, rt, rtr = fold_orth_multi(zo, zt, Xo, Xt, folds)
    nest = nested_delta(y, z_ref_o, ro, folds)
    diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
    fd, pos = fold_deltas(y, z_ref_o, ro, folds, nest["w"] if abs(nest["w"]) > 1e-12 else 0.02)
    h2 = {
        "name": "H2_5F_gpt2med_orth_zref_lm12",
        "controls": ["z_ref"] + ctrl_names,
        **nest,
        "fold_deltas_1e5": fd,
        "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
        "ks_raw": diag["ks_raw"],
    }
    log(
        f"  H2_5F: Δ={h2['delta_1e5']}e-5 w={h2['w']} {h2['pos']}/5 folds={fd} "
        f"disp={h2['dispersion_ratio_raw']} KS={h2['ks_raw']}"
    )
    np.save(ART / "H2_5F_gpt2med_orth_oof.npy", ro.astype(np.float32))
    np.save(ART / "H2_5F_gpt2med_orth_test.npy", rt.astype(np.float32))
    results["H2_5F"] = h2

    # LM consensus = mean of available LM bank members; tree backbone = z_ref
    Lm_o = np.column_stack([lm_bank[k][0] for k in lm_bank])
    Lm_t = np.column_stack([lm_bank[k][1] for k in lm_bank])
    z_lm_c_o = Lm_o.mean(1)
    z_lm_c_t = Lm_t.mean(1)
    # disagreement gate
    def gate(z_lm, z_tree):
        d = z_lm - z_tree
        p = expit(z_tree)
        return np.tanh(np.abs(d)) * p * (1.0 - p) * d

    g_o = gate(z_lm_c_o, z_ref_o)
    g_t = gate(z_lm_c_t, z_st9_t)
    # LOFO orth of gate vs z_ref
    ro_g, rt_g, rtr_g = fold_orth_multi(
        g_o, g_t, z_ref_o.reshape(-1, 1), z_st9_t.reshape(-1, 1), folds
    )
    nest_g = nested_delta(y, z_ref_o, ro_g, folds)
    diag_g = resid_diagnostics(ro_g, rt_g, rtr_g, s9, expit(z_st9_t + rt_g))
    fd_g, _ = fold_deltas(y, z_ref_o, ro_g, folds, nest_g["w"] if abs(nest_g["w"]) > 1e-12 else 0.02)
    gate_row = {
        "name": "LM_vs_Tree_disagreement_gate",
        **nest_g,
        "fold_deltas_1e5": fd_g,
        "dispersion_ratio_raw": diag_g["dispersion_ratio_raw"],
        "ks_raw": diag_g["ks_raw"],
    }
    log(
        f"  DisagreeGate: Δ={gate_row['delta_1e5']}e-5 {gate_row['pos']}/5 "
        f"disp={gate_row['dispersion_ratio_raw']} KS={gate_row['ks_raw']}"
    )
    np.save(ART / "lm_tree_disagree_gate_oof.npy", ro_g.astype(np.float32))
    np.save(ART / "lm_tree_disagree_gate_test.npy", rt_g.astype(np.float32))
    results["disagree_gate"] = gate_row
    return results, lm_bank


# -------------------- Part 2: Native CTR CatBoost --------------------
def exact_token(x):
    if isinstance(x, (float, np.floating)):
        return format(float(x), ".12g")
    return str(x)


def build_x_cat(df: pd.DataFrame) -> pd.DataFrame:
    """48 discrete string keys: 24 raw tokens + 24 Bulsah discrete."""
    out = {}
    for c in RAW_COLS:
        if c not in df.columns:
            continue
        col = df[c]
        kind = getattr(col.dtype, "kind", None)
        if kind in ("f", "i", "u") or pd.api.types.is_numeric_dtype(col):
            out[f"raw_{c}"] = col.map(lambda v: format(float(v), ".12g"))
        else:
            out[f"raw_{c}"] = col.astype(str)

    # ensure 24 raw — pad flags already in RAW_COLS
    # Bulsah discrete tokens
    def cents(series):
        return ((np.round(series.to_numpy(np.float64) * 100).astype(np.int64)) % 100).astype(str)

    def mant_len(series):
        s = pd.Series(np.round(series.to_numpy(np.float64), 8)).astype(str)
        return s.str.split(".").str[-1].str.len().fillna(0).astype(int).astype(str)

    def first2(series):
        s = (
            pd.Series(np.round(series.to_numpy(np.float64), 6))
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace("-", "", regex=False)
        )
        return s.str[:2].fillna("NA")

    def last2(series):
        s = (
            pd.Series(np.round(series.to_numpy(np.float64), 6))
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace("-", "", regex=False)
        )
        return s.str[-2:].fillna("NA")

    conts = [
        "amount_usd",
        "account_balance_usd",
        "velocity_score",
        "merchant_risk_score",
        "distance_from_home_km",
        "hours_since_last_txn",
    ]
    for c in conts:
        v = df[c]
        out[f"cents_{c}"] = cents(v)
        out[f"mantlen_{c}"] = mant_len(v)
        out[f"first2_{c}"] = first2(v)
        out[f"last2_{c}"] = last2(v)

    # step-grid parity + bitmask (fill remaining to 24 bulsah)
    # already 6*4=24 bulsah from above — replace last few with parity/bitmask if needed
    # User asked: step-grid parity bins and flag_bitmask_64 among the 24.
    # Swap mantlen for 2 cols into parity/bitmask for diversity:
    flags = df[FLAGS].to_numpy(np.float64)
    bm = (flags * (2 ** np.arange(flags.shape[1]))).sum(1).astype(np.int64)
    out["flag_bitmask_64"] = pd.Series(bm).astype(str)
    for c, step in [("velocity_score", 0.1), ("merchant_risk_score", 0.02), ("amount_usd", 1.0)]:
        stepv = np.rint(df[c].to_numpy(np.float64) / step).astype(np.int64)
        out[f"parity2_{c}"] = (stepv % 2).astype(str)
        out[f"parity5_{c}"] = (stepv % 5).astype(str)

    # trim/pad to exactly 48
    X = pd.DataFrame(out)
    # prefer: all raw_* + selected bulsah totaling 48
    raw_cols = [c for c in X.columns if c.startswith("raw_")]
    other = [c for c in X.columns if not c.startswith("raw_")]
    # take all raw (should be 24) + first 24 other
    keep = raw_cols[:24] + other[:24]
    if len(keep) < 48:
        keep = list(X.columns)[:48]
    X = X[keep[:48]].astype(str)
    assert X.shape[1] == 48, X.shape
    return X


def fit_cb_native_ctr(y, folds, X_tr, X_te, X_orig, y_orig, z_cal_o, z_cal_t, z_cal_orig):
    from catboost import CatBoostClassifier, Pool

    ntr, nte = len(y), len(X_te)
    cat_idx = list(range(X_tr.shape[1]))
    seeds = [42, 137, 2026]
    oof = np.zeros(ntr, dtype=np.float64)
    te_acc = []

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        max_ctr_complexity=4,
        l2_leaf_reg=25.0,
        learning_rate=0.035,
        iterations=1800,
        random_strength=1.5,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=80,
    )

    for fi, (a, b) in enumerate(folds):
        log(f"  CTR fold {fi}/5 seeds={len(seeds)}")
        # append orig20k to train fold
        Xa = pd.concat([X_tr.iloc[a], X_orig], axis=0, ignore_index=True)
        ya = np.concatenate([y[a], y_orig])
        base_a = np.concatenate([z_cal_o[a], z_cal_orig])
        Xb = X_tr.iloc[b]
        yb = y[b]
        base_b = z_cal_o[b]

        bag_b, bag_t = [], []
        for sd in seeds:
            train_pool = Pool(Xa, label=ya, cat_features=cat_idx, baseline=base_a)
            val_pool = Pool(Xb, label=yb, cat_features=cat_idx, baseline=base_b)
            model = CatBoostClassifier(**params, random_seed=sd)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            # raw margin / logit-ish: use predict_proba then logit, residual vs baseline
            pb = logit(np.clip(model.predict_proba(Xb)[:, 1], 1e-6, 1 - 1e-6)) - base_b
            pt = logit(np.clip(model.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)) - z_cal_t
            bag_b.append(pb)
            bag_t.append(pt)
            best_iter = model.get_best_iteration()
            log(f"    seed={sd} best_iter={best_iter}")
            del model, train_pool, val_pool
            gc.collect()
        oof[b] = np.mean(bag_b, 0)
        te_acc.append(np.mean(bag_t, 0))
        # ECDF align fold test to fold oof chunk scale later globally
        log(f"  CTR fold {fi} done")

    te = np.mean(te_acc, 0)
    # global PCHIP align test to oof
    te = pchip_quantile_map(te, oof, n_knots=N_KNOTS)
    return oof, te


def audit_orth(name, oof, test, y, z_ref_o, z_st9_t, s9, folds, lm_bank):
    names = list(lm_bank.keys())
    Xo = np.column_stack([z_ref_o] + [lm_bank[n][0] for n in names])
    Xt = np.column_stack([z_st9_t] + [lm_bank[n][1] for n in names])
    ro, rt, rtr = fold_orth_multi(oof.astype(np.float64), test.astype(np.float64), Xo, Xt, folds)
    nest = nested_delta(y, z_ref_o, ro, folds)
    diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
    w = nest["w"] if abs(nest["w"]) > 1e-12 else 0.02
    fd, pos = fold_deltas(y, z_ref_o, ro, folds, nest["w"] if abs(nest["w"]) > 1e-12 else w)
    rho = (1.0 - float(spearmanr(z_st9_t + (nest["w"] if abs(nest["w"]) > 1e-12 else 0) * rt, z_st9_t).correlation)) * 1e4
    # step table
    steps = []
    for td in [0.2e-4, 0.35e-4, 0.5e-4, 1.0e-4]:
        ww, got = weight_for_spearman_dist(z_st9_t, rt, td)
        nest_w = nested_at_w(y, z_ref_o, ro, folds, ww)
        fdw, posw = fold_deltas(y, z_ref_o, ro, folds, ww)
        steps.append(
            {
                "target_1e4": round(td * 1e4, 4),
                "achieved_1e4": round(got * 1e4, 4),
                "w": round(ww, 6),
                **nest_w,
                "fold_deltas_1e5": fdw,
                "pos": posw,
            }
        )
    row = {
        "name": name,
        **nest,
        "fold_deltas_1e5": fd,
        "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
        "ks_raw": diag["ks_raw"],
        "spearman_dist_st9_test_1e4": round(rho, 3),
        "steps": steps,
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
    log(
        f"  {name}: Δ={row['delta_1e5']}e-5 {row['pos']}/5 disp={row['dispersion_ratio_raw']} "
        f"KS={row['ks_raw']} gate={row['gate_pass']}"
    )
    return row


def main():
    log("STEP4 START (STRICT NO_SUBMIT) — SoftPseudo/zanch CLOSED")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    z_ref_o, z_st9_t, s9, base = build_z_ref(y, test_ids, folds)
    log(f"z_ref_o={base:.7f}")
    lm_bank = load_lm_bank(y, folds, len(y), len(test_ids), test_ids)

    strike = part1_strike_axis(y, test_ids, z_ref_o, z_st9_t, s9, folds)
    lm_res, lm_bank = part1_lm_h2(y, folds, z_ref_o, z_st9_t, s9, lm_bank)

    # Part 2
    log("PART2: build X_cat 48 discrete keys + native CTR CatBoost")
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    orig = pd.read_csv(ROOT / "original/credit_card_fraud_2026.csv")
    X_tr = build_x_cat(tr)
    X_te = build_x_cat(te)
    X_orig = build_x_cat(orig)
    y_orig = orig["is_fraud"].to_numpy(np.int32)
    log(f"  X_cat cols={list(X_tr.columns)}")

    z_cal_o = calibrate_logit_to_prevalence(z_ref_o, z_ref_o, y)
    z_cal_t = calibrate_logit_to_prevalence(z_st9_t, z_ref_o, y)
    # orig baseline: Stage-1 lookup = mean z_cal (no leakage of test)
    z_cal_orig = np.full(len(y_orig), float(np.mean(z_cal_o)), dtype=np.float64)

    del tr, te, orig
    gc.collect()

    oof, te_pred = fit_cb_native_ctr(
        y, folds, X_tr, X_te, X_orig, y_orig, z_cal_o, z_cal_t, z_cal_orig
    )
    np.save(ART / "cb_native_ctr4_5f_oof.npy", oof.astype(np.float32))
    np.save(ART / "cb_native_ctr4_5f_test.npy", te_pred.astype(np.float32))
    solo = float(fast_auc(y, z_cal_o + oof))
    log(f"  cb raw solo AUC vs y on z_cal+resid ≈ {solo:.7f}")

    cb_row = audit_orth(
        "cb_native_ctr4_5f", oof, te_pred, y, z_ref_o, z_st9_t, s9, folds, lm_bank
    )

    # markdown
    md = [
        "# Step 4 — Non-Distilled Audit + Native CTR CatBoost (NO_SUBMIT)",
        "",
        f"Elapsed **{time.time()-T0:.1f}s**. SoftPseudo/zanch **CLOSED**. z_ref_o **{base:.7f}**.",
        "",
        "## Part 1 — Strike 6→9 axis",
        "",
        "Exact formulas:",
        "",
    ]
    for k, v in strike["formulas"].items():
        md.append(f"- **{k}**: `{v}`")
    md += [
        "",
        f"St9 reconstruct corr: **{strike['st9_reconstruct_corr']}**",
        "",
        "Ladder steps:",
        "",
        "```",
        json.dumps(strike["ladder_steps"], indent=2),
        "```",
        "",
        "### Strike 9.5 / 10_axis continuation candidates (test-space, NO_SUBMIT)",
        "",
        "| a_cont | b_neg | 1-ρ(St9) e-4 | step_rms | formula |",
        "|---:|---:|---:|---:|---|",
    ]
    for r in strike["continuation_cands"]:
        md.append(
            f"| {r['a_cont']} | {r['b_neg_bad5']} | {r['1m_rho_st9_e4']} | {r['step_rms']} | `{r['formula']}` |"
        )

    md += ["", "## Part 1 — LM / H2_5F", ""]
    md.append(f"gpt2_med folds present: **{lm_res['inventory'].get('gpt2_med_5f_assembled')}**")
    md.append(f"LM bank: `{lm_res['inventory'].get('lm_bank_keys')}`")
    if "H2_5F" in lm_res:
        h = lm_res["H2_5F"]
        md.append(
            f"- H2_5F orth[z_ref,lm12]: Δ={h['delta_1e5']}e-5 {h['pos']}/5 "
            f"folds={h['fold_deltas_1e5']} disp={h['dispersion_ratio_raw']} KS={h['ks_raw']}"
        )
    if "disagree_gate" in lm_res:
        g = lm_res["disagree_gate"]
        md.append(
            f"- LM↔Tree disagreement gate: Δ={g['delta_1e5']}e-5 {g['pos']}/5 "
            f"disp={g['dispersion_ratio_raw']} KS={g['ks_raw']}"
        )

    md += [
        "",
        "## Part 2 — `cb_native_ctr4_5f` forensic gate",
        "",
        f"| member | ΔOOF e-5 | w | pos | folds | disp | KS | 1-ρ e-4 | gate |",
        f"|---|---:|---:|---:|---|---:|---:|---:|:---:|",
        f"| {cb_row['name']} | {cb_row['delta_1e5']} | {cb_row['w']} | {cb_row['pos']}/5 | "
        f"{cb_row['fold_deltas_1e5']} | {cb_row['dispersion_ratio_raw']} | {cb_row['ks_raw']} | "
        f"{cb_row['spearman_dist_st9_test_1e4']} | {'PASS' if cb_row['gate_pass'] else 'fail'} |",
        "",
        "### Step grid",
        "",
        "| 1-ρ e-4 | w | ΔOOF e-5 | folds | pos |",
        "|---:|---:|---:|---|---:|",
    ]
    for s in cb_row["steps"]:
        md.append(
            f"| {s['target_1e4']} | {s['w']} | {s['delta_1e5']} | {s['fold_deltas_1e5']} | {s['pos']}/5 |"
        )
    md += ["", f"Artifacts: `{ART}`", "", f"`NO_SUBMIT={NO_SUBMIT}`.", ""]
    (OUT / "STEP4_NON_DISTILLED.md").write_text("\n".join(md))
    payload = {
        "strike": {k: strike[k] for k in ("formulas", "st9_reconstruct_corr", "continuation_cands", "ladder_steps")},
        "lm": lm_res,
        "cb_native_ctr4_5f": cb_row,
        "NO_SUBMIT": NO_SUBMIT,
        "elapsed_s": time.time() - T0,
        "base_auc": base,
    }
    (OUT / "step4_non_distilled.json").write_text(json.dumps(payload, indent=2))
    log("wrote STEP4_NON_DISTILLED.md")
    log("STEP4_COMPLETE")
    print("STEP4_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "STEP4_FAIL.txt").write_text(traceback.format_exc())
        raise
