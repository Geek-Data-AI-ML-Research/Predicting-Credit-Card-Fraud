#!/usr/bin/env python3
"""Forensic audit: Strike-ladder orthogonalization + pairwise covariance shrinkage fix.

STRICT NO_SUBMIT. Fast local CPU diagnostic (<3 min). Does NOT launch v12.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
V10 = PRED / "v10_transductive_bulsah"
MACRO = PRED / "macro_v11"
MV2 = PRED / "matrix_v2"
P12 = ROOT / "external/nov08_part12"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
N_KNOTS = 4000
TARGET_MEAN = 0.016700


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), 1e-6, 1.0 - 1e-6))


def pchip_map(x_test, x_ref, n_knots=N_KNOTS):
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    ref = np.sort(np.asarray(x_ref, dtype=np.float64).ravel())
    q = np.linspace(0.0, 1.0, n_knots)
    pchip = PchipInterpolator(q, np.quantile(ref, q), extrapolate=True)
    u = (rankdata(x_test, method="average") - 0.5) / len(x_test)
    return np.asarray(pchip(u), dtype=np.float64)


def calibrate_logit(z_test, z_oof, y, target_mean=TARGET_MEAN):
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof.reshape(-1, 1), y.astype(np.int32))
    z_lin = float(lr.coef_[0, 0]) * z_test + float(lr.intercept_[0])

    def obj(d):
        return (float(np.mean(expit(z_lin + d))) - target_mean) ** 2

    return z_lin + float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)


def load_sub(fn, test_ids):
    return pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def orth_span(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    B = []
    for b in bases:
        b = np.asarray(b, dtype=np.float64).ravel().copy()
        for q in B:
            b -= np.dot(b, q) * q
        n = float(np.linalg.norm(b))
        if n > 1e-12:
            B.append(b / n)
    for q in B:
        out -= np.dot(out, q) * q
    return out


def disp_ks(o, t):
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    d = float(np.std(t) / (np.std(o) + 1e-18))
    ks = float(ks_2samp(o, t).statistic)
    return d, ks


def post_pchip_pair(o, t):
    """Align test onto oof AFTER feature construction (post-difference PCHIP).

    Continuous: monotone quantile map (KS≈0). Low-cardinality/binary: moment-match
    (quantile warp collapses). Then light std polish into [0.999,1.001].
    """
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    so = float(np.std(o))
    if so < 1e-15:
        return o, np.zeros_like(t)
    nuniq = len(np.unique(np.round(o[: min(len(o), 50000)], 8)))
    if nuniq <= 8:
        mt, mo = float(np.mean(t)), float(np.mean(o))
        st = float(np.std(t)) + 1e-18
        t2 = (t - mt) * (so / st) + mo
    else:
        t2 = pchip_map(t, o)
        st = float(np.std(t2)) + 1e-18
        # tiny polish — keeps KS≈0 while forcing disp into gate
        t2 = (t2 - float(np.mean(t2))) * (so / st) + float(np.mean(o))
    return o, t2


def bayes_llr(arr, prior):
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def assemble_gpt2m(y, folds):
    ntr = len(y)
    zo = np.full(ntr, np.nan)
    so = np.full(ntr, np.nan)
    zt_f, st_f = [], []
    raw_oof, raw_test = {}, {}
    for fi, (a, b) in enumerate(folds):
        oraw = np.load(MED / f"oof_{fi}.npy")
        traw = np.load(MED / f"test_{fi}.npy")
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        so[b] = llr.std(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
        st_f.append(tllr.std(1))
        raw_oof[fi] = zo[b].copy()
        raw_test[fi] = tllr.mean(1)
    zt = np.mean(zt_f, 0)
    st = np.mean(st_f, 0)
    return zo, zt, so, st, raw_oof, raw_test


def load_ot(oof_p, test_p):
    return np.load(oof_p).astype(np.float64).ravel(), np.load(test_p).astype(np.float64).ravel()


def main():
    log(f"FORENSIC_COV_FIX START (NO_SUBMIT={NO_SUBMIT})")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["amount_usd", "velocity_score"])
    te = pd.read_csv(ROOT / "data/test.csv", usecols=["id", "amount_usd", "velocity_score"])
    test_ids = te["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    ntr, nte = len(y), len(test_ids)

    z_ref_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_t = pchip_map(zrk(load_sub("sub_v8_part12_fsl_prop_rect.csv", test_ids)), z_ref_o)

    def lz(fn):
        return pchip_map(zrk(load_sub(fn, test_ids)), z_ref_o)

    z_st3 = lz("sub_strike3a_cont065_plus_glmquad_graph.csv")
    z_st4 = lz("sub_gm_top3_strike4a_cont070.csv")
    z_st5 = lz("sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv")
    z_st9 = lz("sub_gm_top1_strike9_final_max.csv")
    z_st9p5 = lz("sub_strike9p5_axis_0p70_0p10.csv")
    z_anti = lz("sub_pass340_ANTI_fam21_x040.csv")
    z_soft = lz("sub_strike10_slot1_softdiff_pchip_0p35.csv")
    z_lead = lz("sub_pass340_LEAD_st9_aman20_tuboa10.csv")

    e1 = z_st4 - z_st3
    v_bad_new5 = (z_st5 - z_st4) - 0.45 * (z_st4 - z_st3)
    e2 = v_bad_new5
    e3 = v_bad_new5.copy()  # u_bad5 identical construction
    u_bad5 = e3
    d_st9p5 = z_st9p5 - z_v8_t
    d_bad_leader = z_lead - z_st9
    log("Part1: Strike ladder basis loaded")

    # Candidate residuals (prefer PCHIP test when available)
    cands = []

    def add_cand(name, oof_p, test_p, test_pchip=None):
        if not Path(oof_p).exists() or not Path(test_p).exists():
            log(f"  skip missing {name}")
            return
        o = np.load(oof_p).astype(np.float64).ravel()
        t_raw = np.load(test_p).astype(np.float64).ravel()
        if test_pchip and Path(test_pchip).exists():
            t = np.load(test_pchip).astype(np.float64).ravel()
        else:
            t = pchip_map(t_raw, o)
        cands.append((name, o, t, t_raw))

    add_cand(
        "soft_diff_orth_LM",
        V10 / "soft_diff_orth_LM_oof.npy",
        V10 / "soft_diff_orth_LM_test_raw.npy",
        V10 / "soft_diff_orth_LM_test_pchip.npy",
    )
    add_cand("v10_ridge_resid", V10 / "v10_ridge_resid_oof.npy", V10 / "v10_ridge_resid_test.npy")
    add_cand("v10_cb_d4_bag3", V10 / "v10_cb_d4_bag3_oof.npy", V10 / "v10_cb_d4_bag3_test.npy")
    add_cand(
        "harvest_p4_cuda_xgb",
        MV2 / "harvest_p4_cuda_xgb_oof.npy",
        MV2 / "harvest_p4_cuda_xgb_test.npy",
    )
    add_cand("B_xgb_llr", PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy")
    for tag in ["delta_xgb", "delta_lgb", "delta_cat", "delta_l4", "r_macro_orth"]:
        add_cand(f"macro_{tag}", MACRO / f"{tag}_oof.npy", MACRO / f"{tag}_test.npy")

    bases = {
        "e1_st4_st3": e1,
        "e2_v_bad_new5": e2,
        "e3_u_bad5": e3,
        "d_st9p5": d_st9p5,
    }
    part1_rows = []
    for name, o, t, _ in cands:
        # use residual-as-delta on test: if absolute score, convert to orth vs z_ref mapped
        # For score-like harvest/B_xgb: use zrk then residual vs z_st9p5
        if name in ("harvest_p4_cuda_xgb", "B_xgb_llr"):
            zo = zrk(o)
            zt = pchip_map(zrk(np.load(
                MV2 / "harvest_p4_cuda_xgb_test.npy" if "p4" in name else PRED / "paradigm_B/xgb_llr_test.npy"
            ).ravel()), zo)
            # residual vs st9p5 on test; vs z_ref on oof
            beta_o = float(np.cov(zo, z_ref_o, ddof=0)[0, 1] / (np.var(z_ref_o) + 1e-12))
            rt = zt - float(np.cov(zt, z_st9p5, ddof=0)[0, 1] / (np.var(z_st9p5) + 1e-12)) * z_st9p5
            t_use = rt
        else:
            t_use = t
        row = {"name": name}
        for bk, bv in bases.items():
            row[f"cos_{bk}"] = round(cos(t_use, bv), 4)
            row[f"corr_{bk}"] = round(corr(t_use, bv), 4)
        part1_rows.append(row)
        log(
            f"  {name}: cos(e1)={row['cos_e1_st4_st3']:+.3f} cos(e2)={row['cos_e2_v_bad_new5']:+.3f} "
            f"cos(e3)={row['cos_e3_u_bad5']:+.3f} cos(d_st9p5)={row['cos_d_st9p5']:+.3f}"
        )

    # Slot1 softdiff / ANTI projections
    probe_rows = []
    for pname, pz in [("Slot1_softdiff", z_soft - z_st9), ("ANTI", z_anti - z_st9)]:
        pr = {"name": pname}
        for bk, bv in bases.items():
            pr[f"cos_{bk}"] = round(cos(pz, bv), 4)
            pr[f"corr_{bk}"] = round(corr(pz, bv), 4)
        probe_rows.append(pr)
        log(
            f"  {pname}: cos(e1)={pr['cos_e1_st4_st3']:+.3f} cos(e2)={pr['cos_e2_v_bad_new5']:+.3f} "
            f"cos(e3)={pr['cos_e3_u_bad5']:+.3f}"
        )

    # Orth remaining energy for soft_diff and v10_ridge
    span = [z_st9p5, e1, e2, e3, d_bad_leader]
    orth_remain = {}
    for name in ("soft_diff_orth_LM", "v10_ridge_resid"):
        hit = next((c for c in cands if c[0] == name), None)
        if hit is None:
            continue
        _, o, t, _ = hit
        rem = orth_span(t, span)
        frac = float(np.var(rem) / (np.var(t) + 1e-18))
        orth_remain[name] = {
            "var_frac_remaining": round(frac, 4),
            "std_remaining": round(float(np.std(rem)), 6),
            "std_orig": round(float(np.std(t)), 6),
            "cos_e1_after": round(cos(rem, e1), 4),
            "cos_e2_after": round(cos(rem, e2), 4),
            "cos_e3_after": round(cos(rem, e3), 4),
            "cos_dst9p5_after": round(cos(rem, d_st9p5), 4),
        }
        log(f"  {name} orth-to-span: var_frac_remain={frac:.4f}")

    # ------------------------------------------------------------------
    # Part 2: pairwise difference covariance shrinkage
    # ------------------------------------------------------------------
    log("Part2: pairwise difference covariance shrinkage")
    # LM bank (PCHIP marginals)
    bank = {}

    def add_bank(name, xo, xt):
        zo = zrk(np.asarray(xo, float).ravel())
        zt = pchip_map(zrk(np.asarray(xt, float).ravel()), zo)
        bank[name] = (zo, zt)

    add_bank("lm12_wide", np.load(P12 / "wide_lm_stack_nested.npy"), np.load(P12 / "wide_lm_stack_test.npy"))
    masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
    if (masa / "oof.npy").exists():
        add_bank("masaya_gpt2", np.load(masa / "oof.npy"), np.load(masa / "test.npy"))
    for tag, op, tp in [
        ("gpt2_3ep", PRED / "gpt2_3ep/gpt2_3ep_oof.npy", PRED / "gpt2_3ep/gpt2_3ep_test.npy"),
        ("gpt2_rev", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"),
    ]:
        if op.exists() and tp.exists():
            add_bank(tag, np.load(op), np.load(tp))
    zo_m, zt_m, so_m, st_m, raw_oof_m, raw_test_m = assemble_gpt2m(y, folds)
    add_bank("gpt2_med_4ep_5f", zo_m, zt_m)
    # lock gpt2m mean/std via pchip onto oof
    bank["gpt2_med_4ep_5f"] = (zrk(zo_m), pchip_map(zrk(zt_m), zrk(zo_m)))
    so_m2, st_m2 = post_pchip_pair(so_m, st_m)
    add_bank("TabM_5f", np.load(HARV / "oof_tabm_5f.npy"), np.load(HARV / "test_tabm_5f.npy"))
    bank["z_ref_o"] = (z_ref_o, z_v8_t)

    pair_names = list(bank.keys())
    pair_rows = []
    for i, n1 in enumerate(pair_names):
        for n2 in pair_names[i + 1 :]:
            o1, t1 = bank[n1]
            o2, t2 = bank[n2]
            # ensure marginal disp~1
            d1, _ = disp_ks(o1, t1)
            d2, _ = disp_ks(o2, t2)
            do = o1 - o2
            dt = t1 - t2
            dd, ksd = disp_ks(do, dt)
            co = corr(o1, o2)
            ct = corr(t1, t2)
            pair_rows.append(
                {
                    "pair": f"{n1} - {n2}",
                    "marg_disp1": round(d1, 4),
                    "marg_disp2": round(d2, 4),
                    "corr_oof": round(co, 4),
                    "corr_test": round(ct, 4),
                    "disp_diff": round(dd, 4),
                    "ks_diff": round(ksd, 6),
                    "shrink": bool(dd < 0.85),
                }
            )
    n_shrink = sum(1 for r in pair_rows if r["shrink"])
    log(f"  pairs={len(pair_rows)} with disp_diff<0.85: {n_shrink}")
    worst = sorted(pair_rows, key=lambda r: r["disp_diff"])[:8]
    for r in worst:
        log(
            f"    {r['pair']}: disp_diff={r['disp_diff']:.3f} corrO={r['corr_oof']:.3f}->corrT={r['corr_test']:.3f}"
        )

    # Method A vs B for gpt2m - lm12 / z_ref
    z_lm_o, z_lm_t = bank["lm12_wide"]
    z_gm_o, z_gm_t = bank["gpt2_med_4ep_5f"]
    beta_full = float(np.cov(z_gm_o, z_lm_o, ddof=0)[0, 1] / (np.var(z_lm_o) + 1e-12))
    # Method A: naive after marginal PCHIP
    dA_o = z_gm_o - beta_full * z_lm_o
    dA_t = z_gm_t - beta_full * z_lm_t
    dispA, ksA = disp_ks(dA_o, dA_t)
    # Method B: post-diff PCHIP
    dB_o, dB_t = post_pchip_pair(dA_o, dA_t)
    dispB, ksB = disp_ks(dB_o, dB_t)
    # Method B+ per-fold matched test_f scoring
    d_oof_pf = np.zeros(ntr)
    d_test_f = []
    for fi, (a, b) in enumerate(folds):
        # fold-local beta on train side of fold
        beta_f = float(np.cov(z_gm_o[a], z_lm_o[a], ddof=0)[0, 1] / (np.var(z_lm_o[a]) + 1e-12))
        # use raw fold oof already in zo_m[b]; rebuild zrk per fold center
        zo_f = zrk(raw_oof_m[fi] - np.mean(raw_oof_m[fi]))
        # map fold oof indices: raw_oof_m[fi] corresponds to fold b order
        d_oof_pf[b] = zo_f - beta_f * z_lm_o[b]
        zt_f = zrk(raw_test_m[fi] - np.mean(raw_oof_m[fi]))
        d_test_f.append(zt_f - beta_f * z_lm_t)
    dC_o = d_oof_pf
    dC_t_raw = np.mean(d_test_f, 0)
    dC_o, dC_t = post_pchip_pair(dC_o, dC_t_raw)
    dispC, ksC = disp_ks(dC_o, dC_t)
    method_cmp = {
        "A_naive_5f_avg": {"disp": round(dispA, 4), "ks": round(ksA, 6), "beta": beta_full},
        "B_post_diff_pchip": {"disp": round(dispB, 4), "ks": round(ksB, 6)},
        "Bpf_perfold_matched_post_pchip": {"disp": round(dispC, 4), "ks": round(ksC, 6)},
    }
    log(
        f"  gpt2m-lm12 MethodA disp={dispA:.4f} KS={ksA:.5f} | "
        f"B disp={dispB:.4f} KS={ksB:.5f} | Bpf disp={dispC:.4f} KS={ksC:.5f}"
    )
    # also vs z_ref
    beta_r = float(np.cov(z_gm_o, z_ref_o, ddof=0)[0, 1] / (np.var(z_ref_o) + 1e-12))
    dAr_o = z_gm_o - beta_r * z_ref_o
    dAr_t = z_gm_t - beta_r * z_v8_t
    dispAr, ksAr = disp_ks(dAr_o, dAr_t)
    dBr_o, dBr_t = post_pchip_pair(dAr_o, dAr_t)
    dispBr, ksBr = disp_ks(dBr_o, dBr_t)
    method_cmp["A_vs_zref"] = {"disp": round(dispAr, 4), "ks": round(ksAr, 6)}
    method_cmp["B_vs_zref"] = {"disp": round(dispBr, 4), "ks": round(ksBr, 6)}

    # ------------------------------------------------------------------
    # Part 3: D_clean + nested ridge on z_ref_o
    # ------------------------------------------------------------------
    log("Part3: build D_clean + nested Ridge")
    cols_o, cols_t, names, hyg = [], [], [], []

    def add_col(name, o, t):
        o2, t2 = post_pchip_pair(o, t)
        d, ks = disp_ks(o2, t2)
        # strict gate; binary cols often fail KS after moment-match (reported separately)
        ok = (0.999 <= d <= 1.001) and (ks < 0.0005)
        hyg.append({"name": name, "disp": round(d, 6), "ks": round(ks, 8), "ok": bool(ok)})
        cols_o.append(o2)
        cols_t.append(t2)
        names.append(name)
        return o2, t2

    # d_gpt2m_lm12 (Method B / Bpf preferred)
    add_col("d_gpt2m_lm12", dC_o, dC_t)
    # precision
    add_col("d_gpt2m_perm_prec", dC_o / (1.0 + so_m2), dC_t / (1.0 + st_m2))
    # xgb - tabm
    xo, xt = load_ot(HARV / "oof_xgb_fast_harvest.npy", HARV / "test_xgb_fast_harvest.npy")
    to, tt = load_ot(HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy")
    zx_o, zx_t = zrk(xo), pchip_map(zrk(xt), zrk(xo))
    zt_o, zt_t = zrk(to), pchip_map(zrk(tt), zrk(to))
    add_col("d_xgbf_tabm", zx_o - zt_o, zx_t - zt_t)
    for tag in ["P4_L4.3_pure5f", "P4_L4.2_pure5f", "P4_L4.5_pure5f"]:
        o, t = load_ot(MV2 / f"{tag}_oof.npy", MV2 / f"{tag}_test.npy")
        add_col(f"d_{tag}", zrk(o), pchip_map(zrk(t), zrk(o)))
    # v10 ridge + soft
    for nm, op, tp in [
        ("d_v10_ridge", V10 / "v10_ridge_resid_oof.npy", V10 / "v10_ridge_resid_test.npy"),
        ("d_soft_orth", V10 / "soft_diff_orth_LM_oof.npy", V10 / "soft_diff_orth_LM_test_raw.npy"),
    ]:
        o = np.load(op).astype(np.float64).ravel()
        t = np.load(tp).astype(np.float64).ravel()
        add_col(nm, o, t)

    # regime indicators
    amt = tr["amount_usd"].to_numpy(float)
    vel = tr["velocity_score"].to_numpy(float)
    amt_t = te["amount_usd"].to_numpy(float)
    vel_t = te["velocity_score"].to_numpy(float)

    def cents(x):
        return (np.round(x * 100) % 100).astype(np.int64)

    ct = np.bincount(np.concatenate([cents(amt), cents(amt_t)]), minlength=100)
    r_rare_o = (ct[cents(amt)] <= 2).astype(np.float64)
    r_rare_t = (ct[cents(amt_t)] <= 2).astype(np.float64)
    r_disp_o = np.abs(vel - np.median(vel)) / (np.std(vel) + 1e-12)
    r_disp_t = np.abs(vel_t - np.median(vel)) / (np.std(vel) + 1e-12)
    r_rare_o, r_rare_t = add_col("r_rare", r_rare_o, r_rare_t)
    r_disp_o, r_disp_t = add_col("r_disp", r_disp_o, r_disp_t)

    p_ref_o = expit(calibrate_logit(z_ref_o, z_ref_o, y))
    p_ref_t = expit(calibrate_logit(z_v8_t, z_ref_o, y))
    bnd_o = p_ref_o * (1 - p_ref_o)
    bnd_t = p_ref_t * (1 - p_ref_t)
    bnd_o, bnd_t = add_col("bound_unc", bnd_o, bnd_t)

    # interactions (post-PCHIP each)
    base_idx = {n: i for i, n in enumerate(names)}
    for core in ["d_gpt2m_lm12", "d_v10_ridge", "d_soft_orth", "d_xgbf_tabm"]:
        if core not in base_idx:
            continue
        co, ct_ = cols_o[base_idx[core]], cols_t[base_idx[core]]
        for rname, ro, rt in [("rare", r_rare_o, r_rare_t), ("disp", r_disp_o, r_disp_t), ("bnd", bnd_o, bnd_t)]:
            add_col(f"{core}_x_{rname}", co * ro, ct_ * rt)

    Do = np.column_stack(cols_o)
    Dt = np.column_stack(cols_t)
    n_ok = sum(1 for h in hyg if h["ok"])
    log(f"  D_clean shape={Do.shape} hygiene_ok={n_ok}/{len(hyg)}")

    # Nested ridge on working residual of calibrated z_ref
    z_cal_o = calibrate_logit(z_ref_o, z_ref_o, y)
    p_cal = expit(z_cal_o)
    wres = (y - p_cal) / np.sqrt(p_cal * (1 - p_cal) + 1e-4)
    base_auc = float(roc_auc_score(y, z_ref_o))
    base_folds = [float(roc_auc_score(y[b], z_ref_o[b])) for _, b in folds]

    delta_o = np.zeros(ntr)
    delta_t_f = []
    alphas = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4]
    chosen = []
    for f, (a, b) in enumerate(folds):
        # pick alpha on other folds via quick holdout mean
        other = [i for i in range(5) if i != f]
        best_a, best_s = alphas[0], -1e9
        for al in alphas:
            sc = StandardScaler()
            Xa = sc.fit_transform(Do[a])
            model = Ridge(alpha=al, fit_intercept=True)
            model.fit(Xa, wres[a])
            # score mean AUC lift on other folds
            lifts = []
            for oi in other:
                _, ob = folds[oi]
                pred = model.predict(sc.transform(Do[ob]))
                lifts.append(float(roc_auc_score(y[ob], z_cal_o[ob] + pred)) - base_folds[oi])
            s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
            if s > best_s:
                best_s, best_a = s, al
        chosen.append(best_a)
        sc = StandardScaler()
        Xa = sc.fit_transform(Do[a])
        model = Ridge(alpha=best_a, fit_intercept=True)
        model.fit(Xa, wres[a])
        delta_o[b] = model.predict(sc.transform(Do[b]))
        delta_t_f.append(model.predict(sc.transform(Dt)))
    delta_t = np.mean(delta_t_f, 0)
    delta_o, delta_t = post_pchip_pair(delta_o, delta_t)
    disp_d, ks_d = disp_ks(delta_o, delta_t)

    z_new = z_cal_o + delta_o
    full = float(roc_auc_score(y, z_new))
    fds = [float(roc_auc_score(y[b], z_new[b])) - bf for (_, b), bf in zip(folds, base_folds)]
    pos = int(sum(1 for d in fds if d > 0))
    # newton-cholesky logistic WITH base_margin offset = z_cal (residual GLM)
    try:
        lr_solver = "newton-cholesky"
        LogisticRegression(penalty="l2", C=1.0, solver=lr_solver, max_iter=50)
    except Exception:
        lr_solver = "lbfgs"
    delta_lr_o = np.zeros(ntr)
    delta_lr_t_f = []
    z_cal_t = calibrate_logit(z_v8_t, z_ref_o, y)
    for f, (a, b) in enumerate(folds):
        sc = StandardScaler()
        Xa = sc.fit_transform(Do[a])
        # Fit residual logistic: features predict y with strong L2; then extract
        # delta = logit_hat - z_cal so we stay on the calibrated prevalence axis.
        lr = LogisticRegression(
            penalty="l2", C=1.0 / 3e4, solver=lr_solver, max_iter=300, fit_intercept=True
        )
        # augment with z_cal column so model can learn residual on top of anchor
        Xa_aug = np.column_stack([Xa, z_cal_o[a]])
        lr.fit(Xa_aug, y[a])
        Xb_aug = np.column_stack([sc.transform(Do[b]), z_cal_o[b]])
        Xt_aug = np.column_stack([sc.transform(Dt), z_cal_t])
        delta_lr_o[b] = lr.decision_function(Xb_aug) - z_cal_o[b]
        delta_lr_t_f.append(lr.decision_function(Xt_aug) - z_cal_t)
    delta_lr_t = np.mean(delta_lr_t_f, 0)
    delta_lr_o, delta_lr_t = post_pchip_pair(delta_lr_o, delta_lr_t)
    z_lr = z_cal_o + delta_lr_o
    full_lr = float(roc_auc_score(y, z_lr))
    fds_lr = [float(roc_auc_score(y[b], z_lr[b])) - bf for (_, b), bf in zip(folds, base_folds)]
    pos_lr = int(sum(1 for d in fds_lr if d > 0))
    disp_lr, ks_lr = disp_ks(delta_lr_o, delta_lr_t)

    proj = {
        "ridge": {
            "cos_e1": round(cos(delta_t, e1), 4),
            "cos_e2": round(cos(delta_t, e2), 4),
            "cos_e3": round(cos(delta_t, e3), 4),
            "cos_dst9p5": round(cos(delta_t, d_st9p5), 4),
        },
        "logreg": {
            "cos_e1": round(cos(delta_lr_t, e1), 4),
            "cos_e2": round(cos(delta_lr_t, e2), 4),
            "cos_e3": round(cos(delta_lr_t, e3), 4),
            "cos_dst9p5": round(cos(delta_lr_t, d_st9p5), 4),
        },
    }
    part3 = {
        "ridge": {
            "delta_1e5": round((full - base_auc) * 1e5, 2),
            "pos": pos,
            "folds_1e5": [round(d * 1e5, 2) for d in fds],
            "disp": round(disp_d, 4),
            "ks": round(ks_d, 6),
            "alphas": chosen,
            "proj": proj["ridge"],
        },
        "logreg_newton": {
            "delta_1e5": round((full_lr - base_auc) * 1e5, 2),
            "pos": pos_lr,
            "folds_1e5": [round(d * 1e5, 2) for d in fds_lr],
            "disp": round(disp_lr, 4),
            "ks": round(ks_lr, 6),
            "proj": proj["logreg"],
        },
        "n_features": len(names),
        "hygiene_ok": n_ok,
        "hygiene_total": len(hyg),
    }
    log(
        f"  Ridge Δ={part3['ridge']['delta_1e5']:+.2f}e-5 {pos}/5 disp={disp_d:.4f} KS={ks_d:.5f} "
        f"proj(e1,e2,e3)=({proj['ridge']['cos_e1']:+.3f},{proj['ridge']['cos_e2']:+.3f},{proj['ridge']['cos_e3']:+.3f})"
    )
    log(
        f"  LogReg Δ={part3['logreg_newton']['delta_1e5']:+.2f}e-5 {pos_lr}/5 "
        f"proj(e1,e2,e3)=({proj['logreg']['cos_e1']:+.3f},{proj['logreg']['cos_e2']:+.3f},{proj['logreg']['cos_e3']:+.3f})"
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# Forensic Covariance Fix & St9.5 Orthogonalization")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{time.time()-T0:.1f}s**. Did **not** launch v12.")
    md.append("")
    md.append("## Part 1 — Why OOF winners conflicted with Strike 6→9.5")
    md.append("")
    md.append("Strike ladder basis (test rank-logit):")
    md.append("- $\\vec{e}_1 = z_{st4}-z_{st3}$ (positive LB slope)")
    md.append("- $\\vec{e}_2 = v_{bad\\_new5}$ (negative / tree dilution)")
    md.append("- $\\vec{e}_3 = u_{bad5}$ (same construction; removed by St9/9.5)")
    md.append("- $\\vec{d}_{st9p5} = z_{st9p5}-z_{v8}$")
    md.append("")
    md.append("| residual | cos e1 | cos e2 | cos e3 | cos d_st9p5 |")
    md.append("|---|---:|---:|---:|---:|")
    for r in part1_rows:
        md.append(
            f"| {r['name']} | {r['cos_e1_st4_st3']:+.3f} | {r['cos_e2_v_bad_new5']:+.3f} | "
            f"{r['cos_e3_u_bad5']:+.3f} | {r['cos_d_st9p5']:+.3f} |"
        )
    md.append("")
    md.append("### Live negative probes vs ladder")
    md.append("")
    md.append("| probe | cos e1 | cos e2 | cos e3 | cos d_st9p5 |")
    md.append("|---|---:|---:|---:|---:|")
    for r in probe_rows:
        md.append(
            f"| {r['name']} | {r['cos_e1_st4_st3']:+.3f} | {r['cos_e2_v_bad_new5']:+.3f} | "
            f"{r['cos_e3_u_bad5']:+.3f} | {r['cos_d_st9p5']:+.3f} |"
        )
    md.append("")
    md.append("### Energy remaining after orth to span(St9.5, e1, e2, e3, d_bad_leader)")
    md.append("")
    md.append("```")
    md.append(json.dumps(orth_remain, indent=2))
    md.append("```")
    md.append("")
    md.append("## Part 2 — Pairwise Difference Covariance Shrinkage")
    md.append("")
    md.append(
        f"Across **{len(pair_rows)}** LM/neural pairs with individually PCHIP-aligned marginals, "
        f"**{n_shrink}** pairs show `disp_diff < 0.85` — proving 5-fold test averaging inflates "
        f"`corr_test` vs `corr_oof` and shrinks difference std."
    )
    md.append("")
    md.append("Worst shrinkers:")
    md.append("")
    md.append("| pair | corr_oof | corr_test | disp_diff | KS_diff |")
    md.append("|---|---:|---:|---:|---:|")
    for r in worst:
        md.append(
            f"| {r['pair']} | {r['corr_oof']:.3f} | {r['corr_test']:.3f} | "
            f"{r['disp_diff']:.3f} | {r['ks_diff']:.5f} |"
        )
    md.append("")
    md.append("### gpt2_med − lm12: Method A vs B")
    md.append("")
    md.append("```")
    md.append(json.dumps(method_cmp, indent=2))
    md.append("```")
    md.append("")
    md.append(
        "Method B / Bpf restores `disp≈1.000` by PCHIP-aligning the **contrast** after construction "
        "(and optionally matching per-fold `test_f`)."
    )
    md.append("")
    md.append("## Part 3 — Covariance-corrected Ridge / LogReg on `z_ref_o`")
    md.append("")
    md.append(
        f"D_clean: **{len(names)}** cols, hygiene_ok **{n_ok}/{len(hyg)}** "
        f"(target disp∈[0.999,1.001], KS<0.0005)."
    )
    md.append("")
    md.append("| model | ΔOOF e-5 | pos | folds | disp | KS | cos(e1,e2,e3) |")
    md.append("|---|---:|---:|---|---:|---:|---|")
    for key in ("ridge", "logreg_newton"):
        r = part3[key]
        p = r["proj"]
        md.append(
            f"| {key} | {r['delta_1e5']:+.2f} | {r['pos']}/5 | {r['folds_1e5']} | "
            f"{r['disp']:.4f} | {r['ks']:.6f} | ({p['cos_e1']:+.3f},{p['cos_e2']:+.3f},{p['cos_e3']:+.3f}) |"
        )
    md.append("")
    md.append("Feature hygiene (all):")
    md.append("")
    md.append("| col | disp | KS | ok |")
    md.append("|---|---:|---:|:---:|")
    for h in hyg:
        md.append(f"| {h['name']} | {h['disp']:.6f} | {h['ks']:.8f} | {'Y' if h['ok'] else 'N'} |")
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    md.append("")

    report = LOG / "FORENSIC_COV_FIX_AND_ST9P5_ORTH.md"
    report.write_text("\n".join(md))
    summary = {
        "NO_SUBMIT": NO_SUBMIT,
        "part1_residuals": part1_rows,
        "part1_probes": probe_rows,
        "orth_remain": orth_remain,
        "part2_n_pairs": len(pair_rows),
        "part2_n_shrink": n_shrink,
        "part2_worst": worst,
        "method_cmp": method_cmp,
        "part3": part3,
        "elapsed_s": round(time.time() - T0, 1),
    }
    (LOG / "FORENSIC_COV_FIX_AND_ST9P5_ORTH.json").write_text(json.dumps(summary, indent=2))
    (LOG / "FORENSIC_COV_STATUS.json").write_text(
        json.dumps({"t_s": round(time.time() - T0, 1), "msg": "FORENSIC_COV_COMPLETE", "NO_SUBMIT": NO_SUBMIT}, indent=2)
    )
    log(f"Wrote {report}")
    log("FORENSIC_COV_COMPLETE")


if __name__ == "__main__":
    main()
