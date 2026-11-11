#!/usr/bin/env python3
"""Step 1: SoftFamily forensic autopsy + external OOF bank screen.

ZERO submissions. Memory-capped CPU only.
Outputs:
  logs/overnight/recon_and_forensic_audit.json
  logs/overnight/RECON_FORENSIC_AUDIT.md
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

from gm_top3_strike3_expanded7f import (
    HARV,
    N09,
    P12,
    PRED,
    SUB,
    fast_auc,
    orth_vs_v8,
    pchip_quantile_map,
    zrk,
)

OUT = ROOT / "logs" / "overnight"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
N_KNOTS = 2000


def log(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def fold_orth_1d(zo, zt, base_o, base_t, folds):
    """r = z - OLS(base) nested on OOF; test = mean fold models, PCHIP-matched to OOF resid."""
    ro = np.zeros(len(zo))
    rtf = []
    Xo = base_o.reshape(-1, 1)
    Xt = base_t.reshape(-1, 1)
    for a, b in folds:
        ols = LinearRegression().fit(Xo[a], zo[a])
        ua = zo[a] - ols.predict(Xo[a])
        mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
        ro[b] = (zo[b] - ols.predict(Xo[b]) - mu) / sd
        rtf.append((zt - ols.predict(Xt) - mu) / sd)
    rt_raw = np.mean(rtf, 0)
    rt = pchip_quantile_map(rt_raw, ro, n_knots=N_KNOTS)
    return ro, rt, rt_raw


def nested_delta(y, z_base, r, folds, w_grid=None):
    if w_grid is None:
        w_grid = np.linspace(-0.08, 0.08, 33)
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


def resid_diagnostics(ro, rt, rt_raw, s_base_test, s_comp_test):
    std_o = float(np.std(ro) + 1e-12)
    std_t = float(np.std(rt_raw) + 1e-12)  # pre-PCHIP for true scale mismatch
    std_t_matched = float(np.std(rt) + 1e-12)
    ks = float(ks_2samp(ro, rt_raw).statistic)
    ks_matched = float(ks_2samp(ro, rt).statistic)
    dist = (1.0 - float(spearmanr(s_comp_test, s_base_test).correlation)) * 1e4
    return {
        "std_r_oof": round(std_o, 6),
        "std_r_test_raw": round(std_t, 6),
        "std_r_test_pchip": round(std_t_matched, 6),
        "dispersion_ratio_raw": round(std_t / std_o, 4),
        "dispersion_ratio_pchip": round(std_t_matched / std_o, 4),
        "ks_raw": round(ks, 6),
        "ks_pchip": round(ks_matched, 6),
        "spearman_dist_st9_test_1e4": round(dist, 3),
    }


def passes_gate(d_oof, pos, disp, ks):
    return (
        d_oof >= 0.5
        and pos >= 4
        and 0.85 <= disp <= 1.15
        and ks < 0.015
    )


def build_z_ref(y, test_ids, folds):
    """Same St9-era honest basis used historically (z_ref_o ≈ 0.9526612)."""
    v8_sub = (
        pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(float)
    )
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_t = pchip_quantile_map(zrk(v8_sub), z_v8_o, n_knots=N_KNOTS)
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = pchip_quantile_map(zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel()), z_w12_o, n_knots=N_KNOTS)
    z_tabm_o = zrk(np.load(HARV / "oof_tabm_5f.npy").ravel())
    z_tabm_t = pchip_quantile_map(zrk(np.load(HARV / "test_tabm_5f.npy").ravel()), z_tabm_o, n_knots=N_KNOTS)
    z_xgbf_old_o = zrk(np.load(HARV / "oof_xgb_fast_harvest.npy").ravel())
    z_xgbf_old_t = pchip_quantile_map(
        zrk(np.load(HARV / "test_xgb_fast_harvest.npy").ravel()), z_xgbf_old_o, n_knots=N_KNOTS
    )
    r_old_o, r_old_t = orth_vs_v8(z_xgbf_old_o, z_xgbf_old_t, z_v8_o, z_v8_t, folds)
    r_tab_o, r_tab_t = orth_vs_v8(z_tabm_o, z_tabm_t, z_v8_o, z_v8_t, folds)
    del z_tabm_o, z_tabm_t, z_xgbf_old_o, z_xgbf_old_t
    gc.collect()

    sam_o = pd.read_csv(
        N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/oof.csv"
    )["nested_winner"].to_numpy(float)
    sam_t = (
        pd.read_csv(
            N09 / "sameershaikh27_s6e11-fraud-wide-pooled-stack-every-bank-found/submission_cavg.csv"
        )
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(float)
    )
    X_orth_o = np.column_stack([z_v8_o, z_w12_o, r_old_o - r_tab_o])
    X_orth_t = np.column_stack([z_v8_t, z_w12_t, r_old_t - r_tab_t])

    def fold_orth_pair(zo, zt, Xo, Xt):
        ro = np.zeros(len(zo))
        rtf = []
        for a, b in folds:
            ols = LinearRegression().fit(Xo[a], zo[a])
            ua = zo[a] - ols.predict(Xo[a])
            mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
            ro[b] = (zo[b] - ols.predict(Xo[b]) - mu) / sd
            rtf.append((zt - ols.predict(Xt) - mu) / sd)
        return ro, pchip_quantile_map(np.mean(rtf, 0), ro, n_knots=N_KNOTS)

    specs5 = [
        (np.load(PRED / "part11/lab4_tabular_nade_oof.npy").ravel(), np.load(PRED / "part11/lab4_tabular_nade_test.npy").ravel()),
        (sam_o, sam_t),
        (np.load(HARV / "oof_orig20k_knn.npy").ravel(), np.load(HARV / "test_orig20k_knn.npy").ravel()),
        (np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel(), np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel()),
        (np.load(PRED / "part11/lab1_zjoint_graph_lgb4_oof.npy").ravel(), np.load(PRED / "part11/lab1_zjoint_graph_lgb4_test.npy").ravel()),
    ]
    U5_o, U5_t = [], []
    for xo, xt in specs5:
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=N_KNOTS)
        ro, rt = fold_orth_pair(zo, zt, X_orth_o, X_orth_t)
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
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=N_KNOTS)
        ro, rt = fold_orth_pair(zo, zt, X7_o, X7_t)
        U2_o.append(ro)
        U2_t.append(rt)
    U2_o, U2_t = np.column_stack(U2_o), np.column_stack(U2_t)
    w2 = np.array([0.0150, 0.0125])
    z_ref_o = z_v8_o + 0.025 * (r_old_o - r_tab_o) + 0.85 * (U5_o @ w5) + 0.85 * (U2_o @ w2)
    # St9-proxy test: map St9 submission into z_ref OOF quantile space
    s9 = (
        pd.read_csv(SUB / "sub_gm_top1_strike9_final_max.csv")
        .set_index("id")
        .loc[test_ids, "is_fraud"]
        .to_numpy(float)
    )
    z_st9_t = pchip_quantile_map(zrk(s9), z_ref_o, n_knots=N_KNOTS)
    return z_ref_o, z_st9_t, s9, float(fast_auc(y, z_ref_o))


def audit_component(name, oof_path, test_path, y, z_st9_o, z_st9_t, s9, folds, ntr):
    xo = np.load(oof_path).ravel()
    xt = np.load(test_path).ravel()
    if len(xo) != ntr:
        return {"name": name, "error": f"oof len {len(xo)} != {ntr}"}
    zo = zrk(xo)
    zt = pchip_quantile_map(zrk(xt), zo, n_knots=N_KNOTS)
    ro, rt, rt_raw = fold_orth_1d(zo, zt, z_st9_o, z_st9_t, folds)
    nest = nested_delta(y, z_st9_o, ro, folds)
    # test spearman vs St9 probs: use raw xt ranks vs s9
    diag = resid_diagnostics(ro, rt, rt_raw, s9, xt if len(xt) == len(s9) else expit_safe(zt))
    # if xt length mismatch handled above; for spearman use mapped probs approx via ranks of zt
    diag["spearman_dist_st9_test_1e4"] = round(
        (1.0 - float(spearmanr(zt, z_st9_t).correlation)) * 1e4, 3
    )
    row = {
        "name": name,
        "oof": str(oof_path),
        "test": str(test_path),
        **nest,
        **diag,
        "gate_pass": passes_gate(
            nest["delta_1e5"], nest["pos"], diag["dispersion_ratio_raw"], diag["ks_raw"]
        ),
    }
    return row


def expit_safe(z):
    from scipy.special import expit

    return expit(z)


def discover_pairs(dirs, ntr, nte):
    pairs = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        # patterns: *_oof.npy + *_test.npy / *_preds.npy / pred_*.npy / test_*.npy
        for op in sorted(d.rglob("*oof*.npy")):
            if op.name.startswith("._"):
                continue
            stem = op.name
            cands = []
            if "_oof.npy" in stem:
                base = stem.replace("_oof.npy", "")
                cands += [
                    op.with_name(base + "_test.npy"),
                    op.with_name(base + "_preds.npy"),
                    op.with_name(base + "_pred.npy"),
                    op.with_name(base + "_full.npy"),
                    op.with_name("test_" + base + ".npy"),
                    op.with_name("pred_" + base + ".npy"),
                ]
            if stem.startswith("oof_"):
                key = stem[len("oof_") :]
                cands += [
                    op.with_name("test_" + key),
                    op.with_name("pred_" + key),
                    op.with_name("preds_" + key),
                ]
            # subodh style: x_oof + x_preds
            if stem.endswith("_oof.npy"):
                cands.append(op.with_name(stem.replace("_oof.npy", "_preds.npy")))
            tp = next((p for p in cands if p.exists() and not p.name.startswith("._")), None)
            if tp is None:
                continue
            try:
                if op.stat().st_size > 80_000_000 or tp.stat().st_size > 80_000_000:
                    continue
                xo = np.load(op, mmap_mode="r")
                xt = np.load(tp, mmap_mode="r")
                if xo.size != ntr or xt.size != nte:
                    continue
                pairs.append((f"{d.name}:{op.stem}", op, tp))
            except Exception:
                continue
    # dedupe by oof path
    seen = set()
    out = []
    for nm, op, tp in pairs:
        k = str(op.resolve())
        if k in seen:
            continue
        seen.add(k)
        out.append((nm, op, tp))
    return out


def main():
    report = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "St9 has no pure nested OOF; z_ref_o (≈0.9526612) is the St9-stack OOF proxy. Test anchor = PCHIP(St9 submission → z_ref_o).",
        "part_a_softfamily": [],
        "part_b_subodh_features": {},
        "part_b_screen": [],
        "part_b_flagged": [],
    }
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    log(f"loaded y/ids ntr={ntr} nte={nte}")

    # ---------- Part B feature list (cheap) ----------
    fe_path = ROOT / "external/nov10_recon/subodh/train_fe.csv"
    if fe_path.exists():
        cols = pd.read_csv(fe_path, nrows=0).columns.tolist()
        feat_cols = [c for c in cols if c not in ("id", "fold", "is_fraud")]
        report["part_b_subodh_features"] = {
            "n_feature_cols": len(feat_cols),
            "excluded": ["id", "fold", "is_fraud"],
            "features": feat_cols,
            "definitions_inferred": {
                "base_competition_cols": [
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
                ],
                "engineered": {
                    "risk_flag_count": "sum of binary risk flags",
                    "vpn_and_mismatch": "used_vpn AND ip_country_mismatch",
                    "foreign_and_new_merchant": "is_foreign_transaction AND is_new_merchant",
                    "amount_cents": "fractional cents of amount_usd",
                    "amount_is_round": "amount is integer dollars",
                    "amount_is_round_10": "amount multiple of 10",
                    "is_night": "time_of_day_hour in night window",
                    "is_weekend": "day_of_week weekend",
                    "amount_usd_dec / hours_since_last_txn_dec / velocity_score_dec / distance_from_home_km_dec": "decimal/fractional parts",
                    "hours_te": "nested target encoding of hours bucket",
                    "amount_bin200_te": "nested TE of amount binned (~200 bins)",
                    "fold": "StratifiedKFold id (not a model feature)",
                },
            },
        }
        log(f"Subodh features: {len(feat_cols)} cols")

    # ---------- Build St9 anchors ----------
    log("building z_ref / St9 anchors...")
    z_st9_o, z_st9_t, s9, base_auc = build_z_ref(y, test_ids, folds)
    report["st9_proxy_oof_auc"] = round(base_auc, 7)
    log(f"z_ref/St9-proxy OOF AUC={base_auc:.7f}")

    # ---------- Part A: SoftFamily autopsy ----------
    log("Part A: SoftFamily forensic autopsy")
    fam_members = [
        ("soft_diff", PRED / "glm_softpseudo_clean_diff_oof.npy", PRED / "glm_softpseudo_clean_diff_test.npy"),
        ("idf_p3", PRED / "part11/lab1_idf_bayes_p3_m20_oof.npy", PRED / "part11/lab1_idf_bayes_p3_m20_test.npy"),
        ("gb_soft_bag3", PRED / "gb_softpseudo_clean_bag3_oof.npy", PRED / "gb_softpseudo_clean_bag3_test.npy"),
        ("clean035", PRED / "glm_softpseudo_clean_w0.35_oof.npy", PRED / "glm_softpseudo_clean_w0.35_test.npy"),
    ]
    W = {"soft_diff": 0.0600, "idf_p3": 0.0175, "gb_soft_bag3": 0.0425, "clean035": 0.0100}
    U_o, U_t = [], []
    for nm, op, tp in fam_members:
        row = audit_component(nm, op, tp, y, z_st9_o, z_st9_t, s9, folds, ntr)
        report["part_a_softfamily"].append(row)
        log(
            f"  {nm:12s} Δ={row.get('delta_1e5')}e-5 w={row.get('w')} {row.get('pos')}/5 "
            f"disp={row.get('dispersion_ratio_raw')} KS={row.get('ks_raw')} "
            f"1-ρ={row.get('spearman_dist_st9_test_1e4')}e-4"
        )
        zo = zrk(np.load(op).ravel())
        zt = pchip_quantile_map(zrk(np.load(tp).ravel()), zo, n_knots=N_KNOTS)
        ro, rt, _ = fold_orth_1d(zo, zt, z_st9_o, z_st9_t, folds)
        U_o.append(W[nm] * ro)
        U_t.append(W[nm] * rt)
    # combined SoftFamily
    dz_o = np.sum(U_o, axis=0)
    # recompute combined diagnostics properly
    # reconstruct combined z then orth
    zo_c = z_st9_o * 0
    # better: combine member raw z with weights in rank space then orth
    # Use weighted sum of already-orth residuals as SoftFamily pack (matches injection)
    nest_c = nested_delta(y, z_st9_o, dz_o, folds)
    # For disp/KS: build combined raw residual before pchip by summing weighted raw members
    raw_o_parts, raw_t_parts = [], []
    for nm, op, tp in fam_members:
        zo = zrk(np.load(op).ravel())
        zt = pchip_quantile_map(zrk(np.load(tp).ravel()), zo, n_knots=N_KNOTS)
        ro, rt, rt_raw = fold_orth_1d(zo, zt, z_st9_o, z_st9_t, folds)
        raw_o_parts.append(W[nm] * ro)
        raw_t_parts.append(W[nm] * rt_raw)
    ro_c = np.sum(raw_o_parts, axis=0)
    rt_raw_c = np.sum(raw_t_parts, axis=0)
    rt_c = pchip_quantile_map(rt_raw_c, ro_c, n_knots=N_KNOTS)
    diag_c = resid_diagnostics(ro_c, rt_c, rt_raw_c, s9, s9)
    # spearman of SoftFamily inject direction vs St9: use rt_c distance
    diag_c["spearman_dist_st9_test_1e4"] = round(
        (1.0 - float(spearmanr(z_st9_t + rt_c, z_st9_t).correlation)) * 1e4, 3
    )
    comb = {
        "name": "SoftFamily_combined",
        "weights": W,
        **nest_c,
        **diag_c,
        "gate_pass": passes_gate(
            nest_c["delta_1e5"], nest_c["pos"], diag_c["dispersion_ratio_raw"], diag_c["ks_raw"]
        ),
        "public_lb_P2_x085": 0.95306,
        "public_lb_st9": 0.95330,
        "public_delta_P2": -0.00024,
    }
    report["part_a_softfamily"].append(comb)
    log(
        f"  SoftFamily   Δ={comb['delta_1e5']}e-5 {comb['pos']}/5 disp={comb['dispersion_ratio_raw']} "
        f"KS={comb['ks_raw']} | PUBLIC P2 went 0.95330→0.95306"
    )

    # ---------- Part B: screen external pairs ----------
    log("Part B: discovering OOF/test pairs...")
    dirs = [
        ROOT / "external/nov10_recon/subodh",
        ROOT / "external/nov10_recon/sk20220623",
        ROOT / "external/nov10_recon/alex_refresh",
        ROOT / "external/nov10_recon/goodpjw_refresh",
        ROOT / "external/nov09_retake_recon/alex_09524_nov09",
    ]
    pairs = discover_pairs(dirs, ntr, nte)
    log(f"found {len(pairs)} valid oof+test pairs")

    flagged = []
    for i, (nm, op, tp) in enumerate(pairs):
        try:
            row = audit_component(nm, op, tp, y, z_st9_o, z_st9_t, s9, folds, ntr)
            report["part_b_screen"].append(
                {
                    k: row[k]
                    for k in (
                        "name",
                        "delta_1e5",
                        "w",
                        "pos",
                        "dispersion_ratio_raw",
                        "ks_raw",
                        "spearman_dist_st9_test_1e4",
                        "gate_pass",
                    )
                    if k in row
                }
            )
            if row.get("gate_pass"):
                flagged.append(row)
                log(
                    f"  FLAG {nm[:40]:40s} Δ={row['delta_1e5']:+.2f}e-5 {row['pos']}/5 "
                    f"disp={row['dispersion_ratio_raw']} KS={row['ks_raw']}"
                )
            elif i % 15 == 0:
                log(f"  ... scanned {i+1}/{len(pairs)}")
        except Exception as e:
            report["part_b_screen"].append({"name": nm, "error": str(e)[:120]})
        if i % 5 == 0:
            gc.collect()

    # sort flagged / near misses
    report["part_b_flagged"] = sorted(flagged, key=lambda r: -r.get("delta_1e5", -1e9))
    near = [
        r
        for r in report["part_b_screen"]
        if isinstance(r.get("delta_1e5"), (int, float))
        and r.get("delta_1e5", 0) >= 0.5
        and r.get("pos", 0) >= 4
        and not r.get("gate_pass")
    ]
    near = sorted(near, key=lambda r: -r["delta_1e5"])[:20]
    report["part_b_near_miss_oof_ok_but_shift"] = near
    report["elapsed_s"] = round(time.time() - T0, 1)
    report["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    (OUT / "recon_and_forensic_audit.json").write_text(json.dumps(report, indent=2))

    # Markdown tables
    def fmt_a(rows):
        lines = [
            "| component | ΔOOF e-5 | w | pos | disp stdT/stdO | KS | 1-ρ(St9) e-4 | gate |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for r in rows:
            if "error" in r:
                lines.append(f"| {r['name']} | ERR | | | | | | |")
                continue
            lines.append(
                f"| {r['name']} | {r['delta_1e5']} | {r['w']} | {r['pos']}/5 | "
                f"{r['dispersion_ratio_raw']} | {r['ks_raw']} | {r['spearman_dist_st9_test_1e4']} | "
                f"{'PASS' if r.get('gate_pass') else 'fail'} |"
            )
        return "\n".join(lines)

    md = []
    md.append("# Recon & Forensic Audit (ZERO submits)")
    md.append("")
    md.append(f"Elapsed **{report['elapsed_s']}s**. St9-proxy OOF AUC **{base_auc:.7f}**.")
    md.append("")
    md.append("## Part A — SoftFamily autopsy (why P2 → 0.95306)")
    md.append("")
    md.append(report["note"])
    md.append("")
    md.append(fmt_a(report["part_a_softfamily"]))
    md.append("")
    md.append(
        "**Reading:** `dispersion_ratio_raw << 1` or `>> 1` ⇒ train/test residual scale mismatch; "
        "`KS` large ⇒ covariate shift in residual. Combined SoftFamily had strong nested OOF but "
        "public P2×0.85 crashed **−0.00024** — do not trust OOF alone when disp/KS are bad."
    )
    md.append("")
    md.append("## Part B — Subodh 38 features")
    md.append("")
    feats = report["part_b_subodh_features"].get("features", [])
    md.append(f"**Count:** {len(feats)} (excludes id/fold/is_fraud)")
    md.append("")
    md.append("```")
    md.append("\n".join(feats))
    md.append("```")
    md.append("")
    md.append("## Part B — Flagged OOF/Test pairs (strict gate)")
    md.append("")
    md.append("Gate: Δ≥0.5e-5, ≥4/5, 0.85≤disp≤1.15, KS<0.015")
    md.append("")
    if report["part_b_flagged"]:
        md.append(fmt_a(report["part_b_flagged"]))
    else:
        md.append("_None passed the full gate._")
    md.append("")
    md.append("## Near-misses (OOF ok, shift fails gate)")
    md.append("")
    if near:
        md.append(fmt_a(near))
    else:
        md.append("_None._")
    md.append("")
    md.append(f"JSON: `logs/overnight/recon_and_forensic_audit.json`")
    (OUT / "RECON_FORENSIC_AUDIT.md").write_text("\n".join(md))
    log("DONE")
    log(f"flagged={len(report['part_b_flagged'])} near={len(near)} pairs={len(pairs)}")
    print("RECON_FORENSIC_AUDIT_COMPLETE", flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        (OUT / "recon_and_forensic_audit_FATAL.txt").write_text(err)
        raise
