#!/usr/bin/env python3
"""Inspect v8 / z_ref_o lineage + F0/F1-corrupted xgb_gpt2_3ep + 5F lm_super (NO_SUBMIT)."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from recon_and_forensic_audit import build_z_ref

PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
P12 = ROOT / "external/nov08_part12"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), 1e-6, 1.0 - 1e-6))


def pchip_map(xt, xo, n_knots=4000):
    xt = np.asarray(xt, dtype=np.float64).ravel()
    xo = np.sort(np.asarray(xo, dtype=np.float64).ravel())
    q = np.linspace(0.0, 1.0, n_knots)
    pchip = PchipInterpolator(q, np.quantile(xo, q), extrapolate=True)
    u = (rankdata(xt, method="average") - 0.5) / len(xt)
    return np.asarray(pchip(u), dtype=np.float64)


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def fmt_folds(a):
    return "[" + ", ".join(f"{x:.7f}" for x in a) + "]"


def fmt_d1e5(base, cur):
    return "[" + ", ".join(f"{(c - b) * 1e5:+.2f}" for b, c in zip(base, cur)) + "]"


def assemble_gpt2m(y, folds):
    ntr = len(y)
    zo = np.full(ntr, np.nan)
    so = np.full(ntr, np.nan)
    zt_f, st_f = [], []
    for fi, (a, b) in enumerate(folds):
        oraw = np.load(MED / f"oof_{fi}.npy")
        traw = np.load(MED / f"test_{fi}.npy")
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        so[b] = llr.std(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
        st_f.append(tllr.std(1))
    return zo, np.mean(zt_f, 0), so, np.mean(st_f, 0)


def method_b_align(o, t, folds):
    """Per-fold std OOF + full-domain PCHIP test → oof (disp≈1)."""
    o = np.asarray(o, float).ravel()
    t = np.asarray(t, float).ravel()
    o2 = np.zeros_like(o)
    for a, b in folds:
        mu, sd = float(o[b].mean()), float(o[b].std() + 1e-12)
        o2[b] = (o[b] - mu) / sd
    o2 = (o2 - o2.mean()) / (o2.std() + 1e-12)
    t2 = (t - t.mean()) / (t.std() + 1e-12)
    # jitter ties
    def jit(x):
        r = rankdata(x, method="ordinal").astype(float)
        return x + 1e-11 * (r / len(x) - 0.5)

    oj, tj = jit(o2), jit(t2)
    q = np.linspace(0, 1, 10001)
    pchip = PchipInterpolator(np.quantile(tj, q), np.quantile(oj, q), extrapolate=True)
    ta = np.asarray(pchip(tj), float)
    ta = (ta - ta.mean()) / (ta.std() + 1e-12)
    return o2, ta


def main():
    log(f"V8_CORE_INSPECTION START (NO_SUBMIT={NO_SUBMIT})")
    # quarantine note
    qdir = SUB / "_quarantine_nullspace_v3"
    q_files = sorted([p.name for p in qdir.glob("sub_strike10_nullspace_v3_*.csv")]) if qdir.exists() else []
    log(f"quarantined nullspace_v3: {q_files}")

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    idx = {f: b for f, (_, b) in enumerate(folds)}

    # ------------------------------------------------------------------
    # v8
    # ------------------------------------------------------------------
    v8_o = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel()
    z_v8 = zrk(v8_o)
    v8_auc = float(roc_auc_score(y, z_v8))
    v8_folds = fold_aucs(y, z_v8, folds)
    log(f"v8 OOF AUC={v8_auc:.7f} folds={fmt_folds(v8_folds)}")

    v8_formula = {
        "script": "scripts/build_v8_part12_rect_slot2.py",
        "artifact": "preds/P_super_v8_part12_rect_{oof,test}.npy",
        "submission": "submissions/sub_v8_part12_fsl_prop_rect.csv (LB 0.95307)",
        "backbone": "z_w12 = zrk(external/nov08_part12/wide_lm_stack_nested.npy)  # Part12 201-member LM stack",
        "components_KS_rect_orth_vs_w12": [
            "P_super_v4p_hc",
            "paradigm_B/xgb_llr",
            "matrix_v2/P4_L4.3_pure5f",
            "matrix_v2/P4_L4.2_pure5f",
            "matrix_v2/P4_L4.5_pure5f",
            "matrix_v2/P4_L4.3_ext",
            "gpt2_3ep/gpt2_bidir",
            "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6",
        ],
        "formula_oof": "z_v8_oof = z_w12 + R @ w_fold   (nested 5F coord ascent; R = KS-rect orth residuals)",
        "formula_test_slot2": "rk( zrk(0.68*zrk(w12_t)+0.32*zrk(part9_t)) + 1.0 * dz_orth_fsl )",
        "note": "xgb_gpt2_3ep is INSIDE v8 residual basis — F0/F1 gpt2_3ep corruption therefore contaminates v8.",
    }

    # ------------------------------------------------------------------
    # z_ref_o
    # ------------------------------------------------------------------
    z_ref_o, z_st9_t, s9, z_ref_auc = build_z_ref(y, test_ids, folds)
    z_ref_folds = fold_aucs(y, z_ref_o, folds)
    log(f"z_ref_o AUC={z_ref_auc:.7f} folds={fmt_folds(z_ref_folds)}")
    z_ref_formula = {
        "script": "scripts/recon_and_forensic_audit.py::build_z_ref",
        "formula": (
            "z_ref_o = z_v8_o + 0.025*(r_xgb_fast - r_tabm) "
            "+ 0.85*(U5 @ [0.0135, 0.0115, -0.0085, -0.0085, -0.0105]) "
            "+ 0.85*(U2 @ [0.0150, 0.0125])"
        ),
        "U5_orth_vs_[z_v8,z_w12,r_xgb-r_tabm]": [
            "part11/lab4_tabular_nade",
            "sameershaikh27 nested_winner",
            "orig20k_knn",
            "harvest_p4_cuda_xgb",
            "lab1_zjoint_graph_lgb4",
        ],
        "U2_orth_vs_X7": ["ladder/glm_quad", "part11/lab1_glm_plus_graph"],
        "oof_auc": z_ref_auc,
    }

    # ------------------------------------------------------------------
    # LM bank + xgb_gpt2_3ep + gb_gpt2_bag5 + clean harvest
    # ------------------------------------------------------------------
    bank = {}

    def add(name, path_o, path_t=None):
        if not Path(path_o).exists():
            log(f"  MISSING {name}: {path_o}")
            return
        xo = np.load(path_o).ravel()
        zo = zrk(xo)
        bank[name] = {
            "oof": zo,
            "auc": float(roc_auc_score(y, zo)),
            "folds": fold_aucs(y, zo, folds),
            "path": str(path_o),
        }
        log(f"  {name}: AUC={bank[name]['auc']:.7f} folds={fmt_folds(bank[name]['folds'])}")

    add("lm12_wide", P12 / "wide_lm_stack_nested.npy", P12 / "wide_lm_stack_test.npy")
    add("lm12_raw", P12 / "lm_oof.npy", P12 / "lm_test.npy")
    masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
    add("masaya_gpt2", masa / "oof.npy", masa / "test.npy")
    add("gpt2_rev", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy")
    add("gpt2_3ep", PRED / "gpt2_3ep/gpt2_3ep_oof.npy")
    add("lm_gpt2_3ep", PRED / "lm_gpt2_3ep/lm_oof_5f.npy")
    mega = ROOT / "external/banks/megayak_s6e11-target-first-lm"
    add("megayak", mega / "lm_oof.npy")
    add("xgb_gpt2_3ep", PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy")
    add("lgb_gpt2_3ep_fixed", PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy")
    add("gb_gpt2_bag5", PRED / "gb_gpt2_bag5_oof.npy")
    add("xgb_fast_harvest", HARV / "oof_xgb_fast_harvest.npy")
    add("xgb_clean_decorrupt", HARV / "oof_xgb_clean.npy")

    # gpt2-medium 5f
    gmo, gmt, smo, smt = assemble_gpt2m(y, folds)
    z_gm = zrk(gmo)
    bank["gpt2_med_4ep_5f"] = {
        "oof": z_gm,
        "auc": float(roc_auc_score(y, z_gm)),
        "folds": fold_aucs(y, z_gm, folds),
        "path": "preds/gpt2_med_4ep/oof_{0..4}.npy (8-order mean LLR)",
    }
    log(f"  gpt2_med_4ep_5f: AUC={bank['gpt2_med_4ep_5f']['auc']:.7f} folds={fmt_folds(bank['gpt2_med_4ep_5f']['folds'])}")

    # ------------------------------------------------------------------
    # Corruption check: gpt2_3ep vs lm12 on F0/F1
    # ------------------------------------------------------------------
    corr_rows = []
    if "gpt2_3ep" in bank and "lm12_wide" in bank:
        for f in range(5):
            b = idx[f]
            c = float(np.corrcoef(bank["gpt2_3ep"]["oof"][b], bank["lm12_wide"]["oof"][b])[0, 1])
            sp = float(spearmanr(bank["gpt2_3ep"]["oof"][b], bank["lm12_wide"]["oof"][b]).correlation)
            # also vs lm12_raw if available
            c_raw = None
            if "lm12_raw" in bank:
                c_raw = float(np.corrcoef(bank["gpt2_3ep"]["oof"][b], bank["lm12_raw"]["oof"][b])[0, 1])
            corr_rows.append({"fold": f, "pearson_lm12_wide": c, "spearman_lm12_wide": sp, "pearson_lm12_raw": c_raw})
            flag = " *** CORRUPT?" if f in (0, 1) and c > 0.999 else ""
            log(f"  gpt2_3ep vs lm12_wide F{f}: pearson={c:.6f} spearman={sp:.6f}{flag}")

    # ------------------------------------------------------------------
    # lm_super: lm12 + w * r_gpt2m  across all 5 folds
    # ------------------------------------------------------------------
    z_lm = bank["lm12_wide"]["oof"]
    # residual gpt2m vs lm12 (global beta then Method-B style)
    beta = float(np.cov(z_gm, z_lm, ddof=0)[0, 1] / (np.var(z_lm) + 1e-12))
    r_gm = z_gm - beta * z_lm
    r_gm_a, _ = method_b_align(r_gm, r_gm, folds)  # fold-std only on oof side
    # use fold-std residual
    r_gm_f = np.zeros(len(y))
    for a, b in folds:
        rr = r_gm[b]
        r_gm_f[b] = (rr - rr.mean()) / (rr.std() + 1e-12)

    lm_base_folds = bank["lm12_wide"]["folds"]
    w_grid = np.linspace(-0.02, 0.15, 69)
    # per-fold best w (fit on other 4 folds, eval on holdout) + fixed w=0.07
    lm_super_fixed = {}
    lm_super_nested = np.zeros(len(y))
    nested_ws = []
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        best_w, best_s = 0.0, -1e9
        for w in w_grid:
            lifts = [
                float(roc_auc_score(y[idx[oi]], z_lm[idx[oi]] + w * r_gm_f[idx[oi]])) - lm_base_folds[oi]
                for oi in other
            ]
            s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
            if s > best_s:
                best_s, best_w = s, float(w)
        nested_ws.append(best_w)
        lm_super_nested[b] = z_lm[b] + best_w * r_gm_f[b]

    # fixed w=0.07 all folds
    z_fixed = z_lm + 0.07 * r_gm_f
    fixed_folds = fold_aucs(y, z_fixed, folds)
    nested_folds = fold_aucs(y, lm_super_nested, folds)
    lm_super = {
        "beta_gpt2m_vs_lm12": beta,
        "fixed_w": 0.07,
        "fixed_auc": float(roc_auc_score(y, z_fixed)),
        "fixed_folds": fixed_folds,
        "fixed_delta_1e5_vs_lm12": [(a - b) * 1e5 for a, b in zip(fixed_folds, lm_base_folds)],
        "nested_ws": nested_ws,
        "nested_auc": float(roc_auc_score(y, lm_super_nested)),
        "nested_folds": nested_folds,
        "nested_delta_1e5_vs_lm12": [(a - b) * 1e5 for a, b in zip(nested_folds, lm_base_folds)],
    }
    log(
        f"lm_super w=0.07: AUC={lm_super['fixed_auc']:.7f} "
        f"Δfolds_e5={fmt_d1e5(lm_base_folds, fixed_folds)}"
    )
    log(
        f"lm_super nested: AUC={lm_super['nested_auc']:.7f} ws={[round(w,4) for w in nested_ws]} "
        f"Δfolds_e5={fmt_d1e5(lm_base_folds, nested_folds)}"
    )

    # Multi-LM blend: mean of pure LM bank + w * r_gpt2m
    lm_names = [n for n in ["lm12_wide", "masaya_gpt2", "gpt2_rev", "lm_gpt2_3ep"] if n in bank]
    Z_bank = np.column_stack([bank[n]["oof"] for n in lm_names])
    # equal-weight LM consensus then rank-logit
    z_lm_cons = zrk(Z_bank.mean(1))
    beta_c = float(np.cov(z_gm, z_lm_cons, ddof=0)[0, 1] / (np.var(z_lm_cons) + 1e-12))
    r_c = z_gm - beta_c * z_lm_cons
    for a, b in folds:
        rr = r_c[b]
        r_c[b] = (rr - rr.mean()) / (rr.std() + 1e-12)
    cons_folds = fold_aucs(y, z_lm_cons, folds)
    z_cons_super = z_lm_cons + 0.07 * r_c
    cons_super_folds = fold_aucs(y, z_cons_super, folds)
    multi_lm = {
        "members": lm_names,
        "consensus_auc": float(roc_auc_score(y, z_lm_cons)),
        "consensus_folds": cons_folds,
        "super_w0p07_auc": float(roc_auc_score(y, z_cons_super)),
        "super_folds": cons_super_folds,
        "super_delta_1e5_vs_cons": [(a - b) * 1e5 for a, b in zip(cons_super_folds, cons_folds)],
    }
    log(
        f"multiLM cons+0.07*r_gpt2m: AUC={multi_lm['super_w0p07_auc']:.7f} "
        f"Δfolds_e5={fmt_d1e5(cons_folds, cons_super_folds)}"
    )

    # ------------------------------------------------------------------
    # xgb_gpt2_3ep training recipe + clean replacement status
    # ------------------------------------------------------------------
    xgb_recipe = {
        "script": "scripts/post_gpt2_3ep_campaign.py::stage_b",
        "artifact": "preds/gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_{oof,test}.npy",
        "feature_matrix": (
            "X = [KMAT_TE(108 keys from gboost.KMAT), "
            "z_gpt2_3ep, z_gpt2_3ep-z_v4p, z_gpt2*r_rare, z_gpt2*r_disp, "
            "zrk(P4_L4.3/L4.2/L4.5_pure5f), optional h16, gboost.BASE]"
        ),
        "shapes_runtime": "X_o ~ (699960, 108 + 7[+16] + BASE_dim); built via import gboost",
        "base_margin": "Platt-calibrated zrk(P_super_v5_harvest) via calibrate_logit_to_prevalence",
        "hyperparams_xgb": {
            "bag": "3 seeds × depth=2 + 3 seeds × depth=3 (=6)",
            "eta": 0.025,
            "subsample": 0.8,
            "colsample_bynode": 0.35,
            "reg_lambda": 30.0,
            "min_child_weight": 200,
            "num_boost_round": 400,
            "early_stopping_rounds": 40,
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
        },
        "corruption": "gpt2_3ep F0/F1 == lm12 → z_gpt feature in X is poisoned on F0/F1",
        "gboost_stage2_ref": "scripts/gboost.py LGB/XGB shallow (max_depth=2, leaves=4) on KMAT+BASE; GEN flag for LM tokens",
    }

    # Does a clean 5f gpt2m+P4+TabM xgb already exist?
    clean_candidates = {
        "xgb_clean_decorrupt": {
            "exists": (HARV / "oof_xgb_clean.npy").exists(),
            "path": str(HARV / "oof_xgb_clean.npy"),
            "note": "de_corrupt_and_boost_5f.py Step1: LGB depth4 on clean LM block (lm12/masa/gpt2_rev) + TabM + knn24 + KMAT top30 TE — NO gpt2_3ep, NO gpt2-medium, NO P4 pure5f as explicit cols",
            "solo_auc": bank.get("xgb_clean_decorrupt", {}).get("auc"),
            "folds": bank.get("xgb_clean_decorrupt", {}).get("folds"),
        },
        "xgb_gpt2_3ep_d2_d3_bag6": {
            "exists": True,
            "path": "preds/gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy",
            "note": "ORIGINAL corrupted-input booster (uses gpt2_3ep feature)",
            "solo_auc": bank.get("xgb_gpt2_3ep", {}).get("auc"),
            "folds": bank.get("xgb_gpt2_3ep", {}).get("folds"),
        },
        "xgb_lm_super_clean_5f": {
            "exists": False,
            "note": "NOT on disk — proposed homogeneous replacement with gpt2_med_5f + P4_L4.2/3 + TabM_5f, fixing F0/F1",
        },
    }
    log(f"clean xgb_decorrupt exists={clean_candidates['xgb_clean_decorrupt']['exists']} "
        f"AUC={clean_candidates['xgb_clean_decorrupt']['solo_auc']}")

    # Per-fold comparison table vs v8 / z_ref
    compare = {
        "v8": {"auc": v8_auc, "folds": v8_folds},
        "z_ref_o": {"auc": z_ref_auc, "folds": z_ref_folds},
    }
    for n in ["xgb_gpt2_3ep", "gpt2_3ep", "lm12_wide", "gpt2_med_4ep_5f", "xgb_clean_decorrupt", "gb_gpt2_bag5"]:
        if n in bank:
            compare[n] = {"auc": bank[n]["auc"], "folds": bank[n]["folds"]}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# V8 Core Inspection & LM-Super 5F / F0–F1 GBDT Audit")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{time.time()-T0:.1f}s**.")
    md.append("")
    md.append("## 0 — Quarantine `nullspace_v3`")
    md.append("")
    md.append(
        "Moved `sub_strike10_nullspace_v3_0p15e4.csv` and `…_0p30e4.csv` → "
        f"`submissions/_quarantine_nullspace_v3/` ({q_files})."
    )
    md.append("")
    md.append(
        "Reason: `NULLSPACE_STRIKE_ORTH_V3.md` SetA (no `c13_v10_ridge`) = **−1.24e-5 / 2/5**; "
        "SetB gain **+4.14e-5** is entirely `c13_v10_ridge` = SoftPseudo/`zanch_*` family (closed)."
    )
    md.append("")
    md.append("## 1 — Exact `v8` formula")
    md.append("")
    md.append(f"- Script: `{v8_formula['script']}`")
    md.append(f"- Artifact: `{v8_formula['artifact']}` → OOF **{v8_auc:.7f}** (LB submission `{v8_formula['submission']}`)")
    md.append(f"- Backbone: `{v8_formula['backbone']}`")
    md.append(f"- OOF: `{v8_formula['formula_oof']}`")
    md.append(f"- Slot2 test: `{v8_formula['formula_test_slot2']}`")
    md.append("- KS-rect orth components vs Part12:")
    for c in v8_formula["components_KS_rect_orth_vs_w12"]:
        md.append(f"  - `{c}`")
    md.append(f"- **{v8_formula['note']}**")
    md.append(f"- Per-fold AUC: `{fmt_folds(v8_folds)}`")
    md.append("")
    md.append("## 2 — Exact `z_ref_o` formula (≈0.9526612)")
    md.append("")
    md.append(f"- Script: `{z_ref_formula['script']}`")
    md.append(f"- Formula: `{z_ref_formula['formula']}`")
    md.append(f"- Recomputed OOF **{z_ref_auc:.7f}** folds `{fmt_folds(z_ref_folds)}`")
    md.append("- U5:")
    for c in z_ref_formula["U5_orth_vs_[z_v8,z_w12,r_xgb-r_tabm]"]:
        md.append(f"  - `{c}`")
    md.append("- U2:")
    for c in z_ref_formula["U2_orth_vs_X7"]:
        md.append(f"  - `{c}`")
    md.append("")
    md.append("## 3 — Per-fold OOF AUC bank")
    md.append("")
    md.append("| model | OOF AUC | F0 | F1 | F2 | F3 | F4 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in compare.items():
        f = row["folds"]
        md.append(
            f"| {name} | {row['auc']:.7f} | {f[0]:.7f} | {f[1]:.7f} | {f[2]:.7f} | {f[3]:.7f} | {f[4]:.7f} |"
        )
    md.append("")
    md.append("### All LMs")
    md.append("")
    md.append("| LM | OOF AUC | F0 | F1 | F2 | F3 | F4 | path |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for name in [
        "lm12_wide", "lm12_raw", "masaya_gpt2", "gpt2_rev", "gpt2_3ep",
        "lm_gpt2_3ep", "gpt2_med_4ep_5f", "megayak",
    ]:
        if name not in bank:
            md.append(f"| {name} | — | | | | | | MISSING |")
            continue
        f = bank[name]["folds"]
        md.append(
            f"| {name} | {bank[name]['auc']:.7f} | {f[0]:.7f} | {f[1]:.7f} | {f[2]:.7f} | {f[3]:.7f} | {f[4]:.7f} | `{bank[name]['path']}` |"
        )
    md.append("")
    md.append("## 4 — `gpt2_3ep` F0/F1 corruption vs `lm12`")
    md.append("")
    md.append("| fold | pearson(lm12_wide) | spearman | pearson(lm12_raw) |")
    md.append("|---:|---:|---:|---:|")
    for r in corr_rows:
        raw = "—" if r["pearson_lm12_raw"] is None else f"{r['pearson_lm12_raw']:.6f}"
        md.append(
            f"| {r['fold']} | {r['pearson_lm12_wide']:.6f} | {r['spearman_lm12_wide']:.6f} | {raw} |"
        )
    md.append("")
    md.append("## 5 — 5-Fold `lm_super` law: `z_lm12 + w * r_gpt2m_5f`")
    md.append("")
    md.append(
        f"β(gpt2m∼lm12)={lm_super['beta_gpt2m_vs_lm12']:.4f}. "
        f"Residual fold-standardized; Method-B style."
    )
    md.append("")
    md.append("### Fixed `w = +0.070`")
    md.append("")
    md.append(f"- OOF AUC **{lm_super['fixed_auc']:.7f}**")
    md.append(
        f"- Δ vs lm12 ×1e-5 by fold: "
        + ", ".join(f"F{i}={d:+.2f}" for i, d in enumerate(lm_super["fixed_delta_1e5_vs_lm12"]))
    )
    md.append("")
    md.append("### Nested LOFO `w` (fit on other 4 folds)")
    md.append("")
    md.append(f"- ws = `{[round(w, 4) for w in lm_super['nested_ws']]}`")
    md.append(f"- OOF AUC **{lm_super['nested_auc']:.7f}**")
    md.append(
        f"- Δ vs lm12 ×1e-5 by fold: "
        + ", ".join(f"F{i}={d:+.2f}" for i, d in enumerate(lm_super["nested_delta_1e5_vs_lm12"]))
    )
    md.append("")
    md.append("### Multi-LM consensus + `0.07 * r_gpt2m`")
    md.append("")
    md.append(f"- Members: `{multi_lm['members']}`")
    md.append(f"- Consensus AUC **{multi_lm['consensus_auc']:.7f}** → super **{multi_lm['super_w0p07_auc']:.7f}**")
    md.append(
        f"- Δ ×1e-5 by fold: "
        + ", ".join(f"F{i}={d:+.2f}" for i, d in enumerate(multi_lm["super_delta_1e5_vs_cons"]))
    )
    md.append("")
    md.append("## 6 — `xgb_gpt2_3ep` / Stage-2 GBDT surgical upgrade status")
    md.append("")
    md.append(f"- Training script: `{xgb_recipe['script']}`")
    md.append(f"- Feature matrix: `{xgb_recipe['feature_matrix']}`")
    md.append(f"- `base_margin`: `{xgb_recipe['base_margin']}`")
    md.append("- XGB hyperparams:")
    for k, v in xgb_recipe["hyperparams_xgb"].items():
        md.append(f"  - **{k}**: `{v}`")
    md.append(f"- Corruption note: **{xgb_recipe['corruption']}**")
    md.append("")
    md.append("### Existing clean / candidate artifacts")
    md.append("")
    for name, info in clean_candidates.items():
        md.append(f"- **{name}**: exists=`{info['exists']}` — {info['note']}")
        if info.get("solo_auc") is not None:
            md.append(f"  - solo AUC={info['solo_auc']:.7f} folds={fmt_folds(info['folds'])}")
    md.append("")
    md.append(
        "**Conclusion:** No `xgb_lm_super_clean_5f` on disk yet. Closest is "
        "`preds/fast_gpu_harvest/oof_xgb_clean.npy` from `de_corrupt_and_boost_5f.py` "
        "(homogeneous LMs + TabM + knn24 + KMAT TE; **no** gpt2-medium, **no** P4 pure5f cols). "
        "A true replacement of `xgb_gpt2_3ep` should reuse the Stage-B recipe "
        "(KMAT_108 TE + P4_L4.2/3/5 + TabM_5f + **z_gpt2m_5f** instead of corrupted gpt2_3ep, "
        "same bag6 depth2/3 hist XGB, base_margin from a clean Stage-1)."
    )
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    md.append("")

    report = LOG / "V8_CORE_INSPECTION_AND_LM_SUPER_5F.md"
    report.write_text("\n".join(md))
    summary = {
        "NO_SUBMIT": NO_SUBMIT,
        "quarantined": q_files,
        "v8": {"auc": v8_auc, "folds": v8_folds, "formula": v8_formula},
        "z_ref": {"auc": z_ref_auc, "folds": z_ref_folds, "formula": z_ref_formula},
        "bank": {k: {"auc": v["auc"], "folds": v["folds"], "path": v["path"]} for k, v in bank.items()},
        "corruption": corr_rows,
        "lm_super": lm_super,
        "multi_lm": multi_lm,
        "xgb_recipe": xgb_recipe,
        "clean_candidates": clean_candidates,
        "elapsed_s": round(time.time() - T0, 1),
    }
    # json-serialize
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (LOG / "V8_CORE_INSPECTION_AND_LM_SUPER_5F.json").write_text(
        json.dumps(summary, indent=2, default=conv)
    )
    log(f"Wrote {report}")
    log("V8_CORE_INSPECTION_COMPLETE")


if __name__ == "__main__":
    main()
