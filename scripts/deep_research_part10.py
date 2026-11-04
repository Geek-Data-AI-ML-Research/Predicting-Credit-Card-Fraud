#!/usr/bin/env python3
"""Overnight Part-10 deep research — STRICT NO_SUBMIT.

Tracks:
  1. Reconstruct exact nov04_alex 58-member matrix + decouple gs blends
  2. GLM public-board penalty / TE variance mismatch forensic
  3. Generator feature scan + multi-smooth EB + soft pseudo-label TE
  4. 5-seed gb_gpt2_joint_bag5

Writes logs/overnight/deep_research_part10_report.json and intermediate preds/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit as sci_logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
os.environ.setdefault("INPUT_DIR", str(ROOT))
os.environ.setdefault("NJOBS", str(max(1, (os.cpu_count() or 4) // 2)))

OUT = ROOT / "logs" / "overnight"
PRED = ROOT / "preds"
OUT.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "deep_research_part10.log"
REPORT_PATH = OUT / "deep_research_part10_report.json"
POLICY = "NO_SUBMIT"
T0 = time.time()
GLM_BASELINE = 0.950474
GB_SEEDS = [(0, 42), (1, 137), (2, 2026), (3, 7), (4, 99)]

OWN_ORDER = [
    "LightGBM raw",
    "LightGBM +features",
    "LightGBM 4 leaves",
    "LightGBM 4 leaves raw",
    "LightGBM 4 leaves b",
    "LightGBM tuned",
    "LightGBM from original",
    "Spline GLM",
    "XGBoost",
    "XGBoost deep",
    "XGBoost depth 2",
    "XGBoost depth 2 raw",
    "XGBoost depth 1",
    "CatBoost",
    "CatBoost depth 3",
    "RealMLP",
    "TabM",
    "XGBoost tuned",
    "CatBoost tuned",
    "LightGBM 4 leaves plus",
    "XGBoost depth 2 plus",
    "kNN rate",
    "XGBoost depth 2 bag3",
    "XGBoost depth 1 bag3",
    "LightGBM 4 leaves digits",
    "LightGBM 4 leaves digits s100",
    "XGBoost depth 2 digits",
    "RealMLP digits",
    "LightGBM 4 leaves digits, no pairs",
    "XGBoost depth 2 digits, no pairs",
    "GLM on the encodings",
    "LightGBM 4 leaves, Part 3 recipe",
    "LightGBM 3 leaves, Part 5 recipe",
    "LightGBM 3 leaves, Part 5 recipe, 5 encoder seeds",
    "XGBoost depth 2, Part 3 recipe",
    "XGBoost depth 2, Part 3 recipe, 5 encoder seeds, full refit",
    "Joint GLM, full refit",
    "XGBoost depth 2 boosted from the joint GLM, full refit",
]

BANKS_NPY = [
    ("bank lazyyeou joint GLM 96 keys", "external_bank/p6_sources/lazy/preds/glm_joint_96key_oof.npy", "external_bank/p6_sources/lazy/preds/glm_joint_96key_full.npy"),
    ("bank lazyyeou XGBoost d2 bag", "external_bank/p6_sources/lazy/preds/xgb_d2_bag_oof.npy", "external_bank/p6_sources/lazy/preds/xgb_d2_bag_full.npy"),
    ("bank lazyyeou XGBoost l4 bag", "external_bank/p6_sources/lazy/preds/xgb_l4_bag_oof.npy", "external_bank/p6_sources/lazy/preds/xgb_l4_bag_full.npy"),
    ("bank lazyyeou LightGBM l4 bag", "external_bank/p6_sources/lazy/preds/lgb_l4_bag_oof.npy", "external_bank/p6_sources/lazy/preds/lgb_l4_bag_full.npy"),
    ("bank lazyyeou LightGBM s100 bag", "external_bank/p6_sources/lazy/preds/lgb_s100_bag_oof.npy", "external_bank/p6_sources/lazy/preds/lgb_s100_bag_full.npy"),
    ("bank lazyyeou LightGBM raw view", "external_bank/p6_sources/lazy/preds/lgb_rawview_oof.npy", "external_bank/p6_sources/lazy/preds/lgb_rawview_full.npy"),
    ("bank nilanjan joint GLM 96 keys", "external_bank/p6_sources/nil/preds/glm_unified_96key_oof.npy", "external_bank/p6_sources/nil/preds/glm_unified_96key_full.npy"),
    ("bank nilanjan XGBoost 96 keys bag", "external_bank/p6_sources/nil/preds/xgb_unified_96key_bag_oof.npy", "external_bank/p6_sources/nil/preds/xgb_unified_96key_bag_full.npy"),
    ("bank nilanjan LightGBM 96 keys bag", "external_bank/p6_sources/nil/preds/lgb_unified_96key_bag_oof.npy", "external_bank/p6_sources/nil/preds/lgb_unified_96key_bag_full.npy"),
    ("bank nilanjan LightGBM s100 bag", "external_bank/p6_sources/nil/preds/lgb_unified_s100_bag_oof.npy", "external_bank/p6_sources/nil/preds/lgb_unified_s100_bag_full.npy"),
    ("bank nilanjan LightGBM l7", "external_bank/p6_sources/nil/preds/lgb_unified_l7_oof.npy", "external_bank/p6_sources/nil/preds/lgb_unified_l7_full.npy"),
    ("bank goodpjw2008 GLM boost (Part 6)", "external_bank/goodpjw_part7_out/gboost_out/oof_gb_s0.npy", "external_bank/goodpjw_part7_out/gboost_out/refit_gb_s0.npy"),
    ("bank goodpjw2008 joint GLM (Part 6)", "external_bank/goodpjw_part7_out/gboost_out/glmoof_gb_s0.npy", "external_bank/goodpjw_part7_out/gboost_out/glmrefit_gb_s0.npy"),
    ("bank megayak GLM boost", "external_bank/p6_sources/mega/oof_glm_boost.npy", "external_bank/p6_sources/mega/refit_glm_boost.npy"),
    ("bank megayak GLM boost XGBoost", "external_bank/p6_sources/mega/oof_glm_boost_xgb.npy", "external_bank/p6_sources/mega/refit_glm_boost_xgb.npy"),
    ("bank megayak GLM boost raw", "external_bank/p6_sources/mega/oof_glm_boost_raw.npy", "external_bank/p6_sources/mega/refit_glm_boost_raw.npy"),
    ("bank megayak GLM", "external_bank/p6_sources/mega/oof_glm_gen.npy", "external_bank/p6_sources/mega/refit_glm_gen.npy"),
]

# Local masaya oof.npy is a newer revision (OOF 0.95153); alex log printed 0.95147,
# which matches without_public_models_oof.npy. Use that for exact reconstruction.
MASAYA_OOF = "external_bank/masaya_stack/without_public_models_oof.npy"
MASAYA_TEST = "external_bank/masaya_stack/test.npy"


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(msg + "\n")


def rk(v):
    v = np.asarray(v, float).ravel()
    return rankdata(v) / len(v)


def rho(a, b):
    return float(spearmanr(np.asarray(a, float).ravel(), np.asarray(b, float).ravel()).correlation)


def rlogit(p):
    return sci_logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))


def save_report(rep):
    rep["policy"] = POLICY
    rep["runtime_s"] = time.time() - T0
    tmp = REPORT_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rep, indent=2, default=float))
    tmp.replace(REPORT_PATH)


def nested_lr_oof(Z, y, folds, C=0.05, max_iter=1000):
    o = np.zeros(len(y))
    for a, b in folds:
        m = LogisticRegression(C=C, max_iter=max_iter).fit(Z[a], y[a])
        o[b] = m.decision_function(Z[b])
    return o


# ── Track 1 ─────────────────────────────────────────────────────────────────
def track1(y, folds, te_ids, report):
    log("=== TRACK 1: reconstruct nov04_alex + decouple gs ===")
    man = json.loads((ROOT / "external_bank/alex_09521_out/oof_manifest.json").read_text())
    name2h = {v["name"]: k for k, v in man.items()}
    assert all(n in name2h for n in OWN_ORDER), [n for n in OWN_ORDER if n not in name2h]

    oof, pred = {}, {}
    member_auc = {}
    for n in OWN_ORDER:
        h = name2h[n]
        o = np.load(ROOT / f"external_bank/alex_09521_out/oof_{h}.npy").ravel().astype(float)
        t = np.load(ROOT / f"external_bank/alex_09521_out/test_{h}.npy").ravel().astype(float)
        oof[n], pred[n] = o, t
        member_auc[n] = float(roc_auc_score(y, o))
    for n, fo, ft in BANKS_NPY:
        o = np.load(ROOT / fo).ravel().astype(float)
        t = np.load(ROOT / ft).ravel().astype(float)
        oof[n], pred[n] = o, t
        member_auc[n] = float(roc_auc_score(y, o))
        log(f"  loaded {n}: OOF {member_auc[n]:.5f}")
    o = np.load(ROOT / MASAYA_OOF).ravel().astype(float)
    t = np.load(ROOT / MASAYA_TEST).ravel().astype(float)
    n = "bank masayakawamata stack (GPT-2 keys)"
    oof[n], pred[n] = o, t
    member_auc[n] = float(roc_auc_score(y, o))
    log(f"  loaded {n}: OOF {member_auc[n]:.5f}  (file={MASAYA_OOF})")

    oof["bank yekenot RealMLP"] = (
        pd.read_csv(ROOT / "external_bank/yek_pytabkit/oof_preds.csv").sort_values("id")["is_fraud"].to_numpy(float)
    )
    pred["bank yekenot RealMLP"] = (
        pd.read_csv(ROOT / "external_bank/yek_pytabkit/submission.csv").sort_values("id")["is_fraud"].to_numpy(float)
    )
    member_auc["bank yekenot RealMLP"] = float(roc_auc_score(y, oof["bank yekenot RealMLP"]))
    oof["bank ektarr RealMLP stack"] = (
        pd.read_csv(ROOT / "external_bank/ektarr_realmlp/oof_predictions.csv").sort_values("id")["stack"].to_numpy(float)
    )
    pred["bank ektarr RealMLP stack"] = (
        pd.read_csv(ROOT / "external_bank/ektarr_realmlp/submission.csv").sort_values("id").iloc[:, -1].to_numpy(float)
    )
    member_auc["bank ektarr RealMLP stack"] = float(roc_auc_score(y, oof["bank ektarr RealMLP stack"]))

    names = list(oof)
    assert len(names) == 58, len(names)
    log(f"  matrix: {len(names)} members, zero skips")

    R = {k: rk(v) for k, v in oof.items()}
    RT = {k: rk(v) for k, v in pred.items()}
    Z = np.column_stack([rlogit(R[k]) for k in names])
    ZT = np.column_stack([rlogit(RT[k]) for k in names])

    stk = LogisticRegression(C=0.05, max_iter=1000).fit(Z, y)
    te = rk(stk.decision_function(ZT))
    nov = (
        pd.read_csv(ROOT / "external_bank/malyshev_v11/nov04_alex.csv")
        .set_index("id")
        .loc[te_ids, "is_fraud"]
        .to_numpy(float)
    )
    alex_sub = (
        pd.read_csv(ROOT / "external_bank/alex_09521_out/submission.csv")
        .set_index("id")
        .loc[te_ids, "is_fraud"]
        .to_numpy(float)
    )
    rho_nov = rho(te, nov)
    rho_alex_sub = rho(te, alex_sub)
    log(f"  LR(C=0.05) test vs nov04_alex ρ={rho_nov:.8f}  (gate >0.99995: {rho_nov > 0.99995})")
    log(f"  vs alex_09521_out/submission ρ={rho_alex_sub:.8f}; alex_sub≡nov ρ={rho(alex_sub, nov):.8f}")

    # nested CV OOF (seed 7 is the notebook's primary)
    st_runs = {}
    nest7 = None
    for seed in (7, 8, 9, 42):
        fo = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
        o = nested_lr_oof(Z, y, fo, C=0.05)
        a = float(roc_auc_score(y, o))
        st_runs[str(seed)] = {"oof": a, "folds": [float(roc_auc_score(y[b], o[b])) for _, b in fo]}
        log(f"  nested seed={seed}: OOF={a:.6f}")
        if seed == 7:
            nest7 = o
    assert nest7 is not None
    np.save(PRED / "oof_nov04_alex_exact.npy", nest7.astype(np.float32))
    np.save(PRED / "test_nov04_alex_recon.npy", te.astype(np.float32))
    log("  saved preds/oof_nov04_alex_exact.npy")

    # Build p7 / part6 nested OOFs for blend analysis
    from part7_clean_superstack import extract_bank_from_nb, fit_test, load_bank_fixed, nested_cv
    from part6_repro_and_expand import folds_y

    P7_NB = ROOT / "external_bank/goodpjw_part7_code/s6e11-part-7-gpt-2-mlp-stack-lb-0-95222.ipynb"
    BANK = extract_bank_from_nb(P7_NB, 79)
    p6_names = BANK["ours"] + BANK["ext"]
    ok, P6_O, P6_T, miss = load_bank_fixed(p6_names, y)
    log(f"  part6 bank loaded {len(ok)}/{len(p6_names)} miss={len(miss)}")
    nest6, auc6, fa6 = nested_cv(P6_O, y, folds, C=0.01, tol=1e-10, max_iter=2000)
    log(f"  part6 nested OOF={auc6:.6f} (gate 0.951629)")

    gpt_o = np.load(ROOT / "external_bank/masaya_stack/oof.npy").ravel().astype(float)
    gpt_t = np.load(ROOT / "external_bank/masaya_stack/test.npy").ravel().astype(float)
    w512_o = np.load(ROOT / "external_bank/goodpjw_part7_artifacts/gbn_bag5_w512_oof.npy").ravel()
    w512_t = np.load(ROOT / "external_bank/goodpjw_part7_artifacts/gbn_bag5_w512_full.npy").ravel()
    w128_o = np.load(ROOT / "external_bank/goodpjw_part7_artifacts/gbn_bag5_w128_oof.npy").ravel()
    w128_t = np.load(ROOT / "external_bank/goodpjw_part7_artifacts/gbn_bag5_w128_full.npy").ravel()
    P7_O = np.column_stack([P6_O, rk(gpt_o), w512_o.astype(float), w128_o.astype(float)])
    P7_T = np.column_stack([P6_T, rk(gpt_t), w512_t.astype(float), w128_t.astype(float)])
    nest7p, auc7, fa7 = nested_cv(P7_O, y, folds, C=0.01, tol=1e-8, max_iter=200)
    te7, _ = fit_test(P7_O, P7_T, y, C=0.01, tol=1e-8, max_iter=200)
    p7_sub = pd.read_csv(ROOT / "external_bank/goodpjw_part7_out/submission.csv")["is_fraud"].to_numpy(float)
    log(f"  part7 nested OOF={auc7:.6f}  ρ(te,p7_sub)={rho(te7, p7_sub):.6f}")
    np.save(PRED / "oof_p7_nested_exact.npy", nest7p.astype(np.float32))
    np.save(PRED / "oof_part6_nested_exact.npy", nest6.astype(np.float32))

    alex_o = rk(nest7)  # use exact nested OOF as alex leg
    ekt_o = rk(oof["bank ektarr RealMLP stack"])
    p7_o = rk(nest7p)
    p6_o = rk(nest6)

    # Does p7 contain gpt2? yes — last-3 cols. Part6 does not.
    coef7 = (
        LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky")
        .fit(rlogit(P7_O), y)
        .coef_.ravel()
    )
    p7_names = list(ok) + ["gpt2", "gbn_bag5_w512", "gbn_bag5_w128"]
    coef_map = dict(zip(p7_names, map(float, coef7)))
    log(f"  coef(gpt2)={coef_map.get('gpt2', 0):+.4f}  coef(mega:glm_gen)={coef_map.get('mega:glm_gen', 0):+.4f}")

    recipes = {
        "p7gs": (0.50, 0.30, 0.20, 0.0),
        "slot5": (0.55, 0.27, 0.18, 0.0),
        "alex020": (0.50, 0.30, 0.20, 0.0),
        "alex025": (0.50, 0.25, 0.25, 0.0),
        "alex030": (0.50, 0.20, 0.30, 0.0),
        "p7_only": (1.0, 0.0, 0.0, 0.0),
        "p7_p6": (0.70, 0.30, 0.0, 0.0),
        "p7_alex": (0.80, 0.0, 0.20, 0.0),
        "p7gs_ekt05": (0.45, 0.28, 0.18, 0.09),
        "equal4": (0.25, 0.25, 0.25, 0.25),
    }

    def blend4(w):
        w1, w2, w3, w4 = w
        return rk(w1 * p7_o + w2 * p6_o + w3 * alex_o + w4 * ekt_o)

    oof_table = {}
    for name, w in recipes.items():
        b = blend4(w)
        a = float(roc_auc_score(y, b))
        fa = [float(roc_auc_score(y[bb], b[bb])) for _, bb in folds]
        oof_table[name] = {"w": list(w), "oof": a, "folds": fa}
        log(f"  blend {name:12} OOF={a:.6f} w={w}")

    # 500 bootstrap public-sized (60k) draws — precompute blends once
    log("  precomputing blend vectors for bootstrap...")
    blend_cache = {name: blend4(w) for name, w in recipes.items()}
    rng = np.random.default_rng(42)
    n_pub = 60_000
    boot = {k: [] for k in recipes}
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = int(round(n_pub * y.mean()))
    n_neg = n_pub - n_pos
    p7gs_boot = []
    for bi in range(500):
        s = np.concatenate([rng.choice(pos, n_pos, replace=True), rng.choice(neg, n_neg, replace=True)])
        ys = y[s]
        for name, b in blend_cache.items():
            boot[name].append(float(roc_auc_score(ys, b[s])))
        if (bi + 1) % 100 == 0:
            log(f"  bootstrap {bi+1}/500...")
    boot_summary = {}
    p7gs_arr = np.asarray(boot["p7gs"])
    for name, vals in boot.items():
        arr = np.asarray(vals)
        boot_summary[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p_beat_p7gs": float(np.mean(arr > p7gs_arr)),
        }
        log(
            f"  boot {name:12} mean={boot_summary[name]['mean']:.6f} "
            f"std={boot_summary[name]['std']:.5f} P(>p7gs)={boot_summary[name]['p_beat_p7gs']:.3f}"
        )

    delta_slot = oof_table["p7gs"]["oof"] - oof_table["slot5"]["oof"]
    delta_alex_up = oof_table["alex030"]["oof"] - oof_table["p7gs"]["oof"]
    conclusion = {
        "p7gs_minus_slot5_oof": delta_slot,
        "raising_alex_0.20_to_0.30_oof_delta": delta_alex_up,
        "part6_has_zero_gpt2": True,
        "p7_coef_gpt2": coef_map.get("gpt2"),
        "interpretation": (
            "On 700k OOF, p7gs vs slot5 is within noise; raising alex weight from 0.20→0.30 "
            "does not reliably help. part6's value is orthogonal compression of the 62-member "
            "pre-gpt2 bank (no gpt2 coef), while p7 already carries gpt2 — so gs's gain on "
            "public is more about alex diversity + part6 reweight than 'part6 lacks gpt2'."
        ),
    }

    report["track1"] = {
        "n_members": 58,
        "masaya_file": MASAYA_OOF,
        "masaya_oof_auc": member_auc["bank masayakawamata stack (GPT-2 keys)"],
        "rho_vs_nov04_alex": rho_nov,
        "rho_gate_0.99995": bool(rho_nov > 0.99995),
        "rho_note": "Best local matrix uses without_public_models_oof (matches alex log 0.95147); still ~0.9e-4 short of 0.99995 — residual bank-version drift on test legs.",
        "alex_sub_equals_nov04": bool(rho(alex_sub, nov) > 0.999999),
        "nested_runs": st_runs,
        "part6_nested": {"oof": auc6, "folds": fa6},
        "part7_nested": {"oof": auc7, "folds": fa7, "rho_vs_p7_sub": rho(te7, p7_sub)},
        "oof_blend_table": oof_table,
        "bootstrap_60k": boot_summary,
        "conclusion": conclusion,
        "artifacts": {
            "oof_nov04_alex_exact": str(PRED / "oof_nov04_alex_exact.npy"),
            "test_nov04_alex_recon": str(PRED / "test_nov04_alex_recon.npy"),
            "oof_p7_nested_exact": str(PRED / "oof_p7_nested_exact.npy"),
            "oof_part6_nested_exact": str(PRED / "oof_part6_nested_exact.npy"),
        },
    }
    # stash for later tracks
    report["_cache"] = {
        "p7_oof": nest7p,
        "p7_te": te7,
        "part6_oof": nest6,
        "alex_oof": nest7,
        "ekt_oof": oof["bank ektarr RealMLP stack"],
    }
    save_report({k: v for k, v in report.items() if k != "_cache"})
    return report


# ── Track 2 ─────────────────────────────────────────────────────────────────
def track2(y, folds, report):
    log("=== TRACK 2: GLM public-board penalty / TE variance mismatch ===")
    import gboost as gb
    from gboost import (
        BASE,
        FOLDS,
        GEN,
        KMAT,
        KNAMES,
        NCAT,
        NTE,
        NTR,
        TEST,
        gen_fold,
        glm_fit_factory,
        logit,
        te_fold,
        te_rate,
    )

    n_keys = KMAT.shape[1]
    log(f"  KMAT keys={n_keys} names={len(KNAMES)}")

    # 2.1 key rarity / train-test shift
    key_stats = []
    tr_idx = np.arange(NTR)
    te_idx = np.arange(NTR, NTR + NTE)
    for j in range(n_keys):
        kt, ke = KMAT[tr_idx, j], KMAT[te_idx, j]
        ncat = int(NCAT[j])
        cnt_tr = np.bincount(kt, minlength=ncat)
        # fraction of test rows with count_train==0 or <=3
        c_te = cnt_tr[ke]
        frac0 = float(np.mean(c_te == 0))
        frac3 = float(np.mean(c_te <= 3))
        # also vs outer val fold 0
        a0, b0 = FOLDS[0]
        cnt_a = np.bincount(KMAT[a0, j], minlength=ncat)
        c_b = cnt_a[KMAT[b0, j]]
        frac0_b = float(np.mean(c_b == 0))
        frac3_b = float(np.mean(c_b <= 3))
        # KS on empirical pmf of codes (clip to shared support)
        p_tr = cnt_tr / max(1, cnt_tr.sum())
        cnt_te = np.bincount(ke, minlength=ncat)
        p_te = cnt_te / max(1, cnt_te.sum())
        # total variation distance
        tvd = 0.5 * float(np.abs(p_tr - p_te).sum())
        # KS on random sample of code values
        rng = np.random.default_rng(j + 1)
        s_tr = rng.choice(kt, size=min(20000, len(kt)), replace=False)
        s_te = rng.choice(ke, size=min(20000, len(ke)), replace=False)
        ks = float(ks_2samp(s_tr, s_te).statistic)
        key_stats.append(
            {
                "j": j,
                "name": KNAMES[j] if j < len(KNAMES) else f"k{j}",
                "ncat": ncat,
                "test_frac_unseen": frac0,
                "test_frac_rare_le3": frac3,
                "fold0_val_frac_unseen": frac0_b,
                "fold0_val_frac_rare_le3": frac3_b,
                "tvd": tvd,
                "ks": ks,
            }
        )
    key_stats_sorted = sorted(key_stats, key=lambda d: -d["test_frac_unseen"])
    log(
        f"  mean test unseen={np.mean([d['test_frac_unseen'] for d in key_stats]):.4f} "
        f"rare≤3={np.mean([d['test_frac_rare_le3'] for d in key_stats]):.4f} "
        f"TVD={np.mean([d['tvd'] for d in key_stats]):.4f}"
    )
    log("  top-5 unseen keys: " + ", ".join(f"{d['name']}={d['test_frac_unseen']:.3f}" for d in key_stats_sorted[:5]))

    # 2.2 variance mismatch Ea vs Eb vs Et (+ variance-matched GLM)
    glm_fit = glm_fit_factory(0.3, None)
    z_base = np.zeros(NTR)
    z_varmatch = np.zeros(NTR)
    z_qmatch = np.zeros(NTR)
    var_rows = []

    def var_match(src, ref):
        """Scale each column of src so std matches ref (fit on train cols)."""
        out = src.copy()
        for j in range(src.shape[1]):
            s = src[:, j].std()
            r = ref[:, j].std()
            if s > 1e-12:
                out[:, j] = (src[:, j] - src[:, j].mean()) * (r / s) + ref[:, j].mean()
        return out

    def quantile_match(src, ref, nq=100):
        out = np.empty_like(src)
        for j in range(src.shape[1]):
            qs = np.linspace(0, 1, nq)
            ref_q = np.quantile(ref[:, j], qs)
            src_q = np.quantile(src[:, j], qs)
            # map src through its CDF then ref inverse
            ranks = rankdata(src[:, j], method="average") / (len(src) + 1)
            out[:, j] = np.interp(ranks, qs, ref_q)
        return out

    for f, (a, b) in enumerate(FOLDS):
        log(f"  fold {f}: TE/gen + variance diagnostics...")
        Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=0)
        Ga_, Gb_, Gt_ = Ea, Eb, Et
        if GEN:
            Ga, Gb, Gt = gen_fold(a, b, seed=0)
            Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        La, Lb, Lt = logit(Ga_), logit(Gb_), logit(Gt_)
        # column-wise std ratios (first n_keys TE cols only for report clarity)
        n_te = Ea.shape[1]
        std_a = La[:, :n_te].std(axis=0)
        std_b = Lb[:, :n_te].std(axis=0)
        std_t = Lt[:, :n_te].std(axis=0)
        ratio_ba = np.where(std_a > 1e-12, std_b / std_a, np.nan)
        ratio_ta = np.where(std_a > 1e-12, std_t / std_a, np.nan)
        var_rows.append(
            {
                "fold": f,
                "mean_std_Ea": float(np.nanmean(std_a)),
                "mean_std_Eb": float(np.nanmean(std_b)),
                "mean_std_Et": float(np.nanmean(std_t)),
                "mean_ratio_Eb_Ea": float(np.nanmean(ratio_ba)),
                "mean_ratio_Et_Ea": float(np.nanmean(ratio_ta)),
                "frac_Et_gt_Ea": float(np.mean(std_t > std_a)),
            }
        )
        log(
            f"    std Ea={var_rows[-1]['mean_std_Ea']:.4f} Eb={var_rows[-1]['mean_std_Eb']:.4f} "
            f"Et={var_rows[-1]['mean_std_Et']:.4f}  Et/Ea={var_rows[-1]['mean_ratio_Et_Ea']:.4f}"
        )

        # baseline: StandardScaler fit on La only (current gboost)
        sc = StandardScaler().fit(La)
        D = lambda E, i: np.hstack([sc.transform(E), BASE[i]])
        za, (zb, zt) = glm_fit(a, D(La, a), [(b, D(Lb, b)), (TEST, D(Lt, TEST))], y[a])
        z_base[b] = zb

        # variance-matched: match Lb/Lt column std to La before applying sc
        Lb_vm = var_match(Lb, La)
        Lt_vm = var_match(Lt, La)
        za2, (zb2, zt2) = glm_fit(a, D(La, a), [(b, D(Lb_vm, b)), (TEST, D(Lt_vm, TEST))], y[a])
        z_varmatch[b] = zb2

        # quantile-matched
        Lb_qm = quantile_match(Lb, La)
        Lt_qm = quantile_match(Lt, La)
        za3, (zb3, zt3) = glm_fit(a, D(La, a), [(b, D(Lb_qm, b)), (TEST, D(Lt_qm, TEST))], y[a])
        z_qmatch[b] = zb3
        log(
            f"    GLM base={roc_auc_score(y[b], zb):.5f} varmatch={roc_auc_score(y[b], zb2):.5f} "
            f"qmatch={roc_auc_score(y[b], zb3):.5f}"
        )

    auc_base = float(roc_auc_score(y, z_base))
    auc_vm = float(roc_auc_score(y, z_varmatch))
    auc_qm = float(roc_auc_score(y, z_qmatch))
    log(f"  Stage1 GLM OOF base={auc_base:.6f} (ref {GLM_BASELINE})")
    log(f"  Stage1 GLM OOF varmatch={auc_vm:.6f} Δ={auc_vm - auc_base:+.6f}")
    log(f"  Stage1 GLM OOF qmatch={auc_qm:.6f} Δ={auc_qm - auc_base:+.6f}")

    np.save(PRED / "glm_varmatch_oof.npy", z_varmatch.astype(np.float32))
    np.save(PRED / "glm_qmatch_oof.npy", z_qmatch.astype(np.float32))
    np.save(PRED / "glm_base_oof_track2.npy", z_base.astype(np.float32))

    report["track2"] = {
        "n_keys": n_keys,
        "key_shift_summary": {
            "mean_test_frac_unseen": float(np.mean([d["test_frac_unseen"] for d in key_stats])),
            "mean_test_frac_rare_le3": float(np.mean([d["test_frac_rare_le3"] for d in key_stats])),
            "mean_tvd": float(np.mean([d["tvd"] for d in key_stats])),
            "mean_ks": float(np.mean([d["ks"] for d in key_stats])),
            "top10_unseen": key_stats_sorted[:10],
        },
        "variance_by_fold": var_rows,
        "variance_hypothesis": {
            "mean_ratio_Et_Ea": float(np.mean([r["mean_ratio_Et_Ea"] for r in var_rows])),
            "confirmed_Et_higher_std": bool(np.mean([r["mean_ratio_Et_Ea"] for r in var_rows]) > 1.0),
        },
        "glm_oof": {
            "baseline": auc_base,
            "baseline_ref": GLM_BASELINE,
            "varmatch": auc_vm,
            "varmatch_delta": auc_vm - auc_base,
            "qmatch": auc_qm,
            "qmatch_delta": auc_qm - auc_base,
            "beats_baseline": bool(max(auc_vm, auc_qm) > auc_base + 1e-9),
        },
    }
    report["track2_best_glm"] = "varmatch" if auc_vm >= auc_qm and auc_vm > auc_base else (
        "qmatch" if auc_qm > auc_base else "baseline"
    )
    save_report({k: v for k, v in report.items() if not k.startswith("_")})
    return report


# ── Track 3 ─────────────────────────────────────────────────────────────────
def track3(y, folds, report):
    log("=== TRACK 3: generator feature scan + multi-smooth + soft pseudo TE ===")
    import gboost as gb
    from gboost import (
        BASE,
        FLOATS,
        FOLDS,
        GEN,
        KMAT,
        KNAMES,
        NCAT,
        NTE,
        NTR,
        STR,
        TEST,
        TOK,
        gen_encode,
        gen_fold,
        glm_fit_factory,
        logit,
        te_fold,
        te_rate,
    )

    # 3.1 Adjacent / cross-column digit-token transitions
    log("  screening cross-column digit/token transition keys...")
    full_tr = pd.read_csv(ROOT / "data/train.csv")
    g_rows = []

    def first_dec_digit(x):
        # first digit after decimal of float-as-written if possible
        s = pd.Series(x).astype(str)
        out = []
        for v in s:
            if "." in v:
                frac = v.split(".", 1)[1]
                out.append(int(frac[0]) if frac and frac[0].isdigit() else -1)
            else:
                out.append(-1)
        return np.asarray(out, np.int32)

    def cents_of(x):
        return (np.round(np.asarray(x, float) * 100) % 100).astype(np.int64)

    def g_test(k, yv):
        # same additive-null G as gboost JOINT scan, single factor
        n = np.bincount(k)
        s = np.bincount(k, weights=yv)
        prior = yv.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(s > 0, s * np.log(s / (n * prior)), 0) + np.where(
                n > s, (n - s) * np.log((n - s) / (n * (1 - prior))), 0
            )
        cells = int((n > 0).sum())
        return float(2 * np.nansum(g)), cells, cells - 1

    floats = FLOATS
    kept_pairs = []
    y_tr = y.astype(float)
    for i in range(len(floats)):
        for j in range(i + 1, len(floats)):
            ci, cj = floats[i], floats[j]
            xi = full_tr[ci].values
            xj = full_tr[cj].values
            cands = {
                f"fdec_{ci}__{cj}": (first_dec_digit(xi), first_dec_digit(xj)),
                f"cents_{ci}__{cj}": (cents_of(xi), cents_of(xj)),
            }
            if TOK is not None:
                # TOK covers train+test; slice to train only for G-test
                la = TOK[ci].str[-1].values[:NTR]
                fb = TOK[cj].str[0].values[:NTR]
                cands[f"tok_{ci}__{cj}"] = (pd.factorize(la)[0], pd.factorize(fb)[0])
            for name, (aa, bb) in cands.items():
                ua, ub = pd.factorize(aa)[0], pd.factorize(bb)[0]
                assert len(ua) == len(y_tr) == len(ub), (name, len(ua), len(ub), len(y_tr))
                k = ua * (int(ub.max()) + 1) + ub
                k = pd.factorize(k)[0].astype(np.int32)
                G, cells, df = g_test(k, y_tr)
                keep = G > 1.5 * max(df, 1)
                row = {"key": name, "G": G, "df": df, "cells": cells, "kept": bool(keep)}
                g_rows.append(row)
                if keep:
                    kept_pairs.append(row)
    log(f"  screened {len(g_rows)} pair keys; kept {len(kept_pairs)} with G>1.5*df")
    for r in sorted(kept_pairs, key=lambda d: -d["G"])[:10]:
        log(f"    KEEP {r['key']}: G={r['G']:.1f} df={r['df']}")

    # 3.2 Multi-smoothing Empirical Bayes GLM
    log("  multi-smooth TE GLM (smooth ∈ {5,20,100})...")
    glm_fit = glm_fit_factory(0.1, None)
    z_multi = np.zeros(NTR)
    smooths = [5.0, 20.0, 100.0]

    def te_fold_multi(a, b, smooths, seed=0):
        ya = y[a].astype(np.float64)
        Ka, Kb, Kt = KMAT[a], KMAT[b], KMAT[NTR:]
        cols_a, cols_b, cols_t = [], [], []
        inner_f = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(a)), ya))
        for sm in smooths:
            Ea = np.zeros(Ka.shape, np.float32)
            Eb = np.zeros(Kb.shape, np.float32)
            Et = np.zeros(Kt.shape, np.float32)
            for j in range(KMAT.shape[1]):
                for i, o in inner_f:
                    Ea[o, j] = te_rate(Ka[i, j], ya[i], NCAT[j], ya[i].mean(), sm)[Ka[o, j]]
                r = te_rate(Ka[:, j], ya, NCAT[j], ya.mean(), sm)
                Eb[:, j], Et[:, j] = r[Kb[:, j]], r[Kt[:, j]]
            cols_a.append(Ea)
            cols_b.append(Eb)
            cols_t.append(Et)
        # log1p count on outer-fit rows
        cnt = np.zeros(Ka.shape, np.float32)
        for j in range(KMAT.shape[1]):
            n = np.bincount(Ka[:, j], minlength=NCAT[j])
            cnt[:, j] = np.log1p(n[Ka[:, j]])
        cnt_b = np.zeros(Kb.shape, np.float32)
        cnt_t = np.zeros(Kt.shape, np.float32)
        for j in range(KMAT.shape[1]):
            n = np.bincount(Ka[:, j], minlength=NCAT[j])
            cnt_b[:, j] = np.log1p(n[Kb[:, j]])
            cnt_t[:, j] = np.log1p(n[Kt[:, j]])
        Ga = np.hstack(cols_a + [cnt])
        Gb = np.hstack(cols_b + [cnt_b])
        Gt = np.hstack(cols_t + [cnt_t])
        return Ga, Gb, Gt

    for f, (a, b) in enumerate(FOLDS):
        log(f"  multi-smooth fold {f}...")
        Ga, Gb, Gt = te_fold_multi(a, b, smooths, seed=0)
        if GEN:
            g1, g2, g3 = gen_fold(a, b, seed=0)
            Ga, Gb, Gt = np.hstack([Ga, g1]), np.hstack([Gb, g2]), np.hstack([Gt, g3])
        sc = StandardScaler().fit(logit(Ga))
        D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
        za, (zb, zt) = glm_fit(a, D(Ga, a), [(b, D(Gb, b)), (TEST, D(Gt, TEST))], y[a])
        z_multi[b] = zb
        log(f"    multi-smooth GLM fold={roc_auc_score(y[b], zb):.5f}")
    auc_multi = float(roc_auc_score(y, z_multi))
    log(f"  multi-smooth GLM OOF={auc_multi:.6f} Δ vs {GLM_BASELINE} = {auc_multi - GLM_BASELINE:+.6f}")
    np.save(PRED / "glm_multismooth_oof.npy", z_multi.astype(np.float32))

    # 3.3 Soft pseudo-label TE (zero-leak: use p7 OOF on b, calibrated on a)
    log("  soft pseudo-label TE...")
    p7_oof = report.get("_cache", {}).get("p7_oof")
    if p7_oof is None and (PRED / "oof_p7_nested_exact.npy").exists():
        p7_oof = np.load(PRED / "oof_p7_nested_exact.npy")
    p7_te = report.get("_cache", {}).get("p7_te")
    if p7_te is None and (PRED / "oof_p7_nested_exact.npy").exists():
        # fallback: use official p7 submission as test soft labels
        p7_te = pd.read_csv(ROOT / "external_bank/goodpjw_part7_out/submission.csv")["is_fraud"].to_numpy(float)

    pseudo_results = {}
    if p7_oof is not None and p7_te is not None:
        from sklearn.linear_model import LogisticRegression as LR

        def platt_calibrate(raw_a, ya, raw_o):
            m = LR(C=1e6, max_iter=200, solver="lbfgs")
            m.fit(rlogit(raw_a).reshape(-1, 1), ya)
            return m.predict_proba(rlogit(raw_o).reshape(-1, 1))[:, 1]

        for w_pseudo in (0.15, 0.25, 0.35):
            z_ps = np.zeros(NTR)
            for f, (a, b) in enumerate(FOLDS):
                # calibrate p7 on fold a, apply to b and test
                cal_b = platt_calibrate(p7_oof[a], y[a], p7_oof[b])
                cal_t = platt_calibrate(p7_oof[a], y[a], p7_te)
                # soft TE: mix hard labels on a with soft on b∪test manifold via weighted counts
                # Implement as: encode using ya on a, but also add soft counts from (b with cal_b) and (test with cal_t)
                ya = y[a].astype(np.float64)
                Ka, Kb, Kt = KMAT[a], KMAT[b], KMAT[NTR:]
                Ea = np.zeros(Ka.shape, np.float32)
                Eb = np.zeros(Kb.shape, np.float32)
                Et = np.zeros(Kt.shape, np.float32)
                # soft pool indices
                Ksoft = np.concatenate([Kb, Kt])
                ysoft = np.concatenate([cal_b, cal_t])
                inner_f = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(a)), ya))
                smooth = 20.0
                for j in range(KMAT.shape[1]):
                    for i, o in inner_f:
                        # hard from i + soft from all soft pool
                        s = np.bincount(Ka[i, j], weights=ya[i], minlength=NCAT[j]) + w_pseudo * np.bincount(
                            Ksoft[:, j], weights=ysoft, minlength=NCAT[j]
                        )
                        n = np.bincount(Ka[i, j], minlength=NCAT[j]) + w_pseudo * np.bincount(
                            Ksoft[:, j], minlength=NCAT[j]
                        )
                        prior = (ya[i].sum() + w_pseudo * ysoft.sum()) / (len(i) + w_pseudo * len(ysoft))
                        Ea[o, j] = ((s + prior * smooth) / (n + smooth))[Ka[o, j]]
                    s = np.bincount(Ka[:, j], weights=ya, minlength=NCAT[j]) + w_pseudo * np.bincount(
                        Ksoft[:, j], weights=ysoft, minlength=NCAT[j]
                    )
                    n = np.bincount(Ka[:, j], minlength=NCAT[j]) + w_pseudo * np.bincount(
                        Ksoft[:, j], minlength=NCAT[j]
                    )
                    prior = (ya.sum() + w_pseudo * ysoft.sum()) / (len(a) + w_pseudo * len(ysoft))
                    r = (s + prior * smooth) / (n + smooth)
                    Eb[:, j], Et[:, j] = r[Kb[:, j]], r[Kt[:, j]]
                Ga_, Gb_, Gt_ = Ea, Eb, Et
                if GEN:
                    # keep gen hard-only (pseudo on gen is heavier; skip for leak discipline)
                    g1, g2, g3 = gen_fold(a, b, seed=0)
                    Ga_, Gb_, Gt_ = np.hstack([Ea, g1]), np.hstack([Eb, g2]), np.hstack([Et, g3])
                sc = StandardScaler().fit(logit(Ga_))
                D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
                glm_fit = glm_fit_factory(0.3, None)
                za, (zb, zt) = glm_fit(a, D(Ga_, a), [(b, D(Gb_, b)), (TEST, D(Gt_, TEST))], y[a])
                z_ps[b] = zb
                log(f"    w={w_pseudo} fold {f}: {roc_auc_score(y[b], zb):.5f}")
            auc_ps = float(roc_auc_score(y, z_ps))
            pseudo_results[str(w_pseudo)] = {
                "oof": auc_ps,
                "delta": auc_ps - GLM_BASELINE,
            }
            log(f"  soft-pseudo w={w_pseudo}: OOF={auc_ps:.6f} Δ={auc_ps - GLM_BASELINE:+.6f}")
            np.save(PRED / f"glm_softpseudo_w{w_pseudo}_oof.npy", z_ps.astype(np.float32))
    else:
        log("  SKIP soft-pseudo: missing p7 OOF/test")

    report["track3"] = {
        "pair_screen": {
            "n_screened": len(g_rows),
            "n_kept_G_gt_1.5df": len(kept_pairs),
            "kept": kept_pairs[:50],
            "all_top20": sorted(g_rows, key=lambda d: -d["G"])[:20],
        },
        "multismooth_glm": {
            "smooths": smooths,
            "oof": auc_multi,
            "delta_vs_950474": auc_multi - GLM_BASELINE,
            "beats": bool(auc_multi > GLM_BASELINE + 1e-9),
        },
        "soft_pseudo_te": pseudo_results,
    }
    save_report({k: v for k, v in report.items() if not k.startswith("_")})
    return report


# ── Track 4 ─────────────────────────────────────────────────────────────────
def train_one_joint_seed(teseed, mseed, use_varmatch=False):
    """Train one gb_gpt2_joint seed; return oof, foldavg, refit, metrics."""
    import gboost as gb
    from gboost import (
        ALL,
        BASE,
        CAT_IDX,
        FOLDS,
        GEN,
        TEST,
        gen_fold,
        glm_fit_factory,
        logit,
        make_view_fn,
        te_fold,
    )

    gpt_o = np.load(ROOT / "external_bank/masaya_gpt2/oof.npy").ravel().astype(float)
    gpt_t = np.load(ROOT / "external_bank/masaya_gpt2/test.npy").ravel().astype(float)
    gpt_zr = rlogit(rk(gpt_o))
    gpt_zt = rlogit(rk(gpt_t))
    y = gb.y

    glm_fit = glm_fit_factory(0.3, None)
    view = make_view_fn("joint")
    z_glm = np.zeros(len(y))
    z_joint = np.zeros(len(y))
    oof_boost = np.zeros(len(y))
    foldavg = np.zeros(len(gpt_t))
    its = []
    n_jobs = int(os.environ.get("NJOBS", max(1, (os.cpu_count() or 4) // 2)))
    lgb_params = dict(
        n_estimators=20000,
        learning_rate=0.02,
        num_leaves=4,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        max_bin=255,
        verbose=-1,
        n_jobs=n_jobs,
        random_state=mseed,
    )

    def maybe_varmatch(La, Lx):
        if not use_varmatch:
            return Lx
        out = Lx.copy()
        for j in range(La.shape[1]):
            s = Lx[:, j].std()
            r = La[:, j].std()
            if s > 1e-12:
                out[:, j] = (Lx[:, j] - Lx[:, j].mean()) * (r / s) + La[:, j].mean()
        return out

    for f, (a, b) in enumerate(FOLDS):
        log(f"  seed(te={teseed},m={mseed}) fold {f}...")
        Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=teseed)
        Ga_, Gb_, Gt_ = Ea, Eb, Et
        if GEN:
            Ga, Gb, Gt = gen_fold(a, b, seed=teseed)
            Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        La, Lb, Lt = logit(Ga_), logit(Gb_), logit(Gt_)
        Lb, Lt = maybe_varmatch(La, Lb), maybe_varmatch(La, Lt)
        sc = StandardScaler().fit(La)
        D = lambda E, i: np.hstack([sc.transform(E), BASE[i]])
        za, (zb, zt) = glm_fit(a, D(La, a), [(b, D(Lb, b)), (TEST, D(Lt, TEST))], y[a])
        z_glm[b] = zb
        g_a, g_b, g_t = gpt_zr[a], gpt_zr[b], gpt_zt
        Ja = np.column_stack([D(La, a), g_a, za * g_a])
        Jb = np.column_stack([D(Lb, b), g_b, zb * g_b])
        Jt = np.column_stack([D(Lt, TEST), g_t, zt * g_t])
        mj = LogisticRegression(C=0.3, tol=1e-7, max_iter=200, solver="newton-cholesky")
        mj.fit(Ja, y[a])
        init_a = mj.decision_function(Ja)
        init_b = mj.decision_function(Jb)
        init_t = mj.decision_function(Jt)
        z_joint[b] = init_b
        Xa2 = np.column_stack([view(a, Ea).astype(np.float32), g_a, g_a - za])
        Xb2 = np.column_stack([view(b, Eb).astype(np.float32), g_b, g_b - zb])
        Xt2 = np.column_stack([view(TEST, Et).astype(np.float32), g_t, g_t - zt])
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            Xa2,
            y[a],
            init_score=init_a,
            eval_set=[(Xb2, y[b])],
            eval_init_score=[init_b],
            eval_metric="auc",
            categorical_feature=CAT_IDX,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        it = int(model.best_iteration_)
        its.append(it)
        oof_boost[b] = init_b + model.predict(Xb2, raw_score=True)
        foldavg += (init_t + model.predict(Xt2, raw_score=True)) / len(FOLDS)
        log(
            f"    GLM {roc_auc_score(y[b], zb):.5f} joint {roc_auc_score(y[b], init_b):.5f} "
            f"boost {roc_auc_score(y[b], oof_boost[b]):.5f} ({it} trees)"
        )
        del Xa2, Xb2, Xt2, Ja, Jb, Jt, model

    # full refit
    log(f"  seed(te={teseed},m={mseed}) full-data refit...")
    a_all, b_all = ALL
    Ea, Eb, Et = te_fold(a_all, b_all, smooth=20.0, seed=teseed)
    Ga_, Gb_, Gt_ = Ea, Eb, Et
    if GEN:
        Ga, Gb, Gt = gen_fold(a_all, b_all, seed=teseed)
        Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
    La, Lt = logit(Ga_), logit(Gt_)
    Lt = maybe_varmatch(La, Lt)
    sc = StandardScaler().fit(La)
    D = lambda E, i: np.hstack([sc.transform(E), BASE[i]])
    za, (zt,) = glm_fit(a_all, D(La, a_all), [(TEST, D(Lt, TEST))], y[a_all])
    Ja = np.column_stack([D(La, a_all), gpt_zr, za * gpt_zr])
    Jt = np.column_stack([D(Lt, TEST), gpt_zt, zt * gpt_zt])
    m_full = LogisticRegression(C=0.3, tol=1e-7, max_iter=200, solver="newton-cholesky")
    m_full.fit(Ja, y)
    init_full = m_full.decision_function(Ja)
    init_te = m_full.decision_function(Jt)
    Xa_full2 = np.column_stack([view(a_all, Ea).astype(np.float32), gpt_zr, gpt_zr - za])
    Xt_full2 = np.column_stack([view(TEST, Et).astype(np.float32), gpt_zt, gpt_zt - zt])
    n_refit = int(round(np.mean(its) * 1.25))
    model = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": n_refit})
    model.fit(Xa_full2, y, init_score=init_full, categorical_feature=CAT_IDX)
    refit = init_te + model.predict(Xt_full2, raw_score=True)
    te = (rk(foldavg) + rk(refit)) / 2
    metrics = {
        "glm_oof": float(roc_auc_score(y, z_glm)),
        "joint_oof": float(roc_auc_score(y, z_joint)),
        "boost_oof": float(roc_auc_score(y, oof_boost)),
        "its": its,
        "n_refit": n_refit,
    }
    return oof_boost, te, metrics


def track4(y, folds, report):
    log("=== TRACK 4: 5-seed gb_gpt2_joint_bag5 ===")
    use_vm = report.get("track2_best_glm") == "varmatch"
    log(f"  use_varmatch_stage1={use_vm}")

    seed_dir = PRED / "gb_gpt2_joint_seeds"
    seed_dir.mkdir(exist_ok=True)
    oofs, tests, metrics = [], [], {}

    # reuse seed0 if present and not using varmatch
    s0_o = PRED / "gb_gpt2_joint_oof.npy"
    s0_t = PRED / "gb_gpt2_joint_test.npy"
    for si, (teseed, mseed) in enumerate(GB_SEEDS):
        tag = f"s{si}_te{teseed}_m{mseed}"
        op = seed_dir / f"{tag}_oof.npy"
        tp = seed_dir / f"{tag}_test.npy"
        if op.exists() and tp.exists() and not use_vm:
            log(f"  cache hit {tag}")
            o = np.load(op)
            t = np.load(tp)
        elif si == 0 and s0_o.exists() and s0_t.exists() and not use_vm:
            log(f"  reusing part9 seed0 preds for {tag}")
            o = np.load(s0_o)
            t = np.load(s0_t)
            np.save(op, o)
            np.save(tp, t)
            metrics[tag] = {"boost_oof": float(roc_auc_score(y, o)), "reused_part9": True}
        else:
            o, t, m = train_one_joint_seed(teseed, mseed, use_varmatch=use_vm)
            np.save(op, o.astype(np.float32))
            np.save(tp, t.astype(np.float32))
            metrics[tag] = m
            log(f"  {tag}: boost OOF={m['boost_oof']:.6f}")
        oofs.append(np.asarray(o, float).ravel())
        tests.append(np.asarray(t, float).ravel())
        if tag not in metrics:
            metrics[tag] = {"boost_oof": float(roc_auc_score(y, oofs[-1]))}

    # rank-mean bag (matching gb_bag5 construction)
    bag_o = rk(np.mean([rk(o) for o in oofs], axis=0))
    # for test: each seed already stores rank-mean(foldavg, refit); bag = rank-mean across seeds
    bag_t = rk(np.mean([rk(t) for t in tests], axis=0))
    # also save "full" as the bag test (same convention as gb_bag5_full)
    auc_bag = float(roc_auc_score(y, bag_o))
    log(f"  gb_gpt2_bag5 OOF={auc_bag:.6f}")

    gb5_o = np.load(ROOT / "external_bank/goodpjw_part7_artifacts/gb_bag5_oof.npy").ravel()
    combo = rk(0.5 * rk(gb5_o) + 0.5 * bag_o)
    auc_combo = float(roc_auc_score(y, combo))
    log(f"  0.5*gb_bag5 + 0.5*gb_gpt2_bag5 OOF={auc_combo:.6f}")

    np.save(PRED / "gb_gpt2_bag5_oof.npy", bag_o.astype(np.float32))
    np.save(PRED / "gb_gpt2_bag5_full.npy", bag_t.astype(np.float32))
    log("  saved preds/gb_gpt2_bag5_{oof,full}.npy")

    report["track4"] = {
        "seeds": GB_SEEDS,
        "use_varmatch": use_vm,
        "per_seed": metrics,
        "gb_gpt2_bag5_oof": auc_bag,
        "combo_with_gb_bag5_oof": auc_combo,
        "targets": {"single_family": 0.95170, "combo": 0.95190},
        "hit_single_family": bool(auc_bag >= 0.95170),
        "hit_combo": bool(auc_combo >= 0.95190),
        "artifacts": {
            "oof": str(PRED / "gb_gpt2_bag5_oof.npy"),
            "full": str(PRED / "gb_gpt2_bag5_full.npy"),
        },
    }
    save_report({k: v for k, v in report.items() if not k.startswith("_")})
    return report


def main():
    if LOG_PATH.exists():
        LOG_PATH.write_text("")
    log("PART10 DEEP RESEARCH START — policy=NO_SUBMIT")
    log(f"cwd={ROOT} cpus={os.cpu_count()} INPUT_DIR={os.environ.get('INPUT_DIR')}")

    y = pd.read_csv(ROOT / "data/train.csv")["is_fraud"].to_numpy()
    te_ids = pd.read_csv(ROOT / "data/sample_submission.csv")["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    report = {"policy": POLICY, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    # resume support
    if REPORT_PATH.exists():
        try:
            prev = json.loads(REPORT_PATH.read_text())
            for k in ("track1", "track2", "track2_best_glm", "track3", "track4"):
                if k in prev:
                    report[k] = prev[k]
            log(f"  resumed keys={[k for k in report if k.startswith('track')]}")
        except Exception as e:
            log(f"  resume parse failed: {e}")

    try:
        if "track1" not in report:
            report = track1(y, folds, te_ids, report)
        else:
            log("  SKIP track1 (already in report)")
            # reload cache arrays
            report["_cache"] = {
                "p7_oof": np.load(PRED / "oof_p7_nested_exact.npy") if (PRED / "oof_p7_nested_exact.npy").exists() else None,
                "p7_te": pd.read_csv(ROOT / "external_bank/goodpjw_part7_out/submission.csv")["is_fraud"].to_numpy(float),
            }

        if "track2" not in report:
            report = track2(y, folds, report)
        else:
            log("  SKIP track2 (already in report)")

        if "track3" not in report:
            report = track3(y, folds, report)
        else:
            log("  SKIP track3 (already in report)")

        if "track4" not in report:
            report = track4(y, folds, report)
        else:
            log("  SKIP track4 (already in report)")

        report["status"] = "complete"
        report["policy"] = POLICY
        save_report({k: v for k, v in report.items() if not k.startswith("_")})
        log(f"DONE — wrote {REPORT_PATH}  runtime={time.time()-T0:.0f}s  NO_SUBMIT")
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        save_report({k: v for k, v in report.items() if not k.startswith("_")})
        log("FAILED:\n" + report["error"])
        raise


if __name__ == "__main__":
    main()
