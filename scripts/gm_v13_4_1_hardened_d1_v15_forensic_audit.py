#!/usr/bin/env python3
"""GM V13.4.1 HARDENED: pre-PCHIP dispersion, ablation, per-booster audit, trust-region.

Re-fits d1_v15 τ=1.00 with per-family residual dumps + split gains.
STRICT NO_SUBMIT=True. Zero pods.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT is True

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
PRED = ROOT / "preds"
ART = PRED / "gm_v13_4_stageB"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"

MIN_FOLD_E5 = 0.10
SF25_NEED = 18
T25_NEED = 2.80
WILC_P = 0.005
BOOT_LO_NEED = 0.25
RAW_DISP_LO, RAW_DISP_HI = 0.92, 1.08
CORR_MATCH = 0.03

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def pchip_lock(src, ref):
    src = np.asarray(src, float).ravel()
    ref = np.asarray(ref, float).ravel()
    qs = np.linspace(0, 1, 401)
    x, y = np.quantile(src, qs), np.quantile(ref, qs)
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    return PchipInterpolator(x, y, extrapolate=True)(src)


def solve_b(linear, target, lo=-20.0, hi=5.0, n=80):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if float(expit(linear + mid).mean()) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def load_pack():
    log("Load ATH + X_v15")
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from gm_v13_2_overnight_check_and_stageB_v14 import build_ath_proxy

    ath = build_ath_proxy()
    Xo = np.load(ART / "X_v15_oof.npy")
    Xt = np.load(ART / "X_v15_test.npy")
    names = json.loads((ART / "X_v15_names.json").read_text())
    live_p = pd.read_csv(LIVE)["is_fraud"].to_numpy(float)
    ids = pd.read_csv(LIVE)["id"].to_numpy()
    return {
        **ath,
        "Xo": Xo,
        "Xt": Xt,
        "feat_names": names,
        "live_p": live_p,
        "ids": ids,
        "L_LIVE": L(live_p),
    }


def feature_groups(names):
    g1, g2, g3, g4 = [], [], [], []
    for i, n in enumerate(names):
        if n.startswith("KMAT_"):
            g1.append(i)
        elif n.startswith("P4_"):
            g2.append(i)
        elif n == "q_ATH_bin":
            g4.append(i)
        else:
            g3.append(i)
    return {"KMAT_68": g1, "P4": g2, "CONTRASTS": g3, "q_ATH_bin": g4}


# ---------------------------------------------------------------------------
# Train τ=1.00 with per-family dumps
# ---------------------------------------------------------------------------
def train_tau100(pack, col_idx=None, tag="full"):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    y, folds = pack["y"], pack["folds"]
    z_o, z_t = pack["z_ATH_o"], pack["z_ATH_t"]
    if col_idx is None:
        Xo, Xt = pack["Xo"], pack["Xt"]
        feat_names = pack["feat_names"]
    else:
        Xo, Xt = pack["Xo"][:, col_idx], pack["Xt"][:, col_idx]
        feat_names = [pack["feat_names"][i] for i in col_idx]

    oof = {k: np.zeros(len(y)) for k in ("lgb", "xgb", "cat", "init")}
    te = {k: [] for k in ("lgb", "xgb", "cat", "init")}
    gains = {k: np.zeros(Xo.shape[1]) for k in ("lgb", "xgb", "cat")}
    splits = {k: np.zeros(Xo.shape[1]) for k in ("lgb", "xgb", "cat")}

    params_lgb = dict(
        objective="binary",
        metric="auc",
        max_depth=3,
        num_leaves=6,
        min_child_samples=500,
        feature_fraction=0.30,
        lambda_l2=100.0,
        learning_rate=0.015,
        verbosity=-1,
    )
    for f, (tr, te_idx) in enumerate(folds):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(z_o[tr].reshape(-1, 1), y[tr])
        a = float(lr.coef_.ravel()[0])  # τ=1.00
        b = solve_b(a * z_o[tr], float(y[tr].mean()))
        init_tr, init_te, init_tt = a * z_o[tr] + b, a * z_o[te_idx] + b, a * z_t + b
        assert abs(float(expit(init_tr).mean()) - float(y[tr].mean())) < 1e-6
        oof["init"][te_idx] = init_te
        te["init"].append(init_tt)
        log(f"  [{tag}] fold {f}…")

        dtr = lgb.Dataset(Xo[tr], label=y[tr], init_score=init_tr, feature_name=feat_names)
        m = lgb.train(params_lgb, dtr, num_boost_round=180)
        oof["lgb"][te_idx] = m.predict(Xo[te_idx], raw_score=True)
        te["lgb"].append(m.predict(Xt, raw_score=True))
        # gain / split
        for i, (g, s) in enumerate(zip(m.feature_importance("gain"), m.feature_importance("split"))):
            gains["lgb"][i] += g
            splits["lgb"][i] += s

        dtrain = xgb.DMatrix(Xo[tr], label=y[tr], base_margin=init_tr, feature_names=feat_names)
        dval = xgb.DMatrix(Xo[te_idx], base_margin=init_te, feature_names=feat_names)
        dtest = xgb.DMatrix(Xt, base_margin=init_tt, feature_names=feat_names)
        bx = xgb.train(
            dict(
                objective="binary:logistic",
                max_depth=3,
                min_child_weight=500,
                colsample_bytree=0.30,
                subsample=0.8,
                reg_lambda=100.0,
                eta=0.015,
                tree_method="hist",
                verbosity=0,
            ),
            dtrain,
            num_boost_round=180,
        )
        oof["xgb"][te_idx] = bx.predict(dval, output_margin=True)
        te["xgb"].append(bx.predict(dtest, output_margin=True))
        score = bx.get_score(importance_type="gain")
        score_s = bx.get_score(importance_type="weight")
        for j, nm in enumerate(feat_names):
            # xgb may use f0.. or feature names
            gains["xgb"][j] += float(score.get(nm, score.get(f"f{j}", 0.0)))
            splits["xgb"][j] += float(score_s.get(nm, score_s.get(f"f{j}", 0.0)))

        cat = CatBoostClassifier(
            loss_function="Logloss",
            depth=3,
            min_data_in_leaf=500,
            rsm=0.30,
            l2_leaf_reg=100.0,
            learning_rate=0.015,
            iterations=180,
            verbose=False,
            random_seed=42 + f,
            thread_count=2,
        )
        try:
            cat.fit(Xo[tr], y[tr], baseline=init_tr.reshape(-1, 1))
            oof["cat"][te_idx] = cat.predict(
                Xo[te_idx], prediction_type="RawFormulaVal", baseline=init_te.reshape(-1, 1)
            )
            te["cat"].append(
                cat.predict(Xt, prediction_type="RawFormulaVal", baseline=init_tt.reshape(-1, 1))
            )
        except TypeError:
            cat.fit(Xo[tr], y[tr])
            oof["cat"][te_idx] = cat.predict(Xo[te_idx], prediction_type="RawFormulaVal")
            te["cat"].append(cat.predict(Xt, prediction_type="RawFormulaVal"))
        # CatBoost feature importances (PredictionValuesChange or LossFunctionChange)
        try:
            imp = np.asarray(cat.get_feature_importance(), float)
            gains["cat"] += imp
            splits["cat"] += (imp > 0).astype(float)
        except Exception:
            pass

    init_t = np.mean(te["init"], 0)
    raw_o, raw_t = {}, {}
    for fam in ("lgb", "xgb", "cat"):
        raw_o[fam] = oof[fam] - oof["init"]
        raw_t[fam] = np.mean(te[fam], 0) - init_t
    raw_o["mean"] = (raw_o["lgb"] + raw_o["xgb"] + raw_o["cat"]) / 3.0
    raw_t["mean"] = (raw_t["lgb"] + raw_t["xgb"] + raw_t["cat"]) / 3.0

    return {
        "raw_o": raw_o,
        "raw_t": raw_t,
        "init_o": oof["init"],
        "init_t": init_t,
        "gains": gains,
        "splits": splits,
        "feat_names": feat_names,
        "tag": tag,
    }


def lofo_orth(delta_o, delta_t, pack):
    B_o = np.column_stack([unitize(pack["z_ATH_o"]), pack["d1o"]])
    B_t = np.column_stack([unitize(pack["z_ATH_t"]), pack["d1t"]])
    Vo, Vt = unitize(delta_o), unitize(delta_t)
    resid = np.zeros(len(pack["y"]))
    betas = []
    for f in range(5):
        te = pack["fold_id"] == f
        tr = ~te
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B_o[tr], Vo[tr])
        resid[te] = Vo[te] - lr.predict(B_o[te])
        betas.append(np.concatenate([lr.coef_, [lr.intercept_]]))
    bbar = np.mean(betas, 0)
    rt = Vt - (B_t @ bbar[:-1] + bbar[-1])
    # return both pre-pchip and pchip versions of test
    rt_raw = rt.copy()
    rt_pc = unitize(pchip_lock(rt, resid))
    return unitize(resid), rt_raw, rt_pc, bbar


def eval_main(y, folds, base, cand):
    d = np.array(
        [1e5 * (roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])) for _, te in folds]
    )
    rng = np.random.default_rng(42)
    boots = [d[rng.integers(0, 5, size=5)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "pos5": int((d > 0).sum()),
        "mean_e5": float(d.mean()),
        "min_e5": float(d.min()),
        "folds": [round(float(x), 3) for x in d],
        "boot_90ci": [round(float(lo), 3), round(float(hi), 3)],
    }


def eval_25(y, fold_id, base, cand):
    deltas = []
    for f in range(5):
        m = fold_id == f
        yf, bf, cf = y[m], base[m], cand[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for _, te in skf.split(np.arange(len(yf)), yf):
            deltas.append(1e5 * (roc_auc_score(yf[te], cf[te]) - roc_auc_score(yf[te], bf[te])))
    d = np.asarray(deltas, float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    try:
        _, wp = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    return {
        "pos25": int((d > 0).sum()),
        "mean_e5": mean,
        "t_stat": float(mean / (se + 1e-18)),
        "wilcoxon_p": float(wp),
    }


def four_bar(main, sf):
    b1 = main["pos5"] == 5 and main["min_e5"] >= MIN_FOLD_E5
    b2 = sf["pos25"] >= SF25_NEED and sf["t_stat"] >= T25_NEED and sf["wilcoxon_p"] < WILC_P
    b3 = main["boot_90ci"][0] > BOOT_LO_NEED
    return {
        "bar1": b1,
        "bar2": b2,
        "bar3": b3,
        "pass_123": bool(b1 and b2 and b3),
    }


def screen_family(pack, ro, rt_pc, raw_disp, tag, weights=(0.015, 0.020, 0.025, 0.030)):
    rows = []
    for w in weights:
        cand = pack["z_ATH_o"] + w * ro
        main = eval_main(pack["y"], pack["folds"], pack["z_ATH_o"], cand)
        sf = eval_25(pack["y"], pack["fold_id"], pack["z_ATH_o"], cand)
        bars = four_bar(main, sf)
        rows.append(
            {
                "family": tag,
                "w": w,
                "main": main,
                "sf25": sf,
                "bars": bars,
                "raw_disp": round(raw_disp, 4),
            }
        )
        log(
            f"  {tag} w={w}: {main['pos5']}/5 min={main['min_e5']:.2f} "
            f"sf={sf['pos25']}/25 t={sf['t_stat']:.2f} pass123={bars['pass_123']}"
        )
    return rows


def force_to_live(score, live_p):
    order = np.argsort(
        score + 1e-15 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort"
    )
    out = np.empty(len(score), float)
    out[order] = np.sort(live_p)
    return out


def trust_region(pack, ro, rt_pc, w):
    z = pack["z_ATH_t"] + w * unitize(rt_pc)
    cand = force_to_live(z, pack["live_p"])
    dist = float(1 - spearmanr(cand, pack["live_p"]).correlation)
    # top 5% rank shift
    n = len(cand)
    r_live = rankdata(pack["live_p"], method="average")
    r_cand = rankdata(cand, method="average")
    top = r_live >= 0.95 * n
    max_shift = float(np.max(np.abs(r_cand[top] - r_live[top]))) if top.any() else 0.0
    return dist, max_shift, cand


def main():
    assert NO_SUBMIT
    ART.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    log("GM V13.4.1 HARDENED audit · NO_SUBMIT=True · POD_TOUCHES=0")
    pack = load_pack()
    groups = feature_groups(pack["feat_names"])
    log(f"  groups sizes: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    # ---- Full retrain ----
    full = train_tau100(pack, col_idx=None, tag="full")
    # persist per-family
    for fam in ("lgb", "xgb", "cat", "mean"):
        np.save(ART / f"d1_v15_tau1.00_{fam}_raw_oof.npy", full["raw_o"][fam].astype(np.float32))
        np.save(ART / f"d1_v15_tau1.00_{fam}_raw_test.npy", full["raw_t"][fam].astype(np.float32))
    np.save(ART / "d1_v15_tau1.00_init_oof.npy", full["init_o"].astype(np.float32))
    (ART / "d1_v15_tau1.00_gains.json").write_text(
        json.dumps(
            {
                "feat_names": full["feat_names"],
                "gains": {k: v.tolist() for k, v in full["gains"].items()},
                "splits": {k: v.tolist() for k, v in full["splits"].items()},
            }
        )
    )

    # ---- Step 1: pre-PCHIP disp + pre-orth corr ----
    log("STEP1: pre-PCHIP raw_disp + pre-orth correlations")
    step1 = {"raw_disp": {}, "pre_orth_corr": {}, "raw_disp_pass": {}}
    for fam in ("lgb", "xgb", "cat", "mean"):
        rd = float(np.std(full["raw_t"][fam]) / (np.std(full["raw_o"][fam]) + 1e-12))
        step1["raw_disp"][fam] = rd
        step1["raw_disp_pass"][fam] = bool(RAW_DISP_LO <= rd <= RAW_DISP_HI)
        # pre-orth corrs
        co_d1 = float(spearmanr(full["raw_o"][fam], pack["d1o"]).correlation)
        ct_d1 = float(spearmanr(full["raw_t"][fam], pack["d1t"]).correlation)
        co_z = float(spearmanr(full["raw_o"][fam], pack["z_ATH_o"]).correlation)
        ct_z = float(spearmanr(full["raw_t"][fam], pack["z_ATH_t"]).correlation)
        step1["pre_orth_corr"][fam] = {
            "corr_d1_oof": co_d1,
            "corr_d1_test": ct_d1,
            "d1_gap": abs(co_d1 - ct_d1),
            "corr_z_oof": co_z,
            "corr_z_test": ct_z,
            "z_gap": abs(co_z - ct_z),
            "d1_match": abs(co_d1 - ct_d1) <= CORR_MATCH,
            "z_match": abs(co_z - ct_z) <= CORR_MATCH,
        }
        log(
            f"  {fam}: raw_disp={rd:.4f} pass={step1['raw_disp_pass'][fam]} "
            f"ρ(d1) o/t={co_d1:.3f}/{ct_d1:.3f} gap={abs(co_d1-ct_d1):.3f} "
            f"ρ(z) o/t={co_z:.3f}/{ct_z:.3f} gap={abs(co_z-ct_z):.3f}"
        )

    # ---- Step 2: gains + corr symmetry ----
    log("STEP2: split-gain breakdown + feature corr symmetry")
    # aggregate gain across families (sum)
    tot_gain = full["gains"]["lgb"] + full["gains"]["xgb"] + full["gains"]["cat"]
    order = np.argsort(-tot_gain)
    top20 = [
        {
            "rank": i + 1,
            "feat": full["feat_names"][j],
            "gain": float(tot_gain[j]),
            "lgb": float(full["gains"]["lgb"][j]),
            "xgb": float(full["gains"]["xgb"][j]),
            "cat": float(full["gains"]["cat"][j]),
        }
        for i, j in enumerate(order[:20])
    ]
    gsum = float(tot_gain.sum()) + 1e-18
    group_pct = {}
    for gname, idxs in groups.items():
        group_pct[gname] = float(tot_gain[idxs].sum() / gsum * 100.0) if idxs else 0.0
    log(f"  group %gain: { {k: round(v,2) for k,v in group_pct.items()} }")
    for t in top20[:10]:
        log(f"  top{t['rank']}: {t['feat']} gain={t['gain']:.1f}")

    # LOFO orth mean for corr symmetry
    ro_m, rt_raw_m, rt_pc_m, _ = lofo_orth(full["raw_o"]["mean"], full["raw_t"]["mean"], pack)
    # use post-orth unitized oof vs pchip test for feature corr with d1_v15
    d15_o = ro_m
    d15_t = unitize(rt_pc_m)
    corr_sym = []
    for j in order[:15]:
        nm = full["feat_names"][j]
        co = float(spearmanr(d15_o, pack["Xo"][:, j]).correlation)
        ct = float(spearmanr(d15_t, pack["Xt"][:, j]).correlation)
        corr_sym.append(
            {
                "feat": nm,
                "corr_oof": co,
                "corr_test": ct,
                "abs_delta": abs(co - ct),
                "shift_flag": abs(co - ct) > 0.04,
            }
        )
    n_shift = sum(1 for c in corr_sym if c["shift_flag"])
    log(f"  top15 |Δcorr|>0.04: {n_shift}/15")

    # ---- Ablation retrain ----
    log("STEP2b: ablation KMAT-only / CONTRASTS-only / Full")
    abl_results = {}
    # Full already done — screen mean
    for abl_name, idxs in (
        ("KMAT_only", groups["KMAT_68"]),
        ("CONTRASTS_only", groups["CONTRASTS"] + groups["P4"] + groups["q_ATH_bin"]),
        ("FULL", None),
    ):
        if abl_name == "FULL":
            fit = full
        else:
            fit = train_tau100(pack, col_idx=idxs, tag=abl_name)
        ro, rt_raw, rt_pc, _ = lofo_orth(fit["raw_o"]["mean"], fit["raw_t"]["mean"], pack)
        rd = float(np.std(fit["raw_t"]["mean"]) / (np.std(fit["raw_o"]["mean"]) + 1e-12))
        rows = screen_family(pack, ro, rt_pc, rd, abl_name, weights=(0.02, 0.03))
        abl_results[abl_name] = {
            "raw_disp": rd,
            "rows": rows,
            "best_pass": any(r["bars"]["pass_123"] for r in rows),
        }

    # ---- Step 3: per-booster families ----
    log("STEP3: per-booster family audit")
    family_defs = {
        "LGB_only": ["lgb"],
        "XGB_only": ["xgb"],
        "CAT_only": ["cat"],
        "LGB+CAT": ["lgb", "cat"],
        "LGB+XGB": ["lgb", "xgb"],
        "LGB+XGB+CAT": ["lgb", "xgb", "cat"],
    }
    family_rows = []
    family_best = {}
    for fname, parts in family_defs.items():
        ro_raw = sum(full["raw_o"][p] for p in parts) / len(parts)
        rt_raw = sum(full["raw_t"][p] for p in parts) / len(parts)
        rd = float(np.std(rt_raw) / (np.std(ro_raw) + 1e-12))
        ro, rt_r, rt_pc, _ = lofo_orth(ro_raw, rt_raw, pack)
        rows = screen_family(pack, ro, rt_pc, rd, fname)
        family_rows.extend(rows)
        # keep best passer or best mean among pass_123
        passers = [r for r in rows if r["bars"]["pass_123"]]
        if passers:
            best = max(passers, key=lambda r: r["main"]["mean_e5"])
        else:
            best = max(rows, key=lambda r: (r["main"]["pos5"], r["sf25"]["t_stat"]))
        family_best[fname] = {
            **best,
            "ro": ro,
            "rt_pc": rt_pc,
            "rt_raw": rt_r,
            "raw_o": ro_raw,
            "raw_t": rt_raw,
        }

    # Independent family weak bar (>=4/5 and >=16/25 at some w)
    indep = {}
    for fam in ("LGB_only", "XGB_only", "CAT_only"):
        ok = any(
            r["main"]["pos5"] >= 4 and r["sf25"]["pos25"] >= 16
            for r in family_rows
            if r["family"] == fam
        )
        indep[fam] = ok
    log(f"  independent weak-bar (>=4/5 & >=16/25): {indep}")

    # ---- Step 4: trust region + stage ----
    log("STEP4: trust-region + stage NOSUBMIT")
    # primary: LGB+XGB+CAT (verified v13.4)
    primary = family_best["LGB+XGB+CAT"]
    # best family if strictly superior on all 4 bars at its best w
    best_combo_name = "LGB+XGB+CAT"
    for fname, fb in family_best.items():
        if not fb["bars"]["pass_123"]:
            continue
        if fb["sf25"]["pos25"] > primary["sf25"]["pos25"] or (
            fb["sf25"]["pos25"] == primary["sf25"]["pos25"]
            and fb["sf25"]["t_stat"] > primary["sf25"]["t_stat"]
            and fb["main"]["min_e5"] >= primary["main"]["min_e5"]
        ):
            # require pass_123 already
            best_combo_name = fname
            primary = fb
    log(f"  staging family={best_combo_name}")

    trust = []
    staged = []
    for w in (0.015, 0.020, 0.025, 0.030):
        dist, max_shift, cand = trust_region(pack, primary["ro"], primary["rt_pc"], w)
        # also evaluate bars at this w
        main = eval_main(pack["y"], pack["folds"], pack["z_ATH_o"], pack["z_ATH_o"] + w * primary["ro"])
        sf = eval_25(pack["y"], pack["fold_id"], pack["z_ATH_o"], pack["z_ATH_o"] + w * primary["ro"])
        bars = four_bar(main, sf)
        trust.append(
            {
                "w": w,
                "spearman_dist_e4": dist * 1e4,
                "max_rank_shift_top5pct": max_shift,
                "main": main,
                "sf25": sf,
                "bars": bars,
            }
        )
        log(
            f"  w={w}: 1-ρ={dist*1e4:.3f}e-4 max|Δrank|_top5%={max_shift:.1f} "
            f"pass123={bars['pass_123']}"
        )
        if w in (0.020, 0.025, 0.030):
            path = SUB / f"sub_v13p4_verified_d1_v15_tau1.00_w{w:.3f}_NOSUBMIT.csv"
            assert NO_SUBMIT and "NOSUBMIT" in path.name
            pd.DataFrame({"id": pack["ids"], "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"path": str(path), "w": w, "dist_e4": dist * 1e4, "pass123": bars["pass_123"]})
            log(f"  STAGED {path.name}")

    # persist orth mean for reuse
    np.save(ART / "d1_v15_tau1.00_mean_orth_oof.npy", primary["ro"].astype(np.float32))
    np.save(ART / "d1_v15_tau1.00_mean_orth_test.npy", unitize(primary["rt_pc"]).astype(np.float32))

    # ---- Report ----
    md = LOG / "GM_V13_4_1_HARDENED_D1_V15_AUDIT.md"
    lines = []
    lines.append("# GM V13.4.1 HARDENED — d1_v15_tau1.00 Forensic Audit\n\n")
    lines.append(
        f"**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed {time.time()-T0:.1f}s\n\n"
        f"**Script:** `scripts/gm_v13_4_1_hardened_d1_v15_forensic_audit.py`\n\n"
    )

    lines.append("## Step 1 — Pre-PCHIP raw dispersion & pre-orth collinearity\n\n")
    lines.append("| family | raw_disp | in [0.92,1.08] | ρ(d1) oof/test | |Δρ(d1)| | ρ(z) oof/test | |Δρ(z)| |\n")
    lines.append("|---|---:|---|---|---:|---|---:|\n")
    for fam in ("lgb", "xgb", "cat", "mean"):
        c = step1["pre_orth_corr"][fam]
        lines.append(
            f"| `{fam}` | {step1['raw_disp'][fam]:.4f} | {step1['raw_disp_pass'][fam]} | "
            f"{c['corr_d1_oof']:.3f}/{c['corr_d1_test']:.3f} | {c['d1_gap']:.3f} | "
            f"{c['corr_z_oof']:.3f}/{c['corr_z_test']:.3f} | {c['z_gap']:.3f} |\n"
        )
    all_disp_ok = all(step1["raw_disp_pass"].values())
    all_corr_ok = all(
        step1["pre_orth_corr"][f]["d1_match"] and step1["pre_orth_corr"][f]["z_match"]
        for f in ("lgb", "xgb", "cat", "mean")
    )
    lines.append(
        f"\n**Verdict:** raw_disp OK={all_disp_ok} · OOF/Test corr match (±0.03) OK={all_corr_ok}\n"
    )

    lines.append("\n## Step 2 — Split-gain by group + top features\n\n")
    lines.append("| group | % total gain |\n|---|---:|\n")
    for k, v in group_pct.items():
        lines.append(f"| `{k}` | {v:.2f}% |\n")
    lines.append("\n### Top 20 features by gain\n\n")
    lines.append("| # | feature | gain | lgb | xgb | cat |\n|---:|---|---:|---:|---:|---:|\n")
    for t in top20:
        lines.append(
            f"| {t['rank']} | `{t['feat']}` | {t['gain']:.1f} | {t['lgb']:.1f} | {t['xgb']:.1f} | {t['cat']:.1f} |\n"
        )
    lines.append("\n### Top-15 feature corr symmetry (|Δ|>0.04 flagged)\n\n")
    lines.append("| feature | corr_oof | corr_test | |Δ| | shift |\n|---|---:|---:|---:|---|\n")
    for c in corr_sym:
        lines.append(
            f"| `{c['feat']}` | {c['corr_oof']:.3f} | {c['corr_test']:.3f} | "
            f"{c['abs_delta']:.3f} | {c['shift_flag']} |\n"
        )

    lines.append("\n### Ablation (mean family, w∈{0.02,0.03})\n\n")
    lines.append("| abl | w | main | min | sf25 | t | pass123 | raw_disp |\n|---|---:|---:|---:|---:|---:|---|---:|\n")
    for aname, ar in abl_results.items():
        for r in ar["rows"]:
            m, s = r["main"], r["sf25"]
            lines.append(
                f"| `{aname}` | {r['w']} | {m['pos5']}/5 | {m['min_e5']:.2f} | "
                f"{s['pos25']}/25 | {s['t_stat']:.2f} | {r['bars']['pass_123']} | {r['raw_disp']} |\n"
            )

    lines.append("\n## Step 3 — Per-booster family audit\n\n")
    lines.append(
        "| family | w | main | meanΔ | min | folds | sf25 | t | W-p | boot90 | raw_disp | pass123 |\n"
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|\n"
    )
    for r in family_rows:
        m, s = r["main"], r["sf25"]
        lines.append(
            f"| `{r['family']}` | {r['w']} | {m['pos5']}/5 | {m['mean_e5']:.2f} | {m['min_e5']:.2f} | "
            f"{m['folds']} | {s['pos25']}/25 | {s['t_stat']:.2f} | {s['wilcoxon_p']:.3g} | "
            f"{m['boot_90ci']} | {r['raw_disp']} | {r['bars']['pass_123']} |\n"
        )
    lines.append(f"\nIndependent weak-bar (≥4/5 & ≥16/25): `{indep}`\n")
    lines.append(f"Best staging family: **`{best_combo_name}`**\n")

    lines.append("\n## Step 4 — Trust-region vs LIVE 0.95388\n\n")
    lines.append(
        "| w | 1−ρ (e-4) | max\\|Δrank\\| top5% | main | min | sf25 | t | pass123 |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    for t in trust:
        m, s = t["main"], t["sf25"]
        lines.append(
            f"| {t['w']:.3f} | {t['spearman_dist_e4']:.3f} | {t['max_rank_shift_top5pct']:.1f} | "
            f"{m['pos5']}/5 | {m['min_e5']:.2f} | {s['pos25']}/25 | {s['t_stat']:.2f} | "
            f"{t['bars']['pass_123']} |\n"
        )
    lines.append("\n### Staged (NOSUBMIT)\n\n")
    for s in staged:
        lines.append(
            f"- `{Path(s['path']).name}` · w={s['w']:.3f} · 1−ρ={s['dist_e4']:.3f}e-4 · "
            f"pass123={s['pass123']}\n"
        )

    lines.append("\n## Bottom line\n\n")
    lines.append(
        f"1. Pre-PCHIP `raw_disp` {'PASS' if all_disp_ok else 'FAIL'} "
        f"(mean={step1['raw_disp']['mean']:.4f}).\n"
        f"2. Pre-orth OOF/Test corr match {'PASS' if all_corr_ok else 'FAIL'}.\n"
        f"3. Gain mix: KMAT={group_pct['KMAT_68']:.1f}% · CONTRASTS={group_pct['CONTRASTS']:.1f}% · "
        f"P4={group_pct['P4']:.1f}% · q_ATH_bin={group_pct['q_ATH_bin']:.1f}%.\n"
        f"4. Staging family `{best_combo_name}`; NOSUBMIT files at w=0.020/0.025/0.030 written. "
        f"**Still NO_SUBMIT — do not live-submit without human go.**\n"
    )
    md.write_text("".join(lines))

    # JSON (no huge arrays)
    payload = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "elapsed_s": round(time.time() - T0, 2),
        "step1": step1,
        "group_pct_gain": group_pct,
        "top20": top20,
        "corr_sym": corr_sym,
        "ablation": {
            k: {"raw_disp": v["raw_disp"], "best_pass": v["best_pass"], "rows": v["rows"]}
            for k, v in abl_results.items()
        },
        "family_rows": family_rows,
        "indep_weak_bar": indep,
        "best_combo": best_combo_name,
        "trust": trust,
        "staged": staged,
    }
    (LOG / "GM_V13_4_1_HARDENED_D1_V15_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str))
    log(f"WROTE {md}")
    log("DONE")


if __name__ == "__main__":
    main()
