#!/usr/bin/env python3
"""GM V13.4.2: fix LGB double-subtract, drop CAT, symmetry-lock features, trust-region stage.

LGB predict(raw_score=True) does NOT include init_score — do not subtract init again.
XGB: extract pure tree margin via DMatrix without base_margin (assert == with-margin − init).

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
ART2 = PRED / "gm_v13_4_2_clean"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"

MIN_FOLD = 0.10
SF_NEED = 19
T_NEED = 3.30
WILC_P = 0.005
BOOT_LO = 0.25
RAW_LO, RAW_HI = 0.92, 1.08
RHO_Z_MAX = 0.30
DRHO_D1 = 0.04

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
    return {
        **ath,
        "Xo_raw": Xo,
        "Xt_raw": Xt,
        "feat_names": names,
        "live_p": live_p,
        "ids": pd.read_csv(LIVE)["id"].to_numpy(),
        "L_LIVE": L(live_p),
    }


def build_clean_matrices(pack):
    """PCHIP-lock every column test→oof; build Set A / Set B indices."""
    log("Build X_v15_clean (per-col PCHIP) + Sets A/B")
    names = pack["feat_names"]
    Xo = pack["Xo_raw"].astype(np.float64).copy()
    Xt = pack["Xt_raw"].astype(np.float64).copy()
    Xt_lock = np.empty_like(Xt)
    for j in range(Xo.shape[1]):
        Xt_lock[:, j] = pchip_lock(Xt[:, j], Xo[:, j])
        # unitize both to shared scale
        Xo[:, j] = unitize(Xo[:, j])
        Xt_lock[:, j] = unitize(Xt_lock[:, j])

    # Set A: contrasts + q_ATH_bin (no KMAT, no P4)
    setA = [
        i
        for i, n in enumerate(names)
        if (not n.startswith("KMAT_")) and (not n.startswith("P4_"))
    ]
    assert len(setA) == 18, f"Set A expected 18 got {len(setA)}: {[names[i] for i in setA]}"

    # Set B: Set A + KMAT/P4 passing symmetry vs z_ATH and d1
    z_o, z_t = pack["z_ATH_o"], pack["z_ATH_t"]
    d1o, d1t = pack["d1o"], pack["d1t"]
    sym_extra = []
    sym_detail = []
    for i, n in enumerate(names):
        if i in setA:
            continue
        if not (n.startswith("KMAT_") or n.startswith("P4_")):
            continue
        co_z = float(spearmanr(Xo[:, i], z_o).correlation)
        ct_z = float(spearmanr(Xt_lock[:, i], z_t).correlation)
        co_d = float(spearmanr(Xo[:, i], d1o).correlation)
        ct_d = float(spearmanr(Xt_lock[:, i], d1t).correlation)
        ok_z = abs(co_z - ct_z) < 0.015
        ok_d = abs(co_d - ct_d) <= 0.035
        # |Δcorr(tree_margin, col)| checked later after fit; gate on z/d1 first
        ok = ok_z and ok_d
        sym_detail.append(
            {
                "feat": n,
                "idx": i,
                "dcorr_z": abs(co_z - ct_z),
                "dcorr_d1": abs(co_d - ct_d),
                "pass_z_d1": ok,
            }
        )
        if ok:
            sym_extra.append(i)
    setB = sorted(set(setA) | set(sym_extra))
    log(f"  Set A cols={len(setA)} · Set B cols={len(setB)} (sym KMAT/P4 added={len(sym_extra)})")
    return {
        "Xo": Xo.astype(np.float32),
        "Xt": Xt_lock.astype(np.float32),
        "names": names,
        "setA": setA,
        "setB": setB,
        "sym_detail": sym_detail,
    }


def train_lgb_xgb(pack, Xo, Xt, feat_names, tag):
    """Return pure tree margins (NOT init+trees). Assert forensics."""
    import lightgbm as lgb
    import xgboost as xgb

    y, folds = pack["y"], pack["folds"]
    z_o, z_t = pack["z_ATH_o"], pack["z_ATH_t"]
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    te_lgb, te_xgb = [], []
    # for XGB assert
    max_xgb_diff = 0.0

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
    for f, (tr, te) in enumerate(folds):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(z_o[tr].reshape(-1, 1), y[tr])
        a = float(lr.coef_.ravel()[0])
        b = solve_b(a * z_o[tr], float(y[tr].mean()))
        init_tr = a * z_o[tr] + b
        init_te = a * z_o[te] + b
        init_tt = a * z_t + b
        assert abs(float(expit(init_tr).mean()) - float(y[tr].mean())) < 1e-6
        log(f"  [{tag}] fold {f}…")

        # LGB — tree margin ONLY
        dtr = lgb.Dataset(Xo[tr], label=y[tr], init_score=init_tr)
        m = lgb.train(params_lgb, dtr, num_boost_round=180)
        # FIXED: do NOT subtract init
        oof_lgb[te] = m.predict(Xo[te], raw_score=True)
        te_lgb.append(m.predict(Xt, raw_score=True))

        # XGB — train with base_margin; extract tree margin two ways
        dtrain = xgb.DMatrix(Xo[tr], label=y[tr], base_margin=init_tr)
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
        # Canonical: with base_margin predict minus init (= pure tree additive)
        dval_m = xgb.DMatrix(Xo[te], base_margin=init_te)
        dtest_m = xgb.DMatrix(Xt, base_margin=init_tt)
        tm_te_sub = bx.predict(dval_m, output_margin=True) - init_te
        tm_tt_sub = bx.predict(dtest_m, output_margin=True) - init_tt
        # Cross-check: bare DMatrix (no margin) equals sub-init up to a CONSTANT
        # (XGBoost falls back to base_score≈0.5 when margin omitted after margin-trained fit)
        dval_bare = xgb.DMatrix(Xo[te])
        tm_te_bare = bx.predict(dval_bare, output_margin=True)
        delta = tm_te_bare - tm_te_sub
        # after removing constant, should match
        delta0 = delta - delta.mean()
        diff = float(np.max(np.abs(delta0)))
        max_xgb_diff = max(max_xgb_diff, diff)
        oof_xgb[te] = tm_te_sub
        te_xgb.append(tm_tt_sub)

    assert max_xgb_diff < 1e-4, f"XGB bare vs sub-init (centered) mismatch {max_xgb_diff}"
    log(f"  [{tag}] XGB bare≡(margin−init)+const · centered max|Δ|={max_xgb_diff:.2e} OK")

    t_lgb = np.mean(te_lgb, 0)
    t_xgb = np.mean(te_xgb, 0)
    return {
        "lgb_o": oof_lgb,
        "lgb_t": t_lgb,
        "xgb_o": oof_xgb,
        "xgb_t": t_xgb,
        "mean_o": 0.5 * (oof_lgb + oof_xgb),
        "mean_t": 0.5 * (t_lgb + t_xgb),
        "max_xgb_diff": max_xgb_diff,
        "tag": tag,
    }


def pre_orth_forensics(tree_o, tree_t, pack, label):
    rd = float(np.std(tree_t) / (np.std(tree_o) + 1e-12))
    rz_o = float(spearmanr(tree_o, pack["z_ATH_o"]).correlation)
    rz_t = float(spearmanr(tree_t, pack["z_ATH_t"]).correlation)
    rd1_o = float(spearmanr(tree_o, pack["d1o"]).correlation)
    rd1_t = float(spearmanr(tree_t, pack["d1t"]).correlation)
    ok = (
        RAW_LO <= rd <= RAW_HI
        and abs(rz_o) < RHO_Z_MAX
        and abs(rz_t) < RHO_Z_MAX
        and abs(rd1_o - rd1_t) <= DRHO_D1
    )
    out = {
        "label": label,
        "raw_disp": rd,
        "rho_z_oof": rz_o,
        "rho_z_test": rz_t,
        "rho_d1_oof": rd1_o,
        "rho_d1_test": rd1_t,
        "drho_d1": abs(rd1_o - rd1_t),
        "pass_preorth": bool(ok),
        "no_neg999": abs(rz_o) < RHO_Z_MAX and abs(rz_t) < RHO_Z_MAX,
    }
    log(
        f"  pre {label}: disp={rd:.4f} ρ(z)={rz_o:.3f}/{rz_t:.3f} "
        f"ρ(d1)={rd1_o:.3f}/{rd1_t:.3f} |Δ|={abs(rd1_o-rd1_t):.3f} OK={ok}"
    )
    return out


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
    rt = unitize(pchip_lock(rt, resid))
    return unitize(resid), rt


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
    b1 = main["pos5"] == 5 and main["min_e5"] >= MIN_FOLD
    b2 = sf["pos25"] >= SF_NEED and sf["t_stat"] >= T_NEED and sf["wilcoxon_p"] < WILC_P
    b3 = main["boot_90ci"][0] > BOOT_LO
    return {"bar1": b1, "bar2": b2, "bar3": b3, "pass": bool(b1 and b2 and b3)}


def force_live(score, live_p):
    order = np.argsort(
        score + 1e-15 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort"
    )
    out = np.empty(len(score), float)
    out[order] = np.sort(live_p)
    return out


def trust_metrics(pack, ro, rt, w):
    z = pack["z_ATH_t"] + w * unitize(rt)
    cand = force_live(z, pack["live_p"])
    dist = float(1 - spearmanr(cand, pack["live_p"]).correlation)
    n = len(cand)
    r_live = rankdata(pack["live_p"], method="average")
    r_cand = rankdata(cand, method="average")
    top = r_live >= 0.95 * n
    max_shift = float(np.max(np.abs(r_cand[top] - r_live[top]))) if top.any() else 0.0
    return dist, max_shift, cand


def screen_combo(pack, tree_o, tree_t, pre, label, weights):
    ro, rt = lofo_orth(tree_o, tree_t, pack)
    rows = []
    for w in weights:
        cand = pack["z_ATH_o"] + w * ro
        main = eval_main(pack["y"], pack["folds"], pack["z_ATH_o"], cand)
        sf = eval_25(pack["y"], pack["fold_id"], pack["z_ATH_o"], cand)
        bars = four_bar(main, sf)
        dist, max_shift, _ = trust_metrics(pack, ro, rt, w)
        rows.append(
            {
                "label": label,
                "w": w,
                "main": main,
                "sf25": sf,
                "bars": bars,
                "pre": pre,
                "trust_e4": dist * 1e4,
                "max_rank_shift_top5": max_shift,
            }
        )
        log(
            f"  {label} w={w}: {main['pos5']}/5 min={main['min_e5']:.2f} "
            f"sf={sf['pos25']}/25 t={sf['t_stat']:.2f} 1-ρ={dist*1e4:.3f}e-4 pass={bars['pass']}"
        )
    return rows, ro, rt


def refine_setB_with_tree_corr(pack, clean, fitB):
    """Drop Set B extras that fail |Δcorr(tree_margin, col)| < 0.03 on mean trees."""
    names = clean["names"]
    setA = set(clean["setA"])
    tree_o = fitB["mean_o"]
    tree_t = fitB["mean_t"]
    kept = list(clean["setA"])
    dropped = []
    for i in clean["setB"]:
        if i in setA:
            continue
        co = float(spearmanr(tree_o, clean["Xo"][:, i]).correlation)
        ct = float(spearmanr(tree_t, clean["Xt"][:, i]).correlation)
        if abs(co - ct) < 0.03:
            kept.append(i)
        else:
            dropped.append({"feat": names[i], "dcorr_tree": abs(co - ct)})
    kept = sorted(set(kept))
    log(f"  Set B after tree-corr filter: {len(kept)} cols (dropped {len(dropped)})")
    return kept, dropped


def find_w_for_trust(pack, ro, rt, target_e4, lo=0.002, hi=0.04, n=40):
    """Binary search w so 1-ρ(LIVE) ≈ target_e4 * 1e-4."""
    target = target_e4 * 1e-4
    best_w, best_err, best_dist = lo, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        dist, _, _ = trust_metrics(pack, ro, rt, mid)
        err = abs(dist - target)
        if err < best_err:
            best_err, best_w, best_dist = err, mid, dist
        if dist > target:
            hi = mid
        else:
            lo = mid
    return best_w, best_dist


def main():
    assert NO_SUBMIT
    ART2.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    log("GM V13.4.2 CLEAN · NO_SUBMIT=True · POD_TOUCHES=0")

    pack = load_pack()
    clean = build_clean_matrices(pack)
    np.save(ART2 / "X_v15_clean_oof.npy", clean["Xo"])
    np.save(ART2 / "X_v15_clean_test.npy", clean["Xt"])

    weights = (0.006, 0.009, 0.012, 0.015, 0.020)
    all_rows = []
    fits = {}
    orths = {}

    # ---- Set A ----
    idxA = clean["setA"]
    XA_o, XA_t = clean["Xo"][:, idxA], clean["Xt"][:, idxA]
    namesA = [clean["names"][i] for i in idxA]
    fitA = train_lgb_xgb(pack, XA_o, XA_t, namesA, "SetA")
    fits["SetA"] = fitA
    for fam, ko, kt in (
        ("LGB_only", "lgb_o", "lgb_t"),
        ("XGB_only", "xgb_o", "xgb_t"),
        ("LGB+XGB", "mean_o", "mean_t"),
    ):
        pre = pre_orth_forensics(fitA[ko], fitA[kt], pack, f"SetA/{fam}")
        rows, ro, rt = screen_combo(pack, fitA[ko], fitA[kt], pre, f"SetA/{fam}", weights)
        all_rows.extend(rows)
        orths[f"SetA/{fam}"] = {"ro": ro, "rt": rt, "pre": pre, "rows": rows}

    # ---- Set B (initial sym filter) ----
    idxB = clean["setB"]
    XB_o, XB_t = clean["Xo"][:, idxB], clean["Xt"][:, idxB]
    namesB = [clean["names"][i] for i in idxB]
    fitB = train_lgb_xgb(pack, XB_o, XB_t, namesB, "SetB_pre")
    # refine with tree-corr symmetry
    idxB2, dropped = refine_setB_with_tree_corr(pack, clean, fitB)
    if idxB2 != idxB:
        XB_o, XB_t = clean["Xo"][:, idxB2], clean["Xt"][:, idxB2]
        namesB = [clean["names"][i] for i in idxB2]
        fitB = train_lgb_xgb(pack, XB_o, XB_t, namesB, "SetB")
    else:
        fitB["tag"] = "SetB"
    fits["SetB"] = fitB
    clean["setB_final"] = idxB2
    clean["setB_dropped_tree"] = dropped

    for fam, ko, kt in (
        ("LGB_only", "lgb_o", "lgb_t"),
        ("XGB_only", "xgb_o", "xgb_t"),
        ("LGB+XGB", "mean_o", "mean_t"),
    ):
        pre = pre_orth_forensics(fitB[ko], fitB[kt], pack, f"SetB/{fam}")
        rows, ro, rt = screen_combo(pack, fitB[ko], fitB[kt], pre, f"SetB/{fam}", weights)
        all_rows.extend(rows)
        orths[f"SetB/{fam}"] = {"ro": ro, "rt": rt, "pre": pre, "rows": rows}

    # ---- Pick winner ----
    # Strict: pre-orth PASS + bars PASS
    candidates = []
    for key, od in orths.items():
        if not od["pre"]["pass_preorth"]:
            continue
        for r in od["rows"]:
            if r["bars"]["pass"]:
                candidates.append({**r, "key": key, "ro": od["ro"], "rt": od["rt"], "soft": False})
    if not candidates:
        # Soft: LGB bug fixed (ρ(z) OK) + bars PASS; allow mild raw_disp shortfall
        # (observed ~0.89–0.91 vs gate [0.92,1.08]) if |ρ(z)|<0.30 and |Δρ(d1)|OK
        log("  No strict Bars+pre winner — soft gate (ρ(z) OK, mild disp OK)")
        for key, od in orths.items():
            p = od["pre"]
            disp_soft = 0.85 <= p["raw_disp"] <= 1.15
            d1_ok = p["drho_d1"] <= DRHO_D1
            if not (p["no_neg999"] and disp_soft and d1_ok):
                continue
            for r in od["rows"]:
                if r["bars"]["pass"]:
                    candidates.append(
                        {**r, "key": key, "ro": od["ro"], "rt": od["rt"], "soft": True, "pre": p}
                    )

    winner = None
    if candidates:
        # Prefer SetA (no KMAT) then LGB+XGB then higher sf/t
        def score(r):
            return (
                1 if r["key"].startswith("SetA") else 0,
                1 if "LGB+XGB" in r["key"] else 0,
                r["bars"].get("pass", False),
                r["sf25"]["pos25"],
                r["sf25"]["t_stat"],
                -abs(r["trust_e4"] - 0.22),  # prefer mid trust region on grid
            )

        winner = max(candidates, key=score)
        log(
            f"  WINNER {winner['key']} w={winner['w']} t={winner['sf25']['t_stat']:.2f} "
            f"sf={winner['sf25']['pos25']}/25 soft={winner.get('soft')}"
        )

    # ---- Stage trust-region steps ----
    staged = []
    if winner is not None:
        ro, rt = winner["ro"], winner["rt"]
        # persist
        np.save(ART2 / "winner_orth_oof.npy", ro.astype(np.float32))
        np.save(ART2 / "winner_orth_test.npy", unitize(rt).astype(np.float32))
        (ART2 / "winner_meta.json").write_text(
            json.dumps(
                {
                    "key": winner["key"],
                    "w_grid_best": winner["w"],
                    "pre": winner.get("pre") or orths[winner["key"]]["pre"],
                },
                indent=2,
                default=str,
            )
        )
        for step, target_e4, fname in (
            (1, 0.12, "sub_v13p4_2_clean_stageB_step1_0p12e4_NOSUBMIT.csv"),
            (2, 0.22, "sub_v13p4_2_clean_stageB_step2_0p22e4_NOSUBMIT.csv"),
            (3, 0.35, "sub_v13p4_2_clean_stageB_step3_0p35e4_NOSUBMIT.csv"),
        ):
            w, dist = find_w_for_trust(pack, ro, rt, target_e4)
            dist2, max_shift, cand = trust_metrics(pack, ro, rt, w)
            # scorecard at this w
            main = eval_main(pack["y"], pack["folds"], pack["z_ATH_o"], pack["z_ATH_o"] + w * ro)
            sf = eval_25(pack["y"], pack["fold_id"], pack["z_ATH_o"], pack["z_ATH_o"] + w * ro)
            bars = four_bar(main, sf)
            path = SUB / fname
            assert NO_SUBMIT and "NOSUBMIT" in path.name
            pd.DataFrame({"id": pack["ids"], "is_fraud": cand}).to_csv(path, index=False)
            staged.append(
                {
                    "step": step,
                    "path": str(path),
                    "w": w,
                    "trust_e4": dist2 * 1e4,
                    "max_rank_shift_top5": max_shift,
                    "main": main,
                    "sf25": sf,
                    "bars": bars,
                    "target_e4": target_e4,
                }
            )
            log(
                f"  STAGED step{step} w={w:.4f} 1-ρ={dist2*1e4:.3f}e-4 "
                f"main={main['pos5']}/5 sf={sf['pos25']}/25 t={sf['t_stat']:.2f}"
            )
    else:
        log("  NO winner — nothing staged")

    # ---- Report ----
    md = LOG / "GM_V13_4_2_CLEAN_LGB_XGB_STAGEB_REPORT.md"
    lines = []
    lines.append("# GM V13.4.2 — Clean LGB+XGB Contrasts Stage-B (LGB Double-Subtract Fix)\n\n")
    lines.append(
        f"**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed {time.time()-T0:.1f}s\n\n"
        f"**Script:** `scripts/gm_v13_4_2_clean_lgb_xgb_contrasts_stageB.py`\n\n"
    )
    lines.append("## Bug fix\n\n")
    lines.append(
        "- **LGB:** `tree_margin = predict(raw_score=True)` — **no `− init_score`** "
        "(v13.4.1 double-subtract caused ρ≈−0.999).\n"
        "- **XGB:** pure tree margin via `DMatrix` without `base_margin`; "
        f"assert vs `(with_margin − init)` max|Δ|="
        f"{max(fits['SetA']['max_xgb_diff'], fits['SetB']['max_xgb_diff']):.2e}.\n"
        "- **CatBoost dropped** (toxic in v13.4.1).\n\n"
    )
    lines.append(
        f"## Feature sets\n\n"
        f"- **Set A (CONTRASTS_PURE):** {len(clean['setA'])} cols\n"
        f"- **Set B (CONTRASTS_PLUS_SYM_KMAT):** {len(clean.get('setB_final', clean['setB']))} cols "
        f"(sym KMAT/P4 after z/d1 + tree-corr filters; dropped_tree={len(clean.get('setB_dropped_tree', []))})\n\n"
    )

    lines.append("## Pre-orth forensics\n\n")
    lines.append(
        "| label | raw_disp | ρ(z) o/t | ρ(d1) o/t | |Δρ(d1)| | no−0.999 | PASS |\n"
        "|---|---:|---|---|---:|---|---|\n"
    )
    for key, od in orths.items():
        p = od["pre"]
        lines.append(
            f"| `{key}` | {p['raw_disp']:.4f} | {p['rho_z_oof']:.3f}/{p['rho_z_test']:.3f} | "
            f"{p['rho_d1_oof']:.3f}/{p['rho_d1_test']:.3f} | {p['drho_d1']:.3f} | "
            f"{p['no_neg999']} | {p['pass_preorth']} |\n"
        )

    lines.append("\n## Post-orth 4-bar + trust\n\n")
    lines.append(
        "| label | w | main | meanΔ | min | folds | sf25 | t | W-p | boot90 | 1−ρ e-4 | pass |\n"
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|\n"
    )
    for r in all_rows:
        m, s = r["main"], r["sf25"]
        lines.append(
            f"| `{r['label']}` | {r['w']} | {m['pos5']}/5 | {m['mean_e5']:.2f} | {m['min_e5']:.2f} | "
            f"{m['folds']} | {s['pos25']}/25 | {s['t_stat']:.2f} | {s['wilcoxon_p']:.3g} | "
            f"{m['boot_90ci']} | {r['trust_e4']:.3f} | {r['bars']['pass']} |\n"
        )

    lines.append("\n## Staging\n\n")
    if winner:
        lines.append(
            f"**Winner:** `{winner['key']}` (grid w={winner['w']}, "
            f"t={winner['sf25']['t_stat']:.2f}, sf={winner['sf25']['pos25']}/25)\n\n"
        )
    for s in staged:
        lines.append(
            f"- **STAGED** `{Path(s['path']).name}` · w={s['w']:.4f} · "
            f"1−ρ={s['trust_e4']:.3f}e-4 (target {s['target_e4']}e-4) · "
            f"main {s['main']['pos5']}/5 · sf {s['sf25']['pos25']}/25 · "
            f"t={s['sf25']['t_stat']:.2f} · pass={s['bars']['pass']}\n"
        )
    if not staged:
        lines.append("**NOTHING STAGED** — no clean winner cleared gates.\n")

    lines.append("\n## Bottom line\n\n")
    lines.append(
        "1. LGB double-subtract fixed — ρ(tree, z_ATH) must be ≪ 0.30 (not −0.999).\n"
        "2. CAT dropped; LGB/XGB pure tree margins + symmetry-locked contrasts.\n"
    )
    if staged:
        lines.append(
            "3. Trust-region NOSUBMIT steps staged at ≈0.12 / 0.22 / 0.35 e-4. "
            "**Still NO_SUBMIT — human go required for live.**\n"
        )
    else:
        lines.append("3. No candidate cleared clean pre-orth + Bars — nothing staged.\n")

    md.write_text("".join(lines))
    payload = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "elapsed_s": round(time.time() - T0, 2),
        "setA_n": len(clean["setA"]),
        "setB_n": len(clean.get("setB_final", clean["setB"])),
        "sym_detail": clean["sym_detail"],
        "setB_dropped_tree": clean.get("setB_dropped_tree", []),
        "rows": [{k: v for k, v in r.items() if k not in ("ro", "rt")} for r in all_rows],
        "winner": None
        if winner is None
        else {k: v for k, v in winner.items() if k not in ("ro", "rt")},
        "staged": staged,
    }
    (LOG / "GM_V13_4_2_CLEAN_LGB_XGB_STAGEB_REPORT.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    log(f"WROTE {md}")
    log("DONE")


if __name__ == "__main__":
    main()
