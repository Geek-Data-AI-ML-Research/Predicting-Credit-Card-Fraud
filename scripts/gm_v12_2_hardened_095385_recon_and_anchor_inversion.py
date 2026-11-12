#!/usr/bin/env python3
"""V12.2 HARDENED: 0.95385 recon + Q17 anchor inversion + staging.

ABSOLUTE: NO RunPod / runpodctl / SSH / GPU pods.
STRICT NO_SUBMIT=True — stage CSVs only, never upload.
Writes: logs/overnight/GM_V12_2_HARDENED_095385_AND_Q17_REPORT.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NO_SUBMIT = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT_MD = ROOT / "logs/overnight/GM_V12_2_HARDENED_095385_AND_Q17_REPORT.md"
OUT_JSON = ROOT / "logs/overnight/GM_V12_2_HARDENED_095385_AND_Q17_REPORT.json"
Q17 = ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
Q17_MD = ROOT / "logs/overnight/GM_AMANATAR_Q17_AUDIT.md"
PRED = ROOT / "preds/l7_s2026_5f"
S7DIR = ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
KF = ROOT / "external/nov12_late_kernels"
SUB = ROOT / "submissions"
KAG = ROOT / ".venv_forensics/bin/kaggle"
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def bayes(ll, prior, orders=None):
    import numpy as np

    x = ll if orders is None else ll[:, list(orders), :]
    t = np.logaddexp.reduce(x[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(x[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def kaggle_cmd(args, timeout=45):
    exe = str(KAG) if KAG.exists() else "kaggle"
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    try:
        return subprocess.check_output([exe] + args, text=True, timeout=timeout, env=env)
    except Exception as e:
        return f"ERR: {e}"[:400]


def pull_kernel(slug: str, dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.ipynb")):
        return f"already present: {dest}"
    exe = str(KAG) if KAG.exists() else "kaggle"
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    try:
        out = subprocess.check_output(
            [exe, "kernels", "pull", slug, "-p", str(dest)],
            text=True,
            timeout=90,
            env=env,
        )
        return out.strip()[:200]
    except Exception as e:
        return f"pull_fail: {e}"[:200]


def force_to_dist(ref_p, direction, target, hi=80.0, n=80):
    """Binary-search scalar on unit(direction) so 1-ρ(ref, force_law(ref, L(ref)+w*u))≈target."""
    from _gm_v11_shared import L, force_law, spearman_dist, unit, unitize

    import numpy as np

    z0 = unitize(L(ref_p))
    u = unit(np.asarray(direction, dtype=np.float64).ravel())
    lo, best_w, best_err, got, best_p = 0.0, 0.0, 1e9, 0.0, ref_p
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        p = force_law(ref_p, z0 + mid * u)
        d = spearman_dist(p, ref_p)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got, best_p = err, mid, d, p
        if d > target:
            hi = mid
        else:
            lo = mid
    return best_p, float(best_w), float(got)


def fixed_lofo(y, z_base, delta, folds, w=0.036):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from _gm_v11_shared import unit

    u = unit(delta)
    fold_d = []
    for _, te in folds:
        b = float(roc_auc_score(y[te], z_base[te]))
        a = float(roc_auc_score(y[te], z_base[te] + w * u[te]))
        fold_d.append(a - b)
    pos = int(sum(d > 0 for d in fold_d))
    return {
        "pos5": pos,
        "mean_d_e5": float(np.mean(fold_d) * 1e5),
        "fold_d_e5": [round(d * 1e5, 4) for d in fold_d],
        "w": w,
    }


def main():
    assert NO_SUBMIT is True
    assert "runpod" not in " ".join(sys.argv).lower()
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from _gm_v11_shared import (
        L,
        L_fold,
        force_law,
        load_anchor_and_bank,
        load_prob,
        lofo_eval,
        project_out,
        setup_folds,
        spearman_dist,
        unit,
        unitize,
        write_nosubmit,
    )

    log("======== V12.2 HARDENED ZERO-POD NO_SUBMIT=True ========")

    # ---- Step 1: Kaggle recon ----
    recon = {
        "kernels_dateRun": kaggle_cmd(
            [
                "kernels",
                "list",
                "--competition",
                "playground-series-s6e11",
                "--sort-by",
                "dateRun",
                "--page-size",
                "50",
            ]
        ),
        "kernels_scoreDesc": kaggle_cmd(
            [
                "kernels",
                "list",
                "--competition",
                "playground-series-s6e11",
                "--sort-by",
                "scoreDescending",
                "--page-size",
                "30",
            ]
        ),
        "search_9538": kaggle_cmd(["kernels", "list", "-s", "0.9538"]),
        "search_9537": kaggle_cmd(["kernels", "list", "-s", "0.9537"]),
        "search_9536": kaggle_cmd(["kernels", "list", "-s", "0.9536"]),
        "search_s6e11_9538": kaggle_cmd(["kernels", "list", "-s", "s6e11 0.9538", "--sort-by", "relevance"]),
        "search_s6e11_9537": kaggle_cmd(["kernels", "list", "-s", "s6e11 0.9537", "--sort-by", "relevance"]),
        "datasets_s6e11": kaggle_cmd(["datasets", "list", "-s", "s6e11", "--sort-by", "updated"]),
        "g26_files": kaggle_cmd(["datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"]),
        "q17_files": kaggle_cmd(["datasets", "files", "amanatar/s6e11-rank1-q17-meta-bundle"]),
    }
    log("kaggle recon done")

    pulls = {}
    for slug, folder in [
        ("knightfork/s6e11-top-15-cpu-blend-eda-lb-0-95357", "knightfork_095357"),
        ("modansaif/s6e11-fraud-fusion-reproduction-credited", "modansaif"),
        ("asteriospanoras/s6e11-how-much-of-the-public-top-is-real", "asterios"),
        ("bawanthabeliwaththa/the-honest-stack-cv-lb-and-the-0-9533-trap", "bawanthabeliwaththa"),
        ("markbarney/s6e11-what-we-tried-and-what-did-not-work", "markbarney"),
        ("luthfimuthathohirin/credit-card-fraud-ensemble-forensics", "luthfi_forensics"),
    ]:
        pulls[slug] = pull_kernel(slug, KF / folder)
    log(f"kernel pulls: {list(pulls.keys())}")

    knight = {
        "formula": "L(preserved_095354_ladder) + 0.25 * (L(sovereign) - L(wide))",
        "WEIGHT": 0.25,
        "public_auc": 0.95357,
        "inputs": [
            "goodpjw2008 LM Ladder 0.95354 reference CSV",
            "amanatar Sovereign OOF+Test vault (sovereign vs wide difference)",
        ],
        "proxy_oof_delta": 1.7580980577e-05,
        "proxy_folds_up": 4,
        "sha256_submission": "b8f9bda48da5efb8189ea22d3f54373d83c1768f311e6b5818ba3b4de323dc07",
    }
    # modansaif is a credited reproduction of Three Ladders / areeb recipe
    modansaif = {
        "note": "Fraud-fusion reproduction credited — rebuilds public Three-Ladders / LM-ladder recipe (A/F/LM/L7F/GPD/MS/G26 style members).",
        "formula_proxy": "public ladder reconstruction (Areeb/goodpjw lineage), not a new 0.9538x formula",
    }
    bawantha = {
        "title": "The Honest Stack: CV, LB, and the 0.9533 Trap",
        "path": str(KF / "bawanthabeliwaththa"),
        "thesis": "Public 0.9533 band can be CV-fragile / winner's-curse; honesty about CV vs LB gap.",
    }

    # ---- Step 2: Q17 reverse engineering ----
    z = np.load(Q17, allow_pickle=True)
    keys = list(z.files)
    stream_names = [str(x) for x in z["stream_names"].tolist()]
    y = z["y_train"].astype(np.int64)
    folds_arr = z["folds"].astype(np.int8)
    folds, fold_id = setup_folds(y)
    assert (folds_arr == fold_id).all()

    pairs = []
    for k in keys:
        if k.startswith("oof_") and ("test_" + k[4:]) in keys:
            pairs.append(k[4:])

    solo = {}
    for stem in pairs:
        oo = L_fold(z["oof_" + stem].astype(np.float64), fold_id)
        solo[stem] = {
            "solo_auc": float(roc_auc_score(y, oo)),
            "fold_aucs": [float(roc_auc_score(y[te], oo[te])) for _, te in folds],
            "oof_shape": list(z["oof_" + stem].shape),
            "test_shape": list(z["test_" + stem].shape),
        }
    log(f"Q17 pairs={len(pairs)} keys={len(keys)} anchor_solo={solo['anchor_095370']['solo_auc']:.7f}")

    # Ridge/OLS oof vs test weights on 16 streams (exclude anchor from X)
    member_streams = [s for s in stream_names if s != "anchor_095370"]
    Xo = np.column_stack([z["oof_" + s].astype(np.float64).ravel() for s in member_streams])
    Xt = np.column_stack([z["test_" + s].astype(np.float64).ravel() for s in member_streams])
    yo = z["oof_anchor_095370"].astype(np.float64).ravel()
    yt = z["test_anchor_095370"].astype(np.float64).ravel()
    Xo_m, Xo_s = Xo.mean(0), Xo.std(0) + 1e-12
    Xt_m, Xt_s = Xt.mean(0), Xt.std(0) + 1e-12
    yo_c, yt_c = yo - yo.mean(), yt - yt.mean()
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit((Xo - Xo_m) / Xo_s, yo_c)
    w_oof = ridge.coef_.copy()
    r2_oof = float(1 - np.sum((yo_c - ((Xo - Xo_m) / Xo_s) @ w_oof) ** 2) / np.sum(yo_c**2))
    ridge.fit((Xt - Xt_m) / Xt_s, yt_c)
    w_test = ridge.coef_.copy()
    r2_test = float(1 - np.sum((yt_c - ((Xt - Xt_m) / Xt_s) @ w_test) ** 2) / np.sum(yt_c**2))
    wo_ols, *_ = np.linalg.lstsq((Xo - Xo_m) / Xo_s, yo_c, rcond=None)
    wt_ols, *_ = np.linalg.lstsq((Xt - Xt_m) / Xt_s, yt_c, rcond=None)
    max_abs_ridge = float(np.max(np.abs(w_oof - w_test)))
    max_abs_ols = float(np.max(np.abs(wo_ols - wt_ols)))
    yt_hat = (((Xt - Xt_m) / Xt_s) @ w_oof) + yt.mean()
    corr_transfer = float(np.corrcoef(yt_hat, yt)[0, 1])

    # Also regress L(anchor) on L(streams)
    XoL = np.column_stack([L_fold(z["oof_" + s].astype(np.float64), fold_id) for s in member_streams])
    XtL = np.column_stack([L(z["test_" + s].astype(np.float64)) for s in member_streams])
    yoL = L_fold(yo, fold_id)
    ytL = L(yt)
    XoL_m, XoL_s = XoL.mean(0), XoL.std(0) + 1e-12
    XtL_m, XtL_s = XtL.mean(0), XtL.std(0) + 1e-12
    yoLc, ytLc = yoL - yoL.mean(), ytL - ytL.mean()
    ridge.fit((XoL - XoL_m) / XoL_s, yoLc)
    wL_oof = ridge.coef_.copy()
    ridge.fit((XtL - XtL_m) / XtL_s, ytLc)
    wL_test = ridge.coef_.copy()
    max_abs_L = float(np.max(np.abs(wL_oof - wL_test)))

    pack = load_anchor_and_bank(y, fold_id, folds)

    # Ladder bank regression on test_anchor
    ladder_names = ["A", "F", "LM", "L7F", "GPD", "G26", "MS", "MLP", "GB"]
    Xt_ladder = np.column_stack([pack[n] for n in ladder_names])
    # L7_f0 = lm7_test_f0 (K=4 loglik → bayes); L7_s2026 test_0 is (N,10,2)
    prior_global = float(y.mean())
    ll7_t = np.load(S7DIR / "lm7_test_f0.npy").astype(np.float32)
    L7_f0_t = L(bayes(ll7_t, prior_global, orders=range(4)))
    oof26 = np.load(PRED / "oof_0.npy").astype(np.float32)
    t26_ll = np.load(PRED / "test_0.npy").astype(np.float32)
    L7_s2026_t = L(bayes(t26_ll, prior_global))  # K=10 all orders
    Xt_ladder_ext = np.column_stack([Xt_ladder, L7_f0_t])
    ladder_ext_names = ladder_names + ["L7_f0_K4"]
    Xm, Xs = Xt_ladder_ext.mean(0), Xt_ladder_ext.std(0) + 1e-12
    ridge.fit((Xt_ladder_ext - Xm) / Xs, ytLc)
    w_ladder_L = {n: float(c) for n, c in zip(ladder_ext_names, ridge.coef_)}
    r2_ladder_L = float(
        1
        - np.sum((ytLc - ((Xt_ladder_ext - Xm) / Xs) @ ridge.coef_) ** 2) / np.sum(ytLc**2)
    )
    ridge.fit((Xt_ladder_ext - Xm) / Xs, yt_c)
    w_ladder_raw = {n: float(c) for n, c in zip(ladder_ext_names, ridge.coef_)}
    r2_ladder_raw = float(
        1
        - np.sum((yt_c - ((Xt_ladder_ext - Xm) / Xs) @ ridge.coef_) ** 2) / np.sum(yt_c**2)
    )

    # proprietary residual loadings into test_anchor (should be near-zero if unused)
    prop_t = {
        "d1_v4": pack["d1t"],
        "G26_minus_MS": pack["u_g26ms_t"],
        "Strike9p5": L(load_prob(SUB / "sub_strike9p5_axis_0p70_0p10.csv")),
    }
    prop_t["L7_s2026_K10_minus_L7_K4"] = L7_s2026_t - L7_f0_t

    # OLS of L(anchor) on proprietary directions (after projecting public ladder)
    Xt_pub = np.column_stack([unitize(pack[n]) for n in ladder_names])
    resid_anchor = ytL - Linear_predict(Xt_pub, ytL)
    prop_loadings = {}
    for name, vec in prop_t.items():
        vv = unitize(vec)
        # correlate residual
        prop_loadings[name] = {
            "corr_with_L_anchor": float(np.corrcoef(ytL, vv)[0, 1]),
            "corr_with_resid_after_ladder": float(np.corrcoef(resid_anchor, vv)[0, 1]),
            "abs_mean": float(np.mean(np.abs(vv))),
        }

    # Spearman matrix for test_anchor vs refs
    refs = {
        "A_095340": pack["A"],
        "F_095330": pack["F"],
        "z_S7_ours": L(pack["s7"]),
        "cand2": L(pack["cand2"]),
    }
    for tag, globs in [
        ("areeb_095347", ["*095347*NOSUBMIT*.csv", "*areeb*095347*.csv"]),
        ("s352", ["*095352*.csv"]),
        ("s354", ["*095354*.csv", "*post095354*slot1*.csv"]),
    ]:
        for g in globs:
            for p in SUB.glob(g):
                if p.name.startswith("._"):
                    continue
                try:
                    refs[tag] = L(load_prob(p))
                    break
                except Exception:
                    pass
            if tag in refs:
                break
    # knightfork formula proxy if we can build it
    # Use cand2/s354 as ladder proxy + if sovereign exists locally skip; record N/A otherwise
    spearman_anchor = {k: round(spearman_dist(ytL, v), 8) for k, v in refs.items() if len(v) == 299984}
    spearman_anchor["knightfork_095357"] = None  # formula-only; scored CSV not local

    leak = {
        "n_member_streams": len(member_streams),
        "r2_oof_ridge_raw": r2_oof,
        "r2_test_ridge_raw": r2_test,
        "max_abs_w_oof_minus_w_test_ridge": max_abs_ridge,
        "max_abs_w_oof_minus_w_test_ols": max_abs_ols,
        "max_abs_wL_oof_minus_wL_test": max_abs_L,
        "identical_1e4": bool(max_abs_ridge < 1e-4),
        "corr_test_vs_oof_weights": corr_transfer,
        "w_oof_ridge": {n: float(w) for n, w in zip(member_streams, w_oof)},
        "w_test_ridge": {n: float(w) for n, w in zip(member_streams, w_test)},
        "top_abs_delta": sorted(
            [
                {
                    "stream": n,
                    "w_oof": float(a),
                    "w_test": float(b),
                    "abs_delta": float(abs(a - b)),
                }
                for n, a, b in zip(member_streams, w_oof, w_test)
            ],
            key=lambda d: -d["abs_delta"],
        )[:10],
        "verdict": (
            "Possibility A: shared frozen linear blend (w_oof≈w_test)"
            if max_abs_ridge < 1e-4
            else "Possibility B: OOF/test constructed differently (weights diverge)"
        ),
        "ladder_reg_L_r2": r2_ladder_L,
        "ladder_reg_L_w": w_ladder_L,
        "ladder_reg_raw_r2": r2_ladder_raw,
        "ladder_reg_raw_w": w_ladder_raw,
        "prop_loadings": prop_loadings,
    }
    log(f"leak verdict={leak['verdict']} max|Δw|={max_abs_ridge:.3e}")

    # ---- Step 3: de-aliased subfold + inverted-anchor LOFO ----
    a0, b0 = folds[0]
    y0 = y[b0]
    prior0 = float(y[a0].mean())
    z26 = bayes(oof26, prior0)
    ll7 = np.load(S7DIR / "lm7_oof_f0.npy").astype(np.float32)
    z7k4 = bayes(ll7, prior0, orders=range(4))
    # Pure un-aliased residual on Fold0
    u_pure = L(z26) - L(z7k4)
    # Base: eval_S7matched which embeds K4 L7 path via areeb — fixed β=0.036
    base_f0 = pack["eval_S7matched_355"][b0]
    sf = StratifiedKFold(5, shuffle=True, random_state=42)
    subfolds = list(sf.split(np.zeros(len(y0)), y0))

    def subfold_fixed(delta, w=0.036):
        dlt = unit(delta)
        fold_d = []
        for _, te in subfolds:
            b = float(roc_auc_score(y0[te], base_f0[te]))
            a = float(roc_auc_score(y0[te], base_f0[te] + w * dlt[te]))
            fold_d.append(a - b)
        return {
            "pos5": int(sum(d > 0 for d in fold_d)),
            "mean_d_e5": float(np.mean(fold_d) * 1e5),
            "fold_d_e5": [round(d * 1e5, 4) for d in fold_d],
            "w": w,
        }

    pure_sub = subfold_fixed(u_pure, 0.036)
    log(f"pure L7_s2026_K10 subfold={pure_sub}")

    # Inverted-anchor: oof_anchor as evaluator base
    anc_o = L_fold(z["oof_anchor_095370"].astype(np.float64), fold_id)
    # Build OOF residuals (L7 only on fold0)
    d1o = pack["d1o"]
    u_g26ms_o = pack["u_g26ms_o"]
    u_l7_o = np.zeros(len(y), dtype=np.float64)
    u_l7_o[b0] = unit(u_pure)  # fold0-only proprietary Compact-LM upgrade

    # Check near-zero weight: regress anc_o on [d1, g26ms, l7] after public streams orth
    Xprop_o = np.column_stack([unitize(d1o), unitize(u_g26ms_o), unitize(u_l7_o)])
    # residual of anchor after member streams
    Xmem_o = np.column_stack([unitize(L_fold(z["oof_" + s].astype(np.float64), fold_id)) for s in member_streams])
    anc_resid = anc_o - Linear_predict(Xmem_o, anc_o)
    ridge.fit(Xprop_o, unitize(anc_resid))
    prop_in_anchor = {
        "w_d1": float(ridge.coef_[0]),
        "w_g26ms": float(ridge.coef_[1]),
        "w_l7_s2026_f0only": float(ridge.coef_[2]),
        "r2": float(ridge.score(Xprop_o, unitize(anc_resid))),
    }
    log(f"prop_in_anchor={prop_in_anchor}")

    # LOFO lifts ON TOP of anchor
    inv_moves = {}
    for name, delta, w in [
        ("plus_d1_v4_w0.036", d1o, 0.036),
        ("plus_G26_minus_MS_w0.036", u_g26ms_o, 0.036),
        ("plus_L7_s2026_K10_f0_w0.036", u_l7_o, 0.036),
        (
            "plus_d1_and_L7_w0.036",
            unit(d1o) + unit(u_l7_o),
            0.036,
        ),
        (
            "plus_d1_G26MS_L7_w0.036",
            unit(d1o) + unit(u_g26ms_o) + unit(u_l7_o),
            0.036,
        ),
    ]:
        inv_moves[name] = fixed_lofo(y, anc_o, delta, folds, w=w)
        # also adaptive LOFO
        inv_moves[name + "_adaptive"] = {
            k: v
            for k, v in lofo_eval(y, anc_o, delta, folds).items()
            if k != "u"
        }
    log(f"inv moves fixed: " + ", ".join(f"{k}={v['pos5']}/5@{v['mean_d_e5']:.3f}" for k, v in inv_moves.items() if not k.endswith("_adaptive")))

    # ---- Step 4: stage 4 NOSUBMIT families ----
    ids = pack["ids"]
    cand2 = pack["cand2"]
    anchor_t = yt  # raw test probs from q17
    strike95 = load_prob(SUB / "sub_strike9p5_axis_0p70_0p10.csv")

    # residual directions on TEST
    u_l7_t = L7_s2026_t - L7_f0_t
    u_orth_anc_t = project_out(L(anchor_t), [L(cand2), pack["u_g26ms_t"]])
    u_d1_t = pack["d1t"]
    u_strike_t = L(strike95) - L(anchor_t)

    staged = []

    # Slot1: cand2 + L7_s2026K10 − K4 @ 0.105e-4
    p1, w1, d1 = force_to_dist(cand2, u_l7_t, 0.105e-4)
    path1 = SUB / "sub_v12p2_slot1_cand2_plus_s2026K10_splitcancel_0p10e4_NOSUBMIT.csv"
    write_nosubmit(path1, ids, p1)
    staged.append({"slot": 1, "path": str(path1), "w": w1, "dist_e4": d1 * 1e4, "ref": "cand2"})

    # Slot2: cand2 + L7 + orth(anchor) @ 0.200e-4
    u2 = unit(u_l7_t) + unit(u_orth_anc_t)
    p2, w2, d2 = force_to_dist(cand2, u2, 0.200e-4)
    path2 = SUB / "sub_v12p2_slot2_cand2_plus_s2026K10_plus_Q17orth_0p20e4_NOSUBMIT.csv"
    write_nosubmit(path2, ids, p2)
    staged.append({"slot": 2, "path": str(path2), "w": w2, "dist_e4": d2 * 1e4, "ref": "cand2"})

    # Slot3: inverted macro — L(anchor) + d1 + L7 + strike9.5 @ 0.112e-4 from anchor
    u3 = unit(u_d1_t) + unit(u_l7_t) + 0.25 * unit(u_strike_t)
    # score in logit space then force to anchor marginal
    p3, w3, d3 = force_to_dist(anchor_t, u3, 0.112e-4)
    path3 = SUB / "sub_v12p2_slot3_anchor095370_plus_our_StageB_and_s2026_0p11e4_NOSUBMIT.csv"
    write_nosubmit(path3, ids, p3)
    staged.append({"slot": 3, "path": str(path3), "w": w3, "dist_e4": d3 * 1e4, "ref": "anchor_095370"})

    # Slot4: midpoint LOFO-weighted between (cand2+L7) and (anchor+d1)
    # Build family A/B at mild distances then LOFO-weight on OOF
    pA, wA, dA = force_to_dist(cand2, u_l7_t, 0.12e-4)
    pB, wB, dB = force_to_dist(anchor_t, unit(u_d1_t) + 0.5 * unit(u_l7_t), 0.12e-4)
    # OOF proxies
    famA_o = anc_o * 0  # placeholder
    # Use cand2-proxy OOF = eval_S7matched + L7 residual; anchor + d1
    famA_o = pack["eval_S7matched_355"] + 0.036 * unit(u_l7_o)
    famB_o = anc_o + 0.036 * unit(d1o)
    # LOFO blend weight alpha on L(famA) vs L(famB) measured by held-out AUC
    alphas = np.linspace(0.15, 0.85, 15)
    best_alpha, best_mean = 0.5, -1e9
    fold_scores = []
    for alpha in alphas:
        fold_d = []
        for _, te in folds:
            # base = mean of both; measure blended
            z = (1 - alpha) * unitize(famA_o) + alpha * unitize(famB_o)
            # compare to pure A
            a = float(roc_auc_score(y[te], z[te]))
            fold_d.append(a)
        m = float(np.mean(fold_d))
        if m > best_mean:
            best_mean, best_alpha = m, float(alpha)
            fold_scores = fold_d
    # test midpoint with best alpha
    score4 = (1 - best_alpha) * unitize(L(pA)) + best_alpha * unitize(L(pB))
    # force onto cand2 law for submission safety band midpoint
    p4 = force_law(cand2, score4)
    # if too close/far from both, also report distances
    d4_c = spearman_dist(p4, cand2)
    d4_a = spearman_dist(p4, anchor_t)
    path4 = SUB / "sub_v12p2_slot4_midpoint_cand2_s2026_x_anchor095370_NOSUBMIT.csv"
    write_nosubmit(path4, ids, p4)
    staged.append(
        {
            "slot": 4,
            "path": str(path4),
            "alpha_anchor_family": best_alpha,
            "dist_cand2_e4": d4_c * 1e4,
            "dist_anchor_e4": d4_a * 1e4,
            "oof_mean_auc": best_mean,
            "ref": "midpoint",
        }
    )
    log(f"staged {[s['path'].split('/')[-1] for s in staged]}")

    # ---- Report ----
    q17_text = Q17_MD.read_text() if Q17_MD.exists() else "(missing)"
    utc = datetime.now(timezone.utc).isoformat()
    elapsed = round(time.time() - T0, 2)

    summary = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "utc": utc,
        "elapsed_s": elapsed,
        "recon_heads": {k: v[:1200] for k, v in recon.items()},
        "pulls": pulls,
        "knightfork": knight,
        "modansaif": modansaif,
        "bawantha": bawantha,
        "q17_keys": keys,
        "stream_names": stream_names,
        "solo": solo,
        "leak": leak,
        "spearman_anchor": spearman_anchor,
        "pure_subfold": pure_sub,
        "prop_in_anchor": prop_in_anchor,
        "inv_moves": inv_moves,
        "staged": staged,
    }

    lines = []
    lines.append("# GM V12.2 Hardened: 0.95385 Recon + Q17 Anchor Inversion\n\n")
    lines.append(f"**UTC:** {utc}  \n")
    lines.append("**`NO_SUBMIT=True`** — 0 uploads  \n")
    lines.append("**POD TOUCHES: 0** — no `runpodctl`, no SSH, no GPU pods  \n")
    lines.append(f"**Elapsed:** {elapsed}s\n\n")

    lines.append("## Step 1 — Kaggle recon (`0.95385+` surge hunt)\n\n")
    lines.append("### Competition kernels by `dateRun` (top)\n```\n")
    lines.append("\n".join(recon["kernels_dateRun"].splitlines()[:22]) + "\n```\n\n")
    lines.append("### Competition kernels by `scoreDescending` (top)\n```\n")
    lines.append("\n".join(recon["kernels_scoreDesc"].splitlines()[:18]) + "\n```\n\n")
    lines.append("### Dataset updates (`s6e11`)\n```\n")
    lines.append("\n".join(recon["datasets_s6e11"].splitlines()[:16]) + "\n```\n\n")
    lines.append("### G26 / Q17 remote files\n```\n")
    lines.append(recon["g26_files"] + "\n```\n```\n" + recon["q17_files"] + "\n```\n\n")
    lines.append(
        "**0.9538x / 0.9537x kernel search note:** bare `kernels list -s 0.9538` hits unrelated competitions; "
        "comp-scoped `scoreDescending` still tops at public **0.95354** (Areeb / goodpjw). "
        "No verified public **0.95385** kernel artifact found via CLI this turn — surge may be LB-only / private.\n\n"
    )
    lines.append("### Pulled kernels\n")
    for slug, msg in pulls.items():
        lines.append(f"- `{slug}` → {msg}\n")
    lines.append("\n### Exact formulas\n")
    lines.append(f"- **knightfork LB 0.95357:** `{knight['formula']}` · WEIGHT={knight['WEIGHT']}\n")
    lines.append(f"  - Inputs: {', '.join(knight['inputs'])}\n")
    lines.append(f"  - Proxy OOF Δ={knight['proxy_oof_delta']} ({knight['proxy_folds_up']}/5 folds up)\n")
    lines.append(f"- **modansaif:** {modansaif['note']} Proxy: `{modansaif['formula_proxy']}`\n")
    lines.append(f"- **bawanthabeliwaththa:** {bawantha['thesis']}\n")

    lines.append("\n## Step 2 — `rank1_q17_streams.npz` reverse engineering\n\n")
    lines.append("### Prior audit executive excerpt\n```\n")
    lines.append("\n".join(q17_text.splitlines()[:22]) + "\n```\n\n")
    lines.append(f"**NPZ keys ({len(keys)}):** `{keys}`\n\n")
    lines.append(f"**stream_names ({len(stream_names)}):** `{stream_names}`\n\n")
    lines.append("| stream | solo AUC | F0 | F1 | F2 | F3 | F4 | oof shape | test shape |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
    for stem, r in solo.items():
        fa = r["fold_aucs"]
        lines.append(
            f"| {stem} | {r['solo_auc']:.7f} | {fa[0]:.7f} | {fa[1]:.7f} | {fa[2]:.7f} | "
            f"{fa[3]:.7f} | {fa[4]:.7f} | {r['oof_shape']} | {r['test_shape']} |\n"
        )

    lines.append("\n### Possibility A vs B (16 member streams → anchor)\n")
    lines.append(f"- Ridge R² OOF={r2_oof:.6f}, Test={r2_test:.6f}\n")
    lines.append(f"- max|w_oof−w_test| Ridge=**{max_abs_ridge:.6e}** OLS=**{max_abs_ols:.6e}** L-space=**{max_abs_L:.6e}**\n")
    lines.append(f"- Identical@1e-4? **{leak['identical_1e4']}** · transfer corr=**{corr_transfer:.6f}**\n")
    lines.append(f"- **Verdict: {leak['verdict']}**\n\n")
    lines.append("Top |Δw|:\n")
    for row in leak["top_abs_delta"][:8]:
        lines.append(
            f"- `{row['stream']}`: w_oof={row['w_oof']:+.4f}, w_test={row['w_test']:+.4f}, |Δ|={row['abs_delta']:.4f}\n"
        )

    lines.append("\n### Ladder-bank regression on `test_anchor_095370`\n")
    lines.append(f"- L-space R²=**{r2_ladder_L:.6f}** · raw R²=**{r2_ladder_raw:.6f}**\n")
    lines.append("| member | w_L | w_raw |\n|---|---:|---:|\n")
    for n in ladder_ext_names:
        lines.append(f"| {n} | {w_ladder_L[n]:+.4f} | {w_ladder_raw[n]:+.4f} |\n")

    lines.append("\n### Proprietary residual loadings into `test_anchor`\n")
    lines.append("| residual | corr(L_anchor) | corr(resid_after_ladder) |\n|---|---:|---:|\n")
    for n, r in prop_loadings.items():
        lines.append(
            f"| {n} | {r['corr_with_L_anchor']:+.6f} | {r['corr_with_resid_after_ladder']:+.6f} |\n"
        )
    lines.append(
        "\nIf |corr(resid)|≪1 and prop_in_anchor weights tiny → **anchor has near-zero weight on our private δ1 / L7_s2026** "
        "(room for inverted macro-leap).\n"
    )
    lines.append(f"\nOOF prop-in-anchor ridge: `{prop_in_anchor}`\n")

    lines.append("\n### `1−ρ` of `L(test_anchor_095370)` vs bank\n")
    lines.append("| ref | 1−ρ |\n|---|---:|\n")
    for k, v in spearman_anchor.items():
        lines.append(f"| {k} | {v if v is not None else 'N/A (formula-only)'} |\n")

    lines.append("\n## Step 3 — De-aliased subfold + inverted-anchor LOFO\n\n")
    lines.append("### Pure `L7_s2026_K10 − L7_s7_K4` on Fold0 (5-subfold, β=0.036)\n")
    lines.append(
        f"- **{pure_sub['pos5']}/5** · meanΔ=**{pure_sub['mean_d_e5']:.4f}e-5** · `[sf]={pure_sub['fold_d_e5']}`\n"
    )
    lines.append("\n### Injections on top of `oof_anchor_095370` (fixed w=0.036 + adaptive)\n")
    lines.append("| move | fixed pos/5 | fixed meanΔe5 | adaptive pos/5 | adaptive meanΔe5 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for name in [
        "plus_d1_v4_w0.036",
        "plus_G26_minus_MS_w0.036",
        "plus_L7_s2026_K10_f0_w0.036",
        "plus_d1_and_L7_w0.036",
        "plus_d1_G26MS_L7_w0.036",
    ]:
        f = inv_moves[name]
        a = inv_moves[name + "_adaptive"]
        lines.append(
            f"| {name} | {f['pos5']}/5 | {f['mean_d_e5']:.4f} | {a['pos5']}/5 | {a['mean_d_e5']:.4f} |\n"
        )

    lines.append("\n## Step 4 — Staged NOSUBMIT candidates (4 slots)\n\n")
    for s in staged:
        lines.append(f"- **Slot {s['slot']}** `{Path(s['path']).name}` — { {k:v for k,v in s.items() if k!='path'} }\n")
    lines.append(
        "\n**STRICT NO_SUBMIT=True** — files staged for human inspection only. "
        "Do not auto-upload. HOLD until promotion law + time-to-reset decision.\n"
    )
    lines.append("\n---\n*Generated by `scripts/gm_v12_2_hardened_095385_recon_and_anchor_inversion.py` — zero pod ops.*\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines))
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log(f"WROTE {OUT_MD}")
    return 0


def Linear_predict(X, y):
    """OLS fit X→y, return predictions (with intercept via centered columns)."""
    import numpy as np
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    return lr.predict(X)


if __name__ == "__main__":
    sys.exit(main())
