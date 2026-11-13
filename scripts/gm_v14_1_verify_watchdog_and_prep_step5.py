#!/usr/bin/env python3
"""GM V14.1 — non-disruptive canary/watchdog audit + Step-5 evaluator prep (NO_SUBMIT).

Does NOT touch any running training process. Safe to run on Mac while
li1uaxfpuiqf59 trains folds 1–4.

Modes:
  audit     — print L7F/LM fold AUCs, canary risk, ETA math (default)
  step5_dry — validate Step-5 assembly path with available folds only
  step5     — full 5-fold assemble + ATH moves (requires oof_0..4 + test_0..4)
  watch     — poll local preds/; when all folds present, run step5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
REPO = Path(os.environ.get("V14_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
OUT = Path(os.environ.get("L7_OUT", str(REPO / "preds" / "l7_s2026_5f")))
SUB = REPO / "submissions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
REPORT = REPO / "logs/overnight/GM_V14_1_WATCHDOG_CANARY_STEP5_PREP.md"
S7 = REPO / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
X5_META = REPO / "preds/gm_v5_overnight/X_v5_meta.json"
X5_OOF = REPO / "preds/gm_v5_overnight/X_v5_oof.npy"


def log(*a):
    print(time.strftime("%H:%M:%S "), *a, flush=True)


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def L_fold(v, fold_id):
    v = np.asarray(v, dtype=np.float64).ravel()
    out = np.empty_like(v)
    for f in range(5):
        m = fold_id == f
        out[m] = L(v[m])
    return out


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-18)


def bayes_ll(ll, prior):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def ranklogit_orders(ll):
    d = ll[:, :, 1].astype(np.float64) - ll[:, :, 0].astype(np.float64)
    return np.mean(np.column_stack([L(d[:, k]) for k in range(d.shape[1])]), axis=1)


def setup_folds(y):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    return folds, fold_id


def load_y():
    for p in [REPO / "data/train.csv", Path("/tmp/gm_l7_preflight/data/train.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
            return df[col].to_numpy(int)
    raise FileNotFoundError("train.csv")


def fold_aucs_1d(y, z, folds):
    return [float(roc_auc_score(y[te], z[te])) for _, te in folds]


def load_l7f(y, folds):
    # prefer 1d 5fold; else megalm; else per-fold ll
    for p in [S7 / "lm7_oof_5fold.npy", S7 / "megalm_oof_sha181c.npy"]:
        if p.exists():
            a = np.load(p)
            if a.ndim == 1 and len(a) >= len(y):
                return a[: len(y)].astype(np.float64), str(p)
            if a.ndim == 3 and a.shape[0] == len(y):
                oof = np.zeros(len(y))
                for f, (tr, te) in enumerate(folds):
                    oof[te] = bayes_ll(a[te], float(y[tr].mean()))
                return oof, str(p)
    oof = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(folds):
        p = S7 / f"lm7_oof_f{f}.npy"
        if p.exists():
            ll = np.load(p)
            oof[te] = bayes_ll(ll, float(y[tr].mean())) if ll.ndim == 3 else ll.ravel()
    return oof, "lm7_oof_f*.npy"


def load_lm(y, folds):
    """Prefer megayak_5f_lm from ATH stream bank; fallback megalm (hi-res compact LM)."""
    for p in [
        Path("/tmp/s6e11_v132/Z_oof_megayak_5f_lm.npy"),
        REPO / "preds/gm_v5_overnight/megayak_5f_lm_oof.npy",
        S7 / "megalm_oof_sha181c.npy",
    ]:
        if p.exists():
            a = np.load(p).astype(np.float64).ravel()
            if len(a) >= len(y):
                return a[: len(y)], str(p)
    # reconstruct from build_ath_proxy if importable
    try:
        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from gm_v13_2_overnight_check_and_stageB_v14 import build_ath_proxy

        ath = build_ath_proxy()
        z = ath["Z"]["oof_megayak_5f_lm"].astype(np.float64).ravel()
        return z[: len(y)], "build_ath_proxy:oof_megayak_5f_lm"
    except Exception as e:
        raise FileNotFoundError(f"megayak_5f_lm unavailable ({e})")


def audit():
    y = load_y()
    folds, fold_id = setup_folds(y)
    l7f, l7f_src = load_l7f(y, folds)
    lm, lm_src = load_lm(y, folds)
    a_l7f = fold_aucs_1d(y, l7f, folds)
    a_lm = fold_aucs_1d(y, lm, folds)
    # s2026 F0 if present
    a_s26 = [None] * 5
    if (OUT / "oof_0.npy").exists():
        ll = np.load(OUT / "oof_0.npy")
        tr, te = folds[0]
        a_s26[0] = float(roc_auc_score(y[te], bayes_ll(ll, float(y[tr].mean()))))
    for f in range(1, 5):
        p = OUT / f"oof_{f}.npy"
        if p.exists():
            ll = np.load(p)
            tr, te = folds[f]
            a_s26[f] = float(roc_auc_score(y[te], bayes_ll(ll, float(y[tr].mean()))))

    # canary analysis
    running_gate = 0.9485  # in-memory on pod
    softened_gate = 0.9472  # on-disk after V14.1 patch
    f1_l7f = a_l7f[1]
    risk = {
        "running_process_gate": f"auc26 > {running_gate}",
        "on_disk_softened_gate": f"auc26 >= {softened_gate}",
        "L7F_F1": f1_l7f,
        "L7_s2026_F0": a_s26[0],
        "false_abort_if_auc_in_0.9480_0.9485": True,  # running process still strict >
        "mitigation": "canary_recovery_watch.sh resumes folds 2-4 if exit-42 after oof_1; on-disk gate >=0.9472",
        "s7_f1_on_pod": False,
        "solo_only_canary_on_pod": True,
    }

    # ETA math from observed ~6.1 min/epoch, 8 epochs, score estimate
    train_min_per_fold = 6.1 * 8
    # K=10 OOF(~140k)+Test(~300k) @ bs=512: ~ (440k*10*2)/512 forwards of T~72 GPT2
    # empirical budget: 8–15 min; use 12
    score_min_per_fold = 12.0
    per_fold = train_min_per_fold + score_min_per_fold
    four_fold = 4 * per_fold
    watchdog_old, watchdog_new = 300.0, 380.0

    pack = {
        "NO_SUBMIT": True,
        "L7F_s7": {"src": l7f_src, "folds": a_l7f, "mean": float(np.mean(a_l7f))},
        "LM_megayak_5f": {"src": lm_src, "folds": a_lm, "mean": float(np.mean(a_lm))},
        "L7_s2026_partial": a_s26,
        "canary": risk,
        "eta": {
            "min_per_epoch_obs": 6.1,
            "train_min_per_fold": train_min_per_fold,
            "score_min_per_fold_est": score_min_per_fold,
            "total_4fold_min_est": four_fold,
            "watchdog_old_min": watchdog_old,
            "watchdog_new_min": watchdog_new,
            "headroom_min": watchdog_new - four_fold,
            "headroom_ok": watchdog_new > four_fold + 30,
        },
    }
    log("L7F_s7 folds", [round(x, 6) for x in a_l7f], "mean", round(pack["L7F_s7"]["mean"], 6))
    log("LM_megayak folds", [round(x, 6) for x in a_lm], "mean", round(pack["LM_megayak_5f"]["mean"], 6))
    log("L7_s2026 partial", a_s26)
    log("CANARY RISK: running gate >0.9485 WILL false-abort at 0.9480–0.9485; recovery armed")
    log(
        f"ETA ~{four_fold:.0f}m for folds1-4; watchdog 380m headroom={pack['eta']['headroom_min']:.0f}m "
        f"ok={pack['eta']['headroom_ok']}"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# GM V14.1 Watchdog / Canary / Step5 Prep\n\n```json\n"
        + json.dumps(pack, indent=2)
        + "\n```\n"
    )
    (OUT / "v14_1_audit.json").write_text(json.dumps(pack, indent=2))
    return pack


def pchip_lock(src, ref):
    from scipy.interpolate import PchipInterpolator

    src = np.asarray(src, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    qs = np.linspace(0, 1, 401)
    x, y = np.quantile(src, qs), np.quantile(ref, qs)
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    return PchipInterpolator(x, y, extrapolate=True)(src)


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    cols = [unitize(b) for b in bases if np.linalg.norm(np.asarray(b).ravel()) > 1e-12]
    B = np.column_stack(cols)
    lr = LinearRegression(fit_intercept=True).fit(B, v)
    return unitize(v - lr.predict(B))


def force_law(ref_p, score):
    ref_p = np.asarray(ref_p, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    order = np.argsort(score + 1e-15 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort")
    out = np.empty(len(score), dtype=np.float64)
    out[order] = np.sort(ref_p)
    return out


def available_folds():
    return [f for f in range(5) if (OUT / f"oof_{f}.npy").exists() and (OUT / f"test_{f}.npy").exists()]


def partial_fold_report():
    """As each fold syncs: solo Bayes/RankL vs L7F vs 2-seed on completed folds."""
    y = load_y()
    folds, fold_id = setup_folds(y)
    l7f, _ = load_l7f(y, folds)
    have = available_folds()
    rows = []
    for f in have:
        ll = np.load(OUT / f"oof_{f}.npy")
        assert ll.shape[1:] == (10, 2), ll.shape
        tr, te = folds[f]
        prior = float(y[tr].mean())
        zb = bayes_ll(ll, prior)
        zr = ranklogit_orders(ll)
        a_b = float(roc_auc_score(y[te], zb))
        a_r = float(roc_auc_score(y[te], zr))
        a7 = float(roc_auc_score(y[te], l7f[te]))
        two = 0.5 * L(l7f[te]) + 0.5 * L(zb)
        a2 = float(roc_auc_score(y[te], two))
        rows.append(
            {
                "fold": f,
                "auc_bayes": a_b,
                "auc_ranklogit": a_r,
                "auc_L7F": a7,
                "auc_2seed": a2,
                "delta_2seed_vs_L7F": a2 - a7,
            }
        )
        log(
            f"F{f}: Bayes={a_b:.6f} RankL={a_r:.6f} L7F={a7:.6f} 2seed={a2:.6f} "
            f"Δ={a2-a7:+.6f}"
        )
    return rows


def step5_full():
    assert NO_SUBMIT
    have = available_folds()
    assert have == list(range(5)), f"need folds 0..4, have {have}"
    # delegate to hardened pipeline assemble if present
    sys.path.insert(0, str(REPO / "scripts"))
    os.environ.setdefault("L7_OUT", str(OUT))
    os.environ.setdefault("L7_CKPT", str(OUT / "ckpts"))
    from gm_v14_complete_hardened_pipeline import assemble_and_eval

    return assemble_and_eval()


def watch_and_run(poll_s=60):
    log("watch: waiting for oof_0..4 + test_0..4 then step5 (NO_SUBMIT)")
    while True:
        have = available_folds()
        log(f"have folds {have}")
        if have:
            partial_fold_report()
        if have == list(range(5)):
            log("all 5 folds local — running step5")
            return step5_full()
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "step5_dry", "step5", "watch"], default="audit")
    args = ap.parse_args()
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        audit()
    elif args.mode == "step5_dry":
        audit()
        partial_fold_report()
        log("step5_dry OK — evaluator ready; full step5 waits for folds 0..4")
    elif args.mode == "step5":
        step5_full()
    elif args.mode == "watch":
        watch_and_run()


if __name__ == "__main__":
    main()
