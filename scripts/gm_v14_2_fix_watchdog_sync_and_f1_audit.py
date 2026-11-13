#!/usr/bin/env python3
"""GM V14.2 — watchdog/sync notes + Fold1 / F0+F1 local audit (STRICT NO_SUBMIT)."""
from __future__ import annotations

import json
import os
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
# Prefer repo tmp/ on the pics volume — system /tmp is often full.
TMP = REPO / "tmp"
OUT = REPO / "preds" / "l7_s2026_5f"
S7_CANDS = [
    TMP / "v14_2_audit/s7",
    REPO / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7",
    REPO / "external/nov13_recon/compact_lm_seed7",
]
S7 = next((p for p in S7_CANDS if (p / "lm7_oof_5fold.npy").exists() or (p / "megalm_oof_sha181c.npy").exists()), S7_CANDS[0])
REPORT = REPO / "logs/overnight/GM_V14_2_FOLD1_VERIFICATION.md"
LIVE = REPO / "submissions/sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def L_fold(v, fold_id, folds_keep=None):
    v = np.asarray(v, dtype=np.float64).ravel()
    out = np.empty_like(v)
    for f in range(5):
        if folds_keep is not None and f not in folds_keep:
            continue
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


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    cols = [unitize(b) for b in bases if np.linalg.norm(np.asarray(b).ravel()) > 1e-12]
    B = np.column_stack(cols)
    lr = LinearRegression(fit_intercept=True).fit(B, v)
    return unitize(v - lr.predict(B))


def setup(y):
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
    raise FileNotFoundError("train")


def sf10(y, z_base, u, folds_use):
    u = unit(u)
    # sign on first fold
    te0 = folds_use[0][1]
    if roc_auc_score(y[te0], z_base[te0] + 0.01 * u[te0]) < roc_auc_score(y[te0], z_base[te0] - 0.01 * u[te0]):
        u = -u
    deltas = []
    for _, te in folds_use:
        yf, zb, uf = y[te], z_base[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
            base = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - base) * 1e5)
    d = np.asarray(deltas, float)
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    try:
        _, wp = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    return {"pos10": int((d > 0).sum()), "t": float(mean / (se + 1e-18)), "mean_e5": mean, "wilcox": float(wp), "u": u}


def main_2fold(y, z_base, u, folds_use, ws=(0.006, 0.009, 0.012, 0.015, 0.020)):
    u = unit(u)
    te0 = folds_use[0][1]
    if roc_auc_score(y[te0], z_base[te0] + 0.01 * u[te0]) < roc_auc_score(y[te0], z_base[te0] - 0.01 * u[te0]):
        u = -u
    base = [roc_auc_score(y[te], z_base[te]) for _, te in folds_use]
    best = None
    for w in ws:
        d = [(roc_auc_score(y[te], z_base[te] + w * u[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds_use)]
        row = {"w": w, "pos": int(sum(x > 0 for x in d)), "mean_e5": float(np.mean(d)), "folds": [round(x, 3) for x in d]}
        if best is None or (row["pos"], row["mean_e5"]) > (best["pos"], best["mean_e5"]):
            best = row
            best["u"] = u
    return best


def main():
    assert NO_SUBMIT
    y = load_y()
    folds, fold_id = setup(y)
    assert (OUT / "oof_0.npy").exists() and (OUT / "oof_1.npy").exists()
    assert (OUT / "test_0.npy").exists() and (OUT / "test_1.npy").exists()

    l7f = np.load(S7 / "lm7_oof_5fold.npy").astype(np.float64).ravel()[: len(y)]
    lm = np.load(S7 / "megalm_oof_sha181c.npy").astype(np.float64).ravel()[: len(y)]

    # per-fold s2026
    z26 = np.zeros(len(y))
    zr = np.zeros(len(y))
    for f in (0, 1):
        ll = np.load(OUT / f"oof_{f}.npy")
        tr, te = folds[f]
        prior = float(y[tr].mean())
        z26[te] = bayes_ll(ll, prior)
        zr[te] = ranklogit_orders(ll)

    # --- Fold1 solo / 2seed ---
    te = folds[1][1]
    a_b = float(roc_auc_score(y[te], z26[te]))
    a_r = float(roc_auc_score(y[te], zr[te]))
    a7 = float(roc_auc_score(y[te], l7f[te]))
    a_lm = float(roc_auc_score(y[te], lm[te]))
    two = 0.5 * L(l7f[te]) + 0.5 * L(z26[te])
    a2 = float(roc_auc_score(y[te], two))
    delta_bps = (a2 - a7) * 1e4
    corr_oof = float(np.corrcoef(L(z26[te]), L(l7f[te]))[0, 1])

    # test corr / dist
    t0 = bayes_ll(np.load(OUT / "test_0.npy"), float(y.mean()))
    t1 = bayes_ll(np.load(OUT / "test_1.npy"), float(y.mean()))
    # L7F test if present
    l7f_te = None
    for p in [S7 / "lm7_test.npy", S7 / "megalm_test_sha181c.npy"]:
        if p.exists():
            a = np.load(p)
            if a.ndim == 1 and len(a) == len(t1):
                l7f_te = a.astype(np.float64)
                break
            if a.ndim == 3 and a.shape[0] == len(t1):
                l7f_te = bayes_ll(a, float(y.mean()))
                break
    corr_te = float(np.corrcoef(L(t1), L(l7f_te))[0, 1]) if l7f_te is not None else None
    dist_t0_t1 = float(1.0 - spearmanr(t0, t1).correlation)

    f1 = {
        "auc_bayes": a_b,
        "auc_ranklogit": a_r,
        "auc_L7F": a7,
        "auc_LM_megalm": a_lm,
        "auc_2seed": a2,
        "delta_2seed_vs_L7F_bps": delta_bps,
        "corr_L_oof_s2026_vs_L7F": corr_oof,
        "corr_L_test1_vs_L7F_test": corr_te,
        "spearman_dist_test0_vs_test1": dist_t0_t1,
        "ref_F0_2seed_bps": 50.4,
    }

    # --- F0+F1 stack on ATH ---
    ath_cands = [
        TMP / "v14_2_audit/ath/z_stageB_4fam_oof.npy",
        REPO / "preds/gm_v13_4_stageB/ath_095388_oof.npy",
        TMP / "s6e11_v132/ath_095388_oof.npy",
        REPO / "preds/gm_v5_overnight/eval_S7matched_355.npy",
    ]
    z_ath = None
    for p in ath_cands:
        if p.exists():
            z_ath = np.load(p).astype(np.float64).ravel()[: len(y)]
            ath_src = str(p)
            break
    if z_ath is None:
        raise FileNotFoundError("ATH OOF")
    d1_cands = [
        TMP / "v14_2_audit/ath/d1_v4_oof.npy",
        REPO / "preds/gm_v4_polish/d1_v4_oof.npy",
        REPO / "preds/gm_v5_overnight/d1_v4_oof.npy",
        TMP / "s6e11_v132/d1_v4_oof.npy",
    ]
    d1 = next(np.load(p).astype(np.float64).ravel()[: len(y)] for p in d1_cands if p.exists())

    mask = (fold_id == 0) | (fold_id == 1)
    folds_use = [folds[0], folds[1]]
    # L_fold only on F0/F1 then combine
    Ls26 = np.zeros(len(y))
    L7 = np.zeros(len(y))
    LLm = np.zeros(len(y))
    for f in (0, 1):
        m = fold_id == f
        Ls26[m] = L(z26[m])
        L7[m] = L(l7f[m])
        LLm[m] = L(lm[m])

    moves = {}
    for name, delta in [
        ("Move1_L7swap", unitize(Ls26 - L7)),
        ("Move2_LMswap", unitize(Ls26 - LLm)),
        ("Move3_lofo_orth", project_out(Ls26, [z_ath, d1])),
    ]:
        main = main_2fold(y, z_ath, delta, folds_use)
        sf = sf10(y, z_ath, main["u"], folds_use)
        moves[name] = {
            "main": {k: main[k] for k in ("w", "pos", "mean_e5", "folds")},
            "sf10": {k: sf[k] for k in ("pos10", "t", "mean_e5", "wilcox")},
        }

    pack = {
        "NO_SUBMIT": True,
        "watchdog": "sleep 21600 from ~15:31 UTC (360m); PID 257 untouched",
        "sync_monitor": "60s poll; sized pull; delete only when all 8 local",
        "fold1": f1,
        "ath_src": ath_src,
        "moves_F0F1": moves,
    }

    lines = [
        "# GM V14.2 — Fold1 Verification + F0+F1 Stack",
        "",
        "**NO_SUBMIT=True**",
        "",
        "## Ops",
        f"- Watchdog: `{pack['watchdog']}`",
        f"- Sync: `{pack['sync_monitor']}`",
        "",
        "## Fold1 Solo / Two-Seed",
        "",
        f"- Bayes AUC=`{a_b:.6f}` · RankLogit=`{a_r:.6f}`",
        f"- L7F_s7 F1=`{a7:.6f}` · LM(megalm) F1=`{a_lm:.6f}`",
        f"- **2-seed AUC=`{a2:.6f}` · Δ vs L7F = `{delta_bps:+.2f}` bps** (F0 ref +50.4 bps)",
        f"- corr L(OOF s2026, L7F)=`{corr_oof:.6f}`",
        f"- corr L(test1, L7F_test)=`{corr_te}`",
        f"- 1−ρ(test0, test1)=`{dist_t0_t1:.6e}`",
        "",
        "## F0+F1 ATH Stack (2 main / 10 subfolds)",
        "",
        f"- ATH src: `{ath_src}`",
        "",
        "| Move | main pos/2 | meanΔ e-5 | [F0,F1] | sf10 | t | W-p |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in moves.items():
        m, s = r["main"], r["sf10"]
        lines.append(
            f"| {name} | {m['pos']}/2 @w={m['w']} | {m['mean_e5']:.3f} | {m['folds']} | "
            f"{s['pos10']}/10 | {s['t']:.2f} | {s['wilcox']:.3g} |"
        )
    lines += ["", "```json", json.dumps(pack, indent=2), "```", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(REPORT.read_text())
    (OUT / "v14_2_f1_audit.json").write_text(json.dumps(pack, indent=2))


if __name__ == "__main__":
    main()
