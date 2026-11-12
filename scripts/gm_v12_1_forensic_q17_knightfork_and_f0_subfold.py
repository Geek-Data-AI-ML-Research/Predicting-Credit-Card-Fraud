#!/usr/bin/env python3
"""V12.1 forensic: Q17 anchor leak test + knightfork formula + Fold0 5-subfold LOFO.

STRICT NO_SUBMIT=True. Does NOT touch any RunPod pods.
Writes: logs/overnight/GM_V12_1_FORENSIC_Q17_AND_SUBFOLD_REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NO_SUBMIT = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT_MD = ROOT / "logs/overnight/GM_V12_1_FORENSIC_Q17_AND_SUBFOLD_REPORT.md"
OUT_JSON = ROOT / "logs/overnight/GM_V12_1_FORENSIC_Q17_AND_SUBFOLD_REPORT.json"
Q17 = ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
Q17_MD = ROOT / "logs/overnight/GM_AMANATAR_Q17_AUDIT.md"
PRED = ROOT / "preds/l7_s2026_5f"
S7DIR = ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
KF = ROOT / "external/nov12_late_kernels"
T0 = time.time()


def log(m: str):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def bayes(ll, prior, orders=None):
    import numpy as np

    x = ll if orders is None else ll[:, list(orders), :]
    t = np.logaddexp.reduce(x[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(x[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def main():
    assert NO_SUBMIT is True
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import spearmanr

    from _gm_v11_shared import L, L_fold, load_anchor_and_bank, load_prob, project_out, setup_folds, spearman_dist, unit

    log("======== V12.1 forensic NO_SUBMIT=True (NO POD TOUCHES) ========")

    # ---- Step1 note only (no pod ops) ----
    step1 = {
        "pod_ops": "NONE — user forbid all pod touches this turn",
        "vjez42fl07jtrl": "already absent (404) at prior check; not touched",
        "w102kdrd4pmsz8": "observed as current P730 successor; NOT inspected/terminated this turn",
        "lheroidmuab1j4": "never touch",
    }

    # ---- Q17 streams ----
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
    log(f"Q17 pairs={len(pairs)} anchor_solo={solo.get('anchor_095370',{}).get('solo_auc')}")

    # OLS/Ridge leak test
    Xo = np.column_stack([z["oof_" + s].astype(np.float64).ravel() for s in stream_names])
    Xt = np.column_stack([z["test_" + s].astype(np.float64).ravel() for s in stream_names])
    yo = z["oof_anchor_095370"].astype(np.float64).ravel()
    yt = z["test_anchor_095370"].astype(np.float64).ravel()
    # standardize for comparable weights
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
    # also raw OLS via lstsq
    wo_ols, *_ = np.linalg.lstsq((Xo - Xo_m) / Xo_s, yo_c, rcond=None)
    wt_ols, *_ = np.linalg.lstsq((Xt - Xt_m) / Xt_s, yt_c, rcond=None)
    max_abs_ridge = float(np.max(np.abs(w_oof - w_test)))
    max_abs_ols = float(np.max(np.abs(wo_ols - wt_ols)))
    # predict test from oof weights
    yt_hat = (((Xt - Xt_m) / Xt_s) @ w_oof) + yt.mean()
    corr_transfer = float(np.corrcoef(yt_hat, yt)[0, 1])
    leak = {
        "stream_names": stream_names,
        "r2_oof_ridge": r2_oof,
        "r2_test_ridge": r2_test,
        "max_abs_w_oof_minus_w_test_ridge": max_abs_ridge,
        "max_abs_w_oof_minus_w_test_ols": max_abs_ols,
        "identical_1e4": bool(max_abs_ridge < 1e-4),
        "corr_test_vs_oof_weights_applied": corr_transfer,
        "w_oof_ridge": {n: float(w) for n, w in zip(stream_names, w_oof)},
        "w_test_ridge": {n: float(w) for n, w in zip(stream_names, w_test)},
        "top_abs_delta": sorted(
            [
                {"stream": n, "w_oof": float(a), "w_test": float(b), "abs_delta": float(abs(a - b))}
                for n, a, b in zip(stream_names, w_oof, w_test)
            ],
            key=lambda d: -d["abs_delta"],
        )[:8],
        "verdict": (
            "Possibility A: shared frozen linear blend (w_oof≈w_test)"
            if max_abs_ridge < 1e-4
            else "Possibility B: OOF/test constructed differently (weights diverge)"
        ),
    }
    log(f"leak: max|Δw|={max_abs_ridge:.3e} R2o={r2_oof:.4f} R2t={r2_test:.4f} → {leak['verdict']}")

    # spearman distances for test_anchor
    pack = load_anchor_and_bank(y, fold_id, folds)
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
            for p in (ROOT / "submissions").glob(g):
                if p.name.startswith("._"):
                    continue
                try:
                    refs[tag] = L(load_prob(p))
                    break
                except Exception:
                    pass
            if tag in refs:
                break
    Lt = L(yt)
    spearman_anchor = {k: round(spearman_dist(Lt, v), 8) for k, v in refs.items() if len(v) == 299984}
    log(f"anchor test 1-ρ: {spearman_anchor}")

    # ---- knightfork / late kernels ----
    knight = {
        "formula": "L(preserved_095354_ladder) + 0.25 * (L(sovereign) - L(wide))",
        "public": "0.95354 → 0.95357 (+0.00003)",
        "WEIGHT": 0.25,
        "proxy_oof_delta": 0.000017580980577,
        "proxy_folds_improved": 4,
        "inputs": "goodpjw LM Ladder 0.95354 reference + Amanatar Sovereign vault + wide stack",
        "notebook": str(KF / "knightfork_095357/s6e11-top-15-cpu-blend-eda-lb-0-95357.ipynb"),
    }
    asterios = {
        "thesis": "Public top is a dense noisy band; nested CV on wide-stack OOF separates probes that survive private 80%.",
        "method": "Simulate public(~60k)/private(~240k) splits from goodpjw nested wide OOF; replay published ladder moves.",
        "takeaway": "Many tiny public gains are winner's-curse; prefer nested-CV positive moves (e.g. G26−MS) over probes that lose OOF.",
    }
    markbarney = {
        "thesis": "Plain-language postmortem of ~85 own models / dead ends over 12 days; invites collaboration.",
        "compute": "Home lab + RTX 5070/4090 + DGX Sparks + ~$39 Vast.ai",
        "takeaway": "Publishes failures intentionally; lists ideas not yet tested (other LMs, BERT-style).",
    }
    modansaif = {
        "path": str(KF / "modansaif"),
        "note": "Fraud-fusion reproduction / credited — pulled for completeness.",
    }

    # ---- Fold0 5-subfold LOFO with FIXED weights ----
    oof26 = np.load(PRED / "oof_0.npy").astype(np.float32)
    ll7 = np.load(S7DIR / "lm7_oof_f0.npy").astype(np.float32)
    a0, b0 = folds[0]
    y0 = y[b0]
    prior0 = float(y[a0].mean())
    z26 = bayes(oof26, prior0)
    z7k4 = bayes(ll7, prior0, orders=range(4))
    L7F_f0 = L_fold(np.load(S7DIR / "lm7_oof_5fold.npy").astype(np.float64), fold_id)[b0]
    base = pack["eval_S7matched_355"][b0]
    # K4-aware note: base already contains ladder L7 path; we add residuals vs K4 as specified
    uA = L(z26) - L(z7k4)
    uB = 0.65 * L(L7F_f0) + 0.35 * L(z26) - L(z7k4)
    lm_trim = project_out(-(pack["LMo"] - pack["Fo"])[b0], [base, pack["u_g26ms_o"][b0]])
    uC = uB + 0.025 * unit(lm_trim)
    # Q17 anchor residual orth to base on F0
    anc_o = L_fold(z["oof_anchor_095370"].astype(np.float64), fold_id)
    uD_f0 = project_out(anc_o[b0], [base, pack["u_g26ms_o"][b0]])

    sf = StratifiedKFold(5, shuffle=True, random_state=42)
    subfolds = list(sf.split(np.zeros(len(y0)), y0))

    def subfold_lofo(delta, w):
        dlt = unit(delta) if np.std(delta) > 0 else delta
        fold_d = []
        for _, te in subfolds:
            b = float(roc_auc_score(y0[te], base[te]))
            a = float(roc_auc_score(y0[te], base[te] + w * dlt[te]))
            fold_d.append(a - b)
        pos = sum(1 for d in fold_d if d > 0)
        return {
            "pos5": pos,
            "mean_d_e5": float(np.mean(fold_d) * 1e5),
            "fold_d_e5": [round(d * 1e5, 4) for d in fold_d],
            "w": w,
        }

    moves_sf = {
        "MoveA_fixed_w0.036": subfold_lofo(uA, 0.036),
        "MoveB_fixed_w0.036": subfold_lofo(uB, 0.036),
        "MoveC_fixed_B_plus_0.025_LMtrim": subfold_lofo(uC, 1.0),  # uC already includes 0.036? NO
    }
    # Fix Move C: cand2 + 0.036*uB_raw_unit? Spec: Move B_fixed + 0.025*orth(LM_trim)
    # Move B_fixed = base + 0.036 * uB ; Move C = that + 0.025 * orth(LM_trim)
    def subfold_lofo_c():
        dB = unit(uB)
        dL = unit(lm_trim)
        fold_d = []
        for _, te in subfolds:
            b = float(roc_auc_score(y0[te], base[te]))
            a = float(roc_auc_score(y0[te], base[te] + 0.036 * dB[te] + 0.025 * dL[te]))
            fold_d.append(a - b)
        pos = sum(1 for d in fold_d if d > 0)
        return {
            "pos5": pos,
            "mean_d_e5": float(np.mean(fold_d) * 1e5),
            "fold_d_e5": [round(d * 1e5, 4) for d in fold_d],
            "wB": 0.036,
            "wL": 0.025,
        }

    moves_sf["MoveC_fixed_B_plus_0.025_LMtrim"] = subfold_lofo_c()
    moves_sf["MoveD_q17_orth_anchor_w0.036"] = subfold_lofo(uD_f0, 0.036)

    # Move D across full 5 folds
    uD_full = project_out(anc_o, pack["bases_o"])
    fold_d_full = []
    for tr, te in folds:
        b = float(roc_auc_score(y[te], pack["eval_S7matched_355"][te]))
        a = float(roc_auc_score(y[te], pack["eval_S7matched_355"][te] + 0.036 * unit(uD_full)[te]))
        fold_d_full.append(a - b)
    moveD_full = {
        "pos5": sum(1 for d in fold_d_full if d > 0),
        "mean_d_e5": float(np.mean(fold_d_full) * 1e5),
        "fold_d_e5": [round(d * 1e5, 4) for d in fold_d_full],
        "w": 0.036,
    }
    log(f"subfold A={moves_sf['MoveA_fixed_w0.036']} B={moves_sf['MoveB_fixed_w0.036']}")
    log(f"subfold C={moves_sf['MoveC_fixed_B_plus_0.025_LMtrim']} D={moves_sf['MoveD_q17_orth_anchor_w0.036']}")
    log(f"MoveD full5={moveD_full}")

    # Q17 audit text excerpt
    q17_text = Q17_MD.read_text() if Q17_MD.exists() else "(missing)"

    summary = {
        "NO_SUBMIT": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - T0, 2),
        "step1_pods": step1,
        "q17_keys": keys,
        "solo": solo,
        "leak": leak,
        "spearman_anchor": spearman_anchor,
        "knightfork": knight,
        "asterios": asterios,
        "markbarney": markbarney,
        "modansaif": modansaif,
        "moves_subfold": moves_sf,
        "moveD_full5": moveD_full,
    }

    lines = []
    lines.append("# GM V12.1 Forensic: Q17 Leak + knightfork + Fold0 Subfold LOFO\n\n")
    lines.append(f"**UTC:** {summary['utc']}  \n**`NO_SUBMIT=True`** — 0 uploads; **no RunPod pod touches this turn**  \n")
    lines.append(f"**Elapsed:** {summary['elapsed_s']}s\n\n")

    lines.append("## Step 1 — Pods\n")
    lines.append("- **No pod operations executed** (explicit user hard-stop).\n")
    lines.append(f"- `vjez42fl07jtrl`: {step1['vjez42fl07jtrl']}\n")
    lines.append(f"- `w102kdrd4pmsz8`: {step1['w102kdrd4pmsz8']}\n")
    lines.append(f"- Foreign `{step1['lheroidmuab1j4']}` untouched.\n\n")

    lines.append("## Step 2 — Q17 `anchor_095370` leakage & late kernels\n")
    lines.append("### Prior audit (`GM_AMANATAR_Q17_AUDIT.md`) — executive excerpt\n")
    lines.append("```\n" + "\n".join(q17_text.splitlines()[:20]) + "\n```\n\n")
    lines.append("### Stream bank keys / solo AUCs\n")
    lines.append("| stream | solo AUC | F0 | F1 | F2 | F3 | F4 |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for stem, r in solo.items():
        fa = r["fold_aucs"]
        lines.append(
            f"| {stem} | {r['solo_auc']:.7f} | {fa[0]:.7f} | {fa[1]:.7f} | {fa[2]:.7f} | {fa[3]:.7f} | {fa[4]:.7f} |\n"
        )

    lines.append("\n### Possibility A vs B — OOF vs Test blend weights\n")
    lines.append(f"- Ridge R² OOF={r2_oof:.6f}, Test={r2_test:.6f}\n")
    lines.append(f"- max|w_oof − w_test| (Ridge) = **{max_abs_ridge:.6e}** ; OLS = **{max_abs_ols:.6e}**\n")
    lines.append(f"- Identical at 1e-4? **{leak['identical_1e4']}**\n")
    lines.append(f"- Corr(test, apply w_oof on test streams) = **{corr_transfer:.6f}**\n")
    lines.append(f"- **Verdict: {leak['verdict']}**\n")
    lines.append("\nTop |Δw| streams:\n")
    for row in leak["top_abs_delta"]:
        lines.append(
            f"- `{row['stream']}`: w_oof={row['w_oof']:+.4f}, w_test={row['w_test']:+.4f}, |Δ|={row['abs_delta']:.4f}\n"
        )
    lines.append("\n### `test_anchor_095370` Spearman distances (`1−ρ`)\n")
    lines.append("| ref | 1−ρ |\n|---|---:|\n")
    for k, v in spearman_anchor.items():
        lines.append(f"| {k} | {v:.8f} |\n")

    lines.append("\n### knightfork LB 0.95357 — exact formula\n")
    lines.append(f"- **Formula:** `{knight['formula']}`\n")
    lines.append(f"- Public: {knight['public']}\n")
    lines.append(f"- Proxy OOF Δ≈{knight['proxy_oof_delta']} with {knight['proxy_folds_improved']}/5 folds up\n")
    lines.append(f"- Inputs: {knight['inputs']}\n")
    lines.append("\n### asterios / markbarney / modansaif\n")
    lines.append(f"- **asterios:** {asterios['thesis']} Method: {asterios['method']} Takeaway: {asterios['takeaway']}\n")
    lines.append(f"- **markbarney:** {markbarney['thesis']} Compute: {markbarney['compute']}. {markbarney['takeaway']}\n")
    lines.append(f"- **modansaif:** pulled to `{modansaif['path']}` ({modansaif['note']})\n")

    lines.append("\n## Step 3 — Fold-0 5-subfold stratified LOFO (fixed weights)\n")
    lines.append("Base = `eval_S7matched_355[fold0]` · SF = `StratifiedKFold(5, shuffle=True, rs=42)` on 139,992 rows.\n\n")
    lines.append("| move | pos/5 | meanΔe5 | [sf0..sf4] e-5 |\n|---|---:|---:|---|\n")
    for name, r in moves_sf.items():
        lines.append(
            f"| {name} | {r['pos5']}/5 | {r['mean_d_e5']:.4f} | {r['fold_d_e5']} |\n"
        )
    lines.append(
        f"| MoveD_q17 FULL F0..F4 | {moveD_full['pos5']}/5 | {moveD_full['mean_d_e5']:.4f} | {moveD_full['fold_d_e5']} |\n"
    )
    lines.append("\n### Interpretation\n")
    a, b, c = moves_sf["MoveA_fixed_w0.036"], moves_sf["MoveB_fixed_w0.036"], moves_sf["MoveC_fixed_B_plus_0.025_LMtrim"]
    lines.append(
        f"- Prior V12 in-sample Move C (+0.218e-5) used a **2-parameter grid** on all of Fold 0. "
        f"Fixed-weight subfold C is **{c['pos5']}/5**, mean **{c['mean_d_e5']:.4f}e-5**.\n"
    )
    lines.append(
        f"- Move A fixed (0.036× analog of G26−MS): **{a['pos5']}/5**, mean **{a['mean_d_e5']:.4f}e-5**.\n"
    )
    lines.append(
        f"- Move B fixed: **{b['pos5']}/5**, mean **{b['mean_d_e5']:.4f}e-5**.\n"
    )
    lines.append(
        f"- Move D (orth Q17 anchor) subfold **{moves_sf['MoveD_q17_orth_anchor_w0.036']['pos5']}/5**; "
        f"full5 **{moveD_full['pos5']}/5** mean **{moveD_full['mean_d_e5']:.4f}e-5**.\n"
    )
    lines.append("\n**HOLD all 4 daily submissions** until promotion law is satisfied on debiased evidence.\n")
    lines.append("\n---\n*Generated by `scripts/gm_v12_1_forensic_q17_knightfork_and_f0_subfold.py` — no pod ops.*\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines))
    # scrub large arrays from json
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log(f"WROTE {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
