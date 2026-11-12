#!/usr/bin/env python3
"""Stop-loss rescue summary + true K=4→2-seed K=10 Fold-0 upgrade audit.

STRICT NO_SUBMIT=True — stages NOSUBMIT CSVs only; zero Kaggle uploads.
Writes: logs/overnight/GM_V12_STOPLOSS_AND_2SEED_L7_REPORT.md (+ .json)
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

OUT_MD = ROOT / "logs/overnight/GM_V12_STOPLOSS_AND_2SEED_L7_REPORT.md"
OUT_JSON = ROOT / "logs/overnight/GM_V12_STOPLOSS_AND_2SEED_L7_REPORT.json"
SUB = ROOT / "submissions"
PRED = ROOT / "preds/l7_s2026_5f"
S7DIR = ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
Q17_MD = ROOT / "logs/overnight/GM_AMANATAR_Q17_AUDIT.md"
KAG = ROOT / ".venv_forensics/bin/kaggle"
T0 = time.time()


def log(m: str):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def bayes_orders(ll: np.ndarray, prior: float, orders=None) -> np.ndarray:
    """Bayes log-odds from [n,K,2] LL. Optional order subset."""
    x = ll if orders is None else ll[:, list(orders), :]
    t = np.logaddexp.reduce(x[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(x[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def mean_llr(ll: np.ndarray, orders=None) -> np.ndarray:
    x = ll if orders is None else ll[:, list(orders), :]
    return (x[:, :, 1] - x[:, :, 0]).astype(np.float64).mean(1)


def bayes_tau(ll: np.ndarray, prior: float, tau: float, orders=None) -> np.ndarray:
    x = ll if orders is None else ll[:, list(orders), :]
    t = np.logaddexp.reduce(x[:, :, 1].astype(np.float64) / tau, axis=1)
    f = np.logaddexp.reduce(x[:, :, 0].astype(np.float64) / tau, axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def rank_logit_mean(ll: np.ndarray, L_fn, orders=None) -> np.ndarray:
    x = ll if orders is None else ll[:, list(orders), :]
    # per-order LLR then mean of rank-logits
    ks = []
    for k in range(x.shape[1]):
        ks.append(L_fn((x[:, k, 1] - x[:, k, 0]).astype(np.float64)))
    return np.mean(np.stack(ks, 0), 0)


def weight_search(y, base, delta, ws=None):
    from sklearn.metrics import roc_auc_score

    if ws is None:
        ws = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    b = float(roc_auc_score(y, base))
    best = (-1e9, None, b)
    for w in ws:
        a = float(roc_auc_score(y, base + w * delta))
        d = a - b
        if d > best[0]:
            best = (d, w, a)
    return {"base_auc": b, "best_auc": best[2], "best_w": best[1], "delta": best[0], "delta_e5": best[0] * 1e5}


def force_to_ref(ref_p, score):
    from _gm_v11_shared import force_law

    return force_law(ref_p, score)


def main():
    assert NO_SUBMIT is True
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from _gm_v11_shared import (
        L,
        L_fold,
        force_law,
        g26_status,
        load_anchor_and_bank,
        load_prob,
        project_out,
        setup_folds,
        spearman_dist,
        unit,
        write_nosubmit,
    )

    global np  # for helpers typed above
    import numpy as np  # noqa: F811

    log("======== V12 STOPLOSS + K4→2SEED K10 audit NO_SUBMIT=True ========")

    # ---------- Step1 status ----------
    oof26 = np.load(PRED / "oof_0.npy").astype(np.float32)
    te26 = np.load(PRED / "test_0.npy").astype(np.float32)
    assert oof26.shape == (139992, 10, 2) and te26.shape == (299984, 10, 2)
    ck_rep = PRED / "ckpts/replica.pt"
    ck_f0 = PRED / "ckpts/fold_0.pt"
    step1 = {
        "local_oof_0": list(oof26.shape),
        "local_test_0": list(te26.shape),
        "replica_pt_mb": round(ck_rep.stat().st_size / 1e6, 1) if ck_rep.exists() else None,
        "fold_0_pt_mb": round(ck_f0.stat().st_size / 1e6, 1) if ck_f0.exists() else None,
        "pod_id_terminated": "nnts6ddqh5mprf",
        "never_touch": "lheroidmuab1j4",
    }
    log(f"Step1 assets: {step1}")

    # ---------- data / folds ----------
    y_full = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    folds, fold_id = setup_folds(y_full)
    a0, b0 = folds[0]
    assert len(b0) == 139992
    y0 = y_full[b0]
    prior0 = float(y_full[a0].mean())
    pack = load_anchor_and_bank(y_full, fold_id, folds)

    # seed7 tensors
    ll7_oof = np.load(S7DIR / "lm7_oof_f0.npy").astype(np.float32)  # (139992,10,2)
    ll7_te = np.load(S7DIR / "lm7_test_f0.npy").astype(np.float32)  # (299984,4,2)
    L7F_o = L_fold(np.load(S7DIR / "lm7_oof_5fold.npy").astype(np.float64), fold_id)
    L7F_t = L(np.load(S7DIR / "lm7_test_5fold.npy").astype(np.float64))
    assert ll7_oof.shape == (139992, 10, 2) and ll7_te.shape == (299984, 4, 2)

    # reductions
    reductions = {}
    for name, fn in [
        ("R1_mean_llr", lambda ll: mean_llr(ll)),
        ("R2_bayes_tau0.5", lambda ll: bayes_tau(ll, prior0, 0.5)),
        ("R2_bayes_tau1.0", lambda ll: bayes_tau(ll, prior0, 1.0)),
        ("R2_bayes_tau1.5", lambda ll: bayes_tau(ll, prior0, 1.5)),
        ("R2_bayes_tau2.0", lambda ll: bayes_tau(ll, prior0, 2.0)),
        ("R3_ranklogit_mean", lambda ll: rank_logit_mean(ll, L)),
        ("R2_bayes_default", lambda ll: bayes_orders(ll, prior0)),
    ]:
        z = fn(oof26)
        reductions[name] = {
            "auc": float(roc_auc_score(y0, z)),
            "z": z,
        }
        log(f"  s2026 {name} AUC={reductions[name]['auc']:.7f}")

    best_red = max(reductions, key=lambda k: reductions[k]["auc"])
    z26 = reductions[best_red]["z"]
    # also keep default bayes for two-seed (matches prior canary)
    z26_bayes = reductions["R2_bayes_default"]["z"]

    z7_k4 = bayes_orders(ll7_oof, prior0, orders=range(4))
    z7_k10 = bayes_orders(ll7_oof, prior0, orders=range(10))
    # test-side K4 for seed7 (matches lm7_test_f0)
    prior_te = float(y_full.mean())  # not used for AUC; for consistency of Bayes const
    t7_k4 = bayes_orders(ll7_te, prior_te, orders=range(4))
    # s2026 test reduction matching best OOF reducer name
    if best_red.startswith("R1"):
        t26 = mean_llr(te26)
    elif best_red.startswith("R3"):
        t26 = rank_logit_mean(te26, L)
    elif "tau0.5" in best_red:
        t26 = bayes_tau(te26, prior_te, 0.5)
    elif "tau1.5" in best_red:
        t26 = bayes_tau(te26, prior_te, 1.5)
    elif "tau2.0" in best_red:
        t26 = bayes_tau(te26, prior_te, 2.0)
    else:
        t26 = bayes_orders(te26, prior_te)

    solo = {
        "L7_s7_K4": float(roc_auc_score(y0, z7_k4)),
        "L7_s7_K10": float(roc_auc_score(y0, z7_k10)),
        "L7_s2026_best_red": float(roc_auc_score(y0, z26)),
        "L7_s2026_best_red_name": best_red,
        "L7_s2026_bayes": float(roc_auc_score(y0, z26_bayes)),
        "reductions": {k: {"auc": v["auc"]} for k, v in reductions.items()},
    }
    # two-seed ensemble on F0
    z_2seed = 0.5 * L(z7_k10) + 0.5 * L(z26_bayes)
    z_2seed_vs_k4 = 0.5 * L(z7_k4) + 0.5 * L(z26_bayes)
    solo["two_seed_K10_K10"] = float(roc_auc_score(y0, z_2seed))
    solo["two_seed_K4_plus_s2026"] = float(roc_auc_score(y0, z_2seed_vs_k4))
    solo["corr_s2026_vs_s7_K10"] = float(np.corrcoef(z26_bayes, z7_k10)[0, 1])
    solo["corr_s2026_vs_s7_K4"] = float(np.corrcoef(z26_bayes, z7_k4)[0, 1])

    # hybrid with L7F on fold0 rows
    L7F_f0 = L7F_o[b0]
    hybrid_rows = {}
    for g in (0.15, 0.25, 0.35, 0.45, 0.55):
        zh = (1 - g) * L(L7F_f0) + g * L(z26_bayes)
        hybrid_rows[g] = float(roc_auc_score(y0, zh))
    solo["hybrid_L7F_plus_s2026"] = hybrid_rows
    best_g = max(hybrid_rows, key=hybrid_rows.get)
    solo["hybrid_best_gamma"] = best_g
    solo["hybrid_best_auc"] = hybrid_rows[best_g]
    log(f"solo={ {k:v for k,v in solo.items() if k!='reductions'} }")

    # ---------- Full-stack Fold-0 evaluator matching cand2's K=4 L7 entry ----------
    # Rebuild fold0 slice of eval_S7matched but with L7 as K4 instead of whatever is inside z_s7.
    # Practical approach: take pack['eval_S7matched_355'][b0] as base that already includes L7F/LM
    # path via areeb ladder, then measure residual upgrades vs L7_s7_K4 proxy.
    #
    # True Move A: replace K4 contribution with two-seed residual L(s2026)-L(s7_K4)
    base_f0 = pack["eval_S7matched_355"][b0]
    base_auc = float(roc_auc_score(y0, base_f0))

    # Residuals vs the K=4 seed7 member that actually matches lm7_test_f0 on test.
    u_A_raw = unit(L(z26_bayes) - L(z7_k4))
    u_B_raw = unit(0.65 * L(L7F_f0) + 0.35 * L(z26_bayes) - L(z7_k4))

    # LM_trim on F0: -(LM - F) on OOF
    lm_trim_o = -(pack["LMo"] - pack["Fo"])
    u_lm_f0 = unit(project_out(lm_trim_o[b0], [pack["eval_S7matched_355"][b0], pack["u_g26ms_o"][b0]]))

    moveA = weight_search(y0, base_f0, u_A_raw)
    moveB = weight_search(y0, base_f0, u_B_raw)
    # Move C: search joint weights
    bestC = {"delta": -1e9}
    for wB in (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        for wL in (0.005, 0.01, 0.02, 0.04, 0.06, 0.08):
            sc = base_f0 + wB * u_B_raw + wL * u_lm_f0
            a = float(roc_auc_score(y0, sc))
            d = a - base_auc
            if d > bestC["delta"]:
                bestC = {"delta": d, "delta_e5": d * 1e5, "best_auc": a, "wB": wB, "wL": wL, "base_auc": base_auc}

    # Also compare OLD flawed Gate3 residual L(s2026)-L(s7_K10) for documentation
    u_old = unit(L(z26_bayes) - L(z7_k10))
    move_old = weight_search(y0, base_f0, u_old)

    moves = {
        "MoveA_s2026_minus_s7_K4": moveA,
        "MoveB_hybrid65_35_minus_s7_K4": moveB,
        "MoveC_B_plus_LMtrim": bestC,
        "OLD_gate3_s2026_minus_s7_K10": move_old,
        "base_eval_S7matched_f0_auc": base_auc,
    }
    log(f"moves: A={moveA['delta_e5']:.4f}e-5 B={moveB['delta_e5']:.4f}e-5 C={bestC['delta_e5']:.4f}e-5 old={move_old['delta_e5']:.4f}e-5")

    # ---------- Stage NOSUBMIT test CSVs ----------
    ids = pack["ids"]
    cand2 = pack["cand2"]
    # Test residuals
    t7k4 = L(t7_k4)
    t26L = L(t26)
    L7F_tL = L7F_t
    uA_t = unit(t26L - t7k4)
    uB_t = unit(0.65 * L7F_tL + 0.35 * t26L - t7k4)
    # LM_trim test
    u_lm_t = unit(-(pack["LM"] - pack["F"]))

    staged = []
    # Use NON-unit residuals so 1-ρ(cand2) can reach [0.10e-4, 0.14e-4]
    uA_t_raw = L(t26) - t7k4
    uB_t_raw = 0.65 * L7F_tL + 0.35 * L(t26) - t7k4
    for tag, u_t in [
        ("v12_cand1_MoveA_twoseed_K10_minus_K4", uA_t_raw),
        ("v12_cand2_MoveB_hybrid_minus_K4", uB_t_raw),
    ]:
        target = 0.12e-4
        lo, hi = 0.10e-4, 0.14e-4
        best_hit = None
        best_near = (None, 1e9, None, None)
        # coarse then fine search on |w|
        for w in np.concatenate(
            [
                np.linspace(0.001, 0.05, 40),
                np.linspace(0.05, 0.5, 60),
                np.linspace(0.5, 3.0, 50),
                -np.linspace(0.001, 0.5, 40),
            ]
        ):
            sc = L(cand2) + float(w) * u_t
            p = force_law(cand2, sc)
            d = spearman_dist(p, cand2)
            if lo <= d <= hi:
                best_hit = (float(w), d, p)
                break
            if abs(d - target) < best_near[1]:
                best_near = (float(w), abs(d - target), p, d)
        if best_hit is None:
            best_hit = (best_near[0], best_near[3], best_near[2])
        w, d, p = best_hit
        path = SUB / f"sub_{tag}_{d*1e4:.2f}e4_NOSUBMIT.csv"
        write_nosubmit(path, ids, p)
        staged.append(
            {
                "tag": tag,
                "path": str(path),
                "w": float(w),
                "dist_e4": float(d * 1e4),
                "in_band": bool(lo <= d <= hi),
                "NO_SUBMIT": True,
            }
        )
        log(f"  staged {path.name} 1-ρ={d*1e4:.4f}e-4 w={w:.4f} in_band={lo<=d<=hi}")

    # ---------- upstream G26 / Q17 / detective ----------
    g26 = g26_status()
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    exe = str(KAG) if KAG.exists() else "kaggle"

    def kfiles(slug):
        try:
            return subprocess.check_output(
                [exe, "datasets", "files", slug], text=True, timeout=60, env=env
            )
        except Exception as e:
            return str(e)[:200]

    def klist_kernels():
        try:
            return subprocess.check_output(
                [exe, "kernels", "list", "--search", "s6e11", "--sort-by", "dateCreated", "-v"],
                text=True,
                timeout=60,
                env=env,
            )[:2500]
        except Exception as e:
            return str(e)[:300]

    upstream = {
        "g26": g26,
        "g26_files": kfiles("goodpjw2008/s6e11-target-first-gpt2-seed-2026"),
        "q17_files": kfiles("amanatar/s6e11-rank1-q17-meta-bundle"),
        "detective_w": kfiles("aphoristicartist/s6e11-detective-weights"),
        "detective_d": kfiles("aphoristicartist/s6e11-detective-data"),
        "recent_kernels": klist_kernels(),
        "q17_audit_exists": Q17_MD.exists(),
        "q17_audit_path": str(Q17_MD),
    }
    # pull executive from prior audit
    q17_exec = ""
    if Q17_MD.exists():
        lines = Q17_MD.read_text().splitlines()
        # first 20 lines of exec
        q17_exec = "\n".join(lines[:25])

    summary = {
        "NO_SUBMIT": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - T0, 2),
        "step1": step1,
        "solo": solo,
        "moves": moves,
        "staged": staged,
        "upstream": {
            "g26_test5fold": bool(g26.get("has_test_5fold_remote") or g26.get("has_test_5fold")),
            "g26_files_head": upstream["g26_files"][:500],
            "q17_files_head": upstream["q17_files"][:400],
            "detective_w_head": upstream["detective_w"][:400],
            "recent_kernels_head": upstream["recent_kernels"][:800],
        },
        "verdict": {
            "gate3_flaw": "Prior Gate3 compared s2026_K10 vs s7_K10; cand2 test uses lm7_test_f0 = s7_K4 only.",
            "best_move": max(
                [("A", moveA["delta_e5"]), ("B", moveB["delta_e5"]), ("C", bestC["delta_e5"])],
                key=lambda x: x[1],
            ),
        },
    }

    # markdown
    lines = []
    lines.append("# GM V12 Stop-Loss & True 2-Seed L7 Fold-0 Upgrade Audit\n\n")
    lines.append(f"**UTC:** {summary['utc']}  \n**`NO_SUBMIT=True`** — all 4 daily submissions held  \n")
    lines.append(f"**Elapsed:** {summary['elapsed_s']}s\n\n")
    lines.append("## Step 1 — Rescue & terminate\n")
    lines.append(f"- Local `oof_0` `{step1['local_oof_0']}`, `test_0` `{step1['local_test_0']}` intact ✓\n")
    lines.append(f"- Rescued `replica.pt` **{step1['replica_pt_mb']} MB**, `fold_0.pt` **{step1['fold_0_pt_mb']} MB** → `preds/l7_s2026_5f/ckpts/`\n")
    lines.append(f"- Terminated pod `{step1['pod_id_terminated']}` (never touched `{step1['never_touch']}`)\n")
    lines.append("- GPU killed before terminate; billing target **$0.00/hr** for L7 pod\n\n")

    lines.append("## Step 2 — Apples-to-apples Fold-0 (K=4 vs 2-seed K=10)\n")
    lines.append("### Flaw in prior Gate 3\n")
    lines.append("- `cand2` / areeb ladder on **test** uses `lm7_test_f0.npy` = **seed=7, Fold-0 only, K=4**.\n")
    lines.append("- Gate 3 residual used `L(s2026_K10) − L(s7_K10)` — **not** matching the K=4 test member.\n")
    lines.append(f"- Recomputed OLD residual lift: **{move_old['delta_e5']:.4f}e-5** (w={move_old['best_w']}).\n")
    lines.append(f"- True Move A vs **K=4**: **{moveA['delta_e5']:.4f}e-5** (w={moveA['best_w']}).\n\n")

    lines.append("### Solo Fold-0 AUCs\n")
    lines.append(f"| member | AUC |\n|---|---:|\n")
    lines.append(f"| L7_s7_K4 (orders 0..3) | {solo['L7_s7_K4']:.7f} |\n")
    lines.append(f"| L7_s7_K10 | {solo['L7_s7_K10']:.7f} |\n")
    lines.append(f"| L7_s2026 bayes K10 | {solo['L7_s2026_bayes']:.7f} |\n")
    lines.append(f"| L7_s2026 best red (`{best_red}`) | {solo['L7_s2026_best_red']:.7f} |\n")
    lines.append(f"| 0.5·L(s7_K10)+0.5·L(s2026) | {solo['two_seed_K10_K10']:.7f} |\n")
    lines.append(f"| 0.5·L(s7_K4)+0.5·L(s2026) | {solo['two_seed_K4_plus_s2026']:.7f} |\n")
    lines.append(f"| hybrid L7F+s2026 best γ={best_g} | {solo['hybrid_best_auc']:.7f} |\n")
    lines.append(f"| corr(s2026, s7_K10) | {solo['corr_s2026_vs_s7_K10']:.6f} |\n")
    lines.append(f"| corr(s2026, s7_K4) | {solo['corr_s2026_vs_s7_K4']:.6f} |\n\n")

    lines.append("### Reduction table (s2026 K=10)\n")
    lines.append("| reduction | AUC |\n|---|---:|\n")
    for k, v in solo["reductions"].items():
        lines.append(f"| {k} | {v['auc']:.7f} |\n")

    lines.append("\n### Full-stack Fold-0 lift on `eval_S7matched_355[f0]`\n")
    lines.append(f"Base F0 AUC = **{base_auc:.7f}**\n\n")
    lines.append("| move | ΔAUC (e-5) | best w | notes |\n|---|---:|---:|---|\n")
    lines.append(
        f"| A: L(s2026_K10)−L(s7_K4) | {moveA['delta_e5']:.4f} | {moveA['best_w']} | Compact-LM analog of G26−MS |\n"
    )
    lines.append(
        f"| B: 0.65·L(L7F)+0.35·L(s2026)−L(s7_K4) | {moveB['delta_e5']:.4f} | {moveB['best_w']} | 5f seed7 + s2026 vs K4 |\n"
    )
    lines.append(
        f"| C: B + orth(LM_trim) | {bestC['delta_e5']:.4f} | wB={bestC['wB']}, wL={bestC['wL']} | joint |\n"
    )
    lines.append(
        f"| OLD Gate3: L(s2026)−L(s7_K10) | {move_old['delta_e5']:.4f} | {move_old['best_w']} | flawed comparator |\n"
    )

    lines.append("\n## Step 3 — Q17 / detective / G26 upstream\n")
    lines.append(f"- G26 `test_5fold` live: **{summary['upstream']['g26_test5fold']}**\n")
    lines.append("### goodpjw2008/s6e11-target-first-gpt2-seed-2026\n```\n")
    lines.append(summary["upstream"]["g26_files_head"] + "\n```\n")
    lines.append("### amanatar Q17 (prior audit)\n```\n")
    lines.append(q17_exec[:1500] + "\n```\n")
    lines.append("### detective-weights\n```\n")
    lines.append(summary["upstream"]["detective_w_head"] + "\n```\n")
    lines.append("### Recent s6e11 kernels (dateCreated)\n```\n")
    lines.append(summary["upstream"]["recent_kernels_head"] + "\n```\n")

    lines.append("\n## Staged NOSUBMIT candidates\n")
    for s in staged:
        lines.append(f"- `{Path(s['path']).name}` — 1−ρ(cand2)={s['dist_e4']:.4f}e-4, w={s['w']:.4f} **NOSUBMIT**\n")
    lines.append("\n**HOLD all 4 daily submissions.** Do not upload.\n")
    lines.append("\n---\n*Generated by `scripts/gm_stoploss_and_f0_k4_to_2seed_k10_audit.py`*\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines))
    # JSON without huge vectors
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log(f"WROTE {OUT_MD}")
    return 0


# bind np for helpers at import time of functions that reference np in annotations only
import numpy as np  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
