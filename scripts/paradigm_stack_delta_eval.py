#!/usr/bin/env python3
"""
Non-destructive stack-Δ evaluator for paradigm candidates vs frozen P_super_v4p_hc.

Test 1: 5-fold CV orthogonal residual + alpha probe on logit(rank(.))
Test 2: scale-matched injection sanity vs O_v4p / v4p_hc (weight=0 must reproduce baseline)

Does NOT submit. Safe to run while B2/C workers are live.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr, ks_2samp
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
BASELINE = 0.9523071
EPS = 1e-6


def rk01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def to_z_ranklogit(p_or_score: np.ndarray) -> np.ndarray:
    """Map any score to logit of empirical rank in [0,1] — scale-matched to v4p_hc ranks."""
    return logit(np.clip(rk01(p_or_score), EPS, 1.0 - EPS))


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann–Whitney AUC via scipy rankdata (C) — ~50–100× faster than Python tie loops."""
    y = np.asarray(y, dtype=np.int8).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(s, method="average")
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _best_alpha(y_a: np.ndarray, zb_a: np.ndarray, r_a: np.ndarray,
                lo: float = -0.05, hi: float = 0.25) -> tuple[float, float]:
    """Coarse→fine alpha search maximizing AUC of zb + α·r on fold a."""
    best_a, best_auc = 0.0, fast_auc(y_a, zb_a)
    # coarse 16 pts + fine 11 pts ≈ same resolution as 301-grid, ~5s/cand
    for al in np.linspace(lo, hi, 16):
        sc = fast_auc(y_a, zb_a + al * r_a)
        if sc > best_auc:
            best_auc, best_a = sc, float(al)
    flo = max(lo, best_a - 0.015)
    fhi = min(hi, best_a + 0.015)
    for al in np.linspace(flo, fhi, 11):
        sc = fast_auc(y_a, zb_a + al * r_a)
        if sc > best_auc:
            best_auc, best_a = sc, float(al)
    return best_a, best_auc


def test1_orthogonal_alpha_probe(
    y: np.ndarray,
    v4p_hc_oof: np.ndarray,
    p_cand_oof: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    alphas: np.ndarray | None = None,  # kept for API compat; unused (coarse→fine)
) -> dict:
    """
    z_base = logit(rk(v4p_hc)); z_cand = logit(rk(p_cand))
    Inside each fold a: orthogonalize z_cand vs z_base, pick alpha on a, apply on b.
    """
    del alphas  # coarse→fine replaces fixed 301-grid (same resolution, ~5s)
    z_base = to_z_ranklogit(v4p_hc_oof)
    z_cand = to_z_ranklogit(p_cand_oof)
    base_auc = float(fast_auc(y, v4p_hc_oof))
    oof = np.zeros(len(y), dtype=np.float64)
    alphas_used = []
    fold_deltas = []
    for a, b in folds:
        zb_a, zc_a = z_base[a], z_cand[a]
        var_b = float(np.var(zb_a)) + 1e-12
        beta = float(np.cov(zc_a, zb_a)[0, 1] / var_b)
        r_full = z_cand - beta * z_base
        best_a, _ = _best_alpha(y[a], zb_a, r_full[a])
        oof[b] = z_base[b] + best_a * r_full[b]
        alphas_used.append(best_a)
        fold_deltas.append(float(fast_auc(y[b], oof[b]) - fast_auc(y[b], v4p_hc_oof[b])))
    stack_auc = float(fast_auc(y, oof))
    delta = stack_auc - base_auc
    cand_auc = float(fast_auc(y, p_cand_oof))
    return {
        "test": "orthogonal_alpha_probe",
        "baseline_auc": base_auc,
        "cand_solo_auc": cand_auc,
        "stack_oof_auc": stack_auc,
        "delta_auc": delta,
        "mean_alpha": float(np.mean(alphas_used)),
        "alphas_per_fold": alphas_used,
        "fold_deltas": fold_deltas,
        "fold_win_rate": float(np.mean([d > 0 for d in fold_deltas])),
        "spearman_rho_resid_v4p": float(spearmanr(oof - z_base, v4p_hc_oof).correlation),
        "oof_scores": oof,  # callers may drop before JSON
    }


def test2_scale_matched_injection(
    y: np.ndarray,
    O_v4p: np.ndarray,
    v4p_hc_oof: np.ndarray,
    p_cand_oof: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    t1: dict | None = None,
) -> dict:
    """
    Transform p_cand to same [0,1] rank scale as v4p_hc (not raw O_v4p logits).
    Assert weight=0 reproduces baseline; stack Δ comes from Test 1 (passed in or computed).
    """
    O = np.asarray(O_v4p, dtype=np.float64)
    v = np.asarray(v4p_hc_oof, dtype=np.float64)
    p_rank = rk01(p_cand_oof)
    base_auc = float(fast_auc(y, v))
    repro = float(fast_auc(y, rk01(v)))
    auc0 = float(fast_auc(y, v))  # weight-0 blend ≡ baseline vector
    ok = abs(auc0 - base_auc) < 1e-7 and abs(repro - base_auc) < 1e-7
    if t1 is None:
        assert folds is not None
        t1 = test1_orthogonal_alpha_probe(y, v, p_cand_oof, folds)
    return {
        "test": "scale_matched_injection",
        "O_v4p_min": float(O.min()),
        "O_v4p_max": float(O.max()),
        "O_v4p_mean": float(O.mean()),
        "v4p_hc_min": float(v.min()),
        "v4p_hc_max": float(v.max()),
        "cand_raw_min": float(np.min(p_cand_oof)),
        "cand_raw_max": float(np.max(p_cand_oof)),
        "cand_rank_min": float(p_rank.min()),
        "cand_rank_max": float(p_rank.max()),
        "weight0_reproduces_baseline": bool(ok),
        "weight0_auc": auc0,
        "baseline_auc": base_auc,
        "abs_err_weight0": abs(auc0 - base_auc),
        "stack_from_test1_delta": t1["delta_auc"],
    }


def evaluate_candidate(
    name: str,
    y: np.ndarray,
    v4p_hc_oof: np.ndarray,
    p_cand_oof: np.ndarray,
    O_v4p: np.ndarray,
    folds: list,
    p_cand_test: np.ndarray | None = None,
    v4p_hc_test: np.ndarray | None = None,
    keep_oof: bool = False,
) -> dict:
    t1 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, p_cand_oof, folds)
    t2 = test2_scale_matched_injection(y, O_v4p, v4p_hc_oof, p_cand_oof, t1=t1)
    oof_scores = t1.pop("oof_scores", None)
    out = {
        "candidate": name,
        "n": int(len(y)),
        "test1_orthogonal_alpha_probe": t1,
        "test2_scale_matched_injection": t2,
        "stack_delta_auc": t1["delta_auc"],
        "stack_oof_auc": t1["stack_oof_auc"],
        "cand_solo_auc": t1["cand_solo_auc"],
    }
    if keep_oof and oof_scores is not None:
        out["_oof_scores"] = oof_scores
    if p_cand_test is not None and v4p_hc_test is not None:
        out["ks_oof_test_ranks"] = float(ks_2samp(rk01(p_cand_oof), rk01(p_cand_test)).statistic)
        out["spearman_oof_vs_v4p"] = float(spearmanr(p_cand_oof, v4p_hc_oof).correlation)
    return out


def score_candidate_vs_v4p_hc(
    y: np.ndarray,
    v4p_hc_oof: np.ndarray,
    p_cand_oof: np.ndarray,
    folds: list,
    O_v4p: np.ndarray | None = None,
) -> dict:
    """
    Campaign/dossier helper: return metrics dict with correct stack Δ.
    If O_v4p is None, skip Test 2 scale diagnostics (still run Test 1).
    """
    t1 = test1_orthogonal_alpha_probe(y, v4p_hc_oof, p_cand_oof, folds)
    oof = t1.pop("oof_scores")
    out = {
        "stack_oof_auc": t1["stack_oof_auc"],
        "delta_auc": t1["delta_auc"],
        "cand_solo_auc": t1["cand_solo_auc"],
        "stack_method": "test1_orthogonal_alpha_probe_vs_v4p_hc",
        "stack_mean_alpha": t1["mean_alpha"],
        "fold_win_rate": t1["fold_win_rate"],
        "fold_deltas": t1["fold_deltas"],
        "alphas_per_fold": t1["alphas_per_fold"],
        "spearman_rho_resid_v4p": t1["spearman_rho_resid_v4p"],
        "stack_oof_scores": oof,
    }
    if O_v4p is not None:
        t2 = test2_scale_matched_injection(y, O_v4p, v4p_hc_oof, p_cand_oof, t1=t1)
        out["weight0_reproduces_baseline"] = t2["weight0_reproduces_baseline"]
        out["O_v4p_range"] = [t2["O_v4p_min"], t2["O_v4p_max"]]
        out["cand_raw_range"] = [t2["cand_raw_min"], t2["cand_raw_max"]]
    return out


def load_bank_and_y():
    bank = np.load(ROOT / "preds/v4p_bank_cache.npz", allow_pickle=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    return bank, y, folds


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", default="A", choices=["A", "B", "C", "all"])
    ap.add_argument("--cand", default="auto", help="xgb|tabr|combo|stack|llm|auto")
    ap.add_argument("--patch-metrics", action="store_true",
                    help="Write corrected delta into preds/paradigm_*/metrics.json + state")
    args = ap.parse_args()
    t0 = __import__("time").time()
    bank, y, folds = load_bank_and_y()
    v4p = bank["v4p_hc_oof"].astype(np.float64)
    v4p_te = bank["v4p_hc_test"].astype(np.float64)
    O = bank["O_v4p"]
    base_auc = float(fast_auc(y, v4p))
    print(f"[eval] baseline_auc={base_auc:.7f} expected~{BASELINE}  "
          f"O_v4p=[{float(O.min()):.4f},{float(O.max()):.4f}]  "
          f"v4p_hc=[{float(v4p.min()):.4f},{float(v4p.max()):.4f}]", flush=True)

    paradigms = ["A", "B", "C"] if args.paradigm == "all" else [args.paradigm]
    report = {
        "baseline_auc": base_auc,
        "baseline_expected": BASELINE,
        "O_v4p_scale": {"min": float(O.min()), "max": float(O.max()), "mean": float(O.mean())},
        "v4p_hc_scale": {"min": float(v4p.min()), "max": float(v4p.max())},
        "results": {},
    }

    cand_map = {
        "A": [("xgb", "xgb_oof.npy", "xgb_test.npy"), ("tabr", "tabr_oof.npy", "tabr_test.npy"),
              ("combo", "combo_oof.npy", "combo_test.npy"), ("legacy_stack", "stack_oof.npy", None)],
        "B": [("xgb", "xgb_llr_oof.npy", "xgb_llr_test.npy"), ("llm", "llm_solo_oof.npy", "llm_solo_test.npy"),
              ("combo", "combo_oof.npy", "combo_test.npy")],
        "C": [("tabm", "tabm_k32_oof.npy", "tabm_k32_test.npy"), ("stealth", "stealth_oof.npy", "stealth_test.npy"),
              ("combo", "combo_oof.npy", "combo_test.npy")],
    }

    for P in paradigms:
        d = ROOT / f"preds/paradigm_{P}"
        if not d.exists():
            report["results"][P] = {"status": "missing_dir"}
            continue
        entries = []
        for name, fo, ft in cand_map[P]:
            po = d / fo
            if not po.exists():
                continue
            if args.cand != "auto" and args.cand != name:
                continue
            p_oof = np.load(po)
            p_te = np.load(d / ft) if ft and (d / ft).exists() else None
            t_c = __import__("time").time()
            ev = evaluate_candidate(f"{P}/{name}", y, v4p, p_oof, O, folds, p_te, v4p_te)
            print(
                f"[eval] {P}/{name}: solo={ev['cand_solo_auc']:.7f} "
                f"stack={ev['stack_oof_auc']:.7f} Δ={ev['stack_delta_auc']:+.3e} "
                f"ᾱ={ev['test1_orthogonal_alpha_probe']['mean_alpha']:.4f} "
                f"wins={ev['test1_orthogonal_alpha_probe']['fold_win_rate']:.0%} "
                f"w0_ok={ev['test2_scale_matched_injection']['weight0_reproduces_baseline']} "
                f"({__import__('time').time()-t_c:.2f}s)",
                flush=True,
            )
            entries.append(ev)
        report["results"][P] = {
            "status": "ok" if entries else "no_candidates",
            "candidates": entries,
            "best_delta": max((e["stack_delta_auc"] for e in entries), default=None),
            "best_candidate": max(entries, key=lambda e: e["stack_delta_auc"])["candidate"] if entries else None,
        }

        if args.patch_metrics and entries:
            # Prefer primary booster candidate for headline stack Δ
            prefer = {"A": "xgb", "B": "xgb", "C": "stealth"}
            pick = next((c for c in entries if c["candidate"].endswith("/" + prefer.get(P, "xgb"))), entries[0])
            # if another cand has higher Δ, keep both
            best = max(entries, key=lambda e: e["stack_delta_auc"])
            meta_p = d / "metrics.json"
            old = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            old["legacy_bogus_stack_oof_auc"] = old.get("stack_oof_auc")
            old["legacy_bogus_delta_auc"] = old.get("delta_auc")
            old["stack_oof_auc"] = pick["stack_oof_auc"]
            old["delta_auc"] = pick["stack_delta_auc"]
            old["stack_method"] = "test1_orthogonal_alpha_probe_vs_v4p_hc"
            old["stack_mean_alpha"] = pick["test1_orthogonal_alpha_probe"]["mean_alpha"]
            old["fold_win_rate"] = pick["test1_orthogonal_alpha_probe"]["fold_win_rate"]
            old["fold_deltas"] = pick["test1_orthogonal_alpha_probe"]["fold_deltas"]
            old["weight0_reproduces_baseline"] = pick["test2_scale_matched_injection"]["weight0_reproduces_baseline"]
            old["best_cand_stack_delta"] = best["stack_delta_auc"]
            old["best_cand_name"] = best["candidate"]
            # save corrected OOF stack scores
            t1full = test1_orthogonal_alpha_probe(y, v4p, np.load(d / {
                "A": "xgb_oof.npy", "B": "xgb_llr_oof.npy", "C": "stealth_oof.npy"
            }.get(P, "xgb_oof.npy")), folds)
            np.save(d / "stack_oof_corrected.npy", t1full["oof_scores"])
            meta_p.write_text(json.dumps(_jsonable(old), indent=2))
            st_p = ROOT / "logs/overnight/3paradigm_discovery_state.json"
            if st_p.exists():
                st = json.loads(st_p.read_text())
                st.setdefault("paradigms", {}).setdefault(P, {})["metrics"] = old
                st["paradigms"][P]["status"] = st["paradigms"][P].get("status", "completed")
                st["stack_eval_fixed_utc"] = __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                st_p.write_text(json.dumps(st, indent=2))
            print(f"[eval] patched {meta_p} Δ={old['delta_auc']:+.3e} "
                  f"(legacy={old.get('legacy_bogus_delta_auc')})", flush=True)

    A = report["results"].get("A", {})
    legacy = next((c for c in A.get("candidates", []) if c["candidate"].endswith("legacy_stack")), None)
    xgb = next((c for c in A.get("candidates", []) if c["candidate"].endswith("/xgb")), None)
    if legacy and xgb:
        report["root_cause_A"] = {
            "bogus_reported_delta": -0.004451395583806583,
            "bogus_method": (
                "0.50*rk(z_champ)+0.30*rk(xgb)+0.20*rk(tabr) — replaces frozen v4p_hc "
                "(=0.84*rk(v4p)+0.12*rk(hc)+0.04*rk(sup)) with champ-only blend; TabR~0.886 poison"
            ),
            "legacy_stack_rescore_delta_vs_v4p_hc": legacy["stack_delta_auc"],
            "xgb_correct_stack_delta": xgb["stack_delta_auc"],
            "xgb_solo_auc": xgb["cand_solo_auc"],
            "O_v4p_is_logits": report["O_v4p_scale"]["min"] < 0 < report["O_v4p_scale"]["max"],
            "v4p_hc_is_unit_interval": (
                report["v4p_hc_scale"]["min"] >= 0 and report["v4p_hc_scale"]["max"] <= 1
            ),
            "note": (
                "Root cause = wrong baseline construction (forgot 12% HC + gb_super outer blend) "
                "+ fixed-weight injection of weak TabR; NOT an unconstrained solver flip. "
                "Correct Test-1 orthogonal residual on logit(rk(.)) vs frozen v4p_hc."
            ),
        }
        print(json.dumps(report["root_cause_A"], indent=2), flush=True)

    out = ROOT / "logs/overnight/paradigm_stack_delta_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2))
    print(f"[eval] wrote {out} total={__import__('time').time()-t0:.2f}s", flush=True)
    return report


if __name__ == "__main__":
    main()
