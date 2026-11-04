#!/usr/bin/env python3
"""Phase C Pass-1 / Pass-2 nested combiners.

Pass 1: 63 locked Phase-A bank only (run in parallel with Phase B).
Pass 2: bank + Phase B bags; keep B bags only if Pass 2 strictly beats Pass 1
        on nested 5-fold OOF AND wins >= 4/5 folds vs Pass 1.

Scale: within-fold / within-test probit(rank).
Combiners: Nested L2 Logistic C in [0.002..0.1] + batched GPU hillclimb (CPU fallback).
NO auto-submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "logs" / "overnight"
PRED_A = ROOT / "preds" / "phaseA"
PRED_B = ROOT / "preds" / "phaseB"
PRED_C = ROOT / "preds" / "phaseC"
SUBS = ROOT / "submissions"
CKPT = ROOT / "checkpoints" / "overnight"

SINGLE_BEST = 0.951506  # goodpjw gb_bag5
CTRL_OLD = 0.95117
CS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SEED = 42
N_TEST = 299_984
HILL_STEPS = 40

B_BAG_FILES = [
    ("phaseB::oof_glm", PRED_B / "oof_glm.npy", PRED_B / "sub_glm_refit.npy"),
    (
        "phaseB::lgb_glm_boost_l3_l4_refit",
        PRED_B / "oof_lgb_glm_boost_l3_l4_refit.npy",
        PRED_B / "sub_lgb_glm_boost_l3_l4_refit.npy",
    ),
    (
        "phaseB::xgb_cuda_glm_boost_d1_d2_refit",
        PRED_B / "oof_xgb_cuda_glm_boost_d1_d2_refit.npy",
        PRED_B / "sub_xgb_cuda_glm_boost_d1_d2_refit.npy",
    ),
    (
        "phaseB::cat_cuda_glm_boost_d2_refit",
        PRED_B / "oof_cat_cuda_glm_boost_d2_refit.npy",
        PRED_B / "sub_cat_cuda_glm_boost_d2_refit.npy",
    ),
]


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def probit_rank(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average")
    u = np.clip(r / (len(x) + 1.0), 1e-6, 1 - 1e-6)
    return norm.ppf(u)


def within_fold_probit(O: np.ndarray, folds) -> np.ndarray:
    Z = np.zeros_like(O, dtype=np.float64)
    for _, va in folds:
        for j in range(O.shape[1]):
            Z[va, j] = probit_rank(O[va, j])
    return Z


def test_probit(T: np.ndarray) -> np.ndarray:
    Z = np.zeros_like(T, dtype=np.float64)
    for j in range(T.shape[1]):
        col = T[:, j]
        if not np.isfinite(col).all():
            Z[:, j] = 0.0
            continue
        Z[:, j] = probit_rank(col)
    return Z


def fold_aucs(y, oof, folds):
    return [float(roc_auc_score(y[va], oof[va])) for _, va in folds]


def nested_logit(Z, y, folds, C):
    oof = np.zeros(len(y), dtype=np.float64)
    coefs = []
    for tr, va in folds:
        m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-8)
        m.fit(Z[tr], y[tr])
        oof[va] = m.predict_proba(Z[va])[:, 1]
        coefs.append(m.coef_.ravel().copy())
    return oof, coefs


def _hill_one_fold_cpu(Ztr, ytr, max_steps):
    M = Ztr.shape[1]
    w = np.zeros(M, dtype=np.float64)
    best_j = int(np.argmax([roc_auc_score(ytr, Ztr[:, j]) for j in range(M)]))
    w[best_j] = 1.0
    cur_auc = roc_auc_score(ytr, Ztr @ (w / w.sum()))
    for _ in range(max_steps):
        best_gain, best_j = 0.0, None
        for j in range(M):
            w2 = w.copy()
            w2[j] += 1.0
            a = roc_auc_score(ytr, Ztr @ (w2 / w2.sum()))
            if a > cur_auc + 1e-7 and a - cur_auc > best_gain:
                best_gain, best_j = a - cur_auc, j
        if best_j is None:
            break
        w[best_j] += 1.0
        cur_auc = roc_auc_score(ytr, Ztr @ (w / w.sum()))
    return w / max(w.sum(), 1e-12)


def _hill_one_fold_gpu(Ztr, ytr, max_steps, device):
    """Batched GPU trial of +1 weight on each column per step."""
    import torch

    Zt = torch.as_tensor(Ztr, device=device, dtype=torch.float64)
    yt = torch.as_tensor(ytr, device=device, dtype=torch.float64)
    M = Zt.shape[1]
    # AUC via Mann-Whitney on scores (exact for binary)
    pos = yt == 1
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        w = np.zeros(M)
        w[0] = 1.0
        return w

    def auc_batch(scores):  # scores: (B, N)
        # rank within each row
        B, N = scores.shape
        order = torch.argsort(scores, dim=1)
        ranks = torch.empty_like(scores, dtype=torch.float64)
        # average ranks for ties: use dense then adjust — approximate with ordinal for speed
        arange = torch.arange(1, N + 1, device=device, dtype=torch.float64).unsqueeze(0).expand(B, -1)
        ranks.scatter_(1, order, arange)
        # tie correction light: skip (hillclimb relative ok)
        sum_pos = (ranks * pos.unsqueeze(0).to(torch.float64)).sum(dim=1)
        # AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # start from best single
    singles = auc_batch(Zt.T)  # (M,) treating each col as a "batch"
    # Zt.T is (M, N) — good
    best_j = int(torch.argmax(singles).item())
    w = torch.zeros(M, device=device, dtype=torch.float64)
    w[best_j] = 1.0
    cur = auc_batch((Zt @ (w / w.sum())).unsqueeze(0)).item()

    for _ in range(max_steps):
        # trial all M increments in one batch
        W = w.unsqueeze(0).expand(M, -1).clone()
        W[torch.arange(M, device=device), torch.arange(M, device=device)] += 1.0
        Wn = W / W.sum(dim=1, keepdim=True)
        scores = Wn @ Zt.T  # (M, N)
        aucs = auc_batch(scores)
        gain, j = torch.max(aucs - cur, dim=0)
        j = int(j.item())
        if float(gain.item()) <= 1e-7:
            break
        w[j] += 1.0
        cur = float(aucs[j].item())
    wn = (w / w.sum()).detach().cpu().numpy()
    return wn


def nested_hill(Z, y, folds, max_steps=HILL_STEPS, use_gpu=True):
    oof = np.zeros(len(y), dtype=np.float64)
    weights_all = []
    device = None
    backend = "cpu"
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda")
                backend = "cuda"
                print(f"[phaseC] hillclimb backend=cuda device={torch.cuda.get_device_name(0)}", flush=True)
        except Exception as e:
            print(f"[phaseC] GPU hill unavailable ({e}); CPU fallback", flush=True)

    t0 = time.time()
    for fi, (tr, va) in enumerate(folds):
        if device is not None:
            try:
                wn = _hill_one_fold_gpu(Z[tr], y[tr], max_steps, device)
            except Exception as e:
                print(f"[phaseC] GPU hill fold{fi} failed ({e}); CPU", flush=True)
                wn = _hill_one_fold_cpu(Z[tr], y[tr], max_steps)
                backend = "cpu_fallback"
        else:
            wn = _hill_one_fold_cpu(Z[tr], y[tr], max_steps)
        oof[va] = Z[va] @ wn
        weights_all.append(wn)
        print(
            f"[phaseC] hill fold{fi} auc={roc_auc_score(y[va], oof[va]):.6f} backend={backend}",
            flush=True,
        )
    print(f"[phaseC] hill done in {time.time() - t0:.1f}s backend={backend}", flush=True)
    return oof, weights_all, backend


def write_sub(path, ids, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(path, index=False)


def load_bank():
    path = PRED_A / "bank_aligned.npz"
    if not path.exists():
        # fallback locations
        for alt in [
            ROOT / "preds" / "phaseA" / "bank_aligned.npz",
            CKPT / "bank_aligned.npz",
            ROOT / "external_bank" / "bank_aligned.npz",
        ]:
            if alt.exists():
                path = alt
                break
    if not path.exists():
        raise FileNotFoundError(f"bank_aligned.npz not found under {PRED_A}")
    bank = np.load(path, allow_pickle=True)
    names = [str(n) for n in bank["names"]]
    O = bank["O"].astype(np.float64)
    T = bank["T"].astype(np.float64)
    refit_mask = bank["refit_mask"].astype(bool)
    y = bank["y"].astype(int)
    print(f"[phaseC] bank={path} n_members={len(names)} O={O.shape} T={T.shape}", flush=True)
    return names, O, T, refit_mask, y


def inject_phase_b(names, O, T, refit_mask, y, include_glm=True):
    extra_O, extra_T, extra_names, extra_refit = [], [], [], []
    for nm, op, tp in B_BAG_FILES:
        if nm.endswith("oof_glm") and not include_glm:
            # still allow glm if present — include_glm True by default for pass2
            pass
        if op.exists() and tp.exists():
            oo = np.load(op).astype(np.float64).ravel()
            tt = np.load(tp).astype(np.float64).ravel()
            if len(oo) != len(y) or len(tt) != N_TEST:
                print(f"[phaseC] skip {nm} bad shapes {oo.shape} {tt.shape}", flush=True)
                continue
            extra_O.append(oo)
            extra_T.append(tt)
            extra_names.append(nm)
            extra_refit.append(True)
            print(f"[phaseC] + {nm} oof={roc_auc_score(y, oo):.6f}", flush=True)
    if not extra_O:
        return names, O, T, refit_mask, []
    O2 = np.column_stack([O] + extra_O)
    T2 = np.column_stack([T] + extra_T)
    names2 = names + extra_names
    refit2 = np.concatenate([refit_mask, np.array(extra_refit, dtype=bool)])
    return names2, O2, T2, refit2, extra_names


def run_combiners(Z, Zt, y, folds, T, refit_mask, tag):
    results = {}
    # Nested logit sweep
    best_C, best_oof, best_coefs, best_auc = None, None, None, -1.0
    for C in CS:
        oof, coefs = nested_logit(Z, y, folds, C)
        auc = float(roc_auc_score(y, oof))
        fa = fold_aucs(y, oof, folds)
        print(f"[phaseC] {tag} nested_logit C={C} oof={auc:.6f} folds={np.round(fa, 5)}", flush=True)
        if auc > best_auc:
            best_C, best_oof, best_coefs, best_auc = C, oof, coefs, auc
    results["nested_logit"] = {
        "C": best_C,
        "oof": best_auc,
        "folds": fold_aucs(y, best_oof, folds),
        "delta_vs_gb_bag5": best_auc - SINGLE_BEST,
        "delta_vs_ctrl": best_auc - CTRL_OLD,
    }

    # Hillclimb
    hill_oof, hill_w, backend = nested_hill(Z, y, folds, max_steps=HILL_STEPS, use_gpu=True)
    hill_auc = float(roc_auc_score(y, hill_oof))
    hill_fa = fold_aucs(y, hill_oof, folds)
    print(
        f"[phaseC] {tag} nested_hill oof={hill_auc:.6f} folds={np.round(hill_fa, 5)} "
        f"Δgb5={hill_auc - SINGLE_BEST:+.6f} Δctrl={hill_auc - CTRL_OLD:+.6f}",
        flush=True,
    )
    results["nested_hill"] = {
        "oof": hill_auc,
        "folds": hill_fa,
        "backend": backend,
        "delta_vs_gb_bag5": hill_auc - SINGLE_BEST,
        "delta_vs_ctrl": hill_auc - CTRL_OLD,
    }

    # Refit-core subset
    idx = np.where(refit_mask & np.isfinite(T).all(axis=0))[0]
    if len(idx) < 3:
        idx = np.where(np.isfinite(T).all(axis=0))[0]
    Zc = Z[:, idx]
    Tc = T[:, idx]
    Ztc = test_probit(np.nan_to_num(Tc, nan=0.0))
    core_best_C, core_oof, core_coefs, core_auc = None, None, None, -1.0
    for C in CS:
        oof, coefs = nested_logit(Zc, y, folds, C)
        auc = float(roc_auc_score(y, oof))
        if auc > core_auc:
            core_best_C, core_oof, core_coefs, core_auc = C, oof, coefs, auc
    core_fa = fold_aucs(y, core_oof, folds)
    print(
        f"[phaseC] {tag} refit_core n={len(idx)} C={core_best_C} oof={core_auc:.6f} folds={np.round(core_fa, 5)}",
        flush=True,
    )
    results["refit_core"] = {
        "C": core_best_C,
        "n_members": int(len(idx)),
        "oof": core_auc,
        "folds": core_fa,
        "delta_vs_gb_bag5": core_auc - SINGLE_BEST,
        "delta_vs_ctrl": core_auc - CTRL_OLD,
    }

    # Primary = best OOF among three
    cands = [
        ("nested_logit", best_oof, best_coefs, "logit", best_C, Z, Zt),
        ("nested_hill", hill_oof, hill_w, "hill", None, Z, Zt),
        ("refit_core", core_oof, core_coefs, "logit", core_best_C, Zc, Ztc),
    ]
    primary = max(cands, key=lambda t: results[t[0]]["oof"])
    pname, poof, pcoef, pkind, pC, pZ, pZt = primary
    print(
        f"[phaseC] {tag} PRIMARY={pname} oof={results[pname]['oof']:.6f} "
        f"vs gb_bag5={SINGLE_BEST:.6f} ({results[pname]['oof'] - SINGLE_BEST:+.6f}) "
        f"vs ctrl={CTRL_OLD:.6f} ({results[pname]['oof'] - CTRL_OLD:+.6f})",
        flush=True,
    )

    # Test preds
    if pkind == "logit":
        m = LogisticRegression(C=pC, max_iter=3000, solver="lbfgs")
        m.fit(pZ, y)
        sub_A = m.predict_proba(pZt)[:, 1]
    else:
        w = np.mean(np.stack(pcoef), axis=0)
        sub_A = pZt @ w

    m = LogisticRegression(C=core_best_C, max_iter=3000, solver="lbfgs")
    m.fit(Zc, y)
    sub_B = m.predict_proba(Ztc)[:, 1]

    return {
        "results": results,
        "primary": pname,
        "primary_oof": float(results[pname]["oof"]),
        "primary_folds": results[pname]["folds"],
        "oof": poof,
        "sub_A": sub_A,
        "sub_B": sub_B,
        "core_idx": idx,
    }


def pass1(test_ids):
    names, O, T, refit_mask, y = load_bank()
    assert O.shape[1] == len(names)
    folds = locked_folds(y)
    # single-best check
    try:
        j = names.index("goodpjw_artifacts__gb_bag5_oof_npy")
        sb = float(roc_auc_score(y, O[:, j]))
        print(f"[phaseC] pass1 single-best gb_bag5 oof={sb:.6f} (expected ~{SINGLE_BEST})", flush=True)
    except ValueError:
        print("[phaseC] pass1: gb_bag5 name not in bank; using constant", flush=True)

    Z = within_fold_probit(O, folds)
    Zt = test_probit(np.nan_to_num(T, nan=0.0))
    out = run_combiners(Z, Zt, y, folds, T, refit_mask, tag="pass1")

    PRED_C.mkdir(parents=True, exist_ok=True)
    np.save(PRED_C / "pass1_oof.npy", out["oof"])
    np.save(PRED_C / "pass1_sub_A.npy", out["sub_A"])
    np.save(PRED_C / "pass1_sub_B.npy", out["sub_B"])

    write_sub(SUBS / "sub_A_pass1_bank63_nested.csv", test_ids, out["sub_A"])
    write_sub(SUBS / "sub_B_pass1_refit_top.csv", test_ids, out["sub_B"])
    # also keep legacy names for overnight compatibility
    write_sub(SUBS / "sub_A_part6_nested_logit.csv", test_ids, out["sub_A"])
    write_sub(SUBS / "sub_B1_refit_core_only.csv", test_ids, out["sub_B"])
    write_sub(SUBS / "sub_B_refit_core_or_fusion.csv", test_ids, out["sub_B"])

    report = {
        "pass": 1,
        "n_members": len(names),
        "single_best_gb_bag5": SINGLE_BEST,
        "ctrl_old": CTRL_OLD,
        "primary": out["primary"],
        "primary_oof": out["primary_oof"],
        "primary_folds": out["primary_folds"],
        "delta_vs_gb_bag5": out["primary_oof"] - SINGLE_BEST,
        "delta_vs_ctrl": out["primary_oof"] - CTRL_OLD,
        "results": out["results"],
        "files": [
            "submissions/sub_A_pass1_bank63_nested.csv",
            "submissions/sub_B_pass1_refit_top.csv",
        ],
        "policy": "NO_AUTO_SUBMIT",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    (OUT / "phaseC_pass1_summary.json").write_text(json.dumps(report, indent=2))
    (CKPT / "phaseC_pass1_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(
        f"[phaseC] PASS1 DONE nested_oof={out['primary_oof']:.6f} "
        f"vs gb_bag5 {SINGLE_BEST:.6f} ({out['primary_oof'] - SINGLE_BEST:+.6f}) "
        f"vs ctrl {CTRL_OLD:.6f} ({out['primary_oof'] - CTRL_OLD:+.6f})",
        flush=True,
    )
    return 0


def pass2(test_ids):
    names, O, T, refit_mask, y = load_bank()
    folds = locked_folds(y)

    # Require Pass-1 summary
    p1_path = OUT / "phaseC_pass1_summary.json"
    if not p1_path.exists():
        p1_path = CKPT / "phaseC_pass1_summary.json"
    if not p1_path.exists():
        print("[phaseC] pass2: no pass1 summary — running pass1 first", flush=True)
        pass1(test_ids)
        p1_path = OUT / "phaseC_pass1_summary.json"
    p1 = json.loads(p1_path.read_text())
    p1_oof = float(p1["primary_oof"])
    p1_folds = list(p1["primary_folds"])
    print(f"[phaseC] pass2 baseline pass1 oof={p1_oof:.6f} folds={np.round(p1_folds, 5)}", flush=True)

    names2, O2, T2, refit2, added = inject_phase_b(names, O, T, refit_mask, y)
    if not added:
        print("[phaseC] pass2: no Phase B bags on disk — keeping Pass-1", flush=True)
        report = {
            "pass": 2,
            "kept": "pass1",
            "reason": "no_phaseB_bags",
            "pass1_oof": p1_oof,
            "policy": "NO_AUTO_SUBMIT",
        }
        (OUT / "phaseC_pass2_summary.json").write_text(json.dumps(report, indent=2))
        return 0

    Z = within_fold_probit(O2, folds)
    Zt = test_probit(np.nan_to_num(T2, nan=0.0))
    out = run_combiners(Z, Zt, y, folds, T2, refit2, tag="pass2")

    p2_oof = out["primary_oof"]
    p2_folds = out["primary_folds"]
    fold_wins = sum(1 for a, b in zip(p2_folds, p1_folds) if a > b + 1e-12)
    strict_beat = (p2_oof > p1_oof + 1e-12) and (fold_wins >= 4)
    print(
        f"[phaseC] pass2 vs pass1: oof {p2_oof:.6f} vs {p1_oof:.6f} "
        f"(Δ={p2_oof - p1_oof:+.6f}) fold_wins={fold_wins}/5 keep_B={strict_beat}",
        flush=True,
    )

    PRED_C.mkdir(parents=True, exist_ok=True)
    np.save(PRED_C / "pass2_oof.npy", out["oof"])
    np.save(PRED_C / "pass2_sub_A.npy", out["sub_A"])
    np.save(PRED_C / "pass2_sub_B.npy", out["sub_B"])

    if strict_beat:
        write_sub(SUBS / "sub_A_pass2_bank63_plus_B_nested.csv", test_ids, out["sub_A"])
        write_sub(SUBS / "sub_B_pass2_refit_top.csv", test_ids, out["sub_B"])
        write_sub(SUBS / "sub_A_part6_nested_logit.csv", test_ids, out["sub_A"])
        write_sub(SUBS / "sub_B1_refit_core_only.csv", test_ids, out["sub_B"])
        write_sub(SUBS / "sub_B_refit_core_or_fusion.csv", test_ids, out["sub_B"])
        kept = "pass2"
        reason = f"pass2_beat_pass1_oof_and_{fold_wins}/5_folds"
    else:
        # keep pass1 artifacts as primary
        kept = "pass1"
        reason = f"pass2_did_not_strictly_beat (Δ={p2_oof - p1_oof:+.6f}, wins={fold_wins}/5)"
        print(f"[phaseC] KEEP PASS1 — {reason}", flush=True)

    report = {
        "pass": 2,
        "kept": kept,
        "reason": reason,
        "pass1_oof": p1_oof,
        "pass1_folds": p1_folds,
        "pass2_oof": p2_oof,
        "pass2_folds": p2_folds,
        "fold_wins_vs_pass1": fold_wins,
        "phaseB_members_added": added,
        "n_members": len(names2),
        "primary": out["primary"],
        "results": out["results"],
        "delta_vs_gb_bag5": p2_oof - SINGLE_BEST,
        "delta_vs_ctrl": p2_oof - CTRL_OLD,
        "policy": "NO_AUTO_SUBMIT",
    }
    (OUT / "phaseC_pass2_summary.json").write_text(json.dumps(report, indent=2))
    (CKPT / "phaseC_pass2_summary.json").write_text(json.dumps(report, indent=2))
    (OUT / "phaseC_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SUBS.mkdir(parents=True, exist_ok=True)
    PRED_C.mkdir(parents=True, exist_ok=True)
    pass_n = int(os.environ.get("PHASE_C_PASS", "1"))
    test_ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])["id"].values
    assert len(test_ids) == N_TEST
    print(f"[phaseC] PASS={pass_n} start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    if pass_n == 1:
        return pass1(test_ids)
    if pass_n == 2:
        return pass2(test_ids)
    raise SystemExit(f"unknown PHASE_C_PASS={pass_n}")


if __name__ == "__main__":
    sys.exit(main())
