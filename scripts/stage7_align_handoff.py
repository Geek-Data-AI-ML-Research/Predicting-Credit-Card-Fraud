#!/usr/bin/env python3
"""Post-7.1 / post-7.2: fold-align OOF scores, gate vs nested_hill, emit calibrated logits.

Fixes inter-fold scale mismatch that silently hurts pooled roc_auc_score(y, oof).
Writes campaign_best_* for Stage 8.2/8.3 handoff (Platt → logit @ base rate ~0.017).

NO kaggle submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "stage7")
EPS = 1e-6
SEED = 42
BASE_RATE = 0.017  # ≈ train fraud rate; logit ≈ -4.05


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def fold_probit_align(raw: np.ndarray, folds) -> np.ndarray:
    """Per-fold ndtri((rank+0.5)/n_va) so pooled OOF is scale-aligned."""
    out = np.zeros_like(raw, dtype=np.float64)
    for _, va in folds:
        r = rankdata(raw[va], method="average")
        u = np.clip((r - 0.5) / len(va), EPS, 1 - EPS)
        out[va] = ndtri(u)
    return out


def fold_zscore_align(raw: np.ndarray, folds, y: np.ndarray) -> np.ndarray:
    """Per-fold standardize using train-side mean/std of the same raw scores."""
    out = np.zeros_like(raw, dtype=np.float64)
    for tr, va in folds:
        mu, sd = float(raw[tr].mean()), float(raw[tr].std()) + 1e-12
        out[va] = (raw[va] - mu) / sd
    return out


def platt_calibrate_logits(score_oof: np.ndarray, score_te: np.ndarray, y: np.ndarray, folds):
    """Nested Platt: fit 1D LR on tr scores → probabilities, then logit; average test.

    Ensures mean P ≈ base rate rather than 0.5 (critical for XGB base_margin).
    """
    ntr, nte = len(y), len(score_te)
    p_oof = np.zeros(ntr, np.float64)
    p_te = np.zeros(nte, np.float64)
    for tr, va in folds:
        lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
        lr.fit(score_oof[tr].reshape(-1, 1), y[tr])
        p_oof[va] = lr.predict_proba(score_oof[va].reshape(-1, 1))[:, 1]
        p_te += lr.predict_proba(score_te.reshape(-1, 1))[:, 1]
    p_te /= len(folds)
    # optional global shift toward base rate (tiny) — nested Platt usually enough
    mean_p = float(p_oof.mean())
    L_oof = to_logit(p_oof)
    L_te = to_logit(p_te)
    return L_oof, L_te, p_oof, p_te, mean_p


def load_ctrl_hill(y, folds, ntr):
    """Prefer saved ctrl OOF; else folds from 7.1 JSON + rebuild scores; else rebuild."""
    p = os.path.join(PRED, "ctrl_nested_hill_oof.npy")
    if os.path.exists(p):
        o = np.load(p).astype(np.float64)
        folds_auc = [float(roc_auc_score(y[va], o[va])) for _, va in folds]
        return o, float(roc_auc_score(y, o)), folds_auc

    # Use exact fold AUCs / OOF from stage7_1 JSON when vector missing
    jpath = os.path.join(LOGS, "stage7_1_gpu_result.json")
    json_folds = json_oof = None
    if os.path.exists(jpath):
        d = json.load(open(jpath))
        c = d.get("controls", {}).get("nested_hillclimb_mix", {})
        json_folds = c.get("folds")
        json_oof = c.get("oof")

    z = np.load(os.path.join(PRED, "meta_bank.npz"), allow_pickle=True)
    O = z["O"].astype(np.float64)
    Orank = np.column_stack([rankdata(O[:, j]) / ntr for j in range(O.shape[1])])
    Ologit = to_logit(O)
    ctrl = np.zeros(ntr)
    for tr, va in folds:
        # fast: equal-weight top-8 by train AUC instead of full hillclimb
        aucs = [roc_auc_score(y[tr], Orank[tr, j]) for j in range(Orank.shape[1])]
        top = np.argsort(-np.asarray(aucs))[:8]
        w = np.zeros(Orank.shape[1])
        w[top] = 1.0 / len(top)
        hill_va = Orank[va] @ w
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(Ologit[tr], y[tr])
        stack_va = lr.decision_function(Ologit[va])
        ctrl[va] = rankdata(hill_va) / len(va) + rankdata(stack_va) / len(va)
    folds_auc = [float(roc_auc_score(y[va], ctrl[va])) for _, va in folds]
    oof = float(roc_auc_score(y, ctrl))
    # Prefer exact numbers from the live 7.1 run when available
    if json_folds is not None and len(json_folds) == 5:
        folds_auc = [float(x) for x in json_folds]
    if json_oof is not None:
        oof = float(json_oof)
    return ctrl, oof, folds_auc


def candidate_paths():
    """(name, oof_path, test_path) — include nested_hill via ctrl file."""
    names = [
        "nested_nn_logit_probit",
        "nested_probit_hillclimb",
        "nested_hybrid_rank_logit_mix",
        "nested_pruned_ridge_pos",
        "gated_meta_net",
        "meta_xgb_cuda",
        "blend_71_gated_xgb",
        "best_7_1",
        "best_7_2",
    ]
    out = []
    for n in names:
        op, tp = os.path.join(PRED, f"{n}_oof.npy"), os.path.join(PRED, f"{n}_test.npy")
        if os.path.exists(op) and os.path.exists(tp):
            out.append((n, op, tp))
    return out


def main():
    t0 = time.time()
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["is_fraud"])["is_fraud"].astype(int).values
    ntr = len(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))
    ctrl, ctrl_oof, ctrl_folds = load_ctrl_hill(y, folds, ntr)
    np.save(os.path.join(PRED, "ctrl_nested_hill_oof.npy"), ctrl.astype(np.float32))
    print(f"nested_hill OOF={ctrl_oof:.6f} folds={[round(a,6) for a in ctrl_folds]}", flush=True)
    print(
        f"  fold mean={np.mean(ctrl_folds):.6f}  "
        f"f0Δmean={ctrl_folds[0]-np.mean(ctrl_folds):+.6f}  "
        f"f1Δmean={ctrl_folds[1]-np.mean(ctrl_folds):+.6f}",
        flush=True,
    )

    rows = []
    best = None  # (aligned_pooled_auc, name, aligned_oof, test, meta)

    # always consider nested_hill itself as handoff candidate
    ctrl_aligned = fold_probit_align(ctrl, folds)
    ctrl_aligned_auc = float(roc_auc_score(y, ctrl_aligned))
    # test for ctrl: use mean of bank? skip — prefer arm test; for hill use rank of equal blend
    z = np.load(os.path.join(PRED, "meta_bank.npz"), allow_pickle=True)
    T = z["T"].astype(np.float64)
    # proxy test: mean of top-8 train-AUC members
    O = z["O"].astype(np.float64)
    aucs = [roc_auc_score(y, O[:, j]) for j in range(O.shape[1])]
    top = np.argsort(-np.asarray(aucs))[:8]
    ctrl_te = T[:, top].mean(1)
    rows.append(
        {
            "name": "nested_hill",
            "raw_pooled_auc": ctrl_oof,
            "mean_fold_auc": float(np.mean(ctrl_folds)),
            "fold_aucs": ctrl_folds,
            "aligned_probit_pooled_auc": ctrl_aligned_auc,
            "fold_deltas_vs_ctrl": [0.0] * 5,
            "fold_wins": 5,
        }
    )
    best = (ctrl_aligned_auc, "nested_hill", ctrl_aligned, ctrl_te, {"source": "control"})

    for name, op, tp in candidate_paths():
        raw = np.load(op).astype(np.float64)
        te = np.load(tp).astype(np.float64)
        fold_aucs = [float(roc_auc_score(y[va], raw[va])) for _, va in folds]
        raw_pooled = float(roc_auc_score(y, raw))
        aligned = fold_probit_align(raw, folds)
        # also zscore path for reporting
        aligned_z = fold_zscore_align(raw, folds, y)
        ap = float(roc_auc_score(y, aligned))
        az = float(roc_auc_score(y, aligned_z))
        deltas = [fa - c for fa, c in zip(fold_aucs, ctrl_folds)]
        wins = sum(1 for d in deltas if d > 1e-12)
        mean_fold = float(np.mean(fold_aucs))
        gate_raw = promotion_gate(ctrl_folds, fold_aucs, ctrl_oof, raw_pooled)
        gate_aligned = promotion_gate(ctrl_folds, fold_aucs, ctrl_oof, ap)
        # promote on aligned pooled OR mean-fold, with fold wins
        pass_aligned = (ap > ctrl_oof or mean_fold > float(np.mean(ctrl_folds))) and wins >= 4
        rec = {
            "name": name,
            "raw_pooled_auc": raw_pooled,
            "mean_fold_auc": mean_fold,
            "fold_aucs": fold_aucs,
            "aligned_probit_pooled_auc": ap,
            "aligned_zscore_pooled_auc": az,
            "fold_deltas_vs_ctrl": deltas,
            "fold_wins": wins,
            "gate_raw": gate_raw,
            "gate_aligned_pooled": gate_aligned,
            "pass_aligned": pass_aligned,
            "scale_gap_raw_minus_aligned": raw_pooled - ap,
        }
        rows.append(rec)
        print(
            f"{name}: folds={[round(a,6) for a in fold_aucs]} "
            f"Δ={[round(d,6) for d in deltas]} wins={wins}/5 "
            f"raw={raw_pooled:.6f} aligned_p={ap:.6f} meanF={mean_fold:.6f} "
            f"scale_gap={raw_pooled-ap:+.6f}",
            flush=True,
        )
        # fold0/1 spotlight vs nested_hill
        if name in (
            "nested_probit_hillclimb",
            "nested_hybrid_rank_logit_mix",
            "best_7_1",
        ):
            print(
                f"  SPOTLIGHT {name}: fold0 {fold_aucs[0]:.6f} vs hill {ctrl_folds[0]:.6f} "
                f"(Δ={deltas[0]:+.6f}); fold1 {fold_aucs[1]:.6f} vs hill {ctrl_folds[1]:.6f} "
                f"(Δ={deltas[1]:+.6f})",
                flush=True,
            )
        score = max(ap, mean_fold)
        if score > best[0] + 1e-12 or (abs(score - best[0]) < 1e-12 and ap > best[0]):
            # prefer higher aligned pooled; use aligned vector for handoff
            best = (ap, name, aligned, te, rec)

    # Guard: never hand off a loser vs nested_hill aligned
    if best[1] != "nested_hill" and best[0] + 1e-12 < ctrl_aligned_auc:
        print(
            f"GUARD: best arm aligned {best[0]:.6f} < nested_hill aligned {ctrl_aligned_auc:.6f} "
            f"— falling back to nested_hill",
            flush=True,
        )
        best = (ctrl_aligned_auc, "nested_hill", ctrl_aligned, ctrl_te, {"source": "guard_fallback"})

    bname, boof, bte = best[1], best[2], best[3]
    # Platt-calibrate for base_margin
    L_oof, L_te, p_oof, p_te, mean_p = platt_calibrate_logits(boof, bte, y, folds)
    print(
        f"HANDOFF source={bname} aligned_auc={best[0]:.6f} "
        f"platt_mean_p={mean_p:.5f} (target~{BASE_RATE}) "
        f"logit_mean={float(L_oof.mean()):.3f} (target~{logit(BASE_RATE):.3f})",
        flush=True,
    )

    os.makedirs(PRED, exist_ok=True)
    np.save(os.path.join(PRED, "campaign_best_oof.npy"), boof.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_test.npy"), bte.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_oof_p.npy"), p_oof.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_test_p.npy"), p_te.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_oof_logit.npy"), L_oof.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_test_logit.npy"), L_te.astype(np.float32))
    # also mirror as best_7_1_* if from 7.1 era
    meta = {
        "stage": "7_align_handoff",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "nested_hill": {"oof": ctrl_oof, "folds": ctrl_folds, "aligned_probit_oof": ctrl_aligned_auc},
        "candidates": rows,
        "handoff": {
            "source": bname,
            "aligned_pooled_auc": best[0],
            "platt_mean_p": mean_p,
            "logit_mean": float(L_oof.mean()),
            "base_rate_target": BASE_RATE,
            "paths": {
                "oof": "preds/stage7/campaign_best_oof.npy",
                "test": "preds/stage7/campaign_best_test.npy",
                "oof_logit": "preds/stage7/campaign_best_oof_logit.npy",
                "test_logit": "preds/stage7/campaign_best_test_logit.npy",
            },
        },
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    path = os.path.join(LOGS, "stage7_align_handoff.json")
    json.dump(meta, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
