#!/usr/bin/env python3
"""Stage 10.3: Inject Stage 9.2 + Stage 10 OOFs into bounded power-rank hillclimb.

Pre-Submit Audit vs nested_hill 0.951140 and hillclimb_mix_0.95111.csv.
NO kaggle submit.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from stage9_1_full73_power_hillclimb import (  # noqa: E402
    build_bases_nested,
    classic_nested_hill_ranks,
    gpu_bagged_hillclimb,
    to_logit_clip,
    torch_auc,
)
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED7 = os.path.join(ROOT, "preds", "stage7")
PRED9 = os.path.join(ROOT, "preds", "stage9")
PRED10 = os.path.join(ROOT, "preds", "stage10")
SUBS = os.path.join(ROOT, "submissions")
SEED = 42
CTRL_HILL = 0.951140
CTRL_DIGLEAD = 0.95063
EPS = 5e-5


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def collect_extra(ntr, nte, y):
    """Stage 9.2 + Stage 10 OOF/test pairs."""
    extras = {}
    patterns = [
        (PRED9, ["ortho_logit_offset", "ortho_xgb_residual"]),
        (PRED10, None),  # all *_oof.npy in stage10
    ]
    for pred_dir, names in patterns:
        if names is None:
            for op in glob.glob(os.path.join(pred_dir, "*_oof.npy")):
                name = os.path.basename(op).replace("_oof.npy", "")
                tp = os.path.join(pred_dir, f"{name}_test.npy")
                if not os.path.exists(tp):
                    continue
                o, t = np.load(op).astype(np.float64), np.load(tp).astype(np.float64)
                if len(o) != ntr or len(t) != nte:
                    continue
                extras[f"s10:{name}" if "stage10" in pred_dir else f"s9:{name}"] = (o, t)
        else:
            for name in names:
                op = os.path.join(pred_dir, f"{name}_oof.npy")
                tp = os.path.join(pred_dir, f"{name}_test.npy")
                if os.path.exists(op) and os.path.exists(tp):
                    extras[f"s9:{name}"] = (
                        np.load(op).astype(np.float64),
                        np.load(tp).astype(np.float64),
                    )
    return extras


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    y = train[TARGET].astype(int).values
    ntr = len(y)
    nte = len(pd.read_csv(os.path.join(DATA, "sample_submission.csv")))
    folds = locked_folds(y)

    pre = os.path.join(PRED9, "full_bank_unpruned.npz")
    z = np.load(pre, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    O, T = z["O"].astype(np.float64), z["T"].astype(np.float64)
    print(f"base bank m={len(names)}", flush=True)

    extras = collect_extra(ntr, nte, y)
    print(f"extras to inject: {list(extras.keys())}", flush=True)

    # Gate Stage 10 bases vs diglead
    dig_idx = next((i for i, n in enumerate(names) if "lgb_diglead_l3_bag" in n), None)
    dig_oof = O[:, dig_idx] if dig_idx is not None else None
    dig_auc = float(roc_auc_score(y, dig_oof)) if dig_oof is not None else CTRL_DIGLEAD
    dig_folds = (
        [float(roc_auc_score(y[va], dig_oof[va])) for _, va in folds]
        if dig_oof is not None
        else [dig_auc] * 5
    )
    gate_report = {}
    for name, (o, t) in extras.items():
        auc = float(roc_auc_score(y, o))
        folds_a = [float(roc_auc_score(y[va], o[va])) for _, va in folds]
        gate = promotion_gate(dig_folds, folds_a, dig_auc, auc)
        gate_report[name] = {
            "oof_auc": auc,
            "delta_vs_diglead": auc - dig_auc,
            "fold_wins": gate["fold_wins"],
            "gate": gate,
            "inject": True,
        }
        print(
            f"  gate {name}: OOF={auc:.6f} Δ={auc-dig_auc:+.6f} wins={gate['fold_wins']}/5 {gate['reason']}",
            flush=True,
        )
        names.append(name)
        O = np.column_stack([O, o])
        T = np.column_stack([T, t])

    m = O.shape[1]
    os.makedirs(PRED10, exist_ok=True)
    np.savez_compressed(
        os.path.join(PRED10, "bank_plus_stage10.npz"),
        names=np.array(names, dtype=object),
        O=O,
        T=T,
        y=y,
    )
    top15 = [(names[j], float(roc_auc_score(y, O[:, j]))) for j in range(min(15, m))]
    print("=== Top 15 after inject ===", flush=True)
    for i, (n, a) in enumerate(sorted(top15, key=lambda x: -x[1]), 1):
        print(f"  {i:2d}. {a:.6f}  {n}", flush=True)

    # nested_hill control on expanded bank
    print("nested_hill on expanded bank...", flush=True)
    ctrl = classic_nested_hill_ranks(O, y, folds, device=device)
    ctrl_auc = float(roc_auc_score(y, ctrl))
    ctrl_folds = [float(roc_auc_score(y[va], ctrl[va])) for _, va in folds]
    print(f"nested_hill_expanded OOF={ctrl_auc:.6f}", flush=True)

    # ref nested_hill 0.95114
    cref = os.path.join(PRED7, "ctrl_nested_hill_oof.npy")
    ref_auc = CTRL_HILL
    ref_oof = None
    if os.path.exists(cref):
        ref_oof = np.load(cref)
        ref_auc = float(roc_auc_score(y, ref_oof))
    ref_folds = (
        [float(roc_auc_score(y[va], ref_oof[va])) for _, va in folds]
        if ref_oof is not None
        else [ref_auc] * 5
    )

    oof = np.zeros(ntr)
    te_sum = np.zeros(nte)
    fold_aucs = []
    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        Ztr, Zva, Zte = build_bases_nested(O, T, tr, va)
        w = gpu_bagged_hillclimb(Ztr, y[tr], device, iters=60, n_bags=5)
        pred_va = Zva @ w
        pred_te = Zte @ w
        oof[va] = (rankdata(pred_va, method="average") - 0.5) / len(va)
        te_sum += (rankdata(pred_te, method="average") - 0.5) / len(pred_te)
        fa = float(roc_auc_score(y[va], pred_va))
        fold_aucs.append(fa)
        print(f"  power_hill va={fa:.6f}", flush=True)
        torch.cuda.empty_cache()

    te = te_sum / 5.0
    oof_auc = float(roc_auc_score(y, oof))
    gate_exp = promotion_gate(ctrl_folds, fold_aucs, ctrl_auc, oof_auc)
    gate_ref = promotion_gate(ref_folds, fold_aucs, ref_auc, oof_auc)
    print(
        f"power_hill OOF={oof_auc:.6f} vs expanded_hill={ctrl_auc:.6f} vs nested_hill={ref_auc:.6f}",
        flush=True,
    )
    print(f"  gate_vs_expanded: {gate_exp['reason']} wins={gate_exp['fold_wins']}/5", flush=True)
    print(f"  gate_vs_nested_hill: {gate_ref['reason']} wins={gate_ref['fold_wins']}/5", flush=True)

    np.save(os.path.join(PRED10, "stage10_power_hill_oof.npy"), oof.astype(np.float32))
    np.save(os.path.join(PRED10, "stage10_power_hill_test.npy"), te.astype(np.float32))
    np.save(os.path.join(PRED10, "nested_hill_expanded_oof.npy"), ctrl.astype(np.float32))

    # full-data mix submission (same recipe as 9.1)
    Orank = np.column_stack([rankdata(O[:, j]) / ntr for j in range(m)]).astype(np.float64)
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    sub_idx = np.concatenate(
        [
            rng.choice(pos, min(len(pos), 4000), replace=False),
            rng.choice(neg, min(len(neg), 116_000), replace=False),
        ]
    )
    xt = torch.tensor(Orank[sub_idx], device=device, dtype=torch.float64)
    yt = torch.tensor(y[sub_idx].astype(np.float64), device=device)
    w = torch.zeros(m, device=device, dtype=torch.float64)
    singles = [torch_auc(yt, xt[:, j]) for j in range(m)]
    w[int(np.argmax(singles))] = 1.0
    cur = torch_auc(yt, xt @ w)
    for _ in range(40):
        best, bi = cur, None
        for j in range(m):
            w2 = w.clone()
            w2[j] += 1.0
            sc = torch_auc(yt, xt @ w2)
            if sc > best + 1e-8:
                best, bi = sc, j
        if bi is None:
            break
        w[bi] += 1.0
        cur = best
    ww = (w / w.sum()).cpu().numpy()
    Trank = np.column_stack([rankdata(T[:, j]) / nte for j in range(m)])
    te_hill = Trank @ ww
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(to_logit_clip(O), y)
    te_stack = lr.decision_function(to_logit_clip(T))
    te_mix = rankdata(te_hill) / nte + rankdata(te_stack) / nte

    os.makedirs(SUBS, exist_ok=True)
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    candidates = {}

    def write_sub(tag, scores, auc_tag):
        out = sample.copy()
        out["is_fraud"] = rankdata(scores) / nte
        path = os.path.join(SUBS, f"stage10_{tag}_{auc_tag:.5f}.csv")
        out.to_csv(path, index=False)
        candidates[tag] = path
        return out

    sub_power = write_sub("power_hill", te, oof_auc)
    sub_mix = write_sub("nested_mix_expanded", te_mix, ctrl_auc)

    href = os.path.join(SUBS, "hillclimb_mix_0.95111.csv")
    spear_power = spear_mix = None
    if os.path.exists(href):
        ref = pd.read_csv(href)["is_fraud"].values
        spear_power = float(spearmanr(sub_power["is_fraud"], ref).correlation)
        spear_mix = float(spearmanr(sub_mix["is_fraud"], ref).correlation)
        print(f"Spearman vs hillclimb_mix_0.95111: power={spear_power:.6f} mix={spear_mix:.6f}", flush=True)

    audit = {
        "ctrl_nested_hill": ref_auc,
        "ctrl_diglead_l3_bag": dig_auc,
        "expanded_nested_hill_oof": ctrl_auc,
        "power_hill_oof": oof_auc,
        "power_hill_folds": fold_aucs,
        "fold_wins_vs_nested_hill": gate_ref["fold_wins"],
        "fold_wins_vs_expanded_hill": gate_exp["fold_wins"],
        "spearman_power_vs_0.95111": spear_power,
        "spearman_mix_vs_0.95111": spear_mix,
        "gate_vs_nested_hill": gate_ref,
        "gate_vs_expanded_hill": gate_exp,
        "n_members": m,
        "n_injected": len(extras),
    }
    print("=== PRE-SUBMIT AUDIT ===", flush=True)
    print(json.dumps(audit, indent=2), flush=True)

    doc = {
        "stage": "10.3_reinject_hillclimb",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "base_gates_vs_diglead": gate_report,
        "audit": audit,
        "candidates": candidates,
        "top15": [{"name": n, "oof_auc": a} for n, a in sorted(top15, key=lambda x: -x[1])],
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, "stage10_reinject_hillclimb_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    agg = os.path.join(LOGS, "stage10_results.json")
    prev = json.load(open(agg)) if os.path.exists(agg) else {}
    prev["stage10_3"] = doc
    if os.path.exists(os.path.join(LOGS, "stage10_diglead_superbase_result.json")):
        prev["stage10_superbase"] = json.load(
            open(os.path.join(LOGS, "stage10_diglead_superbase_result.json"))
        )
    prev["updated_utc"] = datetime.now(timezone.utc).isoformat()
    json.dump(prev, open(agg, "w"), indent=2)
    print(f"wrote {path} in {time.time()-t0:.0f}s — POD STAYS WARM", flush=True)


if __name__ == "__main__":
    main()
