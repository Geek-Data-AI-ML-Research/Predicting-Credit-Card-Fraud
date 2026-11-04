#!/usr/bin/env python3
"""READ-ONLY Pass-1 weight + split-leak audit. Does not touch master jobs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
PRED_A = ROOT / "preds" / "phaseA"
SUBS = ROOT / "submissions"
OUT = ROOT / "logs" / "overnight"
SEED = 42
GB5 = "goodpjw_artifacts__gb_bag5_oof_npy"


def probit_rank(x):
    r = rankdata(x, method="average")
    u = np.clip(r / (len(x) + 1.0), 1e-6, 1 - 1e-6)
    return norm.ppf(u)


def within_fold_probit(O, folds):
    Z = np.zeros_like(O, dtype=np.float64)
    for _, va in folds:
        for j in range(O.shape[1]):
            Z[va, j] = probit_rank(O[va, j])
    return Z


def nested_hill_weights(Z, y, folds, max_steps=40):
    weights = []
    oof = np.zeros(len(y))
    M = Z.shape[1]
    for tr, va in folds:
        w = np.zeros(M)
        best_j = int(np.argmax([roc_auc_score(y[tr], Z[tr, j]) for j in range(M)]))
        w[best_j] = 1.0
        cur = roc_auc_score(y[tr], Z[tr] @ (w / w.sum()))
        for _ in range(max_steps):
            best_gain, bj = 0.0, None
            for j in range(M):
                w2 = w.copy()
                w2[j] += 1.0
                a = roc_auc_score(y[tr], Z[tr] @ (w2 / w2.sum()))
                if a > cur + 1e-7 and a - cur > best_gain:
                    best_gain, bj = a - cur, j
            if bj is None:
                break
            w[bj] += 1.0
            cur = roc_auc_score(y[tr], Z[tr] @ (w / w.sum()))
        wn = w / max(w.sum(), 1e-12)
        weights.append(wn)
        oof[va] = Z[va] @ wn
    return np.mean(np.stack(weights), axis=0), oof


def main():
    bank = np.load(PRED_A / "bank_aligned.npz", allow_pickle=True)
    names = [str(n) for n in bank["names"]]
    O = bank["O"].astype(np.float64)
    T = bank["T"].astype(np.float64)
    y = bank["y"].astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    assert GB5 in names, names[:5]
    j5 = names.index(GB5)
    o5, t5 = O[:, j5], T[:, j5]
    print(f"[audit] n_members={len(names)} gb_bag5 oof={roc_auc_score(y, o5):.6f}", flush=True)

    Z = within_fold_probit(O, folds)

    # --- 1) nested logit C=0.1 coefs ---
    coefs = []
    for tr, va in folds:
        m = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs", tol=1e-8)
        m.fit(Z[tr], y[tr])
        coefs.append(m.coef_.ravel())
    w = np.mean(np.stack(coefs), axis=0)
    order = np.argsort(-w)
    print("\n=== TOP 15 POSITIVE coef (nested_logit C=0.1) ===", flush=True)
    for r, j in enumerate(order[:15], 1):
        print(f"{r:2d}. {names[j]:60s}  w={w[j]:+.6f}  oof={roc_auc_score(y, O[:, j]):.6f}", flush=True)
    print("\n=== TOP 15 NEGATIVE coef (nested_logit C=0.1) ===", flush=True)
    for r, j in enumerate(order[::-1][:15], 1):
        print(f"{r:2d}. {names[j]:60s}  w={w[j]:+.6f}  oof={roc_auc_score(y, O[:, j]):.6f}", flush=True)

    # also C=0.002 for delta
    coefs2 = []
    for tr, va in folds:
        m = LogisticRegression(C=0.002, max_iter=3000, solver="lbfgs", tol=1e-8)
        m.fit(Z[tr], y[tr])
        coefs2.append(m.coef_.ravel())
    w2 = np.mean(np.stack(coefs2), axis=0)
    dw = w - w2
    print("\n=== TOP 15 |Δw| C=0.1 − C=0.002 (who gained weight) ===", flush=True)
    for r, j in enumerate(np.argsort(-np.abs(dw))[:15], 1):
        print(
            f"{r:2d}. {names[j]:60s}  Δw={dw[j]:+.6f}  w0.1={w[j]:+.6f}  w0.002={w2[j]:+.6f}  oof={roc_auc_score(y, O[:, j]):.6f}",
            flush=True,
        )

    wh, hill_oof = nested_hill_weights(Z, y, folds)
    print(f"\n=== nested_hill mean weights OOF={roc_auc_score(y, hill_oof):.6f} ===", flush=True)
    print("--- TOP 15 POSITIVE hill ---", flush=True)
    for r, j in enumerate(np.argsort(-wh)[:15], 1):
        print(f"{r:2d}. {names[j]:60s}  w={wh[j]:+.6f}  oof={roc_auc_score(y, O[:, j]):.6f}", flush=True)
    print("--- TOP 15 (smallest / most zero) hill ---", flush=True)
    for r, j in enumerate(np.argsort(wh)[:15], 1):
        print(f"{r:2d}. {names[j]:60s}  w={wh[j]:+.6f}  oof={roc_auc_score(y, O[:, j]):.6f}", flush=True)

    # --- 2) OOF vs Test Spearman consistency vs gb_bag5 ---
    print("\n=== LEAK SCREEN: |rho_oof - rho_test| vs gb_bag5 (flag > 0.005) ===", flush=True)
    flags = []
    rows = []
    for j, nm in enumerate(names):
        if not np.isfinite(T[:, j]).all():
            rows.append({"name": nm, "skip": "nonfinite_test"})
            continue
        rho_o = float(spearmanr(O[:, j], o5).correlation)
        rho_t = float(spearmanr(T[:, j], t5).correlation)
        d = abs(rho_o - rho_t)
        auc_j = float(roc_auc_score(y, O[:, j]))
        # subtract test
        r5 = rankdata(o5)
        rj = rankdata(O[:, j])
        a_sub = float(roc_auc_score(y, r5 - 0.2 * rj))
        a_add = float(roc_auc_score(y, r5 + 0.2 * rj))
        base = float(roc_auc_score(y, o5))
        jump_sub = a_sub - base
        jump_add = a_add - base
        leak = d >= 0.005
        weird = (auc_j < 0.9505) and (max(jump_sub, jump_add) > 0.00015)
        if leak or weird:
            flags.append(nm)
        rows.append({
            "name": nm, "oof": auc_j, "rho_oof": rho_o, "rho_test": rho_t, "abs_delta_rho": d,
            "jump_sub0.2": jump_sub, "jump_add0.2": jump_add, "flag_rho": leak, "flag_jump": weird,
        })
        if leak or weird:
            print(
                f"FLAG {nm}: oof={auc_j:.6f} rho_o={rho_o:.4f} rho_t={rho_t:.4f} |Δ|={d:.4f} "
                f"jump_sub={jump_sub:+.6f} jump_add={jump_add:+.6f}",
                flush=True,
            )
    print(f"[audit] flagged {len(flags)}/{len(names)} members", flush=True)
    # top deltas even if under threshold
    rows_ok = [r for r in rows if "abs_delta_rho" in r]
    rows_ok.sort(key=lambda r: -r["abs_delta_rho"])
    print("\n=== TOP 15 |Δrho| (even if <0.005) ===", flush=True)
    for r in rows_ok[:15]:
        print(
            f"{r['name'][:60]:60s}  |Δρ|={r['abs_delta_rho']:.5f}  ρo={r['rho_oof']:.4f} ρt={r['rho_test']:.4f}  oof={r['oof']:.6f}",
            flush=True,
        )

    # --- 3) Spearman vs Part6 submission ---
    print("\n=== SPEARMAN pass1 subs vs Part6 / anchors ===", flush=True)
    subA = pd.read_csv(SUBS / "sub_A_pass1_bank63_nested.csv")["is_fraud"].values
    subB = pd.read_csv(SUBS / "sub_B_pass1_refit_top.csv")["is_fraud"].values
    cands = [
        ("goodpjw_part6_submission", [
            ROOT / "external_bank" / "goodpjw_artifacts" / "submission.csv",
            ROOT / "external_bank" / "goodpjw_p6" / "submission.csv",
            ROOT / "submissions" / "submission_part6.csv",
            ROOT / "external_bank" / "goodpjw_artifacts" / "submission_part6.csv",
        ]),
        ("gb_bag5_full", [ROOT / "external_bank" / "goodpjw_artifacts" / "gb_bag5_full.npy"]),
        ("mega_submission", [ROOT / "external_bank" / "p6_sources" / "mega" / "submission.csv"]),
        ("midblend_95165", [
            ROOT / "submissions" / "sub_lb_hill_mega_midblend.csv",
            ROOT / "submissions" / "sub_lb_hill_mega_midblend_95165.csv",
        ]),
    ]
    for label, paths in cands:
        x = None
        path = None
        for p in paths:
            if not p.exists():
                continue
            path = p
            if p.suffix == ".npy":
                x = np.load(p).astype(float).ravel()
            else:
                df = pd.read_csv(p)
                col = [c for c in df.columns if c != "id"][0]
                x = df[col].values.astype(float)
            break
        if x is None:
            print(f"{label}: MISSING", flush=True)
            continue
        if len(x) != len(subA):
            print(f"{label}: bad len {len(x)} from {path}", flush=True)
            continue
        print(f"{label} <- {path}", flush=True)
        print(f"  vs sub_A_pass1: {spearmanr(subA, x).correlation:.6f}", flush=True)
        print(f"  vs sub_B_pass1: {spearmanr(subB, x).correlation:.6f}", flush=True)
    print(f"sub_A vs sub_B: {spearmanr(subA, subB).correlation:.6f}", flush=True)

    rep = {
        "n_flagged": len(flags),
        "flagged": flags,
        "top_pos_logit": [{"name": names[j], "w": float(w[j])} for j in order[:15]],
        "top_neg_logit": [{"name": names[j], "w": float(w[j])} for j in order[::-1][:15]],
        "top_delta_C": [{"name": names[j], "dw": float(dw[j]), "w01": float(w[j]), "w002": float(w2[j])} for j in np.argsort(-np.abs(dw))[:15]],
        "top_hill": [{"name": names[j], "w": float(wh[j])} for j in np.argsort(-wh)[:15]],
        "rho_rows": rows_ok[:30],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_pass1_leak.json").write_text(json.dumps(rep, indent=2))
    print("\n[audit] wrote logs/overnight/audit_pass1_leak.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
