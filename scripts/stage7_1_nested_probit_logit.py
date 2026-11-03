#!/usr/bin/env python3
"""Stage 7.1: Leak-free nested Probit-Rank + Logit meta-ensemble (3 solvers).

HARD POLICY: never kaggle submit — write candidate CSV + JSON logs only.
Promotion gate Stage 2.2 vs controls hillclimb_mix (0.95111) and mega_logit (0.95115).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from oof_bank_loader import collect_oof_bank, load_y, locked_folds  # noqa: E402
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402

warnings.filterwarnings("ignore")

LOGS = os.path.join(ROOT, "logs")
SUBS = os.path.join(ROOT, "submissions")
EPS = 1e-6
CORR_THRESH = 0.995
CTRL_HILL = 0.95111
CTRL_MEGA = 0.95115
L2_GRID = [1e-4, 5e-4, 1e-3, 5e-3]


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def probit_rank(p: np.ndarray) -> np.ndarray:
    # ndtri((rank+0.5)/N) — Gaussian tail spacing
    r = rankdata(p, method="average")
    n = len(p)
    u = (r - 0.5) / n
    u = np.clip(u, EPS, 1 - EPS)
    return ndtri(u)


def std_logit_fit(p_tr: np.ndarray, p_ap: np.ndarray) -> np.ndarray:
    z = to_logit(p_tr)
    mu, sd = float(z.mean()), float(z.std()) + 1e-12
    return (to_logit(p_ap) - mu) / sd


def build_bases(O: np.ndarray, T: np.ndarray, tr_idx: np.ndarray, ap_oof_idx: np.ndarray):
    """Fit transforms on tr_idx; apply to ap_oof_idx (for O) and full T."""
    n_m = O.shape[1]
    Zo = np.zeros((len(ap_oof_idx), n_m * 2), np.float64)
    Zt = np.zeros((T.shape[0], n_m * 2), np.float64)
    for j in range(n_m):
        # logit std
        Zo[:, j] = std_logit_fit(O[tr_idx, j], O[ap_oof_idx, j])
        Zt[:, j] = std_logit_fit(O[tr_idx, j], T[:, j])
        # probit-rank: fit ranks independently on each slice (no leakage of va into rank of tr)
        # For apply on va: rank within va alone is OK for scoring; for consistency use global
        # rank within the apply set (standard nested OOF practice for rank transforms).
        Zo[:, n_m + j] = probit_rank(O[ap_oof_idx, j])
        Zt[:, n_m + j] = probit_rank(T[:, j])
    return Zo, Zt


def _strat_subsample(y: np.ndarray, n_max: int = 80000, seed: int = 42) -> np.ndarray:
    if len(y) <= n_max:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = min(len(pos), max(2000, int(n_max * y.mean())))
    n_neg = min(len(neg), n_max - n_pos)
    return np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])


def nn_ridge_weights(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Non-negative weights, sum=1, L2 concentration; SLSQP-AUC on stratified subsample."""
    n = X.shape[1]
    w0 = np.ones(n) / n
    idx = _strat_subsample(y, 80000)
    Xs, ys = X[idx], y[idx]

    def neg_auc(w):
        w = np.asarray(w)
        return -roc_auc_score(ys, Xs @ w) + l2 * float(np.dot(w, w))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        neg_auc,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 80, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def coord_ascent_refine(X: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 25) -> np.ndarray:
    """Fast coordinate ascent on discrete mass (hillclimb-style) strictly on train."""
    w = w.astype(np.float64).copy()
    # convert to integer counts for hillclimb stability
    mass = np.maximum(w * 64.0, 0)
    if mass.sum() <= 0:
        mass[int(np.argmax([roc_auc_score(y, X[:, i]) for i in range(X.shape[1])]))] = 1
    cur = roc_auc_score(y, X @ (mass / mass.sum()))
    for _ in range(iters):
        best = cur
        bi = None
        for i in range(X.shape[1]):
            mass2 = mass.copy()
            mass2[i] += 1
            sc = roc_auc_score(y, X @ (mass2 / mass2.sum()))
            if sc > best + 1e-8:
                best, bi = sc, i
        if bi is None:
            break
        mass[bi] += 1
        cur = best
    return mass / mass.sum()


def prune_collinear(O: np.ndarray, names: list[str], y: np.ndarray) -> list[int]:
    Z = to_logit(O)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    corr = np.corrcoef(Z.T)
    order = np.argsort([-roc_auc_score(y, O[:, j]) for j in range(O.shape[1])])
    keep = []
    for j in order:
        if any(corr[j, k] > CORR_THRESH for k in keep):
            continue
        keep.append(int(j))
    return keep


def nested_hillclimb_topk(X: np.ndarray, y: np.ndarray, max_k: int = 12) -> np.ndarray:
    """Greedy add columns maximizing AUC on train; return simplex weights."""
    n = X.shape[1]
    chosen = []
    remaining = list(range(n))
    cur = None
    best_auc = -1.0
    while remaining and len(chosen) < max_k:
        best_i, best_a, best_pred = None, best_auc, None
        for i in remaining:
            if cur is None:
                pred = X[:, i]
                sc = roc_auc_score(y, pred)
            else:
                sc, pred = best_auc, cur
                for a in (0.25, 0.5, 0.75, 1.0):
                    cand = (1 - a) * cur + a * X[:, i]
                    a_sc = roc_auc_score(y, cand)
                    if a_sc > sc + 1e-8:
                        sc, pred = a_sc, cand
            if sc > best_a + 1e-8:
                best_a, best_i, best_pred = sc, i, pred
        if best_i is None:
            break
        chosen.append(best_i)
        remaining.remove(best_i)
        cur = best_pred
        best_auc = best_a
    w = np.zeros(n)
    if not chosen:
        w[int(np.argmax([roc_auc_score(y, X[:, i]) for i in range(n)]))] = 1
        return w
    w[chosen] = 1.0 / len(chosen)
    return w


def fit_temperatures(logits_tr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-model temperature T_j on train logits (scalar grid search)."""
    n_m = logits_tr.shape[1]
    Ts = np.ones(n_m)
    grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for j in range(n_m):
        best_t, best_a = 1.0, -1.0
        z = logits_tr[:, j]
        for t in grid:
            a = roc_auc_score(y, expit(z / t))
            if a > best_a:
                best_a, best_t = a, t
        Ts[j] = best_t
    return Ts


def blend_nn(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    w = nn_ridge_weights(X, y, l2)
    w = coord_ascent_refine(X, y, w)
    return w


def main():
    t0 = time.time()
    y, ntr, nte = load_y(ROOT)
    folds = locked_folds(y)
    sub = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
    assert len(sub) == nte == 299984

    bank = collect_oof_bank(ROOT, ntr, nte, require_test=True, y=y)
    # force-include self models if test materialized
    for alias, opath, tpath in [
        (
            "self:te_xgb_smooth50",
            os.path.join(ROOT, "preds/stage4/te_xgb_smooth50_oof.npy"),
            os.path.join(ROOT, "preds/stage4/te_xgb_smooth50_full.npy"),
        ),
        (
            "self:te_lgbm_leaves3",
            os.path.join(ROOT, "preds/stage5/te_lgbm_leaves3_oof.npy"),
            os.path.join(ROOT, "preds/stage5/te_lgbm_leaves3_full.npy"),
        ),
    ]:
        if os.path.exists(opath) and os.path.exists(tpath):
            bank[alias] = (np.load(opath).astype(np.float64), np.load(tpath).astype(np.float64))
            print(f"included {alias}", flush=True)
        else:
            print(f"WARNING: {alias} missing test preds at {tpath}", flush=True)

    # drop near-dup names preferring self aliases / higher auc
    names = sorted(bank, key=lambda n: -roc_auc_score(y, bank[n][0]))
    # filter auc floor
    names = [n for n in names if roc_auc_score(y, bank[n][0]) >= 0.94]
    O = np.column_stack([bank[n][0] for n in names])
    T = np.column_stack([bank[n][1] for n in names])
    print(f"bank stackable={len(names)}", flush=True)
    for n in names[:15]:
        print(f"  {roc_auc_score(y, bank[n][0]):.6f}  {n}", flush=True)

    keep_idx = prune_collinear(O, names, y)
    names_p = [names[i] for i in keep_idx]
    Op, Tp = O[:, keep_idx], T[:, keep_idx]
    print(f"after corr prune>{CORR_THRESH}: {len(names_p)}", flush=True)

    # ----- controls: nested hillclimb_mix + mega_logit on same bank (raw rank / logit) -----
    def hill_climb(Or, yy, iters=40):
        w = np.zeros(Or.shape[1])
        w[int(np.argmax([roc_auc_score(yy, Or[:, i]) for i in range(Or.shape[1])]))] = 1
        cur = roc_auc_score(yy, Or @ w)
        for _ in range(iters):
            sc, i = max(
                (roc_auc_score(yy, Or @ (w + np.eye(Or.shape[1])[i])), i)
                for i in range(Or.shape[1])
            )
            if sc <= cur + 1e-7:
                break
            w[i] += 1
            cur = sc
        return w / w.sum()

    ctrl_hill_oof = np.zeros(ntr)
    ctrl_mega_oof = np.zeros(ntr)
    Orank = np.column_stack([rankdata(O[:, j]) / ntr for j in range(O.shape[1])])
    Ologit = to_logit(O)
    for tr, va in folds:
        wh = hill_climb(Orank[tr], y[tr])
        ctrl_hill_oof[va] = Orank[va] @ wh
        # mix: hill + logit LR (same as stack_hillclimb mix recipe)
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(Ologit[tr], y[tr])
        hill_va = Orank[va] @ wh
        stack_va = lr.decision_function(Ologit[va])
        ctrl_hill_oof[va] = rankdata(hill_va) / len(va) + rankdata(stack_va) / len(va)
        ctrl_mega_oof[va] = (
            LogisticRegression(C=0.01, max_iter=3000)
            .fit(Ologit[tr], y[tr])
            .decision_function(Ologit[va])
        )

    ctrl_hill_auc = float(roc_auc_score(y, ctrl_hill_oof))
    ctrl_mega_auc = float(roc_auc_score(y, ctrl_mega_oof))
    ctrl_hill_folds = [float(roc_auc_score(y[va], ctrl_hill_oof[va])) for _, va in folds]
    ctrl_mega_folds = [float(roc_auc_score(y[va], ctrl_mega_oof[va])) for _, va in folds]
    print(
        f"controls nested: hillclimb_mix={ctrl_hill_auc:.6f} mega_logit={ctrl_mega_auc:.6f}",
        flush=True,
    )

    solvers = {
        "nn_ridge_probit_logit": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
        "pruned_topk_hillclimb": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
        "temp_scaled_nn_blend": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
    }

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        # bases on pruned bank
        Zo_tr, _ = build_bases(Op, Tp, tr, tr)
        Zo_va, Zt = build_bases(Op, Tp, tr, va)

        # Solver 1: NN ridge + coord ascent — pick best l2 on tr only
        best_l2, best_w, best_a = L2_GRID[0], None, -1.0
        for l2 in L2_GRID:
            w = blend_nn(Zo_tr, y[tr], l2)
            a = roc_auc_score(y[tr], Zo_tr @ w)
            if a > best_a:
                best_a, best_l2, best_w = a, l2, w
        pred_va = Zo_va @ best_w
        pred_te = Zt @ best_w
        solvers["nn_ridge_probit_logit"]["oof"][va] = pred_va
        solvers["nn_ridge_probit_logit"]["test"] += pred_te
        fa = float(roc_auc_score(y[va], pred_va))
        solvers["nn_ridge_probit_logit"]["folds"].append(fa)
        solvers["nn_ridge_probit_logit"]["meta"].append({"l2": best_l2, "tr_auc": best_a, "va_auc": fa})
        print(f"  nn_ridge l2={best_l2} va={fa:.6f}", flush=True)

        # Solver 2: pruned top-k hillclimb on probit+logit
        w2 = nested_hillclimb_topk(Zo_tr, y[tr], max_k=min(12, Zo_tr.shape[1]))
        pred_va = Zo_va @ w2
        pred_te = Zt @ w2
        solvers["pruned_topk_hillclimb"]["oof"][va] = pred_va
        solvers["pruned_topk_hillclimb"]["test"] += pred_te
        fa = float(roc_auc_score(y[va], pred_va))
        solvers["pruned_topk_hillclimb"]["folds"].append(fa)
        solvers["pruned_topk_hillclimb"]["meta"].append({"nnz": int((w2 > 0).sum()), "va_auc": fa})
        print(f"  topk_hill va={fa:.6f} nnz={int((w2>0).sum())}", flush=True)

        # Solver 3: temperature scaling on logits then NN blend of [scaled_logit | probit]
        n_m = Op.shape[1]
        logits_tr = to_logit(Op[tr])
        Ts = fit_temperatures(logits_tr, y[tr])
        # rebuild feature matrix: scaled logit std + probit
        def temp_basis(idx):
            Z = np.zeros((len(idx), n_m * 2))
            for j in range(n_m):
                z = to_logit(Op[idx, j]) / Ts[j]
                # standardize using train stats of scaled logits
                ztr = to_logit(Op[tr, j]) / Ts[j]
                mu, sd = float(ztr.mean()), float(ztr.std()) + 1e-12
                Z[:, j] = (z - mu) / sd
                Z[:, n_m + j] = probit_rank(Op[idx, j])
            return Z

        def temp_basis_test():
            Z = np.zeros((nte, n_m * 2))
            for j in range(n_m):
                z = to_logit(Tp[:, j]) / Ts[j]
                ztr = to_logit(Op[tr, j]) / Ts[j]
                mu, sd = float(ztr.mean()), float(ztr.std()) + 1e-12
                Z[:, j] = (z - mu) / sd
                Z[:, n_m + j] = probit_rank(Tp[:, j])
            return Z

        Xt = temp_basis(tr)
        Xv = temp_basis(va)
        Xte = temp_basis_test()
        w3 = blend_nn(Xt, y[tr], 1e-3)
        pred_va = Xv @ w3
        pred_te = Xte @ w3
        solvers["temp_scaled_nn_blend"]["oof"][va] = pred_va
        solvers["temp_scaled_nn_blend"]["test"] += pred_te
        fa = float(roc_auc_score(y[va], pred_va))
        solvers["temp_scaled_nn_blend"]["folds"].append(fa)
        solvers["temp_scaled_nn_blend"]["meta"].append(
            {"T_mean": float(Ts.mean()), "va_auc": fa}
        )
        print(f"  temp_nn va={fa:.6f}", flush=True)

    # average test over folds
    for s in solvers.values():
        s["test"] /= 5.0
        s["oof_auc"] = float(roc_auc_score(y, s["oof"]))

    results = {}
    promoted = []
    for name, s in solvers.items():
        gate_hill = promotion_gate(ctrl_hill_folds, s["folds"], ctrl_hill_auc, s["oof_auc"])
        gate_mega = promotion_gate(ctrl_mega_folds, s["folds"], ctrl_mega_auc, s["oof_auc"])
        # user rule: OOF > 0.95115 and >=4/5 folds up (vs mega as primary)
        pass_abs = s["oof_auc"] > CTRL_MEGA and gate_mega["fold_wins"] >= 4
        rec = {
            "oof_auc": s["oof_auc"],
            "fold_aucs": s["folds"],
            "meta": s["meta"],
            "gate_vs_hillclimb_mix": gate_hill,
            "gate_vs_mega_logit": gate_mega,
            "pass_abs_vs_0.95115": pass_abs,
        }
        results[name] = rec
        print(
            f"\n{name}: OOF={s['oof_auc']:.6f} "
            f"gate_mega={gate_mega['reason']} abs_pass={pass_abs}",
            flush=True,
        )
        if pass_abs or gate_mega["promote"] or gate_hill["promote"]:
            promoted.append(name)

    # write best candidate if any passes abs gate OR any promote
    candidates_written = []
    hill_ref = None
    href = os.path.join(SUBS, "hillclimb_mix_0.95111.csv")
    if os.path.exists(href):
        hill_ref = pd.read_csv(href)["is_fraud"].values

    os.makedirs(SUBS, exist_ok=True)
    for name in solvers:
        s = solvers[name]
        if not (results[name]["pass_abs_vs_0.95115"] or results[name]["gate_vs_mega_logit"]["promote"]):
            # still write if best overall and OOF > controls for inspection? only if gate pass
            continue
        pred = s["test"]
        assert len(pred) == 299984
        assert np.isfinite(pred).all()
        out = sub.copy()
        # store rank-normalized probs for submission stability
        out["is_fraud"] = rankdata(pred) / nte
        path = os.path.join(SUBS, f"stage7_1_{name}_{s['oof_auc']:.5f}.csv")
        out.to_csv(path, index=False)
        spear = None
        if hill_ref is not None:
            spear = float(spearmanr(out["is_fraud"].values, hill_ref).correlation)
        # verify
        ver = {
            "path": path,
            "n_rows": int(len(out)),
            "n_nan": int(out["is_fraud"].isna().sum()),
            "n_inf": int(np.isinf(out["is_fraud"].values).sum()),
            "id_aligned": bool(np.array_equal(out["id"].values, sub["id"].values)),
            "spearman_vs_hillclimb_mix": spear,
        }
        results[name]["submission"] = ver
        candidates_written.append(ver)
        print(f"WROTE {path} spearman_vs_hill={spearman}", flush=True)

    # always save best solver OOF/test for later even if not promoted
    best_name = max(solvers, key=lambda n: solvers[n]["oof_auc"])
    best = solvers[best_name]
    os.makedirs(os.path.join(ROOT, "preds", "stage7"), exist_ok=True)
    np.save(os.path.join(ROOT, "preds", "stage7", "best_oof.npy"), best["oof"].astype(np.float32))
    np.save(os.path.join(ROOT, "preds", "stage7", "best_test.npy"), best["test"].astype(np.float32))

    # public estimate: prior hillclimb public-cv gap ≈ +0.00052
    gap = 0.95163 - 0.95111
    est_public = {
        n: {
            "cv": solvers[n]["oof_auc"],
            "est_public": solvers[n]["oof_auc"] + gap,
            "method": "cv + (0.95163-0.95111)",
        }
        for n in solvers
    }

    doc = {
        "stage": "7.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_bank": len(names),
        "n_pruned": len(names_p),
        "members_pruned": names_p,
        "self_included": [n for n in names if n.startswith("self:")],
        "controls": {
            "hillclimb_mix": {
                "target_cv": CTRL_HILL,
                "nested_cv": ctrl_hill_auc,
                "fold_aucs": ctrl_hill_folds,
            },
            "mega_logit": {
                "target_cv": CTRL_MEGA,
                "nested_cv": ctrl_mega_auc,
                "fold_aucs": ctrl_mega_folds,
            },
        },
        "solvers": results,
        "best_solver": best_name,
        "promoted": promoted,
        "candidates_written": candidates_written,
        "est_public": est_public,
        "policy": "NO_KAGGLE_SUBMIT",
    }
    os.makedirs(LOGS, exist_ok=True)
    out_path = os.path.join(LOGS, "stage7_1_results.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    print(f"best={best_name} OOF={best['oof_auc']:.6f} est_pub={best['oof_auc']+gap:.6f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
