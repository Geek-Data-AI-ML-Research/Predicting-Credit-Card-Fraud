#!/usr/bin/env python3
"""Stage 7.1: Fast CUDA Probit-Rank + Logit meta-solver (nested SKF rs=42).

NO kaggle submit. Gate vs honest nested hillclimb control (~0.95118).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
SUBS = os.path.join(ROOT, "submissions")
EPS = 1e-6
SEED = 42
CTRL_NESTED_TARGET = 0.95118
L2_GRID = [1e-4, 5e-4, 1e-3, 5e-3]


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def probit_rank(p: np.ndarray) -> np.ndarray:
    r = rankdata(p, method="average")
    u = np.clip((r - 0.5) / len(p), EPS, 1 - EPS)
    return ndtri(u)


def std_cols(X_tr, X_ap):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
    return (X_ap - mu) / sd, mu, sd


def build_Z(O, T, tr, ap):
    """Standardized logits + probit ranks; fit stats on tr only."""
    m = O.shape[1]
    Ltr = to_logit(O[tr])
    mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
    Zap = np.zeros((len(ap), m * 2), np.float64)
    Zt = np.zeros((T.shape[0], m * 2), np.float64)
    Zap[:, :m] = (to_logit(O[ap]) - mu) / sd
    Zt[:, :m] = (to_logit(T) - mu) / sd
    for j in range(m):
        Zap[:, m + j] = probit_rank(O[ap, j])
        Zt[:, m + j] = probit_rank(T[:, j])
    return Zap.astype(np.float32), Zt.astype(np.float32)


def torch_auc(y: torch.Tensor, s: torch.Tensor) -> float:
    """Wilcoxon AUC via argsort (GPU-friendly)."""
    order = torch.argsort(s)
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = torch.arange(1, len(y_sorted) + 1, device=s.device, dtype=torch.float64)
    sum_pos = ranks[y_sorted.bool()].sum()
    return float(((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())


def pairwise_margin_loss(logits: torch.Tensor, y: torch.Tensor, n_pairs: int = 8192) -> torch.Tensor:
    pos = torch.where(y > 0.5)[0]
    neg = torch.where(y < 0.5)[0]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_zeros(())
    n = min(n_pairs, len(pos) * len(neg), max(len(pos), len(neg)) * 4)
    ip = pos[torch.randint(0, len(pos), (n,), device=logits.device)]
    inn = neg[torch.randint(0, len(neg), (n,), device=logits.device)]
    return F.relu(1.0 - (logits[ip] - logits[inn])).mean()


def fit_nn_weights_cuda(X: np.ndarray, y: np.ndarray, l2: float, device: str, steps: int = 120):
    """AdamW on unconstrained u → simplex w=softmax(u); BCE + pairwise + L2."""
    xt = torch.tensor(X, device=device, dtype=torch.float32)
    yt = torch.tensor(y.astype(np.float32), device=device)
    u = torch.zeros(X.shape[1], device=device, requires_grad=True)
    opt = torch.optim.AdamW([u], lr=0.05, weight_decay=0.0)
    best_w, best_auc = None, -1.0
    for _ in range(steps):
        opt.zero_grad()
        w = torch.softmax(u, dim=0)
        score = xt @ w
        # map score → logits via temperature-free linear; BCE on sigmoid(score) unstable if score wide
        # use score as logit after centering
        logits = score - score.mean()
        bce = F.binary_cross_entropy_with_logits(logits, yt)
        pair = pairwise_margin_loss(logits, yt)
        reg = l2 * (w * w).sum()
        loss = bce + 0.2 * pair + reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            auc = torch_auc(yt.double(), score.double())
            if auc > best_auc:
                best_auc = auc
                best_w = w.detach().cpu().numpy().copy()
    return best_w if best_w is not None else np.ones(X.shape[1]) / X.shape[1]


def gpu_coord_ascent(X: np.ndarray, y: np.ndarray, w0: np.ndarray, device: str, iters: int = 40):
    """Discrete mass hillclimb with GPU AUC on stratified ≤120k rows."""
    if len(y) > 120_000:
        rng = np.random.default_rng(SEED)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_pos = min(len(pos), max(3000, int(120_000 * y.mean())))
        n_neg = min(len(neg), 120_000 - n_pos)
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y
    xt = torch.tensor(Xs, device=device, dtype=torch.float64)
    yt = torch.tensor(ys.astype(np.float64), device=device)
    mass = torch.tensor(np.maximum(w0 * 64.0, 0.0), device=device, dtype=torch.float64)
    if mass.sum() <= 0:
        mass[0] = 1.0
    w = mass / mass.sum()
    cur = torch_auc(yt, xt @ w)
    m = X.shape[1]
    for _ in range(iters):
        best, bi = cur, None
        for i in range(m):
            mass2 = mass.clone()
            mass2[i] += 1.0
            sc = torch_auc(yt, xt @ (mass2 / mass2.sum()))
            if sc > best + 1e-8:
                best, bi = sc, i
        if bi is None:
            break
        mass[bi] += 1.0
        cur = best
    return (mass / mass.sum()).cpu().numpy()


def nested_probit_hillclimb(Xp: np.ndarray, y: np.ndarray, max_k: int = 12, n_max: int = 100_000):
    """Greedy blend in probit-only space on stratified subsample."""
    n = Xp.shape[1]
    if len(y) > n_max:
        rng = np.random.default_rng(SEED)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_pos = min(len(pos), max(2500, int(n_max * y.mean())))
        n_neg = min(len(neg), n_max - n_pos)
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        Xs, ys = Xp[idx], y[idx]
    else:
        Xs, ys = Xp, y
    chosen, remaining = [], list(range(n))
    cur, best_auc = None, -1.0
    while remaining and len(chosen) < max_k:
        best_i, best_a, best_pred = None, best_auc, None
        for i in remaining:
            if cur is None:
                pred = Xs[:, i]
                sc = roc_auc_score(ys, pred)
            else:
                sc, pred = best_auc, cur
                for a in (0.25, 0.5, 0.75, 1.0):
                    cand = (1 - a) * cur + a * Xs[:, i]
                    a_sc = roc_auc_score(ys, cand)
                    if a_sc > sc + 1e-8:
                        sc, pred = a_sc, cand
            if sc > best_a + 1e-8:
                best_a, best_i, best_pred = sc, i, pred
        if best_i is None:
            break
        chosen.append(best_i)
        remaining.remove(best_i)
        cur, best_auc = best_pred, best_a
    w = np.zeros(n)
    if not chosen:
        w[int(np.argmax([roc_auc_score(ys, Xs[:, i]) for i in range(n)]))] = 1.0
    else:
        w[chosen] = 1.0 / len(chosen)
    return w


def nnls_ridge_pos(X: np.ndarray, y: np.ndarray, l2: float = 1e-3):
    """Non-negative L-BFGS-B on BCE logits with analytical grad (CPU, fast)."""
    from scipy.optimize import minimize

    n = X.shape[1]
    w0 = np.ones(n) / n

    def loss_and_grad(w):
        w = np.asarray(w)
        s = X @ w
        p = expit(s)
        # BCE + L2
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + l2 * np.dot(w, w)
        grad = (X.T @ (p - y)) / len(y) + 2 * l2 * w
        return loss, grad

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(
        lambda w: loss_and_grad(w)[0],
        w0,
        jac=lambda w: loss_and_grad(w)[1],
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def load_bank():
    path = os.path.join(PRED, "meta_bank.npz")
    assert os.path.exists(path), f"missing {path}"
    z = np.load(path, allow_pickle=True)
    names = list(z["names"])
    O, T, y = z["O"].astype(np.float64), z["T"].astype(np.float64), z["y"].astype(int)
    # append any promoted stage6
    for root, _, files in os.walk(os.path.join(ROOT, "preds", "stage6")):
        for f in files:
            if not f.endswith("_oof.npy"):
                continue
            base = f[:-8]
            op = os.path.join(root, f)
            tp = os.path.join(root, base + "_full.npy")
            if not os.path.exists(tp):
                continue
            o = np.load(op).astype(np.float64)
            t = np.load(tp).astype(np.float64)
            if o.shape[0] != O.shape[0] or t.shape[0] != T.shape[0]:
                continue
            alias = f"stage6:{base}"
            if alias in names:
                continue
            # only keep if reasonably strong
            if roc_auc_score(y, o) < 0.948:
                continue
            names.append(alias)
            O = np.column_stack([O, o])
            T = np.column_stack([T, t])
            print(f"  +bank {alias} auc={roc_auc_score(y, o):.6f}", flush=True)
    return names, O, T, y


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    t0 = time.time()
    names, O, T, y = load_bank()
    ntr, nte, m = O.shape[0], T.shape[0], O.shape[1]
    folds = locked_folds(y)
    print(f"bank m={m} ntr={ntr} device={device}", flush=True)

    # nested controls (fit hillclimb on stratified 120k subsample — full-row AUC too slow)
    Orank = np.column_stack([rankdata(O[:, j]) / ntr for j in range(m)])
    Ologit = to_logit(O)
    ctrl_hill = np.zeros(ntr)
    ctrl_mega = np.zeros(ntr)

    def _strat_idx(yy, n_max=120_000, seed=0):
        if len(yy) <= n_max:
            return np.arange(len(yy))
        rng = np.random.default_rng(seed)
        pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
        n_pos = min(len(pos), max(3000, int(n_max * yy.mean())))
        n_neg = min(len(neg), n_max - n_pos)
        return np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )

    def hill_climb(Or, yy, iters=40, seed=0):
        idx = _strat_idx(yy, 120_000, seed)
        Xs, ys = Or[idx], yy[idx]
        # GPU AUC path
        xt = torch.tensor(Xs, device=device, dtype=torch.float64)
        yt = torch.tensor(ys.astype(np.float64), device=device)
        w = torch.zeros(Or.shape[1], device=device, dtype=torch.float64)
        singles = [torch_auc(yt, xt[:, i]) for i in range(Or.shape[1])]
        w[int(np.argmax(singles))] = 1.0
        cur = torch_auc(yt, xt @ w)
        for _ in range(iters):
            best, bi = cur, None
            for i in range(Or.shape[1]):
                w2 = w.clone()
                w2[i] += 1.0
                sc = torch_auc(yt, xt @ w2)
                if sc > best + 1e-8:
                    best, bi = sc, i
            if bi is None:
                break
            w[bi] += 1.0
            cur = best
        w = w.cpu().numpy()
        return w / w.sum()

    print("computing nested controls...", flush=True)
    for fi, (tr, va) in enumerate(folds):
        wh = hill_climb(Orank[tr], y[tr], seed=SEED + fi)
        hill_va = Orank[va] @ wh
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=-1).fit(
            Ologit[tr], y[tr]
        )
        stack_va = lr.decision_function(Ologit[va])
        ctrl_hill[va] = rankdata(hill_va) / len(va) + rankdata(stack_va) / len(va)
        ctrl_mega[va] = (
            LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs", n_jobs=-1)
            .fit(Ologit[tr], y[tr])
            .decision_function(Ologit[va])
        )
        print(f"  ctrl fold{fi} done", flush=True)
    ctrl_hill_auc = float(roc_auc_score(y, ctrl_hill))
    ctrl_mega_auc = float(roc_auc_score(y, ctrl_mega))
    ctrl_hill_folds = [float(roc_auc_score(y[va], ctrl_hill[va])) for _, va in folds]
    ctrl_mega_folds = [float(roc_auc_score(y[va], ctrl_mega[va])) for _, va in folds]
    print(f"controls nested: hill={ctrl_hill_auc:.6f} mega={ctrl_mega_auc:.6f}", flush=True)
    print(
        f"  nested_hill folds={[round(a, 6) for a in ctrl_hill_folds]} "
        f"mean={np.mean(ctrl_hill_folds):.6f} "
        f"f0Δmean={ctrl_hill_folds[0] - np.mean(ctrl_hill_folds):+.6f} "
        f"f1Δmean={ctrl_hill_folds[1] - np.mean(ctrl_hill_folds):+.6f}",
        flush=True,
    )
    os.makedirs(PRED, exist_ok=True)
    np.save(os.path.join(PRED, "ctrl_nested_hill_oof.npy"), ctrl_hill.astype(np.float32))
    np.save(os.path.join(PRED, "ctrl_nested_mega_oof.npy"), ctrl_mega.astype(np.float32))

    solvers = {
        "nested_nn_logit_probit": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
        "nested_probit_hillclimb": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
        "nested_hybrid_rank_logit_mix": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
        "nested_pruned_ridge_pos": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": [], "meta": []},
    }

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        Ztr, _ = build_Z(O, T, tr, tr)
        Zva, Zte = build_Z(O, T, tr, va)
        m_half = O.shape[1]
        Ztr_p, Zva_p, Zte_p = Ztr[:, m_half:], Zva[:, m_half:], Zte[:, m_half:]

        # 7.1A
        best_l2, best_w, best_a = L2_GRID[0], None, -1.0
        for l2 in L2_GRID:
            w = fit_nn_weights_cuda(Ztr, y[tr], l2, device)
            w = gpu_coord_ascent(Ztr, y[tr], w, device, iters=25)
            a = roc_auc_score(y[tr], Ztr @ w)
            if a > best_a:
                best_a, best_l2, best_w = a, l2, w
        solvers["nested_nn_logit_probit"]["oof"][va] = Zva @ best_w
        solvers["nested_nn_logit_probit"]["test"] += Zte @ best_w
        fa = float(roc_auc_score(y[va], Zva @ best_w))
        solvers["nested_nn_logit_probit"]["folds"].append(fa)
        solvers["nested_nn_logit_probit"]["meta"].append({"l2": best_l2, "tr_auc": best_a, "va_auc": fa})
        print(f"  7.1A nn_logit_probit l2={best_l2} va={fa:.6f}", flush=True)
        w_a = best_w

        # 7.1B probit-only hillclimb
        w_b = nested_probit_hillclimb(Ztr_p, y[tr], max_k=min(12, Ztr_p.shape[1]))
        w_b = gpu_coord_ascent(Ztr_p, y[tr], w_b, device, iters=30)
        solvers["nested_probit_hillclimb"]["oof"][va] = Zva_p @ w_b
        solvers["nested_probit_hillclimb"]["test"] += Zte_p @ w_b
        fa = float(roc_auc_score(y[va], Zva_p @ w_b))
        solvers["nested_probit_hillclimb"]["folds"].append(fa)
        solvers["nested_probit_hillclimb"]["meta"].append({"nnz": int((w_b > 0).sum()), "va_auc": fa})
        print(
            f"  7.1B probit_hill va={fa:.6f} vs hill={ctrl_hill_folds[f]:.6f} "
            f"Δ={fa - ctrl_hill_folds[f]:+.6f}",
            flush=True,
        )

        # 7.1C hybrid alpha mix of A (logit+probit score) and B (probit)
        pred_a_tr, pred_b_tr = Ztr @ w_a, Ztr_p @ w_b
        pred_a_va, pred_b_va = Zva @ w_a, Zva_p @ w_b
        pred_a_te, pred_b_te = Zte @ w_a, Zte_p @ w_b
        best_alpha, best_mix = 0.5, -1.0
        for alpha in np.linspace(0, 1, 21):
            sc = roc_auc_score(y[tr], alpha * pred_a_tr + (1 - alpha) * pred_b_tr)
            if sc > best_mix:
                best_mix, best_alpha = sc, float(alpha)
        mix_va = best_alpha * pred_a_va + (1 - best_alpha) * pred_b_va
        mix_te = best_alpha * pred_a_te + (1 - best_alpha) * pred_b_te
        solvers["nested_hybrid_rank_logit_mix"]["oof"][va] = mix_va
        solvers["nested_hybrid_rank_logit_mix"]["test"] += mix_te
        fa = float(roc_auc_score(y[va], mix_va))
        solvers["nested_hybrid_rank_logit_mix"]["folds"].append(fa)
        solvers["nested_hybrid_rank_logit_mix"]["meta"].append({"alpha": best_alpha, "va_auc": fa})
        print(
            f"  7.1C hybrid alpha={best_alpha:.2f} va={fa:.6f} vs hill={ctrl_hill_folds[f]:.6f} "
            f"Δ={fa - ctrl_hill_folds[f]:+.6f}",
            flush=True,
        )

        # 7.1D ridge_pos on logits + interactions (row mean/std)
        Ltr = to_logit(O[tr])
        mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
        Xl_tr = (Ltr - mu) / sd
        Xl_va = (to_logit(O[va]) - mu) / sd
        Xl_te = (to_logit(T) - mu) / sd
        # temp scale
        Ts = np.ones(m)
        for j in range(m):
            best_t, best_aa = 1.0, -1.0
            for t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                aa = roc_auc_score(y[tr], expit(Ltr[:, j] / t))
                if aa > best_aa:
                    best_aa, best_t = aa, t
            Ts[j] = best_t
        Xt = np.column_stack(
            [
                Xl_tr / Ts,
                Xl_tr.mean(1)[:, None],
                Xl_tr.std(1)[:, None],
            ]
        )
        Xv = np.column_stack([Xl_va / Ts, Xl_va.mean(1)[:, None], Xl_va.std(1)[:, None]])
        Xte = np.column_stack([Xl_te / Ts, Xl_te.mean(1)[:, None], Xl_te.std(1)[:, None]])
        w_d = nnls_ridge_pos(Xt.astype(np.float64), y[tr].astype(np.float64), l2=1e-3)
        solvers["nested_pruned_ridge_pos"]["oof"][va] = Xv @ w_d
        solvers["nested_pruned_ridge_pos"]["test"] += Xte @ w_d
        fa = float(roc_auc_score(y[va], Xv @ w_d))
        solvers["nested_pruned_ridge_pos"]["folds"].append(fa)
        solvers["nested_pruned_ridge_pos"]["meta"].append({"va_auc": fa})
        print(f"  7.1D ridge_pos va={fa:.6f}", flush=True)
        torch.cuda.empty_cache()

    os.makedirs(PRED, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

    def fold_probit_align(raw: np.ndarray) -> np.ndarray:
        out = np.zeros_like(raw, dtype=np.float64)
        for _, va in folds:
            r = rankdata(raw[va], method="average")
            u = np.clip((r - 0.5) / len(va), EPS, 1 - EPS)
            out[va] = ndtri(u)
        return out

    results, promoted = {}, []
    for name, s in solvers.items():
        s["test"] /= 5.0
        s["oof_auc_raw"] = float(roc_auc_score(y, s["oof"]))
        s["aligned"] = fold_probit_align(s["oof"])
        s["oof_auc"] = float(roc_auc_score(y, s["aligned"]))  # aligned pooled is canonical
        s["mean_fold_auc"] = float(np.mean(s["folds"]))
        deltas = [fa - c for fa, c in zip(s["folds"], ctrl_hill_folds)]
        gate = promotion_gate(ctrl_hill_folds, s["folds"], ctrl_hill_auc, s["oof_auc"])
        # also gate on mean-of-folds (immune to scale)
        gate_mean = promotion_gate(
            ctrl_hill_folds, s["folds"], float(np.mean(ctrl_hill_folds)), s["mean_fold_auc"]
        )
        pass_abs = (s["oof_auc"] > ctrl_hill_auc or s["mean_fold_auc"] > float(np.mean(ctrl_hill_folds))) and (
            gate["fold_wins"] >= 4
        )
        results[name] = {
            "oof_auc_raw": s["oof_auc_raw"],
            "oof_auc_aligned_probit": s["oof_auc"],
            "mean_fold_auc": s["mean_fold_auc"],
            "scale_gap_raw_minus_aligned": s["oof_auc_raw"] - s["oof_auc"],
            "fold_aucs": s["folds"],
            "fold_deltas_vs_nested_hill": deltas,
            "meta": s["meta"],
            "gate_vs_nested_hill_aligned": gate,
            "gate_vs_nested_hill_mean_fold": gate_mean,
            "pass_abs": pass_abs,
        }
        np.save(os.path.join(PRED, f"{name}_oof.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(PRED, f"{name}_oof_aligned.npy"), s["aligned"].astype(np.float32))
        np.save(os.path.join(PRED, f"{name}_test.npy"), s["test"].astype(np.float32))
        print(
            f"{name}: folds={[round(a,6) for a in s['folds']]} "
            f"Δ={[round(d,6) for d in deltas]} wins={gate['fold_wins']}/5 "
            f"raw={s['oof_auc_raw']:.6f} aligned={s['oof_auc']:.6f} "
            f"meanF={s['mean_fold_auc']:.6f} gap={s['oof_auc_raw']-s['oof_auc']:+.6f} "
            f"pass={pass_abs}",
            flush=True,
        )
        if pass_abs or gate["promote"] or gate_mean["promote"]:
            promoted.append(name)
            out = sub.copy()
            out["is_fraud"] = rankdata(s["test"]) / nte
            path = os.path.join(SUBS, f"stage7_1_{name}_{s['oof_auc']:.5f}.csv")
            out.to_csv(path, index=False)
            results[name]["submission"] = path
            print(f"WROTE {path}", flush=True)

    # best by aligned pooled AUC
    best_name = max(solvers, key=lambda n: solvers[n]["oof_auc"])
    best = solvers[best_name]
    np.save(os.path.join(PRED, "best_7_1_oof.npy"), best["aligned"].astype(np.float32))
    np.save(os.path.join(PRED, "best_7_1_test.npy"), best["test"].astype(np.float32))
    np.save(os.path.join(PRED, "best_7_1_oof_raw.npy"), best["oof"].astype(np.float32))
    # provisional logits (align_handoff will Platt-calibrate for Stage 8)
    rp = rankdata(best["aligned"]) / ntr
    rt = rankdata(best["test"]) / nte
    np.save(os.path.join(PRED, "best_7_1_oof_p.npy"), rp.astype(np.float32))
    np.save(os.path.join(PRED, "best_7_1_test_p.npy"), rt.astype(np.float32))
    np.save(os.path.join(PRED, "best_7_1_oof_logit.npy"), to_logit(rp).astype(np.float32))
    np.save(os.path.join(PRED, "best_7_1_test_logit.npy"), to_logit(rt).astype(np.float32))
    np.save(os.path.join(PRED, "best_7_1_w_static.npy"), np.ones(m) / m)

    gap = 0.95163 - 0.95111
    doc = {
        "stage": "7.1_gpu",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_members": m,
        "members": names,
        "controls": {
            "nested_hillclimb_mix": {"oof": ctrl_hill_auc, "folds": ctrl_hill_folds},
            "nested_mega_logit": {"oof": ctrl_mega_auc, "folds": ctrl_mega_folds},
            "gate_baseline_oof": ctrl_hill_auc,
        },
        "solvers": results,
        "best_solver": best_name,
        "best_aligned_oof": best["oof_auc"],
        "promoted": promoted,
        "est_public": {n: solvers[n]["oof_auc"] + gap for n in solvers},
        "note": "Canonical OOF is per-fold probit-aligned; run stage7_align_handoff.py before Stage 8",
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    path = os.path.join(LOGS, "stage7_1_gpu_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(f"wrote {path} best={best_name} aligned_OOF={best['oof_auc']:.6f} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
