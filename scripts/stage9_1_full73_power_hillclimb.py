#!/usr/bin/env python3
"""Stage 9.1: Full unpruned bank + bounded power-rank / clipped-logit GPU hillclimb.

NO unbounded ndtri. Gate vs nested_hill 0.951140. NO kaggle submit.
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
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from oof_bank_loader import collect_oof_bank, load_y  # noqa: E402
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "stage9")
PRED7 = os.path.join(ROOT, "preds", "stage7")
SUBS = os.path.join(ROOT, "submissions")
SEED = 42
EPS = 5e-5
CTRL_HILL = 0.951140
POWERS = (0.5, 1.0, 1.5, 2.5, 4.0)


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def to_logit_clip(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def torch_auc(y: torch.Tensor, s: torch.Tensor) -> float:
    order = torch.argsort(s)
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = torch.arange(1, len(y_sorted) + 1, device=s.device, dtype=torch.float64)
    sum_pos = ranks[y_sorted.bool()].sum()
    return float(((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())


def build_bounded_bases(O: np.ndarray, T: np.ndarray, tr: np.ndarray):
    """Expand each model into bounded ranks/powers + clipped std logit (fit on tr)."""
    n_tr, m = len(tr), O.shape[1]
    n_all, n_te = O.shape[0], T.shape[0]
    # columns: for each model: powers on rank + 1 logit = len(POWERS)+1
    d = m * (len(POWERS) + 1)
    Zo = np.zeros((n_all, d), np.float64)
    Zt = np.zeros((n_te, d), np.float64)
    # global ranks for OOF bank (each column independently) — nested: for apply on va we
    # recompute ranks within slice when scoring; for feature matrix used in hillclimb we
    # use train-fit logit stats and full-column ranks (rank within train for tr features,
    # within apply set for va/test).
    col = 0
    Ltr = to_logit_clip(O[tr])
    mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
    for j in range(m):
        # ranks within full O / T for test; for OOF we will rebuild per-fold below
        r_all = (rankdata(O[:, j], method="average") - 0.5) / n_all
        r_te = (rankdata(T[:, j], method="average") - 0.5) / n_te
        for pwr in POWERS:
            Zo[:, col] = np.power(np.clip(r_all, 0, 1), pwr)
            Zt[:, col] = np.power(np.clip(r_te, 0, 1), pwr)
            col += 1
        Zo[:, col] = (to_logit_clip(O[:, j]) - mu[j]) / sd[j]
        Zt[:, col] = (to_logit_clip(T[:, j]) - mu[j]) / sd[j]
        col += 1
    return Zo.astype(np.float32), Zt.astype(np.float32)


def build_bases_nested(O, T, tr, va):
    """Leak-free: ranks within tr / va / test separately; logit std from tr."""
    m = O.shape[1]
    d = m * (len(POWERS) + 1)
    Ztr = np.zeros((len(tr), d), np.float32)
    Zva = np.zeros((len(va), d), np.float32)
    Zte = np.zeros((T.shape[0], d), np.float32)
    Ltr = to_logit_clip(O[tr])
    mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
    col = 0
    for j in range(m):
        r_tr = (rankdata(O[tr, j], method="average") - 0.5) / len(tr)
        r_va = (rankdata(O[va, j], method="average") - 0.5) / len(va)
        r_te = (rankdata(T[:, j], method="average") - 0.5) / len(T)
        for pwr in POWERS:
            Ztr[:, col] = np.power(np.clip(r_tr, 0, 1), pwr)
            Zva[:, col] = np.power(np.clip(r_va, 0, 1), pwr)
            Zte[:, col] = np.power(np.clip(r_te, 0, 1), pwr)
            col += 1
        Ztr[:, col] = ((to_logit_clip(O[tr, j]) - mu[j]) / sd[j]).astype(np.float32)
        Zva[:, col] = ((to_logit_clip(O[va, j]) - mu[j]) / sd[j]).astype(np.float32)
        Zte[:, col] = ((to_logit_clip(T[:, j]) - mu[j]) / sd[j]).astype(np.float32)
        col += 1
    return Ztr, Zva, Zte


def gpu_bagged_hillclimb(X: np.ndarray, y: np.ndarray, device: str, iters: int = 80, n_bags: int = 5):
    """Coordinate +1 / optional -1 shrink; score = mean AUC over bootstrap bags of tr."""
    rng = np.random.default_rng(SEED)
    n, d = X.shape
    bags = []
    for b in range(n_bags):
        idx = rng.choice(n, size=min(n, 120_000), replace=False)
        bags.append(
            (
                torch.tensor(X[idx], device=device, dtype=torch.float64),
                torch.tensor(y[idx].astype(np.float64), device=device),
            )
        )
    mass = torch.zeros(d, device=device, dtype=torch.float64)
    # seed with best single column on bag0
    xt0, yt0 = bags[0]
    singles = [torch_auc(yt0, xt0[:, j]) for j in range(d)]
    mass[int(np.argmax(singles))] = 8.0

    def score(m):
        if m.sum() <= 0:
            return 0.0
        w = m / m.sum()
        return float(np.mean([torch_auc(yt, xt @ w) for xt, yt in bags]))

    cur = score(mass)
    for _ in range(iters):
        best, bi, bsign = cur, None, 1
        for j in range(d):
            for sign, delta in ((1, 1.0), (-1, -1.0)):
                if sign < 0 and mass[j] < 1.0:
                    continue
                m2 = mass.clone()
                m2[j] = m2[j] + delta
                if m2.sum() <= 0:
                    continue
                sc = score(m2)
                # light L2 on weight concentration
                w = (m2 / m2.sum()).cpu().numpy()
                sc_pen = sc - 1e-4 * float(np.dot(w, w))
                if sc_pen > best + 1e-8:
                    best, bi, bsign = sc_pen, j, sign
        if bi is None:
            break
        mass[bi] = mass[bi] + (1.0 if bsign > 0 else -1.0)
        cur = best
    w = mass / mass.sum()
    return w.cpu().numpy()


def classic_nested_hill_ranks(O, y, folds, device="cuda"):
    """Reproduce nested_hill style: rank bank + LR mix (control for full73)."""
    ntr, m = O.shape
    Orank = np.column_stack([rankdata(O[:, j]) / ntr for j in range(m)]).astype(np.float64)
    Ologit = to_logit_clip(O)
    oof = np.zeros(ntr)
    rng = np.random.default_rng(SEED)
    for fi, (tr, va) in enumerate(folds):
        # subsample train for hillclimb speed
        if len(tr) > 120_000:
            pos = tr[y[tr] == 1]
            neg = tr[y[tr] == 0]
            n_pos = min(len(pos), max(3000, int(120_000 * y[tr].mean())))
            n_neg = min(len(neg), 120_000 - n_pos)
            sub = np.concatenate(
                [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
            )
        else:
            sub = tr
        xt = torch.tensor(Orank[sub], device=device, dtype=torch.float64)
        yt = torch.tensor(y[sub].astype(np.float64), device=device)
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
        hill_va = Orank[va] @ ww
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(Ologit[tr], y[tr])
        stack_va = lr.decision_function(Ologit[va])
        oof[va] = rankdata(hill_va) / len(va) + rankdata(stack_va) / len(va)
        print(f"  ctrl73 fold{fi} done", flush=True)
    return oof


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    t0 = time.time()
    y, ntr, nte = load_y(ROOT)
    folds = locked_folds(y)
    pre = os.path.join(PRED, "full_bank_unpruned.npz")
    if os.path.exists(pre):
        z = np.load(pre, allow_pickle=True)
        names = list(z["names"])
        O, T = z["O"].astype(np.float64), z["T"].astype(np.float64)
        if "y" in z.files:
            y = z["y"].astype(int)
        print(f"loaded prebuilt full bank {pre}", flush=True)
    else:
        bank = collect_oof_bank(ROOT, ntr, nte, require_test=True, y=y)
        for alias, op, tp in [
            ("self:te_xgb_smooth50", "preds/stage4/te_xgb_smooth50_oof.npy", "preds/stage4/te_xgb_smooth50_full.npy"),
            ("self:te_lgbm_leaves3", "preds/stage5/te_lgbm_leaves3_oof.npy", "preds/stage5/te_lgbm_leaves3_full.npy"),
        ]:
            op, tp = os.path.join(ROOT, op), os.path.join(ROOT, tp)
            if os.path.exists(op) and os.path.exists(tp):
                bank[alias] = (np.load(op).astype(np.float64), np.load(tp).astype(np.float64))
        names = sorted(bank, key=lambda n: -roc_auc_score(y, bank[n][0]))
        names = [n for n in names if roc_auc_score(y, bank[n][0]) >= 0.94]
        O = np.column_stack([bank[n][0] for n in names])
        T = np.column_stack([bank[n][1] for n in names])
        os.makedirs(PRED, exist_ok=True)
        np.savez_compressed(pre, names=np.array(names, dtype=object), O=O, T=T, y=y)
    m = O.shape[1]
    print(f"FULL BANK unpruned m={m} (target ~73)", flush=True)
    top15 = [(names[j], float(roc_auc_score(y, O[:, j]))) for j in range(min(15, m))]
    print("=== Top 15 standalone ===", flush=True)
    for i, (n, a) in enumerate(top15, 1):
        print(f"  {i:2d}. {a:.6f}  {n}", flush=True)

    # control: nested_hill on full73 + load saved ctrl if present
    print("computing nested_hill_full73 control...", flush=True)
    ctrl73 = classic_nested_hill_ranks(O, y, folds, device=device)
    ctrl73_auc = float(roc_auc_score(y, ctrl73))
    ctrl73_folds = [float(roc_auc_score(y[va], ctrl73[va])) for _, va in folds]
    print(f"nested_hill_full73 OOF={ctrl73_auc:.6f} folds={[round(a,6) for a in ctrl73_folds]}", flush=True)

    # reference nested_hill from stage7 if available
    ctrl_ref = CTRL_HILL
    cref = os.path.join(PRED7, "ctrl_nested_hill_oof.npy")
    if os.path.exists(cref):
        c = np.load(cref)
        ctrl_ref = float(roc_auc_score(y, c))
        print(f"saved nested_hill (pruned48) OOF={ctrl_ref:.6f}", flush=True)

    oof = np.zeros(ntr)
    te_sum = np.zeros(nte)
    fold_aucs = []
    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        Ztr, Zva, Zte = build_bases_nested(O, T, tr, va)
        print(f"  bases d={Ztr.shape[1]}", flush=True)
        w = gpu_bagged_hillclimb(Ztr, y[tr], device, iters=60, n_bags=5)
        pred_va = Zva @ w
        pred_te = Zte @ w
        # bounded rank-align va for pooling
        rva = (rankdata(pred_va, method="average") - 0.5) / len(va)
        oof[va] = rva
        te_sum += (rankdata(pred_te, method="average") - 0.5) / len(pred_te)
        fa = float(roc_auc_score(y[va], pred_va))
        fold_aucs.append(fa)
        print(f"  power_hill va={fa:.6f} vs hill73={ctrl73_folds[f]:.6f} Δ={fa-ctrl73_folds[f]:+.6f}", flush=True)
        torch.cuda.empty_cache()

    te = te_sum / 5.0
    oof_auc = float(roc_auc_score(y, oof))
    gate = promotion_gate(ctrl73_folds, fold_aucs, ctrl73_auc, oof_auc)
    gate_ref = promotion_gate([ctrl_ref] * 5, fold_aucs, ctrl_ref, oof_auc)
    print(f"power_hill OOF={oof_auc:.6f} vs full73_hill={ctrl73_auc:.6f} {gate['reason']}", flush=True)

    os.makedirs(PRED, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    np.save(os.path.join(PRED, "full73_power_hill_oof.npy"), oof.astype(np.float32))
    np.save(os.path.join(PRED, "full73_power_hill_test.npy"), te.astype(np.float32))
    np.save(os.path.join(PRED, "nested_hill_full73_oof.npy"), ctrl73.astype(np.float32))

    # materialize nested_hill_full73 test via same recipe on full data weights averaged
    # approximate test: mean of top-8 by OOF
    top8 = np.argsort([-roc_auc_score(y, O[:, j]) for j in range(m)])[:8]
    te_hill = T[:, top8].mean(1)
    # submission mix: GPU hill on subsample + full LR stack
    Orank = np.column_stack([rankdata(O[:, j]) / ntr for j in range(m)]).astype(np.float64)
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = min(len(pos), 4000)
    n_neg = min(len(neg), 116_000)
    sub = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    xt = torch.tensor(Orank[sub], device=device, dtype=torch.float64)
    yt = torch.tensor(y[sub].astype(np.float64), device=device)
    w = torch.zeros(m, device=device, dtype=torch.float64)
    singles = [torch_auc(yt, xt[:, j]) for j in range(m)]
    w[int(np.argmax(singles))] = 1.0
    cur = torch_auc(yt, xt @ w)
    for _ in range(40):
        best, bi = cur, None
        for j in range(m):
            w2 = w.clone(); w2[j] += 1.0
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
    Ologit = to_logit_clip(O)
    Tlogit = to_logit_clip(T)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(Ologit, y)
    te_stack = lr.decision_function(Tlogit)
    te_mix = rankdata(te_hill) / nte + rankdata(te_stack) / nte
    np.save(os.path.join(PRED, "nested_hill_full73_test.npy"), te_mix.astype(np.float32))

    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    out = sub.copy()
    out["is_fraud"] = rankdata(te_mix) / nte
    path_csv = os.path.join(SUBS, f"nested_hill_full73_{ctrl73_auc:.5f}.csv")
    out.to_csv(path_csv, index=False)
    spear = None
    href = os.path.join(SUBS, "hillclimb_mix_0.95111.csv")
    if os.path.exists(href):
        spear = float(spearmanr(out["is_fraud"], pd.read_csv(href)["is_fraud"]).correlation)
    print(f"WROTE {path_csv} spearman_vs_hillclimb_mix={spear}", flush=True)

    if gate["promote"] or oof_auc > ctrl73_auc:
        out2 = sub.copy()
        out2["is_fraud"] = rankdata(te) / nte
        p2 = os.path.join(SUBS, f"stage9_1_power_hill_{oof_auc:.5f}.csv")
        out2.to_csv(p2, index=False)
        print(f"WROTE {p2}", flush=True)

    # update campaign_best if improved
    best_oof, best_te, best_name = ctrl73, te_mix, "nested_hill_full73"
    if oof_auc > ctrl73_auc:
        best_oof, best_te, best_name = oof, te, "full73_power_hill"
    # Platt
    from sklearn.linear_model import LogisticRegression as LR

    p_oof = np.zeros(ntr)
    p_te = np.zeros(nte)
    for tr, va in folds:
        lr = LR(C=1e6, max_iter=1000, solver="lbfgs")
        lr.fit(best_oof[tr].reshape(-1, 1), y[tr])
        p_oof[va] = lr.predict_proba(best_oof[va].reshape(-1, 1))[:, 1]
        p_te += lr.predict_proba(best_te.reshape(-1, 1))[:, 1]
    p_te /= 5.0
    Lo, Lt = to_logit_clip(p_oof), to_logit_clip(p_te)
    os.makedirs(PRED7, exist_ok=True)
    np.save(os.path.join(PRED7, "campaign_best_oof.npy"), best_oof.astype(np.float32))
    np.save(os.path.join(PRED7, "campaign_best_test.npy"), best_te.astype(np.float32))
    np.save(os.path.join(PRED7, "campaign_best_oof_p.npy"), p_oof.astype(np.float32))
    np.save(os.path.join(PRED7, "campaign_best_test_p.npy"), p_te.astype(np.float32))
    np.save(os.path.join(PRED7, "campaign_best_oof_logit.npy"), Lo.astype(np.float32))
    np.save(os.path.join(PRED7, "campaign_best_test_logit.npy"), Lt.astype(np.float32))
    print(f"campaign_best ← {best_name} auc={roc_auc_score(y, best_oof):.6f} platt_p={p_oof.mean():.5f}", flush=True)

    doc = {
        "stage": "9.1_full73_power_hillclimb",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_members": m,
        "top15": [{"name": n, "oof_auc": a} for n, a in top15],
        "nested_hill_full73": {"oof": ctrl73_auc, "folds": ctrl73_folds, "submission": path_csv, "spearman_vs_0.95111": spear},
        "power_hill": {
            "oof": oof_auc,
            "folds": fold_aucs,
            "gate_vs_full73_hill": gate,
            "gate_vs_ref_0.95114": gate_ref,
            "promote": gate["promote"],
        },
        "campaign_best": best_name,
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    path = os.path.join(LOGS, "stage9_1_full73_power_hillclimb_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    # aggregate stub
    agg = os.path.join(LOGS, "stage9_results.json")
    prev = json.load(open(agg)) if os.path.exists(agg) else {}
    prev["stage9_1"] = doc
    prev["updated_utc"] = datetime.now(timezone.utc).isoformat()
    json.dump(prev, open(agg, "w"), indent=2)
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
