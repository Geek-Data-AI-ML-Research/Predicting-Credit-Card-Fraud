#!/usr/bin/env python3
"""Stage 7.2: CUDA level-2 non-linear meta-stacker over pruned OOF bank.

Expects preds/stage7/meta_bank.npz with keys: names, O (n_train,m), T (n_test,m), y.
Optional context features from data/train.csv columns.
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
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.special import logit, ndtri
from scipy.stats import rankdata, skew
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
CTRL_HILL = 0.95111
CTRL_MEGA = 0.95115
CTX = [
    "merchant_risk_score",
    "velocity_score",
    "ip_country_mismatch",
    "cvv_retry_count",
    "is_ai_generated_scam_attempt",
]


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def probit_rank(p):
    r = rankdata(p, method="average")
    u = np.clip((r - 0.5) / len(p), EPS, 1 - EPS)
    return ndtri(u)


def build_meta_X(O_slice: np.ndarray, ctx_slice: np.ndarray | None, fit_stats=None):
    """Build meta features for a row-block. If fit_stats given, reuse logit mean/std."""
    n, m = O_slice.shape
    L = to_logit(O_slice)
    if fit_stats is None:
        mu, sd = L.mean(0), L.std(0) + 1e-12
        fit_stats = {"mu": mu, "sd": sd}
    Zl = (L - fit_stats["mu"]) / fit_stats["sd"]
    Zp = np.column_stack([probit_rank(O_slice[:, j]) for j in range(m)])
    # row-wise dispersion across models
    row_mean = L.mean(1)
    row_std = L.std(1)
    row_range = L.max(1) - L.min(1)
    row_skew = skew(L, axis=1, bias=False)
    # top5 by column mean on this block (approx) — use global top5 from fit_stats if present
    if "top5" in fit_stats:
        top5 = fit_stats["top5"]
    else:
        top5 = np.argsort(-L.mean(0))[:5]
        fit_stats["top5"] = top5
    top5_std = L[:, top5].std(1)
    feats = [Zl, Zp, row_mean[:, None], row_std[:, None], row_range[:, None], row_skew[:, None], top5_std[:, None]]
    if ctx_slice is not None:
        feats.append(ctx_slice)
    X = np.hstack(feats).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, fit_stats


class BottleneckMeta(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 32):
        super().__init__()
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, 1)
        self.skip = nn.Linear(d_in, 1, bias=False)

    def forward(self, x):
        h = torch.nn.functional.silu(self.fc1(self.drop(x)))
        return self.fc2(h).squeeze(-1) + self.skip(x).squeeze(-1)


def train_meta_mlp(X_tr, y_tr, X_va, y_va, device, epochs=40, lr=2e-3):
    model = BottleneckMeta(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(X_tr, device=device)
    yt = torch.tensor(y_tr.astype(np.float32), device=device)
    xv = torch.tensor(X_va, device=device)
    best_state, best_auc, bad = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).cpu().numpy()
        auc = roc_auc_score(y_va, pv)
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_mlp(model, X, device):
    model.eval()
    return torch.sigmoid(model(torch.tensor(X, device=device))).cpu().numpy()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    t0 = time.time()
    bank_path = os.path.join(PRED, "meta_bank.npz")
    assert os.path.exists(bank_path), f"missing {bank_path} — run export_meta_bank.py first"
    z = np.load(bank_path, allow_pickle=True)
    names = list(z["names"])
    O, T, y = z["O"].astype(np.float64), z["T"].astype(np.float64), z["y"].astype(int)
    ntr, nte, m = O.shape[0], T.shape[0], O.shape[1]
    print(f"bank m={m} ntr={ntr} nte={nte} members={len(names)}", flush=True)

    ctx_tr = ctx_te = None
    tr_csv = os.path.join(DATA, "train.csv")
    te_csv = os.path.join(DATA, "test.csv")
    if os.path.exists(tr_csv):
        tr = pd.read_csv(tr_csv, usecols=["id"] + [c for c in CTX if True])
        te = pd.read_csv(te_csv, usecols=["id"] + CTX)
        ctx_tr = tr[CTX].astype(np.float32).values
        ctx_te = te[CTX].astype(np.float32).values
        # standardize ctx using train
        mu, sd = ctx_tr.mean(0), ctx_tr.std(0) + 1e-12
        ctx_tr = (ctx_tr - mu) / sd
        ctx_te = (ctx_te - mu) / sd

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))
    solvers = {
        "meta_xgb_cuda": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": []},
        "meta_mlp_cuda": {"oof": np.zeros(ntr), "test": np.zeros(nte), "folds": []},
    }

    for f, (tr, va) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        Xtr, stats = build_meta_X(O[tr], None if ctx_tr is None else ctx_tr[tr], fit_stats=None)
        Xva, _ = build_meta_X(O[va], None if ctx_tr is None else ctx_tr[va], fit_stats=stats)
        Xte, _ = build_meta_X(T, ctx_te, fit_stats=stats)

        # Arm A: shallow CUDA XGB
        m = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.015,
            n_estimators=4000,
            colsample_bytree=0.35,
            subsample=0.7,
            reg_lambda=50.0,
            reg_alpha=10.0,
            tree_method="hist",
            device="cuda",
            eval_metric="auc",
            early_stopping_rounds=100,
            random_state=SEED + f,
        )
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], verbose=False)
        pva = m.predict_proba(Xva)[:, 1]
        pte = m.predict_proba(Xte)[:, 1]
        solvers["meta_xgb_cuda"]["oof"][va] = pva
        solvers["meta_xgb_cuda"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        solvers["meta_xgb_cuda"]["folds"].append(fa)
        print(f"  meta_xgb va={fa:.6f}", flush=True)

        # Arm B: bottleneck residual MLP
        mlp = train_meta_mlp(Xtr, y[tr], Xva, y[va], device)
        pva = predict_mlp(mlp, Xva, device)
        pte = predict_mlp(mlp, Xte, device)
        solvers["meta_mlp_cuda"]["oof"][va] = pva
        solvers["meta_mlp_cuda"]["test"] += pte
        fa = float(roc_auc_score(y[va], pva))
        solvers["meta_mlp_cuda"]["folds"].append(fa)
        print(f"  meta_mlp va={fa:.6f}", flush=True)
        torch.cuda.empty_cache()

    results = {}
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

    for name, s in solvers.items():
        s["test"] /= 5.0
        s["oof_auc"] = float(roc_auc_score(y, s["oof"]))
        # gate vs absolute controls (fold vectors unknown — compare OOF + fold mean proxy)
        # use own fold vector vs constant control folds at control OOF level as soft check
        ctrl_folds = [CTRL_MEGA] * 5
        gate = promotion_gate(ctrl_folds, s["folds"], CTRL_MEGA, s["oof_auc"])
        pass_abs = s["oof_auc"] > CTRL_MEGA and gate["fold_wins"] >= 4
        # softer: also vs hillclimb
        gate_h = promotion_gate([CTRL_HILL] * 5, s["folds"], CTRL_HILL, s["oof_auc"])
        results[name] = {
            "oof_auc": s["oof_auc"],
            "fold_aucs": s["folds"],
            "gate_vs_mega_0.95115": gate,
            "gate_vs_hill_0.95111": gate_h,
            "pass_abs_vs_0.95115": pass_abs,
        }
        np.save(os.path.join(PRED, f"{name}_oof.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(PRED, f"{name}_test.npy"), s["test"].astype(np.float32))
        if pass_abs or gate["promote"] or gate_h["promote"]:
            out = sub.copy()
            out["is_fraud"] = rankdata(s["test"]) / nte
            assert len(out) == 299984 and np.isfinite(out["is_fraud"]).all()
            path = os.path.join(SUBS, f"stage7_2_{name}_{s['oof_auc']:.5f}.csv")
            out.to_csv(path, index=False)
            results[name]["submission"] = path
            print(f"WROTE {path}", flush=True)
        print(f"{name}: OOF={s['oof_auc']:.6f} pass_abs={pass_abs}", flush=True)

    gap = 0.95163 - 0.95111
    doc = {
        "stage": "7.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_members": m,
        "members": names,
        "controls": {"hillclimb_mix": CTRL_HILL, "mega_logit": CTRL_MEGA},
        "solvers": results,
        "est_public": {n: solvers[n]["oof_auc"] + gap for n in solvers},
        "policy": "NO_KAGGLE_SUBMIT",
    }
    path = os.path.join(LOGS, "stage7_2_gpu_meta_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
