#!/usr/bin/env python3
"""Stage 7.2: Context-gated attention meta-net + shallow CUDA XGB (+ OOF blend).

NO kaggle submit. Gate vs nested hillclimb control from Stage 7.1 JSON / 0.95118.
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
import torch.nn.functional as F
import xgboost as xgb
from scipy.special import logit
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
TOP_M = 25
CTX = [
    "velocity_score",
    "merchant_risk_score",
    "ip_country_mismatch",
    "cvv_retry_count",
    "distance_from_home_km",
    "is_ai_generated_scam_attempt",
]


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def pairwise_auc_loss(logits, y, n_pairs=8192):
    pos = torch.where(y > 0.5)[0]
    neg = torch.where(y < 0.5)[0]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_zeros(())
    n = min(n_pairs, max(len(pos), len(neg)) * 4)
    ip = pos[torch.randint(0, len(pos), (n,), device=logits.device)]
    inn = neg[torch.randint(0, len(neg), (n,), device=logits.device)]
    return F.relu(1.0 - (logits[ip] - logits[inn])).mean()


class GatedMetaNet(nn.Module):
    """Row-specific gates around static w0 + small residual correction."""

    def __init__(self, n_models: int, n_ctx: int, temp: float = 1.5):
        super().__init__()
        self.temp = temp
        self.ctx1 = nn.Linear(n_ctx, 32)
        self.drop = nn.Dropout(0.25)
        self.ctx2 = nn.Linear(32, n_models)
        self.resid = nn.Sequential(
            nn.Linear(n_models + n_ctx, 32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        self.w0 = nn.Parameter(torch.zeros(n_models), requires_grad=False)

    def set_w0(self, w0: np.ndarray):
        with torch.no_grad():
            self.w0.copy_(torch.tensor(w0, dtype=torch.float32))

    def forward(self, z, c):
        # z: (N, M) standardized logits; c: (N, C)
        h = F.silu(self.ctx1(self.drop(c)))
        delta = self.ctx2(h) / self.temp
        g = torch.softmax(self.w0.unsqueeze(0) + delta, dim=-1)
        blend = (g * z).sum(-1)
        resid = self.resid(torch.cat([z, c], dim=-1)).squeeze(-1)
        return blend + 0.15 * resid


def build_ctx_dispersion(L: np.ndarray, ctx: np.ndarray | None):
    row_std = L.std(1)
    row_range = L.max(1) - L.min(1)
    row_skew = skew(L, axis=1, bias=False)
    top5 = np.argsort(-L.mean(0))[:5]
    top5_std = L[:, top5].std(1)
    top10 = np.argsort(-L.mean(0))[:10]
    bot10 = np.argsort(L.mean(0))[:10]
    topbot = L[:, top10].mean(1) - L[:, bot10].mean(1)
    feats = [row_std, row_range, row_skew, top5_std, topbot]
    C = np.column_stack(feats).astype(np.float32)
    if ctx is not None:
        C = np.hstack([C, ctx.astype(np.float32)])
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return C


def train_gated(z_tr, c_tr, y_tr, z_va, c_va, y_va, w0, device, epochs=60):
    model = GatedMetaNet(z_tr.shape[1], c_tr.shape[1]).to(device)
    model.set_w0(w0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    zt = torch.tensor(z_tr, device=device)
    ct = torch.tensor(c_tr, device=device)
    yt = torch.tensor(y_tr.astype(np.float32), device=device)
    zv = torch.tensor(z_va, device=device)
    cv = torch.tensor(c_va, device=device)
    best_state, best_auc, bad = None, -1.0, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(zt, ct)
        loss = F.binary_cross_entropy_with_logits(logits, yt) + 0.3 * pairwise_auc_loss(logits, yt)
        loss.backward()
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(zv, cv)).cpu().numpy()
        auc = roc_auc_score(y_va, pv)
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_gated(model, z, c, device):
    return torch.sigmoid(
        model(torch.tensor(z, device=device), torch.tensor(c, device=device))
    ).cpu().numpy()


def load_ctrl():
    p = os.path.join(LOGS, "stage7_1_gpu_result.json")
    if os.path.exists(p):
        d = json.load(open(p))
        c = d["controls"]["nested_hillclimb_mix"]
        return float(c["oof"]), list(c["folds"])
    return 0.95118, [0.95118] * 5


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    t0 = time.time()
    z = np.load(os.path.join(PRED, "meta_bank.npz"), allow_pickle=True)
    names = list(z["names"])
    O, T, y = z["O"].astype(np.float64), z["T"].astype(np.float64), z["y"].astype(int)
    # top25 by OOF AUC
    aucs = [roc_auc_score(y, O[:, j]) for j in range(O.shape[1])]
    top = np.argsort(-np.asarray(aucs))[:TOP_M]
    O25, T25 = O[:, top], T[:, top]
    names25 = [names[i] for i in top]
    print(f"top{TOP_M}: {[round(aucs[i], 5) for i in top[:5]]}...", flush=True)

    tr = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id"] + CTX)
    te = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"] + CTX)
    ctx_tr = tr[CTX].astype(np.float32).values
    ctx_te = te[CTX].astype(np.float32).values
    mu, sd = ctx_tr.mean(0), ctx_tr.std(0) + 1e-12
    ctx_tr = (ctx_tr - mu) / sd
    ctx_te = (ctx_te - mu) / sd

    # static w0 from equal / optional best_7_1
    w0 = np.ones(TOP_M, np.float32) / TOP_M
    best71_oof = best71_te = None
    p71 = os.path.join(PRED, "best_7_1_oof.npy")
    if os.path.exists(p71):
        best71_oof = np.load(p71)
        best71_te = np.load(os.path.join(PRED, "best_7_1_test.npy"))

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    ctrl_oof, ctrl_folds = load_ctrl()
    print(f"control nested hill={ctrl_oof:.6f}", flush=True)

    solvers = {
        "gated_meta_net": {"oof": np.zeros(len(y)), "test": np.zeros(len(T)), "folds": []},
        "meta_xgb_cuda": {"oof": np.zeros(len(y)), "test": np.zeros(len(T)), "folds": []},
        "blend_71_gated_xgb": {"oof": np.zeros(len(y)), "test": np.zeros(len(T)), "folds": []},
    }

    for f, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n=== fold {f} ===", flush=True)
        Ltr = to_logit(O25[tr_idx])
        mu_l, sd_l = Ltr.mean(0), Ltr.std(0) + 1e-12
        z_tr = ((Ltr - mu_l) / sd_l).astype(np.float32)
        z_va = ((to_logit(O25[va_idx]) - mu_l) / sd_l).astype(np.float32)
        z_te = ((to_logit(T25) - mu_l) / sd_l).astype(np.float32)
        c_tr = build_ctx_dispersion(Ltr, ctx_tr[tr_idx])
        c_va = build_ctx_dispersion(to_logit(O25[va_idx]), ctx_tr[va_idx])
        c_te = build_ctx_dispersion(to_logit(T25), ctx_te)
        # standardize ctx block using train
        cmu, csd = c_tr.mean(0), c_tr.std(0) + 1e-12
        c_tr = ((c_tr - cmu) / csd).astype(np.float32)
        c_va = ((c_va - cmu) / csd).astype(np.float32)
        c_te = ((c_te - cmu) / csd).astype(np.float32)

        # fit w0 on train via softmax Adam quick
        w0_fold = w0.copy()
        xt = torch.tensor(z_tr, device=device)
        yt = torch.tensor(y[tr_idx].astype(np.float32), device=device)
        u = torch.zeros(TOP_M, device=device, requires_grad=True)
        opt = torch.optim.Adam([u], lr=0.08)
        for _ in range(80):
            opt.zero_grad()
            ww = torch.softmax(u, 0)
            loss = F.binary_cross_entropy_with_logits(xt @ ww, yt)
            loss.backward()
            opt.step()
        w0_fold = torch.softmax(u, 0).detach().cpu().numpy()

        gated = train_gated(z_tr, c_tr, y[tr_idx], z_va, c_va, y[va_idx], w0_fold, device)
        pva_g = predict_gated(gated, z_va, c_va, device)
        pte_g = predict_gated(gated, z_te, c_te, device)
        solvers["gated_meta_net"]["oof"][va_idx] = pva_g
        solvers["gated_meta_net"]["test"] += pte_g
        fa = float(roc_auc_score(y[va_idx], pva_g))
        solvers["gated_meta_net"]["folds"].append(fa)
        print(f"  gated va={fa:.6f}", flush=True)

        # meta XGB on [z | c]
        Xtr = np.hstack([z_tr, c_tr])
        Xva = np.hstack([z_va, c_va])
        Xte = np.hstack([z_te, c_te])
        m = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.015,
            n_estimators=4000,
            subsample=0.7,
            colsample_bytree=0.35,
            reg_lambda=50.0,
            reg_alpha=10.0,
            tree_method="hist",
            device="cuda",
            eval_metric="auc",
            early_stopping_rounds=100,
            random_state=SEED + f,
        )
        m.fit(Xtr, y[tr_idx], eval_set=[(Xva, y[va_idx])], verbose=False)
        pva_x = m.predict_proba(Xva)[:, 1]
        pte_x = m.predict_proba(Xte)[:, 1]
        solvers["meta_xgb_cuda"]["oof"][va_idx] = pva_x
        solvers["meta_xgb_cuda"]["test"] += pte_x
        fa = float(roc_auc_score(y[va_idx], pva_x))
        solvers["meta_xgb_cuda"]["folds"].append(fa)
        print(f"  meta_xgb va={fa:.6f}", flush=True)

        # blend with 7.1 if available — pick alpha on tr via 7.1 oof proxy (use train-side static)
        if best71_oof is not None:
            # nest: use 7.1 OOF on va (already OOF — honest) and optimize blend weights on tr using
            # train predictions approximated by gated/xgb train preds
            pva_71 = best71_oof[va_idx]
            pte_71 = best71_te
            # choose alpha,beta on a small grid using va? NO — use tr predictions
            ptr_g = predict_gated(gated, z_tr, c_tr, device)
            ptr_x = m.predict_proba(Xtr)[:, 1]
            # approximate 7.1 on tr: use rank of best71 on tr (leak-free: it's OOF so ok for blend fit)
            ptr_71 = best71_oof[tr_idx]
            best = (-1.0, 0.7, 0.2, 0.1)
            for a in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
                for b in (0.05, 0.1, 0.15, 0.2, 0.25):
                    c = 1.0 - a - b
                    if c < 0:
                        continue
                    sc = roc_auc_score(y[tr_idx], a * ptr_71 + b * ptr_g + c * ptr_x)
                    if sc > best[0]:
                        best = (sc, a, b, c)
            _, a, b, c = best
            blend_va = a * pva_71 + b * pva_g + c * pva_x
            blend_te = a * pte_71 + b * pte_g + c * pte_x
        else:
            a, b, c = 0.0, 0.55, 0.45
            blend_va = b * pva_g + c * pva_x
            blend_te = b * pte_g + c * pte_x
        solvers["blend_71_gated_xgb"]["oof"][va_idx] = blend_va
        solvers["blend_71_gated_xgb"]["test"] += blend_te
        fa = float(roc_auc_score(y[va_idx], blend_va))
        solvers["blend_71_gated_xgb"]["folds"].append(fa)
        print(f"  blend a={a:.2f} b={b:.2f} c={c:.2f} va={fa:.6f}", flush=True)
        torch.cuda.empty_cache()

    os.makedirs(PRED, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    results, promoted = {}, []
    nte = len(T)
    for name, s in solvers.items():
        s["test"] /= 5.0
        s["oof_auc"] = float(roc_auc_score(y, s["oof"]))
        gate = promotion_gate(ctrl_folds, s["folds"], ctrl_oof, s["oof_auc"])
        pass_abs = s["oof_auc"] > ctrl_oof and gate["fold_wins"] >= 4
        results[name] = {
            "oof_auc": s["oof_auc"],
            "fold_aucs": s["folds"],
            "gate": gate,
            "pass_abs": pass_abs,
        }
        np.save(os.path.join(PRED, f"{name}_oof.npy"), s["oof"].astype(np.float32))
        np.save(os.path.join(PRED, f"{name}_test.npy"), s["test"].astype(np.float32))
        print(f"{name}: OOF={s['oof_auc']:.6f} {gate['reason']} pass={pass_abs}", flush=True)
        if pass_abs or gate["promote"]:
            promoted.append(name)
            out = sub.copy()
            out["is_fraud"] = rankdata(s["test"]) / nte
            path = os.path.join(SUBS, f"stage7_2_{name}_{s['oof_auc']:.5f}.csv")
            out.to_csv(path, index=False)
            results[name]["submission"] = path

    # refresh best for stage 8 if blend wins
    best_name = max(solvers, key=lambda n: solvers[n]["oof_auc"])
    best = solvers[best_name]
    np.save(os.path.join(PRED, "best_7_2_oof.npy"), best["oof"].astype(np.float32))
    np.save(os.path.join(PRED, "best_7_2_test.npy"), best["test"].astype(np.float32))
    # campaign best across 7.1/7.2
    campaign_best_oof, campaign_best_te = best["oof"], best["test"]
    if best71_oof is not None and float(roc_auc_score(y, best71_oof)) > best["oof_auc"]:
        campaign_best_oof, campaign_best_te = best71_oof, best71_te
        best_name = "best_7_1"
    bp = rankdata(campaign_best_oof) / len(y)
    bt = rankdata(campaign_best_te) / nte
    np.save(os.path.join(PRED, "campaign_best_oof.npy"), campaign_best_oof.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_test.npy"), campaign_best_te.astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_oof_logit.npy"), to_logit(bp).astype(np.float32))
    np.save(os.path.join(PRED, "campaign_best_test_logit.npy"), to_logit(bt).astype(np.float32))

    doc = {
        "stage": "7.2_gpu_gated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "top_members": names25,
        "control_oof": ctrl_oof,
        "solvers": results,
        "best_solver": best_name,
        "promoted": promoted,
        "policy": "NO_KAGGLE_SUBMIT",
        "keep_pod_warm": True,
    }
    path = os.path.join(LOGS, "stage7_2_gpu_gated_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(f"wrote {path} best={best_name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
