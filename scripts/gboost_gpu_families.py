#!/usr/bin/env python3
"""Cache exact gboost.py fold matrices (teseed 0,1,2) then train missing GPU families:

  - gb_cat_d2_d3_bag   CatBoost depth 2+3 from GLM margin (CUDA)
  - gb_xgb_cuda_d1_bag XGB max_depth=1 from GLM margin (CUDA)
  - gb_resnet_mlp_bag  PyTorch residual MLP on (RAWV, logit(E), log1p(KCNT)) + z margin

Uses gboost.cache_fold_matrices so TE/gen_encode/GLM run once per teseed.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
# gboost.find() searches INPUT/**/train.csv — point at repo root so data/train.csv resolves
os.environ.setdefault("INPUT_DIR", str(ROOT))
os.environ.setdefault("OUT_DIR", str(ROOT / "cache" / "gboost"))
sys.path.insert(0, str(ROOT / "scripts"))

import gboost as gb  # noqa: E402

CACHE = ROOT / "cache" / "gboost_mats"
PRED = ROOT / "preds" / "part6_expand"
OUT = ROOT / "logs" / "overnight"
TESEEDS = [0, 1, 2]
TARGET = "is_fraud"


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def load_cache(teseed):
    path = gb.cache_fold_matrices(teseed, str(CACHE))
    z = np.load(path)
    idx = np.load(str(path).replace(".npz", "_idx.npz"))
    folds = []
    for f in range(5):
        folds.append({
            "a": idx[f"a{f}"],
            "b": idx[f"b{f}"],
            "za": z[f"f{f}_za"],
            "zb": z[f"f{f}_zb"],
            "zt": z[f"f{f}_zt"],
            "Xa": z[f"f{f}_Xa"],
            "Xb": z[f"f{f}_Xb"],
            "Xt": z[f"f{f}_Xt"],
        })
    full = {"za": z["full_za"], "zt": z["full_zt"], "Xa": z["full_Xa"], "Xt": z["full_Xt"]}
    return folds, full


def arm_cat(y, teseeds=TESEEDS):
    """CatBoost d2+d3 bag across teseeds."""
    from catboost import CatBoostClassifier, Pool

    oof = np.zeros(len(y))
    te_acc = np.zeros(299_984)
    best_iters = []
    n = 0
    for teseed in teseeds:
        folds, full = load_cache(teseed)
        for depth in (2, 3):
            fold_te = np.zeros(299_984)
            for fi, fold in enumerate(folds):
                train_pool = Pool(fold["Xa"], y[fold["a"]], baseline=fold["za"])
                valid_pool = Pool(fold["Xb"], y[fold["b"]], baseline=fold["zb"])
                m = CatBoostClassifier(
                    iterations=12000, learning_rate=0.025, depth=depth, l2_leaf_reg=5.0,
                    task_type="GPU", devices="0", eval_metric="AUC",
                    od_type="Iter", od_wait=300, verbose=False, random_seed=42 + 10 * teseed + depth,
                )
                m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
                bi = int(m.best_iteration_ if m.best_iteration_ is not None else m.tree_count_)
                best_iters.append(bi)
                pb = sigmoid(fold["zb"] + m.predict(fold["Xb"], prediction_type="RawFormulaVal"))
                pt = sigmoid(fold["zt"] + m.predict(fold["Xt"], prediction_type="RawFormulaVal"))
                oof[fold["b"]] += pb
                fold_te += pt
                print(f"[cat] te{teseed} d{depth} fold{fi} auc={roc_auc_score(y[fold['b']], pb):.6f} bi={bi}", flush=True)
            te_acc += fold_te / 5.0
            n += 1
    oof /= n
    te_fold = te_acc / n
    mean_bi = int(np.mean(best_iters)) if best_iters else 500
    ntrees = int(1.25 * mean_bi)
    te_full = np.zeros(299_984)
    nf = 0
    for teseed in teseeds:
        folds, full = load_cache(teseed)
        for depth in (2, 3):
            pool = Pool(full["Xa"], y, baseline=full["za"])
            m = CatBoostClassifier(
                iterations=ntrees, learning_rate=0.025, depth=depth, l2_leaf_reg=5.0,
                task_type="GPU", devices="0", verbose=False, random_seed=99 + teseed + depth,
            )
            m.fit(pool)
            te_full += sigmoid(full["zt"] + m.predict(full["Xt"], prediction_type="RawFormulaVal"))
            nf += 1
    te_full /= nf
    te = 0.65 * te_full + 0.35 * te_fold
    auc = float(roc_auc_score(y, oof))
    print(f"[cat] OOF={auc:.6f} ntrees_refit={ntrees}", flush=True)
    return oof, te, auc


def arm_xgb_d1(y, teseeds=TESEEDS):
    import xgboost as xgb

    oof = np.zeros(len(y))
    te_acc = np.zeros(299_984)
    best_iters = []
    n = 0
    for teseed in teseeds:
        folds, full = load_cache(teseed)
        fold_te = np.zeros(299_984)
        for fi, fold in enumerate(folds):
            m = xgb.XGBClassifier(
                n_estimators=20000, max_depth=1, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.5, tree_method="hist", device="cuda",
                eval_metric="auc", early_stopping_rounds=300, random_state=42 + teseed,
            )
            m.fit(
                fold["Xa"], y[fold["a"]],
                base_margin=fold["za"],
                eval_set=[(fold["Xb"], y[fold["b"]])],
                base_margin_eval_set=[fold["zb"]],
                verbose=False,
            )
            bi = int(m.best_iteration + 1)
            best_iters.append(bi)
            pb = sigmoid(m.predict(fold["Xb"], base_margin=fold["zb"], output_margin=True, iteration_range=(0, bi)))
            pt = sigmoid(m.predict(fold["Xt"], base_margin=fold["zt"], output_margin=True, iteration_range=(0, bi)))
            oof[fold["b"]] += pb
            fold_te += pt
            print(f"[xgb_d1] te{teseed} fold{fi} auc={roc_auc_score(y[fold['b']], pb):.6f} bi={bi}", flush=True)
        te_acc += fold_te / 5.0
        n += 1
    oof /= n
    te_fold = te_acc / n
    mean_bi = int(np.mean(best_iters)) if best_iters else 500
    ntrees = int(1.25 * mean_bi)
    te_full = np.zeros(299_984)
    for teseed in teseeds:
        folds, full = load_cache(teseed)
        m = xgb.XGBClassifier(
            n_estimators=ntrees, max_depth=1, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.5, tree_method="hist", device="cuda", random_state=7 + teseed,
        )
        m.fit(full["Xa"], y, base_margin=full["za"])
        te_full += sigmoid(m.predict(full["Xt"], base_margin=full["zt"], output_margin=True))
    te_full /= len(teseeds)
    te = 0.65 * te_full + 0.35 * te_fold
    auc = float(roc_auc_score(y, oof))
    print(f"[xgb_d1] OOF={auc:.6f} ntrees_refit={ntrees}", flush=True)
    return oof, te, auc


def arm_mlp(y, teseeds=TESEEDS, epochs=40, batch=8192):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlp] device={device}", flush=True)

    class ResMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def prep(X):
        # X already = [RAWV | E | KCNT]; use logit(E) approx via columns after RAW — keep as-is + log1p on trailing KCNT-ish
        # Practical: standardize all; model predicts residual added to z
        return X.astype(np.float32)

    oof = np.zeros(len(y))
    te_acc = np.zeros(299_984)
    n = 0
    for teseed in teseeds:
        folds, full = load_cache(teseed)
        fold_te = np.zeros(299_984)
        for fi, fold in enumerate(folds):
            sc = StandardScaler()
            Xa = sc.fit_transform(prep(fold["Xa"]))
            Xb = sc.transform(prep(fold["Xb"]))
            Xt = sc.transform(prep(fold["Xt"]))
            model = ResMLP(Xa.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()
            ds = TensorDataset(
                torch.as_tensor(Xa), torch.as_tensor(y[fold["a"]].astype(np.float32)),
                torch.as_tensor(fold["za"].astype(np.float32)),
            )
            dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
            model.train()
            for ep in range(epochs):
                for xb, yb, zb in dl:
                    xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
                    opt.zero_grad()
                    # residual on logit scale: z + mlp(x)
                    logits = zb + model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                pb = sigmoid((fold["zb"] + model(torch.as_tensor(Xb).to(device)).cpu().numpy()))
                pt = sigmoid((fold["zt"] + model(torch.as_tensor(Xt).to(device)).cpu().numpy()))
            oof[fold["b"]] += pb
            fold_te += pt
            print(f"[mlp] te{teseed} fold{fi} auc={roc_auc_score(y[fold['b']], pb):.6f}", flush=True)
        te_acc += fold_te / 5.0
        n += 1
    oof /= n
    te_fold = te_acc / n
    # full refit
    te_full = np.zeros(299_984)
    for teseed in teseeds:
        folds, full = load_cache(teseed)
        sc = StandardScaler()
        Xa = sc.fit_transform(prep(full["Xa"]))
        Xt = sc.transform(prep(full["Xt"]))
        model = ResMLP(Xa.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = TensorDataset(
            torch.as_tensor(Xa), torch.as_tensor(y.astype(np.float32)),
            torch.as_tensor(full["za"].astype(np.float32)),
        )
        dl = DataLoader(ds, batch_size=batch, shuffle=True)
        model.train()
        for ep in range(epochs):
            for xb, yb, zb in dl:
                xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
                opt.zero_grad()
                loss = loss_fn(zb + model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            te_full += sigmoid(full["zt"] + model(torch.as_tensor(Xt).to(device)).cpu().numpy())
    te_full /= len(teseeds)
    te = 0.65 * te_full + 0.35 * te_fold
    auc = float(roc_auc_score(y, oof))
    print(f"[mlp] OOF={auc:.6f}", flush=True)
    return oof, te, auc


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    y = gb.y
    print(f"[gpu_families] n_train={len(y)} GEN={gb.GEN} cuda check next", flush=True)

    # 1) cache all teseeds first
    for ts in TESEEDS:
        t0 = time.time()
        gb.cache_fold_matrices(ts, str(CACHE))
        print(f"[cache] teseed={ts} done in {time.time() - t0:.1f}s", flush=True)

    results = {}
    # 2) CatBoost
    try:
        oof, te, auc = arm_cat(y)
        np.save(PRED / "oof_gb_cat_d2_d3_bag.npy", oof)
        np.save(PRED / "sub_gb_cat_d2_d3_bag.npy", te)
        results["gb_cat_d2_d3_bag"] = auc
    except Exception as e:
        print(f"[cat] FAILED: {e}", flush=True)
        results["gb_cat_d2_d3_bag"] = f"FAIL:{e}"

    # 3) XGB d1
    try:
        oof, te, auc = arm_xgb_d1(y)
        np.save(PRED / "oof_gb_xgb_cuda_d1_bag.npy", oof)
        np.save(PRED / "sub_gb_xgb_cuda_d1_bag.npy", te)
        results["gb_xgb_cuda_d1_bag"] = auc
    except Exception as e:
        print(f"[xgb] FAILED: {e}", flush=True)
        results["gb_xgb_cuda_d1_bag"] = f"FAIL:{e}"

    # 4) MLP
    try:
        oof, te, auc = arm_mlp(y)
        np.save(PRED / "oof_gb_resnet_mlp_bag.npy", oof)
        np.save(PRED / "sub_gb_resnet_mlp_bag.npy", te)
        results["gb_resnet_mlp_bag"] = auc
    except Exception as e:
        print(f"[mlp] FAILED: {e}", flush=True)
        results["gb_resnet_mlp_bag"] = f"FAIL:{e}"

    (OUT / "gboost_gpu_families.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    print("[gpu_families] DONE", flush=True)
    return 0


if __name__ == "__main__":
    # Fix double-training bug in first cat draft — arm_cat already clean single pass at end
    sys.exit(main())
