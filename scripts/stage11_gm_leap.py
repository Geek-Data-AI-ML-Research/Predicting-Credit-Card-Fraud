#!/usr/bin/env python3
"""Stage 11 Grandmaster Leap — L1 cross-dig / L2 pseudo / L3 periodic MLP / L4 residual meta.

RULE 0: CUDA batch AUC only; each arm wrapped by caller `timeout 600`.
Auto-submits only if Gate1 (OOF Δ≥+1.5e-4 & ≥4/5 folds) AND Gate2 (Spearman band).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))

from diglead_features import FLOATS, FEATS  # noqa: E402
from diglead_matrix_cache import get_view  # noqa: E402
from unattended_common import (  # noqa: E402
    CTRL_NESTED_HILL,
    LOGS,
    PRED,
    SUBS,
    WS,
    fold_aucs,
    locked_folds,
    promotion_gate_base,
    save_oof_test,
    write_candidate_csv,
)

SEED = 42
EPS = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGE11 = ROOT / "preds" / "stage11"
STAGE11.mkdir(parents=True, exist_ok=True)
(Path(LOGS)).mkdir(parents=True, exist_ok=True)


def _rn(x):
    x = np.asarray(x, dtype=np.float64)
    return (rankdata(x, method="average") - 0.5) / len(x)


def _to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def batch_auc_gpu(cand_mat: torch.Tensor, y_gpu: torch.Tensor) -> torch.Tensor:
    """cand (N,K) → (K,) exact ROC-AUC."""
    pos = y_gpu == 1.0
    n_pos = float(pos.sum().item())
    n_neg = float(len(y_gpu) - n_pos)
    denom = n_pos * n_neg
    offset = n_pos * (n_pos + 1.0) / 2.0
    order = torch.argsort(cand_mat, dim=0)
    ranks = torch.empty_like(cand_mat)
    rank_vals = torch.arange(1, len(y_gpu) + 1, device=cand_mat.device, dtype=torch.float32).unsqueeze(1)
    ranks.scatter_(0, order, rank_vals.expand_as(cand_mat))
    return (ranks[pos].sum(dim=0) - offset) / denom


def load_ctrl(y):
    cref = ROOT / "preds" / "stage7" / "ctrl_nested_hill_oof.npy"
    cte = ROOT / "preds" / "stage7" / "ctrl_nested_hill_test.npy"
    if cref.exists() and cte.exists():
        oof, te = np.load(cref).astype(np.float64), np.load(cte).astype(np.float64)
        return oof, te, float(roc_auc_score(y, oof))
    # fall back: diglead_l3 from bank
    bank = ROOT / "preds" / "stage9" / "full_bank_unpruned.npz"
    z = np.load(bank, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    hits = [i for i, n in enumerate(names) if "lgb_diglead_l3_bag" in n]
    j = hits[0] if hits else 0
    oof, te = z["O"][:, j].astype(np.float64), z["T"][:, j].astype(np.float64)
    print(f"  ctrl fallback bank[{names[j]}] auc={roc_auc_score(y,oof):.6f}", flush=True)
    return oof, te, float(roc_auc_score(y, oof))


def load_member(tag):
    for d in (PRED, ROOT / "preds" / "unattended", Path(WS) / "preds" / "unattended"):
        op, tp = Path(d) / f"oof_{tag}.npy", Path(d) / f"test_{tag}_full.npy"
        if op.exists() and tp.exists():
            return np.load(op).astype(np.float64), np.load(tp).astype(np.float64)
    return None, None


# -------------------- L1: cross dig pairs --------------------
def first2(x):
    i = np.floor(np.abs(np.asarray(x, dtype=np.float64))).astype(np.int64)
    # leading 2 digits via string-free: scale to [10,99]
    i = np.maximum(i, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.floor(np.log10(np.maximum(i, 1))).astype(np.int64) + 1
        lead = (i // np.power(10, np.maximum(L - 2, 0))).astype(np.int64)
    return np.clip(lead, 0, 99)


def frac_zero_flag(x):
    x = np.asarray(x, dtype=np.float64)
    cents = np.round(x * 100) % 100
    return (cents == 0).astype(np.int8)


def decile(x):
    x = np.asarray(x, dtype=np.float64)
    # global decile ranks — frequency on pooled train+test handled by TE fold
    return np.clip((rankdata(x, method="average") / len(x) * 10).astype(np.int8), 0, 9)


def build_cross_keys(df, cols):
    """45 pair keys: lead2_i*100+lead2_j and (frac0_i, decile_j) packed."""
    mats = []
    names = []
    vals = {c: df[c].to_numpy(dtype=np.float64) for c in cols}
    leads = {c: first2(vals[c]) for c in cols}
    f0 = {c: frac_zero_flag(vals[c]) for c in cols}
    dec = {c: decile(vals[c]) for c in cols}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            k1 = leads[a] * 100 + leads[b]
            k2 = f0[a].astype(np.int64) * 10 + dec[b].astype(np.int64)
            mats.append(k1)
            mats.append(k2)
            names.append(f"xd_l2_{a[:8]}_{b[:8]}")
            names.append(f"xd_f0d_{a[:8]}_{b[:8]}")
    return np.column_stack(mats).astype(np.int64), names


def te_smooth(keys_tr, y_tr, keys_others, smooth=30.0):
    prior = float(y_tr.mean())
    s = pd.DataFrame({"k": keys_tr, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    rate = (s["sum"] + prior * smooth) / (s["count"] + smooth)
    fre = s["count"].astype(np.float64)
    out = []
    for k in keys_others:
        out.append(rate.reindex(k).fillna(prior).to_numpy(np.float32))
    freq = []
    for k in keys_others:
        freq.append(np.log1p(fre.reindex(k).fillna(0).to_numpy(np.float32)))
    return out, freq


def arm_l1(view, y, folds, timeout_s=560):
    t0 = time.time()
    print("=== L1_cross_digpairs ===", flush=True)
    cols = [c for c in FLOATS if c in view.train.columns][:10]
    if len(cols) < 6:
        cols = FEATS[:10]
    print(f"  L1 cols={cols}", flush=True)
    Ktr, _ = build_cross_keys(view.train, cols)
    Kte, _ = build_cross_keys(view.test, cols)
    n_keys = Ktr.shape[1]
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    te_xgb = np.zeros(len(view.test))
    te_lgb = np.zeros(len(view.test))
    import lightgbm as lgb
    import xgboost as xgb

    for fi, (tr, va) in enumerate(folds):
        if time.time() - t0 > timeout_s:
            print("  L1 timeout — partial", flush=True)
            break
        Xtr0, Xva0, Xte0 = view.fold_matrix(fi)[2:]
        te_tr_all, te_va_all, te_te_all = [], [], []
        fr_tr_all, fr_va_all, fr_te_all = [], [], []
        for j in range(n_keys):
            tes, frs = te_smooth(Ktr[tr, j], y[tr], [Ktr[tr, j], Ktr[va, j], Kte[:, j]], smooth=30.0)
            te_tr_all.append(tes[0])
            te_va_all.append(tes[1])
            te_te_all.append(tes[2])
            fr_tr_all.append(frs[0])
            fr_va_all.append(frs[1])
            fr_te_all.append(frs[2])
        Xtr = np.hstack([Xtr0, np.column_stack(te_tr_all + fr_tr_all)]).astype(np.float32)
        Xva = np.hstack([Xva0, np.column_stack(te_va_all + fr_va_all)]).astype(np.float32)
        Xte = np.hstack([Xte0, np.column_stack(te_te_all + fr_te_all)]).astype(np.float32)

        # XGB CUDA
        xd = xgb.DMatrix(Xtr, label=y[tr])
        xv = xgb.DMatrix(Xva, label=y[va])
        xt = xgb.DMatrix(Xte)
        params = dict(
            max_leaves=6,
            grow_policy="lossguide",
            max_depth=0,
            eta=0.05,
            subsample=0.8,
            colsample_bytree=0.4,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            seed=SEED + fi,
        )
        bst = xgb.train(
            params,
            xd,
            num_boost_round=400,
            evals=[(xv, "va")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        oof_xgb[va] = bst.predict(xv)
        te_xgb += bst.predict(xt)

        # LGBM
        ds = lgb.Dataset(Xtr, y[tr])
        dv = lgb.Dataset(Xva, y[va], reference=ds)
        params_l = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            num_leaves=5,
            feature_fraction=0.5,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
            seed=SEED + fi,
            num_threads=int(os.environ.get("LGBM_NUM_THREADS", "-1")),
        )
        bo = lgb.train(
            params_l,
            ds,
            num_boost_round=500,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        oof_lgb[va] = bo.predict(Xva)
        te_lgb += bo.predict(Xte)
        print(
            f"  L1 fold{fi} xgb={roc_auc_score(y[va], oof_xgb[va]):.6f} "
            f"lgb={roc_auc_score(y[va], oof_lgb[va]):.6f} X={Xtr.shape}",
            flush=True,
        )

    te_xgb /= 5.0
    te_lgb /= 5.0
    # blend
    oof = 0.55 * oof_xgb + 0.45 * oof_lgb
    te = 0.55 * te_xgb + 0.45 * te_lgb
    auc = float(roc_auc_score(y, oof))
    np.save(STAGE11 / "oof_L1_cross_digpairs.npy", oof.astype(np.float32))
    np.save(STAGE11 / "sub_L1_cross_digpairs.npy", te.astype(np.float32))
    save_oof_test("L1_cross_digpairs", oof, te)
    print(f"L1 DONE OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    return oof, te, auc


# -------------------- L2: pseudo-label --------------------
def arm_l2(view, y, folds, p_lb_test, timeout_s=560):
    t0 = time.time()
    print("=== L2_pseudo_transductive ===", flush=True)
    import lightgbm as lgb
    import xgboost as xgb

    pt = np.asarray(p_lb_test, dtype=np.float64)
    # tails
    thr_lo = np.quantile(pt, 0.25)
    thr_hi = np.quantile(pt, 0.988)
    neg_idx = np.where(pt <= thr_lo)[0]
    pos_idx = np.where(pt >= thr_hi)[0]
    print(f"  pseudo neg={len(neg_idx)} (p<={thr_lo:.6f}) pos={len(pos_idx)} (p>={thr_hi:.6f})", flush=True)

    oof = np.zeros(len(y))
    te_sum = np.zeros(len(view.test))
    for fi, (tr, va) in enumerate(folds):
        if time.time() - t0 > timeout_s:
            break
        # rebuild TE with pseudo: use fold_matrix then append soft — actually rebuild via cache base + TE smooth40 on tr+pseudo
        _, _, Xtr, Xva, Xte = view.fold_matrix(fi)
        # construct pseudo rows from test matrix with labels
        Xpseudo = Xte[np.concatenate([neg_idx, pos_idx])]
        ypseudo = np.concatenate([np.zeros(len(neg_idx)), np.ones(len(pos_idx))])
        wpseudo = np.concatenate([np.full(len(neg_idx), 0.5), np.full(len(pos_idx), 0.7)])
        Xtr_p = np.vstack([Xtr, Xpseudo])
        ytr_p = np.concatenate([y[tr], ypseudo])
        wtr_p = np.concatenate([np.ones(len(tr)), wpseudo])

        xd = xgb.DMatrix(Xtr_p, label=ytr_p, weight=wtr_p)
        xv = xgb.DMatrix(Xva, label=y[va])
        xt = xgb.DMatrix(Xte)
        params = dict(
            max_leaves=4,
            grow_policy="lossguide",
            max_depth=0,
            eta=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            seed=SEED + fi,
        )
        bst = xgb.train(
            params, xd, num_boost_round=350, evals=[(xv, "va")], early_stopping_rounds=40, verbose_eval=False
        )
        px = bst.predict(xv)
        pxt = bst.predict(xt)

        ds = lgb.Dataset(Xtr_p, ytr_p, weight=wtr_p)
        dv = lgb.Dataset(Xva, y[va], reference=ds)
        bo = lgb.train(
            dict(
                objective="binary",
                metric="auc",
                learning_rate=0.05,
                num_leaves=4,
                feature_fraction=0.5,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbose=-1,
                seed=SEED + fi,
            ),
            ds,
            num_boost_round=400,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pl = bo.predict(Xva)
        plt = bo.predict(Xte)
        oof[va] = 0.55 * px + 0.45 * pl
        te_sum += 0.55 * pxt + 0.45 * plt
        print(f"  L2 fold{fi} auc={roc_auc_score(y[va], oof[va]):.6f}", flush=True)

    te = te_sum / 5.0
    auc = float(roc_auc_score(y, oof))
    np.save(STAGE11 / "oof_L2_pseudo.npy", oof.astype(np.float32))
    np.save(STAGE11 / "sub_L2_pseudo.npy", te.astype(np.float32))
    save_oof_test("L2_pseudo", oof, te)
    print(f"L2 DONE OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    return oof, te, auc


# -------------------- L3: periodic residual MLP --------------------
class PeriodicEmb(nn.Module):
    def __init__(self, n_cont, n_freq=8):
        super().__init__()
        self.n_freq = n_freq
        self.w = nn.Parameter(torch.randn(n_cont, n_freq) * 0.1)

    def forward(self, x):
        # x (B, C) → (B, C*2*n_freq)
        ang = 2 * np.pi * x.unsqueeze(-1) * self.w  # (B,C,F)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(1)


class ResidualMLP(nn.Module):
    def __init__(self, n_cont, n_tok=0, n_tok_card=32, hid=256):
        super().__init__()
        self.per = PeriodicEmb(n_cont, n_freq=8)
        in_dim = n_cont * 16 + (n_tok * 8 if n_tok else 0)
        self.tok = nn.Embedding(n_tok_card, 8) if n_tok else None
        self.n_tok = n_tok
        blocks = []
        d = in_dim
        for _ in range(3):
            blocks += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.SiLU(), nn.Dropout(0.25)]
            d = hid
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hid, 1)

    def forward(self, x_cont, x_tok=None):
        h = self.per(x_cont)
        if self.tok is not None and x_tok is not None:
            h = torch.cat([h, self.tok(x_tok.clamp(0, self.tok.num_embeddings - 1)).flatten(1)], dim=1)
        return self.head(self.net(h)).squeeze(-1)


def arm_l3(view, y, folds, ctrl_oof, ctrl_te, timeout_s=560):
    t0 = time.time()
    print("=== L3_periodic_mlp ===", flush=True)
    oof = np.zeros(len(y))
    te_sum = np.zeros(len(view.test))
    for fi, (tr, va) in enumerate(folds):
        if time.time() - t0 > timeout_s:
            break
        _, _, Xtr, Xva, Xte = view.fold_matrix(fi)
        # standardize cont on tr
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr_n = ((Xtr - mu) / sd).astype(np.float32)
        Xva_n = ((Xva - mu) / sd).astype(np.float32)
        Xte_n = ((Xte - mu) / sd).astype(np.float32)
        # ctrl logits
        lc_tr = _to_logit(ctrl_oof[tr]).astype(np.float32)
        lc_va = _to_logit(ctrl_oof[va]).astype(np.float32)
        lc_te = _to_logit(ctrl_te).astype(np.float32)

        model = ResidualMLP(n_cont=Xtr_n.shape[1], hid=192).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        steps = 18 * max(1, len(tr) // 4096)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=max(steps, 18))
        loss_fn = nn.BCEWithLogitsLoss()
        model.train()
        rng = np.random.default_rng(SEED + fi)
        for ep in range(18):
            idx = rng.permutation(len(tr))
            for s in range(0, len(tr), 4096):
                b = idx[s : s + 4096]
                xc = torch.as_tensor(Xtr_n[b], device=DEVICE)
                yt = torch.as_tensor(y[tr][b], device=DEVICE, dtype=torch.float32)
                base = torch.as_tensor(lc_tr[b], device=DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
                    delta = model(xc)
                    logits = base.detach() + 0.2 * delta
                    loss = loss_fn(logits, yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        with torch.no_grad():
            def pred(Xn, lc):
                outs = []
                for s in range(0, len(Xn), 8192):
                    xc = torch.as_tensor(Xn[s : s + 8192], device=DEVICE)
                    base = torch.as_tensor(lc[s : s + 8192], device=DEVICE)
                    outs.append(expit((base + 0.2 * model(xc)).float().cpu().numpy()))
                return np.concatenate(outs)

            oof[va] = pred(Xva_n, lc_va)
            te_sum += pred(Xte_n, lc_te)
        print(f"  L3 fold{fi} auc={roc_auc_score(y[va], oof[va]):.6f}", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    te = te_sum / 5.0
    auc = float(roc_auc_score(y, oof))
    np.save(STAGE11 / "oof_L3_periodic_mlp.npy", oof.astype(np.float32))
    np.save(STAGE11 / "sub_L3_periodic_mlp.npy", te.astype(np.float32))
    save_oof_test("L3_periodic_mlp", oof, te)
    print(f"L3 DONE OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    return oof, te, auc


# -------------------- L4: residual GBDT meta + anchored hill --------------------
def arm_l4(y, folds, members: dict, ctrl_oof, ctrl_te, view, timeout_s=560):
    t0 = time.time()
    print("=== L4_residual_gbdt_meta ===", flush=True)
    import lightgbm as lgb

    names = list(members.keys())
    O = np.column_stack([members[n][0] for n in names])
    T = np.column_stack([members[n][1] for n in names])
    L = _to_logit(O)
    Lt = _to_logit(T)
    # pairwise diffs vs ctrl
    cidx = names.index("ctrl") if "ctrl" in names else 0
    diffs = []
    diffs_t = []
    for j, n in enumerate(names):
        if n in ("L1_cross_digpairs", "L3_periodic_mlp", "t01_lgb_diglead_super"):
            diffs.append(L[:, j] - L[:, cidx])
            diffs_t.append(Lt[:, j] - Lt[:, cidx])
    D = np.column_stack(diffs) if diffs else np.zeros((len(y), 1))
    Dt = np.column_stack(diffs_t) if diffs_t else np.zeros((T.shape[0], 1))
    row_std = L.std(1, keepdims=True)
    row_rng = (L.max(1) - L.min(1)).reshape(-1, 1)
    row_std_t = Lt.std(1, keepdims=True)
    row_rng_t = (Lt.max(1) - Lt.min(1)).reshape(-1, 1)
    # regime: first 8 cols of diglead raw floats if available
    reg = view.train[FLOATS[:6]].to_numpy(np.float32)
    reg_t = view.test[FLOATS[:6]].to_numpy(np.float32)
    reg = np.nan_to_num(reg)
    reg_t = np.nan_to_num(reg_t)

    oof = np.zeros(len(y))
    te_sum = np.zeros(T.shape[0])
    for fi, (tr, va) in enumerate(folds):
        if time.time() - t0 > timeout_s:
            break
        Xtr = np.hstack([L[tr], D[tr], row_std[tr], row_rng[tr], reg[tr]]).astype(np.float32)
        Xva = np.hstack([L[va], D[va], row_std[va], row_rng[va], reg[va]]).astype(np.float32)
        Xte = np.hstack([Lt, Dt, row_std_t, row_rng_t, reg_t]).astype(np.float32)
        init_tr = _to_logit(ctrl_oof[tr])
        init_va = _to_logit(ctrl_oof[va])
        init_te = _to_logit(ctrl_te)
        ds = lgb.Dataset(Xtr, y[tr], init_score=init_tr)
        dv = lgb.Dataset(Xva, y[va], init_score=init_va, reference=ds)
        bo = lgb.train(
            dict(
                objective="binary",
                metric="auc",
                learning_rate=0.015,
                max_depth=2,
                num_leaves=3,
                min_child_samples=2000,
                feature_fraction=0.6,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbose=-1,
                seed=SEED + fi,
            ),
            ds,
            num_boost_round=800,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        # residual on logit: predict returns probability around init
        oof[va] = bo.predict(Xva, raw_score=False)
        te_sum += bo.predict(Xte, raw_score=False)
        print(f"  L4 fold{fi} auc={roc_auc_score(y[va], oof[va]):.6f}", flush=True)

    te = te_sum / 5.0
    auc = float(roc_auc_score(y, oof))
    np.save(STAGE11 / "oof_L4_residual_gbdt_meta.npy", oof.astype(np.float32))
    np.save(STAGE11 / "sub_L4_residual_gbdt_meta.npy", te.astype(np.float32))
    save_oof_test("L4_residual_gbdt_meta", oof, te)
    print(f"L4 DONE OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)

    # GPU anchored hill over {ctrl, T01.., L1,L2,L3,L4}
    print("=== L4b anchored GPU hill ===", flush=True)
    hill_names = list(members.keys()) + ["L4"]
    Oh = np.column_stack([O] + [oof])
    Th = np.column_stack([T] + [te])
    oof_h, te_h = anchored_hill(Oh, Th, y, folds, start_j=cidx, names=hill_names)
    auc_h = float(roc_auc_score(y, oof_h))
    np.save(STAGE11 / "oof_L4_hill_blend.npy", oof_h.astype(np.float32))
    np.save(STAGE11 / "sub_L4_hill_blend.npy", te_h.astype(np.float32))
    print(f"L4hill DONE OOF={auc_h:.6f}", flush=True)
    return oof_h, te_h, auc_h


def anchored_hill(O, T, y, folds, start_j=0, names=None):
    ntr, m = O.shape
    oof = np.zeros(ntr)
    te_sum = np.zeros(T.shape[0])
    names = names or [f"m{i}" for i in range(m)]
    for fi, (tr, va) in enumerate(folds):
        Ltr = _to_logit(O[tr])
        mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
        Ztr = ((Ltr - mu) / sd).astype(np.float32)
        Zva = ((_to_logit(O[va]) - mu) / sd).astype(np.float32)
        Zte = ((_to_logit(T) - mu) / sd).astype(np.float32)
        y_gpu = torch.as_tensor(y[tr], device=DEVICE, dtype=torch.float32)
        Zg = torch.as_tensor(Ztr, device=DEVICE)
        w = np.zeros(m)
        w[start_j] = 1.0
        cur = float(batch_auc_gpu((Zg @ torch.as_tensor(w, device=DEVICE, dtype=torch.float32)).unsqueeze(1), y_gpu)[0])
        for _ in range(40):
            cands = []
            for j in range(m):
                w2 = w.copy()
                w2[j] += 1.0
                ww = w2 / w2.sum()
                cands.append(Zg @ torch.as_tensor(ww, device=DEVICE, dtype=torch.float32))
            aucs = batch_auc_gpu(torch.stack(cands, dim=1), y_gpu)
            bj = int(torch.argmax(aucs).item())
            bs = float(aucs[bj].item())
            if bs > cur + 1e-8:
                w2 = w.copy()
                w2[bj] += 1.0
                cur, w = bs, w2
            else:
                break
        ww = w / w.sum()
        s_va = Zva @ ww
        s_te = Zte @ ww
        oof[va] = _rn(s_va)
        te_sum += _rn(s_te)
        print(f"  hill fold{fi} auc={roc_auc_score(y[va], s_va):.6f} w={[(names[j], float(ww[j])) for j in range(m) if ww[j]>1e-9]}", flush=True)
    return oof, te_sum / 5.0


def main():
    t_all = time.time()
    print(f"STAGE11 start device={DEVICE}", flush=True)
    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_te, ctrl_auc = load_ctrl(y)
    print(f"ctrl OOF={ctrl_auc:.6f} (target≈{CTRL_NESTED_HILL})", flush=True)

    # P_lb for L2 / gate2
    plb_path = ROOT / "submissions" / "hillclimb_mix_0.95111.csv"
    if not plb_path.exists():
        plb_path = Path(SUBS) / "hillclimb_mix_0.95111.csv"
    P_lb = pd.read_csv(plb_path)["is_fraud"].to_numpy(np.float64)

    only = os.environ.get("STAGE11_ONLY", "L1,L2,L3,L4,GATE")
    arms = set(only.split(","))

    l1 = l2 = l3 = None
    if "L1" in arms:
        l1 = arm_l1(view, y, folds)
    if "L2" in arms:
        l2 = arm_l2(view, y, folds, P_lb)
    if "L3" in arms:
        l3 = arm_l3(view, y, folds, ctrl_oof, ctrl_te)

    members = {"ctrl": (ctrl_oof, ctrl_te)}
    for tag in (
        "t01_lgb_diglead_super",
        "t02_xgb_diglead_super",
        "t04_inter_stumps",
        "t05_diglead_glm",
        "t06_orig_aug_diglead",
        "t07_tail_xgb",
        "t08_nested_hill_aligned",
        "t08_anchored_superbase",
    ):
        o, t = load_member(tag)
        if o is not None:
            members[tag] = (o, t)
    if l1:
        members["L1_cross_digpairs"] = (l1[0], l1[1])
    if l2:
        members["L2_pseudo"] = (l2[0], l2[1])
    if l3:
        members["L3_periodic_mlp"] = (l3[0], l3[1])

    final_oof, final_te, final_auc = ctrl_oof, ctrl_te, ctrl_auc
    if "L4" in arms:
        final_oof, final_te, final_auc = arm_l4(y, folds, members, ctrl_oof, ctrl_te, view)

    if "GATE" in arms:
        folds_a = fold_aucs(y, final_oof, folds)
        ctrl_folds = fold_aucs(y, ctrl_oof, folds)
        gate = promotion_gate_base(ctrl_folds, folds_a, ctrl_auc, final_auc)
        # stricter stage11
        delta = final_auc - ctrl_auc
        fold_wins = sum(1 for a, b in zip(folds_a, ctrl_folds) if a > b + 1e-12)
        g1 = delta >= 0.00015 and fold_wins >= 4
        # map te to ranks for spearman vs P_lb
        te_rank = _rn(final_te)
        sp = float(spearmanr(te_rank, P_lb).correlation)
        g2 = 0.9970 <= sp <= 0.9996
        doc = {
            "final_auc": final_auc,
            "ctrl_auc": ctrl_auc,
            "delta": delta,
            "fold_aucs": folds_a,
            "fold_wins": fold_wins,
            "spearman_vs_hillclimb_mix": sp,
            "gate1": g1,
            "gate2": g2,
            "promote": bool(g1 and g2),
            "gate_detail": gate,
        }
        print("=== STAGE11 GATE ===", flush=True)
        print(json.dumps(doc, indent=2), flush=True)
        path = write_candidate_csv(f"sub_stage11_gm_leap_cv_{final_auc:.5f}.csv", final_te)
        doc["submission"] = path
        json.dump(doc, open(Path(LOGS) / "stage11_gm_leap_result.json", "w"), indent=2)
        if g1 and g2:
            print("GATES PASSED — auto-submit", flush=True)
            os.system(
                f'kaggle competitions submit -c playground-series-s6e11 -f "{path}" '
                f'-m "Stage11 GM Leap L1_CrossDig L2_Pseudo L3_PeriodicMLP L4_ResMeta CV={final_auc:.5f}"'
            )
            # poll LB
            for _ in range(40):
                time.sleep(30)
                os.system(
                    "kaggle competitions submissions -c playground-series-s6e11 2>/dev/null | head -8 "
                    f"| tee -a {LOGS}/stage11_lb_result.txt"
                )
                # stop if score present in latest
                if Path(LOGS).joinpath("stage11_lb_result.txt").exists():
                    txt = open(Path(LOGS) / "stage11_lb_result.txt").read()
                    if "sub_stage11_gm_leap" in txt and "0.95" in txt:
                        break
        else:
            print("GATES FAILED — no submit", flush=True)

    print(f"STAGE11 ALL DONE ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
