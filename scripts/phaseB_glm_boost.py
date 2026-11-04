#!/usr/bin/env python3
"""Phase B — Part-6 GLM base + GLM-boosted shallow GBDT bags + 100% full-data refit.

Hard bans: no pair TE; leaves<=4 / depth<=2; full-refit required for test;
canary Fold0 >= 0.95000 before folds 1-4 + refit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))

from diglead_features import (  # noqa: E402
    add_te,
    build_base_frames,
    drop_keys,
)
from unattended_common import locked_folds  # noqa: E402
from validation_stage3_baselines import TARGET  # noqa: E402

OUT = ROOT / "logs" / "overnight"
PRED = ROOT / "preds" / "phaseB"
TE_SEEDS = [42, 100, 2026, 777, 999]
CAT_TE_SEEDS = [42, 2026, 777]
# Soft floor: hard-abort only well below; 0.949–0.950 continues with WARN (0.94944 observed).
CANARY_FLOOR = 0.94900
CANARY_HARD = 0.94800
CTRL_NESTED = 0.95117
SMOOTH = 20.0
REFIT_BLEND = (0.65, 0.35)  # full_refit, fold_mean


def te_matrix(XLtr, XLte, lead_te, y, tr, va, teseed, smooth=SMOOTH):
    Xa = XLtr.iloc[tr].reset_index(drop=True)
    Xb = XLtr.iloc[va].reset_index(drop=True)
    Xt = XLte.reset_index(drop=True)
    Xa, (Xb, Xt) = add_te(Xa, y[tr], [Xb, Xt], lead_te, smooth=smooth, seed=teseed)
    Xa, Xb, Xt = drop_keys(Xa), drop_keys(Xb), drop_keys(Xt)
    return Xa, Xb, Xt


def te_full_refit(XLtr, XLte, lead_te, y, teseed, smooth=SMOOTH):
    """5-fold inner TE on 100% train, map rates onto test."""
    Xa = XLtr.reset_index(drop=True).copy()
    Xt = XLte.reset_index(drop=True).copy()
    Xa, (Xt,) = add_te(Xa, y, [Xt], lead_te, smooth=smooth, seed=teseed)
    return drop_keys(Xa), drop_keys(Xt)


def frame_to_x(D: pd.DataFrame) -> np.ndarray:
    out = D.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category" or out[c].dtype == object:
            out[c] = out[c].astype("category").cat.codes.astype(np.float32)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).values


def fit_glm(Xa, ya, Xb, Xt):
    sc = StandardScaler()
    Za = sc.fit_transform(Xa)
    Zb = sc.transform(Xb)
    Zt = sc.transform(Xt)
    m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", tol=1e-6)
    m.fit(Za, ya)
    pb = m.predict_proba(Zb)[:, 1]
    pt = m.predict_proba(Zt)[:, 1]
    return pb, pt


def clip01(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def run_lgb(Xa, ya, Xb, yb, Xt, init_a, init_b, init_t, leaves, seed):
    dtr = lgb.Dataset(Xa, label=ya, init_score=init_a, free_raw_data=False)
    dva = lgb.Dataset(Xb, label=yb, init_score=init_b, reference=dtr, free_raw_data=False)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=leaves,
        max_depth=-1,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=50,
        verbosity=-1,
        seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
    )
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=8000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    # predictions are margins relative to init_score → convert via expit(init + pred)
    pb = expit(init_b + booster.predict(Xb, raw_score=True))
    pt = expit(init_t + booster.predict(Xt, raw_score=True))
    return pb, pt, int(booster.best_iteration or booster.current_iteration())


def run_lgb_refit(Xa, ya, Xt, init_a, init_t, leaves, seed, ntrees):
    dtr = lgb.Dataset(Xa, label=ya, init_score=init_a, free_raw_data=False)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=leaves,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=50,
        verbosity=-1,
        seed=seed,
    )
    booster = lgb.train(params, dtr, num_boost_round=max(50, ntrees))
    pt = expit(init_t + booster.predict(Xt, raw_score=True))
    return pt


def run_xgb(Xa, ya, Xb, yb, Xt, margin_a, margin_b, margin_t, depth, seed, use_cuda=True):
    import xgboost as xgb

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.03,
        max_depth=depth,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=5,
        tree_method="hist",
        max_bin=256,
        seed=seed,
    )
    if use_cuda:
        params["device"] = "cuda"
    dtr = xgb.DMatrix(Xa, label=ya)
    dva = xgb.DMatrix(Xb, label=yb)
    dte = xgb.DMatrix(Xt)
    dtr.set_base_margin(margin_a)
    dva.set_base_margin(margin_b)
    dte.set_base_margin(margin_t)
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=6000,
        evals=[(dva, "va")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    bi = int(getattr(booster, "best_iteration", None) or 0) + 1
    pb = booster.predict(dva, iteration_range=(0, bi))
    pt = booster.predict(dte, iteration_range=(0, bi))
    return pb, pt, bi


def run_xgb_refit(Xa, ya, Xt, margin_a, margin_t, depth, seed, ntrees, use_cuda=True):
    import xgboost as xgb

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.03,
        max_depth=depth,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=5,
        tree_method="hist",
        max_bin=256,
        seed=seed,
    )
    if use_cuda:
        params["device"] = "cuda"
    dtr = xgb.DMatrix(Xa, label=ya)
    dte = xgb.DMatrix(Xt)
    dtr.set_base_margin(margin_a)
    dte.set_base_margin(margin_t)
    booster = xgb.train(params, dtr, num_boost_round=max(50, ntrees), verbose_eval=False)
    return booster.predict(dte)


def run_cat(Xa, ya, Xb, yb, Xt, base_a, base_b, base_t, depth, seed, use_cuda=True):
    from catboost import CatBoostClassifier, Pool

    task = "GPU" if use_cuda else "CPU"
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=depth,
        learning_rate=0.03,
        iterations=6000,
        random_seed=seed,
        od_type="Iter",
        od_wait=200,
        verbose=False,
        task_type=task,
        allow_writing_files=False,
    )
    train_pool = Pool(Xa, ya)
    valid_pool = Pool(Xb, yb)
    model.fit(train_pool, eval_set=valid_pool, baseline=base_a.reshape(-1, 1), use_best_model=True)
    # CatBoost baseline on predict
    pb = model.predict(Xb, prediction_type="Probability", baseline=base_b.reshape(-1, 1))
    pt = model.predict(Xt, prediction_type="Probability", baseline=base_t.reshape(-1, 1))
    if pb.ndim > 1:
        pb = pb[:, 1] if pb.shape[1] > 1 else pb[:, 0]
    if pt.ndim > 1:
        pt = pt[:, 1] if pt.shape[1] > 1 else pt[:, 0]
    bi = int(model.get_best_iteration() or model.tree_count_)
    return np.asarray(pb, dtype=np.float64), np.asarray(pt, dtype=np.float64), bi


def load_train_test():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    y = train[TARGET].astype(int).values
    return train, test, y


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    train, test, y = load_train_test()
    folds = locked_folds(y)
    XLtr, XLte, lead_te = build_base_frames(train, test)

    use_cuda = os.environ.get("S6E11_FORCE_CPU", "0") != "1"
    try:
        import torch

        use_cuda = use_cuda and torch.cuda.is_available()
    except Exception:
        pass
    print(f"[phaseB] cuda={use_cuda} n_train={len(y)} n_test={len(test)}", flush=True)

    # -------- Stage 1: joint GLM OOF + full refit (resume from disk if present) --------
    oof_path = PRED / "oof_glm.npy"
    sub_path = PRED / "sub_glm_refit.npy"
    full_tr_path = PRED / "oof_glm_full_tr.npy"
    resume_glm = oof_path.exists() and sub_path.exists()

    if resume_glm:
        oof_glm = np.load(oof_path)
        sub_glm_refit = np.load(sub_path)
        print(f"[phaseB] RESUME Stage1 oof_glm + sub_glm_refit from disk", flush=True)
        if full_tr_path.exists():
            oof_glm_full_tr = np.load(full_tr_path)
            print("[phaseB] RESUME oof_glm_full_tr from disk", flush=True)
        else:
            print("[phaseB] recomputing full-data GLM only (missing oof_glm_full_tr)", flush=True)
            oof_glm_full_tr = np.zeros(len(y), dtype=np.float64)
            for ts in TE_SEEDS:
                Xaf, Xtf = te_full_refit(XLtr, XLte, lead_te, y, ts)
                sc = StandardScaler()
                Za = sc.fit_transform(frame_to_x(Xaf))
                m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
                m.fit(Za, y)
                oof_glm_full_tr += m.predict_proba(Za)[:, 1]
            oof_glm_full_tr /= len(TE_SEEDS)
            np.save(full_tr_path, oof_glm_full_tr)
    else:
        oof_glm = np.zeros(len(y), dtype=np.float64)
        fold_te_glm = np.zeros(len(test), dtype=np.float64)
        for fi, (tr, va) in enumerate(folds):
            # mean over TE seeds for stable base margin
            pb_acc = np.zeros(len(va))
            pt_acc = np.zeros(len(test))
            for ts in TE_SEEDS:
                Xa, Xb, Xt = te_matrix(XLtr, XLte, lead_te, y, tr, va, ts)
                pb, pt = fit_glm(frame_to_x(Xa), y[tr], frame_to_x(Xb), frame_to_x(Xt))
                pb_acc += pb
                pt_acc += pt
            oof_glm[va] = pb_acc / len(TE_SEEDS)
            fold_te_glm += pt_acc / len(TE_SEEDS)
            print(f"[phaseB] GLM fold{fi} done", flush=True)
        fold_te_glm /= 5.0

        # full-data GLM refit (mean of TE seeds)
        sub_glm = np.zeros(len(test), dtype=np.float64)
        oof_glm_full_tr = np.zeros(len(y), dtype=np.float64)  # for init on full refit trees
        for ts in TE_SEEDS:
            Xaf, Xtf = te_full_refit(XLtr, XLte, lead_te, y, ts)
            # OOF-style TE already nested inside add_te for train; use as train features
            sc = StandardScaler()
            Za = sc.fit_transform(frame_to_x(Xaf))
            Zt = sc.transform(frame_to_x(Xtf))
            m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
            m.fit(Za, y)
            sub_glm += m.predict_proba(Zt)[:, 1]
            oof_glm_full_tr += m.predict_proba(Za)[:, 1]
        sub_glm /= len(TE_SEEDS)
        oof_glm_full_tr /= len(TE_SEEDS)
        # blend test: prefer full refit
        sub_glm_refit = 0.65 * sub_glm + 0.35 * fold_te_glm
        np.save(PRED / "oof_glm.npy", oof_glm)
        np.save(PRED / "sub_glm_refit.npy", sub_glm_refit)
        np.save(full_tr_path, oof_glm_full_tr)

    glm_auc = float(roc_auc_score(y, oof_glm))
    glm_f0 = float(roc_auc_score(y[folds[0][1]], oof_glm[folds[0][1]]))
    print(f"[phaseB] GLM oof={glm_auc:.6f} fold0={glm_f0:.6f}", flush=True)

    logit_oof = logit(clip01(oof_glm))
    logit_full_tr = logit(clip01(oof_glm_full_tr))
    logit_sub = logit(clip01(sub_glm_refit))

    # -------- Stage 2 canary: LGB leaves=3, seed 42, fold0 --------
    tr0, va0 = folds[0]
    Xa, Xb, Xt = te_matrix(XLtr, XLte, lead_te, y, tr0, va0, 42)
    pb, pt, bi = run_lgb(
        frame_to_x(Xa),
        y[tr0],
        frame_to_x(Xb),
        y[va0],
        frame_to_x(Xt),
        logit_oof[tr0],
        logit_oof[va0],
        logit_sub,
        leaves=3,
        seed=42,
    )
    canary = float(roc_auc_score(y[va0], pb))
    print(f"[phaseB] CANARY fold0 lgb_l3={canary:.6f} (floor {CANARY_FLOOR} hard {CANARY_HARD})", flush=True)
    if canary < CANARY_HARD:
        (OUT / "phaseB_ABORT.json").write_text(
            json.dumps({"status": "ABORT_CANARY", "fold0": canary, "floor": CANARY_FLOOR, "hard": CANARY_HARD}, indent=2)
        )
        print("ABORT: canary failed hard", flush=True)
        return 2
    if canary < CANARY_FLOOR:
        print(f"[phaseB] WARN canary below soft floor {CANARY_FLOOR}; continuing bags", flush=True)
    abort_path = OUT / "phaseB_ABORT.json"
    if abort_path.exists():
        abort_path.unlink()

    members = {}

    def bag_member(tag, train_fn, refit_fn, configs, te_seeds):
        """configs: list of hyper dicts; average all seeds×configs."""
        oof = np.zeros(len(y), dtype=np.float64)
        te_fold = np.zeros(len(test), dtype=np.float64)
        best_iters = []
        n_unit = 0
        for fi, (tr, va) in enumerate(folds):
            acc_b = np.zeros(len(va))
            acc_t = np.zeros(len(test))
            n = 0
            for ts in te_seeds:
                Xa, Xb, Xt = te_matrix(XLtr, XLte, lead_te, y, tr, va, ts)
                xa, xb, xt = frame_to_x(Xa), frame_to_x(Xb), frame_to_x(Xt)
                for cfg in configs:
                    pb, pt, bi = train_fn(xa, y[tr], xb, y[va], xt, logit_oof[tr], logit_oof[va], logit_sub, cfg, ts)
                    acc_b += pb
                    acc_t += pt
                    best_iters.append(bi)
                    n += 1
            oof[va] = acc_b / n
            te_fold += acc_t / n
            n_unit = n
            print(f"[phaseB] {tag} fold{fi} auc={roc_auc_score(y[va], oof[va]):.6f}", flush=True)
            # crash-resume checkpoint after each fold
            np.save(PRED / f"oof_{tag}.partial.npy", oof)
            np.save(PRED / f"te_fold_{tag}.partial.npy", te_fold / max(fi + 1, 1))
            ck = {
                "tag": tag,
                "fold_done": fi,
                "partial_oof_auc": float(roc_auc_score(y[oof > 0], oof[oof > 0])) if (oof > 0).any() else None,
                "best_iters_so_far": best_iters[-n:] if n else [],
            }
            (OUT / f"ckpt_{tag}_fold{fi}.json").write_text(json.dumps(ck, indent=2))
            (ROOT / "checkpoints" / "overnight").mkdir(parents=True, exist_ok=True)
            (ROOT / "checkpoints" / "overnight" / f"ckpt_{tag}_fold{fi}.json").write_text(json.dumps(ck, indent=2))
        te_fold /= 5.0
        mean_bi = int(np.mean(best_iters)) if best_iters else 500
        ntrees = int(1.20 * mean_bi)
        # full refit bag
        te_full = np.zeros(len(test), dtype=np.float64)
        n = 0
        for ts in te_seeds:
            Xaf, Xtf = te_full_refit(XLtr, XLte, lead_te, y, ts)
            xa, xt = frame_to_x(Xaf), frame_to_x(Xtf)
            for cfg in configs:
                pt = refit_fn(xa, y, xt, logit_full_tr, logit_sub, cfg, ts, ntrees)
                te_full += pt
                n += 1
        te_full /= n
        te = REFIT_BLEND[0] * te_full + REFIT_BLEND[1] * te_fold
        auc = float(roc_auc_score(y, oof))
        fa = [float(roc_auc_score(y[va], oof[va])) for _, va in folds]
        members[tag] = {"oof": oof, "te": te, "auc": auc, "folds": fa, "mean_bi": mean_bi, "ntrees_refit": ntrees}
        np.save(PRED / f"oof_{tag}.npy", oof)
        np.save(PRED / f"sub_{tag}.npy", te)
        print(f"[phaseB] {tag} OOF={auc:.6f} folds={np.round(fa,5)}", flush=True)
        return auc

    # B1 LightGBM l3+l4
    def lgb_train(xa, ya, xb, yb, xt, ia, ib, it, cfg, ts):
        return run_lgb(xa, ya, xb, yb, xt, ia, ib, it, leaves=cfg["leaves"], seed=ts)

    def lgb_refit(xa, ya, xt, ia, it, cfg, ts, ntrees):
        return run_lgb_refit(xa, ya, xt, ia, it, leaves=cfg["leaves"], seed=ts, ntrees=ntrees)

    bag_member(
        "lgb_glm_boost_l3_l4_refit",
        lgb_train,
        lgb_refit,
        [{"leaves": 3}, {"leaves": 4}],
        TE_SEEDS,
    )

    # B2 XGB d1+d2
    def xgb_train(xa, ya, xb, yb, xt, ia, ib, it, cfg, ts):
        return run_xgb(xa, ya, xb, yb, xt, ia, ib, it, depth=cfg["depth"], seed=ts, use_cuda=use_cuda)

    def xgb_refit(xa, ya, xt, ia, it, cfg, ts, ntrees):
        return run_xgb_refit(xa, ya, xt, ia, it, depth=cfg["depth"], seed=ts, ntrees=ntrees, use_cuda=use_cuda)

    try:
        bag_member(
            "xgb_cuda_glm_boost_d1_d2_refit",
            xgb_train,
            xgb_refit,
            [{"depth": 1}, {"depth": 2}],
            TE_SEEDS,
        )
    except Exception as e:
        print(f"[phaseB] XGB failed: {e}", flush=True)

    # B3 CatBoost d2 (3 seeds)
    def cat_train(xa, ya, xb, yb, xt, ia, ib, it, cfg, ts):
        return run_cat(xa, ya, xb, yb, xt, ia, ib, it, depth=cfg["depth"], seed=ts, use_cuda=use_cuda)

    def cat_refit(xa, ya, xt, ia, it, cfg, ts, ntrees):
        # CatBoost: retrain with fixed iterations
        from catboost import CatBoostClassifier, Pool

        task = "GPU" if use_cuda else "CPU"
        model = CatBoostClassifier(
            loss_function="Logloss",
            depth=cfg["depth"],
            learning_rate=0.03,
            iterations=max(50, ntrees),
            random_seed=ts,
            verbose=False,
            task_type=task,
            allow_writing_files=False,
        )
        model.fit(Pool(xa, ya), baseline=ia.reshape(-1, 1))
        pt = model.predict(xt, prediction_type="Probability", baseline=it.reshape(-1, 1))
        if np.ndim(pt) > 1:
            pt = pt[:, 1] if pt.shape[1] > 1 else pt[:, 0]
        return np.asarray(pt, dtype=np.float64)

    try:
        bag_member(
            "cat_cuda_glm_boost_d2_refit",
            cat_train,
            cat_refit,
            [{"depth": 2}],
            CAT_TE_SEEDS,
        )
    except Exception as e:
        print(f"[phaseB] CatBoost failed: {e}", flush=True)

    summary = {
        "elapsed_sec": time.time() - t0,
        "cuda": use_cuda,
        "canary_fold0": canary,
        "glm_oof": glm_auc,
        "members": {k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "te")} for k, v in members.items()},
        "policy": "NO_AUTO_SUBMIT",
    }
    (OUT / "phaseB_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
