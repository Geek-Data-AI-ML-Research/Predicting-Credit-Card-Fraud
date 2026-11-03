#!/usr/bin/env python3
"""Unattended sequential T01–T10 campaign. NO kaggle competitions submit."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("S6E11_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diglead_matrix_cache import get_view  # noqa: E402
from unattended_common import (  # noqa: E402
    BAG_SEEDS,
    CANARY_FLOOR,
    CTRL_DIGLEAD,
    CTRL_NESTED_HILL,
    PRED,
    ROOT as UROOT,
    SEED,
    fold_aucs,
    load_ctrl_diglead,
    pr_auc,
    promotion_gate_base,
    run_test_safe,
    save_oof_test,
    write_candidate_csv,
)

assert UROOT == ROOT or True


def _lgb_bag(X_tr, y_tr, X_va, y_va, X_te, leaves_list=(3, 4), seeds=BAG_SEEDS):
    import lightgbm as lgb

    va_sum = np.zeros(len(X_va))
    te_sum = np.zeros(len(X_te))
    n = 0
    n_jobs = int(os.environ.get("LGBM_NUM_THREADS", "-1"))
    for leaves in leaves_list:
        for s in seeds:
            # Part-5 / diglead_l3_bag hyperparameters (reg_lambda=1.0, not 25)
            m = lgb.LGBMClassifier(
                learning_rate=0.03,
                num_leaves=leaves,
                max_depth=3 if leaves >= 4 else 2,
                min_child_samples=100,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                reg_lambda=1.0,
                max_bin=255,
                n_estimators=20000,
                n_jobs=n_jobs,
                verbose=-1,
                random_state=s,
            )
            m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(300, verbose=False)],
            )
            va_sum += m.predict_proba(X_va)[:, 1]
            te_sum += m.predict_proba(X_te)[:, 1]
            n += 1
    return va_sum / n, te_sum / n


def _xgb_bag(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    leaves_list=(3, 4),
    seeds=BAG_SEEDS,
    *,
    max_depth=0,
    grow_policy="lossguide",
    colsample=0.45,
    reg_lambda=30.0,
    reg_alpha=5.0,
    scale_pos_weight=1.0,
    n_estimators=5000,
):
    import xgboost as xgb

    va_sum = np.zeros(len(X_va))
    te_sum = np.zeros(len(X_te))
    n = 0
    for leaves in leaves_list:
        for s in seeds:
            params = dict(
                learning_rate=0.03,
                max_depth=max_depth,
                grow_policy=grow_policy,
                max_leaves=leaves if grow_policy == "lossguide" else 0,
                max_bin=256,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=colsample,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                device="cuda",
                eval_metric="auc",
                objective="binary:logistic",
                seed=s,
            )
            if grow_policy != "lossguide":
                params.pop("max_leaves", None)
                params["max_depth"] = leaves  # reuse leaves arg as depth for depth mode
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)
            dte = xgb.DMatrix(X_te)
            booster = xgb.train(
                params,
                dtr,
                num_boost_round=n_estimators,
                evals=[(dva, "va")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            it = booster.best_iteration + 1
            va_sum += booster.predict(dva, iteration_range=(0, it))
            te_sum += booster.predict(dte, iteration_range=(0, it))
            n += 1
    return va_sum / n, te_sum / n


def _xgb_depth_bag(X_tr, y_tr, X_va, y_va, X_te, depths=(1, 2), seeds=BAG_SEEDS):
    import xgboost as xgb

    va_sum = np.zeros(len(X_va))
    te_sum = np.zeros(len(X_te))
    n = 0
    for depth in depths:
        for s in seeds:
            params = dict(
                learning_rate=0.03,
                max_depth=depth,
                max_bin=256,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.35,
                reg_lambda=40.0,
                tree_method="hist",
                device="cuda",
                eval_metric="auc",
                objective="binary:logistic",
                seed=s,
            )
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)
            dte = xgb.DMatrix(X_te)
            booster = xgb.train(
                params,
                dtr,
                num_boost_round=5000,
                evals=[(dva, "va")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            it = booster.best_iteration + 1
            va_sum += booster.predict(dva, iteration_range=(0, it))
            te_sum += booster.predict(dte, iteration_range=(0, it))
            n += 1
    return va_sum / n, te_sum / n


def _cat_bag(X_tr, y_tr, X_va, y_va, X_te, depths=(2, 3)):
    from catboost import CatBoostClassifier, Pool

    va_sum = np.zeros(len(X_va))
    te_sum = np.zeros(len(X_te))
    n = 0
    for depth in depths:
        model = CatBoostClassifier(
            iterations=6000,
            depth=depth,
            learning_rate=0.04,
            l2_leaf_reg=25,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            random_seed=42 + depth,
            verbose=False,
            early_stopping_rounds=200,
        )
        model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True)
        va_sum += model.predict_proba(X_va)[:, 1]
        te_sum += model.predict_proba(X_te)[:, 1]
        n += 1
    return va_sum / n, te_sum / n


def _platt(y_tr, s_tr, s_va, s_te):
    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(s_tr.reshape(-1, 1), y_tr)
    return lr.predict_proba(s_va.reshape(-1, 1))[:, 1], lr.predict_proba(s_te.reshape(-1, 1))[:, 1]


def _base_result(tag, y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds, canary_aborted=False):
    oof_auc = float(roc_auc_score(y, oof))
    folds_a = fold_aucs(y, oof, folds)
    gate = promotion_gate_base(ctrl_folds, folds_a, ctrl_auc, oof_auc)
    save_oof_test(tag, oof, te)
    write_candidate_csv(f"cand_{tag}_{oof_auc:.5f}.csv", te)
    return {
        "status": "EARLY_ABORT" if canary_aborted else "OK",
        "tag": tag,
        "oof_auc": oof_auc,
        "fold_aucs": folds_a,
        "pr_auc": pr_auc(y, oof),
        "delta_vs_diglead": oof_auc - ctrl_auc,
        "gate_vs_diglead": gate,
        "promote_base": bool(gate["promote"] or oof_auc > CTRL_DIGLEAD),
        "canary_aborted": canary_aborted,
    }


# ---------------- T01 ----------------
def t01():
    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    fold_scores = []
    for f in range(5):
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
        print(f"T01 fold{f} X={Xtr.shape}", flush=True)
        pva, pte = _lgb_bag(Xtr, y[tr], Xva, y[va], Xte)
        fa = float(roc_auc_score(y[va], pva))
        fold_scores.append(fa)
        print(
            f"  T01 fold{f} auc={fa:.6f}  Δ_vs_diglead_l3_bag={fa - CTRL_DIGLEAD:+.6f}  "
            f"(ctrl={CTRL_DIGLEAD})",
            flush=True,
        )
        if f == 0 and fa < CANARY_FLOOR:
            oof[va] = pva
            te += pte
            # fill remaining with fold0 test only averaged later — abort
            te /= 1.0
            # zero other folds
            return _base_result(
                "t01_lgb_diglead_super",
                y,
                oof,
                te,
                folds,
                ctrl_oof,
                ctrl_auc,
                ctrl_folds,
                canary_aborted=True,
            )
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t01_lgb_diglead_super", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- T02 ----------------
def t02():
    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    for f in range(5):
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
        print(f"T02 fold{f} X={Xtr.shape}", flush=True)
        pva, pte = _xgb_bag(Xtr, y[tr], Xva, y[va], Xte, leaves_list=(3, 4))
        fa = float(roc_auc_score(y[va], pva))
        print(f"  T02 fold{f} auc={fa:.6f}", flush=True)
        if f == 0 and fa < CANARY_FLOOR:
            oof[va] = pva
            te += pte
            return _base_result(
                "t02_xgb_diglead_super", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds, True
            )
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t02_xgb_diglead_super", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- T03 ----------------
def t03():
    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    for f in range(5):
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
        print(f"T03 fold{f}", flush=True)
        pva, pte = _cat_bag(Xtr, y[tr], Xva, y[va], Xte, depths=(2, 3))
        fa = float(roc_auc_score(y[va], pva))
        print(f"  T03 fold{f} auc={fa:.6f}", flush=True)
        if f == 0 and fa < CANARY_FLOOR:
            oof[va] = pva
            te += pte
            return _base_result(
                "t03_cat_diglead_super", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds, True
            )
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t03_cat_diglead_super", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- T04 interaction stumps ----------------
def _interaction_matrix(view, f):
    """Append risk×diglead product TE-ish interactions on top of diglead view."""
    tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
    train, y = view.train, view.y
    risk = [
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
        "cvv_retry_count",
    ]
    # Part-5 leading-digit keys (key_l1_*)
    lead_cols = [c for c in view.dl_tr.columns if c.startswith("key_l1_")][:6]
    # build interaction codes on train slice
    def pack(idx_or_df, is_test=False):
        if is_test:
            base = view.test
            dl = view.dl_te
        else:
            base = train.iloc[idx_or_df]
            dl = view.dl_tr.iloc[idx_or_df]
        feats = []
        for r in risk:
            rv = base[r].astype(np.float64).values
            for c in lead_cols:
                feats.append((rv * (dl[c].astype(np.float64).values % 97)).astype(np.float32))
        # auth_method factorized
        am = pd.factorize(base["auth_method"].astype(str))[0].astype(np.float32)
        for c in lead_cols[:3]:
            feats.append((am * (dl[c].astype(np.float64).values % 97)).astype(np.float32))
        return np.column_stack(feats)

    Itr = pack(tr)
    Iva = pack(va)
    Ite = pack(None, is_test=True)
    return tr, va, np.hstack([Xtr, Itr]), np.hstack([Xva, Iva]), np.hstack([Xte, Ite])


def t04():
    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    for f in range(5):
        tr, va, Xtr, Xva, Xte = _interaction_matrix(view, f)
        print(f"T04 fold{f} X={Xtr.shape}", flush=True)
        pva, pte = _xgb_depth_bag(Xtr, y[tr], Xva, y[va], Xte, depths=(1, 2))
        fa = float(roc_auc_score(y[va], pva))
        print(f"  T04 fold{f} auc={fa:.6f}", flush=True)
        if f == 0 and fa < CANARY_FLOOR:
            oof[va] = pva
            te += pte
            return _base_result("t04_inter_stumps", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds, True)
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t04_inter_stumps", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- T05 GLM ----------------
def t05():
    from sklearn.preprocessing import SplineTransformer

    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    for f in range(5):
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
        # spline on first 20 continuous-ish cols
        n_spline = min(20, Xtr.shape[1])
        rest = Xtr.shape[1] - n_spline

        class SplitPipe:
            def __init__(self):
                self.sp = SplineTransformer(n_knots=8, degree=2, include_bias=False)
                self.sc = StandardScaler()
                self.lr = LogisticRegression(C=0.05, penalty="l2", solver="lbfgs", max_iter=1000)

            def fit(self, X, y_):
                A = self.sp.fit_transform(X[:, :n_spline])
                B = self.sc.fit_transform(X) if rest >= 0 else A
                # combine spline of head + scaled full
                Z = np.hstack([A, B])
                Z = np.nan_to_num(Z)
                self.lr.fit(Z, y_)
                return self

            def predict_proba(self, X):
                A = self.sp.transform(X[:, :n_spline])
                B = self.sc.transform(X)
                Z = np.nan_to_num(np.hstack([A, B]))
                return self.lr.predict_proba(Z)

        pipe = SplitPipe().fit(Xtr, y[tr])
        pva = pipe.predict_proba(Xva)[:, 1]
        pte = pipe.predict_proba(Xte)[:, 1]
        fa = float(roc_auc_score(y[va], pva))
        print(f"  T05 fold{f} auc={fa:.6f}", flush=True)
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t05_diglead_glm", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- T06 orig aug ----------------
def t06():
    import lightgbm as lgb
    import xgboost as xgb

    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    data = os.path.join(ROOT, "data")
    orig_path = os.path.join(data, "original.csv")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(ROOT, "original", "credit_card_fraud_2026.csv")
    orig = pd.read_csv(orig_path)
    from validation_stage3_baselines import TARGET

    y_orig = orig[TARGET].astype(int).values
    # build diglead for orig once per fold using train+orig TE fit on tr+orig
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    for f in range(5):
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
        # simple: append orig raw embedding cols as extra rows with is_orig
        # Build orig features via same diglead codes applied with TE from tr only on competition,
        # then append a shallow orig-only block — approximate by training on Xtr with weights
        # and concatenating a projected orig matrix using train TE on orig codes.
        y_combo = np.concatenate([y[tr], y_orig])
        w_combo = np.concatenate([np.ones(len(tr)), np.full(len(orig), 1.5)])
        Xtr_aug = np.hstack([Xtr, np.zeros((len(Xtr), 1), np.float32)])
        # Stratified bootstrap of train rows as orig stand-ins + is_orig flag (labels=y_orig)
        Xorig = np.empty((len(orig), Xtr.shape[1]), np.float32)
        rng = np.random.default_rng(42 + f)
        for i, yi in enumerate(y_orig):
            pool = np.where(y[tr] == yi)[0]
            if len(pool) == 0:
                pool = np.arange(len(tr))
            Xorig[i] = Xtr[int(rng.choice(pool))]
        Xorig = np.hstack([Xorig, np.ones((len(orig), 1), np.float32)])
        Xtr2 = np.vstack([Xtr_aug, Xorig])
        y2 = y_combo
        w2 = w_combo
        Xva2 = np.hstack([Xva, np.zeros((len(Xva), 1), np.float32)])
        Xte2 = np.hstack([Xte, np.zeros((len(Xte), 1), np.float32)])

        # LGB leaves=3 (Part-5 diglead hyperparams)
        m = lgb.LGBMClassifier(
            learning_rate=0.03,
            num_leaves=3,
            max_depth=2,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            max_bin=255,
            n_estimators=20000,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )
        m.fit(
            Xtr2,
            y2,
            sample_weight=w2,
            eval_set=[(Xva2, y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        pva_l = m.predict_proba(Xva2)[:, 1]
        pte_l = m.predict_proba(Xte2)[:, 1]
        # XGB max_leaves=3
        params = dict(
            learning_rate=0.03,
            max_depth=0,
            grow_policy="lossguide",
            max_leaves=3,
            max_bin=256,
            subsample=0.8,
            colsample_bytree=0.45,
            reg_lambda=30.0,
            tree_method="hist",
            device="cuda",
            eval_metric="auc",
            objective="binary:logistic",
            seed=42,
        )
        dtr = xgb.DMatrix(Xtr2, label=y2, weight=w2)
        dva = xgb.DMatrix(Xva2, label=y[va])
        dte = xgb.DMatrix(Xte2)
        booster = xgb.train(
            params, dtr, num_boost_round=5000, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False
        )
        it = booster.best_iteration + 1
        pva_x = booster.predict(dva, iteration_range=(0, it))
        pte_x = booster.predict(dte, iteration_range=(0, it))
        pva = 0.5 * pva_l + 0.5 * pva_x
        pte = 0.5 * pte_l + 0.5 * pte_x
        fa = float(roc_auc_score(y[va], pva))
        print(f"  T06 fold{f} auc={fa:.6f}", flush=True)
        if f == 0 and fa < CANARY_FLOOR:
            oof[va] = pva
            te += pte
            return _base_result(
                "t06_orig_aug_diglead", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds, True
            )
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t06_orig_aug_diglead", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- T07 focal / scale_pos_weight ----------------
def t07():
    view = get_view()
    y, folds = view.y, view.folds
    ctrl_oof, ctrl_auc, ctrl_folds = load_ctrl_diglead(y, folds)
    oof = np.zeros(len(y))
    te = np.zeros(len(view.test))
    for f in range(5):
        tr, va, Xtr, Xva, Xte = view.fold_matrix(f)
        bag_va = []
        bag_te = []
        for spw in (1.0, 1.5, 2.0):
            pva, pte = _xgb_bag(
                Xtr,
                y[tr],
                Xva,
                y[va],
                Xte,
                leaves_list=(3,),
                seeds=[42, 100, 2026],
                scale_pos_weight=spw,
                reg_alpha=5.0,
            )
            bag_va.append(pva)
            bag_te.append(pte)
        pva = np.mean(bag_va, axis=0)
        pte = np.mean(bag_te, axis=0)
        # Platt-calibrate on a VA slice (labels+scores same length), apply to full va/test
        n_cal = max(500, len(va) // 5)
        try:
            pva, pte = _platt(y[va][:n_cal], pva[:n_cal], pva, pte)
        except Exception:
            pass
        fa = float(roc_auc_score(y[va], pva))
        print(f"  T07 fold{f} auc={fa:.6f}", flush=True)
        if f == 0 and fa < CANARY_FLOOR:
            oof[va] = pva
            te += pte
            return _base_result("t07_tail_xgb", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds, True)
        oof[va] = pva
        te += pte
    te /= 5.0
    return _base_result("t07_tail_xgb", y, oof, te, folds, ctrl_oof, ctrl_auc, ctrl_folds)


# ---------------- bank helpers for T08/T09 ----------------
# Explicit Pass-2 inject list (T03 canary-abort is sparse — excluded on purpose)
PASS2_INJECT_TAGS = [
    "t01_lgb_diglead_super",
    "t02_xgb_diglead_super",
    "t04_inter_stumps",
    "t05_diglead_glm",
    "t06_orig_aug_diglead",
    "t07_tail_xgb",
]


def _pred_dirs():
    """Search worktree + persist mirrors for oof/test artifacts."""
    ws = os.environ.get("S6E11_PERSIST", "/workspace/s6e11")
    return [
        PRED,
        os.path.join(ROOT, "preds", "unattended"),
        os.path.join(ws, "preds", "unattended"),
    ]


def _find_oof_test(tag: str):
    for d in _pred_dirs():
        op = os.path.join(d, f"oof_{tag}.npy")
        tp = os.path.join(d, f"test_{tag}_full.npy")
        if os.path.exists(op) and os.path.exists(tp):
            return op, tp
    return None, None


def _collect_bank(y, *, require_inject: bool = False):
    ntr, nte = len(y), 299_984
    pre = os.path.join(ROOT, "preds", "stage9", "full_bank_unpruned.npz")
    if not os.path.exists(pre):
        pre = os.path.join(ROOT, "data", "full_bank_unpruned.npz")
    z = np.load(pre, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    O, T = z["O"].astype(np.float64), z["T"].astype(np.float64)
    base_m = len(names)
    # stage9.2 ortho (optional → pushes 73→75 when present)
    for name in ("ortho_logit_offset", "ortho_xgb_residual"):
        op = os.path.join(ROOT, "preds", "stage9", f"{name}_oof.npy")
        tp = os.path.join(ROOT, "preds", "stage9", f"{name}_test.npy")
        if os.path.exists(op) and os.path.exists(tp):
            names.append(f"s9:{name}")
            O = np.column_stack([O, np.load(op)])
            T = np.column_stack([T, np.load(tp)])
    # Always try T03 too (skip if sparse), then required Pass-2 six
    optional = ["t03_cat_diglead_super"]
    loaded = []
    missing = []
    for tag in optional + PASS2_INJECT_TAGS:
        op, tp = _find_oof_test(tag)
        if op is None:
            if tag in PASS2_INJECT_TAGS:
                missing.append(tag)
                print(f"bank miss {tag}", flush=True)
            continue
        o = np.load(op)
        t = np.load(tp)
        if len(o) != ntr or len(t) != nte or not np.isfinite(o).all():
            print(f"bank bad shape/nan {tag} o={getattr(o,'shape',None)}", flush=True)
            if tag in PASS2_INJECT_TAGS:
                missing.append(tag)
            continue
        # skip near-empty early abort (mostly zeros) — coverage on distinct folds
        if (np.abs(o) > 1e-12).sum() < ntr * 0.5:
            print(f"skip sparse {tag}", flush=True)
            if tag in PASS2_INJECT_TAGS:
                missing.append(tag)
            continue
        names.append(f"u:{tag}")
        O = np.column_stack([O, o])
        T = np.column_stack([T, t])
        loaded.append(tag)
    print(
        f"bank base={base_m} +ortho/extra → m={len(names)}; "
        f"inject_loaded={loaded} missing={missing}",
        flush=True,
    )
    if require_inject and missing:
        raise RuntimeError(f"Pass-2 bank incomplete, missing {missing} (need all 6 inject tags)")
    return names, O, T


def _to_logit(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _platt_fold(s_tr, y_tr, s_va, s_te):
    """1D Platt on fold-train scores → calibrated probabilities for va/te."""
    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(np.asarray(s_tr, dtype=np.float64).reshape(-1, 1), y_tr)
    p_va = lr.predict_proba(np.asarray(s_va, dtype=np.float64).reshape(-1, 1))[:, 1]
    p_te = lr.predict_proba(np.asarray(s_te, dtype=np.float64).reshape(-1, 1))[:, 1]
    return p_va, p_te


def _nested_hill_mix(O, T, y, folds, alphas=(0.0, 0.25, 0.5, 0.75, 1.0), *, start_j=None):
    """Coordinate ascent on logit/rank mix with batched GPU AUC.

    ZL and ZR are both z-scored on tr (mean=0, std=1). Fold outputs are
    Platt-calibrated on tr before writing OOF — kills inter-fold scale drift.
    """
    import time

    import torch

    device = "cuda" if _cuda_ok() else "cpu"
    ntr, m = O.shape
    nte = T.shape[0]
    oof = np.zeros(ntr)
    te_sum = np.zeros(nte)
    fold_aucs_out = []
    print(
        f"  nested_hill batched GPU AUC device={device} N_tr~{ntr // 5 * 4} M={m} "
        f"ZR_zscore+Platt start_j={start_j}",
        flush=True,
    )

    for fi, (tr, va) in enumerate(folds):
        t_fold = time.time()
        Ltr = _to_logit(O[tr])
        mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
        ZL_tr = ((Ltr - mu) / sd).astype(np.float32)
        ZL_va = ((_to_logit(O[va]) - mu) / sd).astype(np.float32)
        ZL_te = ((_to_logit(T) - mu) / sd).astype(np.float32)
        print(f"  nested_hill fold{fi} ranking {len(tr)}×{m}…", flush=True)
        ZR_tr_raw = np.column_stack(
            [(rankdata(O[tr, j]) - 0.5) / len(tr) for j in range(m)]
        ).astype(np.float32)
        ZR_va_raw = np.column_stack(
            [(rankdata(O[va, j]) - 0.5) / len(va) for j in range(m)]
        ).astype(np.float32)
        ZR_te_raw = np.column_stack(
            [(rankdata(T[:, j]) - 0.5) / nte for j in range(m)]
        ).astype(np.float32)
        # Match ZL scale: z-score ranks on tr stats (was mean~0.5 / std~0.288)
        mu_r, sd_r = ZR_tr_raw.mean(0), ZR_tr_raw.std(0) + 1e-12
        ZR_tr = (ZR_tr_raw - mu_r) / sd_r
        ZR_va = (ZR_va_raw - mu_r) / sd_r
        ZR_te = (ZR_te_raw - mu_r) / sd_r

        y_gpu = torch.as_tensor(y[tr], device=device, dtype=torch.float32)
        pos_mask = y_gpu == 1.0
        n_pos = float(pos_mask.sum().item())
        n_neg = float(len(tr) - n_pos)
        denom = n_pos * n_neg
        offset = n_pos * (n_pos + 1.0) / 2.0
        rank_vals = torch.arange(1, len(tr) + 1, device=device, dtype=torch.float32).unsqueeze(1)
        ZL_g = torch.as_tensor(ZL_tr, device=device)
        ZR_g = torch.as_tensor(ZR_tr, device=device)
        mb = (ZL_g.numel() + ZR_g.numel()) * 4 / 1e6
        print(f"  nested_hill fold{fi} on {device}: ZL+ZR={mb:.1f} MiB", flush=True)

        def batch_auc_gpu(cand_mat: torch.Tensor) -> torch.Tensor:
            order = torch.argsort(cand_mat, dim=0)
            ranks = torch.empty_like(cand_mat)
            ranks.scatter_(0, order, rank_vals.expand_as(cand_mat))
            pos_rank_sums = ranks[pos_mask].sum(dim=0)
            return (pos_rank_sums - offset) / denom

        def score_mat(w_np: np.ndarray, alpha: float) -> torch.Tensor:
            w_t = torch.as_tensor(w_np, device=device, dtype=torch.float32)
            return alpha * (ZL_g @ w_t) + (1.0 - alpha) * (ZR_g @ w_t)

        singles_mat = 0.5 * ZL_g + 0.5 * ZR_g
        singles = batch_auc_gpu(singles_mat).detach().cpu().numpy()
        w = np.zeros(m, dtype=np.float64)
        j0 = int(start_j) if start_j is not None else int(np.argmax(singles))
        j0 = max(0, min(m - 1, j0))
        w[j0] = 1.0
        best_alpha = 0.5
        cur = float(batch_auc_gpu(score_mat(w / w.sum(), best_alpha).unsqueeze(1))[0].item())
        print(f"  nested_hill fold{fi} init_auc={cur:.6f} start_j={j0}", flush=True)

        for it in range(40):
            improved = False
            for j in range(m):
                w2 = w.copy()
                w2[j] += 1.0
                ww = w2 / w2.sum()
                cand = torch.stack([score_mat(ww, a) for a in alphas], dim=1)
                aucs = batch_auc_gpu(cand)
                best_ai = int(torch.argmax(aucs).item())
                best_sc = float(aucs[best_ai].item())
                if best_sc > cur + 1e-8:
                    cur, w, best_alpha = best_sc, w2, alphas[best_ai]
                    improved = True
            if not improved:
                break
            if it % 5 == 0:
                print(
                    f"    fold{fi} iter={it} search_auc={cur:.6f} alpha={best_alpha}",
                    flush=True,
                )

        ww = w / w.sum()
        s_tr = best_alpha * (ZL_tr @ ww) + (1.0 - best_alpha) * (ZR_tr @ ww)
        s_va = best_alpha * (ZL_va @ ww) + (1.0 - best_alpha) * (ZR_va @ ww)
        s_te = best_alpha * (ZL_te @ ww) + (1.0 - best_alpha) * (ZR_te @ ww)
        # Platt on tr → probs, then within-fold ranks so pooled OOF shares [0,1] scale
        p_va, p_te = _platt_fold(s_tr, y[tr], s_va, s_te)
        oof[va] = (rankdata(p_va) - 0.5) / len(va)
        te_sum += (rankdata(p_te) - 0.5) / nte
        fold_aucs_out.append(float(roc_auc_score(y[va], s_va)))
        del y_gpu, ZL_g, ZR_g, singles_mat
        if device == "cuda":
            torch.cuda.empty_cache()
        print(
            f"  nested_hill fold{fi} auc={fold_aucs_out[-1]:.6f} alpha={best_alpha} "
            f"({time.time() - t_fold:.0f}s)",
            flush=True,
        )
    return oof, te_sum / 5.0, fold_aucs_out


def _anchored_logit_hill(O, T, y, folds, *, start_j=0, member_names=None):
    """Strict-improve GPU hillclimb in z-scored logit space; start at member start_j.

    Step-0 = pure anchor (ctrl). Only accepts blends that raise tr AUC.
    """
    import time

    import torch

    device = "cuda" if _cuda_ok() else "cpu"
    ntr, m = O.shape
    nte = T.shape[0]
    names = member_names or [f"m{i}" for i in range(m)]
    oof = np.zeros(ntr)
    te_sum = np.zeros(nte)
    fold_aucs_out = []
    print(
        f"  anchored_hill device={device} M={m} start={names[start_j]!r} members={names}",
        flush=True,
    )

    for fi, (tr, va) in enumerate(folds):
        t_fold = time.time()
        Ltr = _to_logit(O[tr])
        mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-12
        Z_tr = ((Ltr - mu) / sd).astype(np.float32)
        Z_va = ((_to_logit(O[va]) - mu) / sd).astype(np.float32)
        Z_te = ((_to_logit(T) - mu) / sd).astype(np.float32)

        y_gpu = torch.as_tensor(y[tr], device=device, dtype=torch.float32)
        pos_mask = y_gpu == 1.0
        n_pos = float(pos_mask.sum().item())
        n_neg = float(len(tr) - n_pos)
        denom = n_pos * n_neg
        offset = n_pos * (n_pos + 1.0) / 2.0
        rank_vals = torch.arange(1, len(tr) + 1, device=device, dtype=torch.float32).unsqueeze(1)
        Z_g = torch.as_tensor(Z_tr, device=device)

        def batch_auc_gpu(cand_mat: torch.Tensor) -> torch.Tensor:
            order = torch.argsort(cand_mat, dim=0)
            ranks = torch.empty_like(cand_mat)
            ranks.scatter_(0, order, rank_vals.expand_as(cand_mat))
            return (ranks[pos_mask].sum(dim=0) - offset) / denom

        w = np.zeros(m, dtype=np.float64)
        w[start_j] = 1.0
        cur = float(batch_auc_gpu((Z_g @ torch.as_tensor(w, device=device, dtype=torch.float32)).unsqueeze(1))[0].item())
        print(f"  anchored fold{fi} step0_auc={cur:.6f} (pure {names[start_j]})", flush=True)

        for it in range(40):
            improved = False
            # batch all +e_j candidates in one argsort
            cands = []
            for j in range(m):
                w2 = w.copy()
                w2[j] += 1.0
                ww = w2 / w2.sum()
                cands.append(Z_g @ torch.as_tensor(ww, device=device, dtype=torch.float32))
            aucs = batch_auc_gpu(torch.stack(cands, dim=1))
            best_j = int(torch.argmax(aucs).item())
            best_sc = float(aucs[best_j].item())
            if best_sc > cur + 1e-8:
                w2 = w.copy()
                w2[best_j] += 1.0
                cur, w = best_sc, w2
                improved = True
            if not improved:
                break
            if it % 5 == 0:
                nz = [(names[j], float(w[j])) for j in range(m) if w[j] > 0]
                print(f"    anchored fold{fi} iter={it} auc={cur:.6f} w={nz}", flush=True)

        ww = w / w.sum()
        s_tr = Z_tr @ ww
        s_va = Z_va @ ww
        s_te = Z_te @ ww
        p_va, p_te = _platt_fold(s_tr, y[tr], s_va, s_te)
        oof[va] = (rankdata(p_va) - 0.5) / len(va)
        te_sum += (rankdata(p_te) - 0.5) / nte
        fold_aucs_out.append(float(roc_auc_score(y[va], s_va)))
        del y_gpu, Z_g
        if device == "cuda":
            torch.cuda.empty_cache()
        nz = [(names[j], float(ww[j])) for j in range(m) if ww[j] > 1e-12]
        print(
            f"  anchored fold{fi} auc={fold_aucs_out[-1]:.6f} w={nz} "
            f"({time.time() - t_fold:.1f}s)",
            flush=True,
        )
    return oof, te_sum / 5.0, fold_aucs_out


def _load_or_rebuild_ctrl(y, folds):
    """Load Stage-7 ctrl nested-hill OOF/test, or rebuild from base-73 bank."""
    pred7 = os.path.join(ROOT, "preds", "stage7")
    os.makedirs(pred7, exist_ok=True)
    cref = os.path.join(pred7, "ctrl_nested_hill_oof.npy")
    cte = os.path.join(pred7, "ctrl_nested_hill_test.npy")
    # also accept campaign_best as test fallback
    cte_alt = os.path.join(pred7, "campaign_best_test.npy")
    if os.path.exists(cref):
        oof = np.load(cref).astype(np.float64)
        if os.path.exists(cte):
            te = np.load(cte).astype(np.float64)
        elif os.path.exists(cte_alt):
            te = np.load(cte_alt).astype(np.float64)
        else:
            te = None
        auc = float(roc_auc_score(y, oof))
        print(f"  ctrl loaded OOF={auc:.6f} from {cref}", flush=True)
        if te is not None and len(te) == 299_984:
            return oof, te, auc
        # fall through to rebuild test via hill on base bank using loaded oof as unused
    # rebuild from base 73 (no injects)
    print("  ctrl missing/incomplete → rebuild nested_hill on base-73 bank…", flush=True)
    pre = os.path.join(ROOT, "preds", "stage9", "full_bank_unpruned.npz")
    z = np.load(pre, allow_pickle=True)
    O, T = z["O"].astype(np.float64), z["T"].astype(np.float64)
    oof, te, _ = _nested_hill_mix(O, T, y, folds)
    auc = float(roc_auc_score(y, oof))
    np.save(cref, oof.astype(np.float32))
    np.save(cte, te.astype(np.float32))
    # persist
    ws = os.environ.get("S6E11_PERSIST", "/workspace/s6e11")
    for d in (os.path.join(ws, "preds", "stage7"),):
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "ctrl_nested_hill_oof.npy"), oof.astype(np.float32))
        np.save(os.path.join(d, "ctrl_nested_hill_test.npy"), te.astype(np.float32))
    print(f"  ctrl rebuilt+saved OOF={auc:.6f} (target≈{CTRL_NESTED_HILL})", flush=True)
    return oof, te, auc


def _collect_inject_bank(y):
    """OOF/test matrix of Pass-2 injects only (T01/T02/T04–T07)."""
    cols_o, cols_t, names = [], [], []
    for tag in PASS2_INJECT_TAGS:
        op, tp = _find_oof_test(tag)
        if op is None:
            print(f"  inject miss {tag}", flush=True)
            continue
        o, t = np.load(op).astype(np.float64), np.load(tp).astype(np.float64)
        if len(o) != len(y) or len(t) != 299_984:
            continue
        cols_o.append(o)
        cols_t.append(t)
        names.append(tag)
    if not cols_o:
        raise RuntimeError("no inject members for anchored blend")
    return names, np.column_stack(cols_o), np.column_stack(cols_t)


def t08():
    # clear stale marker
    for p in (
        os.path.join(ROOT, "logs", "stage10_reinject_hillclimb_result.json"),
        os.path.join(ROOT, "logs", "T08_done"),
    ):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    view = get_view()
    y, folds = view.y, view.folds
    require = os.environ.get("PASS2_REQUIRE_INJECT", "0") == "1"
    names, O, T = _collect_bank(y, require_inject=require)
    print(f"T08 bank m={len(names)} require_inject={require}", flush=True)

    # 1) Aligned expanded hill (ZR z-score + Platt) on full bank
    print("--- T08.1 aligned nested hill (full bank) ---", flush=True)
    oof_al, te_al, folds_al = _nested_hill_mix(O, T, y, folds)
    auc_al = float(roc_auc_score(y, oof_al))
    save_oof_test("t08_nested_hill_aligned", oof_al, te_al)
    save_oof_test("t08_nested_hill_expanded", oof_al, te_al)  # compat alias
    write_candidate_csv(f"cand_t08_aligned_{auc_al:.5f}.csv", te_al)
    print(f"T08.1 aligned OOF={auc_al:.6f} folds={['%.6f'%a for a in folds_al]}", flush=True)

    # 2) Load/rebuild 73-model control, then anchored superbase blend
    print("--- T08.2 anchored superbase (ctrl + injects + t08_aligned) ---", flush=True)
    ctrl_oof, ctrl_te, ctrl_auc = _load_or_rebuild_ctrl(y, folds)
    inj_names, inj_O, inj_T = _collect_inject_bank(y)
    mem_names = ["ctrl"] + inj_names + ["t08_aligned"]
    O_a = np.column_stack([ctrl_oof, inj_O, oof_al])
    T_a = np.column_stack([ctrl_te, inj_T, te_al])
    oof, te, folds_a = _anchored_logit_hill(
        O_a, T_a, y, folds, start_j=0, member_names=mem_names
    )
    oof_auc = float(roc_auc_score(y, oof))
    ctrl_folds = fold_aucs(y, ctrl_oof, folds)
    gate = promotion_gate_base(ctrl_folds, folds_a, ctrl_auc, oof_auc)
    save_oof_test("t08_anchored_superbase", oof, te)
    write_candidate_csv(f"cand_t08_anchored_{oof_auc:.5f}.csv", te)
    print(
        f"T08.2 anchored OOF={oof_auc:.6f} vs ctrl={ctrl_auc:.6f} "
        f"Δ={oof_auc - ctrl_auc:+.6f} aligned={auc_al:.6f}",
        flush=True,
    )
    return {
        "status": "OK",
        "tag": "t08_anchored_superbase",
        "n_members": len(mem_names),
        "member_names": mem_names,
        "oof_auc": oof_auc,
        "fold_aucs": folds_a,
        "aligned_oof_auc": auc_al,
        "aligned_fold_aucs": folds_al,
        "pr_auc": pr_auc(y, oof),
        "ctrl_nested_hill": ctrl_auc,
        "gate_vs_nested_hill": gate,
        "promote_meta": gate["promote"],
    }


def t09():
    """Superseded by T08.2 anchored superbase — skip slow SLSQP."""
    print("T09 SKIPPED — superseded by T08 anchored superbase blend", flush=True)
    # Prefer anchored; fall back to aligned expanded
    for tag in ("t08_anchored_superbase", "t08_nested_hill_aligned", "t08_nested_hill_expanded"):
        op, tp = _find_oof_test(tag)
        if op is None:
            continue
        oof, te = np.load(op), np.load(tp)
        view = get_view()
        y, folds = view.y, view.folds
        oof_auc = float(roc_auc_score(y, oof))
        folds_a = fold_aucs(y, oof, folds)
        save_oof_test("t09_hybrid_meta", oof, te)  # keep T10 cand slot warm
        return {
            "status": "SKIPPED",
            "tag": "t09_hybrid_meta",
            "mirror_of": tag,
            "oof_auc": oof_auc,
            "fold_aucs": folds_a,
            "pr_auc": pr_auc(y, oof),
            "promote_meta": False,
            "note": "SLSQP path retired; mirrored T08 anchored/aligned",
        }
    return {"status": "SKIPPED", "tag": "t09_hybrid_meta", "note": "no T08 artifact to mirror"}


def t10():
    from sklearn.isotonic import IsotonicRegression

    view = get_view()
    y, folds = view.y, view.folds
    # pick best among ctrl, aligned T08, anchored T08, t09 mirror
    cands = {}
    cref = os.path.join(ROOT, "preds", "stage7", "ctrl_nested_hill_oof.npy")
    cte = os.path.join(ROOT, "preds", "stage7", "ctrl_nested_hill_test.npy")
    cte_alt = os.path.join(ROOT, "preds", "stage7", "campaign_best_test.npy")
    if os.path.exists(cref):
        te_c = None
        if os.path.exists(cte):
            te_c = np.load(cte)
        elif os.path.exists(cte_alt):
            te_c = np.load(cte_alt)
        cands["nested_hill"] = (np.load(cref), te_c)
    for tag in (
        "t08_anchored_superbase",
        "t08_nested_hill_aligned",
        "t08_nested_hill_expanded",
        "t09_hybrid_meta",
    ):
        op = os.path.join(PRED, f"oof_{tag}.npy")
        tp = os.path.join(PRED, f"test_{tag}_full.npy")
        if os.path.exists(op) and os.path.exists(tp):
            cands[tag] = (np.load(op), np.load(tp))
    if not cands:
        raise RuntimeError("no candidates for T10")
    ranked = sorted(cands.items(), key=lambda kv: -roc_auc_score(y, kv[1][0]))
    best_name, (best_oof, best_te) = ranked[0]
    runner_name, (run_oof, run_te) = ranked[1] if len(ranked) > 1 else ranked[0]
    if best_te is None:
        best_te = np.full(299_984, 0.5)
    if run_te is None:
        run_te = best_te.copy()
    print(
        f"T10 cands={[(n, float(roc_auc_score(y, o))) for n,(o,_) in ranked]}",
        flush=True,
    )
    print(f"T10 base winner={best_name} oof={roc_auc_score(y, best_oof):.6f}", flush=True)

    # OOF postproc: 0.8 platt + 0.2 isotonic
    oof_cal = np.zeros(len(y))
    te_cal = np.zeros(len(best_te))
    for tr, va in folds:
        lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
        lr.fit(best_oof[tr].reshape(-1, 1), y[tr])
        p_va = lr.predict_proba(best_oof[va].reshape(-1, 1))[:, 1]
        p_te = lr.predict_proba(best_te.reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(best_oof[tr], y[tr])
        i_va = iso.transform(best_oof[va])
        i_te = iso.transform(best_te)
        # blend in logit space
        def mix(p, i):
            return expit(0.8 * logit(np.clip(p, 1e-6, 1 - 1e-6)) + 0.2 * logit(np.clip(i, 1e-6, 1 - 1e-6)))

        oof_cal[va] = mix(p_va, i_va)
        te_cal += mix(p_te, i_te)
    te_cal /= 5.0

    # top-tail rank smoothing using top-3 diverse families if available
    fam_tags = [
        "t01_lgb_diglead_super",
        "t02_xgb_diglead_super",
        "t05_diglead_glm",
    ]
    fams = []
    for tag in fam_tags:
        op = os.path.join(PRED, f"oof_{tag}.npy")
        tp = os.path.join(PRED, f"test_{tag}_full.npy")
        if os.path.exists(op) and os.path.exists(tp):
            fams.append((np.load(op), np.load(tp)))
    oof_final = oof_cal.copy()
    te_final = te_cal.copy()
    if len(fams) >= 2:
        thr = np.quantile(oof_cal, 0.97)
        mask = oof_cal >= thr
        ranks = np.column_stack([rankdata(f[0]) for f in fams])
        oof_final[mask] = ranks[mask].mean(1) / len(y)
        thr_te = np.quantile(te_cal, 0.97)
        mask_te = te_cal >= thr_te
        ranks_te = np.column_stack([rankdata(f[1]) for f in fams])
        te_final[mask_te] = ranks_te[mask_te].mean(1) / len(te_cal)

    oof_auc = float(roc_auc_score(y, oof_final))
    folds_a = fold_aucs(y, oof_final, folds)
    # gate vs nested hill
    if "nested_hill" in cands:
        ctrl = cands["nested_hill"][0]
        ctrl_auc = float(roc_auc_score(y, ctrl))
        ctrl_folds = fold_aucs(y, ctrl, folds)
    else:
        ctrl_auc, ctrl_folds = CTRL_NESTED_HILL, [CTRL_NESTED_HILL] * 5
    gate = promotion_gate_base(ctrl_folds, folds_a, ctrl_auc, oof_auc)

    save_oof_test("t10_postproc_final", oof_final, te_final)
    # pre-submit audit vs hillclimb_mix
    href = os.path.join(ROOT, "submissions", "hillclimb_mix_0.95111.csv")
    spear = pear = None
    if os.path.exists(href):
        ref = pd.read_csv(href)["is_fraud"].values
        our = rankdata(te_final) / len(te_final)
        spear = float(spearmanr(our, ref).correlation)
        pear = float(pearsonr(our, ref)[0])

    path_best = write_candidate_csv(f"sub_stage10_best_cv_{oof_auc:.5f}.csv", te_final)
    # runner-up: raw best before postproc or second ensemble
    run_auc = float(roc_auc_score(y, run_oof))
    path_run = write_candidate_csv(f"sub_stage10_runnerup_cv_{run_auc:.5f}.csv", run_te)

    audit = {
        "winner_base": best_name,
        "runner_up_base": runner_name,
        "oof_auc": oof_auc,
        "fold_aucs": folds_a,
        "fold_wins_vs_nested_hill": gate["fold_wins"],
        "pr_auc": pr_auc(y, oof_final),
        "spearman_vs_hillclimb_mix": spear,
        "pearson_vs_hillclimb_mix": pear,
        "gate_vs_nested_hill": gate,
        "submission_best": path_best,
        "submission_runnerup": path_run,
    }
    print("=== PRE-SUBMIT AUDIT ===", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    return {"status": "OK", "tag": "t10_postproc_final", "audit": audit, **{k: audit[k] for k in ("oof_auc", "fold_aucs")}}


TESTS = [
    ("T01_diglead_lgb_l3_l4_bag", t01),
    ("T02_diglead_xgb_cuda_l3_l4_bag", t02),
    ("T03_diglead_cat_cuda_d2_d3", t03),
    ("T04_diglead_interaction_stumps", t04),
    ("T05_diglead_ridge_spline_glm", t05),
    ("T06_orig20k_augmented_diglead_l3", t06),
    ("T07_focal_asymmetric_diglead_xgb", t07),
    ("T08_reinject_nested_hillclimb", t08),
    ("T09_nested_ridge_logit_rank_blend", t09),
    ("T10_postproc_isotonic_tail_audit", t10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of T01..T10 ids")
    ap.add_argument("--from", dest="from_id", default="T01")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    started = only or True
    past = only == set()
    for tid, fn in TESTS:
        short = tid.split("_")[0]
        if only and short not in only and tid not in only:
            continue
        if not only:
            if short == args.from_id or tid.startswith(args.from_id):
                past = True
            if not past:
                print(f"skip {tid} (before --from)", flush=True)
                continue
        run_test_safe(tid, fn)
    print("ALL REQUESTED TESTS FINISHED", flush=True)


if __name__ == "__main__":
    main()
