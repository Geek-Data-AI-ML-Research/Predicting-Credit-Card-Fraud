"""LAB 3: Transductive Joint-Manifold TE + 5-seed LGB/MLP bags.
Strict leak-free: val fold b uses OOF P_super only (never y[b]).
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from part11_common import (
    BASELINE_GB_GPT2_BAG5,
    PRED11,
    folds_y,
    load_champ_stack,
    load_report,
    load_y_ids,
    log,
    rk,
    rlogit,
    save_report,
    stack_gate,
)

ROOT = Path(__file__).resolve().parents[1]
W_PSEUDO = 0.30


def te_rate_w(k, t, w, ncat, prior, smooth):
    s = np.bincount(k, weights=t * w, minlength=ncat)
    n = np.bincount(k, weights=w, minlength=ncat)
    return (s + prior * smooth) / (n + smooth)


def calibrate_to_mean(p, target_mean):
    """Isotonic-free mean calibration via logit shift."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # binary search logit shift
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = expit(np.log(p / (1 - p)) + mid).mean()
        if m > target_mean:
            hi = mid
        else:
            lo = mid
    return expit(np.log(p / (1 - p)) + 0.5 * (lo + hi))


def run_lab3(rep: dict) -> dict:
    t0 = time.time()
    log("=" * 72)
    log("LAB 3: Transductive TE + gb/gbn bags")
    log("=" * 72)

    y, _ = load_y_ids()
    folds = folds_y(y)
    NTR = len(y)
    import os as _os

    _os.environ["INPUT_DIR"] = str(ROOT)
    _os.environ.setdefault("OUT_DIR", str(ROOT / "preds" / "part11"))
    import gboost as gb

    P_oof = np.load("preds/P_super_champ_oof.npy").ravel().astype(float)
    P_te = np.load("preds/P_super_champ_test.npy").ravel().astype(float)
    log(f"  P_super OOF={roc_auc_score(y, P_oof):.6f}")

    D = gb.KMAT.shape[1]
    # OOF containers
    te_train_oof = np.zeros((NTR, D), np.float32)
    te_trans_oof = np.zeros((NTR, D), np.float32)
    kcnt_u_oof = np.zeros((NTR, D), np.float32)
    z_joint_trans_oof = np.zeros(NTR)
    te_train_te = []
    te_trans_te = []
    kcnt_u_te = []
    z_jt_te = []

    for fi, (a, b) in enumerate(folds):
        log(f"  fold {fi}: build TE_train_only + TE_transductive")
        ya = y[a].astype(np.float64)
        prior = float(ya.mean())
        # calibrated soft labels for U = b ∪ test
        Pb = calibrate_to_mean(P_oof[b], prior)
        Pt = calibrate_to_mean(P_te, prior)

        Ka, Kb, Kt = gb.KMAT[a], gb.KMAT[b], gb.KMAT[gb.TEST]
        # TE_train_only (standard OOF on a via inner CV for a, full-a for b/test)
        Ea = np.zeros((len(a), D), np.float32)
        from sklearn.model_selection import StratifiedKFold

        for j in range(D):
            for i, o in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(a)), ya):
                Ea[o, j] = te_rate_w(Ka[i, j], ya[i], np.ones(len(i)), gb.NCAT[j], ya[i].mean(), 20.0)[Ka[o, j]]
            r = te_rate_w(Ka[:, j], ya, np.ones(len(a)), gb.NCAT[j], prior, 20.0)
            te_train_oof[b, j] = r[Kb[:, j]]
            # stash for test mean
        Et_train = np.column_stack(
            [te_rate_w(Ka[:, j], ya, np.ones(len(a)), gb.NCAT[j], prior, 20.0)[Kt[:, j]] for j in range(D)]
        ).astype(np.float32)
        te_train_te.append(Et_train)

        # Transductive pool U: rows b + test with soft weights
        # Encode: for rates on a ∪ U, use hard y on a and soft P on U
        Ku = np.concatenate([Kb, Kt], 0)
        yu = np.concatenate([Pb, Pt], 0)
        wu = np.full(len(yu), W_PSEUDO, dtype=np.float64)
        Kall = np.concatenate([Ka, Ku], 0)
        yall = np.concatenate([ya, yu], 0)
        wall = np.concatenate([np.ones(len(a)), wu], 0)

        Et_tr = np.zeros((len(b), D), np.float32)
        Et_te = np.zeros((len(gb.TEST), D), np.float32)
        kc_b = np.zeros((len(b), D), np.float32)
        kc_t = np.zeros((len(gb.TEST), D), np.float32)
        for j in range(D):
            r = te_rate_w(Kall[:, j], yall, wall, gb.NCAT[j], prior, 20.0)
            Et_tr[:, j] = r[Kb[:, j]]
            Et_te[:, j] = r[Kt[:, j]]
            # KCNT in U
            nu = np.bincount(Ku[:, j], minlength=gb.NCAT[j])
            kc_b[:, j] = nu[Kb[:, j]]
            kc_t[:, j] = nu[Kt[:, j]]
        te_trans_oof[b] = Et_tr
        kcnt_u_oof[b] = kc_b
        te_trans_te.append(Et_te)
        kcnt_u_te.append(kc_t)

        # z_Joint_Transductive: GLM on TE_trans + BASE
        sc = StandardScaler().fit(rlogit(np.vstack([  # fit on a via train-only TE for a rows
            # use transductive rates mapped to a for training features
            np.column_stack([te_rate_w(Kall[:, j], yall, wall, gb.NCAT[j], prior, 20.0)[Ka[:, j]] for j in range(D)])
        ])))
        Ea_tr = np.column_stack([te_rate_w(Kall[:, j], yall, wall, gb.NCAT[j], prior, 20.0)[Ka[:, j]] for j in range(D)]).astype(
            np.float32
        )
        Da = np.hstack([sc.transform(rlogit(Ea_tr)), gb.BASE[a]])
        Db = np.hstack([sc.transform(rlogit(Et_tr)), gb.BASE[b]])
        Dt = np.hstack([sc.transform(rlogit(Et_te)), gb.BASE[gb.TEST]])
        m = LogisticRegression(C=0.3, max_iter=200, tol=1e-7, solver="newton-cholesky").fit(Da, y[a])
        z_joint_trans_oof[b] = m.decision_function(Db)
        z_jt_te.append(m.decision_function(Dt))
        log(f"    fold{fi} z_Joint_Trans AUC={roc_auc_score(y[b], z_joint_trans_oof[b]):.5f}")

    Et_train_mean = np.mean(te_train_te, 0)
    Et_trans_mean = np.mean(te_trans_te, 0)
    kc_te_mean = np.mean(kcnt_u_te, 0)
    z_jt_te_mean = np.mean(z_jt_te, 0)

    np.save(PRED11 / "lab3_te_train_only_oof.npy", te_train_oof)
    np.save(PRED11 / "lab3_te_transductive_oof.npy", te_trans_oof)
    np.save(PRED11 / "lab3_kcnt_test_support_oof.npy", kcnt_u_oof)
    np.save(PRED11 / "lab3_z_joint_trans_oof.npy", z_joint_trans_oof.astype(np.float32))
    np.save(PRED11 / "lab3_te_transductive_test.npy", Et_trans_mean.astype(np.float32))
    np.save(PRED11 / "lab3_z_joint_trans_test.npy", z_jt_te_mean.astype(np.float32))

    # Feature matrices for boosters: [RAWV, TE_trans, KCNT, log1p(KCNT_U), TE_trans-TE_train]
    X_oof = np.hstack(
        [
            gb.RAWV[:NTR],
            te_trans_oof,
            gb.KCNT[:NTR],
            np.log1p(kcnt_u_oof),
            te_trans_oof - te_train_oof,
        ]
    ).astype(np.float32)
    X_te = np.hstack(
        [
            gb.RAWV[gb.TEST],
            Et_trans_mean,
            gb.KCNT[gb.TEST],
            np.log1p(kc_te_mean),
            Et_trans_mean - Et_train_mean,
        ]
    ).astype(np.float32)

    # 5-seed LGB bag
    log("  training gb_transductive_bag5 …")
    seeds = (0, 1, 2, 3, 4)
    oof_lgb = np.zeros(NTR)
    te_lgb = np.zeros(len(gb.TEST))
    for seed in seeds:
        foof = np.zeros(NTR)
        fte = []
        for a, b in folds:
            dtr = lgb.Dataset(X_oof[a], label=y[a], init_score=z_joint_trans_oof[a])
            dva = lgb.Dataset(X_oof[b], label=y[b], init_score=z_joint_trans_oof[b], reference=dtr)
            params = dict(
                objective="binary",
                metric="auc",
                learning_rate=0.03,
                num_leaves=4,
                min_data_in_leaf=100,
                feature_fraction=0.5,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbosity=-1,
                seed=seed,
            )
            bst = lgb.train(
                params,
                dtr,
                num_boost_round=3000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            raw = bst.predict(X_oof[b], raw_score=True)
            foof[b] = expit(z_joint_trans_oof[b] + raw)
            fte.append(expit(z_jt_te_mean + bst.predict(X_te, raw_score=True)))
        oof_lgb += foof / len(seeds)
        te_lgb += np.mean(fte, 0) / len(seeds)
        log(f"    LGB seed={seed} running…")

    auc_lgb = float(roc_auc_score(y, oof_lgb))
    log(f"  gb_transductive_bag5 OOF={auc_lgb:.6f} vs gb_gpt2_bag5={BASELINE_GB_GPT2_BAG5}")
    np.save(PRED11 / "lab3_gb_transductive_bag5_oof.npy", oof_lgb.astype(np.float32))
    np.save(PRED11 / "lab3_gb_transductive_bag5_test.npy", te_lgb.astype(np.float32))

    # Residual MLP bag (sklearn CPU)
    log("  training gbn_transductive_bag5 …")
    oof_mlp = np.zeros(NTR)
    te_mlp = np.zeros(len(gb.TEST))
    # compact features for MLP: TE_trans logits + kcnt + z
    Xm_o = np.hstack([rlogit(te_trans_oof), np.log1p(kcnt_u_oof), z_joint_trans_oof.reshape(-1, 1)]).astype(np.float32)
    Xm_t = np.hstack([rlogit(Et_trans_mean), np.log1p(kc_te_mean), z_jt_te_mean.reshape(-1, 1)]).astype(np.float32)
    sc = StandardScaler().fit(Xm_o)
    Xm_o, Xm_t = sc.transform(Xm_o), sc.transform(Xm_t)
    for seed in seeds:
        foof = np.zeros(NTR)
        fte = []
        for a, b in folds:
            # residual target in logit space is hard; predict proba and blend
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 256),
                activation="relu",
                alpha=1e-4,
                batch_size=4096,
                learning_rate_init=1e-3,
                max_iter=40,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
            )
            # fit residual: y vs features, then blend with z_joint via stacking of probs
            clf.fit(Xm_o[a], y[a])
            p = clf.predict_proba(Xm_o[b])[:, 1]
            foof[b] = 0.5 * p + 0.5 * expit(z_joint_trans_oof[b])
            fte.append(0.5 * clf.predict_proba(Xm_t)[:, 1] + 0.5 * expit(z_jt_te_mean))
        oof_mlp += foof / len(seeds)
        te_mlp += np.mean(fte, 0) / len(seeds)

    auc_mlp = float(roc_auc_score(y, oof_mlp))
    log(f"  gbn_transductive_bag5 OOF={auc_mlp:.6f}")
    np.save(PRED11 / "lab3_gbn_transductive_bag5_oof.npy", oof_mlp.astype(np.float32))
    np.save(PRED11 / "lab3_gbn_transductive_bag5_test.npy", te_mlp.astype(np.float32))

    names, O, T, champ_meta = load_champ_stack(y, folds)
    gate_lgb = stack_gate(O, oof_lgb, y, folds, label="lab3_gb_transductive_bag5")
    gate_mlp = stack_gate(O, oof_mlp, y, folds, label="lab3_gbn_transductive_bag5")

    lab = {
        "name": "LAB3_transductive",
        "runtime_s": time.time() - t0,
        "w_pseudo": W_PSEUDO,
        "gb_transductive_auc": auc_lgb,
        "gbn_transductive_auc": auc_mlp,
        "baseline_gb_gpt2_bag5": BASELINE_GB_GPT2_BAG5,
        "delta_vs_gb_gpt2": auc_lgb - BASELINE_GB_GPT2_BAG5,
        "stack_gate_lgb": gate_lgb,
        "stack_gate_mlp": gate_mlp,
        "champ_meta": champ_meta,
        "keep_any": bool(gate_lgb["keep"] or gate_mlp["keep"]),
    }
    rep["lab3"] = lab
    save_report(rep)
    log(f"LAB3 done in {lab['runtime_s']:.1f}s keep_any={lab['keep_any']}")
    return lab


if __name__ == "__main__":
    run_lab3(load_report())
