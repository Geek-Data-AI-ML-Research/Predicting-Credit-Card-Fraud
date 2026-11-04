"""Part 9: Part7 recipe dump + two-scale residual-basis stacker + Joint(GLM+GPT2) booster.
NO SUBMIT.
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
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from part6_repro_and_expand import ART, folds_y, rlogit  # noqa: E402
from part7_clean_superstack import (  # noqa: E402
    extract_bank_from_nb,
    fit_test,
    load_bank_fixed,
    nested_cv,
    rank,
)
from part8_proprietary_scan import build_p7_65, ols_beta  # noqa: E402
from part8_orth_superstack import to_stack_col  # noqa: E402

T0 = time.time()
CS = [0.006, 0.008, 0.01, 0.012]


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


def best_nested(O, y, folds, Cs=CS):
    best = (-1.0, None, None, None)
    for C in Cs:
        nest, a, fa = nested_cv(O, y, folds, C=C, tol=1e-8, max_iter=200)
        if a > best[0]:
            best = (a, C, fa, nest)
    return best


def unit(x, eps=1e-12):
    x = np.asarray(x, float).ravel()
    s = x.std()
    return (x - x.mean()) / s if s > eps else np.zeros_like(x)


# ── Step 1: dump Part7 recipe ──────────────────────────────────────────────
def step1_dump():
    log("STEP1: Part7 gpt2 / gbn recipe dump")
    nb = json.loads(
        Path("external_bank/goodpjw_part7_code/s6e11-part-7-gpt-2-mlp-stack-lb-0-95222.ipynb").read_text()
    )
    excerpts = {}
    for i in [82, 83, 84, 85, 86, 87]:
        src = "".join(nb["cells"][i].get("source", []))
        excerpts[f"cell_{i}_{nb['cells'][i]['cell_type']}"] = src
    # MLPRes recipe lives inside the %%writefile gboost.py cell
    gboost_src = "".join(nb["cells"][74].get("source", []))
    mlp_snip = ""
    if "class MLPRes" in gboost_src:
        i0 = gboost_src.index("class MLPRes")
        i1 = gboost_src.find("\ndef ", i0 + 1)
        mlp_snip = gboost_src[i0 : i1 if i1 > 0 else i0 + 2500]
    excerpts["cell_74_MLPRes_snip"] = mlp_snip[:3500]
    summary = {
        "gpt2": {
            "source": "masayakawamata s6e11-gpt-2-target-first-classifier-cv-0-9500 (CC BY 4.0)",
            "recipe": (
                "Chris Deotte target-first GPT-2: train to write rows; continue with label-first "
                "shuffled fields; score True/False likelihoods via Bayes. Loaded as oof.npy/test.npy; "
                "NOT retrained in Part7. OOF 0.95003, ρ(gb_bag5)=0.9598, stack coef ~+0.58."
            ),
            "stack_use": "rankdata(gpt_o)/n appended to P6 bank",
        },
        "gbn_bag5": {
            "recipe": (
                "Residual MLP via gboost.py kind=mlp (MLPRes): same inputs as GLM booster "
                "(raw, encodings, value counts; signed-log + standardized); output ADDED to GLM logit. "
                "Two widths bagged over 5 seeds: 512-512 dropout0.4 lr2e-4 → bag 0.95122; "
                "128-128 dropout0.3 lr3e-4 → bag 0.95122. Loaded as rank-mean oof/full from ART. "
                "NOT trained in the Part7 notebook (CPU); bags precomputed on GPU."
            ),
            "stack_delta": "+0.00004 nested over Part6 alone; with gpt2 → nested 0.95206 / LB 0.95222",
        },
        "stacker7": {
            "C": 0.01,
            "tol": 1e-8,
            "max_iter": 200,
            "solver": "newton-cholesky",
            "n_members": 65,
            "claimed_nested": 0.95206,
            "code": (
                "P7_NAMES = P6_NAMES + [gpt2, gbn_bag5_w512, gbn_bag5_w128]; "
                "nested LR(C=0.01,tol=1e-8,newton-cholesky) on rlogit(P7_O)"
            ),
        },
        "excerpts": excerpts,
    }
    print("\n--- Part7 recipe (compressed) ---")
    print(json.dumps({k: v for k, v in summary.items() if k != "excerpts"}, indent=2))
    print("\n--- Part7 cells 82-86 (markdown+code) ---")
    for k, v in excerpts.items():
        if k.startswith("cell_8"):
            print(f"\n##### {k}\n{v[:2000]}\n...")
    return summary


# ── bank builders ──────────────────────────────────────────────────────────
def build_72(y):
    """P7-65 + gbx_bag3 + alex6 (= keep7 without claiming it's residualized)."""
    P7_names, P7_O, P7_T, _ = build_p7_65(y)
    gbx_o = np.load(ART / "gbx_bag3_oof.npy").ravel().astype(float)
    gbx_t = np.load(ART / "gbx_bag3_full.npy").ravel().astype(float)
    ALEX = [
        "LightGBM tuned",
        "Spline GLM",
        "CatBoost",
        "CatBoost depth 3",
        "CatBoost tuned",
        "Joint GLM, full refit",
    ]
    man = json.loads(Path("external_bank/alex_09521_out/oof_manifest.json").read_text())
    nth = {v["name"]: h for h, v in man.items()}
    ao = [np.load(f"external_bank/alex_09521_out/oof_{nth[n]}.npy").ravel().astype(float) for n in ALEX]
    at = [np.load(f"external_bank/alex_09521_out/test_{nth[n]}.npy").ravel().astype(float) for n in ALEX]
    names = P7_names + ["gbx_bag3"] + [f"alex:{n}" for n in ALEX]
    O = np.column_stack(
        [P7_O, rankdata(gbx_o) / len(y)] + [rankdata(a) / len(y) for a in ao]
    )
    T = np.column_stack(
        [P7_T, rankdata(gbx_t) / len(gbx_t)] + [rankdata(a) / len(a) for a in at]
    )
    assert O.shape[1] == 72
    return names, O, T


def build_8b_champ(y, folds):
    """Rebuild Part8B champion: P7 + dual-resid(gb_bag5, ebm_i25, gbx) + alex6."""
    from part8_orth_superstack import orthogonalize_oof_and_test  # single-anchor ok; use dual

    P7_names, P7_O, P7_T, _ = build_p7_65(y)
    ni = {n: i for i, n in enumerate(P7_names)}
    _, GO, GT, _ = load_bank_fixed(["mega:glm_gen"], y)
    glm_o, glm_t = GO[:, 0], GT[:, 0]
    gpt_o, gpt_t = P7_O[:, ni["gpt2"]], P7_T[:, ni["gpt2"]]
    gbx_o = np.load(ART / "gbx_bag3_oof.npy").ravel().astype(float)
    gbx_t = np.load(ART / "gbx_bag3_full.npy").ravel().astype(float)

    def ols2(yy, x1, x2):
        X = np.column_stack([x1, x2])
        try:
            b = np.linalg.solve(X.T @ X, X.T @ yy)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(X.T @ X, X.T @ yy, rcond=None)[0]
        return float(b[0]), float(b[1])

    def dual(bo, bt):
        n = len(bo)
        br = rlogit(rankdata(bo) / n)
        x1 = rlogit(rankdata(glm_o) / n)
        x2 = rlogit(rankdata(gpt_o) / n)
        e = np.zeros(n)
        for a, b in folds:
            b1, b2 = ols2(br[a], x1[a], x2[a])
            e[b] = br[b] - b1 * x1[b] - b2 * x2[b]
        b1, b2 = ols2(br, x1, x2)
        bt_ = rlogit(rankdata(bt) / len(bt))
        t1 = rlogit(rankdata(glm_t) / len(glm_t))
        t2 = rlogit(rankdata(gpt_t) / len(gpt_t))
        return e, bt_ - b1 * t1 - b2 * t2

    members = ["gb_bag5", "ebm_i25_bag4", "gbx_bag3"]
    cols_o, cols_t, extra = [], [], []
    for n in members:
        if n == "gbx_bag3":
            bo, bt = gbx_o, gbx_t
        else:
            bo, bt = P7_O[:, ni[n]].copy(), P7_T[:, ni[n]].copy()
        eo, et = dual(bo, bt)
        cols_o.append(to_stack_col(eo, True))
        cols_t.append(to_stack_col(et, True))
        extra.append(f"{n}_dualresid_std")
    ALEX = [
        "LightGBM tuned",
        "Spline GLM",
        "CatBoost",
        "CatBoost depth 3",
        "CatBoost tuned",
        "Joint GLM, full refit",
    ]
    man = json.loads(Path("external_bank/alex_09521_out/oof_manifest.json").read_text())
    nth = {v["name"]: h for h, v in man.items()}
    for n in ALEX:
        o = np.load(f"external_bank/alex_09521_out/oof_{nth[n]}.npy").ravel().astype(float)
        t = np.load(f"external_bank/alex_09521_out/test_{nth[n]}.npy").ravel().astype(float)
        cols_o.append(rankdata(o) / len(y))
        cols_t.append(rankdata(t) / len(t))
        extra.append(f"alex:{n}")
    O = np.column_stack([P7_O] + cols_o)
    T = np.column_stack([P7_T] + cols_t)
    return P7_names + extra, O, T


# ── Step 2: two-scale orthogonal basis ─────────────────────────────────────
def transform_alpha(P_tr, P_va, anchor_idx, alpha):
    """Residualize non-anchors vs anchors on train; scale residuals by alpha after unit-var.
    Anchors kept at original rank scale (then rlogit applied outside).
    Returns transformed train/val matrices in rank-[0,1] space for stacker.
    """
    n_tr, p = P_tr.shape
    # work in rlogit space for OLS
    Ztr = rlogit(P_tr)
    Zva = rlogit(P_va)
    Atr = Ztr[:, anchor_idx]
    Ava = Zva[:, anchor_idx]
    # design with intercept-free anchors
    out_tr = np.zeros_like(Ztr)
    out_va = np.zeros_like(Zva)
    # keep anchors as-is (will convert back via expit-ish by using rank of decision later —
    # we store rlogit values then convert to ranks for nested_cv which expects (0,1))
    for j in range(p):
        if j in anchor_idx:
            out_tr[:, j] = Ztr[:, j]
            out_va[:, j] = Zva[:, j]
            continue
        # OLS Zj ~ A
        try:
            beta = np.linalg.lstsq(Atr, Ztr[:, j], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Atr.shape[1])
        e_tr = Ztr[:, j] - Atr @ beta
        e_va = Zva[:, j] - Ava @ beta
        e_tr = unit(e_tr) * alpha
        e_va = unit(e_va) * alpha  # use train std? For val, unit() uses val stats → slight leak
        # Better: scale val by train std
        s = e_tr.std() if False else 1.0  # already unit
        # re-do with train moments
        mu, sd = e_tr.mean(), e_tr.std()
        sd = sd if sd > 1e-12 else 1.0
        e_tr = (e_tr - mu) / sd * alpha
        e_va = ((Zva[:, j] - Ava @ beta) - mu) / sd * alpha
        out_tr[:, j] = e_tr
        out_va[:, j] = e_va
    # map back to (0,1) via rank within fold slice for stacker compatibility
    def to01(M):
        R = np.zeros_like(M)
        for j in range(M.shape[1]):
            R[:, j] = rankdata(M[:, j]) / len(M)
        return R

    return to01(out_tr), to01(out_va)


def transform_alpha_fixed(P_tr, P_va, anchor_idx, alpha):
    """Residualize non-anchors vs anchors in rlogit space; scale residuals to unit var * alpha.

    Returns continuous features (NOT re-ranked) so L2 sees the inflated residual amplitude.
    Anchors stay at raw rlogit scale (~O(1)).
    """
    Ztr = rlogit(P_tr)
    Zva = rlogit(P_va)
    Atr = Ztr[:, list(anchor_idx)]
    Ava = Zva[:, list(anchor_idx)]
    out_tr = np.zeros_like(Ztr)
    out_va = np.zeros_like(Zva)
    for j in range(Ztr.shape[1]):
        if j in anchor_idx:
            out_tr[:, j] = Ztr[:, j]
            out_va[:, j] = Zva[:, j]
            continue
        beta = np.linalg.lstsq(Atr, Ztr[:, j], rcond=None)[0]
        e_tr = Ztr[:, j] - Atr @ beta
        e_va = Zva[:, j] - Ava @ beta
        mu, sd = e_tr.mean(), e_tr.std()
        sd = sd if sd > 1e-12 else 1.0
        out_tr[:, j] = (e_tr - mu) / sd * alpha
        out_va[:, j] = (e_va - mu) / sd * alpha
    return out_tr, out_va


def nested_alpha(P, y, folds, anchor_idx, alpha, Cs=CS):
    best = (-1.0, None, None)
    for C in Cs:
        nested = np.zeros(len(y))
        for a, b in folds:
            Xtr, Xva = transform_alpha_fixed(P[a], P[b], anchor_idx, alpha)
            m = LogisticRegression(C=C, tol=1e-8, max_iter=200, solver="newton-cholesky")
            m.fit(Xtr, y[a])
            nested[b] = rankdata(m.decision_function(Xva)) / len(b)
        auc = float(roc_auc_score(y, nested))
        fa = [float(roc_auc_score(y[b], nested[b])) for _, b in folds]
        if auc > best[0]:
            best = (auc, C, fa)
    return best


def nested_pca_whiten(P, y, folds, gamma, n_keep=None, Cs=CS):
    """PCA on rlogit(P_tr); scale PC_i by var_i^(-gamma) for i>=1 (keep PC0 at 1). Fit on scaled PCs."""
    best = (-1.0, None, None)
    p = P.shape[1]
    n_keep = n_keep or p
    for C in Cs:
        nested = np.zeros(len(y))
        for a, b in folds:
            Ztr = rlogit(P[a])
            Zva = rlogit(P[b])
            pca = PCA(n_components=min(n_keep, Ztr.shape[0] - 1, Ztr.shape[1]), random_state=42)
            Ttr = pca.fit_transform(Ztr)
            Tva = pca.transform(Zva)
            ev = pca.explained_variance_
            scale = np.ones_like(ev)
            for i in range(1, len(ev)):
                scale[i] = (ev[i] + 1e-12) ** (-gamma)
            if len(scale) > 1:
                med = np.median(scale[1:])
                if med > 0:
                    scale[1:] = scale[1:] / med
            Ttr = Ttr * scale
            Tva = Tva * scale
            m = LogisticRegression(C=C, tol=1e-8, max_iter=200, solver="newton-cholesky")
            m.fit(Ttr, y[a])
            nested[b] = rankdata(m.decision_function(Tva)) / len(b)
        auc = float(roc_auc_score(y, nested))
        fa = [float(roc_auc_score(y[b], nested[b])) for _, b in folds]
        if auc > best[0]:
            best = (auc, C, fa)
    return best


def step2(y, folds):
    log("STEP2: two-scale orthogonal-basis stacker on 72-member bank")
    names, O72, T72 = build_72(y)
    base_a, base_C, base_fa, _ = best_nested(O72, y, folds)
    log(f"  plain 72-member nested OOF={base_a:.6f} C={base_C}")

    # anchors: glm_gen, gpt2, ens_mean
    ni = {n: i for i, n in enumerate(names)}
    glm_i = ni["mega:glm_gen"]
    gpt_i = ni["gpt2"]
    # ens_mean as extra column? User said keep 3 anchors including ens_mean —
    # inject ens_mean as column and treat as anchor
    ens_o = O72.mean(axis=1)
    ens_t = T72.mean(axis=1)
    O = np.column_stack([O72, rank(ens_o)])
    T = np.column_stack([T72, rank(ens_t)])
    names_e = names + ["ens_mean"]
    ens_i = len(names_e) - 1
    anchors = {glm_i, gpt_i, ens_i}

    # plain with ens_mean
    plain_a, plain_C, plain_fa, _ = best_nested(O, y, folds)
    log(f"  72+ens_mean plain OOF={plain_a:.6f} C={plain_C}")

    rows = []
    print(f"\n{'config':40} {'C':>6} {'OOF':>10} {'Δvs72':>10}")
    print(f"{'plain_72':40} {base_C:6.3f} {base_a:10.6f} {0.0:10.6f}")
    rows.append({"config": "plain_72", "C": base_C, "oof": base_a, "delta": 0.0})
    print(f"{'plain_72+ens':40} {plain_C:6.3f} {plain_a:10.6f} {plain_a-base_a:+10.6f}")
    rows.append({"config": "plain_72+ens", "C": plain_C, "oof": plain_a, "delta": plain_a - base_a})

    best_alpha = None
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        a, C, fa = nested_alpha(O, y, folds, anchors, alpha)
        d = a - base_a
        print(f"{'alpha='+str(alpha):40} {C:6.3f} {a:10.6f} {d:+10.6f}")
        rows.append({"config": f"alpha={alpha}", "C": C, "oof": a, "delta": d, "folds": fa})
        if best_alpha is None or a > best_alpha[0]:
            best_alpha = (a, C, alpha, fa)

    best_pca = None
    for gamma in [0.15, 0.25, 0.35, 0.50]:
        a, C, fa = nested_pca_whiten(O, y, folds, gamma)
        d = a - base_a
        print(f"{'pca_gamma='+str(gamma):40} {C:6.3f} {a:10.6f} {d:+10.6f}")
        rows.append({"config": f"pca_gamma={gamma}", "C": C, "oof": a, "delta": d, "folds": fa})
        if best_pca is None or a > best_pca[0]:
            best_pca = (a, C, gamma, fa)

    # also vs 8B champ 0.952129
    names8, O8, T8 = build_8b_champ(y, folds)
    a8, C8, fa8, _ = best_nested(O8, y, folds)
    log(f"  Part8B champ confirm OOF={a8:.6f}")
    rows.append({"config": "part8b_champ", "C": C8, "oof": a8, "delta": a8 - base_a})

    beat = max(best_alpha[0], best_pca[0]) > a8
    return {
        "plain_72": base_a,
        "rows": rows,
        "best_alpha": {"alpha": best_alpha[2], "oof": best_alpha[0], "C": best_alpha[1]},
        "best_pca": {"gamma": best_pca[2], "oof": best_pca[0], "C": best_pca[1]},
        "part8b_champ": a8,
        "beats_952129": bool(max(best_alpha[0], best_pca[0]) > a8 + 1e-9),
    }


# ── Step 3: Joint(GLM+GPT2) booster ────────────────────────────────────────
def step3(y, folds, champ_O, champ_names):
    """Inline fold loop (no giant cache_te0.npz on the external volume)."""
    log("STEP3: Joint(GLM+GPT2) base-margin booster (inline, no npz cache)")
    os.environ["INPUT_DIR"] = str(ROOT)

    import gboost as gb
    from gboost import (
        ALL,
        BASE,
        CAT_IDX,
        FOLDS,
        GEN,
        TEST,
        gen_fold,
        glm_fit_factory,
        logit,
        make_view_fn,
        te_fold,
    )

    gpt_o = np.load("external_bank/masaya_gpt2/oof.npy").ravel().astype(float)
    gpt_t = np.load("external_bank/masaya_gpt2/test.npy").ravel().astype(float)
    gpt_zr = rlogit(rankdata(gpt_o) / len(gpt_o))
    gpt_zt = rlogit(rankdata(gpt_t) / len(gpt_t))

    glm_fit = glm_fit_factory(0.3, None)
    view = make_view_fn("joint")
    teseed = 0

    z_glm = np.zeros(len(y))
    z_joint = np.zeros(len(y))
    oof_boost = np.zeros(len(y))
    foldavg = np.zeros(len(gpt_t))
    its = []

    lgb_params = dict(
        n_estimators=20000,
        learning_rate=0.02,
        num_leaves=4,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        max_bin=255,
        verbose=-1,
        n_jobs=max(1, (os.cpu_count() or 4) // 2),
        random_state=42,
    )

    for f, (a, b) in enumerate(FOLDS):
        log(f"  fold {f}: TE/gen + Joint GLM + LGB...")
        Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=teseed)
        Ga_, Gb_, Gt_ = (Ea, Eb, Et)
        if GEN:
            Ga, Gb, Gt = gen_fold(a, b, seed=teseed)
            Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        sc = StandardScaler().fit(logit(Ga_))
        D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
        Da, Db, Dt = D(Ga_, a), D(Gb_, b), D(Gt_, TEST)
        za, (zb, zt) = glm_fit(a, Da, [(b, Db), (TEST, Dt)], y[a])
        z_glm[b] = zb

        g_a, g_b, g_t = gpt_zr[a], gpt_zr[b], gpt_zt
        Ja = np.column_stack([Da, g_a, za * g_a])
        Jb = np.column_stack([Db, g_b, zb * g_b])
        Jt = np.column_stack([Dt, g_t, zt * g_t])
        mj = LogisticRegression(C=0.3, tol=1e-7, max_iter=200, solver="newton-cholesky")
        mj.fit(Ja, y[a])
        init_a = mj.decision_function(Ja)
        init_b = mj.decision_function(Jb)
        init_t = mj.decision_function(Jt)
        z_joint[b] = init_b

        Xa = view(a, Ea).astype(np.float32)
        Xb = view(b, Eb).astype(np.float32)
        Xt = view(TEST, Et).astype(np.float32)
        Xa2 = np.column_stack([Xa, g_a, g_a - za])
        Xb2 = np.column_stack([Xb, g_b, g_b - zb])
        Xt2 = np.column_stack([Xt, g_t, g_t - zt])

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            Xa2,
            y[a],
            init_score=init_a,
            eval_set=[(Xb2, y[b])],
            eval_init_score=[init_b],
            eval_metric="auc",
            categorical_feature=CAT_IDX,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        it = model.best_iteration_
        its.append(int(it))
        oof_boost[b] = init_b + model.predict(Xb2, raw_score=True)
        foldavg += (init_t + model.predict(Xt2, raw_score=True)) / len(FOLDS)
        log(
            f"  fold {f}: GLM {roc_auc_score(y[b], zb):.5f} | joint {roc_auc_score(y[b], init_b):.5f} "
            f"-> boost {roc_auc_score(y[b], oof_boost[b]):.5f} ({it} trees)"
        )
        del Da, Db, Dt, Xa, Xb, Xt, Xa2, Xb2, Xt2, Ja, Jb, Jt, model

    auc_glm = float(roc_auc_score(y, z_glm))
    auc_joint = float(roc_auc_score(y, z_joint))
    auc_boost = float(roc_auc_score(y, oof_boost))
    log(f"  Stage1 z_GLM OOF={auc_glm:.6f}")
    log(f"  Stage1 Joint(GLM+GPT2) OOF={auc_joint:.6f} Δ={auc_joint - auc_glm:+.6f}")
    log(f"  gb_gpt2_joint OOF={auc_boost:.6f}")

    # full-data refit for test
    log("  full-data refit...")
    a_all, b_all = ALL
    Ea, Eb, Et = te_fold(a_all, b_all, smooth=20.0, seed=teseed)
    Ga_, Gb_, Gt_ = (Ea, Eb, Et)
    if GEN:
        Ga, Gb, Gt = gen_fold(a_all, b_all, seed=teseed)
        Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
    sc = StandardScaler().fit(logit(Ga_))
    D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
    Da, Dt = D(Ga_, a_all), D(Gt_, TEST)
    za, (zt,) = glm_fit(a_all, Da, [(TEST, Dt)], y[a_all])
    Ja = np.column_stack([Da, gpt_zr, za * gpt_zr])
    Jt = np.column_stack([Dt, gpt_zt, zt * gpt_zt])
    m_full = LogisticRegression(C=0.3, tol=1e-7, max_iter=200, solver="newton-cholesky")
    m_full.fit(Ja, y)
    init_full = m_full.decision_function(Ja)
    init_te = m_full.decision_function(Jt)
    Xa_full = view(a_all, Ea).astype(np.float32)
    Xt_full = view(TEST, Et).astype(np.float32)
    Xa_full2 = np.column_stack([Xa_full, gpt_zr, gpt_zr - za])
    Xt_full2 = np.column_stack([Xt_full, gpt_zt, gpt_zt - zt])
    n_refit = int(round(np.mean(its) * 1.25))
    model = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": n_refit})
    model.fit(Xa_full2, y, init_score=init_full, categorical_feature=CAT_IDX)
    refit = init_te + model.predict(Xt_full2, raw_score=True)
    te_boost = (rankdata(foldavg) + rankdata(refit)) / (2 * len(gpt_t))

    # save under /tmp then copy small npy to preds/
    Path("preds").mkdir(exist_ok=True)
    np.save("preds/gb_gpt2_joint_oof.npy", oof_boost.astype(np.float32))
    np.save("preds/gb_gpt2_joint_test.npy", te_boost.astype(np.float32))
    log("  saved preds/gb_gpt2_joint_{oof,test}.npy")

    _, GO, GT, _ = load_bank_fixed(["mega:glm_gen"], y)
    glm_o, glm_t = GO[:, 0], GT[:, 0]
    P7_names, P7_O, P7_T, _ = build_p7_65(y)
    gpt_col_o = P7_O[:, P7_names.index("gpt2")]
    gpt_col_t = P7_T[:, P7_names.index("gpt2")]

    def dual(bo, bt, glm_o_, glm_t_, gpt_o_, gpt_t_):
        n = len(bo)
        br = rlogit(rankdata(bo) / n)
        x1 = rlogit(rankdata(glm_o_) / n)
        x2 = rlogit(rankdata(gpt_o_) / n)
        e = np.zeros(n)
        for aa, bb in folds:
            X = np.column_stack([x1[aa], x2[aa]])
            beta = np.linalg.lstsq(X, br[aa], rcond=None)[0]
            e[bb] = br[bb] - np.column_stack([x1[bb], x2[bb]]) @ beta
        X = np.column_stack([x1, x2])
        beta = np.linalg.lstsq(X, br, rcond=None)[0]
        bt_ = rlogit(rankdata(bt) / len(bt))
        t1 = rlogit(rankdata(glm_t_) / len(glm_t_))
        t2 = rlogit(rankdata(gpt_t_) / len(gpt_t_))
        return e, bt_ - np.column_stack([t1, t2]) @ beta

    eo, et = dual(oof_boost, te_boost, glm_o, glm_t, gpt_col_o, gpt_col_t)

    champ_a, champ_C, champ_fa, _ = best_nested(champ_O, y, folds)
    log(f"  Part8B champ baseline OOF={champ_a:.6f}")

    adds = {}
    for label, col_o in [
        ("+gb_gpt2_joint_raw", rankdata(oof_boost) / len(y)),
        ("+gb_gpt2_joint_resid_std", to_stack_col(eo, True)),
        ("+both_raw+resid", None),
    ]:
        if label.startswith("+both"):
            O = np.column_stack(
                [champ_O, rankdata(oof_boost) / len(y), to_stack_col(eo, True)]
            )
        else:
            O = np.column_stack([champ_O, col_o])
        a, C, fa, _ = best_nested(O, y, folds)
        d = a - champ_a
        print(f"  {label:35} OOF={a:.6f} Δ={d:+.6f} C={C}", flush=True)
        adds[label] = {"oof": a, "delta": d, "C": C, "folds": fa, "keep": bool(d > 0)}

    return {
        "stage1_glm_oof": auc_glm,
        "stage1_joint_oof": auc_joint,
        "stage1_delta": auc_joint - auc_glm,
        "gb_gpt2_joint_oof": auc_boost,
        "n_trees_folds": its,
        "n_refit": n_refit,
        "part8b_baseline": champ_a,
        "adds": adds,
    }


def main():
    log("Part9 — NO SUBMIT")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    folds = folds_y(y)

    s1 = step1_dump()
    s2 = step2(y, folds)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    # checkpoint after step2 so a step3 crash does not lose the basis scan
    ckpt = Path("logs/overnight/part9_joint_boost_and_basis.json")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(
        json.dumps(
            conv(
                {
                    "step1_part7_recipe": {k: v for k, v in s1.items() if k != "excerpts"},
                    "step1_excerpts_heads": {k: v[:800] for k, v in s1["excerpts"].items()},
                    "step2_basis": s2,
                    "step3_joint_boost": None,
                    "policy": "NO_SUBMIT",
                    "runtime_s": time.time() - T0,
                    "status": "step2_done",
                }
            ),
            indent=2,
        )
    )
    Path("logs/overnight/part9_part7_excerpts.json").write_text(json.dumps(s1["excerpts"], indent=2))
    log(f"checkpointed {ckpt} after step2")

    names8, O8, T8 = build_8b_champ(y, folds)
    s3 = step3(y, folds, O8, names8)

    summary = {
        "step1_part7_recipe": {k: v for k, v in s1.items() if k != "excerpts"},
        "step1_excerpts_heads": {k: v[:800] for k, v in s1["excerpts"].items()},
        "step2_basis": s2,
        "step3_joint_boost": s3,
        "policy": "NO_SUBMIT",
        "runtime_s": time.time() - T0,
    }

    out = Path("logs/overnight/part9_joint_boost_and_basis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conv(summary), indent=2))
    Path("logs/overnight/part9_part7_excerpts.json").write_text(
        json.dumps(s1["excerpts"], indent=2)
    )
    log(f"wrote {out}")
    print("\n[done] NO_SUBMIT")


if __name__ == "__main__":
    main()
