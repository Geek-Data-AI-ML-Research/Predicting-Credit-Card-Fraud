"""Part-8 GLM-orthogonal super-stack builder. NO SUBMIT."""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from part6_repro_and_expand import ART, PRED, folds_y, rlogit  # noqa: E402
from part7_clean_superstack import (  # noqa: E402
    extract_bank_from_nb,
    fit_test,
    load_bank_fixed,
    nested_cv,
    rank,
    rho,
)
from part8_proprietary_scan import (  # noqa: E402
    CATS,
    FLAGS,
    FLOATS,
    build_keep7,
    build_p7_65,
    first_dec,
    ols_beta,
    token_len_approx,
)

T0 = time.time()
CS = [0.006, 0.008, 0.01, 0.012, 0.015]


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


def r05(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v) - 0.5) / len(v)


def best_nested(O, y, folds, Cs=CS, tol=1e-8, mit=200):
    best = (-1.0, None, None, None)
    for C in Cs:
        nest, a, fa = nested_cv(O, y, folds, C=C, tol=tol, max_iter=mit)
        if a > best[0]:
            best = (a, C, fa, nest)
    return best


def unit_var(x, eps=1e-12):
    x = np.asarray(x, float).ravel()
    s = x.std()
    if s < eps:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def orthogonalize_oof_and_test(booster_o, booster_t, glm_o, glm_t, folds):
    """Fold-wise OOF residual; full-train beta for test."""
    n = len(booster_o)
    br = rlogit(rankdata(booster_o) / n)
    gr = rlogit(rankdata(glm_o) / n)
    e_o = np.zeros(n)
    for a, b in folds:
        beta = ols_beta(br[a], gr[a])
        e_o[b] = br[b] - beta * gr[b]
    beta_full = ols_beta(br, gr)
    bt = rlogit(rankdata(booster_t) / len(booster_t))
    gt = rlogit(rankdata(glm_t) / len(glm_t))
    e_t = bt - beta_full * gt
    return e_o, e_t, float(np.corrcoef(e_o, gr)[0, 1]) if e_o.std() > 0 else 0.0


def to_stack_col(e, standardize: bool):
    """Map residual to stacker column in (0,1)-ish ranks; optional unit-var before rank."""
    if standardize:
        e = unit_var(e)
    return rank(e)


def build_high_g_keys(tr: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rebuild Exp3-style keys; caller filters by G/df>0.80 list."""
    keys = {}
    for a, b in combinations(FLOATS, 2):
        xa, xb = tr[a].values, tr[b].values
        keys[f"toklen_{a}x{b}"] = (
            token_len_approx(xa) * 100 + token_len_approx(xb)
        ).astype(np.int32)
        keys[f"firstdec_{a}x{b}"] = (first_dec(xa) * 10 + first_dec(xb)).astype(np.int32)
        ca = (np.round(xa * 100) % 100).astype(np.int64)
        cb = (np.round(xb * 100) % 100).astype(np.int64)
        keys[f"cents_{a}x{b}"] = (ca * 100 + cb).astype(np.int32)
    for a, b in combinations(FLAGS, 2):
        keys[f"FLAG_{a}x{b}"] = (
            tr[a].astype(np.int8).values * 2 + tr[b].astype(np.int8).values
        ).astype(np.int32)
    for a, b, c in combinations(FLAGS, 3):
        keys[f"FLAG3_{a}x{b}x{c}"] = (
            tr[a].astype(np.int8).values * 4
            + tr[b].astype(np.int8).values * 2
            + tr[c].astype(np.int8).values
        ).astype(np.int32)
    keys["num_flags_set"] = tr[FLAGS].astype(np.int8).sum(axis=1).values.astype(np.int32)
    for a in CATS:
        ca = pd.factorize(tr[a].astype(str))[0]
        for b in FLAGS:
            keys[f"CATxFLAG_{a}x{b}"] = (ca * 2 + tr[b].astype(np.int8).values).astype(np.int32)
        for b, c in combinations(FLAGS, 2):
            keys[f"CxFxF_{a}x{b}x{c}"] = (
                ca * 4 + tr[b].astype(np.int8).values * 2 + tr[c].astype(np.int8).values
            ).astype(np.int32)
    return keys


def build_keys_te(tr: pd.DataFrame, te: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Factorize selected keys jointly over train+test → (n_tr, K) int matrix."""
    all_keys_tr = build_high_g_keys(tr)
    all_keys_te = build_high_g_keys(te)
    cols_tr, cols_te = [], []
    for n in names:
        a = np.concatenate([all_keys_tr[n].astype(object), all_keys_te[n].astype(object)])
        codes = pd.factorize(a)[0].astype(np.int32)
        cols_tr.append(codes[: len(tr)])
        cols_te.append(codes[len(tr) :])
    return np.column_stack(cols_tr), np.column_stack(cols_te)


def joint_key_resid_oof_test(key_tr, key_te, y, folds, z_p7_oof, z_p7_test):
    """Nested OneHot + LR(C=0.05) with offset≈rlogit(p7) via augmented feature + frozen scale.
    Implementation: fit LR on hstack([onehot, ones*0]) predicting y, using
    sample_weight trick — instead fit on residuals in probability space via
    LogisticRegression on onehot with y, then subtract z_p7 from decision_function
    after aligning means. Cleaner: TE of (y - p7_prob) per cell, then LR on TE cols.
    We use OneHot(min_frequency=20)+LR(C=0.05) on features, with offset injected as
    an extra column with coef fixed by using warm constraints:
      decision = coef·onehot + 1.0 * z_p7
    Fit by training on features=onehot to predict soft residual... 

    Practical leak-free method used here:
      For each outer fold, fit OneHot+LR(C=0.05) on train to predict y.
      OOF raw = decision_function(val).
      Then orthogonalize that decision vs z_p7 foldwise (OLS), yielding residual uplift.
    Full-data fit → test residual similarly.
    """
    n = len(y)
    oof_raw = np.zeros(n)
    # sklearn OneHotEncoder dense for modest cardinality after min_frequency
    for a, b in folds:
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
        Xa = enc.fit_transform(key_tr[a])
        Xb = enc.transform(key_tr[b])
        # lbfgs for wide one-hot; newton-cholesky can choke on fat dense one-hots
        m = LogisticRegression(C=0.05, tol=1e-8, max_iter=400, solver="lbfgs")
        m.fit(Xa, y[a])
        oof_raw[b] = m.decision_function(Xb)

    # Orthogonalize key-model decision vs p7 logit (removes double-count of base stack)
    e_o = np.zeros(n)
    for a, b in folds:
        beta = ols_beta(oof_raw[a], z_p7_oof[a])
        e_o[b] = oof_raw[b] - beta * z_p7_oof[b]

    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
    Xtr = enc.fit_transform(key_tr)
    Xte = enc.transform(key_te)
    m = LogisticRegression(C=0.05, tol=1e-8, max_iter=400, solver="lbfgs")
    m.fit(Xtr, y)
    raw_te = m.decision_function(Xte)
    beta_full = ols_beta(oof_raw, z_p7_oof)
    e_t = raw_te - beta_full * z_p7_test
    return e_o, e_t


def main():
    log("Part8 orth superstack — NO SUBMIT")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    te_ids = pd.read_csv("data/sample_submission.csv")["id"].to_numpy()
    N_TE = len(te_ids)
    assert N_TE == 299984
    folds = folds_y(y)

    P7_names, P7_O, P7_T, ok = build_p7_65(y)
    log(f"P7-65 loaded n={P7_O.shape[1]}")

    # mega:glm_gen ranks from bank (same as load_bank_fixed)
    _, G_O, G_T, _ = load_bank_fixed(["mega:glm_gen"], y)
    glm_o, glm_t = G_O[:, 0], G_T[:, 0]

    boosters = {
        "gbx_bag3": (
            np.load(ART / "gbx_bag3_oof.npy").ravel().astype(float),
            np.load(ART / "gbx_bag3_full.npy").ravel().astype(float),
        ),
        "gb_cat_d2_d3_bag": (
            np.load(PRED / "oof_gb_cat_d2_d3_bag.npy").ravel().astype(float),
            np.load(PRED / "sub_gb_cat_d2_d3_bag.npy").ravel().astype(float),
        ),
        "gb_xgb_cuda_d1_bag": (
            np.load(PRED / "oof_gb_xgb_cuda_d1_bag.npy").ravel().astype(float),
            np.load(PRED / "sub_gb_xgb_cuda_d1_bag.npy").ravel().astype(float),
        ),
        "gb_bag5": (
            np.load(ART / "gb_bag5_oof.npy").ravel().astype(float),
            np.load(ART / "gb_bag5_full.npy").ravel().astype(float),
        ),
        "gbn_bag5_w512": (
            np.load("preds/gbn_cache/gbn_bag5_w512_oof.npy").ravel().astype(float),
            np.load("preds/gbn_cache/gbn_bag5_w512_full.npy").ravel().astype(float),
        ),
        "gbn_bag5_w128": (
            np.load("preds/gbn_cache/gbn_bag5_w128_oof.npy").ravel().astype(float),
            np.load("preds/gbn_cache/gbn_bag5_w128_full.npy").ravel().astype(float),
        ),
    }

    resid_o, resid_t, resid_corr = {}, {}, {}
    for name, (bo, bt) in boosters.items():
        eo, et, c = orthogonalize_oof_and_test(bo, bt, glm_o, glm_t, folds)
        resid_o[name] = eo
        resid_t[name] = et
        resid_corr[name] = c
        log(f"  resid {name}: corr(e,glm)={c:.6f} oof_auc={roc_auc_score(y, eo):.6f}")

    base_a, base_C, base_fa, nest_p7 = best_nested(P7_O, y, folds)
    log(f"BASE P7-65: OOF={base_a:.6f} C={base_C} folds={[round(x,5) for x in base_fa]}")

    results = {
        "baseline_p7_65": {"oof": base_a, "C": base_C, "folds": list(map(float, base_fa))},
        "resid_corr": resid_corr,
    }

    def eval_stack(label, cols_o, cols_t, names_extra):
        O = np.column_stack([P7_O] + cols_o) if cols_o else P7_O
        T = np.column_stack([P7_T] + cols_t) if cols_t else P7_T
        a, C, fa, nest = best_nested(O, y, folds)
        d = a - base_a
        print(f"{label:55} n={O.shape[1]:3d} C={C:.3f} OOF={a:.6f} Δ={d:+.6f}")
        te_pred, model = fit_test(O, T, y, C, tol=1e-8, max_iter=200)
        return {
            "label": label,
            "n": int(O.shape[1]),
            "oof": a,
            "C": C,
            "delta": d,
            "folds": list(map(float, fa)),
            "nest": nest,
            "te": te_pred,
            "model": model,
            "O": O,
            "T": T,
            "names": list(P7_names) + names_extra,
        }

    print("\n" + "=" * 72)
    print("1A. GPU residual subsets (raw rank vs unit-var→rank)")
    print("=" * 72)
    gpu_names = ["gbx_bag3", "gb_xgb_cuda_d1_bag", "gb_cat_d2_d3_bag"]
    configs_1a = []
    # singletons + pairs + all3 × {unstd, std}
    from itertools import chain

    subsets = []
    for k in range(1, 4):
        subsets.extend(combinations(gpu_names, k))
    best_1a = None
    for subset in subsets:
        for std in (False, True):
            tag = "std" if std else "raw"
            cols_o = [to_stack_col(resid_o[n], std) for n in subset]
            cols_t = [to_stack_col(resid_t[n], std) for n in subset]
            enames = [f"{n}_resid{'_std' if std else ''}" for n in subset]
            label = f"P7+{'+'.join(n.replace('gb_','').replace('_bag','') for n in subset)}_{tag}"
            r = eval_stack(label, cols_o, cols_t, enames)
            configs_1a.append({k: r[k] for k in ("label", "n", "oof", "C", "delta", "folds")})
            if best_1a is None or r["oof"] > best_1a["oof"]:
                best_1a = r
    log(f"BEST 1A: {best_1a['label']} OOF={best_1a['oof']:.6f}")
    results["step1a"] = {"configs": configs_1a, "best": best_1a["label"], "best_oof": best_1a["oof"]}

    print("\n" + "=" * 72)
    print("1B. Best GPU resids + 6 alex keep7 extras")
    print("=" * 72)
    ALEX = [
        "LightGBM tuned",
        "Spline GLM",
        "CatBoost",
        "CatBoost depth 3",
        "CatBoost tuned",
        "Joint GLM, full refit",
    ]
    man = json.loads(Path("external_bank/alex_09521_out/oof_manifest.json").read_text())
    name_to_h = {v["name"]: h for h, v in man.items()}
    alex_o, alex_t, alex_n = [], [], []
    for n in ALEX:
        h = name_to_h[n]
        alex_o.append(np.load(f"external_bank/alex_09521_out/oof_{h}.npy").ravel().astype(float))
        alex_t.append(np.load(f"external_bank/alex_09521_out/test_{h}.npy").ravel().astype(float))
        alex_n.append(f"alex:{n}")

    # Use best_1a extras + alex
    # Reconstruct best_1a column recipe from label is messy — store recipe on best_1a
    # Re-eval top recipes: all3_std, all3_raw, gbx_std, gbx+xgb_std
    candidates_1b = []
    recipes = [
        (["gbx_bag3"], True),
        (["gbx_bag3"], False),
        (["gbx_bag3", "gb_xgb_cuda_d1_bag"], True),
        (["gbx_bag3", "gb_xgb_cuda_d1_bag", "gb_cat_d2_d3_bag"], True),
        (["gbx_bag3", "gb_xgb_cuda_d1_bag", "gb_cat_d2_d3_bag"], False),
    ]
    best_1b = None
    for subset, std in recipes:
        cols_o = [to_stack_col(resid_o[n], std) for n in subset] + [
            rankdata(a) / len(y) for a in alex_o
        ]
        cols_t = [to_stack_col(resid_t[n], std) for n in subset] + [
            rankdata(a) / len(a) for a in alex_t
        ]
        enames = [f"{n}_resid{'_std' if std else ''}" for n in subset] + alex_n
        label = f"P7+{'+'.join(s[:6] for s in subset)}_{('std' if std else 'raw')}+alex6"
        r = eval_stack(label, cols_o, cols_t, enames)
        candidates_1b.append({k: r[k] for k in ("label", "n", "oof", "C", "delta", "folds")})
        if best_1b is None or r["oof"] > best_1b["oof"]:
            best_1b = r
    log(f"BEST 1B: {best_1b['label']} OOF={best_1b['oof']:.6f}")
    results["step1b"] = {"configs": candidates_1b, "best": best_1b["label"], "best_oof": best_1b["oof"]}

    print("\n" + "=" * 72)
    print("1C. + internal booster resids (gbn512/128, gb_bag5) unit-var")
    print("=" * 72)
    internal = ["gbn_bag5_w512", "gbn_bag5_w128", "gb_bag5"]
    best_1c = None
    configs_1c = []
    # start from best_1b recipe — parse by re-running known good: gbx_std + alex6 as default base
    # Also try best_1a only + internals
    base_recipes = [
        ("1b_best_like", ["gbx_bag3"], True, True),  # gbx_std + alex
        ("1b_all3_std_alex", ["gbx_bag3", "gb_xgb_cuda_d1_bag", "gb_cat_d2_d3_bag"], True, True),
        ("1a_gbx_std", ["gbx_bag3"], True, False),
        ("1a_all3_std", ["gbx_bag3", "gb_xgb_cuda_d1_bag", "gb_cat_d2_d3_bag"], True, False),
    ]
    for tag, subset, std, use_alex in base_recipes:
        for add_internal in (False, True):
            cols_o = [to_stack_col(resid_o[n], std) for n in subset]
            cols_t = [to_stack_col(resid_t[n], std) for n in subset]
            enames = [f"{n}_resid_std" if std else f"{n}_resid" for n in subset]
            if use_alex:
                cols_o += [rankdata(a) / len(y) for a in alex_o]
                cols_t += [rankdata(a) / len(a) for a in alex_t]
                enames += alex_n
            if add_internal:
                for n in internal:
                    cols_o.append(to_stack_col(resid_o[n], True))
                    cols_t.append(to_stack_col(resid_t[n], True))
                    enames.append(f"{n}_resid_std")
            label = f"{tag}{'_+internal3' if add_internal else ''}"
            r = eval_stack(label, cols_o, cols_t, enames)
            configs_1c.append({k: r[k] for k in ("label", "n", "oof", "C", "delta", "folds")})
            if best_1c is None or r["oof"] > best_1c["oof"]:
                best_1c = r
    log(f"BEST 1C: {best_1c['label']} OOF={best_1c['oof']:.6f}")
    results["step1c"] = {"configs": configs_1c, "best": best_1c["label"], "best_oof": best_1c["oof"]}

    # Pick overall best step-1 stack object among best_1a/1b/1c
    step1_best = max([best_1a, best_1b, best_1c], key=lambda r: r["oof"])
    log(f"STEP1 CHAMPION: {step1_best['label']} OOF={step1_best['oof']:.6f} n={step1_best['n']}")

    print("\n" + "=" * 72)
    print("2. Joint multi-key residual corrector (G/df>0.80)")
    print("=" * 72)
    # Keys from exp3 top20 with G/df>0.80 + all toklen pairs + FLAG pairs + num_flags + firstdec
    exp3 = json.loads(Path("logs/overnight/part8_proprietary_scan.json").read_text())[
        "exp3_residual_gtest"
    ]["top20"]
    high = [t["key"] for t in exp3 if t["G_over_df"] > 0.80]
    # ensure all toklen pairs + FLAG pairs + num_flags_set present
    all_keys = build_high_g_keys(tr)
    force = [k for k in all_keys if k.startswith("toklen_") or k.startswith("FLAG_") or k == "num_flags_set" or k.startswith("firstdec_")]
    key_names = sorted(set(high) | set(force))
    # drop keys missing (CAT3 etc. may be in high but we have them if rebuilt)
    key_names = [k for k in key_names if k in all_keys]
    log(f"  joint keys selected: {len(key_names)}")
    for k in key_names[:25]:
        print(f"    {k}")
    if len(key_names) > 25:
        print(f"    ... +{len(key_names)-25} more")

    key_tr, key_te = build_keys_te(tr, te, key_names)
    z_p7 = rlogit(nest_p7)
    # test p7 decision from full fit
    te_p7, _ = fit_test(P7_O, P7_T, y, base_C, tol=1e-8, max_iter=200)
    z_p7_te = rlogit(te_p7)

    jk_o, jk_t = joint_key_resid_oof_test(key_tr, key_te, y, folds, z_p7, z_p7_te)
    log(f"  joint_key_resid standalone AUC={roc_auc_score(y, jk_o):.6f}")

    # Add to step1 best
    cols_o = [step1_best["O"][:, i] for i in range(65, step1_best["O"].shape[1])]
    cols_t = [step1_best["T"][:, i] for i in range(65, step1_best["T"].shape[1])]
    # Actually step1_best["O"] already includes P7 — extract extras only
    extra_o = [step1_best["O"][:, 65:]] if step1_best["O"].shape[1] > 65 else []
    # simpler:
    O2 = np.column_stack([step1_best["O"], to_stack_col(jk_o, True)])
    T2 = np.column_stack([step1_best["T"], to_stack_col(jk_t, True)])
    a2, C2, fa2, nest2 = best_nested(O2, y, folds)
    print(
        f"{'STEP1+joint_key_resid_std':55} n={O2.shape[1]:3d} C={C2:.3f} OOF={a2:.6f} "
        f"Δvs_p7={a2-base_a:+.6f} Δvs_step1={a2-step1_best['oof']:+.6f}"
    )
    te2, model2 = fit_test(O2, T2, y, C2, tol=1e-8, max_iter=200)
    names2 = step1_best["names"] + ["part8_joint_key_resid"]

    # Also try joint key on P7+gbx_resid_std only (lighter)
    O_light = np.column_stack(
        [P7_O, to_stack_col(resid_o["gbx_bag3"], True), to_stack_col(jk_o, True)]
    )
    T_light = np.column_stack(
        [P7_T, to_stack_col(resid_t["gbx_bag3"], True), to_stack_col(jk_t, True)]
    )
    al, Cl, fal, nestl = best_nested(O_light, y, folds)
    print(f"{'P7+gbx_std+joint_key':55} n={O_light.shape[1]:3d} C={Cl:.3f} OOF={al:.6f} Δ={al-base_a:+.6f}")

    # Champion selection
    champions = [
        (step1_best["oof"], step1_best["C"], step1_best["folds"], step1_best["te"], step1_best["model"],
         step1_best["O"], step1_best["T"], step1_best["names"], step1_best["nest"], step1_best["label"]),
        (a2, C2, fa2, te2, model2, O2, T2, names2, nest2, "step1+joint_key"),
        (al, Cl, fal, fit_test(O_light, T_light, y, Cl, tol=1e-8, max_iter=200)[0],
         None, O_light, T_light,
         P7_names + ["gbx_bag3_resid_std", "part8_joint_key_resid"], nestl, "P7+gbx_std+joint_key"),
    ]
    champ = max(champions, key=lambda x: x[0])
    (
        ch_oof,
        ch_C,
        ch_fa,
        ch_te,
        ch_model,
        ch_O,
        ch_T,
        ch_names,
        ch_nest,
        ch_label,
    ) = champ
    if ch_model is None:
        _, ch_model = fit_test(ch_O, ch_T, y, ch_C, tol=1e-8, max_iter=200)

    log(f"CHAMPION: {ch_label} OOF={ch_oof:.6f} C={ch_C} n={ch_O.shape[1]}")

    print("\n" + "=" * 72)
    print("3. Export submissions + p7_official uplift fusion")
    print("=" * 72)
    Path("submissions").mkdir(exist_ok=True)

    # Pure part8 champion
    assert len(ch_te) == N_TE and np.isfinite(ch_te).all()
    p8_path = "submissions/sub_part8_orth_superstack.csv"
    pd.DataFrame({"id": te_ids, "is_fraud": ch_te}).to_csv(p8_path, index=False)
    print(f"saved {p8_path}: rows={N_TE} nans=0 OOF={ch_oof:.6f}")
    print(f"folds={[round(x,5) for x in ch_fa]}")
    coef = pd.Series(ch_model.coef_[0], index=ch_names)
    print("Top 15 |coef|:")
    for n, v in coef.reindex(coef.abs().sort_values(ascending=False).index).head(15).items():
        print(f"  {v:+.4f}  {n}")

    # Local P7-65 test
    te_p7_local = te_p7  # from earlier fit at base_C
    p7_off = pd.read_csv("external_bank/goodpjw_part7_out/submission.csv")
    assert (p7_off["id"].to_numpy() == te_ids).all()
    p7v = p7_off["is_fraud"].to_numpy(float)
    nov_p6 = pd.read_csv("external_bank/malyshev_v11/nov04_part6.csv")["is_fraud"].to_numpy(float)
    nov_alex = pd.read_csv("external_bank/malyshev_v11/nov04_alex.csv")["is_fraud"].to_numpy(float)
    v11 = r05(0.60 * r05(nov_p6) + 0.40 * r05(nov_alex))
    slot5 = pd.read_csv("submissions/sub_slot5_p7off_v11_fusion.csv")["is_fraud"].to_numpy(float)

    delta_part8 = rank(ch_te) - rank(te_p7_local)
    p7_part8_upgraded = rank(rank(p7v) + delta_part8)
    print(f"ρ(delta_part8, p7_off)={rho(delta_part8, p7v):.6f}")
    print(f"ρ(p7_part8_upgraded, p7_off)={rho(p7_part8_upgraded, p7v):.6f}")
    print(f"ρ(p7_part8_upgraded, part8)={rho(p7_part8_upgraded, ch_te):.6f}")

    # Nested OOF proxy: ch_nest stands in for p7_part8_upgraded on OOF
    BANK = extract_bank_from_nb(
        "external_bank/goodpjw_part7_code/s6e11-part-7-gpt-2-mlp-stack-lb-0-95222.ipynb", 79
    )
    _, P6_O, _, _ = load_bank_fixed(BANK["ours"] + BANK["ext"], y)
    oof_p6, _, _ = nested_cv(P6_O, y, folds, C=0.01, tol=1e-8, max_iter=200)
    alex_oof = np.load("preds/alex_09521_nested_oof_rs42.npy").ravel()

    def fold_auc(scores):
        out = np.zeros(len(y))
        for a, b in folds:
            out[b] = rank(scores[b])
        return float(roc_auc_score(y, out))

    # OOF blends: use ch_nest as upgraded p7 proxy
    weight_grid = [
        (0.55, 0.27, 0.18, "0.55/0.27/0.18 (=slot5 style)"),
        (0.65, 0.20, 0.15, "0.65/0.20/0.15"),
        (0.60, 0.25, 0.15, "0.60/0.25/0.15"),
        (0.70, 0.15, 0.15, "0.70/0.15/0.15"),
        (0.50, 0.30, 0.20, "0.50/0.30/0.20"),
    ]
    print(f"\n{'OOF fusion weights':45} {'OOF':>10}")
    best_w = None
    for w7, w6, wa, lab in weight_grid:
        s = w7 * rank(ch_nest) + w6 * rank(oof_p6) + wa * rank(alex_oof)
        a = fold_auc(s)
        print(f"  {lab:43} {a:10.6f}")
        if best_w is None or a > best_w[0]:
            best_w = (a, w7, w6, wa, lab)

    # Save primary slot6 at 0.55/0.27/0.18 and alt 0.65/0.20/0.15
    def blend_test(w7, w6, wa):
        return rank(w7 * rank(p7_part8_upgraded) + w6 * rank(nov_p6) + wa * rank(nov_alex))

    slot6 = blend_test(0.55, 0.27, 0.18)
    slot6b = blend_test(0.65, 0.20, 0.15)
    p6a = "submissions/sub_slot6_p7_part8_upgraded_v11_fusion.csv"
    p6b = "submissions/sub_slot6_alt_065_020_015_fusion.csv"
    pd.DataFrame({"id": te_ids, "is_fraud": slot6}).to_csv(p6a, index=False)
    pd.DataFrame({"id": te_ids, "is_fraud": slot6b}).to_csv(p6b, index=False)

    def verify(path, arr):
        df = pd.read_csv(path)
        ok_ = len(df) == N_TE and df.is_fraud.isna().sum() == 0 and (df.id.to_numpy() == te_ids).all()
        print(
            f"  {path}: rows={len(df)} ok={ok_} "
            f"ρ(slot5)={rho(arr, slot5):.6f} ρ(p7)={rho(arr, p7v):.6f} ρ(v11)={rho(arr, v11):.6f} "
            f"ρ(part8)={rho(arr, ch_te):.6f}"
        )
        return ok_

    print("\nSpearman / verify:")
    verify(p8_path, ch_te)
    verify(p6a, slot6)
    verify(p6b, slot6b)
    print(f"ρ(part8, p7_off)={rho(ch_te, p7v):.6f}")
    print(f"ρ(part8, slot5)={rho(ch_te, slot5):.6f}")
    print(f"ρ(p7_upgraded, slot5)={rho(p7_part8_upgraded, slot5):.6f}")

    # summarize JSON (drop heavy arrays)
    summary = {
        "baseline_p7_65": results["baseline_p7_65"],
        "resid_corr": results["resid_corr"],
        "step1a_best": results["step1a"]["best"],
        "step1a_best_oof": results["step1a"]["best_oof"],
        "step1a_configs": results["step1a"]["configs"],
        "step1b_best": results["step1b"]["best"],
        "step1b_best_oof": results["step1b"]["best_oof"],
        "step1b_configs": results["step1b"]["configs"],
        "step1c_best": results["step1c"]["best"],
        "step1c_best_oof": results["step1c"]["best_oof"],
        "step1c_configs": results["step1c"]["configs"],
        "joint_key_n": len(key_names),
        "joint_key_on_step1": {"oof": a2, "C": C2, "delta_vs_step1": a2 - step1_best["oof"]},
        "joint_key_on_gbx": {"oof": al, "C": Cl},
        "champion": {
            "label": ch_label,
            "oof": ch_oof,
            "C": ch_C,
            "n": int(ch_O.shape[1]),
            "folds": list(map(float, ch_fa)),
            "top15_coef": [
                (n, float(v))
                for n, v in coef.reindex(coef.abs().sort_values(ascending=False).index)
                .head(15)
                .items()
            ],
        },
        "oof_fusions": {
            lab: fold_auc(w7 * rank(ch_nest) + w6 * rank(oof_p6) + wa * rank(alex_oof))
            for w7, w6, wa, lab in weight_grid
        },
        "best_oof_fusion": {"lab": best_w[4], "oof": best_w[0], "w": list(best_w[1:4])},
        "files": [p8_path, p6a, p6b],
        "spearman": {
            "part8|p7_off": float(rho(ch_te, p7v)),
            "part8|slot5": float(rho(ch_te, slot5)),
            "slot6_055|slot5": float(rho(slot6, slot5)),
            "slot6_065|slot5": float(rho(slot6b, slot5)),
            "p7_upgraded|p7_off": float(rho(p7_part8_upgraded, p7v)),
        },
        "policy": "NO_SUBMIT",
        "runtime_s": time.time() - T0,
    }

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

    outp = Path("logs/overnight/part8_orth_superstack.json")
    outp.write_text(json.dumps(conv(summary), indent=2))
    log(f"wrote {outp}")
    print("\n[done] NO_SUBMIT")


if __name__ == "__main__":
    main()
