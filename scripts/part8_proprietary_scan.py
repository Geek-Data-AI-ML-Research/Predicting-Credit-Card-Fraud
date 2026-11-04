"""Part 8 proprietary R&D scan (CPU). NO SUBMIT.
Exp1: GLM-orthogonal residual boosters on intact P7-65
Exp2: orig.csv seed-row prototype matching
Exp3: residual G-test scan vs p7_keep7
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


def fold_auc(scores, y, folds):
    out = np.zeros(len(y))
    for a, b in folds:
        out[b] = rank(scores[b])
    return float(roc_auc_score(y, out))


def best_nested(O, y, folds, Cs=(0.008, 0.01, 0.012, 0.015, 0.02), tol=1e-8, mit=200):
    best = (-1.0, None, None)
    for C in Cs:
        _, a, fa = nested_cv(O, y, folds, C=C, tol=tol, max_iter=mit)
        if a > best[0]:
            best = (a, C, fa)
    return best


def ols_beta(x, z):
    """beta = (x·z) / (z·z) for centered-or-not 1D OLS without intercept (projection onto z)."""
    zz = float(np.dot(z, z))
    if zz < 1e-18:
        return 0.0
    return float(np.dot(x, z) / zz)


def orthogonalize_foldwise(booster, glm, folds):
    """Per-fold: e = rlogit(rank(booster)) - beta * rlogit(rank(glm)), beta fit on train fold."""
    n = len(booster)
    br = rlogit(rankdata(booster) / n)
    gr = rlogit(rankdata(glm) / n)
    e = np.zeros(n)
    for a, b in folds:
        beta = ols_beta(br[a], gr[a])
        e[b] = br[b] - beta * gr[b]
    # full-data residual for test-side construction isn't needed for OOF CV; return OOF resid
    return e, float(np.corrcoef(e, gr)[0, 1]) if e.std() > 0 else 0.0


def build_p7_65(y):
    GBN = Path("preds/gbn_cache")
    BANK = extract_bank_from_nb(
        "external_bank/goodpjw_part7_code/s6e11-part-7-gpt-2-mlp-stack-lb-0-95222.ipynb", 79
    )
    ok, P6_O, P6_T, _ = load_bank_fixed(BANK["ours"] + BANK["ext"], y)
    gpt_o = np.load("external_bank/masaya_gpt2/oof.npy").ravel().astype(float)
    gpt_t = np.load("external_bank/masaya_gpt2/test.npy").ravel().astype(float)
    w512_o = np.load(GBN / "gbn_bag5_w512_oof.npy").ravel().astype(float)
    w512_t = np.load(GBN / "gbn_bag5_w512_full.npy").ravel().astype(float)
    w128_o = np.load(GBN / "gbn_bag5_w128_oof.npy").ravel().astype(float)
    w128_t = np.load(GBN / "gbn_bag5_w128_full.npy").ravel().astype(float)
    names = list(ok) + ["gpt2", "gbn_bag5_w512", "gbn_bag5_w128"]
    O = np.column_stack([P6_O, rankdata(gpt_o) / len(y), w512_o, w128_o])
    T = np.column_stack([P6_T, rankdata(gpt_t) / len(gpt_t), w512_t, w128_t])
    assert O.shape[1] == 65
    return names, O, T, ok


def build_keep7(y, P7_O, P7_T, P7_names):
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
    gbx_o = np.load(ART / "gbx_bag3_oof.npy").ravel().astype(float)
    gbx_t = np.load(ART / "gbx_bag3_full.npy").ravel().astype(float)
    extras_o = [gbx_o]
    extras_t = [gbx_t]
    extra_names = ["gbx_bag3"]
    for n in ALEX:
        h = name_to_h[n]
        extras_o.append(np.load(f"external_bank/alex_09521_out/oof_{h}.npy").ravel().astype(float))
        extras_t.append(np.load(f"external_bank/alex_09521_out/test_{h}.npy").ravel().astype(float))
        extra_names.append(f"alex:{n}")
    O = np.column_stack([P7_O] + extras_o)
    T = np.column_stack([P7_T] + extras_t)
    return P7_names + extra_names, O, T


# ── Exp 1 ──────────────────────────────────────────────────────────────────
def exp1(y, folds, P7_O):
    log("EXP1: GLM-orthogonal residual boosters")
    # mega:glm_gen from bank
    BANK = extract_bank_from_nb(
        "external_bank/goodpjw_part7_code/s6e11-part-7-gpt-2-mlp-stack-lb-0-95222.ipynb", 79
    )
    ok, O6, _, _ = load_bank_fixed(["mega:glm_gen"], y)
    assert ok == ["mega:glm_gen"]
    glm = O6[:, 0]  # already rank-normalized in load_bank_fixed? check
    # load_bank_fixed returns rankdata/n already — use raw if possible
    # Re-load raw for cleaner projection
    # From part7 loader paths — use bank member raw via ART mega pool if needed.
    # O6 is ranked; rlogit(rank()) of already-ranked is monotone OK for projection.
    glm_for_proj = glm

    boosters = {
        "gbx_bag3": np.load(ART / "gbx_bag3_oof.npy").ravel().astype(float),
        "gb_cat_d2_d3_bag": np.load(PRED / "oof_gb_cat_d2_d3_bag.npy").ravel().astype(float),
        "gb_xgb_cuda_d1_bag": np.load(PRED / "oof_gb_xgb_cuda_d1_bag.npy").ravel().astype(float),
    }

    base_a, base_C, base_fa = best_nested(P7_O, y, folds)
    log(f"  baseline P7-65: OOF={base_a:.6f} C={base_C}")

    resid = {}
    corr_after = {}
    results = {"baseline_p7_65": {"oof": base_a, "C": base_C, "folds": list(map(float, base_fa))}}
    print(f"\n{'add':30} {'n':>4} {'C':>6} {'OOF':>10} {'Δ':>10} {'corr(e,glm)':>12} keep?")
    print(f"{'(baseline P7-65)':30} {65:4d} {base_C:6.3f} {base_a:10.6f} {0.0:10.6f} {'—':>12}")

    for name, bo in boosters.items():
        e, c = orthogonalize_foldwise(bo, glm_for_proj, folds)
        resid[name] = e
        corr_after[name] = c
        # also report raw corr before
        br = rlogit(rankdata(bo) / len(y))
        gr = rlogit(rankdata(glm_for_proj) / len(y))
        c_before = float(np.corrcoef(br, gr)[0, 1])
        O = np.column_stack([P7_O, rank(e)])  # stacker expects rank-like [0,1]; use rank of residual
        a, C, fa = best_nested(O, y, folds)
        keep = a > base_a
        print(
            f"+ {name}_resid{'':<12} {O.shape[1]:4d} {C:6.3f} {a:10.6f} {a-base_a:+10.6f} "
            f"{c:12.6f} {'KEEP' if keep else 'reject'}  (pre-corr={c_before:.4f})"
        )
        results[f"+{name}_resid"] = {
            "oof": a,
            "C": C,
            "delta": a - base_a,
            "keep": keep,
            "corr_resid_glm": c,
            "corr_raw_glm": c_before,
        }

    # joint all 3 residuals
    O = np.column_stack([P7_O] + [rank(resid[n]) for n in boosters])
    a, C, fa = best_nested(O, y, folds)
    keep = a > base_a
    print(
        f"+ all3_resid{'':<16} {O.shape[1]:4d} {C:6.3f} {a:10.6f} {a-base_a:+10.6f} "
        f"{'—':>12} {'KEEP' if keep else 'reject'}"
    )
    results["+all3_resid"] = {"oof": a, "C": C, "delta": a - base_a, "keep": keep}

    # compare raw (non-orthogonal) adds for reference
    for name, bo in boosters.items():
        O = np.column_stack([P7_O, rankdata(bo) / len(y)])
        a, C, _ = best_nested(O, y, folds)
        results[f"+{name}_RAW"] = {"oof": a, "C": C, "delta": a - base_a, "keep": a > base_a}
        print(f"+ {name}_RAW{'':<13} {O.shape[1]:4d} {C:6.3f} {a:10.6f} {a-base_a:+10.6f}")

    return results, resid


# ── Exp 2 ──────────────────────────────────────────────────────────────────
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
FLOATS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]
NUMS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
FEATS = NUMS + FLAGS + CATS


def build_token_cols(df):
    """24 base cols + int(x) + cents for floats → list of 1d object/int arrays."""
    cols = {}
    for c in FEATS:
        if c in CATS:
            cols[f"v_{c}"] = df[c].astype(str).values
        elif c in FLAGS:
            cols[f"v_{c}"] = df[c].astype(np.int8).values
        else:
            cols[f"v_{c}"] = df[c].values
    for c in FLOATS:
        x = df[c].values.astype(float)
        cols[f"int_{c}"] = np.floor(x).astype(np.int64)
        cols[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
    return cols


def factorize_joint(tr_cols, te_cols, orig_cols, names):
    """Integer IDs shared across tr/te/orig for each column name."""
    codes = {}
    idf = {}
    n_all = None
    for n in names:
        a = np.concatenate([tr_cols[n], te_cols[n], orig_cols[n]])
        # unify dtype to str for mixed safety on cats; numerics stay numeric via astype object
        if a.dtype == object or a.dtype.kind in "SU":
            a = a.astype(str)
        codes_all, uniques = pd.factorize(a, sort=False)
        n_tr = len(tr_cols[n])
        n_te = len(te_cols[n])
        codes[n] = {
            "tr": codes_all[:n_tr].astype(np.int32),
            "te": codes_all[n_tr : n_tr + n_te].astype(np.int32),
            "orig": codes_all[n_tr + n_te :].astype(np.int32),
            "nuniq": int(len(uniques)),
        }
        # IDF from orig+train frequency of each code (avoid test leakage into idf: use tr+orig)
        cnt = np.bincount(np.concatenate([codes[n]["tr"], codes[n]["orig"]]), minlength=len(uniques))
        w = 1.0 / np.log1p(cnt.astype(np.float64))
        w[~np.isfinite(w)] = 0.0
        idf[n] = w.astype(np.float64)
    return codes, idf


def sim_vs_orig(tr_codes_dict, orig_idx, names, codes, idf, chunk=512):
    """Return (n_tr, n_orig_subset) similarity; computed in chunks over orig axis for memory."""
    n_tr = len(next(iter(tr_codes_dict.values())) if False else codes[names[0]]["tr"])
    n_o = len(orig_idx)
    # We'll compute max and top5 without storing full matrix when n_o large
    return n_tr, n_o


def _sim_vs_small(codes, idf, names, n_tr, orig_idx):
    """Dense (n_tr, |orig_idx|) IDF overlap — for small orig sets (fraud=339)."""
    m = len(orig_idx)
    sim = np.zeros((n_tr, m), dtype=np.float32)
    for n in names:
        tr = codes[n]["tr"]
        og = codes[n]["orig"][orig_idx]
        wt = idf[n][tr].astype(np.float32)
        sim += wt[:, None] * (tr[:, None] == og[None, :])
    return sim


def _max_sim_inverted(codes, idf, names, n_tr, orig_idx, max_df=200):
    """Max IDF overlap vs large orig subset via inverted lists.
    Values appearing in >max_df orig rows are skipped (common flags/cats); rare
    float/int matches dominate the NN score under IDF weighting anyway.
    """
    from collections import defaultdict

    inv = []
    kept_vals = 0
    dropped_vals = 0
    for n in names:
        d = defaultdict(list)
        oc = codes[n]["orig"]
        for local_j, oi in enumerate(orig_idx):
            d[int(oc[oi])].append(local_j)
        frozen = {}
        for k, v in d.items():
            if len(v) <= max_df:
                frozen[k] = np.asarray(v, dtype=np.int32)
                kept_vals += 1
            else:
                dropped_vals += 1
        inv.append(frozen)
    log(f"  inverted lists: kept_vals={kept_vals} dropped_common={dropped_vals} max_df={max_df}")

    n_o = len(orig_idx)
    score = np.zeros(n_o, dtype=np.float64)
    max_s = np.zeros(n_tr, dtype=np.float32)
    tr_cols = [codes[n]["tr"] for n in names]
    idf_cols = [idf[n] for n in names]

    for i in range(n_tr):
        touched = []
        for ci in range(len(names)):
            v = int(tr_cols[ci][i])
            js = inv[ci].get(v)
            if js is None:
                continue
            w = idf_cols[ci][v]
            for j in js:
                j = int(j)
                if score[j] == 0.0:
                    touched.append(j)
                score[j] += w
        if touched:
            tarr = np.asarray(touched, dtype=np.int32)
            max_s[i] = float(score[tarr].max())
            score[tarr] = 0.0
        if i > 0 and i % 100000 == 0:
            log(f"  inverted max-sim {i}/{n_tr}")
    return max_s


def compute_nn_features(codes, idf, names, y_orig, n_tr):
    """max_sim_fraud, mean_top5_fraud, max_sim_legit, orig_nn_llr for train rows."""
    fraud_idx = np.where(y_orig == 1)[0]
    legit_idx = np.where(y_orig == 0)[0]
    log(f"  orig fraud={len(fraud_idx)} legit={len(legit_idx)}")

    sim_f = _sim_vs_small(codes, idf, names, n_tr, fraud_idx)
    max_f = sim_f.max(axis=1)
    k = min(5, len(fraud_idx))
    top5_f = np.partition(sim_f, -k, axis=1)[:, -k:].mean(axis=1)
    del sim_f
    log("  fraud sims done")

    max_l = _max_sim_inverted(codes, idf, names, n_tr, legit_idx)
    log("  legit sims done")

    eps = 1e-6
    llr = np.log((max_f + eps) / (max_l + eps))
    return {
        "max_sim_orig_fraud": max_f,
        "mean_top5_sim_orig_fraud": top5_f,
        "max_sim_orig_legit": max_l,
        "orig_nn_llr": llr,
    }


def exp2(y, folds, keep7_O):
    log("EXP2: orig.csv seed-row prototype matching")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("original/credit_card_fraud_2026.csv")
    # align column names
    if "is_fraud" not in orig.columns:
        raise RuntimeError("orig missing is_fraud")
    y_orig = orig["is_fraud"].astype(int).values
    assert int(y_orig.sum()) == 339

    tr_c = build_token_cols(tr)
    te_c = build_token_cols(te)
    orig_c = build_token_cols(orig)
    names = list(tr_c.keys())
    log(f"  encoded {len(names)} token columns")
    codes, idf = factorize_joint(tr_c, te_c, orig_c, names)
    feats = compute_nn_features(codes, idf, names, y_orig, len(tr))

    out = {}
    for k, v in feats.items():
        a = float(roc_auc_score(y, v))
        out[k] = {"standalone_auc": a}
        log(f"  {k}: standalone AUC={a:.6f}")

    base_a, base_C, _ = best_nested(keep7_O, y, folds)
    log(f"  keep7 baseline: OOF={base_a:.6f} C={base_C}")
    out["keep7_baseline"] = {"oof": base_a, "C": base_C}

    print(f"\n{'add':35} {'OOF':>10} {'Δ':>10} keep?")
    for k, v in feats.items():
        O = np.column_stack([keep7_O, rank(v)])
        a, C, _ = best_nested(O, y, folds)
        keep = a > base_a
        print(f"+ {k:33} {a:10.6f} {a-base_a:+10.6f} {'KEEP' if keep else 'reject'}")
        out[f"+{k}"] = {"oof": a, "C": C, "delta": a - base_a, "keep": keep}

    # joint: llr + max_fraud
    O = np.column_stack([keep7_O, rank(feats["orig_nn_llr"]), rank(feats["max_sim_orig_fraud"])])
    a, C, _ = best_nested(O, y, folds)
    out["+llr_and_maxfraud"] = {"oof": a, "C": C, "delta": a - base_a, "keep": a > base_a}
    print(f"+ llr+maxfraud{'':22} {a:10.6f} {a-base_a:+10.6f} {'KEEP' if a>base_a else 'reject'}")

    # persist train features for later
    Path("preds").mkdir(exist_ok=True)
    np.savez_compressed(
        "preds/orig_nn_train_feats.npz",
        **{k: v.astype(np.float32) for k, v in feats.items()},
    )
    return out


# ── Exp 3 ──────────────────────────────────────────────────────────────────
def g_resid(groups, y, p_hat):
    """Residual G-test vs calibrated p_hat. Returns (G, df, G/df)."""
    # groups: int codes length n
    gmax = int(groups.max()) + 1
    O = np.bincount(groups, weights=y, minlength=gmax).astype(np.float64)
    E = np.bincount(groups, weights=p_hat, minlength=gmax).astype(np.float64)
    N = np.bincount(groups, minlength=gmax).astype(np.float64)
    # drop empty
    mask = N > 0
    O, E, N = O[mask], E[mask], N[mask]
    # clip for numerical stability
    E = np.clip(E, 1e-12, N - 1e-12)
    Oc = np.clip(O, 0, N)
    # cells with O=0 or O=N contribute via the other term; use where
    term1 = np.zeros_like(O)
    nz = Oc > 0
    term1[nz] = Oc[nz] * np.log(Oc[nz] / E[nz])
    term2 = np.zeros_like(O)
    z2 = (N - Oc) > 0
    term2[z2] = (N - Oc)[z2] * np.log((N - Oc)[z2] / (N - E)[z2])
    G = 2.0 * float((term1 + term2).sum())
    df = max(int(mask.sum()) - 1, 1)
    return G, df, G / df


def first_dec(x):
    s = pd.Series(np.floor(x).astype(np.int64).astype(str))
    return s.str[0].astype(int).values


def token_len_approx(x):
    """Digit-string length of float as written with default str — proxy for token length."""
    return pd.Series(x).astype(str).str.len().values.astype(np.int16)


def exp3(y, folds, keep7_O, keep7_nested):
    log("EXP3: residual G-test vs p7_keep7")
    # calibrate
    lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000)
    z = rlogit(rank(keep7_nested)).reshape(-1, 1)
    lr.fit(z, y)
    p_hat = lr.predict_proba(z)[:, 1]
    # scale so sum matches (isotonic-ish via ratio)
    p_hat = p_hat * (y.sum() / p_hat.sum())
    p_hat = np.clip(p_hat, 1e-8, 1 - 1e-8)
    log(f"  sum(p_hat)={p_hat.sum():.2f} sum(y)={y.sum()} calib AUC={roc_auc_score(y,p_hat):.6f}")

    tr = pd.read_csv("data/train.csv")
    orig = pd.read_csv("original/credit_card_fraud_2026.csv")

    keys = {}
    # float pair keys
    for a, b in combinations(FLOATS, 2):
        xa, xb = tr[a].values, tr[b].values
        keys[f"firstdec_{a}x{b}"] = (
            first_dec(xa) * 10 + first_dec(xb)
        ).astype(np.int32)
        ca = (np.round(xa * 100) % 100).astype(np.int64)
        cb = (np.round(xb * 100) % 100).astype(np.int64)
        keys[f"cents_{a}x{b}"] = (ca * 100 + cb).astype(np.int32)
        # l1: len*10+first digit of floor
        def l1(x):
            s = pd.Series(np.floor(x).astype(np.int64).astype(str))
            return (s.str.len().astype(int) * 10 + s.str[0].astype(int)).values

        keys[f"l1_{a}x{b}"] = (l1(xa) * 1000 + l1(xb)).astype(np.int32)
        keys[f"toklen_{a}x{b}"] = (
            token_len_approx(xa) * 100 + token_len_approx(xb)
        ).astype(np.int32)

    # CAT x CAT, FLAG x FLAG, CAT x FLAG
    for a, b in combinations(CATS, 2):
        ca = pd.factorize(tr[a].astype(str))[0]
        cb = pd.factorize(tr[b].astype(str))[0]
        keys[f"CAT_{a}x{b}"] = (ca * (cb.max() + 1) + cb).astype(np.int32)
    for a, b in combinations(FLAGS, 2):
        keys[f"FLAG_{a}x{b}"] = (tr[a].astype(np.int8).values * 2 + tr[b].astype(np.int8).values).astype(
            np.int32
        )
    for a in CATS:
        ca = pd.factorize(tr[a].astype(str))[0]
        for b in FLAGS:
            keys[f"CATxFLAG_{a}x{b}"] = (ca * 2 + tr[b].astype(np.int8).values).astype(np.int32)

    # 3-way: sample of CAT x CAT x FLAG and FLAG triples
    for a, b, c in list(combinations(CATS, 3))[:10]:
        ca = pd.factorize(tr[a].astype(str))[0]
        cb = pd.factorize(tr[b].astype(str))[0]
        cc = pd.factorize(tr[c].astype(str))[0]
        keys[f"CAT3_{a}x{b}x{c}"] = (
            (ca * (cb.max() + 1) + cb) * (cc.max() + 1) + cc
        ).astype(np.int32)
    for a, b, c in list(combinations(FLAGS, 3))[:10]:
        keys[f"FLAG3_{a}x{b}x{c}"] = (
            tr[a].astype(np.int8).values * 4
            + tr[b].astype(np.int8).values * 2
            + tr[c].astype(np.int8).values
        ).astype(np.int32)
    # CAT x FLAG x FLAG
    for a in CATS:
        ca = pd.factorize(tr[a].astype(str))[0]
        for b, c in list(combinations(FLAGS, 2))[:8]:
            keys[f"CxFxF_{a}x{b}x{c}"] = (
                ca * 4 + tr[b].astype(np.int8).values * 2 + tr[c].astype(np.int8).values
            ).astype(np.int32)

    # row-level summaries
    cents = np.column_stack([(np.round(tr[c].values * 100) % 100).astype(np.int64) for c in FLOATS])
    keys["num_zero_cents_in_row"] = ((cents == 0) | ((cents % 10) == 0)).sum(axis=1).astype(np.int32)
    keys["num_flags_set"] = tr[FLAGS].astype(np.int8).sum(axis=1).values.astype(np.int32)

    # values unseen in orig (exact on FEATS subset)
    unseen = np.zeros(len(tr), dtype=np.int32)
    for c in FEATS:
        if c in CATS or c in FLAGS:
            orig_set = set(orig[c].astype(str).values) if c in CATS else set(orig[c].astype(np.int8).values.tolist())
            if c in CATS:
                unseen += ~tr[c].astype(str).isin(orig_set).values
            else:
                unseen += ~np.isin(tr[c].astype(np.int8).values, list(orig_set))
        else:
            orig_set = set(np.round(orig[c].values.astype(float), 6))
            unseen += ~pd.Series(np.round(tr[c].values.astype(float), 6)).isin(orig_set).values
    keys["num_exact_vals_unseen_in_orig"] = unseen.astype(np.int32)

    # digit entropy of row (across float digit strings)
    def row_digit_entropy(df):
        ents = np.zeros(len(df))
        for i in range(0, len(df), 50000):
            sl = slice(i, i + 50000)
            parts = []
            for c in FLOATS:
                parts.append(df[c].iloc[sl].astype(str))
            cat = parts[0].str.cat(parts[1:], sep="")
            # count digit freq per row
            # vectorized-ish via list — keep simple
            for j, s in enumerate(cat.values):
                digits = [ch for ch in s if ch.isdigit()]
                if not digits:
                    continue
                _, cnts = np.unique(digits, return_counts=True)
                p = cnts / cnts.sum()
                ents[i + j] = -np.sum(p * np.log(p + 1e-12))
        return ents

    log("  computing digit entropy...")
    de = row_digit_entropy(tr)
    # bin entropy for G-test key
    keys["digit_entropy_bin"] = pd.qcut(de, q=20, labels=False, duplicates="drop").astype(np.int32)

    log(f"  scanning {len(keys)} keys...")
    scored = []
    for name, g in keys.items():
        # re-factorize to dense 0..K
        g2 = pd.factorize(g)[0].astype(np.int32)
        G, df, gdf = g_resid(g2, y, p_hat)
        scored.append((gdf, G, df, name, int(g2.max()) + 1))
    scored.sort(reverse=True)
    print(f"\nTop 20 keys by G_resid/df:")
    print(f"{'G/df':>10} {'G':>12} {'df':>6} {'ncell':>6}  key")
    top20 = []
    for gdf, G, df, name, ncell in scored[:20]:
        print(f"{gdf:10.3f} {G:12.2f} {df:6d} {ncell:6d}  {name}")
        top20.append({"key": name, "G_over_df": gdf, "G": G, "df": df, "ncell": ncell})

    # quick: add top-1 as a stacked member (rank of cell fraud rate residual)
    top_key = scored[0][3]
    g = pd.factorize(keys[top_key])[0]
    # cell residual rate
    gmax = g.max() + 1
    O = np.bincount(g, weights=y, minlength=gmax)
    E = np.bincount(g, weights=p_hat, minlength=gmax)
    N = np.bincount(g, minlength=gmax).astype(float)
    cell_resid = np.zeros(gmax)
    mask = N > 0
    cell_resid[mask] = (O[mask] - E[mask]) / np.sqrt(N[mask])
    feat = cell_resid[g]
    base_a, base_C, _ = best_nested(keep7_O, y, folds)
    Omat = np.column_stack([keep7_O, rank(feat)])
    a, C, _ = best_nested(Omat, y, folds)
    log(f"  +top_key[{top_key}] resid feature: OOF={a:.6f} Δ={a-base_a:+.6f}")

    return {
        "n_keys_scanned": len(keys),
        "top20": top20,
        "top_key_stack_add": {
            "key": top_key,
            "oof": a,
            "delta": a - base_a,
            "keep": a > base_a,
            "C": C,
        },
        "calib_auc": float(roc_auc_score(y, p_hat)),
    }


def main():
    log("Part8 proprietary scan — NO SUBMIT")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    folds = folds_y(y)

    P7_names, P7_O, P7_T, _ = build_p7_65(y)
    K_names, K_O, K_T = build_keep7(y, P7_O, P7_T, P7_names)
    log(f"P7 n={P7_O.shape[1]} keep7 n={K_O.shape[1]}")

    # nested keep7 for calibration
    keep7_a, keep7_C, keep7_fa = best_nested(K_O, y, folds)
    nest_keep7, _, _ = nested_cv(K_O, y, folds, C=keep7_C, tol=1e-8, max_iter=200)
    log(f"keep7 nested OOF={keep7_a:.6f} C={keep7_C}")
    np.save("preds/oof_p7_keep7_nested.npy", nest_keep7)

    r1, _ = exp1(y, folds, P7_O)
    r2 = exp2(y, folds, K_O)
    r3 = exp3(y, folds, K_O, nest_keep7)

    summary = {
        "keep7": {"oof": keep7_a, "C": keep7_C, "folds": list(map(float, keep7_fa))},
        "exp1_glm_orthogonal_residuals": r1,
        "exp2_orig_prototype_matching": r2,
        "exp3_residual_gtest": r3,
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

    outp = Path("logs/overnight/part8_proprietary_scan.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(conv(summary), indent=2))
    log(f"wrote {outp}")
    print("\n[done] NO_SUBMIT")


if __name__ == "__main__":
    main()
