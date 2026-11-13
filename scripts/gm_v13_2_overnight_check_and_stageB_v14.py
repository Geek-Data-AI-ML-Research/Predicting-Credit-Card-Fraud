#!/usr/bin/env python3
"""GM V13.2: overnight G26/upstream check + retrain δ1_v14 Stage-B on ATH 0.95388.

STRICT NO_SUBMIT=True. Zero pods. Local Mac CPU + Kaggle CLI only.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT is True

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
TMP = Path("/tmp/s6e11_v132")
RECON = ROOT / "external/nov12_095369_recon/datasets"
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
NOV13 = ROOT / "external/nov13_recon"
BANK = RECON / "goodpjw2008__s6e11-oof-bank"
Q17 = TMP / "rank1_q17_streams.npz"
if not Q17.exists():
    Q17 = ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(U(v))


def L_fold(v, fold_id):
    v = np.asarray(v, float).ravel()
    out = np.empty_like(v, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(v[m])
    return out


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def pchip_lock(src, ref):
    src = np.asarray(src, float).ravel()
    ref = np.asarray(ref, float).ravel()
    qs = np.linspace(0, 1, 401)
    x = np.quantile(src, qs)
    y = np.quantile(ref, qs)
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    return PchipInterpolator(x, y, extrapolate=True)(src)


def disp_ratio(a, b):
    return float(np.std(a) / (np.std(b) + 1e-12))


def npy(p):
    return np.load(p).astype(np.float64).ravel()


def kaggle(*args, timeout=90):
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    exe = ["kaggle"]
    try:
        return subprocess.check_output(exe + list(args), text=True, timeout=timeout, env=env)
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Step 1 — upstream
# ---------------------------------------------------------------------------
def step1_upstream() -> dict:
    log("STEP1: overnight upstream check")
    out = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "g26_dataset": "goodpjw2008/s6e11-target-first-gpt2-seed-2026",
        "g26_files_remote": None,
        "g26_has_test_5fold": False,
        "g26_has_test_f1_f4": False,
        "new_datasets": [],
        "top_kernels_by_date": [],
        "top_kernels_by_score": [],
        "downloaded": [],
        "notes": [],
    }
    files = kaggle("datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026")
    out["g26_files_remote"] = files
    low = files.lower()
    out["g26_has_test_5fold"] = "gpt2_s2026_test_5fold" in low or "test_5fold" in low
    out["g26_has_test_f1_f4"] = ("test_f1" in low) and ("test_f4" in low)
    if not out["g26_has_test_5fold"]:
        out["notes"].append(
            "G26 still ABSENT test_5fold / f1..f4 — only oof_5fold + test_f0 (stamp ~2026-11-12 09:14 UTC)."
        )

    ds_s6 = kaggle("datasets", "list", "-s", "s6e11", "--sort-by", "updated")
    ds_gpt = kaggle("datasets", "list", "-s", "gpt2", "--sort-by", "updated")
    out["datasets_s6e11_head"] = "\n".join(ds_s6.splitlines()[:25])
    out["datasets_gpt2_head"] = "\n".join(ds_gpt.splitlines()[:15])

    # flag fresh (Nov 13) s6e11 datasets
    for line in ds_s6.splitlines():
        if "2026-11-13" in line and "s6e11" in line.lower():
            out["new_datasets"].append(line.strip())

    kern_date = kaggle(
        "kernels",
        "list",
        "--competition",
        "playground-series-s6e11",
        "--sort-by",
        "dateRun",
        "--page-size",
        "35",
    )
    kern_score = kaggle(
        "kernels",
        "list",
        "--competition",
        "playground-series-s6e11",
        "--sort-by",
        "scoreDescending",
        "--page-size",
        "25",
    )
    out["top_kernels_by_date"] = [ln.strip() for ln in kern_date.splitlines()[2:12] if ln.strip()]
    out["top_kernels_by_score"] = [ln.strip() for ln in kern_score.splitlines()[2:12] if ln.strip()]

    # already downloaded in driver; record paths
    for tag, p in [
        ("alexchilton/s6e11-oof-predictions", NOV13 / "alexchilton_oof_predictions"),
        ("daoviet/s6e11-gp-dvae-orthogonal-apex", NOV13 / "daoviet_gp_dvae_apex_nov13"),
    ]:
        if p.exists() and any(p.iterdir()):
            out["downloaded"].append({"ref": tag, "path": str(p), "n_files": len(list(p.glob("*")))})

    # quick OOF screen of alexchilton vs ATH will happen after ATH built
    log(
        f"  G26 test_5fold={'YES' if out['g26_has_test_5fold'] else 'NO'} · "
        f"new Nov13 datasets={len(out['new_datasets'])} · downloads={len(out['downloaded'])}"
    )
    return out


# ---------------------------------------------------------------------------
# ATH 0.95388 OOF proxy (bold MIX + 0.03·d1)
# ---------------------------------------------------------------------------
def build_ath_proxy():
    log("Building ATH 0.95388 OOF proxy (bold MIX + 0.03·unitize(d1_v4))")
    Z = np.load(Q17, allow_pickle=False)
    y = Z["y_train"].astype(int)
    fold_id = Z["folds"].astype(int)
    names = [str(s) for s in Z["stream_names"]]
    XO = np.column_stack([L(Z[f"oof_{n}"]) for n in names])
    XT = np.column_stack([L(Z[f"test_{n}"]) for n in names])
    AT = L(Z["test_anchor_095370"])
    W_OOF = L(npy(TMP / "wide_lm_stack_nested.npy" if (TMP / "wide_lm_stack_nested.npy").exists() else ROOT / "external/nov08_part12/wide_lm_stack_nested.npy"))
    W_T = L(npy(TMP / "wide_lm_stack_test.npy" if (TMP / "wide_lm_stack_test.npy").exists() else ROOT / "external/nov08_part12/wide_lm_stack_test.npy"))

    def one(X):
        return np.column_stack([X, np.ones(len(X))])

    A_OOF = one(np.column_stack([XO, W_OOF])) @ np.linalg.lstsq(
        one(np.column_stack([XT, W_T])), AT, rcond=None
    )[0]
    REF = A_OOF + 0.06 * (XO[:, names.index("lm7_5fold")] - XO[:, names.index("megayak_5f_lm")])

    nb_path = TMP / "s6e11-lm-ladder-public-and-cv-up-lb-0-95386.ipynb"
    if not nb_path.exists():
        nb_path = NOV13 / "goodpjw_095386/s6e11-lm-ladder-public-and-cv-up-lb-0-95386.ipynb"
    nb = json.load(open(nb_path))
    src = "".join(nb["cells"][0]["source"])
    i = src.find("RECIPE_JSON = '''")
    j = src.find("'''", i + 17)
    bold_w = json.loads(src[i + 17 : j])["bold"]

    def mem_oof(k):
        if k.startswith("s:"):
            return L(Z[f"oof_{k[2:]}"])
        if k == "wide":
            return W_OOF
        if k.startswith("b:"):
            p = BANK / f"oof_{k[2:]}.npy"
            if not p.exists():
                # try alternate naming
                alts = list(BANK.glob(f"oof_{k[2:]}*.npy"))
                if not alts:
                    raise FileNotFoundError(p)
                p = alts[0]
            return L(npy(p))
        raise KeyError(k)

    def mem_test(k):
        if k.startswith("s:"):
            return L(Z[f"test_{k[2:]}"])
        if k == "wide":
            return W_T
        if k.startswith("b:"):
            p = BANK / f"test_{k[2:]}.npy"
            if not p.exists():
                alts = list(BANK.glob(f"test_{k[2:]}*.npy"))
                if not alts:
                    raise FileNotFoundError(p)
                p = alts[0]
            return L(npy(p))
        raise KeyError(k)

    # Drop missing bank keys gracefully
    usable = {}
    missing = []
    for k, w in bold_w.items():
        try:
            _ = mem_oof(k)
            _ = mem_test(k)
            usable[k] = w
        except Exception:
            missing.append(k)
    if missing:
        log(f"  bold keys missing ({len(missing)}): {missing[:8]}…")

    wsum = sum(usable.values())
    MIX_O = (1.0 - wsum) * REF
    MIX_T = (1.0 - wsum) * AT  # anchor already on test; bold moves from REF-like
    # Better test MIX: rebuild from test members around A_TEST≈AT
    A_T = AT  # test anchor is the public 0.95370 base used in Q17
    REF_T = A_T + 0.06 * (XT[:, names.index("lm7_5fold")] - XT[:, names.index("megayak_5f_lm")])
    MIX_T = (1.0 - wsum) * REF_T
    for k, w in usable.items():
        MIX_O = MIX_O + w * mem_oof(k)
        MIX_T = MIX_T + w * mem_test(k)

    d1o = npy(TMP / "d1_v4_oof.npy" if (TMP / "d1_v4_oof.npy").exists() else PRED / "gm_v4_polish/d1_v4_oof.npy")
    d1t = npy(TMP / "d1_v4_test.npy" if (TMP / "d1_v4_test.npy").exists() else PRED / "gm_v4_polish/d1_v4_test.npy")
    z_ATH_o = MIX_O + 0.03 * unitize(d1o)
    z_ATH_t = MIX_T + 0.03 * unitize(d1t)

    # Prefer LIVE test ATH if present (byte-faithful public)
    live = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
    if live.exists():
        z_ATH_t_live = L(pd.read_csv(live)["is_fraud"].to_numpy(float))
        # keep OOF as reconstructed; test locked to LIVE for staging
        z_ATH_t = z_ATH_t_live
        log(f"  locked test ATH to LIVE {live.name}")

    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    auc = float(roc_auc_score(y, z_ATH_o))
    log(f"  ATH OOF AUC={auc:.7f} · bold_keys_used={len(usable)}/{len(bold_w)} · missing={len(missing)}")
    return {
        "y": y,
        "fold_id": fold_id,
        "folds": folds,
        "z_ATH_o": z_ATH_o.astype(np.float64),
        "z_ATH_t": z_ATH_t.astype(np.float64),
        "d1o": unitize(d1o),
        "d1t": unitize(d1t),
        "MIX_O": MIX_O,
        "Z": Z,
        "names": names,
        "XO": XO,
        "XT": XT,
        "missing_bold": missing,
        "auc_ath_oof": auc,
    }


# ---------------------------------------------------------------------------
# X_v14 feature matrix
# ---------------------------------------------------------------------------
def bayes_from_ll(ll, prior=None):
    """ll: (N,K,2) loglik → bayes posterior P(fraud)."""
    ll = np.asarray(ll, float)
    assert ll.ndim == 3 and ll.shape[-1] == 2
    # log p(y|k) + log prior → normalize over y then average over k (logsumexp)
    if prior is None:
        prior = np.array([0.5, 0.5])
    log_prior = np.log(prior + 1e-12)
    # (N,K,2)
    joint = ll + log_prior.reshape(1, 1, 2)
    # normalize over class
    m = joint.max(axis=-1, keepdims=True)
    ex = np.exp(joint - m)
    post = ex / ex.sum(axis=-1, keepdims=True)
    # average over orders
    return post[:, :, 1].mean(axis=1)


def cross_modal_denoise(target_o, target_t, Xo, Xt, fold_id, y):
    """5-fold Ridge surrogate; PCHIP-lock test to OOF; return denoised contrast with disp≈1."""
    pred_o = np.zeros(len(target_o), float)
    pred_t_acc = []
    for f in range(5):
        te = fold_id == f
        tr = ~te
        ridge = Ridge(alpha=10.0)
        ridge.fit(Xo[tr], target_o[tr])
        pred_o[te] = ridge.predict(Xo[te])
        pred_t_acc.append(ridge.predict(Xt))
    pred_t = np.mean(pred_t_acc, axis=0)
    # blend noisy with surrogate (a*=0.45 default; OOF target is clean so blend mild)
    a = 0.45
    den_o = unitize(a * unitize(target_o) + (1 - a) * unitize(pred_o))
    den_t = unitize(a * unitize(target_t) + (1 - a) * unitize(pred_t))
    den_t = pchip_lock(den_t, den_o)
    den_t = unitize(den_t) * (den_o.std() + 1e-12) + den_o.mean()  # force disp match
    # for folds without real target (e.g. L7_s2026 only f0): replace non-f0 with surrogate
    return den_o, den_t, float(disp_ratio(den_t, den_o))


def build_Xv14(pack):
    log("Building X_v14 feature matrix")
    y, fold_id = pack["y"], pack["fold_id"]
    Z, names = pack["Z"], pack["names"]

    def stream_o(n):
        return L_fold(Z[f"oof_{n}"], fold_id)

    def stream_t(n):
        return L(Z[f"test_{n}"])

    # Core modern streams (Q17 / recon)
    MS_o = stream_o("masaya_gpt2")
    MS_t = stream_t("masaya_gpt2")
    LM_o = stream_o("megayak_5f_lm")
    LM_t = stream_t("megayak_5f_lm")
    L7F_o = stream_o("lm7_5fold")
    L7F_t = stream_t("lm7_5fold")
    GPD_o = stream_o("daoviet_gpdvae")
    GPD_t = stream_t("daoviet_gpdvae")
    # refresh daoviet if nov13 copy differs — prefer Q17 for fold-align; optionally blend
    MLP_o = stream_o("gbn_bag5_w512")
    MLP_t = stream_t("gbn_bag5_w512")
    G26_o = stream_o("gpt2_seed2026")
    G26_t = L(npy(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"))
    # G_med: gpt2 med if present else 0.5*(G26+MS)
    gmed_o_path = Path("/tmp/gm_deep_tech/preds/gm_v10_hardened/gpt2_med_5f_oof_raw.npy")
    gmed_t_path = Path("/tmp/gm_deep_tech/preds/gm_v10_hardened/gpt2_med_5f_test_raw.npy")
    if gmed_o_path.exists() and gmed_t_path.exists():
        Gmed_o = L_fold(npy(gmed_o_path), fold_id)
        Gmed_t = L(npy(gmed_t_path))
        gmed_src = "gpt2_med_5f"
    else:
        Gmed_o = 0.5 * (G26_o + MS_o)
        Gmed_t = 0.5 * (G26_t + MS_t)
        gmed_src = "0.5*(G26+MS)_proxy"

    # Tree mean from shallow/gb family
    Tree_o = 0.5 * (stream_o("megayak_shallow_tree") + stream_o("gb_bag5"))
    Tree_t = 0.5 * (stream_t("megayak_shallow_tree") + stream_t("gb_bag5"))

    # Contrasts (per-fold L already applied)
    feats_o = [
        unitize(MS_o - LM_o),
        unitize(L7F_o - LM_o),
        unitize(GPD_o - MLP_o),
        unitize(Gmed_o - MS_o),
    ]
    feats_t = [
        unitize(pchip_lock(MS_t - LM_t, MS_o - LM_o)),
        unitize(pchip_lock(L7F_t - LM_t, L7F_o - LM_o)),
        unitize(pchip_lock(GPD_t - MLP_t, GPD_o - MLP_o)),
        unitize(pchip_lock(Gmed_t - MS_t, Gmed_o - MS_o)),
    ]
    feat_names = ["MS-LM", "L7F-LM", "GPD-MLP", "Gmed-MS"]

    stack_o = np.column_stack([MS_o, LM_o, L7F_o, GPD_o])
    stack_t = np.column_stack([MS_t, LM_t, L7F_t, GPD_t])
    feats_o.append(unitize(stack_o.std(1)))
    feats_t.append(unitize(pchip_lock(stack_t.std(1), stack_o.std(1))))
    feat_names.append("std(MS,LM,L7F,GPD)")
    lm_mean_o = stack_o[:, :3].mean(1)
    lm_mean_t = stack_t[:, :3].mean(1)
    feats_o.append(unitize(np.abs(lm_mean_o - Tree_o)))
    feats_t.append(unitize(pchip_lock(np.abs(lm_mean_t - Tree_t), np.abs(lm_mean_o - Tree_o))))
    feat_names.append("abs(LM_mean-Tree_mean)")

    # P4 pure5f
    for tag in ("L4.2", "L4.3", "L4.5"):
        op = PRED / f"matrix_v2/P4_{tag}_pure5f_oof.npy"
        tp = PRED / f"matrix_v2/P4_{tag}_pure5f_test.npy"
        if op.exists() and tp.exists():
            oo = L_fold(npy(op), fold_id)
            tt = L(npy(tp))
            feats_o.append(unitize(oo))
            feats_t.append(unitize(pchip_lock(tt, oo)))
            feat_names.append(f"P4_{tag}_pure5f")

    # KMAT_top30 from X_v5[:,0:68]
    Xv5o = np.load(PRED / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
    Xv5t = np.load(PRED / "gm_v5_overnight/X_v5_test.npy", mmap_mode="r")
    meta = json.loads((PRED / "gm_v5_overnight/X_v5_meta.json").read_text())
    top15 = list(meta.get("top15_keys", list(range(15))))
    rest = [j for j in range(68) if j not in top15]
    kmat_idx = (top15 + rest)[:30]
    for col in kmat_idx:
        oo = np.asarray(Xv5o[:, col], dtype=np.float64)
        tt = np.asarray(Xv5t[:, col], dtype=np.float64)
        feats_o.append(unitize(oo))
        feats_t.append(unitize(tt))
        feat_names.append(f"KMAT_{col}")

    # Surrogate design matrix for denoise (lightweight)
    Xo_sur = np.column_stack(feats_o[:8]).astype(np.float64)
    Xt_sur = np.column_stack(feats_t[:8]).astype(np.float64)

    # G26-MS denoise (full OOF exists; test is f0-only → PCHIP + Ridge)
    g26ms_o = unitize(G26_o - MS_o)
    g26ms_t = unitize(G26_t - MS_t)
    den_g_o, den_g_t, disp_g = cross_modal_denoise(g26ms_o, g26ms_t, Xo_sur, Xt_sur, fold_id, y)
    feats_o.append(unitize(den_g_o))
    feats_t.append(unitize(den_g_t))
    feat_names.append("denoise(G26-MS)")
    log(f"  G26-MS denoise disp(test/oof)={disp_g:.4f}")

    # L7_s2026_K10 − L7F (fold0 only real; denoise fills rest)
    l7p = PRED / "l7_s2026_5f"
    l7_note = "absent"
    if (l7p / "oof_0.npy").exists() and (l7p / "test_0.npy").exists():
        ll0 = np.load(l7p / "oof_0.npy")  # (fold0_n, 10, 2)
        llt = np.load(l7p / "test_0.npy")
        # prevalence prior from fold0 labels
        m0 = fold_id == 0
        prior = np.array([1 - y[m0].mean(), y[m0].mean()])
        bay0 = bayes_from_ll(ll0, prior)
        bayt = bayes_from_ll(llt, prior)
        L7s_o = np.zeros(len(y), float)
        L7s_o[m0] = L(bay0)
        # fill other folds with L7F as placeholder before denoise
        L7s_o[~m0] = L7F_o[~m0]
        L7s_t = L(bayt)
        contrast_o = unitize(L7s_o - L7F_o)
        # on non-f0, contrast is ~0 by construction → denoise will learn f0 pattern
        contrast_t = unitize(L7s_t - L7F_t)
        # mask: train denoise primarily on fold0 by upweighting — use Ridge on all with soft target
        den_l_o, den_l_t, disp_l = cross_modal_denoise(contrast_o, contrast_t, Xo_sur, Xt_sur, fold_id, y)
        # blend: keep real fold0 contrast, surrogate elsewhere
        final_o = den_l_o.copy()
        final_o[m0] = contrast_o[m0]
        final_o = unitize(final_o)
        final_t = unitize(pchip_lock(den_l_t, final_o))
        disp_l = disp_ratio(final_t, final_o)
        feats_o.append(final_o)
        feats_t.append(final_t)
        feat_names.append("denoise(L7_s2026_K10-L7F)")
        l7_note = f"fold0_only+denoise disp={disp_l:.4f}"
        log(f"  L7_s2026_K10 denoise {l7_note}")
    else:
        disp_l = None
        log("  L7_s2026 artifacts missing — skipping K10 contrast")

    Xo = np.column_stack(feats_o).astype(np.float32)
    Xt = np.column_stack(feats_t).astype(np.float32)
    # replace any nan/inf
    Xo = np.nan_to_num(Xo, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"  X_v14 shape o={Xo.shape} t={Xt.shape} · gmed={gmed_src}")
    return {
        "Xo": Xo,
        "Xt": Xt,
        "feat_names": feat_names,
        "disp_g26": disp_g,
        "disp_l7": disp_l,
        "gmed_src": gmed_src,
        "l7_note": l7_note,
    }


# ---------------------------------------------------------------------------
# Train Stage-B δ1_v14
# ---------------------------------------------------------------------------
def train_stageB(pack, feats):
    log("Training δ1_v14 Stage-B (lgb+xgb+cat) init_score=z_ATH")
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    y, folds, fold_id = pack["y"], pack["folds"], pack["fold_id"]
    z0 = pack["z_ATH_o"]
    z0t = pack["z_ATH_t"]
    Xo, Xt = feats["Xo"], feats["Xt"]

    params_lgb = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        max_depth=3,
        num_leaves=6,
        min_child_samples=500,
        feature_fraction=0.35,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=80.0,
        learning_rate=0.015,
        verbosity=-1,
    )
    oof_lgb = np.zeros(len(y), float)
    oof_xgb = np.zeros(len(y), float)
    oof_cat = np.zeros(len(y), float)
    te_lgb, te_xgb, te_cat = [], [], []

    for f, (tr, te) in enumerate(folds):
        log(f"  fold {f} fit…")
        # LGB
        dtr = lgb.Dataset(Xo[tr], label=y[tr], init_score=z0[tr])
        model = lgb.train(params_lgb, dtr, num_boost_round=160)
        oof_lgb[te] = model.predict(Xo[te], raw_score=True)
        te_lgb.append(model.predict(Xt, raw_score=True))

        # XGB
        dtrain = xgb.DMatrix(Xo[tr], label=y[tr], base_margin=z0[tr])
        dval = xgb.DMatrix(Xo[te], label=y[te], base_margin=z0[te])
        dtest = xgb.DMatrix(Xt, base_margin=z0t)
        params_x = dict(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=3,
            min_child_weight=500,
            colsample_bytree=0.35,
            subsample=0.8,
            lambda_=80.0,
            eta=0.015,
            tree_method="hist",
            verbosity=0,
        )
        # xgboost uses 'lambda' key
        params_x["lambda"] = params_x.pop("lambda_")
        bx = xgb.train(params_x, dtrain, num_boost_round=160)
        # raw margin predictions
        oof_xgb[te] = bx.predict(dval, output_margin=True)
        te_xgb.append(bx.predict(dtest, output_margin=True))

        # CatBoost — baseline = init logodds
        cat = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            depth=3,
            min_data_in_leaf=500,
            rsm=0.35,
            l2_leaf_reg=80.0,
            learning_rate=0.015,
            iterations=160,
            verbose=False,
            random_seed=42 + f,
            thread_count=2,
        )
        # CatBoost baseline expects shape (n,1) for Logloss in recent versions
        bl_tr = z0[tr].reshape(-1, 1)
        bl_te = z0[te].reshape(-1, 1)
        bl_tt = z0t.reshape(-1, 1)
        try:
            cat.fit(Xo[tr], y[tr], baseline=bl_tr)
            # predict raw / approx
            oof_cat[te] = cat.predict(Xo[te], prediction_type="RawFormulaVal", baseline=bl_te)
            te_cat.append(cat.predict(Xt, prediction_type="RawFormulaVal", baseline=bl_tt))
        except TypeError:
            # older API without baseline on predict
            cat.fit(Xo[tr], y[tr])
            oof_cat[te] = cat.predict(Xo[te], prediction_type="RawFormulaVal")
            te_cat.append(cat.predict(Xt, prediction_type="RawFormulaVal"))

    t_lgb = np.mean(te_lgb, 0)
    t_xgb = np.mean(te_xgb, 0)
    t_cat = np.mean(te_cat, 0)

    # residual vs init (model may return absolute margin ≈ z0 + δ)
    def resid(oof, base):
        r = oof - base
        if np.std(r) < 1e-9:
            # already residual-like or collapsed — use centered oof
            r = unitize(oof)
        return unitize(r)

    d_lgb_o, d_lgb_t = resid(oof_lgb, z0), resid(t_lgb, z0t)
    d_xgb_o, d_xgb_t = resid(oof_xgb, z0), resid(t_xgb, z0t)
    d_cat_o, d_cat_t = resid(oof_cat, z0), resid(t_cat, z0t)

    d1_v14_o = unitize((d_lgb_o + d_xgb_o + d_cat_o) / 3.0)
    d1_v14_t = unitize((d_lgb_t + d_xgb_t + d_cat_t) / 3.0)
    d1_v14_t = unitize(pchip_lock(d1_v14_t, d1_v14_o))

    blend_o = unitize(0.5 * pack["d1o"] + 0.5 * d1_v14_o)
    blend_t = unitize(0.5 * pack["d1t"] + 0.5 * d1_v14_t)
    blend_t = unitize(pchip_lock(blend_t, blend_o))

    art = TMP / "out"
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "d1_v14_oof.npy", d1_v14_o.astype(np.float32))
    np.save(art / "d1_v14_test.npy", d1_v14_t.astype(np.float32))
    np.save(art / "d1_v14_lgb_oof.npy", d_lgb_o.astype(np.float32))
    np.save(art / "d1_v14_xgb_oof.npy", d_xgb_o.astype(np.float32))
    np.save(art / "d1_v14_cat_oof.npy", d_cat_o.astype(np.float32))
    # also persist under preds/
    pout = PRED / "gm_v13_2_stageB"
    pout.mkdir(parents=True, exist_ok=True)
    np.save(pout / "d1_v14_oof.npy", d1_v14_o.astype(np.float32))
    np.save(pout / "d1_v14_test.npy", d1_v14_t.astype(np.float32))
    np.save(pout / "d1_blend_v4_v14_oof.npy", blend_o.astype(np.float32))
    np.save(pout / "d1_blend_v4_v14_test.npy", blend_t.astype(np.float32))

    log("  δ1_v14 saved")
    return {
        "d1_v14_o": d1_v14_o,
        "d1_v14_t": d1_v14_t,
        "blend_o": blend_o,
        "blend_t": blend_t,
        "parts": {
            "lgb": (d_lgb_o, d_lgb_t),
            "xgb": (d_xgb_o, d_xgb_t),
            "cat": (d_cat_o, d_cat_t),
        },
    }


# ---------------------------------------------------------------------------
# Dual gate eval
# ---------------------------------------------------------------------------
def fold_deltas(y, folds, base, cand):
    d = []
    for _, te in folds:
        d.append(1e5 * (roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])))
    return np.asarray(d, float)


def summarize(d, tag):
    pos = int((d > 0).sum())
    rng = np.random.default_rng(42)
    boots = [d[rng.integers(0, 5, size=5)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "tag": tag,
        "pos5": pos,
        "mean_e5": float(d.mean()),
        "min_e5": float(d.min()),
        "folds": [round(float(x), 3) for x in d],
        "boot_mean_90ci": [round(float(lo), 3), round(float(hi), 3)],
        "pass_main_5of5": bool(pos == 5 and d.min() > 0),
    }


def nested_subfold(y, fold_id, base, cand, n_need=3):
    sf_stats = []
    for f in range(5):
        m = fold_id == f
        yf, bf, cf = y[m], base[m], cand[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        deltas = []
        idx = np.arange(len(yf))
        for _, te in skf.split(idx, yf):
            deltas.append(1e5 * (roc_auc_score(yf[te], cf[te]) - roc_auc_score(yf[te], bf[te])))
        deltas = np.asarray(deltas)
        sf_stats.append(
            {
                "main_fold": f,
                "pos5": int((deltas > 0).sum()),
                "mean_e5": float(deltas.mean()),
                "min_e5": float(deltas.min()),
                "folds": [round(float(x), 3) for x in deltas],
            }
        )
    n_clear = sum(1 for s in sf_stats if s["pos5"] == 5 and s["min_e5"] > 0)
    return {
        "n_mainfolds_subfold_5of5": n_clear,
        "per_main_fold": sf_stats,
        "pass_nested": n_clear >= n_need,
    }


def evaluate_candidates(pack, deltas):
    log("Dual-gate evaluation vs ATH")
    y, folds, fold_id = pack["y"], pack["folds"], pack["fold_id"]
    z0 = pack["z_ATH_o"]
    # weight grid for inject
    cands = {}
    for w in (0.02, 0.03, 0.035, 0.04):
        cands[f"ATH+d1_v14_w{w}"] = z0 + w * deltas["d1_v14_o"]
        cands[f"ATH+blend_v4v14_w{w}"] = z0 + w * deltas["blend_o"]
    # Correct d1_v4 retune vs ATH: ATH already = MIX+0.03·d1, so w=0.035 is +0.005·d1
    cands["d1_v4_retune_w0.035_vs_ATH"] = z0 + 0.005 * pack["d1o"]
    cands["d1_v4_retune_w0.04_vs_ATH"] = z0 + 0.010 * pack["d1o"]

    rows = []
    for tag, cand in cands.items():
        # if tag is ATH+d1_v4_w0.03, delta≈0 vs ATH — skip noise
        d = fold_deltas(y, folds, z0, cand)
        s = summarize(d, tag)
        nest = nested_subfold(y, fold_id, z0, cand, n_need=3)
        s["nested_SF"] = nest["n_mainfolds_subfold_5of5"]
        s["pass_nested"] = nest["pass_nested"]
        s["nested_detail"] = nest
        s["pass_dual"] = bool(s["pass_main_5of5"] and s["pass_nested"])
        rows.append(s)
        log(
            f"  {tag}: main {s['pos5']}/5 mean={s['mean_e5']:.3f} nested={s['nested_SF']}/5 "
            f"dual={s['pass_dual']}"
        )
    return rows


def maybe_stage(pack, deltas, rows):
    """Stage NOSUBMIT only if beats d1_w0.035 on meanΔ AND nested SF."""
    base_row = next(r for r in rows if "d1_v4_retune_w0.035" in r["tag"])
    # pick best dual-pass among v14 / blend
    dual = [r for r in rows if r["pass_dual"] and "v14" in r["tag"]]
    staged = None
    if dual:
        # require beat d1_w0.035 on BOTH mean and nested
        beaters = [
            r
            for r in dual
            if r["mean_e5"] > base_row["mean_e5"] and r["nested_SF"] > base_row["nested_SF"]
        ]
        if beaters:
            best = max(beaters, key=lambda r: (r["nested_SF"], r["mean_e5"]))
            # parse weight
            w = float(best["tag"].split("w")[-1])
            use_blend = "blend" in best["tag"]
            u = deltas["blend_t"] if use_blend else deltas["d1_v14_t"]
            z_t = pack["z_ATH_t"] + w * unitize(u)
            # force to LIVE ATH probability law
            live = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
            ids = pd.read_csv(live)["id"].to_numpy()
            ref_p = pd.read_csv(live)["is_fraud"].to_numpy(float)
            order = np.argsort(z_t + 1e-15 * (rankdata(z_t, method="ordinal") / len(z_t)), kind="mergesort")
            out_p = np.empty(len(z_t), float)
            out_p[order] = np.sort(ref_p)
            path = SUB / "sub_v13p2_d1_v14_verified_NOSUBMIT.csv"
            assert NO_SUBMIT and "NOSUBMIT" in path.name
            pd.DataFrame({"id": ids, "is_fraud": out_p}).to_csv(path, index=False)
            staged = {
                "path": str(path),
                "tag": best["tag"],
                "mean_e5": best["mean_e5"],
                "nested_SF": best["nested_SF"],
                "vs_d1_035_mean": best["mean_e5"] - base_row["mean_e5"],
                "vs_d1_035_nested": best["nested_SF"] - base_row["nested_SF"],
            }
            log(f"  STAGED {path.name} ({best['tag']})")
        else:
            log("  dual-pass exists but does NOT beat d1_w0.035 on mean+nested — no stage")
    else:
        log("  no dual-pass v14 candidate — no stage")
    return staged, base_row


def screen_alexchilton(pack):
    """Quick OOF AUC of new alexchilton members (informational)."""
    d = NOV13 / "alexchilton_oof_predictions"
    if not d.exists():
        return []
    y = pack["y"]
    rows = []
    for p in sorted(d.glob("oof_*.npy"))[:30]:
        try:
            a = npy(p)
            if a.shape != y.shape:
                continue
            rows.append({"file": p.name, "auc": float(roc_auc_score(y, a))})
        except Exception:
            continue
    rows.sort(key=lambda r: -r["auc"])
    return rows[:10]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(upstream, pack, feats, rows, staged, alex_rows):
    LOG.mkdir(parents=True, exist_ok=True)
    md = LOG / "GM_V13_2_STAGEB_V14_AND_UPSTREAM_REPORT.md"
    js = LOG / "GM_V13_2_STAGEB_V14_AND_UPSTREAM_REPORT.json"

    best_v14 = max(
        (r for r in rows if "d1_v14" in r["tag"] and "blend" not in r["tag"]),
        key=lambda r: (r["pass_dual"], r["nested_SF"], r["mean_e5"]),
        default=None,
    )
    best_blend = max(
        (r for r in rows if "blend" in r["tag"]),
        key=lambda r: (r["pass_dual"], r["nested_SF"], r["mean_e5"]),
        default=None,
    )
    base035 = next(r for r in rows if "d1_v4_retune_w0.035" in r["tag"])

    lines = []
    lines.append("# GM V13.2 — Overnight Upstream + δ1_v14 Stage-B on ATH 0.95388\n\n")
    lines.append(f"**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed {time.time()-T0:.1f}s\n\n")
    lines.append("## Step 1 — Upstream\n\n")
    lines.append(
        f"- **G26 `gpt2_s2026_test_5fold` / f1–f4:** "
        f"{'PRESENT' if upstream['g26_has_test_5fold'] else '**ABSENT**'} "
        f"(remote still `oof_5fold` + `test_f0` only).\n"
    )
    lines.append(f"- Nov-13-updated s6e11 datasets spotted: **{len(upstream['new_datasets'])}**\n")
    for nd in upstream["new_datasets"][:8]:
        lines.append(f"  - `{nd}`\n")
    lines.append("- Downloaded into `external/nov13_recon/`:\n")
    for d in upstream["downloaded"]:
        lines.append(f"  - `{d['ref']}` → `{d['path']}` ({d['n_files']} files)\n")
    lines.append("\n### Top kernels (dateRun)\n\n")
    for k in upstream["top_kernels_by_date"][:8]:
        lines.append(f"- {k}\n")
    lines.append("\n### Top kernels (scoreDescending)\n\n")
    for k in upstream["top_kernels_by_score"][:8]:
        lines.append(f"- {k}\n")
    if alex_rows:
        lines.append("\n### alexchilton OOF screen (top AUC)\n\n")
        lines.append("| file | OOF AUC |\n|---|---:|\n")
        for r in alex_rows:
            lines.append(f"| `{r['file']}` | {r['auc']:.7f} |\n")
        lines.append(
            f"\nATH OOF AUC for reference: **{pack['auc_ath_oof']:.7f}** "
            f"(alex members are base learners, not ATH-level).\n"
        )

    lines.append("\n## Step 2 — δ1_v14 Stage-B on ATH 0.95388\n\n")
    lines.append(
        f"- ATH OOF proxy: `bold MIX + 0.03·unitize(d1_v4)` · AUC={pack['auc_ath_oof']:.7f}\n"
        f"- X_v14: **{feats['Xo'].shape[1]}** cols "
        f"(KMAT_top30 + P4_L4.2/3/5 + stream contrasts + disagreement + denoise axes)\n"
        f"- G26-MS denoise disp(test/oof)={feats['disp_g26']:.4f}\n"
        f"- L7_s2026: {feats['l7_note']}\n"
        f"- G_med source: `{feats['gmed_src']}`\n"
        f"- Boosters: lgb+xgb+cat · `max_depth=3, num_leaves=6, min_child=500, "
        f"colsample=0.35, λ=80, lr=0.015, n=160` · `init_score=z_ATH`\n"
        f"- Nested gate bar: **≥3/5** main folds with subfold 5/5\n\n"
    )

    lines.append("### Dual-gate scorecard\n\n")
    lines.append(
        "| candidate | 5/5 | meanΔ | min | 90% boot CI | nested SF | dual |\n"
        "|---|---:|---:|---:|---|---:|---|\n"
    )
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {r['pos5']} | {r['mean_e5']:.3f} | {r['min_e5']:.3f} | "
            f"{r['boot_mean_90ci']} | {r['nested_SF']}/5 | {r['pass_dual']} |\n"
        )

    lines.append("\n### vs `d1_w0.035` baseline (failed nested in V13.1)\n\n")
    lines.append(
        f"- `ATH+d1_v4_w0.035`: mean={base035['mean_e5']:.3f}e-5 · nested={base035['nested_SF']}/5\n"
    )
    if best_v14:
        lines.append(
            f"- best `d1_v14`: `{best_v14['tag']}` mean={best_v14['mean_e5']:.3f} · "
            f"nested={best_v14['nested_SF']}/5 · dual={best_v14['pass_dual']}\n"
        )
    if best_blend:
        lines.append(
            f"- best blend 0.5·v4+0.5·v14: `{best_blend['tag']}` mean={best_blend['mean_e5']:.3f} · "
            f"nested={best_blend['nested_SF']}/5 · dual={best_blend['pass_dual']}\n"
        )

    lines.append("\n## Staging decision\n\n")
    if staged:
        lines.append(
            f"**STAGED** `{staged['path']}` — `{staged['tag']}` "
            f"(Δmean vs d1_035={staged['vs_d1_035_mean']:+.3f}, "
            f"Δnested={staged['vs_d1_035_nested']:+d}). Still **NOSUBMIT**.\n"
        )
    else:
        lines.append(
            "**NOT STAGED** — no `d1_v14` / blend candidate cleared dual gate **and** "
            "beat `d1_w0.035` on both meanΔ and nested SF. "
            "`submissions/sub_v13p2_d1_v14_verified_NOSUBMIT.csv` was **not** written.\n"
        )

    lines.append("\n## Bottom line\n\n")
    any_dual = any(r["pass_dual"] and "v14" in r["tag"] for r in rows)
    if any_dual and staged:
        lines.append(
            "Retraining Stage-B on the live ATH base **clears nested** where `δ1_v4` retunes did not; "
            "candidate staged for optional later submit.\n"
        )
    elif any_dual:
        lines.append(
            "Some `δ1_v14` configs clear dual gate but do not dominate `d1_w0.035` on both "
            "mean and nested — hold.\n"
        )
    else:
        lines.append(
            "`δ1_v14` on ATH did **not** clear the dual main+nested bar at the required margin. "
            "G26 5-fold test still missing — highest external lever unchanged.\n"
        )

    md.write_text("".join(lines))
    payload = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "elapsed_s": round(time.time() - T0, 2),
        "upstream": {k: v for k, v in upstream.items() if k not in ("g26_files_remote",)},
        "g26_files_remote": upstream.get("g26_files_remote"),
        "ath_auc_oof": pack["auc_ath_oof"],
        "X_v14_cols": feats["Xo"].shape[1],
        "feat_names": feats["feat_names"],
        "disp_g26": feats["disp_g26"],
        "disp_l7": feats["disp_l7"],
        "rows": [{k: v for k, v in r.items() if k != "nested_detail"} for r in rows],
        "staged": staged,
        "alexchilton_top": alex_rows,
    }
    # attach nested detail separately (smaller)
    payload["nested_detail"] = {r["tag"]: r.get("nested_detail") for r in rows}
    js.write_text(json.dumps(payload, indent=2, default=str))
    log(f"WROTE {md}")
    return md


def main():
    assert NO_SUBMIT
    TMP.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    log("GM V13.2 start · NO_SUBMIT=True · POD_TOUCHES=0")

    upstream = step1_upstream()
    pack = build_ath_proxy()
    alex_rows = screen_alexchilton(pack)
    feats = build_Xv14(pack)
    deltas = train_stageB(pack, feats)
    rows = evaluate_candidates(pack, deltas)
    staged, _ = maybe_stage(pack, deltas, rows)
    write_report(upstream, pack, feats, rows, staged, alex_rows)
    log("DONE")


if __name__ == "__main__":
    main()
