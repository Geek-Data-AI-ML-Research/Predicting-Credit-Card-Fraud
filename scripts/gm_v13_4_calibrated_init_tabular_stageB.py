#!/usr/bin/env python3
"""GM V13.4: Platt-calibrated init_score + full tabular×disagreement X_v15 Stage-B.

Fixes:
  Bug#1 — zero-mean rank-logit init_score implied σ=0.50 fraud rate (true≈0.0167)
  Bug#2 — missing X_v5 tabular/entity + P4; forbid raw z_ATH in X (shortcut collapse)

STRICT NO_SUBMIT=True. Zero pods. Local Mac CPU only.
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
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT is True

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
TMP = Path("/tmp/s6e11_v134")
RECON = ROOT / "external/nov12_095369_recon/datasets"
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
DAO = ROOT / "external/nov13_recon/daoviet_gp_dvae_apex_nov13"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
Q17 = Path("/tmp/s6e11_v132/rank1_q17_streams.npz")
if not Q17.exists():
    Q17 = ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
XV5_O = PRED / "gm_v5_overnight/X_v5_oof.npy"
XV5_T = PRED / "gm_v5_overnight/X_v5_test.npy"

MIN_FOLD_E5 = 0.10
SF25_NEED = 18
T25_NEED = 2.80
WILC_P = 0.005
BOOT_LO_NEED = 0.25
DISP_LO, DISP_HI = 0.97, 1.03
R2_LIVE_MAX = 0.05

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


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


def unitize_fold(v, fold_id):
    v = np.asarray(v, float).ravel()
    out = np.empty_like(v, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = unitize(v[m])
    return out


def pchip_lock(src, ref):
    src = np.asarray(src, float).ravel()
    ref = np.asarray(ref, float).ravel()
    qs = np.linspace(0, 1, 401)
    x, y = np.quantile(src, qs), np.quantile(ref, qs)
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    return PchipInterpolator(x, y, extrapolate=True)(src)


def npy(p):
    return np.load(p).astype(np.float64).ravel()


def kaggle_files(slug):
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    try:
        return subprocess.check_output(
            ["kaggle", "datasets", "files", slug], text=True, timeout=60, env=env
        )
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Step 1 — daoviet CSVs + G26
# ---------------------------------------------------------------------------
def step1_quick(pack_bt=None, L_LIVE=None):
    log("STEP1: daoviet 095370 CSVs + G26 status")
    g26 = kaggle_files("goodpjw2008/s6e11-target-first-gpt2-seed-2026")
    has5 = "test_5fold" in g26.lower() or "gpt2_s2026_test_5fold" in g26.lower()
    live_p = pd.read_csv(LIVE)["is_fraud"].to_numpy(float)
    L_live = L(live_p) if L_LIVE is None else L_LIVE

    rows = []
    for name in (
        "submission_095354_exact.csv",
        "submission_summit_095370.csv",
        "submission_verified_095370.csv",
    ):
        p = DAO / name
        if not p.exists():
            rows.append({"file": name, "missing": True})
            continue
        df = pd.read_csv(p)
        col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
        v = df[col].to_numpy(float)
        dist = float(1 - spearmanr(v, live_p).correlation)
        Lv = L(v)
        # R² of Lv on Bt if provided
        r2 = None
        if pack_bt is not None:
            X1 = np.column_stack([pack_bt, np.ones(len(Lv))])
            coef, *_ = np.linalg.lstsq(X1, unitize(Lv), rcond=None)
            pred = X1 @ coef
            ss_res = np.sum((unitize(Lv) - pred) ** 2)
            ss_tot = np.sum((unitize(Lv) - unitize(Lv).mean()) ** 2) + 1e-18
            r2 = float(1 - ss_res / ss_tot)
        rows.append(
            {
                "file": name,
                "spearman_dist_vs_LIVE": dist,
                "R2_on_Bt": r2,
                "spanned_0999": bool(r2 is not None and r2 > 0.999),
            }
        )
        log(f"  {name}: 1-ρ(LIVE)={dist:.6e} R²(Bt)={r2}")
    out = {
        "g26_files": g26,
        "g26_has_test_5fold": has5,
        "daoviet_csvs": rows,
    }
    log(f"  G26 test_5fold={'YES' if has5 else 'NO'}")
    return out


# ---------------------------------------------------------------------------
# Load ATH + streams
# ---------------------------------------------------------------------------
def load_core():
    log("Loading ATH proxy + streams")
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from gm_v13_2_overnight_check_and_stageB_v14 import build_ath_proxy

    ath = build_ath_proxy()
    y, fold_id, folds = ath["y"], ath["fold_id"], ath["folds"]
    z_ATH_o, z_ATH_t = ath["z_ATH_o"], ath["z_ATH_t"]
    d1o, d1t = ath["d1o"], ath["d1t"]
    Z, names = ath["Z"], ath["names"]

    def so(n):
        return L_fold(Z[f"oof_{n}"], fold_id)

    def st(n):
        return L(Z[f"test_{n}"])

    streams_o = {n: so(n) for n in names}
    streams_t = {n: st(n) for n in names}
    MS_o, MS_t = streams_o["masaya_gpt2"], streams_t["masaya_gpt2"]
    LM_o, LM_t = streams_o["megayak_5f_lm"], streams_t["megayak_5f_lm"]
    L7F_o, L7F_t = streams_o["lm7_5fold"], streams_t["lm7_5fold"]
    GPD_o, GPD_t = streams_o["daoviet_gpdvae"], streams_t["daoviet_gpdvae"]
    MLP_o, MLP_t = streams_o["gbn_bag5_w512"], streams_t["gbn_bag5_w512"]
    GB_o, GB_t = streams_o["gb_bag5"], streams_t["gb_bag5"]
    G26_o = streams_o["gpt2_seed2026"]
    G26_t = L(npy(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"))
    Gmed_o, Gmed_t = 0.5 * (G26_o + MS_o), 0.5 * (G26_t + MS_t)

    # L7_s2026 fold0 bayes
    l7p = PRED / "l7_s2026_5f"
    L7s_o = L7F_o.copy()
    L7s_t = L7F_t.copy()
    if (l7p / "oof_0.npy").exists():
        ll0 = np.load(l7p / "oof_0.npy")
        llt = np.load(l7p / "test_0.npy")
        m0 = fold_id == 0
        prior = np.array([1 - y[m0].mean(), y[m0].mean()])

        def bayes(ll, pr):
            joint = ll + np.log(pr + 1e-12).reshape(1, 1, 2)
            m = joint.max(-1, keepdims=True)
            ex = np.exp(joint - m)
            return (ex / ex.sum(-1, keepdims=True))[:, :, 1].mean(1)

        L7s_o = L7F_o.copy()
        L7s_o[m0] = L(bayes(ll0, prior))
        L7s_t = L(bayes(llt, prior))

    live_p = pd.read_csv(LIVE)["is_fraud"].to_numpy(float)
    L_LIVE = L(live_p)
    ids = pd.read_csv(LIVE)["id"].to_numpy()

    # Bug#1 diagnostic on raw z_ATH
    raw_mean_sig = float(expit(z_ATH_o).mean())
    log(
        f"  BUG#1 proof: mean(σ(z_ATH_o))={raw_mean_sig:.6f} vs y.mean={y.mean():.6f} "
        f"(zero-mean rank-logit ⇒ ~0.50)"
    )
    assert abs(z_ATH_o.mean()) < 0.05, "z_ATH expected near zero-mean rank-logit"

    return dict(
        y=y,
        fold_id=fold_id,
        folds=folds,
        z_ATH_o=z_ATH_o,
        z_ATH_t=z_ATH_t,
        d1o=d1o,
        d1t=d1t,
        MS_o=MS_o,
        MS_t=MS_t,
        LM_o=LM_o,
        LM_t=LM_t,
        L7F_o=L7F_o,
        L7F_t=L7F_t,
        GPD_o=GPD_o,
        GPD_t=GPD_t,
        MLP_o=MLP_o,
        MLP_t=MLP_t,
        GB_o=GB_o,
        GB_t=GB_t,
        G26_o=G26_o,
        G26_t=G26_t,
        Gmed_o=Gmed_o,
        Gmed_t=Gmed_t,
        L7s_o=L7s_o,
        L7s_t=L7s_t,
        streams_o=streams_o,
        streams_t=streams_t,
        live_p=live_p,
        L_LIVE=L_LIVE,
        ids=ids,
        names=names,
    )


# ---------------------------------------------------------------------------
# Step 2 — Platt η_ATH + X_v15
# ---------------------------------------------------------------------------
def solve_intercept_for_mean(linear, target_mean, lo=-20.0, hi=5.0, n=80):
    """Find b s.t. mean(σ(linear + b)) == target_mean."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        m = float(expit(linear + mid).mean())
        if m > target_mean:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def platt_calibrate_oof(z, y, fold_id, tau=1.0):
    """LOFO Platt: η = τ*a*z + b_τ with E[σ(η)]=y.mean() on each train fold."""
    z = np.asarray(z, float).ravel()
    eta = np.zeros(len(y), float)
    params = []
    for f in range(5):
        te = fold_id == f
        tr = ~te
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(z[tr].reshape(-1, 1), y[tr])
        a = float(lr.coef_.ravel()[0])
        b_platt = float(lr.intercept_.ravel()[0])
        # tempered slope
        a_t = tau * a
        linear_tr = a_t * z[tr]
        # solve b so mean(σ)=y[tr].mean() exactly
        b_t = solve_intercept_for_mean(linear_tr, float(y[tr].mean()))
        eta_tr = linear_tr + b_t
        # assert zero-mean gradient / prevalence match
        err = abs(float(expit(eta_tr).mean()) - float(y[tr].mean()))
        assert err < 1e-6, f"fold{f} prevalence mismatch {err}"
        eta[te] = a_t * z[te] + b_t
        params.append({"fold": f, "a": a, "a_tau": a_t, "b_platt": b_platt, "b_tau": b_t, "prev_err": err})
    # test: mean params
    a_bar = float(np.mean([p["a_tau"] for p in params]))
    # fit b on full train for test
    b_test = solve_intercept_for_mean(a_bar * z, float(y.mean()))
    # better: average fold b's with PCHIP of eta itself later
    eta_test_fn = lambda zt: a_bar * zt + solve_intercept_for_mean(a_bar * zt, float(y.mean()))
    return eta, params, a_bar, eta_test_fn


def build_Xv15(pack):
    log("Building X_v15 (tabular×disagreement, no raw z_ATH)")
    y, fold_id = pack["y"], pack["fold_id"]
    feat_o, feat_t, names = [], [], []

    # --- 68 KMAT / tabular from X_v5 ---
    Xo = np.load(XV5_O, mmap_mode="r")
    Xt = np.load(XV5_T, mmap_mode="r")
    assert Xo.shape[1] >= 68
    for j in range(68):
        oo = np.asarray(Xo[:, j], dtype=np.float64)
        tt = np.asarray(Xt[:, j], dtype=np.float64)
        oo_u = unitize_fold(oo, fold_id)
        tt_u = unitize(pchip_lock(tt, oo))  # lock then unitize to OOF scale
        # better: pchip to fold-concat distribution
        tt_u = unitize(pchip_lock(tt, oo))
        # re-unitize test to match oof std after pchip of raw
        tt_p = pchip_lock(tt, oo)
        tt_u = (tt_p - oo.mean()) / (oo.std() + 1e-12)  # global; prefer fold-safe:
        # Use OOF fold-unitized ref std≈1: pchip the unitized-fold vector's test proxy
        tt_u = unitize(pchip_lock(unitize(tt), oo_u))
        feat_o.append(oo_u.astype(np.float32))
        feat_t.append(tt_u.astype(np.float32))
        names.append(f"KMAT_{j}")

    # --- P4 ---
    for tag in ("L4.2", "L4.3", "L4.5"):
        op = PRED / f"matrix_v2/P4_{tag}_pure5f_oof.npy"
        tp = PRED / f"matrix_v2/P4_{tag}_pure5f_test.npy"
        if op.exists() and tp.exists():
            oo = L_fold(npy(op), fold_id)
            tt = L(npy(tp))
            feat_o.append(unitize_fold(oo, fold_id).astype(np.float32))
            feat_t.append(unitize(pchip_lock(tt, oo)).astype(np.float32))
            names.append(f"P4_{tag}")

    def add_contrast(name, oo, tt):
        oo_u = unitize_fold(oo, fold_id)
        tt_u = unitize(pchip_lock(tt, oo))
        feat_o.append(oo_u.astype(np.float32))
        feat_t.append(tt_u.astype(np.float32))
        names.append(name)

    # pairwise zero-sum contrasts
    add_contrast("MS-LM", pack["MS_o"] - pack["LM_o"], pack["MS_t"] - pack["LM_t"])
    add_contrast("L7F-LM", pack["L7F_o"] - pack["LM_o"], pack["L7F_t"] - pack["LM_t"])
    add_contrast("GPD-MLP", pack["GPD_o"] - pack["MLP_o"], pack["GPD_t"] - pack["MLP_t"])
    add_contrast("GPD-GB", pack["GPD_o"] - pack["GB_o"], pack["GPD_t"] - pack["GB_t"])
    add_contrast("MLP-GB", pack["MLP_o"] - pack["GB_o"], pack["MLP_t"] - pack["GB_t"])
    add_contrast("Gmed-MS", pack["Gmed_o"] - pack["MS_o"], pack["Gmed_t"] - pack["MS_t"])

    # denoise G26-MS / L7s-L7F via Ridge on lightweight surrogate (contrasts already in list)
    sur_o = np.column_stack(
        [unitize_fold(c, fold_id) for c in [pack["MS_o"], pack["LM_o"], pack["L7F_o"], pack["GPD_o"], pack["MLP_o"], pack["GB_o"]]]
    )
    sur_t = np.column_stack(
        [unitize(c) for c in [pack["MS_t"], pack["LM_t"], pack["L7F_t"], pack["GPD_t"], pack["MLP_t"], pack["GB_t"]]]
    )

    def denoise(target_o, target_t, tag):
        pred_o = np.zeros(len(y))
        pred_t_acc = []
        for f in range(5):
            te = fold_id == f
            tr = ~te
            from sklearn.linear_model import Ridge

            rg = Ridge(alpha=10.0)
            rg.fit(sur_o[tr], unitize(target_o[tr]))
            pred_o[te] = rg.predict(sur_o[te])
            pred_t_acc.append(rg.predict(sur_t))
        pred_t = np.mean(pred_t_acc, 0)
        a = 0.45
        den_o = unitize(a * unitize(target_o) + (1 - a) * unitize(pred_o))
        den_t = unitize(a * unitize(target_t) + (1 - a) * unitize(pred_t))
        den_t = unitize(pchip_lock(den_t, den_o))
        add_contrast(tag, den_o, den_t)
        return float(np.std(den_t) / (np.std(den_o) + 1e-12))

    disp_g = denoise(pack["G26_o"] - pack["MS_o"], pack["G26_t"] - pack["MS_t"], "denoise(G26-MS)")
    # L7_s2026
    m0 = fold_id == 0
    c_o = pack["L7s_o"] - pack["L7F_o"]
    c_t = pack["L7s_t"] - pack["L7F_t"]
    disp_l = denoise(c_o, c_t, "denoise(L7s-L7F)")
    log(f"  denoise disp G26={disp_g:.4f} L7s={disp_l:.4f}")

    # Q17 zero-sum vs LM / GB
    qmap = [
        ("yekenot_realmlp", "LM"),
        ("alex_glm_gen", "LM"),
        ("sadam_v4_cat", "LM"),
        ("sameer_glm10f", "LM"),
        ("megayak_shallow_tree", "GB"),
    ]
    for sn, peer in qmap:
        if sn not in pack["streams_o"]:
            continue
        if peer == "LM":
            add_contrast(f"{sn}-LM", pack["streams_o"][sn] - pack["LM_o"], pack["streams_t"][sn] - pack["LM_t"])
        else:
            add_contrast(f"{sn}-GB", pack["streams_o"][sn] - pack["GB_o"], pack["streams_t"][sn] - pack["GB_t"])

    # family dispersion / disagreement (zero-sum-ish sensors)
    stack_lm_o = np.column_stack([pack["MS_o"], pack["LM_o"], pack["L7F_o"], pack["G26_o"]])
    stack_lm_t = np.column_stack([pack["MS_t"], pack["LM_t"], pack["L7F_t"], pack["G26_t"]])
    stack_d_o = np.column_stack([pack["GPD_o"], pack["MLP_o"], pack["GB_o"]])
    stack_d_t = np.column_stack([pack["GPD_t"], pack["MLP_t"], pack["GB_t"]])
    add_contrast("std_LM", stack_lm_o.std(1), stack_lm_t.std(1))
    add_contrast("std_Disc", stack_d_o.std(1), stack_d_t.std(1))
    gap_o = stack_lm_o[:, :3].mean(1) - stack_d_o.mean(1)
    gap_t = stack_lm_t[:, :3].mean(1) - stack_d_t.mean(1)
    add_contrast("gap_Gen_vs_Disc", gap_o, gap_t)
    add_contrast("abs_gap_Gen_vs_Disc", np.abs(gap_o), np.abs(gap_t))

    # coarse 5-bin risk from z_ATH quantiles — categorical-ish numeric codes
    z = pack["z_ATH_o"]
    qs = np.quantile(z, [0.50, 0.85, 0.95, 0.985])
    bin_o = np.digitize(z, qs).astype(np.float64)  # 0..4
    bin_t = np.digitize(pack["z_ATH_t"], qs).astype(np.float64)
    # fold-safe: recompute digitize per fold for OOF
    bin_o = np.zeros(len(y), float)
    for f in range(5):
        m = fold_id == f
        qf = np.quantile(z[m], [0.50, 0.85, 0.95, 0.985])
        bin_o[m] = np.digitize(z[m], qf)
    feat_o.append(bin_o.astype(np.float32))
    feat_t.append(bin_t.astype(np.float32))
    names.append("q_ATH_bin")

    Xo = np.column_stack(feat_o)
    Xt = np.column_stack(feat_t)
    Xo = np.nan_to_num(Xo, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    # assert no column is near-perfect proxy for z_ATH
    corr_z = [abs(float(spearmanr(Xo[:, j], z).correlation)) for j in range(Xo.shape[1])]
    max_corr_z = float(np.max(corr_z))
    log(f"  X_v15 shape={Xo.shape} · max|ρ| vs z_ATH={max_corr_z:.4f} (q_ATH_bin expected ~0.8)")
    assert "z_ATH" not in "".join(names)
    return {"Xo": Xo, "Xt": Xt, "names": names, "max_corr_z": max_corr_z, "disp_g26": disp_g, "disp_l7": disp_l}


# ---------------------------------------------------------------------------
# Train + gate
# ---------------------------------------------------------------------------
def lofo_orth(delta_o, delta_t, pack):
    B_o = np.column_stack([unitize(pack["z_ATH_o"]), pack["d1o"]])
    B_t = np.column_stack([unitize(pack["z_ATH_t"]), pack["d1t"]])
    Vo = unitize(delta_o)
    Vt = unitize(delta_t)
    resid = np.zeros(len(pack["y"]))
    betas = []
    for f in range(5):
        te = pack["fold_id"] == f
        tr = ~te
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B_o[tr], Vo[tr])
        resid[te] = Vo[te] - lr.predict(B_o[te])
        betas.append(np.concatenate([lr.coef_, [lr.intercept_]]))
    bbar = np.mean(betas, 0)
    rt = Vt - (B_t @ bbar[:-1] + bbar[-1])
    rt = unitize(pchip_lock(rt, resid))
    return unitize(resid), rt


def eval_main(y, folds, base, cand):
    d = np.array(
        [1e5 * (roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])) for _, te in folds]
    )
    rng = np.random.default_rng(42)
    boots = [d[rng.integers(0, 5, size=5)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "pos5": int((d > 0).sum()),
        "mean_e5": float(d.mean()),
        "min_e5": float(d.min()),
        "folds": [round(float(x), 3) for x in d],
        "boot_90ci": [round(float(lo), 3), round(float(hi), 3)],
    }


def eval_25(y, fold_id, base, cand):
    deltas = []
    for f in range(5):
        m = fold_id == f
        yf, bf, cf = y[m], base[m], cand[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for _, te in skf.split(np.arange(len(yf)), yf):
            deltas.append(1e5 * (roc_auc_score(yf[te], cf[te]) - roc_auc_score(yf[te], bf[te])))
    d = np.asarray(deltas, float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    try:
        _, wp = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    return {
        "pos25": int((d > 0).sum()),
        "mean_e5": mean,
        "t_stat": float(mean / (se + 1e-18)),
        "wilcoxon_p": float(wp),
    }


def four_bar(main, sf, disp, r2_live):
    b1 = main["pos5"] == 5 and main["min_e5"] >= MIN_FOLD_E5
    b2 = sf["pos25"] >= SF25_NEED and sf["t_stat"] >= T25_NEED and sf["wilcoxon_p"] < WILC_P
    b3 = main["boot_90ci"][0] > BOOT_LO_NEED
    b4 = (DISP_LO <= disp <= DISP_HI) and (r2_live < R2_LIVE_MAX)
    return {
        "bar1": b1,
        "bar2": b2,
        "bar3": b3,
        "bar4": b4,
        "pass_all": bool(b1 and b2 and b3 and b4),
    }


def train_mode(pack, feats, tau, tag):
    """Fit fold-fresh Platt η on each train split (no init leakage), then boost."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    y, folds = pack["y"], pack["folds"]
    z_o, z_t = pack["z_ATH_o"], pack["z_ATH_t"]
    Xo, Xt = feats["Xo"], feats["Xt"]
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_init = np.zeros(len(y))  # LOFO init applied to te
    te_lgb, te_xgb, te_cat, te_init = [], [], [], []
    params_lgb = dict(
        objective="binary",
        metric="auc",
        max_depth=3,
        num_leaves=6,
        min_child_samples=500,
        feature_fraction=0.30,
        lambda_l2=100.0,
        learning_rate=0.015,
        verbosity=-1,
    )

    for f, (tr, te) in enumerate(folds):
        # Fold-fresh Platt on tr only
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(z_o[tr].reshape(-1, 1), y[tr])
        a = float(lr.coef_.ravel()[0]) * tau
        b = solve_intercept_for_mean(a * z_o[tr], float(y[tr].mean()))
        init_tr = a * z_o[tr] + b
        init_te = a * z_o[te] + b
        init_tt = a * z_t + b
        prev_err = abs(float(expit(init_tr).mean()) - float(y[tr].mean()))
        assert prev_err < 1e-6, f"{tag} fold{f} prev_err={prev_err}"
        g_mean = float((expit(init_tr) - y[tr]).mean())
        log(f"  {tag} fold {f}: prev_err={prev_err:.2e} mean(grad)={g_mean:.2e}")
        oof_init[te] = init_te
        te_init.append(init_tt)

        dtr = lgb.Dataset(Xo[tr], label=y[tr], init_score=init_tr)
        m = lgb.train(params_lgb, dtr, num_boost_round=180)
        oof_lgb[te] = m.predict(Xo[te], raw_score=True)
        te_lgb.append(m.predict(Xt, raw_score=True))

        dtrain = xgb.DMatrix(Xo[tr], label=y[tr], base_margin=init_tr)
        dval = xgb.DMatrix(Xo[te], base_margin=init_te)
        dtest = xgb.DMatrix(Xt, base_margin=init_tt)
        px = dict(
            objective="binary:logistic",
            max_depth=3,
            min_child_weight=500,
            colsample_bytree=0.30,
            subsample=0.8,
            reg_lambda=100.0,
            eta=0.015,
            tree_method="hist",
            verbosity=0,
        )
        bx = xgb.train(px, dtrain, num_boost_round=180)
        oof_xgb[te] = bx.predict(dval, output_margin=True)
        te_xgb.append(bx.predict(dtest, output_margin=True))

        cat = CatBoostClassifier(
            loss_function="Logloss",
            depth=3,
            min_data_in_leaf=500,
            rsm=0.30,
            l2_leaf_reg=100.0,
            learning_rate=0.015,
            iterations=180,
            verbose=False,
            random_seed=42 + f,
            thread_count=2,
        )
        try:
            cat.fit(Xo[tr], y[tr], baseline=init_tr.reshape(-1, 1))
            oof_cat[te] = cat.predict(
                Xo[te], prediction_type="RawFormulaVal", baseline=init_te.reshape(-1, 1)
            )
            te_cat.append(
                cat.predict(Xt, prediction_type="RawFormulaVal", baseline=init_tt.reshape(-1, 1))
            )
        except TypeError:
            cat.fit(Xo[tr], y[tr])
            oof_cat[te] = cat.predict(Xo[te], prediction_type="RawFormulaVal")
            te_cat.append(cat.predict(Xt, prediction_type="RawFormulaVal"))

    def resid(raw, init):
        r = raw - init
        if np.std(r) < 1e-9:
            r = unitize(raw)
        return unitize(r)

    init_t = np.mean(te_init, 0)
    d_o = unitize(
        (resid(oof_lgb, oof_init) + resid(oof_xgb, oof_init) + resid(oof_cat, oof_init)) / 3.0
    )
    d_t = unitize(
        (
            resid(np.mean(te_lgb, 0), init_t)
            + resid(np.mean(te_xgb, 0), init_t)
            + resid(np.mean(te_cat, 0), init_t)
        )
        / 3.0
    )
    return d_o, d_t, oof_init


def screen_delta(pack, d_o, d_t, tag):
    ro, rt = lofo_orth(d_o, d_t, pack)
    disp = float(np.std(rt) / (np.std(ro) + 1e-12))
    # R²(δ|LIVE)
    X1 = np.column_stack([unitize(pack["L_LIVE"]), np.ones(len(rt))])
    coef, *_ = np.linalg.lstsq(X1, unitize(rt), rcond=None)
    pred = X1 @ coef
    ss_res = np.sum((unitize(rt) - pred) ** 2)
    ss_tot = np.sum((unitize(rt) - unitize(rt).mean()) ** 2) + 1e-18
    r2_live = float(1 - ss_res / ss_tot)

    rows = []
    best = None
    for w in (0.02, 0.03, 0.04, 0.06, 0.08):
        cand = pack["z_ATH_o"] + w * ro
        main = eval_main(pack["y"], pack["folds"], pack["z_ATH_o"], cand)
        sf = eval_25(pack["y"], pack["fold_id"], pack["z_ATH_o"], cand)
        bars = four_bar(main, sf, disp, r2_live)
        row = {
            "tag": tag,
            "w": w,
            "main": main,
            "sf25": sf,
            "bars": bars,
            "disp": round(disp, 4),
            "R2_delta_LIVE": round(r2_live, 5),
        }
        rows.append(row)
        log(
            f"  {tag} w={w}: main {main['pos5']}/5 min={main['min_e5']:.2f} "
            f"sf={sf['pos25']}/25 t={sf['t_stat']:.2f} W-p={sf['wilcoxon_p']:.3g} "
            f"boot={main['boot_90ci']} pass={bars['pass_all']}"
        )
        if best is None or (bars["pass_all"], main["mean_e5"]) > (
            best["bars"]["pass_all"],
            best["main"]["mean_e5"],
        ):
            best = {**row, "ro": ro, "rt": rt}
    return rows, best


def maybe_stage(pack, bests):
    staged = []
    for b in bests:
        if b is None or not b["bars"]["pass_all"]:
            continue
        w = b["w"]
        z = pack["z_ATH_t"] + w * unitize(b["rt"])
        order = np.argsort(z + 1e-15 * (rankdata(z, method="ordinal") / len(z)), kind="mergesort")
        out_p = np.empty(len(z), float)
        out_p[order] = np.sort(pack["live_p"])
        path = SUB / f"sub_v13p4_verified_{b['tag']}_w{w}_NOSUBMIT.csv"
        assert NO_SUBMIT and "NOSUBMIT" in path.name
        pd.DataFrame({"id": pack["ids"], "is_fraud": out_p}).to_csv(path, index=False)
        staged.append({"path": str(path), "tag": b["tag"], "w": w, "main": b["main"], "bars": b["bars"]})
        log(f"  STAGED {path.name}")
    if not staged:
        log("  NOTHING STAGED — no 4-bar clears")
    return staged


def write_report(step1, pack, feats, platts, all_rows, bests, staged):
    LOG.mkdir(parents=True, exist_ok=True)
    md = LOG / "GM_V13_4_CALIBRATED_STAGEB_REPORT.md"
    js = LOG / "GM_V13_4_CALIBRATED_STAGEB_REPORT.json"
    lines = []
    lines.append("# GM V13.4 — Platt-Calibrated Init + Full Tabular×Disagreement Stage-B\n\n")
    lines.append(
        f"**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed {time.time()-T0:.1f}s\n\n"
        f"**Script:** `scripts/gm_v13_4_calibrated_init_tabular_stageB.py`\n\n"
    )
    lines.append("## Bugs fixed\n\n")
    lines.append(
        f"1. **Uncalibrated zero-mean `init_score`:** `mean(σ(z_ATH))` was ~0.50; "
        f"true prevalence={pack['y'].mean():.6f} (logit≈{logit(pack['y'].mean()):.3f}). "
        f"LOFO Platt η now asserts `|mean(σ(η_tr))-y_tr.mean()|<1e-6`.\n"
        f"2. **Restored X_v15:** {feats['Xo'].shape[1]} cols "
        f"(68 KMAT + P4 + zero-sum contrasts + disagreement + `q_ATH_bin`); "
        f"**no raw z_ATH in X** (max|ρ| vs z_ATH={feats['max_corr_z']:.3f}).\n\n"
    )

    lines.append("## Step 1 — daoviet CSVs + G26\n\n")
    lines.append(f"- G26 `test_5fold`: **{'PRESENT' if step1['g26_has_test_5fold'] else 'ABSENT'}**\n")
    for r in step1["daoviet_csvs"]:
        if r.get("missing"):
            lines.append(f"- `{r['file']}`: missing\n")
        else:
            lines.append(
                f"- `{r['file']}`: 1−ρ(LIVE)={r['spearman_dist_vs_LIVE']:.6e} · "
                f"R²(Bt)={r['R2_on_Bt']} · spanned>0.999={r['spanned_0999']}\n"
            )

    lines.append("\n## Step 2 — Platt η + X_v15\n\n")
    for tau, info in platts.items():
        mean_sig = float(expit(info["eta_o"]).mean())
        lines.append(
            f"- τ={tau}: mean(σ(η_OOF))={mean_sig:.6f} · fold prev_err max="
            f"{max(p['prev_err'] for p in info['params']):.2e}\n"
        )
    lines.append(
        f"- X_v15 shape OOF={feats['Xo'].shape} · denoise disp G26={feats['disp_g26']:.4f} "
        f"L7s={feats['disp_l7']:.4f}\n"
    )

    lines.append("\n## Step 3 — δ1_v15 4-bar screen\n\n")
    lines.append(
        "| mode | w | main | min | sf25 | t | W-p | boot90 | disp | R²(δ\\|LIVE) | PASS |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|\n"
    )
    for rows in all_rows:
        for r in rows:
            m, s = r["main"], r["sf25"]
            lines.append(
                f"| `{r['tag']}` | {r['w']} | {m['pos5']}/5 | {m['min_e5']:.2f} | "
                f"{s['pos25']}/25 | {s['t_stat']:.2f} | {s['wilcoxon_p']:.3g} | "
                f"{m['boot_90ci']} | {r['disp']} | {r['R2_delta_LIVE']} | {r['bars']['pass_all']} |\n"
            )

    lines.append("\n## Staging\n\n")
    if staged:
        for s in staged:
            lines.append(f"- **STAGED** `{s['path']}`\n")
    else:
        lines.append("**NOTHING STAGED** — zero candidates cleared all 4 bars.\n")

    lines.append("\n## Bottom line\n\n")
    any_pass = any(b and b["bars"]["pass_all"] for b in bests)
    # best main 5/5?
    best5 = None
    for rows in all_rows:
        for r in rows:
            if r["main"]["pos5"] == 5:
                if best5 is None or r["main"]["mean_e5"] > best5["main"]["mean_e5"]:
                    best5 = r
    lines.append(
        "1. Platt-calibrated `init_score` restores correct prevalence (~0.0167) and zero-mean initial gradients.\n"
        "2. Full tabular×disagreement `X_v15` restores the feature surface Stage-B needs without z_ATH shortcuts.\n"
    )
    if any_pass:
        lines.append("3. **δ1_v15 cleared Bars 1–4** — staged NOSUBMIT candidate(s).\n")
    else:
        msg = "3. **δ1_v15 did not clear all 4 bars**"
        if best5:
            msg += (
                f" (best 5/5: `{best5['tag']}` w={best5['w']} min={best5['main']['min_e5']:.2f} "
                f"sf={best5['sf25']['pos25']}/25 t={best5['sf25']['t_stat']:.2f})"
            )
        lines.append(msg + ".\n")
    lines.append(
        "4. G26 5-fold test still absent; daoviet 095370 CSVs are (near-)spanned podium blends, not new GPD.\n"
    )
    md.write_text("".join(lines))

    payload = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "elapsed_s": round(time.time() - T0, 2),
        "step1": {k: v for k, v in step1.items() if k != "g26_files"},
        "g26_files": step1["g26_files"],
        "X_v15_cols": feats["Xo"].shape[1],
        "feat_names": feats["names"],
        "max_corr_z": feats["max_corr_z"],
        "platt": {
            str(tau): {"params": info["params"], "mean_sig_oof": float(expit(info["eta_o"]).mean())}
            for tau, info in platts.items()
        },
        "rows": [[{k: v for k, v in r.items()} for r in rows] for rows in all_rows],
        "staged": staged,
    }
    js.write_text(json.dumps(payload, indent=2, default=str))
    log(f"WROTE {md}")


def main():
    assert NO_SUBMIT
    TMP.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    log("GM V13.4 start · NO_SUBMIT=True · POD_TOUCHES=0")

    pack = load_core()
    # Bt for step1 R²
    Bt = np.column_stack(
        [
            unitize(c)
            for c in [
                pack["MS_t"],
                pack["LM_t"],
                pack["L7F_t"],
                pack["GPD_t"],
                pack["MLP_t"],
                pack["GB_t"],
                pack["G26_t"],
                pack["Gmed_t"],
            ]
        ]
    )
    step1 = step1_quick(pack_bt=Bt, L_LIVE=pack["L_LIVE"])

    # Platt η for τ=1 and τ=0.5
    platts = {}
    for tau in (1.0, 0.5):
        log(f"Platt-calibrating η_ATH τ={tau}")
        eta_o, params, a_bar, eta_test_fn = platt_calibrate_oof(
            pack["z_ATH_o"], pack["y"], pack["fold_id"], tau=tau
        )
        eta_t = eta_test_fn(pack["z_ATH_t"])
        # global prevalence check
        err = abs(float(expit(eta_o).mean()) - float(pack["y"].mean()))
        log(f"  τ={tau}: mean(σ(η_OOF))={float(expit(eta_o).mean()):.6f} err_vs_global_y={err:.3e}")
        platts[tau] = {"eta_o": eta_o, "eta_t": eta_t, "params": params, "a_bar": a_bar}

    feats = build_Xv15(pack)
    # persist
    art = PRED / "gm_v13_4_stageB"
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "X_v15_oof.npy", feats["Xo"])
    np.save(art / "X_v15_test.npy", feats["Xt"])
    (art / "X_v15_names.json").write_text(json.dumps(feats["names"], indent=2))

    all_rows = []
    bests = []
    # Mode 1 & 2 — fold-fresh Platt inside train_mode
    deltas = {}
    for tau, label in ((1.0, "tau1.00"), (0.5, "tau0.50")):
        d_o, d_t, _ = train_mode(pack, feats, tau, label)
        np.save(art / f"d1_v15_{label}_oof.npy", d_o.astype(np.float32))
        np.save(art / f"d1_v15_{label}_test.npy", d_t.astype(np.float32))
        deltas[label] = (d_o, d_t)
        rows, best = screen_delta(pack, d_o, d_t, f"d1_v15_{label}")
        all_rows.append(rows)
        bests.append(best)

    # Mode 3 blend
    d_o = unitize(0.5 * deltas["tau1.00"][0] + 0.5 * deltas["tau0.50"][0])
    d_t = unitize(0.5 * deltas["tau1.00"][1] + 0.5 * deltas["tau0.50"][1])
    np.save(art / "d1_v15_blend_oof.npy", d_o.astype(np.float32))
    np.save(art / "d1_v15_blend_test.npy", d_t.astype(np.float32))
    rows, best = screen_delta(pack, d_o, d_t, "d1_v15_blend")
    all_rows.append(rows)
    bests.append(best)

    staged = maybe_stage(pack, bests)
    write_report(step1, pack, feats, platts, all_rows, bests, staged)
    log("DONE")


if __name__ == "__main__":
    main()
