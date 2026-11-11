#!/usr/bin/env python3
"""Null-space constrained Method-B contrast stacker v3 (NO_SUBMIT).

Train/test equivalence via feature-space P_perp before nested Ridge + AUC polish.
Target wall-clock < 60s on local Mac CPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from recon_and_forensic_audit import build_z_ref

T0 = time.time()
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
MV2 = PRED / "matrix_v2"
V10 = PRED / "v10_transductive_bulsah"
P12 = ROOT / "external/nov08_part12"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
ART = PRED / "nullspace_v3"
ART.mkdir(parents=True, exist_ok=True)
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
TARGET_MEAN = 0.016700
N_Q = 10001


def log(msg: str) -> None:
    print(f"[{time.time() - T0:5.1f}s] {msg}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), 1e-6, 1.0 - 1e-6))


def calibrate_logit(z_test, z_oof, y, target_mean=TARGET_MEAN):
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof.reshape(-1, 1), y.astype(np.int32))
    z_lin = float(lr.coef_[0, 0]) * z_test + float(lr.intercept_[0])

    def obj(d):
        return (float(np.mean(expit(z_lin + d))) - target_mean) ** 2

    return z_lin + float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)


def load_sub(fn, test_ids):
    return pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(b)
        out -= float(np.dot(out, b)) * b
    return out


def disp_ks(o, t, seed=0):
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def fold_std_oof(v, folds):
    """Per-fold zero-mean unit-variance on OOF."""
    out = np.zeros_like(v, dtype=np.float64)
    for a, b in folds:
        mu, sd = float(v[b].mean()), float(v[b].std() + 1e-12)
        out[b] = (v[b] - mu) / sd
    return out


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


def align_method_b_smooth(oof_vec, test_vec, folds, jitter_scale=1e-11):
    """Full-domain rank-tie-broken PCHIP aligner (no kink at 0).

    Per-fold standardize OOF → unit; map test onto that law; force disp=1.
    """
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = fold_std_oof(oof_vec, folds)
    oof_std = (oof_std - float(oof_std.mean())) / (float(oof_std.std()) + 1e-12)
    test_std = (test_vec - float(test_vec.mean())) / (float(test_vec.std()) + 1e-12)

    oof_j = _jitter(oof_std, jitter_scale)
    test_j = _jitter(test_std, jitter_scale)
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(np.quantile(test_j, q), np.quantile(oof_j, q), extrapolate=True)
    test_al = np.asarray(pchip(test_j), dtype=np.float64)
    test_al = (test_al - float(test_al.mean())) / (float(test_al.std()) + 1e-12)
    test_al = test_al * float(oof_std.std()) + float(oof_std.mean())
    if float(ks_2samp(oof_std, test_al).statistic) >= 0.0005:
        # exact rank→quantile map (KS≈0)
        u = (rankdata(_jitter(test_std, 1e-7), method="average") - 0.5) / len(test_std)
        test_al = np.interp(u, q, np.quantile(oof_std, q))
        test_al = (test_al - float(test_al.mean())) / (float(test_al.std()) + 1e-12)
        test_al = test_al * float(oof_std.std()) + float(oof_std.mean())
    return oof_std, test_al


def align_delta_preserve_scale(delta_o, delta_t, B_bad=None):
    """Align test delta onto OOF delta law WITHOUT unitizing OOF (keeps learned scale)."""
    o = np.asarray(delta_o, dtype=np.float64).ravel()
    t = np.asarray(delta_t, dtype=np.float64).ravel()
    so = float(np.std(o)) + 1e-12
    o_u = (o - float(np.mean(o))) / so
    t_u = (t - float(np.mean(t))) / (float(np.std(t)) + 1e-12)
    o_j, t_j = _jitter(o_u), _jitter(t_u)
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(np.quantile(t_j, q), np.quantile(o_j, q), extrapolate=True)
    t_al = np.asarray(pchip(t_j), dtype=np.float64)
    t_al = (t_al - float(np.mean(t_al))) / (float(np.std(t_al)) + 1e-12)
    t_al = t_al * so + float(np.mean(o))
    if B_bad is not None:
        t_al = orth_against(t_al, [B_bad[:, j] for j in range(B_bad.shape[1])])
        t_al = (t_al - float(np.mean(t_al))) * (so / (float(np.std(t_al)) + 1e-12)) + float(np.mean(o))
    return o, t_al


def lofo_orth_zref(zo, zt, z_ref_o, z_ref_t, folds):
    """LOFO orthogonalize against z_ref; return residual oof/test (not yet Method-B aligned)."""
    ro = np.zeros(len(zo), dtype=np.float64)
    rt_f = []
    for a, b in folds:
        beta = float(np.cov(zo[a], z_ref_o[a], ddof=0)[0, 1] / (np.var(z_ref_o[a]) + 1e-12))
        ra = zo[a] - beta * z_ref_o[a]
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (zo[b] - beta * z_ref_o[b] - mu) / sd
        rt_f.append((zt - beta * z_ref_t - mu) / sd)
    return ro, np.mean(rt_f, 0)


def beta_diff(z1o, z2o, z1t, z2t):
    b = float(np.cov(z1o, z2o, ddof=0)[0, 1] / (np.var(z2o) + 1e-12))
    return z1o - b * z2o, z1t - b * z2t, b


def weight_for_spearman_dist(z_base, r, target, lo=0.0, hi=3.0, n=36):
    best_w, best_err, got = 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        dist = 1.0 - float(spearmanr(z_base, z_base + mid * r).correlation)
        err = abs(dist - target)
        if err < best_err:
            best_err, best_w, got = err, mid, dist
        if dist > target:
            hi = mid
        else:
            lo = mid
    return best_w, got


def assemble_gpt2m(y, folds):
    ntr = len(y)
    zo = np.full(ntr, np.nan)
    so = np.full(ntr, np.nan)
    zt_f, st_f = [], []
    for fi, (a, b) in enumerate(folds):
        oraw = np.load(MED / f"oof_{fi}.npy")
        traw = np.load(MED / f"test_{fi}.npy")
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        so[b] = llr.std(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
        st_f.append(tllr.std(1))
    return zo, np.mean(zt_f, 0), so, np.mean(st_f, 0)


def main():
    log(f"NULLSPACE_V3 START (NO_SUBMIT={NO_SUBMIT})")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    ntr, nte = len(y), len(test_ids)
    idx = {f: b for f, (_, b) in enumerate(folds)}

    # z_ref_o ≈ 0.9526612 + St9.5 test anchor
    z_ref_o, z_st9_t, s9, base_auc = build_z_ref(y, test_ids, folds)
    log(f"z_ref_o AUC={base_auc:.7f}")
    z_v8_t = align_method_b_smooth(
        z_ref_o, zrk(load_sub("sub_v8_part12_fsl_prop_rect.csv", test_ids)), folds
    )[1]

    # Strike ladder bad / good axes on test
    def lz(fn):
        return align_method_b_smooth(z_ref_o, zrk(load_sub(fn, test_ids)), folds)[1]

    z_st3 = lz("sub_strike3a_cont065_plus_glmquad_graph.csv")
    z_st4 = lz("sub_gm_top3_strike4a_cont070.csv")
    z_st5 = lz("sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv")
    z_st9 = lz("sub_gm_top1_strike9_final_max.csv")
    z_st9p5_t = lz("sub_strike9p5_axis_0p70_0p10.csv")
    z_lead = lz("sub_pass340_LEAD_st9_aman20_tuboa10.csv")
    z_soft = lz("sub_strike10_slot1_softdiff_pchip_0p35.csv")

    e1 = z_st4 - z_st3
    v_bad = (z_st5 - z_st4) - 0.45 * (z_st4 - z_st3)
    d_st9p5 = z_st9p5_t - z_v8_t
    d_bad_leader = z_lead - z_st9
    d_bad_soft = z_soft - z_st9

    # Forbidden subspace B_bad (3 cols, orthonormal)
    b1 = unit(v_bad)
    b2 = unit(orth_against(d_bad_leader, [b1]))
    b3 = unit(orth_against(d_bad_soft, [b1, b2]))
    B_bad = np.column_stack([b1, b2, b3])
    e1_u = unit(e1)
    log("ladder axes ready")

    # ------------------------------------------------------------------
    # Load base models -> zrk
    # ------------------------------------------------------------------
    def load_z(oof_p, test_p):
        zo = zrk(np.load(oof_p).ravel())
        zt = zrk(np.load(test_p).ravel())
        return zo, zt

    z_lm_o, z_lm_t = load_z(P12 / "wide_lm_stack_nested.npy", P12 / "wide_lm_stack_test.npy")
    masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
    z_masa_o, z_masa_t = load_z(masa / "oof.npy", masa / "test.npy")
    z_rev_o, z_rev_t = load_z(
        PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"
    )
    z_g3_o, z_g3_t = load_z(PRED / "gpt2_3ep/gpt2_3ep_oof.npy", PRED / "gpt2_3ep/gpt2_3ep_test.npy")
    z_lm_g3_o, z_lm_g3_t = load_z(PRED / "lm_gpt2_3ep/lm_oof_5f.npy", PRED / "lm_gpt2_3ep/lm_test_5f.npy")
    z_gm_o, z_gm_t, s_gm_o, s_gm_t = assemble_gpt2m(y, folds)
    z_gm_o, z_gm_t = zrk(z_gm_o), zrk(z_gm_t)
    s_gm_o = s_gm_o.astype(np.float64)
    s_gm_t = s_gm_t.astype(np.float64)
    z_xgb_o, z_xgb_t = load_z(HARV / "oof_xgb_fast_harvest.npy", HARV / "test_xgb_fast_harvest.npy")
    z_tab_o, z_tab_t = load_z(HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy")
    z_bx_o, z_bx_t = load_z(PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy")

    # de-corrupt gpt2_3ep F0/F1
    train_f = np.concatenate([idx[f] for f in (2, 3, 4)])
    Xtr = np.column_stack([z_lm_o[train_f], z_masa_o[train_f], z_lm_g3_o[train_f]])
    ols_g3 = LinearRegression().fit(Xtr, z_g3_o[train_f])
    z_g3c_o = z_g3_o.copy()
    for f in (0, 1):
        b = idx[f]
        z_g3c_o[b] = ols_g3.predict(np.column_stack([z_lm_o[b], z_masa_o[b], z_lm_g3_o[b]]))
    z_g3c_t = ols_g3.predict(np.column_stack([z_lm_t, z_masa_t, z_lm_g3_t]))
    # mild blend on test: 0.5*orig + 0.5*ols to avoid full distribution jump
    z_g3c_t = 0.5 * z_g3_t + 0.5 * z_g3c_t

    # ------------------------------------------------------------------
    # Build contrast columns (raw), LOFO-orth vs z_ref, Method-B smooth align
    # ------------------------------------------------------------------
    raw_cols = {}

    def add_contrast(name, o, t):
        raw_cols[name] = (np.asarray(o, float).ravel(), np.asarray(t, float).ravel())

    c01o, c01t, b01 = beta_diff(z_gm_o, z_lm_o, z_gm_t, z_lm_t)
    add_contrast("c01_gpt2m_lm12", c01o, c01t)
    add_contrast("c02_gpt2m_prec", c01o / (1.0 + s_gm_o), c01t / (1.0 + s_gm_t))
    c03o, c03t, _ = beta_diff(z_lm_o, z_rev_o, z_lm_t, z_rev_t)
    add_contrast("c03_lm12_rev", c03o, c03t)
    c04o, c04t, _ = beta_diff(z_masa_o, z_lm_o, z_masa_t, z_lm_t)
    add_contrast("c04_masa_lm12", c04o, c04t)
    c05o, c05t, _ = beta_diff(z_g3c_o, z_rev_o, z_g3c_t, z_rev_t)
    add_contrast("c05_gpt2_3ep_clean_rev", c05o, c05t)
    add_contrast("c06_xgbf_tabm", z_xgb_o - z_tab_o, z_xgb_t - z_tab_t)
    add_contrast("c07_gpt2m_tabm", z_gm_o - z_tab_o, z_gm_t - z_tab_t)
    for tag, key in [
        ("c08_P4_L42", "P4_L4.2_pure5f"),
        ("c09_P4_L43", "P4_L4.3_pure5f"),
        ("c10_P4_L45", "P4_L4.5_pure5f"),
    ]:
        o, t = load_z(MV2 / f"{key}_oof.npy", MV2 / f"{key}_test.npy")
        add_contrast(tag, o, t)
    add_contrast("c11_B_xgb_llr", z_bx_o, z_bx_t)
    add_contrast(
        "c12_v10_cb_d4",
        np.load(V10 / "v10_cb_d4_bag3_orth_LM_oof.npy").ravel(),
        np.load(V10 / "v10_cb_d4_bag3_orth_LM_test.npy").ravel(),
    )
    add_contrast(
        "c13_v10_ridge",
        np.load(V10 / "v10_ridge_resid_oof.npy").ravel(),
        np.load(V10 / "v10_ridge_resid_test.npy").ravel(),
    )

    names = [
        "c01_gpt2m_lm12",
        "c02_gpt2m_prec",
        "c03_lm12_rev",
        "c04_masa_lm12",
        "c05_gpt2_3ep_clean_rev",
        "c06_xgbf_tabm",
        "c07_gpt2m_tabm",
        "c08_P4_L42",
        "c09_P4_L43",
        "c10_P4_L45",
        "c11_B_xgb_llr",
        "c12_v10_cb_d4",
        "c13_v10_ridge",
    ]

    cols_o, cols_t, hyg = [], [], []
    for nm in names:
        o, t = raw_cols[nm]
        # LOFO orth vs z_ref (skip if already residual-like with tiny corr — still apply)
        ro, rt = lofo_orth_zref(o, t, z_ref_o, z_v8_t, folds)
        ao, at = align_method_b_smooth(ro, rt, folds)
        d, ks = disp_ks(ao, at)
        if ks >= 0.0005:
            # discrete residual atoms: global z-score + exact rank map (no fold_std)
            ao = (ro - float(ro.mean())) / (float(ro.std()) + 1e-12)
            tu = (rt - float(rt.mean())) / (float(rt.std()) + 1e-12)
            u = (rankdata(_jitter(tu, 1e-7), method="average") - 0.5) / len(tu)
            q = np.linspace(0.0, 1.0, N_Q)
            at = np.interp(u, q, np.quantile(ao, q))
            at = (at - float(at.mean())) / (float(at.std()) + 1e-12)
            d, ks = disp_ks(ao, at)
        ok = (0.999999 <= d <= 1.000001) and (ks < 0.0005)
        hyg.append({"name": nm, "disp": float(d), "ks": float(ks), "ok": bool(ok)})
        cols_o.append(ao)
        cols_t.append(at)
        log(f"  {nm}: disp={d:.6f} KS={ks:.6f} ok={ok}")

    Do = np.column_stack(cols_o)
    Dt = np.column_stack(cols_t)
    n_ok = sum(1 for h in hyg if h["ok"])
    log(f"D shape={Do.shape} hygiene_ok={n_ok}/{len(hyg)}")

    # ------------------------------------------------------------------
    # Null-space projection P_perp from TEST coupling to B_bad
    # ------------------------------------------------------------------
    A = (Dt.T @ B_bad) / float(nte)  # M x 3
    M = Do.shape[1]
    AtA = A.T @ A + 1e-12 * np.eye(A.shape[1])
    P_perp = np.eye(M) - A @ np.linalg.solve(AtA, A.T)
    # symmetrize numerically
    P_perp = 0.5 * (P_perp + P_perp.T)
    Do_p = Do @ P_perp
    Dt_p = Dt @ P_perp
    # verify any random w gives ~0 bad cosines
    rng = np.random.default_rng(0)
    w_chk = rng.normal(size=M)
    delta_chk = Dt_p @ w_chk
    cos_bad = [cos(delta_chk, B_bad[:, j]) for j in range(3)]
    log(f"P_perp check random-w cos(B_bad)={[round(c, 6) for c in cos_bad]}")

    # e1 coupling for weak positive prior (on test, after P_perp)
    a_e1 = (Dt_p.T @ e1_u) / float(nte)  # M,

    z_cal_o = calibrate_logit(z_ref_o, z_ref_o, y)
    p_cal = expit(z_cal_o)
    # working logistic residual in logit units
    resid = (y.astype(np.float64) - p_cal) / (p_cal * (1.0 - p_cal) + 1e-4)
    base_folds = [float(roc_auc_score(y[b], z_ref_o[b])) for _, b in folds]
    base_cal_folds = [float(roc_auc_score(y[b], z_cal_o[b])) for _, b in folds]

    def eval_delta(delta_o, delta_t, tag):
        # CRITICAL: do NOT fold-unitize the stacked delta (that destroyed AUC in v3-buggy).
        delta_o2, delta_t2 = align_delta_preserve_scale(delta_o, delta_t, B_bad=B_bad)
        full = float(roc_auc_score(y, z_ref_o + delta_o2))
        fds = [
            float(roc_auc_score(y[b], z_ref_o[b] + delta_o2[b])) - bf
            for (_, b), bf in zip(folds, base_folds)
        ]
        d, ks = disp_ks(delta_o2, delta_t2)
        return {
            "tag": tag,
            "delta_1e5": round((full - base_auc) * 1e5, 2),
            "pos": int(sum(1 for x in fds if x > 0)),
            "folds_1e5": [round(x * 1e5, 2) for x in fds],
            "disp": round(d, 6),
            "ks": round(ks, 6),
            "cos_e1": round(cos(delta_t2, e1), 4),
            "cos_e2": round(cos(delta_t2, v_bad), 4),
            "cos_e3": round(cos(delta_t2, v_bad), 4),
            "cos_dst9p5": round(cos(delta_t2, d_st9p5), 4),
            "cos_dleader": round(cos(delta_t2, d_bad_leader), 4),
            "cos_dsoft": round(cos(delta_t2, d_bad_soft), 4),
            "delta_o": delta_o2,
            "delta_t": delta_t2,
        }

    def nested_fit(col_idx, set_name, use_e1_prior=False):
        Dop = Do_p[:, col_idx]
        Dtp = Dt_p[:, col_idx]
        a_e1_s = a_e1[col_idx]
        alphas = [100, 500, 2000, 10000, 50000]
        delta_ridge_o = np.zeros(ntr)
        delta_ridge_t_f = []
        delta_polish_o = np.zeros(ntr)
        delta_polish_t_f = []
        w_ridge_folds, w_polish_folds = [], []
        chosen_a = []

        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            best_a, best_s = alphas[0], -1e9
            for al in alphas:
                sc = StandardScaler(with_mean=True, with_std=True)
                Xa = sc.fit_transform(Dop[a])
                model = Ridge(alpha=al, fit_intercept=False)
                model.fit(Xa, resid[a])
                w_sc = model.coef_.ravel()
                w_orig = w_sc / (sc.scale_ + 1e-12)
                lifts = []
                for oi in other:
                    _, ob = folds[oi]
                    pred = Dop[ob] @ w_orig
                    lifts.append(float(roc_auc_score(y[ob], z_ref_o[ob] + pred)) - base_folds[oi])
                s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                if use_e1_prior:
                    s = s + 0.05 * float(np.dot(w_orig, a_e1_s) / (np.linalg.norm(w_orig) + 1e-12))
                if s > best_s:
                    best_s, best_a = s, al
            chosen_a.append(best_a)
            sc = StandardScaler()
            Xa = sc.fit_transform(Dop[a])
            model = Ridge(alpha=best_a, fit_intercept=False)
            model.fit(Xa, resid[a])
            w_orig = model.coef_.ravel() / (sc.scale_ + 1e-12)
            # shrink global scale so delta std stays O(0.05) on train fold
            pred_a = Dop[a] @ w_orig
            scale = 0.05 / (float(np.std(pred_a)) + 1e-12)
            w_orig = w_orig * scale
            w_ridge_folds.append(w_orig)

            delta_ridge_o[b] = Dop[b] @ w_orig
            delta_ridge_t_f.append(Dtp @ w_orig)

            # AUC polish on stratified fold subsamples
            w = w_orig.copy()
            sub_idx = {}
            sub_base = {}
            rng_f = np.random.default_rng(42 + f)
            for oi in other:
                _, ob = folds[oi]
                yb = y[ob]
                pos = ob[yb == 1]
                neg = ob[yb == 0]
                n_pos = min(len(pos), 1500)
                n_neg = min(len(neg), 7500)
                sb = np.concatenate(
                    [rng_f.choice(pos, n_pos, replace=False), rng_f.choice(neg, n_neg, replace=False)]
                )
                sub_idx[oi] = sb
                sub_base[oi] = float(roc_auc_score(y[sb], z_ref_o[sb]))
            for _pass in range(2):
                for j in range(len(w)):
                    best_wj, best_sj = w[j], -1e9
                    base_j = w[j]
                    for step in np.linspace(-1.0, 1.0, 9):
                        trial = w.copy()
                        trial[j] = base_j + step * (abs(base_j) + 0.02)
                        l2 = 0.05 * float(np.sum((trial - w_orig) ** 2))
                        lifts = []
                        for oi in other:
                            sb = sub_idx[oi]
                            pred = Dop[sb] @ trial
                            lifts.append(float(roc_auc_score(y[sb], z_ref_o[sb] + pred)) - sub_base[oi])
                        s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts))) - l2
                        if use_e1_prior:
                            s += 0.05 * float(np.dot(trial, a_e1_s))
                        if s > best_sj:
                            best_sj, best_wj = s, float(trial[j])
                    w[j] = best_wj
            # re-shrink polish weights
            pred_a = Dop[a] @ w
            w = w * (0.05 / (float(np.std(pred_a)) + 1e-12))
            w_polish_folds.append(w)
            delta_polish_o[b] = Dop[b] @ w
            delta_polish_t_f.append(Dtp @ w)

        wr = np.mean(w_ridge_folds, 0)
        wp = np.mean(w_polish_folds, 0)
        row_r = eval_delta(delta_ridge_o, np.mean(delta_ridge_t_f, 0), f"{set_name}/NullSpace_Ridge")
        row_p = eval_delta(delta_polish_o, np.mean(delta_polish_t_f, 0), f"{set_name}/NullSpace_AUC_Polish")
        row_r["alphas"] = chosen_a
        row_r["weights"] = {names[col_idx[i]]: round(float(wr[i]), 6) for i in range(len(col_idx))}
        row_p["weights"] = {names[col_idx[i]]: round(float(wp[i]), 6) for i in range(len(col_idx))}
        row_r["e1_prior"] = use_e1_prior
        row_p["e1_prior"] = use_e1_prior
        return row_r, row_p

    results = []
    # Set A / Set B × {plain, e1 prior}
    for set_name, ncols, prior in [
        ("SetA_c01..c12", 12, False),
        ("SetA_c01..c12_e1prior", 12, True),
        ("SetB_c01..c13", 13, False),
        ("SetB_c01..c13_e1prior", 13, True),
    ]:
        log(f"Nested fit {set_name} …")
        col_idx = list(range(ncols))
        rr, rp = nested_fit(col_idx, set_name, use_e1_prior=prior)
        results.extend([rr, rp])
        for r in (rr, rp):
            log(
                f"  {r['tag']}: Δ={r['delta_1e5']:+.2f}e-5 {r['pos']}/5 "
                f"disp={r['disp']:.4f} KS={r['ks']:.5f} "
                f"cos(e1,e2,leader,soft)=({r['cos_e1']:+.3f},{r['cos_e2']:+.4f},{r['cos_dleader']:+.4f},{r['cos_dsoft']:+.4f})"
            )

    # pick best by minimax: mean fold - 0.5*max(0,-min) with hard gate cos_bad≈0, pos>=4
    def score(r):
        fds = r["folds_1e5"]
        return float(np.mean(fds)) - 0.5 * max(0.0, -float(np.min(fds)))

    gated = [
        r
        for r in results
        if abs(r["cos_e2"]) < 1e-3
        and abs(r["cos_dleader"]) < 1e-3
        and abs(r["cos_dsoft"]) < 1e-3
        and r["pos"] >= 4
        and r["disp"] > 0.99
        and r["ks"] < 0.002
    ]
    winner = max(gated or results, key=score)
    log(f"WINNER {winner['tag']} Δ={winner['delta_1e5']:+.2f}e-5 score={score(winner):.2f}")

    # Stage onto St9.5
    r_t = winner["delta_t"]
    staged = []
    for td, tag in [(0.15e-4, "0p15e4"), (0.30e-4, "0p30e4")]:
        w, got = weight_for_spearman_dist(z_st9p5_t, r_t, td)
        z_final = z_st9p5_t + w * r_t
        p = expit(calibrate_logit(z_final, z_ref_o, y))
        path = SUB / f"sub_strike10_nullspace_v3_{tag}.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(path, index=False)
        staged.append(
            {
                "file": str(path),
                "target_1e4": td * 1e4,
                "achieved_1e4": round(got * 1e4, 4),
                "w": float(w),
                "mean": float(p.mean()),
                "1m_rho_st9p5_1e4": round((1 - float(spearmanr(z_final, z_st9p5_t).correlation)) * 1e4, 4),
                "cos_e2": round(cos(z_final - z_st9p5_t, v_bad), 6),
                "cos_dleader": round(cos(z_final - z_st9p5_t, d_bad_leader), 6),
            }
        )
        log(f"  staged {path.name}: 1-ρ={staged[-1]['1m_rho_st9p5_1e4']:.4f}e-4 mean={p.mean():.6f}")

    # Persist arrays
    np.save(ART / "D_oof_perp.npy", Do_p.astype(np.float32))
    np.save(ART / "D_test_perp.npy", Dt_p.astype(np.float32))
    np.save(ART / "delta_oof_winner.npy", winner["delta_o"].astype(np.float32))
    np.save(ART / "delta_test_winner.npy", winner["delta_t"].astype(np.float32))
    np.save(ART / "P_perp.npy", P_perp.astype(np.float64))

    # Report
    md = []
    md.append("# Null-Space Strike-Orth Contrast Stacker v3")
    md.append("")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{time.time()-T0:.1f}s**. z_ref_o **{base_auc:.7f}**.")
    md.append("")
    md.append("## Per-column Method-B smooth hygiene")
    md.append("")
    md.append("| col | disp | KS | ok |")
    md.append("|---|---:|---:|:---:|")
    for h in hyg:
        md.append(f"| {h['name']} | {h['disp']:.6f} | {h['ks']:.6f} | {'Y' if h['ok'] else 'N'} |")
    md.append("")
    md.append(f"Hygiene pass: **{n_ok}/{len(hyg)}**.")
    md.append("")
    md.append("## Nested comparison (after feature-space $P_\\perp$)")
    md.append("")
    md.append("| model | ΔOOF e-5 | pos | folds | disp | KS | cos(e1,e2,leader,soft) |")
    md.append("|---|---:|---:|---|---:|---:|---|")
    for r in results:
        md.append(
            f"| {r['tag']} | {r['delta_1e5']:+.2f} | {r['pos']}/5 | {r['folds_1e5']} | "
            f"{r['disp']:.4f} | {r['ks']:.6f} | "
            f"({r['cos_e1']:+.3f},{r['cos_e2']:+.4f},{r['cos_dleader']:+.4f},{r['cos_dsoft']:+.4f}) |"
        )
    md.append("")
    md.append(f"**Winner:** `{winner['tag']}` Δ={winner['delta_1e5']:+.2f}e-5 pos={winner['pos']}/5")
    md.append("")
    md.append("### Winner weights")
    md.append("")
    md.append("```")
    md.append(json.dumps(winner.get("weights", {}), indent=2))
    md.append("```")
    md.append("")
    md.append("## Staged onto St9.5 (NOT submitted)")
    md.append("")
    for s in staged:
        md.append(f"- `{s['file']}`: 1−ρ={s['1m_rho_st9p5_1e4']}e-4 w={s['w']:.6f} mean={s['mean']:.6f}")
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    md.append("")

    report = LOG / "NULLSPACE_STRIKE_ORTH_V3.md"
    report.write_text("\n".join(md))
    # strip large arrays from json
    results_json = []
    for r in results:
        results_json.append({k: v for k, v in r.items() if k not in ("delta_o", "delta_t")})
    summary = {
        "NO_SUBMIT": NO_SUBMIT,
        "z_ref_auc": base_auc,
        "hygiene": hyg,
        "hygiene_ok": n_ok,
        "P_perp_random_cos_Bbad": cos_bad,
        "results": results_json,
        "winner": {k: v for k, v in winner.items() if k not in ("delta_o", "delta_t")},
        "staged": staged,
        "elapsed_s": round(time.time() - T0, 1),
    }
    (LOG / "NULLSPACE_STRIKE_ORTH_V3.json").write_text(json.dumps(summary, indent=2))
    (LOG / "NULLSPACE_V3_STATUS.json").write_text(
        json.dumps({"t_s": round(time.time() - T0, 1), "msg": "NULLSPACE_V3_COMPLETE", "NO_SUBMIT": NO_SUBMIT}, indent=2)
    )
    log(f"Wrote {report}")
    log("NULLSPACE_V3_COMPLETE")


if __name__ == "__main__":
    main()
