#!/usr/bin/env python3
"""GM Overnight 8hr Super Factory — STRICT NO_SUBMIT.

Checkpointed multi-phase research/training pipeline.
Artifacts → preds/gm_v5_overnight/ ; report → logs/overnight/GM_V5_OVERNIGHT_8HR_REPORT.md
Resume-safe: skips any family whose {oof,test}.npy already exist.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "1")

warnings.filterwarnings("ignore")
# Relocate workspace to a fast disk via GM_V5_ROOT (e.g. /tmp/gm_v5_work).
ROOT = Path(os.environ.get("GM_V5_ROOT", Path(__file__).resolve().parents[1])).resolve()
os.environ["INPUT_DIR"] = os.environ.get("INPUT_DIR", str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar, nnls
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

T0 = time.time()
PRED = ROOT / "preds"
HARV = PRED / "fast_gpu_harvest"
MED = PRED / "gpt2_med_4ep"
MV2 = PRED / "matrix_v2"
ART2 = PRED / "gm_v2_ladder"
ART3 = PRED / "gm_v3_research"
ART4 = PRED / "gm_v4_polish"
ART41 = PRED / "gm_v4_1"
CACHE41 = PRED / "gm_v4_1_fast_cache.npz"
P12 = ROOT / "external/nov08_part12"
SUB = Path(os.environ.get("GM_V5_SUB", str(ROOT / "submissions")))
LOG = Path(os.environ.get("GM_V5_LOG", str(ROOT / "logs/overnight")))
ART = Path(os.environ.get("GM_V5_ART", str(PRED / "gm_v5_overnight")))
ART.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)
STATE_PATH = ART / "STATE.json"
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
KS_GATE = 0.0002
N_Q = 10001
N_JOBS = 4


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save_state(**kw):
    st = {}
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text())
        except Exception:
            st = {}
    st.update(kw)
    st["elapsed_s"] = round(time.time() - T0, 1)
    STATE_PATH.write_text(json.dumps(st, indent=2, default=float))


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1.0 - EPS))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(b)
        out -= float(np.dot(out, b)) * b
    return out


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


def fold_std_oof(v, folds):
    out = np.zeros_like(v, dtype=np.float64)
    for a, b in folds:
        mu, sd = float(v[b].mean()), float(v[b].std() + 1e-12)
        out[b] = (v[b] - mu) / sd
    return out


def force_exact_law(oof_vec, test_vec):
    o = unitize(oof_vec)
    t = np.asarray(test_vec, dtype=np.float64).ravel()
    o_s = np.sort(o)
    n_o, n_t = len(o_s), len(t)
    order = np.argsort(_jitter(t, 1e-7), kind="mergesort")
    pos = np.empty(n_t, dtype=np.int64)
    pos[order] = np.minimum(((np.arange(n_t) + 0.5) * n_o / n_t).astype(np.int64), n_o - 1)
    t_al = o_s[pos]
    t_al = t_al - float(t_al.mean())
    t_al = t_al * (float(o.std()) / (float(t_al.std()) + 1e-12))
    return o, t_al


def align_method_b_smooth(oof_vec, test_vec, folds, jitter_scale=1e-11):
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = unitize(fold_std_oof(oof_vec, folds))
    test_std = unitize(test_vec)
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(np.quantile(_jitter(test_std, jitter_scale), q), np.quantile(_jitter(oof_std, jitter_scale), q), extrapolate=True)
    test_al = unitize(np.asarray(pchip(_jitter(test_std, jitter_scale)), dtype=np.float64))
    if float(ks_2samp(oof_std, test_al).statistic) >= KS_GATE:
        oof_std, test_al = force_exact_law(oof_std, test_std)
    return oof_std, test_al


def disp_ks(o, t):
    o = np.asarray(o, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    za = np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1)
    ya = np.asarray(y_a, dtype=np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, dtype=np.float64).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def weight_for_spearman_dist(z_base, r, target, lo=0.0, hi=80.0, n=56):
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


def to_probs(z, y):
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    q = np.linspace(0, 1, 10001)
    u = (rankdata(z, method="average") - 0.5) / len(z)
    pchip = PchipInterpolator(q, np.quantile(z_v8_o, q), extrapolate=True)
    zm = np.asarray(pchip(u), float)
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_v8_o.reshape(-1, 1), y.astype(np.int32))
    zlin = float(lr.coef_[0, 0]) * zm + float(lr.intercept_[0])

    def obj(d):
        return (float(np.mean(expit(zlin + d))) - 0.016700) ** 2

    d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
    return expit(zlin + d)


def save_pair(tag, oof, test):
    np.save(ART / f"{tag}_oof.npy", np.asarray(oof, dtype=np.float32))
    np.save(ART / f"{tag}_test.npy", np.asarray(test, dtype=np.float32))
    log(f"  SAVED {tag} oof/test")


def load_pair(tag):
    return np.load(ART / f"{tag}_oof.npy").ravel(), np.load(ART / f"{tag}_test.npy").ravel()


def exists_pair(tag):
    return (ART / f"{tag}_oof.npy").exists() and (ART / f"{tag}_test.npy").exists()


def eb_te_column(ka, ya, kb, kt, prior, m=20.0):
    """Empirical-Bayes TE; returns (te_b, te_t)."""
    if np.issubdtype(ka.dtype, np.floating) or len(np.unique(ka)) > 500:
        edges = np.unique(np.quantile(ka, np.linspace(0, 1, 32)))
        ba = np.digitize(ka, edges)
        bb = np.digitize(kb, edges)
        bt = np.digitize(kt, edges)
    else:
        ba, bb, bt = ka, kb, kt
    tmp = pd.DataFrame({"k": ba, "y": ya})
    gstat = tmp.groupby("k")["y"].agg(["sum", "count"])
    te_map = ((gstat["sum"] + prior * m) / (gstat["count"] + m)).to_dict()
    te_b = np.array([te_map.get(k, prior) for k in bb], dtype=np.float32)
    te_t = np.array([te_map.get(k, prior) for k in bt], dtype=np.float32)
    return te_b, te_t


# ============================================================================
# PHASE 1
# ============================================================================
def phase1(y, folds, test_ids):
    log("======== PHASE 1: z_ref_old + double-poison purge + X_v5 ========")
    save_state(phase="phase1_start")
    ntr = len(y)

    # --- true z_ref_old ---
    if exists_pair("z_ref_old"):
        z_ref_o, z_ref_t = load_pair("z_ref_old")
        z_ref_o = z_ref_o.astype(np.float64)
        log(f"  loaded cached z_ref_old AUC={roc_auc_score(y,z_ref_o):.7f}")
    else:
        log("  building true z_ref_old via recon_and_forensic_audit.build_z_ref …")
        from recon_and_forensic_audit import build_z_ref

        z_ref_o, z_ref_t, _, z_ref_auc = build_z_ref(y, test_ids, folds)
        save_pair("z_ref_old", z_ref_o, z_ref_t)
        log(f"  z_ref_old AUC={z_ref_auc:.7f} (target≈0.9526612)")

    z_ref_auc = float(roc_auc_score(y, z_ref_o))
    z_ref_folds = fold_aucs(y, z_ref_o, folds)
    log(f"  z_ref_old AUC={z_ref_auc:.7f} folds={[round(x,7) for x in z_ref_folds]}")

    # --- non-neg combo of δ1_v4 + d3_lin + d3_bnd on z_ref_old ---
    d1_o = np.load(ART4 / "d1_v4_oof.npy").ravel().astype(np.float64)
    d1_t = np.load(ART4 / "d1_v4_test.npy").ravel().astype(np.float64)
    d3l_o = np.load(ART4 / "d3_v4_linear_oof.npy").ravel().astype(np.float64)
    d3l_t = np.load(ART4 / "d3_v4_linear_test.npy").ravel().astype(np.float64)
    d3b_o = np.load(ART4 / "d3_v4_boundary_oof.npy").ravel().astype(np.float64)
    d3b_t = np.load(ART4 / "d3_v4_boundary_test.npy").ravel().astype(np.float64)
    B_bad = np.load(ART41 / "B_bad_orth.npy")
    Q = np.load(ART41 / "Q_orth_basis.npy")
    q_bad_S3, q_bad_tree, q_bad_ldr, q_bad_soft = Q[:, 3], Q[:, 2], Q[:, 4], Q[:, 5]
    q1, q2, q_S5_win = Q[:, 0], Q[:, 1], Q[:, 6]

    def clean_delta(ro, rt, tag):
        rt_p = orth_against(rt, [B_bad[:, j] for j in range(B_bad.shape[1])])
        o_al, t_al = align_method_b_smooth(ro, rt_p, folds)
        if disp_ks(o_al, t_al)[1] >= KS_GATE:
            o_al, t_al = force_exact_law(o_al, t_al)
        t_al = orth_against(t_al, [B_bad[:, j] for j in range(B_bad.shape[1])])
        o_al, t_al = force_exact_law(o_al, t_al)
        d, ks = disp_ks(o_al, t_al)
        log(f"  clean {tag}: disp={d:.6f} KS={ks:.6f}")
        return o_al, t_al

    d1_o, d1_t = clean_delta(d1_o, d1_t, "d1_v4")
    d3l_o, d3l_t = clean_delta(d3l_o, d3l_t, "d3_lin")
    d3b_o, d3b_t = clean_delta(d3b_o, d3b_t, "d3_bnd")

    R_o = np.column_stack([d1_o, d3l_o, d3b_o])
    R_t = np.column_stack([d1_t, d3l_t, d3b_t])
    keys = ["d1_v4", "d3_lin", "d3_bnd"]
    w_grid = np.linspace(0.0, 0.25, 26)  # NON-NEGATIVE
    base_folds = z_ref_folds
    nested = np.zeros(ntr)
    fold_w = []
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        w = np.zeros(3)
        for _ in range(3):
            for j in range(3):
                bw, bs = w[j], -1e9
                for cw in w_grid:
                    wt = w.copy()
                    wt[j] = cw
                    lifts = [
                        float(roc_auc_score(y[folds[oi][1]], z_ref_o[folds[oi][1]] + R_o[folds[oi][1]] @ wt))
                        - base_folds[oi]
                        for oi in other
                    ]
                    s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                    if s > bs:
                        bs, bw = s, float(cw)
                w[j] = bw
        nested[b] = z_ref_o[b] + R_o[b] @ w
        fold_w.append(w)
        log(f"  NN combo F{f} w={dict(zip(keys, np.round(w,4)))}")
    mean_w = np.mean(fold_w, 0)
    nf = fold_aucs(y, nested, folds)
    deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
    pos = int(sum(x > 0 for x in deltas))
    combo_auc = float(roc_auc_score(y, nested))
    log(f"  δ_v4_combo_NN vs z_ref_old: AUC={combo_auc:.7f} Δe5={[round(x,2) for x in deltas]} pos={pos}/5 w={dict(zip(keys, np.round(mean_w,4)))}")
    d_combo_o = R_o @ mean_w
    d_combo_t = R_t @ mean_w
    d_combo_o, d_combo_t = clean_delta(d_combo_o, d_combo_t, "d_v4_combo_NN")
    save_pair("d_v4_combo_NN", d_combo_o, d_combo_t)

    # --- Positive-only poison purge of Today_S5 ---
    C41 = np.load(CACHE41, allow_pickle=True)
    z_s5 = np.asarray(C41["z_Today_S5"], dtype=np.float64)
    z_s1 = np.asarray(C41["z_Today_S1"], dtype=np.float64)
    delta = z_s5 - z_s1
    c_s3 = float(np.dot(delta, q_bad_S3))
    c_ldr = float(np.dot(delta, q_bad_ldr))
    c_tree = float(np.dot(delta, q_bad_tree))
    c_soft = float(np.dot(delta, q_bad_soft))
    log(f"  S5 vs S1 raw: c_bad_S3={c_s3:+.4f} c_bad_ldr={c_ldr:+.4f} c_bad_tree={c_tree:+.4f} c_bad_soft={c_soft:+.4f}")
    purge = -max(0.0, c_s3) * q_bad_S3 - max(0.0, c_ldr) * q_bad_ldr
    # Do NOT add back tree (keep negative)
    z_s5_purged = z_s5 + purge
    # verify
    d2 = z_s5_purged - z_s1
    log(
        f"  S5 purged: c_bad_S3={np.dot(d2,q_bad_S3):+.4f} c_bad_ldr={np.dot(d2,q_bad_ldr):+.4f} "
        f"c_bad_tree={np.dot(d2,q_bad_tree):+.4f} (kept neg)"
    )
    np.save(ART / "z_S5_positive_purge.npy", z_s5_purged.astype(np.float64))
    np.save(ART / "q_bad_S3.npy", q_bad_S3)
    np.save(ART / "q_bad_ldr.npy", q_bad_ldr)
    np.save(ART / "q_bad_tree.npy", q_bad_tree)
    np.save(ART / "B_bad_orth.npy", B_bad)
    np.save(ART / "Q_orth.npy", Q)

    # --- Build X_v5 ---
    if (ART / "X_v5_oof.npy").exists() and (ART / "X_v5_test.npy").exists() and os.environ.get("REBUILD_X", "0") != "1":
        log("  X_v5 already exists — skipping rebuild")
        meta = json.loads((ART / "X_v5_meta.json").read_text()) if (ART / "X_v5_meta.json").exists() else {}
    else:
        log("  importing gboost for KMAT/BASE …")
        import gboost as gb

        log(f"  KMAT={gb.KMAT.shape} BASE={gb.BASE.shape}")
        K = gb.KMAT
        K_tr, K_te = K[:ntr], K[ntr:]
        n_keys = 68
        # standard 68 TE
        K_oof = np.zeros((ntr, n_keys), dtype=np.float32)
        K_te_folds = []
        for a, b in folds:
            prior = float(y[a].mean())
            cols_b, cols_t = [], []
            for j in range(n_keys):
                te_b, te_t = eb_te_column(K_tr[a, j], y[a], K_tr[b, j], K_te[:, j], prior)
                cols_b.append(te_b)
                cols_t.append(te_t)
            K_oof[b] = np.column_stack(cols_b)
            K_te_folds.append(np.column_stack(cols_t))
            gc.collect()
        K_te_m = np.mean(K_te_folds, 0).astype(np.float32)
        log(f"  KMAT68 TE done shape={K_oof.shape}")

        # 5-seed averaged TE on top-15 high-cardinality keys
        card = [len(np.unique(K_tr[:, j][~pd.isna(K_tr[:, j])][:50000])) for j in range(min(68, K_tr.shape[1]))]
        top15 = np.argsort(card)[::-1][:15]
        log(f"  top15 KMAT keys for 5seed TE: {top15.tolist()} cards={[card[j] for j in top15]}")
        seeds = [42, 101, 202, 303, 404]
        K5_oof = np.zeros((ntr, 15), dtype=np.float32)
        K5_te_acc = []
        for a, b in folds:
            prior = float(y[a].mean())
            te_b_seeds = []
            te_t_seeds = []
            for seed in seeds:
                inner = StratifiedKFold(5, shuffle=True, random_state=seed)
                te_b = np.zeros((len(b), 15), dtype=np.float64)
                te_t = np.zeros((K_te.shape[0], 15), dtype=np.float64)
                cnt_b = np.zeros(len(b))
                # Build TE on each inner train of outer-train `a`, apply to outer-val `b` and test
                # Simpler: average TE maps fit on full `a` with different digitization noise via seeds
                # True nested: for each inner split of `a`, TE on inner-train → apply to all of `b`/`test` and average
                for ia, ib in inner.split(np.zeros(len(a)), y[a]):
                    a_tr = a[ia]
                    for ji, j in enumerate(top15):
                        tb, tt = eb_te_column(K_tr[a_tr, j], y[a_tr], K_tr[b, j], K_te[:, j], float(y[a_tr].mean()))
                        te_b[:, ji] += tb
                        te_t[:, ji] += tt
                    cnt_b += 1
                te_b /= max(cnt_b.max(), 1)
                te_t /= 5.0
                te_b_seeds.append(te_b)
                te_t_seeds.append(te_t)
            K5_oof[b] = np.mean(te_b_seeds, 0).astype(np.float32)
            K5_te_acc.append(np.mean(te_t_seeds, 0).astype(np.float32))
            gc.collect()
            log(f"  5seed TE outer-fold done")
        K5_te = np.mean(K5_te_acc, 0).astype(np.float32)

        # continuous extras
        def ab_load(path_o, path_t, already_z=False):
            xo = np.load(path_o).ravel()
            xt = np.load(path_t).ravel()
            if not already_z:
                xo, xt = zrk(xo), zrk(xt)
            else:
                xo, xt = np.asarray(xo, float), np.asarray(xt, float)
            o, t = align_method_b_smooth(xo, xt, folds)
            if disp_ks(o, t)[1] >= 0.0001:
                o, t = force_exact_law(o, t)
            return o.astype(np.float32), t.astype(np.float32)

        zs_o, zs_t = ab_load(ART4 / "z_lm_raw_super_v4_oof.npy", ART4 / "z_lm_raw_super_v4_test.npy", already_z=True)
        # gpt2m bayes
        if (ART4 / "_cache_gpt2m_bayes_oof.npy").exists():
            zb_o = np.load(ART4 / "_cache_gpt2m_bayes_oof.npy").ravel()
            zb_t = np.load(ART4 / "_cache_gpt2m_bayes_test.npy").ravel()
            zb_o, zb_t = align_method_b_smooth(zb_o, zb_t, folds)
        elif (ART2 / "z_m_bayes_5f_oof.npy").exists():
            zb_o, zb_t = ab_load(ART2 / "z_m_bayes_5f_oof.npy", ART2 / "z_m_bayes_5f_test.npy")
        else:
            zb_o, zb_t = zs_o.copy(), zs_t.copy()
        masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
        zm_o, zm_t = ab_load(masa / "oof.npy", masa / "test.npy")
        zr_o, zr_t = ab_load(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy")
        zbi_o, zbi_t = ab_load(PRED / "gpt2_3ep/gpt2_bidir_oof.npy", PRED / "gpt2_3ep/gpt2_bidir_test.npy")
        zv4_o, zv4_t = ab_load(PRED / "P_super_v4p_hc_oof.npy", PRED / "P_super_v4p_hc_test.npy")
        l43_o, l43_t = ab_load(MV2 / "P4_L4.3_pure5f_oof.npy", MV2 / "P4_L4.3_pure5f_test.npy")
        l42_o, l42_t = ab_load(MV2 / "P4_L4.2_pure5f_oof.npy", MV2 / "P4_L4.2_pure5f_test.npy")
        l45_o, l45_t = ab_load(MV2 / "P4_L4.5_pure5f_oof.npy", MV2 / "P4_L4.5_pure5f_test.npy")
        zt_o, zt_t = ab_load(HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy")

        r_rare = K_oof[:, 0].astype(np.float64)
        r_disp = K_oof[:, 1].astype(np.float64)
        r_rare_t = K_te_m[:, 0].astype(np.float64)
        r_disp_t = K_te_m[:, 1].astype(np.float64)

        extras_o = [
            zs_o, zb_o.astype(np.float32), zm_o, zr_o, zbi_o,
            (zs_o - zv4_o).astype(np.float32),
            (zs_o * r_rare).astype(np.float32), (zs_o * r_disp).astype(np.float32),
            l43_o, l42_o, l45_o, zt_o,
            (zs_o * l42_o).astype(np.float32), (zs_o * zt_o).astype(np.float32),
        ]
        extras_t = [
            zs_t, zb_t.astype(np.float32), zm_t, zr_t, zbi_t,
            (zs_t - zv4_t).astype(np.float32),
            (zs_t * r_rare_t).astype(np.float32), (zs_t * r_disp_t).astype(np.float32),
            l43_t, l42_t, l45_t, zt_t,
            (zs_t * l42_t).astype(np.float32), (zs_t * zt_t).astype(np.float32),
        ]
        extra_names = [
            "z_lm_raw_super_v4", "z_gpt2m_bayes_5f", "z_masa_5f", "z_gpt2_rev_5f", "z_gpt2_bidir_5f",
            "z_lm_minus_v4p", "z_lm_x_rare", "z_lm_x_disp",
            "P4_L4.3", "P4_L4.2", "P4_L4.5", "z_tabm_5f",
            "z_lm_x_L4.2", "z_lm_x_tabm",
        ]
        # re-align each extra
        for j in range(len(extras_o)):
            o, t = align_method_b_smooth(extras_o[j], extras_t[j], folds)
            if disp_ks(o, t)[1] >= 0.0001:
                o, t = force_exact_law(o, t)
            extras_o[j] = o.astype(np.float32)
            extras_t[j] = t.astype(np.float32)

        extra_o = np.column_stack(extras_o)
        extra_t = np.column_stack(extras_t)
        X_o = np.hstack([K_oof, K5_oof, extra_o, gb.BASE[:ntr].astype(np.float32)])
        X_t = np.hstack([K_te_m, K5_te, extra_t, gb.BASE[ntr:].astype(np.float32)])
        np.save(ART / "X_v5_oof.npy", X_o)
        np.save(ART / "X_v5_test.npy", X_t)
        # monotone constraint indices: extras start after 68+15=83
        mono_names = {"z_lm_raw_super_v4", "z_gpt2m_bayes_5f", "P4_L4.2", "P4_L4.3"}
        mono_idx = [83 + extra_names.index(n) for n in mono_names if n in extra_names]
        meta = {
            "shape_o": list(X_o.shape),
            "shape_t": list(X_t.shape),
            "n_kmat68": 68,
            "n_kmat5seed": 15,
            "extra_names": extra_names,
            "mono_idx": mono_idx,
            "top15_keys": top15.tolist(),
            "z_ref_old_auc": z_ref_auc,
            "combo_nn": {"w": dict(zip(keys, map(float, mean_w))), "delta_e5": deltas, "pos": pos, "auc": combo_auc},
            "purge": {"c_s3": c_s3, "c_ldr": c_ldr, "c_tree": c_tree, "c_soft": c_soft},
        }
        (ART / "X_v5_meta.json").write_text(json.dumps(meta, indent=2))
        log(f"  X_v5 shape o={X_o.shape} t={X_t.shape} mono_idx={mono_idx}")
        del K_oof, K_te_m, K5_oof, K5_te, extra_o, extra_t, X_o, X_t
        gc.collect()

    # base margins from v5 harvest
    v5_o = zrk(np.load(PRED / "P_super_v5_harvest_oof.npy").ravel())
    v5_t = zrk(np.load(PRED / "P_super_v5_harvest_test.npy").ravel())
    np.save(ART / "z_v5_harvest_oof.npy", v5_o.astype(np.float32))
    np.save(ART / "z_v5_harvest_test.npy", v5_t.astype(np.float32))

    phase1_out = {
        "z_ref_old_auc": z_ref_auc,
        "z_ref_old_folds": z_ref_folds,
        "combo_nn_auc": combo_auc,
        "combo_nn_delta_e5": deltas,
        "combo_nn_pos": pos,
        "combo_nn_w": dict(zip(keys, map(float, mean_w))),
        "purge": {"c_s3": c_s3, "c_ldr": c_ldr, "c_tree": c_tree},
    }
    save_state(phase="phase1_done", phase1=phase1_out)
    (ART / "phase1.json").write_text(json.dumps(phase1_out, indent=2))
    return phase1_out


# ============================================================================
# Shared training helpers
# ============================================================================
def get_Xy():
    X_o = np.load(ART / "X_v5_oof.npy", mmap_mode="r")
    X_t = np.load(ART / "X_v5_test.npy", mmap_mode="r")
    return X_o, X_t


def get_v5():
    return np.load(ART / "z_v5_harvest_oof.npy").ravel().astype(np.float64), np.load(ART / "z_v5_harvest_test.npy").ravel().astype(np.float64)


# ============================================================================
# PHASE 2 — CatBoost
# ============================================================================
def phase2(y, folds):
    log("======== PHASE 2: CatBoost bag9 ========")
    save_state(phase="phase2_start")
    if exists_pair("cat_stageB_bag9"):
        o, t = load_pair("cat_stageB_bag9")
        log(f"  SKIP — loaded cat_stageB_bag9 AUC={roc_auc_score(y,zrk(o)):.7f}")
        return
    from catboost import CatBoostClassifier, Pool

    X_o, X_t = get_Xy()
    v5_o, v5_t = get_v5()
    ntr = len(y)
    specs = (
        [(42 + i, 3, 25.0, 0.40, 0.035, 1200) for i in range(3)]
        + [(100 + i, 4, 35.0, 0.35, 0.030, 1200) for i in range(3)]
        + [(200 + i, 5, 50.0, 0.30, 0.025, 1000) for i in range(3)]
    )
    oof_bag = np.zeros(ntr, dtype=np.float64)
    te_bag = []
    for bi, (seed, depth, l2, rsm, lr, iters) in enumerate(specs):
        oof_s = np.zeros(ntr)
        te_s = []
        for fi, (a, b) in enumerate(folds):
            tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
            bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
            # CatBoost: use init_score via baseline
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                depth=depth,
                l2_leaf_reg=l2,
                rsm=rsm,
                learning_rate=lr,
                iterations=iters,
                random_seed=seed,
                thread_count=N_JOBS,
                od_type="Iter",
                od_wait=60,
                verbose=False,
                allow_writing_files=False,
            )
            train_pool = Pool(np.asarray(X_o[tr_in]), label=y[tr_in], baseline=bm_tr)
            es_pool = Pool(np.asarray(X_o[es_in]), label=y[es_in], baseline=bm_all[es_in])
            model.fit(train_pool, eval_set=es_pool, use_best_model=True)
            oof_s[b] = model.predict(Pool(np.asarray(X_o[b]), baseline=bm_all[b]), prediction_type="Probability")[:, 1]
            te_s.append(model.predict(Pool(np.asarray(X_t), baseline=bm_te), prediction_type="Probability")[:, 1])
            log(f"  cat bag{bi} d{depth} F{fi} best={model.get_best_iteration()}")
            del model, train_pool, es_pool
            gc.collect()
        oof_bag += oof_s
        te_bag.append(np.mean(te_s, 0))
        # checkpoint partial
        np.save(ART / f"_partial_cat_bag{bi}_oof.npy", oof_s.astype(np.float32))
    oof = oof_bag / len(specs)
    test = np.mean(te_bag, 0)
    save_pair("cat_stageB_bag9", oof, test)
    zo = zrk(oof)
    log(f"  cat_stageB_bag9 AUC={roc_auc_score(y,zo):.7f} folds={[round(x,7) for x in fold_aucs(y,zo,folds)]}")
    save_state(phase="phase2_done")


# ============================================================================
# PHASE 3 — XGB/LGB specialized
# ============================================================================
def phase3(y, folds):
    log("======== PHASE 3: specialized XGB/LGB Stage-B ========")
    save_state(phase="phase3_start")
    tags = [
        "xgb_stageB_monotone_lm_bag6",
        "lgb_stageB_extratrees_bag6",
        "xgb_stageB_tail_specialist_bag6",
        "lgb_stageB_colsub_deep_bag6",
    ]
    if all(exists_pair(t) for t in tags):
        for t in tags:
            o, _ = load_pair(t)
            log(f"  SKIP {t} AUC={roc_auc_score(y,zrk(o)):.7f}")
        save_state(phase="phase3_done")
        return

    import xgboost as xgb
    import lightgbm as lgb

    X_o, X_t = get_Xy()
    v5_o, v5_t = get_v5()
    ntr = len(y)
    p_v5 = expit(v5_o)
    meta = json.loads((ART / "X_v5_meta.json").read_text())
    mono_idx = set(meta.get("mono_idx", []))
    n_feat = X_o.shape[1]
    mono_constraints = tuple(1 if j in mono_idx else 0 for j in range(n_feat))

    # tail weights
    w_tail = 1.0 + 2.5 * expit(2.0 * (v5_o - 1.0)) * np.power(p_v5 * (1.0 - p_v5), 0.35)
    w_tail = w_tail.astype(np.float64)

    def train_xgb(tag, specs, monotone=False, use_tail=False, eta=0.025, colsample=0.35, reg_lambda=35.0, mcw=200, nround=600):
        if exists_pair(tag):
            o, t = load_pair(tag)
            log(f"  SKIP {tag} AUC={roc_auc_score(y,zrk(o)):.7f}")
            return
        oof_bag = np.zeros(ntr)
        te_bag = []
        for bi, (seed, depth) in enumerate(specs):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
                sw = w_tail[tr_in] if use_tail else None
                dtr = xgb.DMatrix(np.asarray(X_o[tr_in]), label=y[tr_in], base_margin=bm_tr, weight=sw)
                des = xgb.DMatrix(np.asarray(X_o[es_in]), label=y[es_in], base_margin=bm_all[es_in])
                dva = xgb.DMatrix(np.asarray(X_o[b]), base_margin=bm_all[b])
                dte = xgb.DMatrix(np.asarray(X_t), base_margin=bm_te)
                params = dict(
                    max_depth=depth, eta=eta, subsample=0.8, colsample_bynode=colsample,
                    reg_lambda=reg_lambda, min_child_weight=mcw, objective="binary:logistic",
                    eval_metric="auc", tree_method="hist", max_bin=256, nthread=N_JOBS, seed=seed,
                )
                if monotone:
                    params["monotone_constraints"] = mono_constraints
                booster = xgb.train(params, dtr, num_boost_round=nround, evals=[(des, "es")], early_stopping_rounds=50, verbose_eval=False)
                oof_s[b] = booster.predict(dva)
                te_s.append(booster.predict(dte))
                log(f"  {tag} bag{bi} d{depth} F{fi} best={booster.best_iteration}")
                del booster, dtr, des, dva, dte
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
        save_pair(tag, oof_bag / len(specs), np.mean(te_bag, 0))
        zo = zrk(oof_bag / len(specs))
        log(f"  {tag} AUC={roc_auc_score(y,zo):.7f} folds={[round(x,7) for x in fold_aucs(y,zo,folds)]}")

    def train_lgb(tag, specs, extra_trees=False, colsample=0.28, lr=0.02, reg_lambda=50.0, reg_alpha=0.0, mcs=300, nround=700, max_depth=-1):
        if exists_pair(tag):
            o, t = load_pair(tag)
            log(f"  SKIP {tag} AUC={roc_auc_score(y,zrk(o)):.7f}")
            return
        oof_bag = np.zeros(ntr)
        te_bag = []
        for bi, (seed, leaves) in enumerate(specs):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
                dtr = lgb.Dataset(np.asarray(X_o[tr_in]), label=y[tr_in], init_score=bm_tr)
                des = lgb.Dataset(np.asarray(X_o[es_in]), label=y[es_in], init_score=bm_all[es_in], reference=dtr)
                params = dict(
                    objective="binary", metric="auc", learning_rate=lr, num_leaves=leaves,
                    feature_fraction=colsample, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=reg_lambda, lambda_l1=reg_alpha, min_child_samples=mcs,
                    verbosity=-1, seed=seed, num_threads=N_JOBS, extra_trees=extra_trees,
                )
                if max_depth > 0:
                    params["max_depth"] = max_depth
                booster = lgb.train(
                    params, dtr, num_boost_round=nround, valid_sets=[des],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                )
                oof_s[b] = expit(booster.predict(np.asarray(X_o[b]), raw_score=True) + bm_all[b])
                te_s.append(expit(booster.predict(np.asarray(X_t), raw_score=True) + bm_te))
                log(f"  {tag} bag{bi} L{leaves} F{fi} best={booster.best_iteration}")
                del booster, dtr, des
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
        save_pair(tag, oof_bag / len(specs), np.mean(te_bag, 0))
        zo = zrk(oof_bag / len(specs))
        log(f"  {tag} AUC={roc_auc_score(y,zo):.7f} folds={[round(x,7) for x in fold_aucs(y,zo,folds)]}")

    train_xgb(
        "xgb_stageB_monotone_lm_bag6",
        [(42 + i, 3) for i in range(3)] + [(100 + i, 4) for i in range(3)],
        monotone=True, eta=0.02, colsample=0.30, reg_lambda=40.0, mcw=250, nround=600,
    )
    train_lgb(
        "lgb_stageB_extratrees_bag6",
        [(42 + i, 6) for i in range(3)] + [(100 + i, 10) for i in range(3)],
        extra_trees=True, colsample=0.28, lr=0.02, reg_lambda=50.0, reg_alpha=10.0, mcs=300, nround=700,
    )
    train_xgb(
        "xgb_stageB_tail_specialist_bag6",
        [(42 + i, 2) for i in range(3)] + [(100 + i, 3) for i in range(3)],
        use_tail=True, eta=0.025, colsample=0.35, reg_lambda=35.0, mcw=200, nround=500,
    )
    train_lgb(
        "lgb_stageB_colsub_deep_bag6",
        [(42 + i, 12) for i in range(6)],
        colsample=0.15, lr=0.02, reg_lambda=60.0, mcs=400, nround=800, max_depth=5,
    )
    save_state(phase="phase3_done")


# ============================================================================
# PHASE 4 — MLP
# ============================================================================
def phase4(y, folds):
    log("======== PHASE 4: Residual Bottleneck MLP ========")
    save_state(phase="phase4_start")
    if exists_pair("mlp_stageB_bag5"):
        o, t = load_pair("mlp_stageB_bag5")
        log(f"  SKIP mlp AUC={roc_auc_score(y,zrk(o)):.7f}")
        return
    # Avoid dual-OpenMP SIGSEGV after LightGBM/XGBoost (libomp vs torch).
    for _k, _v in {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KMP_INIT_AT_FORK": "FALSE",
    }.items():
        os.environ[_k] = _v
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # MPS has hung mid-init on this box; default CPU unless explicitly overridden.
    _dev = os.environ.get("GM_V5_TORCH_DEVICE", "cpu").strip().lower()
    if _dev == "mps" and not torch.backends.mps.is_available():
        _dev = "cpu"
    device = torch.device(_dev)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    log(f"  torch device={device} threads={torch.get_num_threads()}")
    log("  loading X_v5 / z_v5 …")
    X_o, X_t = get_Xy()
    v5_o, v5_t = get_v5()
    ntr = len(y)
    log(f"  X_o={tuple(X_o.shape)} X_t={tuple(X_t.shape)}")

    class ResBottleneck(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.fc1 = nn.Linear(d_in, 128)
            self.ln1 = nn.LayerNorm(128)
            self.fc2 = nn.Linear(128, 32)
            self.ln2 = nn.LayerNorm(32)
            self.fc3 = nn.Linear(32, 1)
            self.drop = nn.Dropout(0.35)
            self.act = nn.SiLU()

        def forward(self, x):
            h = self.drop(self.act(self.ln1(self.fc1(x))))
            h = self.drop(self.act(self.ln2(self.fc2(h))))
            return self.fc3(h).squeeze(-1)

    def rankgauss(train, *others):
        """Per-column RankGauss on train; map others via rank→train quantile."""
        out_tr = np.empty_like(train, dtype=np.float32)
        outs = [np.empty((o.shape[0], train.shape[1]), dtype=np.float32) for o in others]
        from scipy.stats import norm

        for j in range(train.shape[1]):
            col = train[:, j].astype(np.float64)
            r = (rankdata(col, method="average") - 0.5) / len(col)
            out_tr[:, j] = norm.ppf(np.clip(r, 1e-6, 1 - 1e-6)).astype(np.float32)
            qs = np.sort(out_tr[:, j])
            for oi, o in enumerate(others):
                u = (rankdata(o[:, j].astype(np.float64), method="average") - 0.5) / len(o)
                idx = np.minimum((u * len(qs)).astype(np.int64), len(qs) - 1)
                outs[oi][:, j] = qs[idx]
        return (out_tr, *outs)

    def pairwise_rank_loss(logit, y, n_pairs=2048):
        pos = torch.where(y > 0.5)[0]
        neg = torch.where(y < 0.5)[0]
        if len(pos) == 0 or len(neg) == 0:
            return torch.tensor(0.0, device=logit.device)
        n = min(n_pairs, len(pos) * len(neg), 4096)
        pi = pos[torch.randint(0, len(pos), (n,), device=logit.device)]
        ni = neg[torch.randint(0, len(neg), (n,), device=logit.device)]
        return torch.relu(1.0 - (logit[pi] - logit[ni])).mean()

    oof = np.zeros(ntr, dtype=np.float64)
    te_folds = []
    # Slimmer bag: 3 seeds × 12 epochs — enough diversity, less OpenMP/RAM pressure.
    seeds = [42, 101, 202]
    n_epochs = 12
    for fi, (a, b) in enumerate(folds):
        log(f"  mlp fold {fi}: materialize + RankGauss …")
        Xtr_raw = np.asarray(X_o[a], dtype=np.float32)
        Xva_raw = np.asarray(X_o[b], dtype=np.float32)
        Xte_raw = np.asarray(X_t, dtype=np.float32)
        Xtr, Xva, Xte = rankgauss(Xtr_raw, Xva_raw, Xte_raw)
        del Xtr_raw, Xva_raw, Xte_raw
        gc.collect()
        log(f"  mlp fold {fi}: RankGauss done, training {len(seeds)} seeds …")
        bm_a, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[a], y[a], v5_o, v5_t)
        # inner split for early stop
        tr_in, es_in = train_test_split(np.arange(len(a)), test_size=0.10, random_state=42 + fi, stratify=y[a])
        fold_oof_seeds = []
        fold_te_seeds = []
        for seed in seeds:
            log(f"  mlp F{fi} seed{seed} start")
            torch.manual_seed(seed)
            model = ResBottleneck(Xtr.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            bce = nn.BCEWithLogitsLoss()
            xt = torch.tensor(Xtr[tr_in], dtype=torch.float32, device=device)
            yt = torch.tensor(y[a][tr_in].astype(np.float32), device=device)
            bt = torch.tensor(bm_a[tr_in].astype(np.float32), device=device)
            xe = torch.tensor(Xtr[es_in], dtype=torch.float32, device=device)
            ye = torch.tensor(y[a][es_in].astype(np.float32), device=device)
            be = torch.tensor(bm_a[es_in].astype(np.float32), device=device)
            ds = TensorDataset(xt, yt, bt)
            dl = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=0)
            best_state, best_auc = None, -1
            for ep in range(n_epochs):
                model.train()
                for xb, yb, bb in dl:
                    opt.zero_grad(set_to_none=True)
                    residual = model(xb)
                    logit = residual + bb
                    loss = 0.8 * bce(logit, yb) + 0.2 * pairwise_rank_loss(logit, yb)
                    loss.backward()
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    pred = expit((model(xe) + be).cpu().numpy())
                    auc = float(roc_auc_score(ye.cpu().numpy(), pred))
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(Xva, dtype=torch.float32, device=device)
                xtt = torch.tensor(Xte, dtype=torch.float32, device=device)
                bv = torch.tensor(bm_all[b].astype(np.float32), device=device)
                btt = torch.tensor(bm_te.astype(np.float32), device=device)
                pv = expit((model(xv) + bv).cpu().numpy())
                pt = expit((model(xtt) + btt).cpu().numpy())
            fold_oof_seeds.append(pv)
            fold_te_seeds.append(pt)
            log(f"  mlp F{fi} seed{seed} es_auc={best_auc:.5f}")
            del model, opt, dl, ds, xt, yt, bt, xe, ye, be
            gc.collect()
        oof[b] = np.mean(fold_oof_seeds, 0)
        te_folds.append(np.mean(fold_te_seeds, 0))
        np.save(ART / f"_partial_mlp_F{fi}_oof.npy", oof.astype(np.float32))
        del Xtr, Xva, Xte
        gc.collect()
    test = np.mean(te_folds, 0)
    save_pair("mlp_stageB_bag5", oof, test)
    zo = zrk(oof)
    log(f"  mlp_stageB_bag5 AUC={roc_auc_score(y,zo):.7f} folds={[round(x,7) for x in fold_aucs(y,zo,folds)]}")
    save_state(phase="phase4_done")


# ============================================================================
# PHASE 5 — Portfolio + staging
# ============================================================================
def phase5(y, folds, test_ids):
    log("======== PHASE 5: NN portfolio + morning staging ========")
    save_state(phase="phase5_start")
    ntr = len(y)
    z_ref_o, _ = load_pair("z_ref_old")
    z_ref_o = z_ref_o.astype(np.float64)
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_t = zrk(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel())
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())
    B_bad = np.load(ART / "B_bad_orth.npy")
    Q = np.load(ART / "Q_orth.npy")
    q_bad_S3, q_bad_tree, q_bad_ldr = Q[:, 3], Q[:, 2], Q[:, 4]
    z_s5_purged = np.load(ART / "z_S5_positive_purge.npy")

    families = [
        ("xgb_stageB_68_bce", ART3 / "xgb_stageB_68_bce_oof.npy", ART3 / "xgb_stageB_68_bce_test.npy"),
        ("xgb_stageB_68_focal_auc", ART3 / "xgb_stageB_68_focal_auc_oof.npy", ART3 / "xgb_stageB_68_focal_auc_test.npy"),
        ("lgb_stageB_68_clean", ART3 / "lgb_stageB_68_clean_oof.npy", ART3 / "lgb_stageB_68_clean_test.npy"),
        ("lgb_gpt2_3ep_fixed", PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy", PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy"),
        ("cat_stageB_bag9", ART / "cat_stageB_bag9_oof.npy", ART / "cat_stageB_bag9_test.npy"),
        ("xgb_stageB_monotone_lm_bag6", ART / "xgb_stageB_monotone_lm_bag6_oof.npy", ART / "xgb_stageB_monotone_lm_bag6_test.npy"),
        ("lgb_stageB_extratrees_bag6", ART / "lgb_stageB_extratrees_bag6_oof.npy", ART / "lgb_stageB_extratrees_bag6_test.npy"),
        ("xgb_stageB_tail_specialist_bag6", ART / "xgb_stageB_tail_specialist_bag6_oof.npy", ART / "xgb_stageB_tail_specialist_bag6_test.npy"),
        ("lgb_stageB_colsub_deep_bag6", ART / "lgb_stageB_colsub_deep_bag6_oof.npy", ART / "lgb_stageB_colsub_deep_bag6_test.npy"),
        ("mlp_stageB_bag5", ART / "mlp_stageB_bag5_oof.npy", ART / "mlp_stageB_bag5_test.npy"),
    ]
    d3l_o = np.load(ART4 / "d3_v4_linear_oof.npy").ravel().astype(np.float64)
    d3l_t = np.load(ART4 / "d3_v4_linear_test.npy").ravel().astype(np.float64)
    d3b_o = np.load(ART4 / "d3_v4_boundary_oof.npy").ravel().astype(np.float64)
    d3b_t = np.load(ART4 / "d3_v4_boundary_test.npy").ravel().astype(np.float64)

    def orth_vs(zo, zt, bases_o, bases_t):
        Xo = np.column_stack(bases_o)
        Xt = np.column_stack(bases_t)
        ro = np.zeros(len(zo))
        rtf = []
        for a, b in folds:
            beta, *_ = np.linalg.lstsq(Xo[a], zo[a], rcond=None)
            ra = zo[a] - Xo[a] @ beta
            mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
            ro[b] = (zo[b] - Xo[b] @ beta - mu) / sd
            rtf.append((zt - Xt @ beta - mu) / sd)
        return ro, np.mean(rtf, 0)

    def finalize(ro, rt, tag):
        rt_p = orth_against(rt, [B_bad[:, j] for j in range(B_bad.shape[1])])
        o_al, t_al = align_method_b_smooth(ro, rt_p, folds)
        if disp_ks(o_al, t_al)[1] >= KS_GATE:
            o_al, t_al = force_exact_law(o_al, t_al)
        t_al = orth_against(t_al, [B_bad[:, j] for j in range(B_bad.shape[1])])
        o_al, t_al = force_exact_law(o_al, t_al)
        d, ks = disp_ks(o_al, t_al)
        mc = max(abs(cos(t_al, B_bad[:, j])) for j in range(B_bad.shape[1]))
        log(f"  resid {tag}: disp={d:.6f} KS={ks:.6f} max|cos|={mc:.2e}")
        return o_al, t_al

    residuals_o, residuals_t, names = [], [], []
    for tag, op, tp in families:
        if not Path(op).exists():
            log(f"  MISSING {tag} — skip")
            continue
        zo, zt = zrk(np.load(op).ravel()), zrk(np.load(tp).ravel())
        ro, rt = orth_vs(zo, zt, [z_v8_o, z_w12_o], [z_v8_t, z_w12_t])
        o, t = finalize(ro, rt, tag)
        residuals_o.append(o)
        residuals_t.append(t)
        names.append(tag)
        solo = float(roc_auc_score(y, zo))
        log(f"  family {tag} solo AUC={solo:.7f}")

    for tag, ro, rt in [("d3_lin", d3l_o, d3l_t), ("d3_bnd", d3b_o, d3b_t)]:
        o, t = finalize(ro, rt, tag)
        residuals_o.append(o)
        residuals_t.append(t)
        names.append(tag)

    R_o = np.column_stack(residuals_o)
    R_t = np.column_stack(residuals_t)
    M = R_o.shape[1]
    w_grid = np.linspace(0.0, 0.20, 21)  # non-negative

    def nested_nn(z_base, label):
        base_folds = fold_aucs(y, z_base, folds)
        nested = np.zeros(ntr)
        fold_w = []
        for f, (a, b) in enumerate(folds):
            other = [i for i in range(5) if i != f]
            w = np.zeros(M)
            for _ in range(2):
                for j in range(M):
                    bw, bs = w[j], -1e9
                    for cw in w_grid:
                        wt = w.copy()
                        wt[j] = cw
                        lifts = [
                            float(roc_auc_score(y[folds[oi][1]], z_base[folds[oi][1]] + R_o[folds[oi][1]] @ wt))
                            - base_folds[oi]
                            for oi in other
                        ]
                        s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                        if s > bs:
                            bs, bw = s, float(cw)
                    w[j] = bw
            nested[b] = z_base[b] + R_o[b] @ w
            fold_w.append(w)
            log(f"  {label} F{f} nonzero={int(np.sum(w>1e-8))}/{M}")
        mean_w = np.mean(fold_w, 0)
        nf = fold_aucs(y, nested, folds)
        deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
        pos = int(sum(x > 0 for x in deltas))
        auc = float(roc_auc_score(y, nested))
        log(f"  {label}: AUC={auc:.7f} Δe5={[round(x,2) for x in deltas]} pos={pos}/5")
        log(f"  {label} weights: { {n: round(float(w),4) for n,w in zip(names, mean_w) if w>1e-8} }")
        return nested, mean_w, deltas, pos, auc

    nested_ref, w_ref, d_ref, pos_ref, auc_ref = nested_nn(z_ref_o, "on_z_ref_old")
    nested_v8, w_v8, d_v8, pos_v8, auc_v8 = nested_nn(z_v8_o, "on_z_v8_old")

    d1_v5_o = R_o @ w_ref
    d1_v5_t = R_t @ w_ref
    d1_v5_o, d1_v5_t = finalize(d1_v5_o, d1_v5_t, "d1_v5_super")
    save_pair("d1_v5_super", d1_v5_o, d1_v5_t)
    save_pair("z_v5_overnight_ref", nested_ref, z_s5_purged)  # test placeholder; real test below

    # Build test upgrade from purged S5
    axis = unitize(d1_v5_t)
    axis = orth_against(axis, [B_bad[:, j] for j in range(B_bad.shape[1])])
    axis = unitize(axis)
    # reinforce keeping tree negative: do not add +q_bad_tree
    np.save(ART / "delta_v5_axis_test.npy", axis)

    steps = [0.25e-4, 0.45e-4, 0.80e-4, 1.35e-4, 2.00e-4]
    staged = []
    for i, target in enumerate(steps, 1):
        w, got = weight_for_spearman_dist(z_s5_purged, axis, target)
        z = z_s5_purged + w * axis
        # ensure no positive bad_S3 / bad_ldr introduced
        delta = z - z_s5_purged
        for qb in [q_bad_S3, q_bad_ldr]:
            c = float(np.dot(delta, qb))
            if c > 0:
                z = z - c * qb
                delta = z - z_s5_purged
        p = to_probs(z, y)
        fn = f"sub_gm_v5_overnight_slot{i}_NOSUBMIT.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(SUB / fn, index=False)
        np.save(ART / f"slot{i}_z.npy", z)
        np.save(ART / f"slot{i}_p.npy", p)
        tel = {
            "slot": i,
            "file": fn,
            "1mrho": float(1 - spearmanr(z, z_s5_purged).correlation),
            "cos_bad_S3": cos(delta, q_bad_S3),
            "cos_bad_ldr": cos(delta, q_bad_ldr),
            "cos_bad_tree": cos(delta, q_bad_tree),
            "mean_p": float(p.mean()),
        }
        log(f"  staged slot{i}: 1−ρ={tel['1mrho']*1e4:.3f}e-4 cos(S3)={tel['cos_bad_S3']:+.4f} cos(ldr)={tel['cos_bad_ldr']:+.4f} → {fn}")
        staged.append(tel)

    # Update fast cache
    payload = {
        "z_ref_old_oof": z_ref_o.astype(np.float32),
        "d1_v5_super_oof": d1_v5_o.astype(np.float32),
        "d1_v5_super_test": d1_v5_t.astype(np.float32),
        "z_S5_purged": z_s5_purged.astype(np.float32),
        "axis_v5": axis.astype(np.float32),
        "y": y.astype(np.int8),
    }
    np.savez_compressed(PRED / "gm_v5_fast_cache.npz", **payload)
    log(f"  wrote preds/gm_v5_fast_cache.npz")

    report = {
        "z_ref_old_auc": float(roc_auc_score(y, z_ref_o)),
        "v5_on_ref": {"auc": auc_ref, "delta_e5": d_ref, "pos": pos_ref, "w": {n: float(w) for n, w in zip(names, w_ref)}},
        "v5_on_v8": {"auc": auc_v8, "delta_e5": d_v8, "pos": pos_v8, "w": {n: float(w) for n, w in zip(names, w_v8)}},
        "staged": staged,
        "families_used": names,
        "NO_SUBMIT": True,
    }
    (ART / "phase5.json").write_text(json.dumps(report, indent=2, default=float))
    save_state(phase="phase5_done", phase5=report)
    return report


def write_master_report():
    p1 = json.loads((ART / "phase1.json").read_text()) if (ART / "phase1.json").exists() else {}
    p5 = json.loads((ART / "phase5.json").read_text()) if (ART / "phase5.json").exists() else {}
    fam_scores = {}
    for tag in [
        "cat_stageB_bag9", "xgb_stageB_monotone_lm_bag6", "lgb_stageB_extratrees_bag6",
        "xgb_stageB_tail_specialist_bag6", "lgb_stageB_colsub_deep_bag6", "mlp_stageB_bag5",
    ]:
        if exists_pair(tag):
            o, _ = load_pair(tag)
            # need y — load from cache
            y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
            zo = zrk(o)
            fam_scores[tag] = {"auc": float(roc_auc_score(y, zo)), "folds": fold_aucs(y, zo, folds)}

    md = []
    md.append("# GM V5 Overnight 8hr Super Factory")
    md.append("")
    md.append(f"**NO_SUBMIT=True**. Elapsed **{round(time.time()-T0,1)}s**.")
    md.append("")
    md.append("## Phase 1 — z_ref_old + purge + X_v5")
    md.append("")
    md.append(f"- `z_ref_old` AUC=`{p1.get('z_ref_old_auc')}` folds=`{p1.get('z_ref_old_folds')}`")
    md.append(f"- NN δ_v4_combo: AUC=`{p1.get('combo_nn_auc')}` Δe5=`{p1.get('combo_nn_delta_e5')}` pos=`{p1.get('combo_nn_pos')}`/5 w=`{p1.get('combo_nn_w')}`")
    md.append(f"- S5 purge coeffs: `{p1.get('purge')}`")
    if (ART / "X_v5_meta.json").exists():
        md.append(f"- X_v5 meta: `{json.loads((ART/'X_v5_meta.json').read_text())}`")
    md.append("")
    md.append("## Phases 2–4 — New Stage-B families")
    md.append("")
    for tag, sc in fam_scores.items():
        md.append(f"- **{tag}**: AUC={sc['auc']:.7f} folds={sc['folds']}")
    md.append("")
    md.append("## Phase 5 — Portfolio + staging")
    md.append("")
    md.append(f"- on z_ref_old: `{p5.get('v5_on_ref')}`")
    md.append(f"- on z_v8_old: `{p5.get('v5_on_v8')}`")
    md.append("")
    md.append("| slot | 1−ρ | cos(bad_S3) | cos(bad_ldr) | cos(bad_tree) | file |")
    md.append("|---|---:|---:|---:|---:|---|")
    for s in p5.get("staged", []):
        md.append(
            f"| {s['slot']} | {s['1mrho']*1e4:.3f}e-4 | {s['cos_bad_S3']:+.4f} | "
            f"{s['cos_bad_ldr']:+.4f} | {s['cos_bad_tree']:+.4f} | `{s['file']}` |"
        )
    md.append("")
    md.append("`NO_SUBMIT=True`.")
    (LOG / "GM_V5_OVERNIGHT_8HR_REPORT.md").write_text("\n".join(md))
    (LOG / "GM_V5_OVERNIGHT_8HR_REPORT.json").write_text(json.dumps({"phase1": p1, "phase5": p5, "families": fam_scores}, indent=2, default=float))
    log(f"Wrote {LOG / 'GM_V5_OVERNIGHT_8HR_REPORT.md'}")


def main():
    assert NO_SUBMIT, "STRICT NO_SUBMIT"
    log("GM_V5_OVERNIGHT_8HR_SUPER_FACTORY START")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    # Resume control via PHASE_START env (1..5)
    start = int(os.environ.get("PHASE_START", "1"))
    if start <= 1:
        phase1(y, folds, test_ids)
    if start <= 2:
        phase2(y, folds)
    if start <= 3:
        phase3(y, folds)
    if start <= 4:
        phase4(y, folds)
    if start <= 5:
        phase5(y, folds, test_ids)
    write_master_report()
    log(f"GM_V5_OVERNIGHT_8HR_COMPLETE elapsed={round(time.time()-T0,1)}s")


if __name__ == "__main__":
    main()
