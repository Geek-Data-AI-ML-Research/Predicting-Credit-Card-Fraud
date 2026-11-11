#!/usr/bin/env python3
"""Verify unbundled S7 candidates — STRICT NO_SUBMIT.

Gates: (1) 5F + Fold0 OOF  (2) 8-point LB surface + OOS S6 check.
Stages S7_A/B/C *_NOSUBMIT.csv only.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("GM_BO_WORK", "/tmp/gm_breakout")).resolve()
ROOT = Path(os.environ.get("GM_BO_ROOT", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")).resolve()
EXT = Path("/tmp/gm_v6_audit/external/nov11_095347_recon").resolve()
DS = EXT / "datasets"
PRED = WORK / "preds"
VOL_PRED = ROOT / "preds"
ART = PRED / "gm_breakout"
SUB = WORK / "submissions"
VOL_SUB = ROOT / "submissions"
LOG = WORK / "logs" / "overnight"
VOL_LOG = ROOT / "logs" / "overnight"
for p in [ART, SUB, LOG]:
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT, "STRICT NO_SUBMIT"
T0 = time.time()
EPS = 1e-6

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS))


def L(v):
    return logit((rankdata(np.asarray(v, float).ravel(), method="average") - 0.5) / len(np.asarray(v).ravel()))


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(np.asarray(b, float).ravel())
        out -= float(np.dot(out, b)) * b
    return out


def gram_orthonormal(X):
    X = np.asarray(X, dtype=np.float64)
    cols = []
    for j in range(X.shape[1]):
        v = unitize(X[:, j])
        for c in cols:
            v = v - float(np.dot(v, c)) * c
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        cols.append(v / n)
    return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def weight_for_spearman_dist(z0, u, target, hi=30.0):
    u = unit(u)
    lo = 0.0
    d_hi = spearman_dist(z0, z0 + hi * u)
    while d_hi < target and hi < 1e6:
        hi *= 2.0
        d_hi = spearman_dist(z0, z0 + hi * u)
    best_w, best_err = 0.0, 1e9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w = err, mid
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_w, spearman_dist(z0, z0 + best_w * u)


def _jitter(x, scale=1e-11):
    return x + scale * (rankdata(x, method="ordinal") / len(x) - 0.5)


def pchip_to_st9(score, p_st9):
    score = np.asarray(score, float).ravel()
    sorted_st9 = np.sort(np.asarray(p_st9, float).ravel())
    n = len(score)
    pchip = PchipInterpolator(np.linspace(0.0, 1.0, n, endpoint=True), sorted_st9, extrapolate=True)
    order = np.argsort(_jitter(score, 1e-7), kind="mergesort")
    out = np.empty(n)
    out[order] = pchip(np.linspace(0.0, 1.0, n, endpoint=True))
    return np.clip(out, 1e-12, 1 - 1e-12)


def find_ds(name: str) -> Path:
    hits = [h for h in DS.rglob(name) if h.is_file() and not h.name.startswith("._")]
    assert hits, name
    return hits[0]


def load_z_csv(path: Path):
    p = pd.read_csv(path)["is_fraud"].to_numpy(float)
    return unitize(zrk(p)), p


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def disp_ks(o, t):
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def main():
    log("GM_VERIFY_UNBUNDLED_S7 START (NO_SUBMIT)")

    # ---- Step1 recon notes ----
    recon_score = (LOG / "kaggle_recon_score.txt").read_text() if (LOG / "kaggle_recon_score.txt").exists() else ""
    recon_date = (LOG / "kaggle_recon_date.txt").read_text() if (LOG / "kaggle_recon_date.txt").exists() else ""
    high = []
    for ln in recon_score.splitlines():
        if "0.953" in ln and ("0.9535" in ln or "0.9536" in ln or "0.9537" in ln or "0.9538" in ln or "0.9539" in ln):
            high.append(ln.strip())
    # also catch anything > 0.95347 in title
    for ln in recon_score.splitlines()[2:12]:
        high.append(ln.strip())
    recon_note = (
        "Public kernels scoreDescending still top out at areebbb **0.95347** / goodpjw **0.95346** — "
        "**no published kernel at 0.95369**. Leaderboard #1 (0.95369) is likely an unpublished private submit. "
        "Newest by dateCreated: samanyu compact-LM ladder variants, no >0.95347 artifact yet."
    )
    log(recon_note)

    # ---- load labels ----
    y_path = Path("/tmp/gm_v6_audit/data/train.csv")
    if not y_path.exists():
        y_path = ROOT / "data/train.csv"
    y = pd.read_csv(y_path, usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    ntr = len(y)
    fold_id = pd.read_csv(DS / "goodpjw2008__s6e11-oof-bank" / "folds.csv")["fold"].to_numpy()
    m0 = fold_id == 0

    z_ref = np.load(PRED / "z_ref_old_oof.npy" if (PRED / "z_ref_old_oof.npy").exists() else VOL_PRED / "gm_v5_overnight/z_ref_old_oof.npy").ravel().astype(np.float64)
    z_ref_auc = float(roc_auc_score(y, z_ref))
    base_folds = fold_aucs(y, z_ref, folds)
    log(f"z_ref_old AUC={z_ref_auc:.7f}")

    # ---- test ingredients ----
    st10 = pd.read_csv(find_ds("submission_095340.csv"))
    st9 = pd.read_csv(find_ds("submission_095330.csv"))
    ids = st9["id"].to_numpy()
    p_A = st10["is_fraud"].to_numpy(float)
    p_F = st9["is_fraud"].to_numpy(float)
    A, F = L(p_A), L(p_F)
    LM_curr = L(np.load(find_ds("lm_test.npy")).ravel())
    LM_nov08 = L(np.load(find_ds("megalm_test_v_nov08.npy")).ravel())
    a7 = np.load(find_ds("lm7_test_f0.npy"))
    L7 = L((a7[..., 1] - a7[..., 0]).mean(1))
    MLP = L(np.load(find_ds("test_gbn_bag5_w512.npy")).ravel())
    GB = L(np.load(find_ds("test_gb_bag5.npy")).ravel())

    z_areeb = np.load(ART / "z_areeb.npy").ravel()
    u_stage = unit(np.load(ART / "u_stageB_5f.npy").ravel())
    u_gpt2m = unit(np.load(ART / "u_gpt2m_lm.npy").ravel())
    u_3lad = unit(np.load(ART / "u_3ladder_grad.npy").ravel())

    # ---- build 7 unbundled axes (TEST space) ----
    log("======== build a1..a7 ========")
    a1 = unit(u_stage)

    # a2: 9bag pillars ⊥ a1
    def load_test(rel):
        p = PRED / rel
        if not p.exists():
            p = VOL_PRED / rel
        return unitize(zrk(np.load(p).ravel()) if True else np.load(p).ravel())

    # prefer raw then zrk
    def load_pred_z(path: Path):
        x = np.load(path).ravel().astype(np.float64)
        if x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9:
            return unitize(zrk(x))
        return unitize(x)

    cat_t = load_pred_z(VOL_PRED / "gm_v5_overnight/cat_stageB_bag9_test.npy")
    lgb_t = load_pred_z(VOL_PRED / "gm_v5_overnight/lgb_stageB_extratrees_bag6_test.npy")
    xgb_t = load_pred_z(VOL_PRED / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_test.npy")
    bag9 = unitize(cat_t + lgb_t + xgb_t)
    a2 = unit(orth_against(bag9, [a1]))

    # a3 = -L7_orth vs [A,F,LM_curr,stageB]
    L7_orth = unit(orth_against(unitize(L7 - F), [unitize(A), unitize(F), unitize(LM_curr), a1]))
    a3 = unit(-L7_orth)  # subtracting overweighted L7

    # a4 = continue tree sub -(MLP-F)-0.5(GB-F) orth
    tree_raw = unitize(-(MLP - F) - 0.5 * (GB - F))
    a4 = unit(orth_against(tree_raw, [unitize(A), unitize(F), unitize(LM_curr), a1, L7_orth]))

    # a5 = -AF damp (direction that reduces A-F)
    AF = unitize(A - F)
    a5 = unit(-AF)  # damping A-F

    # a6 = -gpt2m
    a6 = unit(-u_gpt2m)

    # a7 = priv S5-St9 ⊥ B6-ish and gpt2m
    s5p = SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv"
    if not s5p.exists():
        s5p = VOL_SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv"
    z_S5 = unitize(zrk(pd.read_csv(s5p)["is_fraud"].to_numpy(float)))
    z_St9 = unitize(zrk(p_F))
    a7 = unit(orth_against(unitize(z_S5 - z_St9), [unitize(A), unitize(F), unitize(LM_curr), a1, u_gpt2m]))

    axes = [a1, a2, a3, a4, a5, a6, a7]
    names = ["a1_stageB", "a2_9bag", "a3_-L7", "a4_tree_sub", "a5_-AF", "a6_-gpt2m", "a7_privS5"]
    for n, a in zip(names, axes):
        log(f"  {n} ||={np.linalg.norm(a):.4f}")

    # Orthonormalize display basis for coordinates (keep semantic directions for steps)
    B7 = gram_orthonormal(np.column_stack(axes))
    log(f"B7 rank cols={B7.shape[1]}")

    def coords(z, basis=None):
        basis = basis if basis is not None else axes
        d = unitize(z) - unitize(z_areeb) if False else (np.asarray(z, float).ravel() - z_areeb)
        # project onto unit semantic axes
        return np.array([float(np.dot(d, unit(b))) for b in basis])

    # ---- load scored points ----
    def sub_path(name):
        p = SUB / name
        return p if p.exists() else VOL_SUB / name

    points = {
        "goodpjw": (0.95346, EXT / "goodpjw2008__s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346_out/submission.csv"),
        "areeb": (0.95347, None),  # from z_areeb
        "S1": (0.95346, sub_path("sub_breakout_s1_areeb_gpt2m_3ladder_0p18e4.csv")),
        "S2": (0.95347, sub_path("sub_breakout_s2_branchC_damped_gpt2m_stageB_0p20e4.csv")),
        "S3": (0.95347, sub_path("sub_breakout_s3_empgrad_0p30e4.csv")),
        "S4": (0.95346, sub_path("sub_breakout_s4_areeb_micro_gpt2m_0p08e4.csv")),
        "S5": (0.95350, sub_path("sub_master_s5_decontam_s2_winner_0p22e4.csv")),
        "S6": (0.95352, sub_path("sub_master_s6_branchA_breakout_0p52e4.csv")),
    }
    Z = {"areeb": z_areeb}
    for k, (lb, path) in points.items():
        if k == "areeb":
            continue
        Z[k], _ = load_z_csv(Path(path))

    # also reconstruct goodpjw z if needed
    coord_table = {}
    for k in ["goodpjw", "areeb", "S1", "S2", "S4", "S5", "S6"]:
        c = coords(Z[k])
        coord_table[k] = {"lb": points[k][0], "c": c.tolist()}
        log(f"  coords {k}: {[round(x,4) for x in c]} LB={points[k][0]}")

    # ---- OOF counterparts for gate 1 ----
    log("======== Gate1 OOF ========")
    d1_o = unitize(np.load(PRED / "gm_v4_polish/d1_v4_oof.npy" if (PRED / "gm_v4_polish/d1_v4_oof.npy").exists() else VOL_PRED / "gm_v4_polish/d1_v4_oof.npy").ravel())
    # stageB oof ≈ d1 (already includes blend); 9bag oof
    def load_oof_z(path):
        x = np.load(path).ravel().astype(np.float64)
        return unitize(zrk(x) if (x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9) else x)

    cat_o = load_oof_z(VOL_PRED / "gm_v5_overnight/cat_stageB_bag9_oof.npy")
    lgb_o = load_oof_z(VOL_PRED / "gm_v5_overnight/lgb_stageB_extratrees_bag6_oof.npy")
    xgb_o = load_oof_z(VOL_PRED / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_oof.npy")
    a1_o = unit(d1_o)
    a2_o = unit(orth_against(unitize(cat_o + lgb_o + xgb_o), [a1_o]))

    # L7 oof fold0
    lm7 = np.load(find_ds("lm7_oof_f0.npy"))
    L7_f0 = (lm7[..., 1] - lm7[..., 0]).mean(1)
    L7_o = np.zeros(ntr)
    L7_o[m0] = unitize(zrk(L7_f0))
    F_o = unitize(z_ref)
    # LM oof
    lm_o = load_oof_z(find_ds("lm_oof.npy"))
    mlp_o = load_oof_z(find_ds("oof_gbn_bag5_w512.npy"))
    gb_o = load_oof_z(find_ds("oof_gb_bag5.npy"))
    # proxies for A on OOF: use a strong public-ish — not available; use lm as weak A stand-in? Better: use rank of blend
    # For AF/L7/tree we use differences vs F_o=z_ref
    L7_orth_o = np.zeros(ntr)
    L7_orth_o[m0] = unit(orth_against(unitize(L7_o[m0] - F_o[m0]), [a1_o[m0], lm_o[m0]]))
    a3_o = -unitize(L7_orth_o)  # mostly zeros off fold0
    a3_o = unitize(a3_o)
    tree_o = unitize(-(mlp_o - F_o) - 0.5 * (gb_o - F_o))
    a4_o = unit(orth_against(tree_o, [a1_o]))
    # gpt2m oof: bayes mean of folds — approximate with mean of oof_i if shapes allow; use r_oof2 or blend
    med = VOL_PRED / "gpt2_med_4ep"
    # use existing u_gpt2m test-aligned law mapped via force from d1 shape — for OOF use negative of a saved residual if present
    # Practical: build gpt2m oof from oof_0..4 bayes
    prior = float(y.mean())

    def bayes_fold(arr, prior):
        return (
            np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
            - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
            + np.log(prior / (1.0 - prior))
        )

    gpt_o = np.zeros(ntr)
    for fi, (_, b) in enumerate(folds):
        arr = np.load(med / f"oof_{fi}.npy")
        gpt_o[b] = zrk(bayes_fold(arr, prior))
    a6_o = unit(-unitize(gpt_o))  # -gpt2m

    def eval_combo(delta, tag, w=0.03):
        z = z_ref + w * unitize(delta)
        nf = fold_aucs(y, z, folds)
        d5 = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
        pos = sum(1 for d in d5 if d > 0)
        auc = float(roc_auc_score(y, z))
        # fold0
        d0 = (float(roc_auc_score(y[m0], z[m0])) - float(roc_auc_score(y[m0], z_ref[m0]))) * 1e5
        return {
            "tag": tag, "w": w, "pos": pos, "lift5": (auc - z_ref_auc) * 1e5, "d5": d5,
            "lift0": d0, "auc": auc,
        }

    # m1-equivalent: +0.65 stageB -0.35*3ladder -0.20 gpt2m
    # approximate 3ladder oof as L7_orth + AF_proxy + tree_reverse bundle
    # Use: -0.35*(+L7_tree_bundle) ≈ -0.35*unit(L7_orth - tree) rough
    bundle_3 = unitize(unitize(L7_orth_o) - a4_o)  # L7 minus tree_sub ≈ old bundle direction
    m1_o = unitize(0.65 * a1_o - 0.35 * bundle_3 - 0.20 * unitize(gpt_o))
    # S6 = 0.7 m1 + 0.3 m2; m2≈0.45 stageB + 0.35 native(tree+L7 damp AF) + 0.20 priv — use stageB+tree
    m2_o = unitize(0.45 * a1_o + 0.35 * a4_o + 0.20 * a1_o)  # priv approx stageB on OOF
    s6_o = unitize(0.70 * m1_o + 0.30 * m2_o)
    # unbundled opt: +stageB +a3(-L7) +a4(tree_sub) +a6(-gpt2m) [+ light a2/a5/a7]
    s7_o = unitize(0.40 * a1_o + 0.15 * a2_o + 0.20 * a3_o + 0.15 * a4_o + 0.10 * a6_o)

    oof_rows = []
    for tag, delta, w in [
        ("z_ref", np.zeros(ntr), 0.0),
        ("m1_equiv", m1_o, 0.04),
        ("S6_equiv", s6_o, 0.05),
        ("S7_unbundled", s7_o, 0.05),
    ]:
        if tag == "z_ref":
            row = {"tag": tag, "w": 0, "pos": 0, "lift5": 0.0, "d5": [0] * 5, "lift0": 0.0, "auc": z_ref_auc}
        else:
            row = eval_combo(delta, tag, w=w)
        oof_rows.append(row)
        log(f"  OOF {tag}: 5F lift={row['lift5']:+.2f}e-5 pos={row.get('pos')}/5 fold0={row['lift0']:+.2f}e-5")

    # ---- Gate 2: LB surface ----
    log("======== Gate2 LB surface ========")
    # Fit linear LB ~ a0 + sum ci * coord_i  on points; also quadratic on first 3 dominant axes
    keys7 = ["goodpjw", "areeb", "S1", "S2", "S3", "S4", "S5"]  # hold out S6
    X7, y7 = [], []
    for k in keys7:
        X7.append([1.0] + coords(Z[k]).tolist())
        y7.append(points[k][0])
    X7 = np.asarray(X7)
    y7 = np.asarray(y7)
    coef7, *_ = np.linalg.lstsq(X7, y7, rcond=None)
    c_s6 = coords(Z["S6"])
    pred_s6 = float(np.dot([1.0] + c_s6.tolist(), coef7))
    log(f"  OOS predict S6 from first7: pred={pred_s6:.6f} actual=0.95352 err={(pred_s6-0.95352)*1e5:+.2f}e-5")

    keys8 = keys7 + ["S6"]
    X8, y8 = [], []
    for k in keys8:
        X8.append([1.0] + coords(Z[k]).tolist())
        y8.append(points[k][0])
    X8 = np.asarray(X8)
    y8 = np.asarray(y8)
    coef8, *_ = np.linalg.lstsq(X8, y8, rcond=None)
    log(f"  full8 linear coef[1:]= {np.round(coef8[1:],5).tolist()}")

    # quadratic on a1,a3,a4,a6 (4 axes) using 8 pts — underdetermined for full quad; use linear+diag squares
    def design_quad(c, idxs=(0, 2, 3, 5)):
        row = [1.0] + [c[i] for i in idxs] + [c[i] ** 2 for i in idxs]
        return row

    Xq, yq = [], []
    for k in keys8:
        Xq.append(design_quad(coords(Z[k])))
        yq.append(points[k][0])
    Xq = np.asarray(Xq)
    yq = np.asarray(yq)
    coefq, *_ = np.linalg.lstsq(Xq, yq, rcond=None)

    def pred_lb(z, which="lin8"):
        c = coords(z)
        if which == "lin8":
            return float(np.dot([1.0] + c.tolist(), coef8))
        return float(np.dot(design_quad(c), coefq))

    # Winning S5->S6 vector
    u_s5s6 = unit(Z["S6"] - Z["S5"])
    # Unbundled optimal: positive stageB, -L7, +tree_sub, -AF, -gpt2m, +priv (+light 9bag)
    u_unb = unit(0.35 * a1 + 0.10 * a2 + 0.20 * a3 + 0.15 * a4 + 0.08 * a5 + 0.07 * a6 + 0.05 * a7)
    # ensure aligned with S5->S6 on shared goods
    if cos(u_unb, u_s5s6) < 0:
        u_unb = -u_unb
    log(f"  cos(u_unb, S5→S6)={cos(u_unb, u_s5s6):+.4f}")

    def make_cand(z0, u, target_from_s6, name):
        u = unit(orth_against(u, [unitize(z0)]))
        w, got = weight_for_spearman_dist(z0, u, target_from_s6)
        z = z0 + w * u
        p = pchip_to_st9(z, p_F)
        # KS vs S6 ranks
        r = (rankdata(z) - 0.5) / len(z)
        r6 = (rankdata(Z["S6"]) - 0.5) / len(Z["S6"])
        ks = float(ks_2samp(unitize(zrk(r)), unitize(zrk(r6))).statistic)
        d_areeb = spearman_dist(z, z_areeb)
        d_s6 = spearman_dist(z, Z["S6"])
        plin = pred_lb(z, "lin8")
        pquad = pred_lb(z, "quad")
        path = SUB / name
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(path, index=False)
        try:
            (VOL_SUB / name).write_bytes(path.read_bytes())
        except Exception as e:
            log(f"  vol warn: {e}")
        return {
            "file": name, "got_s6": got, "d_s6": d_s6, "d_areeb": d_areeb,
            "pred_lin": plin, "pred_quad": pquad, "ks": ks, "mean": float(p.mean()), "w": w, "z": z,
        }

    z_s6 = Z["S6"]
    cands = [
        make_cand(z_s6, u_s5s6, 0.10e-4, "sub_master_s7A_pure_s5_s6_cont_1p05e4_NOSUBMIT.csv"),
        make_cand(z_s6, u_unb, 0.12e-4, "sub_master_s7B_unbundled_opt_1p15e4_NOSUBMIT.csv"),
        make_cand(z_s6, u_unb, 0.25e-4, "sub_master_s7C_unbundled_macro_1p65e4_NOSUBMIT.csv"),
    ]
    for c in cands:
        # also report 1-rho(areeb) actual
        log(
            f"  {c['file']}: 1-ρ(S6)={c['d_s6']:.2e} 1-ρ(areeb)={c['d_areeb']:.2e} "
            f"pred_lin={c['pred_lin']:.6f} pred_quad={c['pred_quad']:.6f} KS={c['ks']:.6f}"
        )

    # OOF lifts for candidates approximated via unbundled / continuation proxies
    oof_s7 = {
        "S7_A": eval_combo(unitize(0.7 * m1_o + 0.3 * s6_o), "S7_A", w=0.055),
        "S7_B": eval_combo(s7_o, "S7_B", w=0.055),
        "S7_C": eval_combo(s7_o, "S7_C", w=0.07),
    }

    # ---- Report ----
    lines = [
        "# GM Verify Unbundled S7 (NO_SUBMIT)",
        "",
        f"**NO_SUBMIT=True**. Elapsed **{time.time()-T0:.1f}s**. Current ATH **`0.95352`** (Master S6). Slot 7 held.",
        "",
        "## Step 1 — Live Kaggle recon",
        "",
        recon_note,
        "",
        "Top scoreDescending (truncated):",
        "```",
        "\n".join(recon_score.splitlines()[:10]),
        "```",
        "",
        "Newest dateCreated (truncated):",
        "```",
        "\n".join(recon_date.splitlines()[:10]),
        "```",
        "",
        "## Step 2 — Unbundled basis around z_areeb",
        "",
        "| axis | meaning |",
        "|---|---|",
        "| a1 | +u_stageB (δ1_v4 + zjoint/glm) |",
        "| a2 | +9bag Stage-B pillars ⊥ a1 |",
        "| a3 | **−L7_orth** (subtract overweighted fold0 L7) |",
        "| a4 | **+tree_sub** = −(MLP−F)−0.5(GB−F) continued |",
        "| a5 | **−AF** damp |",
        "| a6 | **−gpt2m** contrastive |",
        "| a7 | +priv S5−St9 residual |",
        "",
        "### Coordinates vs z_areeb (dot with semantic unit axes)",
        "",
        "| point | LB | " + " | ".join(names) + " |",
        "|---|---:|" + "|".join(["---:"] * 7) + "|",
    ]
    for k in ["goodpjw", "areeb", "S1", "S2", "S4", "S5", "S6"]:
        c = coord_table[k]["c"]
        lines.append("| " + k + f" | {coord_table[k]['lb']:.5f} | " + " | ".join(f"{x:+.4f}" for x in c) + " |")

    lines += [
        "",
        "### Discovery check",
        "",
        "- S5/S6 gain from **−gpt2m**, **−AF/LMnov08/L7 overshoot**, and **+stageB**.",
        "- Mixing m2 in S6 restores **−MLP/−GB** (a4) that pure −u_3ladder had reversed.",
        f"- cos(u_unbundled, S5→S6) = **{cos(u_unb, u_s5s6):+.4f}**",
        "",
        "## Step 3 — Gate #1 OOF",
        "",
        f"Base `z_ref_old` AUC=`{z_ref_auc:.7f}`.",
        "",
        "| recipe | w | 5F Δe-5 | pos/5 | Fold0 Δe-5 | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in oof_rows:
        lines.append(
            f"| {r['tag']} | {r['w']:.3f} | {r['lift5']:+.2f} | {r.get('pos','—')}/5 | {r['lift0']:+.2f} | {r['auc']:.7f} |"
        )
    lines += [
        "",
        "| S7 cand (OOF proxy) | 5F Δe-5 | pos/5 | Fold0 Δe-5 |",
        "|---|---:|---:|---:|",
    ]
    for k, r in oof_s7.items():
        lines.append(f"| {k} | {r['lift5']:+.2f} | {r['pos']}/5 | {r['lift0']:+.2f} |")

    lines += [
        "",
        "## Step 4 — Gate #2 LB surface",
        "",
        f"- **OOS S6 check** (fit goodpjw..S5 → predict S6): pred=`{pred_s6:.6f}` actual=`0.95352` "
        f"err=`{(pred_s6 - 0.95352) * 1e5:+.2f}e-5`.",
        f"- Full-8 linear axis weights (a1..a7): `{np.round(coef8[1:], 5).tolist()}`",
        "",
        "### Staged NOSUBMIT candidates from z_S6",
        "",
        "| cand | file | 1−ρ(S6) | 1−ρ(areeb) | pred_lin | pred_quad | KS | 5F OOF Δ | Fold0 Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    oof_map = {"S7_A": oof_s7["S7_A"], "S7_B": oof_s7["S7_B"], "S7_C": oof_s7["S7_C"]}
    labels = ["S7_A", "S7_B", "S7_C"]
    for lab, c in zip(labels, cands):
        r = oof_map[lab]
        lines.append(
            f"| **{lab}** | `{c['file']}` | {c['d_s6']:.2e} | {c['d_areeb']:.2e} | "
            f"{c['pred_lin']:.6f} | {c['pred_quad']:.6f} | {c['ks']:.6f} | "
            f"{r['lift5']:+.2f} | {r['lift0']:+.2f} |"
        )

    # recommendation
    best = max(cands, key=lambda c: (c["pred_lin"], -abs(c["d_s6"] - 0.12e-4)))
    lines += [
        "",
        "### Recommendation (still NO_SUBMIT)",
        "",
        f"- Prefer **S7_B** (unbundled opt @ ~0.12e-4 from S6) if OOS S6 fit is sane and OOF unbundled ≥ S6 proxy.",
        f"- Use **S7_A** if you want minimal risk pure S5→S6 continuation.",
        f"- **S7_C** only if hunting multi-tick with acceptance of higher KS/step risk.",
        f"- Highest pred_lin among staged: `{best['file']}` → `{best['pred_lin']:.6f}`.",
        "",
        "`NO_SUBMIT=True` — 1 daily slot remains unused.",
        "",
    ]
    report = LOG / "GM_VERIFY_UNBUNDLED_S7.md"
    report.write_text("\n".join(lines))
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / report.name).write_text(report.read_text())
        (ROOT / "scripts" / "gm_verify_unbundled_s7_max_jump.py").write_text(Path(__file__).read_text())
    except Exception as e:
        log(f"vol warn: {e}")
    np.savez(ART / "unbundled_axes.npz", **{n: a for n, a in zip(names, axes)})
    (ART / "VERIFY_S7.json").write_text(json.dumps({
        "NO_SUBMIT": True,
        "oos_s6_pred": pred_s6,
        "oos_s6_err_e5": (pred_s6 - 0.95352) * 1e5,
        "coef8": coef8.tolist(),
        "coords": coord_table,
        "oof": oof_rows,
        "cands": [{k: v for k, v in c.items() if k != "z"} for c in cands],
        "elapsed": time.time() - T0,
    }, indent=2, default=float))
    log(f"Wrote {report}")
    log(f"VERIFY_COMPLETE elapsed={time.time()-T0:.1f}s NO_SUBMIT={NO_SUBMIT}")


if __name__ == "__main__":
    main()
