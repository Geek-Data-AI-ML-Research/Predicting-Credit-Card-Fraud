#!/usr/bin/env python3
"""GM Deep-Tech Research Engine for 0.95360+ — STRICT NO_SUBMIT=True.

Phases 0→4, resume-safe checkpoints under preds/gm_deep_tech_v7/.
Mac CPU/MPS: n_jobs=4, float32. Never calls kaggle competitions submit.
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("GM_V7_ROOT", "/tmp/gm_deep_tech")).resolve()
# Do NOT .resolve()/exists()-probe the volume path — external disk I/O can hang this box.
VOL = Path(os.environ.get("GM_V7_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
os.environ["INPUT_DIR"] = str(WORK)
sys.path.insert(0, str(WORK / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

T0 = time.time()
PRED_V5 = Path(os.environ.get("GM_V7_PRED_V5", "/tmp/gm_v5_work/preds"))
PRED_BO = Path(os.environ.get("GM_V7_PRED_BO", "/tmp/gm_breakout/preds"))
PRED_VOL = VOL / "preds"
EXT = Path(os.environ.get("GM_V7_EXT", "/tmp/gm_v6_audit/external/nov11_095347_recon"))
DS = EXT / "datasets"
DATA = WORK / "data"
ART = Path(os.environ.get("GM_V7_ART", str(WORK / "preds/gm_deep_tech_v7")))
LOG = Path(os.environ.get("GM_V7_LOG", str(WORK / "logs/overnight")))
SUB = Path(os.environ.get("GM_V7_SUB", str(WORK / "submissions")))
VOL_LOG = VOL / "logs/overnight"
VOL_SUB = VOL / "submissions"
VOL_ART = VOL / "preds/gm_deep_tech_v7"
for p in (ART, LOG, SUB):
    p.mkdir(parents=True, exist_ok=True)

STATE_PATH = ART / "STATE.json"
NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
KS_GATE = 0.0002
N_Q = 10001
N_JOBS = int(os.environ.get("GM_V7_NJOBS", "4"))
ATH = 0.95354
Z_REF_TARGET = 0.9526612


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
    st["NO_SUBMIT"] = True
    STATE_PATH.write_text(json.dumps(st, indent=2, default=float))


def sync_artifact(path: Path):
    """Best-effort mirror to volume (disabled by default — volume I/O hangs)."""
    if os.environ.get("GM_V7_SYNC", "0") != "1":
        return
    try:
        rel = None
        for base, dest_root in ((ART, VOL_ART), (LOG, VOL_LOG), (SUB, VOL_SUB)):
            try:
                rel = path.relative_to(base)
                dest = dest_root / rel
                break
            except ValueError:
                continue
        else:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            dest.write_bytes(path.read_bytes())
    except Exception as e:
        log(f"  sync skip {path.name}: {e}")


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
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std, jitter_scale), q),
        np.quantile(_jitter(oof_std, jitter_scale), q),
        extrapolate=True,
    )
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


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


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


def save_pair(tag, oof, test):
    op = ART / f"{tag}_oof.npy"
    tp = ART / f"{tag}_test.npy"
    np.save(op, np.asarray(oof, dtype=np.float32))
    np.save(tp, np.asarray(test, dtype=np.float32))
    log(f"  SAVED {tag} oof/test")
    sync_artifact(op)
    sync_artifact(tp)


def load_pair(tag):
    return np.load(ART / f"{tag}_oof.npy").ravel(), np.load(ART / f"{tag}_test.npy").ravel()


def exists_pair(tag):
    return (ART / f"{tag}_oof.npy").exists() and (ART / f"{tag}_test.npy").exists()


def find_first(*cands: Path) -> Path:
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(cands)


_DS_CACHE = {}


def find_ds(name: str) -> Path:
    if name in _DS_CACHE:
        return _DS_CACHE[name]
    # Prefer known exact paths before any rglob (volume rglob hangs).
    exact = [
        DS / "goodpjw2008__s6e11-compact-lm-seed7" / name,
        DS / "megayak__s6e11-target-first-lm" / name,
        DS / "moranzavdi__s6e11-strike9-lb-095330" / name,
        DS / "amanatar__s6e11-strike10-podium-apex-095340" / name,
        DS / "goodpjw2008__s6e11-fraud-series-npz" / name,
        DS / "goodpjw2008__s6e11-oof-bank" / name,
        Path("/tmp/gm_v5_work/external/banks/megayak_s6e11-target-first-lm") / name,
        Path("/tmp/gm_v5_work/external/nov08_part12") / name,
        PRED_V5 / name,
        PRED_BO / name,
    ]
    for p in exact:
        if p.exists():
            _DS_CACHE[name] = p
            return p
    hits = []
    for root in (DS, Path("/tmp/gm_v5_work/external"), PRED_V5, PRED_BO):
        if not root.exists():
            continue
        hits = [h for h in root.rglob(name) if h.is_file() and not h.name.startswith("._")][:3]
        if hits:
            break
    assert hits, name
    _DS_CACHE[name] = hits[0]
    return hits[0]


def L(p):
    p = np.asarray(p, dtype=np.float64).ravel()
    return unitize(zrk(p))


def pchip_to_st9(score, p_st9):
    score = np.asarray(score, float).ravel()
    sorted_st9 = np.sort(np.asarray(p_st9, float).ravel())
    n = len(score)
    pchip = PchipInterpolator(np.linspace(0.0, 1.0, n, endpoint=True), sorted_st9, extrapolate=True)
    order = np.argsort(_jitter(score, 1e-7), kind="mergesort")
    out = np.empty(n)
    out[order] = pchip(np.linspace(0.0, 1.0, n, endpoint=True))
    return np.clip(out, 1e-12, 1 - 1e-12)


def to_probs_mean0167(z, y, p_st9):
    """Map rank-logit z into St9 order-stat law then shift mean→0.016700."""
    z = np.asarray(z, float).ravel()
    p = pchip_to_st9(z, p_st9)

    def obj(d):
        return (float(np.mean(np.clip(p + d, 1e-12, 1 - 1e-12))) - 0.016700) ** 2

    # shift in probability space via logit nudge
    zl = logit(np.clip(p, 1e-12, 1 - 1e-12))

    def obj2(d):
        return (float(np.mean(expit(zl + d))) - 0.016700) ** 2

    d = float(minimize_scalar(obj2, bounds=(-2, 2), method="bounded").x)
    return expit(zl + d)


def eb_te_maps(ka, ya, keys_list, prior, m):
    """Fit TE map on ka/ya; apply to each key array in keys_list."""
    ka = np.asarray(ka)
    ya = np.asarray(ya, dtype=np.float64)
    ncat = int(max(ka.max(), *(k.max() for k in keys_list)) + 1)
    s = np.bincount(ka.astype(np.int64), weights=ya, minlength=ncat)
    n = np.bincount(ka.astype(np.int64), minlength=ncat)
    te = (s + prior * m) / (n + m)
    return [te[np.asarray(k, dtype=np.int64)] for k in keys_list]


def load_ab(path_o, path_t, folds, already_z=False):
    xo = np.load(path_o).ravel()
    xt = np.load(path_t).ravel()
    if not already_z:
        xo, xt = zrk(xo), zrk(xt)
    o, t = align_method_b_smooth(xo, xt, folds)
    if disp_ks(o, t)[1] >= 0.0001:
        o, t = force_exact_law(o, t)
    return o.astype(np.float32), t.astype(np.float32)


# ============================================================================
# Data bootstrap
# ============================================================================
def load_y_folds():
    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_csv = find_first(
        DS / "goodpjw2008__s6e11-oof-bank" / "folds.csv",
        Path("/tmp/gm_v5_work/external/banks/goodpjw2008_s6e11-oof-bank/folds.csv") if False else Path("/tmp/gm_v6_audit/external/nov11_095347_recon/datasets/goodpjw2008__s6e11-oof-bank/folds.csv"),
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv",
    )
    fold_id = pd.read_csv(fold_csv)["fold"].to_numpy()
    assert len(fold_id) == len(y)
    return y, folds, fold_id


def load_z_ref(folds):
    zo = np.load(find_first(PRED_V5 / "gm_v5_overnight/z_ref_old_oof.npy", ART / "z_ref_old_oof.npy", PRED_VOL / "gm_v5_overnight/z_ref_old_oof.npy")).ravel().astype(np.float64)
    zt = np.load(find_first(PRED_V5 / "gm_v5_overnight/z_ref_old_test.npy", ART / "z_ref_old_test.npy", PRED_VOL / "gm_v5_overnight/z_ref_old_test.npy")).ravel().astype(np.float64)
    return zo, zt


# ============================================================================
# PHASE 0 — Lock ATH + ray sweep
# ============================================================================
def phase0(y, folds, fold_id):
    log("======== PHASE 0: Lock S7=0.95354 ATH + S5→S6→S7 ray OOF ceiling ========")
    save_state(phase="phase0_start", ATH=ATH)
    if (ART / "phase0.json").exists() and os.environ.get("REDO_PHASE0", "0") != "1":
        out = json.loads((ART / "phase0.json").read_text())
        log(f"  SKIP phase0 — loaded t_star={out.get('t_star')}")
        return out

    z_ref, _ = load_z_ref(folds)
    z_ref_auc = float(roc_auc_score(y, z_ref))
    base_folds = fold_aucs(y, z_ref, folds)
    log(f"  z_ref_old AUC={z_ref_auc:.7f} (target {Z_REF_TARGET})")

    # LB surface points (9)
    points = {
        "goodpjw": 0.95346,
        "areeb": 0.95347,
        "S1": 0.95346,
        "S2": 0.95347,
        "S3": 0.95347,
        "S4": 0.95346,
        "S5": 0.95350,
        "S6": 0.95352,
        "S7": ATH,
    }

    # Load test vectors
    z_areeb = np.load(PRED_BO / "gm_breakout/z_areeb.npy").ravel()
    m1 = unit(np.load(PRED_BO / "gm_breakout/m1.npy").ravel())
    m2 = unit(np.load(PRED_BO / "gm_breakout/m2.npy").ravel())
    # S5→S6→S7 continuation direction
    u_ray = unit(0.70 * m1 + 0.30 * m2)

    # Build scored z for surface fit from submission CSVs where available
    sub_map = {
        "S5": find_first(PRED_BO.parent / "submissions/sub_master_s5_decontam_s2_winner_0p22e4.csv", VOL_SUB / "sub_master_s5_decontam_s2_winner_0p22e4.csv", SUB / "sub_master_s5_decontam_s2_winner_0p22e4.csv"),
        "S6": find_first(PRED_BO.parent / "submissions/sub_master_s6_branchA_breakout_0p52e4.csv", VOL_SUB / "sub_master_s6_branchA_breakout_0p52e4.csv"),
        "S7": find_first(PRED_BO.parent / "submissions/sub_master_slot7_verified_s7A.csv", VOL_SUB / "sub_master_slot7_verified_s7A.csv"),
    }
    z_pts = {"areeb": unitize(z_areeb)}
    for k, p in sub_map.items():
        z_pts[k] = unitize(zrk(pd.read_csv(p)["is_fraud"].to_numpy(float)))

    # Features for LB surface: [1, dist_areeb, dist_areeb^2, cos_ray]
    def feats(z):
        d = spearman_dist(z_areeb, z)
        return np.array([1.0, d, d * d, cos(z - z_areeb, u_ray)], dtype=np.float64)

    # Fit on goodpjw..S6 using approximate distances from known 1-rho
    # Use actual submission vectors for S5/S6 + synthetic for earlier points along known coords
    # Reconstruct early points via HISTORY got distances if CSVs exist
    hist_files = {
        "S1": "sub_breakout_s1_areeb_gpt2m_3ladder_0p18e4.csv",
        "S2": "sub_breakout_s2_branchC_damped_gpt2m_stageB_0p20e4.csv",
        "S3": "sub_breakout_s3_empgrad_0p30e4.csv",
        "S4": "sub_breakout_s4_areeb_micro_gpt2m_0p08e4.csv",
    }
    for k, fn in hist_files.items():
        for base in (PRED_BO.parent / "submissions", VOL_SUB, SUB):
            p = base / fn
            if p.exists():
                z_pts[k] = unitize(zrk(pd.read_csv(p)["is_fraud"].to_numpy(float)))
                break
    # goodpjw ≈ areeb without MLP/GB subtract — approximate with areeb + tiny noise axis if missing
    if "goodpjw" not in z_pts:
        # reconstruct from formula ingredients
        try:
            st10 = pd.read_csv(find_ds("submission_095340.csv"))["is_fraud"].to_numpy(float)
            st9 = pd.read_csv(find_ds("submission_095330.csv"))["is_fraud"].to_numpy(float)
            A, F = L(st10), L(st9)
            LM = L(np.load(find_ds("lm_test.npy")).ravel())
            a7 = np.load(find_ds("lm7_test_f0.npy"))
            L7 = L((a7[..., 1] - a7[..., 0]).mean(1))
            LM_nov = L(np.load(find_ds("megalm_test_v_nov08.npy")).ravel())
            score_g = A + 0.6 * (A - F) + 0.12 * (LM_nov - F) + 0.06 * (L7 - F)
            z_pts["goodpjw"] = unitize(zrk(expit(score_g)))  # rank via score directly
            z_pts["goodpjw"] = unitize(score_g)
        except Exception as e:
            log(f"  goodpjw reconstruct failed: {e}; use areeb proxy")
            z_pts["goodpjw"] = z_pts["areeb"].copy()

    names8 = ["goodpjw", "areeb", "S1", "S2", "S3", "S4", "S5", "S6"]
    X8, y8 = [], []
    for n in names8:
        if n not in z_pts:
            log(f"  WARN missing {n} for surface")
            continue
        X8.append(feats(z_pts[n]))
        y8.append(points[n])
    X8 = np.asarray(X8)
    y8 = np.asarray(y8)
    # ridge-ish least squares
    beta, *_ = np.linalg.lstsq(X8, y8, rcond=None)
    pred_s7 = float(feats(z_pts["S7"]) @ beta)
    err_s7 = pred_s7 - points["S7"]
    log(f"  OOS S7 surface: pred={pred_s7:.6f} actual={points['S7']:.5f} err={err_s7*1e5:+.2f}e-5")

    # Refit 9-point for later predictions
    names9 = names8 + ["S7"]
    X9 = np.asarray([feats(z_pts[n]) for n in names9 if n in z_pts])
    y9 = np.asarray([points[n] for n in names9 if n in z_pts])
    beta9, *_ = np.linalg.lstsq(X9, y9, rcond=None)

    # Quadratic surface on t = sqrt(1-rho)*100 along ray from areeb
    # OOF proxy: apply same ray on OOF using stageB/m1 OOF proxies
    d1_o = np.load(find_first(PRED_V5 / "gm_v4_polish/d1_v4_oof.npy", PRED_BO / "gm_v4_polish/d1_v4_oof.npy")).ravel()
    # Approximate OOF ray as unitize(d1) direction (stageB dominant in S5→S7)
    u_oof = unit(unitize(d1_o))
    m0 = fold_id == 0

    def oof_at_w(w):
        z = z_ref + w * u_oof
        auc5 = float(roc_auc_score(y, z))
        d5 = [(a - b) * 1e5 for a, b in zip(fold_aucs(y, z, folds), base_folds)]
        # Fold0
        z0 = z_ref.copy()
        z0[m0] = z_ref[m0] + w * unit(u_oof[m0])
        d0 = (float(roc_auc_score(y[m0], z0[m0])) - float(roc_auc_score(y[m0], z_ref[m0]))) * 1e5
        return auc5, d5, float(np.mean(d5)), d0

    # Known anchors: map t→1-rho(areeb); t=sqrt(1-rho)*100
    # S5 t=0.469 → 1-rho≈2.2e-5; S6 t=0.721 → 5.2e-5; S7 t=1.005 → ~1.01e-4
    known = [
        (0.469, 0.95350, 0.74, 4.43),
        (0.721, 0.95352, 1.81, 6.97),
        (1.005, 0.95354, 2.38, 9.50),
    ]
    # Calibrate OOF weight scale so w at S7 matches ~+2.38e-5 5F
    # Find w_s7 such that mean Δ≈2.38
    def mean_d5(w):
        return oof_at_w(w)[2]

    # coarse search
    w_grid = np.linspace(0.0, 0.20, 41)
    lifts = [mean_d5(w) for w in w_grid]
    # scale so max near known
    # Use w proportional to t: w = c * t
    # Fit c from S7: want mean_d5(c*1.005)≈2.38
    best_c, best_err = 0.05, 1e9
    for c in np.linspace(0.01, 0.25, 49):
        err = abs(mean_d5(c * 1.005) - 2.38)
        if err < best_err:
            best_err, best_c = err, c
    log(f"  OOF scale c={best_c:.4f} err_vs_S7_anchor={best_err:.3f}")

    sweep_t = [1.00, 1.15, 1.30, 1.45, 1.60, 1.80, 2.00, 2.25]
    rows = []
    for t in sweep_t:
        rho1 = (t / 100.0) ** 2
        w = best_c * t
        auc5, d5, mean5, d0 = oof_at_w(w)
        # LB surface pred: move from areeb by spearman dist = rho1 along u_ray
        ww, got = weight_for_spearman_dist(z_areeb, u_ray, rho1, hi=200.0)
        z_t = unitize(z_areeb + ww * u_ray)
        pred_lin = float(feats(z_t) @ beta9)
        # quadratic in t fit on known 3
        # quick local: polyfit LB vs t on known + this pred
        pred_quad = pred_lin
        rows.append(
            {
                "t": t,
                "one_minus_rho": rho1,
                "w_oof": w,
                "auc5": auc5,
                "d5_mean": mean5,
                "d5": d5,
                "d0": d0,
                "pred_lb": pred_lin,
                "got_dist": got,
            }
        )
        log(
            f"  t={t:.2f} 1-ρ={rho1:.2e} 5FΔ={mean5:+.2f}e-5 F0Δ={d0:+.2f}e-5 "
            f"predLB={pred_lin:.6f} pos={sum(x>0 for x in d5)}/5"
        )

    # vertex = max of score = d5_mean + 0.5*d0 among rows with d5_mean>0
    scored = [(r["d5_mean"] + 0.5 * r["d0"], r) for r in rows if r["d5_mean"] > 0]
    if not scored:
        scored = [(r["d5_mean"] + 0.5 * r["d0"], r) for r in rows]
    t_star_row = max(scored, key=lambda x: x[0])[1]
    t_star = t_star_row["t"]
    log(f"  OOF vertex t*={t_star:.2f} 1-ρ(areeb)={t_star_row['one_minus_rho']:.2e} "
        f"5F={t_star_row['d5_mean']:+.2f} F0={t_star_row['d0']:+.2f} predLB={t_star_row['pred_lb']:.6f}")

    out = {
        "ATH": ATH,
        "points": points,
        "oos_s7_pred": pred_s7,
        "oos_s7_err_e5": err_s7 * 1e5,
        "beta8": beta.tolist(),
        "beta9": beta9.tolist(),
        "oof_c": best_c,
        "sweep": rows,
        "t_star": t_star,
        "t_star_row": t_star_row,
        "known_anchors": known,
        "NO_SUBMIT": True,
    }
    (ART / "phase0.json").write_text(json.dumps(out, indent=2, default=float))
    sync_artifact(ART / "phase0.json")
    # Update breakout HISTORY ATH
    try:
        hist_p = PRED_BO / "gm_breakout/HISTORY.json"
        hist = json.loads(hist_p.read_text())
        hist["best"] = ATH
        hist["S7"] = {"LB": ATH, "file": "sub_master_slot7_verified_s7A.csv", "locked": True}
        hist_p.write_text(json.dumps(hist, indent=2))
        mh = PRED_BO / "gm_breakout/MASTER_HISTORY.json"
        master = json.loads(mh.read_text())
        master["best"] = ATH
        master["s7_held"] = False
        master["S7"] = {"LB": ATH, "file": "sub_master_slot7_verified_s7A.csv"}
        for h in master.get("history", []):
            if h.get("slot") == 7:
                h["LB"] = ATH
                h["held"] = False
                h["file"] = "sub_master_slot7_verified_s7A.csv"
        mh.write_text(json.dumps(master, indent=2))
    except Exception as e:
        log(f"  HISTORY update warn: {e}")
    save_state(phase="phase0_done", phase0={"t_star": t_star, "oos_s7_err_e5": err_s7 * 1e5})
    return out


# ============================================================================
# PHASE 1 — Feature condensation
# ============================================================================
def build_raw_keys(ntr):
    """Build categorical / mantissa / joint keys without full gboost import."""
    cache = ART / "raw_keys.npz"
    if cache.exists() and os.environ.get("REBUILD_KEYS", "0") != "1":
        z = np.load(cache, allow_pickle=True)
        log(f"  loaded raw_keys cache cards={len(z['names'])}")
        return z["K"].astype(np.int32), list(z["names"]), z["meta"].item()

    log("  building raw keys from train/test …")
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].to_numpy(np.int32)
    assert len(tr) == ntr
    full = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)
    cats = [c for c in full.columns if full[c].dtype == object or str(full[c].dtype).startswith("int")]
    # known schema
    CAT_CANDS = [
        "merchant_category", "day_of_week", "time_of_day_hour", "customer_age",
        "card_age_months", "txn_count_last_24h", "is_foreign_transaction",
        "ip_country_mismatch", "is_new_merchant", "is_ai_generated_scam_attempt",
    ]
    FLOAT_CANDS = [
        "amount_usd", "merchant_risk_score", "velocity_score", "hours_since_last_txn",
    ]
    cats = [c for c in CAT_CANDS if c in full.columns]
    floats = [c for c in FLOAT_CANDS if c in full.columns]
    keys = {}
    for c in cats:
        keys[f"cat_{c}"] = pd.factorize(full[c].astype(str), sort=True)[0].astype(np.int32)
    for c in floats:
        v = full[c].to_numpy(float)
        keys[f"q20_{c}"] = pd.qcut(v, 20, labels=False, duplicates="drop").astype(np.int32)
        keys[f"q10_{c}"] = pd.qcut(v, 10, labels=False, duplicates="drop").astype(np.int32)
        cents = (np.round(np.abs(v) * 100) % 100).astype(np.int32)
        keys[f"cent_{c}"] = cents
        # last-2 significant digits via string
        s = full[c].astype(str).str.replace(".", "", regex=False).str.replace("-", "", regex=False)
        keys[f"last2_{c}"] = s.str[-2:].fillna("00").map(lambda x: abs(hash(x)) % 100).astype(np.int32)
        mag = np.clip(np.floor(np.log10(np.abs(v) + 1e-12)), -3, 6).astype(np.int32) + 3
        keys[f"mag_{c}"] = mag

    # 2-way joints (categorical × q10) — top by G later; build pool of ~60
    joint_specs = []
    for c in cats[:8]:
        for f in floats:
            joint_specs.append((f"j_{c}_{f}", c, f"q10_{f}" if f"q10_{f}" in keys else None, True))
    # build selected joints
    for name, c, qname, _ in joint_specs:
        if qname is None:
            continue
        a = keys[f"cat_{c}"]
        b = keys[qname]
        keys[name] = (a.astype(np.int64) * (int(b.max()) + 1) + b).astype(np.int32)

    names = list(keys)
    K = np.column_stack([keys[n] for n in names]).astype(np.int32)
    meta = {"cats": cats, "floats": floats, "n_keys": len(names), "ntr": ntr, "nte": len(te)}
    np.savez_compressed(cache, K=K, names=np.array(names, dtype=object), meta=np.array(meta, dtype=object))
    log(f"  raw_keys shape={K.shape}")
    return K, names, meta


def multi_prior_te_block(K_cols, y, folds, seeds, ms, ntr, nte):
    """Return OOF/test TE tensor (n, n_cols * n_m * ...) averaged over seeds via nested TE."""
    n_cols = K_cols.shape[1]
    n_m = len(ms)
    oof = np.zeros((ntr, n_cols, n_m), dtype=np.float32)
    te_acc = np.zeros((nte, n_cols, n_m), dtype=np.float64)
    for fi, (a, b) in enumerate(folds):
        prior = float(y[a].mean())
        te_b = np.zeros((len(b), n_cols, n_m), dtype=np.float64)
        te_t = np.zeros((nte, n_cols, n_m), dtype=np.float64)
        for seed in seeds:
            inner = StratifiedKFold(5, shuffle=True, random_state=seed)
            for ia, _ib in inner.split(np.zeros(len(a)), y[a]):
                a_tr = a[ia]
                pr = float(y[a_tr].mean())
                for j in range(n_cols):
                    for mi, m in enumerate(ms):
                        outs = eb_te_maps(K_cols[a_tr, j], y[a_tr], [K_cols[b, j], K_cols[ntr:, j]], pr, m)
                        te_b[:, j, mi] += outs[0]
                        te_t[:, j, mi] += outs[1]
        denom = float(len(seeds) * 5)
        oof[b] = (te_b / denom).astype(np.float32)
        te_acc += te_t / denom
        log(f"    TE fold {fi} done")
        gc.collect()
    return oof, (te_acc / 5.0).astype(np.float32)


def condense_block(X_o, X_t, y, folds, z_anchor_o, z_anchor_t, tag, ridge_alphas, use_lgb=True, use_cat=False):
    """Condense multi-col TE block into orthogonal preds anchored on Platt(z_ref)."""
    import lightgbm as lgb

    ntr = len(y)
    cols = []
    te_cols = []
    bm_o_cache = {}
    for alpha in ridge_alphas:
        oof = np.zeros(ntr)
        te_folds = []
        for fi, (a, b) in enumerate(folds):
            tr_in, es_in = train_test_split(a, test_size=0.10, random_state=42 + fi, stratify=y[a])
            bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(z_anchor_o[tr_in], y[tr_in], z_anchor_o, z_anchor_t)
            # residual target in logit space relative to anchor
            # Fit ridge on features predicting y with sample; use decision as residual add-on
            model = Ridge(alpha=alpha, fit_intercept=True)
            # Use BM as offset: predict residual probability via linear on features
            model.fit(X_o[tr_in], y[tr_in] - expit(bm_tr))
            oof[b] = expit(bm_all[b]) + model.predict(X_o[b])
            te_folds.append(expit(bm_te) + model.predict(X_t))
        o = np.clip(oof, 1e-6, 1 - 1e-6)
        t = np.clip(np.mean(te_folds, 0), 1e-6, 1 - 1e-6)
        zo, zt = align_method_b_smooth(zrk(o), zrk(t), folds)
        if disp_ks(zo, zt)[1] >= 0.0001:
            zo, zt = force_exact_law(zo, zt)
        cols.append(zo.astype(np.float32))
        te_cols.append(zt.astype(np.float32))
        log(f"  {tag} Ridgeα={alpha} AUC={roc_auc_score(y,zo):.7f} KS={disp_ks(zo,zt)[1]:.6f}")

    if use_lgb:
        oof = np.zeros(ntr)
        te_folds = []
        for fi, (a, b) in enumerate(folds):
            tr_in, es_in = train_test_split(a, test_size=0.10, random_state=77 + fi, stratify=y[a])
            bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(z_anchor_o[tr_in], y[tr_in], z_anchor_o, z_anchor_t)
            dtr = lgb.Dataset(X_o[tr_in], label=y[tr_in], init_score=bm_tr)
            des = lgb.Dataset(X_o[es_in], label=y[es_in], init_score=bm_all[es_in], reference=dtr)
            params = dict(
                objective="binary", metric="auc", learning_rate=0.03, num_leaves=6,
                feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
                min_child_samples=300, verbosity=-1, seed=77, num_threads=N_JOBS,
                lambda_l2=20.0,
            )
            booster = lgb.train(
                params, dtr, num_boost_round=400, valid_sets=[des],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
            )
            oof[b] = expit(booster.predict(X_o[b], raw_score=True) + bm_all[b])
            te_folds.append(expit(booster.predict(X_t, raw_score=True) + bm_te))
            del booster
            gc.collect()
        zo, zt = align_method_b_smooth(zrk(oof), zrk(np.mean(te_folds, 0)), folds)
        if disp_ks(zo, zt)[1] >= 0.0001:
            zo, zt = force_exact_law(zo, zt)
        cols.append(zo.astype(np.float32))
        te_cols.append(zt.astype(np.float32))
        log(f"  {tag} LGBM AUC={roc_auc_score(y,zo):.7f} KS={disp_ks(zo,zt)[1]:.6f}")

    if use_cat:
        from catboost import CatBoostClassifier, Pool

        oof = np.zeros(ntr)
        te_folds = []
        for fi, (a, b) in enumerate(folds):
            tr_in, es_in = train_test_split(a, test_size=0.10, random_state=91 + fi, stratify=y[a])
            bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(z_anchor_o[tr_in], y[tr_in], z_anchor_o, z_anchor_t)
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="AUC", depth=4, l2_leaf_reg=40,
                rsm=0.35, learning_rate=0.03, iterations=800, random_seed=91,
                thread_count=N_JOBS, od_type="Iter", od_wait=50, verbose=False,
                allow_writing_files=False,
            )
            model.fit(Pool(X_o[tr_in], y[tr_in], baseline=bm_tr), eval_set=Pool(X_o[es_in], y[es_in], baseline=bm_all[es_in]), use_best_model=True)
            oof[b] = model.predict(Pool(X_o[b], baseline=bm_all[b]), prediction_type="Probability")[:, 1]
            te_folds.append(model.predict(Pool(X_t, baseline=bm_te), prediction_type="Probability")[:, 1])
            del model
            gc.collect()
        zo, zt = align_method_b_smooth(zrk(oof), zrk(np.mean(te_folds, 0)), folds)
        if disp_ks(zo, zt)[1] >= 0.0001:
            zo, zt = force_exact_law(zo, zt)
        cols.append(zo.astype(np.float32))
        te_cols.append(zt.astype(np.float32))
        log(f"  {tag} Cat AUC={roc_auc_score(y,zo):.7f} KS={disp_ks(zo,zt)[1]:.6f}")

    return np.column_stack(cols), np.column_stack(te_cols)


def phase1(y, folds):
    log("======== PHASE 1: Multi-prior EB feature condensation ========")
    save_state(phase="phase1_start")
    if (ART / "condensed_blocks_oof.npy").exists() and os.environ.get("REDO_PHASE1", "0") != "1":
        log("  SKIP phase1 — condensed_blocks exist")
        return np.load(ART / "condensed_blocks_oof.npy"), np.load(ART / "condensed_blocks_test.npy")

    ntr = len(y)
    z_ref_o, z_ref_t = load_z_ref(folds)
    K, names, meta = build_raw_keys(ntr)
    nte = K.shape[0] - ntr

    # Try gboost KMAT rest if available (cached)
    kmat_path = ART / "kmat_full.npy"
    if kmat_path.exists():
        KMAT = np.load(kmat_path)
        log(f"  loaded KMAT cache {KMAT.shape}")
    else:
        KMAT = None
        try:
            log("  attempting gboost KMAT import (may take a few min) …")
            os.environ["INPUT_DIR"] = str(WORK)
            import gboost as gb  # noqa

            KMAT = gb.KMAT.astype(np.int32)
            np.save(kmat_path, KMAT)
            log(f"  gboost KMAT={KMAT.shape} BASE={gb.BASE.shape}")
        except Exception as e:
            log(f"  gboost unavailable ({e}); using reconstructed keys only")

    # Block 1: KMAT[:,68:] or reconstructed joints + multi-prior TE
    if KMAT is not None and KMAT.shape[1] > 68:
        K1 = KMAT[:, 68:]
        log(f"  Block1 using KMAT[:,68:] -> {K1.shape[1]} cols")
    else:
        joint_idx = [i for i, n in enumerate(names) if n.startswith("j_")]
        # add top cat/q keys
        extra = [i for i, n in enumerate(names) if n.startswith("cat_") or n.startswith("q20_")]
        sel = (joint_idx + extra)[:40]
        K1 = K[:, sel]
        log(f"  Block1 reconstructed {K1.shape[1]} keys")

    # Limit columns for runtime: take top 24 by train cardinality
    cards = [len(np.unique(K1[:ntr, j])) for j in range(K1.shape[1])]
    top = np.argsort(cards)[::-1][:24]
    K1 = K1[:, top]
    log(f"  Block1 top24 cards={[cards[j] for j in top[:5]]}...")

    seeds8 = [11, 29, 47, 67, 89, 101, 151, 199]
    ms = [5.0, 25.0, 100.0]
    log("  Block1 multi-prior TE …")
    te1_o, te1_t = multi_prior_te_block(K1, y, folds, seeds8, ms, ntr, nte)
    X1_o = te1_o.reshape(ntr, -1)
    X1_t = te1_t.reshape(nte, -1)
    b1_o, b1_t = condense_block(X1_o, X1_t, y, folds, z_ref_o, z_ref_t, "block1", [2000.0, 10000.0], use_lgb=True)
    del te1_o, te1_t, X1_o, X1_t
    gc.collect()

    # Block 2: mantissa 2-way
    cent_idx = [i for i, n in enumerate(names) if n.startswith("cent_") or n.startswith("last2_") or n.startswith("mag_")]
    cat_idx = [i for i, n in enumerate(names) if n.startswith("cat_")]
    # build top 80 interactions with support gates
    log("  Block2 building mantissa interactions …")
    inter_cols = []
    # pairwise cents
    cents = [i for i, n in enumerate(names) if n.startswith("cent_")]
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            a = K[:, cents[i]].astype(np.int64)
            b = K[:, cents[j]].astype(np.int64)
            inter_cols.append((a * 101 + b).astype(np.int32))
    for ci in cents:
        for cj in cat_idx[:6]:
            a = K[:, ci].astype(np.int64)
            b = K[:, cj].astype(np.int64)
            inter_cols.append((a * (int(b.max()) + 1) + b).astype(np.int32))
    # support filter
    kept = []
    for col in inter_cols:
        tr_c = np.bincount(col[:ntr], minlength=int(col.max()) + 1)
        te_c = np.bincount(col[ntr:], minlength=int(col.max()) + 1)
        # count keys with train>=5 and test>=2
        ok = (tr_c >= 5) & (te_c[: len(tr_c)] >= 2) if len(te_c) >= len(tr_c) else (tr_c >= 5)
        support = int(ok.sum())
        if support >= 20:
            kept.append((support, col))
    kept.sort(key=lambda x: -x[0])
    kept = kept[:80]
    if not kept:
        # fallback: use raw cent cols
        K2 = K[:, cents[:8]] if cents else K[:, :8]
    else:
        K2 = np.column_stack([c for _, c in kept]).astype(np.int32)
    log(f"  Block2 cols={K2.shape[1]}")
    seeds5 = [13, 37, 71, 103, 139]
    te2_o, te2_t = multi_prior_te_block(K2, y, folds, seeds5, [25.0], ntr, nte)
    # also add log-likelihood style: use m=25 TE only
    X2_o = te2_o.reshape(ntr, -1)
    X2_t = te2_t.reshape(nte, -1)
    b2_o, b2_t = condense_block(X2_o, X2_t, y, folds, z_ref_o, z_ref_t, "block2", [3000.0], use_lgb=True)
    del te2_o, te2_t, X2_o, X2_t
    gc.collect()

    # Block 3: 3-way support-gated
    log("  Block3 3-way tuples …")
    three = []
    base_cats = cat_idx[:5]
    q20 = [i for i, n in enumerate(names) if n.startswith("q20_")]
    for i in range(len(base_cats)):
        for j in range(i + 1, len(base_cats)):
            for q in q20[:3]:
                a, b, c = K[:, base_cats[i]].astype(np.int64), K[:, base_cats[j]].astype(np.int64), K[:, q].astype(np.int64)
                code = ((a * (int(b.max()) + 1) + b) * (int(c.max()) + 1) + c).astype(np.int32)
                tr_c = np.bincount(code[:ntr])
                te_c = np.bincount(code[ntr:], minlength=len(tr_c))
                te_c = te_c[: len(tr_c)]
                ok = (tr_c >= 10) & (te_c >= 4)
                if int(ok.sum()) >= 30:
                    # support ratio feature later; keep code
                    three.append((int(ok.sum()), code))
    three.sort(key=lambda x: -x[0])
    three = three[:100]
    if not three:
        K3 = K[:, (base_cats + q20)[:12]]
        support_ratio_o = np.zeros((ntr, 1), dtype=np.float32)
        support_ratio_t = np.zeros((nte, 1), dtype=np.float32)
    else:
        K3 = np.column_stack([c for _, c in three]).astype(np.int32)
        # support ratio for first 20
        sr_o = []
        sr_t = []
        for _, code in three[:20]:
            tr_c = np.bincount(code[:ntr])
            te_c = np.bincount(code[ntr:], minlength=len(tr_c))[: len(tr_c)]
            ratio = np.log(((te_c + 0.5) / nte) / ((tr_c + 0.5) / ntr))
            sr_o.append(ratio[code[:ntr]])
            sr_t.append(ratio[np.clip(code[ntr:], 0, len(ratio) - 1)])
        support_ratio_o = np.column_stack(sr_o).astype(np.float32)
        support_ratio_t = np.column_stack(sr_t).astype(np.float32)
    log(f"  Block3 cols={K3.shape[1]}")
    te3_o, te3_t = multi_prior_te_block(K3[:, :40] if K3.shape[1] > 40 else K3, y, folds, seeds5, [25.0], ntr, nte)
    X3_o = np.hstack([te3_o.reshape(ntr, -1), support_ratio_o])
    X3_t = np.hstack([te3_t.reshape(nte, -1), support_ratio_t])
    b3_o, b3_t = condense_block(X3_o, X3_t, y, folds, z_ref_o, z_ref_t, "block3", [3000.0], use_lgb=False, use_cat=True)
    del te3_o, te3_t, X3_o, X3_t
    gc.collect()

    blocks_o = np.hstack([b1_o, b2_o, b3_o]).astype(np.float32)
    blocks_t = np.hstack([b1_t, b2_t, b3_t]).astype(np.float32)
    # lift vs z_ref
    lifts = []
    for j in range(blocks_o.shape[1]):
        # small grid
        best = 0.0
        for w in np.linspace(-0.05, 0.15, 21):
            auc = float(roc_auc_score(y, z_ref_o + w * unitize(blocks_o[:, j])))
            best = max(best, (auc - float(roc_auc_score(y, z_ref_o))) * 1e5)
        lifts.append(best)
        d, ks = disp_ks(blocks_o[:, j], blocks_t[:, j])
        log(f"  condensed col{j}: bestΔ={best:+.2f}e-5 disp={d:.4f} KS={ks:.6f}")

    np.save(ART / "condensed_blocks_oof.npy", blocks_o)
    np.save(ART / "condensed_blocks_test.npy", blocks_t)
    meta_out = {"shape_o": list(blocks_o.shape), "shape_t": list(blocks_t.shape), "lifts_e5": lifts}
    (ART / "phase1.json").write_text(json.dumps(meta_out, indent=2))
    sync_artifact(ART / "condensed_blocks_oof.npy")
    sync_artifact(ART / "condensed_blocks_test.npy")
    sync_artifact(ART / "phase1.json")
    save_state(phase="phase1_done", phase1=meta_out)
    return blocks_o, blocks_t


# ============================================================================
# PHASE 2 — L7 denoise
# ============================================================================
def phase2(y, folds, fold_id, blocks_o, blocks_t):
    log("======== PHASE 2: L7 denoise + 5f student ========")
    save_state(phase="phase2_start")
    if exists_pair("l7_denoised") and os.environ.get("REDO_PHASE2", "0") != "1":
        log("  SKIP phase2")
        return load_pair("l7_denoised")

    ntr = len(y)
    nte = blocks_t.shape[0]
    m0 = fold_id == 0
    lm7_o = np.load(find_ds("lm7_oof_f0.npy"))  # (139992,10,2)
    lm7_t = np.load(find_ds("lm7_test_f0.npy"))  # (299984,4,2)
    log(f"  lm7_oof_f0={lm7_o.shape} lm7_test_f0={lm7_t.shape}")

    def llr(arr):
        # arr (..., 2) with logp or p? goodpjw stores logits/probs — use log(p1)-log(p0) safely
        p0 = np.clip(arr[..., 0], 1e-12, 1.0)
        p1 = np.clip(arr[..., 1], 1e-12, 1.0)
        # If values look like log-probs (negative), treat as log
        if np.nanmedian(arr) < 0:
            return arr[..., 1] - arr[..., 0]
        return np.log(p1) - np.log(p0)

    llr_o = llr(lm7_o)  # (139992,10)
    llr_t = llr(lm7_t)  # (299984,4)

    # bayes via logaddexp across perm on probability of class1 in log-space
    def bayes_llr(llr_perm):
        # convert llr to logp1/logp0
        # p1 = sigmoid(llr)
        logp1 = -np.logaddexp(0, -llr_perm)
        logp0 = -np.logaddexp(0, llr_perm)
        # bayes average in prob space via logsumexp
        lp1 = logsumexp(logp1, axis=1) - np.log(llr_perm.shape[1])
        lp0 = logsumexp(logp0, axis=1) - np.log(llr_perm.shape[1])
        return lp1 - lp0

    l7_bayes_o = bayes_llr(llr_o)
    l7_bayes_t = bayes_llr(llr_t)
    l7_mean_o = llr_o.mean(1)
    l7_mean_t = llr_t.mean(1)
    l7_min_o, l7_max_o = llr_o.min(1), llr_o.max(1)
    l7_min_t, l7_max_t = llr_t.min(1), llr_t.max(1)
    l7_std_o = llr_o.std(1)
    l7_std_t = llr_t.std(1)

    # Align std from K=4 test to K=10 fold0 oof via method B on fold0 rows only
    # Build full-length OOF placeholders
    def place_f0(v):
        out = np.zeros(ntr, dtype=np.float64)
        assert v.shape[0] == m0.sum(), (v.shape, m0.sum())
        out[m0] = v
        return out

    stats = {}
    for name, vo, vt in [
        ("l7_bayes", l7_bayes_o, l7_bayes_t),
        ("l7_mean_llr", l7_mean_o, l7_mean_t),
        ("l7_min_llr", l7_min_o, l7_min_t),
        ("l7_max_llr", l7_max_o, l7_max_t),
        ("l7_std_llr", l7_std_o, l7_std_t),
    ]:
        o_full = place_f0(vo)
        # align using fold0 only for law matching
        o_f0 = unitize(vo)
        t_al = force_exact_law(o_f0, vt)[1]
        # For std specifically also try align_method_b on synthetic folds
        stats[name] = (o_full, t_al)
        save_pair(name, o_full, t_al)

    # Student features
    mega_o = zrk(np.load(find_first(
        Path("/tmp/gm_v5_work/external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"),
        DS / "megayak__s6e11-target-first-lm" / "lm_oof.npy",
        VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy",
    )).ravel())
    mega_t = zrk(np.load(find_first(
        Path("/tmp/gm_v5_work/external/banks/megayak_s6e11-target-first-lm/lm_test.npy"),
        DS / "megayak__s6e11-target-first-lm" / "lm_test.npy",
        VOL / "external/banks/megayak_s6e11-target-first-lm/lm_test.npy",
    )).ravel())

    # r_L7_f0 on fold0
    L7_f0 = unitize(zrk(l7_mean_o))
    mega_f0 = unitize(mega_o[m0])
    r_L7_f0 = L7_f0 - mega_f0

    # Feature matrix for student
    feat_list_o = [mega_o]
    feat_list_t = [mega_t]
    # wide lm12
    try:
        w12_o = zrk(np.load(find_first(
            PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_oof.npy",
            Path("/tmp/gm_v5_work/external/nov08_part12/wide_lm_stack_nested.npy"),
        )).ravel()[:ntr] if False else np.load(PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_oof.npy").ravel())
        w12_t = np.load(PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_test.npy").ravel()
        feat_list_o.append(unitize(w12_o))
        feat_list_t.append(unitize(zrk(w12_t) if w12_t.max() <= 1.5 else w12_t))
    except Exception as e:
        log(f"  w12 skip: {e}")

    # gpt2 med 5f — prefer already-stitched bayes; raw oof_i are fold-local (139992,8,2)
    try:
        bayes_o = PRED_V5 / "gm_v2_ladder/z_m_bayes_5f_oof.npy"
        bayes_t = PRED_V5 / "gm_v2_ladder/z_m_bayes_5f_test.npy"
        if bayes_o.exists() and bayes_t.exists():
            o, t = load_ab(bayes_o, bayes_t, folds, already_z=True)
            feat_list_o.append(o)
            feat_list_t.append(t)
            log("  gpt2m: using z_m_bayes_5f")
        else:
            # stitch fold-local logit means into full OOF
            g_o = np.zeros(ntr, dtype=np.float64)
            g_te = []
            for i, (a, b) in enumerate(folds):
                arr = np.load(PRED_V5 / "gpt2_med_4ep" / f"oof_{i}.npy")
                llr = arr[..., 1] - arr[..., 0]
                g_o[b] = llr.mean(1)
                te = np.load(PRED_V5 / "gpt2_med_4ep" / f"test_{i}.npy")
                g_te.append((te[..., 1] - te[..., 0]).mean(1))
            g_t = np.mean(g_te, 0)
            o, t = align_method_b_smooth(zrk(g_o), zrk(g_t), folds)
            feat_list_o.append(o.astype(np.float32))
            feat_list_t.append(t.astype(np.float32))
            log("  gpt2m: stitched fold-local oof_0..4")
    except Exception as e:
        log(f"  gpt2m skip: {e}")

    for tag, po, pt in [
        ("masa", Path("/tmp/gm_v5_work/external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy") if False else find_first(
            Path("/tmp/gm_v5_work").joinpath("external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy"),
            VOL / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy",
        ), find_first(
            Path("/tmp/gm_v5_work").joinpath("external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy"),
            VOL / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy",
        )),
    ]:
        try:
            o, t = load_ab(po, pt, folds)
            feat_list_o.append(o)
            feat_list_t.append(t)
        except Exception as e:
            log(f"  {tag} skip: {e}")

    for rel_o, rel_t in [
        ("gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"),
        ("gpt2_3ep/gpt2_bidir_oof.npy", "gpt2_3ep/gpt2_bidir_test.npy"),
        ("fast_gpu_harvest/oof_tabm_5f.npy", "fast_gpu_harvest/test_tabm_5f.npy"),
        ("matrix_v2/P4_L4.2_pure5f_oof.npy", "matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("matrix_v2/P4_L4.3_pure5f_oof.npy", "matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("matrix_v2/P4_L4.5_pure5f_oof.npy", "matrix_v2/P4_L4.5_pure5f_test.npy"),
    ]:
        try:
            o, t = load_ab(PRED_V5 / rel_o, PRED_V5 / rel_t, folds)
            feat_list_o.append(o)
            feat_list_t.append(t)
        except Exception as e:
            log(f"  skip {rel_o}: {e}")

    # blocks
    for j in range(blocks_o.shape[1]):
        feat_list_o.append(unitize(blocks_o[:, j]))
        feat_list_t.append(unitize(blocks_t[:, j]))

    # X_v5 slice as KMAT proxy
    try:
        Xv = np.load(PRED_V5 / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
        Xt = np.load(PRED_V5 / "gm_v5_overnight/X_v5_test.npy", mmap_mode="r")
        # first 16 TE cols
        for j in range(16):
            feat_list_o.append(unitize(np.asarray(Xv[:, j], dtype=np.float64)))
            feat_list_t.append(unitize(np.asarray(Xt[:, j], dtype=np.float64)))
    except Exception as e:
        log(f"  X_v5 skip: {e}")

    # Drop any mis-shaped features before stack
    keep_o, keep_t = [], []
    for i, (fo, ft) in enumerate(zip(feat_list_o, feat_list_t)):
        fo = np.asarray(fo, dtype=np.float64).ravel()
        ft = np.asarray(ft, dtype=np.float64).ravel()
        if fo.shape[0] != ntr or ft.shape[0] != nte:
            log(f"  drop feat {i}: o={fo.shape} t={ft.shape}")
            continue
        keep_o.append(fo)
        keep_t.append(ft)
    Fo = np.column_stack(keep_o).astype(np.float32)
    Ft = np.column_stack(keep_t).astype(np.float32)
    log(f"  student F shape o={Fo.shape} t={Ft.shape}")

    # Train student on fold0 rows only with inner CV; predict all folds + test
    import lightgbm as lgb
    from catboost import CatBoostRegressor, Pool

    idx0 = np.where(m0)[0]
    y_res = r_L7_f0.astype(np.float64)
    # 5-fold on fold0 to get OOF student residual on fold0, and models for other folds/test
    inner = StratifiedKFold(5, shuffle=True, random_state=42)
    # stratify via y[idx0]
    r_oof = np.zeros(ntr, dtype=np.float64)
    r_te_acc = []
    # Also need predictions for non-fold0 train rows: train on all fold0, predict rest
    # Collect bag of models from inner CV
    models_ridge, models_lgb, models_cat = [], [], []
    oof_f0 = np.zeros(len(idx0))
    for ia, ib in inner.split(np.zeros(len(idx0)), y[idx0]):
        tr, va = idx0[ia], idx0[ib]
        # Ridge
        rg = Ridge(alpha=1000.0)
        rg.fit(Fo[tr], y_res[ia])  # careful: y_res indexed by fold0 positions
        # Fix: y_res is parallel to idx0 order (length 139992)
        # Map: position in idx0
        # Actually y_res[k] corresponds to idx0[k]? place_f0 used m0 boolean order which is sorted indices order
        # idx0 = np.where(m0)[0] is sorted; l7 arrays are in fold0 row order matching folds.csv fold==0 order
        # Verify length
        pass

    # Re-index properly: arrays from lm7 are in the order of rows where fold==0
    assert len(idx0) == len(y_res)
    oof_student_f0 = np.zeros(len(idx0))
    te_bag = []
    rest_bag = []
    for ia, ib in inner.split(np.zeros(len(idx0)), y[idx0]):
        tr_pos, va_pos = ia, ib
        tr_idx, va_idx = idx0[tr_pos], idx0[va_pos]
        ytr = y_res[tr_pos]
        # Ridge
        rg = Ridge(alpha=1000.0)
        rg.fit(Fo[tr_idx], ytr)
        pred_va = rg.predict(Fo[va_idx])
        pred_te = rg.predict(Ft)
        pred_all = rg.predict(Fo)
        # LGBM
        dtr = lgb.Dataset(Fo[tr_idx], label=ytr)
        params = dict(objective="regression", learning_rate=0.03, num_leaves=15, feature_fraction=0.5,
                      bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=42, num_threads=N_JOBS, metric="l2")
        # early stop with va
        dva = lgb.Dataset(Fo[va_idx], label=y_res[va_pos], reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        pred_va_l = booster.predict(Fo[va_idx])
        pred_te_l = booster.predict(Ft)
        pred_all_l = booster.predict(Fo)
        # Cat
        cat = CatBoostRegressor(
            loss_function="RMSE", depth=5, l2_leaf_reg=20, learning_rate=0.05,
            iterations=600, random_seed=42, thread_count=N_JOBS, od_type="Iter", od_wait=40,
            verbose=False, allow_writing_files=False,
        )
        cat.fit(Fo[tr_idx], ytr, eval_set=(Fo[va_idx], y_res[va_pos]), use_best_model=True)
        pred_va_c = cat.predict(Fo[va_idx])
        pred_te_c = cat.predict(Ft)
        pred_all_c = cat.predict(Fo)
        ens_va = (pred_va + pred_va_l + pred_va_c) / 3.0
        ens_te = (pred_te + pred_te_l + pred_te_c) / 3.0
        ens_all = (pred_all + pred_all_l + pred_all_c) / 3.0
        oof_student_f0[va_pos] = ens_va
        te_bag.append(ens_te)
        rest_bag.append(ens_all)
        del booster, cat
        gc.collect()
        log(f"  student inner fold done")

    r_student_o = np.zeros(ntr)
    r_student_o[idx0] = oof_student_f0
    # For other folds: average model predictions trained on fold0
    r_student_o[~m0] = np.mean(rest_bag, 0)[~m0]
    r_student_t = np.mean(te_bag, 0)

    # L7_5f_student = zrk(megayak) + r_student
    L7_student_o = unitize(mega_o + r_student_o)
    L7_student_t = unitize(mega_t + r_student_t)
    # Align
    L7_student_o, L7_student_t = align_method_b_smooth(L7_student_o, L7_student_t, folds)
    if disp_ks(L7_student_o, L7_student_t)[1] >= 0.0001:
        L7_student_o, L7_student_t = force_exact_law(L7_student_o, L7_student_t)

    # Noise vector on test: zrk(L7_test_f0) - zrk(L7_5f_student_test)
    L7_test_raw = unitize(zrk(l7_mean_t))
    e_noise_t = unitize(L7_test_raw - L7_student_t)
    # OOF noise only on fold0
    e_noise_o = np.zeros(ntr)
    e_noise_o[idx0] = unitize(L7_f0 - L7_student_o[idx0])

    # Save stacked denoised pack: [L7_student, e_noise, l7_bayes_aligned]
    pack_o = np.column_stack([L7_student_o, e_noise_o, stats["l7_bayes"][0]])
    pack_t = np.column_stack([L7_student_t, e_noise_t, stats["l7_bayes"][1]])
    np.save(ART / "l7_denoised_oof.npy", pack_o.astype(np.float32))
    np.save(ART / "l7_denoised_test.npy", pack_t.astype(np.float32))
    save_pair("L7_5f_student", L7_student_o, L7_student_t)
    save_pair("e_L7_noise", e_noise_o, e_noise_t)
    meta = {
        "corr_student_vs_r": float(np.corrcoef(oof_student_f0, y_res)[0, 1]),
        "noise_test_std": float(np.std(e_noise_t)),
        "student_auc_as_score": float(roc_auc_score(y, L7_student_o)),
    }
    (ART / "phase2.json").write_text(json.dumps(meta, indent=2))
    sync_artifact(ART / "l7_denoised_oof.npy")
    sync_artifact(ART / "l7_denoised_test.npy")
    sync_artifact(ART / "phase2.json")
    log(f"  phase2 meta={meta}")
    save_state(phase="phase2_done", phase2=meta)
    return pack_o, pack_t


# ============================================================================
# PHASE 3 — Stage-B v7 families
# ============================================================================
def build_X_stageB(y, folds, blocks_o, blocks_t, l7_pack_o, l7_pack_t):
    cache_o = ART / "X_stageB_v7_oof.npy"
    cache_t = ART / "X_stageB_v7_test.npy"
    if cache_o.exists() and cache_t.exists() and os.environ.get("REBUILD_X", "0") != "1":
        log(f"  loaded X_stageB_v7 {np.load(cache_o, mmap_mode='r').shape}")
        return np.load(cache_o, mmap_mode="r"), np.load(cache_t, mmap_mode="r")

    ntr = len(y)
    parts_o, parts_t = [], []
    # Start from X_v5 (already has KMAT68 + extras + BASE)
    Xv_o = np.load(PRED_V5 / "gm_v5_overnight/X_v5_oof.npy")
    Xv_t = np.load(PRED_V5 / "gm_v5_overnight/X_v5_test.npy")
    parts_o.append(Xv_o)
    parts_t.append(Xv_t)
    parts_o.append(blocks_o)
    parts_t.append(blocks_t)
    parts_o.append(l7_pack_o.astype(np.float32))
    parts_t.append(l7_pack_t.astype(np.float32))

    # P5_L5.1
    try:
        o, t = load_ab(PRED_VOL / "gm_recon_p5/P5_L5.1_oof.npy" if (PRED_VOL / "gm_recon_p5/P5_L5.1_oof.npy").exists() else find_first(
            VOL / "preds/gm_recon_p5/P5_L5.1_oof.npy",
            PRED_V5 / "gm_recon_p5/P5_L5.1_oof.npy",
        ), find_first(
            VOL / "preds/gm_recon_p5/P5_L5.1_test.npy",
            PRED_V5 / "gm_recon_p5/P5_L5.1_test.npy",
        ), folds)
        parts_o.append(o.reshape(-1, 1))
        parts_t.append(t.reshape(-1, 1))
    except Exception as e:
        log(f"  P5_L5.1 skip: {e}")

    for rel_o, rel_t, already in [
        ("gm_v4_polish/z_lm_raw_super_v4_oof.npy", "gm_v4_polish/z_lm_raw_super_v4_test.npy", True),
        ("gm_v2_ladder/z_lm12_raw_super_oof.npy", "gm_v2_ladder/z_lm12_raw_super_test.npy", True),
        ("gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy", False),
        ("gpt2_3ep/gpt2_bidir_oof.npy", "gpt2_3ep/gpt2_bidir_test.npy", False),
        ("matrix_v2/P4_L4.2_pure5f_oof.npy", "matrix_v2/P4_L4.2_pure5f_test.npy", False),
        ("matrix_v2/P4_L4.3_pure5f_oof.npy", "matrix_v2/P4_L4.3_pure5f_test.npy", False),
        ("matrix_v2/P4_L4.5_pure5f_oof.npy", "matrix_v2/P4_L4.5_pure5f_test.npy", False),
        ("fast_gpu_harvest/oof_tabm_5f.npy", "fast_gpu_harvest/test_tabm_5f.npy", False),
        ("fast_gpu_harvest/oof_tabm_5f_std.npy", "fast_gpu_harvest/test_tabm_5f_std.npy", False),
        ("fast_gpu_harvest/knn24_oof.npy", "fast_gpu_harvest/knn24_test.npy", False),
        ("fast_gpu_harvest/oof_orig20k_knn.npy", "fast_gpu_harvest/test_orig20k_knn.npy", False),
    ]:
        try:
            o, t = load_ab(PRED_V5 / rel_o, PRED_V5 / rel_t, folds, already_z=already)
            parts_o.append(o.reshape(-1, 1).astype(np.float32))
            parts_t.append(t.reshape(-1, 1).astype(np.float32))
        except Exception as e:
            log(f"  skip {rel_o}: {e}")

    X_o = np.hstack(parts_o).astype(np.float32)
    X_t = np.hstack(parts_t).astype(np.float32)
    # replace nan/inf
    X_o = np.nan_to_num(X_o, nan=0.0, posinf=0.0, neginf=0.0)
    X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(cache_o, X_o)
    np.save(cache_t, X_t)
    (ART / "X_stageB_v7_meta.json").write_text(json.dumps({"shape_o": list(X_o.shape), "shape_t": list(X_t.shape)}))
    log(f"  X_stageB_v7 o={X_o.shape} t={X_t.shape}")
    sync_artifact(cache_o)
    sync_artifact(cache_t)
    return X_o, X_t


def phase3(y, folds):
    log("======== PHASE 3: Stage-B v7 families ========")
    save_state(phase="phase3_start")
    blocks_o = np.load(ART / "condensed_blocks_oof.npy")
    blocks_t = np.load(ART / "condensed_blocks_test.npy")
    l7_o = np.load(ART / "l7_denoised_oof.npy")
    l7_t = np.load(ART / "l7_denoised_test.npy")
    X_o, X_t = build_X_stageB(y, folds, blocks_o, blocks_t, l7_o, l7_t)
    ntr = len(y)

    v5_o = zrk(np.load(PRED_V5 / "P_super_v5_harvest_oof.npy").ravel())
    v5_t = zrk(np.load(PRED_V5 / "P_super_v5_harvest_test.npy").ravel())
    z8_o = zrk(np.load(PRED_V5 / "P_super_v8_part12_rect_oof.npy").ravel())
    z8_t = zrk(np.load(PRED_V5 / "P_super_v8_part12_rect_test.npy").ravel())

    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool

    def train_xgb_dual(tag, depths_seeds, anchors, eta=0.02, colsample=0.28, reg_lambda=35.0, mcw=200, nround=550, pairwise=False):
        if exists_pair(tag):
            o, t = load_pair(tag)
            log(f"  SKIP {tag} AUC={roc_auc_score(y,zrk(o)):.7f}")
            return
        oof_bag = np.zeros(ntr)
        te_bag = []
        n_models = 0
        for (seed, depth), (ao, at, aname) in ((ds, anc) for anc in anchors for ds in depths_seeds):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(ao[tr_in], y[tr_in], ao, at)
                if pairwise:
                    # 50 stratified mini-query groups on tr_in
                    n_q = 50
                    # assign groups by stratified chunks
                    order = np.argsort(ao[tr_in])
                    groups = np.zeros(len(tr_in), dtype=np.int32)
                    # split positives and negatives into buckets
                    pos = np.where(y[tr_in] == 1)[0]
                    neg = np.where(y[tr_in] == 0)[0]
                    for qi in range(n_q):
                        groups[pos[qi::n_q]] = qi
                        groups[neg[qi::n_q]] = qi
                    dtr = xgb.DMatrix(np.asarray(X_o[tr_in]), label=y[tr_in], base_margin=bm_tr)
                    dtr.set_group([int((groups == qi).sum()) for qi in range(n_q)])
                    # For ES use binary auc without group — fallback: use binary logistic eval via separate
                    des = xgb.DMatrix(np.asarray(X_o[es_in]), label=y[es_in], base_margin=bm_all[es_in])
                    # rank:pairwise needs groups on all sets; use trivial group for es
                    des.set_group([len(es_in)])
                    params = dict(
                        max_depth=depth, eta=eta, subsample=0.8, colsample_bynode=0.30,
                        reg_lambda=40.0, min_child_weight=mcw, objective="rank:pairwise",
                        eval_metric="auc", tree_method="hist", max_bin=256, nthread=N_JOBS, seed=seed,
                    )
                else:
                    dtr = xgb.DMatrix(np.asarray(X_o[tr_in]), label=y[tr_in], base_margin=bm_tr)
                    des = xgb.DMatrix(np.asarray(X_o[es_in]), label=y[es_in], base_margin=bm_all[es_in])
                    params = dict(
                        max_depth=depth, eta=eta, subsample=0.8, colsample_bynode=colsample,
                        reg_lambda=reg_lambda, min_child_weight=mcw, objective="binary:logistic",
                        eval_metric="auc", tree_method="hist", max_bin=256, nthread=N_JOBS, seed=seed,
                    )
                dva = xgb.DMatrix(np.asarray(X_o[b]), base_margin=bm_all[b])
                dte = xgb.DMatrix(np.asarray(X_t), base_margin=bm_te)
                booster = xgb.train(params, dtr, num_boost_round=nround, evals=[(des, "es")], early_stopping_rounds=50, verbose_eval=False)
                oof_s[b] = booster.predict(dva)
                te_s.append(booster.predict(dte))
                log(f"  {tag} seed{seed} d{depth} {aname} F{fi} best={booster.best_iteration}")
                del booster, dtr, des, dva, dte
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
            n_models += 1
            np.save(ART / f"_partial_{tag}_{n_models}_oof.npy", oof_s.astype(np.float32))
        save_pair(tag, oof_bag / max(n_models, 1), np.mean(te_bag, 0))
        zo = zrk(oof_bag / max(n_models, 1))
        log(f"  {tag} AUC={roc_auc_score(y,zo):.7f} folds={[round(x,7) for x in fold_aucs(y,zo,folds)]}")

    # 1) dual anchor BCE
    depths = [(42, 2), (43, 3), (44, 4), (45, 2), (46, 3)]  # 5 models
    # Actually 5 on v5 + 5 on z8 with depths 2,3,4 cycling
    depths_v5 = [(42 + i, [2, 3, 4, 2, 3][i]) for i in range(5)]
    depths_z8 = [(100 + i, [2, 3, 4, 2, 3][i]) for i in range(5)]
    # Custom loop combining
    tag = "xgb_v7_bce_dual_anchor"
    if not exists_pair(tag):
        oof_bag = np.zeros(ntr)
        te_bag = []
        specs = [(s, d, v5_o, v5_t, "v5") for s, d in depths_v5] + [(s, d, z8_o, z8_t, "z8") for s, d in depths_z8]
        for bi, (seed, depth, ao, at, aname) in enumerate(specs):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(ao[tr_in], y[tr_in], ao, at)
                dtr = xgb.DMatrix(np.asarray(X_o[tr_in]), label=y[tr_in], base_margin=bm_tr)
                des = xgb.DMatrix(np.asarray(X_o[es_in]), label=y[es_in], base_margin=bm_all[es_in])
                dva = xgb.DMatrix(np.asarray(X_o[b]), base_margin=bm_all[b])
                dte = xgb.DMatrix(np.asarray(X_t), base_margin=bm_te)
                params = dict(
                    max_depth=depth, eta=0.02, subsample=0.8, colsample_bynode=0.28,
                    reg_lambda=35.0, min_child_weight=200, objective="binary:logistic",
                    eval_metric="auc", tree_method="hist", max_bin=256, nthread=N_JOBS, seed=seed,
                )
                booster = xgb.train(params, dtr, num_boost_round=550, evals=[(des, "es")], early_stopping_rounds=50, verbose_eval=False)
                oof_s[b] = booster.predict(dva)
                te_s.append(booster.predict(dte))
                log(f"  {tag} bag{bi} d{depth} {aname} F{fi} best={booster.best_iteration}")
                del booster, dtr, des, dva, dte
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
            np.save(ART / f"_partial_{tag}_{bi}_oof.npy", oof_s.astype(np.float32))
            save_state(phase="phase3_xgb_bce", bag=bi)
        save_pair(tag, oof_bag / len(specs), np.mean(te_bag, 0))
        zo = zrk(oof_bag / len(specs))
        log(f"  {tag} AUC={roc_auc_score(y,zo):.7f}")
    else:
        log(f"  SKIP {tag}")

    # 2) pairwise rank
    tag = "xgb_v7_rank_pairwise"
    if not exists_pair(tag):
        oof_bag = np.zeros(ntr)
        te_bag = []
        specs = [(42 + i, [2, 3, 2, 3, 2, 3][i]) for i in range(6)]
        for bi, (seed, depth) in enumerate(specs):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
                n_q = 50
                groups = np.zeros(len(tr_in), dtype=np.int32)
                pos = np.where(y[tr_in] == 1)[0]
                neg = np.where(y[tr_in] == 0)[0]
                for qi in range(n_q):
                    groups[pos[qi::n_q]] = qi
                    groups[neg[qi::n_q]] = qi
                # reorder by group for XGB
                order = np.argsort(groups, kind="mergesort")
                dtr = xgb.DMatrix(np.asarray(X_o[tr_in][order]), label=y[tr_in][order], base_margin=bm_tr[order])
                dtr.set_group([int((groups == qi).sum()) for qi in range(n_q)])
                des = xgb.DMatrix(np.asarray(X_o[es_in]), label=y[es_in], base_margin=bm_all[es_in])
                des.set_group([len(es_in)])
                dva = xgb.DMatrix(np.asarray(X_o[b]), base_margin=bm_all[b])
                dte = xgb.DMatrix(np.asarray(X_t), base_margin=bm_te)
                params = dict(
                    max_depth=depth, eta=0.02, subsample=0.8, colsample_bynode=0.30,
                    reg_lambda=40.0, min_child_weight=200, objective="rank:pairwise",
                    eval_metric="auc", tree_method="hist", max_bin=256, nthread=N_JOBS, seed=seed,
                )
                try:
                    booster = xgb.train(params, dtr, num_boost_round=550, evals=[(des, "es")], early_stopping_rounds=50, verbose_eval=False)
                except Exception as e:
                    log(f"  pairwise fallback to binary due to {e}")
                    params["objective"] = "binary:logistic"
                    dtr = xgb.DMatrix(np.asarray(X_o[tr_in]), label=y[tr_in], base_margin=bm_tr)
                    des = xgb.DMatrix(np.asarray(X_o[es_in]), label=y[es_in], base_margin=bm_all[es_in])
                    booster = xgb.train(params, dtr, num_boost_round=550, evals=[(des, "es")], early_stopping_rounds=50, verbose_eval=False)
                oof_s[b] = booster.predict(dva)
                te_s.append(booster.predict(dte))
                log(f"  {tag} bag{bi} d{depth} F{fi} best={getattr(booster, 'best_iteration', -1)}")
                del booster
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
            np.save(ART / f"_partial_{tag}_{bi}_oof.npy", oof_s.astype(np.float32))
            save_state(phase="phase3_xgb_rank", bag=bi)
        save_pair(tag, oof_bag / len(specs), np.mean(te_bag, 0))
        log(f"  {tag} AUC={roc_auc_score(y,zrk(oof_bag/len(specs))):.7f}")
    else:
        log(f"  SKIP {tag}")

    # 3) LGB extratrees dual
    tag = "lgb_v7_extratrees_dual"
    if not exists_pair(tag):
        oof_bag = np.zeros(ntr)
        te_bag = []
        leaves = [5, 8, 12, 5, 8]
        specs = [(42 + i, leaves[i], v5_o, v5_t, "v5") for i in range(5)] + [(100 + i, leaves[i], z8_o, z8_t, "z8") for i in range(5)]
        for bi, (seed, nl, ao, at, aname) in enumerate(specs):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(ao[tr_in], y[tr_in], ao, at)
                dtr = lgb.Dataset(np.asarray(X_o[tr_in]), label=y[tr_in], init_score=bm_tr)
                des = lgb.Dataset(np.asarray(X_o[es_in]), label=y[es_in], init_score=bm_all[es_in], reference=dtr)
                params = dict(
                    objective="binary", metric="auc", learning_rate=0.02, num_leaves=nl,
                    feature_fraction=0.25, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l1=15.0, lambda_l2=50.0, min_child_samples=300,
                    verbosity=-1, seed=seed, num_threads=N_JOBS, extra_trees=True,
                )
                booster = lgb.train(params, dtr, num_boost_round=700, valid_sets=[des],
                                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                oof_s[b] = expit(booster.predict(np.asarray(X_o[b]), raw_score=True) + bm_all[b])
                te_s.append(expit(booster.predict(np.asarray(X_t), raw_score=True) + bm_te))
                log(f"  {tag} bag{bi} L{nl} {aname} F{fi} best={booster.best_iteration}")
                del booster, dtr, des
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
            np.save(ART / f"_partial_{tag}_{bi}_oof.npy", oof_s.astype(np.float32))
            save_state(phase="phase3_lgb", bag=bi)
        save_pair(tag, oof_bag / len(specs), np.mean(te_bag, 0))
        log(f"  {tag} AUC={roc_auc_score(y,zrk(oof_bag/len(specs))):.7f}")
    else:
        log(f"  SKIP {tag}")

    # 4) Cat oblivious
    tag = "cat_v7_oblivious_d4_d5"
    if not exists_pair(tag):
        oof_bag = np.zeros(ntr)
        te_bag = []
        specs = [(42 + i, 4) for i in range(3)] + [(100 + i, 5) for i in range(3)]
        for bi, (seed, depth) in enumerate(specs):
            oof_s = np.zeros(ntr)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                tr_in, es_in = train_test_split(a, test_size=0.10, random_state=seed + 17 * fi, stratify=y[a])
                bm_tr, bm_all, bm_te = calibrate_logit_to_prevalence(v5_o[tr_in], y[tr_in], v5_o, v5_t)
                model = CatBoostClassifier(
                    loss_function="Logloss", eval_metric="AUC", depth=depth, l2_leaf_reg=40,
                    rsm=0.28, learning_rate=0.028, iterations=1200, random_seed=seed,
                    thread_count=N_JOBS, od_type="Iter", od_wait=60, verbose=False,
                    allow_writing_files=False, grow_policy="SymmetricTree",
                )
                model.fit(
                    Pool(np.asarray(X_o[tr_in]), y[tr_in], baseline=bm_tr),
                    eval_set=Pool(np.asarray(X_o[es_in]), y[es_in], baseline=bm_all[es_in]),
                    use_best_model=True,
                )
                oof_s[b] = model.predict(Pool(np.asarray(X_o[b]), baseline=bm_all[b]), prediction_type="Probability")[:, 1]
                te_s.append(model.predict(Pool(np.asarray(X_t), baseline=bm_te), prediction_type="Probability")[:, 1])
                log(f"  {tag} bag{bi} d{depth} F{fi} best={model.get_best_iteration()}")
                del model
                gc.collect()
            oof_bag += oof_s
            te_bag.append(np.mean(te_s, 0))
            np.save(ART / f"_partial_{tag}_{bi}_oof.npy", oof_s.astype(np.float32))
            save_state(phase="phase3_cat", bag=bi)
        save_pair(tag, oof_bag / len(specs), np.mean(te_bag, 0))
        log(f"  {tag} AUC={roc_auc_score(y,zrk(oof_bag/len(specs))):.7f}")
    else:
        log(f"  SKIP {tag}")

    save_state(phase="phase3_done")


# ============================================================================
# PHASE 4 — Dual gate + NOSUBMIT staging
# ============================================================================
def phase4(y, folds, fold_id, phase0_out=None):
    log("======== PHASE 4: Dual-gate verification + NOSUBMIT staging ========")
    save_state(phase="phase4_start")
    # Always reload Phase0 from disk (allows mid-run t* patches).
    if (ART / "phase0.json").exists():
        try:
            phase0_out = json.loads((ART / "phase0.json").read_text())
            log(f"  reloaded phase0.json t_star={phase0_out.get('t_star')}")
        except Exception as e:
            log(f"  phase0 reload warn: {e}")
    phase0_out = phase0_out or {}
    ntr = len(y)
    z_ref, _ = load_z_ref(folds)
    base_folds = fold_aucs(y, z_ref, folds)
    m0 = fold_id == 0

    families = [
        "xgb_v7_bce_dual_anchor",
        "xgb_v7_rank_pairwise",
        "lgb_v7_extratrees_dual",
        "cat_v7_oblivious_d4_d5",
    ]
    solo = {}
    for tag in families:
        if not exists_pair(tag):
            log(f"  MISSING {tag}")
            continue
        o, t = load_pair(tag)
        zo = zrk(o)
        auc = float(roc_auc_score(y, zo))
        fa = fold_aucs(y, zo, folds)
        solo[tag] = {"auc": auc, "folds": fa}
        log(f"  solo {tag}: AUC={auc:.7f} folds={[round(x,7) for x in fa]}")

    # baselines
    try:
        xgb_old = zrk(np.load(PRED_V5 / "gm_v5_overnight/xgb_stageB_monotone_lm_bag6_oof.npy").ravel())
        log(f"  baseline xgb_stageB AUC={roc_auc_score(y,xgb_old):.7f}")
    except Exception:
        xgb_old = None
    try:
        lgb_old = zrk(np.load(PRED_V5 / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy").ravel())
        log(f"  baseline lgb_gpt2_3ep_fixed AUC={roc_auc_score(y,lgb_old):.7f}")
    except Exception:
        lgb_old = None

    d1_o = np.load(PRED_V5 / "gm_v4_polish/d1_v4_oof.npy").ravel()
    d1_t = np.load(PRED_V5 / "gm_v4_polish/d1_v4_test.npy").ravel()

    # Blend new families
    mats_o, mats_t = [unitize(d1_o)], [unitize(d1_t)]
    if lgb_old is not None:
        mats_o.append(unitize(lgb_old))
        lt = np.load(PRED_V5 / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy").ravel()
        mats_t.append(unitize(zrk(lt)))
    for tag in families:
        if exists_pair(tag):
            o, t = load_pair(tag)
            mats_o.append(unitize(zrk(o)))
            mats_t.append(unitize(zrk(t)))
    R_o = np.column_stack(mats_o)
    R_t = np.column_stack(mats_t)

    # LOFO non-neg weights maximizing lift on z_ref
    w_grid = np.linspace(0.0, 0.35, 36)
    fold_w = []
    for f, (a, b) in enumerate(folds):
        other = [i for i in range(5) if i != f]
        w = np.zeros(R_o.shape[1])
        for _ in range(2):
            for j in range(R_o.shape[1]):
                bw, bs = w[j], -1e9
                for cw in w_grid:
                    wt = w.copy()
                    wt[j] = cw
                    lifts = [
                        float(roc_auc_score(y[folds[oi][1]], z_ref[folds[oi][1]] + R_o[folds[oi][1]] @ wt)) - base_folds[oi]
                        for oi in other
                    ]
                    s = float(np.mean(lifts)) - 0.5 * max(0.0, -float(np.min(lifts)))
                    if s > bs:
                        bs, bw = s, float(cw)
                w[j] = bw
        fold_w.append(w)
        log(f"  blend F{f} w={np.round(w,4)}")
    mean_w = np.mean(fold_w, 0)
    u_raw_o = R_o @ mean_w
    u_raw_t = R_t @ mean_w

    # Orth against z_v8 and z_w12
    z8_o = zrk(np.load(PRED_V5 / "P_super_v8_part12_rect_oof.npy").ravel())
    z8_t = zrk(np.load(PRED_V5 / "P_super_v8_part12_rect_test.npy").ravel())
    try:
        w12_o = np.load(PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_oof.npy").ravel()
        w12_t = np.load(PRED_V5 / "gm_v2_ladder/z_lm12_raw_super_test.npy").ravel()
        w12_o, w12_t = unitize(w12_o), unitize(w12_t if w12_t.std() > 0 else zrk(w12_t))
    except Exception:
        w12_o, w12_t = z8_o, z8_t
    u_o = orth_against(unitize(u_raw_o), [unitize(z8_o), unitize(w12_o)])
    u_t = orth_against(unitize(u_raw_t), [unitize(z8_t), unitize(w12_t)])
    u_o, u_t = align_method_b_smooth(u_o, u_t, folds)
    if disp_ks(u_o, u_t)[1] >= 0.0002:
        u_o, u_t = force_exact_law(u_o, u_t)
    d, ks = disp_ks(u_o, u_t)
    log(f"  a1_stageB_v7 disp={d:.6f} KS={ks:.6f}")
    save_pair("a1_stageB_v7", u_o, u_t)

    # Gate #1: compare to δ1_v4 and to S7_A OOF levels
    def eval_delta(u, tag):
        best = {"w": 0, "mean": -1e9, "d5": None, "d0": None, "auc": None, "pos": 0}
        for w in np.linspace(0.0, 0.25, 26):
            z = z_ref + w * u
            fa = fold_aucs(y, z, folds)
            d5 = [(a - b) * 1e5 for a, b in zip(fa, base_folds)]
            mean = float(np.mean(d5))
            z0 = z_ref.copy()
            z0[m0] = z_ref[m0] + w * unit(u[m0])
            d0 = (float(roc_auc_score(y[m0], z0[m0])) - float(roc_auc_score(y[m0], z_ref[m0]))) * 1e5
            pos = int(sum(x > 0 for x in d5))
            score = mean + 0.25 * d0
            if score > best["mean"]:
                best = {"w": w, "mean": score, "d5_mean": mean, "d5": d5, "d0": d0, "auc": float(roc_auc_score(y, z)), "pos": pos}
        log(f"  Gate1 {tag}: w={best['w']:.3f} 5FΔ={best['d5_mean']:+.2f}e-5 F0Δ={best['d0']:+.2f}e-5 pos={best['pos']}/5 AUC={best['auc']:.7f}")
        return best

    g_d1 = eval_delta(unitize(d1_o), "delta1_v4")
    g_v7 = eval_delta(unitize(u_o), "a1_stageB_v7")

    # With L7 denoised on fold0
    e_o, e_t = load_pair("e_L7_noise")
    L7s_o, L7s_t = load_pair("L7_5f_student")
    # combine stageB_v7 + small L7 student - noise on F0
    u_comb = unitize(unitize(u_o) + 0.15 * unitize(L7s_o) - 0.10 * unitize(e_o))
    g_comb = eval_delta(u_comb, "stageB_v7+L7denoise")

    # Gate #2: stage candidates from z_S7
    z_areeb = np.load(PRED_BO / "gm_breakout/z_areeb.npy").ravel()
    m1 = unit(np.load(PRED_BO / "gm_breakout/m1.npy").ravel())
    m2 = unit(np.load(PRED_BO / "gm_breakout/m2.npy").ravel())
    u_ray = unit(0.70 * m1 + 0.30 * m2)
    z_s7 = unitize(zrk(pd.read_csv(find_first(
        PRED_BO.parent / "submissions/sub_master_slot7_verified_s7A.csv",
        VOL_SUB / "sub_master_slot7_verified_s7A.csv",
    ))["is_fraud"].to_numpy(float)))
    p_st9 = pd.read_csv(find_ds("submission_095330.csv"))["is_fraud"].to_numpy(float)
    ids = pd.read_csv(find_ds("submission_095330.csv"))["id"].to_numpy()

    # u_stageB_v7 on test + -e_L7_noise
    u_v7_t = unit(u_t)
    e_noise_t = unit(e_t)
    u_s8 = unit(0.75 * u_ray + 0.25 * u_v7_t - 0.15 * e_noise_t)

    t_star = float(phase0_out.get("t_star", 1.15))
    # Convert t* continuation beyond S7: S7 is at t≈1.005 from areeb; step from S7
    # 1-rho(S7) targets
    slot_specs = [
        (1, "pure_ray_tstar", u_ray, max(0.10e-4, ((t_star / 100.0) ** 2) - spearman_dist(z_areeb, z_s7))),
        (2, "s7_stageBv7_L7denoise", unit(0.55 * u_ray + 0.45 * u_v7_t - 0.20 * e_noise_t), 0.15e-4),
        (3, "medium_combo", unit(0.50 * u_ray + 0.50 * u_v7_t - 0.25 * e_noise_t), 0.35e-4),
        (4, "large_combo", unit(0.45 * u_ray + 0.55 * u_v7_t - 0.30 * e_noise_t), 0.65e-4),
        (5, "adaptive_reserve", unit(0.40 * u_ray + 0.60 * u_v7_t - 0.35 * e_noise_t), 0.45e-4),
    ]

    # Simple linear surface using distances from areeb for pred
    def pred_lb(z):
        # local linear using S5,S6,S7
        d = spearman_dist(z_areeb, z)
        # fit LB = a + b*d on S5/S6/S7
        ds = np.array([spearman_dist(z_areeb, unitize(zrk(pd.read_csv(p)["is_fraud"].to_numpy(float)))) for p in [
            find_first(PRED_BO.parent / "submissions/sub_master_s5_decontam_s2_winner_0p22e4.csv", VOL_SUB / "sub_master_s5_decontam_s2_winner_0p22e4.csv"),
            find_first(PRED_BO.parent / "submissions/sub_master_s6_branchA_breakout_0p52e4.csv", VOL_SUB / "sub_master_s6_branchA_breakout_0p52e4.csv"),
            find_first(PRED_BO.parent / "submissions/sub_master_slot7_verified_s7A.csv", VOL_SUB / "sub_master_slot7_verified_s7A.csv"),
        ]])
        lbs = np.array([0.95350, 0.95352, 0.95354])
        b = (lbs[-1] - lbs[0]) / (ds[-1] - ds[0] + 1e-18)
        a = lbs[-1] - b * ds[-1]
        return float(a + b * d)

    cand_rows = []
    for slot, name, u, target in slot_specs:
        w, got = weight_for_spearman_dist(z_s7, u, target, hi=100.0)
        z = unitize(z_s7 + w * u)
        p = to_probs_mean0167(z, y, p_st9)
        assert len(p) == 299984
        assert np.isfinite(p).all()
        ks = float(ks_2samp(z_s7, z).statistic)
        fn = f"sub_gm_v7_post095354_slot{slot}_NOSUBMIT.csv"
        out_p = SUB / fn
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(out_p, index=False)
        # also try volume
        try:
            VOL_SUB.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(VOL_SUB / fn, index=False)
        except Exception:
            pass
        plb = pred_lb(z)
        row = {
            "slot": slot, "name": name, "file": fn, "target_1mrho_S7": target, "got": got,
            "w": w, "mean": float(p.mean()), "ks_vs_s7": ks, "pred_lb": plb,
        }
        cand_rows.append(row)
        log(f"  Slot{slot} {name}: got={got:.2e} mean={p.mean():.6f} KS={ks:.6e} predLB={plb:.6f} NO_SUBMIT")
        sync_artifact(out_p)

    # Report
    gate1_pass = (g_v7["d5_mean"] >= g_d1["d5_mean"]) or (g_comb["d5_mean"] > 2.38 and g_comb["pos"] >= 4)
    md = []
    md.append("# GM Deep-Tech V7 Post-0.95354 Report")
    md.append("")
    md.append("**NO_SUBMIT=True**. ATH locked at **Public LB `0.95354`** (Master Slot 7 / S7_A).")
    md.append("")
    md.append("## Phase 0 — 9-point LB surface & ray OOF ceiling")
    md.append(f"- OOS S7 check (fit goodpjw..S6 → predict S7): pred=`{phase0_out['oos_s7_pred']:.6f}` actual=`0.95354` err=`{phase0_out['oos_s7_err_e5']:+.2f}e-5`.")
    md.append(f"- OOF vertex **t\\*** = `{phase0_out['t_star']}` along S5→S6→S7 ray.")
    md.append("")
    md.append("| t | 1−ρ(areeb) | 5F Δe-5 | F0 Δe-5 | pred LB |")
    md.append("|---:|---:|---:|---:|---:|")
    for r in phase0_out.get("sweep", []):
        md.append(f"| {r['t']:.2f} | {r['one_minus_rho']:.2e} | {r['d5_mean']:+.2f} | {r['d0']:+.2f} | {r['pred_lb']:.6f} |")
    md.append("")
    md.append("## Phase 1 — Condensed blocks")
    if (ART / "phase1.json").exists():
        p1 = json.loads((ART / "phase1.json").read_text())
        md.append(f"- shape o/t = `{p1.get('shape_o')}` / `{p1.get('shape_t')}`")
        md.append(f"- per-col best Δe-5 vs z_ref_old: `{[round(x,2) for x in p1.get('lifts_e5',[])]}`")
    md.append("")
    md.append("## Phase 2 — L7 denoise")
    if (ART / "phase2.json").exists():
        md.append(f"```\n{json.dumps(json.loads((ART/'phase2.json').read_text()), indent=2)}\n```")
    md.append("")
    md.append("## Phase 3 — Stage-B v7 solo OOF")
    md.append("| family | AUC |")
    md.append("|---|---:|")
    for k, v in solo.items():
        md.append(f"| `{k}` | {v['auc']:.7f} |")
    md.append("")
    md.append("## Phase 4 — Dual gates")
    md.append(f"- δ1_v4: 5F Δ=`{g_d1['d5_mean']:+.2f}e-5` F0=`{g_d1['d0']:+.2f}e-5` pos={g_d1['pos']}/5")
    md.append(f"- a1_stageB_v7: 5F Δ=`{g_v7['d5_mean']:+.2f}e-5` F0=`{g_v7['d0']:+.2f}e-5` pos={g_v7['pos']}/5")
    md.append(f"- combo+L7denoise: 5F Δ=`{g_comb['d5_mean']:+.2f}e-5` F0=`{g_comb['d0']:+.2f}e-5` pos={g_comb['pos']}/5")
    md.append(f"- Gate#1 vs S7_A (+2.38 / +9.50): **{'PASS signal' if gate1_pass else 'WEAK / investigate'}**")
    md.append(f"- a1_stageB_v7 alignment: disp=`{d:.6f}` KS=`{ks:.6f}`")
    md.append("")
    md.append("## Tomorrow's 5-slot NOSUBMIT staging (from z_S7=0.95354)")
    md.append("| slot | name | 1−ρ(S7) | got | pred LB | file |")
    md.append("|---:|---|---:|---:|---:|---|")
    for r in cand_rows:
        md.append(f"| {r['slot']} | {r['name']} | {r['target_1mrho_S7']:.2e} | {r['got']:.2e} | {r['pred_lb']:.6f} | `{r['file']}` |")
    md.append("")
    md.append("**NO_SUBMIT=True** — 0 daily submissions remaining; candidates staged only.")
    report = "\n".join(md)
    rp = LOG / "GM_DEEP_TECH_V7_POST_095354.md"
    rp.write_text(report)
    sync_artifact(rp)
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / "GM_DEEP_TECH_V7_POST_095354.md").write_text(report)
    except Exception as e:
        log(f"  volume report sync warn: {e}")
    out = {"solo": solo, "g_d1": g_d1, "g_v7": g_v7, "g_comb": g_comb, "cands": cand_rows, "gate1_pass": gate1_pass}
    (ART / "phase4.json").write_text(json.dumps(out, indent=2, default=float))
    save_state(phase="phase4_done", phase4={"gate1_pass": gate1_pass, "n_cands": len(cand_rows)})
    log("PHASE4 DONE — report written")
    return out


def main():
    assert NO_SUBMIT, "NO_SUBMIT must be True"
    log("======== GM DEEP TECH V7 START (NO_SUBMIT) ATH=0.95354 ========")
    save_state(phase="start", ATH=ATH)
    y, folds, fold_id = load_y_folds()
    log(f"ntr={len(y)} fraud_rate={y.mean():.6f}")

    p0 = phase0(y, folds, fold_id)
    blocks_o, blocks_t = phase1(y, folds)
    phase2(y, folds, fold_id, blocks_o, blocks_t)
    phase3(y, folds)
    phase4(y, folds, fold_id, p0)
    save_state(phase="all_done")
    log("======== ALL PHASES COMPLETE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
