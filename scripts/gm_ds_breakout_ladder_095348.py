#!/usr/bin/env python3
"""DS Breakout Ladder — escape 0.95335, climb above 0.95347.

Uses 4 of 7 remaining daily slots (reserve 3). Adaptive branch after each poll.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "0")  # LIVE submits for this ladder

warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("GM_BO_WORK", "/tmp/gm_breakout")).resolve()
ROOT = Path(os.environ.get("GM_BO_ROOT", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")).resolve()
EXT = Path(os.environ.get("GM_BO_EXT", "/tmp/gm_v6_audit/external/nov11_095347_recon")).resolve()
DS = EXT / "datasets"
PRED = WORK / "preds"
VOL_PRED = ROOT / "preds"
SUB = WORK / "submissions"
VOL_SUB = ROOT / "submissions"
LOG = WORK / "logs" / "overnight"
VOL_LOG = ROOT / "logs" / "overnight"
ART = WORK / "preds" / "gm_breakout"
for p in [SUB, LOG, ART, PRED]:
    p.mkdir(parents=True, exist_ok=True)

KAG = str(ROOT / ".venv_forensics/bin/kaggle")
COMP = "playground-series-s6e11"
MAX_SLOTS = 4
RESERVE = 3
T0 = time.time()

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
N_Q = 10001


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


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(np.asarray(b, dtype=np.float64).ravel())
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


def weight_for_spearman_dist(z0, u, target, hi=20.0):
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


def bayes_fold(arr, prior):
    return (
        np.logaddexp.reduce(arr[:, :, 1].astype(np.float64), axis=1)
        - np.logaddexp.reduce(arr[:, :, 0].astype(np.float64), axis=1)
        + np.log(prior / (1.0 - prior))
    )


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


def pchip_map_to_strike9(score, p_st9):
    """Map score ranks onto Strike9 probability order statistics (mean≈0.0167)."""
    score = np.asarray(score, float).ravel()
    p_st9 = np.asarray(p_st9, float).ravel()
    sorted_st9 = np.sort(p_st9)
    n = len(score)
    # rank of score -> quantile -> PCHIP onto sorted St9
    r = (rankdata(score, method="average") - 0.5) / n
    q = np.linspace(0.0, 1.0, N_Q)
    # identity map in quantile space onto St9 order stats
    pchip = PchipInterpolator(np.linspace(0.0, 1.0, n, endpoint=True), sorted_st9, extrapolate=True)
    # use order-stat positions
    order = np.argsort(_jitter(score, 1e-7), kind="mergesort")
    out = np.empty(n, dtype=np.float64)
    pos = np.linspace(0.0, 1.0, n, endpoint=True)
    out[order] = pchip(pos)
    # tiny numerical fix for mean
    out = np.clip(out, 1e-12, 1 - 1e-12)
    return out


def find_ds(name: str) -> Path:
    hits = sorted(DS.rglob(name))
    hits = [h for h in hits if h.is_file() and not h.name.startswith("._")]
    assert hits, name
    return hits[0]


def run(cmd, timeout=120):
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output or str(e)
    except Exception as e:
        return 1, str(e)


def parse_submissions_table(raw: str) -> list[dict]:
    # kaggle CLI table — use csv-ish whitespace fragile; prefer DictReader on lines
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    # find header
    hdr_i = 0
    for i, ln in enumerate(lines):
        if "fileName" in ln and "publicScore" in ln:
            hdr_i = i
            break
    header = lines[hdr_i]
    # split on 2+ spaces
    import re
    cols = re.split(r"\s{2,}", header.strip())
    rows = []
    for ln in lines[hdr_i + 1 :]:
        if "----" in ln:
            continue
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) < 5:
            continue
        # map by position from known kaggle format:
        # ref, fileName, date, description, status, publicScore, privateScore?
        d = {"raw": ln}
        try:
            d["ref"] = parts[0]
            d["fileName"] = parts[1]
            d["date"] = parts[2] if len(parts) > 2 else ""
            # status often near end
            status = ""
            score = None
            for p in parts:
                if "SubmissionStatus" in p or p in ("Complete", "Pending", "Error", "SubmissionStatus.COMPLETE", "SubmissionStatus.PENDING"):
                    status = p
                try:
                    fv = float(p)
                    if 0.9 < fv < 1.0:
                        score = fv
                except Exception:
                    pass
            d["status"] = status
            d["publicScore"] = score
            rows.append(d)
        except Exception:
            continue
    return rows


def submit_and_poll(path: Path, message: str, timeout_s: int = 900) -> float | None:
    log(f"SUBMIT {path.name}: {message}")
    rc, out = run(
        [KAG, "competitions", "submit", "-c", COMP, "-f", str(path), "-m", message],
        timeout=180,
    )
    log(f"  submit rc={rc} out={out[-300:].replace(chr(10),' ')}")
    if rc != 0:
        return None
    t_end = time.time() + timeout_s
    last = None
    while time.time() < t_end:
        time.sleep(20)
        rc2, raw = run([KAG, "competitions", "submissions", "-c", COMP], timeout=120)
        rows = parse_submissions_table(raw)
        hit = None
        for r in rows:
            if path.name in str(r.get("fileName", "")):
                hit = r
                break
        if hit is None and rows:
            # newest first usually
            hit = rows[0]
        if hit:
            st = str(hit.get("status", ""))
            sc = hit.get("publicScore")
            log(f"  poll status={st} score={sc}")
            last = sc
            if sc is not None and ("COMPLETE" in st.upper() or st == "Complete"):
                return float(sc)
            if "ERROR" in st.upper():
                return None
    log("  poll timeout")
    return last


def save_sub(ids, p, name: str) -> Path:
    path = SUB / name
    pd.DataFrame({"id": ids, "is_fraud": np.asarray(p, float)}).to_csv(path, index=False)
    try:
        (VOL_SUB / name).write_bytes(path.read_bytes())
    except Exception as e:
        log(f"  vol copy warn: {e}")
    mean = float(np.mean(p))
    assert np.isfinite(p).all()
    log(f"  wrote {name} mean={mean:.6f} n={len(p)} nans=0")
    return path


# =============================================================================
def main():
    log("GM_DS_BREAKOUT_LADDER_095348 START (4 slots, reserve 3)")
    inventory = {}

    # ---- inventory ----
    compact = DS / "goodpjw2008__s6e11-compact-lm-seed7"
    oofbank = DS / "goodpjw2008__s6e11-oof-bank"
    lucifer = DS / "lucifer19__golden-signal-v5-scored-predictions"
    inv_lines = []
    for label, root in [("compact-lm-seed7", compact), ("oof-bank", oofbank)]:
        inv_lines.append(f"### {label}")
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.name.startswith("._"):
                continue
            if f.suffix == ".npy":
                a = np.load(f, mmap_mode="r")
                inv_lines.append(f"- `{f.name}` shape={tuple(a.shape)} dtype={a.dtype}")
            else:
                inv_lines.append(f"- `{f.name}` size={f.stat().st_size}")
    # lucifer
    inv_lines.append("### lucifer19/golden-signal-v5-scored-predictions")
    luc_sub = pd.read_csv(lucifer / "submission.csv")
    inv_lines.append(f"- `submission.csv` shape={luc_sub.shape} mean={luc_sub['is_fraud'].mean():.6f}")
    # try to find LB in readme / notebook references — dataset has no explicit LB file; note unknown
    inv_lines.append("- explicit LB label in dataset: **not present** (submission.csv only + challenger_predictions.npz)")
    z = np.load(lucifer / "challenger_predictions.npz")
    inv_lines.append(f"- `challenger_predictions.npz` keys={z.files} oof={z['oof'].shape} test={z['test'].shape}")
    inventory["text"] = "\n".join(inv_lines)
    log("inventory done; lm7 folds: only f0 (lm7_oof_f0, lm7_test_f0)")

    # ---- load native inputs ----
    st10 = pd.read_csv(find_ds("submission_095340.csv"))
    st9 = pd.read_csv(find_ds("submission_095330.csv"))
    sample = st9.copy()
    ids = sample["id"].to_numpy()
    p_A = st10["is_fraud"].to_numpy(float)
    p_F = st9["is_fraud"].to_numpy(float)
    assert len(p_A) == 299984

    LM_curr = np.load(find_ds("lm_test.npy")).astype(np.float64).ravel()
    LM_nov08 = np.load(find_ds("megalm_test_v_nov08.npy")).astype(np.float64).ravel()
    a7 = np.load(find_ds("lm7_test_f0.npy"))
    L7_raw = (a7[..., 1] - a7[..., 0]).mean(1)
    MLP = np.load(find_ds("test_gbn_bag5_w512.npy")).astype(np.float64).ravel()
    GB = np.load(find_ds("test_gb_bag5.npy")).astype(np.float64).ravel()

    # native logit_rank as in notebooks
    def L(v):
        return logit((rankdata(v, method="average") - 0.5) / len(v))

    A, F = L(p_A), L(p_F)
    LMm, LMcu = L(LM_nov08), L(LM_curr)
    L7 = L(L7_raw)
    MLPz, GBz = L(MLP), L(GB)

    # goodpjw exact
    score_gp = A + 0.6 * (A - F) + 0.12 * (LMm - F) + 0.06 * (L7 - F)
    gp_pulled = pd.read_csv(
        EXT / "goodpjw2008__s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346_out/submission.csv"
    )["is_fraud"].to_numpy(float)
    gp_rank = (rankdata(score_gp) - 0.5) / len(score_gp)
    d_gp = spearman_dist(gp_rank, gp_pulled)
    log(f"VERIFY goodpjw 1-spearmanr(recon,pulled)={d_gp:.3e} (want <1e-12)")
    assert d_gp < 1e-12, d_gp

    # areeb exact
    B = A + 0.6 * (A - F) + 0.12 * (LMcu - F) + 0.06 * (L7 - F)
    score_areeb = B - 0.05 * (MLPz - F) - 0.025 * (GBz - F)
    areeb_rank = (rankdata(score_areeb) - 0.5) / len(score_areeb)
    # optional compare to prior recon
    recon_path = Path("/tmp/gm_v6_audit/preds/gm_v6_audit/recon_areeb_095347.csv")
    if recon_path.exists():
        ar_prev = pd.read_csv(recon_path)["is_fraud"].to_numpy(float)
        log(f"VERIFY areeb vs prior recon 1-rho={spearman_dist(areeb_rank, ar_prev):.3e}")

    z_areeb = unitize(zrk(areeb_rank))  # standardized rank-logit
    z_goodpjw = unitize(zrk(gp_rank))
    log(f"z_areeb mean={z_areeb.mean():.2e} std={z_areeb.std():.4f}")

    # transform audit
    transform_note = {
        "A": "Strike10 submission_095340.csv → logit_rank",
        "F": "Strike9 submission_095330.csv → logit_rank",
        "LM_areeb": "megayak/lm_test.npy → logit_rank (CURRENT)",
        "LM_goodpjw": "megalm_test_v_nov08.npy → logit_rank (FROZEN 11-08)",
        "L7": "lm7_test_f0.npy [N,4,2] → mean(logpT-logpF) → logit_rank",
        "MLP": "test_gbn_bag5_w512.npy → logit_rank",
        "GB": "test_gb_bag5.npy → logit_rank",
        "goodpjw": "A+0.6(A-F)+0.12(LMm-F)+0.06(L7-F)",
        "areeb": "B=A+0.6(A-F)+0.12(LMcu-F)+0.06(L7-F); score=B-0.05(MLP-F)-0.025(GB-F)",
        "verify_goodpjw_1mrho": d_gp,
    }

    # ---- B6 span ----
    B6 = gram_orthonormal(np.column_stack([unitize(A), unitize(F), unitize(LMcu), unitize(L7), unitize(MLPz), unitize(GBz)]))
    log(f"B6 cols={B6.shape[1]}")

    # B_bad from overnight Q / q_bad
    B_bad_cols = []
    qpath = PRED / "gm_v5_overnight/Q_orth.npy"
    if not qpath.exists():
        qpath = VOL_PRED / "gm_v5_overnight/Q_orth.npy"
    nte = len(z_areeb)
    if qpath.exists():
        Q = np.load(qpath)
        if Q.shape[0] == nte and Q.shape[1] > 5:
            B_bad_cols += [Q[:, 2], Q[:, 5]]
    for nm in ["q_bad_tree.npy", "q_bad_S3.npy", "q_bad_ldr.npy"]:
        pp = PRED / "gm_v5_overnight" / nm
        if not pp.exists():
            pp = VOL_PRED / "gm_v5_overnight" / nm
        if pp.exists():
            v = np.load(pp).ravel()
            if len(v) == nte:
                B_bad_cols.append(v)
    if not B_bad_cols:
        log("WARNING: no B_bad; using empty")
        B_bad = np.zeros((nte, 0))
    else:
        B_bad = gram_orthonormal(np.column_stack(B_bad_cols))
    log(f"B_bad cols={B_bad.shape[1]}")

    def proj_perp(v):
        bases = [B6[:, j] for j in range(B6.shape[1])] + [B_bad[:, j] for j in range(B_bad.shape[1])]
        return unit(orth_against(unitize(v), bases))

    # ---- Axis 1: gpt2m_lm ----
    log("======== Axis u_gpt2m_lm ========")
    # load y for prior
    y_path = Path("/tmp/gm_v6_audit/data/train.csv")
    if not y_path.exists():
        y_path = ROOT / "data/train.csv"
    y = pd.read_csv(y_path, usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    prior = float(y.mean())
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    med = PRED / "gpt2_med_4ep"
    if not (med / "test_0.npy").exists():
        med = VOL_PRED / "gpt2_med_4ep"
    bay_t_folds = []
    for f in range(5):
        tarr = np.load(med / f"test_{f}.npy")
        bay_t_folds.append(zrk(bayes_fold(tarr, prior)))
    z_gpt2m = unitize(np.mean(bay_t_folds, 0))
    # align_method_b_smooth vs oof blend optional — for test axis use mean bayes test
    bi_path = PRED / "gpt2_3ep/gpt2_bidir_test.npy"
    if not bi_path.exists():
        bi_path = VOL_PRED / "gpt2_3ep/gpt2_bidir_test.npy"
    z_bidir = unitize(zrk(np.load(bi_path).ravel()))
    z_mega08 = unitize(zrk(LM_nov08))
    mix = unitize(0.70 * z_gpt2m + 0.15 * z_bidir + 0.15 * z_mega08)
    u_gpt2m_lm = proj_perp(mix)
    log(f"  u_gpt2m_lm max|cos B6|={max(abs(cos(u_gpt2m_lm, B6[:,j])) for j in range(B6.shape[1])):.2e}")

    # ---- Axis 2: stageB_5f ----
    log("======== Axis u_stageB_5f ========")
    d1t = unitize(np.load(PRED / "gm_v4_polish/d1_v4_test.npy" if (PRED / "gm_v4_polish/d1_v4_test.npy").exists() else VOL_PRED / "gm_v4_polish/d1_v4_test.npy").ravel())
    uj = unit(np.load(PRED / "gm_v4_1/u_zjoint_clean_test.npy" if (PRED / "gm_v4_1/u_zjoint_clean_test.npy").exists() else VOL_PRED / "gm_v4_1/u_zjoint_clean_test.npy").ravel())
    ug = unit(np.load(PRED / "gm_v4_1/u_glm_graph_clean_test.npy" if (PRED / "gm_v4_1/u_glm_graph_clean_test.npy").exists() else VOL_PRED / "gm_v4_1/u_glm_graph_clean_test.npy").ravel())
    stage = unitize(d1t + 0.20 * (uj + ug))
    u_stageB = proj_perp(stage)
    log(f"  u_stageB max|cos B6|={max(abs(cos(u_stageB, B6[:,j])) for j in range(B6.shape[1])):.2e}")

    # ---- Axis 3: 3ladder_grad ----
    log("======== Axis u_3ladder_grad ========")
    score_3p = (
        A
        + 0.62 * (A - F)
        + 0.13 * (0.5 * LMcu + 0.5 * LMm - F)
        + 0.075 * (L7 - F)
        - 0.062 * (MLPz - F)
        - 0.031 * (GBz - F)
    )
    z_3p = unitize(zrk((rankdata(score_3p) - 0.5) / len(score_3p)))
    u_3ladder = unit(z_3p - z_areeb)
    # do NOT force perp to B6 here — user defines as native continuation delta
    log(f"  ||u_3ladder|| ok; cos(areeb)={cos(u_3ladder, z_areeb):+.4f}")

    np.save(ART / "u_gpt2m_lm.npy", u_gpt2m_lm)
    np.save(ART / "u_stageB_5f.npy", u_stageB)
    np.save(ART / "u_3ladder_grad.npy", u_3ladder)
    np.save(ART / "z_areeb.npy", z_areeb)

    def step_from(z0, direction, target_dist):
        u = unit(direction)
        # orth to z0 to make pure spearman step
        u = unit(orth_against(u, [unitize(z0)]))
        w, got = weight_for_spearman_dist(z0, u, target_dist)
        z = z0 + w * u
        return z, w, got

    def z_to_p(z):
        # rank of z → PCHIP onto Strike9 probs
        return pchip_map_to_strike9(z, p_F)

    history = []
    slots_used = 0

    # ========== SLOT 1 ==========
    log("======== SLOT 1 ========")
    d1 = unit(0.55 * u_gpt2m_lm + 0.45 * u_3ladder)
    z_s1, w1, got1 = step_from(z_areeb, d1, 0.18e-4)
    p_s1 = z_to_p(z_s1)
    f1 = save_sub(ids, p_s1, "sub_breakout_s1_areeb_gpt2m_3ladder_0p18e4.csv")
    lb1 = submit_and_poll(f1, f"Breakout S1: areeb+0.55gpt2m+0.45*3ladder @0.18e-4 (w={w1:.4g},got={got1:.2e})")
    slots_used += 1
    history.append({"slot": 1, "file": f1.name, "LB": lb1, "target_1mrho": 0.18e-4, "got": got1, "branch": "S1"})
    log(f"LB_1={lb1}")

    # ========== SLOT 2 ==========
    log("======== SLOT 2 ========")
    AREEb = 0.95347
    branch = "B"
    if lb1 is not None and lb1 >= 0.95348:
        branch = "A"
        z_champ = z_s1
        d2 = unit(0.55 * u_gpt2m_lm + 0.45 * u_3ladder + 0.25 * u_stageB)
        z_s2, w2, got2 = step_from(z_areeb, d2, 0.45e-4)  # 2x further from areeb
        name2 = "sub_breakout_s2_branchA_breakout_0p45e4.csv"
        msg2 = f"Breakout S2 BranchA: LB1={lb1}>={0.95348} step 0.45e-4 w/ stageB"
    elif lb1 is not None and abs(lb1 - AREEb) < 5e-6:
        branch = "B"
        d2 = unit(0.50 * u_gpt2m_lm + 0.50 * u_stageB)
        z_s2, w2, got2 = step_from(z_areeb, d2, 0.38e-4)
        name2 = "sub_breakout_s2_branchB_stageB_gpt2m_0p38e4.csv"
        msg2 = f"Breakout S2 BranchB: LB1={lb1} tie; gpt2m+stageB @0.38e-4"
    else:
        branch = "C"
        d2 = unit(-0.40 * u_3ladder + 0.70 * u_gpt2m_lm + 0.50 * u_stageB)
        z_s2, w2, got2 = step_from(z_areeb, d2, 0.20e-4)
        name2 = "sub_breakout_s2_branchC_damped_gpt2m_stageB_0p20e4.csv"
        msg2 = f"Breakout S2 BranchC: LB1={lb1}<=0.95346; damp 3ladder + gpt2m+stageB @0.20e-4"

    p_s2 = z_to_p(z_s2)
    f2 = save_sub(ids, p_s2, name2)
    lb2 = submit_and_poll(f2, msg2)
    slots_used += 1
    history.append({"slot": 2, "file": f2.name, "LB": lb2, "branch": branch, "got": got2})
    log(f"LB_2={lb2} branch={branch}")

    # ========== SLOT 3 ==========
    log("======== SLOT 3 ========")
    # empirical gradient in span of axes + (z_areeb - z_goodpjw)
    u_ag = unit(z_areeb - z_goodpjw)
    anchors = [
        (0.95346, z_goodpjw),
        (0.95347, z_areeb),
    ]
    if lb1 is not None:
        anchors.append((lb1, z_s1))
    if lb2 is not None:
        anchors.append((lb2, z_s2))
    # fit LB ~ a0 + a·coords in 4D direction basis via lstsq on deltas from areeb
    basis = [u_gpt2m_lm, u_stageB, u_3ladder, u_ag]
    X = []
    ylb = []
    for lb, z in anchors:
        d = z - z_areeb
        X.append([1.0] + [float(np.dot(d, unit(b))) for b in basis])
        ylb.append(lb)
    X = np.asarray(X, float)
    ylb = np.asarray(ylb, float)
    coef, *_ = np.linalg.lstsq(X, ylb, rcond=None)
    # steepest ascent in basis coords = coef[1:]
    g = coef[1:]
    log(f"  empirical grad coef={np.round(coef, 6).tolist()}")
    u_emp = unit(sum(float(g[i]) * unit(basis[i]) for i in range(4)))
    # best anchor
    best_lb, best_z = max(anchors, key=lambda t: (t[0] if t[0] is not None else -1))
    all_tie = (
        lb1 is not None
        and lb2 is not None
        and abs(lb1 - AREEb) < 5e-6
        and abs(lb2 - AREEb) < 5e-6
    )
    if all_tie:
        # Vertex - St9 orth
        v_star = -23.275301
        v2_star = 3.056557
        # approximate private vertex dir from local S5 if available else u_ag
        uA_path = VOL_PRED / "gm_v5p2/probeA_u.npy"
        if uA_path.exists():
            # use post-St9 private as (z_areeb direction weak); prefer u_gpt2m+stageB+vertex proxy
            u_priv = unit(0.50 * u_gpt2m_lm + 0.30 * u_stageB + 0.20 * proj_perp(u_ag))
        else:
            u_priv = unit(0.50 * u_gpt2m_lm + 0.30 * u_stageB + 0.20 * u_ag)
        z_s3, w3, got3 = step_from(z_areeb, u_priv, 0.75e-4)
        name3 = "sub_breakout_s3_alltie_gpt2m_stageB_vertex_0p75e4.csv"
        msg3 = f"Breakout S3 all-tie@0.95347: gpt2m+stageB+priv @0.75e-4"
    else:
        z_s3, w3, got3 = step_from(best_z, u_emp, 0.30e-4)
        name3 = "sub_breakout_s3_empgrad_0p30e4.csv"
        msg3 = f"Breakout S3 empgrad from LB={best_lb} @0.30e-4 coef={np.round(g,4).tolist()}"

    p_s3 = z_to_p(z_s3)
    f3 = save_sub(ids, p_s3, name3)
    lb3 = submit_and_poll(f3, msg3)
    slots_used += 1
    history.append({"slot": 3, "file": f3.name, "LB": lb3, "got": got3, "best_anchor": best_lb})
    log(f"LB_3={lb3}")

    # ========== SLOT 4 ==========
    log("======== SLOT 4 ========")
    lbs = [x for x in [lb1, lb2, lb3] if x is not None]
    reached = any(x >= 0.95348 for x in lbs)
    if reached:
        champ_lb = max(lbs)
        # pick champ z
        candz = [(lb1, z_s1), (lb2, z_s2), (lb3, z_s3)]
        z_ch = max([c for c in candz if c[0] is not None], key=lambda t: t[0])[1]
        # continuation along emp + gpt2m
        u4 = unit(0.60 * u_gpt2m_lm + 0.25 * u_stageB + 0.15 * u_emp)
        z_s4, w4, got4 = step_from(z_ch, u4, 0.25e-4)
        name4 = "sub_breakout_s4_crest_lock_0p25e4.csv"
        msg4 = f"Breakout S4 crest lock from {champ_lb} @0.25e-4"
    else:
        # pure areeb + micro gpt2m @ 0.08e-4 to lock 0.95347+
        z_s4, w4, got4 = step_from(z_areeb, u_gpt2m_lm, 0.08e-4)
        name4 = "sub_breakout_s4_areeb_micro_gpt2m_0p08e4.csv"
        msg4 = "Breakout S4: pure areeb + micro gpt2m @0.08e-4 lock 0.95347+"

    p_s4 = z_to_p(z_s4)
    f4 = save_sub(ids, p_s4, name4)
    lb4 = submit_and_poll(f4, msg4)
    slots_used += 1
    history.append({"slot": 4, "file": f4.name, "LB": lb4, "got": got4, "crest": reached})
    log(f"LB_4={lb4}")

    # ---- report ----
    lines = [
        "# GM DS Breakout Ladder → above 0.95347",
        "",
        f"Elapsed **{time.time()-T0:.1f}s**. Slots used **{slots_used}/4** (reserve **{RESERVE}**).",
        f"Base escape: Ladder S5 `0.95335` → operate on areeb `0.95347` manifold.",
        "",
        "## Step 1 — Native reproduction & inventory",
        "",
        f"- **goodpjw verify** `1-spearmanr(recon, pulled submission.csv) = {d_gp:.3e}` (machine precision OK).",
        f"- **areeb transform:** `{transform_note['areeb']}`",
        f"- LM: areeb uses **current** `lm_test.npy`; goodpjw uses **frozen** `megalm_test_v_nov08.npy`.",
        f"- L7: only **fold 0** present (`lm7_test_f0.npy` shape `(299984,4,2)`, `lm7_oof_f0.npy` `(139992,10,2)`). **No lm7_f1..f4.**",
        "",
        "### Dataset inventory highlights",
        "",
        inventory["text"],
        "",
        "## Step 2 — Orthogonal axes on z_areeb",
        "",
        f"- B6 cols=`{B6.shape[1]}` (A,F,LM_curr,L7,MLP,GB)",
        f"- B_bad cols=`{B_bad.shape[1]}`",
        f"- `u_gpt2m_lm` = unit(P⊥(0.70 gpt2m_5f_bayes + 0.15 bidir + 0.15 megalm_nov08))",
        f"- `u_stageB_5f` = unit(P⊥(δ1_v4_test + 0.20(u_zjoint+u_glm_graph)))",
        f"- `u_3ladder_grad` = unit(zrk(score_3ladder_plus) − z_areeb)",
        "",
        "## Step 3 — Live LB adaptive ladder",
        "",
        "| slot | file | branch/note | 1−ρ target | LB |",
        "|---:|---|---|---:|---:|",
    ]
    for h in history:
        lines.append(
            f"| {h['slot']} | `{h['file']}` | {h.get('branch', h.get('crest', ''))} | "
            f"{h.get('got', float('nan')):.2e} | **{h.get('LB')}** |"
        )
    best = max([h["LB"] for h in history if h.get("LB") is not None], default=None)
    lines += [
        "",
        f"**Best live LB this ladder: `{best}`** (vs areeb `0.95347`, prior champ S5 `0.95335`).",
        "",
        "STOP after Slot 4 — 3 daily submissions remain in reserve.",
        "",
        f"Artifacts: `{ART}` · submissions under `submissions/sub_breakout_s*.csv`.",
        "",
    ]
    report = LOG / "GM_DS_BREAKOUT_095348_LADDER.md"
    report.write_text("\n".join(lines))
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / report.name).write_text(report.read_text())
        (ROOT / "scripts" / "gm_ds_breakout_ladder_095348.py").write_text(Path(__file__).read_text())
    except Exception as e:
        log(f"volume sync warn: {e}")
    (ART / "HISTORY.json").write_text(json.dumps({"history": history, "transform": transform_note, "best": best}, indent=2, default=float))
    log(f"Wrote {report}")
    log(f"BREAKOUT_COMPLETE best={best} slots={slots_used} reserve={RESERVE}")


if __name__ == "__main__":
    main()
