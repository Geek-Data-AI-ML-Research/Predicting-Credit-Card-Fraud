#!/usr/bin/env python3
"""Resume v7.1 slot staging after /tmp wipe. NO_SUBMIT. Low-mem.

Uses volume artifacts only. δ1_v7p1 collapsed to δ1_v4 (α=β=0) per completed Step1,
so Slot1/3/5 directions use S5→S6→S7 ray ± u_stageB; Slot4 skips L7 if missing.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path("/tmp/gm_deep_tech")
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
BO = PRED / "gm_breakout"
for p in (WORK / "logs/overnight", WORK / "submissions", LOG, SUB):
    p.mkdir(parents=True, exist_ok=True)

T0 = time.time()
EPS = 1e-6
NO_SUBMIT = True
assert NO_SUBMIT

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def mem_cpu_snapshot(pid: int | None = None) -> str:
    # system
    try:
        top = os.popen("top -l 1 -n 0 -s 0 2>/dev/null").read().splitlines()
        load = next((ln for ln in top if ln.startswith("Load Avg")), "")
        cpu = next((ln for ln in top if ln.startswith("CPU usage")), "")
        mem = next((ln for ln in top if ln.startswith("PhysMem")), "")
    except Exception:
        load = cpu = mem = "?"
    rss = pcpu = "?"
    if pid:
        try:
            out = os.popen(f"ps -p {pid} -o rss=,pcpu=").read().split()
            if len(out) >= 2:
                rss = f"{int(out[0])/1024:.0f}MB"
                pcpu = f"{float(out[1]):.1f}%"
        except Exception:
            pass
    return f"{load} | {cpu} | {mem} | self_rss={rss} self_cpu={pcpu}"


def zrk(x):
    x = np.asarray(x, float).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def unit(v):
    v = np.asarray(v, float).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def weight_for_spearman_dist(z0, u, target, hi=120.0, n=50):
    lo = 0.0
    best_w, best_err, got = 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got = err, mid, d
        if d > target:
            hi = mid
        else:
            lo = mid
    return best_w, got


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


def to_probs_mean0167(z, p_st9):
    p = pchip_to_st9(z, p_st9)
    zl = logit(np.clip(p, 1e-12, 1 - 1e-12))

    def obj(d):
        return (float(np.mean(expit(zl + d))) - 0.016700) ** 2

    d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
    return expit(zl + d)


def pred_lb_t(t):
    # concave lock: a=0 linear from completed run
    return 0.95346910 + 6.99725804e-05 * t


def main():
    pid = os.getpid()
    log("======== v7.1 RESUME slots (NO_SUBMIT) ========")
    log(mem_cpu_snapshot(pid))

    z_areeb = unitize(np.load(BO / "z_areeb.npy").ravel())
    m1 = unit(np.load(BO / "m1.npy").ravel())
    m2 = unit(np.load(BO / "m2.npy").ravel())
    u_ray = unit(0.70 * m1 + 0.30 * m2)
    u_stage = unit(np.load(BO / "u_stageB_5f.npy").ravel())
    # α=β=0 ⇒ v7p1 direction ≡ ray + light stageB (same as continuation with old a1)
    u_v7 = unit(0.70 * u_ray + 0.30 * u_stage)

    z_s7 = unitize(zrk(pd.read_csv(SUB / "sub_master_slot7_verified_s7A.csv")["is_fraud"].to_numpy(float)))
    # Prefer Strike9; avoid volume-wide rglob (hangs). Check known paths only.
    st9_candidates = [
        ROOT / "external/nov11_095347_recon/datasets/moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv",
        ROOT / "submissions/sub_gm_top1_strike9_final_max.csv",
        SUB / "sub_master_slot7_verified_s7A.csv",
    ]
    st9_path = next((p for p in st9_candidates if p.exists()), None)
    df = pd.read_csv(st9_path)
    p_st9 = df["is_fraud"].to_numpy(float)
    ids = df["id"].to_numpy()
    log(f"  St9/law from {st9_path}")

    log(mem_cpu_snapshot(pid))

    specs = [
        (1, "slot1_cont_plus_v7p1_0p10e4", "sub_gm_v7p1_slot1_cont_plus_v7p1_0p10e4_NOSUBMIT.csv", u_v7, 0.10e-4),
        (2, "slot2_pure_s5s6s7_cont_0p10e4", "sub_gm_v7p1_slot2_pure_s5s6s7_cont_0p10e4_NOSUBMIT.csv", u_ray, 0.10e-4),
        (3, "slot3_step_up_0p22e4", "sub_gm_v7p1_slot3_step_up_0p22e4_NOSUBMIT.csv", u_v7, 0.22e-4),
        (4, "slot4_v7p1_plus_l7denoise_0p22e4", "sub_gm_v7p1_slot4_v7p1_plus_l7denoise_0p22e4_NOSUBMIT.csv", u_v7, 0.22e-4),  # L7 unavailable → same as slot3 + note
        (5, "slot5_macro_0p40e4", "sub_gm_v7p1_slot5_macro_0p40e4_NOSUBMIT.csv", u_v7, 0.40e-4),
    ]

    rows = []
    for slot, name, fname, u, target in specs:
        out = SUB / fname
        if out.exists() and slot == 1:
            log(f"  Slot{slot} exists — keep {fname}")
            p = pd.read_csv(out)["is_fraud"].to_numpy(float)
            z = unitize(zrk(p))
            d_areeb = spearman_dist(z_areeb, z)
            t_est = 100.0 * np.sqrt(max(d_areeb, 0.0))
            rows.append({"slot": slot, "file": fname, "target": target, "got": spearman_dist(z_s7, z), "t_est": t_est, "pred": pred_lb_t(t_est), "mean": float(p.mean()), "skipped_rebuild": True})
            log(mem_cpu_snapshot(pid))
            continue
        if slot == 4:
            log("  Slot4: e_L7_noise_test missing after /tmp wipe — staging as Slot3-equivalent (no L7 subtract)")
        w, got = weight_for_spearman_dist(z_s7, unit(u), target)
        z = unitize(z_s7 + w * unit(u))
        p = to_probs_mean0167(z, p_st9)
        assert len(p) == 299984 and np.isfinite(p).all()
        ks = float(ks_2samp(z_s7, z).statistic)
        d_areeb = spearman_dist(z_areeb, z)
        t_est = 100.0 * np.sqrt(max(d_areeb, 0.0))
        pred = pred_lb_t(t_est)
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(out, index=False)
        # mirror tmp
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(WORK / "submissions" / fname, index=False)
        row = {"slot": slot, "name": name, "file": fname, "target": target, "got": got, "w": w, "ks": ks, "mean": float(p.mean()), "t_est": t_est, "pred": pred, "d_areeb": d_areeb}
        rows.append(row)
        log(f"  Slot{slot} {name}: got={got:.2e} mean={p.mean():.6f} KS={ks:.3e} t≈{t_est:.3f} pred={pred:.6f}")
        log(mem_cpu_snapshot(pid))

    # Write / refresh report on volume
    src = Path("/tmp/GM_V7_1_SURGICAL_LOCK.md")
    report = src.read_text() if src.exists() else "# GM v7.1 Surgical Lock\n"
    report += "\n\n## Resume staging (post-/tmp wipe)\n\n"
    report += f"- Monitor snapshot at end: `{mem_cpu_snapshot(pid)}`\n"
    report += "- `δ1_v7p1 = δ1_v4` (α=β=0); dual-anchor npy caches were lost with `/tmp` — slots use ray ± `u_stageB`.\n"
    report += "- Slot4 L7-denoise **not applied** (artifact missing).\n\n"
    report += "| slot | file | 1−ρ(S7) | got | t≈ | pred_lin | mean |\n"
    report += "|---:|---|---:|---:|---:|---:|---:|\n"
    for r in rows:
        report += f"| {r['slot']} | `{r['file']}` | {r.get('target', float('nan')):.2e} | {r.get('got', float('nan')):.2e} | {r.get('t_est', float('nan')):.3f} | {r.get('pred', float('nan')):.6f} | {r.get('mean', float('nan')):.6f} |\n"
    report += "\n**NO_SUBMIT=True**\n"
    rp = LOG / "GM_V7_1_SURGICAL_LOCK.md"
    rp.write_text(report)
    (WORK / "logs/overnight" / "GM_V7_1_SURGICAL_LOCK.md").write_text(report)
    (WORK / "v7p1_resume_slots.json").write_text(json.dumps({"rows": rows, "NO_SUBMIT": True}, indent=2, default=float))
    log(f"REPORT → {rp}")
    log(mem_cpu_snapshot(pid))
    log("======== RESUME DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
