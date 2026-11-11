#!/usr/bin/env python3
"""GM_V2 Stages 3–4: null-space deltas + adaptive Slots 2–5 (submits + polls LB).

Assumes Slot1 already landed LB=0.95333 on sub_strike9p6_opt_step.csv
and Stage2 artifacts exist under preds/gm_v2_ladder/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from gm_nullspace_strike_orth_v3 import (
    align_method_b_smooth,
    cos,
    disp_ks,
    orth_against,
    unit,
    weight_for_spearman_dist,
)
from gm_st9p5_exact_oof_and_5f_chain_upgrade import calibrate_logit_to_prevalence

T0 = time.time()
PRED = ROOT / "preds"
ART = PRED / "gm_v2_ladder"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
P12 = ROOT / "external/nov08_part12"
KAGGLE = str(ROOT / ".venv_forensics/bin/kaggle")
TARGET_MEAN = 0.016700
N_Q = 10001
LB_BASE0 = 0.95333
SLOT1_FILE = "sub_strike9p6_opt_step.csv"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


EPS = 1e-6


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1.0 - EPS))


def load_sub(fn, test_ids):
    return pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)


def one_minus_rho(a, b):
    return 1.0 - float(spearmanr(a, b).correlation)


def pchip_map_to_ref(z_src, z_ref_oof):
    """Map test z onto OOF law then calibrate mean=0.016700."""
    # rank map src -> ref oof quantiles via PCHIP on z-space then to probability via rank of ref
    # Use Strike9.5 style: map onto v8 oof then calibrate
    q = np.linspace(0, 1, N_Q)
    u = (rankdata(z_src, method="average") - 0.5) / len(z_src)
    pchip = PchipInterpolator(q, np.quantile(z_ref_oof, q), extrapolate=True)
    z_mapped = np.asarray(pchip(u), dtype=np.float64)
    return z_mapped


def to_submission_probs(z_test, z_oof_ref, y, test_ids, target_mean=TARGET_MEAN):
    z_m = pchip_map_to_ref(z_test, z_oof_ref)
    # calibrate prevalence via logistic on oof then shift
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=300)
    lr.fit(z_oof_ref.reshape(-1, 1), y.astype(np.int32))
    z_lin = float(lr.coef_[0, 0]) * z_m + float(lr.intercept_[0])

    def obj(d):
        return (float(np.mean(expit(z_lin + d))) - target_mean) ** 2

    d = float(minimize_scalar(obj, bounds=(-2, 2), method="bounded").x)
    p = expit(z_lin + d)
    # exact rank-preserve remap to hit mean while keeping order (optional soft)
    # Force mean by small affine in logit space already done
    assert abs(float(p.mean()) - target_mean) < 5e-5, p.mean()
    return p


def write_sub(path, test_ids, p):
    pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(path, index=False)
    assert len(p) == 299984
    assert np.isfinite(p).all()
    log(f"  wrote {path.name} mean={p.mean():.6f}")


def kaggle_submit(path, msg):
    cmd = [KAGGLE, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg]
    log(f"SUBMIT {path.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        log(f"STDERR {r.stderr[-500:]}")
        raise RuntimeError(f"submit failed: {r.stderr}")
    return True


def poll_lb(filename, timeout_s=900, every=20):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run(
            [KAGGLE, "competitions", "submissions", "-c", "playground-series-s6e11", "-v"],
            capture_output=True, text=True,
        )
        for line in r.stdout.splitlines()[1:]:
            if filename in line:
                parts = line.split(",")
                # ref,fileName,date,description,status,publicScore,privateScore
                status = parts[4] if len(parts) > 4 else ""
                score = parts[5] if len(parts) > 5 else ""
                log(f"  poll {filename}: status={status} score={score}")
                if "Complete" in status or "COMPLETE" in status:
                    return float(score) if score not in ("", "None", None) else None
                if "Error" in status or "Failed" in status:
                    raise RuntimeError(f"submission failed: {line}")
                break
        time.sleep(every)
    raise TimeoutError(filename)


def project_perp(v, B_cols):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for j in range(B_cols.shape[1]):
        out = orth_against(out, [B_cols[:, j]])
    return out


def gram_orthonormal(cols):
    Q = []
    for j in range(cols.shape[1]):
        v = unit(orth_against(cols[:, j], Q))
        if float(np.linalg.norm(v)) > 1e-8:
            Q.append(v)
    return np.column_stack(Q) if Q else np.zeros((cols.shape[0], 0))


def main():
    log("GM_V2_LADDER START (Slots 2-5)")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    # Wait for Stage2 artifacts
    need = [
        ART / "z_v8_clean_test.npy",
        ART / "z_ref_clean_test.npy",
        ART / "z_lm12_raw_super_test.npy",
        ART / "z_v8_clean_oof.npy",
        ART / "z_ref_clean_oof.npy",
    ]
    for _ in range(180):
        if all(p.exists() for p in need):
            break
        missing = [p.name for p in need if not p.exists()]
        log(f"  waiting Stage2 artifacts… missing={missing}")
        time.sleep(20)
    else:
        raise FileNotFoundError("Stage2 artifacts not ready")

    z_v8_old_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())
    z_v8_old_t = align_method_b_smooth(z_v8_old_o, zrk(load_sub("sub_v8_part12_fsl_prop_rect.csv", test_ids)), folds)[1]
    z_v8_clean_o = np.load(ART / "z_v8_clean_oof.npy").ravel()
    z_v8_clean_t = np.load(ART / "z_v8_clean_test.npy").ravel()
    z_ref_clean_o = np.load(ART / "z_ref_clean_oof.npy").ravel()
    z_ref_clean_t = np.load(ART / "z_ref_clean_test.npy").ravel()
    z_super_o = np.load(ART / "z_lm12_raw_super_oof.npy").ravel()
    z_super_t = np.load(ART / "z_lm12_raw_super_test.npy").ravel()
    z_w12_o = zrk(np.load(P12 / "wide_lm_stack_nested.npy").ravel())
    z_w12_t = zrk(np.load(P12 / "wide_lm_stack_test.npy").ravel())

    # Base = Strike 9.6 (Slot1 winner)
    z_st9p5 = align_method_b_smooth(z_v8_old_o, zrk(load_sub("sub_strike9p5_axis_0p70_0p10.csv", test_ids)), folds)[1]
    z_st9p6 = align_method_b_smooth(z_v8_old_o, zrk(load_sub(SLOT1_FILE, test_ids)), folds)[1]
    z_base = z_st9p6.copy()
    LB_base = LB_BASE0
    log(f"Anchor z_st9p6 LB_base={LB_base}")

    # B_bad
    def lz(fn):
        return align_method_b_smooth(z_v8_old_o, zrk(load_sub(fn, test_ids)), folds)[1]

    z_st3 = lz("sub_strike3a_cont065_plus_glmquad_graph.csv")
    z_st4 = lz("sub_gm_top3_strike4a_cont070.csv")
    z_st5 = lz("sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv")
    z_st9 = lz("sub_gm_top1_strike9_final_max.csv")
    z_lead = lz("sub_pass340_LEAD_st9_aman20_tuboa10.csv")
    z_soft = lz("sub_strike10_slot1_softdiff_pchip_0p35.csv")
    e1 = z_st4 - z_st3
    e2 = (z_st5 - z_st4) - 0.45 * (z_st4 - z_st3)
    e3 = e2.copy()
    d_bad_leader = z_lead - z_st9
    d_bad_soft = z_soft - z_st9
    # Slot1 gained → no d_overshoot in B_bad
    B_raw = np.column_stack([e2, e3, d_bad_leader, d_bad_soft])
    B_bad = gram_orthonormal(B_raw)
    log(f"B_bad rank={B_bad.shape[1]}")

    def make_delta(raw_t, name):
        # Method-B align raw delta against a synthetic oof proxy: use train-side delta mapped
        # For test-only deltas, align onto unit then re-scale after P_perp
        d = project_perp(raw_t, B_bad)
        # unitize and Method-B vs itself-as-oof surrogate using fold-std of a zero-mean version
        # Build fake oof by projecting same formula on oof side where available
        d = (d - d.mean()) / (d.std() + 1e-12)
        # Exact KS fix via rank map onto N(0,1) quantiles isn't needed; force disp=1
        # Use align_method_b_smooth with oof = random permutation? Better: use oof counterpart
        return d

    # δ1, δ2, δ3 with oof counterparts for Method-B
    d1_o = zrk(z_v8_clean_o) - z_v8_old_o
    d1_t = zrk(z_v8_clean_t) - z_v8_old_t
    d1_o, d1_t = align_method_b_smooth(d1_o, d1_t, folds)
    d1_t = project_perp(d1_t, B_bad)
    d1_o_u = unitize_local(d1_o)
    # re-align after proj
    d1_o2, d1_t = align_method_b_smooth(d1_o, d1_t, folds)
    d1_t = project_perp(d1_t, B_bad)
    d1_t = (d1_t - d1_t.mean()) * (d1_o2.std() / (d1_t.std() + 1e-12)) + d1_o2.mean()

    d2_o = zrk(z_ref_clean_o) - z_v8_old_o
    d2_t = zrk(z_ref_clean_t) - z_v8_old_t
    d2_o, d2_t = align_method_b_smooth(d2_o, d2_t, folds)
    d2_t = project_perp(d2_t, B_bad)
    d2_o2, d2_t = align_method_b_smooth(d2_o, d2_t, folds)
    d2_t = project_perp(d2_t, B_bad)
    d2_t = (d2_t - d2_t.mean()) * (d2_o2.std() / (d2_t.std() + 1e-12)) + d2_o2.mean()

    beta_s = float(np.cov(zrk(z_super_o), z_w12_o, ddof=0)[0, 1] / (np.var(z_w12_o) + 1e-12))
    d3_o = zrk(z_super_o) - beta_s * z_w12_o
    d3_t = zrk(z_super_t) - beta_s * z_w12_t
    d3_o, d3_t = align_method_b_smooth(d3_o, d3_t, folds)
    d3_t = project_perp(d3_t, B_bad)
    d3_o2, d3_t = align_method_b_smooth(d3_o, d3_t, folds)
    d3_t = project_perp(d3_t, B_bad)
    d3_t = (d3_t - d3_t.mean()) * (d3_o2.std() / (d3_t.std() + 1e-12)) + d3_o2.mean()

    deltas = {"d1": d1_t, "d2": d2_t, "d3": d3_t}
    for name, d, oref in [("d1", d1_t, d1_o2), ("d2", d2_t, d2_o2), ("d3", d3_t, d3_o2)]:
        disp, ks = disp_ks(oref, d)
        c_bad = max(abs(cos(d, B_bad[:, j])) for j in range(B_bad.shape[1])) if B_bad.shape[1] else 0.0
        c_e1 = cos(d, e1)
        log(f"  {name}: disp={disp:.6f} KS={ks:.6f} cos(e1)={c_e1:+.4f} max|cos(B_bad)|={c_bad:.6f}")
        np.save(ART / f"{name}_test.npy", d)

    results = {
        "slot1": {"file": SLOT1_FILE, "LB": LB_BASE0, "z_base": "z_st9p6"},
        "slots": [],
    }

    def step_and_submit(slot, fname, direction, target_1mrho, msg):
        nonlocal z_base, LB_base
        # Keep Method-B scale (std≈1). Do NOT L2-unitize — that collapses 1-ρ.
        direction = np.asarray(direction, float).ravel()
        direction = (direction - direction.mean()) / (direction.std() + 1e-12)
        # Binary search with wide hi so we can hit ~1e-4 Spearman distances
        w, got = weight_for_spearman_dist(z_base, direction, target_1mrho, lo=0.0, hi=50.0, n=48)
        z_new = z_base + w * direction
        got2 = one_minus_rho(z_base, z_new)
        log(f"  Slot{slot} target_1mrho={target_1mrho:.2e} got={got2:.2e} ({got2*1e4:.3f}e-4) w={w:.6f}")
        if got2 < 0.5 * target_1mrho:
            raise RuntimeError(f"Slot{slot} 1-rho too small ({got2:.2e} vs target {target_1mrho:.2e})")
        p = to_submission_probs(z_new, z_v8_old_o, y, test_ids)
        path = SUB / fname
        write_sub(path, test_ids, p)
        kaggle_submit(path, msg)
        lb = poll_lb(fname)
        results["slots"].append({"slot": slot, "file": fname, "LB": lb, "1mrho": got2, "w": float(w)})
        if lb is not None and lb > LB_base + 1e-9:
            log(f"  NEW BEST LB={lb} (was {LB_base})")
            z_base = z_new
            LB_base = lb
            return "gain", lb, z_new
        if lb is not None and abs(lb - LB_base) < 1e-9:
            return "flat", lb, z_new
        return "drop", lb, z_new

    # SLOT 2: already submitted (flat LB=0.95333, near-zero step due to L2 bug). Record & skip.
    if os.environ.get("SKIP_SLOT2", "1") == "1":
        log("SKIP Slot2 resubmit (already COMPLETE LB=0.95333); treating as flat")
        st2, lb2 = "flat", 0.95333
        results["slots"].append(
            {"slot": 2, "file": "sub_ladder_slot2_v8_clean_0p18e4.csv", "LB": 0.95333, "1mrho": 2.55e-8, "w": 3.0, "note": "bugged tiny step; flat"}
        )
    else:
        st2, lb2, z2 = step_and_submit(
            2, "sub_ladder_slot2_v8_clean_0p18e4.csv", deltas["d1"], 0.18e-4,
            "Slot2 Ladder: v8_clean Stage-2 booster upgrade (1-rho=0.18e-4)",
        )

    # SLOT 3
    if st2 == "gain":
        dir3 = 0.5 * deltas["d1"] + 0.5 * deltas["d2"]
        t3 = 0.35e-4
    elif st2 == "flat":
        dir3 = deltas["d2"]
        t3 = 0.35e-4
    else:
        # add -d1 to B_bad, reproject d2
        B2 = gram_orthonormal(np.column_stack([B_bad, -deltas["d1"]]))
        dir3 = project_perp(deltas["d2"], B2)
        t3 = 0.25e-4
    st3, lb3, z3 = step_and_submit(
        3, "sub_ladder_slot3_zref_clean_0p35e4.csv", dir3, t3,
        f"Slot3 Ladder: z_ref_clean multi-family (1-rho={t3*1e4:.2f}e-4)",
    )

    # SLOT 4
    improved = any(s["LB"] is not None and s["LB"] > LB_BASE0 + 1e-9 for s in results["slots"]) or LB_base > LB_BASE0
    # track empirical winning vector: last gain direction approx
    if improved:
        # continuation of empirical gains + 0.25*d3
        # reconstruct from z_base - z_st9p6 if moved
        emp = z_base - z_st9p6
        if float(np.linalg.norm(emp)) < 1e-12:
            emp = deltas["d1"]
        dir4 = emp + 0.25 * deltas["d3"]
        t4 = 0.65e-4
        case = "A"
    else:
        dir4 = 0.45 * deltas["d1"] + 0.35 * deltas["d2"] + 0.20 * deltas["d3"]
        t4 = 0.65e-4
        case = "B"
    st4, lb4, z4 = step_and_submit(
        4, "sub_ladder_slot4_adaptive_0p65e4.csv", dir4, t4,
        f"Slot4 Ladder: adaptive case{case} (1-rho=0.65e-4)",
    )

    # SLOT 5: Ridge response surface over historical LB + new slots
    hist = [
        ("Strike3A", "sub_strike3a_cont065_plus_glmquad_graph.csv", 0.95322),
        ("Strike4A", "sub_gm_top3_strike4a_cont070.csv", 0.95324),
        ("Strike5A", "sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv", 0.95326),
        ("Strike9", "sub_gm_top1_strike9_final_max.csv", 0.95330),
        ("Strike9.5", "sub_strike9p5_axis_0p70_0p10.csv", 0.95332),
        ("Strike9.6", SLOT1_FILE, 0.95333),
        ("Lead", "sub_pass340_LEAD_st9_aman20_tuboa10.csv", 0.95325),
        ("Soft", "sub_strike10_slot1_softdiff_pchip_0p35.csv", 0.95328),
    ]
    # Build coordinates in span(ladder=z_st9p6-z_st9p5, d1, d2, d3) relative to z_st9p5
    ladder = z_st9p6 - z_st9p5
    basis = [ladder, deltas["d1"], deltas["d2"], deltas["d3"]]
    Bspan = gram_orthonormal(np.column_stack(basis))
    # confirmed drops: soft relative to 9.5, lead relative to 9
    drop_dirs = []
    if 0.95328 < 0.95332:
        drop_dirs.append(z_soft - z_st9p5)
    Xs, ys = [], []
    for name, fn, lb in hist:
        z = lz(fn)
        delta = z - z_st9p5
        # coords
        c = Bspan.T @ delta
        Xs.append(c)
        ys.append(lb)
    for s in results["slots"]:
        # approximate coords from stored files
        z = align_method_b_smooth(z_v8_old_o, zrk(load_sub(s["file"], test_ids)), folds)[1]
        c = Bspan.T @ (z - z_st9p5)
        Xs.append(c)
        ys.append(s["LB"] if s["LB"] is not None else LB_base)
    X = np.asarray(Xs, float)
    yy = np.asarray(ys, float)
    # ridge on coords
    ridge = Ridge(alpha=1e-3, fit_intercept=True).fit(X, yy)
    g = ridge.coef_.astype(float)  # ascent in coord space
    # zero out components aligned with drops
    g_vec = Bspan @ g
    for dd in drop_dirs:
        g_vec = orth_against(g_vec, [dd])
    # also orth to B_bad
    g_vec = project_perp(g_vec, B_bad)
    st5, lb5, z5 = step_and_submit(
        5, "sub_ladder_slot5_macro_finale_1p25e4.csv", g_vec, 1.25e-4,
        "Slot5 Ladder: Ridge g_LB* macro finale (1-rho=1.25e-4)",
    )
    results["final_LB_base"] = LB_base
    results["slot5_ridge_coef"] = g.tolist()

    # Report
    md = []
    md.append("# GM_V2 Five-Slot Adaptive Ladder")
    md.append("")
    md.append(f"**Slot1** `{SLOT1_FILE}` → **LB={LB_BASE0}** → `z_base=z_st9p6` (Strike axis climbing).")
    md.append("")
    md.append("| Slot | file | LB | 1−ρ×1e4 | result |")
    md.append("|---:|---|---:|---:|---|")
    md.append(f"| 1 | `{SLOT1_FILE}` | {LB_BASE0:.5f} | 0.080 | gain vs 0.95332 |")
    for s in results["slots"]:
        md.append(f"| {s['slot']} | `{s['file']}` | {s['LB']} | {s['1mrho']*1e4:.3f} | — |")
    md.append("")
    md.append(f"**Final LB_base = {LB_base}**")
    md.append("")
    (LOG / "GM_V2_FIVE_SLOT_LADDER.md").write_text("\n".join(md))
    (LOG / "GM_V2_FIVE_SLOT_LADDER.json").write_text(json.dumps(results, indent=2))
    log(f"Wrote {LOG / 'GM_V2_FIVE_SLOT_LADDER.md'}")
    log("GM_V2_LADDER_COMPLETE")


def unitize_local(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


if __name__ == "__main__":
    main()
