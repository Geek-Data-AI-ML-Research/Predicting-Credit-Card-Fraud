#!/usr/bin/env python3
"""
DS-OPTIMIZED Slot-1 protocol (local CPU):
  1) Component-level quantile rectification → KS-rect residuals
  2) wide_stack / maly synergy
  3) Candidate matrix + submit Slot 1 (1.0x on ev4) + poll LB
  4) Stage Slot 2/3 branches (NO submit Slot 2)
STRICT: only Slot 1 is submitted.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paradigm_stack_delta_eval import fast_auc, to_z_ranklogit

PRED, SUB, LOG, EXT = ROOT / "preds", ROOT / "submissions", ROOT / "logs/overnight", ROOT / "external/nov07_wide_recon"
EPS = 1e-6


def rk(x):
    x = np.asarray(x, float).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)


def rk01(x):
    x = np.asarray(x, float).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def zrk(x):
    return logit(np.clip(rk01(x), EPS, 1 - EPS))


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def qmap_to_oof(x_test, x_oof):
    """Map test onto exact empirical quantile distribution of OOF (rank-preserving).

    Replaces sorted test values with mid-rank OOF order statistics (fast index,
    not np.quantile over m≈3e5 queries which is ~200s/call on this volume).
    """
    x_test = np.asarray(x_test, float).ravel()
    o = np.sort(np.asarray(x_oof, float).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def ecdf_map(v, ref):
    ref_s = np.sort(np.asarray(ref, float).ravel())
    n = len(ref_s)
    r = rk(v)
    q = np.clip(r * (n - 1), 0.0, n - 1.0)
    lo = np.floor(q).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    w = q - lo
    return (1.0 - w) * ref_s[lo] + w * ref_s[hi]


def build_rect_basis(comps, z_base_o, z_base_t):
    """Return R_o, R_t, metas with component-level quantile rectification."""
    R_o, R_t, metas = [], [], []
    for name, xo, xt, w in comps:
        xt_rect = qmap_to_oof(xt, xo)
        zo, zt = zrk(xo), zrk(xt_rect)
        beta = float(np.dot(zo, z_base_o) / (np.dot(z_base_o, z_base_o) + 1e-12))
        ro = zo - beta * z_base_o
        rt = zt - beta * z_base_t
        sd = float(ro.std() + 1e-12)
        ro = ro / sd
        rt = rt / sd
        # Exact residual quantile-map onto unit-std OOF residual law.
        # Joint (same) scale after qmap preserves KS; never std(rt) alone.
        rt = qmap_to_oof(rt, ro)
        ro = ro - float(ro.mean())
        rt = rt - float(rt.mean())
        s = float(ro.std() + 1e-12)
        ro = ro / s
        rt = rt / s
        ks_raw = float(ks_2samp(xo, xt).statistic)
        ks_score = float(ks_2samp(xo, xt_rect).statistic)
        ks_r = float(ks_2samp(ro, rt).statistic)
        metas.append(dict(
            name=name, w=float(w), beta=beta,
            ks_raw=ks_raw, ks_after_qmap=ks_score, ks_resid=ks_r,
            var_o=float(ro.var()), var_t=float(rt.var()),
            sd_o=float(ro.std()), sd_t=float(rt.std()),
        ))
        R_o.append(ro)
        R_t.append(rt)
        print(f"  {name:16s} w={w:+.4f} β={beta:+.4f} "
              f"KS_raw={ks_raw:.5f}→qmap={ks_score:.5f} KS_resid={ks_r:.6f} "
              f"sd_o={ro.std():.4f} sd_t={rt.std():.4f}", flush=True)
    return np.column_stack(R_o), np.column_stack(R_t), metas


def main():
    t0 = time.time()
    print("=" * 72)
    print("DS SLOT-1 PROTOCOL — KS RECT + WIDE/MALY SYNERGY + SUBMIT")
    print("=" * 72)

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    v4p_o = np.load(PRED / "P_super_v4p_hc_oof.npy").ravel().astype(np.float64)
    v4p_t = np.load(PRED / "P_super_v4p_hc_test.npy").ravel().astype(np.float64)
    v5_o = np.load(PRED / "P_super_v5_harvest_oof.npy").ravel().astype(np.float64)
    v5_t = np.load(PRED / "P_super_v5_harvest_test.npy").ravel().astype(np.float64)
    z_base_o, z_base_t = zrk(v4p_o), zrk(v4p_t)  # harvest was built vs v4p_hc
    # Actually p4_pure used z_base = to_z_ranklogit(v4p). Confirm v5 OOF.
    print(f"v4p_hc OOF={fast_auc(y,v4p_o):.7f}  v5_harvest OOF={fast_auc(y,v5_o):.7f}")

    # ---- Step 1: components ----
    state = json.loads((LOG / "harvest_v5_state.json").read_text())
    mw = state["mean_weights"]
    # map names to files
    comps4 = [
        ("B_xgb_llr", np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel(),
         np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel(), mw["B_xgb_llr"]),
        ("P4_L4.3", np.load(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy").ravel(),
         np.load(PRED / "matrix_v2/P4_L4.3_pure5f_test.npy").ravel(), mw["P4_L4.3"]),
        ("P4_L4.2", np.load(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy").ravel(),
         np.load(PRED / "matrix_v2/P4_L4.2_pure5f_test.npy").ravel(), mw["P4_L4.2"]),
        ("P4_L4.5", np.load(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy").ravel(),
         np.load(PRED / "matrix_v2/P4_L4.5_pure5f_test.npy").ravel(), mw["P4_L4.5"]),
    ]
    # L4.3_ext: w_opt≈+0.024 from audit → 0.5x shrink
    w_ext = 0.5 * 0.024
    comps5 = comps4 + [
        ("P4_L4.3_ext", np.load(PRED / "matrix_v2/P4_L4.3_ext_oof.npy").ravel(),
         np.load(PRED / "matrix_v2/P4_L4.3_ext_test.npy").ravel(), w_ext),
    ]

    print("\n[Step1] Component-level quantile rectification")
    R4_o, R4_t, meta4 = build_rect_basis(comps4, z_base_o, z_base_t)
    R_ext_o, R_ext_t, meta_ext = build_rect_basis([comps5[-1]], z_base_o, z_base_t)
    R5_o = np.column_stack([R4_o, R_ext_o])
    R5_t = np.column_stack([R4_t, R_ext_t])
    meta5 = meta4 + meta_ext
    w4 = np.array([m["w"] for m in meta4])
    w5 = np.array([m["w"] for m in meta5])

    # Reconstruct OOF stacks with rectified basis (OOF unchanged path; test rectified)
    # For OOF we use non-rect train residuals (already matched); for honesty recompute oof from R4_o
    oof_v5_rect = z_base_o + R4_o @ w4
    te_v5_rect = z_base_t + R4_t @ w4
    oof_v5p5_rect = z_base_o + R5_o @ w5
    te_v5p5_rect = z_base_t + R5_t @ w5

    p_v5_rect_o = rk(oof_v5_rect)
    p_v5_rect_t = rk(te_v5_rect)
    p_v5p5_rect_o = rk(oof_v5p5_rect)
    p_v5p5_rect_t = rk(te_v5p5_rect)

    auc_v5r = float(fast_auc(y, p_v5_rect_o))
    auc_v5p5r = float(fast_auc(y, p_v5p5_rect_o))
    print(f"  v5_rect OOF={auc_v5r:.7f} (orig v5={fast_auc(y,v5_o):.7f})")
    print(f"  v5p5_rect (4+0.5*ext) OOF={auc_v5p5r:.7f}")
    max_ks4 = max(m["ks_resid"] for m in meta4)
    max_ks5 = max(m["ks_resid"] for m in meta5)
    print(f"  max KS_resid 4-comp={max_ks4:.6f}  5-comp={max_ks5:.6f}")
    if max_ks4 >= 0.0005:
        raise SystemExit(f"FAIL max KS_resid={max_ks4} >= 0.0005")

    np.save(PRED / "P_super_v5_rect_oof.npy", p_v5_rect_o)
    np.save(PRED / "P_super_v5_rect_test.npy", p_v5_rect_t)
    np.save(PRED / "P_super_v5p5_rect_oof.npy", p_v5p5_rect_o)
    np.save(PRED / "P_super_v5p5_rect_test.npy", p_v5p5_rect_t)

    # Inject onto ev4
    ev4 = pd.read_csv(SUB / "sub_ev4_live10_reverse_ev2_plus_hc_ascent.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_ev4 = zrk(ev4)
    # Use Δz = rect residual sum (already orth vs v4p); inject onto ev4
    dz4 = te_v5_rect - z_base_t  # = R4_t @ w4
    dz4 = ecdf_map(dz4, oof_v5_rect - z_base_o)
    dz5 = ecdf_map(te_v5p5_rect - z_base_t, oof_v5p5_rect - z_base_o)

    sub_old = pd.read_csv(SUB / "sub_v5_harvest_on_ev4.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    cand_1x = rk(z_ev4 + 1.0 * dz4)
    cand_half = rk(z_ev4 + 0.5 * dz4)
    cand_1p5 = rk(z_ev4 + 1.5 * dz5)  # Slot2A staged: 1.5x with ext
    cand_1x_p5 = rk(z_ev4 + 1.0 * dz5)

    # rank difference vs original
    # Spearman distance and mean abs rank delta
    r_old, r_new = rk01(sub_old), rk01(cand_1x)
    mean_abs_rank_delta = float(np.mean(np.abs(r_old - r_new)))
    rho_old_new = float(spearmanr(sub_old, cand_1x).statistic)
    print(f"  vs orig sub_v5_harvest_on_ev4: ρ={rho_old_new:.8f} mean|Δrk|={mean_abs_rank_delta:.3e}")

    p_rect_1x = SUB / "sub_v5_rect_on_ev4_1x.csv"
    p_rect_half = SUB / "sub_v5_rect_on_ev4_half.csv"
    p_rect_1p5 = SUB / "sub_v5p5_rect_on_ev4_1p5x.csv"
    p_rect_1x_p5 = SUB / "sub_v5p5_rect_on_ev4_1x.csv"
    for path, arr in [(p_rect_1x, cand_1x), (p_rect_half, cand_half),
                      (p_rect_1p5, cand_1p5), (p_rect_1x_p5, cand_1x_p5)]:
        pd.DataFrame({"id": test_ids, "is_fraud": arr}).to_csv(path, index=False)
        print(f"  wrote {path.name} sha={sha16(path)}")

    use_rect = mean_abs_rank_delta > 1e-6
    slot1_path = p_rect_1x if use_rect else (SUB / "sub_v5_harvest_on_ev4.csv")
    print(f"  Slot1 file decision: {'RECT' if use_rect else 'ORIG'} → {slot1_path.name} sha={sha16(slot1_path)}")

    # ---- Step 2: wide_stack synergy ----
    print("\n[Step2] wide_stack_nested + v5_rect synergy")
    wide_o = np.load(EXT / "goodpjw_artifacts/wide_stack_nested.npy").ravel().astype(np.float64)
    wide_t = np.load(EXT / "goodpjw_artifacts/wide_stack_test.npy").ravel().astype(np.float64)
    z_wide_o, z_wide_t = zrk(wide_o), zrk(wide_t)
    base_wide = float(fast_auc(y, wide_o))
    print(f"  wide_stack_nested OOF={base_wide:.7f}")

    # Orth our rect residual components against wide (not v4p) for fair stack on wide
    # Use Δ from R4 rebuilt vs wide baseline
    R4w_o, R4w_t, meta_w = build_rect_basis(
        [(n, xo, xt, w) for (n, xo, xt, w) in comps4], z_wide_o, z_wide_t
    )
    # nested alpha search on top of wide with fixed mean_w from harvest? Better: re-fit small grid
    grid = (-0.08, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
    oof_w = np.zeros(len(y))
    fold_w, fold_d = [], []
    for a, b in folds:
        w = np.zeros(4)
        best = float(fast_auc(y[a], z_wide_o[a]))
        for _ in range(2):
            for j in range(4):
                best_wj = w[j]
                for cw in grid:
                    wt = w.copy(); wt[j] = cw
                    sc = float(fast_auc(y[a], z_wide_o[a] + R4w_o[a] @ wt))
                    if sc > best + 1e-12:
                        best, best_wj = sc, cw
                w[j] = best_wj
        oof_w[b] = z_wide_o[b] + R4w_o[b] @ w
        fold_w.append(w)
        fold_d.append(float(fast_auc(y[b], oof_w[b]) - fast_auc(y[b], wide_o[b])))
    mean_ww = np.mean(fold_w, axis=0)
    auc_wide_v5 = float(fast_auc(y, oof_w))
    wins = sum(d > 0 for d in fold_d)
    print(f"  wide+v5_rect nested OOF={auc_wide_v5:.7f} Δ={auc_wide_v5-base_wide:+.3e} wins={wins}/5 w={dict(zip([c[0] for c in comps4], [round(float(x),4) for x in mean_ww]))}")
    wide_v5_t = rk(z_wide_t + R4w_t @ mean_ww)
    np.save(PRED / "P_wide_v5_rect_oof.npy", rk(oof_w))
    np.save(PRED / "P_wide_v5_rect_test.npy", wide_v5_t)
    p_wide_pure = SUB / "sub_wide_v5_rect_pure.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": wide_v5_t}).to_csv(p_wide_pure, index=False)

    # Maly orth injection — tune alpha so ρ≈0.99965
    print("\n[Step2b] proprietary ⊥ maly (target ρ≈0.99965)")
    maly = pd.read_csv(EXT / "malyshev_fsl/submission.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    z_maly = zrk(maly)
    # best prop test vectors: z_ev4+dz4 or wide_v5
    for tag, z_prop in [("ev4_v5_rect", z_ev4 + dz4), ("wide_v5_rect", zrk(wide_v5_t))]:
        beta = float(np.dot(z_prop, z_maly) / (np.dot(z_maly, z_maly) + 1e-12))
        r = z_prop - beta * z_maly
        r = r / (float(r.std()) + 1e-12)
        r = ecdf_map(r, r)  # noop identity shape
        # tune alpha for ρ≈0.99965
        best_a, best_err, best_pred = 0.0, 1e9, None
        for a in np.linspace(0.0, 0.08, 81):
            pred = rk(z_maly + a * r)
            rho = float(spearmanr(pred, maly).statistic)
            err = abs(rho - 0.99965)
            if err < best_err:
                best_err, best_a, best_pred, best_rho = err, float(a), pred, rho
        print(f"  {tag}: β_maly={beta:+.4f} α*={best_a:.4f} ρ={best_rho:.6f} (target 0.99965)")
        if tag == "ev4_v5_rect":
            p_maly = SUB / "sub_v5_rect_orth_on_maly095292.csv"
            pd.DataFrame({"id": test_ids, "is_fraud": best_pred}).to_csv(p_maly, index=False)
            print(f"  wrote {p_maly.name} sha={sha16(p_maly)} α={best_a}")

    # ---- Step 3: candidate matrix ----
    print("\n" + "=" * 72)
    print("CANDIDATE MATRIX")
    print("=" * 72)
    maly_sub = maly
    rows = []
    cands = [
        ("sub_v5_harvest_on_ev4.csv", SUB / "sub_v5_harvest_on_ev4.csv", fast_auc(y, v5_o), fast_auc(y, v5_o)),
        ("sub_v5_harvest_on_ev4_half.csv", SUB / "sub_v5_harvest_on_ev4_half.csv", fast_auc(y, v5_o), fast_auc(y, v5_o)),
        ("sub_v5_rect_on_ev4_1x.csv", p_rect_1x, auc_v5r, auc_v5r),
        ("sub_v5_rect_on_ev4_half.csv", p_rect_half, auc_v5r, auc_v5r),
        ("sub_v5p5_rect_on_ev4_1x.csv", p_rect_1x_p5, auc_v5p5r, auc_v5p5r),
        ("sub_v5p5_rect_on_ev4_1p5x.csv", p_rect_1p5, auc_v5p5r, auc_v5p5r),
        ("sub_v5_rect_orth_on_maly095292.csv", p_maly, auc_v5r, auc_v5r),
        ("sub_wide_v5_rect_pure.csv", p_wide_pure, auc_wide_v5, auc_wide_v5),
        ("sub_v6_gm_on_maly095292.csv", SUB / "sub_v6_gm_on_maly095292.csv", 0.9525275, 0.9525275),
        ("sub_v6_gm_pure_cv095253.csv", SUB / "sub_v6_gm_pure_cv095253.csv", 0.9525275, 0.9525275),
    ]
    print(f"{'File':42s} {'OOFb':>9s} {'ρ_ev4':>9s} {'ρ_maly':>9s} {'maxKS':>8s} sha16")
    for name, path, oof_b, oof_s in cands:
        arr = pd.read_csv(path).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
        print(f"{name:42s} {oof_b:9.7f} {spearmanr(arr,ev4).statistic:+.6f} "
              f"{spearmanr(arr,maly_sub).statistic:+.6f} {max_ks4:8.5f} {sha16(path)}")
        rows.append(dict(file=name, oof=oof_b, rho_ev4=float(spearmanr(arr, ev4).statistic),
                         rho_maly=float(spearmanr(arr, maly_sub).statistic),
                         max_ks=max_ks4, sha=sha16(path)))

    # ---- Step 3b: SUBMIT Slot 1 ----
    print("\n[Step3] SUBMIT Slot 1:", slot1_path)
    msg = f"Slot1 1.0x KS-rect GPU residual on ev4 ({slot1_path.name} sha={sha16(slot1_path)})"
    cmd = [
        str(ROOT / ".venv_forensics/bin/kaggle"),
        "competitions", "submit", "-c", "playground-series-s6e11",
        "-f", str(slot1_path), "-m", msg,
    ]
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()
                     if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")})}
    # scrub proxies that break kaggle
    for k in list(env):
        if "proxy" in k.lower():
            env.pop(k, None)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("  submit stdout:", r.stdout.strip()[-500:])
    print("  submit stderr:", r.stderr.strip()[-500:])
    if r.returncode != 0:
        # fallback system kaggle
        cmd[0] = "/Users/mz/Library/Python/3.14/bin/kaggle"
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print("  fallback rc", r.returncode, r.stdout[-300:], r.stderr[-300:])

    # Poll
    print("\n[Step3] Polling LB …")
    lb1 = None
    kag = cmd[0]
    for i in range(40):
        time.sleep(20)
        p = subprocess.run(
            [kag, "competitions", "submissions", "-c", "playground-series-s6e11"],
            capture_output=True, text=True, env=env,
        )
        out = p.stdout
        # find our file
        hit = None
        for line in out.splitlines():
            if slot1_path.name in line or "KS-rect" in line or "Slot1" in line:
                hit = line
                break
        if hit is None:
            # take first data row
            lines = [l for l in out.splitlines() if l.strip() and not l.startswith("Warning")]
            hit = "\n".join(lines[:6])
        print(f"  poll {i+1}: {hit[:200] if hit else 'no hit'}")
        # parse score
        import re
        # Parse publicScore from kaggle CLI table rows
        for line in out.splitlines():
            if "sub_v5_rect_on_ev4_1x" in line or (use_rect is False and "sub_v5_harvest_on_ev4.csv" in line and "half" not in line):
                nums = re.findall(r"0\.\d{5}", line)
                status_done = ("COMPLETE" in line.upper()) or ("complete" in line and "pending" not in line.lower())
                if nums and (status_done or len(nums) >= 1):
                    # publicScore usually last 0.95xxx on the row
                    cand_scores = [float(x) for x in nums if x.startswith("0.95")]
                    if cand_scores and (status_done or "PENDING" not in line.upper()):
                        if status_done:
                            lb1 = cand_scores[-1]
                            print(f"  >>> LB_1 = {lb1} from: {line[:180]}")
                            break
        if lb1 is not None:
            break

    # ---- Step 4: stage branches ----
    print("\n[Step4] Adaptive Slot 2/3 staging (NOT submitted)")
    if lb1 is None:
        branch = "UNKNOWN (LB pending) — stage both 2A and 2B"
        staged = [p_rect_1p5, p_maly, p_wide_pure]
    elif lb1 > 0.95243:
        branch = "LB_1 > 0.95243 → Stage 2A=1.5x+ext, 2B=orth_on_maly"
        staged = [p_rect_1p5, p_maly]
    elif abs(lb1 - 0.95243) < 1e-9 or lb1 == 0.95243:
        branch = "LB_1 == 0.95243 → Stage wide_v5_rect OR orth_on_maly"
        staged = [p_wide_pure, p_maly]
    else:
        branch = "LB_1 < 0.95243 → Stage half-step or pure wide"
        staged = [SUB / "sub_v5_harvest_on_ev4_half.csv", p_wide_pure]
    print(f"  branch: {branch}")
    for p in staged:
        print(f"  STAGED (no submit): {p.name} sha={sha16(p)}")

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "meta4": meta4,
        "auc_v5_rect": auc_v5r,
        "auc_v5p5_rect": auc_v5p5r,
        "auc_wide_v5_rect": auc_wide_v5,
        "wide_wins": wins,
        "mean_abs_rank_delta_vs_orig": mean_abs_rank_delta,
        "use_rect_for_slot1": use_rect,
        "slot1": str(slot1_path),
        "slot1_sha": sha16(slot1_path),
        "LB_1": lb1,
        "branch": branch,
        "staged": [str(p) for p in staged],
        "candidates": rows,
    }
    (LOG / "ds_slot1_protocol.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[{summary['elapsed_s']}s] wrote logs/overnight/ds_slot1_protocol.json")
    if lb1 is not None:
        print(f"\n*** LIVE PUBLIC LB_1 = {lb1} ***")
    else:
        print("\n*** LB_1 not yet parsed — check kaggle submissions manually; Slot2 NOT submitted ***")


if __name__ == "__main__":
    main()
