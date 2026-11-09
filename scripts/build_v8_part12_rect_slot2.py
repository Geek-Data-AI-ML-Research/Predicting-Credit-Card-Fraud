#!/usr/bin/env python3
"""
Build P_super_v8_part12_rect (Part 12 wide_lm_stack 0.9525651 OOF + our LB-proven 0.95247 KS-rectified
proprietary GPU & 3-epoch GPT-2 residuals -> 0.9525953 Nested OOF / 0.9526675 Mean-W OOF)
and submit Slot 2 (sub_v8_part12_fsl_prop_rect.csv) to Kaggle, then poll live Public LB_2.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
sys.path.insert(0, str(ROOT / "scripts"))
from paradigm_stack_delta_eval import fast_auc

PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
P12 = ROOT / "external/nov08_part12"
W11 = ROOT / "external/nov07_wide_recon"
EPS = 1e-6


def rk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)


def rk01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def zrk(x: np.ndarray) -> np.ndarray:
    return logit(np.clip(rk(x), EPS, 1.0 - EPS))


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def qmap_to_oof(x_test: np.ndarray, x_oof: np.ndarray) -> np.ndarray:
    """Fast exact empirical order-statistic quantile map from test onto OOF."""
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    o = np.sort(np.asarray(x_oof, dtype=np.float64).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def build_rect_basis(specs, z_base_o: np.ndarray, z_base_t: np.ndarray):
    R_o, R_t, metas = [], [], []
    for name, xo, xt in specs:
        xt_rect = qmap_to_oof(xt, xo)
        zo, zt = zrk(xo), zrk(xt_rect)
        beta = float(np.dot(zo, z_base_o) / (np.dot(z_base_o, z_base_o) + 1e-12))
        ro = zo - beta * z_base_o
        rt = zt - beta * z_base_t
        sd = float(ro.std() + 1e-12)
        ro = ro / sd
        rt = rt / sd
        rt = qmap_to_oof(rt, ro)
        ro = ro - float(ro.mean())
        rt = rt - float(rt.mean())
        s = float(ro.std() + 1e-12)
        ro = ro / s
        rt = rt / s
        ks_raw = float(ks_2samp(xo, xt).statistic)
        ks_r = float(ks_2samp(ro, rt).statistic)
        metas.append({
            "name": name,
            "beta_vs_w12": beta,
            "ks_raw": ks_raw,
            "ks_resid": ks_r,
            "sd_o": float(ro.std()),
            "sd_t": float(rt.std()),
        })
        R_o.append(ro)
        R_t.append(rt)
        print(
            f"  {name:16s} β_w12={beta:+.4f} KS_raw={ks_raw:.5f} -> KS_resid={ks_r:.6f} "
            f"sd_o={ro.std():.4f} sd_t={rt.std():.4f}",
            flush=True,
        )
    return np.column_stack(R_o), np.column_stack(R_t), metas


def main():
    t0 = time.time()
    print("=" * 76)
    print("BUILD P_SUPER_V8_PART12_RECT & EXECUTE SLOT 2 SUBMISSION")
    print("=" * 76)

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    # 1. Load Part 12 wide_lm_stack (201 members incl. full 5-fold megayak target-first LM)
    w12_o = np.load(P12 / "wide_lm_stack_nested.npy").ravel().astype(np.float64)
    w12_t = np.load(P12 / "wide_lm_stack_test.npy").ravel().astype(np.float64)
    p9_t = np.load(P12 / "part9_test.npy").ravel().astype(np.float64)
    maly_sub = pd.read_csv(W11 / "malyshev_fsl/submission.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    ev4_sub = pd.read_csv(SUB / "sub_ev4_live10_reverse_ev2_plus_hc_ascent.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    slot1_sub = pd.read_csv(SUB / "sub_v5_rect_on_ev4_1x.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    z_w12_o, z_w12_t = zrk(w12_o), zrk(w12_t)
    base12_auc = float(fast_auc(y, z_w12_o))
    print(f"[Baseline] Part 12 wide_lm_stack_nested (201m) 5F OOF AUC = {base12_auc:.7f}")

    # 2. Build KS-rectified orthogonal basis of our proprietary discoveries vs Part 12
    specs = [
        ("v4p_hc", np.load(PRED / "P_super_v4p_hc_oof.npy").ravel(), np.load(PRED / "P_super_v4p_hc_test.npy").ravel()),
        ("B_xgb_llr", np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel(), np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel()),
        ("P4_L4.3", np.load(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.3_pure5f_test.npy").ravel()),
        ("P4_L4.2", np.load(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.2_pure5f_test.npy").ravel()),
        ("P4_L4.5", np.load(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.5_pure5f_test.npy").ravel()),
        ("P4_L4.3_ext", np.load(PRED / "matrix_v2/P4_L4.3_ext_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.3_ext_test.npy").ravel()),
        ("gpt2_bidir", np.load(PRED / "gpt2_3ep/gpt2_bidir_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/gpt2_bidir_test.npy").ravel()),
        ("xgb_gpt2_3ep", np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_test.npy").ravel()),
    ]
    print("\n[Step 1] Component-Level Quantile Rectification vs Part 12 wide_lm_stack:")
    R_o, R_t, metas = build_rect_basis(specs, z_w12_o, z_w12_t)
    names = [m["name"] for m in metas]
    max_ks = max(m["ks_resid"] for m in metas)
    print(f"  Max component KS_resid = {max_ks:.6f} (Requirement: < 0.0005)")
    if max_ks >= 0.0005:
        raise SystemExit(f"FAIL max_ks={max_ks} >= 0.0005")

    # 3. 5-Fold Nested Coordinate Ascent on top of Part 12 wide_lm_stack
    M = len(names)
    grid = (-0.04, -0.03, -0.022, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.022, 0.03, 0.04)
    oof_v8_z = np.zeros(len(y), dtype=np.float64)
    fold_w, fold_d = [], []
    for fi, (a, b) in enumerate(folds):
        w = np.zeros(M, dtype=np.float64)
        best = float(fast_auc(y[a], z_w12_o[a]))
        for _ in range(3):
            for j in range(M):
                bw = w[j]
                for cw in grid:
                    wt = w.copy()
                    wt[j] = cw
                    sc = float(fast_auc(y[a], z_w12_o[a] + R_o[a] @ wt))
                    if sc > best + 1e-9:
                        best, bw = sc, cw
                w[j] = bw
        oof_v8_z[b] = z_w12_o[b] + R_o[b] @ w
        d = float(fast_auc(y[b], oof_v8_z[b]) - fast_auc(y[b], z_w12_o[b]))
        fold_w.append(w)
        fold_d.append(d)
        print(f"  Fold {fi}: val_ΔAUC = {d:+.3e} | w = {dict(zip(names, [round(float(x), 4) for x in w]))}", flush=True)

    mean_w = np.mean(fold_w, axis=0)
    v8_nested_auc = float(fast_auc(y, oof_v8_z))
    v8_meanw_auc = float(fast_auc(y, z_w12_o + R_o @ mean_w))
    wins = int(sum(d > 0 for d in fold_d))
    print(f"\n==> P_super_v8_part12_rect Nested 5F OOF AUC = {v8_nested_auc:.7f} (Δ vs Part12 = {v8_nested_auc - base12_auc:+.3e}, wins = {wins}/5)")
    print(f"==> P_super_v8_part12_rect Mean-W 5F OOF AUC = {v8_meanw_auc:.7f} (Δ vs Part12 = {v8_meanw_auc - base12_auc:+.3e})")
    print(f"==> Mean weights: {dict(zip(names, [round(float(x), 4) for x in mean_w]))}")

    # Exact ECDF quantile-map of the composite test residual onto the OOF residual
    dz_oof = R_o @ mean_w
    dz_test = qmap_to_oof(R_t @ mean_w, dz_oof)
    ks_dz = float(ks_2samp(dz_oof, dz_test).statistic)
    print(f"==> Composite residual Δz KS(oof, test) = {ks_dz:.6f}, std(dz_test) = {dz_test.std():.4f}")

    p8_oof = rk(oof_v8_z)
    p8_test = rk(z_w12_t + dz_test)
    np.save(PRED / "P_super_v8_part12_rect_oof.npy", p8_oof)
    np.save(PRED / "P_super_v8_part12_rect_test.npy", p8_test)

    # 4. Build Part 12 Baseline & Upgraded v8 Submissions
    # Exact Part 12 public 0.95298 baseline:
    p12_exact_sub = rk(0.70 * rk01(w12_t) + 0.30 * rk01(p9_t))
    # Logit-rank calibrated Part 12 backbone (68/32 logit-rank of w12_t and part9_t):
    z_p12_fsl = zrk(0.68 * zrk(w12_t) + 0.32 * zrk(p9_t))
    z_p12_exact = zrk(p12_exact_sub)

    # Orthogonalize our composite proprietary test prediction (z_w12_t + dz_test) against z_p12_fsl
    # Note: dz_test is already orthogonal to z_w12_t; orthogonalizing against z_p12_fsl removes any tiny
    # overlap with part9_t as well!
    beta_fsl = float(np.dot(dz_test, z_p12_fsl) / (np.dot(z_p12_fsl, z_p12_fsl) + 1e-12))
    dz_orth_fsl = dz_test - beta_fsl * z_p12_fsl
    dz_orth_fsl = qmap_to_oof(dz_orth_fsl, dz_oof)

    # Candidate 1 (SLOT 2 PRIMARY): sub_v8_part12_fsl_prop_rect.csv
    # Upgraded Part 12 logit-rank backbone + 1.0x KS-rectified proprietary GPU/3ep-GPT2 residual
    cand_v8_fsl_1x = rk(z_p12_fsl + 1.0 * dz_orth_fsl)

    # Candidate 2 (STAGED): sub_v8_part12_exact7030_prop_rect.csv
    # Exact Part 12 70/30 rank backbone + 1.0x KS-rectified proprietary residual
    cand_v8_7030_1x = rk(z_p12_exact + 1.0 * dz_test)

    # Candidate 3 (STAGED): sub_v8_pure_oof_cv095260.csv
    # 100% OOF-verified P_super_v8_part12_rect_test (no test-only part9 blend)
    cand_v8_pure = p8_test

    out_files = [
        ("sub_v8_part12_fsl_prop_rect.csv", cand_v8_fsl_1x, v8_nested_auc),
        ("sub_v8_part12_exact7030_prop_rect.csv", cand_v8_7030_1x, v8_nested_auc),
        ("sub_v8_pure_oof_cv095260.csv", cand_v8_pure, v8_nested_auc),
    ]
    print("\n" + "=" * 76)
    print("V8 PART 12 CANDIDATE DIAGNOSTICS")
    print("=" * 76)
    print(f"{'File':42s} {'OOF_AUC':>9s} {'ρ_p12_298':>10s} {'ρ_maly292':>10s} {'ρ_slot1':>10s} {'maxKS':>8s} sha16")
    meta_rows = []
    for fname, arr, oof_sc in out_files:
        fp = SUB / fname
        pd.DataFrame({"id": test_ids, "is_fraud": arr}).to_csv(fp, index=False)
        r_p12 = float(spearmanr(arr, p12_exact_sub).statistic)
        r_maly = float(spearmanr(arr, maly_sub).statistic)
        r_s1 = float(spearmanr(arr, slot1_sub).statistic)
        h = sha16(fp)
        print(f"{fname:42s} {oof_sc:9.7f} {r_p12:+10.6f} {r_maly:+10.6f} {r_s1:+10.6f} {max_ks:8.6f} {h}")
        meta_rows.append({
            "file": fname,
            "oof_auc": oof_sc,
            "rho_part12_095298": r_p12,
            "rho_maly_095292": r_maly,
            "rho_slot1_095247": r_s1,
            "max_ks_resid": max_ks,
            "sha16": h,
        })

    # 5. Submit Slot 2 (sub_v8_part12_fsl_prop_rect.csv)
    slot2_path = SUB / "sub_v8_part12_fsl_prop_rect.csv"
    msg = (
        f"Slot2 v8 Part12(0.95298)+KS-rect GPU/3epGPT2 residual "
        f"(nested OOF {v8_nested_auc:.6f}, meanW {v8_meanw_auc:.6f}, sha={sha16(slot2_path)})"
    )
    print(f"\n[Step 3] SUBMIT Slot 2: {slot2_path.name} (sha={sha16(slot2_path)})")
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    kag = str(ROOT / ".venv_forensics/bin/kaggle")
    cmd = [kag, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(slot2_path), "-m", msg]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("  submit stdout:", r.stdout.strip()[-500:])
    print("  submit stderr:", r.stderr.strip()[-500:])
    if r.returncode != 0:
        raise SystemExit(f"Submit failed rc={r.returncode}")

    # 6. Poll Live Public LB_2
    print("\n[Step 4] Polling Kaggle Public LB for Slot 2 ...")
    lb2 = None
    for i in range(40):
        time.sleep(20)
        p = subprocess.run(
            [kag, "competitions", "submissions", "-c", "playground-series-s6e11"],
            capture_output=True,
            text=True,
            env=env,
        )
        out = p.stdout
        hit = None
        for line in out.splitlines():
            if "sub_v8_part12_fsl_prop_rect.csv" in line:
                hit = line
                break
        print(f"  poll {i+1}: {hit[:180] if hit else 'waiting...'}")
        if hit and "COMPLETE" in hit.upper() and "PENDING" not in hit.upper():
            nums = [float(x) for x in re.findall(r"0\.95\d{3}", hit)]
            if nums:
                lb2 = nums[-1]
                print(f"  >>> LIVE PUBLIC LB_2 = {lb2:.5f} <<<")
                break

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "base_part12_oof_auc": base12_auc,
        "v8_nested_oof_auc": v8_nested_auc,
        "v8_meanw_oof_auc": v8_meanw_auc,
        "folds_won": wins,
        "fold_deltas": fold_d,
        "mean_weights": dict(zip(names, [round(float(x), 5) for x in mean_w])),
        "max_ks_resid": max_ks,
        "ks_composite_dz": ks_dz,
        "slot2_file": str(slot2_path),
        "slot2_sha16": sha16(slot2_path),
        "LB_1": 0.95247,
        "LB_2": lb2,
        "candidates": meta_rows,
    }
    (LOG / "v8_part12_slot2.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{summary['elapsed_s']}s] Saved logs/overnight/v8_part12_slot2.json")


if __name__ == "__main__":
    main()
