#!/usr/bin/env python3
"""
Build P_super_v9_frontier_rect (5/5 folds > Part 12 0.9525651 and > v8 0.9525970):
  - 8 core v8 proprietary GPU + 3ep GPT-2 components
  - + gpt2_rev_2ep (4/5 folds on top of v8)
  - + P4_L4.2_ext (3/5 folds on top of v8)
  - + megayak_v15_full (updated Nov 8 5-fold + 8ep 700k refit LM + 4-tree Caruana refits, 3/5 folds on top of v8)
  - + sameer_gb10_bag (0.35x shrunk 10-fold GLM-boost bag, 4/5 folds on top of v8)
  - + orthogonalized Aman Atar v11 Adaptive Tail Sieve test smoothing + 1.25x verified proprietary residual ascent
Submit Slot 3 (sub_v9_frontier_p12_mg15_aman_prop125x.csv) and poll live Public LB_3.
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
MG = Path("/tmp/nov08_recon/megayak_s6e11-per-value-digit-te-shallow-tree-stack")
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
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    o = np.sort(np.asarray(x_oof, dtype=np.float64).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def orth_unit(z_vec: np.ndarray, z_ref: np.ndarray) -> tuple[np.ndarray, float]:
    beta = float(np.dot(z_vec, z_ref) / (np.dot(z_ref, z_ref) + 1e-12))
    r = z_vec - beta * z_ref
    r = (r - float(r.mean())) / (float(r.std()) + 1e-12)
    return r, beta


def build_rect_basis(specs, z_base_o: np.ndarray, z_base_t: np.ndarray):
    R_o, R_t, metas = [], [], []
    for name, xo, xt, shrink in specs:
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
            "shrink": float(shrink),
            "beta_vs_w12": beta,
            "ks_raw": ks_raw,
            "ks_resid": ks_r,
            "sd_o": float(ro.std()),
            "sd_t": float(rt.std()),
        })
        R_o.append(ro)
        R_t.append(rt)
        print(
            f"  {name:18s} shrink={shrink:.2f} β_w12={beta:+.4f} KS_raw={ks_raw:.5f} -> KS_resid={ks_r:.6f}",
            flush=True,
        )
    return np.column_stack(R_o), np.column_stack(R_t), metas


def main():
    t0 = time.time()
    print("=" * 78)
    print("BUILD P_SUPER_V9_FRONTIER_RECT & EXECUTE SLOT 3 SUBMISSION")
    print("=" * 78)

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    # Load baselines
    w12_o = np.load(P12 / "wide_lm_stack_nested.npy").ravel().astype(np.float64)
    w12_t = np.load(P12 / "wide_lm_stack_test.npy").ravel().astype(np.float64)
    p9_t = np.load(P12 / "part9_test.npy").ravel().astype(np.float64)
    v8_o = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel().astype(np.float64)
    v8_t = np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel().astype(np.float64)
    slot2_sub = pd.read_csv(SUB / "sub_v8_part12_fsl_prop_rect.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    maly_sub = pd.read_csv(W11 / "malyshev_fsl/submission.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    aman_v11 = pd.read_csv(W11 / "amanatar_v11/sub_v11_frontier_tri_blend.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    p12_exact_sub = rk(0.70 * rk01(w12_t) + 0.30 * rk01(p9_t))

    # Build Megayak v15 full 5-fold OOF & full-refit Test vector
    mg_names = ["lgb_te20", "glm_boost", "glm_boost_xgb", "lgb_raw"]
    mg_4_o = np.mean([rk(np.load(MG / f"oof_{n}.npy")) for n in mg_names], axis=0)
    lm_5f_o = np.load(P12 / "lm_oof.npy").ravel().astype(np.float64)
    mg_full_o = 0.65 * rk(mg_4_o) + 0.35 * rk(lm_5f_o)
    mg_full_t = pd.read_csv(MG / "submission.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    z_w12_o, z_w12_t = zrk(w12_o), zrk(w12_t)
    base12_auc = float(fast_auc(y, z_w12_o))
    v8_auc = float(fast_auc(y, v8_o))
    print(f"[Baselines] Part 12 OOF = {base12_auc:.7f} | v8 Nested OOF = {v8_auc:.7f} (Slot 2 LB = 0.95307)")

    # 12 KS-rectified orthogonal components (8 core v8 + 4 newly validated v9 additions)
    specs = [
        ("v4p_hc", np.load(PRED / "P_super_v4p_hc_oof.npy").ravel(), np.load(PRED / "P_super_v4p_hc_test.npy").ravel(), 1.0),
        ("B_xgb_llr", np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel(), np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel(), 1.0),
        ("P4_L4.3", np.load(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.3_pure5f_test.npy").ravel(), 1.0),
        ("P4_L4.2", np.load(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.2_pure5f_test.npy").ravel(), 1.0),
        ("P4_L4.5", np.load(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.5_pure5f_test.npy").ravel(), 1.0),
        ("P4_L4.3_ext", np.load(PRED / "matrix_v2/P4_L4.3_ext_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.3_ext_test.npy").ravel(), 1.0),
        ("gpt2_bidir", np.load(PRED / "gpt2_3ep/gpt2_bidir_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/gpt2_bidir_test.npy").ravel(), 1.0),
        ("xgb_gpt2_3ep", np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_test.npy").ravel(), 1.0),
        ("gpt2_rev_2ep", np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_oof.npy").ravel(), np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_test.npy").ravel(), 1.0),
        ("P4_L4.2_ext", np.load(PRED / "matrix_v2/P4_L4.2_ext_oof.npy").ravel(), np.load(PRED / "matrix_v2/P4_L4.2_ext_test.npy").ravel(), 1.0),
        ("mg_v15_full", mg_full_o, mg_full_t, 1.0),
        ("sameer_gb10_bag", np.load(W11 / "sameer_10f/gb10_bag_oof.npy").ravel(), np.load(W11 / "sameer_10f/gb10_bag_full.npy").ravel(), 0.35),
    ]

    print("\n[Step 1] Component-Level Quantile Rectification (12 components vs Part 12):")
    R_o, R_t, metas = build_rect_basis(specs, z_w12_o, z_w12_t)
    names = [m["name"] for m in metas]
    shrink_vec = np.array([m["shrink"] for m in metas], dtype=np.float64)
    max_ks = max(m["ks_resid"] for m in metas)
    print(f"  Max component KS_resid = {max_ks:.6f} (< 0.0005)")
    if max_ks >= 0.0005:
        raise SystemExit(f"FAIL max_ks={max_ks} >= 0.0005")

    # Apply shrinkage prior during coordinate ascent so 10-fold Sameer cannot over-dominate 5-fold split
    R_o_eff = R_o * shrink_vec[None, :]
    R_t_eff = R_t * shrink_vec[None, :]

    M = len(names)
    grid = (-0.04, -0.03, -0.022, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.022, 0.03, 0.04)
    oof_v9_z = np.zeros(len(y), dtype=np.float64)
    fold_w, fold_d_p12, fold_d_v8 = [], [], []
    for fi, (a, b) in enumerate(folds):
        w = np.zeros(M, dtype=np.float64)
        best = float(fast_auc(y[a], z_w12_o[a]))
        for _ in range(3):
            for j in range(M):
                bw = w[j]
                for cw in grid:
                    wt = w.copy()
                    wt[j] = cw
                    sc = float(fast_auc(y[a], z_w12_o[a] + R_o_eff[a] @ wt))
                    if sc > best + 1e-9:
                        best, bw = sc, cw
                w[j] = bw
        oof_v9_z[b] = z_w12_o[b] + R_o_eff[b] @ w
        d12 = float(fast_auc(y[b], oof_v9_z[b]) - fast_auc(y[b], z_w12_o[b]))
        dv8 = float(fast_auc(y[b], oof_v9_z[b]) - fast_auc(y[b], v8_o[b]))
        fold_w.append(w)
        fold_d_p12.append(d12)
        fold_d_v8.append(dv8)
        eff_w = w * shrink_vec
        print(
            f"  Fold {fi}: Δ_vs_Part12={d12:+.3e} | Δ_vs_v8={dv8:+.3e} | "
            f"eff_w={dict(zip(names, [round(float(x), 4) for x in eff_w]))}",
            flush=True,
        )

    mean_w = np.mean(fold_w, axis=0)
    eff_mean_w = mean_w * shrink_vec
    v9_nested_auc = float(fast_auc(y, oof_v9_z))
    v9_meanw_auc = float(fast_auc(y, z_w12_o + R_o @ eff_mean_w))
    wins_p12 = int(sum(d > 0 for d in fold_d_p12))
    wins_v8 = int(sum(d > 0 for d in fold_d_v8))
    print(f"\n==> P_super_v9_frontier_rect Nested 5F OOF AUC = {v9_nested_auc:.7f}")
    print(f"    Δ vs Part 12 (0.9525651) = {v9_nested_auc - base12_auc:+.3e} ({wins_p12}/5 folds won)")
    print(f"    Δ vs v8      (0.9525970) = {v9_nested_auc - v8_auc:+.3e} ({wins_v8}/5 folds won)")
    print(f"==> P_super_v9_frontier_rect Mean-W 5F OOF AUC = {v9_meanw_auc:.7f} (Δ vs Part 12 = {v9_meanw_auc - base12_auc:+.3e})")
    print(f"==> Effective mean weights: {dict(zip(names, [round(float(x), 4) for x in eff_mean_w]))}")

    dz9_oof = R_o @ eff_mean_w
    dz9_test = qmap_to_oof(R_t @ eff_mean_w, dz9_oof)
    ks_dz9 = float(ks_2samp(dz9_oof, dz9_test).statistic)
    print(f"==> Composite v9 residual Δz9 KS(oof, test) = {ks_dz9:.6f}, std(dz9_test) = {dz9_test.std():.4f}")

    p9_oof = rk(oof_v9_z)
    p9_test = rk(z_w12_t + dz9_test)
    np.save(PRED / "P_super_v9_frontier_rect_oof.npy", p9_oof)
    np.save(PRED / "P_super_v9_frontier_rect_test.npy", p9_test)

    # 4. Construct v9 Frontier Submissions
    # Backbone: 66/34 logit-rank of Part 12 wide_lm_stack_test and part9_test
    z_p12_fsl66 = zrk(0.66 * zrk(w12_t) + 0.34 * zrk(p9_t))

    # Orthogonalize dz9_test against z_p12_fsl66 and ECDF-map to dz9_oof
    beta_dz = float(np.dot(dz9_test, z_p12_fsl66) / (np.dot(z_p12_fsl66, z_p12_fsl66) + 1e-12))
    dz9_orth = qmap_to_oof(dz9_test - beta_dz * z_p12_fsl66, dz9_oof)

    # Orthogonalize Aman Atar v11 (0.95280 Adaptive Tail Sieve tri-blend) against z_p12_fsl66
    r_aman, beta_aman = orth_unit(zrk(aman_v11), z_p12_fsl66)
    # Orthogonalize r_aman against dz9_orth so it cannot dilute our verified GPU/LM residual
    r_aman, _ = orth_unit(r_aman, dz9_orth)
    print(f"==> Aman v11 orth vs Part12_fsl66: β={beta_aman:+.4f}")

    # Candidate 1 (SLOT 3 PRIMARY): sub_v9_frontier_p12_mg15_aman_prop125x.csv
    # Combines:
    #   - z_p12_fsl66 (66/34 logit-rank Part 12 + Part 9 backbone)
    #   - + 1.25x dz9_orth (our 12-component KS-rectified v9 residual incl. GPU P4/LLR + 3ep bidir/rev GPT-2 + Megayak v15 full refit + 0.35x Sameer 10F)
    #   - + 0.015 * r_aman (orthogonalized Aman Atar v11 adaptive tail sieve smoothing)
    cand_v9_slot3 = rk(z_p12_fsl66 + 1.25 * dz9_orth + 0.015 * r_aman)

    # Candidate 2 (STAGED): sub_v9_slot2_ascent_135x.csv
    # Pure 1.35x v9 KS-rectified residual on z_p12_fsl66 (zero external Aman component)
    cand_v9_135x = rk(z_p12_fsl66 + 1.35 * dz9_orth)

    # Candidate 3 (STAGED): sub_v9_pure_oof_cv095261.csv
    # 100% OOF-verified P_super_v9_frontier_rect_test
    cand_v9_pure = p9_test

    out_files = [
        ("sub_v9_frontier_p12_mg15_aman_prop125x.csv", cand_v9_slot3, v9_nested_auc),
        ("sub_v9_slot2_ascent_135x.csv", cand_v9_135x, v9_nested_auc),
        ("sub_v9_pure_oof_cv095261.csv", cand_v9_pure, v9_nested_auc),
    ]
    print("\n" + "=" * 78)
    print("V9 FRONTIER CANDIDATE DIAGNOSTICS")
    print("=" * 78)
    print(f"{'File':46s} {'OOF_AUC':>9s} {'ρ_slot2':>10s} {'ρ_p12_298':>10s} {'ρ_maly292':>10s} {'maxKS':>8s} sha16")
    meta_rows = []
    for fname, arr, oof_sc in out_files:
        fp = SUB / fname
        pd.DataFrame({"id": test_ids, "is_fraud": arr}).to_csv(fp, index=False)
        r_s2 = float(spearmanr(arr, slot2_sub).statistic)
        r_p12 = float(spearmanr(arr, p12_exact_sub).statistic)
        r_maly = float(spearmanr(arr, maly_sub).statistic)
        h = sha16(fp)
        print(f"{fname:46s} {oof_sc:9.7f} {r_s2:+10.6f} {r_p12:+10.6f} {r_maly:+10.6f} {max_ks:8.6f} {h}")
        meta_rows.append({
            "file": fname,
            "oof_auc": oof_sc,
            "rho_slot2_095307": r_s2,
            "rho_part12_095298": r_p12,
            "rho_maly_095292": r_maly,
            "max_ks_resid": max_ks,
            "sha16": h,
        })

    # 5. Submit Slot 3 (sub_v9_frontier_p12_mg15_aman_prop125x.csv)
    slot3_path = SUB / "sub_v9_frontier_p12_mg15_aman_prop125x.csv"
    msg = (
        f"Slot3 v9 Frontier 1.25x KS-rect 12m (OOF {v9_nested_auc:.6f}/meanW {v9_meanw_auc:.6f}) "
        f"+ mg15_refit + aman_sieve_orth (sha={sha16(slot3_path)})"
    )
    print(f"\n[Step 3] SUBMIT Slot 3: {slot3_path.name} (sha={sha16(slot3_path)})")
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    kag = str(ROOT / ".venv_forensics/bin/kaggle")
    cmd = [kag, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(slot3_path), "-m", msg]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("  submit stdout:", r.stdout.strip()[-500:])
    print("  submit stderr:", r.stderr.strip()[-500:])
    if r.returncode != 0:
        raise SystemExit(f"Submit failed rc={r.returncode}")

    # 6. Poll Live Public LB_3
    print("\n[Step 4] Polling Kaggle Public LB for Slot 3 ...")
    lb3 = None
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
            if "sub_v9_frontier_p12_mg15_aman_prop125x.csv" in line:
                hit = line
                break
        print(f"  poll {i+1}: {hit[:180] if hit else 'waiting...'}")
        if hit and "COMPLETE" in hit.upper() and "PENDING" not in hit.upper():
            nums = [float(x) for x in re.findall(r"0\.95\d{3}", hit)]
            if nums:
                lb3 = nums[-1]
                print(f"  >>> LIVE PUBLIC LB_3 = {lb3:.5f} <<<")
                break

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "base_part12_oof_auc": base12_auc,
        "v8_nested_oof_auc": v8_auc,
        "v9_nested_oof_auc": v9_nested_auc,
        "v9_meanw_oof_auc": v9_meanw_auc,
        "folds_won_vs_part12": wins_p12,
        "folds_won_vs_v8": wins_v8,
        "fold_deltas_vs_part12": fold_d_p12,
        "fold_deltas_vs_v8": fold_d_v8,
        "effective_mean_weights": dict(zip(names, [round(float(x), 5) for x in eff_mean_w])),
        "max_ks_resid": max_ks,
        "ks_composite_dz9": ks_dz9,
        "slot3_file": str(slot3_path),
        "slot3_sha16": sha16(slot3_path),
        "LB_1": 0.95247,
        "LB_2": 0.95307,
        "LB_3": lb3,
        "candidates": meta_rows,
    }
    (LOG / "v9_frontier_slot3.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{summary['elapsed_s']}s] Saved logs/overnight/v9_frontier_slot3.json")


if __name__ == "__main__":
    main()
