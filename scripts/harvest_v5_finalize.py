#!/usr/bin/env python3
"""Fast finalize P_super_v5_harvest from verified Step1 signed B_xgb_llr + optional p4. STRICT NO_SUBMIT."""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/workspace/s6e11")
sys.path.insert(0, str(ROOT / "scripts"))
from paradigm_stack_delta_eval import (
    fast_auc, to_z_ranklogit, score_candidate_vs_v4p_hc, test1_orthogonal_alpha_probe,
)

LOG = ROOT / "logs/overnight/harvest_v5_finalize.log"
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
STATE = ROOT / "logs/overnight/harvest_v5_state.json"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def rk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)

def nested_signed_blend(y, v4p, cand, folds, alphas_grid=None):
    if alphas_grid is None:
        alphas_grid = (-0.20, -0.15, -0.10, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.10, 0.15)
    z = to_z_ranklogit(v4p)
    zc = to_z_ranklogit(cand)
    oof = np.zeros(len(y))
    als = []
    for a, b in folds:
        zc_s_a = (zc[a] - zc[a].mean()) / (zc[a].std() + 1e-12)
        best_a, best_sc = 0.0, float(fast_auc(y[a], v4p[a]))
        for al in alphas_grid:
            sc = float(fast_auc(y[a], z[a] + al * zc_s_a))
            if sc > best_sc:
                best_sc, best_a = sc, al
        zc_s_b = (zc[b] - zc[a].mean()) / (zc[a].std() + 1e-12)
        oof[b] = z[b] + best_a * zc_s_b
        als.append(best_a)
    return rk(oof), float(np.mean(als)), als

def apply_alpha_test(v4p_t, cand_t, mean_alpha, v4p_o=None, cand_o=None):
    z = to_z_ranklogit(v4p_t)
    zc = to_z_ranklogit(cand_t)
    # standardize using OOF stats if provided
    if cand_o is not None:
        zc_o = to_z_ranklogit(cand_o)
        mu, sd = float(zc_o.mean()), float(zc_o.std() + 1e-12)
    else:
        mu, sd = float(zc.mean()), float(zc.std() + 1e-12)
    zc_s = (zc - mu) / sd
    return rk(z + mean_alpha * zc_s)

def bootstrap_delta(y, base, cand, n=400, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(float(fast_auc(y[idx], cand[idx]) - fast_auc(y[idx], base[idx])))
    return float(np.mean(d)), float(np.percentile(d, 5)), float(np.percentile(d, 95))

def main():
    t0 = time.time()
    LOG.write_text("")
    log("=== HARVEST V5 FINALIZE (STRICT NO_SUBMIT) ===")
    bank = np.load(PRED / "v4p_bank_cache.npz", allow_pickle=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    v4p_o = bank["v4p_hc_oof"].astype(np.float64)
    v4p_t = bank["v4p_hc_test"].astype(np.float64)
    base = float(fast_auc(y, v4p_o))
    a0, b0 = folds[0]

    # Step1 candidates
    cands = {
        "B_xgb_llr": (np.load(PRED/"paradigm_B/xgb_llr_oof.npy").ravel(),
                      np.load(PRED/"paradigm_B/xgb_llr_test.npy").ravel()),
        "A_tabr": (np.load(PRED/"paradigm_A/tabr_oof.npy").ravel(),
                   np.load(PRED/"paradigm_A/tabr_test.npy").ravel()),
        "B_llm_solo": (np.load(PRED/"paradigm_B/llm_solo_oof.npy").ravel(),
                       np.load(PRED/"paradigm_B/llm_solo_test.npy").ravel()),
        "A_combo": (np.load(PRED/"paradigm_A/combo_oof.npy").ravel(),
                    np.load(PRED/"paradigm_A/combo_test.npy").ravel()),
    }
    rows = []
    for name, (o, t) in cands.items():
        t1 = score_candidate_vs_v4p_hc(y, v4p_o, o, folds)
        ks = float(ks_2samp(rk(o), rk(t)).statistic)
        drho = abs(float(spearmanr(rk(o), rk(v4p_o)).statistic) - float(spearmanr(rk(t), rk(v4p_t)).statistic))
        gate = (t1["delta_auc"] > 0) and (ks < 0.008) and (drho < 0.0020)
        log(f"STEP1 {name}: Test1Δ={t1['delta_auc']:+.3e} ᾱ={t1['stack_mean_alpha']:+.4f} KS={ks:.4f} |Δρ|={drho:.5f} gate={gate}")
        rows.append({"name": name, "delta": float(t1["delta_auc"]), "alpha": float(t1["stack_mean_alpha"]),
                     "ks": ks, "drho": drho, "gate": gate, "solo": float(t1["cand_solo_auc"])})

    # Best single: B_xgb_llr (only gate=True typically)
    o_llr, t_llr = cands["B_xgb_llr"]
    t1_llr = score_candidate_vs_v4p_hc(y, v4p_o, o_llr, folds)
    # Use official Test1 OOF stack scores
    s1_oof = np.asarray(t1_llr["stack_oof_scores"], dtype=np.float64)
    # If stack scores are decision/logit-like, convert via rk for submission scale
    s1_oof_rk = rk(s1_oof)
    mean_al = float(t1_llr["stack_mean_alpha"])
    # Also nested signed blend
    s1_signed, al_signed, al_list = nested_signed_blend(y, v4p_o, o_llr, folds)
    s1_te_signed = apply_alpha_test(v4p_t, t_llr, al_signed, v4p_o, o_llr)
    # Prefer higher raw AUC between Test1 stack and signed
    auc_t1 = float(fast_auc(y, s1_oof_rk))
    auc_sig = float(fast_auc(y, s1_signed))
    log(f"STEP1 Test1-stack AUC={auc_t1:.7f} Δraw={auc_t1-base:+.3e} Test1Δ={t1_llr['delta_auc']:+.3e}")
    log(f"STEP1 signed-blend AUC={auc_sig:.7f} Δraw={auc_sig-base:+.3e} mean_α={al_signed:+.4f}")
    if auc_sig >= auc_t1:
        s1_oof_f, s1_te_f, s1_method, s1_delta_metric = s1_signed, s1_te_signed, "signed", auc_sig - base
        s1_test1 = float(score_candidate_vs_v4p_hc(y, v4p_o, s1_signed, folds)["delta_auc"])
    else:
        # rebuild test from fold alphas of Test1 (use mean_al on z)
        s1_te_f = apply_alpha_test(v4p_t, t_llr, mean_al, v4p_o, o_llr)
        s1_oof_f, s1_method, s1_delta_metric = s1_oof_rk, "test1_stack", auc_t1 - base
        s1_test1 = float(t1_llr["delta_auc"])
    log(f"STEP1 chosen={s1_method} rawΔ={s1_delta_metric:+.3e} Test1Δ={s1_test1:+.3e}")

    # Step2 joint F0 + load residual xgb
    levers = ["P4_L4.3","P4_L4.2","P4_L4.5","P3_L3.3","P3_L3.4","P3_L3.5","P4_L4.4"]
    z_champ = bank["z_champ_oof"].astype(np.float64)
    skf2 = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(b0)), y[b0]))
    from sklearn.linear_model import LogisticRegression
    cols = []
    f0_rows = []
    for k in levers:
        d = np.load(PRED/f"matrix_v2/{k}_fold0.npz")["delta_fold0"].astype(np.float64)
        cols.append(rk(d))
        cand = z_champ[b0] + d
        t1 = test1_orthogonal_alpha_probe(y[b0], v4p_o[b0], cand, skf2)
        f0_rows.append({"key": k, "delta": float(t1["delta_auc"]), "solo": float(fast_auc(y[b0], cand))})
        log(f"  F0 {k}: Test1Δ={t1['delta_auc']:+.3e} solo={float(fast_auc(y[b0],cand)):.6f}")
    Xf = np.column_stack(cols)
    oof_f0 = np.zeros(len(b0))
    for sa, sb in skf2:
        m = LogisticRegression(C=0.05, solver="lbfgs", max_iter=400).fit(
            np.column_stack([z_champ[b0][sa], Xf[sa]]), y[b0][sa])
        oof_f0[sb] = m.decision_function(np.column_stack([z_champ[b0][sb], Xf[sb]]))
    t1j = test1_orthogonal_alpha_probe(y[b0], v4p_o[b0], oof_f0, skf2)
    log(f"JOINT F0 nested_LR7 Test1Δ={t1j['delta_auc']:+.3e}")

    p4_o = np.load(PRED/"matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel()
    p4_t = np.load(PRED/"matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel()
    t1x = score_candidate_vs_v4p_hc(y, v4p_o, p4_o, folds)
    log(f"STEP2 p4_xgb Test1Δ={t1x['delta_auc']:+.3e} solo={t1x['cand_solo_auc']:.6f}")

    # Final: start from s1, try add p4 with nested signed
    p_cur = s1_oof_f.copy()
    # sequential add p4
    p_with_p4, al_p4, _ = nested_signed_blend(y, p_cur, p4_o, folds)  # treat p_cur as base ranks... need v4p-like
    # Better: blend in z-space from v4p with both cands greedily
    # cand list: llr then p4
    z = to_z_ranklogit(v4p_o).copy()
    oof = np.zeros(len(y))
    # multi: for each fold pick (a_llr, a_p4) on coarse grid
    grid = (-0.15, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10)
    te_z = to_z_ranklogit(v4p_t).copy()
    z_llr_o = to_z_ranklogit(o_llr); z_llr_t = to_z_ranklogit(t_llr)
    z_p4_o = to_z_ranklogit(p4_o); z_p4_t = to_z_ranklogit(p4_t)
    al_pairs = []
    for a, b in folds:
        best = (0.0, 0.0); best_sc = float(fast_auc(y[a], v4p_o[a]))
        s_llr = (z_llr_o[a] - z_llr_o[a].mean()) / (z_llr_o[a].std() + 1e-12)
        s_p4 = (z_p4_o[a] - z_p4_o[a].mean()) / (z_p4_o[a].std() + 1e-12)
        zb = to_z_ranklogit(v4p_o[a])
        for a1 in grid:
            for a2 in grid:
                sc = float(fast_auc(y[a], zb + a1 * s_llr + a2 * s_p4))
                if sc > best_sc:
                    best_sc = sc; best = (a1, a2)
        s_llr_b = (z_llr_o[b] - z_llr_o[a].mean()) / (z_llr_o[a].std() + 1e-12)
        s_p4_b = (z_p4_o[b] - z_p4_o[a].mean()) / (z_p4_o[a].std() + 1e-12)
        oof[b] = to_z_ranklogit(v4p_o[b]) + best[0] * s_llr_b + best[1] * s_p4_b
        al_pairs.append(best)
    p5_o = rk(oof)
    mean_a1 = float(np.mean([p[0] for p in al_pairs]))
    mean_a2 = float(np.mean([p[1] for p in al_pairs]))
    s_llr_t = (z_llr_t - z_llr_o.mean()) / (z_llr_o.std() + 1e-12)
    s_p4_t = (z_p4_t - z_p4_o.mean()) / (z_p4_o.std() + 1e-12)
    p5_t = rk(to_z_ranklogit(v4p_t) + mean_a1 * s_llr_t + mean_a2 * s_p4_t)
    t15 = score_candidate_vs_v4p_hc(y, v4p_o, p5_o, folds)
    auc5 = float(fast_auc(y, p5_o))
    log(f"FINAL dual-α (llr={mean_a1:+.3f}, p4={mean_a2:+.3f}) AUC={auc5:.7f} rawΔ={auc5-base:+.3e} Test1Δ={t15['delta_auc']:+.3e}")

    # If dual worse than s1 alone, keep s1
    if auc5 < float(fast_auc(y, s1_oof_f)) - 1e-9:
        log("FINAL fallback to Step1-only (dual did not improve raw AUC)")
        p5_o, p5_t = s1_oof_f, s1_te_f
        t15 = score_candidate_vs_v4p_hc(y, v4p_o, p5_o, folds)
        auc5 = float(fast_auc(y, p5_o))
        mean_a2 = 0.0
        method = "s1_only_" + s1_method
        w_report = {"B_xgb_llr_alpha": al_signed if s1_method == "signed" else mean_al}
    else:
        method = "dual_llr_p4"
        w_report = {"B_xgb_llr_alpha": mean_a1, "p4_xgb_alpha": mean_a2}

    np.save(PRED / "P_super_v5_harvest_oof.npy", p5_o.astype(np.float64))
    np.save(PRED / "P_super_v5_harvest_test.npy", p5_t.astype(np.float64))
    # ensure p4 files exist (already)
    SUB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_ids, "is_fraud": p5_t}).to_csv(SUB / "sub_v5_harvest_pure.csv", index=False)

    rho_ev4 = None; best_a = 0.0
    ev4_path = SUB / "sub_ev4.csv"
    if ev4_path.exists():
        ev4 = pd.read_csv(ev4_path).set_index("id").loc[test_ids, "is_fraud"].to_numpy(np.float64)
        z_ev4 = to_z_ranklogit(ev4)
        best_a, best_sc = 0.0, base
        for a in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            sc = float(fast_auc(y, (1 - a) * rk(v4p_o) + a * rk(p5_o)))
            if sc > best_sc:
                best_sc, best_a = sc, a
        p_blend = rk((1 - best_a) * z_ev4 + best_a * to_z_ranklogit(p5_t))
        pd.DataFrame({"id": test_ids, "is_fraud": p_blend}).to_csv(SUB / "sub_v5_harvest_on_ev4.csv", index=False)
        rho_ev4 = float(spearmanr(p_blend, ev4).statistic)
        log(f"on_ev4 α={best_a} ρ={rho_ev4:.5f}")

    seed_list = []
    for sd in (42, 7, 2024):
        fl = list(StratifiedKFold(5, shuffle=True, random_state=sd).split(np.zeros(len(y)), y))
        seed_list.append(float(test1_orthogonal_alpha_probe(y, v4p_o, p5_o, fl)["delta_auc"]))
    dmean, dlo, dhi = bootstrap_delta(y, v4p_o, p5_o)

    log("=" * 90)
    log("FINAL HARVEST TABLE")
    log(f"{'Model':42s} | {'OOF AUC':10s} | {'Δ / Test1':14s}")
    log(f"{'v4p_hc baseline':42s} | {base:.7f} | +0.000e+00")
    log(f"{'Step1 B_xgb_llr ('+s1_method+')':42s} | {float(fast_auc(y,s1_oof_f)):.7f} | {s1_test1:+.3e}")
    log(f"{'Step2 P4 residual CUDA XGB':42s} | {float(fast_auc(y,p4_o)):.7f} | {t1x['delta_auc']:+.3e}")
    log(f"{'P_super_v5_harvest ('+method+')':42s} | {auc5:.7f} | {t15['delta_auc']:+.3e}")
    log(f"weights: {w_report}")
    log(f"JOINT F0 matrix_v2 LR7 Test1Δ={t1j['delta_auc']:+.3e}")
    log(f"3-seed mean Test1 Δ = {float(np.mean(seed_list)):+.3e}  {seed_list}")
    log(f"90% boot ΔAUC mean={dmean:+.3e} CI=[{dlo:+.3e},{dhi:+.3e}]")
    if rho_ev4 is not None:
        log(f"Spearman ρ(on_ev4, sub_ev4)={rho_ev4:.5f}")
    log(f"STRICT NO_SUBMIT wall={time.time()-t0:.1f}s")

    st = {
        "status": "DONE_FINALIZE",
        "baseline": base,
        "step1": {"method": s1_method, "test1_delta": s1_test1, "raw_delta": s1_delta_metric, "rows": rows},
        "step2": {"test1_delta": float(t1x["delta_auc"]), "joint_f0": float(t1j["delta_auc"]), "f0_rows": f0_rows},
        "final": {"method": method, "auc": auc5, "test1_delta": float(t15["delta_auc"]),
                  "raw_delta": auc5 - base, "weights": w_report,
                  "seed_deltas": seed_list, "seed_mean": float(np.mean(seed_list)),
                  "boot_delta": [dmean, dlo, dhi], "rho_ev4": rho_ev4, "on_ev4_alpha": best_a},
        "wall_s": time.time() - t0,
    }
    STATE.write_text(json.dumps(st, indent=2, default=float))
    # sha256 for sync verify
    import hashlib
    files = [
        PRED/"P_super_v5_harvest_oof.npy", PRED/"P_super_v5_harvest_test.npy",
        PRED/"matrix_v2/harvest_p4_cuda_xgb_oof.npy", PRED/"matrix_v2/harvest_p4_cuda_xgb_test.npy",
        PRED/"paradigm_B/xgb_llr_oof.npy", PRED/"paradigm_B/xgb_llr_test.npy",
        SUB/"sub_v5_harvest_pure.csv", SUB/"sub_v5_harvest_on_ev4.csv",
    ]
    hashes = {}
    for f in files:
        if f.exists():
            h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            hashes[str(f.relative_to(ROOT))] = {"sha256_16": h, "bytes": f.stat().st_size}
            log(f"HASH {f.name} {f.stat().st_size} {h}")
    (ROOT/"logs/overnight/harvest_v5_hashes.json").write_text(json.dumps(hashes, indent=2))
    log("=== FINALIZE DONE ===")

if __name__ == "__main__":
    main()
