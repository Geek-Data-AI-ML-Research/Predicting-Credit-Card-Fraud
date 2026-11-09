#!/usr/bin/env python3
"""
60s local CPU audit + P_super_v5p5_clean surgical blend (STRICT NO_SUBMIT).
"""
from __future__ import annotations

import hashlib
import json
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
from paradigm_stack_delta_eval import fast_auc, score_candidate_vs_v4p_hc, to_z_ranklogit

PRED, SUB, LOG = ROOT / "preds", ROOT / "submissions", ROOT / "logs/overnight"
V4P_BASE, V5_BASE = 0.9523071, 0.9523401
EPS = 1e-6


def rk(x):
    x = np.asarray(x, float).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)


def zrk(x):
    return logit(np.clip(rk01(x), EPS, 1 - EPS))


def rk01(x):
    x = np.asarray(x, float).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_pair(oof_p, te_p):
    o = np.load(oof_p).astype(np.float64).ravel()
    t = np.load(te_p).astype(np.float64).ravel()
    return o, t


def orth_against(zo, zt, ref_o, ref_t):
    beta = float(np.dot(zo, ref_o) / (np.dot(ref_o, ref_o) + 1e-12))
    ro, rt = zo - beta * ref_o, zt - beta * ref_t
    sd = float(ro.std() + 1e-12)
    return ro / sd, rt / sd, beta


def ecdf_map_to_ref(v, ref):
    ref_s = np.sort(np.asarray(ref, float).ravel())
    n = len(ref_s)
    r = rk(v)
    q = np.clip(r * (n - 1), 0.0, n - 1.0)
    lo = np.floor(q).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    w = q - lo
    return (1.0 - w) * ref_s[lo] + w * ref_s[hi]


def main():
    t0 = time.time()
    print("=" * 72)
    print("60s LOCAL CPU AUDIT + P_super_v5p5_clean (NO_SUBMIT)")
    print("=" * 72)

    # ---- CAMPAIGN_DONE ----
    done = json.loads((LOG / "CAMPAIGN_DONE.json").read_text())
    print(f"\n[CAMPAIGN_DONE] status={done.get('status')} ready={done.get('ready_to_terminate')} finished={done.get('finished_utc')}")

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    v4p_o = np.load(PRED / "P_super_v4p_hc_oof.npy").ravel().astype(np.float64)
    v4p_t = np.load(PRED / "P_super_v4p_hc_test.npy").ravel().astype(np.float64)
    v5_o = np.load(PRED / "P_super_v5_harvest_oof.npy").ravel().astype(np.float64)
    v5_t = np.load(PRED / "P_super_v5_harvest_test.npy").ravel().astype(np.float64)
    print(f"baseline v4p_hc OOF={fast_auc(y,v4p_o):.7f} (claim {V4P_BASE})")
    print(f"baseline v5_harvest OOF={fast_auc(y,v5_o):.7f} (claim {V5_BASE})")

    # ---- 1. Score overnight .npy ----
    cands = [
        ("O_v4p_gpt2_3ep_replace", PRED / "gpt2_3ep/O_v4p_gpt2_3ep_replace_oof.npy", PRED / "gpt2_3ep/O_v4p_gpt2_3ep_replace_test.npy"),
        ("gpt2_3ep", PRED / "gpt2_3ep/gpt2_3ep_oof.npy", PRED / "gpt2_3ep/gpt2_3ep_test.npy"),
        ("gpt2_bidir", PRED / "gpt2_3ep/gpt2_bidir_oof.npy", PRED / "gpt2_3ep/gpt2_bidir_test.npy"),
        ("gb_gpt2_3ep_super(xgb-only)", PRED / "gpt2_3ep/gb_gpt2_3ep_super_oof.npy", PRED / "gpt2_3ep/gb_gpt2_3ep_super_test.npy"),
        ("xgb_gpt2_3ep_d2_d3_bag6", PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy", PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_test.npy"),
        ("P4_L4.2_ext", PRED / "matrix_v2/P4_L4.2_ext_oof.npy", PRED / "matrix_v2/P4_L4.2_ext_test.npy"),
        ("P4_L4.3_ext", PRED / "matrix_v2/P4_L4.3_ext_oof.npy", PRED / "matrix_v2/P4_L4.3_ext_test.npy"),
        ("gpt2_rev_2ep", PRED / "gpt2_3ep/gpt2_rev_2ep_oof.npy", PRED / "gpt2_3ep/gpt2_rev_2ep_test.npy"),
    ]
    print("\n" + "=" * 72)
    print("OVERNIGHT .npy  |  solo_AUC  |  Δvs_v4p  |  Δvs_v5  |  KS(oof,test)  |  win@v5")
    print("=" * 72)
    scored = {}
    for name, op, tp in cands:
        o, t = load_pair(op, tp)
        solo = float(fast_auc(y, o))
        ev4 = score_candidate_vs_v4p_hc(y, v4p_o, o, folds)
        ev5 = score_candidate_vs_v4p_hc(y, v5_o, o, folds)
        ks = float(ks_2samp(rk(o), rk(t)).statistic)
        win5 = sum(1 for d in ev5["fold_deltas"] if d > 0)
        scored[name] = dict(o=o, t=t, solo=solo, d4=ev4["delta_auc"], d5=ev5["delta_auc"],
                            ks=ks, win5=win5, fold_d5=ev5["fold_deltas"], a5=ev5["stack_mean_alpha"])
        print(f"{name:30s}  {solo:.7f}  {ev4['delta_auc']:+.3e}  {ev5['delta_auc']:+.3e}  {ks:.5f}  {win5}/5")

    # ---- CSV Spearman matrix ----
    ev4_path = SUB / "sub_ev4_live10_reverse_ev2_plus_hc_ascent.csv"
    ev4 = pd.read_csv(ev4_path).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    half = pd.read_csv(SUB / "sub_v5_harvest_on_ev4_half.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
    full = pd.read_csv(SUB / "sub_v5_harvest_on_ev4.csv").set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)

    csvs = [
        "sub_v5_harvest_on_ev4_half.csv",
        "sub_v5_harvest_on_ev4.csv",
        "sub_v5_harvest_pure.csv",
        "sub_v6_gm_pure_cv095253.csv",
        "sub_v6_gm_on_maly095292.csv",
        "sub_v6_gm_triad_maly095292_ev4_p4gpu.csv",
        "sub_v6_on_ev4_half.csv",
        "sub_v6_on_ev4_1x.csv",
        "sub_v6_pure.csv",
    ]
    recipes = {
        "sub_v5_harvest_on_ev4_half.csv": "rk(z_ev4 + 0.5*(z_v5 - proj))",
        "sub_v5_harvest_on_ev4.csv": "rk(z_ev4 + 1.0*Δz_v5_orth)",
        "sub_v5_harvest_pure.csv": "rk(P_super_v5_harvest_test)",
        "sub_v6_gm_pure_cv095253.csv": "wide_stack + GPU orth + megayak LM (nested)",
        "sub_v6_gm_on_maly095292.csv": "maly0.95292 + full GPU orth residual",
        "sub_v6_gm_triad_maly095292_ev4_p4gpu.csv": "0.88*maly + 0.12*ev4 + P4/LLR orth",
        "sub_v6_on_ev4_half.csv": "rk(z_ev4 + 0.5*(z_v6camp - z_v5))",
        "sub_v6_on_ev4_1x.csv": "rk(z_ev4 + 1.0*(z_v6camp - z_v5))",
        "sub_v6_pure.csv": "rk(P_super_v6_campaign_test)",
    }
    print("\n" + "=" * 72)
    print("SUBMISSION CSV  |  sha16  |  ρ(ev4)  |  ρ(v5_half)  |  ρ(v5_1x)  |  recipe")
    print("=" * 72)
    for fn in csvs:
        p = SUB / fn
        arr = pd.read_csv(p).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
        print(f"{fn:42s}  {sha16(p)}  {spearmanr(arr,ev4).statistic:+.6f}  "
              f"{spearmanr(arr,half).statistic:+.6f}  {spearmanr(arr,full).statistic:+.6f}  | {recipes.get(fn,'?')}")

    # ---- 2. Greedy v5p5 clean ----
    print("\n" + "=" * 72)
    print("GREEDY KEEP-ONLY-IF-IMPROVES on v5_harvest (gate: Nested↑, ≥4/5 folds, KS<0.003)")
    print("=" * 72)
    z_base_o, z_base_t = zrk(v5_o), zrk(v5_t)
    base_auc = float(fast_auc(y, v5_o))
    assert abs(base_auc - V5_BASE) < 5e-5, (base_auc, V5_BASE)

    # Prefer fixed xgb-only gb_super; also allow O_v4p replace, bidir, L4.2/3_ext
    pool = [
        ("O_v4p_gpt2_3ep_replace", scored["O_v4p_gpt2_3ep_replace"]),
        ("gb_gpt2_3ep_super", scored["gb_gpt2_3ep_super(xgb-only)"]),
        ("gpt2_bidir", scored["gpt2_bidir"]),
        ("P4_L4.2_ext", scored["P4_L4.2_ext"]),
        ("P4_L4.3_ext", scored["P4_L4.3_ext"]),
    ]

    # Precompute ECDF-aligned orthogonal residuals vs z_v5
    basis = []
    for name, sc in pool:
        ks = sc["ks"]
        zo, zt = zrk(sc["o"]), zrk(sc["t"])
        ro, rt, beta = orth_against(zo, zt, z_base_o, z_base_t)
        # ECDF-align test residual to oof residual (kill KS inflation)
        rt_al = ecdf_map_to_ref(rt, ro)
        ks_r = float(ks_2samp(ro, rt_al).statistic)
        basis.append(dict(name=name, ro=ro, rt=rt_al, beta=beta, ks=ks, ks_r=ks_r, solo=sc["solo"]))
        print(f"  basis {name:28s} beta={beta:+.4f} KS_score={ks:.5f} KS_resid={ks_r:.5f} solo={sc['solo']:.7f}")

    grid = (-0.12, -0.08, -0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)

    def nested_auc_and_folds(active_idx, fixed_w=None):
        """5-fold nested coord ascent on active basis; return oof_z, mean_w, fold_deltas vs v5."""
        M = len(active_idx)
        if M == 0:
            return z_base_o.copy(), np.zeros(0), [0.0] * 5, base_auc
        R_o = np.column_stack([basis[j]["ro"] for j in active_idx])
        oof_z = np.zeros(len(y))
        fold_w, fold_d = [], []
        for a, b in folds:
            if fixed_w is not None:
                w = np.asarray(fixed_w, float).copy()
            else:
                w = np.zeros(M)
                best = float(fast_auc(y[a], z_base_o[a]))
                for _ in range(3):
                    for j in range(M):
                        best_wj = w[j]
                        for cw in grid:
                            wt = w.copy(); wt[j] = cw
                            sc = float(fast_auc(y[a], z_base_o[a] + R_o[a] @ wt))
                            if sc > best + 1e-12:
                                best, best_wj = sc, cw
                        w[j] = best_wj
            oof_z[b] = z_base_o[b] + R_o[b] @ w
            fold_w.append(w)
            fold_d.append(float(fast_auc(y[b], oof_z[b]) - fast_auc(y[b], v5_o[b])))
        mean_w = np.mean(fold_w, axis=0)
        return oof_z, mean_w, fold_d, float(fast_auc(y, oof_z))

    # Greedy forward selection with strict gate
    active = []
    cur_auc = base_auc
    rejected = []
    print(f"\n  start AUC={cur_auc:.7f}")
    improved = True
    while improved:
        improved = False
        best_j, best_auc, best_info = None, cur_auc, None
        for j, b in enumerate(basis):
            if j in active:
                continue
            # Gate on ECDF-aligned residual KS (raw score KS may be high for L4.3_ext)
            if b["ks_r"] >= 0.003:
                if (b["name"], f"KS_resid={b['ks_r']:.5f}>=0.003") not in rejected:
                    rejected.append((b["name"], f"KS_resid={b['ks_r']:.5f}>=0.003"))
                continue
            trial = active + [j]
            oof_z, mean_w, fold_d, auc = nested_auc_and_folds(trial)
            wins = sum(1 for d in fold_d if d > 0)
            gate = (auc > cur_auc + 1e-9) and (wins >= 4)
            print(f"  try +{b['name']:28s} → {auc:.7f} (Δ={auc-base_auc:+.3e}) wins={wins}/5 "
                  f"{'KEEP?' if gate else 'reject'} w={dict(zip([basis[i]['name'] for i in trial], [round(float(x),4) for x in mean_w]))}")
            if gate and auc > best_auc:
                best_j, best_auc, best_info = j, auc, (mean_w, fold_d, oof_z)
        if best_j is not None:
            active.append(best_j)
            cur_auc = best_auc
            improved = True
            print(f"  >>> ACCEPTED {basis[best_j]['name']}  nested={cur_auc:.7f}")
        else:
            print("  >>> no further component passes gate")

    # Final fit with accepted set
    if active:
        oof_z, mean_w, fold_d, final_auc = nested_auc_and_folds(active)
        R_t = np.column_stack([basis[j]["rt"] for j in active])
        test_z = z_base_t + R_t @ mean_w
        names = [basis[j]["name"] for j in active]
        weights = {n: float(w) for n, w in zip(names, mean_w)}
    else:
        oof_z, test_z, final_auc = z_base_o.copy(), z_base_t.copy(), base_auc
        names, weights, fold_d = [], {}, [0.0] * 5

    v5p5_o = rk(oof_z)
    v5p5_t = rk(test_z)
    np.save(PRED / "P_super_v5p5_clean_oof.npy", v5p5_o.astype(np.float64))
    np.save(PRED / "P_super_v5p5_clean_test.npy", v5p5_t.astype(np.float64))

    # Inject onto ev4 (true orth residual of v5p5 vs ev4 scale via Δ vs v5 mapped)
    z_ev4 = zrk(ev4)
    dz = zrk(v5p5_t) - z_base_t  # residual vs v5 on test in rank-logit
    # ECDF align dz to oof residual for KS
    dz_oof = zrk(v5p5_o) - z_base_o
    dz_al = ecdf_map_to_ref(dz, dz_oof)
    half_pred = rk(z_ev4 + 0.5 * dz_al)
    full_pred = rk(z_ev4 + 1.0 * dz_al)
    p_half = SUB / "sub_v5p5_clean_on_ev4_half.csv"
    p_full = SUB / "sub_v5p5_clean_on_ev4_1x.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": half_pred}).to_csv(p_half, index=False)
    pd.DataFrame({"id": test_ids, "is_fraud": full_pred}).to_csv(p_full, index=False)

    print("\n" + "=" * 72)
    print("FINAL COMPARISON (pick Slot 1)")
    print("=" * 72)
    print(f"  P_super_v5_harvest           OOF={base_auc:.7f}")
    print(f"  P_super_v5p5_clean           OOF={final_auc:.7f}  Δ={final_auc-base_auc:+.3e}  folds_won={sum(d>0 for d in fold_d)}/5")
    print(f"  weights={weights}")
    print(f"  fold_deltas_vs_v5={[round(d, 8) for d in fold_d]}")
    print(f"  rejected={rejected}")
    print(f"  saved {PRED/'P_super_v5p5_clean_oof.npy'} sha={sha16(PRED/'P_super_v5p5_clean_oof.npy')}")
    print(f"  saved {PRED/'P_super_v5p5_clean_test.npy'} sha={sha16(PRED/'P_super_v5p5_clean_test.npy')}")
    print(f"  {p_half.name} sha={sha16(p_half)}  ρ(ev4)={spearmanr(half_pred,ev4).statistic:.6f}  ρ(v5_half)={spearmanr(half_pred,half).statistic:.6f}")
    print(f"  {p_full.name} sha={sha16(p_full)}  ρ(ev4)={spearmanr(full_pred,ev4).statistic:.6f}  ρ(v5_1x)={spearmanr(full_pred,full).statistic:.6f}")
    print(f"  KS(v5p5 oof,test)={ks_2samp(v5p5_o, v5p5_t).statistic:.6f}")

    # Slot recommendation
    print("\nSLOT-1 RECOMMENDATION:")
    if final_auc > base_auc + 1e-7:
        print(f"  Prefer sub_v5p5_clean_on_ev4_half.csv (OOF lift {final_auc-base_auc:+.3e} over v5_harvest)")
    else:
        print("  No clean OOF lift — prefer existing sub_v5_harvest_on_ev4_half.csv / sub_v6_gm_on_maly095292.csv")

    out = {
        "v5_oof": base_auc,
        "v5p5_oof": final_auc,
        "delta": final_auc - base_auc,
        "weights": weights,
        "fold_deltas": fold_d,
        "accepted": names,
        "rejected": rejected,
        "elapsed_s": round(time.time() - t0, 2),
        "subs": {
            "sub_v5p5_clean_on_ev4_half.csv": sha16(p_half),
            "sub_v5p5_clean_on_ev4_1x.csv": sha16(p_full),
        },
    }
    (LOG / "v5p5_clean_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{out['elapsed_s']}s] wrote logs/overnight/v5p5_clean_audit.json  NO_SUBMIT")


if __name__ == "__main__":
    main()
