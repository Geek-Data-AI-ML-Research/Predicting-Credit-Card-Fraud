#!/usr/bin/env python3
"""LOCAL CPU: V11 pull verify, Pass-1 forensic, sub_B rebuild, 2 candidate subs.

NO AUTO-SUBMIT.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from part6_repro_and_expand import (  # noqa: E402
    BANK_EXT,
    BANK_OURS,
    ART,
    PRED,
    folds_y,
    fit_test_ranks,
    load_bank,
    load_extra_gpu_families,
    nested_rank_cv,
)

V11 = ROOT / "external_bank" / "malyshev_v11"
SUBS = ROOT / "submissions"
OUT = ROOT / "logs" / "overnight"
TARGET = "is_fraud"
N_TEST = 299_984


def rank(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return (rankdata(v) - 0.5) / len(v)


def load_csv_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["id"].to_numpy(), df["is_fraud"].to_numpy(dtype=float)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    SUBS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. V11
    man = json.loads((V11 / "nov04_manifest.json").read_text())
    print("=== 1. V11 dataset ===")
    print(json.dumps(man, indent=2))
    pred_te = {}
    for name, info in man.items():
        path = V11 / f"nov04_{name}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest == info["sha256"]
        print(f"  {name}: sha256={'OK' if ok else 'MISMATCH'}  {digest[:16]}…")
        if not ok:
            raise SystemExit(f"sha mismatch for {name}")
        ids, vals = load_csv_pred(path)
        assert len(vals) == N_TEST and np.isfinite(vals).all()
        pred_te[name] = rank(vals)
        pred_te[f"{name}_raw"] = vals

    v11_base = rank(0.6 * pred_te["part6"] + 0.4 * pred_te["alex"])
    print(f"  v11_base = rank(0.6*rank(part6)+0.4*rank(alex))  n={len(v11_base)}")

    # ------------------------------------------------------------------ load data / banks
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    te_ids = pd.read_csv(ROOT / "data" / "sample_submission.csv")["id"].to_numpy()
    folds = folds_y(y)
    bank_names = BANK_OURS + BANK_EXT
    ok, P6_O, P6_T, miss = load_bank(bank_names, y)
    print(f"\nPart6 bank loaded {len(ok)}/{len(bank_names)}; missing {len(miss)}")
    if miss:
        print("  misses:", miss[:5])

    extras = load_extra_gpu_families(y)
    extra_map = {tag: (o, t) for tag, o, t in extras}
    for tag in ("gb_cat_d2_d3_bag", "gb_xgb_cuda_d1_bag", "gbx_bag3", "gb_resnet_mlp_bag"):
        if tag in extra_map:
            o, _ = extra_map[tag]
            print(f"  GPU/extra {tag}: OOF AUC={roc_auc_score(y, o):.6f}")

    # nested Part-6 repro @ C=0.01 and clean sub_B bank @ C=0.02
    nest_p6, auc_p6, _ = nested_rank_cv(P6_O, y, folds, C=0.01, transform="rlogit")
    print(f"  Part6 nested C=0.01 OOF={auc_p6:.6f}")

    # sub_B_clean_local: 62 + gbx + cat + xgb (drop resmlp), C=0.02
    b_names = list(ok)
    b_O = [P6_O]
    b_T = [P6_T]
    for tag in ("gbx_bag3", "gb_cat_d2_d3_bag", "gb_xgb_cuda_d1_bag"):
        if tag not in extra_map:
            raise FileNotFoundError(tag)
        o, t = extra_map[tag]
        b_names.append(tag)
        b_O.append(o[:, None] if o.ndim == 1 else o)
        b_T.append(t[:, None] if t.ndim == 1 else t)
    B_O = np.column_stack(b_O)
    B_T = np.column_stack(b_T)
    nest_B, auc_B, fa_B = nested_rank_cv(B_O, y, folds, C=0.02, transform="rlogit")
    sub_B_te, _ = fit_test_ranks(B_O, B_T, y, 0.02, "rlogit")
    sub_B_path = SUBS / "sub_B_clean_local.csv"
    pd.DataFrame({"id": te_ids, "is_fraud": sub_B_te}).to_csv(sub_B_path, index=False)
    print(f"  sub_B_clean_local n={B_O.shape[1]} C=0.02 OOF={auc_B:.6f} folds={[round(x,5) for x in fa_B]}")
    print(f"  wrote {sub_B_path}")

    # ------------------------------------------------------------------ 2. Pass-1 forensic
    print("\n=== 2. Pass-1 forensic audit ===")
    weights = json.loads((OUT / "phaseC_pass1_weights.json").read_text())
    leak = json.loads((OUT / "audit_pass1_leak.json").read_text())
    pass1_ids, pass1_te = load_csv_pred(SUBS / "sub_A_pass1_bank63_nested.csv")
    expand_ids, expand_te = load_csv_pred(SUBS / "sub_A_part6_plus_gpu_families.csv")
    assert np.array_equal(pass1_ids, te_ids) and np.array_equal(expand_ids, te_ids)

    print(f"  Pass-1 primary={weights['primary']} C={weights['C']}  (reported OOF 0.952085)")
    print(f"  sub_B_clean_local OOF={auc_B:.6f}  Δ(pass1−subB)≈{0.952085 - auc_B:+.6f}")
    print(f"  leak flagged members: {leak['n_flagged']} → {leak['flagged']}")

    # top ± weights
    top10 = weights["top10_nested_logit"]
    print("\n  Top-10 |weight| (Pass-1 nested logit C=0.1):")
    print(f"  {'rank':>4}  {'weight':>9}  {'oof':>8}  {'name'}")
    for row in top10:
        print(f"  {row['rank']:>4}  {row['weight']:+9.4f}  {row['oof']:.6f}  {row['name']}")

    # abs(rho_oof - rho_test) vs gb_bag5 for top-weighted members
    # Prefer rows already in leak audit; also recompute vs gb_bag5 if bank available
    gb_o = np.load(ART / "gb_bag5_oof.npy").ravel().astype(float)
    gb_t = np.load(ART / "gb_bag5_full.npy").ravel().astype(float)
    assert len(gb_o) == len(y) and len(gb_t) == N_TEST

    leak_by = {r["name"]: r for r in leak.get("rho_rows", [])}

    print("\n  Top-weighted members vs gb_bag5 (ρ_oof / ρ_test / |Δ|):")
    print(f"  {'|w|':>7}  {'ρ_oof':>7}  {'ρ_te':>7}  {'|Δρ|':>7}  flag  name")
    flagged_set = set(leak.get("flagged", []))
    dirty = False
    for row in top10:
        name = row["name"]
        # resolve OOF/test vectors for rho vs gb_bag5
        rho_o = rho_t = abs_d = float("nan")
        src = leak_by.get(name)
        if src and "rho_oof" in src:
            rho_o = src["rho_oof"]
            rho_t = src.get("rho_test", src.get("rho_te", float("nan")))
            abs_d = src.get("abs_delta_rho", abs(rho_o - rho_t))
        else:
            # try load from bank_aligned / artifacts by name pattern
            try:
                # goodpjw_artifacts__X_oof_npy → ART/X_oof.npy (strip _oof_npy)
                if name.startswith("goodpjw_artifacts__") and name.endswith("_oof_npy"):
                    stem = name[len("goodpjw_artifacts__") : -len("_oof_npy")]
                    o = np.load(ART / f"{stem}_oof.npy").ravel()
                    t = np.load(ART / f"{stem}_full.npy").ravel()
                elif name.startswith("alex_oof__") and name.endswith("_npy"):
                    h = name[len("alex_oof__oof_") : -len("_npy")] if "oof_" in name else None
                    # name like alex_oof__oof_HASH_npy
                    parts = name.replace("alex_oof__", "").replace("_npy", "")
                    # oof_HASH
                    h = parts.replace("oof_", "", 1) if parts.startswith("oof_") else parts
                    o = np.load(ROOT / "external_bank" / "alex_oof" / f"oof_{h}.npy").ravel()
                    tp = ROOT / "external_bank" / "alex_oof" / f"test_{h}.npy"
                    if not tp.exists():
                        tp = ROOT / "external_bank" / "alex_oof" / f"full_{h}.npy"
                    t = np.load(tp).ravel()
                elif name.startswith("masaya_gpt2__"):
                    o = np.load(ROOT / "external_bank" / "masaya_gpt2" / "oof.npy").ravel()
                    t = np.load(ROOT / "external_bank" / "masaya_gpt2" / "test.npy").ravel()
                else:
                    raise FileNotFoundError(name)
                rho_o = float(spearmanr(o, gb_o).correlation)
                rho_t = float(spearmanr(t, gb_t).correlation)
                abs_d = abs(rho_o - rho_t)
            except Exception as e:
                print(f"  skip {name}: {e}")
                continue
        is_flag = name in flagged_set or abs_d >= 0.01
        if is_flag and abs(row["weight"]) > 0.2:
            dirty = True
        print(
            f"  {abs(row['weight']):7.4f}  {rho_o:7.4f}  {rho_t:7.4f}  {abs_d:7.4f}  "
            f"{'FLAG' if is_flag else 'ok  '}  {name}"
        )

    verdict = (
        "DIRTY — Pass-1 0.952085 is inflated by out-of-split / unstable OOF members "
        f"(flagged={leak['n_flagged']}, masaya_gpt2 |Δρ| large, C=0.1 overfits)."
        if dirty or leak["n_flagged"] > 0
        else "CLEAN — no leak flags; 0.952085 usable for sub_2."
    )
    print(f"\n  VERDICT: {verdict}")
    use_pass1_for_sub2 = leak["n_flagged"] == 0 and not dirty
    print(f"  → sub_2 source: {'pass1 bank63' if use_pass1_for_sub2 else 'sub_B_clean_local'}")

    # ------------------------------------------------------------------ pairwise Spearman (TEST)
    print("\n=== Pairwise Spearman (TEST ranks) ===")
    vectors = {
        "v11_base": v11_base,
        "nov04_part6": pred_te["part6"],
        "nov04_alex": pred_te["alex"],
        "sub_A_expand66": rank(expand_te),
        "sub_B_clean": rank(sub_B_te),
        "pass1_bank63": rank(pass1_te),
    }
    keys = list(vectors.keys())
    print("  " + "".join(f"{k[:12]:>13}" for k in keys))
    mat = {}
    for i, a in enumerate(keys):
        row = []
        for b in keys:
            r = float(spearmanr(vectors[a], vectors[b]).correlation)
            mat[(a, b)] = r
            row.append(f"{r:13.4f}")
        print(f"  {a[:12]:12}" + "".join(row))

    # ------------------------------------------------------------------ 3. CV of blends
    print("\n=== 3. 5-fold OOF blend verification ===")
    # alex OOF proxy: member (or rank-mean of alex cols) best matching nov04_alex on TEST
    alex_idx = [i for i, n in enumerate(ok) if n.startswith("alex:")]
    best_alex_i, best_rho = None, -1.0
    for i in alex_idx:
        r = float(spearmanr(P6_T[:, i], pred_te["alex_raw"]).correlation)
        if r > best_rho:
            best_rho, best_alex_i = r, i
    # also try rank-mean of all alex test cols
    alex_mean_te = rank(P6_T[:, alex_idx].mean(axis=1)) if alex_idx else None
    rho_mean = float(spearmanr(alex_mean_te, pred_te["alex"]).correlation) if alex_mean_te is not None else -1
    if rho_mean > best_rho:
        oof_alex = rank(P6_O[:, alex_idx].mean(axis=1))
        alex_proxy = "rankmean_all_alex_bank"
        print(f"  alex proxy={alex_proxy}  ρ(test vs nov04_alex)={rho_mean:.4f}")
    else:
        oof_alex = rank(P6_O[:, best_alex_i])
        alex_proxy = ok[best_alex_i]
        print(f"  alex proxy={alex_proxy}  ρ(test vs nov04_alex)={best_rho:.4f}")

    oof_p6 = nest_p6  # already fold-ranked
    oof_cat = rank(extra_map["gb_cat_d2_d3_bag"][0])
    oof_xgb = rank(extra_map["gb_xgb_cuda_d1_bag"][0])
    oof_gbx = rank(extra_map["gbx_bag3"][0])

    def fold_auc(scores: np.ndarray) -> float:
        # re-rank within folds for fair OOF
        out = np.zeros(len(y))
        for a, b in folds:
            out[b] = rank(scores[b])
        return float(roc_auc_score(y, out))

    blend_v11_oof = 0.60 * rank(oof_p6) + 0.40 * rank(oof_alex)
    auc_v11_proxy = fold_auc(blend_v11_oof)
    print(f"  OOF 0.60*rank(part6)+0.40*rank(alex_proxy) = {auc_v11_proxy:.6f}")

    # add GPU boosters (equal small weight, then re-normalize)
    # orthogonal add: 0.50 part6 + 0.30 alex + 0.20/3 each GPU
    w_gpu = 0.20 / 3
    blend_gpu = (
        0.50 * rank(oof_p6)
        + 0.30 * rank(oof_alex)
        + w_gpu * oof_cat
        + w_gpu * oof_xgb
        + w_gpu * oof_gbx
    )
    auc_gpu = fold_auc(blend_gpu)
    print(f"  OOF 0.50*p6+0.30*alex+0.20*(cat+xgb+gbx)/3 = {auc_gpu:.6f}")

    # verify proposed sub_1 weights on OOF analogues
    # oof_subB = nest_B
    cand_w = [
        (0.52, 0.36, 0.12),
        (0.50, 0.35, 0.15),
        (0.55, 0.35, 0.10),
        (0.45, 0.40, 0.15),
        (0.48, 0.38, 0.14),
        (0.60, 0.40, 0.00),
    ]
    print("  weight grid (part6, alex, subB) → OOF AUC:")
    best_w, best_a = (0.52, 0.36, 0.12), -1.0
    for wp, wa, wb in cand_w:
        s = wp * rank(oof_p6) + wa * rank(oof_alex) + wb * rank(nest_B)
        a = fold_auc(s)
        mark = ""
        if a > best_a:
            best_a, best_w = a, (wp, wa, wb)
            mark = "  ← best"
        print(f"    {wp:.2f}/{wa:.2f}/{wb:.2f}  {a:.6f}{mark}")
    print(f"  selected weights for sub_1: {best_w}  OOF={best_a:.6f}")

    # ------------------------------------------------------------------ export subs
    print("\n=== Export candidates (NO SUBMIT) ===")
    wp, wa, wb = best_w
    # TEST side: use nov04_part6 / nov04_alex (true V11 vectors) + sub_B_clean_local
    sub1 = rank(wp * pred_te["part6"] + wa * pred_te["alex"] + wb * rank(sub_B_te))
    p1 = SUBS / "sub_1_part6_gpu_cat_xgb_plus_alex.csv"
    pd.DataFrame({"id": te_ids, "is_fraud": sub1}).to_csv(p1, index=False)
    assert len(sub1) == N_TEST and np.isfinite(sub1).all() and not np.isnan(sub1).any()
    print(f"  {p1.name}: n={len(sub1)} nan=0  weights={best_w}  OOF_proxy={best_a:.6f}")
    print(f"    ρ(sub1,v11_base)={spearmanr(sub1, v11_base).correlation:.6f}")
    print(f"    ρ(sub1,nov04_part6)={spearmanr(sub1, pred_te['part6']).correlation:.6f}")
    print(f"    ρ(sub1,sub_B)={spearmanr(sub1, rank(sub_B_te)).correlation:.6f}")

    if use_pass1_for_sub2:
        sub2 = rank(pass1_te)
        src2 = "sub_A_pass1_bank63_nested.csv"
        oof2 = 0.952085
    else:
        sub2 = rank(sub_B_te)
        src2 = "sub_B_clean_local.csv"
        oof2 = auc_B
    p2 = SUBS / "sub_2_cv_pure_expanded_stack.csv"
    pd.DataFrame({"id": te_ids, "is_fraud": sub2}).to_csv(p2, index=False)
    assert len(sub2) == N_TEST and np.isfinite(sub2).all()
    print(f"  {p2.name}: from {src2}  OOF={oof2:.6f}  n={len(sub2)} nan=0")
    print(f"    ρ(sub2,v11_base)={spearmanr(sub2, v11_base).correlation:.6f}")
    print(f"    ρ(sub2,pass1)={spearmanr(sub2, rank(pass1_te)).correlation:.6f}")

    report = {
        "v11_dataset": "malyshevdanil/s6e11-nov04-v11-selected-inputs",
        "v11_recipe": {"part6": 0.6, "alex": 0.4, "lb": 0.95222},
        "sha_ok": True,
        "part6_nested_C01": auc_p6,
        "sub_B_clean": {"n": int(B_O.shape[1]), "C": 0.02, "oof": auc_B, "folds": fa_B},
        "pass1": {
            "reported_oof": 0.952085,
            "C": 0.1,
            "n_flagged": leak["n_flagged"],
            "flagged": leak["flagged"],
            "verdict": "DIRTY" if not use_pass1_for_sub2 else "CLEAN",
        },
        "spearman_test": {f"{a}|{b}": mat[(a, b)] for a in keys for b in keys if a <= b},
        "cv": {
            "alex_proxy": alex_proxy,
            "v11_proxy_oof": auc_v11_proxy,
            "gpu_boost_oof": auc_gpu,
            "sub1_weights": list(best_w),
            "sub1_oof": best_a,
        },
        "files": [str(p1.relative_to(ROOT)), str(p2.relative_to(ROOT)), str(sub_B_path.relative_to(ROOT))],
        "policy": "NO_AUTO_SUBMIT",
        "runtime_s": time.time() - t0,
    }
    (OUT / "nov04_local_audit_summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] {time.time() - t0:.1f}s  summary → logs/overnight/nov04_local_audit_summary.json")
    print("[policy] NO_AUTO_SUBMIT")
    return 0


if __name__ == "__main__":
    # fit_test_ranks may live in module — ensure import
    try:
        from part6_repro_and_expand import fit_test_ranks  # noqa: F401
    except ImportError:
        pass
    raise SystemExit(main())
