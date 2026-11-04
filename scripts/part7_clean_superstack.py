#!/usr/bin/env python3
"""Part7 pull verify + clean |Δρ|<0.003 newton-cholesky super-stack. NO SUBMIT."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from part6_repro_and_expand import (  # noqa: E402
    ART,
    BANK_EXT,
    BANK_OURS,
    PRED,
    folds_y,
    load_bank,
    load_member,
    rlogit,
    stacker,
)

YEK_TAB = ROOT / "external_bank" / "yek_pytabkit"
YEK_TORCH = ROOT / "external_bank" / "yek_pytorch"
P7_ART = ROOT / "external_bank" / "goodpjw_part7_artifacts"
P7_OUT = ROOT / "external_bank" / "goodpjw_part7_out"
P7_NB = ROOT / "external_bank" / "goodpjw_part7_code" / "s6e11-part-7-gpt-2-mlp-stack-lb-0-95222.ipynb"
ALEX_OUT = ROOT / "external_bank" / "alex_09521_out"
MAS = ROOT / "external_bank" / "masaya_stack"
EKT = ROOT / "external_bank" / "ektarr_realmlp"
V11 = ROOT / "external_bank" / "malyshev_v11"
SUBS = ROOT / "submissions"
OUT = ROOT / "logs" / "overnight"
TARGET = "is_fraud"
SEED = 42
CS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
RHO_GATE = 0.003


def rank(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def rho(a, b):
    return float(spearmanr(np.asarray(a, float).ravel(), np.asarray(b, float).ravel()).correlation)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_yek_real(name: str, y):
    """Replace yek fallbacks with real kernel outputs."""
    if "pytabk" in name or "pytabkit" in name:
        d = YEK_TAB
    else:
        d = YEK_TORCH
    o = pd.read_csv(d / "oof_preds.csv").sort_values("id").iloc[:, -1].to_numpy(float)
    t = pd.read_csv(d / "submission.csv").sort_values("id")["is_fraud"].to_numpy(float)
    assert len(o) == len(y) and len(t) == 299_984
    return o, t


def load_member_fixed(name, y):
    if name.startswith("yek:"):
        return load_yek_real(name, y)
    return load_member(name, y)


def load_bank_fixed(names, y):
    O, T, ok, miss = [], [], [], []
    for n in names:
        try:
            o, t = load_member_fixed(n, y)
            O.append(o)
            T.append(t)
            ok.append(n)
        except Exception as e:
            miss.append({"name": n, "error": str(e)})
            print(f"MISS {n}: {e}", flush=True)
    return ok, np.column_stack(O), np.column_stack(T), miss


def nested_cv(P, y, folds, C=0.01, tol=1e-10, max_iter=2000):
    nested = np.zeros(len(y), dtype=float)
    X = rlogit(P)
    for a, b in folds:
        m = LogisticRegression(C=C, tol=tol, max_iter=max_iter, solver="newton-cholesky").fit(X[a], y[a])
        nested[b] = rankdata(m.decision_function(X[b])) / len(b)
    auc = float(roc_auc_score(y, nested))
    fa = [float(roc_auc_score(y[b], nested[b])) for _, b in folds]
    return nested, auc, fa


def fit_test(P_o, P_t, y, C, tol=1e-10, max_iter=2000):
    m = LogisticRegression(C=C, tol=tol, max_iter=max_iter, solver="newton-cholesky").fit(rlogit(P_o), y)
    te = rankdata(m.decision_function(rlogit(P_t))) / len(P_t)
    return te, m


def extract_bank_from_nb(nb_path, cell_idx=79):
    nb = json.loads(Path(nb_path).read_text())
    src = "".join(nb["cells"][cell_idx]["source"])
    i = src.find("{", src.find("BANK ="))
    depth = 0
    for k, ch in enumerate(src[i:], i):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(src[i : k + 1])
    raise RuntimeError("BANK not found")


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    SUBS.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    te_ids = pd.read_csv(ROOT / "data" / "sample_submission.csv")["id"].to_numpy()
    folds = folds_y(y)
    gb_o = np.load(ART / "gb_bag5_oof.npy").ravel()
    gb_t = np.load(ART / "gb_bag5_full.npy").ravel()

    print("=" * 72)
    print("1. PART 7 INSPECTION")
    print("=" * 72)
    print("Part7 adds to Part6's 62: gpt2 (masaya target-first), gbn_bag5_w512, gbn_bag5_w128")
    print("Claimed nested CV 0.95206 · public LB 0.95222")
    print("stacker7: C=0.01, tol=1e-8, max_iter=200, newton-cholesky")

    p7_sub = P7_OUT / "submission.csv"
    p6_sub = P7_OUT / "submission_part6.csv"
    nov_p6 = V11 / "nov04_part6.csv"
    nov_alex = V11 / "nov04_alex.csv"
    print(f"p7 submission sha={sha(p7_sub)[:32]}…")
    print(f"p7 submission_part6 sha={sha(p6_sub)[:32]}…")
    print(f"nov04_part6 sha={sha(nov_p6)[:32]}…")
    print(f"p7_sub == nov04_part6? {p7_sub.read_bytes()==nov_p6.read_bytes()}")
    print(f"p6_sub == nov04_part6? {p6_sub.read_bytes()==nov_p6.read_bytes()}")
    p7_te = pd.read_csv(p7_sub)["is_fraud"].to_numpy(float)
    nov_p6_te = pd.read_csv(nov_p6)["is_fraud"].to_numpy(float)
    nov_alex_te = pd.read_csv(nov_alex)["is_fraud"].to_numpy(float)
    v11_base = rank(0.6 * rank(nov_p6_te) + 0.4 * rank(nov_alex_te))
    print(f"ρ(p7_sub, nov04_part6)={rho(p7_te, nov_p6_te):.6f}")
    print(f"ρ(p7_sub, nov04_alex)={rho(p7_te, nov_alex_te):.6f}")
    print(f"ρ(p7_sub, v11_base)={rho(p7_te, v11_base):.6f}")

    # yek real AUCs
    print("\nReal yekenot OOFs:")
    for label, d in [("pytabkit", YEK_TAB), ("pytorch", YEK_TORCH)]:
        o = pd.read_csv(d / "oof_preds.csv").sort_values("id").iloc[:, -1].to_numpy(float)
        print(f"  {label}: OOF={roc_auc_score(y, o):.6f}  n={len(o)}")

    # ------------------------------------------------------------------ Part6 with real yek
    print("\n" + "=" * 72)
    print("2. PART6 / PART7 REPRODUCTION")
    print("=" * 72)
    BANK = extract_bank_from_nb(P7_NB, 79)
    names = BANK["ours"] + BANK["ext"]
    ok, P6_O, P6_T, miss = load_bank_fixed(names, y)
    print(f"loaded {len(ok)}/{len(names)} miss={len(miss)}")
    our_auc = {n: float(roc_auc_score(y, P6_O[:, i])) for i, n in enumerate(ok)}
    diffs = []
    for n in ok:
        b = BANK["auc"][n]
        d = our_auc[n] - b
        if abs(d) > 2e-6:
            diffs.append((n, our_auc[n], b, d))
    print(f"BANK auc |Δ|>2e-6: {len(diffs)}")
    for n, o, b, d in diffs:
        print(f"  {n}: ours={o:.6f} BANK={b:.6f} Δ={d:+.6e}")
    nest6, auc6, fa6 = nested_cv(P6_O, y, folds, C=0.01, tol=1e-10, max_iter=2000)
    print(f"Part6 nested C=0.01 tol=1e-10: {auc6:.6f} (gate 0.951629) folds={[round(x,5) for x in fa6]}")
    coef6 = pd.Series(
        LogisticRegression(C=0.01, tol=1e-10, max_iter=2000, solver="newton-cholesky")
        .fit(rlogit(P6_O), y)
        .coef_.ravel(),
        index=ok,
    )
    print(f"coef mega:glm_gen={coef6['mega:glm_gen']:+.4f}")

    # Part7 bank
    gpt_o = np.load(MAS / "oof.npy").ravel().astype(float)
    gpt_t = np.load(MAS / "test.npy").ravel().astype(float)
    w512_o = np.load(P7_ART / "gbn_bag5_w512_oof.npy").ravel()
    w512_t = np.load(P7_ART / "gbn_bag5_w512_full.npy").ravel()
    w128_o = np.load(P7_ART / "gbn_bag5_w128_oof.npy").ravel()
    w128_t = np.load(P7_ART / "gbn_bag5_w128_full.npy").ravel()
    print(f"gpt2 OOF={roc_auc_score(y, gpt_o):.6f}")
    print(f"gbn_w512 OOF={roc_auc_score(y, w512_o):.6f}")
    print(f"gbn_w128 OOF={roc_auc_score(y, w128_o):.6f}")

    # Part7 uses already-ranked gpt and MLP bags; P6 cols as-is then rlogit
    # From notebook: gpt appended as rankdata/len; MLP oofs already rank means
    P7_names = list(ok) + ["gpt2", "gbn_bag5_w512", "gbn_bag5_w128"]
    P7_O = np.column_stack(
        [P6_O, rankdata(gpt_o) / len(y), w512_o.astype(float), w128_o.astype(float)]
    )
    P7_T = np.column_stack(
        [P6_T, rankdata(gpt_t) / len(gpt_t), w512_t.astype(float), w128_t.astype(float)]
    )
    nest7, auc7, fa7 = nested_cv(P7_O, y, folds, C=0.01, tol=1e-8, max_iter=200)
    print(f"Part7 nested C=0.01 tol=1e-8: {auc7:.6f} (claim 0.95206) folds={[round(x,5) for x in fa7]}")
    te7, m7 = fit_test(P7_O, P7_T, y, C=0.01, tol=1e-8, max_iter=200)
    print(f"ρ(our_p7_test, p7_submission)={rho(te7, p7_te):.6f}")
    coef7 = pd.Series(m7.coef_.ravel(), index=P7_names)
    print("Part7 top coefs:")
    for n, c in coef7.reindex(coef7.abs().sort_values(ascending=False).index).head(12).items():
        print(f"  {c:+.3f}  {n}")

    # ------------------------------------------------------------------ clean pool
    print("\n" + "=" * 72)
    print("3. CLEAN SUPER-STACK |Δρ|<0.003 vs gb_bag5")
    print("=" * 72)

    def gate(name, o, t):
        ro, rt = rho(o, gb_o), rho(t, gb_t)
        d = abs(ro - rt)
        ok_g = d < RHO_GATE
        return ok_g, ro, rt, d, float(roc_auc_score(y, o))

    pool_o, pool_t, pool_n = [], [], []

    def try_add(name, o, t):
        ok_g, ro, rt, d, a = gate(name, o, t)
        flag = "PASS" if ok_g else "DROP"
        print(f"  {flag} |Δρ|={d:.6f} OOF={a:.5f}  {name}")
        if ok_g:
            pool_o.append(np.asarray(o, float).ravel())
            pool_t.append(np.asarray(t, float).ravel())
            pool_n.append(name)
        return ok_g

    print("\n-- Part7 full bank --")
    for i, n in enumerate(P7_names):
        try_add(f"p7:{n}", P7_O[:, i], P7_T[:, i])

    print("\n-- masaya / ektarr --")
    try_add("masaya_gpt2", gpt_o, gpt_t)  # may dup gpt2 already in p7
    ekt_o = pd.read_csv(EKT / "oof_predictions.csv").sort_values("id")["stack"].to_numpy(float)
    ekt_t = pd.read_csv(EKT / "submission.csv").sort_values("id").iloc[:, -1].to_numpy(float)
    try_add("ektarr_realmlp_stack", ekt_o, ekt_t)

    print("\n-- alex_09521 own members --")
    man = json.loads((ALEX_OUT / "oof_manifest.json").read_text())
    for h, info in man.items():
        op, tp = ALEX_OUT / f"oof_{h}.npy", ALEX_OUT / f"test_{h}.npy"
        if not op.exists() or not tp.exists():
            continue
        try_add(f"alex:{info['name']}", np.load(op).ravel(), np.load(tp).ravel())

    print("\n-- GPU boosters --")
    gpu = {
        "gbx_bag3": (ART / "gbx_bag3_oof.npy", ART / "gbx_bag3_full.npy"),
        "gb_cat_d2_d3_bag": (PRED / "oof_gb_cat_d2_d3_bag.npy", PRED / "sub_gb_cat_d2_d3_bag.npy"),
        "gb_xgb_cuda_d1_bag": (PRED / "oof_gb_xgb_cuda_d1_bag.npy", PRED / "sub_gb_xgb_cuda_d1_bag.npy"),
    }
    gpu_ok = {}
    for n, (op, tp) in gpu.items():
        o, t = np.load(op).ravel(), np.load(tp).ravel()
        gpu_ok[n] = try_add(f"gpu:{n}", o, t)

    print(f"\nClean pool size: {len(pool_n)}")

    # dedupe by exact OOF equality (masaya vs p7:gpt2)
    keep = []
    seen = []
    for i, n in enumerate(pool_n):
        o = pool_o[i]
        dup = False
        for j in seen:
            if np.allclose(o, pool_o[j], rtol=0, atol=1e-12):
                print(f"  dedupe {n} == {pool_n[j]}")
                dup = True
                break
        if not dup:
            keep.append(i)
            seen.append(i)
    pool_o = [pool_o[i] for i in keep]
    pool_t = [pool_t[i] for i in keep]
    pool_n = [pool_n[i] for i in keep]
    print(f"After dedupe: {len(pool_n)}")

    Super_O = np.column_stack(pool_o)
    Super_T = np.column_stack(pool_t)

    # variants
    def idx_without(prefixes_or_names):
        return [i for i, n in enumerate(pool_n) if not any(x in n for x in prefixes_or_names)]

    variants = {
        "part6_exact": (ok, P6_O, P6_T, 0.01, 1e-10, 2000),
        "part7_exact": (P7_names, P7_O, P7_T, 0.01, 1e-8, 200),
    }

    # part7 + masaya + ektarr + clean alex (no gpu) — rebuild from named filters
    no_gpu_idx = [i for i, n in enumerate(pool_n) if not n.startswith("gpu:")]
    # If masaya/ektarr/alex already in pool after p7 gate
    variants["p7_masa_ekt_alex_nogpu"] = (
        [pool_n[i] for i in no_gpu_idx],
        Super_O[:, no_gpu_idx],
        Super_T[:, no_gpu_idx],
        None,
        1e-10,
        2000,
    )
    variants["full_clean_super"] = (pool_n, Super_O, Super_T, None, 1e-10, 2000)
    # without cat/gbx specifically
    no_cat_gbx = [i for i, n in enumerate(pool_n) if n not in ("gpu:gb_cat_d2_d3_bag", "gpu:gbx_bag3")]
    variants["super_no_cat_gbx"] = (
        [pool_n[i] for i in no_cat_gbx],
        Super_O[:, no_cat_gbx],
        Super_T[:, no_cat_gbx],
        None,
        1e-10,
        2000,
    )

    print("\nNested CV table:")
    print(f"{'variant':32} {'n':>4} {'bestC':>6} {'OOF':>10} {'folds'}")
    results = {}
    for vname, (names_v, Ov, Tv, Cfix, tol, mit) in variants.items():
        best_c, best_a, best_fa, best_nest = None, -1, None, None
        Cs = [Cfix] if Cfix is not None else CS
        for C in Cs:
            nest, a, fa = nested_cv(Ov, y, folds, C=C, tol=tol, max_iter=mit)
            if a > best_a:
                best_c, best_a, best_fa, best_nest = C, a, fa, nest
        te, model = fit_test(Ov, Tv, y, best_c, tol=tol, max_iter=mit)
        results[vname] = {
            "n": int(Ov.shape[1]),
            "C": best_c,
            "oof": best_a,
            "folds": best_fa,
            "te": te,
            "coef": pd.Series(model.coef_.ravel(), index=list(names_v) if not isinstance(names_v[0], str) else names_v),
        }
        print(
            f"{vname:32} {Ov.shape[1]:4d} {best_c:6.3f} {best_a:10.6f} {[round(x,5) for x in best_fa]}"
        )

    # exports
    # sub_clean_superstack_no_extra_glm = p7+masa+ekt+alex (no gpu)
    # sub_clean_superstack_with_part7_alex = full clean (user asked two files; map sensibly)
    exports = [
        ("sub_clean_superstack_no_extra_glm.csv", "p7_masa_ekt_alex_nogpu"),
        ("sub_clean_superstack_with_part7_alex.csv", "full_clean_super"),
    ]
    print("\nExports:")
    for fname, key in exports:
        te = results[key]["te"]
        path = SUBS / fname
        pd.DataFrame({"id": te_ids, "is_fraud": te}).to_csv(path, index=False)
        assert len(te) == 299_984 and np.isfinite(te).all()
        print(f"  {fname}: n={len(te)} nan=0  OOF={results[key]['oof']:.6f} C={results[key]['C']}")
        print(f"    ρ(p7_sub)={rho(te, p7_te):.6f}  ρ(nov04_alex)={rho(te, nov_alex_te):.6f}  ρ(v11)={rho(te, v11_base):.6f}")

    print("\nTop15 coefs full_clean_super:")
    coef = results["full_clean_super"]["coef"]
    for n, c in coef.reindex(coef.abs().sort_values(ascending=False).index).head(15).items():
        print(f"  {c:+.3f}  {n}")

    summary = {
        "part6_nested": results["part6_exact"]["oof"],
        "part6_gate": 0.951629,
        "part6_bank_diffs": len(diffs),
        "glm_gen_coef": float(coef6["mega:glm_gen"]),
        "part7_nested": results["part7_exact"]["oof"],
        "part7_claim": 0.95206,
        "rho_our_p7_vs_official": float(rho(te7, p7_te)),
        "p7_vs_nov04_part6_sha_equal": p7_sub.read_bytes() == nov_p6.read_bytes(),
        "clean_pool_n": len(pool_n),
        "variants": {
            k: {"n": v["n"], "C": v["C"], "oof": v["oof"], "folds": v["folds"]}
            for k, v in results.items()
        },
        "files": [a for a, _ in exports],
        "policy": "NO_SUBMIT",
        "runtime_s": time.time() - t0,
    }
    (OUT / "part7_clean_superstack_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.1f}s  NO_SUBMIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
