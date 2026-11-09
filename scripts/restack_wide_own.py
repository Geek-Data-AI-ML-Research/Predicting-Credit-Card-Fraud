#!/usr/bin/env python3
"""
Own wide stack: every public OOF bank on the shared 5-fold split + our proprietary members,
one converged L2 logistic stack on logit-ranks (goodpjw Part 11/12 recipe), nested 5-fold CV.

Compares PUBLIC-only pool vs PUBLIC+OURS on identical folds and solver.
Writes preds/P_restack_{pub,own}_{oof,test}.npy and logs/overnight/restack_wide_own.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paradigm_stack_delta_eval import fast_auc

B = ROOT / "external/banks"
W11 = ROOT / "external/nov07_wide_recon"
MG = Path("/tmp/nov08_recon/megayak_s6e11-per-value-digit-te-shallow-tree-stack")
PRED = ROOT / "preds"
LOG = ROOT / "logs/overnight"
NTR, NTE = 699_960, 299_984
MIN_AUC = 0.944
DUP_RHO = 0.9995
C = 0.01


def lg_rank(v: np.ndarray) -> np.ndarray:
    r = (rankdata(v, method="average") - 0.5) / len(v)
    return logit(np.clip(r, 1e-6, 1 - 1e-6)).astype(np.float32)


def stacker() -> LogisticRegression:
    return LogisticRegression(C=C, tol=1e-8, max_iter=200, solver="newton-cholesky")


def collect_public(train_ids, test_ids) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    m: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name, po, pt):
        if Path(po).exists() and Path(pt).exists():
            m[name] = (np.load(po).ravel(), np.load(pt).ravel())

    for d, tag in [(B / "goodpjw2008_s6e11-oof-bank", "pjw"),
                   (B / "alexchilton_s6e11-oof-predictions", "alex"),
                   (B / "mathsonchain_s6e11-fraud-oof-test-17-single-models", "moc"),
                   (MG, "mega")]:
        for po in sorted(d.glob("oof_*.npy")):
            if po.name.startswith("._"):
                continue
            n = po.name[4:-4]
            add(f"{tag}:{n}", po, d / f"test_{n}.npy")

    art = B / "goodpjw2008_s6e11-fraud-series-artifacts"
    for po in sorted(art.glob("*_oof.npy")):
        if po.name.startswith("._") or "glmoof" in po.name:
            continue
        n = po.name[:-8]
        for suf in ("_full.npy", "_test.npy"):
            if (art / f"{n}{suf}").exists():
                add(f"pjwart:{n}", po, art / f"{n}{suf}")
                break

    sad = B / "sadamtorres_ps-s6e11-tabpfn-gbdt-oof-test-pairs"
    for po in sorted(sad.glob("pair_oof_*.csv")):
        n = po.name[9:-4]
        pt = sad / f"pair_test_{n}.csv"
        if pt.exists():
            o = pd.read_csv(po).set_index("id").reindex(train_ids)["is_fraud"].to_numpy(float)
            t = pd.read_csv(pt).set_index("id").reindex(test_ids)["is_fraud"].to_numpy(float)
            m[f"sad:{n}"] = (o, t)

    mas = B / "masayakawamata_s6e11-gpt2-target-first-oof"
    add("masa:gpt2", mas / "oof.npy", mas / "test.npy")
    mlm = B / "megayak_s6e11-target-first-lm"
    add("mega:lm", mlm / "lm_oof.npy", mlm / "lm_test.npy")

    s10 = W11 / "sameer_10f"
    for n in ("gb10_bag", "gb10_s0", "gb10_s1"):
        add(f"sam10:{n}", s10 / f"{n}_oof.npy", s10 / f"{n}_full.npy")
    return m


def collect_ours() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pairs = {
        "our:v4p_hc": ("P_super_v4p_hc_oof.npy", "P_super_v4p_hc_test.npy"),
        "our:B_xgb_llr": ("paradigm_B/xgb_llr_oof.npy", "paradigm_B/xgb_llr_test.npy"),
        "our:P4_L4.3": ("matrix_v2/P4_L4.3_pure5f_oof.npy", "matrix_v2/P4_L4.3_pure5f_test.npy"),
        "our:P4_L4.2": ("matrix_v2/P4_L4.2_pure5f_oof.npy", "matrix_v2/P4_L4.2_pure5f_test.npy"),
        "our:P4_L4.5": ("matrix_v2/P4_L4.5_pure5f_oof.npy", "matrix_v2/P4_L4.5_pure5f_test.npy"),
        "our:P4_L4.3_ext": ("matrix_v2/P4_L4.3_ext_oof.npy", "matrix_v2/P4_L4.3_ext_test.npy"),
        "our:P4_L4.2_ext": ("matrix_v2/P4_L4.2_ext_oof.npy", "matrix_v2/P4_L4.2_ext_test.npy"),
        "our:gpt2_3ep": ("gpt2_3ep/gpt2_3ep_oof.npy", "gpt2_3ep/gpt2_3ep_test.npy"),
        "our:gpt2_bidir": ("gpt2_3ep/gpt2_bidir_oof.npy", "gpt2_3ep/gpt2_bidir_test.npy"),
        "our:xgb_gpt2_3ep": ("gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy", "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_test.npy"),
        "our:lgb_gpt2_3ep": ("gpt2_3ep/lgb_gpt2_3ep_leaves4_bag6_oof.npy", "gpt2_3ep/lgb_gpt2_3ep_leaves4_bag6_test.npy"),
        "our:gb_gpt2_3ep_super": ("gpt2_3ep/gb_gpt2_3ep_super_oof.npy", "gpt2_3ep/gb_gpt2_3ep_super_test.npy"),
        "our:p4_cuda_xgb": ("matrix_v2/harvest_p4_cuda_xgb_oof.npy", "matrix_v2/harvest_p4_cuda_xgb_test.npy"),
    }
    m = {}
    for k, (o, t) in pairs.items():
        if (PRED / o).exists() and (PRED / t).exists():
            m[k] = (np.load(PRED / o).ravel(), np.load(PRED / t).ravel())
    return m


def screen(members, y, min_auc=MIN_AUC):
    ok = {}
    for k, (o, t) in members.items():
        if len(o) != NTR or len(t) != NTE or not (np.isfinite(o).all() and np.isfinite(t).all()):
            print(f"  skip {k}: shape/finite")
            continue
        a = float(fast_auc(y, o))
        if a < min_auc:
            continue
        ok[k] = (o, t, a)
    return ok


def dedupe(ok, y, protect=()):
    names = sorted(ok, key=lambda k: -ok[k][2])
    rng = np.random.default_rng(0)
    sub = rng.choice(NTR, 120_000, replace=False)
    Rs = {k: rankdata(ok[k][0][sub]) for k in names}
    kept = []
    for k in names:
        dup = False
        for j in kept:
            if np.corrcoef(Rs[k], Rs[j])[0, 1] > DUP_RHO and (k not in protect or j in protect):
                dup = True
                break
        if not dup:
            kept.append(k)
    return kept


def nested_stack(names, ok, y, folds, tag):
    t0 = time.time()
    X = np.column_stack([lg_rank(ok[n][0]) for n in names])
    Xt = np.column_stack([lg_rank(ok[n][1]) for n in names])
    nested = np.zeros(NTR)
    fold_auc = []
    for a, b in folds:
        dfun = stacker().fit(X[a], y[a]).decision_function(X[b])
        nested[b] = dfun
        fold_auc.append(float(fast_auc(y[b], dfun)))
    full = stacker().fit(X, y)
    test = full.decision_function(Xt)
    auc = float(fast_auc(y, nested))
    print(f"[{tag}] members={len(names)} nested OOF={auc:.7f} folds={[round(f, 5) for f in fold_auc]} ({time.time()-t0:.0f}s)", flush=True)
    coef = dict(zip(names, full.coef_[0].astype(float)))
    return auc, nested, test, fold_auc, coef


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "is_fraud"])
    y = tr["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))

    pub = screen(collect_public(tr["id"].to_numpy(), test_ids), y)
    p12 = ROOT / "external/nov08_part12"
    pub.update(screen({"pjw:wide_lm_stack": (np.load(p12 / "wide_lm_stack_nested.npy").ravel(),
                                             np.load(p12 / "wide_lm_stack_test.npy").ravel())}, y))
    ours = screen(collect_ours(), y, min_auc=0.0)
    print(f"public members passing AUC≥{MIN_AUC}: {len(pub)} | ours: {len(ours)}")
    for k in sorted(ours, key=lambda k: -ours[k][2]):
        print(f"  {k:24s} OOF={ours[k][2]:.6f}")

    pub_names = dedupe(pub, y)
    allm = {**pub, **ours}
    own_names = dedupe(allm, y, protect=tuple(ours))
    print(f"after dedupe: public={len(pub_names)} | public+ours={len(own_names)} "
          f"(ours kept: {sum(n in ours for n in own_names)})")

    ref = np.load(B / "goodpjw2008_s6e11-fraud-series-artifacts/wide_lm_stack_nested.npy").ravel() \
        if (B / "goodpjw2008_s6e11-fraud-series-artifacts/wide_lm_stack_nested.npy").exists() \
        else np.load(ROOT / "external/nov08_part12/wide_lm_stack_nested.npy").ravel()
    ref_auc = float(fast_auc(y, ref))
    ref_folds = [float(fast_auc(y[b], ref[b])) for _, b in folds]
    print(f"[ref] goodpjw Part 12 wide_lm_stack nested OOF={ref_auc:.7f}")

    a_pub, n_pub, t_pub, f_pub, _ = nested_stack(pub_names, pub, y, folds, "PUBLIC")
    a_own, n_own, t_own, f_own, coef = nested_stack(own_names, allm, y, folds, "PUBLIC+OURS")

    win_pub = sum(o > p for o, p in zip(f_own, f_pub))
    win_ref = sum(o > r for o, r in zip(f_own, ref_folds))
    print(f"\nΔ own vs public-only = {a_own - a_pub:+.3e} ({win_pub}/5 folds)")
    print(f"Δ own vs Part 12 ref = {a_own - ref_auc:+.3e} ({win_ref}/5 folds)")
    print("our member coefficients:", {k: round(v, 4) for k, v in coef.items() if k.startswith("our:")})

    np.save(PRED / "P_restack_pub_oof.npy", n_pub)
    np.save(PRED / "P_restack_pub_test.npy", t_pub)
    np.save(PRED / "P_restack_own_oof.npy", n_own)
    np.save(PRED / "P_restack_own_test.npy", t_own)
    out = dict(
        ref_part12=ref_auc, public=a_pub, own=a_own,
        folds_ref=ref_folds, folds_public=f_pub, folds_own=f_own,
        n_public=len(pub_names), n_own=len(own_names),
        members_own=own_names, coef_own=coef, elapsed_s=round(time.time() - t0, 1),
    )
    (LOG / "restack_wide_own.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{out['elapsed_s']}s] wrote logs/overnight/restack_wide_own.json")


if __name__ == "__main__":
    main()
