"""Part10 local CPU: upgrade P_super_champ (0.9521425) with gb_gpt2_bag5 + glm_quad + testbag.

STRICT NO_SUBMIT. Writes:
  preds/P_super_v3_{oof,test}.npy
  logs/overnight/part10_upgrade_psuper.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from part6_repro_and_expand import ART, folds_y, rlogit  # noqa: E402
from part7_clean_superstack import load_bank_fixed  # noqa: E402
from part8_orth_superstack import to_stack_col  # noqa: E402
from part9_joint_boost_and_basis import build_8b_champ  # noqa: E402

T0 = time.time()
POLICY = "NO_SUBMIT"
BASELINE_CLAIM = 0.9521425
C_OPT, HW_OPT, SW_OPT = 0.005, 0.12, 0.08


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


def rk(v):
    v = np.asarray(v, float).ravel()
    return rankdata(v) / len(v)


def rho(a, b):
    return float(spearmanr(np.asarray(a, float).ravel(), np.asarray(b, float).ravel()).correlation)


def d_e4(a, b):
    return (1.0 - rho(a, b)) * 1e4


def fast_auc(s, yy, bins=1 << 20):
    lo, hi = float(s.min()), float(s.max())
    b = ((s - lo) * ((bins - 1) / (hi - lo + 1e-12))).astype(np.int64)
    pos = np.bincount(b, weights=yy, minlength=bins)
    neg = np.bincount(b, minlength=bins) - pos
    return float((pos * (np.cumsum(neg) - neg)).sum() + 0.5 * (pos * neg).sum()) / (
        pos.sum() * neg.sum() + 1e-30
    )


def hill_climb(O, yy, iters=80):
    w = np.zeros(O.shape[1])
    w[int(np.argmax([fast_auc(O[:, i], yy) for i in range(O.shape[1])]))] = 1.0
    s = O @ w
    cur = fast_auc(s, yy)
    for _ in range(iters):
        sc, i = max((fast_auc(s + O[:, i], yy), i) for i in range(O.shape[1]))
        if sc <= cur + 1e-7:
            break
        w[i] += 1
        s = s + O[:, i]
        cur = sc
    return w / w.sum()


def ols_fit(yy, *cols):
    X = np.column_stack(cols)
    try:
        return np.linalg.solve(X.T @ X, X.T @ yy)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X.T @ X, X.T @ yy, rcond=None)[0]


def dual_orth_foldwise(bo, bt, a1_o, a2_o, a1_t, a2_t, folds):
    """Fold-wise OLS residual of rlogit(rank(bo)) vs two anchors; full-fit for test."""
    n = len(bo)
    br = rlogit(rk(bo))
    x1, x2 = rlogit(rk(a1_o)), rlogit(rk(a2_o))
    e = np.zeros(n)
    for a, b in folds:
        b1, b2 = ols_fit(br[a], x1[a], x2[a])
        e[b] = br[b] - b1 * x1[b] - b2 * x2[b]
    b1, b2 = ols_fit(br, x1, x2)
    et = rlogit(rk(bt)) - b1 * rlogit(rk(a1_t)) - b2 * rlogit(rk(a2_t))
    return e, et


def single_orth_foldwise(bo, bt, a_o, a_t, folds):
    n = len(bo)
    br = rlogit(rk(bo))
    x = rlogit(rk(a_o))
    e = np.zeros(n)
    for a, b in folds:
        (coef,) = ols_fit(br[a], x[a])
        e[b] = br[b] - coef * x[b]
    (coef,) = ols_fit(br, x)
    et = rlogit(rk(bt)) - coef * rlogit(rk(a_t))
    return e, et


def nest_newton(O, y, folds, C=C_OPT):
    st = np.zeros(len(y))
    X = rlogit(O)
    for a, b in folds:
        m = LogisticRegression(C=C, tol=1e-8, max_iter=200, solver="newton-cholesky")
        m.fit(X[a], y[a])
        st[b] = rk(m.decision_function(X[b]))
    return st, float(roc_auc_score(y, st))


def nest_hill(O, y, folds):
    hc = np.zeros(len(y))
    for a, b in folds:
        w = hill_climb(O[a], y[a])
        hc[b] = rk(O[b] @ w)
    return hc, float(roc_auc_score(y, hc))


def p_super_from_parts(st_oof, hc_oof, sup_o, y, folds, hw=HW_OPT, sw=SW_OPT):
    mix = rk((1.0 - hw) * st_oof + hw * hc_oof)
    p = rk((1.0 - sw) * mix + sw * rk(sup_o))
    auc = float(roc_auc_score(y, p))
    folds_auc = [float(roc_auc_score(y[b], p[b])) for _, b in folds]
    return p, auc, folds_auc, float(roc_auc_score(y, mix))


def fit_test_stack(O, T, y, C=C_OPT):
    m = LogisticRegression(C=C, tol=1e-8, max_iter=200, solver="newton-cholesky")
    m.fit(rlogit(O), y)
    st_te = rk(m.decision_function(rlogit(T)))
    w = hill_climb(O, y)
    hc_te = rk(T @ w)
    return st_te, hc_te


def wire_testbags(names, T):
    """Part-9 recipe: replace three test columns with multi-refit rank bags."""
    T = T.copy()
    ni = {n: i for i, n in enumerate(names)}
    mapping = {
        "gb_bag5": ART / "gb_testbag_full.npy",
        "gbn_bag5_w512": ART / "gbn_w512_testbag_full.npy",
        "gbn_bag5_w128": ART / "gbn_w128_testbag_full.npy",
    }
    info = {}
    for m, path in mapping.items():
        if m not in ni:
            raise KeyError(f"{m} missing from 8B names")
        arr = np.load(path).ravel().astype(float)
        assert arr.shape == (T.shape[0],) and not np.isnan(arr).any()
        old = T[:, ni[m]].copy()
        T[:, ni[m]] = rk(arr)
        info[m] = {"file": path.name, "rho_vs_old": rho(rk(arr), old)}
        log(f"  testbag wire {m} ← {path.name}  ρ(vs old col)={info[m]['rho_vs_old']:.6f}")
    return T, info


def main():
    log(f"Part10 upgrade P_super — {POLICY}")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    folds = folds_y(y)
    sub_ids = pd.read_csv("data/sample_submission.csv")["id"].to_numpy()
    N, Nt = len(y), len(sub_ids)

    # ── 74-member Part-8B bank ────────────────────────────────────────────
    log("build_8b_champ (74 cols) …")
    names8, O8, T8 = build_8b_champ(y, folds)
    assert O8.shape[1] == 74, O8.shape
    log(f"  O8={O8.shape} T8={T8.shape}")

    log("wire Part-9 testbags into T8 …")
    T8w, wire_info = wire_testbags(names8, T8)

    # Anchors for orth
    _, G_o, G_t, _ = load_bank_fixed(["mega:glm_gen"], y)
    glm_o, glm_t = G_o[:, 0], G_t[:, 0]
    # gpt2 already in bank as ranked col — also load raw for dual orth anchors
    gpt_o = np.load("external_bank/masaya_gpt2/oof.npy").ravel().astype(float)
    gpt_t = np.load("external_bank/masaya_gpt2/test.npy").ravel().astype(float)

    gb5_o = np.load(ART / "gb_bag5_oof.npy").ravel().astype(float)
    gb5_t = np.load(ART / "gb_bag5_full.npy").ravel().astype(float)
    # testbag-wired gb for gb_super test side
    gb5_t_bag = np.load(ART / "gb_testbag_full.npy").ravel().astype(float)

    gj_o = np.load("preds/gb_gpt2_joint_oof.npy").ravel().astype(float)
    gj_t = np.load("preds/gb_gpt2_joint_test.npy").ravel().astype(float)
    bag_o = np.load("preds/gb_gpt2_bag5_oof.npy").ravel().astype(float)
    bag_t = np.load("preds/gb_gpt2_bag5_full.npy").ravel().astype(float)
    gq_o = np.load("preds/ladder/glm_quad_oof.npy").ravel().astype(float)
    gq_t = np.load("preds/ladder/glm_quad_test.npy").ravel().astype(float)

    log(
        f"members: joint OOF={roc_auc_score(y, gj_o):.6f}  "
        f"gb_gpt2_bag5={roc_auc_score(y, bag_o):.6f}  glm_quad={roc_auc_score(y, gq_o):.6f}"
    )

    # gb_super variants
    sup_joint_o = 0.50 * rk(gb5_o) + 0.50 * rk(gj_o)
    sup_joint_t = 0.50 * rk(gb5_t) + 0.50 * rk(gj_t)
    # Upgrade1: 5-seed bag replaces 1-seed joint; test uses testbag for gb_bag5 half
    sup_bag_o = 0.50 * rk(gb5_o) + 0.50 * rk(bag_o)
    sup_bag_t = 0.50 * rk(gb5_t_bag) + 0.50 * rk(bag_t)
    log(
        f"gb_super joint={roc_auc_score(y, sup_joint_o):.6f}  "
        f"gb_super bag5={roc_auc_score(y, sup_bag_o):.6f} (claimed combo 0.951851)"
    )

    # Extra bank columns
    bag_rk_o, bag_rk_t = rk(bag_o), rk(bag_t)
    gq_rk_o, gq_rk_t = rk(gq_o), rk(gq_t)

    e_bag_o, e_bag_t = dual_orth_foldwise(bag_o, bag_t, glm_o, gpt_o, glm_t, gpt_t, folds)
    e_gq_o, e_gq_t = single_orth_foldwise(gq_o, gq_t, glm_o, glm_t, folds)
    bag_orth_o = to_stack_col(e_bag_o, True)
    bag_orth_t = to_stack_col(e_bag_t, True)
    gq_orth_o = to_stack_col(e_gq_o, True)
    gq_orth_t = to_stack_col(e_gq_t, True)

    banks = {
        "8B74": (O8, T8w, names8),
        "8B74+bag5": (
            np.column_stack([O8, bag_rk_o]),
            np.column_stack([T8w, bag_rk_t]),
            names8 + ["gb_gpt2_bag5"],
        ),
        "8B74+bag5+orth": (
            np.column_stack([O8, bag_rk_o, bag_orth_o]),
            np.column_stack([T8w, bag_rk_t, bag_orth_t]),
            names8 + ["gb_gpt2_bag5", "gb_gpt2_bag5_dualresid"],
        ),
        "8B74+glm_quad": (
            np.column_stack([O8, gq_rk_o]),
            np.column_stack([T8w, gq_rk_t]),
            names8 + ["glm_quad"],
        ),
        "8B74+glm_quad+orth": (
            np.column_stack([O8, gq_rk_o, gq_orth_o]),
            np.column_stack([T8w, gq_rk_t, gq_orth_t]),
            names8 + ["glm_quad", "glm_quad_orth_glm"],
        ),
        "8B74+bag5+glm_quad": (
            np.column_stack([O8, bag_rk_o, gq_rk_o]),
            np.column_stack([T8w, bag_rk_t, gq_rk_t]),
            names8 + ["gb_gpt2_bag5", "glm_quad"],
        ),
        "8B74+bag5+orth+glm_quad+orth": (
            np.column_stack([O8, bag_rk_o, bag_orth_o, gq_rk_o, gq_orth_o]),
            np.column_stack([T8w, bag_rk_t, bag_orth_t, gq_rk_t, gq_orth_t]),
            names8
            + [
                "gb_gpt2_bag5",
                "gb_gpt2_bag5_dualresid",
                "glm_quad",
                "glm_quad_orth_glm",
            ],
        ),
    }

    # Nested newton + hill for each bank (cached on 8B74)
    nest = {}
    for label, (O, T, names) in banks.items():
        log(f"nested newton+hill on {label} (n={O.shape[1]}) …")
        st, auc_st = nest_newton(O, y, folds)
        hc, auc_hc = nest_hill(O, y, folds)
        nest[label] = {"st": st, "hc": hc, "auc_newton": auc_st, "auc_hill": auc_hc, "O": O, "T": T, "names": names}
        log(f"  newton={auc_st:.7f}  hill={auc_hc:.7f}")

    # Ablation table: P_super recipes
    rows = []

    def add_row(name, st, hc, sup_o, note=""):
        p, auc, fa, auc_mix = p_super_from_parts(st, hc, sup_o, y, folds)
        d = auc - BASELINE_CLAIM
        rows.append(
            {
                "name": name,
                "nested_auc": auc,
                "delta_vs_02521425": d,
                "newton_hill_mix": auc_mix,
                "folds": fa,
                "note": note,
            }
        )
        log(f"  {name:48s}  AUC={auc:.7f}  Δ={d:+.7e}  {note}")
        return p, auc

    print("\n" + "=" * 88)
    print("ABLATION vs P_super_champ claimed 0.9521425")
    print("=" * 88)

    # Baseline exact recipe (joint gb_super; T unused for nested)
    n0 = nest["8B74"]
    p_base, auc_base = add_row(
        "baseline_8B74 +12%hc +8%gb_super(joint)",
        n0["st"],
        n0["hc"],
        sup_joint_o,
        note="reproduce part9b",
    )

    # Upgrade 1a: swap joint→bag5 inside gb_super only
    p_u1a, _ = add_row(
        "8B74 +12%hc +8%gb_super(bag5)",
        n0["st"],
        n0["hc"],
        sup_bag_o,
        note="U1: replace joint with 5-seed bag5 in gb_super",
    )

    # Upgrade 1b/c: add bag5 (±orth) into newton bank, keep bag5 gb_super
    for lab in ("8B74+bag5", "8B74+bag5+orth"):
        n = nest[lab]
        add_row(
            f"{lab} +12%hc +8%gb_super(bag5)",
            n["st"],
            n["hc"],
            sup_bag_o,
            note="U1 bank inject",
        )

    # Upgrade 2: glm_quad ± orth
    for lab in ("8B74+glm_quad", "8B74+glm_quad+orth"):
        n = nest[lab]
        add_row(
            f"{lab} +12%hc +8%gb_super(bag5)",
            n["st"],
            n["hc"],
            sup_bag_o,
            note="U2 glm_quad",
        )

    # Combined
    for lab in ("8B74+bag5+glm_quad", "8B74+bag5+orth+glm_quad+orth"):
        n = nest[lab]
        add_row(
            f"{lab} +12%hc +8%gb_super(bag5)",
            n["st"],
            n["hc"],
            sup_bag_o,
            note="U1+U2",
        )

    # Pure newton (no hill/super) for reference on best bank
    best_row = max(rows, key=lambda r: r["nested_auc"])
    log(f"BEST nested so far: {best_row['name']}  {best_row['nested_auc']:.7f}")

    # Also report newton-only deltas on expanded banks
    print("\nNewton-only (C=0.005) on banks (no hill/super):")
    for lab, n in nest.items():
        d = n["auc_newton"] - 0.952130  # part9b claimed pure newton
        log(f"  {lab:40s} newton={n['auc_newton']:.7f}  Δvs0.952130={d:+.7e}")

    # Pick champion = best nested AUC among recipes
    champ = max(rows, key=lambda r: r["nested_auc"])
    # Map champ name → bank key + super
    def bank_key_from_name(name: str) -> str:
        for k in (
            "8B74+bag5+orth+glm_quad+orth",
            "8B74+bag5+glm_quad",
            "8B74+bag5+orth",
            "8B74+glm_quad+orth",
            "8B74+glm_quad",
            "8B74+bag5",
            "8B74",
        ):
            if k in name:
                return k
        return "8B74"

    bk = bank_key_from_name(champ["name"])
    use_bag_super = "gb_super(bag5)" in champ["name"] or (
        "joint" not in champ["name"] and "baseline" not in champ["name"]
    )
    nC = nest[bk]
    # Rebuild champ OOF explicitly
    P_v3_oof, auc_v3, fa_v3, auc_mix_v3 = p_super_from_parts(
        nC["st"], nC["hc"], sup_bag_o if use_bag_super else sup_joint_o, y, folds
    )
    assert abs(auc_v3 - champ["nested_auc"]) < 1e-12

    # Test with wired T + matching super
    st_te, hc_te = fit_test_stack(nC["O"], nC["T"], y)
    mix_te = rk((1.0 - HW_OPT) * st_te + HW_OPT * hc_te)
    sup_t = sup_bag_t if use_bag_super else sup_joint_t
    P_v3_te = rk((1.0 - SW_OPT) * mix_te + SW_OPT * rk(sup_t))

    Path("preds").mkdir(exist_ok=True)
    np.save("preds/P_super_v3_oof.npy", P_v3_oof.astype(np.float32))
    np.save("preds/P_super_v3_test.npy", P_v3_te.astype(np.float32))
    log(f"saved preds/P_super_v3_{{oof,test}}.npy  OOF={auc_v3:.7f}  bank={bk}  super={'bag5' if use_bag_super else 'joint'}")

    # ── Amanatar formula + Spearman queue ─────────────────────────────────
    print("\n" + "=" * 88)
    print("AMANATAR Sovereign Fusion (0.95237 LB) — exact formula")
    print("=" * 88)
    amanatar_formula = {
        "lb": 0.95237,
        "inner_stack": {
            "n_members": 48,
            "selection": "unbundle 117 OOFs → prune Pearson r>0.9995 → 48",
            "transform": "rank → logit(clip)",
            "solver": "LogisticRegression newton-cholesky C∈[0.01,0.10] (reported nested 0.951982)",
            "top_coefs": [
                ("masaya_gpt2_target_first", 0.586934),
                ("goodpjw_gb_bag5", 0.208229),
                ("goodpjw_gb_ohk", 0.175712),
                ("alexchilton_oof_00107", 0.169397),
                ("goodpjw_lgb_diglead_l3", 0.122832),
            ],
        },
        "post_stack_fusion": {
            "formula": "0.50 * P_unbundled_newton_cholesky + 0.50 * P_titan_triad",
            "titan_triad_file": "sub_titan_triad_40_30_30.csv",
            "unbundled_file": "sub_unbundled_newton_cholesky.csv",
            "note": "Titan Triad is a 40/30/30 public CSV blend (LB 0.95232); 50/50 with unbundled stack → 0.95237",
        },
    }
    print(json.dumps(amanatar_formula, indent=2))

    # Rebuild public refs for Spearman (test-side)
    log("rebuild goodpjw_part9 / part10 / p7gs for Spearman …")
    # Part9 = rankmean(stack65(testbag), stack62(testbag)) on P7 bank — approximate via cached
    # Use same construction as ladder audit: fit 8B74 newton on O8 with T8w is NOT part9.
    # Part9 uses P7-65. Rebuild quickly.
    from part8_proprietary_scan import build_p7_65  # noqa: E402
    from part7_clean_superstack import fit_test  # noqa: E402

    p7_names, P7_O, P7_T, _ = build_p7_65(y)
    P9_T, _ = wire_testbags(p7_names, P7_T)
    te65, _ = fit_test(P7_O, P9_T, y, C=0.01, tol=1e-8, max_iter=200)
    te62, _ = fit_test(P7_O[:, :62], P9_T[:, :62], y, C=0.01, tol=1e-8, max_iter=200)
    goodpjw_part9 = rk(rk(te65) + rk(te62))
    amanatar = (
        pd.read_csv("external_bank/amanatar_sovereign/submission.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    goodpjw_part10 = rk(rk(goodpjw_part9) + rk(amanatar))
    p7 = (
        pd.read_csv("external_bank/goodpjw_part7_out/submission.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    part6 = (
        pd.read_csv("external_bank/malyshev_v11/nov04_part6.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    alex = (
        pd.read_csv("external_bank/malyshev_v11/nov04_alex.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    gs = rk(0.6 * rk(part6) + 0.4 * rk(alex))
    p7gs = rk(0.5 * rk(p7) + 0.5 * rk(gs))

    vecs = {
        "P_super_v3": P_v3_te,
        "goodpjw_part9": goodpjw_part9,
        "amanatar_sovereign": amanatar,
        "goodpjw_part10": goodpjw_part10,
        "p7gs": p7gs,
    }
    print("\n" + "=" * 88)
    print("TEST Spearman: (1-ρ)×1e4  (lower = closer)")
    print("=" * 88)
    keys = list(vecs.keys())
    hdr = f"{'':22s}" + "".join(f"{k[:12]:>12s}" for k in keys)
    print(hdr)
    dist = {}
    for a in keys:
        dist[a] = {}
        row = f"{a:22s}"
        for b in keys:
            d = 0.0 if a == b else d_e4(vecs[a], vecs[b])
            dist[a][b] = d
            row += f"{d:12.2f}"
        print(row)

    print("\nTomorrow candidate queue (from distances + nested):")
    queue = [
        {
            "id": "A",
            "desc": "Submit/hold P_super_v3 alone (nested best local)",
            "nested": auc_v3,
            "d_e4_vs_part10": dist["P_super_v3"]["goodpjw_part10"],
        },
        {
            "id": "B",
            "desc": "0.5*rk(P_super_v3)+0.5*rk(amanatar) — mirror Sovereign 50/50",
            "nested": None,
            "d_e4_components": dist["P_super_v3"]["amanatar_sovereign"],
        },
        {
            "id": "C",
            "desc": "0.5*rk(P_super_v3)+0.5*rk(goodpjw_part9)",
            "nested": None,
            "d_e4_components": dist["P_super_v3"]["goodpjw_part9"],
        },
        {
            "id": "D",
            "desc": "0.35*P_super_v3 + 0.15*p7 + 0.44*gs + 0.06*pb (part9c recipe w/ v3)",
            "nested": None,
        },
        {
            "id": "E",
            "desc": "Watch: if nested still <0.9521425, do NOT replace P_super_champ cache",
            "nested_gap": auc_v3 - BASELINE_CLAIM,
        },
    ]
    for q in queue:
        print(f"  [{q['id']}] {q['desc']}")

    report = {
        "policy": POLICY,
        "baseline_claimed": BASELINE_CLAIM,
        "baseline_reproduced": auc_base,
        "baseline_delta_repro": auc_base - BASELINE_CLAIM,
        "wire_info": wire_info,
        "gb_super": {
            "joint_oof": float(roc_auc_score(y, sup_joint_o)),
            "bag5_oof": float(roc_auc_score(y, sup_bag_o)),
            "bag5_member_oof": float(roc_auc_score(y, bag_o)),
            "glm_quad_oof": float(roc_auc_score(y, gq_o)),
        },
        "newton_only": {k: v["auc_newton"] for k, v in nest.items()},
        "ablation": rows,
        "champion": {
            "name": champ["name"],
            "bank": bk,
            "gb_super": "bag5" if use_bag_super else "joint",
            "nested_auc": auc_v3,
            "delta_vs_baseline": auc_v3 - BASELINE_CLAIM,
            "folds": fa_v3,
            "files": ["preds/P_super_v3_oof.npy", "preds/P_super_v3_test.npy"],
        },
        "amanatar_formula": amanatar_formula,
        "test_distance_e4": dist,
        "tomorrow_queue": queue,
        "runtime_s": time.time() - T0,
    }
    out = Path("logs/overnight/part10_upgrade_psuper.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    log(f"wrote {out}")
    log("DONE — NO_SUBMIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
