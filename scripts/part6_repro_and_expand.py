#!/usr/bin/env python3
"""Reproduce goodpjw Part 6 nested stack (gate 0.951629) then expand with new GPU families.

Stacker (exact Part 6 §6.4):
  nested[b] = rankdata(stacker(C).fit(rlogit(P6_O[a]), y[a]).decision_function(rlogit(P6_O[b]))) / len(b)
  stacker = LogisticRegression(C=C, tol=1e-10, max_iter=2000, solver='newton-cholesky')

NO auto-submit.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "logs" / "overnight"
SUBS = ROOT / "submissions"
PRED = ROOT / "preds" / "part6_expand"
P6 = ROOT / "external_bank" / "p6_sources"
ART = ROOT / "external_bank" / "goodpjw_artifacts"
ALEX = ROOT / "external_bank" / "alex_oof"
MALY = ROOT / "external_bank" / "malyshev_pooled"
TARGET = "is_fraud"
P6_NESTED_GATE = 0.951629
SEED = 42

# Exact 62-member BANK from Part 6 cell 77
BANK_OURS = [
    "lgb_diglead_l3_bag", "glm_ohk2_bag", "ebm_i25_bag4", "lgb_dig_s100_bag5",
    "lgb_rawdig_l4", "lgb_dig_l7", "gb_bag5",
]
BANK_EXT = [
    "mega:glm_boost", "mega:glm_boost_xgb", "mega:glm_boost_raw",
    "nil:lgb_unified_96key_bag", "lazy:xgb_d2_bag", "nil:xgb_unified_96key_bag",
    "lazy:xgb_l4_bag", "lazy:lgb_s100_bag", "nil:lgb_unified_s100_bag",
    "mega:lgb_te20", "lazy:lgb_l4_bag",
    "alex:LightGBM_3_leaves,_Part_5_recipe,_5_encoder_seeds",
    "mega:xgb_te20",
    "alex:XGBoost_depth_2,_Part_3_recipe,_5_encoder_seeds,_full_refit",
    "alex:LightGBM_3_leaves,_Part_5_recipe",
    "maly:full_lgb_te20", "mega:glm_gen",
    "alex:XGBoost_depth_2_boosted_from_the_joint_GLM,_full_refit",
    "maly:full_xgb_te20", "alex:LightGBM_4_leaves,_Part_3_recipe",
    "alex:XGBoost_depth_2,_Part_3_recipe", "nil:lgb_unified_l7",
    "maly:joint_token", "alex:Joint_GLM,_full_refit", "nil:glm_unified_96key",
    "alex:LightGBM_4_leaves_digits_s100", "alex:LightGBM_4_leaves_digits",
    "alex:XGBoost_depth_2_digits", "alex:LightGBM_4_leaves_digits,_no_pairs",
    "alex:XGBoost_depth_2_digits,_no_pairs", "alex:GLM_on_the_encodings",
    "yek:ps-s6-e11-realmlp-pytorc", "maly:lgb_enriched_s100",
    "maly:xgb_enriched_d1_s100", "yek:ps-s6-e11-realmlp-pytabk",
    "maly:lgb_te_s20_cnt", "maly:xgb_te_s20", "maly:lgb_te_s100",
    "alex:LightGBM_4_leaves_plus", "mega:lgb_raw", "alex:XGBoost_depth_2_bag3",
    "maly:ebm_private_int5", "alex:LightGBM_4_leaves_b", "alex:XGBoost_depth_1_bag3",
    "alex:LightGBM_tuned", "alex:LightGBM_4_leaves", "alex:XGBoost_depth_2",
    "alex:XGBoost_depth_2_plus", "alex:XGBoost_tuned", "alex:XGBoost_depth_1",
    "maly:ebm_raw", "lazy:lgb_rawview", "alex:RealMLP_digits",
    "alex:XGBoost_depth_2_raw", "alex:LightGBM_4_leaves_raw",
]

CS_RLOGIT = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]
CS_PROBIT = [0.005, 0.01, 0.015]


def folds_y(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def rlogit(r):
    r = np.clip(r, 1e-6, 1 - 1e-6)
    return np.log(r / (1 - r))


def probit_rank_cols(X):
    Z = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        r = rankdata(X[:, j], method="average")
        u = np.clip(r / (len(X) + 1.0), 1e-6, 1 - 1e-6)
        Z[:, j] = norm.ppf(u)
    return Z


def stacker(C=0.01):
    return LogisticRegression(C=C, tol=1e-10, max_iter=2000, solver="newton-cholesky")


def nested_rank_cv(P, y, folds, C=0.01, transform="rlogit"):
    nested = np.zeros(len(y), dtype=np.float64)
    X = rlogit(P) if transform == "rlogit" else probit_rank_cols(P)
    for a, b in folds:
        m = stacker(C).fit(X[a], y[a])
        nested[b] = rankdata(m.decision_function(X[b])) / len(b)
    auc = float(roc_auc_score(y, nested))
    fa = [float(roc_auc_score(y[b], nested[b])) for _, b in folds]
    return nested, auc, fa


def _load_npy(path):
    return np.asarray(np.load(path), np.float32).ravel().astype(float)


def find_alex_hash(name_key: str):
    """Map Part6 alex name (spaces→underscores) to oof hash via manifest."""
    for man_path in [
        P6 / "alex" / "oof_manifest.json",
        P6 / "alex_ds" / "oof_manifest.json",
        ALEX / "oof_manifest.json",
        ROOT / "external_bank" / "local_public_alex" / "oof_manifest.json",
    ]:
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        for h, v in man.items():
            if isinstance(v, dict) and v.get("name", "").replace(" ", "_") == name_key:
                return man_path.parent, h
            if isinstance(v, str) and v.replace(" ", "_") == name_key:
                return man_path.parent, h
    # fallback: scan alex_oof dir for matching sidecar? none — try hashed files already in bank
    return None, None


def load_member(name, y):
    p, b = name.split(":", 1) if ":" in name else ("ours", name)
    if p == "ours":
        o = _load_npy(ART / f"{b}_oof.npy")
        t = _load_npy(ART / f"{b}_full.npy")
    elif p in ("lazy", "nil"):
        d = P6 / p / "preds"
        # lazy may use slightly different names
        cand_o = [d / f"{b}_oof.npy", P6 / p / f"{b}_oof.npy"]
        cand_t = [d / f"{b}_full.npy", P6 / p / f"{b}_full.npy"]
        # lazy bags sometimes without _bag suffix variants
        if b == "xgb_d2_bag":
            cand_o.append(d / "xgb_d2_bag_oof.npy")
        o = t = None
        for co, ct in zip(cand_o, cand_t):
            if co.exists() and ct.exists():
                o, t = _load_npy(co), _load_npy(ct)
                break
        if o is None:
            raise FileNotFoundError(f"{name}: tried {cand_o}")
    elif p == "mega":
        d = P6 / "mega"
        o = _load_npy(d / f"oof_{b}.npy")
        t = _load_npy(d / f"refit_{b}.npy")
    elif p == "alex":
        d, h = find_alex_hash(b)
        if d is None:
            # try matching already-aligned phaseA bank member by fuzzy name later
            raise FileNotFoundError(f"alex manifest miss for {b}")
        o = _load_npy(d / f"oof_{h}.npy")
        tpath = d / f"test_{h}.npy"
        if not tpath.exists():
            tpath = d / f"full_{h}.npy"
        t = _load_npy(tpath)
    elif p == "yek":
        # notebook truncates: pytorc / pytabk — kernels may be private; fall back to goodpjw realmlp artifacts
        if "pytorc" in b:
            d = P6 / "yek_torch"
            fb_o, fb_t = ART / "realmlp_dig_oof.npy", ART / "realmlp_dig_test.npy"
        else:
            d = P6 / "yek_tab"
            fb_o, fb_t = ART / "realmlp_te_oof.npy", ART / "realmlp_te_test.npy"
        if (d / "oof_preds.csv").exists() and (d / "submission.csv").exists():
            o = pd.read_csv(d / "oof_preds.csv").iloc[:, -1].values.astype(float)
            t = pd.read_csv(d / "submission.csv").iloc[:, -1].values.astype(float)
        elif fb_o.exists() and fb_t.exists():
            print(f"[part6] yek fallback {b} -> {fb_o.name}", flush=True)
            o, t = _load_npy(fb_o), _load_npy(fb_t)
        else:
            raise FileNotFoundError(f"yek missing {d} and fallback {fb_o}")
    else:  # maly
        d = P6 / "maly" if (P6 / "maly" / "pool_bank.npz").exists() else MALY
        z = np.load(d / "pool_bank.npz", allow_pickle=False)
        names = json.loads((d / "pool_manifest.json").read_text())["names"]
        key = "own_" + b
        if key not in names:
            # try without own_
            if b in names:
                key = b
            else:
                raise FileNotFoundError(f"maly missing {key} in {names[:10]}...")
        j = names.index(key)
        o, t = z["oof"][:, j].astype(float), z["test"][:, j].astype(float)
    assert len(o) == len(y) and len(t) == 299_984, (name, len(o), len(t))
    return o, t


def load_bank(names, y):
    O, T, ok, miss = [], [], [], []
    for n in names:
        try:
            o, t = load_member(n, y)
            O.append(o)
            T.append(t)
            ok.append(n)
        except Exception as e:
            miss.append({"name": n, "error": str(e)})
            print(f"[part6] MISS {n}: {e}", flush=True)
    if not O:
        raise RuntimeError("no members loaded")
    return ok, np.column_stack(O), np.column_stack(T), miss


def load_extra_gpu_families(y):
    """Optional new arms from preds/part6_expand."""
    extras = []
    for tag in ["gb_cat_d2_d3_bag", "gb_xgb_cuda_d1_bag", "gb_resnet_mlp_bag", "gbx_bag3"]:
        if tag == "gbx_bag3":
            op, tp = ART / "gbx_bag3_oof.npy", ART / "gbx_bag3_full.npy"
        else:
            op, tp = PRED / f"oof_{tag}.npy", PRED / f"sub_{tag}.npy"
        if op.exists() and tp.exists():
            o, t = _load_npy(op), _load_npy(tp)
            extras.append((tag, o, t))
            print(f"[part6] +extra {tag} oof={roc_auc_score(y, o):.6f}", flush=True)
    return extras


def fit_test_ranks(P_oof, P_te, y, C, transform="rlogit"):
    X = rlogit(P_oof) if transform == "rlogit" else probit_rank_cols(P_oof)
    Xt = rlogit(P_te) if transform == "rlogit" else probit_rank_cols(P_te)
    m = stacker(C).fit(X, y)
    return rankdata(m.decision_function(Xt)) / len(Xt), m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SUBS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=[TARGET])[TARGET].astype(int).values
    te_ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])["id"].values
    folds = folds_y(y)

    names = BANK_OURS + BANK_EXT
    assert len(names) == 62, len(names)
    ok, P6_O, P6_T, miss = load_bank(names, y)
    print(f"[part6] loaded {len(ok)}/62 members; missing {len(miss)}", flush=True)
    (OUT / "part6_member_load.json").write_text(json.dumps({"ok": ok, "miss": miss}, indent=2))

    # Gate: reproduce on whatever we have; prefer exact 62
    nested, auc, fa = nested_rank_cv(P6_O, y, folds, C=0.01, transform="rlogit")
    print(
        f"[part6] REPRO stack n={P6_O.shape[1]} nested={auc:.6f} "
        f"(gate {P6_NESTED_GATE}) folds={np.round(fa, 5)}",
        flush=True,
    )
    gate_ok = abs(auc - P6_NESTED_GATE) < 5e-5 or auc >= P6_NESTED_GATE - 1e-4
    # sweep C on rlogit
    best = {"auc": -1}
    table = []
    for C in CS_RLOGIT:
        _, a, f = nested_rank_cv(P6_O, y, folds, C=C, transform="rlogit")
        row = {"transform": "rlogit", "C": C, "n": P6_O.shape[1], "oof": a, "folds": f,
               "delta_vs_p6": a - P6_NESTED_GATE}
        table.append(row)
        print(f"[part6] rlogit C={C} oof={a:.6f} Δp6={a - P6_NESTED_GATE:+.6f}", flush=True)
        if a > best.get("auc", best.get("oof", -1)):
            best = {**row, "auc": a, "kind": "rlogit"}
    for C in CS_PROBIT:
        _, a, f = nested_rank_cv(P6_O, y, folds, C=C, transform="probit")
        row = {"transform": "probit", "C": C, "n": P6_O.shape[1], "oof": a, "folds": f,
               "delta_vs_p6": a - P6_NESTED_GATE}
        table.append(row)
        print(f"[part6] probit C={C} oof={a:.6f} Δp6={a - P6_NESTED_GATE:+.6f}", flush=True)
        if a > best.get("auc", best.get("oof", -1)):
            best = {**row, "auc": a, "kind": "probit"}

    # Expand with GPU families + gbx_bag3
    extras = load_extra_gpu_families(y)
    if extras:
        Pe_O = np.column_stack([P6_O] + [e[1] for e in extras])
        Pe_T = np.column_stack([P6_T] + [e[2] for e in extras])
        enames = ok + [e[0] for e in extras]
        print(f"[part6] expanded n={Pe_O.shape[1]}", flush=True)
        best_e = {"auc": -1}
        for C in CS_RLOGIT:
            _, a, f = nested_rank_cv(Pe_O, y, folds, C=C, transform="rlogit")
            row = {"transform": "rlogit", "C": C, "n": Pe_O.shape[1], "oof": a, "folds": f,
                   "delta_vs_p6": a - P6_NESTED_GATE, "members": enames}
            table.append({**row, "expanded": True})
            print(f"[part6] EXP rlogit C={C} oof={a:.6f} Δp6={a - P6_NESTED_GATE:+.6f}", flush=True)
            if a > best_e.get("auc", best_e.get("oof", -1)):
                best_e = {**row, "auc": a, "kind": "rlogit"}
        for C in CS_PROBIT:
            _, a, f = nested_rank_cv(Pe_O, y, folds, C=C, transform="probit")
            row = {"transform": "probit", "C": C, "n": Pe_O.shape[1], "oof": a, "folds": f,
                   "delta_vs_p6": a - P6_NESTED_GATE, "members": enames}
            table.append({**row, "expanded": True})
            print(f"[part6] EXP probit C={C} oof={a:.6f} Δp6={a - P6_NESTED_GATE:+.6f}", flush=True)
            if a > best_e.get("auc", best_e.get("oof", -1)):
                best_e = {**row, "auc": a, "kind": "probit"}
    else:
        Pe_O, Pe_T, enames, best_e = P6_O, P6_T, ok, best

    # Rank-blend: best rlogit + best probit on expanded
    def best_of(transform, mat_O):
        bC, bA = CS_RLOGIT[0], -1
        cs = CS_RLOGIT if transform == "rlogit" else CS_PROBIT
        for C in cs:
            _, a, _ = nested_rank_cv(mat_O, y, folds, C=C, transform=transform)
            if a > bA:
                bC, bA = C, a
        return bC, bA

    c_r, a_r = best_of("rlogit", Pe_O)
    c_p, a_p = best_of("probit", Pe_O)
    sub_r, _ = fit_test_ranks(Pe_O, Pe_T, y, c_r, "rlogit")
    sub_p, _ = fit_test_ranks(Pe_O, Pe_T, y, c_p, "probit")
    # disagreement features: std across family columns if present
    fam_idx = []
    for i, n in enumerate(enames):
        if any(k in n for k in ("gb_bag5", "gb_cat", "ebm_i25", "realmlp", "glm_gen", "gbx_bag3", "gb_xgb", "gb_resnet")):
            fam_idx.append(i)
    if len(fam_idx) >= 2:
        std_te = Pe_T[:, fam_idx].std(axis=1)
        std_te = (rankdata(std_te) / (len(std_te) + 1))
    else:
        std_te = np.full(len(sub_r), 0.5)
    blend = 0.55 * sub_r + 0.35 * sub_p + 0.10 * std_te
    blend = rankdata(blend) / len(blend)

    # sub_A = expanded best single transform test ranks (use best_e)
    tr_e = best_e.get("transform", best_e.get("kind", "rlogit"))
    if tr_e == "rlogit":
        sub_A, model_A = fit_test_ranks(Pe_O, Pe_T, y, best_e["C"], "rlogit")
    else:
        sub_A, model_A = fit_test_ranks(Pe_O, Pe_T, y, best_e["C"], "probit")
    # prefer blend if we can measure OOF blend roughly
    # build OOF blend
    nest_r, _, _ = nested_rank_cv(Pe_O, y, folds, C=c_r, transform="rlogit")
    nest_p, _, _ = nested_rank_cv(Pe_O, y, folds, C=c_p, transform="probit")
    if len(fam_idx) >= 2:
        std_o = Pe_O[:, fam_idx].std(axis=1)
        std_o = np.zeros(len(y))
        for a, b in folds:
            std_o[b] = rankdata(Pe_O[b][:, fam_idx].std(axis=1)) / (len(b) + 1)
    else:
        std_o = np.full(len(y), 0.5)
    nest_blend = 0.55 * nest_r + 0.35 * nest_p + 0.10 * std_o
    # re-rank within folds for fair AUC
    nest_blend_r = np.zeros(len(y))
    for a, b in folds:
        nest_blend_r[b] = rankdata(nest_blend[b]) / len(b)
    auc_blend = float(roc_auc_score(y, nest_blend_r))
    print(f"[part6] rank-blend OOF={auc_blend:.6f} (rlogit={a_r:.6f} probit={a_p:.6f})", flush=True)
    if auc_blend >= best_e["auc"]:
        sub_A = blend
        primary_oof = auc_blend
        primary = "rank_blend"
    else:
        primary_oof = best_e["auc"]
        primary = f"{tr_e}_C{best_e['C']}"

    # sub_B = clean 62 Part-6 + gbx_bag3 + gb_cat if present
    b_names = list(ok)
    b_O, b_T = [P6_O], [P6_T]
    for tag, o, t in extras:
        if tag in ("gbx_bag3", "gb_cat_d2_d3_bag", "gb_xgb_cuda_d1_bag"):
            b_names.append(tag)
            b_O.append(o[:, None] if o.ndim == 1 else o)
            b_T.append(t[:, None] if t.ndim == 1 else t)
    if len(b_O) > 1:
        B_O = np.column_stack(b_O)
        B_T = np.column_stack(b_T)
    else:
        B_O, B_T = P6_O, P6_T
    # best C on B bank
    b_best_C, b_best_a = 0.01, -1
    for C in CS_RLOGIT:
        _, a, _ = nested_rank_cv(B_O, y, folds, C=C, transform="rlogit")
        if a > b_best_a:
            b_best_C, b_best_a = C, a
    sub_B, _ = fit_test_ranks(B_O, B_T, y, b_best_C, "rlogit")
    print(f"[part6] sub_B bank n={B_O.shape[1]} nested={b_best_a:.6f} C={b_best_C}", flush=True)

    pd.DataFrame({"id": te_ids, "is_fraud": sub_A}).to_csv(SUBS / "sub_A_part6_plus_gpu_families.csv", index=False)
    pd.DataFrame({"id": te_ids, "is_fraud": sub_B}).to_csv(SUBS / "sub_B_part6_repro_plus_cat_xgb.csv", index=False)

    # coefs top10 for primary expanded rlogit
    X = rlogit(Pe_O)
    m = stacker(best_e["C"] if tr_e == "rlogit" else c_r).fit(X, y)
    coef = pd.Series(m.coef_.ravel(), index=enames)
    top10 = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)

    report = {
        "repro_n": int(P6_O.shape[1]),
        "repro_nested_C01": auc,
        "repro_folds": fa,
        "gate": P6_NESTED_GATE,
        "gate_ok": bool(gate_ok or auc >= P6_NESTED_GATE - 1e-4),
        "missing": miss,
        "expanded_n": int(Pe_O.shape[1]),
        "primary": primary,
        "primary_oof": primary_oof,
        "sub_B_oof": b_best_a,
        "sub_B_C": b_best_C,
        "blend_oof": auc_blend,
        "table": table,
        "top10_coef": {k: float(v) for k, v in top10.items()},
        "files": [
            "submissions/sub_A_part6_plus_gpu_families.csv",
            "submissions/sub_B_part6_repro_plus_cat_xgb.csv",
        ],
        "policy": "NO_AUTO_SUBMIT",
    }
    (OUT / "part6_expand_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "table"}, indent=2), flush=True)
    print("[part6] DONE — NO SUBMIT", flush=True)
    return 0 if report["repro_n"] >= 50 else 2


if __name__ == "__main__":
    sys.exit(main())
