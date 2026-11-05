"""High-EV research loop vs true 76-member P_super_v3 (0.9521490). STRICT NO_SUBMIT.

Gates:
  Nested CV must beat V3_NESTED=0.9521490
  Fold-0 stack must beat V3_FOLD0=0.9511199
Dead ends logged into logs/overnight/high_ev_paradigm_loop.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
os.environ.setdefault("INPUT_DIR", str(ROOT))

from part6_repro_and_expand import ART, folds_y, rlogit  # noqa: E402
from part7_clean_superstack import load_bank_fixed  # noqa: E402
from part8_orth_superstack import to_stack_col  # noqa: E402
from part9_joint_boost_and_basis import build_8b_champ  # noqa: E402
from part10_upgrade_psuper import (  # noqa: E402
    C_OPT,
    HW_OPT,
    SW_OPT,
    dual_orth_foldwise,
    fit_test_stack,
    hill_climb,
    nest_hill,
    nest_newton,
    p_super_from_parts,
    rk,
    rho,
    single_orth_foldwise,
    wire_testbags,
)

T0 = time.time()
V3_NESTED = 0.9521489693921668
V3_FOLD0 = 0.9511199364214583
OUT = Path("logs/overnight/high_ev_paradigm_loop.json")
PRED = Path("preds")
POLICY = "NO_SUBMIT"


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


def d_e4(a, b):
    return (1.0 - rho(a, b)) * 1e4


def _jsonable(x):
    """Strip ndarray payloads and coerce numpy scalars for JSON."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("oof", "test") and isinstance(v, np.ndarray):
                continue
            out[k] = _jsonable(v)
        return out
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.size <= 64:
            return x.tolist()
        return f"<ndarray shape={list(x.shape)} dtype={str(x.dtype)}>"
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def save_state(state):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state["runtime_s"] = time.time() - T0
    state["policy"] = POLICY
    OUT.write_text(json.dumps(_jsonable(state), indent=2))


def mark_dead(state, name, reason, metrics=None):
    entry = {"name": name, "reason": reason, "metrics": metrics or {}}
    state.setdefault("dead_ends", []).append(entry)
    log(f"DEAD_END: {name} — {reason}")
    save_state(state)


# ── helpers for P_super_v3 bank (74 + glm_quad + orth) ─────────────────────
def build_v3_bank(y, folds):
    names8, O8, T8 = build_8b_champ(y, folds)
    T8w, wire_info = wire_testbags(names8, T8)
    _, G_o, G_t, _ = load_bank_fixed(["mega:glm_gen"], y)
    glm_o, glm_t = G_o[:, 0], G_t[:, 0]
    gq_o = np.load("preds/ladder/glm_quad_oof.npy").ravel().astype(float)
    gq_t = np.load("preds/ladder/glm_quad_test.npy").ravel().astype(float)
    e_o, e_t = single_orth_foldwise(gq_o, gq_t, glm_o, glm_t, folds)
    O = np.column_stack([O8, rk(gq_o), to_stack_col(e_o, True)])
    T = np.column_stack([T8w, rk(gq_t), to_stack_col(e_t, True)])
    names = names8 + ["glm_quad", "glm_quad_orth_glm"]
    assert O.shape[1] == 76, O.shape
    return names, O, T, wire_info, glm_o, glm_t


def nested_psuper(O, T, y, folds, gb5_o, bag_o, gb5_t_bag, bag_t):
    st, auc_st = nest_newton(O, y, folds)
    hc, _ = nest_hill(O, y, folds)
    sup_o = 0.50 * rk(gb5_o) + 0.50 * rk(bag_o)
    p, auc, fa, auc_mix = p_super_from_parts(st, hc, sup_o, y, folds)
    a0, b0 = folds[0]
    fold0 = float(roc_auc_score(y[b0], p[b0]))
    st_te, hc_te = fit_test_stack(O, T, y)
    mix_te = rk((1.0 - HW_OPT) * st_te + HW_OPT * hc_te)
    sup_t = 0.50 * rk(gb5_t_bag) + 0.50 * rk(bag_t)
    te = rk((1.0 - SW_OPT) * mix_te + SW_OPT * rk(sup_t))
    return {
        "nested": auc,
        "fold0": fold0,
        "newton": auc_st,
        "mix": auc_mix,
        "folds": fa,
        "oof": p,
        "test": te,
        "delta_nested": auc - V3_NESTED,
        "delta_fold0": fold0 - V3_FOLD0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LEVER 1 — unbundling recon
# ═══════════════════════════════════════════════════════════════════════════
def lever1(state, y, folds, gb5_o, gb5_t, v3_oof, names_v3, O_v3, T_v3, bank_ctx):
    log("=" * 72)
    log("LEVER 1: unpack external OOF/Test vectors")
    log("=" * 72)
    a0, b0 = folds[0]
    candidates = []

    # masaya_stack
    ms = Path("external_bank/masaya_stack")
    pairs = [
        ("masaya_stack_main", ms / "oof.npy", ms / "test.npy"),
    ]
    # ablations — OOF only (no test) → skip nested inject, report only
    for name in (
        "without_public_models_oof.npy",
        "with_shallow_trees_oof.npy",
        "without_gpt2_target_first_oof.npy",
        "two_member_oof.npy",
    ):
        p = ms / name
        if p.exists():
            pairs.append((f"masaya_{p.stem}", p, None))

    # dhananjay
    dhan = Path("external_bank/dhananjay_part4_out/oof")
    for oof_p in sorted(dhan.glob("*_oof.npy")):
        if oof_p.name.startswith("._"):
            continue
        te_p = dhan / oof_p.name.replace("_oof.npy", "_test.npy")
        candidates_name = f"dhan_{oof_p.stem}"
        pairs.append((candidates_name, oof_p, te_p if te_p.exists() else None))

    # subh / amanatar — submission only (test)
    subh = Path("external_bank/subh_ensemble_out/submission.csv")
    aman = Path("external_bank/amanatar_sovereign/submission.csv")
    # amanatar rank3 CSVs (test only)
    for csv in Path("external_bank/amanatar_rank3_artifacts").glob("*.csv"):
        pairs.append((f"amanatar_{csv.stem}", None, csv))

    if subh.exists():
        pairs.append(("subh_ensemble_sub", None, subh))
    if aman.exists():
        pairs.append(("amanatar_sovereign_sub", None, aman))

    sub_ids = pd.read_csv("data/sample_submission.csv")["id"].to_numpy()
    reports = []
    inject = []

    for name, oof_path, te_path in pairs:
        # skip macOS AppleDouble resource forks
        paths = [x for x in (oof_path, te_path) if x is not None]
        if any(Path(x).name.startswith("._") for x in paths) or "._" in name:
            continue
        rec = {"name": name, "oof_path": str(oof_path) if oof_path else None, "test_path": str(te_path) if te_path else None}
        try:
            oof = te = None
            if oof_path is not None:
                oof = np.load(oof_path, allow_pickle=False).ravel().astype(float)
                if oof.shape != (len(y),):
                    rec["status"] = f"bad_oof_shape={oof.shape}"
                    reports.append(rec)
                    continue
                rec["oof_auc"] = float(roc_auc_score(y, oof))
                rec["rho_v3"] = rho(oof, v3_oof)
                rec["rho_gb5_oof"] = rho(oof, gb5_o)
            if te_path is not None:
                if str(te_path).endswith(".csv"):
                    df = pd.read_csv(te_path)
                    col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
                    if "id" in df.columns:
                        te = df.set_index("id").loc[sub_ids, col].to_numpy(float)
                    else:
                        te = df[col].to_numpy(float)
                else:
                    te = np.load(te_path, allow_pickle=False).ravel().astype(float)
                if te.shape[0] != len(sub_ids):
                    rec["status"] = f"bad_test_shape={te.shape}"
                    reports.append(rec)
                    continue
                rec["rho_gb5_test"] = rho(te, gb5_t)
            if oof is not None and te is not None:
                rec["delta_rho"] = abs(rec["rho_gb5_oof"] - rec["rho_gb5_test"])
                rec["aligned"] = bool(rec["delta_rho"] < 0.003)
                # duplicate vs existing v3 members: max |pearson| on ranks vs each col
                ro = rk(oof)
                max_r = max(abs(float(np.corrcoef(ro, O_v3[:, j])[0, 1])) for j in range(O_v3.shape[1]))
                rec["max_r_vs_v3_bank"] = max_r
                rec["duplicate"] = bool(max_r >= 0.9995)
                if rec["aligned"] and not rec["duplicate"]:
                    inject.append((name, oof, te, rec))
                    rec["status"] = "candidate_inject"
                else:
                    rec["status"] = "skip_align_or_dup"
            elif oof is not None:
                rec["status"] = "oof_only_no_test"
            else:
                rec["status"] = "test_only"
            reports.append(rec)
            log(
                f"  {name}: status={rec['status']} "
                f"oof_auc={rec.get('oof_auc')} |Δρ|={rec.get('delta_rho')} "
                f"max_r={rec.get('max_r_vs_v3_bank')}"
            )
        except Exception as e:
            rec["status"] = f"error:{e}"
            reports.append(rec)
            log(f"  {name}: ERROR {e}")

    # Nested inject each clean candidate + joint all
    inject_results = []
    gb5_o = bank_ctx["gb5_o"]
    bag_o = bank_ctx["bag_o"]
    gb5_t_bag = bank_ctx["gb5_t_bag"]
    bag_t = bank_ctx["bag_t"]

    for name, oof, te, meta in inject:
        O2 = np.column_stack([O_v3, rk(oof)])
        T2 = np.column_stack([T_v3, rk(te)])
        res = nested_psuper(O2, T2, y, folds, gb5_o, bag_o, gb5_t_bag, bag_t)
        row = {
            "name": name,
            "oof_auc": meta["oof_auc"],
            "delta_rho": meta["delta_rho"],
            "nested": res["nested"],
            "fold0": res["fold0"],
            "delta_nested": res["delta_nested"],
            "delta_fold0": res["delta_fold0"],
            "promoted": bool(res["delta_nested"] > 1e-7 and res["delta_fold0"] > 0),
        }
        inject_results.append(row)
        log(
            f"  INJECT {name}: nested={res['nested']:.7f} (Δ={res['delta_nested']:+.3e}) "
            f"fold0={res['fold0']:.7f} (Δ={res['delta_fold0']:+.3e}) "
            f"{'PROMOTE' if row['promoted'] else 'DEAD'}"
        )
        if not row["promoted"]:
            mark_dead(
                state,
                f"L1_inject:{name}",
                "no lift vs P_super_v3 nested/fold0",
                row,
            )

    # joint inject of all promoted
    promoted = [x for x in inject if any(r["name"] == x[0] and r["promoted"] for r in inject_results)]
    joint = None
    if promoted:
        O2 = np.column_stack([O_v3] + [rk(o) for _, o, _, _ in promoted])
        T2 = np.column_stack([T_v3] + [rk(t) for _, _, t, _ in promoted])
        joint = nested_psuper(O2, T2, y, folds, gb5_o, bag_o, gb5_t_bag, bag_t)
        log(f"  JOINT promoted inject nested={joint['nested']:.7f} Δ={joint['delta_nested']:+.3e}")

    state["lever1"] = {
        "amanatar_dataset_sources": ["amanatar/s6e11-rank3-artifacts"],
        "amanatar_rank3_files": sorted(p.name for p in Path("external_bank/amanatar_rank3_artifacts").glob("*")),
        "amanatar_note": "dataset ships only fusion CSVs (unbundled/triad/sovereign); 117 raw OOFs NOT published",
        "reports": reports,
        "inject_results": inject_results,
        "joint_promoted": joint,
        "lb_scan_note": "see state['lb_top']",
    }
    save_state(state)
    return [x for x in inject if any(r["name"] == x[0] and r.get("promoted") for r in inject_results)]


# ═══════════════════════════════════════════════════════════════════════════
# LEVER 2 — glm_cubic_synergy
# ═══════════════════════════════════════════════════════════════════════════
def lever2(state, y, folds, bank_ctx, names_v3, O_v3, T_v3):
    log("=" * 72)
    log("LEVER 2: glm_cubic_synergy (Fold-0 gate first)")
    log("=" * 72)
    import gboost as gb

    gpt_o = np.load("external_bank/masaya_gpt2/oof.npy").ravel().astype(float)
    gpt_t = np.load("external_bank/masaya_gpt2/test.npy").ravel().astype(float)
    gq_o = np.load("preds/ladder/glm_quad_oof.npy").ravel().astype(float)
    gq_t = np.load("preds/ladder/glm_quad_test.npy").ravel().astype(float)

    def te_stage1_fold(fi, a, b):
        log(f"  TE fold {fi} …")
        Ea, Eb, Et = gb.te_fold(a, b, smooth=20.0, seed=0)
        if gb.GEN:
            Ga, Gb, Gt = gb.gen_fold(a, b, seed=0)
            Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        ybar = float(y[a].mean())
        scz = StandardScaler().fit(rlogit(Ea) - rlogit(ybar))
        Za = scz.transform(rlogit(Ea) - rlogit(ybar))
        Zb = scz.transform(rlogit(Eb) - rlogit(ybar))
        Zt = scz.transform(rlogit(Et) - rlogit(ybar))
        Xa = np.hstack([Za, gb.BASE[a]])
        Xb = np.hstack([Zb, gb.BASE[b]])
        Xt = np.hstack([Zt, gb.BASE[gb.TEST]])
        m = LogisticRegression(C=0.3, max_iter=200, solver="newton-cholesky").fit(Xa, y[a])
        log(f"    D={Za.shape[1]} fold{fi} stage1 done")
        return {
            "a": a, "b": b, "Za": Za, "Zb": Zb, "Zt": Zt, "fi": fi,
            "z_b": m.decision_function(Xb), "z_t": m.decision_function(Xt),
        }

    # Fold-0 first for gate (avoid 5× TE cost on dead ends)
    parts = []
    z_stage1 = np.zeros(len(y))
    zte_f = []
    p0 = te_stage1_fold(0, *folds[0])
    parts.append(p0)
    z_stage1[p0["b"]] = p0["z_b"]
    zte_f.append(p0["z_t"])
    # provisional joint using fold0 stage1 + full gpt2 (fold0 gate only needs b0)
    z_joint = z_stage1 + rlogit(rk(gpt_o))
    z_te_joint = None  # filled after full TE if promoted

    # Fold-0 screen expanded basis
    p0 = parts[0]
    a, b, Za, Zb, Zt = p0["a"], p0["b"], p0["Za"], p0["Zb"], p0["Zt"]
    d = Za.shape[1]
    resid_quad = y[a] - gq_o[a]  # residual vs glm_quad probs if calibrated; use decision-ish
    # glm_quad is probability — convert
    resid = y[a] - gq_o[a]
    Zc = Za - Za.mean(0)
    r = resid - resid.mean()
    wr = Zc * r[:, None]

    # (a) top-1500 2-way by |mean(Zj*Zk*r)|
    scores2, pairs2 = [], []
    for j in range(d):
        s = (wr[:, j : j + 1] * Zc[:, j + 1 :]).mean(0)
        for t, k in enumerate(range(j + 1, d)):
            scores2.append(abs(float(s[t])))
            pairs2.append((j, k))
    pairs2 = [pairs2[i] for i in np.argsort(-np.asarray(scores2))[:1500]]

    # (b) top-400 cubics among top-30 |coef|-proxy columns (corr with resid)
    col_score = np.abs((Zc * r[:, None]).mean(0))
    top30 = np.argsort(-col_score)[:30]
    scores3, trips = [], []
    for ii, j in enumerate(top30):
        for jj, k in enumerate(top30[ii + 1 :], ii + 1):
            for l in top30[jj + 1 :]:
                s = float(np.mean(Zc[:, j] * Zc[:, k] * Zc[:, l] * r))
                scores3.append(abs(s))
                trips.append((int(j), int(k), int(l)))
    trips = [trips[i] for i in np.argsort(-np.asarray(scores3))[:400]] if scores3 else []

    # (c) count-confidence: z_j * std_scaled(log1p(KCNT_j)) — use first min(d, KCNT.shape[1]) cols
    kcnt = gb.KCNT  # n_train+test? check
    # KCNT is train+test stacked in gboost — index carefully
    n_tr = len(y)
    KCNT_tr = gb.KCNT[:n_tr]
    KCNT_te = gb.KCNT[n_tr:]
    d_k = min(d, KCNT_tr.shape[1])

    def build_basis(Z, idx, kcnt_rows):
        Z = np.asarray(Z, np.float32)
        n = Z.shape[0]
        n2, n3 = len(pairs2), len(trips)
        out = np.empty((n, n2 + n3 + d_k + 2 * d), np.float32)
        if n2:
            jj = np.fromiter((j for j, _ in pairs2), np.int32, n2)
            kk = np.fromiter((k for _, k in pairs2), np.int32, n2)
            out[:, :n2] = Z[:, jj] * Z[:, kk]
        if n3:
            jj = np.fromiter((j for j, _, _ in trips), np.int32, n3)
            kk = np.fromiter((k for _, k, _ in trips), np.int32, n3)
            ll = np.fromiter((l for _, _, l in trips), np.int32, n3)
            out[:, n2 : n2 + n3] = Z[:, jj] * Z[:, kk] * Z[:, ll]
        c = n2 + n3
        kc_mat = np.log1p(kcnt_rows[idx, :d_k].astype(np.float32))
        kc_mat = (kc_mat - kc_mat.mean(0)) / (kc_mat.std(0) + 1e-12)
        out[:, c : c + d_k] = Z[:, :d_k] * kc_mat
        c += d_k
        out[:, c : c + d] = np.sign(Z) * np.log1p(np.abs(Z))
        c += d
        out[:, c : c + d] = np.tanh(Z / 2.0)
        return out

    # C sweep on Fold0 — subsample train fold for speed/memory; score on full b
    log(f"  building Fold0 cubic basis (pairs={len(pairs2)} trips={len(trips)}) …")
    Ia = build_basis(Za, a, KCNT_tr)
    Ib = build_basis(Zb, b, KCNT_tr)
    log(f"  basis shape Ia={Ia.shape} Ib={Ib.shape} dtype={Ia.dtype}")
    scI = StandardScaler()
    rng = np.random.RandomState(0)
    n_sub = min(120_000, Ia.shape[0])
    sub = rng.choice(Ia.shape[0], size=n_sub, replace=False)
    Ia_sub = scI.fit_transform(Ia[sub])
    Ib_s = scI.transform(Ib)
    y_sub = y[a][sub]
    del Ia  # free ~5GB
    best_c, best_auc = None, -1.0
    for C in (0.0005, 0.001, 0.002, 0.005):
        m = LogisticRegression(C=C, max_iter=200, solver="newton-cholesky").fit(Ia_sub, y_sub)
        pred = expit(z_joint[b] + 0.5 * m.decision_function(Ib_s))
        auc = float(roc_auc_score(y[b], pred))
        log(f"  Fold0 C={C} cubic AUC={auc:.6f} (glm_quad fold0={roc_auc_score(y[b], gq_o[b]):.6f})")
        if auc > best_auc:
            best_auc, best_c = auc, C

        gq_f0 = float(roc_auc_score(y[b], gq_o[b]))
    gate_solo = best_auc > gq_f0 + 1e-6
    log(f"  best C={best_c} Fold0={best_auc:.6f} vs glm_quad Fold0={gq_f0:.6f} solo_gate={gate_solo}")

    # quick Fold0 stack probe: replace glm_quad col with cubic fold0 scores in a temporary full oof
    # Build full 5-fold cubic if solo gate OR if close — user said must beat glm_quad Fold0 AND lift stack Fold0
    # First build fold0-only cubic into a full vector filled with gq elsewhere for stack probe
    cubic_oof = gq_o.copy()
    m = LogisticRegression(C=best_c, max_iter=200, solver="newton-cholesky").fit(Ia_sub, y_sub)
    cubic_oof[b] = expit(z_joint[b] + 0.5 * m.decision_function(Ib_s))

    # Fold0 stack probe: swap glm_quad column in O_v3
    ni = {n: i for i, n in enumerate(names_v3)}
    O_probe = O_v3.copy()
    O_probe[:, ni["glm_quad"]] = rk(cubic_oof)
    # recompute orth vs glm for probe fold0 only is imperfect; still use for Fold0 stack score
    st_p, _ = nest_newton(O_probe, y, folds)
    hc_p, _ = nest_hill(O_probe, y, folds)
    sup_o = 0.50 * rk(bank_ctx["gb5_o"]) + 0.50 * rk(bank_ctx["bag_o"])
    p_probe, _, _, _ = p_super_from_parts(st_p, hc_p, sup_o, y, folds)
    fold0_stack = float(roc_auc_score(y[b], p_probe[b]))
    log(f"  Fold0 stack with cubic-swap={fold0_stack:.7f} vs V3_FOLD0={V3_FOLD0:.7f}")

    lever2_res = {
        "best_C": best_c,
        "fold0_cubic_auc": best_auc,
        "fold0_glm_quad_auc": gq_f0,
        "fold0_stack_probe": fold0_stack,
        "solo_gate": gate_solo,
        "stack_gate": fold0_stack > V3_FOLD0,
        "n_pairs2": len(pairs2),
        "n_trips": len(trips),
        "basis_dim": int(Ib_s.shape[1]),
    }

    if not (gate_solo and fold0_stack > V3_FOLD0):
        mark_dead(
            state,
            "L2_glm_cubic_synergy",
            "failed Fold0 gate vs glm_quad and/or P_super_v3 Fold0",
            lever2_res,
        )
        state["lever2"] = {**lever2_res, "promoted": False}
        save_state(state)
        return None

    # Full 5-fold + refit
    log("  Fold0 PASSED — running remaining TE folds + 5-fold cubic …")
    for fi, (a, b) in enumerate(folds):
        if fi == 0:
            continue
        p = te_stage1_fold(fi, a, b)
        parts.append(p)
        z_stage1[p["b"]] = p["z_b"]
        zte_f.append(p["z_t"])
    z_joint = z_stage1 + rlogit(rk(gpt_o))
    z_te_joint = np.mean(zte_f, 0) + rlogit(rk(gpt_t))
    oof = np.zeros(len(y))
    te_fs = []
    for p in parts:
        a, b, Za, Zb, Zt = p["a"], p["b"], p["Za"], p["Zb"], p["Zt"]
        log(f"    cubic fit fold {p['fi']} …")
        Ia = build_basis(Za, a, KCNT_tr)
        scI = StandardScaler().fit(Ia)
        # subsample fit for memory; evaluate on full fold-b / test
        rng = np.random.RandomState(100 + p["fi"])
        n_sub = min(200_000, Ia.shape[0])
        sub = rng.choice(Ia.shape[0], size=n_sub, replace=False)
        m = LogisticRegression(C=best_c, max_iter=200, solver="newton-cholesky").fit(
            scI.transform(Ia[sub]), y[a][sub]
        )
        del Ia
        Ib = scI.transform(build_basis(Zb, b, KCNT_tr))
        It = scI.transform(build_basis(Zt, np.arange(len(KCNT_te)), KCNT_te))
        oof[b] = expit(z_joint[b] + 0.5 * m.decision_function(Ib))
        te_fs.append(expit(z_te_joint + 0.5 * m.decision_function(It)))
        del Ib, It
    te = np.mean(te_fs, 0)
    np.save("preds/glm_cubic_oof.npy", oof.astype(np.float32))
    np.save("preds/glm_cubic_test.npy", te.astype(np.float32))
    oof_auc = float(roc_auc_score(y, oof))
    log(f"  glm_cubic OOF={oof_auc:.6f}")

    # Replace glm_quad in v3 bank with cubic (+orth)
    glm_o, glm_t = bank_ctx["glm_o"], bank_ctx["glm_t"]
    e_o, e_t = single_orth_foldwise(oof, te, glm_o, glm_t, folds)
    O2 = O_v3.copy()
    T2 = T_v3.copy()
    O2[:, ni["glm_quad"]] = rk(oof)
    T2[:, ni["glm_quad"]] = rk(te)
    O2[:, ni["glm_quad_orth_glm"]] = to_stack_col(e_o, True)
    T2[:, ni["glm_quad_orth_glm"]] = to_stack_col(e_t, True)
    res = nested_psuper(
        O2, T2, y, folds, bank_ctx["gb5_o"], bank_ctx["bag_o"], bank_ctx["gb5_t_bag"], bank_ctx["bag_t"]
    )
    promoted = res["delta_nested"] > 1e-7
    lever2_res.update({"oof_auc": oof_auc, "nested_res": res, "promoted": promoted})
    if not promoted:
        mark_dead(state, "L2_glm_cubic_full", "5-fold nested did not beat P_super_v3", lever2_res)
    state["lever2"] = lever2_res
    save_state(state)
    return (oof, te, res) if promoted else None


# ═══════════════════════════════════════════════════════════════════════════
# LEVER 3 — boost from z_Super
# ═══════════════════════════════════════════════════════════════════════════
def lever3(state, y, folds, bank_ctx, names_v3, O_v3, T_v3, cubic=None):
    log("=" * 72)
    log("LEVER 3: gb_superquad from z_Super (Fold-0 gate)")
    log("=" * 72)
    import gboost as gb

    gq_o = np.load("preds/ladder/glm_quad_oof.npy").ravel().astype(float)
    gq_t = np.load("preds/ladder/glm_quad_test.npy").ravel().astype(float)
    if cubic is not None:
        gq_o, gq_t = cubic[0], cubic[1]
        log("  using glm_cubic as quad source")
    gpt_o = np.load("external_bank/masaya_gpt2/oof.npy").ravel().astype(float)
    gpt_t = np.load("external_bank/masaya_gpt2/test.npy").ravel().astype(float)
    _, G_o, G_t, _ = load_bank_fixed(["mega:glm_gen"], y)
    glm_o, glm_t = G_o[:, 0], G_t[:, 0]

    # 3-source super-margin OOF via nested LR on ranks
    X3_o = np.column_stack([rlogit(rk(glm_o)), rlogit(rk(gq_o)), rlogit(rk(gpt_o))])
    X3_t = np.column_stack([rlogit(rk(glm_t)), rlogit(rk(gq_t)), rlogit(rk(gpt_t))])
    z_super = np.zeros(len(y))
    for a, b in folds:
        m = LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky").fit(X3_o[a], y[a])
        z_super[b] = m.decision_function(X3_o[b])
    m_full = LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky").fit(X3_o, y)
    z_super_te = m_full.decision_function(X3_t)
    super_auc = float(roc_auc_score(y, z_super))
    log(f"  z_Super 3-source OOF AUC={super_auc:.6f}")

    # Fold0 LGB from init_score
    a0, b0 = folds[0]
    Ea, Eb, Et = gb.te_fold(a0, b0, smooth=20.0, seed=0)
    if gb.GEN:
        Ga, Gb, Gt = gb.gen_fold(a0, b0, seed=0)
        Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
    # gboost tree view
    sc = StandardScaler().fit(rlogit(Ea))
    def D(E, i):
        return np.hstack([sc.transform(rlogit(E)), gb.BASE[i]])

    Xa, Xb = D(Ea, a0), D(Eb, b0)
    # disagreement cols
    def diffs(idx, o_glm, o_gq, o_gpt):
        return np.column_stack(
            [
                rlogit(rk(o_gq))[idx] - rlogit(rk(o_glm))[idx],
                rlogit(rk(o_gpt))[idx] - rlogit(rk(o_glm))[idx],
                rlogit(rk(o_gpt))[idx] - rlogit(rk(o_gq))[idx],
            ]
        )

    Xa = np.hstack([Xa, diffs(a0, glm_o, gq_o, gpt_o)])
    Xb = np.hstack([Xb, diffs(b0, glm_o, gq_o, gpt_o)])
    za, zb = z_super[a0], z_super[b0]

    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=4,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(Xa, y[a0], init_score=za)
    # raw margin = init + tree
    raw_b = zb + clf.predict(Xb, raw_score=True)
    pred_b = expit(raw_b)
    auc_f0 = float(roc_auc_score(y[b0], pred_b))
    # baselines
    gb_s0_path = ART / "gb_s0_oof.npy"
    gb_s0_auc_f0 = float(roc_auc_score(y[b0], np.load(gb_s0_path).ravel()[b0])) if gb_s0_path.exists() else 0.95039
    log(f"  gb_superquad_s0 Fold0 AUC={auc_f0:.6f} vs gb_s0 Fold0≈{gb_s0_auc_f0:.6f}")

    # stack Fold0 probe: put scores into full vector
    tmp = np.load("preds/gb_gpt2_bag5_oof.npy").ravel().astype(float).copy()
    tmp[b0] = pred_b
    # blend into gb_super for fold0
    sup_new = 0.50 * rk(bank_ctx["gb5_o"]) + 0.50 * rk(tmp)
    # rebuild p_super fold0 using existing nest of v3 but swap super — approximate via mixing
    v3_oof = np.load("preds/P_super_v3_oof.npy").ravel()
    # extract mix ≈ remove 8% old super: rough probe — compare fold0 of new super alone vs gb_super
    old_sup = 0.50 * rk(bank_ctx["gb5_o"]) + 0.50 * rk(bank_ctx["bag_o"])
    # Fold0 stack: replace 8% component
    # v3 = 0.92*mix + 0.08*sup → new = 0.92*mix + 0.08*sup_new; mix≈(v3-0.08*sup)/0.92
    mix_approx = (v3_oof - 0.08 * rk(old_sup)) / 0.92
    p_new = rk(0.92 * mix_approx + 0.08 * rk(sup_new))
    fold0_stack = float(roc_auc_score(y[b0], p_new[b0]))
    log(f"  Fold0 stack probe with new gb_super={fold0_stack:.7f} vs {V3_FOLD0:.7f}")

    gate = (auc_f0 > gb_s0_auc_f0 + 1e-5) and (fold0_stack > V3_FOLD0)
    res = {
        "z_super_oof_auc": super_auc,
        "fold0_lgb_auc": auc_f0,
        "fold0_gb_s0_auc": gb_s0_auc_f0,
        "fold0_stack_probe": fold0_stack,
        "promoted": False,
        "gate": gate,
    }
    if not gate:
        mark_dead(state, "L3_gb_superquad", "Fold0 gate failed vs gb_s0 / P_super_v3 Fold0", res)
        state["lever3"] = res
        save_state(state)
        return None

    log("  Fold0 PASSED — 5-seed bag (this may take a while) …")
    # Full 5 seeds for LGB only (time box)
    oof = np.zeros(len(y))
    te_acc = []
    for seed_i, (a, b) in enumerate(folds):
        Ea, Eb, Et = gb.te_fold(a, b, smooth=20.0, seed=seed_i)
        if gb.GEN:
            Ga, Gb, Gt = gb.gen_fold(a, b, seed=seed_i)
            Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        sc = StandardScaler().fit(rlogit(Ea))
        def D(E, i, sc=sc):
            return np.hstack([sc.transform(rlogit(E)), gb.BASE[i]])
        Xa = np.hstack([D(Ea, a), diffs(a, glm_o, gq_o, gpt_o)])
        Xb = np.hstack([D(Eb, b), diffs(b, glm_o, gq_o, gpt_o)])
        Xt = np.hstack([D(Et, gb.TEST), diffs(np.arange(len(gq_t)), glm_t, gq_t, gpt_t)])
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=4, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42 + seed_i, n_jobs=-1, verbosity=-1,
        )
        clf.fit(Xa, y[a], init_score=z_super[a])
        oof[b] = expit(z_super[b] + clf.predict(Xb, raw_score=True))
        te_acc.append(expit(z_super_te + clf.predict(Xt, raw_score=True)))
        log(f"    seed/fold {seed_i} done")
    te = np.mean(te_acc, 0)
    np.save("preds/gb_superquad_bag5_oof.npy", oof.astype(np.float32))
    np.save("preds/gb_superquad_bag5_full.npy", te.astype(np.float32))
    bag_auc = float(roc_auc_score(y, oof))
    log(f"  gb_superquad_bag5 OOF={bag_auc:.6f}")

    # Update gb_super inside v3 recipe
    st, _ = nest_newton(O_v3, y, folds)
    hc, _ = nest_hill(O_v3, y, folds)
    sup_o = 0.50 * rk(bank_ctx["gb5_o"]) + 0.50 * rk(oof)
    p, auc, fa, _ = p_super_from_parts(st, hc, sup_o, y, folds)
    fold0 = float(roc_auc_score(y[b0], p[b0]))
    promoted = auc > V3_NESTED + 1e-7
    res.update(
        {
            "bag5_oof_auc": bag_auc,
            "nested": auc,
            "fold0": fold0,
            "delta_nested": auc - V3_NESTED,
            "promoted": promoted,
            "oof": oof,
            "test": te,
        }
    )
    if not promoted:
        mark_dead(state, "L3_gb_superquad_full", "5-fold nested did not beat P_super_v3", {k: v for k, v in res.items() if k not in ("oof", "test")})
    state["lever3"] = {k: v for k, v in res.items() if k not in ("oof", "test")}
    save_state(state)
    return (oof, te, auc) if promoted else None


# ═══════════════════════════════════════════════════════════════════════════
# LEVER 4 — regime-conditioned stacker
# ═══════════════════════════════════════════════════════════════════════════
def lever4(state, y, folds, bank_ctx, names_v3, O_v3, T_v3):
    log("=" * 72)
    log("LEVER 4: row-wise uncertainty meta-stacker")
    log("=" * 72)
    import gboost as gb

    n = len(y)
    nt = T_v3.shape[0]
    # r_rare from float KCNT — first 6 floatish keys in KMAT if available
    kcnt_tr = gb.KCNT[:n]
    kcnt_te = gb.KCNT[n:]
    n_float = min(6, kcnt_tr.shape[1])
    r_rare_tr = (kcnt_tr[:, :n_float] <= 3).mean(1)
    r_rare_te = (kcnt_te[:, :n_float] <= 3).mean(1)

    def stdize(tr, te):
        mu, sd = tr.mean(), tr.std() + 1e-12
        return (tr - mu) / sd, (te - mu) / sd

    r_rare_tr, r_rare_te = stdize(r_rare_tr, r_rare_te)

    # family logits from bank / files
    ni = {n: i for i, n in enumerate(names_v3)}
    def col_or_file(name, path, Tpath=None):
        if name in ni:
            return O_v3[:, ni[name]], T_v3[:, ni[name]]
        o = rk(np.load(path).ravel())
        t = rk(np.load(Tpath).ravel()) if Tpath else np.full(nt, 0.5)
        return o, t

    fam = {}
    fam["gb_bag5"] = col_or_file("gb_bag5", ART / "gb_bag5_oof.npy", ART / "gb_bag5_full.npy")
    # gpt2 is named gpt2 in P7 portion
    gpt_name = "gpt2" if "gpt2" in ni else None
    if gpt_name:
        fam["masaya_gpt2"] = (O_v3[:, ni[gpt_name]], T_v3[:, ni[gpt_name]])
    else:
        fam["masaya_gpt2"] = (rk(np.load("external_bank/masaya_gpt2/oof.npy").ravel()), rk(np.load("external_bank/masaya_gpt2/test.npy").ravel()))
    fam["glm_quad"] = (O_v3[:, ni["glm_quad"]], T_v3[:, ni["glm_quad"]])
    ebm_name = "ebm_i25_bag4" if "ebm_i25_bag4" in ni else None
    if ebm_name:
        fam["ebm_i25"] = (O_v3[:, ni[ebm_name]], T_v3[:, ni[ebm_name]])
    else:
        fam["ebm_i25"] = fam["gb_bag5"]
    gbn = "gbn_bag5_w512" if "gbn_bag5_w512" in ni else None
    if gbn:
        fam["gbn_w512"] = (O_v3[:, ni[gbn]], T_v3[:, ni[gbn]])
    # ektarr if present else skip
    fam_keys = ["gb_bag5", "masaya_gpt2", "glm_quad", "ebm_i25", "gbn_w512"]

    mats_tr = np.column_stack([rlogit(fam[k][0]) for k in fam_keys[:5]])
    mats_te = np.column_stack([rlogit(fam[k][1]) for k in fam_keys[:5]])
    r_disp_tr = mats_tr.std(1)
    r_disp_te = mats_te.std(1)
    r_disp_tr, r_disp_te = stdize(r_disp_tr, r_disp_te)

    v3_o = np.load("preds/P_super_v3_oof.npy").ravel()
    v3_t = np.load("preds/P_super_v3_test.npy").ravel()
    r_tail_tr = np.abs(rlogit(rk(v3_o)))
    r_tail_te = np.abs(rlogit(rk(v3_t)))
    r_tail_tr, r_tail_te = stdize(r_tail_tr, r_tail_te)

    regimes_tr = {"r_rare": r_rare_tr, "r_disp": r_disp_tr, "r_tail": r_tail_tr}
    regimes_te = {"r_rare": r_rare_te, "r_disp": r_disp_te, "r_tail": r_tail_te}
    mod_fams = ["gb_bag5", "masaya_gpt2", "glm_quad", "ebm_i25", "gbn_w512"]

    extra_o, extra_t = [], []
    for rk_name in regimes_tr:
        for fk in mod_fams:
            extra_o.append(regimes_tr[rk_name] * rlogit(fam[fk][0]))
            extra_t.append(regimes_te[rk_name] * rlogit(fam[fk][1]))
    Eo = np.column_stack(extra_o)
    Et = np.column_stack(extra_t)
    # stacker on rlogit(O_v3) + extras (already logit-space mods)
    Xo = np.hstack([rlogit(O_v3), Eo])
    Xt = np.hstack([rlogit(T_v3), Et])

    best = {"nested": -1, "C": None}
    for C in (0.003, 0.005, 0.008):
        nested = np.zeros(n)
        for a, b in folds:
            m = LogisticRegression(C=C, tol=1e-8, max_iter=300, solver="newton-cholesky").fit(Xo[a], y[a])
            nested[b] = rk(m.decision_function(Xo[b]))
        auc = float(roc_auc_score(y, nested))
        log(f"  regime newton C={C} nested={auc:.7f} Δ={auc - V3_NESTED:+.3e}")
        if auc > best["nested"]:
            best = {"nested": auc, "C": C, "oof": nested}

    a0, b0 = folds[0]
    fold0 = float(roc_auc_score(y[b0], best["oof"][b0]))
    promoted = best["nested"] > V3_NESTED + 1e-7
    # test
    m = LogisticRegression(C=best["C"], tol=1e-8, max_iter=300, solver="newton-cholesky").fit(Xo, y)
    te = rk(m.decision_function(Xt))
    res = {
        "best_C": best["C"],
        "nested": best["nested"],
        "fold0": fold0,
        "delta_nested": best["nested"] - V3_NESTED,
        "delta_fold0": fold0 - V3_FOLD0,
        "promoted": promoted,
        "n_extra": Eo.shape[1],
    }
    if not promoted:
        mark_dead(state, "L4_regime_stacker", "nested did not beat P_super_v3", res)
        state["lever4"] = res
        save_state(state)
        return None
    state["lever4"] = res
    save_state(state)
    return best["oof"], te, res


def main():
    log(f"HIGH-EV LOOP vs P_super_v3 nested={V3_NESTED:.7f} fold0={V3_FOLD0:.7f} — {POLICY}")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    folds = folds_y(y)
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
        except Exception:
            prev = {}
    state = {
        "gates": {"V3_NESTED": V3_NESTED, "V3_FOLD0": V3_FOLD0},
        "dead_ends": [
            {
                "name": "Rung3_delta_LGB",
                "reason": "Fold0 ΔS vs 3-member fallback only; LGB+δ 0.95011 < gb_s0 0.95039; not vs true P_super_v3 Fold0 0.95112",
            }
        ],
        "lb_top": [],
    }
    # preserve completed lever blocks from prior run
    for k in ("lever1", "lever2", "lever3", "lever4", "v3_verify"):
        if k in prev:
            state[k] = prev[k]
    for d in prev.get("dead_ends", []):
        if d.get("name") != "Rung3_delta_LGB" and d not in state["dead_ends"]:
            state["dead_ends"].append(d)

    # LB scan already done externally — embed snapshot
    state["lb_top"] = [
        "areebbb/s6e11-glm-boost-residual-mlp-pool-lb-0-95239",
        "malyshevdanil/predicting-credit-card-fraud-getting-started",
        "nina2025/ps-s6e11-eos-4",
        "amanatar/s6e11-sovereign-fusion-solution",
        "goodpjw2008/s6e11-part-10-test-bag-fusion-lb-0-95239",
    ]
    save_state(state)

    log("building 76-member P_super_v3 bank …")
    names_v3, O_v3, T_v3, wire_info, glm_o, glm_t = build_v3_bank(y, folds)
    gb5_o = np.load(ART / "gb_bag5_oof.npy").ravel().astype(float)
    gb5_t = np.load(ART / "gb_bag5_full.npy").ravel().astype(float)
    gb5_t_bag = np.load(ART / "gb_testbag_full.npy").ravel().astype(float)
    bag_o = np.load("preds/gb_gpt2_bag5_oof.npy").ravel().astype(float)
    bag_t = np.load("preds/gb_gpt2_bag5_full.npy").ravel().astype(float)
    v3_oof = np.load("preds/P_super_v3_oof.npy").ravel()
    bank_ctx = {
        "gb5_o": gb5_o,
        "gb5_t": gb5_t,
        "gb5_t_bag": gb5_t_bag,
        "bag_o": bag_o,
        "bag_t": bag_t,
        "glm_o": glm_o,
        "glm_t": glm_t,
        "wire_info": wire_info,
    }

    # verify v3 nested quickly
    verify = nested_psuper(O_v3, T_v3, y, folds, gb5_o, bag_o, gb5_t_bag, bag_t)
    log(f"verify P_super_v3 rebuild nested={verify['nested']:.7f} fold0={verify['fold0']:.7f}")
    state["v3_verify"] = {"nested": verify["nested"], "fold0": verify["fold0"]}
    save_state(state)

    # Resume Lever-1 if prior JSON already has inject_results (skip ~5min recompute)
    if state.get("lever1", {}).get("inject_results") and os.environ.get("FORCE_L1", "0") != "1":
        log("resuming Lever-1 from prior JSON (set FORCE_L1=1 to recompute)")
        promoted_members = []
        path_by_name = {r["name"]: r for r in state["lever1"].get("reports", [])}
        for row in state["lever1"]["inject_results"]:
            if not row.get("promoted"):
                continue
            meta = path_by_name.get(row["name"], {})
            oof_p, te_p = meta.get("oof_path"), meta.get("test_path")
            if not oof_p or not te_p:
                continue
            oof = np.load(oof_p).ravel().astype(float)
            te = np.load(te_p).ravel().astype(float)
            promoted_members.append((row["name"], oof, te, row))
        log(f"  resumed {len(promoted_members)} promoted L1 members")
        save_state(state)
    else:
        promoted_members = lever1(state, y, folds, gb5_o, gb5_t, v3_oof, names_v3, O_v3, T_v3, bank_ctx)
    cubic = lever2(state, y, folds, bank_ctx, names_v3, O_v3, T_v3)
    l3 = lever3(state, y, folds, bank_ctx, names_v3, O_v3, T_v3, cubic=cubic[:2] if cubic else None)
    l4 = lever4(state, y, folds, bank_ctx, names_v3, O_v3, T_v3)

    # Build P_super_v4 from best promoted combination
    log("=" * 72)
    log("Assemble P_super_v4")
    log("=" * 72)
    O4, T4 = O_v3.copy(), T_v3.copy()
    names4 = list(names_v3)
    if cubic is not None:
        oof, te, _ = cubic
        ni = {n: i for i, n in enumerate(names4)}
        e_o, e_t = single_orth_foldwise(oof, te, glm_o, glm_t, folds)
        O4[:, ni["glm_quad"]] = rk(oof)
        T4[:, ni["glm_quad"]] = rk(te)
        O4[:, ni["glm_quad_orth_glm"]] = to_stack_col(e_o, True)
        T4[:, ni["glm_quad_orth_glm"]] = to_stack_col(e_t, True)
        names4 = [n if n != "glm_quad" else "glm_cubic" for n in names4]
    for name, oof, te, _ in promoted_members or []:
        O4 = np.column_stack([O4, rk(oof)])
        T4 = np.column_stack([T4, rk(te)])
        names4.append(name)

    bag_o_use, bag_t_use = bag_o, bag_t
    if l3 is not None:
        bag_o_use, bag_t_use = l3[0], l3[1]

    if l4 is not None:
        # regime stacker replaces full p_super
        p_oof, p_te, meta = l4
        auc = meta["nested"]
        log(f"P_super_v4 = regime stacker nested={auc:.7f}")
    else:
        res = nested_psuper(O4, T4, y, folds, gb5_o, bag_o_use, gb5_t_bag, bag_t_use)
        p_oof, p_te, auc = res["oof"], res["test"], res["nested"]
        log(f"P_super_v4 nested={auc:.7f} Δ={auc - V3_NESTED:+.3e} fold0={res['fold0']:.7f}")
        state["v4_nested_res"] = {k: v for k, v in res.items() if k not in ("oof", "test")}

    # If no improvement, still save v4 as copy of v3 with note
    if auc <= V3_NESTED + 1e-9:
        log("No nested lift — P_super_v4 := P_super_v3 (no regression)")
        p_oof = np.load("preds/P_super_v3_oof.npy").ravel()
        p_te = np.load("preds/P_super_v3_test.npy").ravel()
        auc = V3_NESTED
        state["v4_note"] = "identical to v3 — no promoted lever beat nested gate"

    np.save("preds/P_super_v4_oof.npy", np.asarray(p_oof, np.float32))
    np.save("preds/P_super_v4_test.npy", np.asarray(p_te, np.float32))

    # Spearman distances
    sub_ids = pd.read_csv("data/sample_submission.csv")["id"].to_numpy()
    from part8_proprietary_scan import build_p7_65  # noqa: E402
    from part7_clean_superstack import fit_test  # noqa: E402

    p7_names, P7_O, P7_T, _ = build_p7_65(y)
    P9_T, _ = wire_testbags(p7_names, P7_T)
    te65, _ = fit_test(P7_O, P9_T, y, C=0.01, tol=1e-8, max_iter=200)
    te62, _ = fit_test(P7_O[:, :62], P9_T[:, :62], y, C=0.01, tol=1e-8, max_iter=200)
    part9 = rk(rk(te65) + rk(te62))
    amanatar = pd.read_csv("external_bank/amanatar_sovereign/submission.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    part10 = rk(rk(part9) + rk(amanatar))
    p7 = pd.read_csv("external_bank/goodpjw_part7_out/submission.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    part6 = pd.read_csv("external_bank/malyshev_v11/nov04_part6.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    alex = pd.read_csv("external_bank/malyshev_v11/nov04_alex.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    p7gs = rk(0.5 * rk(p7) + 0.5 * rk(0.6 * rk(part6) + 0.4 * rk(alex)))

    vecs = {
        "P_super_v4": p_te,
        "goodpjw_part9": part9,
        "amanatar_sovereign": amanatar,
        "goodpjw_part10": part10,
        "p7gs": p7gs,
    }
    dist = {a: {b: (0.0 if a == b else d_e4(vecs[a], vecs[b])) for b in vecs} for a in vecs}
    state["v4"] = {
        "nested": float(roc_auc_score(y, p_oof)),
        "files": ["preds/P_super_v4_oof.npy", "preds/P_super_v4_test.npy"],
        "test_distance_e4": dist,
    }
    save_state(state)

    print("\n" + "=" * 88)
    print("SUMMARY — HIGH EV LOOP (NO_SUBMIT)")
    print("=" * 88)
    print(f"dead_ends ({len(state['dead_ends'])}):")
    for d in state["dead_ends"]:
        print(f"  - {d['name']}: {d['reason']}")
    print(f"P_super_v4 nested={state['v4']['nested']:.7f}")
    print("test (1-ρ)×1e4:")
    keys = list(vecs)
    print(f"{'':22s}" + "".join(f"{k[:12]:>12s}" for k in keys))
    for a in keys:
        print(f"{a:22s}" + "".join(f"{dist[a][b]:12.2f}" for b in keys))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
