"""Part-11 Grandmaster AUC Engine — STRICT NO_SUBMIT.

Steps:
  1) Unbundle dhan/masaya/areebbb/nina + Ridge orth(blend | Bank_76)
  2) S_extremal row-wise order-stat features (Fold-0 gate)
  3) L4_v2 two-stage differentially-regularized regime stacker
  4) L5 overlap-focal / RankNet AUC meta
  5) Save P_super_v5_{pure,full} + anchor audit

Outputs: logs/overnight/part11_grandmaster_auc_engine.json
         preds/P_super_v5_{pure,full}_{oof,test}.npy
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
from scipy.special import expit, logit as _logit
from scipy.stats import kurtosis, rankdata, skew, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
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
    fit_test_stack,
    nest_hill,
    nest_newton,
    p_super_from_parts,
    rk,
    rho,
    single_orth_foldwise,
    wire_testbags,
)
from high_ev_paradigm_loop import (  # noqa: E402
    V3_FOLD0,
    V3_NESTED,
    _jsonable,
    build_v3_bank,
    d_e4,
    nested_psuper,
)

T0 = time.time()
OUT = Path("logs/overnight/part11_grandmaster_auc_engine.json")
POLICY = "NO_SUBMIT"
V4_NESTED = 0.9522570184189607
L4_V3_NESTED = 0.9521794373741388


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


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


def load_pair(oof_p, te_p, sub_ids, y):
    oof = np.load(oof_p).ravel().astype(float)
    if str(te_p).endswith(".csv"):
        df = pd.read_csv(te_p)
        col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
        if "id" in df.columns:
            te = df.set_index("id").loc[sub_ids, col].to_numpy(float)
        else:
            te = df[col].to_numpy(float)
    else:
        te = np.load(te_p).ravel().astype(float)
    assert oof.shape == (len(y),), oof.shape
    assert te.shape == (len(sub_ids),), te.shape
    return oof, te


def foldwise_ridge_orth(blend_o, blend_t, O_bank, T_bank, folds, alpha=100.0):
    """Residual rlogit(blend) - Ridge(rlogit(Bank_76)). Fold-wise OOF + full-refit test."""
    Xo = rlogit(np.asarray(O_bank, float))
    Xt = rlogit(np.asarray(T_bank, float))
    yo = rlogit(rk(blend_o))
    yt = rlogit(rk(blend_t))
    e_o = np.zeros(len(yo))
    for a, b in folds:
        m = Ridge(alpha=alpha, fit_intercept=True).fit(Xo[a], yo[a])
        e_o[b] = yo[b] - m.predict(Xo[b])
    m = Ridge(alpha=alpha, fit_intercept=True).fit(Xo, yo)
    e_t = yt - m.predict(Xt)
    return e_o, e_t


def eval_inject(O, T, y, folds, bank_ctx, label):
    res = nested_psuper(
        O, T, y, folds, bank_ctx["gb5_o"], bank_ctx["bag_o"], bank_ctx["gb5_t_bag"], bank_ctx["bag_t"]
    )
    log(
        f"  {label}: nested={res['nested']:.7f} (Δ={res['delta_nested']:+.3e}) "
        f"fold0={res['fold0']:.7f} (Δ={res['delta_fold0']:+.3e})"
    )
    return res


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — unbundle + orth
# ═══════════════════════════════════════════════════════════════════════════
def step1(state, y, folds, names_v3, O_v3, T_v3, bank_ctx, sub_ids, part10_te, amanatar_te):
    log("=" * 72)
    log("STEP 1: unbundle + orth(blend | Bank_76)")
    log("=" * 72)

    recon = {
        "dhananjay_formulas": {
            "18_blend_pooled2": (
                "nested-CV wider pool (~116 members) best method=mix3 "
                "(rank(hill)+rank(stack C=1)+rank(stack C=0.03)); "
                "weights fit OOF via nested folds (honest OOF). "
                "Hill weights in runs.json include masaya logreg, gb_bag10, tabpfn lgb, gpt2."
            ),
            "18_blend_lm50": (
                "50/50 rank mix of nested stack WITH LM members vs WITHOUT LM members; "
                "both stacks have nested OOF → soft_oof is honest. No raw 18 base npy in part4_out."
            ),
            "raw_base_npy_in_part4_out": False,
            "dataset_sources": json.loads(
                Path("external_bank/dhananjay_part4_code/kernel-metadata.json").read_text()
            ).get("dataset_sources"),
            "kernel_sources": [
                k
                for k in json.loads(
                    Path("external_bank/dhananjay_part4_code/kernel-metadata.json").read_text()
                ).get("kernel_sources", [])
                if k
            ],
        },
        "masaya_stack": {
            "members": json.loads(Path("external_bank/masaya_stack/manifest.json").read_text())["members"],
            "weights": json.loads(Path("external_bank/masaya_stack/manifest.json").read_text())["weights"],
            "kernel_sources": [
                k
                for k in json.loads(
                    Path("external_bank/masaya_stack_code/kernel-metadata.json").read_text()
                ).get("kernel_sources", [])
                if k
            ],
            "without_public_models_oof_auc": float(
                roc_auc_score(y, np.load("external_bank/masaya_stack/without_public_models_oof.npy").ravel())
            ),
            "note": "without_public_models is OOF-only ablation (no test); raw members partially pulled",
        },
        "areebbb_0_95239": {
            "formula": "submission.csv = rank(0.4 * goodpjw_part9 + 0.6 * amanatar_sovereign)",
            "was": "0.95231 = 0.5*part7 + 0.5*getting_started anchors",
            "new_oofs": False,
            "datasets": json.loads(
                Path("external_bank/areebbb_95239_code/kernel-metadata.json").read_text()
            ).get("dataset_sources"),
            "kernels": [
                k
                for k in json.loads(
                    Path("external_bank/areebbb_95239_code/kernel-metadata.json").read_text()
                ).get("kernel_sources", [])
                if k
            ],
        },
        "nina2025_eos4": {
            "note": "EoS(4) — public fusion kernel; only submission.csv published (test-only)",
            "kernels": [
                k
                for k in json.loads(
                    Path("external_bank/nina2025_eos4_code/kernel-metadata.json").read_text()
                ).get("kernel_sources", [])
                if k
            ],
            "datasets": json.loads(
                Path("external_bank/nina2025_eos4_code/kernel-metadata.json").read_text()
            ).get("dataset_sources"),
        },
    }
    state["step1_recon"] = recon
    save_state(state)

    # --- load composite blends ---
    blends = {}
    for name, oof_p, te_p in [
        (
            "masaya_stack_main",
            "external_bank/masaya_stack/oof.npy",
            "external_bank/masaya_stack/test.npy",
        ),
        (
            "dhan_18_blend_lm50",
            "external_bank/dhananjay_part4_out/oof/18_blend_lm50_oof.npy",
            "external_bank/dhananjay_part4_out/oof/18_blend_lm50_test.npy",
        ),
        (
            "dhan_18_blend_pooled2",
            "external_bank/dhananjay_part4_out/oof/18_blend_pooled2_oof.npy",
            "external_bank/dhananjay_part4_out/oof/18_blend_pooled2_test.npy",
        ),
    ]:
        oof, te = load_pair(oof_p, te_p, sub_ids, y)
        blends[name] = (oof, te)
        log(f"  blend {name}: oof_auc={roc_auc_score(y, oof):.6f}")

    # --- unbundled masaya raw members (where available) ---
    raw_members = []
    cand_raw = [
        (
            "masaya_logreg_te",
            "external_bank/masaya_members/masayakawamata_s6e11-single-logreg-te-features-cv-0-9502/oof.npy",
            "external_bank/masaya_members/masayakawamata_s6e11-single-logreg-te-features-cv-0-9502/test.npy",
        ),
        (
            "masaya_xgb_te",
            "external_bank/masaya_members/masayakawamata_s6e11-single-xgb-te-features-cv-0-9504/oof.npy",
            "external_bank/masaya_members/masayakawamata_s6e11-single-xgb-te-features-cv-0-9504/test.npy",
        ),
        (
            "masaya_gpt2",
            "external_bank/masaya_gpt2/oof.npy",
            "external_bank/masaya_gpt2/test.npy",
        ),
    ]
    # sameer csv — NEVER use is_fraud (label); prefer seed_4 / avg5
    sameer_dir = Path("external_bank/masaya_members/sameershaikh27_s6e11-fraud-seed-and-fold-averaging-measured")
    if (sameer_dir / "oof.csv").exists():
        oof_df = pd.read_csv(sameer_dir / "oof.csv")
        te_df = pd.read_csv(sameer_dir / "test_preds.csv")
        for ocol, tcol, tag in (
            ("seed_4", "seed_4", "sameer_seed_4"),
            ("avg5", "seed_4", "sameer_avg5"),  # test has no avg5 → use seed_4 as proxy pair later filtered
        ):
            if ocol not in oof_df.columns or tcol not in te_df.columns:
                continue
            train_ids = pd.read_csv("data/train.csv")["id"].to_numpy()
            oof = oof_df.set_index("id").loc[train_ids, ocol].to_numpy(float)
            te = te_df.set_index("id").loc[sub_ids, tcol].to_numpy(float)
            auc = float(roc_auc_score(y, oof))
            if auc > 0.999:
                log(f"  SKIP {tag}: suspicious oof_auc={auc:.6f}")
                continue
            # defer align/dedupe to shared path via cand_raw-style append after checks
            ro = rk(oof)
            max_r = max(abs(float(np.corrcoef(ro, O_v3[:, j])[0, 1])) for j in range(O_v3.shape[1]))
            delta_rho = abs(rho(oof, bank_ctx["gb5_o"]) - rho(te, bank_ctx["gb5_t"]))
            rec = {
                "name": tag,
                "oof_auc": auc,
                "max_r_vs_v3": max_r,
                "delta_rho": float(delta_rho),
                "aligned": bool(delta_rho < 0.003),
                "duplicate": bool(max_r >= 0.9995),
            }
            log(
                f"  raw {tag}: auc={auc:.6f} max_r={max_r:.4f} |Δρ|={delta_rho:.4f} "
                f"{'skip' if (rec['duplicate'] or not rec['aligned']) else 'ok'}"
            )
            state.setdefault("step1_raw_members", []).append(rec)
            if rec["aligned"] and not rec["duplicate"]:
                raw_members.append((tag, oof, te))
            break  # only one sameer member

    for name, oof_p, te_p in cand_raw:
        if oof_p is None:
            continue
        if not Path(oof_p).exists():
            continue
        oof, te = load_pair(oof_p, te_p, sub_ids, y)
        # skip near-duplicates vs bank
        ro = rk(oof)
        max_r = max(abs(float(np.corrcoef(ro, O_v3[:, j])[0, 1])) for j in range(O_v3.shape[1]))
        delta_rho = abs(rho(oof, bank_ctx["gb5_o"]) - rho(te, bank_ctx["gb5_t"]))
        rec = {
            "name": name,
            "oof_auc": float(roc_auc_score(y, oof)),
            "max_r_vs_v3": max_r,
            "delta_rho": float(delta_rho),
            "aligned": bool(delta_rho < 0.003),
            "duplicate": bool(max_r >= 0.9995),
        }
        log(
            f"  raw {name}: auc={rec['oof_auc']:.6f} max_r={max_r:.4f} |Δρ|={delta_rho:.4f} "
            f"{'DUP' if rec['duplicate'] else 'ok'}"
        )
        if rec["aligned"] and not rec["duplicate"]:
            raw_members.append((name, oof, te))
        state.setdefault("step1_raw_members", []).append(rec)

    # --- orth composites ---
    orth_members = []
    orth_cmp = []
    for name, (oof, te) in blends.items():
        e_o, e_t = foldwise_ridge_orth(oof, te, O_v3, T_v3, folds, alpha=100.0)
        # stackable residual column
        o_col = to_stack_col(e_o, True)
        t_col = to_stack_col(e_t, True)
        orth_members.append((f"orth_{name}", o_col, t_col, e_o, e_t))

        # raw inject
        O_raw = np.column_stack([O_v3, rk(oof)])
        T_raw = np.column_stack([T_v3, rk(te)])
        res_raw = eval_inject(O_raw, T_raw, y, folds, bank_ctx, f"RAW {name}")
        # orth inject
        O_or = np.column_stack([O_v3, o_col])
        T_or = np.column_stack([T_v3, t_col])
        res_or = eval_inject(O_or, T_or, y, folds, bank_ctx, f"ORTH {name}")

        # test distances of resulting p_super test vs anchors
        def te_dist(res):
            return {
                "vs_part10": float(d_e4(res["test"], part10_te)),
                "vs_amanatar": float(d_e4(res["test"], amanatar_te)),
            }

        row = {
            "name": name,
            "raw": {
                "nested": res_raw["nested"],
                "fold0": res_raw["fold0"],
                "delta_nested": res_raw["delta_nested"],
                **te_dist(res_raw),
            },
            "orth": {
                "nested": res_or["nested"],
                "fold0": res_or["fold0"],
                "delta_nested": res_or["delta_nested"],
                **te_dist(res_or),
            },
            "prefer_orth": bool(res_or["nested"] >= res_raw["nested"] - 1e-9),
        }
        orth_cmp.append(row)
        log(
            f"    prefer={'ORTH' if row['prefer_orth'] else 'RAW'} "
            f"raw_d10={row['raw']['vs_part10']:.2f} orth_d10={row['orth']['vs_part10']:.2f}"
        )

    # joint orth of all 3
    O_j = np.column_stack([O_v3] + [c for _, c, _, _, _ in orth_members])
    T_j = np.column_stack([T_v3] + [c for _, _, c, _, _ in orth_members])
    res_j = eval_inject(O_j, T_j, y, folds, bank_ctx, "JOINT ORTH 3 blends")
    # joint raw (for comparison — collinear)
    O_jr = np.column_stack([O_v3] + [rk(blends[n][0]) for n in blends])
    T_jr = np.column_stack([T_v3] + [rk(blends[n][1]) for n in blends])
    res_jr = eval_inject(O_jr, T_jr, y, folds, bank_ctx, "JOINT RAW 3 blends")

    # raw unbundled inject
    res_raw_u = None
    if raw_members:
        O_u = np.column_stack([O_v3] + [rk(o) for _, o, _ in raw_members])
        T_u = np.column_stack([T_v3] + [rk(t) for _, _, t in raw_members])
        res_raw_u = eval_inject(O_u, T_u, y, folds, bank_ctx, f"RAW unbundled ({len(raw_members)})")

    state["step1"] = {
        "orth_comparison": orth_cmp,
        "joint_orth": {k: v for k, v in res_j.items() if k not in ("oof", "test")},
        "joint_raw": {k: v for k, v in res_jr.items() if k not in ("oof", "test")},
        "joint_orth_dist": {
            "vs_part10": float(d_e4(res_j["test"], part10_te)),
            "vs_amanatar": float(d_e4(res_j["test"], amanatar_te)),
        },
        "joint_raw_dist": {
            "vs_part10": float(d_e4(res_jr["test"], part10_te)),
            "vs_amanatar": float(d_e4(res_jr["test"], amanatar_te)),
        },
        "unbundled_raw": None
        if res_raw_u is None
        else {k: v for k, v in res_raw_u.items() if k not in ("oof", "test")},
        "n_raw_members": len(raw_members),
    }
    save_state(state)
    # persist for resume
    np.savez_compressed(
        "preds/part11_step1.npz",
        O_orth=O_j,
        T_orth=T_j,
        O_unbundled=np.column_stack([O_v3] + [rk(o) for _, o, _ in raw_members]) if raw_members else O_v3,
        T_unbundled=np.column_stack([T_v3] + [rk(t) for _, _, t in raw_members]) if raw_members else T_v3,
        names_unbundled=np.array(list(names_v3) + [n for n, _, _ in raw_members], dtype=object),
        names_orth=np.array(list(names_v3) + [n for n, *_ in orth_members], dtype=object),
    )
    return {
        "blends": blends,
        "orth_members": [(n, oc, tc) for n, oc, tc, _, _ in orth_members],
        "raw_members": raw_members,
        "joint_orth_oof": res_j["oof"],
        "joint_orth_te": res_j["test"],
        "O_orth": O_j,
        "T_orth": T_j,
        "O_unbundled": np.column_stack([O_v3] + [rk(o) for _, o, _ in raw_members]) if raw_members else O_v3.copy(),
        "T_unbundled": np.column_stack([T_v3] + [rk(t) for _, _, t in raw_members]) if raw_members else T_v3.copy(),
        "names_unbundled": list(names_v3) + [n for n, _, _ in raw_members],
        "names_orth": list(names_v3) + [n for n, *_ in orth_members],
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — S_extremal
# ═══════════════════════════════════════════════════════════════════════════
def extremal_feats(Z, kcnt_rows, fam_logits=None):
    """~35 row-wise extremal / concentration features from Z (n,d)."""
    Z = np.asarray(Z, np.float32)
    n, d = Z.shape
    # sort ascending per row
    Zs = np.sort(Z, axis=1)
    feats = []
    # top-5 largest (end) + partial sums
    for k in range(1, 6):
        feats.append(Zs[:, -k])
    feats.append(Zs[:, -5:].sum(1))
    feats.append(Zs[:, -3:].sum(1))
    # top-5 most negative
    for k in range(5):
        feats.append(Zs[:, k])
    feats.append(Zs[:, :5].sum(1))
    # concentration / moments
    mu = Z.mean(1)
    sd = Z.std(1) + 1e-4
    feats.append(Zs[:, -1] / sd)  # max / std
    feats.append(skew(Z, axis=1, bias=False))
    feats.append(kurtosis(Z, axis=1, bias=False, fisher=True))
    feats.append(((Z - mu[:, None]) / sd[:, None] > 2).sum(1).astype(np.float32))
    feats.append(((Z - mu[:, None]) / sd[:, None] > 3).sum(1).astype(np.float32))
    # support-weighted extremes
    if kcnt_rows is not None:
        d_k = min(d, kcnt_rows.shape[1])
        w = np.log1p(kcnt_rows[:, :d_k].astype(np.float32))
        Zw = Z[:, :d_k] * w
        feats.append(Zw.max(1))
        feats.append(Zw.min(1))
    else:
        feats.append(Zs[:, -1])
        feats.append(Zs[:, 0])
    # first-24-key block extremes (main columns proxy)
    d24 = min(24, d)
    Z24 = np.sort(Z[:, :d24], axis=1)
    feats.append(Z24[:, -1])
    feats.append(Z24[:, -3:].sum(1))
    feats.append(Z24[:, 0])
    feats.append(Z24[:, :3].sum(1))
    # cross-family range
    if fam_logits is not None:
        F = np.asarray(fam_logits, np.float32)
        feats.append(F.max(1) - F.min(1))
        feats.append(F.std(1))
    else:
        feats.append(Zs[:, -1] - Zs[:, 0])
        feats.append(sd)
    X = np.column_stack(feats).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def step2(state, y, folds, names_v3, O_v3, T_v3, bank_ctx):
    log("=" * 72)
    log("STEP 2: S_extremal order-stat features (Fold-0 gate)")
    log("=" * 72)
    log("  importing gboost (TE keys) …")
    import gc
    gc.collect()
    import gboost as gb
    log(f"  gboost ready GEN={gb.GEN} BASE={gb.BASE.shape}")

    glm_o, glm_t = bank_ctx["glm_o"], bank_ctx["glm_t"]
    n = len(y)
    n_tr = n
    KCNT_tr = gb.KCNT[:n_tr]
    KCNT_te = gb.KCNT[n_tr:]

    # family logits for cross-family range
    ni = {n: i for i, n in enumerate(names_v3)}
    fam_names = [x for x in ("gb_bag5", "gpt2", "glm_quad", "ebm_i25_bag4", "gbn_bag5_w512", "gbx_bag3", "ektarr", "realmlp") if x in ni]
    # fallback names present in 8B
    if len(fam_names) < 4:
        fam_names = list(names_v3[:8])

    def te_z(a, b):
        Ea, Eb, Et = gb.te_fold(a, b, smooth=20.0, seed=0)
        if gb.GEN:
            Ga, Gb, Gt = gb.gen_fold(a, b, seed=0)
            Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        ybar = float(y[a].mean())
        scz = StandardScaler().fit(rlogit(Ea) - rlogit(ybar))
        Za = scz.transform(rlogit(Ea) - rlogit(ybar)).astype(np.float32)
        Zb = scz.transform(rlogit(Eb) - rlogit(ybar)).astype(np.float32)
        Zt = scz.transform(rlogit(Et) - rlogit(ybar)).astype(np.float32)
        return Za, Zb, Zt

    # Fold-0 first
    a0, b0 = folds[0]
    log("  TE fold 0 for extremal …")
    Za, Zb, Zt = te_z(a0, b0)
    log(f"  D={Za.shape[1]}")
    fam_o = np.column_stack([rlogit(O_v3[:, ni[n]]) for n in fam_names if n in ni])
    fam_t = np.column_stack([rlogit(T_v3[:, ni[n]]) for n in fam_names if n in ni])
    Xa = extremal_feats(Za, KCNT_tr[a0], fam_o[a0])
    Xb = extremal_feats(Zb, KCNT_tr[b0], fam_o[b0])
    # also append BASE-ish: use stage1 offset
    # GLM + extremal
    sc = StandardScaler().fit(Xa)
    Xa_s, Xb_s = sc.transform(Xa), sc.transform(Xb)
    # nest as extra cols: train LR on [rlogit(glm), extremal] with init from glm
    Xa_m = np.hstack([rlogit(rk(glm_o))[a0, None], Xa_s])
    Xb_m = np.hstack([rlogit(rk(glm_o))[b0, None], Xb_s])
    m = LogisticRegression(C=0.01, max_iter=200, solver="newton-cholesky").fit(Xa_m, y[a0])
    pred_glm = expit(m.decision_function(Xb_m))
    auc_glm = float(roc_auc_score(y[b0], pred_glm))
    # LGB from z_GLM
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=4, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
    )
    z_glm_a = rlogit(rk(glm_o))[a0]
    z_glm_b = rlogit(rk(glm_o))[b0]
    clf.fit(Xa_s, y[a0], init_score=z_glm_a)
    pred_lgb = expit(z_glm_b + clf.predict(Xb_s, raw_score=True))
    auc_lgb = float(roc_auc_score(y[b0], pred_lgb))
    log(f"  Fold0 S_extremal+GLM={auc_glm:.6f}  S_extremal+LGB(init=z_GLM)={auc_lgb:.6f}")

    # cheap Fold0 stack probe: blend extremal into existing V3 OOF on fold0 only
    v3_oof = np.load("preds/P_super_v3_oof.npy").ravel()
    ext_b = 0.5 * rk(pred_glm) + 0.5 * (rankdata(pred_lgb) / len(pred_lgb))
    # rank-mean of V3 with extremal on fold0
    blend_b = rk(0.92 * rk(v3_oof[b0]) + 0.08 * ext_b)
    fold0_stack = float(roc_auc_score(y[b0], blend_b))
    fold0_v3 = float(roc_auc_score(y[b0], v3_oof[b0]))
    log(f"  Fold0 stack probe={fold0_stack:.7f} vs V3_FOLD0={fold0_v3:.7f} (gate {V3_FOLD0:.7f})")
    # also require solo not catastrophic vs glm_quad fold0
    gq_f0 = float(roc_auc_score(y[b0], np.load("preds/ladder/glm_quad_oof.npy").ravel()[b0]))
    gate = (fold0_stack > V3_FOLD0 + 1e-7) and (max(auc_glm, auc_lgb) > gq_f0 - 5e-4)
    res = {
        "fold0_glm_auc": auc_glm,
        "fold0_lgb_auc": auc_lgb,
        "fold0_stack_probe": fold0_stack,
        "n_feats": int(Xa.shape[1]),
        "D": int(Za.shape[1]),
        "promoted": False,
    }
    if not gate:
        mark_dead(state, "S2_S_extremal", "Fold0 stack did not beat P_super_v3 Fold0", res)
        state["step2"] = res
        save_state(state)
        return None

    log("  Fold0 PASSED — full 5-fold S_extremal …")
    oof = np.zeros(n)
    te_acc = []
    for fi, (a, b) in enumerate(folds):
        log(f"  TE+extremal fold {fi} …")
        Za, Zb, Zt = te_z(a, b)
        Xa = extremal_feats(Za, KCNT_tr[a], fam_o[a])
        Xb = extremal_feats(Zb, KCNT_tr[b], fam_o[b])
        Xt = extremal_feats(Zt, KCNT_te, fam_t)
        sc = StandardScaler().fit(Xa)
        Xa, Xb, Xt = sc.transform(Xa), sc.transform(Xb), sc.transform(Xt)
        Xa_m = np.hstack([rlogit(rk(glm_o))[a, None], Xa])
        Xb_m = np.hstack([rlogit(rk(glm_o))[b, None], Xb])
        Xt_m = np.hstack([rlogit(rk(glm_t))[:, None], Xt])
        m = LogisticRegression(C=0.01, max_iter=200, solver="newton-cholesky").fit(Xa_m, y[a])
        p_glm_b = expit(m.decision_function(Xb_m))
        p_glm_t = expit(m.decision_function(Xt_m))
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=4, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42 + fi, n_jobs=-1, verbosity=-1,
        )
        clf.fit(Xa, y[a], init_score=rlogit(rk(glm_o))[a])
        p_lgb_b = expit(rlogit(rk(glm_o))[b] + clf.predict(Xb, raw_score=True))
        p_lgb_t = expit(rlogit(rk(glm_t)) + clf.predict(Xt, raw_score=True))
        oof[b] = 0.5 * rk(p_glm_b) + 0.5 * (rankdata(p_lgb_b) / len(p_lgb_b))
        te_acc.append(0.5 * rk(p_glm_t) + 0.5 * (rankdata(p_lgb_t) / len(p_lgb_t)))
    te = np.mean(te_acc, 0)
    np.save("preds/s_extremal_oof.npy", oof.astype(np.float32))
    np.save("preds/s_extremal_test.npy", te.astype(np.float32))
    oof_auc = float(roc_auc_score(y, oof))
    O2 = np.column_stack([O_v3, rk(oof)])
    T2 = np.column_stack([T_v3, rk(te)])
    nest = eval_inject(O2, T2, y, folds, bank_ctx, "S_extremal inject")
    promoted = nest["delta_nested"] > 1e-7 and nest["delta_fold0"] > 0
    res.update({"oof_auc": oof_auc, "nested": nest["nested"], "fold0": nest["fold0"], "promoted": promoted})
    if not promoted:
        mark_dead(state, "S2_S_extremal_full", "5-fold nested did not beat V3", res)
    state["step2"] = res
    save_state(state)
    return (oof, te, nest) if promoted else None


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — L4_v2 two-stage
# ═══════════════════════════════════════════════════════════════════════════
def build_regimes(y, names, O, T, bank_ctx, extremal_oof=None, extremal_te=None):
    import gboost as gb

    n, nt = O.shape[0], T.shape[0]
    ni = {nm: i for i, nm in enumerate(names)}
    kcnt_tr = gb.KCNT[:n]
    kcnt_te = gb.KCNT[n:]
    n_float = min(6, kcnt_tr.shape[1])

    def stdize(tr, te):
        mu, sd = float(tr.mean()), float(tr.std()) + 1e-12
        return (tr - mu) / sd, (te - mu) / sd

    r_rare_tr, r_rare_te = stdize(
        (kcnt_tr[:, :n_float] <= 3).mean(1), (kcnt_te[:, :n_float] <= 3).mean(1)
    )
    r_high_tr, r_high_te = stdize(
        (kcnt_tr[:, :n_float] >= np.percentile(kcnt_tr[:, :n_float], 90)).mean(1),
        (kcnt_te[:, :n_float] >= np.percentile(kcnt_tr[:, :n_float], 90)).mean(1),
    )

    def col(name, fallback=None):
        if name in ni:
            return O[:, ni[name]], T[:, ni[name]]
        return fallback

    gb5 = col("gb_bag5", (rk(bank_ctx["gb5_o"]), rk(bank_ctx["gb5_t_bag"])))
    gpt = col("gpt2", (rk(np.load("external_bank/masaya_gpt2/oof.npy").ravel()), rk(np.load("external_bank/masaya_gpt2/test.npy").ravel())))
    gq = col("glm_quad")
    ebm = col("ebm_i25_bag4", gb5)
    gbn = col("gbn_bag5_w512", gb5)
    # optional extras
    fams = [gb5, gpt, gq if gq else gb5, ebm, gbn]
    mats_tr = np.column_stack([rlogit(f[0]) for f in fams])
    mats_te = np.column_stack([rlogit(f[1]) for f in fams])
    r_disp_tr, r_disp_te = stdize(mats_tr.std(1), mats_te.std(1))
    r_gpt_tr, r_gpt_te = stdize(rlogit(gpt[0]) - rlogit(gb5[0]), rlogit(gpt[1]) - rlogit(gb5[1]))
    r_gpt2_tr, r_gpt2_te = stdize(r_gpt_tr ** 2, r_gpt_te ** 2)
    # mlp vs tree: gbn vs gb5
    r_mlp_tr, r_mlp_te = stdize(rlogit(gbn[0]) - rlogit(gb5[0]), rlogit(gbn[1]) - rlogit(gb5[1]))
    # tails vs v3-ish mean of bank
    base_o = O.mean(1)
    base_t = T.mean(1)
    p90 = np.percentile(base_o, 90)
    p20 = np.percentile(base_o, 20)
    r_pos_tr, r_pos_te = stdize((base_o >= p90).astype(float), (base_t >= p90).astype(float))
    r_neg_tr, r_neg_te = stdize((base_o <= p20).astype(float), (base_t <= p20).astype(float))

    regs_tr = {
        "r_rare_float": r_rare_tr,
        "r_high_cnt": r_high_tr,
        "r_disp_all": r_disp_tr,
        "r_gpt_vs_tree": r_gpt_tr,
        "r_gpt_vs_tree_sq": r_gpt2_tr,
        "r_mlp_vs_tree": r_mlp_tr,
        "r_pos_tail_p90": r_pos_tr,
        "r_neg_tail_p20": r_neg_tr,
    }
    regs_te = {
        "r_rare_float": r_rare_te,
        "r_high_cnt": r_high_te,
        "r_disp_all": r_disp_te,
        "r_gpt_vs_tree": r_gpt_te,
        "r_gpt_vs_tree_sq": r_gpt2_te,
        "r_mlp_vs_tree": r_mlp_te,
        "r_pos_tail_p90": r_pos_te,
        "r_neg_tail_p20": r_neg_te,
    }
    if extremal_oof is not None:
        # proxy max/min/kurt from extremal score itself + simple
        regs_tr["r_max1_z"] = stdize(rlogit(rk(extremal_oof)), rlogit(rk(extremal_te)))[0]
        regs_te["r_max1_z"] = stdize(rlogit(rk(extremal_oof)), rlogit(rk(extremal_te)))[1]
        regs_tr["r_min1_z"] = -regs_tr["r_max1_z"]
        regs_te["r_min1_z"] = -regs_te["r_max1_z"]
        regs_tr["r_kurt_z"] = regs_tr["r_disp_all"]
        regs_te["r_kurt_z"] = regs_te["r_disp_all"]

    mod_fams = [("gb_bag5", gb5), ("gpt2", gpt), ("glm_quad", gq if gq else gb5), ("ebm", ebm), ("gbn", gbn)]
    return regs_tr, regs_te, mod_fams


def step3(state, y, folds, names, O, T, bank_ctx, extremal=None):
    log("=" * 72)
    log("STEP 3: L4_v2 two-stage differentially-regularized regime")
    log("=" * 72)
    n = len(y)
    # Stage A
    st, auc_st = nest_newton(O, y, folds, C=0.005)
    hc, auc_hc = nest_hill(O, y, folds)
    sup_o = 0.50 * rk(bank_ctx["gb5_o"]) + 0.50 * rk(bank_ctx["bag_o"])
    p_main, auc_main, fa_main, auc_mix = p_super_from_parts(st, hc, sup_o, y, folds)
    a0, b0 = folds[0]
    fold0_main = float(roc_auc_score(y[b0], p_main[b0]))
    log(f"  StageA p_super nested={auc_main:.7f} fold0={fold0_main:.7f} newton={auc_st:.7f}")

    # z_main as decision-ish: use rlogit of nested newton only for residual base
    z_main = rlogit(rk(st))
    st_te, hc_te = fit_test_stack(O, T, y, C=0.005)
    mix_te = rk((1.0 - HW_OPT) * st_te + HW_OPT * hc_te)
    sup_t = 0.50 * rk(bank_ctx["gb5_t_bag"]) + 0.50 * rk(bank_ctx["bag_t"])
    p_main_te = rk((1.0 - SW_OPT) * mix_te + SW_OPT * rk(sup_t))
    z_main_te = rlogit(rk(st_te))

    ext_o = extremal[0] if extremal else None
    ext_t = extremal[1] if extremal else None
    regs_tr, regs_te, mod_fams = build_regimes(y, names, O, T, bank_ctx, ext_o, ext_t)

    # interaction design: regime × family logits, scaled by gamma
    best = {"nested": -1.0}
    for gamma in (0.10, 0.20, 0.35, 0.50):
        inter_o = []
        inter_t = []
        for rn, rv in regs_tr.items():
            for fn, (fo, ft) in mod_fams:
                inter_o.append(gamma * rv * rlogit(fo))
                inter_t.append(gamma * regs_te[rn] * rlogit(ft))
        Io = np.column_stack(inter_o)
        It = np.column_stack(inter_t)
        # Stage B: residual logistic on [z_main, interactions] — heavier L2 on interactions via gamma scale
        Xo = np.column_stack([z_main, Io])
        Xt = np.column_stack([z_main_te, It])
        nested = np.zeros(n)
        for a, b in folds:
            m = LogisticRegression(C=0.008, tol=1e-8, max_iter=300, solver="newton-cholesky").fit(Xo[a], y[a])
            nested[b] = rk(m.decision_function(Xo[b]))
        auc = float(roc_auc_score(y, nested))
        # also mix with hill+super
        p_mix, auc_mix2, _, _ = p_super_from_parts(nested, hc, sup_o, y, folds)
        fold0 = float(roc_auc_score(y[b0], p_mix[b0]))
        log(f"  L4_v2 γ={gamma}: newton={auc:.7f} +hc+super={auc_mix2:.7f} fold0={fold0:.7f}")
        if auc_mix2 > best["nested"]:
            m_full = LogisticRegression(C=0.008, tol=1e-8, max_iter=300, solver="newton-cholesky").fit(Xo, y)
            te_n = rk(m_full.decision_function(Xt))
            mix_te2 = rk((1.0 - HW_OPT) * te_n + HW_OPT * hc_te)
            te = rk((1.0 - SW_OPT) * mix_te2 + SW_OPT * rk(sup_t))
            best = {
                "nested": auc_mix2,
                "newton_only": auc,
                "fold0": fold0,
                "gamma": gamma,
                "oof": p_mix,
                "test": te,
                "delta_nested": auc_mix2 - V3_NESTED,
                "delta_fold0": fold0 - V3_FOLD0,
            }

    promoted = best["nested"] > max(L4_V3_NESTED, V3_NESTED) + 1e-7
    res = {k: v for k, v in best.items() if k not in ("oof", "test")}
    res["promoted"] = promoted
    if not promoted:
        mark_dead(state, "L4_v2_TwoStage", "did not beat L4_v3 / V3 nested", res)
    state["step3"] = res
    save_state(state)
    log(f"  BEST L4_v2 γ={best['gamma']} nested={best['nested']:.7f} promoted={promoted}")
    return (best["oof"], best["test"], best) if True else None  # always return best for L5 base


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — L5 AUC direct
# ═══════════════════════════════════════════════════════════════════════════
def step4(state, y, folds, names, O, T, bank_ctx, l4_oof, l4_te, base_nested):
    log("=" * 72)
    log("STEP 4: L5 overlap-focal / RankNet AUC meta")
    log("=" * 72)
    ni = {n: i for i, n in enumerate(names)}
    n = len(y)

    def get(name, path_o=None, path_t=None):
        if name in ni:
            return O[:, ni[name]], T[:, ni[name]]
        if path_o and Path(path_o).exists():
            return rk(np.load(path_o).ravel()), rk(np.load(path_t).ravel())
        return None, None

    cols_o, cols_t, col_names = [], [], []
    for name, po, pt in [
        ("l4", None, None),
        ("gb_super", None, None),
        ("gpt2", "external_bank/masaya_gpt2/oof.npy", "external_bank/masaya_gpt2/test.npy"),
        ("glm_quad", "preds/ladder/glm_quad_oof.npy", "preds/ladder/glm_quad_test.npy"),
        ("gbn_bag5_w512", None, None),
        ("ebm_i25_bag4", None, None),
    ]:
        if name == "l4":
            o, t = rk(l4_oof), rk(l4_te)
        elif name == "gb_super":
            o = rk(0.5 * rk(bank_ctx["gb5_o"]) + 0.5 * rk(bank_ctx["bag_o"]))
            t = rk(0.5 * rk(bank_ctx["gb5_t_bag"]) + 0.5 * rk(bank_ctx["bag_t"]))
        else:
            o, t = get(name, po, pt)
            if o is None:
                continue
            if name in ("glm_quad",) and name not in ni:
                o, t = rk(np.load(po).ravel()), rk(np.load(pt).ravel())
        cols_o.append(rlogit(o))
        cols_t.append(rlogit(t))
        col_names.append(name)
    Xo = np.column_stack(cols_o)
    Xt = np.column_stack(cols_t)
    z_base = rlogit(rk(l4_oof))
    z_base_te = rlogit(rk(l4_te))

    # 4A overlap-weighted newton
    best_a = {"nested": -1.0}
    for alpha in (-0.3, -0.15, 0.15, 0.30, 0.50):
        p = expit(z_base)
        w = np.clip(p * (1 - p), 1e-6, None) ** alpha
        nested = np.zeros(n)
        for a, b in folds:
            m = LogisticRegression(C=0.01, tol=1e-8, max_iter=300, solver="newton-cholesky")
            m.fit(Xo[a], y[a], sample_weight=w[a])
            nested[b] = rk(m.decision_function(Xo[b]))
        auc = float(roc_auc_score(y, nested))
        log(f"  L5-4A α={alpha:+.2f} nested={auc:.7f} Δ={auc - base_nested:+.3e}")
        if auc > best_a["nested"]:
            m = LogisticRegression(C=0.01, tol=1e-8, max_iter=300, solver="newton-cholesky")
            m.fit(Xo, y, sample_weight=w)
            best_a = {"nested": auc, "alpha": alpha, "oof": nested, "test": rk(m.decision_function(Xt))}

    # 4B pairwise RankNet (mini-batch all-pairs approx via histogram / subsample)
    def ranknet_loss_and_grad(w, Xp, Xn, tau=1.0):
        # score = X @ w
        sp = Xp @ w
        sn = Xn @ w
        # subsample pairs if huge
        rng = np.random.RandomState(abs(hash(w.tobytes())) % (2**31))
        n_pos, n_neg = len(sp), len(sn)
        n_pair = min(200_000, n_pos * n_neg)
        ip = rng.randint(0, n_pos, size=n_pair)
        jn = rng.randint(0, n_neg, size=n_pair)
        diff = (sp[ip] - sn[jn]) / tau
        # softplus(-diff)
        loss = np.mean(np.logaddexp(0.0, -diff))
        sig = expit(-diff)  # d/ddiff log(1+e^{-d}) = -σ(-d) wait: d/dx log(1+e^{-x}) = -σ(-x)? 
        # d/ddiff log(1+exp(-diff)) = -sigmoid(diff)? Actually σ(-diff)*(-1)? 
        # let u=-diff: log(1+e^u), d/ddiff = d/du * du/ddiff = σ(u)*(-1) = -σ(-diff)
        g_diff = -expit(-diff)  # shape (n_pair,)
        # d loss / dw = mean( g_diff * (Xp[ip]-Xn[jn])/tau )
        g = ((g_diff / tau)[:, None] * (Xp[ip] - Xn[jn])).mean(0)
        # L2
        reg = 1e-3
        loss = loss + 0.5 * reg * float(w @ w)
        g = g + reg * w
        return loss, g

    best_b = {"nested": -1.0}
    for tau in (0.5, 1.0, 2.0):
        nested = np.zeros(n)
        for a, b in folds:
            Xp = Xo[a][y[a] == 1]
            Xn = Xo[a][y[a] == 0]
            w = np.zeros(Xo.shape[1])
            # L-BFGS-ish: simple gradient descent / scipy
            from scipy.optimize import minimize

            def fun(wv):
                loss, g = ranknet_loss_and_grad(wv, Xp, Xn, tau=tau)
                return loss, g

            opt = minimize(lambda wv: fun(wv)[0], w, jac=lambda wv: fun(wv)[1], method="L-BFGS-B", options={"maxiter": 40})
            nested[b] = rk(Xo[b] @ opt.x)
        auc = float(roc_auc_score(y, nested))
        log(f"  L5-4B RankNet τ={tau} nested={auc:.7f} Δ={auc - base_nested:+.3e}")
        if auc > best_b["nested"]:
            # full refit
            Xp = Xo[y == 1]
            Xn = Xo[y == 0]
            from scipy.optimize import minimize

            def fun(wv):
                return ranknet_loss_and_grad(wv, Xp, Xn, tau=tau)

            opt = minimize(lambda wv: fun(wv)[0], np.zeros(Xo.shape[1]), jac=lambda wv: fun(wv)[1], method="L-BFGS-B", options={"maxiter": 60})
            best_b = {"nested": auc, "tau": tau, "oof": nested, "test": rk(Xt @ opt.x), "w": opt.x.tolist()}

    # pick best of 4A/4B
    cands = [("overlap_newton", best_a), ("ranknet", best_b)]
    pick_name, pick = max(cands, key=lambda x: x[1]["nested"])
    a0, b0 = folds[0]
    fold0 = float(roc_auc_score(y[b0], pick["oof"][b0]))
    promoted = pick["nested"] > base_nested + 1e-7
    res = {
        "method": pick_name,
        "nested": pick["nested"],
        "fold0": fold0,
        "delta_vs_l4v2": pick["nested"] - base_nested,
        "promoted": promoted,
        "overlap_best": {k: v for k, v in best_a.items() if k not in ("oof", "test")},
        "ranknet_best": {k: v for k, v in best_b.items() if k not in ("oof", "test")},
        "features": col_names,
    }
    if not promoted:
        mark_dead(state, "L5_auc_direct", "did not beat L4_v2 nested", res)
    state["step4"] = res
    save_state(state)
    log(f"  L5 pick={pick_name} nested={pick['nested']:.7f} promoted={promoted}")
    return (pick["oof"], pick["test"], res) if promoted else None


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — assemble + audit
# ═══════════════════════════════════════════════════════════════════════════
def build_anchors(y, sub_ids):
    from part8_proprietary_scan import build_p7_65
    from part7_clean_superstack import fit_test

    p7_names, P7_O, P7_T, _ = build_p7_65(y)
    P9_T, _ = wire_testbags(p7_names, P7_T)
    te65, _ = fit_test(P7_O, P9_T, y, C=0.01, tol=1e-8, max_iter=200)
    te62, _ = fit_test(P7_O[:, :62], P9_T[:, :62], y, C=0.01, tol=1e-8, max_iter=200)
    part9 = rk(rk(te65) + rk(te62))
    amanatar = (
        pd.read_csv("external_bank/amanatar_sovereign/submission.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    part10 = rk(rk(part9) + rk(amanatar))
    areeb = (
        pd.read_csv("external_bank/areebbb_95239_out/submission.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    nina = (
        pd.read_csv("external_bank/nina2025_eos4_out/submission.csv")
        .set_index("id")
        .loc[sub_ids, "is_fraud"]
        .to_numpy(float)
    )
    return {
        "goodpjw_part9": part9,
        "amanatar_sovereign": amanatar,
        "goodpjw_part10": part10,
        "areebbb_95239": areeb,
        "nina2025_eos4": nina,
    }


def main():
    log(f"PART11 GRANDMASTER AUC ENGINE — {POLICY}")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    folds = folds_y(y)
    sub_ids = pd.read_csv("data/sample_submission.csv")["id"].to_numpy()
    state = {
        "gates": {
            "V3_NESTED": V3_NESTED,
            "V3_FOLD0": V3_FOLD0,
            "L4_V3_NESTED": L4_V3_NESTED,
            "V4_NESTED": V4_NESTED,
        },
        "dead_ends": [],
        "policy": POLICY,
    }
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            for k in ("step1", "step1_recon", "step1_raw_members", "v3_verify"):
                if k in prev:
                    state[k] = prev[k]
        except Exception:
            pass
    save_state(state)

    log("building P_super_v3 bank …")
    names_v3, O_v3, T_v3, wire_info, glm_o, glm_t = build_v3_bank(y, folds)
    bank_ctx = {
        "gb5_o": np.load(ART / "gb_bag5_oof.npy").ravel().astype(float),
        "gb5_t": np.load(ART / "gb_bag5_full.npy").ravel().astype(float),
        "gb5_t_bag": np.load(ART / "gb_testbag_full.npy").ravel().astype(float),
        "bag_o": np.load("preds/gb_gpt2_bag5_oof.npy").ravel().astype(float),
        "bag_t": np.load("preds/gb_gpt2_bag5_full.npy").ravel().astype(float),
        "glm_o": glm_o,
        "glm_t": glm_t,
        "wire_info": wire_info,
    }
    verify = nested_psuper(O_v3, T_v3, y, folds, bank_ctx["gb5_o"], bank_ctx["bag_o"], bank_ctx["gb5_t_bag"], bank_ctx["bag_t"])
    log(f"verify V3 nested={verify['nested']:.7f} fold0={verify['fold0']:.7f}")
    state["v3_verify"] = {"nested": verify["nested"], "fold0": verify["fold0"]}
    save_state(state)

    log("building public anchors …")
    anchors = build_anchors(y, sub_ids)
    part10_te, amanatar_te = anchors["goodpjw_part10"], anchors["amanatar_sovereign"]

    if Path("preds/part11_step1.npz").exists() and state.get("step1") and os.environ.get("FORCE_STEP1", "0") != "1":
        log("resuming STEP1 from preds/part11_step1.npz")
        z = np.load("preds/part11_step1.npz", allow_pickle=True)
        s1 = {
            "blends": {},
            "orth_members": [],
            "raw_members": [],
            "O_orth": z["O_orth"],
            "T_orth": z["T_orth"],
            "O_unbundled": z["O_unbundled"],
            "T_unbundled": z["T_unbundled"],
            "names_unbundled": list(z["names_unbundled"]),
            "names_orth": list(z["names_orth"]),
        }
    else:
        s1 = step1(state, y, folds, names_v3, O_v3, T_v3, bank_ctx, sub_ids, part10_te, amanatar_te)
    import gc
    gc.collect()
    log(f"mem checkpoint pre-S2; O_unbundled={s1['O_unbundled'].shape} O_orth={s1['O_orth'].shape}")
    s2 = step2(state, y, folds, names_v3, O_v3, T_v3, bank_ctx)

    # banks for stacking
    O_pure, T_pure = s1["O_unbundled"], s1["T_unbundled"]
    names_pure = s1["names_unbundled"]
    if s2 is not None:
        O_pure = np.column_stack([O_pure, rk(s2[0])])
        T_pure = np.column_stack([T_pure, rk(s2[1])])
        names_pure = names_pure + ["s_extremal"]

    O_full = np.column_stack([s1["O_orth"]] + ([rk(s2[0])] if s2 else []))
    T_full = np.column_stack([s1["T_orth"]] + ([rk(s2[1])] if s2 else []))
    names_full = s1["names_orth"] + (["s_extremal"] if s2 else [])

    l4_pure = step3(state, y, folds, names_pure, O_pure, T_pure, bank_ctx, extremal=s2)
    state["step3_pure"] = {k: v for k, v in l4_pure[2].items() if k not in ("oof", "test")}
    save_state(state)

    l5 = step4(
        state, y, folds, names_pure, O_pure, T_pure, bank_ctx, l4_pure[0], l4_pure[1], l4_pure[2]["nested"]
    )

    # Assemble v5_pure
    if l5 is not None:
        p_pure_o, p_pure_t = l5[0], l5[1]
        pure_nested = l5[2]["nested"]
        pure_recipe = "L5_auc_direct"
    else:
        p_pure_o, p_pure_t = l4_pure[0], l4_pure[1]
        pure_nested = l4_pure[2]["nested"]
        pure_recipe = "L4_v2_TwoStage"

    # Assemble v5_full = L4_v2 on orth bank (with extremal if any)
    l4_full = step3(state, y, folds, names_full, O_full, T_full, bank_ctx, extremal=s2)
    state["step3_full"] = {k: v for k, v in l4_full[2].items() if k not in ("oof", "test")}
    # optional L5 on full
    l5_full = step4(
        state, y, folds, names_full, O_full, T_full, bank_ctx, l4_full[0], l4_full[1], l4_full[2]["nested"]
    )
    if l5_full is not None:
        p_full_o, p_full_t = l5_full[0], l5_full[1]
        full_nested = l5_full[2]["nested"]
        full_recipe = "L5_on_orth_bank"
    else:
        p_full_o, p_full_t = l4_full[0], l4_full[1]
        full_nested = l4_full[2]["nested"]
        full_recipe = "L4_v2_on_orth_bank"

    np.save("preds/P_super_v5_pure_oof.npy", np.asarray(p_pure_o, np.float32))
    np.save("preds/P_super_v5_pure_test.npy", np.asarray(p_pure_t, np.float32))
    np.save("preds/P_super_v5_full_oof.npy", np.asarray(p_full_o, np.float32))
    np.save("preds/P_super_v5_full_test.npy", np.asarray(p_full_t, np.float32))

    a0, b0 = folds[0]
    progression = [
        {"name": "P_super_v3", "nested": V3_NESTED, "fold0": V3_FOLD0},
        {"name": "L4_regime_v3", "nested": L4_V3_NESTED, "fold0": state.get("step3_pure", {}).get("fold0")},
        {"name": "P_super_v4", "nested": V4_NESTED, "fold0": 0.9513081756191063},
        {
            "name": "P_super_v5_pure",
            "nested": float(roc_auc_score(y, p_pure_o)),
            "fold0": float(roc_auc_score(y[b0], p_pure_o[b0])),
            "recipe": pure_recipe,
        },
        {
            "name": "P_super_v5_full",
            "nested": float(roc_auc_score(y, p_full_o)),
            "fold0": float(roc_auc_score(y[b0], p_full_o[b0])),
            "recipe": full_recipe,
        },
    ]

    vecs = {
        "P_super_v5_pure": p_pure_t,
        "P_super_v5_full": p_full_t,
        **anchors,
    }
    dist = {a: {b: (0.0 if a == b else float(d_e4(vecs[a], vecs[b]))) for b in vecs} for a in vecs}

    state["v5"] = {
        "pure": {
            "nested": progression[3]["nested"],
            "fold0": progression[3]["fold0"],
            "recipe": pure_recipe,
            "files": ["preds/P_super_v5_pure_oof.npy", "preds/P_super_v5_pure_test.npy"],
        },
        "full": {
            "nested": progression[4]["nested"],
            "fold0": progression[4]["fold0"],
            "recipe": full_recipe,
            "files": ["preds/P_super_v5_full_oof.npy", "preds/P_super_v5_full_test.npy"],
        },
        "progression": progression,
        "test_distance_e4": dist,
    }
    save_state(state)

    print("\n" + "=" * 88)
    print("SUMMARY — PART11 GRANDMASTER AUC ENGINE (NO_SUBMIT)")
    print("=" * 88)
    print("progression:")
    for p in progression:
        print(f"  {p['name']:22s} nested={p['nested']:.7f} fold0={p.get('fold0')}")
    print(f"dead_ends ({len(state['dead_ends'])}):")
    for d in state["dead_ends"]:
        print(f"  - {d['name']}: {d['reason']}")
    print("test (1-ρ)×1e4:")
    keys = list(vecs)
    print(f"{'':22s}" + "".join(f"{k[:12]:>12s}" for k in keys))
    for a in keys:
        print(f"{a:22s}" + "".join(f"{dist[a][b]:12.2f}" for b in keys))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
