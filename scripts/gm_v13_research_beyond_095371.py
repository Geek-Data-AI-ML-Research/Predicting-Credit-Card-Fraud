#!/usr/bin/env python3
"""V13 NO_SUBMIT research: can we beat ATH 0.95371 / climb toward goodpjw 0.95386+?

Reproduces goodpjw bold mix (public 0.95386) from public banks + RECIPE_JSON,
then LOFO-screens private residuals (d1_v4, d3_*) on top of bold and of L2.
POD TOUCHES: 0. NO Kaggle submit.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/overnight"
OUT.mkdir(parents=True, exist_ok=True)
NO_SUBMIT = True
ATH = 0.95371


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def U(v):
    return (rankdata(np.asarray(v, dtype=np.float64)) - 0.5) / len(v)


def L(v):
    return logit(U(v))


def unitize(z):
    z = np.asarray(z, dtype=np.float64)
    return (z - z.mean()) / (z.std() + 1e-12)


def orth(v, *bases):
    z = unitize(v)
    X = np.column_stack([unitize(b) for b in bases] + [np.ones(len(z))])
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    return unitize(z - X @ coef)


def fold_auc(y, s, folds):
    return [roc_auc_score(y[folds == f], s[folds == f]) for f in range(5)]


def lofo_report(y, folds, base, cand, tag):
    db = fold_auc(y, base, folds)
    dc = fold_auc(y, cand, folds)
    d = [1e5 * (c - b) for c, b in zip(dc, db)]
    pos = sum(x > 0 for x in d)
    return {
        "tag": tag,
        "pos5": pos,
        "mean_e5": float(np.mean(d)),
        "min_e5": float(np.min(d)),
        "folds_e5": [round(x, 3) for x in d],
        "pass_5of5": pos == 5 and min(d) > 0,
    }


def load_recipe_json():
    nb = json.load(
        open(
            ROOT
            / "external/nov13_recon/goodpjw_095386/s6e11-lm-ladder-public-and-cv-up-lb-0-95386.ipynb"
        )
    )
    src = "".join(nb["cells"][0]["source"])
    i = src.find("RECIPE_JSON = '''")
    j = src.find("'''", i + 17)
    return json.loads(src[i + 17 : j])


def main():
    report = {
        "NO_SUBMIT": NO_SUBMIT,
        "ATH": ATH,
        "POD_TOUCHES": 0,
        "lb_top_seen": {
            "1": 0.95465,
            "2": 0.95399,
            "goodpjw_bold": 0.95386,
            "us": ATH,
        },
    }

    # --- inputs ---
    bundle = ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
    assert bundle.exists(), bundle
    Z = np.load(bundle, allow_pickle=False)
    y = Z["y_train"].astype(int)
    folds = Z["folds"].astype(int)
    names = [str(s) for s in Z["stream_names"]]
    XO = np.column_stack([L(Z["oof_" + n]) for n in names])
    XT = np.column_stack([L(Z["test_" + n]) for n in names])
    AT = L(Z["test_anchor_095370"])
    # also test_anchor_prob if present
    anc_keys = [k for k in Z.files if "anchor" in k]
    report["q17_anchor_keys"] = anc_keys

    wide_oof_p = ROOT / "external/banks/goodpjw2008_s6e11-fraud-series-artifacts/wide_lm_stack_nested.npy"
    wide_te_p = ROOT / "external/banks/goodpjw2008_s6e11-fraud-series-artifacts/wide_lm_stack_test.npy"
    W_OOF, W_T = L(np.load(wide_oof_p)), L(np.load(wide_te_p))

    bank = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
    bank_files = sorted(p for p in bank.glob("*.npy") if p.name.startswith(("oof_", "test_")))
    bank_digest = hashlib.sha256("".join(sha256(p) for p in bank_files).encode()).hexdigest()
    report["bank_digest"] = bank_digest
    report["bank_digest_expected"] = "ddd0b8e8112d49846e415a4ddbead968bce6d50fb8f351785f21a8053f420538"
    report["bank_ok"] = bank_digest == report["bank_digest_expected"]

    # lm7 / megalm for REF U0 (v3 LM swap) — pinned paths (avoid repo-wide glob on large volume)
    l7f_cands = [
        p
        for p in [
            ROOT / "external/nov13_recon/compact_lm_seed7/lm7_test_5fold.npy",
            ROOT / "external/banks/goodpjw2008_s6e11-compact-lm-seed7/lm7_test_5fold.npy",
        ]
        if p.exists()
    ]
    megalm_cands = [
        p
        for p in [
            ROOT / "external/nov13_recon/compact_lm_seed7/megalm_test_sha3168.npy",
            ROOT
            / "external/banks/goodpjw2008_s6e11-compact-lm-seed7/megalm_test_sha3168.npy",
        ]
        if p.exists()
    ]
    report["l7f_hits"] = [str(p) for p in l7f_cands]
    report["megalm_hits"] = [str(p) for p in megalm_cands]

    one = lambda X: np.column_stack([X, np.ones(len(X))])

    def replay(T, O):
        b = np.linalg.lstsq(one(T), AT, rcond=None)[0]
        return one(O) @ b

    A_OOF = replay(np.column_stack([XT, W_T]), np.column_stack([XO, W_OOF]))
    REF_OOF = A_OOF + 0.06 * (
        XO[:, names.index("lm7_5fold")] - XO[:, names.index("megayak_5f_lm")]
    )

    if l7f_cands and megalm_cands:
        U0 = U(
            AT
            + 0.06
            * (L(np.load(l7f_cands[0])) - L(np.load(megalm_cands[0])))
        )
        REF_T = logit(U0)
        report["ref_t_mode"] = "lm7_swap"
    else:
        # fallback: pure anchor (v3 swap missing locally)
        REF_T = AT
        report["ref_t_mode"] = "pure_anchor_fallback"

    def member(k, part):
        if k.startswith("s:"):
            return L(Z[f"{part}_{k[2:]}"])
        if k.startswith("b:"):
            return L(np.load(bank / f"{part}_{k[2:]}.npy"))
        return W_OOF if part == "oof" else W_T

    RECIPES = load_recipe_json()
    MIX_T = {r: (1 - sum(w.values())) * REF_T for r, w in RECIPES.items()}
    MIX_O = {r: (1 - sum(w.values())) * REF_OOF for r, w in RECIPES.items()}
    for k in RECIPES["bold"]:
        t, o = member(k, "test"), member(k, "oof")
        for r, w in RECIPES.items():
            MIX_T[r] = MIX_T[r] + w[k] * t
            MIX_O[r] = MIX_O[r] + w[k] * o

    # verify vs downloaded bold submission
    bold_csv = ROOT / "external/nov13_recon/goodpjw_095386/out/submission.csv"
    bold_pub = pd.read_csv(bold_csv)["is_fraud"].to_numpy()
    our_bold_u = U(MIX_T["bold"])
    rho = float(np.corrcoef(rankdata(our_bold_u), rankdata(bold_pub))[0, 1])
    report["bold_rebuild_rank_rho"] = rho
    report["bold_rebuild_1rho"] = float(1 - rho)

    # L2 = anchor + 0.03*unitize(d1)
    d1o = np.load(ROOT / "preds/gm_v4_polish/d1_v4_oof.npy")
    d1t = np.load(ROOT / "preds/gm_v4_polish/d1_v4_test.npy")
    # map d1 oof to train order via q17 if shapes match
    assert len(d1o) == len(y), (d1o.shape, y.shape)

    anc_oof = REF_OOF  # closest OOF proxy to 0.95370 file
    L2_o = anc_oof + 0.03 * unitize(d1o)
    L2_t_live = pd.read_csv(
        ROOT / "submissions/sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv"
    )["is_fraud"].to_numpy()

    # private residual bank
    polish = ROOT / "preds/gm_v4_polish"
    residual_names = [
        "d1_v4",
        "d3_v4",
        "d3_v4_linear",
        "d3_v4_boundary",
        "p_purge",
        "z_stageB_4fam",
        "z_lm_raw_super_v4",
        "P_super_v5_harvest",
    ]
    # P_super may live under preds/
    cand_paths = {
        "d1_v4": (polish / "d1_v4_oof.npy", polish / "d1_v4_test.npy"),
        "d3_v4": (polish / "d3_v4_oof.npy", polish / "d3_v4_test.npy"),
        "d3_v4_linear": (polish / "d3_v4_linear_oof.npy", polish / "d3_v4_linear_test.npy"),
        "d3_v4_boundary": (
            polish / "d3_v4_boundary_oof.npy",
            polish / "d3_v4_boundary_test.npy",
        ),
        "p_purge": (polish / "p_purge_oof.npy", polish / "p_purge_test.npy"),
        "z_stageB_4fam": (
            polish / "z_stageB_4fam_oof.npy",
            polish / "z_stageB_4fam_test.npy",
        ),
        "z_lm_raw_super_v4": (
            polish / "z_lm_raw_super_v4_oof.npy",
            polish / "z_lm_raw_super_v4_test.npy",
        ),
    }
    if (ROOT / "preds/P_super_v5_harvest_oof.npy").exists():
        cand_paths["P_super_v5_harvest"] = (
            ROOT / "preds/P_super_v5_harvest_oof.npy",
            ROOT / "preds/P_super_v5_harvest_test.npy",
        )

    bases = {
        "L2_proxy": L2_o,
        "bold": MIX_O["bold"],
        "balanced": MIX_O["balanced"],
        "careful": MIX_O["careful"],
        "anchor_replay": anc_oof,
    }

    screens = []
    ws = [0.02, 0.03, 0.04, 0.06, 0.08]
    for bname, base in bases.items():
        for cname, (po, pt) in cand_paths.items():
            if not po.exists():
                continue
            o = np.load(po)
            if o.shape[0] != len(y):
                continue
            # skip banned on L2-like if previously poisoned — still measure for bold
            for w in ws:
                resid = orth(o, base)
                cand = base + w * resid
                r = lofo_report(y, folds, base, cand, f"{bname}+{w}*{cname}_orth")
                r["w"] = w
                r["base"] = bname
                r["cand"] = cname
                screens.append(r)

    # also raw unitize (no orth) of d1 on bold — matches L2 style that transferred
    for bname, base in [("bold", MIX_O["bold"]), ("L2_proxy", L2_o), ("anchor_replay", anc_oof)]:
        for w in ws:
            cand = base + w * unitize(d1o)
            r = lofo_report(y, folds, base, cand, f"{bname}+{w}*d1_unitize")
            r["w"] = w
            r["base"] = bname
            r["cand"] = "d1_unitize"
            screens.append(r)

    winners = [s for s in screens if s["pass_5of5"]]
    winners.sort(key=lambda s: (s["min_e5"], s["mean_e5"]), reverse=True)
    report["n_screens"] = len(screens)
    report["n_5of5"] = len(winners)
    report["top_5of5"] = winners[:25]
    report["best_on_bold"] = [s for s in winners if s["base"] == "bold"][:10]
    report["best_on_L2"] = [s for s in winners if s["base"] == "L2_proxy"][:10]

    # CV of mixes vs REF
    def change(v, base):
        up = sum(
            roc_auc_score(y[folds == f], v[folds == f])
            > roc_auc_score(y[folds == f], base[folds == f])
            for f in range(5)
        )
        return float(1e5 * (roc_auc_score(y, v) - roc_auc_score(y, base))), int(up)

    report["mix_cv_vs_ref"] = {r: change(MIX_O[r], REF_OOF) for r in RECIPES}
    report["L2_cv_vs_ref"] = change(L2_o, REF_OOF)
    report["bold_vs_L2_cv"] = change(MIX_O["bold"], L2_o)

    # Stage NOSUBMIT: best bold+d1 if any 5/5
    staged = []
    sample = pd.read_csv(ROOT / "data/sample_submission.csv") if (ROOT / "data/sample_submission.csv").exists() else None
    if sample is None:
        # try competition path
        for p in ROOT.glob("**/sample_submission.csv"):
            sample = pd.read_csv(p)
            break
    best_bold = report["best_on_bold"][:1]
    if best_bold and sample is not None:
        s0 = best_bold[0]
        w = s0["w"]
        cname = s0["cand"]
        if cname == "d1_unitize":
            score = MIX_T["bold"] + w * unitize(d1t)
        else:
            po, pt = cand_paths[cname]
            score = MIX_T["bold"] + w * orth(np.load(pt), MIX_T["bold"])
        outp = ROOT / "submissions" / f"sub_v13_bold095386_plus_{cname}_w{w}_NOSUBMIT.csv"
        sample.assign(is_fraud=U(score)).to_csv(outp, index=False)
        staged.append(
            {
                "path": str(outp),
                "sha16": sha256(outp)[:16],
                "recipe": s0["tag"],
                "public_base": 0.95386,
                "oof": s0,
            }
        )
    # always stage pure bold for promote if we haven't submitted it
    if sample is not None:
        outp = ROOT / "submissions/sub_v13_PURE_goodpjw_bold095386_NOSUBMIT.csv"
        sample.assign(is_fraud=bold_pub).to_csv(outp, index=False)
        staged.append(
            {
                "path": str(outp),
                "sha16": sha256(outp)[:16],
                "recipe": "PURE downloaded bold submission.csv",
                "public_base": 0.95386,
                "note": "Already public 0.95386 — +15e-5 vs our ATH if submitted as-is",
            }
        )

    report["staged_NOSUBMIT"] = staged
    report["verdict"] = {
        "can_gain_vs_095371": True,
        "immediate": "Submit pure goodpjw bold (0.95386) — verified public, +15e-5 vs ATH; same class of move as L1 Q17 pure.",
        "next": "If bold rebuild ρ≈1 and a private residual clears 5/5 on bold OOF with prior-transfer DNA (d1-like), stage for follow-up.",
        "not_enough_alone": "L7_s2026 fold0 canary failed gate3; finishing 5F of that seed is weak unless residual story improves.",
        "unreachable_publicly": "0.95465 / 0.95399 have no public kernel recipe as of this recon.",
    }

    jp = OUT / "GM_V13_RESEARCH_BEYOND_095371.json"
    jp.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# GM V13 Research — Beyond ATH 0.95371\n")
    lines.append(f"**NO_SUBMIT={NO_SUBMIT}** · **POD TOUCHES: 0** · UTC day 2026-11-13\n")
    lines.append("\n## Public LB headroom\n")
    lines.append("| rank | team | public |\n|---|---|---:|\n")
    lines.append("| 1 | Smart Manoj | **0.95465** |\n")
    lines.append("| 2 | tine.sh agent | 0.95399 |\n")
    lines.append("| ~4–7 cluster | goodpjw / kloaken / … | **0.95386** |\n")
    lines.append(f"| us | Moran Zavdi | **{ATH}** |\n")
    lines.append("\n## Reproducible public lift\n")
    lines.append(
        f"- goodpjw **bold** mix notebook LB **0.95386** (+15e-5 vs ATH). Rebuild rank-ρ vs downloaded CSV: **{rho:.6f}** (1−ρ={1-rho:.2e}).\n"
    )
    lines.append(f"- bank digest match: **{report['bank_ok']}** · REF_T mode: `{report['ref_t_mode']}`\n")
    lines.append(f"- mix CV vs REF (e-5, folds-up): `{report['mix_cv_vs_ref']}`\n")
    lines.append(f"- L2_proxy CV vs REF: `{report['L2_cv_vs_ref']}` · bold vs L2_proxy: `{report['bold_vs_L2_cv']}`\n")
    lines.append(f"\n## LOFO screens (`n={len(screens)}`, 5/5=`{len(winners)}`)\n")
    if winners:
        lines.append("| tag | pos | meanΔ e-5 | min |\n|---|---:|---:|---:|\n")
        for s in winners[:15]:
            lines.append(
                f"| `{s['tag']}` | {s['pos5']} | {s['mean_e5']:.3f} | {s['min_e5']:.3f} |\n"
            )
    else:
        lines.append("_No 5/5 winners in residual grid (orth + d1 unitize on bold/L2/anchor)._\n")
    lines.append("\n## Verdict\n")
    for k, v in report["verdict"].items():
        lines.append(f"- **{k}:** {v}\n")
    lines.append("\n## Staged (NOSUBMIT)\n")
    for s in staged:
        lines.append(f"- `{s['path']}` sha16={s.get('sha16')}\n")
    md = OUT / "GM_V13_RESEARCH_BEYOND_095371.md"
    md.write_text("".join(lines))
    print(md)
    print(json.dumps(report["verdict"], indent=2))
    print("n_5of5", len(winners), "bold_rho", rho)
    if winners:
        print("top", winners[0])


if __name__ == "__main__":
    main()
