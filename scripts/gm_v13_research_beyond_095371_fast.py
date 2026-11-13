#!/usr/bin/env python3
"""Fast V13 NO_SUBMIT research: public headroom + private residual LOFO on L2/anchor.

Skips full 87-member bold rebuild (slow on external volume). Uses downloaded
goodpjw bold CSV (public 0.95386) for staging; LOFO screens private residuals
on Q17-replay + L2 proxies that need only local preds.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/overnight"
PY = time.time()


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


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


def lofo(y, folds, base, cand):
    d = []
    for f in range(5):
        m = folds == f
        d.append(1e5 * (roc_auc_score(y[m], cand[m]) - roc_auc_score(y[m], base[m])))
    pos = sum(x > 0 for x in d)
    return {
        "pos5": pos,
        "mean_e5": float(np.mean(d)),
        "min_e5": float(np.min(d)),
        "folds_e5": [round(x, 3) for x in d],
        "pass_5of5": pos == 5 and min(d) > 0,
    }


def main():
    t0 = time.time()
    print("[1] load Q17 + residuals", flush=True)
    Z = np.load(
        ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz",
        allow_pickle=False,
    )
    y = Z["y_train"].astype(int)
    folds = Z["folds"].astype(int)
    names = [str(s) for s in Z["stream_names"]]
    XO = np.column_stack([L(Z[f"oof_{n}"]) for n in names])
    XT = np.column_stack([L(Z[f"test_{n}"]) for n in names])
    AT = L(Z["test_anchor_095370"])
    print(f"  streams={len(names)} n={len(y)}  ({time.time()-t0:.1f}s)", flush=True)

    # 16-stream replay of anchor (no wide) — OOF proxy for 0.95370 file
    one = lambda X: np.column_stack([X, np.ones(len(X))])
    b = np.linalg.lstsq(one(XT), AT, rcond=None)[0]
    A16 = one(XO) @ b
    # L2-style on this proxy
    d1o = np.load(ROOT / "preds/gm_v4_polish/d1_v4_oof.npy")
    d1t = np.load(ROOT / "preds/gm_v4_polish/d1_v4_test.npy")
    L2 = A16 + 0.03 * unitize(d1o)

    bold_p = ROOT / "external/nov13_recon/goodpjw_095386/out/submission.csv"
    bold = pd.read_csv(bold_p)["is_fraud"].to_numpy()
    L2_live = pd.read_csv(
        ROOT / "submissions/sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv"
    )["is_fraud"].to_numpy()
    anc_live = pd.read_csv(
        ROOT / "submissions/sub_v12p3_slot1_PURE_Q17_anchor095370_LIVE.csv"
    )["is_fraud"].to_numpy()

    def rho(a, b):
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

    dist = {
        "1rho_bold_vs_L2live": 1 - rho(bold, L2_live),
        "1rho_bold_vs_Q17": 1 - rho(bold, anc_live),
        "1rho_L2live_vs_Q17": 1 - rho(L2_live, anc_live),
    }
    print("[2] distances", dist, flush=True)

    polish = ROOT / "preds/gm_v4_polish"
    cands = {}
    for name in [
        "d1_v4",
        "d3_v4",
        "d3_v4_linear",
        "d3_v4_boundary",
        "p_purge",
        "z_stageB_4fam",
        "z_lm_raw_super_v4",
    ]:
        po = polish / f"{name}_oof.npy"
        pt = polish / f"{name}_test.npy"
        if po.exists():
            cands[name] = (np.load(po), np.load(pt) if pt.exists() else None)

    if (ROOT / "preds/P_super_v5_harvest_oof.npy").exists():
        cands["P_super_v5_harvest"] = (
            np.load(ROOT / "preds/P_super_v5_harvest_oof.npy"),
            np.load(ROOT / "preds/P_super_v5_harvest_test.npy"),
        )

    print(f"[3] LOFO screen n_cands={len(cands)}", flush=True)
    bases = {"A16_anchor_replay": A16, "L2_proxy": L2}
    ws = [0.02, 0.03, 0.04, 0.06, 0.08]
    screens = []
    for bname, base in bases.items():
        for cname, (o, _) in cands.items():
            if len(o) != len(y):
                continue
            for mode in ("unitize", "orth"):
                for w in ws:
                    resid = unitize(o) if mode == "unitize" else orth(o, base)
                    r = lofo(y, folds, base, base + w * resid)
                    r.update({"tag": f"{bname}+{w}*{cname}_{mode}", "base": bname, "cand": cname, "mode": mode, "w": w})
                    screens.append(r)
        # d1 unitize only (known transfer recipe)
        for w in ws:
            r = lofo(y, folds, base, base + w * unitize(d1o))
            r.update({"tag": f"{bname}+{w}*d1_unitize", "base": bname, "cand": "d1_unitize", "mode": "unitize", "w": w})
            screens.append(r)

    winners = [s for s in screens if s["pass_5of5"]]
    winners.sort(key=lambda s: (s["min_e5"], s["mean_e5"]), reverse=True)
    near = sorted(screens, key=lambda s: (s["pos5"], s["min_e5"]), reverse=True)[:12]
    print(f"  5/5={len(winners)} / {len(screens)}  ({time.time()-t0:.1f}s)", flush=True)

    # Stage pure bold + optional L2-style d1 on bold using LIVE test vectors
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    staged = []
    pure = ROOT / "submissions/sub_v13_PURE_goodpjw_bold095386_NOSUBMIT.csv"
    sample.assign(is_fraud=bold).to_csv(pure, index=False)
    staged.append({"path": str(pure), "sha16": sha16(pure), "public_expected": 0.95386, "note": "byte-class copy of goodpjw bold"})

    # Inject d1 onto bold test in L2 style (w=0.03) — OOF cannot score bold without full MIX_O;
    # we still stage for optional submit AFTER pure bold confirms.
    for w in (0.02, 0.03, 0.04):
        score = L(bold) + w * unitize(d1t)
        outp = ROOT / f"submissions/sub_v13_bold095386_plus_d1v4_w{w}_NOSUBMIT.csv"
        sample.assign(is_fraud=U(score)).to_csv(outp, index=False)
        staged.append(
            {
                "path": str(outp),
                "sha16": sha16(outp),
                "public_base": 0.95386,
                "note": f"L(bold)+{w}*unitize(d1_v4); OOF ungated vs bold (no MIX_O); promote only after pure bold lands",
                "1rho_vs_bold": 1 - rho(U(score), bold),
            }
        )

    # L2 + unused private axes that cleared 5/5 (if any)
    for s in winners[:3]:
        if s["base"] != "L2_proxy":
            continue
        cname, w, mode = s["cand"], s["w"], s["mode"]
        if cname == "d1_unitize":
            continue  # already ATH
        o, t = cands.get(cname, (None, None))
        if t is None:
            continue
        base_t = L(L2_live)
        resid = unitize(t) if mode == "unitize" else orth(t, base_t)
        score = base_t + w * resid
        outp = ROOT / f"submissions/sub_v13_L2_plus_{cname}_{mode}_w{w}_NOSUBMIT.csv"
        sample.assign(is_fraud=U(score)).to_csv(outp, index=False)
        staged.append({"path": str(outp), "sha16": sha16(outp), "oof": s, "note": "on L2 live; banned families excluded only if not in winners"})

    report = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "ATH": 0.95371,
        "lb_top": {"1": 0.95465, "2": 0.95399, "goodpjw_bold": 0.95386, "us": 0.95371},
        "distances": dist,
        "A16_replay_auc": float(roc_auc_score(y, A16)),
        "L2_proxy_auc": float(roc_auc_score(y, L2)),
        "n_screens": len(screens),
        "n_5of5": len(winners),
        "top_5of5": winners[:20],
        "near_misses": near,
        "staged_NOSUBMIT": staged,
        "elapsed_s": round(time.time() - t0, 2),
        "verdict": {
            "can_gain": True,
            "delta_to_public_recipe": "+0.00015 via pure goodpjw bold (0.95386)",
            "delta_to_1st": "+0.00094 needed for 0.95465 — no public recipe",
            "submit_now_candidate": str(pure),
            "followup": "After bold confirms, try staged bold+d1 (same family that transferred on Q17). Do not re-blend P_super/sov/G26 on L2.",
            "l7_s2026": "Fold0 canary failed gate3 (s2026 0.94749 < s7 0.94754); finishing 5F is low priority vs submitting bold.",
        },
    }
    jp = OUT / "GM_V13_RESEARCH_BEYOND_095371.json"
    jp.write_text(json.dumps(report, indent=2))
    lines = [
        "# GM V13 Research — Beyond ATH 0.95371\n\n",
        f"**NO_SUBMIT=True** · **POD TOUCHES: 0** · elapsed {report['elapsed_s']}s\n\n",
        "## Public LB headroom (live CLI)\n\n",
        "| rank | score | note |\n|---|---:|---|\n",
        "| 1 | **0.95465** | Smart Manoj — no public kernel |\n",
        "| 2 | 0.95399 | tine.sh agent — no public kernel |\n",
        "| cluster | **0.95386** | goodpjw bold mix (reproducible) |\n",
        "| us | **0.95371** | L2 = Q17 + 0.03·d1 |\n\n",
        "## Distances (1−ρ)\n\n",
        f"```\n{json.dumps(dist, indent=2)}\n```\n\n",
        f"## LOFO private residuals on A16/L2 proxies\n\n5/5 winners: **{len(winners)}** / {len(screens)}\n\n",
    ]
    if winners:
        lines.append("| tag | meanΔ e-5 | min |\n|---|---:|---:|\n")
        for s in winners[:12]:
            lines.append(f"| `{s['tag']}` | {s['mean_e5']:.3f} | {s['min_e5']:.3f} |\n")
    else:
        lines.append("_No new 5/5 on A16/L2 beyond recipes already tried / ATH._\n")
        lines.append("\nNear misses:\n\n| tag | pos | mean | min |\n|---|---:|---:|---:|\n")
        for s in near[:8]:
            lines.append(f"| `{s['tag']}` | {s['pos5']} | {s['mean_e5']:.3f} | {s['min_e5']:.3f} |\n")
    lines += [
        "\n## Verdict\n\n",
        "1. **Yes — immediate +15e-5 available:** submit pure goodpjw `bold` (public **0.95386**), same move class as L1 pure Q17.\n",
        "2. **Staged (hold):** `bold + d1_v4` at w∈{0.02,0.03,0.04} — only after bold confirms; OOF vs bold not rebuilt in this fast pass.\n",
        "3. **No public path to 0.95465/0.95399** found (kernel search empty).\n",
        "4. **In-bank L2 remixes** remain saturated / previously poisoned (P_super, sov+G26).\n\n",
        "## Staged NOSUBMIT\n\n",
    ]
    for s in staged:
        lines.append(f"- `{s['path']}` sha16=`{s['sha16']}` — {s.get('note','')}\n")
    md = OUT / "GM_V13_RESEARCH_BEYOND_095371.md"
    md.write_text("".join(lines))
    print(md, flush=True)
    print(json.dumps(report["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
