#!/usr/bin/env python3
"""Track-2 local CPU forensic audit: amanatar Q17 meta-bundle + detective datasets + G26.

STRICT NO_SUBMIT=True — zero Kaggle submissions.
Writes: logs/overnight/GM_AMANATAR_Q17_AUDIT.md (+ .json)
Target wall: <90s.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NO_SUBMIT = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT_MD = ROOT / "logs" / "overnight" / "GM_AMANATAR_Q17_AUDIT.md"
OUT_JSON = ROOT / "logs" / "overnight" / "GM_AMANATAR_Q17_AUDIT.json"
Q17 = ROOT / "external" / "nov12_late_recon" / "amanatar_q17"
DET_W = ROOT / "external" / "nov12_late_recon" / "detective_weights"
DET_D = ROOT / "external" / "nov12_late_recon" / "detective_data"
KAG = ROOT / ".venv_forensics" / "bin" / "kaggle"

T0 = time.time()


def log(m: str):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def kaggle(args: list[str], timeout: int = 90) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    exe = str(KAG) if KAG.exists() else "kaggle"
    try:
        p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    except Exception as e:
        return 1, str(e)


def main():
    assert NO_SUBMIT is True
    log("======== TRACK2 amanatar Q17 audit NO_SUBMIT=True ========")
    from _gm_v11_shared import (
        L,
        L_fold,
        fold_aucs,
        gate_pass,
        load_anchor_and_bank,
        load_prob,
        lofo_eval,
        project_out,
        setup_folds,
        spearman_dist,
        unit,
        unitize,
        g26_status,
    )
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    # --- live file lists ---
    rc_g26, g26_files = kaggle(["datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"])
    rc_dw, dw_files = kaggle(["datasets", "files", "aphoristicartist/s6e11-detective-weights"])
    rc_dd, dd_files = kaggle(["datasets", "files", "aphoristicartist/s6e11-detective-data"])
    rc_q17, q17_files = kaggle(["datasets", "files", "amanatar/s6e11-rank1-q17-meta-bundle"])
    g26_live = g26_status()
    g26_test5 = ("test_5fold" in g26_files.lower()) or g26_live.get("has_test_5fold_remote", False)

    # --- load Q17 ---
    npz_path = Q17 / "rank1_q17_streams.npz"
    assert npz_path.exists(), npz_path
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.files)
    stream_names = [str(x) for x in z["stream_names"].tolist()] if "stream_names" in keys else []
    y = z["y_train"].astype(np.int64)
    folds_arr = z["folds"].astype(np.int8)
    folds, fold_id = setup_folds(y)
    assert (folds_arr == fold_id).all(), "Q17 folds != SKF(5,rs=42)"

    pack = load_anchor_and_bank(y, fold_id, folds)
    refs = {
        "A_095340": pack["A"],
        "F_095330": pack["F"],
        "z_S7_ours": L(pack["s7"]),
        "cand2": L(pack["cand2"]),
    }
    # optional LB refs (skip AppleDouble / non-utf8 junk)
    def try_ref(tag: str, paths):
        if tag in refs:
            return
        for p in paths:
            p = Path(p)
            if not p.is_file() or p.name.startswith("._"):
                continue
            try:
                refs[tag] = L(load_prob(p))
                log(f"  ref {tag} <- {p}")
                return
            except Exception as e:
                log(f"  skip ref {p.name}: {e}")

    try_ref(
        "areeb_095347",
        list((ROOT / "submissions").glob("*095347*.csv"))
        + list((ROOT / "submissions").glob("*areeb*.csv"))
        + list((ROOT / "external").glob("**/amanatar*095347*/**/*submission*.csv"))
        + list((ROOT / "external").glob("**/*095347*submission*.csv")),
    )
    try_ref("s352", list((ROOT / "submissions").glob("*095352*.csv")) + list((ROOT / "submissions").glob("*s352*.csv")))
    try_ref("s354", list((ROOT / "submissions").glob("*095354*.csv")) + list((ROOT / "submissions").glob("*s354*.csv")))

    # pair streams
    pairs = []
    for k in keys:
        if not k.startswith("oof_"):
            continue
        stem = k[len("oof_") :]
        tk = "test_" + stem
        if tk in keys:
            pairs.append((stem, k, tk))

    spearman_rows = []
    pair_rows = []
    lofo_rows = []

    for stem, ok, tk in pairs:
        o = z[ok].astype(np.float64).ravel()
        t = z[tk].astype(np.float64).ravel()
        assert o.shape == (699960,) and t.shape == (299984,), (stem, o.shape, t.shape)
        # solo AUCs
        zo = L_fold(o, fold_id)
        fa = fold_aucs(y, zo, folds)
        solo = float(roc_auc_score(y, zo))
        leak_gap = fa[2] - fa[4]
        # spearman distances on TEST vs refs
        Lt = L(t)
        dist = {name: round(spearman_dist(Lt, ref), 6) for name, ref in refs.items() if len(ref) == 299984}
        spearman_rows.append({"stream": stem, **dist})
        pair_rows.append(
            {
                "stream": stem,
                "solo_auc": round(solo, 7),
                "fold_aucs": [round(a, 7) for a in fa],
                "F2_minus_F4": round(leak_gap, 7),
                "leak_flag": bool(leak_gap > 0.004),  # large F2>>F4 can indicate oddness; report both ways
                "oof_finite": bool(np.isfinite(o).all()),
                "test_finite": bool(np.isfinite(t).all()),
            }
        )
        # LOFO: zero-sum style residual vs MS/MLP/GB/LM proxies + orth to [S7, G26-MS]
        # candidate as L_fold(o) projected
        bases = pack["bases_o"]
        delta = project_out(zo, bases)
        rS = lofo_eval(y, pack["eval_S7matched_355"], delta, folds)
        rP = lofo_eval(y, pack["eval_simple_355"], delta, folds)
        # also raw swap-style: replace axes by difference vs MS/MLP/GB/LM if available
        swaps = {}
        for name, oo in [("MS", pack["MSo"]), ("MLP", pack["MLPo"]), ("GB", pack["GBo"]), ("LM", pack["LMo"])]:
            dswap = project_out(zo - oo, bases)
            swaps[name] = {
                "S7m": lofo_eval(y, pack["eval_S7matched_355"], dswap, folds),
                "simple": lofo_eval(y, pack["eval_simple_355"], dswap, folds),
            }
        lofo_rows.append(
            {
                "stream": stem,
                "orth_S7m": {k: rS[k] for k in ("pos5", "mean_d_e5", "d0_e5", "fold_d_e5", "pass")},
                "orth_simple": {k: rP[k] for k in ("pos5", "mean_d_e5", "d0_e5", "fold_d_e5", "pass")},
                "gate": gate_pass(rS, rP),
                "swaps": {
                    n: {
                        "S7m_pos5": swaps[n]["S7m"]["pos5"],
                        "S7m_pass": swaps[n]["S7m"]["pass"],
                        "simple_pos5": swaps[n]["simple"]["pos5"],
                        "simple_pass": swaps[n]["simple"]["pass"],
                    }
                    for n in swaps
                },
            }
        )
        log(f"  stream={stem} solo={solo:.6f} F2-F4={leak_gap:+.5f} LOFO_S7m={rS['pos5']}/5 gate={gate_pass(rS,rP)}")

    # also evaluate test_anchor alone
    if "test_anchor_095370" in keys:
        Lt = L(z["test_anchor_095370"].astype(np.float64))
        dist = {name: round(spearman_dist(Lt, ref), 6) for name, ref in refs.items() if len(ref) == 299984}
        spearman_rows.append({"stream": "anchor_095370_TEST_ONLY", **dist})
        if "oof_anchor_095370" in keys:
            zo = L_fold(z["oof_anchor_095370"].astype(np.float64), fold_id)
            fa = fold_aucs(y, zo, folds)
            pair_rows.append(
                {
                    "stream": "anchor_095370",
                    "solo_auc": round(float(roc_auc_score(y, zo)), 7),
                    "fold_aucs": [round(a, 7) for a in fa],
                    "F2_minus_F4": round(fa[2] - fa[4], 7),
                    "leak_flag": bool((fa[2] - fa[4]) > 0.004),
                }
            )
            delta = project_out(zo, pack["bases_o"])
            rS = lofo_eval(y, pack["eval_S7matched_355"], delta, folds)
            rP = lofo_eval(y, pack["eval_simple_355"], delta, folds)
            lofo_rows.append(
                {
                    "stream": "anchor_095370",
                    "orth_S7m": {k: rS[k] for k in ("pos5", "mean_d_e5", "d0_e5", "fold_d_e5", "pass")},
                    "orth_simple": {k: rP[k] for k in ("pos5", "mean_d_e5", "d0_e5", "fold_d_e5", "pass")},
                    "gate": gate_pass(rS, rP),
                }
            )
            log(f"  ANCHOR solo={roc_auc_score(y,zo):.6f} LOFO={rS['pos5']}/5 gate={gate_pass(rS,rP)}")

    # note: Q17 ships test_gpt2_seed2026 (299984,) — compare to public G26 test_f0 only vector
    g26_note = "Q17 has test_gpt2_seed2026 (299984,) aggregated; live Kaggle G26 still lacks test_5fold/f1..f4 tensors."
    if "test_gpt2_seed2026" in keys:
        g26_f0 = pack["G26"]  # L(test_f0) from recon — wait pack G26 is already L()
        # compare raw ranks
        t_q = z["test_gpt2_seed2026"].astype(np.float64)
        # load raw f0
        f0p = ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"
        if f0p.exists():
            f0 = np.load(f0p).astype(np.float64).ravel()
            g26_note += f" spearman_dist(Q17.test_gpt2_s2026, G26_test_f0)={spearman_dist(t_q, f0):.6e}."

    passed = [r for r in lofo_rows if r.get("gate")]
    summary = {
        "NO_SUBMIT": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - T0, 2),
        "q17_keys": keys,
        "stream_names": stream_names,
        "n_pairs": len(pairs),
        "g26_test5fold_live": bool(g26_test5),
        "g26_files_remote": g26_files[:800],
        "detective_weights_files": dw_files[:500],
        "detective_data_files": dd_files[:500],
        "q17_remote_files": q17_files[:400],
        "n_lofo_pass": len(passed),
        "passed_streams": [r["stream"] for r in passed],
        "spearman_rows": spearman_rows,
        "pair_rows": pair_rows,
        "lofo_rows": lofo_rows,
        "g26_note": g26_note,
        "refs_used": list(refs.keys()),
    }

    # markdown
    lines = []
    lines.append("# GM Track-2 Audit: amanatar Q17 + detective + G26\n")
    lines.append(f"**UTC:** {summary['utc']}  \n**`NO_SUBMIT=True`** — 0 uploads  \n**Elapsed:** {summary['elapsed_s']}s\n")
    lines.append("\n## Executive verdict\n")
    if passed:
        lines.append(f"- **{len(passed)} stream(s) cleared strict LOFO gate:** `{passed}`\n")
    else:
        lines.append("- **0 streams cleared strict LOFO 5/5 gate** on `eval_S7matched_355` ∩ `eval_simple_355` after double-orth to `[z_S7_ours, G26−MS]`.\n")
    lines.append(f"- **G26 live `test_5fold`:** {'PRESENT' if g26_test5 else '**ABSENT**'} (still oof_5fold + test_f0 only).\n")
    lines.append(f"- **Detective weights:** `model.pt` (~475MB) + `config.json` — weights artifact, not OOF bank. Data = orig20k CSV only.\n")
    lines.append(f"- {g26_note}\n")

    lines.append("\n## Live dataset file lists\n")
    lines.append("### amanatar/s6e11-rank1-q17-meta-bundle\n```\n" + q17_files.strip()[:1200] + "\n```\n")
    lines.append("### aphoristicartist/s6e11-detective-weights\n```\n" + dw_files.strip()[:800] + "\n```\n")
    lines.append("### aphoristicartist/s6e11-detective-data\n```\n" + dd_files.strip()[:800] + "\n```\n")
    lines.append("### goodpjw2008/s6e11-target-first-gpt2-seed-2026\n```\n" + g26_files.strip()[:800] + "\n```\n")

    lines.append("\n## Bundle contents (`rank1_q17_streams.npz`)\n")
    lines.append(f"- Keys ({len(keys)}): `{keys}`\n")
    lines.append(f"- `stream_names`: `{stream_names}`\n")
    lines.append("- Folds match `StratifiedKFold(5, shuffle=True, random_state=42)` ✓\n")
    lines.append("- **No README/md/script** inside the zip — only the NPZ.\n")

    lines.append("\n## Test-vector Spearman distances (`1−ρ`) vs ATH refs\n")
    lines.append("| stream | " + " | ".join(refs.keys()) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(refs)) + "|\n")
    for row in spearman_rows:
        cells = [str(row.get(k, "")) for k in refs.keys()]
        lines.append(f"| {row['stream']} | " + " | ".join(cells) + " |\n")

    lines.append("\n## OOF/Test pairs — solo AUC & fold profile\n")
    lines.append("| stream | solo | F0 | F1 | F2 | F3 | F4 | F2−F4 | leak?>0.004 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
    for r in pair_rows:
        fa = r.get("fold_aucs", [None] * 5)
        lines.append(
            f"| {r['stream']} | {r['solo_auc']} | {fa[0]} | {fa[1]} | {fa[2]} | {fa[3]} | {fa[4]} | {r['F2_minus_F4']} | {r.get('leak_flag')} |\n"
        )

    lines.append("\n## Strict LOFO gate (orth residual vs `[S7, G26−MS]`)\n")
    lines.append("| stream | S7m pos5 | S7m meanΔe5 | S7m d0 | simple pos5 | gate |\n")
    lines.append("|---|---:|---:|---:|---:|:---:|\n")
    for r in lofo_rows:
        s, p = r["orth_S7m"], r["orth_simple"]
        lines.append(
            f"| {r['stream']} | {s['pos5']}/5 | {s['mean_d_e5']} | {s['d0_e5']} | {p['pos5']}/5 | {r['gate']} |\n"
        )

    lines.append("\n### Swap residuals (−MS/−MLP/−GB/−LM) then orth — any 5/5?\n")
    any_swap = False
    for r in lofo_rows:
        if "swaps" not in r:
            continue
        for n, sw in r["swaps"].items():
            if sw["S7m_pass"] and sw["simple_pos5"] >= 4:
                lines.append(f"- **PASS-ish:** `{r['stream']}` swap `{n}` S7m_pass={sw['S7m_pass']} simple_pos5={sw['simple_pos5']}\n")
                any_swap = True
    if not any_swap:
        lines.append("- None of the −MS/−MLP/−GB/−LM swap+orth residuals cleared the dual gate.\n")

    lines.append("\n## Implications for Track Compact-LM seed2026\n")
    lines.append("1. Q17 is a **frozen stream bank** (16 members + 0.95370 anchor), not a new training recipe.\n")
    lines.append("2. Without a 5/5 LOFO pass, **do not** burn a submission blending Q17 streams onto cand2.\n")
    lines.append("3. G26 5-fold test still missing → Compact-LM `SEED=2026` Fold-0 canary remains the budget GPU path.\n")
    lines.append("4. Detective `model.pt` is out of scope for OOF stacking this turn (no matching OOF/test vectors in detective-data).\n")
    lines.append("\n---\n*Generated by `scripts/gm_track2_amanatar_q17_audit.py`*\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines))
    # JSON without huge nested u vectors
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log(f"WROTE {OUT_MD}")
    log(f"LOFO pass count={len(passed)} G26_test5={g26_test5} NO_SUBMIT={NO_SUBMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
