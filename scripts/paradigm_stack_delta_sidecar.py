#!/usr/bin/env python3
"""
Non-destructive sidecar: wait for campaign C/dossier artifacts, then re-score
A/B/C with Test-1 stack Δ and rewrite dossier. Never kills the GPU worker.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
STATE = ROOT / "logs/overnight/3paradigm_discovery_state.json"
DOSSIER = ROOT / "logs/overnight/PARADIGM_DECISION_DOSSIER.json"
C_METRICS = ROOT / "preds/paradigm_C/metrics.json"
MARKER = ROOT / "logs/overnight/stack_delta_sidecar_done.json"


def stage() -> str:
    if not STATE.exists():
        return ""
    return json.loads(STATE.read_text()).get("stage", "")


def campaign_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "paradigm_abc_cuda_campaign.py"], capture_output=True)
    return r.returncode == 0


def main():
    print(f"[sidecar] start stage={stage()} alive={campaign_alive()}", flush=True)
    # Wait until C metrics exist AND campaign has exited (or dossier written by worker)
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        st = stage()
        alive = campaign_alive()
        print(f"[sidecar] poll stage={st} alive={alive} c_metrics={C_METRICS.exists()} dossier={DOSSIER.exists()}", flush=True)
        if C_METRICS.exists() and (not alive or st in ("C_completed", "dossier", "DONE", "done")):
            # Give worker a moment to finish writing dossier if mid-write
            if alive and st not in ("dossier", "DONE", "done", "C_completed"):
                time.sleep(15)
                continue
            time.sleep(3)
            break
        time.sleep(30)
    else:
        print("[sidecar] timeout waiting for C", flush=True)
        return 1

    env = os.environ.copy()
    env["S6E11_ROOT"] = str(ROOT)
    env["NO_SUBMIT"] = "1"
    # Re-score all paradigms with corrected evaluator
    for P in ("A", "B", "C"):
        cmd = [sys.executable, "-u", str(ROOT / "scripts/paradigm_stack_delta_eval.py"),
               "--paradigm", P, "--patch-metrics"]
        print(f"[sidecar] running {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)

    # Rebuild dossier tables from corrected metrics (lightweight, no GPU)
    from paradigm_stack_delta_eval import load_bank_and_y, score_candidate_vs_v4p_hc, _jsonable
    import numpy as np
    from sklearn.metrics import roc_auc_score

    bank, y, folds = load_bank_and_y()
    v4p = bank["v4p_hc_oof"].astype(np.float64)
    table1 = []
    preds = {}
    for P, fo in [("A", "xgb_oof.npy"), ("B", "xgb_llr_oof.npy"), ("C", "stealth_oof.npy")]:
        p = ROOT / f"preds/paradigm_{P}" / fo
        if not p.exists():
            # C may only have tabm
            alt = ROOT / f"preds/paradigm_{P}" / ("tabm_k32_oof.npy" if P == "C" else fo)
            if alt.exists():
                p = alt
            else:
                continue
        res = score_candidate_vs_v4p_hc(y, v4p, np.load(p), folds)
        oof = res.pop("stack_oof_scores")
        preds[P] = oof
        np.save(ROOT / f"preds/paradigm_{P}/stack_oof_corrected.npy", oof)
        meta = {}
        mp = ROOT / f"preds/paradigm_{P}/metrics.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        table1.append({
            "paradigm": P,
            "cand_file": p.name,
            "cand_solo_auc": res["cand_solo_auc"],
            "stack_oof_auc": res["stack_oof_auc"],
            "delta_auc": res["delta_auc"],
            "stack_mean_alpha": res["stack_mean_alpha"],
            "fold_win_rate": res["fold_win_rate"],
            "gpu_minutes": meta.get("gpu_minutes"),
            "legacy_bogus_delta_auc": meta.get("legacy_bogus_delta_auc"),
            "stack_method": res["stack_method"],
        })

    # For C, also try tabm and keep better
    c_tabm = ROOT / "preds/paradigm_C/tabm_k32_oof.npy"
    if c_tabm.exists() and "C" in preds:
        res_t = score_candidate_vs_v4p_hc(y, v4p, np.load(c_tabm), folds)
        oof_t = res_t.pop("stack_oof_scores")
        c_row = next(r for r in table1 if r["paradigm"] == "C")
        if res_t["delta_auc"] > c_row["delta_auc"]:
            preds["C"] = oof_t
            np.save(ROOT / "preds/paradigm_C/stack_oof_corrected.npy", oof_t)
            c_row.update({
                "cand_file": "tabm_k32_oof.npy",
                "cand_solo_auc": res_t["cand_solo_auc"],
                "stack_oof_auc": res_t["stack_oof_auc"],
                "delta_auc": res_t["delta_auc"],
                "stack_mean_alpha": res_t["stack_mean_alpha"],
                "fold_win_rate": res_t["fold_win_rate"],
            })

    dossier = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_oof": float(0.9523071),
        "no_submit": True,
        "stack_eval": "test1_orthogonal_alpha_probe_vs_v4p_hc",
        "table1_head_to_head": table1,
        "ranked": sorted(table1, key=lambda r: r["delta_auc"], reverse=True),
        "note": "Rebuilt by stack_delta_sidecar after live C worker (old in-memory stack) finished.",
    }
    # Preserve any prior dossier extras if present
    if DOSSIER.exists():
        try:
            old = json.loads(DOSSIER.read_text())
            for k in ("table2_orthogonality", "table3_combinations_shapley", "table4_error_strata", "table5_decision"):
                if k in old:
                    dossier[k] = old[k]
        except Exception:
            pass
    DOSSIER.write_text(json.dumps(_jsonable(dossier), indent=2))

    # Patch state live_metrics / paradigms deltas
    if STATE.exists():
        st = json.loads(STATE.read_text())
        for row in table1:
            P = row["paradigm"]
            st.setdefault("paradigms", {}).setdefault(P, {}).setdefault("metrics", {})
            st["paradigms"][P]["metrics"]["delta_auc"] = row["delta_auc"]
            st["paradigms"][P]["metrics"]["stack_oof_auc"] = row["stack_oof_auc"]
            st["paradigms"][P]["metrics"]["stack_method"] = row["stack_method"]
        st["stage"] = "DONE_STACK_FIXED"
        st["status_line"] = "dossier rebuilt with Test-1 stack Δ"
        st["sidecar_fixed_utc"] = dossier["utc"]
        STATE.write_text(json.dumps(st, indent=2))

    MARKER.write_text(json.dumps(dossier, indent=2))
    print(f"[sidecar] DONE wrote {DOSSIER}", flush=True)
    print(json.dumps(table1, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
