#!/usr/bin/env python3
"""Part 11 orchestrator — 4 sequential research labs. Policy: NO_SUBMIT."""
from __future__ import annotations

import time
import traceback

from part11_common import BASELINE_STACK, POLICY, T0, load_report, log, save_report


def main():
    rep = load_report()
    rep["policy"] = POLICY
    rep["baseline_stack"] = BASELINE_STACK
    rep["target_nested"] = 0.95235
    rep["target_lb"] = 0.9525
    save_report(rep)

    labs = [
        ("lab1", "part11_lab1_idf_graph", "run_lab1"),
        ("lab2", "part11_lab2_interactions", "run_lab2"),
        ("lab3", "part11_lab3_transductive", "run_lab3"),
        ("lab4", "part11_lab4_nade", "run_lab4"),
    ]
    for key, modname, fn in labs:
        if rep.get(key, {}).get("runtime_s"):
            log(f"SKIP {key} (already complete)")
            continue
        log(f"START {key}")
        try:
            mod = __import__(modname)
            getattr(mod, fn)(rep)
            rep = load_report()
        except Exception as e:
            log(f"FAIL {key}: {e}")
            traceback.print_exc()
            rep = load_report()
            rep[key] = {"error": str(e), "traceback": traceback.format_exc()}
            save_report(rep)
            # continue to next lab
            continue

    # Summary
    rep = load_report()
    keeps = []
    for k in ("lab1", "lab2", "lab3", "lab4"):
        lab = rep.get(k) or {}
        if lab.get("keep_any"):
            keeps.append(k)
    rep["summary"] = {
        "labs_kept": keeps,
        "any_keep": bool(keeps),
        "runtime_s": time.time() - T0,
        "policy": POLICY,
        "note": "NO_SUBMIT — net nested CV lift gates only",
    }
    save_report(rep)
    log("=" * 72)
    log(f"PART11 DONE keeps={keeps} runtime={rep['summary']['runtime_s']:.0f}s")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
