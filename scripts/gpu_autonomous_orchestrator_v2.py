#!/usr/bin/env python3
"""
Autonomous GPU orchestrator v2 — runs Rungs 1-8 on dedicated pod s6e11-0953-campaign.
STRICT NO_SUBMIT until Stage 9 gates; self-healing; 5-min state updates; spend cap $20.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("INPUT_DIR", str(ROOT))

from gpu_matrix_v2_core import (  # noqa: E402
    BLACKLIST_POD,
    append_watchdog,
    assert_pod_allowed,
    load_state,
    save_state,
)
from gpu_matrix_v2_runner import Ctx, LEVER_GENERATORS, evaluate_lever_5fold, evaluate_lever_fold0  # noqa: E402

RUNG_ORDER = [
    # Priority: rescue CPU Fold-0 winners first (highest EV), then P3 transformers
    ("2", ["P4_L4.3", "P4_L4.1", "P4_L4.4", "P4_L4.2", "P4_L4.5"]),
    ("7", ["P1_L1.3", "P1_L1.4"]),
    ("5", ["P5_L5.1"]),
    ("6", ["P6_L6.2"]),
    ("4", ["P2_L2.3"]),
    ("3", ["P7_L7.4"]),
    ("1", ["P3_L3.3", "P3_L3.4", "P3_L3.5", "P3_L3.1", "P3_L3.2"]),
]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def estimate_spend(state: dict, cost_hr: float) -> float:
    started = state.get("started_utc")
    if not started:
        return float(state.get("spend_usd") or 0.0)
    try:
        t0 = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - t0).total_seconds() / 3600.0
        return round(hours * cost_hr, 4)
    except Exception:
        return float(state.get("spend_usd") or 0.0)


def update_progress(state: dict, status: str, cost_hr: float) -> None:
    state["last_progress_utc"] = utc()
    state["status_line"] = status
    state["spend_usd"] = estimate_spend(state, cost_hr)
    # ETA from remaining rungs
    etas = state.get("rung_etas_min") or {}
    cur = int(state.get("current_rung") or 0)
    remaining = 0.0
    for r, _ in RUNG_ORDER:
        if int(r) > cur:
            remaining += float(etas.get(r, 35))
        elif int(r) == cur:
            done = int(state.get("levers_done_in_rung") or 0)
            total = int(state.get("levers_total_in_rung") or 5)
            frac = max(0, total - done) / max(total, 1)
            remaining += float(etas.get(r, 35)) * frac
    remaining += float(etas.get("9", 20))
    state["eta_minutes"] = round(remaining, 1)
    save_state(state)
    # also write chat-friendly snapshot
    snap = {
        "utc": state["last_progress_utc"],
        "pod_id": state.get("gpu_pod_id"),
        "pod_name": state.get("gpu_pod_name"),
        "rung": state.get("current_rung"),
        "lever": state.get("current_lever"),
        "levers_done": state.get("levers_done_in_rung"),
        "levers_total": state.get("levers_total_in_rung"),
        "completed": state.get("completed_levers"),
        "promoted": state.get("promoted_levers"),
        "spend_usd": state.get("spend_usd"),
        "eta_minutes": state.get("eta_minutes"),
        "status": status,
        "best_lb": state.get("best_lb"),
        "champ_oof": state.get("v_champ_oof_auc"),
        "gap_to_953": round(0.953000 - float(state.get("best_lb") or 0.95243), 6),
    }
    out = ROOT / "logs/overnight/progress_snapshot_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2))
    line = (
        f"[{snap['utc']}] Rung {snap['rung']} lever={snap['lever']} "
        f"done={snap['levers_done']}/{snap['levers_total']} "
        f"promoted={len(snap['promoted'] or [])} spend=${snap['spend_usd']} "
        f"ETA={snap['eta_minutes']}m | {status}"
    )
    print(line, flush=True)
    with open(ROOT / "logs/overnight/watchdog_5min.jsonl", "a") as fh:
        fh.write(json.dumps({"utc": snap["utc"], "event": "progress5m", **snap}) + "\n")


def run() -> None:
    pod_id = os.environ.get("POD_ID", "")
    cost_hr = float(os.environ.get("COST_USD_HR", "0.59"))
    spend_cap = float(os.environ.get("SPEND_CAP_USD", "20"))
    assert_pod_allowed(pod_id)
    if pod_id == BLACKLIST_POD:
        raise SystemExit(2)

    state = load_state()
    state["gpu_pod_id"] = pod_id
    state["gpu_pod_name"] = os.environ.get("POD_NAME", "s6e11-0953-campaign")
    if not state.get("started_utc"):
        state["started_utc"] = utc()
    save_state(state)
    update_progress(state, "loading_ctx", cost_hr)

    ctx = Ctx()
    update_progress(state, "ctx_ready_starting_rungs", cost_hr)

    completed = set(state.get("completed_levers") or [])
    promoted = list(state.get("promoted_levers") or [])

    for rung_id, levers in RUNG_ORDER:
        if estimate_spend(state, cost_hr) >= spend_cap - 0.5:
            update_progress(state, "SPEND_CAP_STOP", cost_hr)
            break
        state["current_rung"] = int(rung_id)
        state["levers_total_in_rung"] = len(levers)
        state["levers_done_in_rung"] = 0
        strikes = int((state.get("strikes_by_paradigm") or {}).get(f"P{rung_id}", 0))
        for li, key in enumerate(levers):
            if key in completed:
                state["levers_done_in_rung"] = li + 1
                continue
            if key not in LEVER_GENERATORS:
                append_watchdog("skip_no_gen", {"key": key})
                completed.add(key)
                continue
            state["current_lever"] = key
            update_progress(state, f"running_fold0_{key}", cost_hr)
            paradigm, lever = key.split("_", 1)
            try:
                meta0 = evaluate_lever_fold0(ctx, paradigm, lever)
            except Exception as e:
                append_watchdog("fold0_exception", {"key": key, "err": str(e), "tb": traceback.format_exc()[-1500:]})
                strikes += 1
                state.setdefault("strikes_by_paradigm", {})[f"R{rung_id}"] = strikes
                completed.add(key)
                state["completed_levers"] = sorted(completed)
                state["levers_done_in_rung"] = li + 1
                update_progress(state, f"fold0_exc_{key}_{type(e).__name__}:{str(e)[:80]}", cost_hr)
                if strikes >= 5:
                    update_progress(state, f"strikeout_rung_{rung_id}", cost_hr)
                    break
                continue

            if not meta0.get("passed_fold0"):
                strikes += 1
                state.setdefault("strikes_by_paradigm", {})[f"R{rung_id}"] = strikes
                completed.add(key)
                state["completed_levers"] = sorted(completed)
                state["levers_done_in_rung"] = li + 1
                update_progress(state, f"fold0_fail_{key}_d={meta0.get('delta_auc')}", cost_hr)
                if strikes >= 5:
                    break
                continue

            update_progress(state, f"running_5fold_{key}", cost_hr)
            try:
                meta5 = evaluate_lever_5fold(ctx, paradigm, lever)
            except Exception as e:
                append_watchdog("fivefold_exception", {"key": key, "err": str(e)})
                completed.add(key)
                state["completed_levers"] = sorted(completed)
                state["levers_done_in_rung"] = li + 1
                continue

            completed.add(key)
            state["completed_levers"] = sorted(completed)
            if meta5.get("promoted"):
                promoted.append({"key": key, **{k: meta5[k] for k in ("delta_oof", "new_auc", "delta_rho") if k in meta5}})
                state["promoted_levers"] = promoted
            state["levers_done_in_rung"] = li + 1
            update_progress(state, f"done_{key}_promoted={meta5.get('promoted')}", cost_hr)

            # local backup hint file
            (ROOT / "logs/overnight/backup_request.flag").write_text(utc())

        else:
            continue
        # strikeout break inner
        if strikes >= 5:
            continue

    state["current_rung"] = 9
    update_progress(state, f"phase_complete_promoted={len(promoted)}_NO_SUBMIT", cost_hr)
    append_watchdog("orchestrator_complete_phase", {"promoted": promoted, "spend": state.get("spend_usd")})


if __name__ == "__main__":
    # Single-instance lock — prevent watchdog double-start races
    lock_path = ROOT / "logs/overnight/orchestrator_v2.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(lock_path, "w")
    try:
        import fcntl

        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another orchestrator holds the lock; exiting", flush=True)
        raise SystemExit(0)
    lock_fh.write(str(os.getpid()))
    lock_fh.flush()
    try:
        run()
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fh.close()
