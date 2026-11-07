#!/usr/bin/env python3
"""Emit a 5-minute research progress snapshot (chat-friendly + JSONL append)."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT / "logs/overnight/watchdog_5min.jsonl"
STATE = ROOT / "logs/overnight/matrix_8x5_state.json"
WAVE2 = ROOT / "logs/overnight/nov07_wave2.json"
LADDER = ROOT / "logs/overnight/nov07_submit_ladder.json"
SNAP = ROOT / "logs/overnight/progress_snapshot_latest.json"


def mem() -> dict:
    page = 16384
    avail = -1.0
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        d: dict[str, int] = {}
        for line in out.splitlines():
            m = re.match(r"([^:]+):\s+(\d+)", line)
            if m:
                d[m.group(1).strip()] = int(m.group(2))
            if "page size of" in line:
                mm = re.search(r"page size of (\d+)", line)
                if mm:
                    page = int(mm.group(1))
        avail = (
            d.get("Pages free", 0)
            + d.get("Pages inactive", 0)
            + d.get("Pages speculative", 0)
        ) * page / 1e6
    except Exception:
        pass
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = (-1.0, -1.0, -1.0)
    return {
        "avail_mb": round(avail),
        "load1": round(load[0], 2),
        "load5": round(load[1], 2),
        "cpus": os.cpu_count() or 1,
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def active_python_jobs() -> list[str]:
    """Best-effort: pgrep may be TCC-blocked; return empty on failure."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-fl", "nov07_|run_local_matrix|research_progress|wave2"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return lines[:8]
    except Exception:
        return []


def main() -> None:
    now = datetime.now(timezone.utc)
    next_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    eta_reset_h = (next_reset - now).total_seconds() / 3600.0

    st = load_json(STATE)
    w2 = load_json(WAVE2)
    ladder = load_json(LADDER)
    m = mem()
    jobs = active_python_jobs()

    best_lb = max(
        float(w2.get("best_lb") or 0),
        float(ladder.get("best_lb") or 0),
        0.95243,
    )
    champ_oof = float(st.get("v_champ_oof_auc") or 0.9523071)
    completed = st.get("completed_levers") or []
    promoted = st.get("promoted_levers") or []
    strikes = st.get("strikes_by_paradigm") or {}
    gap_to_953 = 0.953000 - best_lb

    # crude slot estimate: midnight UTC reset
    slots_left_est = 0 if now.date() == datetime(2026, 11, 7, tzinfo=timezone.utc).date() else 10
    # Prefer reading last submit ladder length for today
    today_subs = len(ladder.get("ladder") or []) + len(w2.get("ladder") or [])
    if today_subs >= 10:
        slots_left_est = 0

    phase = "IDLE_PLATEAU"
    next_action = "Wait for UTC reset (~10 new slots) OR start Paradigm-3 GPU / new orthogonal OOF head (logit remix exhausted)."
    if jobs:
        phase = "ACTIVE_LOCAL_JOB"
        next_action = f"Monitor active job(s): {jobs[0][:80]}"
    if st.get("gpu_pod_id"):
        phase = "GPU_POD_ACTIVE"
        next_action = f"Monitor GPU pod {st.get('gpu_pod_id')} for Paradigm 3"

    snap = {
        "utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": phase,
        "best_public_lb": best_lb,
        "champ_oof": round(champ_oof, 7),
        "gap_to_0953": round(gap_to_953, 6),
        "slots_used_today_est": today_subs,
        "slots_left_est": slots_left_est,
        "eta_kaggle_reset_hours": round(eta_reset_h, 2),
        "eta_kaggle_reset_utc": next_reset.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matrix_levers_done": len(completed),
        "matrix_promoted": len(promoted),
        "strikes": strikes,
        "p3_pending": True if not st.get("gpu_pod_id") else False,
        "mps_last_oof": w2.get("mps_oof"),
        "mps_last_delta": w2.get("mps_delta"),
        "mem": m,
        "active_jobs": jobs,
        "next_action": next_action,
    }
    SNAP.write_text(json.dumps(snap, indent=2))
    WATCH.parent.mkdir(parents=True, exist_ok=True)
    with WATCH.open("a") as f:
        f.write(json.dumps(snap) + "\n")

    # Chat-friendly block
    print("=== RESEARCH PROGRESS SNAPSHOT ===")
    print(f"UTC:              {snap['utc']}")
    print(f"Phase:            {phase}")
    print(f"Best Public LB:   {best_lb:.5f}   (gap to 0.953000 = {gap_to_953:+.5f})")
    print(f"Champ Nested OOF: {champ_oof:.7f}")
    print(f"Slots today:      used≈{today_subs}  left≈{slots_left_est}")
    print(f"ETA Kaggle reset: {eta_reset_h:.2f}h  ({snap['eta_kaggle_reset_utc']})")
    print(f"Matrix 8x5:       {len(completed)}/40 levers done, promoted={len(promoted)}, P3 GPU pending={snap['p3_pending']}")
    print(f"Host:             avail_mb={m['avail_mb']}  load1={m['load1']}  cpus={m['cpus']}")
    if w2.get("mps_oof") is not None:
        print(f"Last MPS RankNet: OOF={w2['mps_oof']:.7f}  Δ={w2['mps_delta']:+.2e}")
    print(f"Active jobs:      {jobs if jobs else 'none'}")
    print(f"Next:             {next_action}")
    print("=== END SNAPSHOT ===")


if __name__ == "__main__":
    main()
