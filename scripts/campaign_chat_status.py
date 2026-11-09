#!/usr/bin/env python3
"""Pull campaign status for 5-min chat ticks. Prints a compact multi-line report."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "logs/overnight/campaign_pod.env"


def load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV.exists():
        return out
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def ssh_cmd(env: dict[str, str], remote: str) -> str:
    key = env.get("SSH_KEY", str(Path.home() / ".ssh/casmi_runpod"))
    host = env.get("SSH_DIRECT_HOST", "194.68.245.67")
    port = env.get("SSH_DIRECT_PORT", "22068")
    user = env.get("SSH_USER", "root")
    cmd = [
        "ssh",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(port),
        f"{user}@{host}",
        remote,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        return (r.stdout or "") + (("\n" + r.stderr) if r.returncode else "")
    except Exception as e:
        return f"SSH_FAIL: {e}"


REMOTE = r'''
python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
s = json.load(open("/workspace/s6e11/logs/overnight/matrix_v2_state.json"))
start = datetime.strptime(s["started_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
hrs = (now - start).total_seconds() / 3600.0
cost = float((__import__("os").environ.get("COST_USD_HR") or "0.59"))
print("UTC", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
print("STATUS", s.get("status_line"))
print("RUNG", s.get("current_rung"), "LEVER", s.get("current_lever"))
print("DONE", s.get("levers_done_in_rung"), "/", s.get("levers_total_in_rung"))
print("COMPLETED", ",".join(s.get("completed_levers") or []) or "-")
prom = s.get("promoted_levers") or []
print("PROMOTED_N", len(prom))
print("PROMOTED", json.dumps(prom)[:400])
print("SPEND", round(hrs * 0.59, 4))
print("ETA_MIN", s.get("eta_minutes"))
print("BEST_LB", s.get("best_lb"))
print("GAP", round(0.953 - float(s.get("best_lb") or 0.95243), 6))
wp = Path("/workspace/s6e11/logs/overnight/watchdog_5min.jsonl")
pass_n = fail_n = exc_n = 0
passes, fails = [], []
if wp.exists():
    for line in wp.read_text().splitlines()[-300:]:
        try:
            o = json.loads(line)
        except Exception:
            continue
        ev = o.get("event")
        if ev == "fold0_done":
            if o.get("passed_fold0"):
                pass_n += 1
                passes.append("%s d=%.2e path=%s" % (o.get("key"), float(o.get("delta_auc") or 0), o.get("best_path")))
            else:
                fail_n += 1
                fails.append("%s d=%s" % (o.get("key"), o.get("delta_auc")))
        elif ev == "fold0_exception":
            exc_n += 1
            fails.append("%s EXC:%s" % (o.get("key"), str(o.get("err") or "")[:70]))
print("PASS_N", pass_n, "FAIL_N", fail_n, "EXC_N", exc_n)
print("RECENT_PASS", " | ".join(passes[-5:]) or "-")
print("RECENT_FAIL", " | ".join(fails[-6:]) or "-")
PY
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
(pgrep -af gpu_autonomous_orchestrator_v2 || echo ORCH_DEAD)
tail -5 /workspace/s6e11/logs/overnight/orchestrator_v2.log
'''


def main() -> int:
    env = load_env()
    raw = ssh_cmd(env, REMOTE)
    print(raw.rstrip())
    # also write local snapshot for chat ticks
    out = ROOT / "logs/overnight/chat_tick_latest.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
