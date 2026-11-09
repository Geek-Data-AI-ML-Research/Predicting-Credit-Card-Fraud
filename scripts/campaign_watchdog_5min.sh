#!/usr/bin/env bash
# Every 5 minutes: pull status, backup, print ETA, restart orchestrator if dead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/logs/overnight/campaign_pod.env"
SSH="ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p ${SSH_DIRECT_PORT} ${SSH_USER}@${SSH_DIRECT_HOST}"
INTERVAL="${INTERVAL_SEC:-300}"

echo "[watchdog] armed interval=${INTERVAL}s pod=$POD_ID"
while true; do
  STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  # Pull state
  rsync -az --no-owner --timeout=90 -e "ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${SSH_DIRECT_PORT}" \
    "${SSH_USER}@${SSH_DIRECT_HOST}:/workspace/s6e11/logs/overnight/" "$ROOT/logs/overnight/" 2>/dev/null || true
  rsync -az --no-owner --timeout=120 -e "ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${SSH_DIRECT_PORT}" \
    "${SSH_USER}@${SSH_DIRECT_HOST}:/workspace/s6e11/preds/matrix_v2/" "$ROOT/preds/matrix_v2/" 2>/dev/null || true

  START_EPOCH=$(python3 -c "from datetime import datetime,timezone; print(int(datetime.strptime('$STARTED_UTC','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()))")
  NOW_EPOCH=$(date -u +%s)
  SPEND=$(python3 -c "print(round(($NOW_EPOCH-$START_EPOCH)/3600.0*$COST_USD_HR, 4))")

  SNAP="$ROOT/logs/overnight/progress_snapshot_latest.json"
  if [[ -f "$SNAP" ]]; then
    python3 - <<PY
import json
from pathlib import Path
s=json.loads(Path("$SNAP").read_text())
print(f"[{s.get('utc','?')}] CAMPAIGN {s.get('pod_name')} Rung {s.get('rung')} lever={s.get('lever')} "
      f"{s.get('levers_done')}/{s.get('levers_total')} promoted={len(s.get('promoted') or [])} "
      f"spend=\${s.get('spend_usd')} (local=\$$SPEND) ETA={s.get('eta_minutes')}m | {s.get('status')}")
PY
  else
    echo "[$STAMP] CAMPAIGN pod=$POD_ID spend=\$$SPEND waiting for snapshot"
  fi

  # Health + auto-heal
  REMOTE=$($SSH 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; pgrep -af gpu_autonomous_orchestrator_v2 || echo ORCH_DEAD' 2>/dev/null || echo SSH_FAIL)
  echo "[$STAMP] remote: $REMOTE" | tr '\n' ' '; echo

  # Only heal when truly dead (not loading_ctx). Lock file also blocks double-start.
  if echo "$REMOTE" | grep -q ORCH_DEAD; then
    LIVE=$($SSH 'pgrep -f gpu_autonomous_orchestrator_v2.py || true' 2>/dev/null || echo "")
    if [[ -n "${LIVE// /}" ]]; then
      echo "[$STAMP] HEAL skipped — process still alive: $LIVE"
    else
      echo "[$STAMP] HEAL: restarting orchestrator on pod"
      $SSH 'bash -s' <<'REMOTE'
set -euo pipefail
source /workspace/venv/bin/activate
export S6E11_ROOT=/workspace/s6e11 INPUT_DIR=/workspace/s6e11
export POD_ID=ht590jc0rpb9kz POD_NAME=s6e11-0953-campaign
export COST_USD_HR=0.59 SPEND_CAP_USD=20 PYTHONUNBUFFERED=1
export BLACKLIST_POD=lheroidmuab1j4 NO_SUBMIT=1
export DUAL_TREE_ROUNDS_A=160 DUAL_TREE_ROUNDS_B=100
cd /workspace/s6e11
# flock inside orch exits 0 if already held
nohup python -u scripts/gpu_autonomous_orchestrator_v2.py >> logs/overnight/orchestrator_v2.log 2>&1 &
echo restarted_pid=$!
REMOTE
    fi
  fi

  bash "$ROOT/scripts/campaign_backup_sync.sh" || true
  sleep "$INTERVAL"
done
