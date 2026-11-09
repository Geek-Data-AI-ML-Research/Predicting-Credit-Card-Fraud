#!/usr/bin/env bash
# Local 5-minute progress + backup loop (prints chat-friendly status)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/logs/overnight/campaign_pod.env"
SSH="ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p ${SSH_DIRECT_PORT} ${SSH_USER}@${SSH_DIRECT_HOST}"
INTERVAL="${INTERVAL_SEC:-300}"

echo "[progress_loop] start interval=${INTERVAL}s pod=$POD_ID"
while true; do
  STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  # Pull latest state
  rsync -az --no-owner --timeout=60 -e "ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${SSH_DIRECT_PORT}" \
    "${SSH_USER}@${SSH_DIRECT_HOST}:/workspace/s6e11/logs/overnight/" "$ROOT/logs/overnight/" 2>/dev/null || true
  rsync -az --no-owner --timeout=120 -e "ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${SSH_DIRECT_PORT}" \
    "${SSH_USER}@${SSH_DIRECT_HOST}:/workspace/s6e11/preds/matrix_v2/" "$ROOT/preds/matrix_v2/" 2>/dev/null || true

  # Spend estimate
  START_EPOCH=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$STARTED_UTC" +%s 2>/dev/null || python3 -c "from datetime import datetime,timezone; print(int(datetime.strptime('$STARTED_UTC','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()))")
  NOW_EPOCH=$(date -u +%s)
  SPEND=$(python3 -c "print(round(($NOW_EPOCH-$START_EPOCH)/3600.0*$COST_USD_HR, 4))")

  SNAP="$ROOT/logs/overnight/progress_snapshot_latest.json"
  if [[ -f "$SNAP" ]]; then
    python3 - <<PY
import json
from pathlib import Path
s=json.loads(Path("$SNAP").read_text())
print(f"[{s.get('utc','?')}] CAMPAIGN {s.get('pod_name')} id={s.get('pod_id')} "
      f"Rung {s.get('rung')} lever={s.get('lever')} {s.get('levers_done')}/{s.get('levers_total')} "
      f"promoted={len(s.get('promoted') or [])} spend=\${s.get('spend_usd')} (local_est=\$$SPEND) "
      f"ETA={s.get('eta_minutes')}m gap_to_953={s.get('gap_to_953')} | {s.get('status')}")
PY
  else
    echo "[$STAMP] CAMPAIGN pod=$POD_ID spend_est=\$$SPEND waiting for first snapshot"
  fi

  # Remote health
  $SSH 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null; pgrep -af gpu_autonomous\|gpu_matrix | head -3' 2>/dev/null | sed "s/^/[$STAMP] remote: /" || echo "[$STAMP] remote: ssh_health_fail"

  bash "$ROOT/scripts/campaign_backup_sync.sh" || true
  sleep "$INTERVAL"
done
