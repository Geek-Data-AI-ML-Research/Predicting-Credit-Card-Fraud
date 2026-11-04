#!/usr/bin/env bash
# Pull overnight checkpoints from A40 → local so a pod crash is recoverable.
set -euo pipefail
ROOT="${S6E11_ROOT:-/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/casmi_runpod}"
HOST="${RUNPOD_HOST:-194.68.245.44}"
PORT="${RUNPOD_PORT:-22056}"
RSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT"
REMOTE=root@$HOST
DEST="$ROOT/checkpoints/from_pod"
mkdir -p "$DEST"/{logs,preds_phaseA,preds_phaseB,submissions,heartbeat}

rsync -az --partial --inplace -e "$RSH" \
  "$REMOTE:/workspace/s6e11/logs/overnight/" "$DEST/logs/" || true
rsync -az --partial --inplace -e "$RSH" \
  "$REMOTE:/workspace/s6e11/preds/phaseA/" "$DEST/preds_phaseA/" || true
rsync -az --partial --inplace -e "$RSH" \
  "$REMOTE:/workspace/s6e11/preds/phaseB/" "$DEST/preds_phaseB/" || true
rsync -az --partial --inplace -e "$RSH" \
  --include 'sub_A*' --include 'sub_B*' --exclude '*' \
  "$REMOTE:/workspace/s6e11/submissions/" "$DEST/submissions/" || true
rsync -az --partial --inplace -e "$RSH" \
  "$REMOTE:/workspace/s6e11/checkpoints/overnight/" "$DEST/heartbeat/" || true

# mirror into working tree for resume
mkdir -p "$ROOT/logs/overnight" "$ROOT/preds/phaseA" "$ROOT/preds/phaseB" "$ROOT/submissions"
rsync -a "$DEST/logs/" "$ROOT/logs/overnight/" || true
rsync -a "$DEST/preds_phaseA/" "$ROOT/preds/phaseA/" || true
rsync -a "$DEST/preds_phaseB/" "$ROOT/preds/phaseB/" || true
rsync -a "$DEST/submissions/" "$ROOT/submissions/" || true

python3 - <<PY
import json, time
from pathlib import Path
root=Path("$ROOT")
dest=Path("$DEST")
hb=dest/"heartbeat"/"heartbeat.json"
status={
  "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "phaseA_npy": sorted(p.name for p in (root/"preds/phaseA").glob("*")),
  "phaseB_npy": sorted(p.name for p in (root/"preds/phaseB").glob("*.npy")),
  "subs": sorted(p.name for p in (root/"submissions").glob("sub_[AB]*")),
  "logs": {p.name: p.stat().st_size for p in (root/"logs/overnight").glob("*") if p.is_file()},
}
if hb.exists():
  try: status["pod_heartbeat"]=json.loads(hb.read_text())
  except Exception: pass
(dest/"last_pull.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status, indent=2))
PY
