#!/usr/bin/env bash
set -uo pipefail
export PYTHONUNBUFFERED=1 TMPDIR=/tmp/s6e11_tmp S6E11_ROOT=/tmp/s6e11_work S6E11_PERSIST=/workspace/s6e11
ROOT=/tmp/s6e11_work; LOG=/tmp/s6e11_logs; WS=/workspace/s6e11; PY=/tmp/venv/bin/python
mkdir -p "$TMPDIR" "$LOG" "$ROOT"/{logs,preds/unattended,preds/stage7,preds/stage9,submissions} "$WS"/{logs,preds/unattended,submissions}
cd "$ROOT"
echo "[orch10] START $(date -u)" | tee -a "$LOG/unattended_10tests.log"
persist() {
  mkdir -p "$WS"/{logs,preds/unattended,submissions}
  cp -a logs/*.json "$WS/logs/" 2>/dev/null || true
  cp -a preds/unattended/* "$WS/preds/unattended/" 2>/dev/null || true
  cp -a submissions/* "$WS/submissions/" 2>/dev/null || true
}
$PY -c "import torch,lightgbm,xgboost,catboost,pandas; assert torch.cuda.is_available()" || {
  echo "[orch10] FATAL deps" | tee -a "$LOG/unattended_10tests.log"; exit 2; }
set +e
$PY -u scripts/run_unattended_10tests.py "$@" 2>&1 | tee -a "$LOG/unattended_10tests.run.log"
rc=${PIPESTATUS[0]}
persist
date -u > logs/CAMPAIGN_10TESTS_COMPLETE
# Pass-1 must NOT stop the pod — T06/T07 may still need a re-run + Pass-2 T08–T10.
# Set ALLOW_POD_STOP=1 only after Pass-2 finishes.
if [ "${ALLOW_POD_STOP:-0}" = "1" ]; then
  echo stop > logs/REQUEST_POD_STOP
  echo "[orch10] COMPLETE rc=$rc REQUEST_POD_STOP $(date -u)" | tee -a "$LOG/unattended_10tests.log"
else
  rm -f logs/REQUEST_POD_STOP
  echo "[orch10] COMPLETE rc=$rc NO_POD_STOP (ALLOW_POD_STOP!=1) $(date -u)" | tee -a "$LOG/unattended_10tests.log"
fi
persist
exit 0
