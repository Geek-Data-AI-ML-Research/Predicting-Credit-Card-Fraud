#!/usr/bin/env bash
# KEEP WARM — Stage 8.3 canary → 8.1 → Stage 9.1 / 9.2 → Stage 10 diglead
set -euo pipefail
export TMPDIR=/tmp/s6e11_tmp PYTHONUNBUFFERED=1
mkdir -p /tmp/s6e11_tmp /tmp/s6e11_logs /tmp/s6e11_work/{logs,preds/stage6,preds/stage7,preds/stage8,preds/stage9,preds/stage10,submissions}
cd /tmp/s6e11_work
PY=/tmp/venv/bin/python
LOG=/tmp/s6e11_logs
WS=/workspace/s6e11

persist() {
  mkdir -p "$WS/logs" "$WS/preds/stage6" "$WS/preds/stage7" "$WS/preds/stage8" "$WS/preds/stage9" "$WS/preds/stage10" "$WS/submissions"
  cp -a logs/*.json "$WS/logs/" 2>/dev/null || true
  cp -a preds/stage6/* "$WS/preds/stage6/" 2>/dev/null || true
  cp -a preds/stage7/* "$WS/preds/stage7/" 2>/dev/null || true
  cp -a preds/stage8/* "$WS/preds/stage8/" 2>/dev/null || true
  cp -a preds/stage9/* "$WS/preds/stage9/" 2>/dev/null || true
  cp -a preds/stage10/* "$WS/preds/stage10/" 2>/dev/null || true
  cp -a submissions/* "$WS/submissions/" 2>/dev/null || true
}

echo "[orch] PIVOT START $(date -u)" | tee -a "$LOG/orch_campaign.log"

# Write 8.3 EARLY_ABORT from observed fold0 if no result yet
if [[ ! -f logs/stage8_3_pseudolabel_result.json ]]; then
  $PY - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
Path("logs").mkdir(exist_ok=True)
doc = {
  "stage": "8.3_pseudolabel",
  "created_utc": datetime.now(timezone.utc).isoformat(),
  "status": "EARLY_ABORT_FOLD0_max_0.9434",
  "fold0": {"ArmA": 0.943448, "ArmB": 0.938263},
  "fold1_partial": {"ArmA": 0.945065, "ArmB": 0.941890},
  "promote": False,
  "reason": "Fold0 max 0.9434 << 0.9500; te_smooth50 saturated (killed mid fold2). Pivot to Stage 8.1 / 9.",
  "policy": "NO_KAGGLE_SUBMIT",
  "keep_pod_warm": True,
  "next": ["stage8_1_raw_dual_realmlp", "stage9_1_full73_power_hillclimb", "stage9_2_orig20k_orthogonal_residual"],
}
Path("logs/stage8_3_pseudolabel_result.json").write_text(json.dumps(doc, indent=2))
print("wrote EARLY_ABORT 8.3")
PY
fi

run_step() {
  local name="$1"
  local script="$2"
  local marker="$3"
  if [[ -n "$marker" && -f "$marker" ]]; then
    # allow re-run of 8.3 only if COMPLETE (not EARLY_ABORT) — skip abort markers
    if [[ "$name" == "stage8_3_pseudo" ]]; then
      echo "[orch] SKIP $name (abort/result marker exists)" | tee -a "$LOG/orch_campaign.log"
      return 0
    fi
    echo "[orch] SKIP $name (marker $marker)" | tee -a "$LOG/orch_campaign.log"
    return 0
  fi
  echo "[orch] START $name $(date -u)" | tee -a "$LOG/orch_campaign.log"
  $PY -u "$script" 2>&1 | tee "$LOG/${name}.run.log"
  echo "[orch] DONE $name $(date -u)" | tee -a "$LOG/orch_campaign.log"
  persist
}

# Skip already-done 8.2
run_step stage8_3_pseudo scripts/stage8_3_gpu_pseudolabel.py logs/stage8_3_pseudolabel_result.json
run_step stage8_1_realmlp scripts/stage8_1_raw_dual_realmlp.py logs/stage8_1_raw_dual_realmlp_result.json
run_step stage9_1_power scripts/stage9_1_full73_power_hillclimb.py logs/stage9_1_full73_power_hillclimb_result.json
run_step stage9_2_ortho scripts/stage9_2_orig20k_orthogonal_residual.py logs/stage9_2_orig20k_orthogonal_residual_result.json
run_step stage10_diglead scripts/stage10_diglead_superbase.py logs/stage10_diglead_superbase_result.json
run_step stage10_reinject scripts/stage10_reinject_hillclimb.py logs/stage10_reinject_hillclimb_result.json

$PY - <<'PY'
import json, glob, os
from datetime import datetime, timezone
logs = {}
for p in sorted(glob.glob("logs/*.json")):
    try:
        logs[os.path.basename(p)] = json.load(open(p))
    except Exception as e:
        logs[os.path.basename(p)] = {"error": str(e)}
doc = {
    "campaign": "stage9_pivot_plus_stage10_diglead",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "pod": "83f0f0fwi39ccy",
    "keep_warm": True,
    "policy": "NO_KAGGLE_SUBMIT",
    "results": logs,
}
json.dump(doc, open("logs/stage7_and_8_gpu_campaign.json", "w"), indent=2)
s9 = {"updated_utc": datetime.now(timezone.utc).isoformat(), "policy": "NO_KAGGLE_SUBMIT"}
for k in ("stage9_1_full73_power_hillclimb_result.json", "stage9_2_orig20k_orthogonal_residual_result.json"):
    if k in logs:
        s9[k.replace("_result.json","")] = logs[k]
json.dump(s9, open("logs/stage9_results.json", "w"), indent=2)
s10 = {"updated_utc": datetime.now(timezone.utc).isoformat(), "policy": "NO_KAGGLE_SUBMIT"}
for k in ("stage10_diglead_superbase_result.json", "stage10_reinject_hillclimb_result.json"):
    if k in logs:
        s10[k.replace("_result.json","")] = logs[k]
json.dump(s10, open("logs/stage10_results.json", "w"), indent=2)
print("campaign aggregate written")
PY
persist
date -u > logs/CAMPAIGN_COMPLETE
echo "[orch] STAGE10 COMPLETE — SHUTDOWN REQUESTED $(date -u)" | tee -a "$LOG/orch_campaign.log"
# signal local stop watcher; do not self-terminate (needs account API)
echo stop > logs/REQUEST_POD_STOP
