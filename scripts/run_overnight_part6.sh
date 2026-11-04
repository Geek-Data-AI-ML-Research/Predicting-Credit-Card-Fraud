#!/usr/bin/env bash
# Overnight unattended: Phase A → B → C → D. NO kaggle submit.
set -euo pipefail
export S6E11_ROOT="${S6E11_ROOT:-/workspace/s6e11}"
cd "$S6E11_ROOT"
mkdir -p logs/overnight preds/phaseA preds/phaseB submissions

PY="${PYTHON:-python3}"
echo "[overnight] root=$S6E11_ROOT py=$PY $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee logs/overnight/runner.log

# Prefer venv if present
if [[ -x /workspace/s6e11/.venv/bin/python ]]; then
  PY=/workspace/s6e11/.venv/bin/python
elif [[ -x /workspace/.venv/bin/python ]]; then
  PY=/workspace/.venv/bin/python
fi

echo "[overnight] PHASE A audit" | tee -a logs/overnight/runner.log
$PY -u scripts/phaseA_audit_bank.py 2>&1 | tee logs/overnight/phaseA.log

echo "[overnight] PHASE B glm-boost" | tee -a logs/overnight/runner.log
set +e
$PY -u scripts/phaseB_glm_boost.py 2>&1 | tee logs/overnight/phaseB.log
B_RC=${PIPESTATUS[0]}
set -e
if [[ "$B_RC" -eq 2 ]]; then
  echo "[overnight] PHASE B canary ABORT — still running Phase C on bank-only" | tee -a logs/overnight/runner.log
elif [[ "$B_RC" -ne 0 ]]; then
  echo "[overnight] PHASE B failed rc=$B_RC — continuing to Phase C on whatever exists" | tee -a logs/overnight/runner.log
fi

echo "[overnight] PHASE C nested combiner" | tee -a logs/overnight/runner.log
set +e
$PY -u scripts/phaseC_nested_combiner.py 2>&1 | tee logs/overnight/phaseC.log
C_RC=${PIPESTATUS[0]}
set -e
echo "[overnight] PHASE C rc=$C_RC" | tee -a logs/overnight/runner.log

# integrity
$PY - <<'PY' 2>&1 | tee -a logs/overnight/runner.log
import pandas as pd, numpy as np, json
from pathlib import Path
root=Path('.')
subs=['submissions/sub_A_part6_nested_logit.csv','submissions/sub_B1_refit_core_only.csv','submissions/sub_B_refit_core_or_fusion.csv']
rep={}
for s in subs:
    p=root/s
    if not p.exists():
        rep[s]={'missing':True}; continue
    df=pd.read_csv(p)
    col=[c for c in df.columns if c!='id'][0]
    x=df[col].to_numpy()
    rep[s]={'n':len(df),'nan':int(np.isnan(x).sum()),'finite':bool(np.isfinite(x).all()),'min':float(np.min(x)),'max':float(np.max(x))}
    assert len(df)==299984 and np.isfinite(x).all()
Path('logs/overnight/phaseD_integrity.json').write_text(json.dumps(rep,indent=2))
print(json.dumps(rep,indent=2))
print('[overnight] DONE — NO SUBMIT')
PY

echo "[overnight] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/overnight/runner.log
