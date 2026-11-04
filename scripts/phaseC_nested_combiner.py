#!/usr/bin/env python3
"""Phase C entrypoint — delegates to phaseC_pass12.

Default after overnight Phase B: Pass 2 (bank63 + B bags, gated vs Pass 1).
For Pass 1 only: PHASE_C_PASS=1 python -u scripts/phaseC_pass12.py
NO auto-submit.
"""
from __future__ import annotations

import os
import sys

# Overnight runner historically called this script after Phase B.
# Always run Pass 2 here so the B→C handoff applies the keep-gate vs Pass 1.
os.environ.setdefault("PHASE_C_PASS", "2")

from phaseC_pass12 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
