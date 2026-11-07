#!/usr/bin/env python3
"""Submit Nov07 research ladder via kaggle_doh (DoH/fallback IP)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_doh  # noqa: E402

ORDER = [
    (
        "submissions/nov07_ladder/sub_s2_reverse_ev2_7hc.csv",
        "Nov07-S2: Reinforced reverse-ev2 + hc/7hc from 0.95243",
    ),
    (
        "submissions/nov07_ladder/sub_s1_peak_strong_ascent.csv",
        "Nov07-S1: Stronger ascent from 0.95243 + hc/7hc",
    ),
    (
        "submissions/nov07_ladder/sub_s7_anti_soft_toward_hc.csv",
        "Nov07-S7: Anti-softpseudo toward pure v4p_hc",
    ),
    (
        "submissions/nov07_ladder/sub_s5_peak_plus_blend.csv",
        "Nov07-S5: Peak + OOF logit blend + ascent",
    ),
    (
        "submissions/nov07_ladder/sub_s3_peak_master_resid.csv",
        "Nov07-S3: Peak + hc_master residual + 7hc",
    ),
    (
        "submissions/nov07_ladder/sub_s8_live10_prop_axes.csv",
        "Nov07-S8: Live10 + proprietary multi-axis",
    ),
    (
        "submissions/nov07_ladder/sub_s4_ev456_consensus.csv",
        "Nov07-S4: EV4/5/6 consensus + ascent",
    ),
]


def main() -> None:
    n, best, _rows = kaggle_doh.today_utc_count()
    print(f"QUOTA today={n} left~={max(0, 10 - n)} best={best}", flush=True)
    left = min(7, max(0, 10 - n))
    ladder = []
    best_lb = max(best, 0.95243)
    for path, msg in ORDER[:left]:
        p = ROOT / path
        print("SUBMIT", p.name, flush=True)
        resp = kaggle_doh.submit(p, msg)
        print("RESP", resp[:350], flush=True)
        try:
            j = json.loads(resp)
            if j.get("code"):
                print("API_ERR", j, flush=True)
                ladder.append({"file": p.name, "ok": False, "err": j})
                break
        except Exception:
            pass
        sc = kaggle_doh.poll_file(p.name, timeout_s=540, sleep_s=10)
        print("SCORE", p.name, sc, flush=True)
        if sc is not None:
            best_lb = max(best_lb, sc)
        ladder.append({"file": p.name, "public_lb": sc, "msg": msg})
        (ROOT / "logs/overnight/nov07_submit_ladder.json").write_text(
            json.dumps({"best_lb": best_lb, "ladder": ladder}, indent=2)
        )
        if sc is not None and sc >= 0.953000:
            print("GOAL HIT", flush=True)
            break
    print("FINAL_BEST", best_lb, flush=True)
    print(json.dumps(ladder, indent=2), flush=True)


if __name__ == "__main__":
    main()
