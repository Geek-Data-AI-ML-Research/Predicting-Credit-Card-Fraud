# GM V13.1 Research — Beyond ATH 0.95388

**NO_SUBMIT** · **POD TOUCHES: 0** · 2026-11-13

## Public headroom
| rank | score | public recipe? |
|---|---:|---|
| 1 | 0.95465 | no |
| 2 | 0.95399 | no |
| 3 | 0.95395 | no |
| **us** | **0.95388** | bold + 0.03·d1 |

Gap to #2: **+11e-5**. No new kernels/datasets above our ATH.

## OOF screen vs ATH proxy (`bold_MIX + 0.03·d1`)
`n=43` probes · **16** clear LOFO 5/5 (includes d1-on-bold baselines).

### Best lifts that beat current ATH weight
| candidate | OOF vs ATH | meanΔ e-5 | min fold |
|---|---|---:|---:|
| **d1 w=0.04** on bold | 5/5 | +0.78 | +0.015 |
| **d1 w=0.035** on bold | 5/5 | +0.42 | +0.030 |
| **newton_cholesky orth w=0.04** | 5/5 | +1.42 | +0.66 |
| **sovereign orth w=0.04** | 5/5 | +2.26 | +0.66 |
| sovereign orth w=0.03 | 5/5 | +1.74 | +0.54 |

### Dead / weak
- d3 / p_purge orth: 0/5 (hurt)
- d1 w≥0.05: 4/5 only (fails a fold)
- Q17 member re-injects (sameer / megayak shallow): 4/5 near-misses

## Interpretation
1. **Most credible next tick:** retune `d1` to **0.035–0.04** — same residual that already transferred Q17→L2 and bold→0.95388. OOF edge is small (min +0.015–0.03 e-5) → public may be flat or +1 tick.
2. **Higher OOF but unproven public:** sovereign / newton injects on ATH — stronger OOF, but new family on this base (L3/L4 taught OOF≠public).
3. **No path to 0.95465** from open banks.

## Staged NOSUBMIT
See `GM_V13_1_STAGED.json` / `sub_v13p1_*_NOSUBMIT.csv`.

## Verdict
**Yes, a small further lift is plausible** via `d1` retune; sovereign/newton are secondary research bets. Do **not** expect the 0.95399–0.95465 band without private top-team signal.
