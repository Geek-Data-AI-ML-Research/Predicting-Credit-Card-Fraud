# GM Slot 7 Verified S7_A Submit

**LIVE SUBMIT**. Elapsed **192.6s**.
**LB_7 = `0.95354`** (prior ATH S6 = `0.95352`).

## Step 1 — samanyu1808 quick check

Both kernels reuse **goodpjw’s exact ladder skeleton** with the same public datasets (Strike9/10, compact-lm-seed7, oof-bank, megayak):

```
source_score  = A + 0.6*(A-F) + 0.12*(mega_old-F) + 0.06*(LM7-F)          # goodpjw 0.95346
updated_score = A + 0.6*(A-F) + 0.12*(mega_updated-F) + 0.06*(LM7-F)
# mega_updated = 0.5*current_megayak_LM + 0.5*GLM-boost LGB from oof-bank
```

No new private artifact beyond goodpjw’s bank + megayak LM blend — research note for overnight, not a `0.95369` source.

## Step 2 — OOF step sweep along S7_A

| +a2? | 1−ρ(S6) | Δ5F vs S6 | ΔF0 vs S6 | score | abs 5F | abs F0 |
|---|---:|---:|---:|---:|---:|---:|
| False | 8.00e-06 | -0.05 | +0.00 | -0.05 | +3.36 | +5.42 |
| False ** | 1.00e-05 | -0.06 | +0.07 | -0.03 | +3.35 | +5.49 |
| False | 1.20e-05 | -0.08 | +0.16 | +0.00 | +3.34 | +5.58 |
| False | 1.50e-05 | -0.10 | +0.20 | +0.01 | +3.32 | +5.62 |
| False | 1.80e-05 | -0.12 | +0.25 | -0.00 | +3.29 | +5.67 |
| False | 2.20e-05 | -0.17 | +0.38 | +0.02 | +3.25 | +5.80 |
| True | 8.00e-06 | -0.05 | +0.00 | -0.05 | +3.36 | +5.42 |
| True | 1.00e-05 | -0.06 | +0.07 | -0.03 | +3.35 | +5.49 |
| True | 1.20e-05 | -0.08 | +0.16 | +0.00 | +3.34 | +5.58 |
| True | 1.50e-05 | -0.10 | +0.20 | +0.01 | +3.32 | +5.62 |
| True | 1.80e-05 | -0.12 | +0.25 | +0.00 | +3.29 | +5.67 |
| True | 2.20e-05 | -0.17 | +0.38 | +0.02 | +3.25 | +5.80 |

**Selected:** 1−ρ(S6)=`1.00e-05` plus_a2=`False` (got=`1.00e-05`, 1−ρ(areeb)=`1.08e-04`).

## Step 3 — Submit

- file: `sub_master_slot7_verified_s7A.csv`
- n=`299984` mean=`0.016700` KS vs S6=`3.334e-06` nans=0
- message: `Master Slot 7: OOF+OOS-verified S7_A (S5->S6 continuation, t=1.00e-05, +3.35e-5 5F / +5.49e-5 F0 OOF)`
- **LB_7 = 0.95354**

Δ vs S6 ATH: **+2.0e-5**
