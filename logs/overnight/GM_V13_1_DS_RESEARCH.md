# GM V13.1 DS Research — Can we raise 0.95388?

**NO_SUBMIT** · elapsed 46.95s · public noise ref ±13e-5 (60k)

## Decision

**YES_SMALL** (confidence: **low**)

- d1 w=0.035 clears 5/5; smaller step than 0.04
- WARNING: d1 retune failed nested subfold bar — OOF main-fold may overstate
- No public artifact exists for 0.95395–0.95465; cannot close +11e-5 to #2 from open banks alone
- Public LB differences of ~1e-5 are within noise; only submit if cost of a slot is acceptable for a possible +1 tick

## Main-fold LOFO vs ATH proxy

| candidate | 5/5 | meanΔ | min | 90% boot CI | nested SF | DS pass |
|---|---:|---:|---:|---|---:|---|
| `d1_w0.035` | 5 | 0.424 | 0.030 | [0.194, 0.686] | 0/5 | True |
| `d1_w0.040` | 5 | 0.777 | 0.015 | [0.337, 1.266] | 1/5 | True |
| `d1_w0.045` | 4 | 1.051 | -0.201 | [0.376, 1.826] | —/5 | False |
| `ATH+sov0.02` | 5 | 1.165 | 0.430 | [0.803, 1.508] | —/5 | True |
| `ATH+sov0.03` | 5 | 1.736 | 0.542 | [1.15, 2.284] | 2/5 | True |
| `ATH+sov0.04` | 5 | 2.264 | 0.657 | [1.492, 2.983] | —/5 | True |
| `ATH+newton0.03` | 5 | 1.098 | 0.490 | [0.757, 1.416] | —/5 | True |
| `ATH+newton0.04` | 5 | 1.417 | 0.661 | [0.995, 1.81] | 2/5 | True |

## Nested subfold (within each main fold)

- **d1_w0.035**: 0/5 main folds have subfold 5/5 · nested_pass=False
- **d1_w0.040**: 1/5 main folds have subfold 5/5 · nested_pass=False
- **ATH+sov0.03**: 2/5 main folds have subfold 5/5 · nested_pass=False
- **ATH+newton0.04**: 2/5 main folds have subfold 5/5 · nested_pass=False

## If submitting (optional)

1. `sub_v13p1_bold_d1_w0.035_NOSUBMIT.csv` — conservative d1 retune (E[public]≈0.233 e-5)

## Avoid / hold

- `ATH+sov*`: OOF may clear but nested=False / CI=[1.15, 2.284]; no public-transfer prior on this base (L3/L4 lesson)
- `ATH+newton*`: fails nested stability and/or no transfer prior

## Bottom line

A **1-tick** public bump is plausible via `d1∈{0.035,0.04}`; a climb to **0.95399+** is **not** supported by current banks.
