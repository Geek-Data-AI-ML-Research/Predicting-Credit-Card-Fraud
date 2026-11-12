# GM V12.4 Further Lift Research + Last Submit

**Base ATH:** 0.95371 (L2 = Q17 anchor + 0.03·d1_v4)  
**POD TOUCHES: 0**

## Research findings
1. **No third axis** clears 5/5 on SAFE = L2+sov+G26 (0 singles, 0 pairs). P_super/stageB remain **banned** (L3 poison).
2. **Sameer LM logit mix** on L2 fails hard (F0 −7 to −18e-5) — do not use on this base.
3. **Public top 0.95385** still has no reproducible kernel/dataset recipe.
4. **Only lever left in-bank:** joint retune of `(d1, sov, G26)`.
5. Alone, `sov−wide` fails F0 and `G26−MS` fails F2; together they clear OOF 5/5 (complementary folds) — but **public did not reward** that pair (L4).

## Candidates (OOF vs L2)

| recipe | OOF pos/5 | meanΔ e-5 | min fold | live public |
|---|---:|---:|---:|---:|
| SAFE `d1=0.03, sov=0.01, G26=0.015` | 5/5 | +2.36 | +0.39 | (not submitted) |
| Robust `d1=0.028, sov=0.012, G26=0.018` | 5/5 | +2.62 | +0.42 | **0.95370** (L4) |

## Last submit (ref `58285309`)
- File: `submissions/sub_v12p4_L2plus_d10.028_sov0.012_G260.018_robust_NOSUBMIT.csv`
- Formula: `L(anchor)+0.028*unitize(d1_v4)+0.012*orth(sov−wide)+0.018*orth(G26−MS)`
- **Public LB: 0.95370** (−1e-5 vs ATH)
- Quota after: **0 remaining** that UTC day

## Ceiling / next
Within local/public banks, the `(d1,sov,G26)` plane is saturated for *public* lift even when OOF 5/5. Beating **0.95371** toward 0.95377–0.95385 needs **new private signal** (complete 5-fold Compact-LM seed2026 K=10, new LM, or undisclosed top-team members)—not more blending of the current bank.

JSON: `GM_V12_4_FURTHER_LIFT_SEARCH.json`, `GM_V12_4_FURTHER_LIFT_BEST.json`.
