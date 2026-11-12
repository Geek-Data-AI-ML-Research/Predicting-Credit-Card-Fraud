# GM V12.3 Live Ladder — 2026-11-12

**POD TOUCHES: 0** · Day start ATH **0.95355** (cand2)

## Live results (public LB)

| # | file | gate | public LB | Δ vs then-ATH |
|---|---|---|---:|---:|
| 1 | `sub_v12p3_slot1_PURE_Q17_anchor095370_LIVE.csv` | external 0.95370 + dual LOFO | **0.95370** | +15e-5 |
| 2 | `sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv` | OOF LOFO 5/5 d1@0.03 mean+1.82e-5 | **0.95371** | +1e-5 |
| 3 | `sub_v12p3_slot3_L2_Ps0.03_sb0.01_5of5_LIVE.csv` | OOF LOFO 5/5 vs L2 mean+1.75e-5 | **0.95366** | −5e-5 |
| 4 | `sub_v12p4_L2plus_d10.028_sov0.012_G260.018_robust_NOSUBMIT.csv` | OOF 5/5 vs L2 mean+2.62e-5 (retuned d1/sov/G26) | **0.95370** | −1e-5 |

Kaggle refs: L1=`58284730` · L2=`58284807` · L3=`58284886` · L4=`58285309`.

## Autopsy
- **L1** exact match to Q17 public apex — file integrity OK.
- **L2** δ1_v4 transferred (~55% of OOF). **Day ATH = 0.95371** (best public; team board shows this).
- **L3** P_super_v5 + stageB_4fam: OOF green, public red → **ban those axes** on Q17/L2 base.
- **L4** sov+G26 retune: OOF predicted +2.62e-5 vs L2; public **0.95370** (no transfer). Complementary-fold theory did not survive public 20%.

## Locked ATH recipe (do not dilute)
```
L(Q17 test_anchor_prob) + 0.03 * unitize(d1_v4)
```
File: `submissions/sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv` · public **0.95371**.

## Current record
**Public ATH 0.95371** · daily quota exhausted after L4 · next work needs **new private signal** (full 5F L7_s2026), not more bank blends.

See also: `GM_V12_4_FURTHER_LIFT_REPORT.md`, `GM_V12_DAY_SUMMARY_2026-11-12.md`.
