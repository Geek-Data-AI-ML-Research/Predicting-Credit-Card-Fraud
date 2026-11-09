# Day-2 Submit Plan (Strike 10 SoftPseudo Orth Leap)

Base 7M OOF: `0.9526612` → Soft-only OOF: `0.9527775` (Δ `+11.63e-5`, pos `5/5`).
Soft+IDF grid OOF: `0.9527663` (Δ `+10.52e-5`).

## Recommended order
1. **Slot A** — `sub_day2_slotA_softpseudo_x040.csv` (0.40× softpseudo — lead)
2. **Slot B** — `sub_day2_slotB_softpseudo_x055.csv` (0.55× if A gains)
3. **Slot E** — `sub_day2_slotE_soft_x0p40_plus_cont.csv` (soft + cont gradient)
4. **Slot F** — `sub_day2_slotF_soft_idf_x0p40.csv` (soft+IDF mid)
5. Hold remaining for LB-gradient follow-up; use C/H if A regresses; D only if mid-scale wins big.

## Submit commands (quota reset)
```bash
kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotA_softpseudo_x040.csv -m "Day2 SlotA: St9(0.95330)+0.40*SoftPseudoDiff orth"
kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotB_softpseudo_x055.csv -m "Day2 SlotB: St9+0.55*SoftPseudoDiff"
kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotE_soft_x0p40_plus_cont.csv -m "Day2 SlotE: St9+0.40*Soft+0.35*(St9-St8)"
```

Generated in 65.0s. Do **not** submit until daily quota resets.

## Boost research (Strike 10B) — use AFTER Slot A lands
SoftPseudo **family pack** OOF Δ `+19.67e-5` (5/5) vs soft-only `+11.63e-5`.
Weights: soft_diff=0.045, idf_p3=0.0225, gb_soft_bag3=0.025, soft_clean035=0.01.

1. Keep **Slot A** (`soft x040`) as first submit (single-axis).
2. If A ≥ 0.95330: **Slot I** `sub_day2_slotI_softfamily_x035.csv` (family boost).
3. Then **Slot J** / **Slot M** (family step-up / +cont).
4. **Slot L** if family regresses (soft@0.06 only).

```bash
kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotI_softfamily_x035.csv -m "Day2 SlotI: St9+0.35*SoftFamily(diff+idf+gb+c035)"
kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotJ_softfamily_x050.csv -m "Day2 SlotJ: St9+0.50*SoftFamily"
kaggle competitions submit -c playground-series-s6e11 -f submissions/sub_day2_slotM_softfamily_x040_cont.csv -m "Day2 SlotM: St9+0.40*SoftFamily+0.30*(St9-St8)"
```
