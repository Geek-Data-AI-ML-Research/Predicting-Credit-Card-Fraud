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
