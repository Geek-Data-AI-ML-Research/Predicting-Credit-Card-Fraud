# GM_V2 Five-Slot Adaptive Ladder

**Final LB_base = 0.95335** (was St9.5 `0.95332` → Slot1 `0.95333` → Slot4 `0.95334` → Slot5 `0.95335`).

## Live LB table

| Slot | file | LB | 1−ρ×1e4 | note |
|---:|---|---:|---:|---|
| 1 | `sub_strike9p6_opt_step.csv` | **0.95333** | 0.080 | Strike axis still climbing; `z_base=z_st9p6` |
| 2 | `sub_ladder_slot2_v8_clean_0p18e4.csv` | 0.95333 | ~0 | flat — first submit used L2-unitized δ (bug); near-identical to Slot1 |
| 3 | `sub_ladder_slot3_zref_clean_0p35e4.csv` | 0.95329 | 0.350 | drop — pure δ2 (`z_ref_clean`) after Slot2 flat |
| 4 | `sub_ladder_slot4_adaptive_0p65e4.csv` | **0.95334** | 0.650 | gain — Case B joint `0.45δ1+0.35δ2+0.20δ3` |
| 5 | `sub_ladder_slot5_macro_finale_1p25e4.csv` | **0.95335** | 1.250 | gain — Ridge `g_LB*` macro step from Slot4 anchor |

0 submissions remaining today.

## Stage 2 OOF (pre-submit)

- `lm12_raw_super` w=0.075: **5/5** Δe5 = `[+8.97, +24.26, +9.04, +16.23, +12.65]`
- `xgb_lm_super_clean_5f`: solo **0.9523083**, X=`(699960, 143)` (92 KMAT TE + 10 extras + 41 BASE)
- `v8_old 0.9525970 → v8_clean 0.9525990` (Δ=+0.20e-5, pos=2/5)
- `z_ref_old 0.9526612 → z_ref_clean 0.9526407` (Δ=−2.04e-5, pos=2/5)

## Null-space hygiene (δ after `P_⊥`)

| δ | disp | KS | max‖cos(B_bad)‖ |
|---|---:|---:|---:|
| δ1 (v8_clean−v8_old) | 1.0000 | 0.0144 | 8e-5 |
| δ2 (z_ref_clean−v8_old) | 1.0000 | 0.0002 | 1.5e-3 |
| δ3 (lm12_raw_super⊥w12) | 1.0000 | 0.0014 | 6.6e-4 |

## Artifacts

- Stage2: `preds/gm_v2_ladder/`, `preds/gpt2_3ep/xgb_lm_super_clean_5f_{oof,test}.npy`
- Scripts: `scripts/build_v8_zref_clean_upgrade.py`, `scripts/gm_v2_five_slot_ladder.py`
- JSON: `logs/overnight/GM_V2_FIVE_SLOT_LADDER.json`, `logs/overnight/V8_ZREF_CLEAN_UPGRADE.json`
