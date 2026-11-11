# GM Deep-Tech V7 Post-0.95354 Report

**NO_SUBMIT=True**. ATH locked at **Public LB `0.95354`** (Master Slot 7 / S7_A).

## Phase 0 — 9-point LB surface & ray OOF ceiling
- OOS S7 check (fit goodpjw..S6 → predict S7): pred=`0.953514` actual=`0.95354` err=`-2.55e-5`.
- OOF vertex **t\*** = `1.15` along S5→S6→S7 ray.

| t | 1−ρ(areeb) | 5F Δe-5 | F0 Δe-5 | pred LB |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00e-04 | +0.04 | +0.05 | 0.953532 |
| 1.15 | 1.32e-04 | +0.05 | +0.06 | 0.953566 |
| 1.30 | 1.69e-04 | +0.06 | +0.07 | 0.953623 |
| 1.45 | 2.10e-04 | +0.06 | +0.08 | 0.953711 |
| 1.60 | 2.56e-04 | +0.07 | +0.08 | 0.953835 |
| 1.80 | 3.24e-04 | +0.08 | +0.10 | 0.954075 |
| 2.00 | 4.00e-04 | +0.09 | +0.11 | 0.954421 |
| 2.25 | 5.06e-04 | +0.10 | +0.15 | 0.955042 |

## Phase 1 — Condensed blocks
- shape o/t = `[699960, 7]` / `[299984, 7]`
- per-col best Δe-5 vs z_ref_old: `[0.07, 0.01, 0.31, 0.01, 0.43, 0.74, 0.07]`

## Phase 2 — L7 denoise
```
{
  "corr_student_vs_r": 0.5834840621986976,
  "noise_test_std": 0.9999999999943455,
  "student_auc_as_score": 0.950754043051992
}
```

## Phase 3 — Stage-B v7 solo OOF
| family | AUC |
|---|---:|
| `xgb_v7_bce_dual_anchor` | 0.9524929 |
| `xgb_v7_rank_pairwise` | 0.9522965 |
| `lgb_v7_extratrees_dual` | 0.9525161 |
| `cat_v7_oblivious_d4_d5` | 0.9523034 |

## Phase 4 — Dual gates
- δ1_v4: 5F Δ=`+3.22e-5` F0=`-0.01e-5` pos=5/5
- a1_stageB_v7: 5F Δ=`+3.26e-5` F0=`-0.01e-5` pos=4/5
- combo+L7denoise: 5F Δ=`+2.96e-5` F0=`-0.01e-5` pos=5/5
- Gate#1 vs S7_A (+2.38 / +9.50): **PASS signal**
- a1_stageB_v7 alignment: disp=`1.000000` KS=`0.000333`

## Tomorrow's 5-slot NOSUBMIT staging (from z_S7=0.95354)
| slot | name | 1−ρ(S7) | got | pred LB | file |
|---:|---|---:|---:|---:|---|
| 1 | pure_ray_tstar | 2.47e-05 | 2.47e-05 | 0.953600 | `sub_gm_v7_post095354_slot1_NOSUBMIT.csv` |
| 2 | s7_stageBv7_L7denoise | 1.50e-05 | 1.50e-05 | 0.953581 | `sub_gm_v7_post095354_slot2_NOSUBMIT.csv` |
| 3 | medium_combo | 3.50e-05 | 3.50e-05 | 0.953606 | `sub_gm_v7_post095354_slot3_NOSUBMIT.csv` |
| 4 | large_combo | 6.50e-05 | 6.50e-05 | 0.953636 | `sub_gm_v7_post095354_slot4_NOSUBMIT.csv` |
| 5 | adaptive_reserve | 4.50e-05 | 4.50e-05 | 0.953613 | `sub_gm_v7_post095354_slot5_NOSUBMIT.csv` |

**NO_SUBMIT=True** — 0 daily submissions remaining; candidates staged only.