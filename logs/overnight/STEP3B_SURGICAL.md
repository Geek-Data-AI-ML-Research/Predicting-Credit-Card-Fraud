# Step 3b — Surgical no-zanch probe (NO_SUBMIT)

Elapsed **438.5s**. Holding `sub_strike10A` — PassOnly lift was zanch-masked.

## Why hold Strike10A

Dropping `zanch_*` killed ridge pure-FE (+6.03→0). Portfolio C `disp=0.958` was soft_diff (`disp=1.028`) masking PassOnly (`disp=0.914`).

## zanch_* train/test bias

- bad std_ratio (<0.92 or >1.08): **0/140**
- KS>0.01: **0/140**
- eb20 mean-only PCHIP-clean (0.97–1.03, KS<0.005): **35/35**

## Bulsah-only / soft_diff PCHIP

| member | ΔOOF e-5 | pos | disp | KS | PassOnly gate |
|---|---:|---:|---:|---:|:---:|
| v10_ridge_bulsah_only | 0.03 | 3/5 | 0.8674 | 0.019389 | False |
| v10_lr_bulsah_only | 0.15 | 3/5 | 1.1319 | 0.065604 | False |
| bulsah+pchip_eb20 | 0.87 | 3/5 | 0.8778 | 0.016997 | False |
| soft_diff_orth_LM PCHIP | (see steps) |  | 1.0000 | 0.001435 | n/a |

### soft_diff PCHIP step grid

| 1-ρ e-4 | w | ΔOOF e-5 | folds | pos | disp | KS |
|---:|---:|---:|---|---:|---:|---:|
| 0.2 | 0.010728 | 3.65 | [1.68, 1.67, 6.07, 4.54, 4.87] | 5/5 | 1.0 | 0.001435 |
| 0.3 | 0.013141 | 4.38 | [2.02, 1.93, 7.29, 5.44, 5.93] | 5/5 | 1.0 | 0.001435 |
| 0.35 | 0.014194 | 4.71 | [2.18, 2.11, 7.86, 5.77, 6.43] | 5/5 | 1.0 | 0.001435 |
| 0.5 | 0.016969 | 5.52 | [2.58, 2.32, 9.27, 6.77, 7.65] | 5/5 | 1.0 | 0.001435 |

## Staged (NOT submitted)

- `sub_strike10_slot1_softdiff_pchip_0p35.csv`: Δ=4.71e-5 pos=5/5 disp=0.9999999999980289 KS=0.0014350700033601838 @1-ρ=0.35e-4
- `sub_strike10_slot2_clean_combo_0p45.csv`: **NOT WRITTEN** (bulsah failed PassOnly alone)

`NO_SUBMIT=True`. Await confirmation before Slot-1 probe.
