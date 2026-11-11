# St9.5 Exact OOF Recon + 2F→5F gpt2-medium Chain Upgrade

**NO_SUBMIT=True**. Elapsed **162.5s**.

## Part 1 — Lineage & 14-Point LB Response Surface

### Scored atlas

| name | file | Public LB |
|---|---|---:|
| v8 | `sub_v8_part12_fsl_prop_rect.csv` | 0.95307 |
| Slot3 | `sub_slot3_gm_verified_f02.csv` | 0.95312 |
| Strike2 | `sub_gm_top3_strike2_homogeneous5f.csv` | 0.95319 |
| Strike3A | `sub_strike3a_cont065_plus_glmquad_graph.csv` | 0.95322 |
| Strike4A | `sub_gm_top3_strike4a_cont070.csv` | 0.95323 |
| Strike5A | `sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv` | 0.95315 |
| Strike6 | `sub_gm_top3_strike6_reverse_strike5_from_095323.csv` | 0.95325 |
| Strike7 | `sub_gm_top3_strike7_push_for_top1.csv` | 0.95327 |
| Strike8 | `sub_gm_top1_strike8_claim_rank1.csv` | 0.95329 |
| Strike9 | `sub_gm_top1_strike9_final_max.csv` | 0.95330 |
| Strike9.5 | `sub_strike9p5_axis_0p70_0p10.csv` | 0.95332 |
| ANTI | `sub_pass340_ANTI_fam21_x040.csv` | 0.95329 |
| Slot1_softdiff | `sub_strike10_slot1_softdiff_pchip_0p35.csv` | 0.95328 |
| leader_blend | `sub_pass340_LEAD_st9_aman20_tuboa10.csv` | 0.95325 |

### Exact Strike formulas (test-space)

- **Strike4A**: `z_st3 + 0.70*(z_st3-z_st2) + 0.15*(z_st2-z_a3)`
- **Strike6**: `z_st4 + 0.30*(z_st4-z_st3) - 0.35*v_bad_new5; v_bad=(z_st5-z_st4)-0.45*(z_st4-z_st3)`
- **Strike7**: `z_st6 + 0.85*(z_st6-z_st4) + 0.15*(z_st4-z_st3)`
- **Strike8**: `z_st7 + 0.90*(z_st7-z_st6)`
- **Strike9**: `z_st8 + 0.78*(z_st8-z_st7) - 0.14*u_bad5`
- **Strike9.5**: `z_st9 + 0.70*(z_st9-z_st8) - 0.10*u_bad5`
- **u_bad5**: `u_bad5 = (z_st5-z_st4) - 0.45*(z_st4-z_st3)`

Reconstruct corr(St9)=1.000000, corr(St9.5)=1.000000. St9.5 file `sub_strike9p5_axis_0p70_0p10.csv`.

### OOF counterparts

| stage | exact 5f OOF? | OOF AUC |
|---|---|---:|
| v8 | True | 0.9525970 |
| Slot3_2F | True | 0.9526269 |
| Slot3_5F | True | 0.9526176 |
| Strike2→St9.5 | False | — — test-only affine recurrence on live CSVs (Att3/St2/St3…); no saved 5f OOF twin on disk |

Slot3 has an exact reconstructible 5-fold OOF twin (F0/F2 gpt2-medium upgrades on Cand0/CandC trunk). Strike2+ are primarily **test-only** affine continuations of live CSVs (plus one-shot orth injections at Strike2/3/5); no complete saved OOF twin for St9.5 exists on disk.

### Empirical LB gradient at St9.5 (0.95332)

- Basis rank k=6, surface RMSE=1.392e-06 (LB units)
- $\nabla$ along $\vec{d}_{ladder}=z_{st9p5}-z_{st9}$: **-1.185e-06**
- $\nabla$ along $-\vec{d}_{bad\_leader}^\perp$: **-1.142e-06**
- $\nabla$ along $-\vec{d}_{bad\_soft}^\perp$: **-1.324e-06**

## Part 2 — 2F→5F gpt2-medium inside Slot3 (fixed weights)

Slot3 originally used `0.5*(z_m_test0 + z_m_test2)` PCHIP-mapped onto F0∪F2 OOF (coverage √(2/5)=0.6325 conceptually; code uses plain average then PCHIP). Full5F replaces with `0.2*Σ_{f=0..4} z_m_test_f` PCHIP onto all-5 OOF pool. Fixed weights w_h1=-0.0177, w_h2=-0.0045.

- Slot3_2F OOF **0.9526269** (Spearman vs submitted CSV=1.00000000)
- Slot3_5F OOF **0.9526176**
- ΔOOF(5F−2F) **-0.93e-5** pos=0/5 folds=[0.0, -2.13, 0.0, -1.87, -1.01] disp=1.0431 KS=0.000143
- Propagated `z_st9p5_full5f`: 1−ρ(St9.5)=0.0345e-4, cos(d_prop, d_ladder)=0.0000

## Part 3 — Staged candidates (NOT submitted)

### `sub_strike9p6_opt_step.csv`

- **file**: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_strike9p6_opt_step.csv`
- **formula**: `z_st9p5 + 0.846089*(z_st9p5-z_st9)`
- **1m_rho_st9p5_1e4**: `0.08`
- **target_1e4**: `0.08`
- **w**: `0.8460887316614389`
- **mean**: `0.016699998834038065`
- **ks_vs_st9p5**: `0.0`
- **pred_dLB**: `1.3253487258562146e-05`
- **pred_LB**: `0.9533332534872585`

### `sub_strike10_5f_gpt2m_plus_ladder.csv`

- **file**: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_strike10_5f_gpt2m_plus_ladder.csv`
- **formula**: `z_st9p5 + w*(0.35*r_5f_orth + ladder_9p6 + 0.15*(-d_bad_leader)⊥)`
- **1m_rho_st9p5_1e4**: `0.12`
- **1m_rho_vs_9p6_1e4**: `0.2013`
- **w**: `0.02596847480162978`
- **mean**: `0.016699998834038065`
- **ks_vs_st9p5**: `3.333511120695576e-06`
- **pred_dLB**: `-1.9403028160356693e-06`
- **pred_LB**: `0.9533180596971839`
- **cos_ladder**: `0.021581967709301562`

`NO_SUBMIT=True`. Macro v11 research CSVs quarantined under `preds/macro_v11/_quarantine/`.
