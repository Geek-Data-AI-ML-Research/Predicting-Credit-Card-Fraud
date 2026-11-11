# GM v4.1 — Fast Cache + Orthonormal Lock

**NO_SUBMIT=True**. Elapsed **288.7s**.

## Step 1 — Fast binary cache

- Cache: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/gm_v4_1_fast_cache.npz` (81.05 MB)
- Contents: 19 LB rank-logit vectors + scores, δ1_v4 / δ3_v4_lin / δ3_v4_bnd OOF+test, u_zjoint, u_glm_graph, z_ref_o, z_v8_old_oof, z_w12_oof, y, folds

## Step 2 — Gram-Schmidt orthonormal basis (vs St9.5 / S1)

- max|⟨qi,qj⟩| off-diag = `3.33e-15` (target 0)

| sub | LB | c_q1 | c_q2 | c_bad_tree | c_bad_S3 | c_bad_ldr | c_bad_soft | c_S5_win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Today_S1 | 0.95333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Today_S3 | 0.95329 | -7.3789 | +0.9397 | +0.9595 | +7.1491 | -0.0000 | +0.0000 | -0.0000 |
| Today_S4 | 0.95334 | -1.2785 | +0.4724 | +0.2665 | +3.0641 | +0.2615 | -0.1251 | +3.4026 |
| Today_S5 | 0.95335 | +6.1490 | +1.5764 | -3.2780 | +2.2705 | +1.8500 | -0.4254 | +52.1284 |

**Poison check:** Today_S4 `c_bad_S3=+3.0641`, Today_S5 `c_bad_S3=+2.2705`, Today_S5 `c_S5_win=+52.1284`.

## Step 3 — Joint 5/5 OOF synergy on `z_ref_o`

- `z_ref_o` AUC=0.9526407166623628
- mean weights: `{'d1_v4': 0.03799999999999999, 'd3_lin': 0.032, 'd3_bnd': -0.05}`
- **δ_v4_combo** Δe5=`[2.1905520065090833, 5.176564795084104, 2.1076013808407623, 11.007502066617825, 2.5699500252107654]` pos=5/5 AUC=0.9526885006843173
- hygiene: `{'d1_v4': {'disp': 0.9999999999989989, 'ks': 0.0003512705011943784, 'max_cos_bad': 0.000574809546267773}, 'd3_lin': {'disp': 0.999999999999, 'ks': 1.5879278043362532e-05, 'max_cos_bad': 0.0004467228749603503}, 'd3_bnd': {'disp': 0.999999999999, 'ks': 0.00012983426700896583, 'max_cos_bad': 0.00017850303305045568}, 'u_zjoint': {'disp': 0.999999999999, 'ks': 3.381112569744715e-05, 'max_cos_bad': 0.03797032677545463}, 'u_glm_graph': {'disp': 0.999999999999, 'ks': 1.055777587449791e-05, 'max_cos_bad': 0.03603431515821967}}`
- combo disp/KS/maxcos: `[0.999999999998999, 0.0002639948024306671, 0.0006405727235310377]`

## Step 4 — Staged candidates (`NO_SUBMIT`)

| slot | 1−ρ(S5) | cos(q1) | cos(bad_S3) | cos(bad_tree) | max\|cos B_bad\| | KS | file |
|---|---:|---:|---:|---:|---:|---:|---|
| slot1_depoison_S5 | 0.220e-4 | +0.447 | -0.0000 | -0.0000 | 1.54e-16 | 0.000329 | `sub_gm_v4p1_slot1_depoison_S5_NOSUBMIT.csv` |
| slot2_v4combo_micro_0p40e4 | 0.400e-4 | +0.268 | -0.0000 | -0.0000 | 1.75e-16 | 0.000044 | `sub_gm_v4p1_slot2_v4combo_micro_0p40e4_NOSUBMIT.csv` |
| slot3_v4combo_med_0p75e4 | 0.750e-4 | +0.267 | +0.0000 | +0.0000 | 1.38e-16 | 0.000133 | `sub_gm_v4p1_slot3_v4combo_med_0p75e4_NOSUBMIT.csv` |
| slot4_v4combo_large_1p30e4 | 1.300e-4 | +0.228 | +0.0000 | +0.0000 | 1.00e-16 | 0.000046 | `sub_gm_v4p1_slot4_v4combo_large_1p30e4_NOSUBMIT.csv` |
| slot5_adaptive_reserve | 2.000e-4 | +0.221 | -0.0000 | +0.0000 | 9.77e-17 | 0.000077 | `sub_gm_v4p1_slot5_adaptive_reserve_NOSUBMIT.csv` |

`NO_SUBMIT=True`.