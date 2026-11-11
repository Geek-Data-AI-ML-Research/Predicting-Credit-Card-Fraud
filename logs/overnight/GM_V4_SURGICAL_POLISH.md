# GM v4 Surgical Polish

**NO_SUBMIT=True**. In-script elapsed **315.0s** (volume I/O; compute after LB cache ≈3–4 min). Target was <90s on warm local disk.

## Fix 1 — δ1_v4 Stage-B 4-family residual

- Root cause of v3 `KS=0.0194`: FSL-mixed test vs pre-FSL backbone + P⊥ after PCHIP without re-align. v4 does **P⊥ → Method-B → exact order-stat map**.
- Chosen: **`v8_v3_plus_lgbf_λ0.5`** (v8_v3 residual + 0.5×`lgb_gpt2_3ep_fixed` residual) — flips Fold 3 to **5/5**
- ΔOOF e-5: `[+3.92, +4.26, +0.76, +1.54, +2.94]` pos=**5/5**
- disp=1.0000 KS≈**3.5e-4** (λ0.75 reaches KS≈1.9e-4 but drops to 4/5; kept 5/5)
- variants: `{'v8_v3_minus_old': {'disp': 0.9999999999989985, 'ks': 0.00040668835671237025}, 'stageB_vs_v8_w12': {'disp': 0.999999999999, 'ks': 3.1682585789005735e-05}, 'lgb_fixed_vs_v8': {'disp': 0.9999999999989997, 'ks': 1.666755560303379e-05}, 'v8_v3_plus_lgbf_λ0.25': {'disp': 0.9999999999989984, 'ks': 0.000453357512400665}, 'v8_v3_plus_lgbf_λ0.5': {'disp': 0.9999999999989988, 'ks': 0.0003500186676622663}, 'v8_v3_plus_lgbf_λ0.75': {'disp': 0.9999999999989994, 'ks': 0.00018882779373308178}, 'v8_v3_plus_lgbf_λ1.0': {'disp': 0.9999999999989997, 'ks': 0.00018992010488039224}}`

## Fix 2 — lm_raw_super_v4 + δ3_v4

- **masaya rescued**: align + exact order-stat map **before** KS gate → KS=7.5e-5 (was 0.0017 reject in v3), **5/5** kept at w≈0.116
- Kept: `[{'name': 'r_gpt2m_bayes', 'w': 0.075}, {'name': 'r_masa_raw', 'w': 0.116}]`
- `z_lm_raw_super_v4` AUC=0.9508614 (super0=0.9505715, +29e-5)
- Trials: `[{'name': 'r_masa_raw', 'disp': 0.9999999999989998, 'ks': 7.533213170118902e-05, 'delta_e5': [53.297526029971735, 34.39743412410046, 13.572480601009573, 13.693971407446881, 23.860048656221267], 'pos': 5, 'kept': True, 'w': 0.11600000000000002}, {'name': 'r_rev_raw', 'disp': 0.999999999999, 'ks': 3.5133443206658566e-05, 'delta_e5': [0.0, 0.0, -2.3934621018129043, 0.0, 0.0], 'pos': 0, 'kept': False, 'w': 0.003}]`
- d3_linear: `{'name': 'd3_v4_linear', 'auc': 0.9509394521222575, 'delta_vs_super0_1e5': [47.67426992128287, 35.84444866496783, 25.420475330939496, 42.017175928998896, 34.48226018587519], 'pos': 5}` disp/KS=`[0.9999999999989996, 0.0001799540515181608]`
- d3_boundary: `{'name': 'd3_v4_boundary', 'auc': 0.9509499273127988, 'delta_vs_super0_1e5': [47.33606634770293, 33.903081507724586, 25.50809785629715, 46.77986411497636, 36.209542520548865], 'pos': 5}` disp/KS=`[0.999999999999, 0.00012983426700896583]`
- Chosen δ3: **`boundary`**

## Fix 3 — Anti-poison purge + historical Ridge

- S3 coords (hist basis): `{'d_ladder': 0.29079609777907983, 'e1': -0.47124042274840816, 'e2': 0.339397615586736, 'd2_bad': 0.00973152480773956, 'd1_old': 0.00037580624652272884, 'd3_old': 0.0006700344515091257}`
- S5 coords (hist basis): `{'d_ladder': -5.229235541891899, 'e1': 1.3045423493455344, 'e2': -2.1787236067215465, 'd2_bad': 0.0016053096066865466, 'd1_old': -0.02735682812329384, 'd3_old': 0.03126620682031832}`
- p_purge disp/KS=`[0.9999999999989995, 0.00022001173395913565]` cos(e1)=0.03437216025691789 cos(e2)=0.0024058324091503444 cos(d2_bad)=0.009062572498830435
- Hist Ridge LOO MAE=2.836244926360108e-05 coefs=`{'d_ladder': 1.4443058272960717e-05, 'e1': 2.552710754135977e-05, 'e2': -4.594848391595557e-05, 'd2_bad': 0.00022685031359447408, 'd1_old': 5.432272913055959e-05, 'd3_old': -0.00012743986764730486}`

## Step 4 — Staged candidates (`NO_SUBMIT`)

| slot | 1−ρ(S5) | pred_LB | cos(e1) | cos(e2) | cos(δ2_bad) | disp | KS | file |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| slot1_pure_s5_depoison | 0.250e-4 | 0.95337 | +0.029 | +0.000 | +0.000 | 1.0000 | 0.000225 | `sub_gm_v4_slot1_pure_s5_depoison_NOSUBMIT.csv` |
| slot2_stageB_4fam_ks0 | 0.450e-4 | 0.95337 | +0.017 | +0.006 | -0.000 | 1.0000 | 0.000357 | `sub_gm_v4_slot2_stageB_4fam_ks0_NOSUBMIT.csv` |
| slot3_step_up_0p85e4 | 0.850e-4 | 0.95337 | +0.074 | +0.002 | -0.000 | 1.0000 | 0.000291 | `sub_gm_v4_slot3_step_up_0p85e4_NOSUBMIT.csv` |
| slot4_macro_1p50e4 | 1.500e-4 | 0.95337 | +0.090 | +0.001 | -0.000 | 1.0000 | 0.000274 | `sub_gm_v4_slot4_macro_1p50e4_NOSUBMIT.csv` |

`NO_SUBMIT=True`.