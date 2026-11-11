# GM V5 Overnight 8hr Super Factory

**NO_SUBMIT=True**. Elapsed **1040.2s**.

## Phase 1 — z_ref_old + purge + X_v5

- `z_ref_old` AUC=`0.9526611588522081` folds=`[0.951956887685966, 0.9525950268462368, 0.9573680137861428, 0.9528724158361718, 0.9485100678219476]`
- NN δ_v4_combo: AUC=`0.9526828591835981` Δe5=`[-0.48570136372827477, 6.308511106489778, 0.19870427290236137, 0.5861698371023571, 2.8816003546983104]` pos=`4`/5 w=`{'d1_v4': 0.046, 'd3_lin': 0.004, 'd3_bnd': 0.0}`
- S5 purge coeffs: `{'c_s3': 2.2705297295487377, 'c_ldr': 1.8499755476712862, 'c_tree': -3.2780325006230657}`
- X_v5 meta: `{'shape_o': [699960, 138], 'shape_t': [299984, 138], 'n_kmat68': 68, 'n_kmat5seed': 15, 'extra_names': ['z_lm_raw_super_v4', 'z_gpt2m_bayes_5f', 'z_masa_5f', 'z_gpt2_rev_5f', 'z_gpt2_bidir_5f', 'z_lm_minus_v4p', 'z_lm_x_rare', 'z_lm_x_disp', 'P4_L4.3', 'P4_L4.2', 'P4_L4.5', 'z_tabm_5f', 'z_lm_x_L4.2', 'z_lm_x_tabm'], 'mono_idx': [84, 92, 91, 83], 'top15_keys': [6, 39, 0, 60, 3, 37, 1, 27, 47, 11, 38, 64, 25, 67, 43], 'z_ref_old_auc': 0.9526611588522081, 'combo_nn': {'w': {'d1_v4': 0.046, 'd3_lin': 0.004, 'd3_bnd': 0.0}, 'delta_e5': [-0.48570136372827477, 6.308511106489778, 0.19870427290236137, 0.5861698371023571, 2.8816003546983104], 'pos': 4, 'auc': 0.9526828591835981}, 'purge': {'c_s3': 2.2705297295487377, 'c_ldr': 1.8499755476712862, 'c_tree': -3.2780325006230657, 'c_soft': -0.42544765755950215}}`

## Phases 2–4 — New Stage-B families

- **cat_stageB_bag9**: AUC=0.9523050 folds=[0.9515029737279702, 0.9524775272147795, 0.9569920587767426, 0.9527731351649043, 0.9480013460435626]
- **xgb_stageB_monotone_lm_bag6**: AUC=0.9523114 folds=[0.9515104294381841, 0.952515029344908, 0.9569994103687625, 0.9527550202953035, 0.9479641561852002]
- **lgb_stageB_extratrees_bag6**: AUC=0.9523231 folds=[0.9515007791349285, 0.9524954944932474, 0.956989346207075, 0.9527769554448657, 0.9479831379585093]
- **xgb_stageB_tail_specialist_bag6**: AUC=0.9523021 folds=[0.9514999072038403, 0.9524991252351221, 0.9570004870072387, 0.9527447432916661, 0.9479497839160167]
- **lgb_stageB_colsub_deep_bag6**: AUC=0.9523012 folds=[0.9514945823626494, 0.9524918963767813, 0.9569737046541469, 0.9527561248984153, 0.9479744891181089]
- **mlp_stageB_bag5**: AUC=0.9517065 folds=[0.9508032000710227, 0.9516403576823571, 0.9568451807418416, 0.9525807151670974, 0.9473609854031687]

## Phase 5 — Portfolio + staging

- on z_ref_old: `{'auc': 0.9526582063462632, 'delta_e5': [0.0, 0.0, -1.4350518911299126, 0.0, 0.0], 'pos': 0, 'w': {'xgb_stageB_68_bce': 0.0, 'xgb_stageB_68_focal_auc': 0.0, 'lgb_stageB_68_clean': 0.0, 'lgb_gpt2_3ep_fixed': 0.002, 'cat_stageB_bag9': 0.0, 'xgb_stageB_monotone_lm_bag6': 0.0, 'lgb_stageB_extratrees_bag6': 0.0, 'xgb_stageB_tail_specialist_bag6': 0.0, 'lgb_stageB_colsub_deep_bag6': 0.0, 'mlp_stageB_bag5': 0.0, 'd3_lin': 0.0, 'd3_bnd': 0.0}}`
- on z_v8_old: `{'auc': 0.9525922669018082, 'delta_e5': [0.0, 0.0, -1.447635977203099, 0.0, -1.2093151369851896], 'pos': 0, 'w': {'xgb_stageB_68_bce': 0.0, 'xgb_stageB_68_focal_auc': 0.0, 'lgb_stageB_68_clean': 0.0, 'lgb_gpt2_3ep_fixed': 0.004, 'cat_stageB_bag9': 0.0, 'xgb_stageB_monotone_lm_bag6': 0.0, 'lgb_stageB_extratrees_bag6': 0.0, 'xgb_stageB_tail_specialist_bag6': 0.0, 'lgb_stageB_colsub_deep_bag6': 0.0, 'mlp_stageB_bag5': 0.0, 'd3_lin': 0.0, 'd3_bnd': 0.0}}`

| slot | 1−ρ | cos(bad_S3) | cos(bad_ldr) | cos(bad_tree) | file |
|---|---:|---:|---:|---:|---|
| 1 | 0.250e-4 | +0.0000 | +0.0000 | +0.0000 | `sub_gm_v5_overnight_slot1_NOSUBMIT.csv` |
| 2 | 0.450e-4 | +0.0000 | -0.0000 | +0.0000 | `sub_gm_v5_overnight_slot2_NOSUBMIT.csv` |
| 3 | 0.800e-4 | -0.0000 | -0.0000 | +0.0000 | `sub_gm_v5_overnight_slot3_NOSUBMIT.csv` |
| 4 | 1.350e-4 | +0.0000 | -0.0000 | +0.0000 | `sub_gm_v5_overnight_slot4_NOSUBMIT.csv` |
| 5 | 2.000e-4 | +0.0000 | -0.0000 | -0.0000 | `sub_gm_v5_overnight_slot5_NOSUBMIT.csv` |

`NO_SUBMIT=True`.