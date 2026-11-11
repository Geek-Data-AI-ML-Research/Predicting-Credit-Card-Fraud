# GM DS Breakout Ladder → above 0.95347

Elapsed **582.3s**. Slots used **4/4** (reserve **3**).
Base escape: Ladder S5 `0.95335` → operate on areeb `0.95347` manifold.

## Step 1 — Native reproduction & inventory

- **goodpjw verify** `1-spearmanr(recon, pulled submission.csv) = 1.865e-14` (machine precision OK).
- **areeb transform:** `B=A+0.6(A-F)+0.12(LMcu-F)+0.06(L7-F); score=B-0.05(MLP-F)-0.025(GB-F)`
- LM: areeb uses **current** `lm_test.npy`; goodpjw uses **frozen** `megalm_test_v_nov08.npy`.
- L7: only **fold 0** present (`lm7_test_f0.npy` shape `(299984,4,2)`, `lm7_oof_f0.npy` `(139992,10,2)`). **No lm7_f1..f4.**

### Dataset inventory highlights

### compact-lm-seed7
- `README.md` size=1019
- `lm7_oof_f0.npy` shape=(139992, 10, 2) dtype=float32
- `lm7_test_f0.npy` shape=(299984, 4, 2) dtype=float32
- `megalm_test_v_nov08.npy` shape=(299984,) dtype=float32
### oof-bank
- `README.md` size=4341
- `folds.csv` size=6188538
- `manifest.csv` size=13602
- `oof_ebm_all_bag.npy` shape=(699960,) dtype=float32
- `oof_ebm_dig.npy` shape=(699960,) dtype=float32
- `oof_ebm_dig_bag.npy` shape=(699960,) dtype=float32
- `oof_ebm_dig_r7.npy` shape=(699960,) dtype=float32
- `oof_ebm_diglead_bag.npy` shape=(699960,) dtype=float32
- `oof_ebm_diglead_i25.npy` shape=(699960,) dtype=float32
- `oof_ebm_diglead_r11.npy` shape=(699960,) dtype=float32
- `oof_ebm_diglead_r13.npy` shape=(699960,) dtype=float32
- `oof_ebm_i25_bag.npy` shape=(699960,) dtype=float32
- `oof_ebm_i25_bag4.npy` shape=(699960,) dtype=float32
- `oof_ebm_lead4_bag.npy` shape=(699960,) dtype=float32
- `oof_gb_bag10.npy` shape=(699960,) dtype=float32
- `oof_gb_bag5.npy` shape=(699960,) dtype=float32
- `oof_gb_l3.npy` shape=(699960,) dtype=float32
- `oof_gb_ohk.npy` shape=(699960,) dtype=float32
- `oof_gbc_bag3.npy` shape=(699960,) dtype=float32
- `oof_gbe_bag3.npy` shape=(699960,) dtype=float32
- `oof_gbn_bag5_w128.npy` shape=(699960,) dtype=float32
- `oof_gbn_bag5_w512.npy` shape=(699960,) dtype=float32
- `oof_gbr_bag3.npy` shape=(699960,) dtype=float32
- `oof_gbx_bag3.npy` shape=(699960,) dtype=float32
- `oof_gbx_bag5.npy` shape=(699960,) dtype=float32
- `oof_glm_dig.npy` shape=(699960,) dtype=float32
- `oof_glm_dl.npy` shape=(699960,) dtype=float32
- `oof_glm_dl_bag.npy` shape=(699960,) dtype=float32
- `oof_glm_dl_s50.npy` shape=(699960,) dtype=float32
- `oof_glm_dl_s50_bag.npy` shape=(699960,) dtype=float32
- `oof_glm_ohk.npy` shape=(699960,) dtype=float32
- `oof_glm_ohk2.npy` shape=(699960,) dtype=float32
- `oof_glm_ohk2_bag.npy` shape=(699960,) dtype=float32
- `oof_glm_ohk_c01.npy` shape=(699960,) dtype=float32
- `oof_glm_ohkl.npy` shape=(699960,) dtype=float32
- `oof_glm_ohkl_bag.npy` shape=(699960,) dtype=float32
- `oof_glm_sb.npy` shape=(699960,) dtype=float32
- `oof_glm_te.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_l4.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_l7.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_s100.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_s100_bag.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_s100_bag5.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_s100_l3.npy` shape=(699960,) dtype=float32
- `oof_lgb_dig_s100_l3_bag.npy` shape=(699960,) dtype=float32
- `oof_lgb_diglead_bag.npy` shape=(699960,) dtype=float32
- `oof_lgb_diglead_bag5.npy` shape=(699960,) dtype=float32
- `oof_lgb_diglead_l2.npy` shape=(699960,) dtype=float32
- `oof_lgb_diglead_l3.npy` shape=(699960,) dtype=float32
- `oof_lgb_diglead_l3_bag.npy` shape=(699960,) dtype=float32
- `oof_lgb_diglead_l4.npy` shape=(699960,) dtype=float32
- `oof_lgb_dk_l4.npy` shape=(699960,) dtype=float32
- `oof_lgb_raw_l31.npy` shape=(699960,) dtype=float32
- `oof_lgb_raw_l4.npy` shape=(699960,) dtype=float32
- `oof_lgb_raw_l4_cnt.npy` shape=(699960,) dtype=float32
- `oof_lgb_rawdig_l4.npy` shape=(699960,) dtype=float32
- `oof_lgb_te_l4.npy` shape=(699960,) dtype=float32
- `oof_lgb_te_l4_s100.npy` shape=(699960,) dtype=float32
- `oof_lgb_te_l4_s300.npy` shape=(699960,) dtype=float32
- `oof_lgb_te_l4_s5.npy` shape=(699960,) dtype=float32
- `oof_realmlp_allcat.npy` shape=(699960,) dtype=float32
- `oof_realmlp_cnt.npy` shape=(699960,) dtype=float32
- `oof_realmlp_dig.npy` shape=(699960,) dtype=float32
- `oof_realmlp_dig_s100.npy` shape=(699960,) dtype=float32
- `oof_realmlp_dig_s2.npy` shape=(699960,) dtype=float32
- `oof_realmlp_rawdig.npy` shape=(699960,) dtype=float32
- `oof_realmlp_te.npy` shape=(699960,) dtype=float32
- `oof_realmlp_te_cnt.npy` shape=(699960,) dtype=float32
- `oof_realmlp_te_s2.npy` shape=(699960,) dtype=float32
- `oof_realmlp_v1.npy` shape=(699960,) dtype=float32
- `oof_xgb_dig_d2.npy` shape=(699960,) dtype=float32
- `oof_xgb_raw_d3.npy` shape=(699960,) dtype=float32
- `oof_xgb_te_d2.npy` shape=(699960,) dtype=float32
- `test_ebm_all_bag.npy` shape=(299984,) dtype=float32
- `test_ebm_dig.npy` shape=(299984,) dtype=float32
- `test_ebm_dig_bag.npy` shape=(299984,) dtype=float32
- `test_ebm_dig_r7.npy` shape=(299984,) dtype=float32
- `test_ebm_diglead_bag.npy` shape=(299984,) dtype=float32
- `test_ebm_diglead_i25.npy` shape=(299984,) dtype=float32
- `test_ebm_diglead_r11.npy` shape=(299984,) dtype=float32
- `test_ebm_diglead_r13.npy` shape=(299984,) dtype=float32
- `test_ebm_i25_bag.npy` shape=(299984,) dtype=float32
- `test_ebm_i25_bag4.npy` shape=(299984,) dtype=float32
- `test_ebm_lead4_bag.npy` shape=(299984,) dtype=float32
- `test_gb_bag10.npy` shape=(299984,) dtype=float32
- `test_gb_bag5.npy` shape=(299984,) dtype=float32
- `test_gb_l3.npy` shape=(299984,) dtype=float32
- `test_gb_ohk.npy` shape=(299984,) dtype=float32
- `test_gbc_bag3.npy` shape=(299984,) dtype=float32
- `test_gbe_bag3.npy` shape=(299984,) dtype=float32
- `test_gbn_bag5_w128.npy` shape=(299984,) dtype=float32
- `test_gbn_bag5_w512.npy` shape=(299984,) dtype=float32
- `test_gbr_bag3.npy` shape=(299984,) dtype=float32
- `test_gbx_bag3.npy` shape=(299984,) dtype=float32
- `test_gbx_bag5.npy` shape=(299984,) dtype=float32
- `test_glm_dig.npy` shape=(299984,) dtype=float32
- `test_glm_dl.npy` shape=(299984,) dtype=float32
- `test_glm_dl_bag.npy` shape=(299984,) dtype=float32
- `test_glm_dl_s50.npy` shape=(299984,) dtype=float32
- `test_glm_dl_s50_bag.npy` shape=(299984,) dtype=float32
- `test_glm_ohk.npy` shape=(299984,) dtype=float32
- `test_glm_ohk2.npy` shape=(299984,) dtype=float32
- `test_glm_ohk2_bag.npy` shape=(299984,) dtype=float32
- `test_glm_ohk_c01.npy` shape=(299984,) dtype=float32
- `test_glm_ohkl.npy` shape=(299984,) dtype=float32
- `test_glm_ohkl_bag.npy` shape=(299984,) dtype=float32
- `test_glm_sb.npy` shape=(299984,) dtype=float32
- `test_glm_te.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_l4.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_l7.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_s100.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_s100_bag.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_s100_bag5.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_s100_l3.npy` shape=(299984,) dtype=float32
- `test_lgb_dig_s100_l3_bag.npy` shape=(299984,) dtype=float32
- `test_lgb_diglead_bag.npy` shape=(299984,) dtype=float32
- `test_lgb_diglead_bag5.npy` shape=(299984,) dtype=float32
- `test_lgb_diglead_l2.npy` shape=(299984,) dtype=float32
- `test_lgb_diglead_l3.npy` shape=(299984,) dtype=float32
- `test_lgb_diglead_l3_bag.npy` shape=(299984,) dtype=float32
- `test_lgb_diglead_l4.npy` shape=(299984,) dtype=float32
- `test_lgb_dk_l4.npy` shape=(299984,) dtype=float32
- `test_lgb_raw_l31.npy` shape=(299984,) dtype=float32
- `test_lgb_raw_l4.npy` shape=(299984,) dtype=float32
- `test_lgb_raw_l4_cnt.npy` shape=(299984,) dtype=float32
- `test_lgb_rawdig_l4.npy` shape=(299984,) dtype=float32
- `test_lgb_te_l4.npy` shape=(299984,) dtype=float32
- `test_lgb_te_l4_s100.npy` shape=(299984,) dtype=float32
- `test_lgb_te_l4_s300.npy` shape=(299984,) dtype=float32
- `test_lgb_te_l4_s5.npy` shape=(299984,) dtype=float32
- `test_realmlp_allcat.npy` shape=(299984,) dtype=float32
- `test_realmlp_cnt.npy` shape=(299984,) dtype=float32
- `test_realmlp_dig.npy` shape=(299984,) dtype=float32
- `test_realmlp_dig_s100.npy` shape=(299984,) dtype=float32
- `test_realmlp_dig_s2.npy` shape=(299984,) dtype=float32
- `test_realmlp_rawdig.npy` shape=(299984,) dtype=float32
- `test_realmlp_te.npy` shape=(299984,) dtype=float32
- `test_realmlp_te_cnt.npy` shape=(299984,) dtype=float32
- `test_realmlp_te_s2.npy` shape=(299984,) dtype=float32
- `test_realmlp_v1.npy` shape=(299984,) dtype=float32
- `test_xgb_dig_d2.npy` shape=(299984,) dtype=float32
- `test_xgb_raw_d3.npy` shape=(299984,) dtype=float32
- `test_xgb_te_d2.npy` shape=(299984,) dtype=float32
### lucifer19/golden-signal-v5-scored-predictions
- `submission.csv` shape=(299984, 2) mean=0.016748
- explicit LB label in dataset: **not present** (submission.csv only + challenger_predictions.npz)
- `challenger_predictions.npz` keys=['oof', 'test'] oof=(699960,) test=(299984,)

## Step 2 — Orthogonal axes on z_areeb

- B6 cols=`6` (A,F,LM_curr,L7,MLP,GB)
- B_bad cols=`4`
- `u_gpt2m_lm` = unit(P⊥(0.70 gpt2m_5f_bayes + 0.15 bidir + 0.15 megalm_nov08))
- `u_stageB_5f` = unit(P⊥(δ1_v4_test + 0.20(u_zjoint+u_glm_graph)))
- `u_3ladder_grad` = unit(zrk(score_3ladder_plus) − z_areeb)

## Step 3 — Live LB adaptive ladder

| slot | file | branch/note | 1−ρ target | LB |
|---:|---|---|---:|---:|
| 1 | `sub_breakout_s1_areeb_gpt2m_3ladder_0p18e4.csv` | S1 | 1.80e-05 | **0.95346** |
| 2 | `sub_breakout_s2_branchC_damped_gpt2m_stageB_0p20e4.csv` | C | 2.00e-05 | **0.95347** |
| 3 | `sub_breakout_s3_empgrad_0p30e4.csv` |  | 3.00e-05 | **0.95347** |
| 4 | `sub_breakout_s4_areeb_micro_gpt2m_0p08e4.csv` | False | 8.00e-06 | **0.95346** |

**Best live LB this ladder: `0.95347`** (vs areeb `0.95347`, prior champ S5 `0.95335`).

STOP after Slot 4 — 3 daily submissions remain in reserve.

Artifacts: `/private/tmp/gm_breakout/preds/gm_breakout` · submissions under `submissions/sub_breakout_s*.csv`.
