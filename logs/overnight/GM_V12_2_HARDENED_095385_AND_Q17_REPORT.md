# GM V12.2 Hardened: 0.95385 Recon + Q17 Anchor Inversion

**UTC:** 2026-11-12T20:11:29.583941+00:00  
**`NO_SUBMIT=True`** — 0 uploads  
**POD TOUCHES: 0** — no `runpodctl`, no SSH, no GPU pods  
**Elapsed:** 180.54s

## Step 1 — Kaggle recon (`0.95385+` surge hunt)

### Competition kernels by `dateRun` (top)
```
ref                                                               title                                               author                        lastRunTime                 totalVotes  
----------------------------------------------------------------  --------------------------------------------------  ----------------------------  --------------------------  ----------  
markbarney/s6e11-what-we-tried-and-what-did-not-work              S6E11 | What We Tried and What Did Not Work         Mark Barney                   2026-11-12 19:26:19.613000           0  
modansaif/s6e11-fraud-fusion-reproduction-credited                S6E11 Fraud Fusion Reproduction (credited)          SAIF MODAN                    2026-11-12 18:36:26.740000           0  
luthfimuthathohirin/credit-card-fraud-ensemble-forensics          Credit Card Fraud Ensemble Forensics                Luthfi Muthathohirin          2026-11-12 18:09:43.623000           1  
evgendvorkin/s6e11-deep-eda-artifacts-xgb-oof-0-950285            S6E11 | Deep EDA & Artifacts | XGB OOF 0.950285     Дворкин Евгений Владимирович  2026-11-12 17:55:03.580000          60  
bawanthabeliwaththa/the-honest-stack-cv-lb-and-the-0-9533-trap    The Honest Stack: CV, LB, and the 0.9533 Trap       Bawantha Beliwaththa          2026-11-12 17:38:52.200000           0  
logeshwari12345/predicting-credit-card-fraud                      Predicting Credit Card Fraud                        Logeshwari Somasundaram       2026-11-12 14:56:42.633000           4  
asteriospanoras/s6e11-how-much-of-the-public-top-is-real          S6E11 | How Much of the Public Top Is Real?         Asterios Panoras              2026-11-12 14:46:56.213000           0  
saamhm/eda-blend-lb-0-95239-credit-card-fraud                     EDA | blend | LB-0.95239 | Credit Card Fraud        samsul hoque mondal           2026-11-12 14:28:49.137000          53  
knightfork/s6e11-top-15-cpu-blend-eda-lb-0-95357                  S6E11 | Top 15 CPU Blend + EDA | LB 0.95357         Knight Fork                   2026-11-12 13:52:53.687000           0  
sriram00777/s6e11-top-15-cpu-blend-eda-lb-0-95357                 S6E11 | Top 15 CPU Blend + EDA | LB 0.95357         Sriram S                      2026-11-12 13:47:10.680000           2  
saqibishtiaq699/complete-code-lightgbm-with-smote                 Complete Code | LightGBM with SMOTE                 Saqib Ishtiaq                 2026-11-12 13:16:04.540000           6  
malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion               Fraud Signal Lab | GPT-2 and GLM Fusion             Malyshev Danil                2026-11-12 13:09:05.370000          30  
sameershaikh27/s6e11-fraud-public-submission-fusion-no-oof        S6E11 Fraud: Public Submission Fusion (No OOF)      Sameer Shaikh                 2026-11-12 13:01:40.447000           2  
daoviet/s6e11-on-the-journey-to-your-first-top-100                S6E11 | On the journey to your first top 100 🚀      daoviet                       2026-11-12 12:43:28.657000           7  
deepakkumbhar16/beginners-friendly-blending-gbdt-lb-0-95071       📚Beginners-Friendly Blending GBDT | LB - 0.95071    Deepak Kumbhar                2026-11-12 12:34:08.843000           2  
areebbb/s6e11-three-ladders-two-seeds-lb-0-95354                  S6E11 | Three Ladders, Two Seeds | LB 0.95354       Areeb Ali                     2026-11-12 10:11:25.853000          47  
rohansardar/s6e11-predicting-card-fraud-xgb-5-fold-cv             S6E11 | Predicting Card Fraud | XGB | 5-fold CV     Rohan Sardar                  2026-11-12 09:32:05.957000           4  
goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354            S6E11 LM Ladder | Two GPT-2 Seeds | LB 0.95354      goodpjw2008                   2026-11-12 09:17:28.553000          19  
nagendragouda125/credit-card-fraud-detection                      credit card fraud detection                         Nagendra Gouda                2026-11-12 09:03:18.263000           4  
prafultiwari1/predicting-credit-card-fraud                        predicting credit card fraud                        Praful Tiwari1                2026-11-12 08:53:02.667000           0  
```

### Competition kernels by `scoreDescending` (top)
```
ref                                                             title                                               author                            lastRunTime                 totalVotes  
--------------------------------------------------------------  --------------------------------------------------  --------------------------------  --------------------------  ----------  
areebbb/s6e11-three-ladders-two-seeds-lb-0-95354                S6E11 | Three Ladders, Two Seeds | LB 0.95354       Areeb Ali                         2026-11-12 10:11:25.853000          47  
malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion             Fraud Signal Lab | GPT-2 and GLM Fusion             Malyshev Danil                    2026-11-12 13:09:05.370000          30  
goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354          S6E11 LM Ladder | Two GPT-2 Seeds | LB 0.95354      goodpjw2008                       2026-11-12 09:17:28.553000          19  
asteriospanoras/s6e11-how-much-of-the-public-top-is-real        S6E11 | How Much of the Public Top Is Real?         Asterios Panoras                  2026-11-12 14:46:56.213000           0  
modansaif/s6e11-fraud-fusion-reproduction-credited              S6E11 Fraud Fusion Reproduction (credited)          SAIF MODAN                        2026-11-12 18:36:26.740000           0  
saamhm/eda-blend-lb-0-95239-credit-card-fraud                   EDA | blend | LB-0.95239 | Credit Card Fraud        samsul hoque mondal               2026-11-12 14:28:49.137000          53  
daoviet/s6e11-on-the-journey-to-your-first-top-100              S6E11 | On the journey to your first top 100 🚀      daoviet                           2026-11-12 12:43:28.657000           7  
amanatar/s6e11-podium-apex-public-lb-0-95340                    S6E11 | Podium Apex | Public LB 0.95340             Aman Atar                         2026-11-10 13:58:10.947000           3  
mzishran/lb-0-95330-s6e11-ahh-shit-here-we-go-again             LB 0.95330 |S6E11 | Ahh Shit ! Here we GO AGAIN !   Mirza Ishran                      2026-11-09 12:57:32.353000           7  
alexchilton/credit-card-fraud-lb-0-9530-shallow-tree-stack      Credit Card Fraud | LB 0.9530 | Shallow-Tree Stack  alex chilton                      2026-11-12 07:03:20.347000          52  
tuboa2/0-95301-rank-logit-ensembling-guide                      [0.95301] Rank-Logit Ensembling Guide               tuboa                             2026-11-09 10:43:40.097000           2  
lucifer19/the-golden-signal-consensus-blend-v22                 🐈‍⬛ The Golden Signal | Consensus Blend V22         Adaptive Agent Systems            2026-11-11 19:41:59.390000          23  
mzishran/s6e11-intro-to-playground-0-95300                      S6E11 | INTRO TO PLAYGROUND | 0.95300               Mirza Ishran                      2026-11-09 08:33:56.543000           7  
goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298          S6E11 Part 12 | Wide Stack + 2nd LM | LB 0.95298    goodpjw2008                       2026-11-08 00:06:34.660000          53  
amanatar/s6e11-sovereign-fusion-solution                        S6E11 Sovereign Fusion Solution                     Aman Atar                         2026-11-06 20:14:33.773000          12  
nina2025/ps-s6e11-eos-7                                         PS-s6e11 | EoS (7)                                  F.A.Nina                          2026-11-06 18:12:55.297000          19  
```

### Dataset updates (`s6e11`)
```
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
amanatar/s6e11-rank1-q17-meta-bundle                    S6E11 Rank #1 Q17 Meta Bundle                      64592945  2026-11-12 14:04:45.277000             16          1  0.125            
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal Apex                  11617844  2026-11-12 12:40:02.780000             62          0  0.29411766       
goodpjw2008/s6e11-target-first-gpt2-seed-2026           S6E11 Target-First GPT-2 Seed 2026                  7465705  2026-11-12 09:14:44.553000             63          1  0.375            
alexchilton/s6e11-oof-predictions                       S6E11 OOF + test predictions (15 models)          229776751  2026-11-12 08:22:15.010000           1209          0  0.5              
goodpjw2008/s6e11-compact-lm-seed7                      S6E11 Compact Target-First LM Seed 7               29140672  2026-11-12 08:04:21.997000            240          0  0.375            
amanatar/s6e11-sovereign-oof-test-and-orthogonal-vault  S6E11 Sovereign OOF+Test & Orthogonal Vault        65342999  2026-11-11 02:54:55.807000            276          0  0.64705884       
aphoristicartist/s6e11-detective-weights                s6e11-detective-weights                           462714314  2026-11-10 23:11:38.847000             47          0  0.375            
keshigami/model-vs-one-line-rule-six-kaggle-starters    Model vs one-line rule: 6 Kaggle starters              1675  2026-11-10 15:56:19.047000             12          0  1                
aphoristicartist/s6e11-detective-data                   s6e11-detective-data                                 665173  2026-11-10 15:55:57.537000             31          0  0.23529412       
dariushafshar/s6e11-golem-oof-library                   S6E11 OOF library - 9 models, frozen folds         76962207  2026-11-10 15:38:57.440000            165          0  0.9411765        
amanatar/s6e11-strike10-podium-apex-095340              S6E11 Strike 10 Podium Apex LB 0.95340              9104774  2026-11-10 13:22:20.547000            110          0  0.3529412        
rajibbahar/ps-s6e11-experiment-results                  PS S6E11 Fraud Experiment Results                      1814  2026-11-09 21:37:53.863000             27          0  0.8235294        
sometimessubodh/ps6e11-dataset                          PS6E11 38 Features and Leak-Safe OOF/Test Preds   131372929  2026-11-09 11:51:13.290000            354          4  1                
sk20220623/s6e11-oof-predictions                        S6E11 OOF Predictions Library                     173095936  2026-11-09 11:43:36.793000            107          1  0.1875           
```

### G26 / Q17 remote files
```
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

```
```
name                       size  creationDate                
---------------------  --------  --------------------------  
rank1_q17_streams.npz  64574677  2026-11-12 14:04:47.476000  

```

**0.9538x / 0.9537x kernel search note:** bare `kernels list -s 0.9538` hits unrelated competitions; comp-scoped `scoreDescending` still tops at public **0.95354** (Areeb / goodpjw). No verified public **0.95385** kernel artifact found via CLI this turn — surge may be LB-only / private.

### Pulled kernels
- `knightfork/s6e11-top-15-cpu-blend-eda-lb-0-95357` → already present: /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/knightfork_095357
- `modansaif/s6e11-fraud-fusion-reproduction-credited` → already present: /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/modansaif
- `asteriospanoras/s6e11-how-much-of-the-public-top-is-real` → already present: /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/asterios
- `bawanthabeliwaththa/the-honest-stack-cv-lb-and-the-0-9533-trap` → already present: /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/bawanthabeliwaththa
- `markbarney/s6e11-what-we-tried-and-what-did-not-work` → already present: /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/markbarney
- `luthfimuthathohirin/credit-card-fraud-ensemble-forensics` → already present: /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/luthfi_forensics

### Exact formulas
- **knightfork LB 0.95357:** `L(preserved_095354_ladder) + 0.25 * (L(sovereign) - L(wide))` · WEIGHT=0.25
  - Inputs: goodpjw2008 LM Ladder 0.95354 reference CSV, amanatar Sovereign OOF+Test vault (sovereign vs wide difference)
  - Proxy OOF Δ=1.7580980577e-05 (4/5 folds up)
- **modansaif:** Fraud-fusion reproduction credited — rebuilds public Three-Ladders / LM-ladder recipe (A/F/LM/L7F/GPD/MS/G26 style members). Proxy: `public ladder reconstruction (Areeb/goodpjw lineage), not a new 0.9538x formula`
- **bawanthabeliwaththa:** Public 0.9533 band can be CV-fragile / winner's-curse; honesty about CV vs LB gap.

## Step 2 — `rank1_q17_streams.npz` reverse engineering

### Prior audit executive excerpt
```
# GM Track-2 Audit: amanatar Q17 + detective + G26
**UTC:** 2026-11-12T16:32:36.735189+00:00  
**`NO_SUBMIT=True`** — 0 uploads  
**Elapsed:** 516.57s

## Executive verdict
- **`anchor_095370` cleared the dual LOFO gate** (S7m **5/5**, simple **4/5**, `d0>0`). Orth residual meanΔ ≈ **+0.035e-5** on S7m. **All 16 Q17 member streams failed** the dual gate (several hit S7m 5/5 alone — `gbn_bag5_w512`, `gb_bag5` — but simple ≤3/5).
- **HOLD all 4 submits** (`NO_SUBMIT=True`): anchor is already the public 0.95370 apex; do **not** burn a cand2↔anchor blend without a dedicated promotion plan.
- **F2−F4 > 0.004 on every stream** including known-clean public OOFs → structural fold difficulty, **not** cross-fold leak.
- **G26 live `test_5fold`:** **ABSENT** (still oof_5fold + test_f0 only).
- **Q17 `test_gpt2_seed2026` ≡ G26 `test_f0`** (`1−ρ = 0`) — not a full 5-fold test tensor.
- **Detective:** `model.pt` (~475MB) + `config.json`; data = orig20k CSV only — no OOF/test bank.

## Live dataset file lists
### amanatar/s6e11-rank1-q17-meta-bundle
```
name                       size  creationDate                
---------------------  --------  --------------------------  
rank1_q17_streams.npz  64574677  2026-11-12 14:04:47.476000
```
### aphoristicartist/s6e11-detective-weights
```
```

**NPZ keys (38):** `['y_train', 'folds', 'oof_anchor_095370', 'test_anchor_095370', 'test_anchor_prob', 'stream_names', 'oof_megayak_5f_lm', 'test_megayak_5f_lm', 'oof_masaya_gpt2', 'test_masaya_gpt2', 'oof_lm7_5fold', 'test_lm7_5fold', 'oof_gpt2_seed2026', 'test_gpt2_seed2026', 'oof_tabpfn_slimypunk_tp1', 'test_tabpfn_slimypunk_tp1', 'oof_sadam_tabpfn_100k_tp3', 'test_sadam_tabpfn_100k_tp3', 'oof_yekenot_realmlp', 'test_yekenot_realmlp', 'oof_alex_glm_gen', 'test_alex_glm_gen', 'oof_daoviet_gpdvae', 'test_daoviet_gpdvae', 'oof_gbn_bag5_w512', 'test_gbn_bag5_w512', 'oof_sameer_glm10f', 'test_sameer_glm10f', 'oof_sadam_v6_gb_xgb', 'test_sadam_v6_gb_xgb', 'oof_sadam_v6_gb_lgb', 'test_sadam_v6_gb_lgb', 'oof_sadam_v4_cat', 'test_sadam_v4_cat', 'oof_gb_bag5', 'test_gb_bag5', 'oof_megayak_shallow_tree', 'test_megayak_shallow_tree']`

**stream_names (16):** `['megayak_5f_lm', 'masaya_gpt2', 'lm7_5fold', 'gpt2_seed2026', 'tabpfn_slimypunk_tp1', 'sadam_tabpfn_100k_tp3', 'yekenot_realmlp', 'alex_glm_gen', 'daoviet_gpdvae', 'gbn_bag5_w512', 'sameer_glm10f', 'sadam_v6_gb_xgb', 'sadam_v6_gb_lgb', 'sadam_v4_cat', 'gb_bag5', 'megayak_shallow_tree']`

| stream | solo AUC | F0 | F1 | F2 | F3 | F4 | oof shape | test shape |
|---|---:|---:|---:|---:|---:|---:|---|---|
| anchor_095370 | 0.9525514 | 0.9517749 | 0.9525318 | 0.9572653 | 0.9528037 | 0.9483810 | [699960] | [299984] |
| megayak_5f_lm | 0.9504517 | 0.9485056 | 0.9506818 | 0.9560850 | 0.9506043 | 0.9463809 | [699960] | [299984] |
| masaya_gpt2 | 0.9500480 | 0.9488823 | 0.9503172 | 0.9553120 | 0.9498742 | 0.9458540 | [699960] | [299984] |
| lm7_5fold | 0.9494440 | 0.9475445 | 0.9497671 | 0.9551662 | 0.9490652 | 0.9456760 | [699960] | [299984] |
| gpt2_seed2026 | 0.9501731 | 0.9491911 | 0.9503701 | 0.9552357 | 0.9502097 | 0.9458585 | [699960] | [299984] |
| tabpfn_slimypunk_tp1 | 0.9493429 | 0.9481552 | 0.9503329 | 0.9547382 | 0.9491845 | 0.9443034 | [699960] | [299984] |
| sadam_tabpfn_100k_tp3 | 0.9488896 | 0.9475724 | 0.9508788 | 0.9529789 | 0.9490339 | 0.9439834 | [699960] | [299984] |
| yekenot_realmlp | 0.9501145 | 0.9500496 | 0.9510404 | 0.9549457 | 0.9503984 | 0.9441386 | [699960] | [299984] |
| alex_glm_gen | 0.9505035 | 0.9494181 | 0.9506593 | 0.9555146 | 0.9514767 | 0.9454483 | [699960] | [299984] |
| daoviet_gpdvae | 0.9465294 | 0.9447321 | 0.9476896 | 0.9514630 | 0.9466615 | 0.9420998 | [699960] | [299984] |
| gbn_bag5_w512 | 0.9512429 | 0.9499157 | 0.9516058 | 0.9560761 | 0.9518553 | 0.9467612 | [699960] | [299984] |
| sameer_glm10f | 0.9516046 | 0.9508096 | 0.9520959 | 0.9563718 | 0.9520656 | 0.9466798 | [699960] | [299984] |
| sadam_v6_gb_xgb | 0.9507703 | 0.9506578 | 0.9504680 | 0.9558513 | 0.9510173 | 0.9458569 | [699960] | [299984] |
| sadam_v6_gb_lgb | 0.9509402 | 0.9506461 | 0.9506616 | 0.9560987 | 0.9513073 | 0.9459875 | [699960] | [299984] |
| sadam_v4_cat | 0.9501135 | 0.9491888 | 0.9505731 | 0.9550723 | 0.9510023 | 0.9447305 | [699960] | [299984] |
| gb_bag5 | 0.9515369 | 0.9504322 | 0.9515890 | 0.9562735 | 0.9523948 | 0.9469945 | [699960] | [299984] |
| megayak_shallow_tree | 0.9515043 | 0.9503825 | 0.9518021 | 0.9562093 | 0.9525071 | 0.9466198 | [699960] | [299984] |

### Possibility A vs B (16 member streams → anchor)
- Ridge R² OOF=0.994034, Test=0.982663
- max|w_oof−w_test| Ridge=**9.116272e-01** OLS=**9.141872e-01** L-space=**8.567963e-01**
- Identical@1e-4? **False** · transfer corr=**0.989263**
- **Verdict: Possibility B: OOF/test constructed differently (weights diverge)**

Top |Δw|:
- `megayak_shallow_tree`: w_oof=+0.5745, w_test=-0.3371, |Δ|=0.9116
- `gb_bag5`: w_oof=+0.3435, w_test=+1.1050, |Δ|=0.7614
- `sadam_v6_gb_xgb`: w_oof=+0.5859, w_test=+1.2160, |Δ|=0.6301
- `megayak_5f_lm`: w_oof=+0.4063, w_test=+0.8609, |Δ|=0.4546
- `sameer_glm10f`: w_oof=+0.3658, w_test=-0.0657, |Δ|=0.4315
- `sadam_v6_gb_lgb`: w_oof=-0.1337, w_test=-0.5405, |Δ|=0.4068
- `masaya_gpt2`: w_oof=+0.4427, w_test=+0.1026, |Δ|=0.3400
- `tabpfn_slimypunk_tp1`: w_oof=-0.0053, w_test=+0.2103, |Δ|=0.2156

### Ladder-bank regression on `test_anchor_095370`
- L-space R²=**0.994104** · raw R²=**0.994104**
| member | w_L | w_raw |
|---|---:|---:|
| A | +1.4988 | +1.4988 |
| F | +0.3917 | +0.3917 |
| LM | +0.2525 | +0.2525 |
| L7F | +0.2504 | +0.2504 |
| GPD | +0.0515 | +0.0515 |
| G26 | +0.1193 | +0.1193 |
| MS | -0.3426 | -0.3426 |
| MLP | -0.0842 | -0.0842 |
| GB | -0.3082 | -0.3082 |
| L7_f0_K4 | -0.0311 | -0.0311 |

### Proprietary residual loadings into `test_anchor`
| residual | corr(L_anchor) | corr(resid_after_ladder) |
|---|---:|---:|
| d1_v4 | +0.000882 | +0.150126 |
| G26_minus_MS | -0.005616 | +0.000000 |
| Strike9p5 | +0.995596 | +0.000537 |
| L7_s2026_K10_minus_L7_K4 | -0.014469 | -0.042827 |

If |corr(resid)|≪1 and prop_in_anchor weights tiny → **anchor has near-zero weight on our private δ1 / L7_s2026** (room for inverted macro-leap).

OOF prop-in-anchor ridge: `{'w_d1': -0.05097036519860978, 'w_g26ms': -0.001283033692295512, 'w_l7_s2026_f0only': -0.028033085131395057, 'r2': 0.0033891585190568563}`

### `1−ρ` of `L(test_anchor_095370)` vs bank
| ref | 1−ρ |
|---|---:|
| A_095340 | 0.00165884 |
| F_095330 | 0.00179762 |
| z_S7_ours | 0.00104637 |
| cand2 | 0.0009896 |
| areeb_095347 | 0.00115785 |
| s354 | 0.0010722 |
| knightfork_095357 | N/A (formula-only) |

## Step 3 — De-aliased subfold + inverted-anchor LOFO

### Pure `L7_s2026_K10 − L7_s7_K4` on Fold0 (5-subfold, β=0.036)
- **4/5** · meanΔ=**0.0264e-5** · `[sf]=[-0.031, 0.1087, 0.0156, 0.0078, 0.0311]`

### Injections on top of `oof_anchor_095370` (fixed w=0.036 + adaptive)
| move | fixed pos/5 | fixed meanΔe5 | adaptive pos/5 | adaptive meanΔe5 |
|---|---:|---:|---:|---:|
| plus_d1_v4_w0.036 | 4/5 | 0.0031 | 4/5 | 0.0090 |
| plus_G26_minus_MS_w0.036 | 2/5 | -0.0008 | 4/5 | 0.0100 |
| plus_L7_s2026_K10_f0_w0.036 | 1/5 | 0.0001 | 0/5 | -0.0010 |
| plus_d1_and_L7_w0.036 | 4/5 | 0.0020 | 3/5 | 0.0100 |
| plus_d1_G26MS_L7_w0.036 | 3/5 | 0.0031 | 4/5 | 0.0090 |

## Step 4 — Staged NOSUBMIT candidates (4 slots)

- **Slot 1** `sub_v12p2_slot1_cand2_plus_s2026K10_splitcancel_0p10e4_NOSUBMIT.csv` — {'slot': 1, 'w': 2.3592915758490562, 'dist_e4': 0.1049999983948613, 'ref': 'cand2'}
- **Slot 2** `sub_v12p2_slot2_cand2_plus_s2026K10_plus_Q17orth_0p20e4_NOSUBMIT.csv` — {'slot': 2, 'w': 4.128386154770851, 'dist_e4': 0.19999999761877163, 'ref': 'cand2'}
- **Slot 3** `sub_v12p2_slot3_anchor095370_plus_our_StageB_and_s2026_0p11e4_NOSUBMIT.csv` — {'slot': 3, 'w': 2.5797586143016815, 'dist_e4': 0.11199999958022566, 'ref': 'anchor_095370'}
- **Slot 4** `sub_v12p2_slot4_midpoint_cand2_s2026_x_anchor095370_NOSUBMIT.csv` — {'slot': 4, 'alpha_anchor_family': 0.6499999999999999, 'dist_cand2_e4': 4.217739657551567, 'dist_anchor_e4': 1.342809029106773, 'oof_mean_auc': 0.9525883399463781, 'ref': 'midpoint'}

**STRICT NO_SUBMIT=True** — files staged for human inspection only. Do not auto-upload. HOLD until promotion law + time-to-reset decision.

---
*Generated by `scripts/gm_v12_2_hardened_095385_recon_and_anchor_inversion.py` — zero pod ops.*
