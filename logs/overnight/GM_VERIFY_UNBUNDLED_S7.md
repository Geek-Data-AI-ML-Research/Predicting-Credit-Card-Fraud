# GM Verify Unbundled S7 (NO_SUBMIT)

**NO_SUBMIT=True**. Elapsed **137.0s**. Current ATH **`0.95352`** (Master S6). Slot 7 held.

## Step 1 — Live Kaggle recon

Public kernels scoreDescending still top out at areebbb **0.95347** / goodpjw **0.95346** — **no published kernel at 0.95369**. Leaderboard #1 (0.95369) is likely an unpublished private submit. Newest by dateCreated: samanyu compact-LM ladder variants, no >0.95347 artifact yet.

Top scoreDescending (truncated):
```
ref                                                             title                                               author                            lastRunTime                 totalVotes  
--------------------------------------------------------------  --------------------------------------------------  --------------------------------  --------------------------  ----------  
areebbb/s6e11-three-ladders-two-steps-lb-0-95347                S6E11 | Three Ladders, Two Steps | LB 0.95347       Areeb Ali                         2026-11-11 06:44:54.873000          29  
saamhm/eda-blend-lb-0-95239-credit-card-fraud                   EDA | blend | LB-0.95239 | Credit Card Fraud        samsul hoque mondal               2026-11-11 09:49:03.260000          51  
goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346       S6E11 LM Ladder Part 1 | Own LM Step | LB 0.95346   goodpjw2008                       2026-11-11 02:56:46.540000           7  
malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion             Fraud Signal Lab | GPT-2 and GLM Fusion             Malyshev Danil                    2026-11-10 16:13:04.133000          30  
amanatar/s6e11-podium-apex-public-lb-0-95340                    S6E11 | Podium Apex | Public LB 0.95340             Aman Atar                         2026-11-10 13:58:10.947000           2  
mzishran/lb-0-95330-s6e11-ahh-shit-here-we-go-again             LB 0.95330 |S6E11 | Ahh Shit ! Here we GO AGAIN !   Mirza Ishrannnnn                  2026-11-09 12:57:32.353000           4  
alexchilton/credit-card-fraud-lb-0-9524-shallow-tree-stack      Credit Card Fraud | LB 0.9524 | Shallow-Tree Stack  alex chilton                      2026-11-10 07:04:25.950000          49  
tuboa2/0-95301-rank-logit-ensembling-guide                      [0.95301] Rank-Logit Ensembling Guide               tuboa                             2026-11-09 10:43:40.097000           1  
```

Newest dateCreated (truncated):
```
ref                                                          title                                              author                lastRunTime                 totalVotes  
-----------------------------------------------------------  -------------------------------------------------  --------------------  --------------------------  ----------  
zhangyaqikaggle/notebook5ebb288e12                           notebook5ebb288e12                                 zhangyaqikaggle       2026-11-11 11:46:31.193000           0  
samanyu1808/s6e11-compact-lm-ladder-updated                  S6E11 | Compact LM Ladder Updated                  samanyu1808           2026-11-11 07:10:50.703000           0  
samanyu1808/s6e11-compact-lm-ladder-glm-boost                S6E11 | Compact LM Ladder + GLM-Boost              samanyu1808           2026-11-11 07:08:19.337000           0  
kazuhiroyogo/credit-kazuhiro-yogo-hte                        Credit_kazuhiro_yogo_HTE                           Kazuhiro_Yogo         2026-11-11 05:47:13.960000           0  
venkateshprasad107/predicting-credit-card-fraud-xg-boosting  predicting credit card fraud XG Boosting           Venkatesh_Prasad_107  2026-11-11 10:54:56.083000           0  
gauravduttaiiitb/ex13-s6e11-flaml-roc-auc                    Ex13 : S6E11 : FLAML : roc_auc                     Gaurav Dutta          2026-11-11 04:29:58.627000           1  
goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346    S6E11 LM Ladder Part 1 | Own LM Step | LB 0.95346  goodpjw2008           2026-11-11 02:56:46.540000           7  
keshigami/credit-card-fraud-eight-checks-before-any-model    Credit card fraud: eight checks before any model   Keshigami             2026-11-10 21:05:25.893000           0  
```

## Step 2 — Unbundled basis around z_areeb

| axis | meaning |
|---|---|
| a1 | +u_stageB (δ1_v4 + zjoint/glm) |
| a2 | +9bag Stage-B pillars ⊥ a1 |
| a3 | **−L7_orth** (subtract overweighted fold0 L7) |
| a4 | **+tree_sub** = −(MLP−F)−0.5(GB−F) continued |
| a5 | **−AF** damp |
| a6 | **−gpt2m** contrastive |
| a7 | +priv S5−St9 residual |

### Coordinates vs z_areeb (dot with semantic unit axes)

| point | LB | a1_stageB | a2_9bag | a3_-L7 | a4_tree_sub | a5_-AF | a6_-gpt2m | a7_privS5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goodpjw | 0.95346 | +0.1396 | +0.4009 | -2.2897 | -5.4149 | +2.7740 | +0.7223 | -0.0155 |
| areeb | 0.95347 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| S1 | 0.95346 | +0.1819 | -0.0117 | -1.0624 | +1.0778 | -0.2409 | -2.3227 | +0.1508 |
| S2 | 0.95347 | +2.0464 | +0.0254 | +1.7635 | -0.1331 | +0.2495 | -2.7614 | -0.1046 |
| S4 | 0.95346 | +0.1646 | +0.0049 | +0.4498 | +0.3230 | +0.0268 | -2.0695 | -0.0069 |
| S5 | 0.95350 | +3.5217 | +0.0237 | +1.3313 | -0.8705 | +0.2645 | +0.7826 | -0.0350 |
| S6 | 0.95352 | +6.4642 | -0.0019 | +0.6723 | -0.4738 | +0.2359 | +1.0264 | +0.5585 |

### Discovery check

- S5/S6 gain from **−gpt2m**, **−AF/LMnov08/L7 overshoot**, and **+stageB**.
- Mixing m2 in S6 restores **−MLP/−GB** (a4) that pure −u_3ladder had reversed.
- cos(u_unbundled, S5→S6) = **+0.7264**

## Step 3 — Gate #1 OOF

Base `z_ref_old` AUC=`0.9526612`.

| recipe | w | 5F Δe-5 | pos/5 | Fold0 Δe-5 | AUC |
|---|---:|---:|---:|---:|---:|
| z_ref | 0.000 | +0.00 | 0/5 | +0.00 | 0.9526612 |
| m1_equiv | 0.040 | +0.74 | 2/5 | +4.43 | 0.9526685 |
| S6_equiv | 0.050 | +1.81 | 3/5 | +6.97 | 0.9526792 |
| S7_unbundled | 0.050 | +0.02 | 3/5 | -0.73 | 0.9526613 |

| S7 cand (OOF proxy) | 5F Δe-5 | pos/5 | Fold0 Δe-5 |
|---|---:|---:|---:|
| S7_A | +0.57 | 2/5 | +2.53 |
| S7_B | -0.42 | 3/5 | -2.96 |
| S7_C | -2.11 | 4/5 | -11.38 |

## Step 4 — Gate #2 LB surface

- **OOS S6 check** (fit goodpjw..S5 → predict S6): pred=`0.953513` actual=`0.95352` err=`-0.74e-5`.
- Full-8 linear axis weights (a1..a7): `[1e-05, 0.00039, 1e-05, 2e-05, -1e-05, 1e-05, 1e-05]`

### Staged NOSUBMIT candidates from z_S6

| cand | file | 1−ρ(S6) | 1−ρ(areeb) | pred_lin | pred_quad | KS | 5F OOF Δ | Fold0 Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **S7_A** | `sub_master_s7A_pure_s5_s6_cont_1p05e4_NOSUBMIT.csv` | 1.00e-05 | 1.01e-04 | 0.953536 | 0.953534 | 0.000003 | +0.57 | +2.53 |
| **S7_B** | `sub_master_s7B_unbundled_opt_1p15e4_NOSUBMIT.csv` | 1.20e-05 | 1.04e-04 | 0.953532 | 0.953535 | 0.000003 | -0.42 | -2.96 |
| **S7_C** | `sub_master_s7C_unbundled_macro_1p65e4_NOSUBMIT.csv` | 2.50e-05 | 1.34e-04 | 0.953538 | 0.953541 | 0.000003 | -2.11 | -11.38 |

### Recommendation (still NO_SUBMIT)

**Gate verdict:**
- Gate #2 OOS S6 check **PASSED** (pred 0.953513 vs actual 0.95352, err −0.74e-5).
- Gate #1 OOF: **S7_A wins** (+0.57e-5 / +2.53 fold0). Unbundled S7_B/C **fail OOF** (−0.42 / −2.11) despite slightly higher surface preds — do **not** trust surface-alone for B/C.
- Coord evidence: S5→S6 is mostly **+a1 stageB** (+3.52→+6.46) and **+a7 privS5** (−0.04→+0.56), with **+a6 (−gpt2m)** already on; a4 tree_sub went more negative (bad for “continue −MLP”).

**If/when submitting the last slot:** prefer **`S7_A`** (pure S5→S6 continuation @ 0.10e-4 from S6). Hold **S7_B/C** unless a revised unbundle fixes Fold0 L7/OOF.

`NO_SUBMIT=True` — 1 daily slot remains unused.
