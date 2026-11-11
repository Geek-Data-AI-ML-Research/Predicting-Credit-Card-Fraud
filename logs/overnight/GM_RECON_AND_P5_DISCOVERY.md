# GM Recon & P5 Discovery Report

**NO_SUBMIT=True**. Elapsed **378.9s**.
Base for probes: `z_ref_old + δ1_v4` OOF AUC≈`0.9526906` (target 0.9526906).
Champion held: Ladder S5 **0.95335**. No submissions made.

## Part 1 — Live Kaggle + unused residual probes

### High-signal kernels (scoreDescending)

- **HIGH_KERNEL>0.95347: areebbb/s6e11-three-ladders-two-steps-lb-0-95347                S6E11 | Three Ladders, Two Steps | LB 0.95347       Aree**
- **HIGH_KERNEL>0.95346: goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346       S6E11 LM Ladder Part 1 | Own LM Step | LB 0.95346   good**
- **HIGH_KERNEL>0.95340: amanatar/s6e11-podium-apex-public-lb-0-95340                    S6E11 | Podium Apex | Public LB 0.95340             Aman**

Top kernels (truncated):
```
areebbb/s6e11-three-ladders-two-steps-lb-0-95347                S6E11 | Three Ladders, Two Steps | LB 0.95347       Areeb Ali                         2026-11-11
goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346       S6E11 LM Ladder Part 1 | Own LM Step | LB 0.95346   goodpjw2008                       2026-11-11
saamhm/eda-blend-lb-0-95239-credit-card-fraud                   EDA | blend | LB-0.95239 | Credit Card Fraud        samsul hoque mondal               2026-11-11
malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion             Fraud Signal Lab | GPT-2 and GLM Fusion             Malyshev Danil                    2026-11-10
amanatar/s6e11-podium-apex-public-lb-0-95340                    S6E11 | Podium Apex | Public LB 0.95340             Aman Atar                         2026-11-10
mzishran/lb-0-95330-s6e11-ahh-shit-here-we-go-again             LB 0.95330 |S6E11 | Ahh Shit ! Here we GO AGAIN !   Mirza Ishrannnnn                  2026-11-09
alexchilton/credit-card-fraud-lb-0-9524-shallow-tree-stack      Credit Card Fraud | LB 0.9524 | Shallow-Tree Stack  alex chilton                      2026-11-10
tuboa2/0-95301-rank-logit-ensembling-guide                      [0.95301] Rank-Logit Ensembling Guide               tuboa                             2026-11-09
lucifer19/the-golden-signal-consensus-blend-v22                 🐈‍⬛ The Golden Signal | Consensus Blend V22         Agent Trainer                     2026-11-10
mzishran/s6e11-intro-to-playground-0-95300                      S6E11 | INTRO TO PLAYGROUND | 0.95300               Mirza Ishrannnnn                  2026-11-09
goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298          S6E11 Part 12 | Wide Stack + 2nd LM | LB 0.95298    goodpjw2008                       2026-11-08
amanatar/s6e11-sovereign-fusion-solution                        S6E11 Sovereign Fusion Solution                     Aman Atar                         2026-11-06
```

### Unused OOF probes on base (top by pos/lift)

| key | pos/5 | lift e-5 | mean Δe5 | ρ(base) | KS |
|---|---:|---:|---:|---:|---:|
| `/tmp/gm_recon/preds_bank/gm_v5_overnight__z_v5_harvest` | 3/5 | +0.24 | +0.35 | 0.9972 | 0.00007 |
| `/tmp/gm_recon/preds_bank/preds__P_super_v5_harvest` | 3/5 | +0.24 | +0.35 | 0.9972 | 0.00007 |
| `/tmp/gm_recon/preds_bank/goodpjw2008_s6e11-fraud-series-artifacts__realmlp_te` | 3/5 | -0.21 | -0.09 | 0.9538 | 0.00008 |
| `/tmp/gm_recon/preds_bank/goodpjw_artifacts__realmlp_te` | 3/5 | -0.21 | -0.09 | 0.9538 | 0.00008 |
| `/tmp/gm_recon/preds_bank/goodpjw_p6__realmlp_te` | 3/5 | -0.21 | -0.09 | 0.9538 | 0.00008 |
| `/tmp/gm_recon/preds_bank/goodpjw2008_s6e11-fraud-series-artifacts__realmlp_dig` | 3/5 | -0.21 | -0.18 | 0.9600 | 0.00018 |
| `/tmp/gm_recon/preds_bank/goodpjw_artifacts__realmlp_dig` | 3/5 | -0.21 | -0.18 | 0.9600 | 0.00018 |
| `/tmp/gm_recon/preds_bank/goodpjw_p6__realmlp_dig` | 3/5 | -0.21 | -0.18 | 0.9600 | 0.00018 |
| `/tmp/gm_recon/preds_bank/goodpjw2008_s6e11-fraud-series-artifacts__ebm_dig` | 3/5 | -0.35 | -0.38 | 0.9775 | 0.00020 |
| `/tmp/gm_recon/preds_bank/goodpjw_artifacts__ebm_dig` | 3/5 | -0.35 | -0.38 | 0.9775 | 0.00020 |
| `/tmp/gm_recon/preds_bank/goodpjw_p6__ebm_dig` | 3/5 | -0.35 | -0.38 | 0.9775 | 0.00020 |
| `/tmp/gm_recon/preds_bank/goodpjw2008_s6e11-fraud-series-artifacts__ebm_b4` | 3/5 | -0.38 | -0.31 | 0.9622 | 0.00005 |
| `/tmp/gm_recon/preds_bank/goodpjw_artifacts__ebm_b4` | 3/5 | -0.38 | -0.31 | 0.9622 | 0.00005 |
| `/tmp/gm_recon/preds_bank/goodpjw_p6__ebm_b4` | 3/5 | -0.38 | -0.31 | 0.9622 | 0.00005 |
| `/tmp/gm_recon/preds_bank/goodpjw_part7_artifacts__ebm_b4` | 3/5 | -0.38 | -0.31 | 0.9622 | 0.00005 |

## Part 2 — Quadratic vertex (v1=S5−S4, v2=u_A⊥v1)

| point | v1 | v2 | LB |
|---|---:|---:|---:|
| S1 | -51.514439 | -0.977659 | 0.95333 |
| S4 | -50.476949 | -0.000000 | 0.95334 |
| S5 | +0.000000 | +0.000000 | 0.95335 |
| A | -0.149574 | +7.466945 | 0.95334 |
| B | +6.954619 | -1.449618 | 0.95332 |

- **Vertex:** `(v1*, v2*) = (-23.275301, +3.056557)`
- Fit RMSE=`2.37e-15`

- peak damped BACK toward S4 (v1*<0): do not push past S5 along S5-S4
- peak on +u_A side (v2*>0)

## Part 3 — New P5 levers (individual, P4_pure5f recipe)

| lever | pos/5 | lift e-5 | Δe5 folds | disp | KS | w |
|---|---:|---:|---|---:|---:|---:|
| **P5_L5.1** | 4/5 | +0.22 | `[0.33, 0.71, 1.29, -1.71, 0.94]` | 1.0000 | 0.000028 | 0.0100 |
| **P5_L5.2** | 2/5 | -0.07 | `[-0.6, -0.29, -0.14, 1.09, 0.17]` | 1.0000 | 0.000111 | 0.0100 |
| **P5_L5.3** | 3/5 | +0.86 | `[0.5, -0.06, 2.63, -0.07, 0.6]` | 1.0000 | 0.000111 | -0.0100 |

### Notes

- Avoided unregularized GLM (Ridge α=1000 + depth-3 LGB residual on `calibrate_logit(z_ref)`).
- Test aligned via Method-B + exact order-stat map (KS gate 0.0002).
- Artifacts: `preds/gm_recon_p5/P5_L5.{1,2,3}_{oof,test}.npy`.

`NO_SUBMIT=True` — 7 Kaggle slots remain unused.
