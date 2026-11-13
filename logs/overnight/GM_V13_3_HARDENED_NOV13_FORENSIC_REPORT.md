# GM V13.3 HARDENED — Nov13 Forensic Novelty / Symmetry / Double-Count Gate

**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed 330.0s

**Script:** `scripts/gm_v13_3_hardened_nov13_forensic_and_doublecount_gate.py`

## Step 1a — `daoviet` refresh (05:29 UTC)

**Verdict:** GPD oof/test BYTE-IDENTICAL to prior pull; refresh only added submission CSVs

| file | shape | in prior | 1−ρ | max\|Δ\| | identical |
|---|---|---|---:|---:|---|
| `oof_ours.npy` | [699960] | True | 0.0 | 0.0 | True |
| `submission_095351.csv` | None | True | None | None | None |
| `submission_095354_exact.csv` | None | False | None | None | None |
| `submission_summit_095370.csv` | None | False | None | None | None |
| `submission_verified_095370.csv` | None | False | None | None | None |
| `test_ours.npy` | [299984] | True | 2.220446049250313e-16 | 0.0 | True |

Added vs prior pull: `['submission_095354_exact.csv', 'submission_summit_095370.csv', 'submission_verified_095370.csv']`

→ **GPD stream itself was NOT updated** — only podium submission CSVs (`095354_exact`, `summit_095370`, `verified_095370`) were added. No new GPD signal for ATH inject.

## Step 1b — `alexchilton` (08:51 UTC) forensic A/B/C

- vs prior pull: added `[]` · removed `[]` · sha-changed `[]` · manifest entries=38

| tag | AUC | R²oof | R²test | disp | max\|Δw\| | G26 gap | R²(V\|LIVE) | flags | survive |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `4188470eded4cdec` | 0.95066 | 0.9914 | 0.9917 | 1.000 | 0.146 | 0.002 | 0.961 | — | True |
| `82e9a79659bdf9d8` | 0.95054 | 0.9897 | 0.9876 | 1.000 | 0.070 | 0.002 | 0.963 | — | True |
| `21b4d1316c55b429` | 0.95052 | 0.9916 | 0.9912 | 1.000 | 0.162 | 0.004 | 0.961 | — | True |
| `00107284da228571` | 0.95042 | 0.9900 | 0.9904 | 1.000 | 0.061 | 0.001 | 0.962 | — | True |
| `ff71516344a5cf6a` | 0.95039 | 0.9898 | 0.9893 | 1.000 | 0.171 | 0.004 | 0.959 | — | True |
| `55b2e2d2f9b615b9` | 0.95036 | 0.9889 | 0.9899 | 1.000 | 0.175 | 0.004 | 0.960 | — | True |
| `29b8383a5886512b` | 0.95011 | 0.9894 | 0.9905 | 1.000 | 0.023 | 0.001 | 0.953 | — | True |
| `9d26cfb564230caa` | 0.94951 | 0.9780 | 0.9838 | 1.000 | 0.143 | 0.005 | 0.956 | — | True |
| `37984c74f5880f09` | 0.94951 | 0.9791 | 0.9839 | 1.000 | 0.163 | 0.005 | 0.955 | — | True |
| `59abebeed6b8fa2d` | 0.94951 | 0.9796 | 0.9848 | 1.000 | 0.180 | 0.005 | 0.955 | — | True |
| `47ec80c2946cc751` | 0.94936 | 0.9769 | 0.9827 | 1.000 | 0.163 | 0.005 | 0.953 | — | True |
| `581c04132b5940af` | 0.94934 | 0.9765 | 0.9828 | 1.000 | 0.184 | 0.006 | 0.956 | — | True |
| `e0f351ee76b3a71e` | 0.94916 | 0.9725 | 0.9753 | 1.000 | 0.114 | 0.002 | 0.934 | — | True |
| `d5bd0671dd68108f` | 0.94868 | 0.9671 | 0.9761 | 1.000 | 0.093 | 0.005 | 0.945 | — | True |
| `0958aef8e20ba93c` | 0.94855 | 0.9704 | 0.9783 | 1.000 | 0.090 | 0.004 | 0.939 | — | True |
| `b890168ab4ab0fdc` | 0.94845 | 0.9698 | 0.9780 | 1.000 | 0.073 | 0.004 | 0.941 | — | True |
| `ddfa69a32874765c` | 0.94844 | 0.9741 | 0.9806 | 1.000 | 0.042 | 0.003 | 0.935 | — | True |
| `91652b3e9b245296` | 0.94839 | 0.9753 | 0.9812 | 1.000 | 0.037 | 0.003 | 0.936 | — | True |
| `278d2f3f2c3631c2` | 0.94838 | 0.9685 | 0.9769 | 1.000 | 0.068 | 0.004 | 0.943 | — | True |
| `a9fffc89ca5f057f` | 0.94837 | 0.9692 | 0.9776 | 1.000 | 0.080 | 0.004 | 0.938 | — | True |

**daoviet GPD forensic:** R²oof=1.0000 R²test=1.0000 flags=['REPACKAGED_BLEND'] survive=False

**Step1 survivors:** 20

## Step 2 — LOFO orth + 4-bar gate

Bars: (1) main 5/5 & min≥+0.10e-5 · (2) ≥18/25 sf, t≥2.80, Wilcoxon p<0.005 · (3) boot90 lo>+0.25e-5 · (4) Step1 B&C clean

| tag | mode | main | min | sf25 | t | W-p | boot90 | bars | PASS |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `4188470eded4cd` | `orth_lofo_w0.04` | 3/5 | -12.49 | 14/25 | 1.18 | 0.105 | [-3.037, 9.672] | 0001 | False |
| `82e9a79659bdf9` | `orth_lofo_w0.04` | 3/5 | -15.50 | 14/25 | 0.83 | 0.1763 | [-4.591, 9.603] | 0001 | False |
| `21b4d1316c55b4` | `orth_lofo_w0.04` | 3/5 | -12.13 | 17/25 | 1.69 | 0.04516 | [-2.962, 11.071] | 0001 | False |
| `00107284da2285` | `orth_lofo_w0.02` | 3/5 | -5.51 | 15/25 | 0.46 | 0.2891 | [-1.948, 3.751] | 0001 | False |
| `ff71516344a5cf` | `orth_lofo_w0.04` | 3/5 | -16.05 | 15/25 | 1.34 | 0.11 | [-4.323, 11.859] | 0001 | False |
| `55b2e2d2f9b615` | `orth_lofo_w0.04` | 3/5 | -15.55 | 14/25 | 1.05 | 0.11 | [-4.532, 10.29] | 0001 | False |
| `29b8383a588651` | `orth_lofo_w0.02` | 3/5 | -4.49 | 11/25 | -0.20 | 0.6245 | [-2.128, 1.991] | 0001 | False |
| `9d26cfb564230c` | `orth_lofo_w0.03` | 3/5 | -10.17 | 14/25 | 0.50 | 0.3264 | [-3.456, 6.005] | 0001 | False |
| `37984c74f5880f` | `orth_lofo_w0.02` | 4/5 | -6.09 | 15/25 | 0.99 | 0.1905 | [-1.669, 4.432] | 0001 | False |
| `59abebeed6b8fa` | `orth_lofo_w0.02` | 3/5 | -5.60 | 14/25 | 0.81 | 0.1979 | [-1.848, 4.081] | 0001 | False |
| `47ec80c2946cc7` | `orth_lofo_w0.03` | 3/5 | -9.86 | 16/25 | 0.69 | 0.2371 | [-3.279, 5.904] | 0001 | False |
| `581c04132b5940` | `orth_lofo_w0.02` | 3/5 | -5.29 | 14/25 | 0.88 | 0.1833 | [-1.579, 3.975] | 0001 | False |
| `e0f351ee76b3a7` | `orth_lofo_w0.03` | 4/5 | -7.99 | 16/25 | 1.20 | 0.08648 | [-1.803, 6.417] | 0001 | False |
| `d5bd0671dd6810` | `orth_lofo_w0.02` | 1/5 | -3.77 | 10/25 | 0.08 | 0.6245 | [-2.19, 3.668] | 0001 | False |
| `0958aef8e20ba9` | `orth_lofo_w0.02` | 1/5 | -6.29 | 10/25 | -0.23 | 0.7018 | [-2.857, 2.505] | 0001 | False |
| `b890168ab4ab0f` | `orth_lofo_w0.02` | 1/5 | -6.95 | 9/25 | -0.42 | 0.7546 | [-3.455, 2.907] | 0001 | False |
| `ddfa69a3287476` | `orth_lofo_w0.02` | 2/5 | -6.80 | 10/25 | -0.34 | 0.6736 | [-3.016, 2.355] | 0001 | False |
| `91652b3e9b2452` | `orth_lofo_w0.02` | 2/5 | -7.13 | 10/25 | -0.55 | 0.7199 | [-3.326, 2.256] | 0001 | False |
| `278d2f3f2c3631` | `orth_lofo_w0.02` | 1/5 | -6.43 | 9/25 | -0.32 | 0.7288 | [-3.086, 2.479] | 0001 | False |
| `a9fffc89ca5f05` | `orth_lofo_w0.02` | 2/5 | -6.22 | 9/25 | -0.47 | 0.7064 | [-3.104, 2.111] | 0001 | False |

## Step 3 — δ1_v14b (`init_score = 0.25·z_ATH`)

- Nov13 residual features used: `['4188470eded4', '82e9a79659bd', '21b4d1316c55', '00107284da22', 'ff71516344a5']`
- delta disp=1.0 · R²(δ\|LIVE)=0.0008

| mode | main | min | sf25 | t | W-p | boot90 | PASS |
|---|---:|---:|---:|---:|---:|---|---|
| `soft0.25_orth_w0.02` | 5/5 | 0.15 | 14/25 | 0.78 | 0.2712 | [0.255, 0.692] | False |
| `soft0.25_orth_w0.03` | 4/5 | -0.01 | 13/25 | 0.34 | 0.5 | [0.209, 0.805] | False |
| `soft0.25_orth_w0.04` | 4/5 | -0.19 | 14/25 | 0.19 | 0.5316 | [0.134, 0.888] | False |
| `soft0.25_orth_w0.06` | 3/5 | -0.88 | 11/25 | -0.19 | 0.6736 | [-0.359, 0.902] | False |
| `soft0.25_orth_w0.08` | 3/5 | -1.84 | 10/25 | -0.49 | 0.7109 | [-1.064, 0.729] | False |

## Staging

**NOTHING STAGED.** Zero candidates cleared Bars 1–4. `sub_v13p3_verified_*_NOSUBMIT.csv` not written.

## Bottom line

1. **daoviet 05:29 refresh did not update GPD** — `oof_ours`/`test_ours` byte-identical to prior pull; only new submission CSVs (`095354_exact`, `summit_095370`, `verified_095370`).
2. **alexchilton 08:51 vs prior local pull:** no added/removed/sha-changed `.npy` files (same 38 members). Top AUCs sit at **R²≈0.989–0.992** on our basis (just under the 0.992 REPACKAGED cut) with **R²(V|LIVE)≈0.96** and dominant weights on `megayak_shallow_tree`/`GB` — near-collinear with streams already inside ATH. OOF/Test symmetry is clean (disp=1.00, no G26 trap). After LOFO orth to `[z_ATH, d1_v4]` every inject **fails Bars 1–3** (typical min fold −5 to −16 e-5).
3. **Softened Stage-B works partially:** `init_score=0.25·z_ATH` lifts `δ1_v14b` to **main 5/5** at w=0.02 (min=+0.15e-5; boot90 lo=+0.255) vs V13.2’s 3/5 under full `z_ATH` — but **Bar 2 fails** (14/25, t=0.78 ≪ 2.80). Hessian fix ≠ new signal.
4. **NOTHING STAGED.** Public ATH remains **0.95388**; G26 5-fold test still the top external lever.
