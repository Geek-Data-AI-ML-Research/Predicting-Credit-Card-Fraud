# GM v11 Macro-Leap Report — toward 0.95370+

`NO_SUBMIT=True` · baseline **cand2 = 0.95355** · 4 submits remain · $0.00/hr GPU

## Executive verdict

1. **External bank screen:** 189 pairs → 155 clean → **0 strict 5/5 LOFO** winners on top of cand2 after removing the `(G26−MS)` axis. The two-seed plateau is saturated (`α=0.036` and `0.060` both LB **0.95355**).
2. **Macro lever that is still open:** inject the **official `s354` climb into cand2** (true combination Slot3 never submitted). OOF on the proxy climb is weak (≤3/5) — treat live as a **trust-region probe**, not a 5/5 lock.
3. **G26 folds 1–4 test still missing.** OOF already complete at **AUC 0.950140**; `0.5*G26+0.5*MS` = **0.950439**. Completing test folds ≈ 50.4 GPU-h (over remaining ~$6.50 budget at A40 rates unless cheaper GPU).
4. Staged 3 NOSUBMIT probes + true-mid archive (below).

## Step 1 — GPT-2 / G26

| model | OOF AUC | folds |
|---|---:|---|
| G26 seed2026 | **0.950140** | 0.94919 / 0.95037 / 0.95524 / 0.95021 / 0.94586 |
| MS (masaya) | 0.950027 | 0.94888 / 0.95032 / 0.95531 / 0.94987 / 0.94585 |
| 0.5 blend | **0.950439** | 0.94935 / 0.95068 / 0.95561 / 0.95035 / 0.9462 |

- Kernel: `external/gpt2_target_first_09501/` (GPT-2 small; GReaT replica ~5h; classifier 8ep AdamW 5e-5 batch32 bf16 ~55h; 10 feature orders; ~60h train + ~62h score).
- Dataset README: test is **Fold-0 only**; folds 1–4 ~12.6 GPU-h each still running upstream.
- Fraud-series artifacts synced into recon; oof-bank present.

## Step 2 — `v17→s352` transplant + true fusion

Test identity holds: `Δ=s352−v17=−0.03(LM−F)+0.09(L7F−F)−0.06(L7f0−F)+0.05(GPD−F)`.
`s354` recon vs official `1−ρ ≈ 2.1e-8`.

On cand2-locked OOF (`eval_S7matched_355`), combined `Δ` / s354-climb proxies do **not** clear 5/5 (typical ≤2–3/5). Soft-orth to cand2+(G26−MS) removes most public-ladder residual — private Stage-B already overlaps.

True 50/50 mid `cand2×s354`: `1−ρ(cand2)=8.410154e-05` (outside 0.16e-4 trust region → archive only).

## Step 3 — Bank LOFO screen

| | |
|---|---|
| discovered | 189 |
| clean | 155 |
| screened | 80 |
| quick survivors | 4 |
| **strict 5/5** | **0** |

No external `V−MS/MLP/GB/LM` swap beat cand2 under the orth+LOFO gate.

## Step 4 — Staged NOSUBMIT (calibrated on force_law 1−ρ)

Axis: `unitize(L(s354)) − unitize(L(cand2))` added in unitize-logit space, then `force_law` onto cand2 multiset.

| file | target 1−ρ | got | w | sha16 |
|---|---:|---:|---:|---|
| `sub_v11_cand1_top_5of5_swap_0p11e4_NOSUBMIT.csv` | 1.120e-05 | 1.120000e-05 | 2.7628 | `eaf339d1bf3527b3` |
| `sub_v11_cand2_multi_5of5_orth_0p14e4_NOSUBMIT.csv` | 1.400e-05 | 1.400000e-05 | 3.0889 | `9b549aab349b1a83` |
| `sub_v11_cand3_conservative_0p08e4_NOSUBMIT.csv` | 8.000e-06 | 8.000000e-06 | 2.3352 | `2507241b9ae317be` |

| `sub_v11_cand_TRUE_MID_cand2_s354_NOSUBMIT.csv` | mid | 8.410154e-05 | — | `1edf63803ba50b5b` |

Pairwise: 1↔2=`1.593e-07`, 1↔3=`2.727e-07`, 2↔3=`8.408e-07`.

### Path to 0.95370+
1. Live-probe cand1 (`0.112e-4` s354-climb) only if accepting non-5/5 OOF.
2. Finish G26 test folds 1–4 (or wait for goodpjw upload) then re-run two-seed at α∈[0.036,0.06] on 5-fold G26.
3. Do **not** burn GPU on L7 5F until G26 5-fold test exists — higher leverage per dollar.

---
`NO_SUBMIT=True` — no Kaggle upload. 4 daily submits remain.
