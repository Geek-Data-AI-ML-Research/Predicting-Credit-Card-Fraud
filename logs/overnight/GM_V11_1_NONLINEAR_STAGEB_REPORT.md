# GM v11.1 Non-Linear Stage-B / Wilcoxon / G26 Denoiser

`NO_SUBMIT=True` · immutable **cand2=0.95355** · G26 test5fold=**False** (upstream still Fold-0 only)

## Executive verdict

1. **No pillar cleared the strict 5/5 LOFO gate** on `eval_S7matched_355` after double-orth to `[S7, G26−MS]`.
2. **Best near-miss: `orth(LM_trim)`** — LOFO **4/5**, meanΔ=0.007e-5, folds=[0.018, 0.007, -0.004, 0.006, 0.007], simple 3/5. (Earlier quick-scan 5/5 was optimistic vs full LOFO.)
3. **G26 James-Stein denoise** and **Wilcoxon** and **Stage-B_v11** all failed the gate (denoise≈null; Wilcoxon 1/5; StageB 3/5).
4. Staged cand2±LM_trim probes anyway as **diagnostic NOSUBMIT** (not “verified 5/5”).

## Promotion scorecard

| pillar | S7m | meanΔe-5 | d0 | folds | GATE |
|---|---:|---:|---:|---|---|
| P1 LM_trim | 4/5 | 0.007 | 0.018 | [0.018, 0.007, -0.004, 0.006, 0.007] | FAIL (4/5) |
| P1 G26_denoise | 0/5 | ~0 | 0 | — | FAIL |
| P2 Wilcoxon | 1/5 | -0.004 | 0 | — | FAIL |
| P3 StageB_v11 | 3/5 | ~0 | 0 | — | FAIL |

## G26 upstream
- local/remote `test_5fold`: **False** / **False**
- files: `['README.md', 'gpt2_s2026_oof_5fold.npy', 'gpt2_s2026_test_f0.npy']`

## Staged NOSUBMIT (cand2 + w·orth(LM_trim))

| file | target | got | w | sha16 |
|---|---:|---:|---:|---|
| `sub_v11p1_slot1_verified_5of5_pillars_0p11e4_NOSUBMIT.csv` | 1.120e-05 | 1.120000e-05 | 2.5618 | `10db902de943e6e2` |
| `sub_v11p1_slot2_verified_5of5_pillars_0p16e4_NOSUBMIT.csv` | 1.600e-05 | 1.600000e-05 | 3.0614 | `4d1900af6197130d` |
| `sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv` | 1.000e-05 | 1.000000e-05 | 2.4208 | `46299dab908d5989` |

---
`NO_SUBMIT=True`. Do **not** submit prior v11 s354-climb files.
