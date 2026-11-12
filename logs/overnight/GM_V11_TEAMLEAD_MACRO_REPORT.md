# GM v11 Team-Lead Modular Macro-Leap Report

`NO_SUBMIT=True` · immutable **cand2 = 0.95355** · $0.00/hr GPU

## Executive verdict

1. **All 4 modules FAIL the independent 5/5 LOFO gate** after double-orthogonality to `[L(S7), G26−MS]` + PCHIP contrast law.
2. **Best near-miss remains Module A / `orth(LM_trim)`: 4/5** (meanΔ=0.007e-5, folds=[0.018, 0.007, -0.004, 0.006, 0.007], simple 3/5, ΔF0=0.018>0). Not promoted (needs 5/5).
3. **G26 `test_5fold` still not on Kaggle** (Fold-0 only). Est. ~50.4 GPU-h to finish folds 1–4.
4. Staged **diagnostic** cand2+LM_trim probes at exact trust-region distances (not “macro_pass verified”).

## G26 recipe / status
- arch: GPT-2 small target-first (Deotte / masaya / goodpjw seed-2026)
- classifier: 8ep AdamW 5e-5→0, batch32, bf16; 10 feature orders
- train ~60 GPU-h; test score ~12.6 GPU-h / fold
- has_test_5fold local/remote: **False** / **False**
- files: `['README.md', 'gpt2_s2026_oof_5fold.npy', 'gpt2_s2026_test_f0.npy']`

## 4-Module Promotion Scorecard

| Module | S7m pos | meanΔe-5 | d0 | folds | simple | GATE |
|---|---:|---:|---:|---|---:|---|
| A_zerosum (LM_trim best) | 4/5 | 0.007 | 0.018 | [0.018, 0.007, -0.004, 0.006, 0.007] | 3/5 | FAIL |
| B_wilcoxon | 1/5 | -0.004 | 0 | [0,0,0,0,-0.02] | 0/5 | FAIL |
| C_moe | 0/5 | -0.001 | ~0 | — | 0/5 | FAIL |
| D_stageB | 3/5 | ~0 | 0 | — | 3/5 | FAIL |
| A2 LM+G26denoise | 4/5 | 0.007 | 0.006 | — | 3/5 | FAIL |

Passed modules: `[]` → no `Δ_macro_pass`.

## Staged NOSUBMIT (cand2 + w·orth(LM_trim) diagnostic)

| file | target | got | w | sha16 |
|---|---:|---:|---:|---|
| `sub_v11_slot1_macro_pass_0p11e4_NOSUBMIT.csv` | 1.120e-05 | 1.120000e-05 | 2.5618 | `10db902de943e6e2` |
| `sub_v11_slot2_macro_pass_0p22e4_NOSUBMIT.csv` | 2.200e-05 | 2.200000e-05 | 3.5896 | `32845f9ec31de93d` |
| `sub_v11_slot3_top_single_module_0p12e4_NOSUBMIT.csv` | 1.200e-05 | 1.200000e-05 | 2.6517 | `fa59aad7e2313701` |

### Notes
- Do **not** submit earlier `sub_v11_cand1..3` s354-climb files (≤3/5 OOF).
- Highest leverage remaining: **G26 test 5-fold** upload/recompute, then re-open two-seed + Stage-B with matched dispersion.
- LM_trim is the only consistently positive-mean direction; one more fold flip away from a 5/5 gate.

---
`NO_SUBMIT=True` — no Kaggle upload. 4 daily submits remain.
