# GM v7.1 Surgical Lock — Dual-Anchor Extraction + Concave LB

**NO_SUBMIT=True**. ATH remains **`0.95354`**.

## Bugs fixed

1. Dropped weak `0.95230` bags; kept only `lgb_v7_extratrees_dual` (`0.9525161`) + `xgb_v7_bce_dual_anchor` (`0.9524929`) + Phase-1 top cols.
2. Tested LOFO against full 3-vector base-margin span `[z_v8, z_w12, v5_harvest]` **and** tree-logit residuals / beyond-δ1 alternatives.
3. Fold-0 evaluator restored to VERIFY-style (`z_ref + w·unitize(δ)` on `fold_id==0`).
4. LB surface re-fit with **concave constraint `a ≤ 0`** (no convex explosion to 0.955).

## Step 1 — δ1_v7p1

### Residual diagnostics

| mode | 5F Δe-5 | pos/5 | F0 Δe-5 |
|---|---:|---:|---:|
| LOFO ⊥ `[z8,w12,v5]` (user-spec) | +0.00 | 0/5 | +0.00 |
| LOFO ⊥ `[z_ref,δ1]` | +0.00 | 0/5 | +0.00 |
| LOFO ⊥ `[z8,lgb_gpt2]` (Fix1) | +0.00 | 0/5 | +0.00 |
| **tree_logit ⊥ `[z_ref,δ1]`** | **+0.74** | **4/5** | **+1.48** |

**Finding:** Dual-anchor boosters live almost entirely inside the base-margin span in rank-logit space. Stripping `[z8,w12,v5]` nulls them. The only recoverable innovation is the **tree logit add-on** (`logit(p) − Platt(margin)`), at **+0.74e-5 / 4/5** — not enough alone to beat δ1_v4 under a strict 5/5 gate.

### δ1_v4 (Fold-0 bug FIXED)

| base | 5F Δe-5 | pos/5 | F0 Δe-5 | AUC | per-fold ×1e-5 |
|---|---:|---:|---:|---:|---|
| z_ref | **+3.22** | **5/5** | **+5.42** | 0.9526953 | `[+5.42, +6.99, +0.20, +0.59, +2.88]` |
| v8_old | **+3.54** | **5/5** | **+6.65** | — | — |

This matches the VERIFY / V5P2 Fold-0 regime (no more bogus `F0=−0.01e-5`).

### Best blend

- Selected residual mode: `tree_logit`
- Best α,β under strict 5/5 on z_ref **and** v8: **`α=0.000`, `β=0.000`** → δ1_v7p1 **collapses to δ1_v4**
- per-fold GAIN vs δ1_v4: `[0, 0, 0, 0, 0]`
- Any α>0 loses the 5/5 / beat-δ1 gate (tree residual is 4/5)
- **Strict 5/5 Gate for a true upgrade: FAIL** (honest null: dual-anchor solo AUC does not transfer as 5/5 residual beyond δ1_v4)

P5_L5.1 on disk was sign-inverted (`AUC≈0.09` → flipped to `0.909`); still weak vs Stage-B and not used in the winning blend.

## Step 2 — VERIFY OOF proxy past S7

Exact VERIFY anchors reproduced:

| recipe | 5F Δe-5 | pos/5 | F0 Δe-5 | per-fold |
|---|---:|---:|---:|---|
| m1@0.04 | **+0.74** | 2/5 | **+4.43** | `[+4.43, −0.69, −0.42, +0.04, −0.17]` |
| S6@0.05 | **+1.81** | 3/5 | **+6.97** | `[+6.97, +0.47, −0.19, −1.68, +2.21]` |
| S6v7@0.05 | +1.81 | 3/5 | +6.97 | (identical — α=β=0) |

S7_proxy (s6_ray@w=0.055): 5F=`+1.62` / F0=`+6.30` (VERIFY S7_A was +2.38/+9.50 at a slightly different w).

### Steps past S7 (all worse than S7_proxy)

| kind | 1−ρ(S7) | w_L7 | 5F | vsS7 5F | F0 | vsS7 F0 |
|---|---:|---:|---:|---:|---:|---:|
| pure / v7p1 | 0.06e-4 | 0.00 | +1.38 | −0.24 | +5.14 | −1.16 |
| pure / v7p1 | 0.10e-4 | 0.00 | +1.17 | −0.45 | +4.30 | −2.01 |
| pure / v7p1 | 0.15e-4 | 0.00 | +0.87 | −0.75 | +3.15 | −3.15 |
| + L7 denoise | any | −0.05/−0.10 | worse | worse | worse | worse |

**OOF max on this ray is at / before S7 — not past it.** Pure continuation and L7-denoise both degrade OOF past `t≈1.005`.

## Step 3 — Concave LB law + staged slots

### Fit on areeb→S5→S6→S7

- **linear:** `LB = 0.95346910 + 6.99725804e-05 · t`
- **concave quad (`a≤0`):** `a = 0` (linear is optimal; no positive curvature)

| t | pred_lin ≈ pred_quad |
|---:|---:|
| 1.005 (S7) | **0.953540** |
| 1.26 | **0.953558** |
| 1.48 | **0.953574** |
| 1.65 | **0.953585** |
| 2.00 | 0.953610 |
| 2.25 | 0.953627 |

Convex Phase-0 bug (`0.955042` @ t=2.25) is **gone**.

### NOSUBMIT slots (materialized, 299,984 rows each)

| slot | file | 1−ρ(S7) | t≈ | pred_lin | note |
|---:|---|---:|---:|---:|---|
| 1 | `sub_gm_v7p1_slot1_cont_plus_v7p1_0p10e4_NOSUBMIT.csv` | 0.10e-4 | 1.344 | **0.953563** | cont + v7p1 (=δ1_v4) |
| 2 | `sub_gm_v7p1_slot2_pure_s5s6s7_cont_0p10e4_NOSUBMIT.csv` | 0.10e-4 | 1.353 | **0.953564** | pure ray control |
| 3 | `sub_gm_v7p1_slot3_step_up_0p22e4_NOSUBMIT.csv` | 0.22e-4 | 1.503 | **0.953574** | next rung |
| 4 | `sub_gm_v7p1_slot4_v7p1_plus_l7denoise_0p22e4_NOSUBMIT.csv` | 0.22e-4 | 1.503 | **0.953574** | L7 noise missing post-/tmp wipe → Slot3-equivalent |
| 5 | `sub_gm_v7p1_slot5_macro_0p40e4_NOSUBMIT.csv` | 0.40e-4 | 1.665 | **0.953586** | speculative |

**OOF caution for tomorrow:** Step 2 says OOF already peaks at S7; surface still predicts +1–2 ticks. Prefer Slot1/2 micro-steps; treat Slot4/5 as speculative.

## Verdict

| item | result |
|---|---|
| Fold-0 evaluator | **FIXED** (δ1_v4 F0=+5.42e-5, 5/5) |
| Concave LB lock | **FIXED** (a=0; S7 pred=0.95354; t=2.25→0.95363 not 0.955) |
| Dual-anchor → 5/5 residual upgrade | **FAIL** (null transfer; α=β=0) |
| Tree-logit crumb | +0.74e-5 @ 4/5 — useful diagnostic, not a gate-pass |
| Past-S7 OOF | **declining** — do not trust large steps |
| 5 NOSUBMIT slots | **staged** under `submissions/sub_gm_v7p1_slot{1..5}_*_NOSUBMIT.csv` |

**NO_SUBMIT=True** — no Kaggle submit performed.
