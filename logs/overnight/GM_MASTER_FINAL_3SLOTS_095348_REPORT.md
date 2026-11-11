# Master Final 3-Slot + Full Submission Postmortem

Elapsed **~200s**. Submitted **Slots 5–6 only**. **Slot 7 HELD** (1 daily reserve left).  
Entered day at private S5 **`0.95335`**; entered this master phase tied at public **`0.95347`**.

---

## Executive verdict

| Milestone | LB |
|---|---:|
| Prior private champ (Ladder S5) | 0.95335 |
| Public areebbb / our S2–S3 | 0.95347 |
| **Master Slot 5 (m1)** | **0.95350** (+3e-5 vs areeb) |
| **Master Slot 6 (Branch A)** | **0.95352** (+5e-5 vs areeb) |
| Slot 7 | **HELD — not submitted** |

**Cleared `0.95348`? YES.** New personal / session ATH = **`0.95352`**.

---

## Part A — Every live submission: what worked / what didn’t

### Public genealogy (not our submits)

| File | Essence | LB | Lesson |
|---|---|---:|---|
| goodpjw | `A+0.60(A−F)+0.12(LM_nov08−F)+0.06(L7−F)` | 0.95346 | Frozen megaLM ladder works |
| areebbb | switch to **LM_curr** + `−0.05(MLP−F)−0.025(GB−F)` | **0.95347** | **+1 tick** from current LM + push away from MLP/GB |

### Breakout Slots 1–4

| Slot | Direction | 1−ρ | LB | vs 0.95347 | Verdict |
|---:|---|---:|---:|---:|---|
| 1 | `+0.55 gpt2m + 0.45 3ladder` | 0.18e-4 | **0.95346** | −1 | **FAIL** — overshoot mix |
| 2 | BranchC `−0.40 3ladder + 0.70 gpt2m + 0.50 stageB` | 0.20e-4 | **0.95347** | 0 | **RECOVERY** — anti-3ladder + stageB saves the tick |
| 3 | empgrad (flat fit) | 0.30e-4 | **0.95347** | 0 | **NEUTRAL** — no new direction |
| 4 | pure `+gpt2m` | 0.08e-4 | **0.95346** | −1 | **CONFIRMED DRAG** — gpt2m alone is toxic on areeb |

**S1–S4 geometry takeaways**

1. `+u_gpt2m_lm` is a **confirmed −1 tick axis** (S4).
2. Further `+u_3ladder_grad` from areeb **hurts** (S1), even though its dominant subcomponent is `part_L7_tree` (cos≈0.87) — meaning **past areeb, more L7/tree continuation overshoots**.
3. Useful move is **`−u_3ladder + u_stageB`**, not more LM stacking (S2).
4. Empgrad over `{0.95346,0.95347}` cannot invent a breakout (S3).

### Master Slots 5–7

| Slot | Vector | 1−ρ | LB | Verdict |
|---:|---|---:|---:|---|
| **5** | **m1** = `+0.65 stageB − 0.35 3ladder − 0.20 gpt2m` | 0.22e-4 | **0.95350** | **ATH breakout** — decontaminated S2 winner |
| **6** | BranchA `0.70 m1 + 0.30 m2` | 0.52e-4 | **0.95352** | **ATH step-up** — 2.2× continuation works |
| 7 | staged crest candidate | ~0.15e-4 | — | **NOT SUBMITTED** (reserve) |

**Why S5/S6 worked**

- Stripped the Slot-4 gpt2m **drag** (`−0.20` contrastive, cos(m1,gpt2m)=−0.19).
- Kept Slot-2’s anti-3ladder + stageB mass (m1 cos(stageB)=+0.85).
- S6 scaled the same geometry further and blended a little native/private residual (m2) without reintroducing `+gpt2m`.

---

## Part B — OOF-bank / lm7 audit

`z_ref_old` OOF AUC = **`0.9526612`**.

### Files

- `lm7_oof_f0.npy` `(139992,10,2)` — **Fold 0 only** (mapped via `folds.csv`).
- `lm7_test_f0.npy` `(299984,4,2)` — no f1–f4.
- Full OOF present for `oof_gbn_bag5_w512`, `oof_gb_bag5`, `megayak/lm_oof`.

### Best OOF lifts on `z_ref_old`

| direction | scope | pos/5 | lift e-5 | w |
|---|---|---:|---:|---:|
| `+δ1_v4 / stageB` | 5f | **5/5** | **+3.42** | +0.050 |
| `+(L7−F)` | fold0 | — | +5.32 | −0.030† |
| `−(MLP−F)` | 5f | 3/5 | +0.41 | −0.010 |
| `−(GB−F)` | 5f | 2/5 | +0.32 | −0.010 |
| `+(LM_curr−F)` | 5f | 0/5 | 0.00 | 0.000 |

† Fold0 L7 sign at best-w is negative on this z_ref probe (local optimum quirky); live ladder still used `+L7` successfully in areeb — treat fold0 L7 OOF as incomplete (single fold, not 5f nested).

### `u_3ladder` decomposition vs areeb

| sub-part | cos with `u_3ladder` |
|---|---:|
| part_AF (`0.60→0.62`) | +0.13 |
| part_LMnov08 (mix frozen LM) | +0.46 |
| **part_L7_tree** (`+L7 −MLP/GB`) | **+0.87** |

So S1’s 3ladder was mostly “more of areeb’s winning L7/tree move.” Live LB says **that move is already saturated at areeb** — further push drops to 0.95346; reversing it (S2/S5) helps when paired with stageB.

---

## Part C — Master vector recipes used

```
m1 = unit(+0.65*u_stageB − 0.35*u_3ladder − 0.20*u_gpt2m)
m2 = unit(0.45*u_stageB + 0.35*z_native_diff + 0.20*u_priv_S5)
     score_native = A + 0.585(A−F) + 0.125(LM_curr−F) + 0.072(L7−F)
                    − 0.062(MLP−F) − 0.031(GB−F)   # LM_curr only, no LM_nov08
m3 = unit(P⊥_{B6 ∪ gpt2m}(u_stageB))
```

PCHIP↔rank map check: `1−ρ = 0` (rank-invariant).

---

## Part D — Last reserve slot (guidance only)

Current ATH **`0.95352`**. One submit left.

- Prefer a **tiny crest** along `m1` or `0.7 m1 + 0.3 m2` at **≤0.12e-4** from Slot-6 — not a large step.
- Do **not** re-add `+u_gpt2m`.
- If unsure, **hold** the reserve; `0.95352` already beats areeb by +5e-5.

---

Artifacts: `preds/gm_breakout/` · script: `scripts/gm_master_final_3slots_095348.py` · submissions: `sub_master_s5_*.csv`, `sub_master_s6_branchA_*.csv`, `sub_master_s7_HELD_NOSUBMIT.csv`.
