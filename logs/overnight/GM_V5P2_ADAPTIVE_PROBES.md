# GM V5.2 Adaptive Probes — Untouched Ladder S5 Upgrade

Champion held: Ladder S5 **0.95335** (`sub_ladder_slot5_macro_finale_1p25e4.csv`).  
Script: `scripts/gm_v5p2_hierarchical_s5_upgrade.py`. Artifacts: `preds/gm_v5p2/`.

---

## Step 1 — δ1_v5p2 (4-pillar / NEW-overnight blend into δ1_v4)

- **MLP dropped** (`0.95171`).
- Pillars built (zrk + Method-B): XGB_24, LGB_18, CAT_9, LGB_Fixed; NEW residuals LOFO-orth vs `[z_v8, z_w12]`.
- Grid search under **strict 5/5 on v8_old + lift on z_ref_old** selected:
  - key=`(0,0,0)` ≡ **pure δ1_v4** (overnight NEW bags did not beat δ1_v4 under the gate).
- On `v8_old` (0.9525970): pos **5/5** Δe5=`[+4.54, +4.80, +0.77, +1.54, +3.30]` → AUC **0.9526289**
- On `z_ref_old` (0.9526612): pos **5/5** Δe5=`[+3.93, +5.46, +0.45, +1.35, +2.70]` → AUC **0.9526906**

## Step 2 — δ_v5p2_combo + novelty over untouched S5

- Tight nonneg mix of `(δ1_v5p2, δ3_lin, δ3_bnd)` → best `(w1,w3l,w3b)=(1.0,0.0,0.0)` @ scale `0.030` (**δ1-only**; δ3 did not improve 5/5 on z_ref_old here).
- `z_ref_clean` (0.9526407): pos **4/5** Δe5=`[+5.33, +4.28, +0.07, −0.41, +1.89]`
- **No S5 purge.** `B_strict_bad = orth_span(e2, d_bad_soft, z_v5_drop−z_S5)` (3 cols).
- `δ_novel = P⊥(combo − OLS[δ1_v4, δ3_old])` with β≈`[0.771, −0.136]`, cos(drop)=0.
- Probe axis: `u_A = unit(0.60·combo⊥ + 0.25·novel + 0.15·(zjoint+glm_graph)⊥)` then ⊥drop.

## Step 3 — Adaptive 2-slot results

| probe | file | 1−ρ(S5) | Public LB | Δ vs 0.95335 |
|---|---|---:|---:|---:|
| **A** | `sub_gm_v5p2_probeA_unpurged_s5_plus_v5combo_0p25e4.csv` | 0.250e−4 | **0.95334** | **−1.0e−5** |
| **B** (branch **3**) | `sub_gm_v5p2_probeB_branch3_0.22e4.csv` | 0.220e−4 | **0.95332** | **−3.0e−5** |

**Branch rule:** `LB_A=0.95334 ≤ 0.95334` → **Branch 3**  
`z_S5 + w·unit((z_S5−z_S4) − 0.20·u_A)` at `1−ρ(S5)=0.22e−4`, mapped via exact S5 PCHIP order-stat probs (KS=0, mean=0.016700).

### Verdict

1. Untouched-S5 + v5p2 micro-step **did not break out** (ProbeA −1e−5).
2. Branch3 reverse/continuation **also dipped** (ProbeB −3e−5).
3. **Keep Ladder S5 `0.95335` as champion.** Stop — **7 submits remain today**.
4. Overnight NEW Stage-B bags remain useful diversity but did not unlock a public residual beyond δ1_v4 under w≥0 / 5/5 gates on this probe.

Refs: ProbeA `58228702`, ProbeB `58228935`.
