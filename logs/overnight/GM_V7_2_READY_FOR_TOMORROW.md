# GM v7.2 Ready for Tomorrow — Dedup + Decoupled Stage-B

**NO_SUBMIT=True**. Low-RAM sequential float32 pipeline. ATH **`0.95354`**.

- tree_logit source: `fallback_overnight_extratrees+monotone` (**v7 dual-anchor npy caches were wiped from `/tmp`**)
- block3 source: `fallback_X_v5_TE24_ridge_LOFO` (condensed_blocks missing; this proxy was **OOF-negative** `@0.04 → −9e-5`, so Ray B gains are driven by `δ1/u_stageB + tree_logit`, not Block3)
- observed process RSS during run: **~530–580 MB** (briefly over 500 MB while `X_v5` mmap warmed for Block3 fallback); idle after finish
- CPU: single Python worker, `OMP_NUM_THREADS=2`; monitor load ~3.3–4.3

## Step 1 — Ray A (coupled S5→S7) vs Ray B (freeze −u_3ladder, advance stageB_plus)

S7_proxy: 5F=`+2.69e-5` F0=`+12.67e-5` pos=3/5

| 1−ρ(S7) | A Δ5F vsS7 | A ΔF0 vsS7 | B Δ5F vsS7 | B ΔF0 vsS7 | B pos |
|---:|---:|---:|---:|---:|---:|
| 6.00e-06 | -0.12 | +0.85 | +0.20 | +0.46 | 3/5 |
| 8.00e-06 | -0.17 | +1.08 | +0.22 | +0.54 | 3/5 |
| 1.00e-05 | -0.24 | +1.33 | +0.21 | +0.65 | 3/5 |
| 1.50e-05 | -0.42 | +1.93 | +0.04 | +0.84 | 2/5 |
| 1.80e-05 | -0.56 | +2.27 | -0.10 | +0.88 | 2/5 |
| 2.20e-05 | -0.79 | +2.61 | -0.41 | +0.79 | 2/5 |
| 3.00e-05 | -1.32 | +3.25 | -1.33 | +0.47 | 2/5 |

**Winner:** Ray B @ `1.00e-05` (vsS7 5F `+0.21` / F0 `+0.65`) vs Ray A vsS7 5F `-0.24`.

## Step 2 — 4 distinct NOSUBMIT slots from z_S7

Pairwise uniqueness gate `1−ρ > 0.02e-4` (=`2e-6`): **PASS** (min pair `1−ρ≈2.2e-6` slot2↔3; max `1.2e-5` slot1↔4 — distinct directions, still close in rank space)

| slot | file | 1−ρ(S7) | t≈ | pred LB | mean |
|---:|---|---:|---:|---:|---:|
| 1 | `sub_gm_v7p2_slot1_pure_ray_0p08e4_NOSUBMIT.csv` | 8.00e-06 | 1.319 | 0.953561 | 0.016700 |
| 2 | `sub_gm_v7p2_slot2_decoupled_stageB_plus_treelogit_0p10e4_NOSUBMIT.csv` | 1.00e-05 | 1.311 | 0.953561 | 0.016700 |
| 3 | `sub_gm_v7p2_slot3_hybrid_rayA_rayB_0p18e4_NOSUBMIT.csv` | 1.80e-05 | 1.446 | 0.953570 | 0.016700 |
| 4 | `sub_gm_v7p2_slot4_adaptive_stepup_0p30e4_NOSUBMIT.csv` | 3.00e-05 | 1.522 | 0.953576 | 0.016700 |

**NO_SUBMIT=True** — fire Slot1/2 first tomorrow; Slot4 only if Slot1 or Slot2 ≥ 0.95355.