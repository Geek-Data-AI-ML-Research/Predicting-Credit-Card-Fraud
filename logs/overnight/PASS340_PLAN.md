# Pass 0.95340+ — TOMORROW READY

**Goal:** 0.95340+ (+0.00010 from St9 0.95330; leader 0.95334)  
**Quota:** opens ~00:00 UTC — do not submit until then  
**Helper:** `scripts/pass340_tomorrow.sh` (shows LB + copy-paste; does not auto-submit)

## Calibration (critical)
| Move | 1-ρ(prev) | ΔLB |
|------|-----------|-----|
| St ladder wins | 0.1–0.5e-4 | +1–3e-5 |
| Strike5A crash | **4.5e-4** | **−15e-5** |
| SAFE fam×0.35 | 2.4e-4 | first probe |
| SAFE fam×0.45 | 3.9e-4 | near St5A dist — only if ×0.35 up |
| P2 fam×0.85 | 14e-4 | scale-up only after proof |

SoftFamily is OOF-gated (unlike St5A junk orth) but **start below crash distance**.

## CPU campaign add-ons (2026-11-09)
Cumulative OOF after CPU levers: **+21.98e-5** (FAM21 +0.61 alex neg). Transductive/20k/soft-real: no new gain (RAM-gated). New files: `L1/L2` multi-leader, `R1–R3` alexrestack, `CPU_x055/x070`. Full report: `logs/overnight/CPU_CAMPAIGN_REPORT.md`. **GPU phase is READY.**

## Slot plan (5 submits)

### Slot 1 — SAFE SoftFamily (default)
`submissions/sub_pass340_S_fam21_x035.csv`  
1-ρ=2.39e-4 · FAM21 OOF +21.37e-5 · mean=0.01670

```bash
.venv_forensics/bin/kaggle competitions submit -c playground-series-s6e11 \
  -f submissions/sub_pass340_S_fam21_x035.csv \
  -m "Pass340 S1: St9+0.35*FAM21(+21.37e-5 OOF 5/5)"
```

### Slot 2 — branch on Slot1
| Slot1 LB | File |
|----------|------|
| **≥0.95334** (clear up) | `sub_pass340_S_fam21_x055.csv` then later P2 |
| **0.95331–0.95333** (tiny up) | `sub_pass340_S_fam21_x045.csv` |
| **0.95330** flat | `sub_pass340_F1_famp24_x070.csv` (+own_p24) |
| **≤0.95329** down | `sub_pass340_S_fam21_x025.csv` (cut) |

### Slot 3–5 — climb to 340+
- Climbing: `S_fam21_x055` → `P2_fam21_x085` → `F2_famp24_x085` / `N1_noproxy_x070`
- Flat after fam: `C1_cleanjoint_x055` (soft_w015 proxy) → `C2_cleanjoint_x070`
- Down twice: stop orth; `A1_st9_aman_w12` salvage (caps ~leader, not 340 alone)
- **Never** repeat a direction that lost ≥0.00005
- If any hit **0.95338+**: next slot full `P3_fam21_x100` or `C2` for 340+

## OOF pack ladder (all 5/5)
| Pack | ΔOOF | Best first files |
|------|------|------------------|
| FAM21 | +21.37e-5 | S_×035 / S_×045 / P2 |
| FAM+own_p24 | +25.41e-5 | F1 / F2 |
| NOPROXY | +28.67e-5 | N1 |
| CLEAN+proxy | +34.08e-5 | C1 / C2 |

## Why this can clear 0.95340
Aman-blend alone cannot pass 340 (caps near 0.95334). SoftFamily is the only sized honest lever. Probe below St5A crash distance, then scale only if LB confirms.
