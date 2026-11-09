# Retake Plan — beat 0.95332 (need 0.95334+)

**Status:** 강북구집돌이 `0.95332` (#1). Us `0.95330` (#2).  
**Quota:** exhausted until **00:00 UTC** (≈ **02:00 IST**). Probe submit returned HTTP 400.

SoftFamily nested ΔOOF **`+19.67e-5` (5/5 folds)** on the 7M stack — ~10× Strike2’s OOF lift.

## Submit order when quota resets

| # | File | Role | `1−ρ(St9)` |
|---|---|---|---:|
| 1 | `sub_retake_R0_softfamily_x060.csv` | Mid opener (safer leap) | ~3.9e-4 |
| 2 | `sub_retake_R1_softfamily_x080.csv` | **Primary retake** | ~6.9e-4 |
| 3 | `sub_retake_R4_softfamily_x070_cont.csv` | Family + cont | ~5.2e-4 |
| 4 | `sub_retake_R6_softfamily_x075.csv` | Mid-high family | ~6.1e-4 |
| 5 | `sub_retake_R2_softfamily_x100.csv` | Full family | ~10.8e-4 |
| 6 | `sub_retake_R5` / `R7` / `R3` | Max push / soft@0.07 alt | — |

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  .venv_forensics/bin/kaggle competitions submit -c playground-series-s6e11 \
  -f submissions/sub_retake_R0_softfamily_x060.csv \
  -m "Retake R0: St9+0.60*SoftFamily vs 0.95332"

env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  .venv_forensics/bin/kaggle competitions submit -c playground-series-s6e11 \
  -f submissions/sub_retake_R1_softfamily_x080.csv \
  -m "Retake R1: St9+0.80*SoftFamily primary leap"
```

If R0/R1 land ≥ `0.95333`, ride LB-gradient with R4/R6. If they drop, fall back to Day2 Slot A/I (softer).
