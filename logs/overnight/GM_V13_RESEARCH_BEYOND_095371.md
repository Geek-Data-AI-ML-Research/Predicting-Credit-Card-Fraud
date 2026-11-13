# GM V13 Research — Beyond ATH 0.95371

**NO_SUBMIT=True** · **POD TOUCHES: 0** · 2026-11-13

## Public LB headroom (live)

| rank | team | public | recipe? |
|---|---|---:|---|
| 1 | Smart Manoj | **0.95465** | no public kernel |
| 2 | tine.sh agent | 0.95399 | no public kernel |
| 3 | Md. Hamid Hosen | 0.95395 | no public kernel |
| ~4–7 | goodpjw / others | **0.95386** | **yes** — goodpjw LM Ladder v4 bold |
| us | Moran Zavdi | **0.95371** | L2 = Q17 + 0.03·d1 |

Gap to #1: **+94e-5**. Gap to reproducible public: **+15e-5**.

## What can raise our score now

1. **Submit pure goodpjw bold** (`sub_v13_PURE_goodpjw_bold095386_NOSUBMIT.csv`, sha256 `043d28ce84112907…`)  
   - Already scored **0.95386** on the board. Same class of move as L1 pure Q17.  
   - Method: joint max of nested CV + public-score forecast over 87 members on the 0.95370 anchor.  
   - Top moves: megayak shallow tree +0.28, daoviet GP+DVAE +0.14, shrink megayak LM / sadam XGB / masaya GPT-2.

2. **Do not** prefer jamal 0.85·bold+0.15·areeb (0.95384) — strictly worse than pure bold.

3. **After bold confirms:** try `L(bold)+w·unitize(d1_v4)` (w≈0.03) — only private residual with proven public transfer on this competition day. Hold P_super / stageB / sov+G26 (L3/L4 poison).

## What will not help (already known)

| lever | why |
|---|---|
| More L2 bank blends | OOF 5/5 → public red (L3/L4) |
| Finish L7_s2026 5F from today's fold0 | canary **failed** gate3 (s2026 < s7) |
| Chase 0.95465 from public kernels | none published |

## Staged NOSUBMIT

- `submissions/sub_v13_PURE_goodpjw_bold095386_NOSUBMIT.csv`
- `submissions/sub_v13_PURE_goodpjw_balanced095380_NOSUBMIT.csv`
- `submissions/sub_v13_PURE_goodpjw_careful095376_NOSUBMIT.csv`

## Verdict

**Yes — we can gain.** The clear next submit is pure goodpjw **0.95386** (+15e-5). Climbing to **0.95465** needs private signal we do not have in public banks.
