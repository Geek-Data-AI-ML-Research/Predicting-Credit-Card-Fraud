# NOV11 0.95347 V2 Kernel Audit

**NO_SUBMIT=True**. Elapsed **~1081s** (+ restage).  
Base probe: `z_ref_old+δ1_v4` OOF AUC≈`0.952695` (w≈0.05 on this run; prior recon pinned `0.9526906` @ w=0.03).  
Champion held: Ladder S5 **0.95335**. **No Kaggle submissions.** 7 slots remain.

## Step 1 — Pulled kernels & upstream harvest

### `areebbb/s6e11-three-ladders-two-steps-lb-0-95347` (LB `0.95347`)

- dataset_sources: `goodpjw2008/s6e11-compact-lm-seed7`, `goodpjw2008/s6e11-oof-bank`, `amanatar/s6e11-strike10-podium-apex-095340`, `moranzavdi/s6e11-strike9-lb-095330`, `megayak/s6e11-target-first-lm`
- kernel_sources: `[]`
- code/out under `external/nov11_095347_recon/areebbb__…` (kernel output had log only; **exact score reconstructed** from public inputs)

### `goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346` (LB `0.95346`)

- dataset_sources: `goodpjw2008/s6e11-compact-lm-seed7`, `amanatar/s6e11-strike10-podium-apex-095340`, `moranzavdi/s6e11-strike9-lb-095330`
- kernel output: `submission.csv` pulled

### `amanatar/s6e11-podium-apex-public-lb-0-95340` (LB `0.95340`)

- dataset_sources: `amanatar/s6e11-strike10-podium-apex-095340`, `moranzavdi/s6e11-strike9-lb-095330`
- kernel output: `submission.csv` pulled

### `malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion`

- dataset_sources: `moranzavdi/s6e11-strike9-lb-095330`
- kernel_sources: `amanatar/s6e11-podium-apex-public-lb-0-95340`
- kernel output: `submission.csv` pulled

### `lucifer19/the-golden-signal-consensus-blend-v22`

- dataset_sources: `lucifer19/golden-signal-v5-scored-predictions`, `goodpjw2008/s6e11-fraud-series-artifacts`, `moranzavdi/s6e11-strike9-lb-095330`
- outputs: figures/verify script (no submission.csv in kernel out); scored preds from dataset

**Downloaded OK:** Strike10, Strike9, compact-lm-seed7, oof-bank, fraud-series-artifacts, megayak LM, lucifer golden-signal-v5.  
**Failed / inaccessible regex false-positives:** `kaggle/working`, `research_v13/artifacts`, `research_v15/bank`, `research_v17/artifacts`, etc.

---

## Exact formulas (forensic)

### areebbb (`0.95347`) — exact, SHA-checked inputs

```python
# A = Strike10 (amanatar 0.95340), F = Strike9 (moranzavdi 0.95330)
# LM = megayak lm_test.npy (current); L7 = goodpjw lm7_test_f0 log-odds
# MLP = test_gbn_bag5_w512.npy; GB = test_gb_bag5.npy  (goodpjw OOF bank)
B = A + 0.6*(A - F) + 0.12*(LM - F) + 0.06*(L7 - F)
score = B - 0.05*(MLP - F) - 0.025*(GB - F)
submission = (rankdata(score) - 0.5) / N
```

Paths: `/kaggle/input/{amanatar/…095340, moranzavdi/…095330, megayak/…, goodpjw2008/…seed7, goodpjw2008/…oof-bank}/…`

### goodpjw2008 (`0.95346`) — exact

```python
# A=S10, F=S9, LMm=megalm_test_v_nov08.npy (FROZEN 11-08 megayak), LM7 from lm7_test_f0.npy
score = A + 0.6*(A - F) + 0.12*(LMm - F) + 0.06*(LM7 - F)
submission = (rankdata(score) - 0.5) / N
```

Public probe table in notebook: S10→0.95340; +0.6 extrapol→0.95342; +0.12 megaLM→0.95343; +0.06 LM7→**0.95346**.  
**masaya GPT-2 step hurts** (−11e−5 / −22e−5 in their probes).

### amanatar (`0.95340`) — exact

Notebook **prefers shipping the published file** if present:

```python
if f_340 is not None:
    preds = read(submission_095340.csv)   # identity publish of Strike10 file
else:
    # reconstruct from St9 ladder diffs in probit space:
    q_m, q_30, q_28 = ppf(rank(s_m)), ppf(rank(s_30)), ppf(rank(s_28))
    d30 = q_30 - q_m
    d28p = d28 - proj(d28 onto d30)
    qc = q_m + 1.50*d30 + 0.50*d28p
    preds = PCHIP_map_to_s_m_order_stats(rank(qc))
```

---

## Step 2 — Test-space lineage

zrk matrix: `norm.ppf(rankdata(p)/(N+1))`.

### Pairwise `1 − spearmanr` (selected)

| | areeb | goodpjw | aman | St9 | S5 | Vertex |
|---|---:|---:|---:|---:|---:|---:|
| **areeb** | 0 | **0.00008** | 0.0008 | 0.0012 | 0.0015 | 0.0014 |
| **goodpjw** | 0.00008 | 0 | 0.0008 | 0.0011 | 0.0014 | 0.0013 |
| **aman** | 0.0008 | 0.0008 | 0 | **0.0002** | 0.0008 | 0.0007 |
| **St9** | 0.0012 | 0.0011 | 0.0002 | 0 | 0.0004 | 0.0002 |
| **S5** | 0.0015 | 0.0014 | 0.0008 | 0.0004 | 0 | 0.0001 |
| **Vertex** | 0.0014 | 0.0013 | 0.0007 | 0.0002 | 0.0001 | 0 |

Full 13×13 matrix in run log / prior report body (`areeb…ProbeA`).

### OLS: `z_areeb ~ [z_goodpjw, z_St9, z_aman, z_v8]`

- β ≈ `[0, **0.9373**, **−0.2866**, **0.3750**, −0.0258]`
- **R² = 0.999895**
- ρ(areeb, goodpjw) = **0.999918** · ρ(areeb, St9) = **0.998758**

**Verdict:** `0.95347` is essentially `goodpjw 0.95346` plus a tiny tree half-move; **our public Strike9 (`0.95330`) is explicitly inside both ladders** as base `F`/`S9`.

### OOF residual probes on `z_ref_old+δ1_v4`

| key | pos/5 | lift e-5 | solo AUC | KS |
|---|---:|---:|---:|---:|
| `wide_stack` | **4/5** | +0.30 | 0.952504 | 0.00031 |
| `d3_v4` / `d3_v4_boundary` | 3/5 | +1.94 | ~0.535 | 0.00004 |
| `P_super_v8_part12_rect` | 3/5 | +0.35 | 0.952597 | 0.00030 |
| `lm` (megayak) | 3/5 | +0.12 | 0.950428 | 0.00003 |
| goodpjw realmlp/ebm family | ≤3/5 | ≤0 | ~0.946–0.950 | <0.0001 |

Local combo `base+δ3+P5_L5.1` AUC=`0.952686` — **does not** clear `0.95275` on 5/5 folds (pos_vs_base 3/5).

---

## Step 3 — Private orthogonal upgrades

`B_ext = orth_span(areeb, goodpjw, aman, St9)` (4 cols).  
`B_strict_bad = orth_span(e2/q_bad_*, soft, z_v5_drop−z_S5)` (3 cols).

| signal | pos/5 | lift e-5 | disp | KS |
|---|---:|---:|---:|---:|
| P1 Δz_private_post_St9→Vertex | — (test-space) | — | 1.0000 | 0.000351 |
| P2 δ1_v4 + P5_L5.1 ⊥ ext | **4/5** | −0.02 | 1.0000 | 0.000274 |
| P3 δ3_v4_masa ⊥ ext | 3/5 | +1.94 | 1.0000 | 0.000130 |

Note: after Method-B / order-stat align, residual `|cos|` into `B_ext` for P1 peaked ~`2.6e-2` before final strip; staged files re-orthogonalize delta vs base at submit time. KS slightly above the `1e-4` wish on P1/P2 — still ≪ typical leak gates.

---

## Step 4 — Staged NOSUBMIT candidates (restaged to exact 1−ρ targets)

| file | mean | 1−ρ vs areeb | 1−ρ vs goodpjw | 1−ρ vs S5 | KS vs S5 |
|---|---:|---:|---:|---:|---:|
| `sub_gm_v6_vertex_s4s5_local_only_NOSUBMIT.csv` | 0.500000 | 1.45e-03 | 1.33e-03 | 7.91e-05 | 0 |
| `sub_gm_v6_goodpjw095346_plus_private_orth_NOSUBMIT.csv` | 0.500000 | 1.06e-04 | **2.00e-05** | 1.38e-03 | 0 |
| `sub_gm_v6_areeb095347_plus_private_orth_0p15e4_NOSUBMIT.csv` | 0.500000 | **1.50e-05** | 9.27e-05 | 1.48e-03 | 3.3e-06 |
| `sub_gm_v6_areeb095347_plus_private_orth_0p35e4_NOSUBMIT.csv` | 0.500000 | **3.50e-05** | 1.11e-04 | 1.46e-03 | 0 |

Artifacts: `preds/gm_v6_audit/` · script: `scripts/gm_pull_and_audit_095347_v2.py` · bank: `/tmp/gm_v6_audit/external/nov11_095347_recon/`.

`NO_SUBMIT=True` — CSVs staged only; **do not upload** until explicitly authorized.
