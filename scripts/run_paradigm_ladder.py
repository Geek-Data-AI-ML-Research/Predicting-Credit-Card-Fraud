#!/usr/bin/env python3
"""Fold-0 Gated Autonomous Research Ladder → target 0.9530+ Nested CV.
Policy: NO_SUBMIT. Progress → logs/overnight/paradigm_ladder_progress.json
Pod: 5llt7gof2prbur (RTX A5000). Kill Lab2/3 first; shut down pod after Rung 5.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

PRED = ROOT / "preds" / "ladder"
OUT = ROOT / "logs" / "overnight"
PROGRESS = OUT / "paradigm_ladder_progress.json"
PRED.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "is_fraud"
ID = "id"
POLICY = "NO_SUBMIT"
GATE1_AUC, GATE1_DS = 0.94935, 4e-5
GATE2_AUC, GATE2_DS = 0.94980, 8e-5
GATE3_AUC = 0.95080
GATE4_AUC = 0.95080
DS_PROMOTE = 3e-5
T0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "paradigm_ladder.log", "a") as fh:
        fh.write(msg + "\n")


def rk(v):
    v = np.asarray(v, float).ravel()
    return rankdata(v) / len(v)


def rlogit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def folds_y(y, seed=42):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def save_progress(rep):
    rep["policy"] = POLICY
    rep["runtime_s"] = time.time() - T0
    rep["device"] = str(DEVICE)
    tmp = PROGRESS.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rep, indent=2, default=float))
    tmp.replace(PROGRESS)


def load_progress():
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except Exception:
            pass
    return {"policy": POLICY, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rungs": {}}


# ── Masaya tabtext helpers (vendored for pod) ──────────────────────────────
def _ensure_tabtext():
    repo = ROOT / "external_bank" / "masaya_gpt2_repo"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def make_orders_k8(n_cols: int):
    """8 orthogonal-ish column permutations for Rung 1."""
    base = np.arange(n_cols, dtype=np.int16)
    orders = [
        base.copy(),  # 0 canonical
        base[::-1].copy(),  # 1 reverse
    ]
    # float-first: first 6 floats (amount..merchant_risk among NUMS) — use even indices first
    floatish = np.array([0, 1, 3, 6, 8, 11], dtype=np.int16)  # approximate float positions in FEATS
    floatish = floatish[floatish < n_cols]
    rest = np.array([i for i in range(n_cols) if i not in set(floatish)], dtype=np.int16)
    orders.append(np.concatenate([floatish, rest]))
    # cat/flag-first
    orders.append(np.concatenate([rest, floatish]))
    # 4 cyclic shifts
    for s in (3, 7, 11, 17):
        orders.append(np.roll(base, s))
    return [o.astype(np.int16) for o in orders[:8]]


# Inference batch for Rung 2 / Rung 5 / post-Gate1 enrich (A5000 has ~20GB free at score time)
SCORE_INFER_BS = int(os.environ.get("LADDER_SCORE_BS", "1536"))


@torch.inference_mode()
def fast_score_k_orders(
    model,
    tok,
    tokens,
    prefix_t,
    prefix_f,
    orders,
    score_idx: np.ndarray,
    n_rows_held: int,
    n_cols: int,
    *,
    bs: int | None = None,
    tag: str = "score",
):
    """Score held rows under K column orders with dual y=1/y=0 prefixes in one forward.

    Concatenates prefix batches along batch dim, bfloat16 + inference_mode.
    Returns (llr[n_held], delta[n_held,n_cols], llr_per_order[K,n_held]).
    """
    bs = int(bs or SCORE_INFER_BS)
    n_prefix = len(prefix_t)
    K = len(orders)
    llr = np.zeros(n_rows_held, dtype=np.float64)
    delta = np.zeros((n_rows_held, n_cols), dtype=np.float64)
    llr_po = np.zeros((K, n_rows_held), dtype=np.float64)
    score_idx = np.asarray(score_idx, dtype=np.int64)
    eos = tok.eos_token_id

    def body_spans(r, order):
        pos = 0
        sp = []
        for jpos, j in enumerate(order):
            plen = len(tokens.pieces[j][tokens.codes[r, j]])
            sp.append((pos, pos + plen))
            pos += plen
            if jpos < len(order) - 1:
                pos += len(tokens.comma_ids)
        return sp

    for oi, order in enumerate(orders):
        t0 = time.time()
        log(f"    {tag} order {oi}/{K} bs={bs} …")
        for s0 in range(0, len(score_idx), bs):
            rows = [int(x) for x in score_idx[s0 : s0 + bs]]
            bodies = [tokens.body_ids(r, order) for r in rows]
            spans = [body_spans(r, order) for r in rows]
            L = max(len(b) for b in bodies)
            B = len(rows)
            # Dual prefix: [y=1 batch || y=0 batch] along batch dim
            ids = torch.full((2 * B, n_prefix + L), eos, dtype=torch.long, device=DEVICE)
            for i, body in enumerate(bodies):
                seq_t = prefix_t + body
                seq_f = prefix_f + body
                ids[i, : len(seq_t)] = torch.tensor(seq_t, dtype=torch.long, device=DEVICE)
                ids[B + i, : len(seq_f)] = torch.tensor(seq_f, dtype=torch.long, device=DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                hidden = model.transformer(input_ids=ids, use_cache=False).last_hidden_state
            # token logprobs for feature body (after prefix)
            h = hidden[:, n_prefix - 1 : -1].float()
            tgt = ids[:, n_prefix:]
            flat_h = h.reshape(-1, h.shape[-1])
            flat_t = tgt.reshape(-1)
            tok_lp = torch.empty(flat_t.shape, dtype=torch.float64, device=DEVICE)
            for s in range(0, flat_h.shape[0], 8192):
                e = min(flat_h.shape[0], s + 8192)
                lp = model.lm_head(flat_h[s:e]).log_softmax(-1)
                tok_lp[s:e] = lp.gather(1, flat_t[s:e, None])[:, 0].double()
            tok_lp = tok_lp.view(2 * B, -1)
            lp_t, lp_f = tok_lp[:B], tok_lp[B:]
            for i, r in enumerate(rows):
                blen = len(bodies[i])
                d = float(lp_t[i, :blen].sum() - lp_f[i, :blen].sum())
                llr[r] += d
                llr_po[oi, r] = d
                for cj, (lo, hi) in enumerate(spans[i]):
                    delta[r, int(order[cj])] += float(lp_t[i, lo:hi].sum() - lp_f[i, lo:hi].sum())
            del ids, hidden, h, tok_lp, lp_t, lp_f
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        log(f"    {tag} order {oi}/{K} done in {time.time() - t0:.1f}s")
    llr /= K
    delta /= K
    return llr, delta, llr_po


def ridge_oof_delta(delta_si, y_si, seed=0):
    rng = np.random.default_rng(seed)
    inner = rng.permutation(len(y_si))
    mid = len(y_si) // 2
    tr_i, va_i = inner[:mid], inner[mid:]
    sc = StandardScaler().fit(delta_si[tr_i])
    score = np.zeros(len(y_si))
    score[va_i] = Ridge(alpha=1.0).fit(sc.transform(delta_si[tr_i]), y_si[tr_i]).predict(
        sc.transform(delta_si[va_i])
    )
    score[tr_i] = Ridge(alpha=1.0).fit(sc.transform(delta_si[va_i]), y_si[va_i]).predict(
        sc.transform(delta_si[tr_i])
    )
    return score, float(roc_auc_score(y_si, score))


def enrich_rung1_perm_curve(rep: dict) -> dict:
    """Post-Gate1: fast re-score cached Fold-0 GPT-2 to log cumulative K AUC curve.

    Non-destructive: does not retrain; uses rung1_fold0_gpt2 + saved score_idx if present.
    """
    r1 = (rep.get("rungs") or {}).get("rung1") or {}
    if r1.get("perm_auc_curve"):
        log("  perm_auc_curve already logged — skip enrich")
        return rep
    model_dir = PRED / "rung1_fold0_gpt2"
    if not (model_dir / "config.json").exists():
        log("  enrich skip: no cached rung1 model")
        return rep
    log("=" * 72)
    log("POST-GATE1 ENRICH: cumulative K∈{1,2,4,6,8} perm AUC + δ24 Ridge (fast bs)")
    log("=" * 72)
    _ensure_tabtext()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tabtext import RowTokens, feature_columns, label_prefix_ids, read_csv_as_strings

    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]
    train = read_csv_as_strings(ROOT / "data" / "train.csv")
    cols = feature_columns(train, ID, TARGET)
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE).eval()
    held = train.iloc[b0].reset_index(drop=True)
    tokens = RowTokens(held, cols, tok)
    prefix_t, prefix_f = label_prefix_ids(tok, TARGET)
    orders = make_orders_k8(len(cols))
    n_b = len(held)
    idx_p = PRED / "rung1_score_idx.npy"
    if idx_p.exists():
        score_idx = np.load(idx_p).astype(np.int64)
    else:
        SCORE_N = min(n_b, 16_000)
        rng_s = np.random.default_rng(7)
        y_b = y[b0]
        pos, neg = np.where(y_b == 1)[0], np.where(y_b == 0)[0]
        n_pos = min(len(pos), max(200, int(SCORE_N * y_b.mean())))
        n_neg = min(len(neg), SCORE_N - n_pos)
        score_idx = np.concatenate(
            [rng_s.choice(pos, n_pos, replace=False), rng_s.choice(neg, n_neg, replace=False)]
        )
        rng_s.shuffle(score_idx)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    enrich_bs = min(SCORE_INFER_BS, 256)
    llr = delta = llr_po = None
    last_err = None
    for try_bs in (enrich_bs, 128, 64, 32):
        try:
            log(f"  enrich scoring n={len(score_idx)} K=8 bs={try_bs}")
            llr, delta, llr_po = fast_score_k_orders(
                model,
                tok,
                tokens,
                prefix_t,
                prefix_f,
                orders,
                score_idx,
                n_b,
                len(cols),
                bs=try_bs,
                tag="enrich",
            )
            enrich_bs = try_bs
            break
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            log(f"  enrich OOM at bs={try_bs}; retrying smaller")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if llr is None:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        raise last_err or RuntimeError("enrich scoring failed")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    prior = float(y[a0].mean())
    logp = math.log(prior / (1 - prior))
    si = score_idx
    y_si = y[b0][si]
    curve = {}
    for k in (1, 2, 4, 6, 8):
        # cumulative mean of first k order LLRs
        s = llr_po[:k, si].mean(0) + logp
        curve[f"K={k}"] = float(roc_auc_score(y_si, s))
        log(f"  cum-perm AUC {k=} → {curve[f'K={k}']:.5f}")
    score_w_si, auc_w = ridge_oof_delta(delta[si], y_si)
    log(f"  δ24 Ridge Fold-0 AUC={auc_w:.5f}")
    full = np.load(ROOT / "external_bank" / "masaya_gpt2" / "oof.npy").ravel().astype(np.float64)
    full[b0[si]] = score_w_si
    S_with, dS = stack_delta_full(full, y, folds, rep)
    S_base = float(rep.get("S_fold0_base") or 0.950841)
    log(f"  S_fold0={S_with:.6f}  S_fold0_base={S_base:.6f}  ΔS={dS:+.6f}")

    np.save(PRED / "delta24_fold0.npy", delta.astype(np.float32))
    np.save(PRED / "rung1_llr_per_order.npy", llr_po.astype(np.float32))
    np.save(PRED / "rung1_score_idx.npy", si.astype(np.int32))
    np.save(PRED / "rung1_score_full.npy", full.astype(np.float32))
    np.save(PRED / "rung1_llr_fold0.npy", (llr).astype(np.float32))

    r1 = dict(r1)
    r1["perm_auc_curve"] = curve
    r1["auc_sum_llr"] = curve.get("K=8", r1.get("auc_sum_llr"))
    r1["auc_weighted_delta"] = auc_w
    r1["S_fold0"] = S_with
    r1["Delta_S_fold0"] = dS
    r1["S_fold0_base"] = S_base
    r1["score_n"] = int(len(si))
    r1["enrich_bs"] = enrich_bs
    r1["passed"] = bool(auc_w >= GATE1_AUC or dS >= GATE1_DS)
    rep.setdefault("rungs", {})["rung1"] = r1
    save_progress(rep)
    log(f"  GATE1 enrich logged  passed={r1['passed']}  curve={curve}")
    return rep


# ═══════════════════════════════════════════════════════════════════════════
# RUNG 0
# ═══════════════════════════════════════════════════════════════════════════
def rung0(rep: dict) -> dict:
    log("=" * 72)
    log("RUNG 0: Harvest glm_quad · inspect Masaya · Fold-0 stack baseline")
    log("=" * 72)
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]

    # Harvest glm_quad: recompute K=800 C=0.001 if missing
    gq_oof_p = PRED / "glm_quad_oof.npy"
    gq_te_p = PRED / "glm_quad_test.npy"
    harvest = {"had_npy": gq_oof_p.exists(), "logged_auc": 0.950836}
    if not gq_oof_p.exists():
        log("  recomputing glm_quad K=800 C=0.001 (harvest) …")
        os.environ["INPUT_DIR"] = str(ROOT)
        import gboost as gb
        from scipy.special import expit

        gpt_o = np.load(ROOT / "external_bank" / "masaya_gpt2" / "oof.npy").ravel()
        gpt_t = np.load(ROOT / "external_bank" / "masaya_gpt2" / "test.npy").ravel()
        parts = []
        z_joint = np.zeros(len(y))
        zte_f = []
        for fi, (a, b) in enumerate(folds):
            Ea, Eb, Et = gb.te_fold(a, b, smooth=20.0, seed=0)
            if gb.GEN:
                Ga, Gb, Gt = gb.gen_fold(a, b, seed=0)
                Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
            ybar = float(y[a].mean())
            scz = StandardScaler().fit(rlogit(Ea) - rlogit(ybar))
            Za = scz.transform(rlogit(Ea) - rlogit(ybar))
            Zb = scz.transform(rlogit(Eb) - rlogit(ybar))
            Zt = scz.transform(rlogit(Et) - rlogit(ybar))
            Xa = np.hstack([Za, gb.BASE[a]])
            Xb = np.hstack([Zb, gb.BASE[b]])
            Xt = np.hstack([Zt, gb.BASE[gb.TEST]])
            m = LogisticRegression(C=0.3, max_iter=200, solver="newton-cholesky").fit(Xa, y[a])
            z_joint[b] = m.decision_function(Xb) + rlogit(rk(gpt_o))[b]
            zte_f.append(m.decision_function(Xt) + rlogit(rk(gpt_t)))
            parts.append((a, b, Za, Zb, Zt))
            log(f"    fold{fi} harvest TE ready D={Za.shape[1]}")
        z_te = np.mean(zte_f, 0)
        # screen products on fold-0 train
        a, b, Za, Zb, Zt = parts[0]
        resid = y[a] - expit(z_joint[a])
        Zc = Za - Za.mean(0)
        r = resid - resid.mean()
        wr = Zc * r[:, None]
        scores, pairs = [], []
        d = Za.shape[1]
        for j in range(d):
            s = (wr[:, j : j + 1] * Zc[:, j + 1 :]).mean(0)
            for t, k in enumerate(range(j + 1, d)):
                scores.append(abs(float(s[t])))
                pairs.append((j, k))
        pairs = [pairs[i] for i in np.argsort(-np.asarray(scores))[:800]]
        log(f"  screened {len(pairs)} product pairs")
        oof = np.zeros(len(y))
        te_fs = []
        for a, b, Za, Zb, Zt in parts:
            Ia = np.column_stack([Za[:, j] * Za[:, k] for j, k in pairs])
            Ib = np.column_stack([Zb[:, j] * Zb[:, k] for j, k in pairs])
            It = np.column_stack([Zt[:, j] * Zt[:, k] for j, k in pairs])
            scI = StandardScaler().fit(Ia)
            Ia, Ib, It = scI.transform(Ia), scI.transform(Ib), scI.transform(It)
            m = LogisticRegression(C=0.001, max_iter=400, solver="lbfgs").fit(Ia, y[a])
            oof[b] = expit(z_joint[b] + 0.5 * m.decision_function(Ib))
            te_fs.append(expit(z_te + 0.5 * m.decision_function(It)))
        te = np.mean(te_fs, 0)
        auc = float(roc_auc_score(y, oof))
        np.save(gq_oof_p, oof.astype(np.float32))
        np.save(gq_te_p, te.astype(np.float32))
        harvest["recomputed_auc"] = auc
        log(f"  glm_quad harvested OOF={auc:.6f}")
    else:
        oof = np.load(gq_oof_p).ravel()
        harvest["recomputed_auc"] = float(roc_auc_score(y, oof))
        log(f"  glm_quad cached OOF={harvest['recomputed_auc']:.6f}")

    # Masaya inventory
    code_dir = ROOT / "external_bank" / "masaya_gpt2_code"
    out_dir = ROOT / "external_bank" / "masaya_gpt2_out"
    repo_dir = ROOT / "external_bank" / "masaya_gpt2_repo"
    masaya = {
        "code_files": sorted(p.name for p in code_dir.glob("*") if not p.name.startswith("._")),
        "out_files": sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("._"))[:20],
        "repo_scripts": sorted(p.name for p in repo_dir.glob("*.py")) if repo_dir.exists() else [],
        "row_text_rule": "is_fraud is {True|False}, {col} is {csv_value}, ...<|endoftext|>",
        "combine": "log-sum-exp over orders per label, then difference + fold prior",
        "fold0_auc": 0.9488823016904258,
        "oof_auc": 0.9500272037932737,
        "orders_in_public": 10,
    }
    gpt_oof = np.load(ROOT / "external_bank" / "masaya_gpt2" / "oof.npy").ravel()
    masaya["fold0_auc_verified"] = float(roc_auc_score(y[b0], gpt_oof[b0]))
    log(f"  Masaya Fold-0 AUC verified={masaya['fold0_auc_verified']:.5f} (ref 0.94888)")

    # Fold-0 stack gate baseline: Part9B-style stacker fit on folds 1-4
    log("  building Fold-0 stack baseline S_fold0_base …")
    try:
        from part9_joint_boost_and_basis import build_8b_champ

        names, O, T = build_8b_champ(y, folds)
        X = rlogit(O)
        fit_idx = np.concatenate([folds[i][1] for i in range(1, 5)])  # use OOF rows of folds 1-4 as train?
        # Correct protocol: fit stacker on train indices of folds≠0 i.e. rows in a0, predict b0
        # But O is OOF — for stacker we fit on OOF of folds 1-4 (held-out from those folds) using their labels,
        # OR fit on a0 using in-fold predictions. Part7 nested uses: for outer fold b, fit on a.
        # Here Fold-0 gate: fit LogisticRegression on O[a0], y[a0], eval on O[b0].
        m = LogisticRegression(C=0.005, tol=1e-8, max_iter=200, solver="newton-cholesky").fit(X[a0], y[a0])
        s_b0 = m.decision_function(X[b0])
        S_fold0_base = float(roc_auc_score(y[b0], s_b0))
        stack_meta = {"source": "8b_champ", "n_members": len(names), "C": 0.005}
    except Exception as e:
        log(f"  8B unavailable ({e}); using P_super + gpt2 + glm_quad ranks")
        cols = [rk(np.load(ROOT / "preds" / "P_super_champ_oof.npy").ravel())]
        cols.append(rk(gpt_oof))
        cols.append(rk(np.load(gq_oof_p).ravel()))
        O = np.column_stack(cols)
        X = rlogit(O)
        m = LogisticRegression(C=0.005, tol=1e-8, max_iter=200, solver="newton-cholesky").fit(X[a0], y[a0])
        s_b0 = m.decision_function(X[b0])
        S_fold0_base = float(roc_auc_score(y[b0], s_b0))
        stack_meta = {"source": "p_super_fallback", "n_members": O.shape[1], "C": 0.005, "error": str(e)}
        # stash for later gates
        np.save(PRED / "stack_O_fallback.npy", O.astype(np.float32))

    np.save(PRED / "fold0_stack_base_scores.npy", s_b0.astype(np.float32))
    log(f"  S_fold0_base={S_fold0_base:.6f}")

    out = {
        "name": "rung0",
        "harvest": harvest,
        "masaya": masaya,
        "S_fold0_base": S_fold0_base,
        "stack_meta": stack_meta,
        "n_b0": int(len(b0)),
        "passed": True,
    }
    rep["rungs"]["rung0"] = out
    rep["S_fold0_base"] = S_fold0_base
    save_progress(rep)
    return out


def _stack_delta(new_oof_b0, y_b0, a0, b0, y, rep):
    """Fit stacker on a0 with [base_cols | new], eval Δ on b0."""
    S_base = rep["S_fold0_base"]
    # Reload base matrix used in rung0
    try:
        from part9_joint_boost_and_basis import build_8b_champ

        folds = folds_y(y)
        _, O, _ = build_8b_champ(y, folds)
    except Exception:
        O = np.load(PRED / "stack_O_fallback.npy")
    X = rlogit(O)
    Xn = np.column_stack([X, rlogit(rk(new_oof_b0) if False else rk(
        # need full-length column: put scores on b0, zeros elsewhere then rank — better pass full oof
        new_oof_b0
    ))])
    # If new_oof_b0 is only fold0 length, build full vector
    if len(new_oof_b0) == len(b0):
        full = np.zeros(len(y))
        full[b0] = new_oof_b0
        # For fitting on a0 we need a0 scores — use mean of new as weak; require caller pass full OOF
        # Expect caller passes full-length scores with a0 filled via inner OOF or train scores
        col = rk(full)
    else:
        col = rk(new_oof_b0)
    X2 = np.column_stack([X, rlogit(col)])
    m = LogisticRegression(C=0.005, tol=1e-8, max_iter=200, solver="newton-cholesky").fit(X2[a0], y[a0])
    auc = float(roc_auc_score(y[b0], m.decision_function(X2[b0])))
    return auc, auc - S_base


def stack_delta_full(new_full_oof, y, folds, rep):
    a0, b0 = folds[0]
    try:
        from part9_joint_boost_and_basis import build_8b_champ

        _, O, _ = build_8b_champ(y, folds)
    except Exception:
        O = np.load(PRED / "stack_O_fallback.npy")
    X = rlogit(O)
    X2 = np.column_stack([X, rlogit(rk(new_full_oof))])
    m = LogisticRegression(C=0.005, tol=1e-8, max_iter=200, solver="newton-cholesky").fit(X2[a0], y[a0])
    auc = float(roc_auc_score(y[b0], m.decision_function(X2[b0])))
    return auc, auc - rep["S_fold0_base"]


# ═══════════════════════════════════════════════════════════════════════════
# RUNG 1 — Multi-perm LLR + column-weighted δ
# ═══════════════════════════════════════════════════════════════════════════
def rung1(rep: dict) -> dict:
    log("=" * 72)
    log("RUNG 1: Multi-perm Bayes LLR K=8 + column-weighted δ")
    log("=" * 72)
    _ensure_tabtext()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tabtext import RowTokens, feature_columns, label_prefix_ids, read_csv_as_strings

    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]
    train = read_csv_as_strings(ROOT / "data" / "train.csv")
    cols = feature_columns(train, ID, TARGET)

    # Train Fold-0 GPT-2 classifier from gpt2 base (skip GReaT replica for gate speed)
    model_dir = PRED / "rung1_fold0_gpt2"
    model_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    if not (model_dir / "config.json").exists():
        # Fold-0 gate budget: stratified subsample of a0 (~24k) × 2 epochs ≈ 8-10 min on A5000
        GATE_N = 24_000
        rng = np.random.default_rng(42)
        y_a0 = y[a0]
        pos = np.where(y_a0 == 1)[0]
        neg = np.where(y_a0 == 0)[0]
        n_pos = min(len(pos), max(500, int(GATE_N * y_a0.mean())))
        n_neg = min(len(neg), GATE_N - n_pos)
        sub = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(sub)
        fit = train.iloc[a0[sub]].reset_index(drop=True)
        log(f"  fine-tuning GPT-2 on Fold-0 subsample n={len(fit)} (2 ep, bf16, bs=16) …")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
        labels = (fit[TARGET].map(lambda v: v.strip() in ("1", "1.0", "True")).to_numpy())
        tokens = RowTokens(fit, cols, tok)
        prefix_t, prefix_f = label_prefix_ids(tok, TARGET)
        n_prefix = len(prefix_t)
        n_rows = len(fit)
        bs = 16 if DEVICE.type == "cuda" else 4
        steps_per_ep = math.ceil(n_rows / bs)
        total_steps = steps_per_ep * 2
        step = 0
        use_bf16 = DEVICE.type == "cuda"
        for ep in range(2):
            perm = rng.permutation(n_rows)
            for i0 in range(0, n_rows, bs):
                rows = perm[i0 : i0 + bs]
                orders = np.array([rng.permutation(len(cols)) for _ in rows], dtype=np.int16)
                seqs = []
                for i, r in enumerate(rows):
                    pref = prefix_t if labels[r] else prefix_f
                    seqs.append(pref + tokens.body_ids(int(r), orders[i]))
                L = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), L), tok.eos_token_id, dtype=torch.long, device=DEVICE)
                attn = torch.zeros((len(seqs), L), dtype=torch.long, device=DEVICE)
                targets = torch.full((len(seqs), L), -100, dtype=torch.long, device=DEVICE)
                for i, s in enumerate(seqs):
                    input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=DEVICE)
                    attn[i, : len(s)] = 1
                    targets[i, : len(s)] = input_ids[i, : len(s)]
                lr = 5e-5 * max(0.0, (total_steps - step) / total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                    logits = out.logits
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    targets[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 200 == 0:
                    log(f"    step {step}/{total_steps} loss={float(loss.detach()):.4f}")
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        log(f"  saved {model_dir}")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log(f"  loading cached {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE)

    model.eval()
    # Score Fold-0 val with K=8 orders + per-column δ
    log("  scoring Fold-0 with K=8 orders + per-column LLR …")
    held = train.iloc[b0].reset_index(drop=True)
    tokens = RowTokens(held, cols, tok)
    prefix_t, prefix_f = label_prefix_ids(tok, TARGET)
    n_prefix = len(prefix_t)
    orders = make_orders_k8(len(cols))
    n_b = len(held)
    n_c = len(cols)
    # δ[row, col] and total LLR
    delta = np.zeros((n_b, n_c), dtype=np.float64)
    llr = np.zeros(n_b, dtype=np.float64)

    @torch.inference_mode()
    def seq_token_logprobs(input_ids, n_pref):
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            hidden = model.transformer(input_ids=input_ids, use_cache=False).last_hidden_state
        hidden = hidden[:, n_pref - 1 : -1].float()
        targets = input_ids[:, n_pref:]
        # chunked lm_head
        flat_h = hidden.reshape(-1, hidden.shape[-1])
        flat_t = targets.reshape(-1)
        out = torch.empty(flat_t.shape, dtype=torch.float64, device=DEVICE)
        for s in range(0, flat_t.shape[0], 4096):
            e = s + 4096
            lp = model.lm_head(flat_h[s:e]).log_softmax(-1)
            out[s:e] = lp.gather(1, flat_t[s:e, None])[:, 0].double()
        return out.view(targets.shape)  # B x T_body

    # Score-time VRAM is low (~3–4GB at bs=8); raise batch for gate throughput.
    # Stratified subsample of b0 for Fold-0 gate (~40k) — full 5-fold promotion rescoring later.
    SCORE_N = min(n_b, 16_000)  # ~2 min/order on A5000; gate-quality AUC
    if SCORE_N < n_b:
        rng_s = np.random.default_rng(7)
        y_b = y[b0]
        pos = np.where(y_b == 1)[0]
        neg = np.where(y_b == 0)[0]
        n_pos = min(len(pos), max(200, int(SCORE_N * y_b.mean())))
        n_neg = min(len(neg), SCORE_N - n_pos)
        score_idx = np.concatenate(
            [rng_s.choice(pos, n_pos, replace=False), rng_s.choice(neg, n_neg, replace=False)]
        )
        rng_s.shuffle(score_idx)
        log(f"  gate-score subsample {len(score_idx)}/{n_b} of Fold-0 val")
    else:
        score_idx = np.arange(n_b)
    bs = 96 if DEVICE.type == "cuda" else 8
    for oi, order in enumerate(orders):
        log(f"    order {oi}/8 …")
        for s0 in range(0, len(score_idx), bs):
            rows = [int(x) for x in score_idx[s0 : s0 + bs]]
            bodies = [tokens.body_ids(r, order) for r in rows]
            spans = []
            for r in rows:
                pos = 0
                sp = []
                for jpos, j in enumerate(order):
                    plen = len(tokens.pieces[j][tokens.codes[r, j]])
                    sp.append((pos, pos + plen))
                    pos += plen
                    if jpos < len(order) - 1:
                        pos += len(tokens.comma_ids)
                spans.append(sp)
            L = max(len(b) for b in bodies)

            def pack(pref):
                ids = torch.full((len(rows), len(pref) + L), tok.eos_token_id, dtype=torch.long, device=DEVICE)
                for i, body in enumerate(bodies):
                    seq = pref + body
                    ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=DEVICE)
                return ids

            ids_t = pack(prefix_t)
            lp_t = seq_token_logprobs(ids_t, n_prefix)
            del ids_t
            ids_f = pack(prefix_f)
            lp_f = seq_token_logprobs(ids_f, n_prefix)
            del ids_f
            for i, r in enumerate(rows):
                blen = len(bodies[i])
                llr[r] += float(lp_t[i, :blen].sum() - lp_f[i, :blen].sum())
                for cj, (lo, hi) in enumerate(spans[i]):
                    col = int(order[cj])
                    delta[r, col] += float(lp_t[i, lo:hi].sum() - lp_f[i, lo:hi].sum())
            del lp_t, lp_f
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    # Restrict metrics to scored rows
    scored_mask = np.zeros(n_b, dtype=bool)
    scored_mask[score_idx] = True

    llr /= len(orders)
    delta /= len(orders)
    prior = float(y[a0].mean())
    si = score_idx  # local indices into b0 / held
    y_si = y[b0][si]
    score_sum = llr[si] + math.log(prior / (1 - prior))
    auc_sum = float(roc_auc_score(y_si, score_sum))
    log(f"  K=8 sum-LLR Fold-0 AUC={auc_sum:.5f} (n={len(si)})")

    # Ridge over δ with inner half-split on scored rows
    rng = np.random.default_rng(0)
    inner = rng.permutation(len(si))
    mid = len(si) // 2
    tr_i, va_i = inner[:mid], inner[mid:]
    d_si = delta[si]
    sc = StandardScaler().fit(d_si[tr_i])
    ridge = Ridge(alpha=1.0).fit(sc.transform(d_si[tr_i]), y_si[tr_i])
    score_w_si = np.zeros(len(si))
    score_w_si[va_i] = ridge.predict(sc.transform(d_si[va_i]))
    ridge2 = Ridge(alpha=1.0).fit(sc.transform(d_si[va_i]), y_si[va_i])
    score_w_si[tr_i] = ridge2.predict(sc.transform(d_si[tr_i]))
    auc_w = float(roc_auc_score(y_si, score_w_si))
    log(f"  column-weighted δ Fold-0 AUC={auc_w:.5f}")

    score_w = np.zeros(n_b)
    score_w[si] = score_w_si
    # Map scored Fold-0 rows into full OOF; fill rest with masaya gpt proxy
    full_scores = np.load(ROOT / "external_bank" / "masaya_gpt2" / "oof.npy").ravel().astype(np.float64)
    full_scores[b0[si]] = score_w_si
    S_with, dS = stack_delta_full(full_scores, y, folds, rep)
    log(f"  stack Fold-0 AUC={S_with:.6f} ΔS={dS:+.6f}")

    np.save(PRED / "delta24_fold0.npy", delta.astype(np.float32))
    np.save(PRED / "rung1_score_idx.npy", si.astype(np.int32))
    np.save(PRED / "rung1_llr_fold0.npy", score_w.astype(np.float32))
    np.save(PRED / "rung1_score_full.npy", full_scores.astype(np.float32))

    passed = bool(auc_w >= GATE1_AUC or dS >= GATE1_DS)
    out = {
        "name": "rung1",
        "auc_sum_llr": auc_sum,
        "auc_weighted_delta": auc_w,
        "S_fold0": S_with,
        "Delta_S_fold0": dS,
        "gate_auc": GATE1_AUC,
        "gate_ds": GATE1_DS,
        "passed": passed,
        "n_orders": 8,
        "score_n": int(len(si)),
    }
    rep["rungs"]["rung1"] = out
    save_progress(rep)
    log(f"  GATE1 {'PASS' if passed else 'FAIL'}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RUNG 2 — Transductive pretrain + contrastive LLR
# ═══════════════════════════════════════════════════════════════════════════
def rung2(rep: dict) -> dict:
    log("=" * 72)
    log("RUNG 2: Transductive test pretrain + contrastive LLR loss")
    log("=" * 72)
    _ensure_tabtext()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tabtext import RowTokens, feature_columns, label_prefix_ids, read_csv_as_strings

    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]
    train = read_csv_as_strings(ROOT / "data" / "train.csv")
    test = read_csv_as_strings(ROOT / "data" / "test.csv")
    cols = feature_columns(train, ID, TARGET)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    model_dir = PRED / "rung2_fold0_contrastive"
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "config.json").exists():
        # init from rung1 if available else gpt2
        init = PRED / "rung1_fold0_gpt2"
        src = str(init) if (init / "config.json").exists() else "gpt2"
        log(f"  loading init={src}")
        model = AutoModelForCausalLM.from_pretrained(src).to(DEVICE)
        # Phase A: 1 epoch unconditional on test + a0 (target-first with random labels? use LM on features only)
        # Simpler: LM on target-first with TRUE label for a0 and False-prior for test soft
        log("  Phase A: 1-epoch transductive LM on sampled test∪a0 (gate budget) …")
        fit = train.iloc[a0].reset_index(drop=True)
        # Cap unlabeled test to 40k for Fold-0 gate speed (~15-20 min total rung)
        rng = np.random.default_rng(1)
        te_idx = rng.choice(len(test), size=min(40_000, len(test)), replace=False)
        te_s = test.iloc[te_idx].reset_index(drop=True)
        pool = pd.concat([fit[cols + [TARGET]], te_s[cols].assign(**{TARGET: "0"})], ignore_index=True)
        labels = (pool[TARGET].map(lambda v: str(v).strip() in ("1", "1.0", "True")).to_numpy())
        tokens = RowTokens(pool, cols, tok)
        prefix_t, prefix_f = label_prefix_ids(tok, TARGET)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
        model.train()
        bs = 16 if DEVICE.type == "cuda" else 4
        n = len(pool)
        perm = rng.permutation(n)
        log(f"    pool n={n} bs={bs}")
        for i0 in range(0, n, bs):
            rows = perm[i0 : i0 + bs]
            orders = np.array([rng.permutation(len(cols)) for _ in rows], dtype=np.int16)
            seqs = []
            for i, r in enumerate(rows):
                pref = prefix_t if labels[r] else prefix_f
                seqs.append(pref + tokens.body_ids(int(r), orders[i]))
            L = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), L), tok.eos_token_id, dtype=torch.long, device=DEVICE)
            attn = torch.zeros_like(input_ids)
            targets = torch.full_like(input_ids, -100)
            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, device=DEVICE)
                attn[i, : len(s)] = 1
                targets[i, : len(s)] = input_ids[i, : len(s)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            opt.step()
            if (i0 // bs) % 400 == 0:
                log(f"    pretrain {i0}/{n} loss={float(loss):.4f}")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        # Phase B: contrastive LLR on a0
        log("  Phase B: contrastive LLR fine-tune on a0 …")
        fit = train.iloc[a0].reset_index(drop=True)
        y_a = (fit[TARGET].map(lambda v: str(v).strip() in ("1", "1.0", "True")).to_numpy().astype(np.float32))
        tokens = RowTokens(fit, cols, tok)
        prior = float(y_a.mean())
        log_prior = math.log(prior / (1 - prior))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        n = len(fit)
        perm = rng.permutation(n)
        n_prefix = len(prefix_t)
        bs = 8 if DEVICE.type == "cuda" else 2  # dual forward — keep tiny
        max_steps = min(n, bs * 600)  # ~600 steps gate budget
        for i0 in range(0, max_steps, bs):
            rows = perm[i0 : i0 + bs]
            if len(rows) == 0:
                break
            orders = np.array([rng.permutation(len(cols)) for _ in rows], dtype=np.int16)
            bodies = [tokens.body_ids(int(r), orders[i]) for i, r in enumerate(rows)]
            L = max(len(b) for b in bodies)

            def pack(pref):
                ids = torch.full((len(rows), len(pref) + L), tok.eos_token_id, dtype=torch.long, device=DEVICE)
                for i, body in enumerate(bodies):
                    seq = pref + body
                    ids[i, : len(seq)] = torch.tensor(seq, device=DEVICE)
                return ids

            ids_t, ids_f = pack(prefix_t), pack(prefix_f)
            ya = torch.tensor(y_a[rows], device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                ids_true = torch.where(ya[:, None] > 0.5, ids_t, ids_f)
                logits = model(input_ids=ids_true, use_cache=False).logits
                targets = ids_true.clone()
                loss_lm = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    targets[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                h_t = model.transformer(input_ids=ids_t, use_cache=False).last_hidden_state
                h_f = model.transformer(input_ids=ids_f, use_cache=False).last_hidden_state

            def nll_sum(h, ids):
                hidden = h[:, n_prefix - 1 : -1].float()
                tgt = ids[:, n_prefix:]
                flat = hidden.reshape(-1, hidden.shape[-1])
                # chunk lm_head to limit peak VRAM
                out = torch.empty(tgt.shape[0] * tgt.shape[1], device=DEVICE, dtype=torch.float32)
                flat_t = tgt.reshape(-1)
                for s in range(0, flat.shape[0], 2048):
                    e = min(flat.shape[0], s + 2048)
                    lp = model.lm_head(flat[s:e]).log_softmax(-1)
                    out[s:e] = -lp.gather(1, flat_t[s:e, None])[:, 0]
                return out.view(tgt.shape).sum(1)

            nll_t = nll_sum(h_t, ids_t)
            nll_f = nll_sum(h_f, ids_f)
            llr = -nll_t + nll_f
            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(llr + log_prior, ya)
            loss = 0.3 * loss_lm + 0.7 * loss_bce
            loss.backward()
            opt.step()
            if (i0 // bs) % 100 == 0:
                log(f"    contrastive {i0} loss={float(loss):.4f} bce={float(loss_bce):.4f}")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE)
        tok = AutoTokenizer.from_pretrained(model_dir)

    # Score Fold-0 with K=8 — dual-prefix; fall back bs on OOM (1536 OOMs on A5000)
    model.eval()
    held = train.iloc[b0].reset_index(drop=True)
    tokens = RowTokens(held, cols, tok)
    prefix_t, prefix_f = label_prefix_ids(tok, TARGET)
    orders = make_orders_k8(len(cols))
    n_b = len(held)
    SCORE_N = min(n_b, 16_000)
    rng_s = np.random.default_rng(7)
    if SCORE_N < n_b:
        y_b = y[b0]
        pos, neg = np.where(y_b == 1)[0], np.where(y_b == 0)[0]
        n_pos = min(len(pos), max(200, int(SCORE_N * y_b.mean())))
        n_neg = min(len(neg), SCORE_N - n_pos)
        score_idx = np.concatenate(
            [rng_s.choice(pos, n_pos, replace=False), rng_s.choice(neg, n_neg, replace=False)]
        )
        rng_s.shuffle(score_idx)
        log(f"  gate-score subsample {len(score_idx)}/{n_b}")
    else:
        score_idx = np.arange(n_b)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    llr = delta = llr_po = None
    last_err = None
    score_bs = min(SCORE_INFER_BS, 256)
    for try_bs in (score_bs, 128, 64, 32):
        try:
            log(f"  scoring contrastive model Fold-0 K=8 (bs={try_bs}, dual-prefix) …")
            llr, delta, llr_po = fast_score_k_orders(
                model,
                tok,
                tokens,
                prefix_t,
                prefix_f,
                orders,
                score_idx,
                n_b,
                len(cols),
                bs=try_bs,
                tag="rung2",
            )
            score_bs = try_bs
            break
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            log(f"  rung2 score OOM at bs={try_bs}; retrying smaller")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if llr is None:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        raise last_err or RuntimeError("rung2 scoring failed")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    prior = float(y[a0].mean())
    logp = math.log(prior / (1 - prior))
    si = score_idx
    y_si = y[b0][si]
    curve = {f"K={k}": float(roc_auc_score(y_si, llr_po[:k, si].mean(0) + logp)) for k in (1, 2, 4, 6, 8)}
    for k, v in curve.items():
        log(f"  cum-perm AUC {k} → {v:.5f}")
    score_w_si, auc = ridge_oof_delta(delta[si], y_si)
    log(f"  contrastive weighted Fold-0 AUC={auc:.5f} (n={len(si)})")

    full = np.load(ROOT / "external_bank" / "masaya_gpt2" / "oof.npy").ravel().astype(np.float64)
    full[b0[si]] = score_w_si
    S_with, dS = stack_delta_full(full, y, folds, rep)
    beat_r1 = auc > float(rep.get("rungs", {}).get("rung1", {}).get("auc_weighted_delta", 0))
    passed = bool((auc >= GATE2_AUC or dS >= GATE2_DS) and beat_r1)
    np.save(PRED / "delta24_contrastive_fold0.npy", delta.astype(np.float32))
    np.save(PRED / "rung2_llr_per_order.npy", llr_po.astype(np.float32))
    np.save(PRED / "rung2_score_full.npy", full.astype(np.float32))
    out = {
        "name": "rung2",
        "auc_weighted_delta": auc,
        "auc_sum_llr": curve.get("K=8"),
        "perm_auc_curve": curve,
        "S_fold0": S_with,
        "Delta_S_fold0": dS,
        "beats_rung1": beat_r1,
        "passed": passed,
        "score_bs": score_bs,
        "score_n": int(len(si)),
    }
    rep["rungs"]["rung2"] = out
    save_progress(rep)
    log(f"  GATE2 {'PASS' if passed else 'FAIL'} ΔS={dS:+.6f}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RUNG 3 — Inject δ24 into gboost views
# ═══════════════════════════════════════════════════════════════════════════
def rung3(rep: dict) -> dict:
    log("=" * 72)
    log("RUNG 3: Inject δ24 into GLM + 4-leaf LGB + XGB + MLPRes")
    log("=" * 72)
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]
    # Prefer contrastive δ if present
    dpath = PRED / "delta24_contrastive_fold0.npy"
    if not dpath.exists():
        dpath = PRED / "delta24_fold0.npy"
    if not dpath.exists():
        log("  no delta24 — FAIL gate")
        out = {"name": "rung3", "passed": False, "reason": "missing_delta24"}
        rep["rungs"]["rung3"] = out
        save_progress(rep)
        return out
    delta_b0 = np.load(dpath)
    # Need δ on a0 as well — train a cheap proxy: use zeros on a0 for injection into models
    # trained only on a0 with δ=0, then apply real δ on b0 (honest for tree that can use δ at val)
    # Better: score a0 with rung1/2 model quickly — if missing, tile mean
    delta_a0 = np.zeros((len(a0), delta_b0.shape[1]), dtype=np.float32)

    os.environ["INPUT_DIR"] = str(ROOT)
    import lightgbm as lgb
    import gboost as gb
    from scipy.special import expit

    Ea, Eb, Et = gb.te_fold(a0, b0, smooth=20.0, seed=0)
    if gb.GEN:
        Ga, Gb, Gt = gb.gen_fold(a0, b0, seed=0)
        Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
    sc = StandardScaler().fit(rlogit(Ea))
    # Stage-1 GLM with delta
    Da = np.hstack([sc.transform(rlogit(Ea)), gb.BASE[a0], delta_a0])
    Db = np.hstack([sc.transform(rlogit(Eb)), gb.BASE[b0], delta_b0])
    m = LogisticRegression(C=0.3, max_iter=300, solver="newton-cholesky").fit(Da, y[a0])
    z_b = m.decision_function(Db)
    z_a = m.decision_function(Da)
    auc_glm = float(roc_auc_score(y[b0], z_b))
    log(f"  Stage1 GLM+δ Fold-0 AUC={auc_glm:.5f} (ref harvest fold0 GLM 0.94942)")

    # glm_quad logit if available
    gq = np.load(PRED / "glm_quad_oof.npy").ravel() if (PRED / "glm_quad_oof.npy").exists() else None

    # 4-leaf LGB
    Xa = np.hstack([gb.RAWV[a0], Ea, gb.KCNT[a0], delta_a0])
    Xb = np.hstack([gb.RAWV[b0], Eb, gb.KCNT[b0], delta_b0])
    if gq is not None:
        Xa = np.hstack([Xa, gq[a0].reshape(-1, 1)])
        Xb = np.hstack([Xb, gq[b0].reshape(-1, 1)])
    dtr = lgb.Dataset(Xa, label=y[a0], init_score=z_a)
    dva = lgb.Dataset(Xb, label=y[b0], init_score=z_b, reference=dtr)
    bst = lgb.train(
        dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=4, min_data_in_leaf=100, verbosity=-1),
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p_lgb = expit(z_b + bst.predict(Xb, raw_score=True))
    auc_lgb = float(roc_auc_score(y[b0], p_lgb))
    log(f"  LGB4+δ Fold-0 AUC={auc_lgb:.5f}")

    # XGB depth 2
    try:
        import xgboost as xgb

        dtrain = xgb.DMatrix(Xa, label=y[a0])
        dtrain.set_base_margin(z_a)
        dval = xgb.DMatrix(Xb, label=y[b0])
        dval.set_base_margin(z_b)
        xm = xgb.train(
            dict(objective="binary:logistic", eval_metric="auc", max_depth=2, eta=0.03, tree_method="hist", device="cuda" if DEVICE.type == "cuda" else "cpu"),
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "v")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        p_xgb = xm.predict(dval)
        auc_xgb = float(roc_auc_score(y[b0], p_xgb))
    except Exception as e:
        log(f"  XGB skip: {e}")
        p_xgb, auc_xgb = p_lgb, auc_lgb
    log(f"  XGB2+δ Fold-0 AUC={auc_xgb:.5f}")

    # MLPRes residual
    import torch.nn as nn

    class MLPRes(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.3), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    scX = StandardScaler().fit(Xa)
    Xat, Xbt = scX.transform(Xa), scX.transform(Xb)
    mlp = MLPRes(Xat.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
    Ya = torch.tensor(y[a0], dtype=torch.float32, device=DEVICE)
    Za = torch.tensor(z_a, dtype=torch.float32, device=DEVICE)
    Xt = torch.tensor(Xat, dtype=torch.float32, device=DEVICE)
    for ep in range(8):
        mlp.train()
        perm = torch.randperm(len(a0), device=DEVICE)
        for i0 in range(0, len(a0), 4096):
            idx = perm[i0 : i0 + 4096]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(Za[idx] + mlp(Xt[idx]), Ya[idx])
            loss.backward()
            opt.step()
    mlp.eval()
    with torch.no_grad():
        p_mlp = expit(z_b + mlp(torch.tensor(Xbt, dtype=torch.float32, device=DEVICE)).cpu().numpy())
    auc_mlp = float(roc_auc_score(y[b0], p_mlp))
    log(f"  MLPRes+δ Fold-0 AUC={auc_mlp:.5f}")

    best = max(auc_lgb, auc_xgb, auc_mlp)
    full = np.zeros(len(y))
    # use best model scores on b0; a0 from glm
    full[b0] = p_lgb if auc_lgb >= best - 1e-12 else (p_xgb if auc_xgb >= auc_mlp else p_mlp)
    full[a0] = expit(z_a)
    S_with, dS = stack_delta_full(full, y, folds, rep)
    passed = bool(best >= GATE3_AUC or dS >= DS_PROMOTE)
    np.save(PRED / "rung3_score_full.npy", full.astype(np.float32))
    out = {
        "name": "rung3",
        "auc_glm": auc_glm,
        "auc_lgb": auc_lgb,
        "auc_xgb": auc_xgb,
        "auc_mlp": auc_mlp,
        "best": best,
        "S_fold0": S_with,
        "Delta_S_fold0": dS,
        "passed": passed,
    }
    rep["rungs"]["rung3"] = out
    save_progress(rep)
    log(f"  GATE3 {'PASS' if passed else 'FAIL'} best={best:.5f} ΔS={dS:+.6f}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RUNG 4 — BiTabGPT
# ═══════════════════════════════════════════════════════════════════════════
def rung4(rep: dict) -> dict:
    log("=" * 72)
    log("RUNG 4: BiTabGPT bidirectional encoder")
    log("=" * 72)
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]
    os.environ["INPUT_DIR"] = str(ROOT)
    import gboost as gb

    Ea, Eb, Et = gb.te_fold(a0, b0, smooth=20.0, seed=0)
    if gb.GEN:
        Ga, Gb, Gt = gb.gen_fold(a0, b0, seed=0)
        Ea, Eb, Et = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
    # Use first 24 TE cols as token continuous feats
    n_tok = 24
    te_a = StandardScaler().fit_transform(rlogit(Ea[:, :n_tok]))
    te_b = StandardScaler().fit(rlogit(Ea[:, :n_tok])).transform(rlogit(Eb[:, :n_tok]))
    k_a = np.log1p(gb.KCNT[a0, :n_tok])
    k_b = np.log1p(gb.KCNT[b0, :n_tok])
    dpath = PRED / "delta24_contrastive_fold0.npy"
    if not dpath.exists():
        dpath = PRED / "delta24_fold0.npy"
    if dpath.exists():
        delta_b = np.load(dpath)[:, :n_tok]
        delta_a = np.zeros_like(delta_b[: len(a0)]) if False else np.zeros((len(a0), n_tok), np.float32)
    else:
        delta_a = np.zeros((len(a0), n_tok), np.float32)
        delta_b = np.zeros((len(b0), n_tok), np.float32)

    # Stage1 init score
    sc = StandardScaler().fit(rlogit(Ea))
    Da = np.hstack([sc.transform(rlogit(Ea)), gb.BASE[a0]])
    Db = np.hstack([sc.transform(rlogit(Eb)), gb.BASE[b0]])
    glm = LogisticRegression(C=0.3, max_iter=200, solver="newton-cholesky").fit(Da, y[a0])
    z_a = glm.decision_function(Da)
    z_b = glm.decision_function(Db)

    # Token features: [te, log1p(kcnt), delta] → 3 dims per token
    def pack(te, k, d):
        return np.stack([te, StandardScaler().fit_transform(k) if False else (k - k.mean(0)) / (k.std(0) + 1e-6), d], -1).astype(np.float32)

    # fit scaler on a0
    ka_s = (k_a - k_a.mean(0)) / (k_a.std(0) + 1e-6)
    kb_s = (k_b - k_a.mean(0)) / (k_a.std(0) + 1e-6)
    Xa = np.stack([te_a, ka_s, delta_a], -1).astype(np.float32)  # N x 24 x 3
    Xb = np.stack([te_b, kb_s, delta_b], -1).astype(np.float32)

    import torch.nn as nn

    class BiTabGPT(nn.Module):
        def __init__(self, n_tok=24, d_in=3, d_model=192, nhead=6, nlayers=4, dim_ff=512, drop=0.25):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.col = nn.Embedding(n_tok, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, drop, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, nlayers)
            self.head = nn.Linear(d_model, 1)
            self.n_tok = n_tok

        def forward(self, x, mask_ratio=0.0):
            # x: B x T x 3
            B, T, _ = x.shape
            h = self.proj(x) + self.col.weight[:T].unsqueeze(0)
            if mask_ratio > 0 and self.training:
                m = (torch.rand(B, T, device=x.device) < mask_ratio).unsqueeze(-1)
                h = h.masked_fill(m, 0.0)
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, h], 1)
            h = self.enc(h)
            return self.head(h[:, 0]).squeeze(-1)

    model = BiTabGPT().to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    n_ep = 12
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    Ya = torch.tensor(y[a0], dtype=torch.float32, device=DEVICE)
    Za = torch.tensor(z_a, dtype=torch.float32, device=DEVICE)
    Xat = torch.tensor(Xa, device=DEVICE)
    bs = 512
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(len(a0), device=DEVICE)
        total = 0.0
        for i0 in range(0, len(a0), bs):
            idx = perm[i0 : i0 + bs]
            opt.zero_grad()
            xin = Xat[idx]
            if DEVICE.type == "cuda":
                xin = xin.to(dtype=torch.bfloat16)
            delta = model(xin, mask_ratio=0.15)
            loss = nn.functional.binary_cross_entropy_with_logits(Za[idx] + delta.float(), Ya[idx])
            loss.backward()
            opt.step()
            total += float(loss)
        sched.step()
        log(f"    ep{ep} loss={total / max(1, len(a0) // bs):.4f}")

    model.eval()
    with torch.no_grad():
        Xbt = torch.tensor(Xb, device=DEVICE)
        if DEVICE.type == "cuda":
            Xbt = Xbt.to(dtype=torch.bfloat16)
        preds = []
        for i0 in range(0, len(b0), bs):
            preds.append(model(Xbt[i0 : i0 + bs]).float().cpu().numpy())
        raw = np.concatenate(preds)
    from scipy.special import expit

    p = expit(z_b + raw)
    auc = float(roc_auc_score(y[b0], p))
    log(f"  BiTabGPT Fold-0 AUC={auc:.5f}")
    full = np.zeros(len(y))
    full[b0] = p
    full[a0] = expit(z_a)
    S_with, dS = stack_delta_full(full, y, folds, rep)
    passed = bool(auc >= GATE4_AUC or dS >= DS_PROMOTE)
    np.save(PRED / "rung4_score_full.npy", full.astype(np.float32))
    torch.save(model.state_dict(), PRED / "rung4_bitabgpt.pt")
    out = {"name": "rung4", "auc": auc, "S_fold0": S_with, "Delta_S_fold0": dS, "passed": passed}
    rep["rungs"]["rung4"] = out
    save_progress(rep)
    log(f"  GATE4 {'PASS' if passed else 'FAIL'} ΔS={dS:+.6f}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RUNG 5 — promote passers
# ═══════════════════════════════════════════════════════════════════════════
def rung5(rep: dict) -> dict:
    log("=" * 72)
    log("RUNG 5: Promote gate-passers to full 5-fold (lightweight promotion)")
    log("=" * 72)
    y = pd.read_csv(ROOT / "data" / "train.csv")[TARGET].to_numpy()
    folds = folds_y(y)
    passers = []
    for k in ("rung1", "rung2", "rung3", "rung4"):
        r = rep.get("rungs", {}).get(k) or {}
        ds = float(r.get("Delta_S_fold0") or -1)
        if r.get("passed") and ds >= DS_PROMOTE:
            passers.append((ds, k, r))
        elif r.get("passed"):
            # also promote if passed AUC gate even if tiny ΔS
            passers.append((ds, k, r))
    passers.sort(reverse=True)
    log(f"  promoting {[k for _, k, _ in passers]}")

    # For each passer, export Fold-0 scores into bank-shaped full OOF by stitching
    # masaya gpt as a0 filler + fold0 scores — then nest with champ
    promoted = []
    try:
        from part9_joint_boost_and_basis import build_8b_champ
        from part7_clean_superstack import nested_cv

        names, O, T = build_8b_champ(y, folds)
    except Exception as e:
        log(f"  champ unavailable: {e}")
        names, O, T = None, None, None
        nested_cv = None

    for ds, k, r in passers:
        sp = PRED / f"{k}_score_full.npy"
        if not sp.exists():
            continue
        col = np.load(sp).ravel()
        # nest
        if O is not None:
            O2 = np.column_stack([O, rk(col)])
            best_a, best_C = -1, None
            for C in (0.005, 0.008, 0.01, 0.012):
                _, a, fa = nested_cv(O2, y, folds, C=C)
                if a > best_a:
                    best_a, best_C = a, C
            log(f"  {k} nested CV={best_a:.6f} C={best_C} (Δ vs 0.95214={best_a-0.9521425:+.6f})")
            promoted.append({"rung": k, "nested_auc": best_a, "C": best_C, "Delta_S_fold0": ds})
            np.save(PRED / f"{k}_promoted_oof.npy", col.astype(np.float32))
        else:
            promoted.append({"rung": k, "nested_auc": None, "Delta_S_fold0": ds})

    out = {"name": "rung5", "promoted": promoted, "passed": True}
    rep["rungs"]["rung5"] = out
    rep["summary"] = {
        "promoted": promoted,
        "best_nested": max((p.get("nested_auc") or 0) for p in promoted) if promoted else None,
        "policy": POLICY,
        "note": "NO_SUBMIT — shut down pod after sync",
    }
    save_progress(rep)
    return out


def main():
    rep = load_progress()
    log(f"DEVICE={DEVICE} ROOT={ROOT} SCORE_INFER_BS={SCORE_INFER_BS}")
    order = [
        ("rung0", rung0),
        ("rung1", rung1),
        ("rung2", rung2),
        ("rung3", rung3),
        ("rung4", rung4),
        ("rung5", rung5),
    ]
    for key, fn in order:
        prev = (rep.get("rungs") or {}).get(key) or {}
        # Skip only fully completed gate runs (passed True, or False with a real metric)
        if key == "rung0" and prev.get("passed") is True and "S_fold0_base" in prev:
            log(f"SKIP {key} (complete) S_fold0_base={prev.get('S_fold0_base')}")
            continue
        if key != "rung0" and key != "rung5":
            if prev.get("passed") is True or (
                prev.get("passed") is False
                and not prev.get("error")
                and (prev.get("auc_weighted_delta") is not None or prev.get("auc") is not None)
            ):
                log(f"SKIP {key} (already ran passed={prev.get('passed')})")
                # After Gate1, ensure cumulative K-curve is logged before later rungs
                if key == "rung1" and not prev.get("perm_auc_curve"):
                    try:
                        rep = enrich_rung1_perm_curve(load_progress())
                    except Exception as e:
                        log(f"  enrich after skip failed: {e}")
                continue
        try:
            fn(rep)
            rep = load_progress()
            if key == "rung1":
                # Always attach cumulative perm curve + refreshed ΔS after Gate1
                try:
                    rep = enrich_rung1_perm_curve(rep)
                except Exception as e:
                    log(f"  enrich after rung1 failed: {e}")
        except Exception as e:
            log(f"FAIL {key}: {e}")
            import traceback

            traceback.print_exc()
            rep = load_progress()
            rep.setdefault("rungs", {})[key] = {"error": str(e), "passed": False}
            save_progress(rep)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue
    log("LADDER COMPLETE")
    save_progress(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
