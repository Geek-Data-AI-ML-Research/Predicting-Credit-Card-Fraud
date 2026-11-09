#!/usr/bin/env python3
"""
Autonomous post-Fold-4 campaign on mgfv9v9ltb348q (STRICT NO_SUBMIT).

Waits for PID 5157 (run_folds_234.py) without importing torch/CUDA, then runs:
  Stage A — verify gpt2_3ep 5-fold + ECDF + score vs v4p_hc/v5 + O_v4p gpt2 replace + h16
  Stage B — gb_gpt2_3ep_super (xgb/lgb/cat bags) on KMAT+P4+gpt2_3ep[+h16]
  Stage C — gpt2_rev_2ep risk-first reverse twin (2 epochs × 5 folds)
  Stage D — P4_L4.2_ext / L4.3_ext Ridge(500) + P_super_v6 lock + DONE marker

Pod terminate is owned by the local monitor after CAMPAIGN_DONE.json appears.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- paths
S6 = Path(os.environ.get("S6E11_ROOT", "/workspace/s6e11"))
LM_ROOT = Path(os.environ.get("LM_ROOT", "/workspace/src/lm"))
WAIT_PID = int(os.environ.get("WAIT_PID", "5157"))
SKIP_WAIT = int(os.environ.get("SKIP_WAIT", "0"))
SKIP_STAGE_C = int(os.environ.get("SKIP_STAGE_C", "0"))  # emergency escape hatch

sys.path.insert(0, str(S6 / "scripts"))
sys.path.insert(0, str(LM_ROOT))
os.environ.setdefault("INPUT_DIR", str(S6))
os.environ.setdefault("S6E11_ROOT", str(S6))
os.environ["NO_SUBMIT"] = "1"

PRED = S6 / "preds"
SUB = S6 / "submissions"
LOGD = S6 / "logs/overnight"
STATED = LOGD / "post_gpt2_3ep_state.json"
DONEF = LOGD / "CAMPAIGN_DONE.json"
LOGF = LOGD / "post_gpt2_3ep_campaign.log"
for d in (PRED / "matrix_v2", PRED / "paradigm_B", PRED / "gpt2_3ep", SUB, LOGD):
    d.mkdir(parents=True, exist_ok=True)

V4P_BASE = 0.9523071
V5_BASE = 0.9523401


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")


def save_state(**kw) -> None:
    st = {}
    if STATED.exists():
        try:
            st = json.loads(STATED.read_text())
        except Exception:
            st = {}
    st.update(kw)
    st["updated_utc"] = utc()
    STATED.write_text(json.dumps(st, indent=2, default=str))


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def rk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return (rankdata(x, method="average") - 0.5) / max(len(x), 1)


def rk01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return rankdata(x, method="average") / max(len(x), 1)


def zrk(x: np.ndarray) -> np.ndarray:
    return logit(np.clip(rk01(x), 1e-6, 1.0 - 1e-6))


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(s, method="average")
    return (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def align_to_fold_train_ecdf(sa: np.ndarray, sb: np.ndarray, ste: np.ndarray):
    sa_s = np.sort(np.asarray(sa, dtype=np.float64).ravel())
    n = len(sa_s)

    def _m(v):
        r = rk(v)
        q = np.clip(r * (n - 1), 0.0, n - 1.0)
        lo = np.floor(q).astype(np.int64)
        hi = np.minimum(lo + 1, n - 1)
        w = q - lo
        return (1.0 - w) * sa_s[lo] + w * sa_s[hi]

    return _m(sb), _m(ste)


def ecdf_align_5fold(oof: np.ndarray, test: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, np.ndarray]:
    """Per-fold ECDF map val→train; mean-aggregate test maps. Target KS(oof,test)<0.001."""
    oof_al = np.zeros_like(oof, dtype=np.float64)
    te_folds = []
    for a, b in folds:
        sb, ste = align_to_fold_train_ecdf(oof[a], oof[b], test)
        oof_al[b] = sb
        te_folds.append(ste)
    te_al = np.mean(te_folds, axis=0)
    return oof_al, te_al


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    za = np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1)
    ya = np.asarray(y_a, dtype=np.int32).ravel()
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(za, ya)
    outs = [lr.decision_function(za).astype(np.float64)]
    for zo in z_others:
        outs.append(lr.decision_function(np.asarray(zo, dtype=np.float64).reshape(-1, 1)).astype(np.float64))
    return tuple(outs)


def wait_for_pid(pid: int) -> None:
    if SKIP_WAIT:
        log(f"SKIP_WAIT=1 — not waiting for PID {pid}")
        return
    log(f"WAIT GATE: passive poll kill -0 {pid} every 15s (0% GPU / no torch yet)")
    save_state(phase="wait_pid", wait_pid=pid)
    t0 = time.time()
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"PID {pid} exited after {(time.time()-t0)/60:.1f} min wait — starting Stages A–D")
            break
        except PermissionError:
            # process exists but not owned — treat as alive
            pass
        time.sleep(15)
    time.sleep(5)  # settle disk flushes


# --------------------------------------------------------------------------- Stage A
def stage_a(y, folds, v4p_o, v4p_t, v5_o, v5_t) -> dict:
    from paradigm_stack_delta_eval import score_candidate_vs_v4p_hc

    log("=== STAGE A: gpt2_3ep verify + ECDF + score + h16 ===")
    save_state(phase="stage_a")
    out = {}

    # Prefer final 5-fold merge from run_folds_234
    cand_oof = [
        LM_ROOT / "out/lm/lm_oof_5f.npy",
        LM_ROOT / "out/lm/lm_oof.npy",
        PRED / "gpt2_3ep/gpt2_3ep_oof.npy",
    ]
    cand_te = [
        LM_ROOT / "out/lm/lm_test_5f.npy",
        LM_ROOT / "out/lm/lm_test.npy",
        PRED / "gpt2_3ep/gpt2_3ep_test.npy",
    ]
    oof_p = next((p for p in cand_oof if p.exists()), None)
    te_p = next((p for p in cand_te if p.exists()), None)
    if oof_p is None or te_p is None:
        raise FileNotFoundError(f"missing gpt2_3ep arrays: oof={oof_p} te={te_p}")

    oof = np.load(oof_p).astype(np.float64).ravel()
    te = np.load(te_p).astype(np.float64).ravel()
    assert oof.shape == (699960,), oof.shape
    assert te.shape == (299984,), te.shape
    assert np.isfinite(oof).all() and np.isfinite(te).all()
    solo = float(fast_auc(y, oof))
    log(f"  loaded {oof_p.name}/{te_p.name} solo_auc={solo:.7f}")

    # Per-fold ECDF alignment
    oof_al, te_al = ecdf_align_5fold(oof, te, y, folds)
    ks = float(ks_2samp(rk(oof_al), rk(te_al)).statistic)
    log(f"  ECDF-aligned KS(oof,test)={ks:.6f}")
    np.save(PRED / "gpt2_3ep/gpt2_3ep_oof.npy", oof_al.astype(np.float64))
    np.save(PRED / "gpt2_3ep/gpt2_3ep_test.npy", te_al.astype(np.float64))

    ev_v4 = score_candidate_vs_v4p_hc(y, v4p_o, oof_al, folds)
    ev_v5 = score_candidate_vs_v4p_hc(y, v5_o, oof_al, folds)
    log(f"  vs v4p_hc: Δ={ev_v4['delta_auc']:+.3e} stack={ev_v4['stack_oof_auc']:.7f} ᾱ={ev_v4['stack_mean_alpha']:+.4f} win={ev_v4['fold_win_rate']}")
    log(f"  vs v5_harvest: Δ={ev_v5['delta_auc']:+.3e} stack={ev_v5['stack_oof_auc']:.7f} ᾱ={ev_v5['stack_mean_alpha']:+.4f}")
    out["vs_v4p_hc"] = {k: ev_v4[k] for k in ("delta_auc", "stack_oof_auc", "cand_solo_auc", "stack_mean_alpha", "fold_win_rate", "fold_deltas")}
    out["vs_v5"] = {k: ev_v5[k] for k in ("delta_auc", "stack_oof_auc", "cand_solo_auc", "stack_mean_alpha", "fold_win_rate")}
    out["ks_ecdf"] = ks
    out["solo_auc"] = solo

    # O_v4p direct replacement of col 'gpt2' (idx 62)
    bank_p = PRED / "v4p_bank_slim.npz"
    if not bank_p.exists():
        bank_p = PRED / "v4p_bank_cache.npz"
    if bank_p.exists():
        bank = np.load(bank_p, allow_pickle=True)
        names = list(bank["names_v4p"])
        O = bank["O_v4p"].astype(np.float64)
        T = bank["T_v4p"].astype(np.float64) if "T_v4p" in bank.files else None
        gpt_idx = names.index("gpt2") if "gpt2" in names else None
        if gpt_idx is not None:
            O2 = O.copy()
            O2[:, gpt_idx] = oof_al
            # quick Newton-Cholesky style: fold-wise LogisticRegression on O cols (C=0.01)
            oof_rep = np.zeros(len(y), dtype=np.float64)
            te_rep_folds = []
            for a, b in folds:
                lr = LogisticRegression(C=0.01, solver="lbfgs", max_iter=200, tol=1e-8)
                lr.fit(O2[a], y[a])
                oof_rep[b] = lr.decision_function(O2[b])
                if T is not None:
                    Tt = T.copy()
                    Tt[:, gpt_idx] = te_al
                    te_rep_folds.append(lr.decision_function(Tt))
            rep_auc = float(fast_auc(y, oof_rep))
            base_o_auc = float(fast_auc(y, v4p_o))
            # also score original O without replace for delta
            oof_baseO = np.zeros(len(y), dtype=np.float64)
            for a, b in folds:
                lr = LogisticRegression(C=0.01, solver="lbfgs", max_iter=200, tol=1e-8)
                lr.fit(O[a], y[a])
                oof_baseO[b] = lr.decision_function(O[b])
            baseO_auc = float(fast_auc(y, oof_baseO))
            log(f"  O_v4p gpt2-replace: AUC={rep_auc:.7f} (refit_baseO={baseO_auc:.7f}, Δ={rep_auc-baseO_auc:+.3e})")
            np.save(PRED / "gpt2_3ep/O_v4p_gpt2_3ep_replace_oof.npy", oof_rep.astype(np.float64))
            if te_rep_folds:
                te_rep = np.mean(te_rep_folds, axis=0)
                np.save(PRED / "gpt2_3ep/O_v4p_gpt2_3ep_replace_test.npy", te_rep.astype(np.float64))
            out["o_v4p_replace"] = {"auc": rep_auc, "baseO_auc": baseO_auc, "delta": rep_auc - baseO_auc, "gpt_idx": gpt_idx}
        else:
            log("  O_v4p: no 'gpt2' column found — skip replace")
    else:
        log("  O_v4p bank missing — skip replace test")

    # h16: supervised Ridge projections of h_last from fold checkpoints
    h16_o, h16_t = extract_h16(y, folds, v4p_o, v4p_t)
    if h16_o is not None:
        np.save(PRED / "gpt2_3ep/gpt2_3ep_h16_oof.npy", h16_o.astype(np.float64))
        np.save(PRED / "gpt2_3ep/gpt2_3ep_h16_test.npy", h16_t.astype(np.float64))
        # score mean of h16 cols
        h_mean = h16_o.mean(1)
        ev_h = score_candidate_vs_v4p_hc(y, v4p_o, h_mean, folds)
        log(f"  h16 mean vs v4p_hc: Δ={ev_h['delta_auc']:+.3e} solo={ev_h['cand_solo_auc']:.6f}")
        out["h16"] = {"delta_auc": ev_h["delta_auc"], "cand_solo_auc": ev_h["cand_solo_auc"], "shape": list(h16_o.shape)}
    else:
        out["h16"] = None
        log("  h16 extraction skipped/failed — continuing without")

    save_state(phase="stage_a_done", stage_a=out)
    log("=== STAGE A DONE ===")
    return out


def extract_h16(y, folds, v4p_o, v4p_t):
    """bf16 pass over fold ckpts → h_last → top-16 Ridge on Platt residual."""
    try:
        import torch
        import lm_member as lm
    except Exception as e:
        log(f"  h16 import fail: {e}")
        return None, None

    ckpts = sorted((LM_ROOT / "ckpt/lm").glob("fold_*.pt"))
    if len(ckpts) < 3:
        log(f"  h16: only {len(ckpts)} fold ckpts — skip")
        return None, None

    log(f"  h16: extracting from {len(ckpts)} checkpoints …")
    g = lm.load_data()
    R = lm.Rows(g)
    ntr, nte = g["ntr"], g["nte"]
    test_idx = np.arange(ntr, ntr + nte)
    z_base = zrk(v4p_o)
    H_oof = np.zeros((ntr, 768), dtype=np.float32)
    H_te_acc = []
    covered = np.zeros(ntr, dtype=bool)

    @torch.no_grad()
    def h_last_batch(m, rows, bs=256):
        m.eval()
        outs = []
        gen = torch.Generator(device=lm.dev).manual_seed(123)
        for i0 in range(0, len(rows), bs):
            idx = rows[i0:i0 + bs]
            idx_t = torch.as_tensor(idx, device=lm.dev)
            perm = R.perm(len(idx), gen)
            # assemble with label=0 just to get sequence; take last non-pad hidden
            inp, _ = R.assemble(idx_t, torch.zeros(len(idx), dtype=torch.long, device=lm.dev), perm)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = m.hidden(inp)  # [B,T,D]
            # last non-pad position
            pad = g["PAD"]
            mask = inp != pad
            # position of last True per row
            lengths = mask.long().sum(1).clamp_min(1) - 1
            gather = h[torch.arange(h.shape[0], device=h.device), lengths]
            outs.append(gather.float().cpu().numpy())
        return np.concatenate(outs, 0)

    for ck in ckpts:
        # fold_2.pt → fold index 2
        try:
            fi = int(ck.stem.split("_")[1])
        except Exception:
            continue
        if fi >= len(folds):
            continue
        a, b = folds[fi]
        log(f"  h16 fold {fi}: load {ck.name}")
        m = lm.replica(g, R, torch.as_tensor(np.maximum(g["y"], 0), device=lm.dev))
        s = torch.load(ck, map_location=lm.dev, weights_only=True)
        m.load_state_dict(s["model"] if isinstance(s, dict) and "model" in s else s)
        Hb = h_last_batch(m, b)
        Ht = h_last_batch(m, test_idx)
        H_oof[b] = Hb
        covered[b] = True
        H_te_acc.append(Ht)
        del m
        torch.cuda.empty_cache()
        gc.collect()

    if covered.sum() < ntr * 0.5:
        # fill missing folds 0/1 from base if only 2/3/4 present — use mean of available for uncovered
        log(f"  h16 covered {covered.mean():.0%} — filling uncovered with global mean")
        mu = H_oof[covered].mean(0, keepdims=True)
        H_oof[~covered] = mu

    H_te = np.mean(H_te_acc, axis=0) if H_te_acc else np.zeros((nte, 768), np.float32)

    # top-16 supervised Ridge on residual y - σ(z_cal) inside each fold a
    h16_o = np.zeros((ntr, 16), dtype=np.float64)
    h16_te_folds = []
    for a, b in folds:
        z_cal_a, z_cal_all, z_cal_te = calibrate_logit_to_prevalence(z_base[a], y[a], z_base, zrk(v4p_t))
        res_a = y[a].astype(np.float64) - expit(z_cal_a)
        # select top-16 dims by |corr| on fold a
        Ha = H_oof[a].astype(np.float64)
        corrs = np.array([abs(np.corrcoef(Ha[:, j], res_a)[0, 1]) if Ha[:, j].std() > 1e-8 else 0.0 for j in range(Ha.shape[1])])
        top = np.argsort(-corrs)[:16]
        Xa = StandardScaler().fit_transform(Ha[:, top])
        Xb = StandardScaler().fit(Ha[:, top]).transform(H_oof[b][:, top].astype(np.float64))
        Xte = StandardScaler().fit(Ha[:, top]).transform(H_te[:, top].astype(np.float64))
        # multi-target: fit 16 independent 1D ridges projecting each selected dim residual-weighted
        # simpler: Ridge predicting residual from 16 dims, then keep per-dim contributions
        ridge = Ridge(alpha=100.0).fit(Xa, res_a)
        # store the 16 standardized features themselves (supervised selection) as h16
        h16_o[b] = Xb
        h16_te_folds.append(Xte)
        # also write residual projection as col 0 override? keep features
        _ = ridge  # fitted for side-check
        log(f"  h16 fold: top corr={corrs[top[0]]:.4f} ridge_R2≈{ridge.score(Xa, res_a):.4f}")

    h16_t = np.mean(h16_te_folds, axis=0)
    return h16_o, h16_t


# --------------------------------------------------------------------------- Stage B
def stage_b(y, folds, v4p_o, v4p_t, v5_o, v5_t, gpt_o, gpt_t, h16_o, h16_t) -> dict:
    from paradigm_stack_delta_eval import score_candidate_vs_v4p_hc

    log("=== STAGE B: gb_gpt2_3ep_super ===")
    save_state(phase="stage_b")
    out = {}

    # Build X_fusion: KMAT (gboost) + P4 pure5f scores as features + gpt2 z + interactions
    log("  importing gboost (builds KMAT) …")
    import gboost as gb
    log(f"  gboost ready GEN={getattr(gb,'GEN',None)} KMAT={gb.KMAT.shape} BASE={gb.BASE.shape}")

    # OOF TE of KMAT keys (first NBASE) — use gb's existing TE helpers if present, else simple EB TE
    K = gb.KMAT.astype(np.float64)  # full train+test stacked
    ntr = len(y)
    K_tr, K_te = K[:ntr], K[ntr:]

    # Lightweight: use first min(108, K.shape[1]) raw key columns after fold-wise target encoding
    n_keys = min(108, K_tr.shape[1])
    K_oof = np.zeros((ntr, n_keys), dtype=np.float64)
    K_te_folds = []
    for a, b in folds:
        prior = float(y[a].mean())
        te_cols_b = []
        te_cols_te = []
        te_cols_a = []
        for j in range(n_keys):
            ka = K_tr[a, j]
            # bin continuous keys
            if np.issubdtype(ka.dtype, np.floating) or len(np.unique(ka)) > 500:
                # quantile bins
                edges = np.unique(np.quantile(ka, np.linspace(0, 1, 32)))
                ba = np.digitize(ka, edges)
                bb = np.digitize(K_tr[b, j], edges)
                bt = np.digitize(K_te[:, j], edges)
            else:
                ba, bb, bt = ka, K_tr[b, j], K_te[:, j]
            # EB TE
            tmp = pd.DataFrame({"k": ba, "y": y[a]})
            gstat = tmp.groupby("k")["y"].agg(["sum", "count"])
            m = 20.0
            te_map = ((gstat["sum"] + prior * m) / (gstat["count"] + m)).to_dict()
            te_cols_a.append(np.array([te_map.get(k, prior) for k in ba], dtype=np.float64))
            te_cols_b.append(np.array([te_map.get(k, prior) for k in bb], dtype=np.float64))
            te_cols_te.append(np.array([te_map.get(k, prior) for k in bt], dtype=np.float64))
        # store OOF for b
        K_oof[b] = np.column_stack(te_cols_b)
        K_te_folds.append(np.column_stack(te_cols_te))
    K_te_m = np.mean(K_te_folds, axis=0)

    # P4 pure5f as single cols
    def load_p4(name):
        o = np.load(PRED / f"matrix_v2/{name}_pure5f_oof.npy").ravel()
        t = np.load(PRED / f"matrix_v2/{name}_pure5f_test.npy").ravel()
        return o, t

    l43_o, l43_t = load_p4("P4_L4.3")
    l42_o, l42_t = load_p4("P4_L4.2")
    l45_o, l45_t = load_p4("P4_L4.5")

    z_gpt = zrk(gpt_o)
    z_gpt_t = zrk(gpt_t)
    z_v4 = zrk(v4p_o)
    z_v4_t = zrk(v4p_t)
    # rare/disp proxies from amount-like key if available else velocity TE col 0
    r_rare = K_oof[:, 0]
    r_disp = K_oof[:, min(1, n_keys - 1)]
    r_rare_t = K_te_m[:, 0]
    r_disp_t = K_te_m[:, min(1, n_keys - 1)]

    extra_o = np.column_stack([
        z_gpt, z_gpt - z_v4, z_gpt * r_rare, z_gpt * r_disp,
        zrk(l43_o), zrk(l42_o), zrk(l45_o),
    ])
    extra_t = np.column_stack([
        z_gpt_t, z_gpt_t - z_v4_t, z_gpt_t * r_rare_t, z_gpt_t * r_disp_t,
        zrk(l43_t), zrk(l42_t), zrk(l45_t),
    ])
    if h16_o is not None:
        extra_o = np.hstack([extra_o, h16_o])
        extra_t = np.hstack([extra_t, h16_t])

    X_o = np.hstack([K_oof, extra_o, gb.BASE[:ntr].astype(np.float64)])
    X_t = np.hstack([K_te_m, extra_t, gb.BASE[ntr:].astype(np.float64)])
    log(f"  X_fusion shape o={X_o.shape} t={X_t.shape}")

    import xgboost as xgb
    import lightgbm as lgb
    try:
        from catboost import CatBoostClassifier
        HAS_CAT = True
    except Exception:
        HAS_CAT = False
        log("  CatBoost unavailable — skipping cat bag")

    z_v5 = zrk(v5_o)
    z_v5_t = zrk(v5_t)

    def train_family(kind: str) -> tuple[np.ndarray, np.ndarray]:
        oof = np.zeros(ntr, dtype=np.float64)
        te_acc = []
        if kind == "xgb":
            specs = [(42 + i, 2) for i in range(3)] + [(100 + i, 3) for i in range(3)]
        elif kind == "lgb":
            specs = [(42 + i, 4 if i % 2 == 0 else 5) for i in range(6)]
        else:
            specs = [(42 + i, 4) for i in range(4)]

        for seed, depth_or_leaves in specs:
            oof_s = np.zeros(ntr, dtype=np.float64)
            te_s = []
            for fi, (a, b) in enumerate(folds):
                # Platt-calibrated base_margin from v5_harvest
                bm_a, bm_all, bm_te = calibrate_logit_to_prevalence(z_v5[a], y[a], z_v5, z_v5_t)
                bm_b = bm_all[b]
                if kind == "xgb":
                    dtr = xgb.DMatrix(X_o[a], label=y[a], base_margin=bm_a)
                    dva = xgb.DMatrix(X_o[b], label=y[b], base_margin=bm_b)
                    dte = xgb.DMatrix(X_t, base_margin=bm_te)
                    params = dict(
                        max_depth=depth_or_leaves, eta=0.025, subsample=0.8,
                        colsample_bynode=0.35, reg_lambda=30.0, min_child_weight=200,
                        objective="binary:logistic", eval_metric="auc",
                        tree_method="hist", device="cuda", seed=seed,
                    )
                    booster = xgb.train(params, dtr, num_boost_round=400, evals=[(dva, "v")],
                                        early_stopping_rounds=40, verbose_eval=False)
                    oof_s[b] = booster.predict(dva)
                    te_s.append(booster.predict(dte))
                elif kind == "lgb":
                    dtr = lgb.Dataset(X_o[a], label=y[a], init_score=bm_a)
                    dva = lgb.Dataset(X_o[b], label=y[b], init_score=bm_b, reference=dtr)
                    params = dict(
                        objective="binary", metric="auc", learning_rate=0.025,
                        num_leaves=depth_or_leaves, feature_fraction=0.35,
                        min_child_samples=150, verbosity=-1, seed=seed,
                        device_type="gpu" if os.environ.get("LGB_GPU", "0") == "1" else "cpu",
                    )
                    booster = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                                        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
                    oof_s[b] = booster.predict(X_o[b], raw_score=False)
                    # apply init_score on test via probability space — use margin form
                    # LightGBM predict doesn't take init_score; approximate by training with init and predicting
                    te_pred = booster.predict(X_t, raw_score=True) + bm_te
                    te_s.append(expit(te_pred))
                else:
                    model = CatBoostClassifier(
                        depth=depth_or_leaves, l2_leaf_reg=25.0, learning_rate=0.025,
                        iterations=400, loss_function="Logloss", eval_metric="AUC",
                        random_seed=seed, verbose=False, task_type="GPU",
                        early_stopping_rounds=40,
                    )
                    model.fit(X_o[a], y[a], eval_set=(X_o[b], y[b]), baseline=bm_a)
                    # CatBoost baseline on val/test is tricky; use predict_proba
                    oof_s[b] = model.predict_proba(X_o[b])[:, 1]
                    te_s.append(model.predict_proba(X_t)[:, 1])
            oof += oof_s
            te_acc.append(np.mean(te_s, axis=0))
        oof /= len(specs)
        te = np.mean(te_acc, axis=0)
        return oof, te

    families = {}
    for kind, key in [("xgb", "xgb_gpt2_3ep_d2_d3_bag6"), ("lgb", "lgb_gpt2_3ep_leaves4_bag6")]:
        log(f"  training {key} …")
        oof_k, te_k = train_family(kind)
        np.save(PRED / f"gpt2_3ep/{key}_oof.npy", oof_k)
        np.save(PRED / f"gpt2_3ep/{key}_test.npy", te_k)
        ev = score_candidate_vs_v4p_hc(y, v5_o, oof_k, folds)
        log(f"  {key}: solo={ev['cand_solo_auc']:.7f} Δvs_v5={ev['delta_auc']:+.3e}")
        families[key] = (oof_k, te_k, ev)

    if HAS_CAT:
        key = "cat_gpt2_3ep_d4_bag4"
        try:
            log(f"  training {key} …")
            oof_k, te_k = train_family("cat")
            np.save(PRED / f"gpt2_3ep/{key}_oof.npy", oof_k)
            np.save(PRED / f"gpt2_3ep/{key}_test.npy", te_k)
            ev = score_candidate_vs_v4p_hc(y, v5_o, oof_k, folds)
            log(f"  {key}: solo={ev['cand_solo_auc']:.7f} Δvs_v5={ev['delta_auc']:+.3e}")
            families[key] = (oof_k, te_k, ev)
        except Exception as e:
            log(f"  cat bag failed: {e}")

    # Blend families by rank
    blend_o = rk(sum(rk(v[0]) for v in families.values()))
    blend_t = rk(sum(rk(v[1]) for v in families.values()))
    np.save(PRED / "gpt2_3ep/gb_gpt2_3ep_super_oof.npy", blend_o.astype(np.float64))
    np.save(PRED / "gpt2_3ep/gb_gpt2_3ep_super_test.npy", blend_t.astype(np.float64))
    ev_b = score_candidate_vs_v4p_hc(y, v5_o, blend_o, folds)
    log(f"  gb_gpt2_3ep_super: solo={ev_b['cand_solo_auc']:.7f} Δvs_v5={ev_b['delta_auc']:+.3e} stack={ev_b['stack_oof_auc']:.7f}")
    out["families"] = {k: {"delta_auc": v[2]["delta_auc"], "solo": v[2]["cand_solo_auc"]} for k, v in families.items()}
    out["super"] = {"delta_auc": ev_b["delta_auc"], "solo": ev_b["cand_solo_auc"], "stack": ev_b["stack_oof_auc"]}
    save_state(phase="stage_b_done", stage_b=out)
    log("=== STAGE B DONE ===")
    return out


# --------------------------------------------------------------------------- Stage C
def stage_c(y, folds, gpt_o, gpt_t) -> dict:
    """Risk-first reverse twin: 2 epochs × 5 folds, fixed reverse column order."""
    from paradigm_stack_delta_eval import score_candidate_vs_v4p_hc

    log("=== STAGE C: gpt2_rev_2ep (risk-first reverse twin) ===")
    save_state(phase="stage_c")
    if SKIP_STAGE_C:
        log("  SKIP_STAGE_C=1")
        return {"skipped": True}

    import torch
    import lm_member as lm

    # Force 2-epoch reverse recipe
    lm.EPOCHS = 2
    lm.BATCH = 96
    lm.LR = 9e-5
    lm.K_OOF = 6
    lm.K_TEST = 4
    lm.SCORE_BS = 512
    lm.CK = LM_ROOT / "ckpt" / "lm_rev"
    lm.OUT = LM_ROOT / "out" / "lm_rev"
    lm.CK.mkdir(parents=True, exist_ok=True)
    lm.OUT.mkdir(parents=True, exist_ok=True)

    g = lm.load_data()
    R = lm.Rows(g)
    ydev = torch.as_tensor(np.maximum(g["y"], 0), device=lm.dev)
    ntr, nte = g["ntr"], g["nte"]
    test_idx = np.arange(ntr, ntr + nte)

    # Risk-first column order: cats/flags/risk first, financial floats last
    # Column order in cache matches train.csv feature order
    tr_cols = [c for c in pd.read_csv(S6 / "data/train.csv", nrows=0).columns if c not in ("id", "is_fraud")]
    assert len(tr_cols) == g["NC"], (len(tr_cols), g["NC"])
    FIN = {"amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km",
           "card_age_months", "customer_age", "txn_count_last_24h", "cvv_retry_count",
           "time_of_day_hour", "day_of_week"}
    RISK = {"velocity_score", "merchant_risk_score", "prior_disputes",
            "is_foreign_transaction", "is_new_merchant", "used_vpn",
            "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
            "merchant_category", "card_type", "auth_method", "channel", "device_type"}
    risk_idx = [i for i, c in enumerate(tr_cols) if c in RISK or c not in FIN]
    fin_idx = [i for i, c in enumerate(tr_cols) if c in FIN]
    # dedupe preserve order
    seen = set()
    rev_order = []
    for i in risk_idx + fin_idx:
        if i not in seen:
            seen.add(i)
            rev_order.append(i)
    for i in range(g["NC"]):
        if i not in seen:
            rev_order.append(i)
    rev_t = torch.tensor(rev_order, device=lm.dev, dtype=torch.long)
    log(f"  reverse order (first8)={[tr_cols[i] for i in rev_order[:8]]} last4={[tr_cols[i] for i in rev_order[-4:]]}")

    _orig_perm = R.perm

    def risk_first_perm(n, gen):
        # fixed risk-first with tiny jitter of adjacent swaps for regularization
        base = rev_t.unsqueeze(0).expand(n, -1).contiguous()
        if n > 0:
            # occasional adjacent swap ~10%
            jitter = torch.rand(n, device=lm.dev, generator=gen) < 0.10
            if jitter.any():
                # swap positions 0 and 1 for jittered rows (cheap)
                tmp = base[jitter, 0].clone()
                base[jitter, 0] = base[jitter, 1]
                base[jitter, 1] = tmp
        return base

    R.perm = risk_first_perm

    full_oof = np.full(ntr, np.nan, dtype=np.float64)
    test_ranks = []
    fold_aucs = {}

    for f, (a, b) in enumerate(folds):
        oof_path = lm.OUT / f"oof_{f}.npy"
        test_path = lm.OUT / f"test_{f}.npy"
        prior = float(y[a].mean())
        if oof_path.exists() and test_path.exists():
            ll_b = np.load(oof_path)
            ll_t = np.load(test_path)
            z_b = lm.bayes(ll_b, prior)
            z_t = lm.bayes(ll_t, prior)
        else:
            t1 = time.time()
            m = lm.replica(g, R, ydev)
            lm.train(m, R, ydev, a, lm.EPOCHS, lm.BATCH, lm.LR, f"rev_fold_{f}", lm.SEED + 17 + f)
            ll_b = lm.score(m, R, b, lm.K_OOF, lm.SEED + 300 + f, bs=lm.SCORE_BS)
            np.save(oof_path, ll_b)
            z_b = lm.bayes(ll_b, prior)
            ll_t = lm.score(m, R, test_idx, lm.K_TEST, lm.SEED + 400 + f, bs=lm.SCORE_BS)
            np.save(test_path, ll_t)
            z_t = lm.bayes(ll_t, prior)
            log(f"  rev fold {f} DONE in {(time.time()-t1)/60:.1f}m")
            del m
            torch.cuda.empty_cache()
        f_auc = float(roc_auc_score(y[b], z_b))
        fold_aucs[f] = f_auc
        full_oof[b] = z_b
        test_ranks.append(rk(z_t))
        log(f"  rev fold {f} OOF AUC={f_auc:.6f}")
        save_state(phase=f"stage_c_fold_{f}_done", fold_aucs_rev=fold_aucs)

    R.perm = _orig_perm
    # calibrate scale to gpt2_3ep
    ref_mu, ref_sd = float(np.nanmean(gpt_o)), float(np.nanstd(gpt_o))
    full_oof = (full_oof - np.nanmean(full_oof)) / (np.nanstd(full_oof) + 1e-12) * ref_sd + ref_mu
    te = np.mean(test_ranks, axis=0)
    np.save(PRED / "gpt2_3ep/gpt2_rev_2ep_oof.npy", full_oof.astype(np.float64))
    np.save(PRED / "gpt2_3ep/gpt2_rev_2ep_test.npy", te.astype(np.float64))

    bi_o = rk(0.5 * rk(gpt_o) + 0.5 * rk(full_oof))
    bi_t = rk(0.5 * rk(gpt_t) + 0.5 * rk(te))
    np.save(PRED / "gpt2_3ep/gpt2_bidir_oof.npy", bi_o.astype(np.float64))
    np.save(PRED / "gpt2_3ep/gpt2_bidir_test.npy", bi_t.astype(np.float64))

    v4p_o = np.load(PRED / "P_super_v4p_hc_oof.npy").ravel()
    ev_rev = score_candidate_vs_v4p_hc(y, v4p_o, full_oof, folds)
    ev_bi = score_candidate_vs_v4p_hc(y, v4p_o, bi_o, folds)
    log(f"  gpt2_rev_2ep: solo={ev_rev['cand_solo_auc']:.7f} Δ={ev_rev['delta_auc']:+.3e}")
    log(f"  bidir 0.5/0.5: solo={ev_bi['cand_solo_auc']:.7f} Δ={ev_bi['delta_auc']:+.3e}")
    out = {
        "fold_aucs": fold_aucs,
        "rev": {"solo": ev_rev["cand_solo_auc"], "delta": ev_rev["delta_auc"]},
        "bidir": {"solo": ev_bi["cand_solo_auc"], "delta": ev_bi["delta_auc"]},
    }
    save_state(phase="stage_c_done", stage_c=out)
    log("=== STAGE C DONE ===")
    return out


# --------------------------------------------------------------------------- Stage D
def stage_d(y, folds, v4p_o, v4p_t, v5_o, v5_t) -> dict:
    from paradigm_stack_delta_eval import score_candidate_vs_v4p_hc
    from gpu_matrix_v2_core import per_feature_ks_filter

    log("=== STAGE D: P4_L4.2_ext + L4.3_ext + P_super_v6 lock ===")
    save_state(phase="stage_d")
    tr = pd.read_csv(S6 / "data/train.csv")
    te_df = pd.read_csv(S6 / "data/test.csv")
    ntr = len(y)
    test_ids = te_df["id"].to_numpy()

    # --- L4.2_ext: ratio sub-float ASCII strings ---
    def ratio_bulsah(df):
        amt = df["amount_usd"].to_numpy(np.float64)
        bal = df["account_balance_usd"].to_numpy(np.float64)
        # avg_30d may not exist — use txn_count proxy
        avg = df["txn_count_last_24h"].to_numpy(np.float64) if "txn_count_last_24h" in df.columns else np.ones(len(df))
        lim = df["card_age_months"].to_numpy(np.float64) + 1.0  # proxy if no credit_limit
        if "credit_limit" in df.columns:
            lim = df["credit_limit"].to_numpy(np.float64)
        vel = df["velocity_score"].to_numpy(np.float64)
        mr = df["merchant_risk_score"].to_numpy(np.float64)
        ratios = [
            amt / (bal + 1), amt / (avg + 1), amt / (lim + 1), bal / (lim + 1),
            vel / (mr + 1), amt / (vel + 1), bal / (mr + 1), avg / (lim + 1),
        ]
        cols = []
        for r in ratios:
            s = pd.Series(np.round(r, 6).astype(str))
            has_dot = s.str.contains(r"\.", regex=True).astype(np.float64).to_numpy()
            frac_len = s.str.split(".").str[-1].str.len().fillna(0).astype(np.float64).to_numpy()
            n_digits = s.str.replace(r"\D", "", regex=True).str.len().astype(np.float64).to_numpy()
            cols.extend([has_dot, frac_len, n_digits, np.log1p(np.abs(r))])
        return np.column_stack(cols)

    def ridge_lever_oof(feat_fn, name, n_inner_seeds=1):
        oof = np.zeros(ntr, dtype=np.float64)
        te_folds = []
        z_base = zrk(v5_o)
        z_base_t = zrk(v5_t)
        for fi, (a, b) in enumerate(folds):
            Fa = feat_fn(tr.iloc[a])
            Fb = feat_fn(tr.iloc[b])
            Fte = feat_fn(te_df)
            Fa, Fb, Fte, _ = per_feature_ks_filter(Fa, Fb, Fte)
            # multi-seed inner TE stabilization for L4.3_ext path handled outside
            sc = StandardScaler().fit(np.nan_to_num(Fa, 0.0))
            Xa = sc.transform(np.nan_to_num(Fa, 0.0))
            Xb = sc.transform(np.nan_to_num(Fb, 0.0))
            Xte = sc.transform(np.nan_to_num(Fte, 0.0))
            z_cal_a, z_cal_b, z_cal_te = calibrate_logit_to_prevalence(z_base[a], y[a], z_base[b], z_base_t)
            res_a = y[a].astype(np.float64) - expit(z_cal_a)
            preds_b, preds_te = [], []
            for sd in range(n_inner_seeds):
                # optional row subsample for seed diversity
                rs = np.random.RandomState(42 + fi * 17 + sd)
                if n_inner_seeds > 1:
                    idx = rs.choice(len(a), size=int(0.9 * len(a)), replace=False)
                else:
                    idx = np.arange(len(a))
                ridge = Ridge(alpha=500.0).fit(Xa[idx], res_a[idx])
                preds_b.append(ridge.predict(Xb))
                preds_te.append(ridge.predict(Xte))
            pb = np.mean(preds_b, axis=0)
            pte = np.mean(preds_te, axis=0)
            pa = Ridge(alpha=500.0).fit(Xa, res_a).predict(Xa)
            pb_al, pte_al = align_to_fold_train_ecdf(pa, pb, pte)
            oof[b] = pb_al
            te_folds.append(pte_al)
        te_pred = np.mean(te_folds, axis=0)
        ks = float(ks_2samp(rk(oof), rk(te_pred)).statistic)
        ev = score_candidate_vs_v4p_hc(y, v4p_o, oof, folds)
        log(f"  {name}: solo={ev['cand_solo_auc']:.7f} Δ={ev['delta_auc']:+.3e} KS={ks:.5f}")
        if ks > 0.005:
            log(f"  WARN {name} KS={ks:.5f} > 0.005 — still saving but flagging")
        np.save(PRED / f"matrix_v2/{name}_oof.npy", oof.astype(np.float64))
        np.save(PRED / f"matrix_v2/{name}_test.npy", te_pred.astype(np.float64))
        return oof, te_pred, ev, ks

    # L4.3_ext: 8-seed step-grid + 3-way cross
    FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]

    def l43_ext_feats(df):
        vel = df["velocity_score"].to_numpy(np.float64)
        mr = df["merchant_risk_score"].to_numpy(np.float64)
        v_step = np.rint(vel / 0.1)
        m_step = np.rint(mr / 0.02)
        flags = df[FLAGS].to_numpy(np.float64)
        bitmask = (flags * (2 ** np.arange(flags.shape[1]))).sum(1)  # 0..63
        three = (v_step % 10) * 1000 + (m_step % 10) * 100 + (bitmask % 64)
        return np.column_stack([
            v_step % 2, v_step % 5, v_step % 10,
            m_step % 2, m_step % 5, m_step % 10,
            np.abs(vel - v_step * 0.1) * 1e4,
            np.abs(mr - m_step * 0.02) * 1e4,
            (v_step % 2) * (m_step % 2),
            three,
            bitmask,
            (v_step % 5) * (m_step % 5),
        ]).astype(np.float64)

    l42e_o, l42e_t, ev42, ks42 = ridge_lever_oof(ratio_bulsah, "P4_L4.2_ext", n_inner_seeds=1)
    l43e_o, l43e_t, ev43, ks43 = ridge_lever_oof(l43_ext_feats, "P4_L4.3_ext", n_inner_seeds=8)

    # Load campaign members
    def maybe(path):
        p = Path(path)
        return np.load(p).ravel().astype(np.float64) if p.exists() else None

    gpt_o = maybe(PRED / "gpt2_3ep/gpt2_3ep_oof.npy")
    gpt_t = maybe(PRED / "gpt2_3ep/gpt2_3ep_test.npy")
    gb_o = maybe(PRED / "gpt2_3ep/gb_gpt2_3ep_super_oof.npy")
    gb_t = maybe(PRED / "gpt2_3ep/gb_gpt2_3ep_super_test.npy")
    rev_o = maybe(PRED / "gpt2_3ep/gpt2_rev_2ep_oof.npy")
    rev_t = maybe(PRED / "gpt2_3ep/gpt2_rev_2ep_test.npy")
    bi_o = maybe(PRED / "gpt2_3ep/gpt2_bidir_oof.npy")
    bi_t = maybe(PRED / "gpt2_3ep/gpt2_bidir_test.npy")

    # Orthogonal coordinate ascent vs v5_harvest
    z_base_o = zrk(v5_o)
    z_base_t = zrk(v5_t)
    base_auc = float(fast_auc(y, v5_o))
    candidates = [("gpt2_3ep", gpt_o, gpt_t), ("gb_super", gb_o, gb_t),
                  ("gpt2_rev", rev_o, rev_t), ("bidir", bi_o, bi_t),
                  ("L4.2_ext", l42e_o, l42e_t), ("L4.3_ext", l43e_o, l43e_t)]
    R_o, R_t, names = [], [], []
    for name, co, ct in candidates:
        if co is None or ct is None:
            continue
        zo, zt = zrk(co), zrk(ct)
        beta = float(np.dot(zo, z_base_o) / (np.dot(z_base_o, z_base_o) + 1e-12))
        ro = (zo - beta * z_base_o)
        rt = (zt - beta * z_base_t)
        sd = float(ro.std() + 1e-12)
        R_o.append(ro / sd)
        R_t.append(rt / sd)
        names.append(name)
        log(f"  basis {name}: beta={beta:+.4f}")

    R_o = np.column_stack(R_o)
    R_t = np.column_stack(R_t)
    M = R_o.shape[1]
    grid = (-0.08, -0.05, -0.03, -0.015, -0.008, 0.0, 0.008, 0.015, 0.03, 0.05, 0.08, 0.12)
    oof_z = np.zeros(ntr, dtype=np.float64)
    fold_w = []
    for a, b in folds:
        w = np.zeros(M)
        best = float(fast_auc(y[a], z_base_o[a]))
        for _ in range(3):
            for j in range(M):
                best_wj = w[j]
                for cw in grid:
                    wt = w.copy(); wt[j] = cw
                    sc = float(fast_auc(y[a], z_base_o[a] + R_o[a] @ wt))
                    if sc > best + 1e-9:
                        best, best_wj = sc, cw
                w[j] = best_wj
        oof_z[b] = z_base_o[b] + R_o[b] @ w
        fold_w.append(w)
    mean_w = np.mean(fold_w, axis=0)
    v6_oof = rk(oof_z)
    v6_te = rk(z_base_t + R_t @ mean_w)
    v6_auc = float(fast_auc(y, v6_oof))
    log(f"  P_super_v6_campaign nested AUC={v6_auc:.7f} (Δvs_v5={v6_auc-base_auc:+.3e}) w={dict(zip(names, [round(float(x),4) for x in mean_w]))}")
    np.save(PRED / "P_super_v6_campaign_oof.npy", v6_oof.astype(np.float64))
    np.save(PRED / "P_super_v6_campaign_test.npy", v6_te.astype(np.float64))

    # Submissions on ev4
    ev4_p = SUB / "sub_ev4.csv"
    if not ev4_p.exists():
        # try alternate name staged earlier
        alts = list(SUB.glob("sub_ev4*.csv"))
        ev4_p = alts[0] if alts else None
    maly_p = PRED / "submission.csv"  # malyshev staged here
    subs_meta = {}
    if ev4_p and ev4_p.exists():
        ev4 = pd.read_csv(ev4_p).set_index("id").loc[test_ids, "is_fraud"].to_numpy(np.float64)
        z_ev4 = zrk(ev4)
        delta = zrk(v6_te) - zrk(v6_te).mean()  # already rank; use residual vs mean
        # true orth residual of v6 vs ev4 on test via oof-fit beta
        if maly_p.exists():
            maly = pd.read_csv(maly_p).set_index("id").loc[test_ids, "is_fraud"].to_numpy(np.float64)
        else:
            maly = ev4
        # inject mean_w residual magnitude onto anchors
        dz = (zrk(v6_te) - zrk(v5_t))
        for fname, pred in [
            ("sub_v6_on_ev4_half.csv", rk(z_ev4 + 0.5 * dz)),
            ("sub_v6_on_ev4_1x.csv", rk(z_ev4 + 1.0 * dz)),
            ("sub_v6_pure.csv", v6_te),
        ]:
            fp = SUB / fname
            pd.DataFrame({"id": test_ids, "is_fraud": pred}).to_csv(fp, index=False)
            h = sha16(fp)
            subs_meta[fname] = {"sha256_16": h, "bytes": fp.stat().st_size}
            log(f"  wrote {fname} sha={h}")

    out = {
        "v6_auc": v6_auc,
        "delta_vs_v5": v6_auc - base_auc,
        "weights": dict(zip(names, map(float, mean_w))),
        "L4.2_ext": {"delta": ev42["delta_auc"], "ks": ks42},
        "L4.3_ext": {"delta": ev43["delta_auc"], "ks": ks43},
        "subs": subs_meta,
    }
    save_state(phase="stage_d_done", stage_d=out)
    log("=== STAGE D DONE ===")
    return out


def write_done(summary: dict) -> None:
    # Manifest of artifacts + sha
    arts = []
    for p in sorted(PRED.rglob("*.npy")):
        if "gpt2" in p.name or "v6_campaign" in p.name or "L4.2_ext" in p.name or "L4.3_ext" in p.name:
            arts.append({"path": str(p), "sha256_16": sha16(p), "bytes": p.stat().st_size})
    for p in sorted(SUB.glob("sub_v6*.csv")):
        arts.append({"path": str(p), "sha256_16": sha16(p), "bytes": p.stat().st_size})
    payload = {
        "status": "COMPLETE",
        "finished_utc": utc(),
        "summary": summary,
        "artifacts": arts,
        "pod_id": "mgfv9v9ltb348q",
        "ready_to_terminate": True,
    }
    DONEF.write_text(json.dumps(payload, indent=2, default=str))
    log(f"WROTE {DONEF} — local monitor may terminate pod")


def main():
    stage_only = os.environ.get("STAGE_ONLY", "").strip().upper()  # e.g. D
    if not stage_only:
        LOGF.write_text("")
    log(f"=== post_gpt2_3ep_campaign START (NO_SUBMIT) stage_only={stage_only or 'ALL'} ===")
    if not stage_only:
        wait_for_pid(WAIT_PID)

    y = pd.read_csv(S6 / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    v4p_o = np.load(PRED / "P_super_v4p_hc_oof.npy").ravel().astype(np.float64)
    v4p_t = np.load(PRED / "P_super_v4p_hc_test.npy").ravel().astype(np.float64)
    v5_o = np.load(PRED / "P_super_v5_harvest_oof.npy").ravel().astype(np.float64)
    v5_t = np.load(PRED / "P_super_v5_harvest_test.npy").ravel().astype(np.float64)

    summary = {}
    if STATED.exists():
        try:
            st = json.loads(STATED.read_text())
            for k in ("stage_a", "stage_b", "stage_c", "stage_d"):
                if k in st and st[k]:
                    summary[k[-1].upper()] = st[k]
        except Exception:
            pass
    try:
        run_a = stage_only in ("", "A", "ALL")
        run_b = stage_only in ("", "B", "ALL")
        run_c = stage_only in ("", "C", "ALL")
        run_d = stage_only in ("", "D", "ALL")
        if stage_only == "D":
            run_a = run_b = run_c = False
            run_d = True

        if run_a:
            summary["A"] = stage_a(y, folds, v4p_o, v4p_t, v5_o, v5_t)
        gpt_o = np.load(PRED / "gpt2_3ep/gpt2_3ep_oof.npy").ravel()
        gpt_t = np.load(PRED / "gpt2_3ep/gpt2_3ep_test.npy").ravel()
        h16_o = h16_t = None
        hp = PRED / "gpt2_3ep/gpt2_3ep_h16_oof.npy"
        if hp.exists():
            h16_o = np.load(hp)
            h16_t = np.load(PRED / "gpt2_3ep/gpt2_3ep_h16_test.npy")
        if run_b:
            summary["B"] = stage_b(y, folds, v4p_o, v4p_t, v5_o, v5_t, gpt_o, gpt_t, h16_o, h16_t)
        if run_c:
            summary["C"] = stage_c(y, folds, gpt_o, gpt_t)
        if run_d:
            summary["D"] = stage_d(y, folds, v4p_o, v4p_t, v5_o, v5_t)
        write_done(summary)
        log("=== CAMPAIGN COMPLETE SUCCESS ===")
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        save_state(phase="FATAL", error=str(e))
        # still write a DONE-with-error so monitor can decide
        DONEF.write_text(json.dumps({
            "status": "ERROR", "finished_utc": utc(), "error": str(e),
            "ready_to_terminate": False, "summary": summary,
        }, indent=2, default=str))
        raise


if __name__ == "__main__":
    main()
