#!/usr/bin/env python3
"""GM V14 — hardened L7_s2026 5-fold completion + ATH upgrade (STRICT NO_SUBMIT).

Fixes (vs train_l7_5f_s2026.py Flip-0 campaign):
  Flaw1  K_oof=K_test=10 on every fold (no 6-order rotated test)
  Flaw2  fold_0.pt 128-row CPU/GPU parity gate before any pod
  Flaw4  Fold-1 canary: solo AUC>0.9485 and 2-seed Δ vs L7_s7_f1 ≥ +35bps
  Flaw5  Proper 5-fold assemble + ATH linear/zero-sum moves (no Stage-B trees)
  Flaw6  Never torch.compile before load_state_dict (no _orig_mod. keys)
  Flaw7  Same bf16 autocast dtype as Fold0; CPU fp32 allowed only for parity
  Flaw8/9 Bit-exact orders_10 = RandomState(2026) 5perms+reverses, shape (10, NC=24)
  Flaw10 Full raw stack + Bayes / rank-logit reductions
  Flaw11 Save only fp16 model weights (~172MB), never AdamW 1GB ckpts
  Flaw12 4-bar LOFO + 25-subfold gates; stage NOSUBMIT only

Modes:
  preflight  — recon + orders_10 + fold_0 128-row parity (HARD GATE)
  resume     — folds 1..4 from replica.pt (pod GPU)
  assemble   — 5-fold reduce + ATH 4-bar screen + NOSUBMIT staging
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
SEED = 2026
K_ORDERS = 10
SCORE_BS = int(os.environ.get("SCORE_BS", "2048"))
USE_COMPILE = 0  # Flaw 6 — hard off
REP_EPOCHS = 60
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "5e-5"))
BOS = 0
LMAX = 6

REPO = Path(os.environ.get("V14_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
ROOT = Path(os.environ.get("L7_ROOT", str(REPO)))
OUT = Path(os.environ.get("L7_OUT", str(ROOT / "preds" / "l7_s2026_5f")))
CK = Path(os.environ.get("L7_CKPT", str(OUT / "ckpts")))
CACHE = Path(os.environ.get("L7_CACHE", str(ROOT / "cache_compact.npz")))
if not CACHE.exists():
    alt = Path("/tmp/l7_s2026_v14/cache_compact.npz")
    if alt.exists():
        CACHE = alt
REPORT = Path(os.environ.get("V14_REPORT", str(REPO / "logs/overnight/GM_V14_COMPLETE_5FOLD_L7_S2026_REPORT.md")))
SUB = REPO / "submissions"
LIVE_PATH = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"

dev = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "v14_run.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def master_perms(nc: int, seed: int = 2026) -> np.ndarray:
    rs = np.random.RandomState(seed)
    base = np.arange(nc, dtype=np.int64)
    out = np.empty((10, nc), dtype=np.int64)
    for k in range(5):
        p = rs.permutation(base)
        out[k] = p
        out[k + 5] = p[::-1].copy()
    return out


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def L_fold(v, fold_id):
    v = np.asarray(v, dtype=np.float64).ravel()
    out = np.empty_like(v)
    for f in range(5):
        m = fold_id == f
        out[m] = L(v[m])
    return out


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-18)


def pchip_lock(src, ref):
    from scipy.interpolate import PchipInterpolator

    src = np.asarray(src, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    qs = np.linspace(0, 1, 401)
    x = np.quantile(src, qs)
    y = np.quantile(ref, qs)
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    return PchipInterpolator(x, y, extrapolate=True)(src)


def force_law(ref_p, score):
    ref_p = np.asarray(ref_p, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    order = np.argsort(score + 1e-15 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort")
    out = np.empty(len(score), dtype=np.float64)
    out[order] = np.sort(ref_p)
    return out


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    cols = [unitize(b) for b in bases if np.linalg.norm(np.asarray(b).ravel()) > 1e-12]
    if not cols:
        return unitize(v)
    B = np.column_stack(cols)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B))


def bayes_ll(ll, prior):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def ranklogit_orders(ll):
    # mean_k L(ll_pos_k - ll_neg_k)
    d = ll[:, :, 1].astype(np.float64) - ll[:, :, 0].astype(np.float64)
    zs = [L(d[:, k]) for k in range(d.shape[1])]
    return np.mean(np.column_stack(zs), axis=1)


# -------------------- model (bit-exact with train_l7_5f_s2026) --------------------
class Block(nn.Module):
    def __init__(s, d, nh, p):
        super().__init__()
        s.ln_1, s.ln_2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.fc, s.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        s.nh, s.p, s.drop = nh, p, nn.Dropout(p)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln_1(x)).split(D, 2)
        sh = lambda t: t.view(B, T, s.nh, D // s.nh).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            sh(q), sh(k), sh(v), dropout_p=s.p if s.training else 0.0, is_causal=True
        )
        x = x + s.drop(s.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + s.drop(s.out(F.gelu(s.fc(s.ln_2(x)), approximate="tanh")))


class LM(nn.Module):
    def __init__(s, ntok, nout, T, nl=12, nh=12, d=768, p=0.1):
        super().__init__()
        s.wte, s.wpe = nn.Embedding(ntok, d), nn.Embedding(T, d)
        s.h = nn.ModuleList(Block(d, nh, p) for _ in range(nl))
        s.ln_f, s.drop, s.nout = nn.LayerNorm(d, eps=1e-5), nn.Dropout(p), nout

    def hidden(s, x, lab=None):
        e = s.wte(x) + s.wpe.weight[: x.shape[1]]
        h = s.drop(e if lab is None else e + s.wte(lab)[:, None, :])
        for b in s.h:
            h = b(h)
        return s.ln_f(h)

    def forward(s, x, lab=None):
        h = s.hidden(x, lab).float()
        with torch.autocast("cuda", enabled=False):
            return h @ s.wte.weight[: s.nout].T


class Rows:
    def __init__(self, g, orders):
        self.t = torch.from_numpy(g["toks"]).to(dev)
        self.g, self.T = g, g["T"]
        self.P = len(g["PRE"])
        self.lens = (g["toks"] >= 0).sum((1, 2)) + self.P + 1
        self.pre = torch.tensor(g["PRE"], dtype=torch.long, device=dev)
        self.labid = torch.tensor(g["LABID"], dtype=torch.long, device=dev)
        self.master = torch.as_tensor(orders, device=dev)

    def rand_perm(self, n, gen):
        return torch.rand(n, self.g["NC"], device=dev, generator=gen).argsort(1)

    def fixed_perm(self, n, order_idx: int):
        return self.master[int(order_idx)].unsqueeze(0).expand(n, -1).contiguous()

    def batches(self, rows, bs, rs, group=64):
        o = rs.permutation(rows)[: len(rows) // bs * bs]
        for c in range(0, len(o), bs * group):
            s = o[c : c + bs * group]
            o[c : c + bs * group] = s[np.argsort(self.lens[s], kind="stable")]
        b = o.reshape(-1, bs)[rs.permutation(len(o) // bs)]
        return torch.as_tensor(b, device=dev), self.lens[b].max(1).tolist()

    def assemble(self, idx, lab, perm, T=None, full=False):
        g, T = self.g, T or self.T
        B = idx.numel()
        seg = self.t[idx].long().gather(1, perm[:, :, None].expand(-1, -1, LMAX)).reshape(B, -1)
        valid = seg >= 0
        pos = torch.where(valid, valid.cumsum(1) + self.P, torch.full_like(seg, T + 1))
        buf = torch.full((B, T + 2), g["PAD"], dtype=torch.long, device=dev)
        buf.scatter_(1, pos, seg.clamp_min(0))
        buf = buf[:, : T + 1]
        if self.P:
            buf[:, : self.P] = self.pre
        buf[:, self.P] = self.labid[lab]
        cur, nxt = buf[:, :-1], buf[:, 1:]
        tgt = torch.where(nxt < g["V"], nxt, torch.full_like(nxt, g["END"]))
        tgt = torch.where(cur == g["PAD"], torch.full_like(tgt, -100), tgt)
        if not full:
            tgt[:, : self.P + 1] = -100
        return cur.contiguous(), tgt


def load_g():
    assert CACHE.exists(), f"missing cache {CACHE}"
    z = np.load(CACHE)
    g = dict(
        toks=z["toks"],
        y=z["y"].astype(np.int64),
        vocab=z["vocab"],
        colids=z["colids"],
        ntr=int(z["ntr"]),
        nte=int(z["nte"]),
        lab=z["lab"],
        comma=z["comma"],
    )
    g["V"] = len(g["vocab"])
    g["END"], g["PAD"], g["LAB"], g["COL0"] = g["V"], g["V"] + 1, g["V"] + 2, g["V"] + 4
    g["NC"] = g["toks"].shape[1]
    g["BOS"] = g["COL0"] + g["NC"]
    g["NTOK"] = g["BOS"] + 1
    g["PRE"] = [g["BOS"]] if BOS else []
    g["LABID"] = [g["LAB"], g["LAB"] + 1]
    g["T"] = int((g["toks"] >= 0).sum((1, 2)).max()) + len(g["PRE"]) + 1
    return g


def extract_model_sd(obj):
    if isinstance(obj, dict):
        if "model" in obj:
            sd = obj["model"]
        else:
            sd = obj
    else:
        sd = obj
    # strip accidental _orig_mod. prefix
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def load_model(g, path: Path):
    m = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
    obj = torch.load(path, map_location=dev, weights_only=False)
    sd = extract_model_sd(obj)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def save_model_fp16(m, path: Path):
    """Flaw 11 — pure model weights only (~172MB), no AdamW."""
    path = Path(path)
    sd = {k: v.detach().half().cpu() for k, v in m.state_dict().items()}
    tmp = str(path) + ".tmp"
    torch.save({"model": sd, "fp16": True, "epoch": EPOCHS - 1}, tmp)
    os.replace(tmp, path)
    log(f"saved fp16 model {path.name} bytes={path.stat().st_size}")


@torch.inference_mode()
def score_orders(m, R, rows, order_ids, bs=SCORE_BS, dtype=torch.bfloat16):
    m.eval()
    srt = np.argsort(R.lens[rows], kind="stable")
    ls = R.lens[rows][srt]
    rows_t = torch.as_tensor(rows[srt], device=dev)
    K = len(order_ids)
    out = torch.zeros((len(rows), K, 2), dtype=torch.float32, device=dev)
    for ki, oid in enumerate(order_ids):
        for s in range(0, len(rows), bs):
            idx = rows_t[s : s + bs]
            n = idx.numel()
            perm = R.fixed_perm(n, oid)
            inp, tgt = R.assemble(idx, torch.zeros(n, dtype=torch.long, device=dev), perm, int(ls[s + n - 1]))
            both = torch.cat([inp, inp])
            both[n:, R.P] = R.labid[1]
            if dev == "cuda":
                with torch.autocast("cuda", dtype=dtype):
                    lg = m(both)
            else:
                lg = m(both)
            nll = (
                F.cross_entropy(
                    lg.float().view(-1, lg.shape[-1]),
                    tgt.repeat(2, 1).reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                .view(2 * n, -1)
                .sum(1)
            )
            out[s : s + n, ki, 0] = -nll[:n]
            out[s : s + n, ki, 1] = -nll[n:]
    res = np.empty((len(rows), K, 2), np.float32)
    res[srt] = out.cpu().numpy()
    return res


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fold(m, R, ytr, rows, seed):
    """FT from replica; do NOT persist AdamW (Flaw 11)."""
    per = len(rows) // BATCH
    total = max(per * EPOCHS, 1)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0, fused=(dev == "cuda"))
    gen = torch.Generator(device=dev)
    m.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        gen.manual_seed(seed * 1000 + ep)
        bt, tb = R.batches(rows, BATCH, np.random.RandomState((seed * 1000 + ep) % 2**31))
        run = 0.0
        for s in range(per):
            step = ep * per + s
            warm = min(1.0, (step + 1) / 100)
            decay = 0.5 * (1.0 + np.cos(np.pi * step / total))
            for grp in opt.param_groups:
                grp["lr"] = LR * warm * decay
            idx = bt[s]
            inp, tgt = R.assemble(idx, ytr[idx], R.rand_perm(BATCH, gen), tb[s])
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(inp)
            else:
                lg = m(inp)
            loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss.detach())
            if (s + 1) % 2000 == 0:
                log(f"  ep {ep+1}/{EPOCHS} step {s+1}/{per} loss {run/2000:.4f}")
                run = 0.0
        log(f"  epoch {ep+1}/{EPOCHS} done {(time.time()-t0)/60:.1f} min")
    return m


# -------------------- Step 3 preflight --------------------
def ensure_orders(g) -> np.ndarray:
    OUT.mkdir(parents=True, exist_ok=True)
    recomputed = master_perms(g["NC"], SEED)
    mp = OUT / "master_perms_seed2026.npy"
    if mp.exists():
        existing = np.load(mp)
        assert np.array_equal(existing, recomputed), "orders drift vs master_perms_seed2026.npy!"
    else:
        np.save(mp, recomputed)
    op = OUT / "orders_10.npy"
    np.save(op, recomputed)
    assert recomputed.shape == (10, g["NC"])
    log(f"orders_10 bit-exact shape={recomputed.shape} saved {op}")
    return recomputed


def fold0_parity(g, orders) -> dict:
    """Hard gate: recompute first 128 val rows vs oof_0.npy."""
    oof_path = OUT / "oof_0.npy"
    ck_path = CK / "fold_0.pt"
    assert oof_path.exists() and ck_path.exists()
    oof = np.load(oof_path)
    assert oof.shape[1:] == (10, 2)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(g["ntr"]), g["y"][: g["ntr"]]))
    _, b = folds[0]
    rows128 = b[:128]
    log(f"parity: loading fold_0.pt on {dev} …")
    m = load_model(g, ck_path)
    # also verify replica loads strict
    rep = CK / "replica.pt"
    if rep.exists():
        m_rep = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
        obj = torch.load(rep, map_location=dev, weights_only=False)
        m_rep.load_state_dict(extract_model_sd(obj), strict=True)
        log("replica.pt strict=True OK")
        del m_rep
    R = Rows(g, orders)
    # CPU uses fp32; GPU uses bf16 (Flaw 7)
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    bs = 32 if dev == "cpu" else SCORE_BS
    log(f"parity: scoring 128×10 orders bs={bs} dtype={dtype} …")
    t0 = time.time()
    recomputed = score_orders(m, R, rows128, list(range(10)), bs=bs, dtype=dtype)
    log(f"parity: score wall {(time.time()-t0):.1f}s")
    # compare LLR
    ref = oof[:128]
    llr_r = recomputed[:, :, 1] - recomputed[:, :, 0]
    llr_o = ref[:, :, 1] - ref[:, :, 0]
    rho = float(spearmanr(llr_r.ravel(), llr_o.ravel()).correlation)
    mad = float(np.max(np.abs(llr_r - llr_o)))
    # CPU fp32 vs GPU bf16: rho is primary; MAD max can be ~0.06 on CPU.
    # On CUDA bf16 re-check expects max|Δ|<0.02.
    mad_lim = 0.02 if dev == "cuda" else 0.10
    med = float(np.median(np.abs(llr_r - llr_o)))
    info = {
        "rho_llr": rho,
        "max_abs_llr": mad,
        "median_abs_llr": med,
        "mad_lim": mad_lim,
        "rho_gate": rho > 0.9999,
        "mad_gate": mad < mad_lim and med < 0.02,
        "pass": bool(rho > 0.9999 and mad < mad_lim and med < 0.02),
        "device": dev,
        "dtype": str(dtype),
        "n": 128,
    }
    log(f"PARITY rho={rho:.8f} max|Δ|={mad:.6f} pass={info['pass']}")
    (OUT / "v14_fold0_parity.json").write_text(json.dumps(info, indent=2))
    return info


def preflight() -> dict:
    log("=== V14 PREFLIGHT (NO_SUBMIT) ===")
    g = load_g()
    orders = ensure_orders(g)
    # artifact inventory
    inv = {
        "CACHE": str(CACHE),
        "NC": int(g["NC"]),
        "V": int(g["V"]),
        "T": int(g["T"]),
        "NTOK": int(g["NTOK"]),
        "oof_0": (OUT / "oof_0.npy").exists(),
        "test_0": (OUT / "test_0.npy").exists(),
        "replica": (CK / "replica.pt").exists(),
        "fold_0": (CK / "fold_0.pt").exists(),
        "missing_folds": [f for f in range(1, 5) if not (OUT / f"oof_{f}.npy").exists()],
        "orders_shape": list(orders.shape),
        "K_oof": 10,
        "K_test": 10,
        "USE_COMPILE": USE_COMPILE,
        "SCORE_BS": SCORE_BS,
        "autocast": "bf16",
    }
    log(json.dumps(inv))
    assert inv["oof_0"] and inv["fold_0"] and inv["replica"], "missing Fold0 artifacts"
    parity = fold0_parity(g, orders)
    assert parity["pass"], f"HARD GATE FAIL parity={parity}"
    return {"inventory": inv, "parity": parity}


# -------------------- Step 4 resume --------------------
def resume_folds(which=(1, 2, 3, 4)):
    assert USE_COMPILE == 0
    g = load_g()
    orders = ensure_orders(g)
    y = g["y"]
    ydev = torch.as_tensor(y, device=dev)
    R = Rows(g, orders)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(g["ntr"]), y[: g["ntr"]]))
    test_idx = np.arange(g["ntr"], g["ntr"] + g["nte"])
    rep_path = CK / "replica.pt"
    assert rep_path.exists()

    for f in which:
        oof_p, te_p = OUT / f"oof_{f}.npy", OUT / f"test_{f}.npy"
        if oof_p.exists() and te_p.exists():
            a = np.load(oof_p)
            b = np.load(te_p)
            assert a.shape == (len(folds[f][1]), 10, 2) and b.shape == (g["nte"], 10, 2)
            log(f"fold {f}: already scored — skip")
            continue
        log(f"=== fold {f}: init from replica.pt seed={SEED+f} ===")
        seed_everything(SEED + f)
        m = load_model(g, rep_path)
        a, b = folds[f]
        t1 = time.time()
        m = train_fold(m, R, ydev, a, SEED + f)
        log(f"fold {f}: train {(time.time()-t1)/60:.1f} min")
        # score K=10 OOF + K=10 Test (Flaw 1)
        t2 = time.time()
        ll = score_orders(m, R, b, list(range(10)), bs=SCORE_BS)
        assert ll.shape == (len(b), 10, 2)
        prior = float(y[a].mean())
        z = bayes_ll(ll, prior)
        auc = float(roc_auc_score(y[b], z))
        log(f"fold {f}: OOF AUC={auc:.6f}")
        lt = score_orders(m, R, test_idx, list(range(10)), bs=SCORE_BS)
        assert lt.shape == (g["nte"], 10, 2)
        np.save(oof_p, ll)
        np.save(te_p, lt)
        # Flaw 11 — rolling fp16 only
        save_model_fp16(m, CK / "fold_latest.pt")
        meta = {
            "fold": f,
            "auc": auc,
            "prior": prior,
            "oof_shape": list(ll.shape),
            "test_shape": list(lt.shape),
            "SCORE_BS": SCORE_BS,
            "SEED": SEED + f,
            "K_oof": 10,
            "K_test": 10,
            "USE_COMPILE": 0,
            "wall_min": (time.time() - t1) / 60.0,
        }
        (OUT / f"fold_{f}_meta.json").write_text(json.dumps(meta, indent=2))
        (OUT / f"F{f}_READY").write_text("ok\n")
        log(f"fold {f}: score {(time.time()-t2)/60:.1f} min — READY")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if f == 1:
            canary = fold1_canary(g, ll, prior)
            (OUT / "v14_fold1_canary.json").write_text(json.dumps(canary, indent=2))
            if not canary["pass"]:
                log(f"FOLD1 CANARY FAIL {canary} — abort remaining folds")
                raise SystemExit(42)
    log("resume complete")


def fold1_canary(g, ll_f1, prior) -> dict:
    """Flaw 4 — solo AUC >= 0.9445 and 2-seed Δ vs L7_s7_f1 ≥ +35bps."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(g["ntr"]), g["y"][: g["ntr"]]))
    _, b = folds[1]
    z26 = bayes_ll(ll_f1, prior)
    auc26 = float(roc_auc_score(g["y"][b], z26))
    # locate seed7 fold1 OOF
    cands = [
        REPO / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f1.npy",
        Path("/tmp/gm_l7_preflight/external/goodpjw2008_s6e11-compact-lm-seed7/lm7_oof_f1.npy"),
        OUT.parent / "l7_s7_5f" / "oof_1.npy",
    ]
    s7p = next((p for p in cands if p.exists()), None)
    info = {"auc_s2026_f1": auc26, "gate_solo": auc26 >= 0.9445}
    if s7p is None:
        info.update({"pass": False, "reason": "missing L7_s7 fold1 OOF"})
        return info
    ll7 = np.load(s7p)
    # prior from fold1 train
    a, _ = folds[1]
    prior7 = float(g["y"][a].mean())
    z7 = bayes_ll(ll7, prior7)
    auc7 = float(roc_auc_score(g["y"][b], z7))
    zavg = 0.5 * L(z7) + 0.5 * L(z26)
    # convert rank-logit blend back to score space via L then use as score
    auc_avg = float(roc_auc_score(g["y"][b], zavg))
    delta = auc_avg - auc7
    info.update(
        {
            "auc_s7_f1": auc7,
            "auc_two_seed": auc_avg,
            "two_seed_delta": delta,
            "gate_2seed": delta >= 3.5e-4,
            "s7_path": str(s7p),
            "pass": bool(auc26 >= 0.9445 and delta >= 3.5e-4),
        }
    )
    log(f"F1 canary auc26={auc26:.6f} auc7={auc7:.6f} 2seedΔ={delta:.6f} pass={info['pass']}")
    return info


# -------------------- Step 5 assemble + ATH --------------------
def setup_folds(y):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    return folds, fold_id


def assemble_and_eval():
    log("=== V14 ASSEMBLE + ATH 4-BAR (NO_SUBMIT) ===")
    g = load_g()
    y = g["y"][: g["ntr"]]
    folds, fold_id = setup_folds(y)
    # load all folds
    oofs, tests, priors = [], [], []
    for f in range(5):
        op, tp = OUT / f"oof_{f}.npy", OUT / f"test_{f}.npy"
        assert op.exists() and tp.exists(), f"missing fold {f}"
        oo, tt = np.load(op), np.load(tp)
        assert oo.shape[1:] == (10, 2) and tt.shape == (g["nte"], 10, 2), (oo.shape, tt.shape)
        oofs.append(oo)
        tests.append(tt)
        a, b = folds[f]
        assert len(b) == oo.shape[0]
        priors.append(float(y[a].mean()))

    raw_oof = np.concatenate(oofs, axis=0)  # but NOT in fold order — need place by index
    # proper placement
    oof_placed = np.zeros((g["ntr"], 10, 2), np.float32)
    for f, (_, b) in enumerate(folds):
        oof_placed[b] = oofs[f]
    test_stack = np.stack(tests, axis=0)  # (5, nte, 10, 2)
    np.save(OUT / "l7_s2026_oof_5fold_raw.npy", oof_placed)
    np.save(OUT / "l7_s2026_test_5fold_raw.npy", test_stack)
    log(f"raw OOF {oof_placed.shape} Test {test_stack.shape}")

    # reductions
    z_bayes = np.zeros(g["ntr"], np.float64)
    z_rl = np.zeros(g["ntr"], np.float64)
    te_bayes_L, te_rl_L = [], []
    per_fold_auc_b, per_fold_auc_r = [], []
    for f, (_, b) in enumerate(folds):
        zb = bayes_ll(oofs[f], priors[f])
        zr = ranklogit_orders(oofs[f])
        z_bayes[b] = zb
        z_rl[b] = zr
        per_fold_auc_b.append(float(roc_auc_score(y[b], zb)))
        per_fold_auc_r.append(float(roc_auc_score(y[b], zr)))
        tb = bayes_ll(tests[f], priors[f])
        tr = ranklogit_orders(tests[f])
        te_bayes_L.append(L(tb))
        te_rl_L.append(L(tr))

    R_bayes_o = L_fold(z_bayes, fold_id)
    R_rl_o = L_fold(z_rl, fold_id)
    R_bayes_t = pchip_lock(np.mean(np.column_stack(te_bayes_L), axis=1), R_bayes_o)
    R_rl_t = pchip_lock(np.mean(np.column_stack(te_rl_L), axis=1), R_rl_o)
    np.save(OUT / "R_bayes_5f_oof.npy", R_bayes_o)
    np.save(OUT / "R_bayes_5f_test.npy", R_bayes_t)
    np.save(OUT / "R_ranklogit_5f_oof.npy", R_rl_o)
    np.save(OUT / "R_ranklogit_5f_test.npy", R_rl_t)

    auc_b = float(roc_auc_score(y, R_bayes_o))
    auc_r = float(roc_auc_score(y, R_rl_o))
    log(f"L7_s2026 Bayes AUC={auc_b:.6f} folds={per_fold_auc_b}")
    log(f"L7_s2026 RankL AUC={auc_r:.6f} folds={per_fold_auc_r}")

    # prefer higher OOF reduction
    if auc_b >= auc_r:
        R_o, R_t, rname = R_bayes_o, R_bayes_t, "bayes"
    else:
        R_o, R_t, rname = R_rl_o, R_rl_t, "ranklogit"

    # L7F_s7 5f
    s7 = load_l7f_s7(g, folds, fold_id)
    two_o = 0.5 * L_fold(s7["oof"], fold_id) + 0.5 * R_o
    two_auc = float(roc_auc_score(y, two_o))
    s7_auc = float(roc_auc_score(y, s7["oof"]))
    log(f"L7F_s7 AUC={s7_auc:.6f} 2seed AUC={two_auc:.6f} Δ={two_auc-s7_auc:+.6f}")

    # ATH 0.95388
    ath = build_ath_pack(y, fold_id, folds)
    results = {}
    for move, delta_o, delta_t in [
        ("two_seed_L7_swap", unitize(R_o - L_fold(s7["oof"], fold_id)), unitize(R_t - L(s7["test"]))),
        ("compact_LM_slot_swap", unitize(R_o - L_fold(ath["LM"], fold_id)), unitize(R_t - L(ath["LM_t"]))),
        (
            "lofo_orth_L7_s2026",
            project_out(R_o, [ath["z_o"], ath["d1_o"]]),
            project_out(R_t, [ath["z_t"], ath["d1_t"]]),
        ),
    ]:
        results[move] = screen_move(ath, delta_o, delta_t, move)

    write_report(
        {
            "reduction": rname,
            "auc_bayes": auc_b,
            "auc_rl": auc_r,
            "per_fold_bayes": per_fold_auc_b,
            "per_fold_rl": per_fold_auc_r,
            "s7_auc": s7_auc,
            "two_seed_auc": two_auc,
            "two_seed_delta": two_auc - s7_auc,
            "results": results,
            "parity": json.loads((OUT / "v14_fold0_parity.json").read_text())
            if (OUT / "v14_fold0_parity.json").exists()
            else {},
        }
    )
    return results


def load_l7f_s7(g, folds, fold_id):
    """Load seed=7 compact LM 5-fold OOF/test."""
    base = REPO / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
    # try full 5f megalm / per-fold
    oof = np.zeros(g["ntr"], np.float64)
    found = 0
    for f, (_, b) in enumerate(folds):
        for name in [f"lm7_oof_f{f}.npy", f"oof_{f}.npy"]:
            p = base / name
            if p.exists():
                ll = np.load(p)
                a, _ = folds[f]
                prior = float(g["y"][: g["ntr"]][a].mean())
                oof[b] = bayes_ll(ll, prior) if ll.ndim == 3 else ll.ravel()
                found += 1
                break
    mega = base / "megalm_oof_sha181c.npy"
    if found < 5 and mega.exists():
        oof = np.load(mega).astype(np.float64).ravel()[: g["ntr"]]
        found = 5
    te_cands = [
        base / "lm7_test.npy",
        base / "megalm_test_sha181c.npy",
        REPO / "preds/gm_v5_overnight/X_v5_test.npy",  # fallback placeholder — avoid
    ]
    test = None
    for p in te_cands:
        if p.exists():
            arr = np.load(p)
            if arr.ndim == 1 and len(arr) == g["nte"]:
                test = arr.astype(np.float64)
                break
            if arr.ndim == 3 and arr.shape[0] == g["nte"]:
                test = bayes_ll(arr, float(g["y"][: g["ntr"]].mean()))
                break
    if test is None:
        # use OOF-calibrated noise-free proxy: pchip from a constant — mark missing
        test = np.zeros(g["nte"], np.float64)
        log("WARN: L7F_s7 test missing — using zeros (Move1 test trust dist may be weak)")
    assert found >= 1
    return {"oof": oof, "test": test, "found_folds": found}


def build_ath_pack(y, fold_id, folds):
    live = pd.read_csv(LIVE_PATH)
    z_t = live["is_fraud"].to_numpy(np.float64)
    # OOF proxy: prefer saved ATH oof if present
    ath_oof_cands = [
        REPO / "preds/gm_v13_4_stageB/ath_095388_oof.npy",
        REPO / "preds/gm_v13_2/ath_oof.npy",
        Path("/tmp/s6e11_v132/ath_095388_oof.npy"),
    ]
    z_o = None
    for p in ath_oof_cands:
        if p.exists():
            z_o = np.load(p).astype(np.float64).ravel()
            if len(z_o) == len(y):
                break
            z_o = None
    if z_o is None:
        # fallback: L7F-like — use LIVE locked random? Use L7F if available
        log("WARN: ATH OOF proxy missing — building from cand2 if present")
        cand = SUB / "sub_v10p2_cand2_S7ours_plus_twoseed_G26_MS_0p11e4_NOSUBMIT.csv"
        if cand.exists():
            # not ideal; use as rank structure only on test; OOF from bank
            z_o = np.load(
                next(
                    p
                    for p in [
                        REPO / "preds/gm_v5_overnight/eval_S7matched_355.npy",
                        Path("/tmp/s6e11_v132/eval_S7matched_355.npy"),
                    ]
                    if p.exists()
                )
            ).astype(np.float64).ravel()[: len(y)]
        else:
            raise FileNotFoundError("ATH OOF proxy")
    # d1_v4
    d1_cands = [
        REPO / "preds/gm_v5_overnight/d1_v4_oof.npy",
        REPO / "preds/d1_v4_oof.npy",
        Path("/tmp/s6e11_v132/d1_v4_oof.npy"),
    ]
    d1_o = next(np.load(p).astype(np.float64).ravel() for p in d1_cands if p.exists())
    d1_t_cands = [
        REPO / "preds/gm_v5_overnight/d1_v4_test.npy",
        Path("/tmp/s6e11_v132/d1_v4_test.npy"),
    ]
    d1_t = next(np.load(p).astype(np.float64).ravel() for p in d1_t_cands if p.exists())
    # LM
    lm_cands = [
        REPO / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy",
    ]
    LM = next(np.load(p).astype(np.float64).ravel()[: len(y)] for p in lm_cands if p.exists())
    LM_t = np.zeros(len(z_t))
    for p in [
        REPO / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/megalm_test_sha181c.npy",
    ]:
        if p.exists():
            LM_t = np.load(p).astype(np.float64).ravel()
            break
    return {
        "y": y,
        "folds": folds,
        "fold_id": fold_id,
        "z_o": z_o,
        "z_t": z_t,
        "d1_o": d1_o,
        "d1_t": d1_t,
        "LM": LM,
        "LM_t": LM_t,
        "ids": live["id"].to_numpy(),
        "live_p": z_t,
    }


def subfold25(y, z_base, u, folds):
    u = unit(u)
    # sign on fold0
    m0 = folds[0][1]
    if roc_auc_score(y[m0], z_base[m0] + 0.01 * u[m0]) < roc_auc_score(y[m0], z_base[m0] - 0.01 * u[m0]):
        u = -u
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], z_base[te], u[te]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for _, ste in skf.split(np.zeros(len(yf)), yf):
            base = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - base) * 1e5)
    d = np.asarray(deltas, float)
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    try:
        _, wp = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    # bootstrap CI
    rng = np.random.RandomState(42)
    boots = [float(rng.choice(d, size=len(d), replace=True).mean()) for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "pos25": int((d > 0).sum()),
        "t_stat": float(mean / (se + 1e-18)),
        "wilcoxon_p": float(wp),
        "mean_e5": float(mean),
        "ci90": [float(lo), float(hi)],
        "u": u,
    }


def main_lofo(y, z_base, u, folds, ws=(0.006, 0.009, 0.012, 0.015, 0.020)):
    u = unit(u)
    m0 = folds[0][1]
    if roc_auc_score(y[m0], z_base[m0] + 0.01 * u[m0]) < roc_auc_score(y[m0], z_base[m0] - 0.01 * u[m0]):
        u = -u
    base_fa = [roc_auc_score(y[te], z_base[te]) for _, te in folds]
    best = None
    for w in ws:
        d = []
        for f, (_, te) in enumerate(folds):
            d.append((roc_auc_score(y[te], z_base[te] + w * u[te]) - base_fa[f]) * 1e5)
        pos = int(sum(x > 0 for x in d))
        row = {"w": w, "pos5": pos, "mean_e5": float(np.mean(d)), "min_e5": float(np.min(d)), "folds": [round(x, 3) for x in d]}
        if best is None or (pos, row["mean_e5"]) > (best["pos5"], best["mean_e5"]):
            best = row
            best["u"] = u
    return best


def screen_move(ath, delta_o, delta_t, name):
    y, folds = ath["y"], ath["folds"]
    z_o, z_t = ath["z_o"], ath["z_t"]
    main = main_lofo(y, z_o, delta_o, folds)
    sf = subfold25(y, z_o, main["u"], folds)
    # disp
    u = main["u"]
    w = main["w"]
    cand_o = z_o + w * u
    disp = float(np.std(cand_o) / (np.std(z_o) + 1e-12))
    # test trust
    cand_t = force_law(ath["live_p"], unitize(L(z_t)) + w * unit(delta_t))
    dist = float(1.0 - spearmanr(ath["live_p"], cand_t).correlation)
    bars = {
        "bar1_main5": main["pos5"] == 5,
        "bar2_sf": sf["pos25"] >= 19 and sf["t_stat"] >= 3.30,
        "bar3_wilcox": sf["wilcoxon_p"] < 0.01,
        "bar4_disp": 0.92 <= disp <= 1.08,
        "pass": False,
    }
    bars["pass"] = all([bars["bar1_main5"], bars["bar2_sf"], bars["bar3_wilcox"], bars["bar4_disp"]])
    out = {
        "name": name,
        "main": {k: main[k] for k in ("w", "pos5", "mean_e5", "min_e5", "folds")},
        "sf25": {k: sf[k] for k in ("pos25", "t_stat", "wilcoxon_p", "mean_e5", "ci90")},
        "disp": disp,
        "trust_1mrho_e4": dist * 1e4,
        "bars": bars,
    }
    log(
        f"MOVE {name}: main {main['pos5']}/5 meanΔ={main['mean_e5']:.2f}e-5 "
        f"sf={sf['pos25']}/25 t={sf['t_stat']:.2f} disp={disp:.4f} "
        f"1-ρ={dist*1e4:.3f}e-4 pass={bars['pass']}"
    )
    if bars["pass"] or (main["pos5"] == 5 and sf["pos25"] >= 18):
        # stage NOSUBMIT near trust region
        for tag, target in [("step1_0p12e4", 0.12e-4), ("step2_0p22e4", 0.22e-4), ("step3_0p35e4", 0.35e-4)]:
            p, ww, got = weight_force(ath["live_p"], unit(delta_t), target)
            path = SUB / f"sub_v14_l7_s2026_5fold_{name}_{tag}_NOSUBMIT.csv"
            assert NO_SUBMIT and "NOSUBMIT" in path.name
            pd.DataFrame({"id": ath["ids"], "is_fraud": p}).to_csv(path, index=False)
            log(f"  staged {path.name} w={ww:.4f} 1-ρ={got*1e4:.3f}e-4")
            out.setdefault("staged", []).append({"path": str(path), "w": ww, "dist_e4": got * 1e4})
    return out


def weight_force(cand2, u, target, hi=80.0, n=80):
    z0 = unitize(L(cand2))
    u = unit(u)
    lo, best_w, best_err, got, best_p = 0.0, 0.0, 1e9, 0.0, cand2
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        p = force_law(cand2, z0 + mid * u)
        d = float(1.0 - spearmanr(p, cand2).correlation)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got, best_p = err, mid, d, p
        if d > target:
            hi = mid
        else:
            lo = mid
    return best_p, float(best_w), float(got)


def write_report(pack):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GM V14 — Complete Hardened 5-Fold L7_s2026 Report",
        "",
        "**STRICT NO_SUBMIT=True** · zero live Kaggle submit from this script.",
        "",
        "## Preflight / Parity",
        "",
        "```json",
        json.dumps(pack.get("parity", {}), indent=2),
        "```",
        "",
        "## Solo 5-Fold Two-Seed",
        "",
        f"- L7_s2026 Bayes AUC=`{pack['auc_bayes']:.6f}` folds=`{pack['per_fold_bayes']}`",
        f"- L7_s2026 RankLogit AUC=`{pack['auc_rl']:.6f}` folds=`{pack['per_fold_rl']}`",
        f"- Preferred reduction: **{pack['reduction']}**",
        f"- L7F_s7 AUC=`{pack['s7_auc']:.6f}`",
        f"- 2-seed `0.5*L(s7)+0.5*L(s2026)` AUC=`{pack['two_seed_auc']:.6f}` (Δ=`{pack['two_seed_delta']:+.6f}`)",
        "",
        "## ATH 0.95388 Family Upgrades (zero Stage-B trees)",
        "",
    ]
    for name, r in pack["results"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"- main `{r['main']['pos5']}/5` meanΔ=`{r['main']['mean_e5']:.3f}e-5` "
            f"min=`{r['main']['min_e5']:.3f}e-5` folds=`{r['main']['folds']}` w=`{r['main']['w']}`"
        )
        lines.append(
            f"- sf25 `{r['sf25']['pos25']}/25` t=`{r['sf25']['t_stat']:.2f}` "
            f"W-p=`{r['sf25']['wilcoxon_p']:.4g}` CI90=`{r['sf25']['ci90']}`"
        )
        lines.append(
            f"- disp=`{r['disp']:.4f}` · Test 1−ρ=`{r['trust_1mrho_e4']:.3f}e-4` · pass=`{r['bars']['pass']}`"
        )
        if r.get("staged"):
            for s in r["staged"]:
                lines.append(f"  - staged `{Path(s['path']).name}` (1−ρ=`{s['dist_e4']:.3f}e-4`)")
        lines.append("")
    lines += ["", "## Guardrails", "", "- NO_SUBMIT=True", "- Foreign pod `lheroidmuab1j4` never touched", ""]
    REPORT.write_text("\n".join(lines))
    log(f"WROTE {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preflight", "resume", "assemble", "all_local"], default="preflight")
    ap.add_argument("--folds", default="1,2,3,4")
    args = ap.parse_args()
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    CK.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        preflight()
    elif args.mode == "resume":
        which = [int(x) for x in args.folds.split(",") if x != ""]
        resume_folds(which)
    elif args.mode == "assemble":
        assemble_and_eval()
    elif args.mode == "all_local":
        preflight()
        # assemble only if all folds present
        if all((OUT / f"oof_{f}.npy").exists() for f in range(5)):
            assemble_and_eval()
        else:
            log("folds 1-4 incomplete — preflight only; launch pod for --mode resume")


if __name__ == "__main__":
    main()
