#!/usr/bin/env python3
"""Compact-token target-first LM supporting gpt2 (124M) and gpt2-medium (355M) + incremental orthogonal screening vs v8 + Part 12."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6_ROOT", "/workspace/s6e11"))
HERE = Path(__file__).resolve().parent
TAG = os.environ.get("TAG", "gpt2_med_4ep")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2-medium")  # "gpt2" or "gpt2-medium"
OUT = HERE / "out" / TAG
CK = HERE / "ckpt" / TAG
BOS = int(os.environ.get("BOS", 0))
LABEMB = int(os.environ.get("LABEMB", 0))
NATURAL = int(os.environ.get("NATURAL", 0))
PARITY = int(os.environ.get("PARITY", 0))
CACHE = HERE / ("cache_nat.npz" if NATURAL else "cache.npz")
FMT = f"{MODEL_NAME}_b{BOS}_l{LABEMB}_n{NATURAL}"
for d in (OUT, CK):
    d.mkdir(parents=True, exist_ok=True)
dev = "cuda"
LMAX = 16 if NATURAL else 6
EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH = int(os.environ.get("BATCH", 64))
LR = float(os.environ.get("LR", 5e-5))
REP_EPOCHS = int(os.environ.get("REP_EPOCHS", 45))
REP_BATCH = int(os.environ.get("REP_BATCH", 32))
K_OOF = int(os.environ.get("K_OOF", 8))
K_TEST = int(os.environ.get("K_TEST", 4))
K_REFIT = int(os.environ.get("K_REFIT", 8))
SCORE_BS = int(os.environ.get("SCORE_BS", 256))
SEED = int(os.environ.get("SEED", 777))
torch.backends.cuda.matmul.allow_tf32 = True

MODEL_SPECS = {
    "gpt2": dict(nl=12, nh=12, d=768),
    "gpt2-medium": dict(nl=24, nh=16, d=1024),
    "gpt2-large": dict(nl=36, nh=20, d=1280),
}


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def build_cache():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    d = ROOT / "data"
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    tr_p = d / "playground-series-s6e11/train.csv"
    if not tr_p.exists():
        tr_p, te_p = d / "train.csv", d / "test.csv"
    else:
        te_p = d / "playground-series-s6e11/test.csv"
    og_p = d / "original/credit_card_fraud_2026.csv"
    if not og_p.exists():
        og_p = ROOT / "original/credit_card_fraud_2026.csv"
    tr, te, og = rd(tr_p), rd(te_p), rd(og_p)
    cols = [c for c in tr.columns if c not in ("id", "is_fraud")]
    allv = pd.concat([tr[cols], te[cols], og[cols]], ignore_index=True)
    used, per = set(), []
    pre = tok.encode("is_fraud is")
    lab = [tok.encode(" False")[0], tok.encode(" True")[0]]
    if NATURAL:
        used.update(pre + lab)
    for c in cols:
        codes, uniq = pd.factorize(allv[c], sort=False)
        head = tok.encode(",") + tok.encode(" " + c) + tok.encode(" is") if NATURAL else []
        ids = [head + tok.encode(" " + (str(v).strip() or "None")) for v in uniq]
        assert codes.min() >= 0 and max(map(len, ids)) <= (LMAX if NATURAL else LMAX - 1), c
        used.update(i for x in ids for i in x)
        per.append((codes, ids))
    vocab = np.array(sorted(used))
    remap = {g: i for i, g in enumerate(vocab)}
    V = len(vocab)
    COL0 = V + 4
    toks = np.full((len(allv), len(cols), LMAX), -1, np.int16)
    for j, (codes, ids) in enumerate(per):
        t = np.full((len(ids), LMAX), -1, np.int16)
        o = 0 if NATURAL else 1
        if not NATURAL:
            t[:, 0] = COL0 + j
        for i, x in enumerate(ids):
            t[i, o:o + len(x)] = [remap[g] for g in x]
        toks[:, j, :] = t[codes]
    y = np.r_[tr.is_fraud.astype(float).values, np.full(len(te), -1.0), og.is_fraud.astype(float).values].astype(np.int8)
    colids = np.full((len(cols), 8), -1, np.int64)
    for j, c in enumerate(cols):
        x = tok.encode(" " + c)[:8]
        colids[j, :len(x)] = x
    np.savez(CACHE, toks=toks, y=y, vocab=vocab, colids=colids, ntr=len(tr), nte=len(te), lab=np.array(lab),
             comma=np.array(tok.encode(",")), pre=np.array([remap[i] for i in pre] if NATURAL else [], np.int64),
             labn=np.array([remap[i] for i in lab] if NATURAL else [0, 0], np.int64))


def load_data():
    if not CACHE.exists():
        log("building cache.npz …")
        build_cache()
    z = np.load(CACHE)
    g = dict(toks=z["toks"], y=z["y"].astype(np.int64), vocab=z["vocab"], colids=z["colids"], ntr=int(z["ntr"]),
             nte=int(z["nte"]), lab=z["lab"], comma=z["comma"])
    g["V"] = len(g["vocab"])
    g["END"], g["PAD"], g["LAB"], g["COL0"] = g["V"], g["V"] + 1, g["V"] + 2, g["V"] + 4
    g["NC"] = g["toks"].shape[1]
    g["BOS"] = g["COL0"] + g["NC"]
    g["NTOK"] = g["BOS"] + 1
    pre = [int(i) for i in z["pre"]] if "pre" in z.files else []
    g["PRE"] = pre if NATURAL else ([g["BOS"]] if BOS else [])
    g["LABID"] = [int(i) for i in z["labn"]] if NATURAL else [g["LAB"], g["LAB"] + 1]
    g["T"] = int((g["toks"] >= 0).sum((1, 2)).max()) + len(g["PRE"]) + 1
    return g


class Rows:
    def __init__(self, g):
        self.t = torch.from_numpy(g["toks"]).to(dev)
        self.g, self.T = g, g["T"]
        self.P = len(g["PRE"])
        self.lens = (g["toks"] >= 0).sum((1, 2)) + self.P + 1
        self.pre = torch.tensor(g["PRE"], dtype=torch.long, device=dev)
        self.labid = torch.tensor(g["LABID"], dtype=torch.long, device=dev)

    def perm(self, n, gen):
        return torch.rand(n, self.g["NC"], device=dev, generator=gen).argsort(1)

    def batches(self, rows, bs, rs, group=64):
        group = 1 if PARITY else group
        o = rs.permutation(rows)[:len(rows) // bs * bs]
        for c in range(0, len(o), bs * group):
            s = o[c:c + bs * group]
            o[c:c + bs * group] = s[np.argsort(self.lens[s], kind="stable")]
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
        buf = buf[:, :T + 1]
        buf[:, :self.P] = self.pre
        buf[:, self.P] = self.labid[lab]
        cur, nxt = buf[:, :-1], buf[:, 1:]
        tgt = torch.where(nxt < g["V"], nxt, torch.full_like(nxt, g["END"]))
        tgt = torch.where(cur == g["PAD"], torch.full_like(tgt, -100), tgt)
        if not full:
            tgt[:, :self.P + (0 if NATURAL else 1)] = -100
        return cur.contiguous(), tgt


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
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), dropout_p=s.p if s.training else 0.0, is_causal=True)
        x = x + s.drop(s.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + s.drop(s.out(F.gelu(s.fc(s.ln_2(x)), approximate="tanh")))


class LM(nn.Module):
    def __init__(s, ntok, nout, T, nl=12, nh=12, d=768, p=0.1):
        super().__init__()
        s.wte, s.wpe = nn.Embedding(ntok, d), nn.Embedding(T, d)
        s.h = nn.ModuleList(Block(d, nh, p) for _ in range(nl))
        s.ln_f, s.drop, s.nout = nn.LayerNorm(d, eps=1e-5), nn.Dropout(p), nout

    def hidden(s, x, lab=None):
        e = s.wte(x) + s.wpe.weight[:x.shape[1]]
        h = s.drop(e if lab is None else e + s.wte(lab)[:, None, :])
        for b in s.h:
            h = b(h)
        return s.ln_f(h)

    def forward(s, x, lab=None):
        h = s.hidden(x, lab).float()
        with torch.autocast("cuda", enabled=False):
            return h @ s.wte.weight[:s.nout].T


def copy_blocks(m, sd):
    P = "transformer."
    with torch.no_grad():
        for i, b in enumerate(m.h):
            q = f"{P}h.{i}."
            for name, key in (("ln_1", "ln_1"), ("ln_2", "ln_2")):
                getattr(b, name).weight.copy_(sd[q + key + ".weight"])
                getattr(b, name).bias.copy_(sd[q + key + ".bias"])
            for name, key in (("qkv", "attn.c_attn"), ("proj", "attn.c_proj"), ("fc", "mlp.c_fc"), ("out", "mlp.c_proj")):
                getattr(b, name).weight.copy_(sd[q + key + ".weight"].T)
                getattr(b, name).bias.copy_(sd[q + key + ".bias"])
        m.ln_f.weight.copy_(sd[P + "ln_f.weight"])
        m.ln_f.bias.copy_(sd[P + "ln_f.bias"])


def from_gpt2(g):
    from transformers import GPT2LMHeadModel
    spec = MODEL_SPECS[MODEL_NAME]
    sd = GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    m = LM(g["NTOK"], g["V"] + 1, g["T"], **spec)
    copy_blocks(m, sd)
    w = sd["transformer.wte.weight"]
    with torch.no_grad():
        m.wpe.weight.copy_(sd["transformer.wpe.weight"][:g["T"]])
        m.wte.weight[:g["V"]] = w[torch.from_numpy(g["vocab"])]
        m.wte.weight[g["END"]] = w[50256 if NATURAL else int(g["comma"][0])]
        m.wte.weight[g["BOS"]] = w[50256]
        m.wte.weight[g["PAD"]] = w.mean(0)
        m.wte.weight[g["LAB"]:g["LAB"] + 2] = w[torch.from_numpy(g["lab"])]
        for j, r in enumerate(g["colids"]):
            m.wte.weight[g["COL0"] + j] = w[torch.from_numpy(r[r >= 0])].mean(0)
    return m.to(dev)


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(m, R, ytr, rows, epochs, bs, lr, name, seed, max_steps=None):
    per = len(rows) // bs
    total = per * epochs
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0, fused=True)
    ck, start = CK / f"{name}.pt", 0
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=True)
        m.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        start = int(s["epoch"]) + 1
        log(f"{name}: resumed after epoch {start}")
    gen = torch.Generator(device=dev)
    m.train()
    t0, seen, run = time.time(), 0, torch.zeros((), device=dev)
    for ep in range(start, epochs):
        run.zero_()
        gen.manual_seed(seed * 1000 + ep)
        bt, tb = R.batches(rows, bs, np.random.RandomState((seed * 1000 + ep) % 2**31))
        for s in range(per):
            step = ep * per + s
            for grp in opt.param_groups:
                grp["lr"] = lr * min(1.0, (step + 1) / 100) * max(0.0, 1 - step / total)
            idx = bt[s]
            inp, tgt = R.assemble(idx, ytr[idx], R.perm(bs, gen), tb[s], full=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(inp, R.labid[ytr[idx]] if LABEMB else None)
            loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.detach()
            seen += bs
            if (s + 1) % 2000 == 0:
                log(f"{name} ep {ep + 1}/{epochs} step {s + 1}/{per} loss {run.item() / 2000:.4f} "
                    f"{seen / (time.time() - t0):.0f} rows/s")
                run.zero_()
            if max_steps and s + 1 >= max_steps:
                torch.cuda.synchronize()
                return seen / (time.time() - t0)
        save_atomic({"model": m.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ck)
        log(f"{name}: epoch {ep + 1}/{epochs} done, {(time.time() - t0) / 60:.1f} min")
    return None


@torch.no_grad()
def score(m, R, rows, K, seed, bs=256, dtype=torch.bfloat16):
    m.eval()
    torch.cuda.empty_cache()
    srt = np.argsort(R.lens[rows], kind="stable")
    ls = R.lens[rows][srt]
    rows_t = torch.as_tensor(rows[srt], device=dev)
    out = torch.zeros((len(rows), K, 2), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for k in range(K):
        for s in range(0, len(rows), bs):
            idx = rows_t[s:s + bs]
            n = idx.numel()
            perm = R.perm(n, gen)
            inp, tgt = R.assemble(idx, torch.zeros(n, dtype=torch.long, device=dev), perm, int(ls[s + n - 1]))
            both = torch.cat([inp, inp])
            both[n:, R.P] = R.labid[1]
            lab2 = R.labid.repeat_interleave(n) if LABEMB else None
            with torch.autocast("cuda", dtype=dtype):
                lg = m(both, lab2)
            nll = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.repeat(2, 1).reshape(-1), ignore_index=-100,
                                  reduction="none").view(2 * n, -1).sum(1)
            out[s:s + n, k, 0] = -nll[:n]
            out[s:s + n, k, 1] = -nll[n:]
    res = np.empty((len(rows), K, 2), np.float32)
    res[srt] = out.cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return res


def bayes(ll, prior):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1 - prior))


def replica(g, R, ydev):
    p = CK.parent / f"replica_{FMT}_s{SEED}.pt"
    spec = MODEL_SPECS[MODEL_NAME]
    if p.exists():
        m = LM(g["NTOK"], g["V"] + 1, g["T"], **spec).to(dev)
        m.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
        return m
    m = from_gpt2(g)
    rows = np.arange(g["ntr"] + g["nte"], len(g["y"]))
    train(m, R, ydev, rows, REP_EPOCHS, REP_BATCH, LR, "replica", SEED)
    save_atomic(m.state_dict(), p)
    (CK / "replica.pt").unlink(missing_ok=True)
    return m


def evaluate_against_refs(y_b, z_b, fold_idx, b_idx):
    """Compare fold OOF against Part 12 lm_oof, wide_lm_stack, and v8 on this fold."""
    refs = {}
    for name, rel in [("lm12", "external/nov08_part12/lm_oof.npy"),
                      ("g3", "preds/gpt2_3ep/gpt2_3ep_oof.npy"),
                      ("w12", "external/nov08_part12/wide_lm_stack_nested.npy"),
                      ("v8", "preds/P_super_v8_part12_rect_oof.npy")]:
        fp = ROOT / rel
        if fp.exists():
            arr = np.load(fp).ravel()[b_idx]
            refs[name] = dict(auc=float(roc_auc_score(y_b, arr)), corr=float(np.corrcoef(zrk(z_b), zrk(arr))[0, 1]))
            # best orthogonal blend on this fold
            zb_r, zr = zrk(z_b), zrk(arr)
            beta = float(np.dot(zb_r, zr) / np.dot(zr, zr))
            r = (zb_r - beta * zr)
            r = r / (r.std() + 1e-12)
            best_w, best_auc = 0.0, refs[name]["auc"]
            for w in np.linspace(-0.08, 0.08, 33):
                sc = float(roc_auc_score(y_b, zr + w * r))
                if sc > best_auc:
                    best_auc, best_w = sc, float(w)
            refs[name]["best_orth_w"] = best_w
            refs[name]["best_orth_delta"] = best_auc - refs[name]["auc"]
    return refs


def run(g, which):
    R = Rows(g)
    y = g["y"]
    ydev = torch.as_tensor(np.maximum(y, 0), device=dev)
    ntr, nte = g["ntr"], g["nte"]
    test = np.arange(ntr, ntr + nte)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    t0 = time.time()
    replica(g, R, ydev)
    log(f"replica ready ({MODEL_NAME}), {(time.time() - t0) / 60:.1f} min")
    status = {}
    st_path = OUT / "status.json"
    if st_path.exists():
        status = json.loads(st_path.read_text())

    for f in which:
        done = OUT / f"test_{f}.npy"
        oof_p = OUT / f"oof_{f}.npy"
        fi = int(f)
        a, b = folds[fi]
        prior = float(y[a].mean())
        if done.exists() and oof_p.exists():
            ll = np.load(oof_p)
            z = bayes(ll, prior)
            log(f"fold {f}: already scored, OOF AUC {roc_auc_score(y[b], z):.6f}")
            continue
        t1 = time.time()
        m = replica(g, R, ydev)
        train(m, R, ydev, a, EPOCHS, BATCH, LR, f"fold_{f}", SEED + fi)
        t2 = time.time()
        ll = score(m, R, b, K_OOF, SEED + 100 + fi, bs=SCORE_BS)
        np.save(oof_p, ll)
        z = bayes(ll, prior)
        f_auc = float(roc_auc_score(y[b], z))
        one_auc = float(roc_auc_score(y[b], ll[:, 0, 1] - ll[:, 0, 0]))
        refs = evaluate_against_refs(y[b], z, fi, b)
        log(f"fold {f}: OOF AUC {f_auc:.6f} (1-order {one_auc:.6f}) | refs={json.dumps(refs)}")
        lt = score(m, R, test, K_TEST, SEED + 200 + fi, bs=SCORE_BS)
        np.save(done, lt)
        log(f"fold {f}: train {(t2 - t1) / 60:.1f} min, score {(time.time() - t2) / 60:.1f} min")
        status[f"fold_{f}"] = dict(auc=f_auc, one_order_auc=one_auc, refs=refs,
                                   train_min=round((t2 - t1) / 60, 2), score_min=round((time.time() - t2) / 60, 2))
        st_path.write_text(json.dumps(status, indent=2))
        del m
        torch.cuda.empty_cache()

    # Assemble full 5-fold OOF & test if all 5 exist
    if all((OUT / f"oof_{i}.npy").exists() and (OUT / f"test_{i}.npy").exists() for i in range(5)):
        full_oof = np.empty(ntr, dtype=np.float64)
        te_list = []
        for i, (a, b) in enumerate(folds):
            pr = float(y[a].mean())
            full_oof[b] = bayes(np.load(OUT / f"oof_{i}.npy"), pr)
            te_list.append(bayes(np.load(OUT / f"test_{i}.npy"), pr))
        full_te = np.mean(te_list, axis=0)
        np.save(OUT / f"{TAG}_oof.npy", full_oof)
        np.save(OUT / f"{TAG}_test.npy", full_te)
        log(f"ALL 5 FOLDS DONE: pooled OOF AUC = {roc_auc_score(y[:ntr], full_oof):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "run"])
    ap.add_argument("--folds", default="2,0,1,3,4")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    g = load_data()
    if a.mode == "bench":
        R = Rows(g)
        ydev = torch.as_tensor(np.maximum(g["y"], 0), device=dev)
        m = from_gpt2(g)
        r = train(m, R, ydev, np.arange(g["ntr"]), 1, BATCH, LR, "bench", 1, max_steps=100)
        log(f"BENCH {MODEL_NAME} bs={BATCH}: {r:.0f} rows/s, peak={torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    else:
        log(f"START model={MODEL_NAME} tag={TAG} ep={EPOCHS} bs={BATCH} lr={LR} K={K_OOF}/{K_TEST} seed={SEED}")
        run(g, [x.strip() for x in a.folds.split(",") if x.strip()])
