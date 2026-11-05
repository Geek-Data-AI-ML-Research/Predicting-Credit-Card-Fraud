"""LAB 4: Fast Tabular NADE / Masked Pseudolikelihood Ratio scorer (CPU or GPU).
Policy: NO_SUBMIT. ~48 discrete tokens/row; Leave-Block-Out inference ×16 masks.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from part11_common import (
    CATS,
    FEATS,
    FLAGS,
    FLOATS,
    NUMS,
    PRED11,
    folds_y,
    load_champ_stack,
    load_report,
    load_y_ids,
    log,
    rk,
    save_report,
    stack_gate,
)

ROOT = Path(__file__).resolve().parents[1]


VOCAB_CAP = 256  # keep GPU softmax tractable (A5000 24GB)


def build_token_table(tr: pd.DataFrame, te: pd.DataFrame):
    """48 discrete categorical fields; each hashed/capped to VOCAB_CAP ids."""
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    cols = []
    names = []

    def add(name, series):
        codes, _ = pd.factorize(series.astype(str), sort=False)
        vc = pd.Series(codes).value_counts()
        remap = {int(c): i + 1 for i, c in enumerate(vc.index[: VOCAB_CAP - 1])}
        out = pd.Series(codes).map(remap).fillna(0).astype(np.int32).to_numpy()
        cols.append(out)
        names.append(name)

    for c in CATS + FLAGS:
        add(c, full[c])
    for c in NUMS:
        if c in FLOATS:
            v = full[c].to_numpy(float)
            add(f"{c}__bin50", pd.qcut(v, q=50, duplicates="drop").astype(str))
            add(f"{c}__int", np.floor(v).astype(np.int64).astype(str))
            cents = np.round(v * 100).astype(np.int64)
            add(f"{c}__cents", cents.astype(str))
            add(f"{c}__ten", (cents // 10).astype(str))
            add(f"{c}__l2", (cents % 100).astype(str))
        else:
            add(c, full[c].astype(str))
    Tok = np.column_stack(cols)
    if Tok.shape[1] > 48:
        Tok = Tok[:, :48]
        names = names[:48]
    elif Tok.shape[1] < 48:
        pad = np.zeros((len(Tok), 48 - Tok.shape[1]), np.int32)
        Tok = np.hstack([Tok, pad])
        names += [f"pad{i}" for i in range(48 - len(names))]
    ncat = np.full(Tok.shape[1], VOCAB_CAP, dtype=np.int32)
    log(f"  token table shape={Tok.shape} vocab_cap={VOCAB_CAP}")
    return Tok, names, ncat


def run_nade_torch(Tok, y, folds, ncat, n_masks=16, seeds=(0, 1, 2), epochs=8):
    """Masked tabular classifier + optional field reconstruction via tied emb dots."""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  NADE device={device}")
    NTR = len(y)
    NTE = Tok.shape[0] - NTR
    F = Tok.shape[1]
    emb_dim = 32
    max_cat = int(VOCAB_CAP)
    bs = 1024 if device.type == "cuda" else 4096
    Tok64 = Tok.astype(np.int64)

    class MaskedNADE(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(max_cat, emb_dim)
            self.field = nn.Embedding(F, emb_dim)
            hid = 256
            self.net = nn.Sequential(
                nn.Linear(emb_dim * 2, hid),
                nn.SiLU(),
                nn.Dropout(0.25),
                nn.Linear(hid, hid),
                nn.SiLU(),
            )
            self.y_head = nn.Linear(hid, 1)
            # project h → emb space for tied reconstruction (no huge softmax)
            self.to_emb = nn.Linear(hid + emb_dim, emb_dim)

        def forward(self, tok, mask):
            e = self.emb(tok) + self.field.weight.unsqueeze(0)
            e = e * mask.unsqueeze(-1)
            denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
            pooled = e.sum(1) / denom
            x = torch.cat([pooled, (1.0 - mask).mean(-1, keepdim=True).expand(-1, emb_dim)], -1)
            h = self.net(x)
            return self.y_head(h).squeeze(-1), h

        def recon_nll(self, h, tok, mask):
            """Mean squared error of predicted emb vs true token emb on masked fields (pseudolikelihood proxy)."""
            B = tok.shape[0]
            h_exp = h.unsqueeze(1).expand(-1, F, -1)
            f_emb = self.field.weight.unsqueeze(0).expand(B, -1, -1)
            pred = self.to_emb(torch.cat([h_exp, f_emb], -1))
            true = self.emb(tok)
            mse = ((pred - true) ** 2).mean(-1)  # B x F
            inv = 1.0 - mask
            return (mse * inv).sum() / inv.sum().clamp_min(1.0)

    det_masks = []
    block = max(1, F // n_masks)
    for m in range(n_masks):
        mask = np.ones(F, np.float32)
        lo = m * block
        hi = F if m == n_masks - 1 else min(F, (m + 1) * block)
        mask[lo:hi] = 0.0
        det_masks.append(mask)

    oof = np.zeros(NTR)
    te_acc = []

    for seed in seeds:
        torch.manual_seed(seed)
        fold_te = []
        for fi, (a, b) in enumerate(folds):
            model = MaskedNADE().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            Ta = torch.as_tensor(Tok64[a], device=device)
            ya = torch.as_tensor(y[a], dtype=torch.float32, device=device)
            model.train()
            for ep in range(epochs):
                perm = torch.randperm(len(a), device=device)
                for i0 in range(0, len(a), bs):
                    idx = perm[i0 : i0 + bs]
                    vis = torch.empty(len(idx), F, device=device).uniform_(0.15, 0.85)
                    mask = (torch.rand(len(idx), F, device=device) < vis).float()
                    mask[:, 0] = 1.0
                    opt.zero_grad()
                    logit_y, h = model(Ta[idx], mask)
                    loss = nn.functional.binary_cross_entropy_with_logits(logit_y, ya[idx])
                    loss = loss + 0.2 * model.recon_nll(h, Ta[idx], mask)
                    loss.backward()
                    opt.step()
            model.eval()

            def score_rows(rows):
                rows = np.asarray(rows)
                out = np.zeros(len(rows), np.float64)
                chunk = 8192
                with torch.no_grad():
                    for s in range(0, len(rows), chunk):
                        sl = rows[s : s + chunk]
                        T = torch.as_tensor(Tok64[sl], device=device)
                        acc = torch.zeros(len(sl), device=device)
                        for m in det_masks:
                            mask = torch.as_tensor(m, device=device).unsqueeze(0).expand(len(sl), -1)
                            logit_y, h = model(T, mask)
                            nll = model.recon_nll(h, T, mask)  # scalar — use y logit only for score
                            acc += logit_y
                        out[s : s + len(sl)] = (acc / len(det_masks)).cpu().numpy()
                return out

            oof_fold = score_rows(b)
            oof[b] += oof_fold / len(seeds)
            fold_te.append(score_rows(np.arange(NTR, NTR + NTE)))
            log(f"  NADE seed={seed} fold={fi} valAUC={roc_auc_score(y[b], oof_fold):.5f}")
        te_acc.append(np.mean(fold_te, 0))

        # full-data refit
        model = MaskedNADE().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Tall = torch.as_tensor(Tok64[:NTR], device=device)
        yall = torch.as_tensor(y, dtype=torch.float32, device=device)
        model.train()
        for ep in range(epochs + 2):
            perm = torch.randperm(NTR, device=device)
            for i0 in range(0, NTR, bs):
                idx = perm[i0 : i0 + bs]
                vis = torch.empty(len(idx), F, device=device).uniform_(0.15, 0.85)
                mask = (torch.rand(len(idx), F, device=device) < vis).float()
                mask[:, 0] = 1.0
                opt.zero_grad()
                logit_y, h = model(Tall[idx], mask)
                loss = nn.functional.binary_cross_entropy_with_logits(logit_y, yall[idx])
                loss = loss + 0.2 * model.recon_nll(h, Tall[idx], mask)
                loss.backward()
                opt.step()
        model.eval()
        full_te = score_rows(np.arange(NTR, NTR + NTE))
        te_acc[-1] = 0.5 * te_acc[-1] + 0.5 * full_te

    return oof, np.mean(te_acc, 0)


def run_lab4(rep: dict) -> dict:
    t0 = time.time()
    log("=" * 72)
    log("LAB 4: Masked Tabular NADE / Pseudolikelihood scorer")
    log("=" * 72)

    y, _ = load_y_ids()
    folds = folds_y(y)
    NTR = len(y)
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")
    Tok, names, ncat = build_token_table(tr, te)

    oof, te_s = run_nade_torch(Tok, y, folds, ncat)
    # map to [0,1] ranks for stacking
    oof_p = rk(oof)
    te_p = rk(te_s)
    auc = float(roc_auc_score(y, oof))
    log(f"  tabular_nade OOF AUC={auc:.6f} (target >0.9503)")
    np.save(PRED11 / "lab4_tabular_nade_oof.npy", oof.astype(np.float32))
    np.save(PRED11 / "lab4_tabular_nade_test.npy", te_s.astype(np.float32))
    np.save(PRED11 / "lab4_tabular_nade_oof_rank.npy", oof_p.astype(np.float32))
    np.save(PRED11 / "lab4_tabular_nade_test_rank.npy", te_p.astype(np.float32))

    corr = {}
    for label, path in [
        ("masaya_gpt2", None),
        ("gb_bag5", None),
        ("gb_gpt2_bag5", "preds/gb_gpt2_bag5_oof.npy"),
    ]:
        if path and Path(path).exists():
            corr[label] = float(np.corrcoef(oof_p, rk(np.load(path).ravel()))[0, 1])
    try:
        from part9_joint_boost_and_basis import build_p7_65

        names7, PO, PT, _ = build_p7_65(y)
        ni = {n: i for i, n in enumerate(names7)}
        if "gpt2" in ni:
            corr["masaya_gpt2"] = float(np.corrcoef(oof_p, rk(PO[:, ni["gpt2"]]))[0, 1])
        if "gb_bag5" in ni:
            corr["gb_bag5"] = float(np.corrcoef(oof_p, rk(PO[:, ni["gb_bag5"]]))[0, 1])
    except Exception as e:
        log(f"  corr skip: {e}")

    names_s, O, T, champ_meta = load_champ_stack(y, folds)
    gate = stack_gate(O, oof_p, y, folds, label="lab4_tabular_nade")

    lab = {
        "name": "LAB4_tabular_nade",
        "runtime_s": time.time() - t0,
        "n_tokens": int(Tok.shape[1]),
        "standalone_auc": auc,
        "target_auc": 0.9503,
        "beats_target": bool(auc > 0.9503),
        "correlations": corr,
        "stack_gate": gate,
        "champ_meta": champ_meta,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "keep_any": bool(gate["keep"]),
    }
    rep["lab4"] = lab
    save_report(rep)
    log(f"LAB4 done in {lab['runtime_s']:.1f}s keep_any={lab['keep_any']}")
    return lab


if __name__ == "__main__":
    run_lab4(load_report())
