"""LAB 2: Exhaustive pairwise logit interactions + DCN-v2 residual bag.
Policy: NO_SUBMIT. Uses gboost TE cache or builds on the fly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from part11_common import (
    PRED11,
    best_nested,
    folds_y,
    load_champ_stack,
    load_report,
    load_y_ids,
    log,
    rk,
    rlogit,
    save_report,
    stack_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def _find_te_cache():
    cands = list(ROOT.glob("**/cache_te*.npz")) + list(Path("/tmp").glob("**/cache_te*.npz"))
    # prefer gen glm caches with za
    for p in sorted(cands, key=lambda x: -x.stat().st_mtime):
        return p
    return None


def _winsor_z(Z, lo=0.001, hi=0.999):
    Z = Z.astype(np.float64)
    out = np.empty_like(Z)
    for j in range(Z.shape[1]):
        col = Z[:, j]
        a, b = np.quantile(col, [lo, hi])
        out[:, j] = np.clip(col, a, b)
    sc = StandardScaler()
    return sc.fit_transform(out), sc


def screen_interactions(Za, resid, K_keep, include_absdiff=True):
    """Vectorized residual correlation of Z_j*Z_k and |Z_j-Z_k| on fold-train."""
    n, d = Za.shape
    # center
    Zc = Za - Za.mean(0, keepdims=True)
    r = resid - resid.mean()
    # products: for each pair (j,k), corr(Zj*Zk, r)
    # Use: cov(Zj*Zk, r) = mean(Zj*Zk*r) since centered approx
    # Efficient: for fixed j, score_k = mean(Zj * Zk * r) = (Zj*r) · Zk / n
    scores = []
    pairs = []
    wr = Zc * r[:, None]  # n x d
    # products
    for j in range(d):
        # score vs k>j
        s = (wr[:, j : j + 1] * Zc[:, j + 1 :]).mean(0)  # d-j-1
        for t, k in enumerate(range(j + 1, d)):
            scores.append(abs(float(s[t])))
            pairs.append(("prod", j, k))
    if include_absdiff:
        for j in range(d):
            diff = np.abs(Zc[:, j : j + 1] - Zc[:, j + 1 :])
            s = (diff * r[:, None]).mean(0)
            for t, k in enumerate(range(j + 1, d)):
                scores.append(abs(float(s[t])))
                pairs.append(("abs", j, k))
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    keep = order[:K_keep]
    return [pairs[i] for i in keep], scores[keep]


def build_inter_matrix(Z, pair_list):
    cols = []
    for kind, j, k in pair_list:
        if kind == "prod":
            cols.append(Z[:, j] * Z[:, k])
        else:
            cols.append(np.abs(Z[:, j] - Z[:, k]))
    return np.column_stack(cols).astype(np.float64)


def run_dcn_bag(X_oof_list, y, folds, z_init_oof, X_te_list, z_init_te, seeds=(0, 1, 2, 3, 4)):
    """Lightweight DCN-v2 + residual MLP in torch (CPU/GPU)."""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  DCN device={device}")
    # CPU: fewer seeds/epochs; GPU: full bag
    if device.type == "cpu":
        seeds = tuple(seeds)[:3]
        n_epochs_cv, n_epochs_full, bs = 12, 15, 16384
    else:
        n_epochs_cv, n_epochs_full, bs = 25, 30, 8192

    class DCNv2(nn.Module):
        def __init__(self, d, rank=16, n_cross=3, hidden=256, drop=0.35):
            super().__init__()
            self.cross_U = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(n_cross)])
            self.cross_V = nn.ModuleList([nn.Linear(rank, d, bias=True) for _ in range(n_cross)])
            self.mlp = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Dropout(drop),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(drop),
                nn.Linear(hidden, 1),
            )
            self.head = nn.Linear(d + 1, 1)

        def forward(self, x0):
            x = x0
            for U, V in zip(self.cross_U, self.cross_V):
                x = x0 * V(U(x)) + x
            m = self.mlp(x0)
            return (self.head(torch.cat([x, m], -1))).squeeze(-1)

    X_oof = X_oof_list
    X_te = X_te_list
    n, d = X_oof.shape
    oof = np.zeros(n)
    te_bags = []
    Xt = torch.tensor(X_te, dtype=torch.float32, device=device)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        fold_te = []
        for fi, (a, b) in enumerate(folds):
            model = DCNv2(d).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs_cv)
            Xa = torch.tensor(X_oof[a], dtype=torch.float32, device=device)
            ya = torch.tensor(y[a], dtype=torch.float32, device=device)
            za = torch.tensor(z_init_oof[a], dtype=torch.float32, device=device)
            Xb = torch.tensor(X_oof[b], dtype=torch.float32, device=device)
            model.train()
            for ep in range(n_epochs_cv):
                perm = torch.randperm(len(a), device=device)
                for i0 in range(0, len(a), bs):
                    idx = perm[i0 : i0 + bs]
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(za[idx] + model(Xa[idx]), ya[idx])
                    loss.backward()
                    opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                oof[b] += expit(z_init_oof[b] + model(Xb).cpu().numpy()) / len(seeds)
                fold_te.append(expit(z_init_te + model(Xt).cpu().numpy()))
        te_bags.append(np.mean(fold_te, 0))
        # full refit
        model = DCNv2(d).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs_full)
        Xall = torch.tensor(X_oof, dtype=torch.float32, device=device)
        yall = torch.tensor(y, dtype=torch.float32, device=device)
        zall = torch.tensor(z_init_oof, dtype=torch.float32, device=device)
        model.train()
        for ep in range(n_epochs_full):
            perm = torch.randperm(n, device=device)
            for i0 in range(0, n, bs):
                idx = perm[i0 : i0 + bs]
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(zall[idx] + model(Xall[idx]), yall[idx])
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            full_te = expit(z_init_te + model(Xt).cpu().numpy())
        te_bags[-1] = 0.5 * te_bags[-1] + 0.5 * full_te
        log(f"  DCN seed={seed} done")

    te = np.mean(te_bags, 0)
    return oof, te


def run_lab2(rep: dict) -> dict:
    t0 = time.time()
    log("=" * 72)
    log("LAB 2: Pairwise logit interactions + DCN-v2")
    log("=" * 72)

    y, _ = load_y_ids()
    folds = folds_y(y)
    NTR = len(y)

    # Prefer cached TE logits from gboost cache; else rebuild via import gboost
    cache = _find_te_cache()
    Z_oof = None
    z_glm_oof = np.zeros(NTR)
    z_glm_te = None
    Z_te = None

    if cache is not None:
        log(f"  loading TE cache {cache}")
        z = np.load(cache, allow_pickle=True)
        # reconstruct OOF TE logits from fold Da/Db — actually want Ea logits
        # Use zb (GLM) and try to recover TE block from Da if present
        # Fallback path: use glm_joint + build TE via gboost import
        keys = list(z.keys())
        log(f"  cache keys sample: {keys[:12]}")
        if "f0_za" in z:
            for fi, (a, b) in enumerate(folds):
                z_glm_oof[b] = z[f"f{fi}_zb"]
            z_glm_te = np.mean([z[f"f{fi}_zt"] for fi in range(5)], 0)
            if "full_zt" in z:
                z_glm_te = 0.5 * z_glm_te + 0.5 * z["full_zt"]
            # TE features: take first 92 cols of Da after scaler — not stored as Ea alone
            # Use Xa which is view(a,Ea) = [RAWV, E, KCNT] — extract E block if we know dims
            # Safer: recompute TE via gboost below
    # Always build TE matrices via gboost for exact 92-key logits
    log("  computing 5-fold TE logits via gboost.te_fold …")
    import os as _os

    _os.environ["INPUT_DIR"] = str(ROOT)
    _os.environ.setdefault("OUT_DIR", str(ROOT / "preds" / "part11"))
    import gboost as gb

    D = None
    Z_parts = []
    Zt_parts = []
    z_glm_oof = np.zeros(NTR)
    zt_folds = []
    for fi, (a, b) in enumerate(folds):
        Ea, Eb, Et = gb.te_fold(a, b, smooth=20.0, seed=0)
        if gb.GEN:
            Ga, Gb, Gt = gb.gen_fold(a, b, seed=0)
            Ea = np.hstack([Ea, Ga])
            Eb = np.hstack([Eb, Gb])
            Et = np.hstack([Et, Gt])
        if D is None:
            D = Ea.shape[1]
            log(f"  TE dim D={D}")
        ybar = float(y[a].mean())
        # Z = logit(E) - logit(ybar)
        def toZ(E):
            return rlogit(E) - rlogit(np.full(E.shape[1], ybar) if False else ybar)

        # broadcast subtract
        Za = rlogit(Ea) - rlogit(ybar)
        Zb = rlogit(Eb) - rlogit(ybar)
        Zt = rlogit(Et) - rlogit(ybar)
        # standardize using fold-train
        sc = StandardScaler().fit(np.clip(Za, np.quantile(Za, 0.001, 0), np.quantile(Za, 0.999, 0)))
        Za_s, Zb_s, Zt_s = sc.transform(Za), sc.transform(Zb), sc.transform(Zt)
        Z_parts.append((a, b, Za_s, Zb_s, Zt_s, Ea, Eb, Et))
        # GLM joint init: use cached glm_joint if available
        if Path("preds/glm_joint_oof.npy").exists():
            pass
        # Stage-1 style GLM on TE+BASE for init_score
        from sklearn.preprocessing import StandardScaler as SS

        scE = SS().fit(rlogit(Ea))
        Da = np.hstack([scE.transform(rlogit(Ea)), gb.BASE[a]])
        Db = np.hstack([scE.transform(rlogit(Eb)), gb.BASE[b]])
        Dt = np.hstack([scE.transform(rlogit(Et)), gb.BASE[gb.TEST]])
        m = LogisticRegression(C=0.3, max_iter=200, tol=1e-7, solver="newton-cholesky").fit(Da, y[a])
        z_glm_oof[b] = m.decision_function(Db)
        zt_folds.append(m.decision_function(Dt))
        log(f"  fold{fi} GLM AUC={roc_auc_score(y[b], z_glm_oof[b]):.5f} D={D}")

    z_glm_te = np.mean(zt_folds, 0)
    # GPT2 joint
    gpt_o = np.load("preds/gb_gpt2_bag5_oof.npy").ravel() if Path("preds/gb_gpt2_bag5_oof.npy").exists() else None
    # Actually masaya gpt2 from bank
    try:
        from part9_joint_boost_and_basis import build_p7_65

        names, PO, PT, _ = build_p7_65(y)
        ni = {n: i for i, n in enumerate(names)}
        gpt_o, gpt_t = PO[:, ni["gpt2"]], PT[:, ni["gpt2"]]
    except Exception:
        gpt_o = np.load("preds/gb_gpt2_joint_oof.npy").ravel()
        gpt_t = np.load("preds/gb_gpt2_joint_test.npy").ravel()

    z_joint_o = z_glm_oof + rlogit(rk(gpt_o))
    z_joint_t = z_glm_te + rlogit(rk(gpt_t))

    # Assemble OOF Z matrix
    Z_oof = np.zeros((NTR, D), np.float64)
    Z_te_acc = []
    for a, b, Za_s, Zb_s, Zt_s, *_ in Z_parts:
        Z_oof[b] = Zb_s
        Z_te_acc.append(Zt_s)
    Z_te = np.mean(Z_te_acc, 0)

    # Screen interactions on each fold-train, union top-K, fit ridge-logit residual
    results_K = {}
    best_quad = None
    # Prior partial run: K=300 max 0.950481; K=800 C=0.001 → 0.950836. Focus remaining grid.
    for K in (800, 1500):
        pair_union = {}
        for a, b, Za_s, Zb_s, Zt_s, *_ in Z_parts:
            resid = y[a] - expit(z_joint_o[a])
            pairs, _ = screen_interactions(Za_s, resid, K_keep=K, include_absdiff=(K <= 800))
            for p in pairs:
                pair_union[p] = pair_union.get(p, 0) + 1
        ranked = sorted(pair_union.items(), key=lambda kv: -kv[1])[:K]
        pairs = [p for p, _ in ranked]
        log(f"  K={K}: using {len(pairs)} interaction terms")

        for C in (0.001, 0.005, 0.02):
            oof = np.zeros(NTR)
            te_fs = []
            for a, b, Za_s, Zb_s, Zt_s, *_ in Z_parts:
                Ia = build_inter_matrix(Za_s, pairs)
                Ib = build_inter_matrix(Zb_s, pairs)
                It = build_inter_matrix(Zt_s, pairs)
                scI = StandardScaler().fit(Ia)
                Ia, Ib, It = scI.transform(Ia), scI.transform(Ib), scI.transform(It)
                m = LogisticRegression(
                    C=C, max_iter=300, tol=1e-5, solver="saga", penalty="l2", n_jobs=-1
                ).fit(Ia, y[a])
                delta_b = m.decision_function(Ib)
                oof[b] = expit(z_joint_o[b] + 0.5 * delta_b)
                te_fs.append(expit(z_joint_t + 0.5 * m.decision_function(It)))
            auc = float(roc_auc_score(y, oof))
            log(f"  glm_quad K={K} C={C} OOF={auc:.6f}")
            results_K[f"K{K}_C{C}"] = auc
            if best_quad is None or auc > best_quad[0]:
                best_quad = (auc, K, C, oof, np.mean(te_fs, 0), pairs)
            # early abandon if clearly worse than known 0.950836@K800C0.001
            if K == 800 and C == 0.001:
                log(f"  (resume baseline) K800 C0.001={auc:.6f}")

    auc_q, K_b, C_b, oof_q, te_q, pairs_b = best_quad
    np.save(PRED11 / "lab2_glm_quad_inter_oof.npy", oof_q.astype(np.float32))
    np.save(PRED11 / "lab2_glm_quad_inter_test.npy", te_q.astype(np.float32))
    log(f"  BEST glm_quad_inter OOF={auc_q:.6f} K={K_b} C={C_b}")

    # DCN-v2 on [Z, log1p(KCNT), gpt2_logit]
    log("  training dcn_gpt2_bag5 …")
    KCNT = gb.KCNT[:NTR]
    KCNT_te = gb.KCNT[gb.TEST]
    gpt_logit_o = rlogit(rk(gpt_o))
    gpt_logit_t = rlogit(rk(gpt_t))
    X_oof = np.hstack([Z_oof, np.log1p(KCNT), gpt_logit_o.reshape(-1, 1)]).astype(np.float32)
    X_te = np.hstack([Z_te, np.log1p(KCNT_te), gpt_logit_t.reshape(-1, 1)]).astype(np.float32)
    # standardize
    scX = StandardScaler().fit(X_oof)
    X_oof, X_te = scX.transform(X_oof).astype(np.float32), scX.transform(X_te).astype(np.float32)

    try:
        oof_dcn, te_dcn = run_dcn_bag(X_oof, y, folds, z_joint_o, X_te, z_joint_t)
        auc_dcn = float(roc_auc_score(y, oof_dcn))
        log(f"  dcn_gpt2_bag5 OOF={auc_dcn:.6f}")
        np.save(PRED11 / "lab2_dcn_gpt2_bag5_oof.npy", oof_dcn.astype(np.float32))
        np.save(PRED11 / "lab2_dcn_gpt2_bag5_test.npy", te_dcn.astype(np.float32))
    except Exception as e:
        log(f"  DCN failed: {e}")
        oof_dcn = oof_q
        te_dcn = te_q
        auc_dcn = auc_q

    # Correlations
    corr = {}
    for name, path in [("gb_bag5", "preds/public_artifacts/gb_bag5_oof.npy"), ("gb_gpt2_bag5", "preds/gb_gpt2_bag5_oof.npy")]:
        p = Path(path)
        if not p.exists():
            # try ART
            alts = list(Path("preds").rglob(f"*{name}*oof*.npy"))
            p = alts[0] if alts else None
        if p and p.exists():
            v = np.load(p).ravel()
            corr[f"quad_vs_{name}"] = float(np.corrcoef(oof_q, v)[0, 1])
            corr[f"dcn_vs_{name}"] = float(np.corrcoef(oof_dcn, v)[0, 1])

    # Stack gates
    names, O, T, champ_meta = load_champ_stack(y, folds)
    gate_q = stack_gate(O, oof_q, y, folds, label="lab2_glm_quad_inter")
    gate_d = stack_gate(O, oof_dcn, y, folds, label="lab2_dcn_gpt2_bag5")

    lab = {
        "name": "LAB2_pairwise_DCN",
        "runtime_s": time.time() - t0,
        "TE_dim": int(D),
        "glm_quad_best_auc": auc_q,
        "glm_quad_best_K": K_b,
        "glm_quad_best_C": C_b,
        "glm_quad_grid": results_K,
        "dcn_auc": auc_dcn,
        "correlations": corr,
        "stack_gate_quad": gate_q,
        "stack_gate_dcn": gate_d,
        "champ_meta": champ_meta,
        "keep_any": bool(gate_q["keep"] or gate_d["keep"]),
    }
    rep["lab2"] = lab
    save_report(rep)
    log(f"LAB2 done in {lab['runtime_s']:.1f}s keep_any={lab['keep_any']}")
    return lab


if __name__ == "__main__":
    run_lab2(load_report())
