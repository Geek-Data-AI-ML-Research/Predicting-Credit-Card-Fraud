"""LAB 1: IDF Exact-Value Collision Graph + Strict OOF 2-Hop Belief Propagation.
Policy: NO_SUBMIT. Artifacts → preds/part11/; gates → part11_paradigm_shift.json.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from part11_common import (
    BASELINE_GLM,
    CATS,
    FEATS,
    FLAGS,
    FLOATS,
    NUMS,
    OUT,
    PRED11,
    TARGET,
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


def _token_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Exact + int + cents tokens for all raw columns."""
    toks: dict[str, np.ndarray] = {}
    for c in CATS + FLAGS:
        toks[c] = df[c].astype(str).to_numpy()
    for c in NUMS:
        if c in FLOATS:
            v = df[c].to_numpy(dtype=float)
            toks[c] = np.asarray([f"{x:.8g}" for x in v], dtype=object)
            toks[f"{c}__int"] = np.floor(v).astype(np.int64).astype(str)
            toks[f"{c}__cents"] = np.round(v * 100).astype(np.int64).astype(str)
        else:
            toks[c] = df[c].astype(str).to_numpy()
    return toks


def build_incidence(all_df: pd.DataFrame, max_freq: int = 500, min_freq: int = 2):
    """Sparse transaction→value incidence with IDF weights for rare exact values."""
    log("LAB1: building token incidence over train+test+orig …")
    toks = _token_series(all_df)
    N = len(all_df)
    row_blocks, col_blocks, data_blocks = [], [], []
    value_meta = []
    col_id = 0
    for name, arr in toks.items():
        codes, uniques = pd.factorize(arr, sort=False)
        vc = pd.Series(codes).value_counts()
        keep = vc[(vc >= min_freq) & (vc <= max_freq)]
        if keep.empty:
            continue
        rare = keep.index.to_numpy()
        idf_of = {int(c): float(np.log(N / float(keep.loc[c]))) for c in rare}
        cid_of = {}
        for c in rare:
            value_meta.append(
                {"col": name, "val": str(uniques[c]), "count": int(keep.loc[c]), "idf": idf_of[int(c)], "cid": col_id}
            )
            cid_of[int(c)] = col_id
            col_id += 1
        mask = np.isin(codes, rare)
        idx = np.nonzero(mask)[0]
        cc = codes[idx]
        lut_c = pd.Series(cid_of)
        lut_w = pd.Series(idf_of)
        row_blocks.append(idx.astype(np.int32))
        col_blocks.append(lut_c.loc[cc].to_numpy(dtype=np.int32))
        data_blocks.append(lut_w.loc[cc].to_numpy(dtype=np.float32))
        log(f"    {name}: {len(rare)} rare values, nnz+={len(idx)}")
    X = sparse.csr_matrix(
        (np.concatenate(data_blocks), (np.concatenate(row_blocks), np.concatenate(col_blocks))),
        shape=(N, max(col_id, 1)),
    )
    log(f"  single-col rare values: {col_id}  nnz={X.nnz}  density={X.nnz/(N*max(col_id,1)):.3e}")
    return X, value_meta, toks


def build_pair_incidence(toks: dict, N: int, top_pairs: int = 15, max_freq: int = 100, min_freq: int = 2):
    """Rare 2-column exact collisions for top-15 column pairs (vectorized)."""
    raw_names = [n for n in toks if "__" not in n]
    pair_cands = []
    for i, a in enumerate(raw_names):
        for b in raw_names[i + 1 :]:
            score = min(len(set(toks[a][:20000])), len(set(toks[b][:20000])))
            pair_cands.append((score, a, b))
    pair_cands.sort(reverse=True)
    chosen = [(a, b) for _, a, b in pair_cands[:top_pairs]]
    log(f"  pair collisions on {len(chosen)} column pairs (vectorized) …")

    row_blocks, col_blocks, data_blocks = [], [], []
    meta = []
    cid0 = 0
    for a, b in chosen:
        # integer codes for pair via factorize of structured view
        ca, _ = pd.factorize(toks[a], sort=False)
        cb, _ = pd.factorize(toks[b], sort=False)
        # unique pair id: cantor-like packing (safe for int64)
        pair_id = ca.astype(np.int64) * (int(cb.max()) + 1) + cb.astype(np.int64)
        vc = pd.Series(pair_id).value_counts()
        keep = vc[(vc >= min_freq) & (vc <= max_freq)]
        if keep.empty:
            continue
        # map rare pair_id → local cid
        rare_ids = keep.index.to_numpy()
        idf_map = {int(pid): (cid0 + k, float(np.log(N / float(keep.loc[pid])))) for k, pid in enumerate(rare_ids)}
        for pid, cnt in keep.items():
            meta.append({"pair": f"{a}|{b}", "val": str(int(pid)), "count": int(cnt), "idf": float(np.log(N / float(cnt))), "cid": idf_map[int(pid)][0]})
        mask = np.isin(pair_id, rare_ids)
        idx = np.nonzero(mask)[0]
        pids = pair_id[idx]
        # lookup via Series map
        lut_c = pd.Series({pid: cid for pid, (cid, _) in idf_map.items()})
        lut_w = pd.Series({pid: w for pid, (_, w) in idf_map.items()})
        row_blocks.append(idx.astype(np.int32))
        col_blocks.append(lut_c.loc[pids].to_numpy(dtype=np.int32))
        data_blocks.append(lut_w.loc[pids].to_numpy(dtype=np.float32))
        cid0 += len(rare_ids)
        log(f"    pair {a}|{b}: {len(rare_ids)} rare keys, nnz+={len(idx)}")
    if not row_blocks:
        return sparse.csr_matrix((N, 1), dtype=np.float32), meta, chosen
    Xp = sparse.csr_matrix(
        (np.concatenate(data_blocks), (np.concatenate(row_blocks), np.concatenate(col_blocks))),
        shape=(N, max(cid0, 1)),
    )
    log(f"  pair rare values: {cid0}  nnz={Xp.nnz}")
    return Xp, meta, chosen


def propagate_fold(
    X: sparse.csr_matrix,
    y: np.ndarray,
    a: np.ndarray,
    query: np.ndarray,
    y_bar: float,
    p_list=(1, 2, 3),
    m_list=(5, 20),
    y_orig: np.ndarray | None = None,
    orig_idx: np.ndarray | None = None,
):
    """2-hop BP: for query rows, IDF-weighted scores from labeled fold-train `a`.
    Similarity S(i,j) = sum_v w_v^2 * 1[i and j share v]  (dot of IDF rows).
    Self-edges excluded.
    """
    # Restrict to values that appear on at least one labeled fraud or any labeled row for rates
    Xa = X[a]
    ya = y[a].astype(np.float64)
    # Value → aggregated fraud mass: for each value col, sum of IDF * y over a
    # Xa.T @ (ya)  gives sum_i∈a X_{i,v} * y_i = sum idf * y for fraud rows sharing v
    # For idf_pos_score of query i: sum_v X_{i,v} * (sum_{j∈a,fraud} X_{j,v})  but WITHOUT self if i∈a
    # = (X @ (Xa.T @ ya_fraud))[query] minus self contribution

    fraud_a = a[ya > 0.5]
    if len(fraud_a) == 0:
        nq = len(query)
        nfeat = 2 + len(p_list) * len(m_list) + 3
        return np.zeros((nq, nfeat), dtype=np.float64)

    Xf = X[fraud_a]
    # value mass for fraud: sum of IDF weights from fraud rows
    v_fraud = np.asarray(Xf.sum(axis=0)).ravel()  # shape (n_vals,)
    idf_pos = np.asarray(X[query].dot(v_fraud)).ravel()

    # Self-correction: if query row is also in fraud_a, subtract its own contribution
    # For row i: contribution of self = sum_v X_{i,v}^2  (since v_fraud includes X_{i,v})
    q_set = set(int(i) for i in query)
    fraud_set = set(int(i) for i in fraud_a)
    self_hits = q_set & fraud_set
    if self_hits:
        for i in self_hits:
            # find position in query
            # slower but rare relative — use map
            pass
        q_pos = {int(q): k for k, q in enumerate(query)}
        for i in self_hits:
            row = X.getrow(i)
            self_c = float(row.multiply(row).sum())
            idf_pos[q_pos[i]] -= self_c

    # Bayesian rates via neighbor aggregation through values
    # Neighbor similarity S(i,j) = (X_i · X_j); for p-power we use (X^p) style via elementwise on weights
    feats = [idf_pos]
    for p in p_list:
        if p == 1:
            Xp = X
        else:
            Xp = X.copy()
            Xp.data = np.power(Xp.data, p)
        Xpa = Xp[a]
        # For each value, sum of similarity-weights * y and sum of weights from fold-train
        # num_v = Xpa.T @ ya ; den_v = Xpa.T @ 1
        num_v = np.asarray(Xpa.T.dot(ya)).ravel()
        den_v = np.asarray(Xpa.T.dot(np.ones(len(a)))).ravel()
        for m in m_list:
            # score_i = (X_i · num_v + m*ybar) / (X_i · den_v + m)  with self removed
            num_i = np.asarray(Xp[query].dot(num_v)).ravel()
            den_i = np.asarray(Xp[query].dot(den_v)).ravel()
            if self_hits:
                # if i∈a, we double-counted the self edge through shared values
                for i in self_hits:
                    row = Xp.getrow(i)
                    # self contribution to num: sum_v X_{i,v} * (X_{i,v} * y_i) = y_i * ||row||^2
                    # self contribution to den: ||row||^2
                    sq = float(row.multiply(row).sum())
                    yi = float(y[i])
                    num_i[q_pos[i]] -= yi * sq
                    den_i[q_pos[i]] -= sq
            rate = (num_i + m * y_bar) / (den_i + m)
            feats.append(rate)

    # max_sim to fraud train / orig
    # max_j∈fraud S(i,j) ≈ max over fraud rows of X_i·X_j — expensive; approximate via
    # top-value channel: for each query, take max over its values of (max IDF among fraud sharing that value)
    # Better cheap approx: idf-weighted max value fraud-count
    # max-sim proxy: presence-weighted IDF mass on values that appear in ≥1 fraud row
    v_fraud_bin = (v_fraud > 0).astype(np.float64)
    # recover per-value IDF ≈ any row's weight on that col; use max over a sample of rows sharing the value
    # cheap: use v_fraud itself normalized by fraud count on value
    max_sim_fraud = np.asarray(X[query].dot(v_fraud_bin * np.maximum(v_fraud, 0))).ravel()

    max_sim_orig = np.zeros(len(query))
    orig_exact_match = np.zeros(len(query))
    if y_orig is not None and orig_idx is not None and len(orig_idx):
        of = orig_idx[y_orig > 0.5]
        if len(of):
            Xof = X[of]
            v_of = np.asarray(Xof.sum(axis=0)).ravel()
            max_sim_orig = np.asarray(X[query].dot((v_of > 0).astype(np.float64) * v_of)).ravel()
            Xbin = X.copy()
            Xbin.data = np.ones_like(Xbin.data)
            v_of_bin = (np.asarray(Xbin[of].sum(axis=0)).ravel() > 0).astype(np.float64)
            orig_exact_match = np.asarray(Xbin[query].dot(v_of_bin)).ravel()

    feats.extend([max_sim_fraud, max_sim_orig, orig_exact_match])
    return np.column_stack(feats).astype(np.float64)


FEAT_NAMES = (
    ["idf_pos_score"]
    + [f"idf_bayes_p{p}_m{m}" for p in (1, 2, 3) for m in (5, 20)]
    + ["max_sim_fraud_train", "max_sim_fraud_orig", "orig_exact_match_count"]
)


def run_lab1(rep: dict) -> dict:
    t0 = time.time()
    log("=" * 72)
    log("LAB 1: IDF Exact-Value Collision Graph + OOF 2-Hop BP")
    log("=" * 72)

    y, te_ids = load_y_ids()
    folds = folds_y(y)
    NTR = len(y)

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")
    orig_path = ROOT / "original" / "credit_card_fraud_2026.csv"
    orig = pd.read_csv(orig_path)
    # align columns
    for c in FEATS:
        if c not in orig.columns:
            raise RuntimeError(f"orig missing {c}")
    y_orig = orig[TARGET].to_numpy().astype(int)
    NTE = len(te)
    NORIG = len(orig)

    all_df = pd.concat(
        [tr[FEATS].reset_index(drop=True), te[FEATS].reset_index(drop=True), orig[FEATS].reset_index(drop=True)],
        ignore_index=True,
    )
    log(f"  rows: train={NTR} test={NTE} orig={NORIG} all={len(all_df)}")

    cache_X = PRED11 / "lab1_incidence.npz"
    cache_meta = PRED11 / "lab1_value_meta.json"
    if cache_X.exists() and cache_meta.exists():
        log("  loading cached incidence …")
        z = np.load(cache_X, allow_pickle=True)
        X = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        meta = json.loads(cache_meta.read_text())
        toks = None
    else:
        X1, meta1, toks = build_incidence(all_df)
        Xp, meta_p, pairs = build_pair_incidence(toks, len(all_df))
        # hstack single + pair
        X = sparse.hstack([X1, Xp], format="csr")
        meta = {"single": meta1, "pairs": meta_p, "pair_cols": pairs, "n_vals": int(X.shape[1])}
        np.savez_compressed(
            cache_X,
            data=X.data,
            indices=X.indices,
            indptr=X.indptr,
            shape=np.array(X.shape),
        )
        cache_meta.write_text(json.dumps(meta, indent=2)[:2_000_000])
        log(f"  saved incidence → {cache_X}  shape={X.shape}")

    train_idx = np.arange(NTR)
    test_idx = np.arange(NTR, NTR + NTE)
    orig_idx = np.arange(NTR + NTE, NTR + NTE + NORIG)

    # OOF feature matrix for train + test
    n_feat = len(FEAT_NAMES)
    F_oof = np.zeros((NTR, n_feat), dtype=np.float64)
    F_te_folds = []

    for fi, (a, b) in enumerate(folds):
        log(f"  fold {fi}: propagate |a|={len(a)} |b|={len(b)}")
        ybar = float(y[a].mean())
        # features for train-val b and for test (and we also need a for completeness — skip)
        Fb = propagate_fold(X, y, a, b, ybar, y_orig=y_orig, orig_idx=orig_idx)
        Ft = propagate_fold(X, y, a, test_idx, ybar, y_orig=y_orig, orig_idx=orig_idx)
        F_oof[b] = Fb
        F_te_folds.append(Ft)
        log(f"    fold{fi} idf_pos AUC vs y[b]={roc_auc_score(y[b], Fb[:,0]):.5f}")

    F_te = np.mean(np.stack(F_te_folds, 0), axis=0)

    # Full-data refit features for test (a = all train)
    log("  full-refit propagate for test …")
    F_te_full = propagate_fold(X, y, train_idx, test_idx, float(y.mean()), y_orig=y_orig, orig_idx=orig_idx)
    # Blend fold-avg and full
    F_te_use = 0.5 * F_te + 0.5 * F_te_full

    for j, nm in enumerate(FEAT_NAMES):
        np.save(PRED11 / f"lab1_{nm}_oof.npy", F_oof[:, j].astype(np.float32))
        np.save(PRED11 / f"lab1_{nm}_test.npy", F_te_use[:, j].astype(np.float32))

    # Standalone predictor: logistic on graph feats
    log("  standalone graph GLM (OOF) …")
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(F_oof, nan=0.0, posinf=0.0, neginf=0.0))
    Xt = sc.transform(np.nan_to_num(F_te_use, nan=0.0, posinf=0.0, neginf=0.0))
    graph_oof = np.zeros(NTR)
    graph_te_folds = []
    for a, b in folds:
        m = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000).fit(Xs[a], y[a])
        graph_oof[b] = m.predict_proba(Xs[b])[:, 1]
        graph_te_folds.append(m.predict_proba(Xt)[:, 1])
    m_full = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000).fit(Xs, y)
    graph_te = 0.5 * np.mean(graph_te_folds, 0) + 0.5 * m_full.predict_proba(Xt)[:, 1]
    graph_auc = float(roc_auc_score(y, graph_oof))
    log(f"  graph standalone OOF AUC={graph_auc:.6f}")
    np.save(PRED11 / "lab1_graph_glm_oof.npy", graph_oof.astype(np.float32))
    np.save(PRED11 / "lab1_graph_glm_test.npy", graph_te.astype(np.float32))

    # Gate A: add to Stage-1 GLM baseline if available
    glm_path = ROOT / "preds" / "mega_glm_gen_oof.npy"
    alt = list((ROOT / "preds").glob("*glm*oof*.npy"))
    stage1 = {"available": False}
    # try bank
    try:
        from part6_bank_load import load_bank_fixed  # type: ignore
    except Exception:
        pass
    try:
        from part9_joint_boost_and_basis import load_bank_fixed

        _, GO, GT, _ = load_bank_fixed(["mega:glm_gen"], y)
        glm_o, glm_t = GO[:, 0], GT[:, 0]
        glm_auc = float(roc_auc_score(y, glm_o))
        # Fit Stage-1 style: concat glm logit + graph feats
        Z = np.column_stack([rlogit(rk(glm_o)), Xs])
        Zt = np.column_stack([rlogit(rk(glm_t)), Xt])
        oof = np.zeros(NTR)
        te_fs = []
        for a, b in folds:
            m = LogisticRegression(C=0.05, solver="lbfgs", max_iter=2000).fit(Z[a], y[a])
            oof[b] = m.predict_proba(Z[b])[:, 1]
            te_fs.append(m.predict_proba(Zt)[:, 1])
        auc = float(roc_auc_score(y, oof))
        delta_glm = auc - glm_auc
        stage1 = {
            "available": True,
            "glm_base_auc": glm_auc,
            "glm_plus_graph_auc": auc,
            "delta": delta_glm,
            "baseline_ref": BASELINE_GLM,
        }
        log(f"  Stage1 GLM {glm_auc:.6f} + graph → {auc:.6f} Δ={delta_glm:+.6f}")
        np.save(PRED11 / "lab1_glm_plus_graph_oof.npy", oof.astype(np.float32))
        np.save(
            PRED11 / "lab1_glm_plus_graph_test.npy",
            (0.5 * np.mean(te_fs, 0) + 0.5 * LogisticRegression(C=0.05, solver="lbfgs", max_iter=2000).fit(Z, y).predict_proba(Zt)[:, 1]).astype(
                np.float32
            ),
        )
    except Exception as e:
        log(f"  Stage1 GLM gate skipped: {e}")
        stage1 = {"available": False, "error": str(e)}

    # Gate B: 4-leaf LGB from z_Joint (glm+gpt2) + graph feats
    lgb_gate = {"available": False}
    try:
        from part9_joint_boost_and_basis import build_p7_65, load_bank_fixed

        P7_names, P7_O, P7_T, _ = build_p7_65(y)
        ni = {n: i for i, n in enumerate(P7_names)}
        _, GO, GT, _ = load_bank_fixed(["mega:glm_gen"], y)
        z_joint_o = rlogit(rk(GO[:, 0])) + rlogit(rk(P7_O[:, ni["gpt2"]]))
        z_joint_t = rlogit(rk(GT[:, 0])) + rlogit(rk(P7_T[:, ni["gpt2"]]))
        # residualize: LGB on graph feats with init_score = z_joint
        oof_lgb = np.zeros(NTR)
        te_lgb = []
        for a, b in folds:
            dtr = lgb.Dataset(Xs[a], label=y[a], init_score=z_joint_o[a])
            dva = lgb.Dataset(Xs[b], label=y[b], init_score=z_joint_o[b], reference=dtr)
            params = dict(
                objective="binary",
                metric="auc",
                learning_rate=0.05,
                num_leaves=4,
                min_data_in_leaf=200,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                verbosity=-1,
                seed=42,
            )
            booster = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            # predict = sigmoid(z_joint + raw)
            raw = booster.predict(Xs[b], raw_score=True)
            oof_lgb[b] = 1 / (1 + np.exp(-(z_joint_o[b] + raw)))
            raw_t = booster.predict(Xt, raw_score=True)
            te_lgb.append(1 / (1 + np.exp(-(z_joint_t + raw_t))))
        lgb_auc = float(roc_auc_score(y, oof_lgb))
        # baseline 4-leaf without graph: just sigmoid(z_joint)
        base_auc = float(roc_auc_score(y, 1 / (1 + np.exp(-z_joint_o))))
        lgb_gate = {
            "available": True,
            "z_joint_auc": base_auc,
            "z_joint_plus_graph_lgb4_auc": lgb_auc,
            "delta": lgb_auc - base_auc,
        }
        log(f"  z_Joint {base_auc:.6f} + graph LGB4 → {lgb_auc:.6f} Δ={lgb_auc-base_auc:+.6f}")
        np.save(PRED11 / "lab1_zjoint_graph_lgb4_oof.npy", oof_lgb.astype(np.float32))
        np.save(PRED11 / "lab1_zjoint_graph_lgb4_test.npy", np.mean(te_lgb, 0).astype(np.float32))
    except Exception as e:
        log(f"  LGB gate skipped: {e}")
        lgb_gate = {"available": False, "error": str(e)}

    # Gate C: add standalone graph predictor into 8B/champ stack
    log("  stack gate vs 8B champ …")
    names, O, T, champ_meta = load_champ_stack(y, folds)
    gate_stack = stack_gate(O, graph_oof, y, folds, label="lab1_graph_glm")
    # also try top bayesian feats
    top_cols = F_oof[:, [0, 1, 2, -3, -2, -1]]  # pos + 2 bayes + max sims
    gate_feats = stack_gate(O, top_cols, y, folds, label="lab1_graph_feats6")

    lab = {
        "name": "LAB1_IDF_collision_graph",
        "runtime_s": time.time() - t0,
        "n_values": int(X.shape[1]),
        "nnz": int(X.nnz),
        "feat_names": FEAT_NAMES,
        "graph_standalone_auc": graph_auc,
        "stage1_gate": stage1,
        "lgb4_gate": lgb_gate,
        "stack_gate_graph_glm": gate_stack,
        "stack_gate_feats6": gate_feats,
        "champ_meta": champ_meta,
        "keep_any": bool(gate_stack["keep"] or gate_feats["keep"] or (stage1.get("delta") or 0) > 1e-9),
    }
    rep["lab1"] = lab
    save_report(rep)
    log(f"LAB1 done in {lab['runtime_s']:.1f}s  keep_any={lab['keep_any']}")
    return lab


if __name__ == "__main__":
    rep = load_report()
    run_lab1(rep)
