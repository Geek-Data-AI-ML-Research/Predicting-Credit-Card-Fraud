"""GLM-boost experiments for Part 6 (built on megayak's public notebook v7, Apache 2.0:
https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack).

Keys, out-of-fold TE, joint keys and the token-pooling GLM inputs are megayak's code (copied from
src/megayak_v7.py); the driver at the bottom is ours: inner-TE seed / model seed, leaves, learning rate, GLM
design and the test-prediction kind are options, so a member can be bagged and varied.

Usage: INPUT_DIR=cache/kin python src/gboost.py <name> [teseed=0] [mseed=42] [leaves=4] [lr=0.02] [kind=lgb]
                                                [glm=gen|gen_ohk] [C=0.3] [ohkC=...] [view=joint|raw] [cs=0.5]
Writes cache/gboost/{oof,foldavg,refit,glmoof,glmrefit}_<name>.npy
"""
display = print
import os, glob, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

INPUT = os.environ.get("INPUT_DIR", "/kaggle/input")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
NJOBS = int(os.environ.get("NJOBS", os.cpu_count()))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def find(name):
    # Prefer flat / data/ lookups — never recurse through preds symlinks onto slow volumes.
    cands = [
        os.path.join(INPUT, name),
        os.path.join(INPUT, "data", name),
        os.path.join(os.path.dirname(INPUT), name),
        os.path.join(os.path.dirname(INPUT), "data", name),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    hits = sorted(glob.glob(os.path.join(INPUT, name)))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


TARGET = "is_fraud"
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
         "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
FEATS = NUMS + FLAGS + CATS
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]

tr = pd.read_csv(find("train.csv"))
te = pd.read_csv(find("test.csv"))
for df in (tr, te):
    df[FLAGS] = df[FLAGS].astype("int8")
y = tr[TARGET].astype(int).values
NTR, NTE = len(tr), len(te)
full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
OOF, PRED = {}, {}
def key_table(df):
    K = {}
    for c in FEATS:
        K[f"v_{c}"] = df[c].astype(str).values if c in CATS else df[c].values
    for c in FLOATS:
        x = df[c].values
        K[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
        K[f"int_{c}"] = np.floor(x).astype(np.int64)
        K[f"ten_{c}"] = np.floor(x / 10).astype(np.int64)
        K[f"dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
    for c in FLOATS:
        i = pd.Series(np.floor(df[c].values).astype(np.int64)).astype(str)
        K[f"l1_{c}"] = (i.str.len() * 10 + i.str[0].astype(int)).values
        K[f"l2_{c}"] = (i.str.len() * 100 + i.str[:2].astype(int)).values
    return K


STR, TOK = None, None     # the floats as written, and their GPT-2 tokens (kept for section 3)


def token_keys():
    global STR, TOK
    try:
        from tokenizers import Tokenizer
        gpt2 = Tokenizer.from_file(find("gpt2-tokenizer.json"))
    except Exception as e:
        log(f"GPT-2 tokenizer unavailable ({type(e).__name__}), token keys skipped")
        return {}
    STR = pd.concat([pd.read_csv(find("train.csv"), usecols=FLOATS, dtype=str),
                     pd.read_csv(find("test.csv"), usecols=FLOATS, dtype=str)], ignore_index=True)
    TOK, K = {}, {}
    for c in FLOATS:
        p = TOK[c] = STR[c].map({v: gpt2.encode(" " + v).tokens for v in STR[c].unique()})
        K[f"t1_{c}"] = p.str[0].values
        K[f"t2_{c}"] = p.map(lambda t: t[1] if len(t) > 1 else "").values
        K[f"tl_{c}"] = p.str[-1].values
        K[f"tn_{c}"] = p.str.len().values
    return K


KEYS = key_table(full)
KEYS.update(token_keys())
KNAMES = list(KEYS)
codes = [pd.factorize(pd.Series(KEYS[n]))[0].astype(np.int32) for n in KNAMES]
KMAT = np.column_stack(codes)
NCAT = np.array([c.max() + 1 for c in codes])
del KEYS, codes

# raw view: the 24 columns as numbers (categoricals as codes) plus the cents of each float
RAW = full[NUMS + FLAGS].copy()
for c in CATS:
    RAW[c] = pd.factorize(full[c], sort=True)[0].astype(np.int16)
for c in FLOATS:
    RAW[f"cents_{c}"] = (np.round(full[c].values * 100) % 100).astype(np.int16)
RAWV = RAW.values.astype(np.float32)
CAT_IDX = [list(RAW.columns).index(c) for c in CATS]
log(f"{len(KNAMES)} target-encoded keys, {RAW.shape[1]} raw columns")
pass

# the 13 pairs from the offline scan; "q:" = deciles of a float (category x risk was scanned with risk deciles and
# is keyed on the exact score, which the G check below and the GLM both prefer)
JOINT = [("j_category_risk", ("merchant_category", "merchant_risk_score")),
         ("j_foreign_ip", ("is_foreign_transaction", "ip_country_mismatch")),
         ("j_category_amount", ("merchant_category", "q:amount_usd")),
         ("j_txn_velocity", ("txn_count_last_24h", "q:velocity_score")),
         ("j_category_age", ("merchant_category", "customer_age")),
         ("j_amount_risk", ("q:amount_usd", "q:merchant_risk_score")),
         ("j_hours_velocity", ("q:hours_since_last_txn", "q:velocity_score")),
         ("j_age_risk", ("customer_age", "q:merchant_risk_score")),
         ("j_newmerchant_weekday", ("is_new_merchant", "day_of_week")),
         ("j_scam_velocity", ("is_ai_generated_scam_attempt", "q:velocity_score")),
         ("j_age_cardage", ("customer_age", "card_age_months")),
         ("j_hour_weekday", ("time_of_day_hour", "day_of_week")),
         ("j_txn_age", ("txn_count_last_24h", "customer_age"))]


def joint_col(df, c):
    return pd.qcut(df[c[2:]], 10, labels=False, duplicates="drop") if c.startswith("q:") else df[c]


def joint_code(df, cols):
    k = np.zeros(len(df), np.int64)
    for c in cols:
        u = pd.factorize(joint_col(df, c))[0]
        k = k * (u.max() + 1) + u
    return pd.factorize(k)[0]


lgt1 = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
rows, JCODES = [], {}
for name, cols in JOINT:
    JCODES[name] = joint_code(full, cols).astype(np.int32)
    k = JCODES[name][:NTR]
    add = np.full(NTR, -(len(cols) - 1) * lgt1(y.mean()))
    dof = 0
    for c in cols:
        u = pd.factorize(joint_col(full, c))[0][:NTR]
        add += lgt1(np.bincount(u, weights=y) / np.bincount(u))[u]
        dof += u.max()
    s, n = np.bincount(k, weights=y), np.bincount(k)
    r = np.bincount(k, weights=1 / (1 + np.exp(-add))) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(s > 0, s * np.log(s / (n * r)), 0) + np.where(n > s, (n - s) * np.log((n - s) / (n * (1 - r))), 0)
    rows.append(dict(key=name, cells=(n > 0).sum(), G=2 * g.sum(), df=(n > 0).sum() - 1 - dof))
GT = pd.DataFrame(rows).set_index("key")
GT["kept"] = GT["G"] > 1.2 * GT["df"]
USED = list(GT.index[GT["kept"]])
NBASE = KMAT.shape[1]          # the trees use the first NBASE keys, the GLM all of them (section 3)
KMAT = np.column_stack([KMAT] + [JCODES[n] for n in USED])
NCAT = np.append(NCAT, [JCODES[n].max() + 1 for n in USED])
KNAMES += USED
log(f"{len(USED)} joint keys kept -> {len(KNAMES)} target-encoded keys")

tab = pd.DataFrame({"foreign": tr["is_foreign_transaction"].values, "ip mismatch": tr["ip_country_mismatch"].values,
                    "y": y})
obs = tab.groupby(["foreign", "ip mismatch"])["y"].agg(["mean", "size"])
ra, rb = tab.groupby("foreign")["y"].mean(), tab.groupby("ip mismatch")["y"].mean()
obs["additive"] = [1 / (1 + np.exp(-(lgt1(ra[i]) + lgt1(rb[j]) - lgt1(y.mean())))) for i, j in obs.index]

def te_rate(k, t, ncat, prior, smooth):
    s = np.bincount(k, weights=t, minlength=ncat)
    n = np.bincount(k, minlength=ncat)
    return (s + prior * smooth) / (n + smooth)


def te_fold(a, b, smooth=20.0, inner=5, seed=0):
    ya = y[a].astype(np.float64)
    Ka, Kb, Kt = KMAT[a], KMAT[b], KMAT[NTR:]
    Ea = np.zeros(Ka.shape, np.float32)
    Eb = np.zeros(Kb.shape, np.float32)
    Et = np.zeros(Kt.shape, np.float32)
    inner_f = list(StratifiedKFold(inner, shuffle=True, random_state=seed).split(np.zeros(len(a)), ya))
    for j in range(KMAT.shape[1]):
        for i, o in inner_f:
            Ea[o, j] = te_rate(Ka[i, j], ya[i], NCAT[j], ya[i].mean(), smooth)[Ka[o, j]]
        r = te_rate(Ka[:, j], ya, NCAT[j], ya.mean(), smooth)
        Eb[:, j], Et[:, j] = r[Kb[:, j]], r[Kt[:, j]]
    return Ea, Eb, Et


t = time.time()
log(f"one outer fold encoded in {time.time() - t:.1f}s")
LGB = dict(n_estimators=20000, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, max_bin=255, verbose=-1)
XGB = dict(n_estimators=20000, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.5,
           min_child_weight=5, reg_lambda=1.0, max_bin=256, tree_method="hist", eval_metric="auc",
           early_stopping_rounds=300)
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


POOL2 = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]
POOL1 = ["velocity_score", "merchant_risk_score"]
GEN = TOK is not None
if GEN:
    dec = lambda c: STR[c].str.split(".").str[1].astype(str)
    lit = lambda c: TOK[c].map(lambda t: t[t.index(".") - 1] if "." in t else t[-1])
    GROUPS = []
    for mem, fn in [(POOL2, dec), (POOL2, lambda c: dec(c).str[0]), (POOL1, dec), (POOL1, lit)]:
        v = pd.concat([fn(c) for c in mem], ignore_index=True)
        GROUPS.append(pd.factorize(v)[0].astype(np.int32).reshape(len(mem), -1).T.copy())
    MARKOV = [pd.factorize(lit(c) + "|" + dec(c))[0].astype(np.int32) for c in POOL2]


def gen_encode(i, o, m0=50.0, m1=2000.0, m2=200.0):
    yi = y[i].astype(np.float64)
    prior, cols, field0 = yi.mean(), [], []
    for g, C in enumerate(GROUPS):
        ncat = int(C.max()) + 1
        s = [np.bincount(C[i, j], weights=yi, minlength=ncat) for j in range(C.shape[1])]
        n = [np.bincount(C[i, j], minlength=ncat) for j in range(C.shape[1])]
        pool = (sum(s) + prior * m0) / (sum(n) + m0)
        for j in range(C.shape[1]):
            h = ((s[j] + m1 * pool) / (n[j] + m1))[C[o, j]]
            cols.append(h)
            if g == 0:
                field0.append(h)
    for j, M in enumerate(MARKOV):
        ncat = int(M.max()) + 1
        s, n = np.bincount(M[i], weights=yi, minlength=ncat), np.bincount(M[i], minlength=ncat)
        cols.append((s[M[o]] + m2 * field0[j]) / (n[M[o]] + m2))
    return np.column_stack(cols).astype(np.float32)


def gen_fold(a, b, inner=5, seed=0):
    Ga = None
    for i, o in StratifiedKFold(inner, shuffle=True, random_state=seed).split(np.zeros(len(a)), y[a]):
        E = gen_encode(a[i], a[o])
        Ga = np.zeros((len(a), E.shape[1]), np.float32) if Ga is None else Ga
        Ga[o] = E
    E = gen_encode(a, np.concatenate([b, np.arange(NTR, NTR + NTE)]))
    return Ga, E[:len(b)], E[len(b):]


ONEHOT = pd.get_dummies(full[CATS].astype(str), dtype=np.float32).values
BASE = np.hstack([full[FLAGS].values.astype(np.float32), ONEHOT])


# ---------------------------------------------------------------- shared helpers (safe to import)
import sys
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

ALL = (np.arange(NTR), np.arange(0))
TEST = np.arange(NTR, NTR + NTE)
KCNT = np.column_stack([np.bincount(KMAT[:, j])[KMAT[:, j]] for j in range(KMAT.shape[1])]).astype(np.float32)
CNT = np.column_stack([full[c].map(full[c].value_counts()).values for c in NUMS]).astype(np.float32)


def make_view_fn(view_mode="joint"):
    def view(idx, E):
        return np.hstack([RAWV[idx], CNT[idx]]) if view_mode == "raw" else np.hstack([RAWV[idx], E, KCNT[idx]])
    return view


def glm_fit_factory(C, OHK=None):
    def glm_fit(idx_a, Xa_dense, Xo_list, ya):
        """Fit the GLM on rows idx_a; return its logit on the fit rows and on each (idx, dense) in Xo_list."""
        if OHK is None:
            m = LogisticRegression(C=C, max_iter=200, tol=1e-7, solver="newton-cholesky").fit(Xa_dense, ya)
            return m.decision_function(Xa_dense), [m.decision_function(X) for _, X in Xo_list]
        Xa = sparse.hstack([sparse.csr_matrix(Xa_dense), OHK[idx_a]]).tocsr()
        m = LogisticRegression(C=C, max_iter=3000, tol=1e-6, solver="lbfgs").fit(Xa, ya)
        outs = [m.decision_function(sparse.hstack([sparse.csr_matrix(X), OHK[i]]).tocsr()) for i, X in Xo_list]
        return m.decision_function(Xa), outs
    return glm_fit


def cache_fold_matrices(teseed, out_dir, glm_C=0.3, view_mode="joint", glm="gen"):
    """Cache za/zb/zt + view(a,Ea)/view(b,Eb)/view(TEST,Et) for each outer fold + full refit.

    Writes out_dir/cache_te{teseed}.npz once (~few minutes CPU). Returns path.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cache_te{teseed}.npz")
    if os.path.exists(path):
        log(f"cache hit {path}")
        return path
    OHK = None
    if glm == "gen_ohk":
        OHK = OneHotEncoder(handle_unknown="ignore", min_frequency=5, dtype=np.float32).fit_transform(KMAT).tocsr()
        OHK = OHK * 0.1
    glm_fit = glm_fit_factory(glm_C, OHK)
    view = make_view_fn(view_mode)
    payload = {"teseed": teseed, "folds": [], "full": None}
    its_placeholder = []
    for f, (a, b) in enumerate(FOLDS):
        Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=teseed)
        Ga_, Gb_, Gt_ = (Ea, Eb, Et)
        if GEN:
            Ga, Gb, Gt = gen_fold(a, b, seed=teseed)
            Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        sc = StandardScaler().fit(logit(Ga_))
        D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
        za, (zb, zt) = glm_fit(a, D(Ga_, a), [(b, D(Gb_, b)), (TEST, D(Gt_, TEST))], y[a])
        Da, Db, Dt = D(Ga_, a), D(Gb_, b), D(Gt_, TEST)
        payload["folds"].append({
            "f": f, "a": a, "b": b,
            "za": za.astype(np.float32), "zb": zb.astype(np.float32), "zt": zt.astype(np.float32),
            "Da": Da.astype(np.float32), "Db": Db.astype(np.float32), "Dt": Dt.astype(np.float32),
            "Xa": view(a, Ea).astype(np.float32),
            "Xb": view(b, Eb).astype(np.float32),
            "Xt": view(TEST, Et).astype(np.float32),
        })
        log(f"cache teseed={teseed} fold {f}: GLM {roc_auc_score(y[b], zb):.5f}")
    # full-data
    a, b = ALL
    Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=teseed)
    Ga_, Gb_, Gt_ = (Ea, Eb, Et)
    if GEN:
        Ga, Gb, Gt = gen_fold(a, b, seed=teseed)
        Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
    sc = StandardScaler().fit(logit(Ga_))
    D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
    za, (zt,) = glm_fit(a, D(Ga_, a), [(TEST, D(Gt_, TEST))], y[a])
    Da, Dt = D(Ga_, a), D(Gt_, TEST)
    payload["full"] = {
        "za": za.astype(np.float32), "zt": zt.astype(np.float32),
        "Da": Da.astype(np.float32), "Dt": Dt.astype(np.float32),
        "Xa": view(a, Ea).astype(np.float32),
        "Xt": view(TEST, Et).astype(np.float32),
    }
    np.savez_compressed(path, **{
        **{f"f{f}_{k}": v for f, fold in enumerate(payload["folds"]) for k, v in fold.items() if k not in ("f",)},
        **{f"full_{k}": v for k, v in payload["full"].items()},
        "teseed": np.array([teseed]),
        "n_folds": np.array([len(FOLDS)]),
    })
    # also store fold index arrays separately for clarity
    meta = {"teseed": teseed, "folds": [{"f": f, "a": fold["a"], "b": fold["b"]} for f, fold in enumerate(payload["folds"])]}
    import json
    with open(path.replace(".npz", "_meta.json"), "w") as fh:
        json.dump({k: ([{kk: vv.tolist() if hasattr(vv, "tolist") else vv for kk, vv in d.items()} for d in v] if k == "folds" else v)
                   for k, v in meta.items()}, fh)
    # rewrite meta more simply
    with open(path.replace(".npz", "_meta.json"), "w") as fh:
        json.dump({"teseed": teseed, "folds": [{"f": f, "n_a": int(len(fold["a"])), "n_b": int(len(fold["b"]))} for f, fold in enumerate(payload["folds"])]}, fh)
    # save fold indices
    np.savez_compressed(path.replace(".npz", "_idx.npz"),
                        **{f"a{f}": fold["a"] for f, fold in enumerate(payload["folds"])},
                        **{f"b{f}": fold["b"] for f, fold in enumerate(payload["folds"])})
    log(f"wrote {path}")
    return path


def run_driver(argv=None):
    """Original Part-6 CLI driver."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ARGS = dict(a.split("=", 1) for a in argv[1:])
    NAME = argv[0]
    TESEED, MSEED = int(ARGS.get("teseed", 0)), int(ARGS.get("mseed", 42))
    LEAVES, LR, KIND = int(ARGS.get("leaves", 4)), float(ARGS.get("lr", 0.02)), ARGS.get("kind", "lgb")
    GLMV, C, VIEW = ARGS.get("glm", "gen"), float(ARGS.get("C", 0.3)), ARGS.get("view", "joint")
    CS, MCS = float(ARGS.get("cs", 0.5)), int(ARGS.get("mcs", 100))
    OUTD = os.environ.get("GB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "gboost"))
    os.makedirs(OUTD, exist_ok=True)
    OHK = None
    if GLMV == "gen_ohk":
        OHK = OneHotEncoder(handle_unknown="ignore", min_frequency=5, dtype=np.float32).fit_transform(KMAT).tocsr()
        OHK = OHK * float(ARGS.get("ohks", 0.1))
    log(f"{NAME}: teseed {TESEED} mseed {MSEED} leaves {LEAVES} lr {LR} kind {KIND} glm {GLMV} C {C} view {VIEW}")
    glm_fit = glm_fit_factory(C, OHK)
    view = make_view_fn(VIEW)

    def boost_params():
        if KIND == "lgb":
            return dict(n_estimators=20000, learning_rate=LR, num_leaves=LEAVES, min_child_samples=MCS, subsample=0.8,
                        subsample_freq=1, colsample_bytree=CS, reg_lambda=1.0, max_bin=255, verbose=-1,
                        n_jobs=NJOBS, random_state=MSEED)
        return dict(n_estimators=20000, learning_rate=LR, max_depth=int(ARGS.get("depth", 2)), subsample=0.8,
                    colsample_bytree=CS, min_child_weight=5, reg_lambda=1.0, max_bin=256, tree_method="hist",
                    eval_metric="auc", n_jobs=NJOBS, random_state=MSEED,
                    **({"device": ARGS["device"]} if "device" in ARGS else {}))

    def bfit(Xa, ya, za, n=None, Xb=None, yb=None, zb=None):
        P = boost_params()
        if KIND == "lgb":
            if n:
                return lgb.LGBMClassifier(**{**P, "n_estimators": n}).fit(Xa, ya, init_score=za, categorical_feature=CAT_IDX), n
            m = lgb.LGBMClassifier(**P)
            m.fit(Xa, ya, init_score=za, eval_set=[(Xb, yb)], eval_init_score=[zb], eval_metric="auc",
                  categorical_feature=CAT_IDX, callbacks=[lgb.early_stopping(300, verbose=False)])
            return m, m.best_iteration_
        if n:
            return xgb.XGBClassifier(**{**P, "n_estimators": n}).fit(Xa, ya, base_margin=za), n
        m = xgb.XGBClassifier(**P, early_stopping_rounds=300)
        m.fit(Xa, ya, base_margin=za, eval_set=[(Xb, yb)], base_margin_eval_set=[zb], verbose=False)
        return m, m.best_iteration + 1

    def bmargin(m, X, z, it):
        if KIND == "lgb":
            return z + m.predict(X, raw_score=True)
        return m.predict(X, base_margin=z, output_margin=True, iteration_range=(0, it))

    oof, foldavg, glm_oof, its = np.zeros(NTR), np.zeros(NTE), np.zeros(NTR), []
    refit = None
    for f, (a, b) in list(enumerate(FOLDS)) + [("all", ALL)]:
        Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=TESEED)
        Ga_, Gb_, Gt_ = (Ea, Eb, Et)
        if GEN:
            Ga, Gb, Gt = gen_fold(a, b, seed=TESEED)
            Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])
        sc = StandardScaler().fit(logit(Ga_))
        D = lambda E, i: np.hstack([sc.transform(logit(E)), BASE[i]])
        if f == "all":
            za, (zt,) = glm_fit(a, D(Ga_, a), [(TEST, D(Gt_, TEST))], y[a])
            n = int(round(np.mean(its) * 1.25))
            m, _ = bfit(view(a, Ea), y, za, n=n)
            refit = bmargin(m, view(TEST, Et), zt, n)
            np.save(os.path.join(OUTD, f"glmrefit_{NAME}.npy"), zt.astype(np.float32))
            log(f"refit on all rows with {n} trees")
            break
        za, (zb, zt) = glm_fit(a, D(Ga_, a), [(b, D(Gb_, b)), (TEST, D(Gt_, TEST))], y[a])
        glm_oof[b] = zb
        m, it = bfit(view(a, Ea), y[a], za, Xb=view(b, Eb), yb=y[b], zb=zb)
        its.append(it)
        oof[b] = bmargin(m, view(b, Eb), zb, it)
        foldavg += bmargin(m, view(TEST, Et), zt, it) / len(FOLDS)
        log(f"{NAME} fold {f}: GLM {roc_auc_score(y[b], zb):.5f} -> boosted {roc_auc_score(y[b], oof[b]):.5f} ({it} trees)")

    for k, v in (("oof", oof), ("foldavg", foldavg), ("refit", refit), ("glmoof", glm_oof)):
        np.save(os.path.join(OUTD, f"{k}_{NAME}.npy"), v.astype(np.float32))
    np.save(os.path.join(OUTD, f"test_{NAME}.npy"), ((rankdata(foldavg) + rankdata(refit)) / (2 * NTE)).astype(np.float32))
    fa = [roc_auc_score(y[b], oof[b]) for _, b in FOLDS]
    log(f"==> {NAME}: OOF {roc_auc_score(y, oof):.6f} (GLM {roc_auc_score(y, glm_oof):.6f}) | folds " +
        " ".join(f"{x:.5f}" for x in fa))
    with open(os.path.join(OUTD, "results.csv"), "a") as fh:
        fh.write(f"{time.strftime('%m-%d %H:%M')},{NAME},{roc_auc_score(y, oof):.6f},{roc_auc_score(y, glm_oof):.6f},"
                 f"{' '.join(argv[1:])}\n")


if __name__ == "__main__":
    run_driver()
