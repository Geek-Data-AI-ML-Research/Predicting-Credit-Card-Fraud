
# %% [markdown] cell 0
"""
# S6E11 Credit Card Fraud | Highest Public LB 0.95154 | Self-Trained

**Public LB 0.95154 · 5-fold CV 0.95084** — the highest public-notebook score on the board (Nov 2, tied).
Every model in this notebook is trained here, from the competition data and the original dataset only:
no external predictions.

What the score is built on, each measured on the same `StratifiedKFold(5, shuffle=True, random_state=42)`:

1. **Every value has its own fraud rate** (my [EDA notebook](https://www.kaggle.com/code/sidhaarthshree/s6e11-credit-card-fraud-a-complete-eda)).
   Each column is target-encoded value by value, out of fold.
2. **The digits carry the label.** The rows were written like text, number by number, so the *parts* of a
   number have their own fraud rate: integer part, decimals, `int(x/10)`, `int(10x)`, and the leading one
   and two digits with the length of the integer part. Each part is a key with its own encoding.
3. **The signal is close to additive**: 4-leaf LightGBM and depth-2 XGBoost beat deeper trees, and a
   logistic regression that fits one coefficient per value (jointly, on top of the encodings) is a strong,
   different member.
4. **Less noise on the test side.** Each tree member is averaged over seeds (model seed *and* the inner folds
   of the encoding), and every member's test prediction comes from a refit on all 700k training rows
   (1.25x the mean best iteration).

| member | model | features |
|---|---|---|
| `lgb_k` | LightGBM, 4 leaves, 3 seeds | value + digit-key encodings, decimals, value counts |
| `lgb_k_s100` | LightGBM, 4 leaves, 2 seeds | same, encodings smoothed 100 (a second view) |
| `lgb_k_l7` | LightGBM, 7 leaves | same as `lgb_k` |
| `xgb_k` | XGBoost, depth 2 | same as `lgb_k` |
| `lgb_raw` | LightGBM, 4 leaves | raw columns + decimals + value counts, no encodings |
| `glm_joint` | logistic regression | encoding logits + one-hot of every value and every digit key |
| `realmlp_k` | RealMLP (GPU) | value + digit-key encodings |

A regularised logistic regression on the members' logits makes the submission. Out-of-fold and refit test
predictions are saved in `preds/`. Digit keys, refits and seed bags were discussed openly in this
competition's notebooks and forum; the code and every model here are my own.
"""

# %% cell 1
import glob, os, time, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
INPUT = os.environ.get("INPUT_DIR", "/kaggle/input")
def find(name, hint=""):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True))
    return ([h for h in hits if hint in h] or hits)[0]

SMOKE = os.environ.get("SMOKE") == "1"       # tiny dry run
SEED, N_FOLDS, TARGET = 42, 5, "is_fraud"
N_JOBS = os.cpu_count()
try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
FEATS = NUMS + FLAGS + CATS
DEC = {"amount_usd": 2, "account_balance_usd": 2, "hours_since_last_txn": 2, "distance_from_home_km": 2,
       "velocity_score": 1, "merchant_risk_score": 1}      # float columns and their decimals
FLOATS = list(DEC)

tr = pd.read_csv(find("train.csv", "playground-series-s6e11"))
te = pd.read_csv(find("test.csv", "playground-series-s6e11"))
og = pd.read_csv(find("credit_card_fraud_2026.csv")).drop(columns="transaction_id")
if SMOKE:
    tr, te = tr.iloc[:40000].copy(), te.iloc[:10000].copy()
for d in (tr, te, og):
    d[FLAGS] = d[FLAGS].astype("int8")
y = tr[TARGET].astype(int).values
for c in CATS:
    cats = sorted(pd.concat([tr[c], te[c], og[c]]).unique())
    for d in (tr, te):
        d[c] = pd.Categorical(d[c], categories=cats)
full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
NTR = len(tr)

FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
OOF, FULL, ITS = {}, {}, {}        # out-of-fold, refit test prediction, best iterations
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
print(f"train {tr.shape} | test {te.shape} | fraud rate {y.mean():.4%} | {N_JOBS} CPUs | GPU {GPU}")

# %% [markdown] cell 2
"""
## 1. The digits carry the label

Fraud rate by the first decimal of each two-decimal column. A physical quantity should not care about its
first decimal, yet the same digits stand out in every column. That is the fingerprint of a generator that
writes numbers as text.
"""

# %% cell 3
D4 = FLOATS[:4]
first_dec = pd.DataFrame({c: pd.Series(y).groupby((np.round(tr[c].values * 100).astype(np.int64) // 10) % 10).mean()
                          for c in D4}).T
fig, ax = plt.subplots(figsize=(10, 3.2))
lift = first_dec.values / y.mean() - 1
im = ax.imshow(lift, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
ax.set_xticks(range(10), [f".{d}" for d in range(10)]); ax.set_yticks(range(len(D4)), D4)
for i in range(lift.shape[0]):
    for j in range(lift.shape[1]):
        ax.text(j, i, f"{first_dec.values[i, j]:.2%}", ha="center", va="center", fontsize=8)
ax.set_title(f"Fraud rate by first decimal digit (base {y.mean():.2%})", loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, format=plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
plt.tight_layout(); plt.show()

# %% [markdown] cell 4
"""
## 2. Keys

For each float column, six keys read from the number as written (`1371.37`):

| key | value | groups rows by |
|---|---|---|
| `int` | 1371 | integer part |
| `frac` | 37 | the decimals |
| `ten` | 137 | integer part without its last digit |
| `dec` | 13713 | the value cut to one decimal |
| `l1` | 4·1 | length of the integer part + its first digit |
| `l2` | 4·13 | length + first two digits |

Plus three plain numbers per float column for the trees: the decimals, and the first and second decimal digit.
"""

# %% cell 5
def key_frame(D):
    out = {}
    for c, k in DEC.items():
        s = D[c].map(lambda v: f"{v:.{k}f}")
        ip, fp = s.str.split(".").str[0], s.str.split(".").str[1]
        L = ip.str.len().astype(str)
        out[f"int_{c}"], out[f"frac_{c}"] = ip, fp
        out[f"ten_{c}"] = np.floor(D[c] / 10).astype("int64").astype(str)
        out[f"dec_{c}"] = np.floor(D[c] * 10 + 1e-6).astype("int64").astype(str)
        out[f"l1_{c}"], out[f"l2_{c}"] = L + "|" + ip.str[:1], L + "|" + ip.str[:2]
    return pd.DataFrame(out)
KF = key_frame(full)
KEYS = list(KF.columns)

# integer codes of every column and key over train + test (unseen-in-train codes fall back to the prior)
ENC_COLS = FEATS + KEYS
CODES = np.column_stack([pd.factorize(full[c].astype(str) if c in CATS else full[c])[0] for c in FEATS] +
                        [pd.factorize(KF[k])[0] for k in KEYS]).astype(np.int32)
NCODE = CODES.max(0) + 1

# numeric frame for the trees: raw columns, value counts, decimals and decimal digits
NUM = full.copy()
for c in FLOATS:
    vc = pd.concat([full[c], og[c]]).value_counts()
    NUM[f"cnt_{c}"] = full[c].map(vc).values.astype(np.float32)
    k = np.round(full[c].values * 10 ** DEC[c]).astype(np.int64)
    NUM[f"frac_{c}"] = (k % 10 ** DEC[c]).astype(np.int16)
    NUM[f"d1_{c}"] = ((k // 10 ** (DEC[c] - 1)) % 10).astype(np.int8)
    if DEC[c] == 2:
        NUM[f"d2_{c}"] = (k % 10).astype(np.int8)
print(f"{len(KEYS)} keys, {NUM.shape[1]} numeric columns")
print(KF.iloc[:NTR].nunique().sort_values(ascending=False).head(12).to_string())

# %% [markdown] cell 6
"""
## 3. Out-of-fold target encoding

Inside each training fold an inner 5-fold split encodes the training rows; the validation fold and the test
rows get encodings fitted on the whole training fold. For the refits, "training fold" is all of train.
Smoothing pulls rare values toward the base rate. Counting with `np.bincount` on integer codes keeps this
fast enough to run a dozen times.
"""

# %% cell 7
def te_block(rows_fit, rows_apply, smooth, prior):
    """Encoding of every column for rows_apply, fitted on rows_fit (train indices)."""
    out = np.empty((len(rows_apply), len(ENC_COLS)), np.float32)
    yf = y[rows_fit]
    for j in range(len(ENC_COLS)):
        kf, ka = CODES[rows_fit, j], CODES[rows_apply, j]
        s = np.bincount(kf, weights=yf, minlength=NCODE[j]); n = np.bincount(kf, minlength=NCODE[j])
        out[:, j] = (s[ka] + prior * smooth) / (n[ka] + smooth)
    return out

def encode(train_idx, apply_idx, smooth=20.0, seed=0):
    """(encodings of train_idx via inner folds, encodings of apply_idx = indices into train+test)."""
    prior = y[train_idx].mean()
    inner = np.empty((len(train_idx), len(ENC_COLS)), np.float32)
    for i, j in StratifiedKFold(5, shuffle=True, random_state=seed).split(train_idx, y[train_idx]):
        inner[j] = te_block(train_idx[i], train_idx[j], smooth, prior)
    return inner, te_block(train_idx, apply_idx, smooth, prior)

TE_NAMES = ["te_" + c for c in ENC_COLS]
TEST = np.arange(NTR, NTR + len(te))
_TE = {}
def encoded(f, smooth=20.0, seed=0):
    """Encodings for fold f (train, valid, test) or f="full" (train, None, test). Cached."""
    if (f, smooth, seed) not in _TE:
        a, b = (np.arange(NTR), None) if f == "full" else FOLDS[f]
        ea, eo = encode(a, TEST if b is None else np.concatenate([b, TEST]), smooth, seed)
        _TE[(f, smooth, seed)] = (ea, None, eo) if b is None else (ea, eo[:len(b)], eo[len(b):])
    return _TE[(f, smooth, seed)]

def clear():
    _TE.clear(); gc.collect()

# %% [markdown] cell 8
"""
## 4. Tree members: CV, then one refit on every training row

Early stopping on each validation fold gives the out-of-fold predictions and five best iterations. The refit
trains on all rows with 1.25x their mean (25% more rows), no early stopping. A *bag* repeats this with other
model seeds and other inner folds for the encodings and averages.
"""

# %% cell 9
LR = 0.03
LGB = dict(learning_rate=LR, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=N_JOBS, verbose=-1)
XGB = dict(learning_rate=LR, max_depth=2, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
           reg_lambda=1.0, tree_method="hist", eval_metric="auc", n_jobs=N_JOBS, enable_categorical=True,
           max_cat_to_onehot=1, **(dict(device="cuda") if GPU else {}))
NUM_TR, NUM_TE = NUM.iloc[:NTR].reset_index(drop=True), NUM.iloc[NTR:].reset_index(drop=True)

def frame(rows, enc, use_te):
    D = NUM_TR.iloc[rows] if rows is not None else NUM_TE
    D = D.reset_index(drop=True)
    return pd.concat([D, pd.DataFrame(enc, columns=TE_NAMES)], axis=1) if use_te else D

def fit(kind, n, seed, Xa, ya, Xb=None, yb=None, **over):
    if kind == "lgb":
        m = lgb.LGBMClassifier(**{**LGB, **over}, n_estimators=n, random_state=seed)
        if Xb is None:
            return m.fit(Xa, ya), n
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
        return m, m.best_iteration_
    m = xgb.XGBClassifier(**{**XGB, **over}, n_estimators=n, random_state=seed,
                          **({} if Xb is None else dict(early_stopping_rounds=300)))
    if Xb is None:
        return m.fit(Xa, ya, verbose=False), n
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m, m.best_iteration + 1

def run_tree(name, kind, smooth=20.0, seed=SEED, teseed=0, use_te=True, **over):
    oof, its, t0 = np.zeros(NTR), [], time.time()
    for f, (a, b) in enumerate(FOLDS):
        ea, eb, _ = encoded(f, smooth, teseed) if use_te else (None, None, None)
        m, it = fit(kind, 30000, seed, frame(a, ea, use_te), y[a], frame(b, eb, use_te), y[b], **over)
        oof[b] = m.predict_proba(frame(b, eb, use_te))[:, 1]
        its.append(it)
        print(f"{name} fold {f}: AUC {roc_auc_score(y[b], oof[b]):.5f}, {it} trees ({time.time() - t0:.0f}s)")
    n = int(round(np.mean(its) * 1.25))
    ea, _, et = encoded("full", smooth, teseed) if use_te else (None, None, None)
    m, _ = fit(kind, n, seed, frame(np.arange(NTR), ea, use_te), y, **over)
    OOF[name], FULL[name], ITS[name] = oof, m.predict_proba(frame(None, et, use_te))[:, 1], its
    print(f"==> {name}: CV AUC {roc_auc_score(y, oof):.5f} | refit {n} trees ({time.time() - t0:.0f}s)\n")

def bag(name, parts):
    OOF[name] = np.mean([OOF.pop(p) for p in parts], axis=0)
    FULL[name] = np.mean([FULL.pop(p) for p in parts], axis=0)
    print(f"==> {name} ({len(parts)} seeds): CV AUC {roc_auc_score(y, OOF[name]):.5f}\n")

# %% [markdown] cell 10
"""
### Encodings with smoothing 20, inner seed 0: LightGBM (4 and 7 leaves) and XGBoost
"""

# %% cell 11
run_tree("lgb_k_0", "lgb")
run_tree("lgb_k_l7", "lgb", num_leaves=7)
run_tree("xgb_k", "xgb")

# %% [markdown] cell 12
"""
### The joint GLM

A logistic regression over: standardized numbers and their B-splines, flags, one-hot categoricals, the
standardized logits of all the encodings, and one-hot columns for every exact value of every numeric column
and every digit key (values seen fewer than 5 times share a column). Fitting all of them together under one
L2 penalty lets each value's coefficient correct its marginal encoding for the other columns, and lets a rare
value fall back on its coarser keys. With ~100k columns it needs a strong penalty (`C=0.01`).
"""

# %% cell 13
num = full[NUMS].astype(float).copy()
for c in ("amount_usd", "account_balance_usd", "hours_since_last_txn"):
    num[c] = np.log1p(num[c])
Z = StandardScaler().fit_transform(num)
ONEHOT = sparse.hstack([
    sparse.csr_matrix(Z), sparse.csr_matrix(full[FLAGS].values.astype(float)),
    OneHotEncoder(handle_unknown="ignore").fit_transform(full[CATS].astype(str)),
    sparse.csr_matrix(SplineTransformer(n_knots=8, degree=3, knots="quantile").fit_transform(Z)),
    OneHotEncoder(handle_unknown="ignore", min_frequency=5).fit_transform(full[NUMS]),
    OneHotEncoder(handle_unknown="ignore", min_frequency=5).fit_transform(KF),
]).tocsr()
OH_TR, OH_TE = ONEHOT[:NTR], ONEHOT[NTR:]

def glm_design(Ms, Es):
    sc = StandardScaler().fit(logit(Es[0]))
    return [sparse.hstack([M, sparse.csr_matrix(sc.transform(logit(E)))]).tocsr() for M, E in zip(Ms, Es)]

glm = lambda: LogisticRegression(C=0.01, max_iter=4000, tol=1e-6)
oof, t0 = np.zeros(NTR), time.time()
for f, (a, b) in enumerate(FOLDS):
    ea, eb, _ = encoded(f)
    Da, Db = glm_design([OH_TR[a], OH_TR[b]], [ea, eb])
    oof[b] = glm().fit(Da, y[a]).predict_proba(Db)[:, 1]
    print(f"glm_joint fold {f}: AUC {roc_auc_score(y[b], oof[b]):.5f} ({time.time() - t0:.0f}s)")
ea, _, et = encoded("full")
Da, Dt = glm_design([OH_TR, OH_TE], [ea, et])
OOF["glm_joint"], FULL["glm_joint"] = oof, glm().fit(Da, y).predict_proba(Dt)[:, 1]
print(f"==> glm_joint {Da.shape}: CV AUC {roc_auc_score(y, oof):.5f} ({time.time() - t0:.0f}s)\n")

# %% [markdown] cell 14
"""
### RealMLP on the same encodings (GPU)

4 ensemble members, 4 epochs, no early stopping; the test prediction is the 5-fold average.
"""

# %% cell 15
if GPU:
    os.system("pip install -q pytabkit")
    from pytabkit import RealMLP_TD_Classifier
    REALMLP = dict(
        n_ens=4, n_epochs=4, batch_size=512, use_early_stopping=False, lr=0.04, wd=0.0236, sq_mom=0.988,
        lr_sched="lin_cos_log_15", wd_sched="cos_log_15", first_layer_lr_factor=0.25, embedding_size=5,
        max_one_hot_cat_size=18, hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="expm4t",
        plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33, ls_eps=0.01,
        ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
        tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"])
    oof, pred, t0 = np.zeros(NTR), np.zeros(len(te)), time.time()
    for f, (a, b) in enumerate(FOLDS):
        ea, eb, et = encoded(f)
        Xa, Xb, XT = (pd.concat([D[FEATS].reset_index(drop=True), pd.DataFrame(E, columns=TE_NAMES)], axis=1)
                      for D, E in ((NUM_TR.iloc[a], ea), (NUM_TR.iloc[b], eb), (NUM_TE, et)))
        m = RealMLP_TD_Classifier(device="cuda", random_state=SEED, val_metric_name="1-auc_ovr", verbosity=0, **REALMLP)
        m.fit(Xa, y[a], X_val=Xb, y_val=y[b], cat_col_names=CATS)
        oof[b] = m.predict_proba(Xb)[:, 1]; pred += m.predict_proba(XT)[:, 1] / N_FOLDS
        print(f"realmlp_k fold {f}: AUC {roc_auc_score(y[b], oof[b]):.5f} ({time.time() - t0:.0f}s)")
    OOF["realmlp_k"], FULL["realmlp_k"] = oof, pred
    print(f"==> realmlp_k: CV AUC {roc_auc_score(y, oof):.5f}\n")
else:
    print("no GPU: RealMLP skipped")
clear()

# %% [markdown] cell 16
"""
### Two more seeds for the main LightGBM, and the smoothing-100 view
"""

# %% cell 17
run_tree("lgb_k_1", "lgb", seed=1, teseed=1); clear()
run_tree("lgb_k_2", "lgb", seed=2, teseed=2); clear()
bag("lgb_k", ["lgb_k_0", "lgb_k_1", "lgb_k_2"])
run_tree("lgb_k_s100_0", "lgb", smooth=100.0); clear()
run_tree("lgb_k_s100_1", "lgb", smooth=100.0, seed=1, teseed=1); clear()
bag("lgb_k_s100", ["lgb_k_s100_0", "lgb_k_s100_1"])

# %% [markdown] cell 18
"""
### The raw view: no encodings

Raw columns, value counts and the decimals as plain numbers. Weaker alone, but it sees the data without any
encoding, which is what a stack needs next to the encoded members.
"""

# %% cell 19
run_tree("lgb_raw", "lgb", use_te=False)

# %% [markdown] cell 20
"""
## 5. Stack

A logistic regression on the members' logits (`C=0.01`), scored on a second 5-fold split so it never scores
rows it was fitted on, then refitted on all out-of-fold rows and applied to the refit test predictions.
"""

# %% cell 21
NAMES = list(OOF)
Lo = np.column_stack([logit(OOF[n]) for n in NAMES])
Lf = np.column_stack([logit(FULL[n]) for n in NAMES])
st = np.zeros(NTR)
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Lo, y):
    st[b] = LogisticRegression(C=0.01, max_iter=3000).fit(Lo[a], y[a]).decision_function(Lo[b])
stack = LogisticRegression(C=0.01, max_iter=3000).fit(Lo, y)
res = pd.DataFrame({"CV AUC": [roc_auc_score(y, OOF[n]) for n in NAMES] + [roc_auc_score(y, st)],
                    "stack weight": list(stack.coef_[0]) + [np.nan]}, index=NAMES + ["STACK (cross-validated)"])
print(res.round({"CV AUC": 5, "stack weight": 3}).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
r = res.iloc[:-1].sort_values("CV AUC")
ax.barh(r.index, r["CV AUC"], color="#2a78d6")
ax.axvline(res.iloc[-1, 0], color="#eb6834", lw=2, label=f"stack {res.iloc[-1, 0]:.5f}")
ax.set_xlim(r["CV AUC"].min() - 0.001, res.iloc[-1, 0] + 0.0005)
ax.set_xlabel("5-fold CV AUC"); ax.legend(loc="lower right", frameon=False)
ax.set_title("Members and the stack", loc="left")
plt.tight_layout(); plt.show()

# %% cell 22
sub = pd.DataFrame({"id": te["id"], TARGET: 1 / (1 + np.exp(-stack.decision_function(Lf)))})
assert len(sub) == len(te) and sub[TARGET].between(0, 1).all()
sub.to_csv("submission.csv", index=False)
os.makedirs("preds", exist_ok=True)
for n in NAMES:
    np.save(f"preds/{n}_oof.npy", OOF[n].astype(np.float32)); np.save(f"preds/{n}_full.npy", FULL[n].astype(np.float32))
sub.head()

# %% [markdown] cell 23
"""
## What did not help (5-fold CV unless noted)

| idea | result |
|---|---|
| original 20k rows appended to training | 0.94864 vs 0.94886 |
| original rows as an extra encoding source | 0.94875 vs 0.94886 |
| 10 inner folds for the encodings | 0.94853 vs 0.94886 |
| CatBoost (depth 4) on the digit encodings | 0.94909, no gain in the stack |
| synthetic rows copying an original row (3+ exact matching values) | 296 rows, original label AUC 0.49: no memorisation |

If this helped you, an upvote is appreciated.
"""
