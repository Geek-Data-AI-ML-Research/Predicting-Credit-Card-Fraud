import os, glob, time, json, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
try:
    display
except NameError:
    display = print
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
ROOT = os.environ.get("KAGGLE_INPUT", "/kaggle/input"); OUT = os.environ.get("KAGGLE_OUT", "/kaggle/working")
files = sorted(glob.glob(f"{ROOT}/**/*.csv", recursive=True))
comp = [f for f in files if "/competitions/" in f] or files
TRAIN = [f for f in comp if os.path.basename(f) == "train.csv"][0]
SUB = [f for f in comp if os.path.basename(f) == "sample_submission.csv"][0]
log("train:", TRAIN)
tr = pd.read_csv(TRAIN, usecols=["id", "is_fraud"], dtype=str, keep_default_na=False)
BOOL_MAP = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0}
y = tr["is_fraud"].str.strip().str.lower().map(BOOL_MAP)
log("label values:", tr["is_fraud"].str.strip().str.lower().value_counts().head(5).to_dict())
y = y.astype(int).values
NTR = len(y); sub = pd.read_csv(SUB); NTE = len(sub)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))

#====

paths = {}; FULL = {}
GENERIC = {"oof", "preds", "artifacts", "output", "outputs", "working"}
def owner_of(dn):
    b = os.path.basename(dn)
    return os.path.basename(os.path.dirname(dn)) if b in GENERIC else b
for f in sorted(glob.glob(f"{ROOT}/**/*.npy", recursive=True)):
    bn, dn = os.path.basename(f), os.path.dirname(f)
    if bn.startswith("oof_"): base, t = bn[4:-4], os.path.join(dn, "test_" + bn[4:])
    elif bn.endswith("_oof.npy"):
        base = bn[:-8]; t = os.path.join(dn, base + "_test.npy")
        if not os.path.exists(t): t = os.path.join(dn, base + "_full.npy")   # some public runs only save refit test preds
    else: continue
    n = owner_of(dn)[:12] + ":" + base
    if os.path.exists(t) and n not in paths:
        o_, t_ = np.load(f), np.load(t)
        if len(o_) == NTR and len(t_) == NTE:
            paths[n] = (o_, t_)
            fl = os.path.join(dn, "full_" + base + ".npy")
            if os.path.exists(fl): FULL[n] = np.load(fl)
            if t.endswith("_full.npy"): FULL[n] = t_
        else: log("skip (shape)", n, o_.shape, t_.shape)
# .npz banks: pair every array of length NTR whose key mentions oof with a test/full array of length NTE
for f in sorted(glob.glob(f"{ROOT}/**/*.npz", recursive=True)):
    try:
        z = np.load(f, allow_pickle=False); keys = list(z.keys())
        log("npz", os.path.basename(f), {k: z[k].shape for k in keys})
        for k in keys:
            if "oof" not in k.lower() or z[k].ndim != 1 or len(z[k]) != NTR: continue
            for alt in (k.replace("oof", "test"), k.replace("oof", "full"), k.replace("oof", "tst"), k.replace("OOF", "TEST")):
                if alt in z and z[alt].ndim == 1 and len(z[alt]) == NTE:
                    n = owner_of(os.path.dirname(f))[:12] + ":" + k.replace("_oof", "").replace("oof_", "")
                    paths[n] = (z[k], z[alt]); log("  npz member", n, "<-", k, "+", alt); break
    except Exception as e:
        log("npz skip", f, str(e)[:120])
names = list(paths)
O = np.column_stack([rankdata(paths[n][0]) / NTR for n in names])
Tm = np.column_stack([rankdata(paths[n][1]) / NTE for n in names])
auc = {n: float(roc_auc_score(y, O[:, i])) for i, n in enumerate(names)}
log("members with full-data refit test predictions:", list(FULL))
log(f"{len(names)} members"); print(pd.Series(auc).sort_values(ascending=False).round(5).to_string(), flush=True)
# drop weak members to keep the search sane (they rarely help and slow hill climbing)
thr = max(auc.values()) - 0.006
keep = [i for i, n in enumerate(names) if auc[n] >= thr]
names = [names[i] for i in keep]; O = O[:, keep]; Tm = Tm[:, keep]
log(f"{len(names)} members kept (OOF AUC >= {thr:.5f})")

#====

import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
s_auc = pd.Series({n: auc[n] for n in names}).sort_values()
fig, ax = plt.subplots(figsize=(9, 0.32 * len(s_auc) + 1))
ax.barh(s_auc.index, s_auc.values, color="#2a6f97"); ax.set_xlim(s_auc.min() - 0.001, s_auc.max() + 0.0006)
ax.set_xlabel("out-of-fold AUC"); ax.set_title("Members used in the blend"); plt.tight_layout(); plt.show()
corr = pd.DataFrame(O, columns=names).corr()
fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(corr.values, cmap="viridis")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
plt.colorbar(im, ax=ax); ax.set_title("Rank correlation between members (lower = more diverse)"); plt.tight_layout(); plt.show()

#====

def hill_climb(O, yy, iters=80):
    w = np.zeros(O.shape[1]); w[int(np.argmax([roc_auc_score(yy, O[:, i]) for i in range(O.shape[1])]))] = 1
    cur = roc_auc_score(yy, O @ w)
    for _ in range(iters):
        sc, i = max((roc_auc_score(yy, O @ (w + np.eye(O.shape[1])[i])), i) for i in range(O.shape[1]))
        if sc <= cur + 1e-7: break
        w[i] += 1; cur = sc
    return w / w.sum()
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
cv = {k: np.zeros(NTR) for k in ["mean", "hill", "stack", "stack_c01", "mix"]}
for a, b in FOLDS:
    cv["mean"][b] = O[b].mean(1)
    cv["hill"][b] = O[b] @ hill_climb(O[a], y[a])
    cv["stack"][b] = LogisticRegression(C=1.0, max_iter=3000).fit(logit(O[a]), y[a]).decision_function(logit(O[b]))
    cv["stack_c01"][b] = LogisticRegression(C=0.01, max_iter=3000).fit(logit(O[a]), y[a]).decision_function(logit(O[b]))
    cv["mix"][b] = rankdata(cv["hill"][b]) / len(b) + rankdata(cv["stack"][b]) / len(b)
scores = {k: float(roc_auc_score(y, v)) for k, v in cv.items()}
log("nested-CV blend AUC:", {k: round(float(v), 6) for k, v in scores.items()})
best = max(scores, key=scores.get)
w = hill_climb(O, y)
st = LogisticRegression(C=1.0, max_iter=3000).fit(logit(O), y)
st01 = LogisticRegression(C=0.01, max_iter=3000).fit(logit(O), y)
final = {"mean": Tm.mean(1), "hill": Tm @ w, "stack": st.decision_function(logit(Tm)),
         "stack_c01": st01.decision_function(logit(Tm))}
final["mix"] = rankdata(final["hill"]) / NTE + rankdata(final["stack"]) / NTE
log("hill weights:", {n: round(float(v), 3) for n, v in zip(names, w) if v > 0})
sub_fold = sub.copy(); sub_fold["is_fraud"] = rankdata(final[best]) / NTE
sub_fold.to_csv(f"{OUT}/submission_foldavg.csv", index=False)
# same weights, but the bagged members' test columns come from the full-data refit
Tm2 = Tm.copy()
for j, n in enumerate(names):
    if n in FULL: Tm2[:, j] = rankdata(FULL[n]) / NTE
f2 = {"mean": Tm2.mean(1), "hill": Tm2 @ w, "stack": st.decision_function(logit(Tm2)), "stack_c01": st01.decision_function(logit(Tm2))}
f2["mix"] = rankdata(f2["hill"]) / NTE + rankdata(f2["stack"]) / NTE
sub["is_fraud"] = rankdata(f2[best]) / NTE
sub.to_csv(f"{OUT}/submission.csv", index=False)   # the CV-validated blend
log("refit swapped for", [n for n in names if n in FULL], "| rank corr refit vs fold-avg:",
    round(float(np.corrcoef(sub["is_fraud"], sub_fold["is_fraud"])[0, 1]), 5))
json.dump(dict(members=auc, kept=names, blend=scores, best=best), open(f"{OUT}/summary.json", "w"), indent=1)
log(f"BEST BLEND = {best} | nested CV {scores[best]:.6f} | rows {len(sub)}")

#====

res = pd.Series(scores).sort_values(ascending=False).round(6)
display(res.to_frame("nested-CV AUC"))
wd = pd.Series({n: v for n, v in zip(names, w) if v > 0}).sort_values()
fig, ax = plt.subplots(figsize=(8, 0.35 * len(wd) + 1)); ax.barh(wd.index, wd.values, color="#d1495b")
ax.set_xlabel("weight"); ax.set_title("Hill-climbing weights (part of the final 'mix')"); plt.tight_layout(); plt.show()

#====

anc = [f for f in files if os.path.basename(f) == "submission.csv" and "96-key" in f.lower()]
if anc:
    a_rank = rankdata(pd.read_csv(anc[0]).set_index("id").loc[sub["id"], "is_fraud"].values) / NTE
    ours = sub["is_fraud"].values.copy()
    log("anchor:", anc[0], "| rank corr with our validated blend:", round(float(np.corrcoef(a_rank, ours)[0, 1]), 5))
    fused = sub.copy(); fused["is_fraud"] = rankdata(0.15 * ours + 0.85 * a_rank) / NTE
    fused.to_csv(f"{OUT}/submission_fusion.csv", index=False)
    log("submission_fusion.csv = 85/15 anchor fusion (public LB 0.95180); submission.csv = CV-validated blend (0.95170)")
else:
    log("anchor not attached: no fusion file written")