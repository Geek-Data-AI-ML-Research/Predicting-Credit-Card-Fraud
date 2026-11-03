#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
ALEX = ROOT / "preds" / "public_alex"
SUBS = ROOT / "submissions"
DATA = ROOT / "data"

def load_y():
    y=[]
    with open(DATA/"train.csv") as f:
        for row in csv.DictReader(f):
            y.append(int(float(row["is_fraud"])))
    return np.asarray(y, np.int8)

def read_sub(path):
    with open(path) as f:
        return np.array([float(row["is_fraud"]) for row in csv.DictReader(f)], np.float64)

def rn(x):
    x=np.asarray(x, np.float64)
    order=np.argsort(x, kind="mergesort")
    ranks=np.empty(len(x), np.float64)
    ranks[order]=np.arange(1, len(x)+1, dtype=np.float64)
    return (ranks-0.5)/len(x)

def auc_exact(y, s):
    y=np.asarray(y); s=np.asarray(s, np.float64)
    order=np.argsort(s, kind="mergesort")
    ranks=np.empty(len(s), np.float64)
    ranks[order]=np.arange(1, len(s)+1, dtype=np.float64)
    pos=y==1
    n_pos=int(pos.sum()); n_neg=len(y)-n_pos
    return (ranks[pos].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

print("loading y…", flush=True)
y=load_y()
print("y", y.shape, float(y.mean()), flush=True)

members=[]
for op in sorted(ALEX.glob("oof_*.npy")):
    if op.name.startswith("._"): continue
    tag=op.name[4:-4]
    tp=ALEX/f"test_{tag}.npy"
    if not tp.exists(): continue
    o=np.load(op); t=np.load(tp)
    if len(o)!=len(y) or len(t)!=299984: continue
    if not np.isfinite(o).all(): continue
    a=auc_exact(y,o)
    members.append((tag, a, o.astype(np.float64), t.astype(np.float64)))
    print(f"  {tag[:20]} auc={a:.6f}", flush=True)

bank=np.load(ROOT/"preds/stage9/full_bank_unpruned.npz", allow_pickle=True)
names=[str(n) for n in bank["names"]]
O=bank["O"].astype(np.float64); T=bank["T"].astype(np.float64)
for i,n in enumerate(names):
    if "diglead_l3_bag" in n:
        a=auc_exact(y,O[:,i])
        members.append((f"bank:{n}", a, O[:,i], T[:,i]))
        print(f"  bank {n} auc={a:.6f}", flush=True)

members.sort(key=lambda x: -x[1])
print(f"loaded {len(members)} best={members[0][1]:.6f}", flush=True)

m=len(members)
R=np.column_stack([rn(members[j][2]) for j in range(m)])
Rt=np.column_stack([rn(members[j][3]) for j in range(m)])
w=np.zeros(m); w[0]=1.0
cur=auc_exact(y, R@w)
print(f"start auc={cur:.6f}", flush=True)
for it in range(80):
    best_sc, best_j = cur, None
    for j in range(m):
        w2=w.copy(); w2[j]+=1; ww=w2/w2.sum()
        sc=auc_exact(y, R@ww)
        if sc>best_sc+1e-8:
            best_sc, best_j = sc, j
    if best_j is None:
        break
    w[best_j]+=1; cur=best_sc
    if it%5==0:
        nz=[(members[j][0][:14], float(w[j])) for j in range(m) if w[j]>0]
        print(f"  iter{it} auc={cur:.6f} {nz}", flush=True)
ww=w/w.sum()
oof_blend=R@ww
te_blend=Rt@ww
print(f"hill OOF={auc_exact(y,oof_blend):.6f}", flush=True)

H=read_sub(SUBS/"hillclimb_mix_0.95111.csv")
ids=[row["id"] for row in csv.DictReader(open(DATA/"sample_submission.csv"))]
rh, ra = rn(H), rn(te_blend)

def write_csv(name, pred):
    path=SUBS/name
    with open(path,"w",newline="") as f:
        wri=csv.writer(f); wri.writerow(["id","is_fraud"])
        for i,v in zip(ids, pred):
            wri.writerow([i, f"{float(v):.10f}"])
    print("WROTE", path, flush=True)
    return path

chosen=[]
for a in [0.97,0.95,0.93,0.90,0.85]:
    br=np.clip(a*rh+(1-a)*ra,0,1)
    pred=np.quantile(H, br)
    sp=float(np.corrcoef(rn(pred), rh)[0,1])
    print(f"a={a:.2f} sp={sp:.6f}", flush=True)
    chosen.append((a,pred,sp))

# primary: a=0.93 — meaningful alex inject, still very close to hill
write_csv("sub_lb_lift_hill93_alex07.csv", chosen[2][1])
write_csv("sub_lb_lift_hill97_alex03.csv", chosen[0][1])
# also pure alex ranked mapped to hill quantiles (max diversity under hill marginal)
write_csv("sub_lb_lift_alex_qmap_hill.csv", np.quantile(H, ra))

json.dump({
  "alex_hill_oof": float(auc_exact(y,oof_blend)),
  "weights": {members[j][0]: float(ww[j]) for j in range(m) if ww[j]>1e-12},
}, open(ROOT/"logs/lb_lift_alex_hill_meta.json","w"), indent=2)
print("DONE", flush=True)
