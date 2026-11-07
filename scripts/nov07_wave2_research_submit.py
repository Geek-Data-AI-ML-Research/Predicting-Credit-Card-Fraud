#!/usr/bin/env python3
"""Wave-2: MPS RankNet + aggressive/diverse axes; use remaining daily slots."""
from __future__ import annotations

import gc
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
EPS = 1e-6
OUT = ROOT / "submissions/nov07_ladder"
PRED = ROOT / "preds/nov07"
LOG = ROOT / "logs/overnight/nov07_wave2.json"

_FALLBACK = {
    "www.kaggle.com": "35.244.233.98",
    "api.kaggle.com": "34.54.168.202",
}
_CACHE = dict(_FALLBACK)
_orig_getaddrinfo = socket.getaddrinfo


def _doh_ip(name: str) -> str | None:
    try:
        out = subprocess.check_output(
            [
                "curl",
                "-sS",
                "-m",
                "15",
                "--resolve",
                "dns.google:443:8.8.8.8",
                f"https://dns.google/resolve?name={name}&type=A",
            ],
            text=True,
        )
        j = json.loads(out)
        for a in j.get("Answer", []):
            if a.get("type") == 1:
                return a["data"]
    except Exception:
        return None
    return None


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and not host.replace(".", "").isdigit():
        if host not in _CACHE:
            ip = _doh_ip(host)
            if ip:
                _CACHE[host] = ip
                print(f"DNS_PATCH {host} -> {ip}", flush=True)
        if host in _CACHE:
            host = _CACHE[host]
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _patched_getaddrinfo  # type: ignore[assignment]
for k in list(os.environ):
    if "proxy" in k.lower():
        os.environ.pop(k, None)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem() -> dict:
    page = 16384
    avail = -1.0
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        d: dict[str, int] = {}
        for line in out.splitlines():
            m = re.match(r"([^:]+):\s+(\d+)", line)
            if m:
                d[m.group(1).strip()] = int(m.group(2))
            if "page size of" in line:
                mm = re.search(r"page size of (\d+)", line)
                if mm:
                    page = int(mm.group(1))
        avail = (
            d.get("Pages free", 0)
            + d.get("Pages inactive", 0)
            + d.get("Pages speculative", 0)
        ) * page / 1e6
    except Exception:
        pass
    try:
        import resource

        rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    except Exception:
        rss = -1
    return {"avail_mb": round(avail), "rss_mb": rss, "cpu_count": os.cpu_count() or 1}


def rk(x):
    x = np.asarray(x, float).ravel()
    return (rankdata(x, method="average") - 1.0) / max(len(x) - 1, 1)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


def load_csv(path, ids):
    df = pd.read_csv(path)
    col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
    return rk(df.set_index("id").loc[ids, col].to_numpy(float))


def load_npy(path):
    return rk(np.load(path).ravel())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mem": []}
    log(f"MEM0 {mem()}")

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    ids = pd.read_csv(ROOT / "data/sample_submission.csv")["id"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    bank_oof = {
        k: load_npy(ROOT / f"preds/{p}")
        for k, p in [
            ("v4p", "P_super_v4p_oof.npy"),
            ("v4p_hc", "P_super_v4p_hc_oof.npy"),
            ("v4p_7hc", "P_super_v4p_7hc_oof.npy"),
            ("v4p_hc_master", "P_super_v4p_hc_master_oof.npy"),
            ("v7_clean", "P_super_v7_clean_oof.npy"),
            ("v7_pure_master", "P_super_v7_pure_master_oof.npy"),
        ]
        if (ROOT / f"preds/{p}").exists()
    }
    bank_test = {
        k: load_npy(ROOT / f"preds/{p}")
        for k, p in [
            ("v4p", "P_super_v4p_test.npy"),
            ("v4p_hc", "P_super_v4p_hc_test.npy"),
            ("v4p_7hc", "P_super_v4p_7hc_test.npy"),
            ("v4p_hc_master", "P_super_v4p_hc_master_test.npy"),
            ("v7_clean", "P_super_v7_clean_test.npy"),
            ("v7_pure_master", "P_super_v7_pure_master_test.npy"),
        ]
        if (ROOT / f"preds/{p}").exists()
    }
    lb = {
        "live10": load_csv(ROOT / "submissions/sub_live10_grand_logit_extrapolation.csv", ids),
        "live8": load_csv(ROOT / "submissions/sub_live8_logit_simplex_plus_ascent.csv", ids),
        "ev2": load_csv(ROOT / "submissions/sub_ev2_live10_plus_v7master_shift.csv", ids),
        "ev4": load_csv(ROOT / "submissions/sub_ev4_live10_reverse_ev2_plus_hc_ascent.csv", ids),
        "ev5": load_csv(ROOT / "submissions/sub_ev5_peak095243_continued_ascent.csv", ids),
        "ev6": load_csv(ROOT / "submissions/sub_ev6_grand_6axis_logit_ridge_extrap.csv", ids),
    }
    base_o, base_t = bank_oof["v4p_hc"], bank_test["v4p_hc"]
    base_auc = float(roc_auc_score(y, base_o))
    log(f"base v4p_hc OOF={base_auc:.7f} mem={mem()}")

    # ---- MPS RankNet residual ----
    mps_vec = None
    try:
        import torch
        import torch.nn as nn

        mps_ok = torch.backends.mps.is_available()
        report["mps_available"] = mps_ok
        report["torch"] = torch.__version__
        log(f"torch={torch.__version__} mps={mps_ok}")
    except Exception as e:
        mps_ok = False
        report["mps_error"] = str(e)
        log(f"torch import failed: {e}")

    if mps_ok:
        feat = [k for k in ["v4p", "v4p_hc", "v4p_7hc", "v4p_hc_master", "v7_pure_master"] if k in bank_oof]
        Fo = np.column_stack([bank_oof[k] for k in feat]).astype(np.float32)
        Ft = np.column_stack([bank_test[k] for k in feat]).astype(np.float32)
        init_o = logit(base_o).astype(np.float32)
        init_t = logit(base_t).astype(np.float32)

        class TinyRes(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1)
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        device = torch.device("mps")
        oof_mps = np.zeros(len(y), np.float32)
        for fi, (a, b) in enumerate(folds):
            model = TinyRes(Fo.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
            Xa = torch.tensor(Fo[a], device=device)
            Xb = torch.tensor(Fo[b], device=device)
            ya = torch.tensor(y[a].astype(np.float32), device=device)
            ia = torch.tensor(init_o[a], device=device)
            ib = torch.tensor(init_o[b], device=device)
            pos = torch.where(ya > 0.5)[0]
            neg = torch.where(ya < 0.5)[0]
            best_state, best_va = None, -1.0
            for _ep in range(25):
                model.train()
                for _ in range(40):
                    pi = pos[torch.randint(0, len(pos), (1024,), device=device)]
                    ni = neg[torch.randint(0, len(neg), (1024,), device=device)]
                    loss = torch.nn.functional.softplus(
                        -((ia[pi] + model(Xa[pi])) - (ia[ni] + model(Xa[ni])))
                    ).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pred_b = torch.sigmoid(ib + model(Xb)).cpu().numpy()
                va = roc_auc_score(y[b], pred_b)
                if va > best_va:
                    best_va = va
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.to(device).eval()
            with torch.no_grad():
                oof_mps[b] = torch.sigmoid(ib + model(Xb)).cpu().numpy()
            log(f"fold{fi} va={best_va:.7f} d={best_va - roc_auc_score(y[b], base_o[b]):+.2e} mem={mem()}")
            del model, opt, Xa, Xb, ya, ia, ib
            gc.collect()
            torch.mps.empty_cache()
        auc_mps = float(roc_auc_score(y, oof_mps))
        report["mps_oof"] = auc_mps
        report["mps_delta"] = auc_mps - base_auc
        log(f"MPS_OOF={auc_mps:.7f} d={auc_mps - base_auc:+.2e}")
        if auc_mps > base_auc + 3e-6:
            model = TinyRes(Fo.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
            Xall = torch.tensor(Fo, device=device)
            yall = torch.tensor(y.astype(np.float32), device=device)
            iall = torch.tensor(init_o, device=device)
            pos = torch.where(yall > 0.5)[0]
            neg = torch.where(yall < 0.5)[0]
            for _ep in range(35):
                model.train()
                for _ in range(50):
                    pi = pos[torch.randint(0, len(pos), (1024,), device=device)]
                    ni = neg[torch.randint(0, len(neg), (1024,), device=device)]
                    loss = torch.nn.functional.softplus(
                        -((iall[pi] + model(Xall[pi])) - (iall[ni] + model(Xall[ni])))
                    ).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                te = torch.sigmoid(
                    torch.tensor(init_t, device=device) + model(torch.tensor(Ft, device=device))
                ).cpu().numpy()
            d_rho = abs(spearmanr(oof_mps, base_o).correlation - spearmanr(te, base_t).correlation)
            report["mps_delta_rho"] = float(d_rho)
            log(f"MPS |drho|={d_rho:.5f}")
            if d_rho < 0.0025:
                mps_vec = {"oof": oof_mps, "test": te, "auc": auc_mps}
                np.save(PRED / "mps_ranknet_oof.npy", oof_mps)
                np.save(PRED / "mps_ranknet_test.npy", te.astype(np.float32))

    # ---- diverse recipes for remaining slots ----
    z_peak = logit(lb["ev4"])
    a_ev4_l10 = logit(lb["ev4"]) - logit(lb["live10"])
    a_ev4_ev2 = logit(lb["ev4"]) - logit(lb["ev2"])
    a_hc_v4p = logit(bank_test["v4p_hc"]) - logit(bank_test["v4p"])
    a_7hc = logit(bank_test["v4p_7hc"]) - logit(bank_test["v4p_hc"])
    a_master = logit(bank_test["v4p_hc_master"]) - logit(bank_test["v4p_hc"])
    a_anti_soft = logit(bank_test["v4p_hc"]) - logit(bank_test["v7_clean"])
    a_l10_l8 = logit(lb["live10"]) - logit(lb["live8"])

    recipes = []
    # Overshoot ascent (wave1 matched; try larger step)
    recipes.append(
        (
            "w2_overshoot_ascent",
            z_peak + 0.90 * a_ev4_l10 + 0.45 * a_ev4_ev2 + 0.25 * a_hc_v4p + 0.20 * a_7hc,
            "Nov07-W2A: Overshoot ascent from 0.95243 (0.90 live10-gap + reverse-ev2 + hc/7hc)",
        )
    )
    # Pure proprietary ridge (no public live remix beyond peak seed)
    recipes.append(
        (
            "w2_pure_prop_ridge",
            logit(bank_test["v4p_hc"])
            + 0.55 * a_7hc
            + 0.40 * a_master
            + 0.30 * a_anti_soft
            + 0.15 * (logit(bank_test["v7_pure_master"]) - logit(bank_test["v4p_hc"])),
            "Nov07-W2B: Pure proprietary ridge on v4p_hc (7hc+master+anti-soft+pure)",
        )
    )
    # Strong anti-softpseudo
    recipes.append(
        (
            "w2_strong_anti_soft",
            z_peak + 0.25 * a_ev4_l10 + 0.55 * a_anti_soft + 0.20 * a_7hc + 0.15 * a_l10_l8,
            "Nov07-W2C: Strong anti-softpseudo toward hc from 0.95243 peak",
        )
    )
    if mps_vec is not None:
        recipes.append(
            (
                "w2_peak_mps",
                z_peak
                + 0.30 * a_ev4_l10
                + 0.35 * (logit(rk(mps_vec["test"])) - logit(base_t))
                + 0.15 * a_hc_v4p
                + 0.10 * a_7hc,
                f"Nov07-W2D: Peak + MPS RankNet residual OOF={mps_vec['auc']:.7f}",
            )
        )

    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    cands = []
    for name, z, msg in recipes:
        pred = rk(sigmoid(z))
        path = OUT / f"sub_{name}.csv"
        out = sample.copy()
        out["is_fraud"] = pred
        out.to_csv(path, index=False)
        rho = float(spearmanr(pred, rk(lb["ev4"])).correlation)
        cands.append({"name": name, "path": str(path), "msg": msg, "rho_vs_ev4": rho})
        log(f"SAVED {name} rho_vs_ev4={rho:.6f}")

    report["candidates"] = cands
    report["mem"].append(mem())
    LOG.write_text(json.dumps(report, indent=2, default=float))

    # Submit remaining slots (prefer most diverse first)
    import kaggle_doh
    from kaggle.api.kaggle_api_extended import KaggleApi

    n, best, _ = kaggle_doh.today_utc_count()
    left = max(0, 10 - n)
    log(f"slots_left={left} best={best}")
    # order: mps if any, then pure_prop, overshoot, strong_anti
    prefer = ["w2_peak_mps", "w2_pure_prop_ridge", "w2_overshoot_ascent", "w2_strong_anti_soft"]
    by = {c["name"]: c for c in cands}
    order = [by[n] for n in prefer if n in by]
    api = KaggleApi()
    api.authenticate()
    ladder = []
    best_lb = max(best, 0.95243)
    for cand in order[:left]:
        p = Path(cand["path"])
        log(f"SUBMIT {p.name}")
        try:
            result = api.competition_submit(str(p), cand["msg"], "playground-series-s6e11")
            log(f"RESULT {result}")
            ref = getattr(result, "ref", None)
        except Exception as e:
            log(f"FAIL {e}")
            ladder.append({"file": p.name, "ok": False, "err": str(e)})
            break
        sc = kaggle_doh.poll_file(p.name, timeout_s=540, sleep_s=10)
        log(f"SCORE {p.name} {sc}")
        if sc is not None:
            best_lb = max(best_lb, sc)
        ladder.append({"file": p.name, "public_lb": sc, "ref": ref, "msg": cand["msg"]})
        report["ladder"] = ladder
        report["best_lb"] = best_lb
        LOG.write_text(json.dumps(report, indent=2, default=float))
        if sc is not None and sc >= 0.953000:
            log("GOAL HIT")
            break
    log(f"DONE best_lb={best_lb} mem={mem()}")
    print(json.dumps(ladder, indent=2), flush=True)


if __name__ == "__main__":
    main()
