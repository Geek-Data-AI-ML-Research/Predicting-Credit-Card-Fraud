#!/usr/bin/env python3
"""Malyshev V10 rank-displacement operator (exact).

  midrank(x) = (rankdata(x) - 0.5) / n
  out = midrank( before + scale * (after - before) )

Default scale=3.0 as published for 0.95197 → 0.95200.
No OOF/CV claimed for this operator alone.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def midrank(v: np.ndarray) -> np.ndarray:
    return (rankdata(v) - 0.5) / len(v)

def displace(before: np.ndarray, after: np.ndarray, scale: float = 3.0) -> np.ndarray:
    b, a = midrank(before), midrank(after)
    return midrank(b + scale * (a - b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--id-from", default=None, help="CSV providing id column (default: --before)")
    args = ap.parse_args()
    b = pd.read_csv(args.before)
    a = pd.read_csv(args.after)
    ids = pd.read_csv(args.id_from or args.before)
    pc = lambda df: df[[c for c in df.columns if c.lower() != "id"][0]].to_numpy()
    out = displace(pc(b), pc(a), args.scale)
    sub = pd.DataFrame({"id": ids["id"], "is_fraud": out})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} scale={args.scale} n={len(sub)}")

if __name__ == "__main__":
    main()
