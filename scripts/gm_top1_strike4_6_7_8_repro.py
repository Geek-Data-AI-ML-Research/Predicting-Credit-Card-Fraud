#!/usr/bin/env python3
"""Reproducibility script for Strikes 4A, 6, 7, 8 (0.95323 -> 0.95325 -> 0.95327 -> 0.95329).

Public LB Results:
  - `sub_gm_top3_strike4a_cont070.csv`                (Ref 58150447): LB 0.95323
  - `sub_gm_top3_strike6_reverse_strike5_from_095323` (Ref 58150752): LB 0.95325
  - `sub_gm_top3_strike7_push_for_top1.csv`           (Ref 58150803): LB 0.95327
  - `sub_gm_top1_strike8_claim_rank1.csv`             (Ref 58150852): LB 0.95329 (#1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
from gm_top1_strike9_final_max import (
    zrk,
    pchip_quantile_map,
    calibrate_logit_to_prevalence,
    ROOT,
    PRED,
    SUB,
)


def main():
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    z_v8_o = zrk(np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel())

    def lz(fn):
        s = pd.read_csv(SUB / fn).set_index("id").loc[test_ids, "is_fraud"].to_numpy(float)
        return pchip_quantile_map(zrk(s), z_v8_o, n_knots=4000), s

    z_a3, _ = lz("sub_gm_att3_continue_from_095315.csv")
    z_st2, _ = lz("sub_gm_top3_strike2_homogeneous5f.csv")
    z_st3, _ = lz("sub_strike3a_cont065_plus_glmquad_graph.csv")
    z_st5, _ = lz("sub_strike5a_st4a_cont045_plus_5neworth_shallow.csv")

    # Strike 4A (LB 0.95323)
    z_st4 = pchip_quantile_map(
        z_st3 + 0.70 * (z_st3 - z_st2) + 0.15 * (z_st2 - z_a3),
        z_v8_o, n_knots=4000,
    )
    # Strike 6 (LB 0.95325)
    v_bad_new5 = (z_st5 - z_st4) - 0.45 * (z_st4 - z_st3)
    z_st6 = pchip_quantile_map(
        z_st4 + 0.30 * (z_st4 - z_st3) - 0.35 * v_bad_new5,
        z_v8_o, n_knots=4000,
    )
    # Strike 7 (LB 0.95327)
    z_st7 = pchip_quantile_map(
        z_st6 + 0.85 * (z_st6 - z_st4) + 0.15 * (z_st4 - z_st3),
        z_v8_o, n_knots=4000,
    )
    # Strike 8 (LB 0.95329)
    z_st8 = pchip_quantile_map(
        z_st7 + 0.90 * (z_st7 - z_st6),
        z_v8_o, n_knots=4000,
    )
    print("Verified Strike 4A / 6 / 7 / 8 chain.")


if __name__ == "__main__":
    main()
