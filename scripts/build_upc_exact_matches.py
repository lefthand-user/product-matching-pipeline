# scripts/build_upc_exact_matches.py
"""
Build Stage 0: UPC exact matches for national brands.

Inputs:
    data/grocery_store_a_clean_phase2.csv
    data/grocery_store_b_clean_phase2.csv

Outputs:
    artifacts/upc_exact_matches.csv

Usage:
    (.venv) python scripts/build_upc_exact_matches.py
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"

A_PATH = os.path.join(DATA_DIR, "grocery_store_a_clean_phase2.csv")
B_PATH = os.path.join(DATA_DIR, "grocery_store_b_clean_phase2.csv")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "upc_exact_matches.csv")


def _choose_main_size(
    ml: pd.Series, g: pd.Series, count: pd.Series
) -> np.ndarray:
    """
    Pick a single numeric size value as "main size" for tie-break:
        1) Prefer ml if available
        2) Else prefer g
        3) Else use count
        4) Else NaN
    """
    ml = ml.astype("float64")
    g = g.astype("float64")
    count = count.astype("float64")

    main = np.full_like(ml, np.nan, dtype="float64")

    mask_ml = ~ml.isna()
    main[mask_ml] = ml[mask_ml]

    mask_unset = np.isnan(main)
    mask_g = mask_unset & ~g.isna()
    main[mask_g] = g[mask_g]

    mask_unset = np.isnan(main)
    mask_c = mask_unset & ~count.isna()
    main[mask_c] = count[mask_c]

    return main


def build_upc_exact() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"[UPC] Loading A clean data from: {A_PATH}")
    df_a = pd.read_csv(A_PATH)

    print(f"[UPC] Loading B clean data from: {B_PATH}")
    df_b = pd.read_csv(B_PATH)

    # ------------------ basic field checks ------------------
    for col in ["bb_id", "upc_clean_l0", "is_private_label"]:
        if col not in df_a.columns:
            raise ValueError(f"A is missing required column: {col}")

    for col in ["bb_id", "upc_clean", "is_private_label"]:
        if col not in df_b.columns:
            raise ValueError(f"B is missing required column: {col}")

    # ------------------ filter: has UPC & non private label ------------------
    a_use = df_a[
        df_a["upc_clean_l0"].notna()
        & (df_a["upc_clean_l0"] != "")
        & (df_a["is_private_label"] == False)
    ].copy()

    b_use = df_b[
        df_b["upc_clean"].notna()
        & (df_b["upc_clean"] != "")
        & (df_b["is_private_label"] == False)
    ].copy()

    print(f"[UPC] A candidates (with UPC + national brand): {len(a_use)}")
    print(f"[UPC] B candidates (with UPC + national brand): {len(b_use)}")

    # Prepare size main value and normalized titles for tie-break
    for df, side in [(a_use, "A"), (b_use, "B")]:
        for col in ["size_ml_final", "size_g_final", "size_count_final"]:
            if col not in df.columns:
                df[col] = np.nan

        df[f"size_main_{side}"] = _choose_main_size(
            df["size_ml_final"], df["size_g_final"], df["size_count_final"]
        )

        title_col = "title_normalized_l0"
        if title_col not in df.columns:
            # fallback: use name if no normalized title
            df[title_col] = df.get("name", "").astype(str)

        df[f"title_norm_{side}"] = (
            df[title_col].fillna("").astype(str).str.lower().str.strip()
        )

    # ------------------ join on UPC ------------------
    print("[UPC] Joining A and B on UPC ...")
    df_join = a_use.merge(
        b_use,
        left_on="upc_clean_l0",
        right_on="upc_clean",
        how="inner",
        suffixes=("_A", "_B"),
    )

    print(f"[UPC] Raw UPC join rows: {len(df_join)}")
    if len(df_join) == 0:
        print("[UPC] No UPC exact matches found. Writing empty file.")
        pd.DataFrame(
            columns=[
                "a_bb_id",
                "b_bb_id",
                "upc_clean",
                "match_type",
            ]
        ).to_csv(OUTPUT_PATH, index=False)
        return

    # ------------------ tie-break per A: choose best B ------------------
    def _compute_tie_break(row):
        # size_penalty: |size_main_A - size_main_B|, smaller is better
        size_a = row.get("size_main_A", np.nan)
        size_b = row.get("size_main_B", np.nan)
        if pd.isna(size_a) or pd.isna(size_b):
            size_penalty = 9999.0  # big penalty when no size info
        else:
            size_penalty = float(abs(size_a - size_b))

        # title token overlap: more overlap is better
        t_a = str(row.get("title_norm_A", "")).split()
        t_b = str(row.get("title_norm_B", "")).split()
        overlap = len(set(t_a) & set(t_b))

        return pd.Series(
            {
                "size_penalty": size_penalty,
                "title_overlap": overlap,
            }
        )

    df_join[["size_penalty", "title_overlap"]] = df_join.apply(
        _compute_tie_break, axis=1
    )

    # For each A bb_id, pick the best B:
    #   1) smallest size_penalty
    #   2) largest title_overlap
    df_join_sorted = df_join.sort_values(
        by=["bb_id_A", "size_penalty", "title_overlap"],
        ascending=[True, True, False],
    )

    best_idx = df_join_sorted.groupby("bb_id_A").head(1).index
    df_best = df_join_sorted.loc[best_idx].copy().reset_index(drop=True)

    print(f"[UPC] After tie-break, exact pairs: {len(df_best)}")

    # ------------------ build output ------------------
    out = pd.DataFrame(
        {
            "a_bb_id": df_best["bb_id_A"],
            "b_bb_id": df_best["bb_id_B"],
            "upc_clean": df_best["upc_clean"],
            "match_type": "upc_exact",
        }
    )

    print(f"[UPC] Saving UPC exact matches to: {OUTPUT_PATH}")
    out.to_csv(OUTPUT_PATH, index=False)
    print("[UPC] Done.")


if __name__ == "__main__":
    build_upc_exact()
