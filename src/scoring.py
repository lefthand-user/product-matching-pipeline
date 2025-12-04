# src/scoring.py
"""
This module:
    - Loads A/B clean_phase2 data
    - Loads candidates_topk (A × topK B candidates)
    - Adds feature columns (embedding/brand/category/size/size_strict)
    - Computes a weighted total score

Outputs:
    artifacts/candidates_scored.csv or .parquet
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
CANDIDATES_PATH = os.path.join(ARTIFACTS_DIR, "candidates_topk.csv")

CANDIDATES_SCORED_PARQUET = os.path.join(ARTIFACTS_DIR, "candidates_scored.parquet")
CANDIDATES_SCORED_CSV = os.path.join(ARTIFACTS_DIR, "candidates_scored.csv")


# ----------------------------- helpers --------------------------------


def _normalize_str_series(s: pd.Series) -> pd.Series:
    """
    Normalize string columns: lowercase + strip. NaN -> empty string.
    """
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def _choose_best_size_pair(
    a_ml: pd.Series,
    b_ml: pd.Series,
    a_g: pd.Series,
    b_g: pd.Series,
    a_count: pd.Series,
    b_count: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose the best pair of size values to compare for size similarity.

    Strategy:
        1. If both sides have ml, use ml
        2. Else if both sides have g, use g
        3. Else if both sides have count, use count
        4. Otherwise mark as NaN (we cannot compare size)

    Returns:
        (size_a, size_b) as numpy arrays of float64 (may contain NaN)
    """
    a_ml = a_ml.astype("float64")
    b_ml = b_ml.astype("float64")
    a_g = a_g.astype("float64")
    b_g = b_g.astype("float64")
    a_count = a_count.astype("float64")
    b_count = b_count.astype("float64")

    size_a = np.full_like(a_ml, np.nan, dtype="float64")
    size_b = np.full_like(b_ml, np.nan, dtype="float64")

    #  use ml if both sides have it
    mask_ml = (~a_ml.isna()) & (~b_ml.isna())
    size_a[mask_ml] = a_ml[mask_ml]
    size_b[mask_ml] = b_ml[mask_ml]

    #  for rows not yet assigned, try g
    mask_unset = np.isnan(size_a)
    mask_g = mask_unset & (~a_g.isna()) & (~b_g.isna())
    size_a[mask_g] = a_g[mask_g]
    size_b[mask_g] = b_g[mask_g]

    #  for still-unset rows, try count
    mask_unset = np.isnan(size_a)
    mask_count = mask_unset & (~a_count.isna()) & (~b_count.isna())
    size_a[mask_count] = a_count[mask_count]
    size_b[mask_count] = b_count[mask_count]

    return size_a, size_b


def _compute_size_similarity(size_a: np.ndarray, size_b: np.ndarray) -> np.ndarray:
    """
    Compute a simple similarity score based on size ratio.

    Idea:
        ratio = max(a, b) / min(a, b)

        if ratio <= 1.25 -> 1.0
        elif ratio <= 1.5 -> 0.7
        elif ratio <= 2.0 -> 0.4
        else -> 0.0

    NaN values -> 0.0.
    """
    size_feat = np.zeros_like(size_a, dtype="float32")

    valid = (~np.isnan(size_a)) & (~np.isnan(size_b)) & (size_a > 0) & (size_b > 0)
    if not valid.any():
        return size_feat

    a_valid = size_a[valid]
    b_valid = size_b[valid]

    ratio = np.maximum(a_valid, b_valid) / np.minimum(a_valid, b_valid)

    feat_valid = np.zeros_like(ratio, dtype="float32")
    feat_valid[ratio <= 1.25] = 1.0
    feat_valid[(ratio > 1.25) & (ratio <= 1.5)] = 0.7
    feat_valid[(ratio > 1.5) & (ratio <= 2.0)] = 0.4

    size_feat[valid] = feat_valid
    return size_feat


# --------------------------- main logic -------------------------------


def add_features_and_weighted_score() -> None:
    """
    Main entry point:

         Load A/B clean data + candidates_topk
         Merge A/B attributes into candidates
         Compute brand/category/size/size_strict features
         Compute weighted total score
         Save candidates_scored.*
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    #  Load data
    # ------------------------------------------------------------------
    print(f"[Scoring] Loading A data from: {A_PATH}")
    df_a = pd.read_csv(A_PATH)
    df_a = df_a.reset_index(drop=True)
    df_a["a_row_index"] = np.arange(len(df_a), dtype="int32")

    print(f"[Scoring] Loading B data from: {B_PATH}")
    df_b = pd.read_csv(B_PATH)
    df_b = df_b.reset_index(drop=True)
    df_b["b_row_index"] = np.arange(len(df_b), dtype="int32")

    print(f"[Scoring] Loading candidates from: {CANDIDATES_PATH}")
    candidates = pd.read_csv(CANDIDATES_PATH)
    print(f"[Scoring] Candidates shape: {candidates.shape}")

    # ------------------------------------------------------------------
    #  Select & rename fields from A/B
    # ------------------------------------------------------------------
    a_cols = [
        "a_row_index",
        "bb_id",
        "side",
        "customer_item_id",
        "title_normalized_l0",
        "brand_final",
        "category_lvl1_final",
        "size_ml_final",
        "size_g_final",
        "size_count_final",
    ]
    a_cols = [c for c in a_cols if c in df_a.columns]
    df_a_feat = df_a[a_cols].copy()
    rename_map_a = {c: f"a_{c}" for c in df_a_feat.columns if c != "a_row_index"}
    df_a_feat = df_a_feat.rename(columns=rename_map_a)

    b_cols = [
        "b_row_index",
        "bb_id",
        "side",
        "customer_item_id",
        "title_normalized_l0",
        "brand_final",
        "category_lvl1_final",
        "size_ml_final",
        "size_g_final",
        "size_count_final",
    ]
    b_cols = [c for c in b_cols if c in df_b.columns]
    df_b_feat = df_b[b_cols].copy()
    rename_map_b = {c: f"b_{c}" for c in df_b_feat.columns if c != "b_row_index"}
    df_b_feat = df_b_feat.rename(columns=rename_map_b)

    print("[Scoring] Merging A attributes ...")
    df = candidates.merge(df_a_feat, on="a_row_index", how="left")

    print("[Scoring] Merging B attributes ...")
    df = df.merge(df_b_feat, on="b_row_index", how="left")

    print(f"[Scoring] After merge, shape: {df.shape}")

    # ------------------------------------------------------------------
    #  feature_brand
    # ------------------------------------------------------------------
    if "a_brand_final" in df.columns and "b_brand_final" in df.columns:
        a_brand_norm = _normalize_str_series(df["a_brand_final"])
        b_brand_norm = _normalize_str_series(df["b_brand_final"])
        same_brand = (a_brand_norm != "") & (b_brand_norm != "") & (
            a_brand_norm == b_brand_norm
        )
        df["feature_brand"] = same_brand.astype("float32")
    else:
        df["feature_brand"] = 0.0

    # ------------------------------------------------------------------
    #  feature_category
    # ------------------------------------------------------------------
    if "a_category_lvl1_final" in df.columns and "b_category_lvl1_final" in df.columns:
        a_cat_norm = _normalize_str_series(df["a_category_lvl1_final"])
        b_cat_norm = _normalize_str_series(df["b_category_lvl1_final"])
        same_cat = (a_cat_norm != "") & (b_cat_norm != "") & (a_cat_norm == b_cat_norm)
        df["feature_category"] = same_cat.astype("float32")
    else:
        df["feature_category"] = 0.0

    # ------------------------------------------------------------------
    #  feature_size  + feature_size_strict
    # ------------------------------------------------------------------
    size_a_ml = df.get("a_size_ml_final", pd.Series(dtype="float64"))
    size_b_ml = df.get("b_size_ml_final", pd.Series(dtype="float64"))
    size_a_g = df.get("a_size_g_final", pd.Series(dtype="float64"))
    size_b_g = df.get("b_size_g_final", pd.Series(dtype="float64"))
    size_a_count = df.get("a_size_count_final", pd.Series(dtype="float64"))
    size_b_count = df.get("b_size_count_final", pd.Series(dtype="float64"))

    size_a_ml = size_a_ml.astype("float64")
    size_b_ml = size_b_ml.astype("float64")
    size_a_g = size_a_g.astype("float64")
    size_b_g = size_b_g.astype("float64")
    size_a_count = size_a_count.astype("float64")
    size_b_count = size_b_count.astype("float64")

    size_a, size_b = _choose_best_size_pair(
        size_a_ml,
        size_b_ml,
        size_a_g,
        size_b_g,
        size_a_count,
        size_b_count,
    )
    df["feature_size"] = _compute_size_similarity(size_a, size_b)

    size_strict = np.zeros_like(size_a, dtype="float32")
    valid = (~np.isnan(size_a)) & (~np.isnan(size_b)) & (size_a > 0) & (size_b > 0)
    ratio = np.zeros_like(size_a, dtype="float64")
    ratio[valid] = np.maximum(size_a[valid], size_b[valid]) / np.minimum(
        size_a[valid], size_b[valid]
    )
    size_strict[valid] = (ratio[valid] <= 1.05).astype("float32")
    df["feature_size_strict"] = size_strict

    # ------------------------------------------------------------------
    #  feature_embedding
    # ------------------------------------------------------------------
    if "similarity_embedding" not in df.columns:
        raise ValueError(
            "Column `similarity_embedding` is missing in candidates. "
            "Please check retrieve_topk.py output."
        )
    df["feature_embedding"] = np.clip(
        df["similarity_embedding"].astype("float32"), 0.0, 1.0
    )

    # ------------------------------------------------------------------
    #  Weighted total score (now includes size_strict)
    # ------------------------------------------------------------------
    w_emb = 0.55
    w_brand = 0.15
    w_cat = 0.15
    w_size = 0.05
    w_size_strict = 0.10

    df["score_total"] = (
        w_emb * df["feature_embedding"]
        + w_brand * df["feature_brand"]
        + w_cat * df["feature_category"]
        + w_size * df["feature_size"]
        + w_size_strict * df["feature_size_strict"]
    ).astype("float32")

    print("[Scoring] Feature summary:")
    print(
        df[
            [
                "feature_embedding",
                "feature_brand",
                "feature_category",
                "feature_size",
                "feature_size_strict",
                "score_total",
            ]
        ].describe()
    )

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    saved = False
    try:
        df.to_parquet(CANDIDATES_SCORED_PARQUET, index=False)
        print(
            f"[Scoring] Saved candidates_scored to Parquet: {CANDIDATES_SCORED_PARQUET}"
        )
        saved = True
    except Exception as e:
        print(f"[Scoring] Failed to save Parquet ({e}), falling back to CSV.")

    if not saved:
        df.to_csv(CANDIDATES_SCORED_CSV, index=False)
        print(f"[Scoring] Saved candidates_scored to CSV: {CANDIDATES_SCORED_CSV}")

    print("[Scoring] Done.")
