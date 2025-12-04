# scripts/build_matches_v0.py
"""
Step 5 of Phase 2 (strict + size-aware):
Use scored candidates to build matches_v0.csv, keeping:
    - high-score, high-embedding, same-category matches
    - plus: same-category pairs whose sizes differ by ≤ 5% (size_strict)

Inputs:
    artifacts/candidates_scored.csv

Output:
    artifacts/matches_v0.csv

Run:
    python scripts/build_matches_v0.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACTS_DIR = "artifacts"
CANDIDATES_SCORED_CSV = os.path.join(ARTIFACTS_DIR, "candidates_scored.csv")
MATCHES_V0_CSV = os.path.join(ARTIFACTS_DIR, "matches_v0.csv")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"[Matches v0] Loading scored candidates from: {CANDIDATES_SCORED_CSV}")
    df = pd.read_csv(CANDIDATES_SCORED_CSV)
    print(f"[Matches v0] Input shape: {df.shape}")

    # ------------------------------------------------------------------
    # For each A product, keep only the best B candidate
    # ------------------------------------------------------------------
    df_sorted = df.sort_values(
        by=["a_row_index", "score_total"], ascending=[True, False]
    ).reset_index(drop=True)

    best_idx = df_sorted.groupby("a_row_index")["score_total"].idxmax()
    df_top1 = df_sorted.loc[best_idx].copy().reset_index(drop=True)
    print(f"[Matches v0] After top-1 per A, shape: {df_top1.shape}")

    # ------------------------------------------------------------------
    # STRICT rule-based filtering + size_strict 
    # ------------------------------------------------------------------
    score = df_top1["score_total"].fillna(0.0).astype("float32")
    emb = df_top1["feature_embedding"].fillna(0.0).astype("float32")
    cat = df_top1["feature_category"].fillna(0.0).astype("float32")
    brand = df_top1["feature_brand"].fillna(0.0).astype("float32")
    size_f = df_top1["feature_size"].fillna(0.0).astype("float32")
    size_strict = df_top1.get(
        "feature_size_strict", pd.Series(0.0, index=df_top1.index)
    ).astype("float32")

    # threthhold
    MIN_SCORE_MAIN = 0.70
    MIN_EMB_MAIN = 0.65

    # main mask
    mask_main = (
        (score >= MIN_SCORE_MAIN) &
        (emb >= MIN_EMB_MAIN) &
        (cat >= 0.5)
    )

    # size mask : size（≤5%）
    MIN_SCORE_SIZE = MIN_SCORE_MAIN - 0.05  # 0.65
    MIN_EMB_SIZE = MIN_EMB_MAIN - 0.05      # 0.60

    mask_size_strict = (
        (cat >= 0.5) &
        (size_strict >= 0.5) &      # feature_size_strict == 1
        (score >= MIN_SCORE_SIZE) &
        (emb >= MIN_EMB_SIZE)
    )

    mask_keep = mask_main | mask_size_strict

    df_kept = df_top1.loc[mask_keep].copy().reset_index(drop=True)

    print(
        f"[Matches v0] After STRICT+SIZE filter, kept {len(df_kept)} matches."
    )

    # ------------------------------------------------------------------
    # Select output columns for matches_v0
    # ------------------------------------------------------------------
    out_cols = []

    # A-side identifiers
    for col in [
        "a_row_index",
        "a_bb_id",
        "a_customer_item_id",
        "a_title_normalized_l0",
        "a_brand_final",
        "a_category_lvl1_final",
        "a_size_ml_final",
        "a_size_g_final",
        "a_size_count_final",
    ]:
        if col in df_kept.columns:
            out_cols.append(col)

    # B-side identifiers
    for col in [
        "b_row_index",
        "b_bb_id",
        "b_customer_item_id",
        "b_title_normalized_l0",
        "b_brand_final",
        "b_category_lvl1_final",
        "b_size_ml_final",
        "b_size_g_final",
        "b_size_count_final",
    ]:
        if col in df_kept.columns:
            out_cols.append(col)

    # Scoring features
    for col in [
        "feature_embedding",
        "feature_brand",
        "feature_category",
        "feature_size",
        "feature_size_strict",
        "score_total",
    ]:
        if col in df_kept.columns:
            out_cols.append(col)

    if not out_cols:
        out_cols = list(df_kept.columns)

    df_out = df_kept[out_cols].copy()

    print(f"[Matches v0] Output shape: {df_out.shape}")
    print(f"[Matches v0] Saving to: {MATCHES_V0_CSV}")
    df_out.to_csv(MATCHES_V0_CSV, index=False)

    print("[Matches v0] Done.")


if __name__ == "__main__":
    main()
