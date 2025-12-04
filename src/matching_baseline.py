from pathlib import Path
from typing import Set

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TOPK_FEATURE_INPUT_PATH = Path("artifacts/faiss_v0/topk_with_features.csv")
MATCHES_V0_OUTPUT_PATH = Path("artifacts/matches_v0.csv")


# ---------------------------------------------------------------------
# Rule-based final match selection (allowing "no match")
# ---------------------------------------------------------------------

def apply_rule_filter_and_select_best(
    topk_with_features_path: Path = TOPK_FEATURE_INPUT_PATH,
    matches_output_path: Path = MATCHES_V0_OUTPUT_PATH,
    min_sim_embedding_01: float = 0.45,
    min_weighted_score: float = 0.55,
) -> None:
    """
    Step 4: Apply rules on top-K candidate pairs and select at most
    one match for each A product.

    IMPORTANT:
        - Some A products will have NO match at all if none of their
          candidates meet the quality thresholds. This is intentional.

    Rules (baseline):
        For each candidate (A, B), keep it only if
            sim_embedding_01 >= min_sim_embedding_01
        AND weighted_score       >= min_weighted_score
        AND (category_soft_match >= 0.05 OR title_token_overlap >= 0.10)

    Then for each A (bb_id_A), pick the remaining candidate with the
    highest weighted_score. If no candidates remain for that A, it will
    simply NOT appear in matches_v0.csv.

    Output CSV columns:
        - bb_id_A
        - bb_id_B
        - weighted_score
        - sim_embedding
        - title_token_overlap
        - brand_match_score
        - size_ratio
        - category_soft_match
    """
    if not topk_with_features_path.exists():
        raise FileNotFoundError(
            f"Top-K features file not found at: {topk_with_features_path}"
        )

    df = pd.read_csv(topk_with_features_path)

    required_cols = [
        "bb_id_A",
        "bb_id_B",
        "weighted_score",
        "sim_embedding",
        "sim_embedding_01",
        "title_token_overlap",
        "brand_match_score",
        "size_ratio",
        "category_soft_match",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' is missing from {topk_with_features_path}"
            )

    # ---------------------------------------------------------
    # 1) Sort by A and weighted_score (descending)
    # ---------------------------------------------------------
    df = df.sort_values(
        ["bb_id_A", "weighted_score"], ascending=[True, False]
    ).reset_index(drop=True)

    all_a_ids: Set[int] = set(df["bb_id_A"].tolist())
    num_a_total = len(all_a_ids)

    # ---------------------------------------------------------
    # 2) Apply quality thresholds (this is the ONLY filter now)
    # ---------------------------------------------------------
    sim_ok = df["sim_embedding_01"] >= min_sim_embedding_01
    score_ok = df["weighted_score"] >= min_weighted_score
    cat_or_title_ok = (df["category_soft_match"] >= 0.05) | (
        df["title_token_overlap"] >= 0.10
    )

    keep_mask = sim_ok & score_ok & cat_or_title_ok
    df_kept = df[keep_mask].copy()

    print(f"[INFO] Number of A products: {num_a_total}")
    print(f"[INFO] Candidate rows before rules: {len(df)}")
    print(f"[INFO] Candidate rows after rules: {len(df_kept)}")

    if df_kept.empty:
        print("[WARN] No candidates survived the rules. "
              "You may need to lower min_sim_embedding_01 or min_weighted_score.")
        matches_out = df.iloc[0:0].copy()
    else:
        # -----------------------------------------------------
        # 3) For each A, pick the best candidate by weighted_score
        # -----------------------------------------------------
        best_per_a = (
            df_kept.sort_values(["bb_id_A", "weighted_score"], ascending=[True, False])
            .groupby("bb_id_A", as_index=False)
            .head(1)
        )

        matched_a_ids: Set[int] = set(best_per_a["bb_id_A"].tolist())
        num_matched = len(matched_a_ids)

        print(f"[INFO] Final matched A products: {num_matched} / {num_a_total}")

        # -----------------------------------------------------
        # 4) Prepare output
        # -----------------------------------------------------
        out_cols = [
            "bb_id_A",
            "bb_id_B",
            "weighted_score",
            "sim_embedding",
            "title_token_overlap",
            "brand_match_score",
            "size_ratio",
            "category_soft_match",
        ]
        matches_out = best_per_a[out_cols].copy()
        matches_out = matches_out.sort_values("weighted_score", ascending=False)

    matches_output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_out.to_csv(matches_output_path, index=False)

    print(f"[OK] Saved matches_v0 to {matches_output_path}")
    print(f"[OK] Total match rows: {len(matches_out)}")
