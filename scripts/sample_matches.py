# scripts/sample_matches.py
"""
Quality Check
Randomly sample N matches from matches_v0.csv
and save for manual inspection.

Run:
    python scripts/sample_matches.py --n 50
"""

from __future__ import annotations
import os
import sys
import pandas as pd
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MATCHES_PATH = os.path.join("artifacts", "matches_v0.csv")
SAMPLE_OUTPUT = os.path.join("artifacts", "sample_matches_v0.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of samples")
    args = parser.parse_args()
    n = args.n

    print(f"[QC] Loading: {MATCHES_PATH}")
    df = pd.read_csv(MATCHES_PATH)
    print(f"[QC] Total matches: {len(df)}")

    if len(df) == 0:
        print("[QC] No matches to sample. Exiting.")
        return

    # random sample
    n = min(n, len(df))
    df_s = df.sample(n=n, random_state=42).copy()

    print(f"[QC] Sampling {n} rows for manual review...")

    # Reorder columns for readability
    col_order = [
        # A-side
        "a_bb_id", "a_customer_item_id", "a_title_normalized_l0",
        "a_brand_final", "a_category_lvl1_final", "a_size_ml_final",
        "a_size_g_final", "a_size_count_final",

        # B-side
        "b_bb_id", "b_customer_item_id", "b_title_normalized_l0",
        "b_brand_final", "b_category_lvl1_final", "b_size_ml_final",
        "b_size_g_final", "b_size_count_final",

        # scores
        "feature_embedding", "feature_brand", "feature_category",
        "feature_size", "score_total"
    ]

    col_order = [c for c in col_order if c in df_s.columns]
    df_s = df_s[col_order]

    print(f"[QC] Saving sample to: {SAMPLE_OUTPUT}")
    df_s.to_csv(SAMPLE_OUTPUT, index=False)

    print("[QC] Done.")
    print("You can open artifacts/sample_matches_v0.csv to inspect quality.")


if __name__ == "__main__":
    main()
