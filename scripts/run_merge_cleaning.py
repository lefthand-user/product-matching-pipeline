"""
Entry script for L2 final cleaning / merge.

This script:
- Loads the L1 tables for side A and B:
    data/grocery_store_a_clean_l1.csv
    data/grocery_store_b_clean_l1.csv
- Applies L2 fusion logic to build *_final columns.
- Writes:
    data/grocery_store_a_clean_final.csv
    data/grocery_store_b_clean_final.csv
"""

from __future__ import annotations

from pathlib import Path

from src.preprocess.merge_cleaning import apply_l2_cleaning_for_side


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    # ----- Side A -----
    apply_l2_cleaning_for_side(
        side="A",
        l1_csv_path=DATA_DIR / "grocery_store_a_clean_l1.csv",
        output_csv_path=DATA_DIR / "grocery_store_a_clean_final.csv",
    )

    # ----- Side B -----
    apply_l2_cleaning_for_side(
        side="B",
        l1_csv_path=DATA_DIR / "grocery_store_b_clean_l1.csv",
        output_csv_path=DATA_DIR / "grocery_store_b_clean_final.csv",
    )


if __name__ == "__main__":
    main()
