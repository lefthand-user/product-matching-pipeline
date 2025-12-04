"""
run_cleaning_v2.py
L0 Deterministic Cleaning

This script loads raw A/B data, applies the L0 cleaning rules
(normalized title, cleaned UPC, size parsing & normalization, raw brand/category retention),
and outputs intermediate L0 cleaned CSV files.

"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from pathlib import Path

from src.preprocess.l0_rules import L0Config, apply_l0_cleaning


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def run_for_side(
    side: str,
    raw_filename: str,
    output_filename: str,
) -> None:
    """
    Apply L0 cleaning to one side of the dataset.

    Parameters:
        side: "A" or "B"
        raw_filename: input file (under data/)
        output_filename: output cleaned file (under data/)
    """
    raw_path = DATA_DIR / raw_filename
    out_path = DATA_DIR / output_filename

    print(f"[L0] Loading raw data for side {side}: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    # --------------------------------------------------------------------
    # A/B Raw Columns include:
    #   bb_id
    #   name
    #   upc_raw
    #   brand_raw
    #   department   ← we treat this as category_raw
    #
    # config maps are IDENTICAL for A & B except 'side'.
    # --------------------------------------------------------------------

    config = L0Config(
        side=side,
        product_id_col="bb_id",    # keep bb_id as unique ID
        title_col="name",          # original product name/title
        upc_col="upc_raw",         # raw UPC
        brand_col="brand_raw",     # raw brand
        category_col="department", # raw category-like field
    )

    # ---------------------------------------------------------
    # Apply L0 cleaning
    # ---------------------------------------------------------
    df_l0 = apply_l0_cleaning(df_raw, config)

    print(f"[L0] Saving cleaned output for side {side}: {out_path}")
    df_l0.to_csv(out_path, index=False)
    print(f"[L0] Finished side {side}.\n")


def main() -> None:
    """
    Pipeline entry point.
    Convert raw A/B CSVs → L0 cleaned CSVs.
    """
    run_for_side(
        side="A",
        raw_filename="grocery_store_a_raw_data.csv",
        output_filename="grocery_store_a_clean_l0.csv",
    )

    run_for_side(
        side="B",
        raw_filename="grocery_store_b_raw_data.csv",
        output_filename="grocery_store_b_clean_l0.csv",
    )


if __name__ == "__main__":
    main()
