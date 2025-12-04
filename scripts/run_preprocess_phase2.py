# scripts/run_preprocess_phase2.py
"""
Build `text_for_embedding` for A/B clean_final data and save new CSVs.

Input:
    data/grocery_store_a_clean_final.csv
    data/grocery_store_b_clean_final.csv

Output:
    data/grocery_store_a_clean_phase2.csv
    data/grocery_store_b_clean_phase2.csv

Run:
    python scripts/run_preprocess_phase2.py
"""

import os
import math
import pandas as pd


DATA_DIR = "data"
A_INPUT = os.path.join(DATA_DIR, "grocery_store_a_clean_final.csv")
B_INPUT = os.path.join(DATA_DIR, "grocery_store_b_clean_final.csv")

A_OUTPUT = os.path.join(DATA_DIR, "grocery_store_a_clean_phase2.csv")
B_OUTPUT = os.path.join(DATA_DIR, "grocery_store_b_clean_phase2.csv")


def _is_missing(value) -> bool:
    """Return True if the value should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    if text == "":
        return True
    # GPT cleaning used these special tokens
    if text.upper() in ("UNKNOWN", "NONE"):
        return True
    return False


def build_size_display(row) -> str:
    """
    Build a human-readable size string from *_final columns.

    Examples:
        "40 count"
        "473 ml"
        "12 oz, 355 ml"
    We keep it simple: at most one volume + one weight + one count.
    """
    parts = []

    size_count = row.get("size_count_final")
    size_ml = row.get("size_ml_final")
    size_g = row.get("size_g_final")

    # count
    if not _is_missing(size_count):
        try:
            val = float(size_count)
            if abs(val - round(val)) < 1e-6:
                val_str = str(int(round(val)))
            else:
                val_str = str(val)
            parts.append(f"{val_str} count")
        except Exception:
            # fall back to raw text
            parts.append(f"{size_count} count")

    # ml
    if not _is_missing(size_ml):
        try:
            val = float(size_ml)
            if abs(val - round(val)) < 1e-6:
                val_str = str(int(round(val)))
            else:
                val_str = str(val)
            parts.append(f"{val_str} ml")
        except Exception:
            parts.append(f"{size_ml} ml")

    # g
    if not _is_missing(size_g):
        try:
            val = float(size_g)
            if abs(val - round(val)) < 1e-6:
                val_str = str(int(round(val)))
            else:
                val_str = str(val)
            parts.append(f"{val_str} g")
        except Exception:
            parts.append(f"{size_g} g")

    return ", ".join(parts)


def build_text_for_embedding(row) -> str:
    """
    Build the `text_for_embedding` string for a single row.

    We always start from title_normalized_l0,
    then append brand / core / flavor / size / category if available.
    """
    # Base title
    base_title = str(row.get("title_normalized_l0", "")).strip()
    if base_title == "":
        # fall back to title_raw or name
        base_title = str(row.get("title_raw", "") or row.get("name", "")).strip()

    pieces = [base_title]

    brand = row.get("brand_final")
    if not _is_missing(brand):
        pieces.append(f"brand: {brand}")

    core = row.get("product_core_final")
    if not _is_missing(core):
        pieces.append(f"core: {core}")

    flavor = row.get("flavor_final")
    if not _is_missing(flavor):
        pieces.append(f"flavor: {flavor}")

    size_display = build_size_display(row)
    if size_display:
        pieces.append(f"size: {size_display}")

    cat = row.get("category_lvl1_final")
    if not _is_missing(cat):
        pieces.append(f"category: {cat}")

    # Join with ", " 
    return ", ".join(pieces)


def process_one_file(input_path: str, output_path: str) -> None:
    print(f"[Phase2-Preprocess] Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Sanity check: make sure key columns exist
    required_cols = [
        "title_normalized_l0",
        "brand_final",
        "product_core_final",
        "flavor_final",
        "category_lvl1_final",
        "size_ml_final",
        "size_g_final",
        "size_count_final",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    print(f"[Phase2-Preprocess] Building text_for_embedding for {len(df)} rows...")
    df["text_for_embedding"] = df.apply(build_text_for_embedding, axis=1)

    print(f"[Phase2-Preprocess] Example rows:")
    print(df[["bb_id", "side", "text_for_embedding"]].head(5))

    print(f"[Phase2-Preprocess] Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("[Phase2-Preprocess] Done.\n")


def main():
    # Process A side
    process_one_file(A_INPUT, A_OUTPUT)
    # Process B side
    process_one_file(B_INPUT, B_OUTPUT)


if __name__ == "__main__":
    main()
