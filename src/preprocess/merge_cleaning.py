"""
L2 final cleaning / merge layer.

This module takes the L1 output tables (L0 + GPT fields) and
produces a final, canonical clean version of each row with
*_final columns.

Key ideas:
- Prefer GPT fields when they are confident and not UNKNOWN/NONE.
- Fall back to L0 raw / heuristic mapping when GPT is weak or missing.
- Keep things deterministic and fast.

Inputs:
    data/grocery_store_a_clean_l1.csv
    data/grocery_store_b_clean_l1.csv

Outputs:
    data/grocery_store_a_clean_final.csv
    data/grocery_store_b_clean_final.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Helpers for category fallback
# ---------------------------------------------------------------------


def is_null_or_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def map_department_to_category_lvl1(department: str) -> str:
    """
    Same heuristic as in L1 fallback: map department / category_raw
    into our unified top-level category space.
    """
    if not isinstance(department, str):
        return "other_unknown"

    d = department.lower()

    if "produce" in d or "fruit" in d or "vegetable" in d:
        return "fresh_produce"
    if "meat" in d:
        return "meat"
    if "seafood" in d or "fish" in d:
        return "seafood"
    if "dairy" in d or "milk" in d or "cheese" in d:
        return "dairy"
    if "egg" in d:
        return "eggs"
    if "bakery" in d or "bread" in d:
        return "bakery"
    if "frozen" in d:
        return "frozen_food"
    if "alcohol" in d or "wine" in d or "beer" in d or "liquor" in d:
        return "alcohol"
    if "beverage" in d or "drink" in d or "soda" in d or "juice" in d:
        return "beverage_non_alcohol"
    if "snack" in d or "chips" in d or "candy" in d or "cookies" in d:
        return "snacks"
    if "cereal" in d or "breakfast" in d:
        return "breakfast_cereal"
    if "baby" in d:
        return "baby"
    if "pet" in d:
        return "pet"
    if "household" in d or "clean" in d or "paper" in d:
        return "household"
    if "personal" in d or "beauty" in d or "hair" in d or "skin" in d:
        return "personal_care"

    # default grocery-like
    return "pantry"


# ---------------------------------------------------------------------
# Column-level fusion helpers (row-wise)
# ---------------------------------------------------------------------


def choose_brand_final(row: pd.Series) -> Tuple[str, str]:
    """
    Choose brand_final and its source.

    Priority:
        1) GPT brand (not UNKNOWN, high/medium confidence)
        2) brand_raw from L0 (non-empty)
        3) UNKNOWN
    """
    gpt_brand = row.get("brand", None)
    gpt_conf = str(row.get("brand_confidence", "low")).lower()
    raw_brand = row.get("brand_raw", None)

    #  GPT brand
    if isinstance(gpt_brand, str):
        b = gpt_brand.strip()
        if b != "" and b.upper() != "UNKNOWN" and gpt_conf in ("high", "medium"):
            return b, "gpt"

    #  raw brand
    if isinstance(raw_brand, str):
        b2 = raw_brand.strip()
        if b2 != "":
            return b2, "raw"

    #  fallback
    return "UNKNOWN", "none"


def choose_product_core_final(row: pd.Series) -> Tuple[str, str]:
    """
    Choose product_core_final and its source.

    Priority:
        1) GPT product_core (not UNKNOWN, high/medium)
        2) UNKNOWN
    """
    gpt_core = row.get("product_core", None)
    gpt_conf = str(row.get("product_core_confidence", "low")).lower()

    if isinstance(gpt_core, str):
        c = gpt_core.strip()
        if c != "" and c.upper() != "UNKNOWN" and gpt_conf in ("high", "medium"):
            return c, "gpt"

    return "UNKNOWN", "none"


def choose_flavor_final(row: pd.Series) -> Tuple[str, str]:
    """
    Choose flavor_final.

    Priority:
        1) GPT flavor (not NONE, high/medium)
        2) NONE
    """
    gpt_flavor = row.get("flavor", None)
    gpt_conf = str(row.get("flavor_confidence", "low")).lower()

    if isinstance(gpt_flavor, str):
        f = gpt_flavor.strip()
        if f != "" and f.upper() != "NONE" and gpt_conf in ("high", "medium"):
            return f, "gpt"

    return "NONE", "none"


def choose_category_final(row: pd.Series) -> Tuple[str, str, str]:
    """
    Choose category_lvl1_final / category_lvl2_final / source.

    Priority:
        1) GPT category_lvl1 with high/medium confidence and not other_unknown.
           Use GPT category_lvl2 as well (even if NONE).
        2) Map department/category_raw to category_lvl1 via heuristic;
           set category_lvl2_final = NONE.
    """
    gpt_cat1 = row.get("category_lvl1", None)
    gpt_cat1_conf = str(row.get("category_lvl1_confidence", "low")).lower()
    gpt_cat2 = row.get("category_lvl2", None)

    #  Use GPT category if reliable
    if isinstance(gpt_cat1, str):
        c1 = gpt_cat1.strip()
        if c1 != "" and c1 != "other_unknown" and gpt_cat1_conf in ("high", "medium"):
            if isinstance(gpt_cat2, str) and gpt_cat2.strip() != "":
                c2 = gpt_cat2.strip()
            else:
                c2 = "NONE"
            return c1, c2, "gpt"

    #  Fallback: department/category_raw
    dept = row.get("department", row.get("category_raw", None))
    if isinstance(dept, str):
        c1_fb = map_department_to_category_lvl1(dept)
    else:
        c1_fb = "other_unknown"

    return c1_fb, "NONE", "department_fallback"


def choose_size_final(row: pd.Series) -> Tuple[float | None, float | None, float | None, str]:
    """
    Choose size_ml_final / size_g_final / size_count_final / size_final_source.

    Priority:
         GPT size (ml/g/count) when not null and size_confidence == high
         L0 size_*_l0 when not null
         None (for each dimension)
    """
    # GPT size
    size_ml_gpt = row.get("size_ml", None)
    size_g_gpt = row.get("size_g", None)
    size_count_gpt = row.get("size_count", None)
    size_conf = str(row.get("size_confidence", "")).lower()

    # L0 size
    size_ml_l0 = row.get("size_ml_l0", None)
    size_g_l0 = row.get("size_g_l0", None)
    size_count_l0 = row.get("size_count_l0", None)

    # Default output
    ml_final = None
    g_final = None
    count_final = None
    source = "none"

    #  GPT high confidence
    if size_conf == "high":
        used_gpt = False

        if not pd.isna(size_ml_gpt) if isinstance(size_ml_gpt, float) else size_ml_gpt not in (None, ""):
            ml_final = float(size_ml_gpt)
            used_gpt = True
        if not pd.isna(size_g_gpt) if isinstance(size_g_gpt, float) else size_g_gpt not in (None, ""):
            g_final = float(size_g_gpt)
            used_gpt = True
        if not pd.isna(size_count_gpt) if isinstance(size_count_gpt, float) else size_count_gpt not in (None, ""):
            count_final = float(size_count_gpt)
            used_gpt = True

        if used_gpt:
            source = "gpt"

    # 2) L0 fallback for any dimension still missing
    if ml_final is None and not is_null_or_empty(size_ml_l0):
        try:
            ml_final = float(size_ml_l0)
            if source == "none":
                source = "l0"
        except Exception:
            pass

    if g_final is None and not is_null_or_empty(size_g_l0):
        try:
            g_final = float(size_g_l0)
            if source == "none":
                source = "l0"
        except Exception:
            pass

    if count_final is None and not is_null_or_empty(size_count_l0):
        try:
            count_final = float(size_count_l0)
            if source == "none":
                source = "l0"
        except Exception:
            pass

    return ml_final, g_final, count_final, source


# ---------------------------------------------------------------------
# Main L2 function
# ---------------------------------------------------------------------


def apply_l2_cleaning_for_side(
    side: str,
    l1_csv_path: Path,
    output_csv_path: Path,
) -> None:
    """
    Load the L1 table for a given side (A or B), build *_final columns,
    and save a new CSV.
    """
    print(f"[L2] side={side} loading L1 table from {l1_csv_path} ...")
    df = pd.read_csv(l1_csv_path)

    # --- brand_final ---
    print(f"[L2] side={side} computing brand_final ...")
    brand_final_values = df.apply(
        lambda row: choose_brand_final(row),
        axis=1,
        result_type="expand",
    )
    df["brand_final"] = brand_final_values[0]
    df["brand_final_source"] = brand_final_values[1]

    # --- product_core_final ---
    print(f"[L2] side={side} computing product_core_final ...")
    core_final_values = df.apply(
        lambda row: choose_product_core_final(row),
        axis=1,
        result_type="expand",
    )
    df["product_core_final"] = core_final_values[0]
    df["product_core_final_source"] = core_final_values[1]

    # --- flavor_final ---
    print(f"[L2] side={side} computing flavor_final ...")
    flavor_final_values = df.apply(
        lambda row: choose_flavor_final(row),
        axis=1,
        result_type="expand",
    )
    df["flavor_final"] = flavor_final_values[0]
    df["flavor_final_source"] = flavor_final_values[1]

    # --- category_final (lvl1 + lvl2) ---
    print(f"[L2] side={side} computing category_lvl1_final / category_lvl2_final ...")
    cat_final_values = df.apply(
        lambda row: choose_category_final(row),
        axis=1,
        result_type="expand",
    )
    df["category_lvl1_final"] = cat_final_values[0]
    df["category_lvl2_final"] = cat_final_values[1]
    df["category_final_source"] = cat_final_values[2]

    # --- size_final (ml/g/count) ---
    print(f"[L2] side={side} computing size_*_final ...")
    size_final_values = df.apply(
        lambda row: choose_size_final(row),
        axis=1,
        result_type="expand",
    )
    df["size_ml_final"] = size_final_values[0]
    df["size_g_final"] = size_final_values[1]
    df["size_count_final"] = size_final_values[2]
    df["size_final_source"] = size_final_values[3]

    # --- Save result ---
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[L2] side={side} saved final clean table to {output_csv_path}.")
