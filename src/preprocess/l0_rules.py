"""
L0 deterministic cleaning rules.

Implements:
- Title normalization
- UPC cleaning
- Size parsing & normalization
- Retention of raw brand/category

This module is used by run_cleaning_v2.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import re
import pandas as pd


Side = Literal["A", "B"]


# ---------------------------------------------------------
# L0 Configuration
# ---------------------------------------------------------

@dataclass
class L0Config:
    """Configuration for L0 cleaning."""

    side: Side
    product_id_col: str
    title_col: str
    upc_col: Optional[str] = None
    brand_col: Optional[str] = None
    category_col: Optional[str] = None


# ---------------------------------------------------------
# Text helpers
# ---------------------------------------------------------

WHITESPACE_RE = re.compile(r"\s+")

def normalize_title(title: str) -> str:
    """Normalize text title: lowercase, strip, collapse spaces."""
    if not isinstance(title, str):
        return ""
    t = title.strip().lower()
    t = WHITESPACE_RE.sub(" ", t)
    return t


# ---------------------------------------------------------
# UPC cleaning
# ---------------------------------------------------------

def clean_upc(upc: str) -> tuple[Optional[str], bool]:
    """Extract digits from UPC."""
    if not isinstance(upc, str):
        return None, False
    digits = re.sub(r"\D", "", upc)
    if digits == "":
        return None, False
    return digits, True


# ---------------------------------------------------------
# Size parsing & normalization
# ---------------------------------------------------------

SIZE_RE = re.compile(
    r"(?P<value>\d+(\.\d+)?)\s*(?P<unit>[a-zA-Z]+)",
    re.IGNORECASE,
)

def parse_size_from_title(title: str) -> tuple[Optional[float], Optional[str]]:
    if not isinstance(title, str):
        return None, None
    m = SIZE_RE.search(title)
    if not m:
        return None, None
    return float(m.group("value")), m.group("unit").lower()


def normalize_size(value: Optional[float], unit: Optional[str]):
    """Return normalized sizes: (ml, g, count, normalized_unit)."""
    if value is None or unit is None:
        return None, None, None, None

    u = unit.lower()

    # Volume
    if u == "ml":
        return value, None, None, "ml"
    if u in ["l", "lt", "liter"]:
        return value * 1000.0, None, None, "ml"

    # Weight
    if u == "g":
        return None, value, None, "g"
    if u in ["kg", "kgs"]:
        return None, value * 1000.0, None, "g"
    if u in ["oz", "ounce", "ounces"]:
        return None, value * 28.3495, None, "g"
    if u in ["lb", "lbs"]:
        return None, value * 453.592, None, "g"

    # Count
    if u in ["ct", "cnt", "count", "pk", "pack"]:
        return None, None, value, "count"

    return None, None, None, None


# ---------------------------------------------------------
# Main L0 Cleaning Function
# ---------------------------------------------------------

def apply_l0_cleaning(df: pd.DataFrame, config: L0Config) -> pd.DataFrame:
    """
    Apply deterministic cleaning rules to DataFrame.
    Returns a new DataFrame with added *_l0 columns.
    """
    result = df.copy()

    # Title
    result["title_raw"] = result[config.title_col]
    result["title_normalized_l0"] = result[config.title_col].apply(normalize_title)

    # UPC
    if config.upc_col and config.upc_col in result.columns:
        cleaned = result[config.upc_col].apply(clean_upc)
        result["upc_clean_l0"] = cleaned.apply(lambda x: x[0])
        result["has_upc_l0"] = cleaned.apply(lambda x: x[1])
    else:
        result["upc_clean_l0"] = None
        result["has_upc_l0"] = False

    # Raw brand & category passthrough
    if config.brand_col and config.brand_col in result.columns:
        result["brand_raw"] = result[config.brand_col]
    else:
        result["brand_raw"] = None

    if config.category_col and config.category_col in result.columns:
        result["category_raw"] = result[config.category_col]
    else:
        result["category_raw"] = None

    # Size parsing
    parsed_sizes = result["title_normalized_l0"].apply(parse_size_from_title)
    result["size_value_raw_l0"] = parsed_sizes.apply(lambda x: x[0])
    result["size_unit_raw_l0"] = parsed_sizes.apply(lambda x: x[1])

    normalized = parsed_sizes.apply(lambda x: normalize_size(x[0], x[1]))
    result["size_ml_l0"] = normalized.apply(lambda x: x[0])
    result["size_g_l0"] = normalized.apply(lambda x: x[1])
    result["size_count_l0"] = normalized.apply(lambda x: x[2])
    result["size_unit_norm_l0"] = normalized.apply(lambda x: x[3])

    # Attach metadata
    result["side"] = config.side
    result["product_id"] = result[config.product_id_col]

    return result
