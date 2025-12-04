"""
L1 GPT-based semantic cleaning 

This module:
- Takes L0 cleaned tables (A/B).
- Decides which rows really need GPT (hard rows).
- For hard rows: call GPT in parallel using ThreadPoolExecutor.
- For easy rows: build a fallback record using simple rules
  with no GPT call to save cost and time.
- Merges GPT/FALLBACK results back into the L0 table.

It also uses a shorter prompt to reduce tokens and speed up calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


# --------- Data classes ---------


@dataclass
class GPTCleaningConfig:
    """
    Configuration for running GPT cleaning on one side (A or B).
    """
    side: str  # "A" or "B"
    max_rows: Optional[int] = None        # for testing; None = all rows
    max_workers: int = 5                  # number of parallel GPT calls
    enable_skip_simple_rows: bool = True  # whether to skip easy rows
    sleep_seconds_between_batches: float = 0.0  # optional throttle between batches


# --------- Prompts & schema (shortened) ---------


SYSTEM_PROMPT = """
You are an expert data cleaning assistant for grocery products.

Goal:
- Read noisy product data (titles, raw brand, department, size_l0, etc.).
- Output EXACTLY ONE JSON object per request.
- Follow the required keys and value rules strictly.
- If unsure, use the fallback values (UNKNOWN / NONE / other_unknown).

Important:
- Do NOT output explanations, comments, or multiple JSON objects.
- Only output the final JSON object.
""".strip()


# Describe the required keys and allowed values in a compact way.
OUTPUT_SCHEMA_KEYS: Dict[str, str] = {
    # identity
    "product_id": "string (copy from input)",
    "side": "string, 'A' or 'B'",

    # brand
    "brand": "string; canonical brand or 'UNKNOWN'",
    "brand_confidence": "one of: high, medium, low",
    "is_private_label": "true, false, or null",

    # product core
    "product_core": "string; generic product name or 'UNKNOWN'",
    "product_core_confidence": "one of: high, medium, low",

    # flavor
    "flavor": "string; flavor or 'NONE'",
    "flavor_confidence": "one of: high, medium, low",

    # category
    "category_lvl1": (
        "one of: fresh_produce, meat, seafood, dairy, eggs, bakery, "
        "frozen_food, beverage_non_alcohol, alcohol, snacks, pantry, "
        "breakfast_cereal, baby, pet, household, personal_care, other_unknown"
    ),
    "category_lvl1_confidence": "one of: high, medium, low",
    "category_lvl2": "short sub-category phrase or 'NONE'",
    "category_lvl2_confidence": "one of: high, medium, low",

    # notes
    "notes": "string; can be empty",

    # size refinement
    "size_ml": "float or null; volume in ml",
    "size_g": "float or null; weight in g",
    "size_count": "float or null; count/pieces",
    "size_confidence": "one of: high, medium, low, or null when not applicable",
}


def build_user_message_for_row(row: pd.Series, side: str) -> str:
    """
    Build the user message that will be sent to GPT for a single row.

    We keep it compact to save tokens:
    - Input JSON with core noisy fields.
    - Short description of required keys.
    - Clear instructions about size behavior.
    """
    input_payload = {
        "product_id": str(row.get("bb_id", "")),
        "side": side,
        "title_normalized_l0": row.get("title_normalized_l0", ""),
        "title_raw": row.get("title_raw", row.get("name", "")),
        "brand_raw": row.get("brand_raw", ""),
        "department": row.get("department", row.get("category_raw", "")),
        "category_raw": row.get("category_raw", ""),
        "description": row.get("description", ""),
        "permanent_tags": row.get("permanent_tags", ""),
        "is_private_label_raw": row.get("is_private_label", None),
        "is_fresh_raw": row.get("is_fresh", None),
        "size_value_raw_l0": row.get("size_value_raw_l0", None),
        "size_unit_raw_l0": row.get("size_unit_raw_l0", None),
        "size_ml_l0": row.get("size_ml_l0", None),
        "size_g_l0": row.get("size_g_l0", None),
        "size_count_l0": row.get("size_count_l0", None),
        "size_unit_norm_l0": row.get("size_unit_norm_l0", None),
    }

    parts: List[str] = []

    parts.append("You will receive noisy product information as JSON.")
    parts.append("Use it to infer clean, normalized attributes.")
    parts.append("")
    parts.append("INPUT JSON:")
    parts.append(json.dumps(input_payload, ensure_ascii=False))
    parts.append("")
    parts.append("REQUIRED OUTPUT:")
    parts.append("Return exactly ONE JSON object with these keys:")
    parts.append(json.dumps(OUTPUT_SCHEMA_KEYS, ensure_ascii=False, indent=2))
    parts.append("")
    parts.append(
        "Rules:\n"
        "- Use 'UNKNOWN' for unknown brand or product_core.\n"
        "- Use 'NONE' for missing flavor or category_lvl2.\n"
        "- Use 'other_unknown' for category_lvl1 when unsure.\n"
        "- Use true/false/null for is_private_label.\n"
        "- For size fields, prefer the existing *_l0 values when they look reasonable.\n"
        "- Only change or infer size when the text clearly indicates a better value.\n"
        "- Do NOT add extra keys. Do NOT output anything except the JSON object."
    )

    return "\n".join(parts)


# --------- Simple heuristics:when to call GPT for this row ---------


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
    Heuristic mapping from department text to our top-level category.

    This is only used for "easy rows" where we skip GPT.
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


def needs_gpt(row: pd.Series) -> bool:
    """
    Decide whether this row really needs GPT.

    We treat a row as "easy" (no GPT needed) if:
    - brand_raw is non-empty, AND
    - department/category_raw is non-empty.

    Everything else is "hard" and will be sent to GPT.
    """
    brand_raw = row.get("brand_raw", None)
    dept = row.get("department", row.get("category_raw", None))

    brand_missing = is_null_or_empty(brand_raw)
    dept_missing = is_null_or_empty(dept)

    # If either brand or department/category is missing, call GPT.
    if brand_missing or dept_missing:
        return True

    # Otherwise consider it "easy".
    return False


def build_fallback_record(row: pd.Series, side: str) -> Dict[str, Any]:
    """
    Build a non-GPT fallback record for "easy" rows.

    This uses:
    - brand_raw as brand (medium confidence).
    - department/category_raw mapped to category_lvl1 (medium).
    - L0 size fields copied as-is (no extra inference).
    The goal is not to be perfect, but to produce a reasonable default
    without spending GPT tokens.
    """
    brand_raw = row.get("brand_raw", "")
    brand = "UNKNOWN"
    if isinstance(brand_raw, str) and brand_raw.strip() != "":
        brand = brand_raw.strip()

    dept = row.get("department", row.get("category_raw", ""))
    category_lvl1 = map_department_to_category_lvl1(dept) if isinstance(dept, str) else "other_unknown"

    return {
        "product_id": str(row.get("bb_id", "")),
        "side": side,

        "brand": brand,
        "brand_confidence": "medium" if brand != "UNKNOWN" else "low",
        "is_private_label": row.get("is_private_label", None),

        "product_core": "UNKNOWN",
        "product_core_confidence": "low",

        "flavor": "NONE",
        "flavor_confidence": "low",

        "category_lvl1": category_lvl1,
        "category_lvl1_confidence": "medium" if category_lvl1 != "other_unknown" else "low",

        "category_lvl2": "NONE",
        "category_lvl2_confidence": "low",

        "notes": "SKIPPED_GPT_EASY_ROW",

        "size_ml": row.get("size_ml_l0", None),
        "size_g": row.get("size_g_l0", None),
        "size_count": row.get("size_count_l0", None),
        "size_confidence": None,
    }


# --------- Core GPT call helpers ---------


def call_gpt_for_row(
    client: OpenAI,
    model: str,
    row: pd.Series,
    side: str,
) -> Dict[str, Any]:
    """
    Call the GPT model for a single row and return the parsed JSON dict.

    - Any API error or unexpected return type leads to a fallback object.
    - Any JSON parsing error also leads to a fallback object.
    This way, one bad call will not crash the whole batch.
    """
    user_message = build_user_message_for_row(row, side)

    # ----  API  ----
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        if hasattr(response, "choices"):
            content = response.choices[0].message.content
        else:
            content = str(response)

    except Exception as exc:
        print(
            f"[WARN] GPT API error for product_id={row.get('bb_id')} "
            f"side={side}: {exc}"
        )
        fallback: Dict[str, Any] = {
            "product_id": str(row.get("bb_id", "")),
            "side": side,
            "brand": "UNKNOWN",
            "brand_confidence": "low",
            "is_private_label": None,
            "product_core": "UNKNOWN",
            "product_core_confidence": "low",
            "flavor": "NONE",
            "flavor_confidence": "low",
            "category_lvl1": "other_unknown",
            "category_lvl1_confidence": "low",
            "category_lvl2": "NONE",
            "category_lvl2_confidence": "low",
            "notes": "GPT_API_ERROR",
            "size_ml": row.get("size_ml_l0", None),
            "size_g": row.get("size_g_l0", None),
            "size_count": row.get("size_count_l0", None),
            "size_confidence": None,
        }
        return fallback

    # ----  JSON ----
    try:
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("Model output is not a JSON object")
        return obj
    except Exception as exc:
        print(
            f"[WARN] Failed to parse GPT JSON for product_id={row.get('bb_id')} "
            f"side={side}: {exc}"
        )
        fallback: Dict[str, Any] = {
            "product_id": str(row.get("bb_id", "")),
            "side": side,
            "brand": "UNKNOWN",
            "brand_confidence": "low",
            "is_private_label": None,
            "product_core": "UNKNOWN",
            "product_core_confidence": "low",
            "flavor": "NONE",
            "flavor_confidence": "low",
            "category_lvl1": "other_unknown",
            "category_lvl1_confidence": "low",
            "category_lvl2": "NONE",
            "category_lvl2_confidence": "low",
            "notes": "GPT_JSON_PARSE_ERROR",
            "size_ml": row.get("size_ml_l0", None),
            "size_g": row.get("size_g_l0", None),
            "size_count": row.get("size_count_l0", None),
            "size_confidence": None,
        }
        return fallback



def _gpt_task(
    client: OpenAI,
    model: str,
    row_index: int,
    row: pd.Series,
    side: str,
) -> tuple[int, Dict[str, Any]]:
    """
    Small wrapper for concurrent GPT calls.

    Returns (row_index, gpt_obj).
    """
    print(f"[L1-GPT] side={side} row_index={row_index} bb_id={row.get('bb_id')}")
    gpt_obj = call_gpt_for_row(client, model, row, side)
    return row_index, gpt_obj


def run_gpt_cleaning_for_side(
    side: str,
    l0_csv_path: Path,
    client: OpenAI,
    model: str,
    output_csv_path: Path,
    output_jsonl_path: Optional[Path] = None,
    config: Optional[GPTCleaningConfig] = None,
) -> None:
    """
    Run GPT cleaning for one side (A or B), with:
    - skipping of easy rows,
    - concurrent GPT calls (max_workers).
    """
    if config is None:
        config = GPTCleaningConfig(side=side)

    print(f"[L1] Loading L0 data for side {side} from {l0_csv_path} ...")
    df_l0 = pd.read_csv(l0_csv_path)

    if config.max_rows is not None:
        df_l0 = df_l0.head(config.max_rows)
        print(f"[L1] Restricting to first {config.max_rows} rows for side {side}.")

    n_rows = len(df_l0)
    print(f"[L1] side={side} total rows to process (after limit) = {n_rows}")

    # Prepare list for GPT / fallback records, aligned with df_l0 index.
    gpt_records: List[Optional[Dict[str, Any]]] = [None] * n_rows

    # Decide which rows need GPT.
    need_gpt_flags: List[bool] = []
    for idx, row in df_l0.iterrows():
        flag = needs_gpt(row) if config.enable_skip_simple_rows else True
        need_gpt_flags.append(flag)
        if not flag:
            # Build fallback record immediately for easy rows.
            fallback_obj = build_fallback_record(row, side)
            gpt_records[idx] = fallback_obj

    total_need_gpt = sum(need_gpt_flags)
    total_skip = n_rows - total_need_gpt
    print(
        f"[L1] side={side} rows needing GPT = {total_need_gpt}, "
        f"rows using fallback (no GPT) = {total_skip}"
    )

    # simple sleep before heavy GPT load.
    if config.sleep_seconds_between_batches > 0:
        time.sleep(config.sleep_seconds_between_batches)

    # Run GPT calls in parallel for the rows that need GPT.
    if total_need_gpt > 0:
        max_workers = max(1, config.max_workers)
        print(f"[L1] side={side} starting GPT calls with max_workers={max_workers} ...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, row in df_l0.iterrows():
                if not need_gpt_flags[idx]:
                    continue
                futures.append(
                    executor.submit(_gpt_task, client, model, idx, row, side)
                )

            for future in as_completed(futures):
                row_index, gpt_obj = future.result()
                gpt_records[row_index] = gpt_obj

    # Sanity check: ensure all records are filled.
    for i, record in enumerate(gpt_records):
        if record is None:
            print(f"[WARN] side={side} row_index={i} missing GPT/fallback record; "
                  f"using emergency fallback.")
            row = df_l0.iloc[i]
            gpt_records[i] = build_fallback_record(row, side)

    # Convert GPT/FALLBACK records to DataFrame.
    df_gpt = pd.DataFrame(gpt_records)

    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open("w", encoding="utf-8") as f_jsonl:
            for obj in gpt_records:
                f_jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[L1] side={side} wrote raw GPT/FALLBACK JSONL to {output_jsonl_path}")

    # --- Prepare for merge  ---

    # Reset index to ensure align rows purely by position.
    df_l0_reset = df_l0.reset_index(drop=True)
    df_gpt_reset = df_gpt.reset_index(drop=True)

    # To avoid duplicated columns, drop product_id/side from GPT part,
    cols_to_drop = [c for c in ["product_id", "side"] if c in df_gpt_reset.columns]
    df_gpt_reset = df_gpt_reset.drop(columns=cols_to_drop, errors="ignore")

    # Horizontal concat: each row i in df_l0 matches gpt_records[i]
    df_merged = pd.concat([df_l0_reset, df_gpt_reset], axis=1)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv_path, index=False)
    print(f"[L1] side={side} saved merged L0+L1 output to {output_csv_path}.")
