"""
Entry script for L1 GPT-based cleaning.

This script:
- Loads OpenAI credentials from openai_creds.yaml.
- Creates an OpenAI client using the Azure OpenAI endpoint.
- Runs GPT cleaning for side A and side B on top of the L0 tables,
  with:
    * skipping easy rows (no GPT),
    * parallel GPT calls (max_workers).

- set max_rows=None to process all rows.
- adjust max_workers depending on rate limits.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from openai import OpenAI

from src.preprocess.gpt_cleaning import (
    run_gpt_cleaning_for_side,
    GPTCleaningConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CREDS_PATH = PROJECT_ROOT / "openai_creds.yaml"


def load_openai_client_from_yaml(creds_path: Path) -> tuple[OpenAI, str]:
    """
    Load OpenAI client configuration from the YAML file.
    """
    with creds_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    openai_cfg = cfg["openai"]
    endpoint = openai_cfg["endpoint"]
    api_key = openai_cfg["api_key"]
    deployment_name = openai_cfg["deployment_name"]

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )

    return client, deployment_name


def main() -> None:
    print(f"[L1] Using credentials from {CREDS_PATH}")
    client, model_name = load_openai_client_from_yaml(CREDS_PATH)

    # ===== Side A =====
    config_a = GPTCleaningConfig(
        side="A",
        max_rows=None,                
        max_workers=25,              
        enable_skip_simple_rows=True,
        sleep_seconds_between_batches=0.0,
    )
    run_gpt_cleaning_for_side(
        side="A",
        l0_csv_path=DATA_DIR / "grocery_store_a_clean_l0.csv",
        client=client,
        model=model_name,
        output_csv_path=DATA_DIR / "grocery_store_a_clean_l1.csv",
        output_jsonl_path=DATA_DIR / "grocery_store_a_l1_gpt.jsonl",
        config=config_a,
    )

    # ===== Side B =====
    config_b = GPTCleaningConfig(
        side="B",
        max_rows=None,                
        max_workers=25,          
        enable_skip_simple_rows=True,
        sleep_seconds_between_batches=0.0,
    )
    run_gpt_cleaning_for_side(
        side="B",
        l0_csv_path=DATA_DIR / "grocery_store_b_clean_l0.csv",
        client=client,
        model=model_name,
        output_csv_path=DATA_DIR / "grocery_store_b_clean_l1.csv",
        output_jsonl_path=DATA_DIR / "grocery_store_b_l1_gpt.jsonl",
        config=config_b,
    )


if __name__ == "__main__":
    main()
