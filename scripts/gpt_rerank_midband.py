# scripts/gpt_rerank_midband.py
"""
GPT mid-band reranking for BetterBasket matching.

This script:
- Loads OpenAI credentials from openai_creds.yaml (same as L1 cleaning).
- Loads mid-band candidate matches (score_total in [0.65, 0.70)).
- Calls GPT for semantic evaluation.
- Produces a new file with gpt_label, gpt_score, and final_score.

No Chinese text is included in prompts or code.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI


# -----------------------------------------------------------
# Paths 
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CREDS_PATH = PROJECT_ROOT / "openai_creds.yaml"

MATCHES_FOR_GPT = ARTIFACTS_DIR / "matches_for_gpt.csv"
MATCHES_GPT_REVIEW = ARTIFACTS_DIR / "matches_gpt_review.csv"

# Weights to combine rule score and GPT score
WEIGHT_RULE = 0.6
WEIGHT_GPT = 0.4


# ------------------------------------------------------------
# loading pattern
# ------------------------------------------------------------

def load_openai_client_from_yaml(creds_path: Path) -> tuple[OpenAI, str]:
    """
    Load OpenAI client configuration from a YAML file.

    Expected YAML structure:
        openai:
          endpoint: https://.../openai/v1/
          api_key: YOUR_API_KEY
          deployment_name: your-deployment-name
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


# ------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------

def build_prompt(row: pd.Series) -> str:
    """
    Build an English prompt describing A and B products to GPT.
    GPT is asked to label: exact / substitute / none + score 0~1.
    """

    def info(prefix: str):
        return {
            "title": row.get(f"{prefix}_title_normalized_l0", ""),
            "brand": row.get(f"{prefix}_brand_final", ""),
            "category": row.get(f"{prefix}_category_lvl1_final", ""),
            "size_ml": row.get(f"{prefix}_size_ml_final", ""),
            "size_g": row.get(f"{prefix}_size_g_final", ""),
            "size_count": row.get(f"{prefix}_size_count_final", ""),
        }

    A = info("a")
    B = info("b")

    # JSON format description as a plain string 
    json_format_block = (
        "Return ONLY a JSON object, no explanation. Format:\n"
        "{\n"
        '  "label": "exact" or "substitute" or "none",\n'
        '  "score": <a number between 0 and 1>\n'
        "}\n"
    )

    prompt = (
        "You are a product matching assistant for grocery items.\n\n"
        "You will be given two products:\n"
        "- Product A from store A\n"
        "- Product B from store B\n\n"
        "Your task:\n"
        "1. Determine how well Product B matches Product A.\n"
        '2. Choose exactly ONE label:\n'
        '   - "exact": nearly the same SKU (same brand/series, same or very similar size).\n'
        '   - "substitute": not identical but a strong substitute most customers would accept.\n'
        '   - "none": not a suitable match (different type, ingredient, usage, etc.).\n'
        "3. Provide a numeric match score between 0 and 1.\n\n"
        f"{json_format_block}\n"
        f"Product A:\n"
        f"- title: {A['title']}\n"
        f"- brand: {A['brand']}\n"
        f"- category: {A['category']}\n"
        f"- size_ml: {A['size_ml']}\n"
        f"- size_g: {A['size_g']}\n"
        f"- size_count: {A['size_count']}\n\n"
        f"Product B:\n"
        f"- title: {B['title']}\n"
        f"- brand: {B['brand']}\n"
        f"- category: {B['category']}\n"
        f"- size_ml: {B['size_ml']}\n"
        f"- size_g: {B['size_g']}\n"
        f"- size_count: {B['size_count']}\n"
    )

    return prompt


# ------------------------------------------------------------
# GPT call function
# ------------------------------------------------------------

def call_gpt(client: OpenAI, model_name: str, prompt: str) -> dict:
    """
    Calls GPT using Azure OpenAI client.
    Returns: {"label": ..., "score": ...}
    On failure: {"label": "none", "score": 0.0}
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content
        data = json.loads(content)

        label = str(data.get("label", "none")).lower()
        score = float(data.get("score", 0.0))

        if label not in {"exact", "substitute", "none"}:
            label = "none"
        score = max(0.0, min(1.0, score))  # clamp

        return {"label": label, "score": score}

    except Exception as e:
        print(f"[GPT] Error: {e}")
        return {"label": "none", "score": 0.0}


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main() -> None:
    print(f"[GPTReview] Loading credentials from {CREDS_PATH}")
    client, model_name = load_openai_client_from_yaml(CREDS_PATH)

    print(f"[GPTReview] Loading mid-band matches: {MATCHES_FOR_GPT}")
    df = pd.read_csv(MATCHES_FOR_GPT)

    if "score_total" not in df.columns:
        raise ValueError("Column `score_total` not found in matches_for_gpt.csv.")

    gpt_labels = []
    gpt_scores = []
    final_scores = []

    total = len(df)
    for idx, row in df.iterrows():
        print(f"[GPTReview] Processing {idx + 1}/{total} ...")

        prompt = build_prompt(row)
        result = call_gpt(client, model_name, prompt)

        rule_score = float(row["score_total"])
        gpt_label = result["label"]
        gpt_score = result["score"]

        final_score = WEIGHT_RULE * rule_score + WEIGHT_GPT * gpt_score

        gpt_labels.append(gpt_label)
        gpt_scores.append(gpt_score)
        final_scores.append(final_score)

        time.sleep(0.01)

    df["gpt_label"] = gpt_labels
    df["gpt_score"] = gpt_scores
    df["final_score"] = final_scores

    print(f"[GPTReview] Saving to {MATCHES_GPT_REVIEW}")
    df.to_csv(MATCHES_GPT_REVIEW, index=False)
    print("[GPTReview] Done.")


if __name__ == "__main__":
    main()
