# scripts/select_gpt_band.py
"""
Select pairs in the mid-confidence band for GPT review.

Input:
    artifacts/matches_v0.csv
Output:
    artifacts/matches_for_gpt.csv
"""

import os
import pandas as pd

ARTIFACTS_DIR = "artifacts"
MATCHES_V0 = os.path.join(ARTIFACTS_DIR, "matches_v0.csv")
MATCHES_FOR_GPT = os.path.join(ARTIFACTS_DIR, "matches_for_gpt.csv")

MIN_SCORE = 0.70
MAX_SCORE = 0.71  
MAX_ROWS = 2000    

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"[SelectGPT] Loading matches from: {MATCHES_V0}")
    df = pd.read_csv(MATCHES_V0)

    if "score_total" not in df.columns:
        raise ValueError("Column `score_total` not found in matches_v0.csv.")

    mask = (df["score_total"] >= MIN_SCORE) & (df["score_total"] < MAX_SCORE)
    df_band = df.loc[mask].copy().reset_index(drop=True)
    print(f"[SelectGPT] In score band [{MIN_SCORE}, {MAX_SCORE}): {len(df_band)} rows")

    if len(df_band) > MAX_ROWS:
        df_band = df_band.head(MAX_ROWS)
        print(f"[SelectGPT] Truncated to first {MAX_ROWS} rows for GPT demo.")

    print(f"[SelectGPT] Saving to: {MATCHES_FOR_GPT}")
    df_band.to_csv(MATCHES_FOR_GPT, index=False)
    print("[SelectGPT] Done.")


if __name__ == "__main__":
    main()
