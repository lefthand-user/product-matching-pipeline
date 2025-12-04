# scripts/build_final_matches.py
"""
Build final matches after GPT review.

Steps:
 Load baseline matches_v0 from artifacts/matches_v0.csv.
 Load GPT mid-band review from artifacts/matches_gpt_review.csv.
 Remove all pairs where GPT label == "none" (for the mid-band subset).
 Join back bb_id_A and bb_id_B from the Phase 2 cleaned tables:
       data/grocery_store_a_clean_phase2.csv
       data/grocery_store_b_clean_phase2.csv
   using a_row_index / b_row_index.
 Save:
   - data/Final_matches.csv                # full table after GPT filtering (with bb_id_A, bb_id_B)
   - result/final_matches_bb_ids.csv       # only (bb_id_A, bb_id_B)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULT_DIR = PROJECT_ROOT / "result"

MATCHES_V0_PATH = ARTIFACTS_DIR / "matches_v0.csv"
MATCHES_GPT_PATH = ARTIFACTS_DIR / "matches_gpt_review.csv"

A_PHASE2_PATH = DATA_DIR / "grocery_store_a_clean_phase2.csv"
B_PHASE2_PATH = DATA_DIR / "grocery_store_b_clean_phase2.csv"

FINAL_MATCHES_PATH = DATA_DIR / "Final_matches.csv"
FINAL_IDS_PATH = RESULT_DIR / "final_matches_bb_ids.csv"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    #  load baseline matches_v0
    print(f"[FinalMatches] Loading baseline matches_v0: {MATCHES_V0_PATH}")
    df_v0 = pd.read_csv(MATCHES_V0_PATH)

    required_v0 = {"a_row_index", "b_row_index", "score_total"}
    missing_v0 = required_v0 - set(df_v0.columns)
    if missing_v0:
        raise ValueError(
            f"matches_v0.csv is missing required columns: {missing_v0}. "
            f"Available columns: {list(df_v0.columns)}"
        )

    #  load GPT review 
    print(f"[FinalMatches] Loading GPT review: {MATCHES_GPT_PATH}")
    df_gpt = pd.read_csv(MATCHES_GPT_PATH)

    required_gpt = {"a_row_index", "b_row_index", "gpt_label"}
    missing_gpt = required_gpt - set(df_gpt.columns)
    if missing_gpt:
        raise ValueError(
            f"matches_gpt_review.csv is missing required columns: {missing_gpt}. "
            f"Available columns: {list(df_gpt.columns)}"
        )

    # build pair key for join
    df_v0["pair_key"] = list(zip(df_v0["a_row_index"], df_v0["b_row_index"]))
    df_gpt["pair_key"] = list(zip(df_gpt["a_row_index"], df_gpt["b_row_index"]))

    # keys with gpt_label == "none" removed
    bad_keys = set(df_gpt.loc[df_gpt["gpt_label"] == "none", "pair_key"])
    print(f"[FinalMatches] Pairs with gpt_label == 'none': {len(bad_keys)}")

    before = len(df_v0)
    df_filtered = df_v0[~df_v0["pair_key"].isin(bad_keys)].copy()
    after = len(df_filtered)

    print(f"[FinalMatches] Baseline matches_v0 rows: {before}")
    print(f"[FinalMatches] Rows removed by GPT filter: {before - after}")
    print(f"[FinalMatches] Rows kept after GPT filter: {after}")

    #  drop helper key
    df_filtered = df_filtered.drop(columns=["pair_key"])

    #  join back bb_id_A and bb_id_B from phase2 tables
    print(f"[FinalMatches] Loading Phase2 A table: {A_PHASE2_PATH}")
    df_a = pd.read_csv(A_PHASE2_PATH)
    print(f"[FinalMatches] Loading Phase2 B table: {B_PHASE2_PATH}")
    df_b = pd.read_csv(B_PHASE2_PATH)

    if "bb_id" not in df_a.columns or "bb_id" not in df_b.columns:
        raise ValueError("Phase2 tables must contain column 'bb_id'.")

    # create mapping 
    df_a_ids = (
        df_a.reset_index()
        .rename(columns={"index": "a_row_index", "bb_id": "bb_id_A"})
        [["a_row_index", "bb_id_A"]]
    )
    df_b_ids = (
        df_b.reset_index()
        .rename(columns={"index": "b_row_index", "bb_id": "bb_id_B"})
        [["b_row_index", "bb_id_B"]]
    )

    print("[FinalMatches] Joining bb_id_A and bb_id_B ...")
    df_final = df_filtered.merge(df_a_ids, on="a_row_index", how="left")
    df_final = df_final.merge(df_b_ids, on="b_row_index", how="left")

    # simple sanity check
    if df_final["bb_id_A"].isna().any() or df_final["bb_id_B"].isna().any():
        missing_a = df_final["bb_id_A"].isna().sum()
        missing_b = df_final["bb_id_B"].isna().sum()
        print(
            f"[FinalMatches] WARNING: missing bb_id_A: {missing_a}, "
            f"missing bb_id_B: {missing_b}"
        )

    # save full table to data/
    print(f"[FinalMatches] Saving full final matches to: {FINAL_MATCHES_PATH}")
    df_final.to_csv(FINAL_MATCHES_PATH, index=False)

    # extract only (bb_id_A, bb_id_B) to result/
    df_ids = df_final[["bb_id_A", "bb_id_B"]].dropna().drop_duplicates().reset_index(drop=True)
    print(f"[FinalMatches] Saving ID pairs to: {FINAL_IDS_PATH}")
    df_ids.to_csv(FINAL_IDS_PATH, index=False)

    print("[FinalMatches] Done.")


if __name__ == "__main__":
    main()
