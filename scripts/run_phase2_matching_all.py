# scripts/run_phase2_matching_all.py
"""
Phase 2 entry script: run full matching pipeline on cleaned data.

This script assumes that Phase 1 has already produced:
    data/grocery_store_a_clean_final.csv
    data/grocery_store_b_clean_final.csv

It will:
1) Run Phase 2 preprocessing.
2) Build B-side embeddings and FAISS index.
3) Retrieve top-K candidates for each A product.
4) Add features and compute weighted scores.
5) Build UPC exact matches (Stage 0).
6) Build baseline matches_v0 with rule filter.
7) (Optional) Sample matches for manual QA.
8) (Optional) Run GPT mid-band review on medium-confidence pairs.
9) (Optional) Build final match lists (with bb_id_A, bb_id_B) after GPT filtering.

IMPORTANT:
- GPT mid-band review is also relatively slow and consumes tokens.
  You can disable it by setting ENABLE_GPT_RERANK = False below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Toggle this flag if you want to skip GPT mid-band review
ENABLE_GPT_RERANK = True


def run_step(script_name: str, project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    cmd = [sys.executable, str(scripts_dir / script_name)]
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(project_root))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    print("=======================================================")
    print("  Phase 2: MATCHING PIPELINE (from *_clean_final)")
    print("=======================================================")
    print("This script starts from cleaned final tables in data/")
    print("and runs the full matching baseline + optional GPT review.")
    print("=======================================================\n")

    # 1) Phase 2 preprocessing (from *_clean_final to *_clean_phase2)
    run_step("run_preprocess_phase2.py", project_root)

    # 2) Build B-side embeddings + FAISS index
    run_step("build_b_index.py", project_root)

    # 3) Retrieve top-K candidates for each A
    run_step("retrieve_topk.py", project_root)

    # 4) Add features + weighted scores
    run_step("score_candidates.py", project_root)

    # 5) UPC exact matches (Stage 0)
    run_step("build_upc_exact_matches.py", project_root)

    # 6) Build baseline matches_v0 with strict rule filter
    run_step("build_matches_v0.py", project_root)

    # 7) Optional: sample matches for manual QA
    try:
        run_step("sample_matches.py", project_root)
    except FileNotFoundError:
        print("sample_matches.py not found, skipping manual QA sampling step.")

    # 8) Optional: GPT mid-band review
    if ENABLE_GPT_RERANK:
        print("\n=======================================================")
        print("  GPT MID-BAND REVIEW ENABLED")
        print("=======================================================")
        print("This step will call GPT on medium-confidence matches ")
        print("(score_total in [0.65, 0.70)).")
        print("It may take some time and will consume tokens.")
        print("If you want to skip this step, set:")
        print("  ENABLE_GPT_RERANK = False")
        print("at the top of this script.\n")

        run_step("select_gpt_band.py", project_root)
        run_step("gpt_rerank_midband.py", project_root)

        # 9) Build final match list (uses GPT results to filter out 'none')
        run_step("build_final_matches.py", project_root)

    else:
        print("\n=======================================================")
        print("  GPT MID-BAND REVIEW DISABLED")
        print("=======================================================")
        print("Skipping select_gpt_band.py, gpt_rerank_midband.py and")
        print("build_final_matches.py (which depends on GPT labels).")
        print("You still get the baseline matches here:")
        print("  artifacts/matches_v0.csv")
        print("If you later decide to run GPT review, you can run:")
        print("  python scripts/select_gpt_band.py")
        print("  python scripts/gpt_rerank_midband.py")
        print("  python scripts/build_final_matches.py")

    print("\n=== Phase 2 finished ===")
    print("Main outputs:")
    print("  artifacts/matches_v0.csv            # baseline matches after rule filter")
    print("  artifacts/upc_exact_matches.csv     # UPC-based exact matches (Stage 0)")
    print("  artifacts/sample_matches_v0.csv     # manual QA sample (if generated)")
    if ENABLE_GPT_RERANK:
        print("  artifacts/matches_gpt_review.csv    # GPT-reviewed mid-band matches")
        print("  data/Final_matches.csv              # final matches after GPT filtering")
        print("  result/final_matches_bb_ids.csv     # (bb_id_A, bb_id_B) pairs")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
