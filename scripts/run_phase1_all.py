# scripts/run_phase1_all.py
"""
Phase 1 entry script: run full data cleaning pipeline (L0 -> L1 GPT -> L2 final).

This script:
- Calls scripts/run_cleaning_v2.py.
- Produces:
    data/grocery_store_a_clean_final.csv
    data/grocery_store_b_clean_final.csv

IMPORTANT:
- This step is SLOW and EXPENSIVE because it uses GPT heavily.
- If you only want to reproduce the matching results, you can SKIP this
  step and directly use the pre-cleaned *_clean_final.csv files
  provided in the repo, and then run:
      python scripts/run_phase2_matching_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"

    print("=======================================================")
    print("  Phase 1: FULL DATA CLEANING PIPELINE")
    print("=======================================================")
    print("This step may take a long time and uses GPT extensively.")
    print("If you only want to run matching on pre-cleaned data,")
    print("you can skip this and go directly to Phase 2:")
    print("  python scripts/run_phase2_matching_all.py")
    print("-------------------------------------------------------\n")

    cmd = [sys.executable, str(scripts_dir / "run_cleaning_v2.py")]
    print(f"[Phase1-All] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(project_root))

    print("\n[Phase1-All] Done.")
    print("Cleaned final tables should now exist at:")
    print("  data/grocery_store_a_clean_final.csv")
    print("  data/grocery_store_b_clean_final.csv")
    print("You can now run Phase 2 matching:")
    print("  python scripts/run_phase2_matching_all.py")


if __name__ == "__main__":
    main()
