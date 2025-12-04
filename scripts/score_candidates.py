# scripts/score_candidates.py
"""
Add features and weighted score to A×B candidate pairs.

Run:
    python scripts/score_candidates.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scoring import add_features_and_weighted_score  # noqa: E402


def main() -> None:
    add_features_and_weighted_score()


if __name__ == "__main__":
    main()
