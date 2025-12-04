BetterBasket Product Matching Project

A complete two-phase pipeline for cleaning, embedding-based retrieval, hybrid scoring, and GPT-assisted reranking.


1. Overview
This project implements a production-ready product matching pipeline for two grocery catalogs (Store A and Store B).
It consists of:

Phase 1 — Data Cleaning
Rule-based extraction (L0)
GPT-based semantic normalization (L1)
Deterministic consolidation (L2 → clean_final)

Phase 2 — Matching
Embedding generation (Sentence-Transformers)
FAISS vector search (top-K retrieval)
Feature engineering + weighted scoring
Strict rule filtering
Optional GPT mid-band reranking (for ambiguous matches)
Final result exported as bb_id_A → bb_id_B pairs

You can either:
run the entire pipeline from raw data, or
skip Phase 1 and directly use the pre-cleaned data included in this repository.


2. Installation
Tested with Python 3.9–3.12.
Creating a virtual environment is recommended.

(1). Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# OR
.\.venv\Scripts\activate      # Windows PowerShell

(2). Install dependencies
pip install -r requirements.txt

(3). (Optional) Faster Parquet support
pip install pyarrow

(4). Provide OpenAI credentials
openai_creds.yaml


3. How to Run the Pipeline

There are two official entry scripts depending on whether you wish to re-run GPT cleaning.
OPTION A — Run the Full Pipeline (Phase 1 + Phase 2)
Warning: Phase 1 is slow and expensive
It uses GPT heavily and can take a long time.

Step 1 — Phase 1: Full Data Cleaning
python scripts/run_phase1_all.py
This produces:
data/grocery_store_a_clean_final.csv
data/grocery_store_b_clean_final.csv

Step 2 — Phase 2: Matching
python scripts/run_phase2_matching_all.py
Outputs include:
artifacts/matches_v0.csv — baseline results
artifacts/upc_exact_matches.csv — strong matches via UPC
artifacts/matches_gpt_review.csv — GPT evaluation (mid-band only)
data/Final_matches.csv — final cleaned matches
result/final_matches_bb_ids.csv — deliverable mapping file


OPTION B — Skip Phase 1 (Recommended for fast reproduction)
The repository already includes:
grocery_store_a_clean_final.csv
grocery_store_b_clean_final.csv

So you can directly run:
python scripts/run_phase2_matching_all.py

Toggle GPT reranking
Inside run_phase2_matching_all.py:
ENABLE_GPT_RERANK = True
Set to False if you only want the baseline matcher:
artifacts/matches_v0.csv


4. Project Structure
betterbasket-matching/
├── data/                    # Raw, intermediate, and final cleaned tables
├── artifacts/               # Embeddings, FAISS index, candidates, GPT output
├── result/                  # Final deliverable (bb_id_A → bb_id_B)
│
├── scripts/                 # Pipeline entrypoints
│   ├── run_phase1_all.py
│   ├── run_phase2_matching_all.py
│   │
│   ├── run_preprocess_phase2.py
│   ├── build_b_index.py
│   ├── retrieve_topk.py
│   ├── score_candidates.py
│   ├── build_matches_v0.py
│   ├── build_upc_exact_matches.py
│   ├── select_gpt_band.py
│   ├── gpt_rerank_midband.py
│   ├── build_final_matches.py
│   └── sample_matches.py
│
├── src/
│   ├── embeddings.py        # Sentence-transformers + FAISS utilities
│   ├── retrieval.py         # Vector search logic
│   ├── scoring.py           # Feature engineering & weighted scoring
│   └── preprocess/          # L0/L1/L2 cleaning modules
│
├── openai_creds.yaml        # User-provided GPT credential file (ignored by Git)
├── requirements.txt
└── README.md


5. System Design Summary
(1). Embedding + Vector Retrieval
All products generate a semantic vector via sentence-transformers.
Store B vectors are indexed using FAISS for fast similarity search.
(2). Hybrid Scoring
Matches are evaluated using:
embedding similarity
brand consistency
category agreement
size/volume ratio
strict size outlier detection
(3). Rule Filter
Hard rules prune impossible matches (e.g., mismatched units/categories).
(4). GPT Reranking 
For ambiguous candidates (score 0.70–0.71), GPT evaluates:
Are they exact matches?
Are they reasonable substitutes?
Or should they be rejected?
This step dramatically improves precision with minimal cost.
(5). Deterministic Final Output
GPT-reviewed pairs + deterministic baseline →
Final mapping file:
result/final_matches_bb_ids.csv



6. Final Deliverables
result/final_matches_bb_ids.csv



7. Additional useful artifacts
data/Final_matches.csv — full-featured match records
artifacts/matches_v0.csv — strict rule-based baseline
artifacts/upc_exact_matches.csv — highest-confidence UPC matches
