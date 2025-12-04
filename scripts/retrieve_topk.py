# scripts/retrieve_topk.py
"""
Step 3 of Phase 2:
Compute embeddings for A-side products and retrieve top-K candidates
from the B-side FAISS index.

Inputs:
    data/grocery_store_a_clean_phase2.csv
        - must contain column: text_for_embedding

    artifacts/b_index.faiss
    artifacts/b_metadata.csv

Outputs:
    artifacts/a_embeddings.npy       # normalized A-side embeddings (N_A, D)
    artifacts/candidates_topk.parquet  # flattened candidate pairs for A × topK

Run:
    python scripts/retrieve_topk.py
"""

from __future__ import annotations

import os
import sys

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embeddings import compute_embeddings_for_texts  # noqa: E402

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
DATA_A_PATH = os.path.join("data", "grocery_store_a_clean_phase2.csv")

ARTIFACTS_DIR = "artifacts"
A_EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "a_embeddings.npy")

B_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "b_index.faiss")
B_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "b_metadata.csv")

CANDIDATES_PATH_PARQUET = os.path.join(ARTIFACTS_DIR, "candidates_topk.parquet")
CANDIDATES_PATH_CSV = os.path.join(ARTIFACTS_DIR, "candidates_topk.csv")


TOP_K = 50


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load A-side data
    # ------------------------------------------------------------------
    print(f"[Retrieve TopK] Loading A data from: {DATA_A_PATH}")
    df_a = pd.read_csv(DATA_A_PATH)

    if "text_for_embedding" not in df_a.columns:
        raise ValueError(
            "Column `text_for_embedding` is missing in A-side data. "
            "Please run `run_preprocess_phase2.py` first."
        )

    print(f"[Retrieve TopK] Number of A products: {len(df_a)}")

    df_a = df_a.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Compute embeddings for A-side products
    # ------------------------------------------------------------------
    texts_a = df_a["text_for_embedding"].astype(str).tolist()
    embeddings_a = compute_embeddings_for_texts(texts_a, batch_size=128)

    if embeddings_a.shape[0] != len(df_a):
        raise RuntimeError(
            f"Embedding row count mismatch: got {embeddings_a.shape[0]}, "
            f"but A dataframe has {len(df_a)} rows."
        )

    print("[Retrieve TopK] Normalizing A embeddings (L2) ...")
    faiss.normalize_L2(embeddings_a)

    print(f"[Retrieve TopK] Saving A embeddings to: {A_EMBEDDINGS_PATH}")
    np.save(A_EMBEDDINGS_PATH, embeddings_a)

    # ------------------------------------------------------------------
    # Load B-side FAISS index + metadata
    # ------------------------------------------------------------------
    print(f"[Retrieve TopK] Loading FAISS index from: {B_INDEX_PATH}")
    if not os.path.exists(B_INDEX_PATH):
        raise FileNotFoundError(
            f"Cannot find B index at {B_INDEX_PATH}. "
            f"Please run `python scripts/build_b_index.py` first."
        )

    index = faiss.read_index(B_INDEX_PATH)

    print(f"[Retrieve TopK] Loading B metadata from: {B_METADATA_PATH}")
    df_b_meta = pd.read_csv(B_METADATA_PATH)
    df_b_meta = df_b_meta.reset_index(drop=True)

    if "b_row_index" not in df_b_meta.columns:
        raise ValueError(
            "Column `b_row_index` is missing in B metadata. "
            "It should be created in build_b_index.py."
        )

    n_a, emb_dim = embeddings_a.shape
    print(f"[Retrieve TopK] A embeddings shape: {embeddings_a.shape}")
    print(f"[Retrieve TopK] B index size: {index.ntotal}, dim: {index.d}")

    if index.d != emb_dim:
        raise RuntimeError(
            f"FAISS index dimension mismatch: index.d = {index.d}, "
            f"but A embeddings have dim = {emb_dim}."
        )

    # ------------------------------------------------------------------
    # Search top-K for each A product
    # ------------------------------------------------------------------
    print(f"[Retrieve TopK] Searching top-{TOP_K} candidates for each A product ...")
    # similarities: shape (N_A, TOP_K)
    similarities, b_indices = index.search(embeddings_a, TOP_K)

    # ------------------------------------------------------------------
    # Flatten results to a long DataFrame
    # ------------------------------------------------------------------
    print("[Retrieve TopK] Building candidates DataFrame ...")

    # indices:
    #   a_row_index: 0..N_A-1, repeated TOP_K times
    #   b_row_index: taken from FAISS search result
    a_row_index = np.repeat(np.arange(n_a, dtype="int32"), TOP_K)
    b_row_index = b_indices.reshape(-1).astype("int32")
    sim_flat = similarities.reshape(-1).astype("float32")

    candidates = pd.DataFrame(
        {
            "a_row_index": a_row_index,
            "b_row_index": b_row_index,
            "similarity_embedding": sim_flat,
        }
    )

    def safe_lookup(series: pd.Series, idx: np.ndarray):
        # Helper to avoid index alignment issues
        return series.iloc[idx].to_numpy()

    # A-side info
    if "bb_id" in df_a.columns:
        candidates["a_bb_id"] = safe_lookup(df_a["bb_id"], a_row_index)
    if "side" in df_a.columns:
        candidates["a_side"] = safe_lookup(df_a["side"], a_row_index)
    if "customer_item_id" in df_a.columns:
        candidates["a_customer_item_id"] = safe_lookup(
            df_a["customer_item_id"], a_row_index
        )
    if "title_normalized_l0" in df_a.columns:
        candidates["a_title"] = safe_lookup(df_a["title_normalized_l0"], a_row_index)

    # B-side info (from metadata)
    if "bb_id" in df_b_meta.columns:
        candidates["b_bb_id"] = safe_lookup(df_b_meta["bb_id"], b_row_index)
    if "side" in df_b_meta.columns:
        candidates["b_side"] = safe_lookup(df_b_meta["side"], b_row_index)
    if "customer_item_id" in df_b_meta.columns:
        candidates["b_customer_item_id"] = safe_lookup(
            df_b_meta["customer_item_id"], b_row_index
        )
    if "title_normalized_l0" in df_b_meta.columns:
        candidates["b_title"] = safe_lookup(
            df_b_meta["title_normalized_l0"], b_row_index
        )

    print(f"[Retrieve TopK] Candidates shape: {candidates.shape}")

    # ------------------------------------------------------------------
    # Save candidates 
    # ------------------------------------------------------------------
    # Try Parquet first (if pyarrow or fastparquet is installed)
    saved = False
    try:
        candidates.to_parquet(CANDIDATES_PATH_PARQUET, index=False)
        print(f"[Retrieve TopK] Saved candidates to Parquet: {CANDIDATES_PATH_PARQUET}")
        saved = True
    except Exception as e:
        print(f"[Retrieve TopK] Failed to save Parquet ({e}), will fall back to CSV.")

    if not saved:
        candidates.to_csv(CANDIDATES_PATH_CSV, index=False)
        print(f"[Retrieve TopK] Saved candidates to CSV: {CANDIDATES_PATH_CSV}")

    print("[Retrieve TopK] Done.")


if __name__ == "__main__":
    main()
