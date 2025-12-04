# scripts/build_b_index.py
"""
Build embeddings + FAISS index for B-side products.

Input:
    data/grocery_store_b_clean_phase2.csv
        - must contain column: text_for_embedding

Output (in artifacts/):
    artifacts/b_embeddings.npy        # normalized embedding matrix (N, D)
    artifacts/b_index.faiss           # FAISS index for fast similarity search
    artifacts/b_metadata.csv          # mapping from row index -> product info

Run:
    python scripts/build_b_index.py
"""

from __future__ import annotations

import os

import faiss
import numpy as np
import pandas as pd

from src.embeddings import compute_embeddings_for_texts


DATA_PATH = os.path.join("data", "grocery_store_b_clean_phase2.csv")
ARTIFACTS_DIR = "artifacts"

EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "b_embeddings.npy")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "b_index.faiss")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "b_metadata.csv")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load B-side clean data
    # -------------------------------------------------------------------------
    print(f"[Build B Index] Loading B data from: {DATA_PATH}")
    df_b = pd.read_csv(DATA_PATH)

    if "text_for_embedding" not in df_b.columns:
        raise ValueError(
            "Column `text_for_embedding` is missing in B-side data. "
            "Please run `run_preprocess_phase2.py` first."
        )

    texts = df_b["text_for_embedding"].astype(str).tolist()
    print(f"[Build B Index] Number of B products: {len(texts)}")

    # -------------------------------------------------------------------------
    # Compute embeddings
    # -------------------------------------------------------------------------
    embeddings = compute_embeddings_for_texts(texts, batch_size=64)
    if embeddings.shape[0] != len(df_b):
        raise RuntimeError(
            f"Embedding row count mismatch: got {embeddings.shape[0]}, "
            f"but B dataframe has {len(df_b)} rows."
        )

    # -------------------------------------------------------------------------
    # Normalize embeddings (for cosine similarity) & save them
    # -------------------------------------------------------------------------
    print("[Build B Index] Normalizing embeddings (L2) ...")
    faiss.normalize_L2(embeddings)

    print(f"[Build B Index] Saving normalized embeddings to: {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH, embeddings)

    # -------------------------------------------------------------------------
    # Build FAISS index (Inner Product = cosine similarity after L2 norm)
    # -------------------------------------------------------------------------
    n_items, dim = embeddings.shape
    print(f"[Build B Index] Building FAISS IndexFlatIP with dim={dim} ...")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # add all B-side embeddings to the index
    print(f"[Build B Index] Index contains {index.ntotal} vectors.")

    print(f"[Build B Index] Saving FAISS index to: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    # -------------------------------------------------------------------------
    # Save metadata  map FAISS row index -> product info
    # -------------------------------------------------------------------------
    print(f"[Build B Index] Saving metadata to: {METADATA_PATH}")

    # We keep only the most useful columns here; add more if needed.
    meta_cols = [
        "bb_id",
        "side",
        "customer_item_id",
        "name",
        "title_normalized_l0",
        "brand_final",
        "product_core_final",
        "flavor_final",
        "category_lvl1_final",
        "text_for_embedding",
    ]
    meta_cols = [c for c in meta_cols if c in df_b.columns]

    df_meta = df_b[meta_cols].copy()
    df_meta["b_row_index"] = np.arange(len(df_meta), dtype="int32")

    df_meta.to_csv(METADATA_PATH, index=False)

    print("[Build B Index] Done.")
    print(f"  - Embeddings: {EMBEDDINGS_PATH}")
    print(f"  - FAISS index: {INDEX_PATH}")
    print(f"  - Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
