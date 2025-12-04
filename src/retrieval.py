from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

A_CLEAN_PATH = Path("data/processed/A_clean.csv")
B_INDEX_DIR = Path("artifacts/faiss_v0")
TOPK_OUTPUT_PATH = Path("artifacts/faiss_v0/topk_a_to_b.csv")


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load the same embedding model that was used for building the B index.

    It is critical that A and B use the *same* model, otherwise the
    vector space will not be comparable and FAISS search will be meaningless.
    """
    print(f"[INFO] Loading embedding model for retrieval: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def load_b_index_and_ids(
    index_dir: Path = B_INDEX_DIR,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Load the FAISS index and the mapping from index position -> B bb_id.

    Parameters
    ----------
    index_dir : Path
        Directory where `b_index.faiss` and `b_bb_ids.npy` are stored.

    Returns
    -------
    index : faiss.Index
        Loaded FAISS index ready for similarity search.
    b_ids : np.ndarray
        Array of BetterBasket bb_ids aligned with index rows, i.e.
        b_ids[i] is the bb_id of the i-th vector stored in the index.
    """
    index_path = index_dir / "b_index.faiss"
    ids_path = index_dir / "b_bb_ids.npy"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"B bb_id mapping not found at: {ids_path}")

    print(f"[INFO] Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"[INFO] Loading B bb_id mapping from {ids_path}")
    b_ids = np.load(ids_path)

    if index.ntotal != len(b_ids):
        raise ValueError(
            f"Index size ({index.ntotal}) does not match number of B ids ({len(b_ids)})"
        )

    return index, b_ids


def build_a_embeddings_and_search_topk(
    a_csv_path: Path = A_CLEAN_PATH,
    text_col: str = "text_for_embedding",
    id_col: str = "bb_id",
    index_dir: Path = B_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = 50,
    batch_size: int = 256,
    output_path: Path = TOPK_OUTPUT_PATH,
) -> None:
    """
    For each A product, compute embeddings and retrieve top-K candidates from B.

    This function:
        1. Reads cleaned A data (A_clean.csv).
        2. Encodes `text_for_embedding` using the same model as B.
        3. Normalizes A embeddings for cosine similarity.
        4. Searches the B FAISS index for top-K nearest neighbors.
        5. Writes a long-format CSV containing candidate pairs and scores.

    The output CSV has one row per (A, B_candidate) pair:

        bb_id_A, bb_id_B, rank, sim_embedding

    where:
        - bb_id_A : BetterBasket id for store A product
        - bb_id_B : BetterBasket id for store B product
        - rank    : 1..K (1 = most similar according to embedding)
        - sim_embedding : cosine similarity score in [-1, 1]
    """
    # -------------------------------------------------------------
    #  Load cleaned A data
    # -------------------------------------------------------------
    if not a_csv_path.exists():
        raise FileNotFoundError(f"A_clean file not found at: {a_csv_path}")

    df_a = pd.read_csv(a_csv_path)

    if text_col not in df_a.columns:
        raise KeyError(f"Column '{text_col}' not found in {a_csv_path}")
    if id_col not in df_a.columns:
        raise KeyError(f"Column '{id_col}' not found in {a_csv_path}")

    a_texts: List[str] = df_a[text_col].fillna("").astype(str).tolist()
    a_ids = df_a[id_col].tolist()
    num_a = len(a_texts)

    print(f"[INFO] Number of A products: {num_a}")

    # -------------------------------------------------------------
    #  Load embedding model and B index
    # -------------------------------------------------------------
    model = load_embedding_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {dim}")

    index, b_ids = load_b_index_and_ids(index_dir)
    print(f"[INFO] B index size (ntotal): {index.ntotal}")

    # -------------------------------------------------------------
    #  Encode A products in batches and search top-K
    # -------------------------------------------------------------
    all_rows = []

    for start in range(0, num_a, batch_size):
        end = min(start + batch_size, num_a)
        batch_texts = a_texts[start:end]
        batch_ids_a = a_ids[start:end]

        print(f"[INFO] Encoding & searching batch {start} ~ {end}")

        # Encode current batch of A texts
        batch_embs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        # Normalize for cosine similarity 
        faiss.normalize_L2(batch_embs)

        # Perform top-K similarity search against B index
        # distances: (batch_size, top_k) similarity scores
        # indices  : (batch_size, top_k) index positions in B index
        distances, indices = index.search(batch_embs, top_k)

        # Convert results into long-form rows
        for i, a_bb_id in enumerate(batch_ids_a):
            sims = distances[i]
            idxs = indices[i]

            for rank, (sim, b_idx) in enumerate(zip(sims, idxs), start=1):
                b_bb_id = b_ids[b_idx]
                all_rows.append(
                    {
                        "bb_id_A": a_bb_id,
                        "bb_id_B": b_bb_id,
                        "rank": rank,
                        "sim_embedding": float(sim),
                    }
                )

    # -------------------------------------------------------------
    #  Save all candidate pairs to CSV
    # -------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_path, index=False)

    print(f"[OK] Saved A→B top-{top_k} candidates to {output_path}")
    print(f"[OK] Total candidate pairs: {len(df_out)}")
