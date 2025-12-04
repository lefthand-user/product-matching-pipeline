# src/embeddings.py
"""
Utility functions for computing text embeddings using a local
sentence-transformers model (no external API required).

This module is shared by Phase 2 scripts:
    - build_b_index.py
    - retrieve_topk.py
    - (optional) scoring / analysis scripts

We use a general-purpose English semantic model by default:
    "sentence-transformers/all-MiniLM-L6-v2"

You can switch to another model (e.g. BGE) by changing MODEL_NAME below.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily load and cache the sentence-transformers model.

    The first call will download the model (if not cached on disk yet),
    which may take a few minutes. Subsequent calls are fast.
    """
    global _MODEL
    if _MODEL is None:
        print(f"[Embeddings] Loading sentence-transformers model: {MODEL_NAME}")
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def compute_embeddings_for_texts(
    texts: Iterable[str],
    batch_size: int = 128,
) -> np.ndarray:
    """
    Compute embeddings for a list/iterable of texts using a local model.

    Args:
        texts: Iterable of input strings (one per product).
        batch_size: Number of texts to encode in each forward pass.

    Returns:
        A NumPy array of shape (N, D), where:
            N = number of texts
            D = embedding dimension of the chosen model.
    """
    model = _get_model()

    texts_list: List[str] = [str(t) if t is not None else "" for t in texts]
    if not texts_list:
        return np.zeros((0, 0), dtype="float32")

    print(f"[Embeddings] Encoding {len(texts_list)} texts with batch_size={batch_size} ...")

    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype("float32")

    print(f"[Embeddings] Final embedding matrix shape: {embeddings.shape}")
    return embeddings
