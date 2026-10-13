"""Embedding backends.

The retriever depends on this small interface, not on MiniLM itself.
A later milestone could swap in another local or API embedder.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that turns a list of strings into a 2D numpy array."""

    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class MiniLMEmbedder:
    """Local sentence embedding model. No Groq or other API calls."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
