"""Vector index backends.

Milestone 2 uses exact cosine similarity. That is enough for 53 documents
and keeps ranking independent of FAISS.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        ...

    def size(self) -> int:
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each vector so cosine similarity becomes a simple dot product."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


class CosineVectorIndex:
    """Stores document vectors and returns the nearest neighbors by cosine similarity."""

    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if len(doc_ids) != len(vectors):
            raise ValueError("doc_ids and vectors must have the same length.")
        normalized = _l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._doc_ids = list(doc_ids)
        self._vectors = normalized

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self._doc_ids:
            return []
        normalized_query = _l2_normalize(np.asarray(query_vector, dtype=np.float32))
        scores = (self._vectors @ normalized_query.T).ravel()
        k = min(top_k, len(self._doc_ids))
        # argpartition is faster than a full sort; we then sort the shortlist.
        top_indices = np.argpartition(scores, -k)[-k:]
        ordered = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self._doc_ids[i], float(scores[i])) for i in ordered]

    def size(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors
