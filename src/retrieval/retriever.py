"""Retrieve ranked FAQ candidates for a customer query.

This module does not generate a reply and does not use the issue classifier.
"""

from __future__ import annotations

from src.knowledge.schema import KnowledgeDocument, RetrievalHit, RetrievalResult
from src.retrieval.embedders import Embedder
from src.retrieval.index import VectorIndex


class SupportRetriever:
    def __init__(
        self,
        documents: list[KnowledgeDocument],
        embedder: Embedder,
        index: VectorIndex,
    ) -> None:
        if not documents:
            raise ValueError("Cannot build a retriever with an empty document list.")
        self.documents = {document.doc_id: document for document in documents}
        self.embedder = embedder
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        known_knowledge_gap: bool = False,
    ) -> RetrievalResult:
        """Return up to top_k candidates. Hits are not assumed to be valid answers."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embedder.encode([query])[0]
        neighbors = self.index.search(query_vector, top_k=top_k)

        hits: list[RetrievalHit] = []
        for rank, (doc_id, score) in enumerate(neighbors, start=1):
            document = self.documents[doc_id]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    similarity_score=score,
                    document=document,
                )
            )

        evidence_status, evidence_notes = self._evidence_status(
            hits=hits,
            known_knowledge_gap=known_knowledge_gap,
        )

        return RetrievalResult(
            query=query,
            top_k=top_k,
            embedder_name=self.embedder.name,
            corpus_size=self.index.size(),
            hits=hits,
            evidence_status=evidence_status,
            evidence_notes=evidence_notes,
        )

    def _evidence_status(
        self,
        hits: list[RetrievalHit],
        known_knowledge_gap: bool,
    ) -> tuple[str, str]:
        """Option B: flag empty results and known gaps. No similarity cutoff yet."""
        if not hits:
            return (
                "none",
                "No documents were retrieved. The knowledge base is empty or search returned nothing.",
            )
        if known_knowledge_gap:
            return (
                "insufficient",
                "This query is a known knowledge-base gap (for example zipper/hardware, "
                "color fade, stitching, or refund/return). Candidate FAQs may still be "
                "listed for inspection, but similarity is not treated as a valid answer.",
            )
        return (
            "candidate",
            "Returned the nearest prototype FAQs by cosine similarity. "
            "Similarity is a relevance clue, not proof of correctness or official policy.",
        )
