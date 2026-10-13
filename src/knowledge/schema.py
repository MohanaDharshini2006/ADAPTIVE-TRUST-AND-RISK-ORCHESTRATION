"""Reusable knowledge-document schema for Technical Support retrieval.

These fields stay the same whether the text comes from the current prototype
FAQs or from manuals and product specs added later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VerificationMetadata:
    """How trustworthy a source is. Prototype FAQs are not official policy."""

    status: str
    is_official_policy: bool
    is_verified_troubleshooting: bool
    origin: str
    notes: str
    verified_by: str | None = None
    verified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeDocument:
    """One searchable support document with provenance."""

    doc_id: str
    source_id: str
    source_type: str
    category: str | None
    title: str | None
    text: str
    question: str | None
    answer: str | None
    source_url: str | None
    verification: VerificationMetadata
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class RetrievalHit:
    """One ranked candidate. Similarity is not proof the answer is correct."""

    rank: int
    similarity_score: float
    document: KnowledgeDocument

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "similarity_score": self.similarity_score,
            "document": self.document.to_dict(),
        }


@dataclass
class RetrievalResult:
    """Top-k candidates plus an evidence-status flag."""

    query: str
    top_k: int
    embedder_name: str
    corpus_size: int
    hits: list[RetrievalHit]
    evidence_status: str
    evidence_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "embedder_name": self.embedder_name,
            "corpus_size": self.corpus_size,
            "hits": [hit.to_dict() for hit in self.hits],
            "evidence_status": self.evidence_status,
            "evidence_notes": self.evidence_notes,
        }
