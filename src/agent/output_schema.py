"""ATRO-facing output contract for the Technical Support Agent.

Unimplemented milestone fields stay null so teammates can already depend
on the shape of this object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.knowledge.schema import RetrievalResult


@dataclass
class TechnicalSupportAgentOutput:
    agent_id: str
    query: str
    predicted_category: str | None
    category_confidence: float | None
    response: str | None
    evidence: list[dict[str, Any]]
    evidence_status: str
    evidence_notes: str
    hallucination_check: dict[str, Any] | None
    evidence_quality: dict[str, Any] | None
    trust_signal: float | None
    risk_signal: float | None
    conflict_signal: dict[str, Any] | None
    recommended_action: str | None
    embedder_name: str
    corpus_size: int
    top_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "query": self.query,
            "predicted_category": self.predicted_category,
            "category_confidence": self.category_confidence,
            "response": self.response,
            "evidence": self.evidence,
            "evidence_status": self.evidence_status,
            "evidence_notes": self.evidence_notes,
            "hallucination_check": self.hallucination_check,
            "evidence_quality": self.evidence_quality,
            "trust_signal": self.trust_signal,
            "risk_signal": self.risk_signal,
            "conflict_signal": self.conflict_signal,
            "recommended_action": self.recommended_action,
            "embedder_name": self.embedder_name,
            "corpus_size": self.corpus_size,
            "top_k": self.top_k,
        }

    @classmethod
    def from_retrieval(
        cls,
        retrieval: RetrievalResult,
    ) -> TechnicalSupportAgentOutput:
        """Fill only retrieval fields. Later milestones add response and checks."""
        return cls(
            agent_id="technical_support",
            query=retrieval.query,
            predicted_category=None,
            category_confidence=None,
            response=None,
            evidence=[hit.to_dict() for hit in retrieval.hits],
            evidence_status=retrieval.evidence_status,
            evidence_notes=retrieval.evidence_notes,
            hallucination_check=None,
            evidence_quality=None,
            trust_signal=None,
            risk_signal=None,
            conflict_signal=None,
            recommended_action=None,
            embedder_name=retrieval.embedder_name,
            corpus_size=retrieval.corpus_size,
            top_k=retrieval.top_k,
        )
