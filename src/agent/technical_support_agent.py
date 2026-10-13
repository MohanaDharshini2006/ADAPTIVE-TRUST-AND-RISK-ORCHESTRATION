"""Thin Technical Support Agent wrapper for Milestone 2.

This class only runs retrieval and returns the ATRO-facing schema.
It does not classify issues, call Groq, score evidence, or choose ATRO actions.
"""

from __future__ import annotations

from src.agent.output_schema import TechnicalSupportAgentOutput
from src.agent.router_contracts import TechnicalSupportClassifierInput
from src.retrieval.retriever import SupportRetriever


class TechnicalSupportAgent:
    def __init__(self, retriever: SupportRetriever) -> None:
        self.retriever = retriever

    def handle_query(
        self,
        query: str,
        top_k: int = 5,
        known_knowledge_gap: bool = False,
    ) -> TechnicalSupportAgentOutput:
        validated_query = TechnicalSupportClassifierInput(
            user_message=query,
            route_context="technical_support",
        )

        retrieval = self.retriever.retrieve(
            query=validated_query.user_message,
            top_k=top_k,
            known_knowledge_gap=known_knowledge_gap,
        )
        return TechnicalSupportAgentOutput.from_retrieval(retrieval)
