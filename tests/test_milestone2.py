"""Lightweight Milestone 2 checks that do not call Groq or the classifier.

Run from the project root:
    python tests/test_milestone2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.loaders import EXPECTED_SUPPORT_QA_ROWS, load_support_qa_documents
from src.knowledge.schema import KnowledgeDocument, RetrievalHit, VerificationMetadata
from src.agent.technical_support_agent import TechnicalSupportAgent
from src.agent.output_schema import TechnicalSupportAgentOutput
from src.retrieval.index import CosineVectorIndex
from src.retrieval.retriever import SupportRetriever
import numpy as np

CSV_PATH = PROJECT_ROOT / "data" / "technical_support_unified_dataset_FINAL.csv"
EVAL_PATH = PROJECT_ROOT / "eval" / "retrieval_sample_queries.json"
CLASSIFIER_MODEL = PROJECT_ROOT / "models" / "tfidf_logreg_issue_classifier.joblib"
MILESTONE1_RESULTS = [
    PROJECT_ROOT / "results" / "classification_metrics.json",
    PROJECT_ROOT / "results" / "classification_metrics.csv",
    PROJECT_ROOT / "results" / "confusion_matrix.csv",
]


def test_loader_excludes_amazon_and_marks_unverified() -> None:
    documents = load_support_qa_documents(CSV_PATH)
    assert len(documents) == EXPECTED_SUPPORT_QA_ROWS
    assert all(document.source_type == "support_qa" for document in documents)
    assert all(document.verification.status == "unverified" for document in documents)
    assert all(document.verification.is_official_policy is False for document in documents)
    assert all(document.verification.origin == "prototype_faq" for document in documents)
    assert all(document.question and document.answer for document in documents)
    assert all(document.question in document.text and document.answer in document.text for document in documents)
    assert all(document.source_url is None for document in documents)
    print("PASS test_loader_excludes_amazon_and_marks_unverified")


def test_eval_file_has_answerable_and_gap_queries() -> None:
    spec = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    coverages = {item["expected_coverage"] for item in spec["queries"]}
    assert "answerable" in coverages
    assert "gap" in coverages
    print("PASS test_eval_file_has_answerable_and_gap_queries")


def test_empty_index_is_visible() -> None:
    """Empty search results must be flagged as none, not filled with a fake answer."""

    class DummyEmbedder:
        name = "dummy"
        dimension = 2

        def encode(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32)

    document = KnowledgeDocument(
        doc_id="demo",
        source_id="test",
        source_type="support_qa",
        category="Support",
        title="Demo",
        text="Demo question Demo answer",
        question="Demo question",
        answer="Demo answer",
        source_url=None,
        verification=VerificationMetadata(
            status="unverified",
            is_official_policy=False,
            is_verified_troubleshooting=False,
            origin="prototype_faq",
            notes="test",
        ),
    )
    index = CosineVectorIndex()
    retriever = SupportRetriever([document], DummyEmbedder(), index)
    # Index was never populated, so search returns nothing.
    result = retriever.retrieve("hello", top_k=5)
    assert result.evidence_status == "none"
    assert result.hits == []
    output = TechnicalSupportAgentOutput.from_retrieval(result)
    assert output.response is None
    print("PASS test_empty_index_is_visible")


def test_gap_flag_does_not_invent_a_response() -> None:
    class DummyEmbedder:
        name = "dummy"
        dimension = 2

        def encode(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32)

    document = KnowledgeDocument(
        doc_id="demo",
        source_id="test",
        source_type="support_qa",
        category="Support",
        title="Demo",
        text="Demo question Demo answer",
        question="Demo question",
        answer="Demo answer",
        source_url=None,
        verification=VerificationMetadata(
            status="unverified",
            is_official_policy=False,
            is_verified_troubleshooting=False,
            origin="prototype_faq",
            notes="test",
        ),
    )
    index = CosineVectorIndex()
    index.add(["demo"], np.array([[1.0, 0.0]], dtype=np.float32))
    retriever = SupportRetriever([document], DummyEmbedder(), index)
    result = retriever.retrieve("zipper broke", top_k=5, known_knowledge_gap=True)
    assert result.evidence_status == "insufficient"
    output = TechnicalSupportAgentOutput.from_retrieval(result)
    assert output.response is None
    assert output.predicted_category is None
    assert output.recommended_action is None
    print("PASS test_gap_flag_does_not_invent_a_response")


def test_technical_support_agent_valid_stitching_query_reaches_retrieval() -> None:
    class DummyRetriever:
        def retrieve(self, query, top_k=5, known_knowledge_gap=False):
            assert query == "The stitching on my product is coming apart."
            assert top_k == 3
            assert known_knowledge_gap is False
            return type(
                "RetrievalStub",
                (),
                {
                    "query": query,
                    "top_k": top_k,
                    "embedder_name": "dummy",
                    "corpus_size": 1,
                    "hits": [],
                    "evidence_status": "candidate",
                    "evidence_notes": "candidate retrieval",
                },
            )()

    agent = TechnicalSupportAgent(retriever=DummyRetriever())
    output = agent.handle_query(
        "The stitching on my product is coming apart.",
        top_k=3,
        known_knowledge_gap=False,
    )

    assert output.query == "The stitching on my product is coming apart."
    assert output.response is None
    assert output.predicted_category is None
    assert output.evidence_status == "candidate"
    print("PASS test_technical_support_agent_valid_stitching_query_reaches_retrieval")


def test_technical_support_agent_rejects_blank_or_invalid_inputs_before_retrieval() -> None:
    retrieval_calls = []

    class DummyRetriever:
        def retrieve(self, query, top_k=5, known_knowledge_gap=False):
            retrieval_calls.append((query, top_k, known_knowledge_gap))
            raise AssertionError("retrieve() must not be called for blank input")

    agent = TechnicalSupportAgent(retriever=DummyRetriever())

    try:
        agent.handle_query("   ")
        raise AssertionError("Expected ValueError for blank user_message")
    except ValueError:
        pass

    assert retrieval_calls == []

    print("PASS test_technical_support_agent_rejects_blank_or_invalid_inputs_before_retrieval")


def test_technical_support_agent_preserves_output_contract_and_evidence_status() -> None:
    class DummyRetriever:
        def retrieve(self, query, top_k=5, known_knowledge_gap=False):
            document = KnowledgeDocument(
                doc_id="doc-1",
                source_id="qa-1",
                source_type="support_qa",
                category="support",
                title="Stitching issue",
                text="Question: stitches unravel? Answer: inspect seam quality.",
                question="stitches unravel?",
                answer="inspect seam quality.",
                source_url=None,
                verification=VerificationMetadata(
                    status="unverified",
                    is_official_policy=False,
                    is_verified_troubleshooting=False,
                    origin="prototype_faq",
                    notes="test",
                ),
            )
            return type(
                "RetrievalStub",
                (),
                {
                    "query": query,
                    "top_k": top_k,
                    "embedder_name": "dummy",
                    "corpus_size": 42,
                    "hits": [
                        RetrievalHit(
                            rank=1,
                            similarity_score=0.82,
                            document=document,
                        )
                    ],
                    "evidence_status": "candidate",
                    "evidence_notes": "candidate retrieval",
                },
            )()

    agent = TechnicalSupportAgent(retriever=DummyRetriever())
    output = agent.handle_query("The stitching on the seam is failing.")

    assert output.agent_id == "technical_support"
    assert output.query == "The stitching on the seam is failing."
    assert output.predicted_category is None
    assert output.category_confidence is None
    assert output.response is None
    assert output.hallucination_check is None
    assert output.evidence_quality is None
    assert output.trust_signal is None
    assert output.risk_signal is None
    assert output.conflict_signal is None
    assert output.recommended_action is None
    assert output.evidence_status == "candidate"
    assert output.evidence_notes == "candidate retrieval"
    assert output.evidence[0]["rank"] == 1
    assert output.corpus_size == 42
    print("PASS test_technical_support_agent_preserves_output_contract_and_evidence_status")


def test_milestone1_artifacts_still_present() -> None:
    assert CLASSIFIER_MODEL.exists(), "Milestone 1 classifier model is missing."
    for path in MILESTONE1_RESULTS:
        assert path.exists(), f"Missing Milestone 1 result file: {path}"
    print("PASS test_milestone1_artifacts_still_present")


def main() -> None:
    test_loader_excludes_amazon_and_marks_unverified()
    test_eval_file_has_answerable_and_gap_queries()
    test_empty_index_is_visible()
    test_gap_flag_does_not_invent_a_response()
    test_technical_support_agent_valid_stitching_query_reaches_retrieval()
    test_technical_support_agent_rejects_blank_or_invalid_inputs_before_retrieval()
    test_technical_support_agent_preserves_output_contract_and_evidence_status()
    test_milestone1_artifacts_still_present()
    print("All Milestone 2 unit checks passed.")


if __name__ == "__main__":
    main()
