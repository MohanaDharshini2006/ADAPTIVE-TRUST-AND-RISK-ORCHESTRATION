"""Run Milestone 2 retrieval on the manual sample queries.

Run from the project root after building the index:
    python src/run_retrieval_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.technical_support_agent import TechnicalSupportAgent
from src.knowledge.schema import KnowledgeDocument, VerificationMetadata
from src.retrieval.embedders import MiniLMEmbedder
from src.retrieval.index import CosineVectorIndex
from src.retrieval.retriever import SupportRetriever

INDEX_DIR = PROJECT_ROOT / "models" / "retrieval"
EVAL_PATH = PROJECT_ROOT / "eval" / "retrieval_sample_queries.json"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_saved_index(index_dir: Path) -> tuple[list[KnowledgeDocument], np.ndarray, dict]:
    knowledge_path = index_dir / "knowledge_base.json"
    embeddings_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "index_meta.json"
    missing = [path for path in (knowledge_path, embeddings_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Retrieval index not found. Run: python -m src.retrieval.build_index\n"
            f"Missing: {missing}"
        )

    raw_documents = json.loads(knowledge_path.read_text(encoding="utf-8"))
    documents = []
    for item in raw_documents:
        verification = VerificationMetadata(**item["verification"])
        documents.append(
            KnowledgeDocument(
                doc_id=item["doc_id"],
                source_id=item["source_id"],
                source_type=item["source_type"],
                category=item.get("category"),
                title=item.get("title"),
                text=item["text"],
                question=item.get("question"),
                answer=item.get("answer"),
                source_url=item.get("source_url"),
                verification=verification,
                extra=item.get("extra") or {},
            )
        )
    embeddings = np.load(embeddings_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return documents, embeddings, metadata


def hit_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    return float(any(doc_id in gold_ids for doc_id in retrieved_ids[:k]))


def precision_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    relevant = sum(1 for doc_id in top if doc_id in gold_ids)
    return relevant / len(top)


def reciprocal_rank(gold_ids: list[str], retrieved_ids: list[str]) -> float:
    if not gold_ids:
        return float("nan")
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    valid = [value for value in values if value == value]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    spec = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    top_k = int(spec.get("default_top_k", 5))

    documents, embeddings, index_meta = load_saved_index(INDEX_DIR)
    if any(document.source_type != "support_qa" for document in documents):
        raise ValueError("Index contains non-support_qa documents; Amazon reviews must stay out.")

    embedder = MiniLMEmbedder(model_name=index_meta["embedder_name"])
    index = CosineVectorIndex()
    index.add([document.doc_id for document in documents], embeddings)
    agent = TechnicalSupportAgent(SupportRetriever(documents, embedder, index))

    per_query_rows = []
    sample_outputs = []

    for item in spec["queries"]:
        is_gap = item["expected_coverage"] == "gap"
        output = agent.handle_query(
            query=item["query"],
            top_k=top_k,
            known_knowledge_gap=is_gap,
        )
        retrieved_ids = [hit["document"]["doc_id"] for hit in output.evidence]
        gold_ids = item.get("gold_doc_ids") or []

        row = {
            "query_id": item["query_id"],
            "query": item["query"],
            "expected_coverage": item["expected_coverage"],
            "gold_doc_ids": gold_ids,
            "retrieved_doc_ids": retrieved_ids,
            "similarity_scores": [hit["similarity_score"] for hit in output.evidence],
            "evidence_status": output.evidence_status,
            "hit_at_5": hit_at_k(gold_ids, retrieved_ids, 5),
            "precision_at_3": precision_at_k(gold_ids, retrieved_ids, 3),
            "precision_at_5": precision_at_k(gold_ids, retrieved_ids, 5),
            "reciprocal_rank": reciprocal_rank(gold_ids, retrieved_ids),
            "response_is_null": output.response is None,
            "predicted_category_is_null": output.predicted_category is None,
        }
        per_query_rows.append(row)
        sample_outputs.append(output.to_dict())

        print(f"=== {item['query_id']} ({item['expected_coverage']}) ===")
        print(f"Query: {item['query']}")
        print(f"evidence_status: {output.evidence_status}")
        for hit in output.evidence:
            print(
                f"  #{hit['rank']} {hit['document']['doc_id']} "
                f"score={hit['similarity_score']:.4f} "
                f"| {hit['document']['title']}"
            )
        print()

    answerable = [row for row in per_query_rows if row["expected_coverage"] == "answerable"]
    gaps = [row for row in per_query_rows if row["expected_coverage"] == "gap"]
    summary = {
        "embedder_name": index_meta["embedder_name"],
        "corpus_size": len(documents),
        "top_k": top_k,
        "n_answerable_queries": len(answerable),
        "n_gap_queries": len(gaps),
        "answerable_hit_at_5": mean([row["hit_at_5"] for row in answerable]),
        "answerable_mrr": mean([row["reciprocal_rank"] for row in answerable]),
        "answerable_precision_at_3": mean([row["precision_at_3"] for row in answerable]),
        "answerable_precision_at_5": mean([row["precision_at_5"] for row in answerable]),
        "gap_queries_flagged_insufficient": all(
            row["evidence_status"] == "insufficient" for row in gaps
        ),
        "all_responses_null": all(row["response_is_null"] for row in per_query_rows),
        "notes": [
            "Similarity is a relevance clue, not factual correctness or official policy.",
            "No cosine similarity cutoff was applied in Milestone 2.",
            "Gap queries still return candidate FAQs so scores can be inspected later.",
        ],
    }

    print("=== Summary (answerable queries) ===")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("answerable") or k.startswith("n_")}, indent=2))
    print("gap_queries_flagged_insufficient:", summary["gap_queries_flagged_insufficient"])

    output_path = RESULTS_DIR / "retrieval_sample_output.json"
    eval_path = RESULTS_DIR / "retrieval_eval.json"
    output_path.write_text(json.dumps(sample_outputs, indent=2), encoding="utf-8")
    eval_path.write_text(
        json.dumps({"summary": summary, "per_query": per_query_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {eval_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
