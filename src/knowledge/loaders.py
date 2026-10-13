"""Load knowledge documents from source files.

Milestone 2 only loads support Q&A rows. Amazon reviews are excluded.
Later loaders can add manuals or specs into the same KnowledgeDocument schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.knowledge.schema import KnowledgeDocument, VerificationMetadata

EXPECTED_SUPPORT_QA_ROWS = 53
PROTOTYPE_FAQ_NOTES = (
    "Prototype FAQ from the project dataset (technical_support_unified_dataset_FINAL.csv). "
    "Not official store policy and not verified troubleshooting."
)


def load_support_qa_documents(csv_path: Path) -> list[KnowledgeDocument]:
    """Turn the 53 support_qa rows into KnowledgeDocument objects."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    required_columns = [
        "doc_id",
        "source",
        "category",
        "query_text",
        "reference_response",
    ]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    support_qa = dataframe[dataframe["source"] == "support_qa"].copy()
    amazon_count = int((dataframe["source"] == "amazon_review").sum())

    if len(support_qa) != EXPECTED_SUPPORT_QA_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SUPPORT_QA_ROWS} support_qa rows, found {len(support_qa)}."
        )

    documents: list[KnowledgeDocument] = []
    for _, row in support_qa.iterrows():
        question = str(row["query_text"]).strip()
        answer = str(row["reference_response"]).strip()
        if not question or not answer or answer.lower() == "nan":
            raise ValueError(f"Empty question or answer for doc_id={row['doc_id']}")

        combined_text = f"{question} {answer}".strip()
        documents.append(
            KnowledgeDocument(
                doc_id=str(row["doc_id"]),
                source_id=csv_path.name,
                source_type="support_qa",
                category=str(row["category"]) if pd.notna(row["category"]) else None,
                title=question,
                text=combined_text,
                question=question,
                answer=answer,
                source_url=None,
                verification=VerificationMetadata(
                    status="unverified",
                    is_official_policy=False,
                    is_verified_troubleshooting=False,
                    origin="prototype_faq",
                    notes=PROTOTYPE_FAQ_NOTES,
                ),
            )
        )

    document_ids = [document.doc_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate doc_id values found in support_qa records.")

    print(
        f"Loaded {len(documents)} support_qa documents "
        f"(excluded {amazon_count} amazon_review rows from the retrieval corpus)."
    )
    return documents
