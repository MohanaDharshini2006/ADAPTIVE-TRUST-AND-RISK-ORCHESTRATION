"""Build and save the support Q&A retrieval index.

Run from the project root:
    python -m src.retrieval.build_index
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.loaders import load_support_qa_documents
from src.retrieval.embedders import DEFAULT_EMBEDDER_NAME, MiniLMEmbedder
from src.retrieval.index import CosineVectorIndex

DATA_PATH = PROJECT_ROOT / "data" / "technical_support_unified_dataset_FINAL.csv"
INDEX_DIR = PROJECT_ROOT / "models" / "retrieval"


def build_and_save_index(
    csv_path: Path = DATA_PATH,
    index_dir: Path = INDEX_DIR,
) -> CosineVectorIndex:
    documents = load_support_qa_documents(csv_path)
    embedder = MiniLMEmbedder()
    texts = [document.text for document in documents]
    print(f"Embedding {len(texts)} documents with {embedder.name} ...")
    vectors = embedder.encode(texts)

    index = CosineVectorIndex()
    index.add([document.doc_id for document in documents], vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / "embeddings.npy"
    knowledge_path = index_dir / "knowledge_base.json"
    meta_path = index_dir / "index_meta.json"

    np.save(embeddings_path, vectors)
    knowledge_path.write_text(
        json.dumps([document.to_dict() for document in documents], indent=2),
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(
            {
                "embedder_name": DEFAULT_EMBEDDER_NAME,
                "embedding_input": "question_plus_answer",
                "index_type": "cosine_exact",
                "corpus_source_filter": "support_qa",
                "corpus_size": len(documents),
                "vector_dimension": int(vectors.shape[1]),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_file": csv_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved embeddings: {embeddings_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved knowledge base: {knowledge_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved index metadata: {meta_path.relative_to(PROJECT_ROOT)}")
    return index


def main() -> None:
    build_and_save_index()
    print("Milestone 2 index build finished.")


if __name__ == "__main__":
    main()
