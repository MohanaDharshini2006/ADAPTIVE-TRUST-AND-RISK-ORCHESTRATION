# ATRO Technical Support Agent (Milestone 1)

This folder currently implements **Milestone 1 only**: dataset checks and a baseline six-category issue classifier for the Technical Support Agent.

The classifier predicts an issue category from customer review text. It does **not** generate a customer reply, retrieve support articles, or talk to the ATRO decision layer yet.

## Setup

Use Python 3.11 or similar. From this project directory:

```bash
pip install -r requirements.txt
```

Your environment already had pandas, scikit-learn, numpy, joblib, and matplotlib during Milestone 1 development. Seaborn is not required.

## Dataset

- File: `data/technical_support_unified_dataset_FINAL.csv`
- Expected size: 1,853 rows (1,800 Amazon reviews + 53 support Q&A records)
- Training uses **Amazon reviews only** (`source == amazon_review`)
- Support Q&A records are reserved for later retrieval work and are **not** used to train this classifier

Amazon issue categories in the CSV:

- `color_fade`
- `fabric_material`
- `fastener_hardware`
- `general_quality_complaint`
- `sizing_fit`
- `stitching_durability`

## Run Milestone 1

From the project root:

```bash
python src/train_issue_classifier.py
```

What the script does:

1. Loads and verifies the CSV (row counts, sources, six Amazon categories).
2. Splits Amazon reviews 80% train / 20% test, stratified by category, `random_state=42`.
3. Fits TF-IDF (`ngram_range=(1, 2)`) and Logistic Regression on **training data only**.
4. Evaluates on the held-out test set.
5. Saves the model and result files.

Expected console output includes dataset verification, train/test sizes (1,440 / 360 if the 1,800 Amazon rows split 80:20), metric values, a per-class classification report, and paths of saved files.

## Output files

| File | Contents |
|------|----------|
| `models/tfidf_logreg_issue_classifier.joblib` | Trained scikit-learn pipeline (TF-IDF + Logistic Regression) |
| `results/classification_metrics.json` | Accuracy, macro precision, macro recall, macro-F1, weighted-F1 |
| `results/classification_metrics.csv` | Same metrics in one CSV row |
| `results/confusion_matrix.csv` | Test-set confusion matrix counts |
| `results/confusion_matrix.png` | Confusion matrix plot |

The original dataset under `data/` is not modified. `notebooks/` is unused in this milestone.

## Milestone 2 — semantic retrieval

Milestone 2 adds a **search component** over the 53 `support_qa` records only. Amazon reviews are not indexed. The issue classifier is not called or retrained.

Longer design notes: `docs/ARCHITECTURE.md`.

### Run Milestone 2

```bash
pip install -r requirements.txt
python tests/test_milestone2.py
python -m src.retrieval.build_index
python src/run_retrieval_eval.py
```

Expected: unit checks pass; 53 FAQ embeddings are saved under `models/retrieval/`; ten sample queries print top-5 IDs and scores; gap queries are labeled `insufficient`; `response` stays null.

### Milestone 2 outputs

| File | Contents |
|------|----------|
| `models/retrieval/embeddings.npy` | FAQ vectors |
| `models/retrieval/knowledge_base.json` | FAQ metadata and provenance |
| `models/retrieval/index_meta.json` | Embedder name and index settings |
| `results/retrieval_sample_output.json` | Full agent JSON for each sample query |
| `results/retrieval_eval.json` | Hit@5 / MRR / Precision on answerable queries |

Similarity means “these FAQs look related.” It does not mean the FAQ is correct, complete, or official policy. This corpus has no dedicated FAQs for zippers, color fade, stitching repairs, or refunds.

## Limitations 

- Amazon category labels were created with **keyword rules**, so some reviews may be tagged incorrectly. High accuracy can partly mean “the model learned the same keywords,” not that it understands products.
- The 1,800 reviews are a **balanced sample** (300 per class), not real complaint frequencies.
- Reviews describe customer experience. They are **not** official troubleshooting steps or store policy.
- The 53 Q&A rows are a small retrieval corpus for later milestones, not a complete knowledge base.
- This baseline does not detect hallucinations, score evidence quality, or produce ATRO actions (`ACCEPT` / `VERIFY` / `RE-ROUTE` / `ESCALATE`).
