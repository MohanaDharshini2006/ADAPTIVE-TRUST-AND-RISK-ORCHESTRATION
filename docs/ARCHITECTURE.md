# Technical Support Agent architecture

This document describes how the Technical Support Agent is split into modules and how later pieces (Groq, evidence checking, ATRO) should connect. Milestone 2 implements retrieval only.

## What this agent is not

The **shared router** (Billing vs Order vs Return vs Technical Support) is a teammate component. This agent assumes it already received a technical-support query.

The **six-category issue classifier** from Milestone 1 lives *inside* this agent. It labels `sizing_fit`, `color_fade`, and similar issue types. It is **not** the shared router. Milestone 2 does not call or retrain it.

## Pipeline

```text
Shared router (not this milestone)
        |
        v
Technical Support Agent
  1. Issue classifier     (Milestone 1, saved; unused in Milestone 2)
  2. Retriever            (Milestone 2, implemented)
  3. Groq generator       (Milestone 3, not implemented)
  4. Evidence verifier    (Milestone 4, not implemented)
  5. ATRO-facing JSON     (schema exists; most fields null in Milestone 2)
        |
        v
ATRO decision layer (Milestone 5, not implemented)
```

## Module boundaries

| Module | Responsibility | Must not do |
|--------|----------------|-------------|
| `src/knowledge/` | Document schema and loaders | Embed, rank, generate replies |
| `src/retrieval/embedders.py` | Turn text into vectors | Search or call Groq |
| `src/retrieval/index.py` | Store vectors and nearest-neighbor lookup | Know about FAQs or LLMs |
| `src/retrieval/retriever.py` | Rank documents and set evidence_status | Write customer replies |
| `src/agent/` | Call retrieval and emit the ATRO JSON | Router, billing, returns, Groq, ATRO actions |
| `src/train_issue_classifier.py` | Milestone 1 training (frozen) | Retrieval |

## Knowledge document

Every indexed item uses the same fields: `doc_id`, `source_id`, `source_type`, `category`, `title`, `text`, `question`, `answer`, `source_url`, and `verification`.

The current 53 `support_qa` rows are loaded with:

- `verification.status = "unverified"`
- `is_official_policy = false`
- `origin = "prototype_faq"`

Amazon reviews are **not** indexed. Later loaders may add manuals, specs, or care sheets without changing the retriever.

## Retrieval (Milestone 2)

- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (local, no API)
- Index: exact cosine similarity
- Default `top_k = 5`
- Embedded text: question + answer; original question and answer are stored separately
- Similarity is a **relevance clue**, not correctness or official policy

### evidence_status (option B)

- `none`: no hits (empty index or empty search)
- `insufficient`: caller marked a **known knowledge-base gap** (eval set: zipper/hardware, color fade, stitching, refund)
- `candidate`: nearest FAQs were returned; they are not assumed to be valid answers

No numeric similarity cutoff is applied yet. Scores are recorded so a threshold can be calibrated later from manual review.

## ATRO-facing output

`TechnicalSupportAgentOutput` is the contract teammates can plan around:

- Filled in Milestone 2: `query`, `evidence` (with provenance), `evidence_status`, `evidence_notes`, `embedder_name`, `corpus_size`, `top_k`
- Explicitly null in Milestone 2: `predicted_category`, `response`, `hallucination_check`, `evidence_quality`, `trust_signal`, `risk_signal`, `conflict_signal`, `recommended_action`

ATRO, not this agent, should later choose ACCEPT / VERIFY / RE-ROUTE / ESCALATE.

## Future connections (not implemented)

**Groq (Milestone 3)** should receive the same `evidence` list. If `evidence_status` is `none` or `insufficient`, the generator should refuse to invent policy instead of guessing.

**Verifier (Milestone 4)** should compare the generated reply to that same evidence list (SUPPORTED / CONTRADICTED / UNVERIFIED). It should not run a separate undocumented search.

**Classifier (later)** can fill `predicted_category` by loading `models/tfidf_logreg_issue_classifier.joblib` without retraining.

## Frozen Milestone 1 artifacts

Do not overwrite:

- `src/train_issue_classifier.py`
- `models/tfidf_logreg_issue_classifier.joblib`
- `results/classification_metrics.json`
- `results/classification_metrics.csv`
- `results/confusion_matrix.csv`
- `results/confusion_matrix.png`
- `data/technical_support_unified_dataset_FINAL.csv`
