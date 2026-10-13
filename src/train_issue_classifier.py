"""
Milestone 1: Train and evaluate a six-category technical issue classifier.

This script:
  1. Loads the unified Technical Support dataset.
  2. Verifies row counts, sources, and Amazon issue categories.
  3. Trains TF-IDF (unigrams + bigrams) + Logistic Regression on Amazon reviews only.
  4. Evaluates on a held-out stratified test set (never used during fitting).
  5. Saves the trained pipeline, metrics, and confusion matrix.

Run from the project root:
    python src/train_issue_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Paths are relative to the project root (parent of src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "technical_support_unified_dataset_FINAL.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = MODELS_DIR / "tfidf_logreg_issue_classifier.joblib"

EXPECTED_TOTAL_ROWS = 1853
EXPECTED_AMAZON_ROWS = 1800
EXPECTED_SUPPORT_QA_ROWS = 53
EXPECTED_ROWS_PER_AMAZON_CATEGORY = 300

# Labels stored in the CSV (do not rename the dataset).
AMAZON_CATEGORIES = [
    "color_fade",
    "fabric_material",
    "fastener_hardware",
    "general_quality_complaint",
    "sizing_fit",
    "stitching_durability",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and confirm the expected columns exist."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    expected_columns = [
        "doc_id",
        "source",
        "category",
        "query_text",
        "reference_response",
        "rating",
    ]
    missing_columns = [col for col in expected_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    return dataframe


def verify_dataset(dataframe: pd.DataFrame) -> None:
    """Print checks and stop if Amazon training data does not match expectations."""
    print("=== Dataset verification ===")
    print(f"Total rows: {len(dataframe)} (expected {EXPECTED_TOTAL_ROWS})")
    print("Source counts:")
    print(dataframe["source"].value_counts(dropna=False).to_string())
    print("Category counts (all sources):")
    print(dataframe["category"].value_counts(dropna=False).to_string())

    amazon_reviews = dataframe[dataframe["source"] == "amazon_review"]
    support_qa = dataframe[dataframe["source"] == "support_qa"]

    print(f"Amazon review rows: {len(amazon_reviews)} (expected {EXPECTED_AMAZON_ROWS})")
    print(f"Support Q&A rows: {len(support_qa)} (expected {EXPECTED_SUPPORT_QA_ROWS})")
    print("Amazon category counts:")
    print(amazon_reviews["category"].value_counts().sort_index().to_string())

    issues: list[str] = []
    if len(dataframe) != EXPECTED_TOTAL_ROWS:
        issues.append(
            f"Total row count is {len(dataframe)}, expected {EXPECTED_TOTAL_ROWS}."
        )
    if len(amazon_reviews) != EXPECTED_AMAZON_ROWS:
        issues.append(
            f"Amazon row count is {len(amazon_reviews)}, expected {EXPECTED_AMAZON_ROWS}."
        )
    if len(support_qa) != EXPECTED_SUPPORT_QA_ROWS:
        issues.append(
            f"Support Q&A row count is {len(support_qa)}, expected {EXPECTED_SUPPORT_QA_ROWS}."
        )

    amazon_category_counts = amazon_reviews["category"].value_counts()
    unexpected_categories = sorted(
        set(amazon_reviews["category"].unique()) - set(AMAZON_CATEGORIES)
    )
    missing_categories = sorted(
        set(AMAZON_CATEGORIES) - set(amazon_reviews["category"].unique())
    )
    if unexpected_categories:
        issues.append(f"Unexpected Amazon categories: {unexpected_categories}")
    if missing_categories:
        issues.append(f"Missing Amazon categories: {missing_categories}")
    for category_name in AMAZON_CATEGORIES:
        count = int(amazon_category_counts.get(category_name, 0))
        if count != EXPECTED_ROWS_PER_AMAZON_CATEGORY:
            issues.append(
                f"Category '{category_name}' has {count} rows, "
                f"expected {EXPECTED_ROWS_PER_AMAZON_CATEGORY}."
            )

    empty_queries = (
        amazon_reviews["query_text"].fillna("").astype(str).str.strip().eq("").sum()
    )
    if empty_queries:
        issues.append(f"Found {empty_queries} Amazon reviews with empty query_text.")

    if issues:
        raise ValueError("Dataset verification failed:\n- " + "\n- ".join(issues))

    print("Dataset verification passed. Training will use Amazon reviews only.")
    print()


def build_pipeline() -> Pipeline:
    """TF-IDF unigrams+bigrams followed by multinomial logistic regression."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Compute the Milestone 1 metrics on the test set only."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "n_test_examples": int(len(y_true)),
        "n_classes": int(len(labels)),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: list[str],
    labels: list[str],
    results_dir: Path,
) -> None:
    """Save the confusion matrix as CSV (counts) and PNG (plot)."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_dataframe = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_dataframe.index.name = "true_label"
    csv_path = results_dir / "confusion_matrix.csv"
    matrix_dataframe.to_csv(csv_path)

    figure, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=True, xticks_rotation=45)
    axis.set_title("Issue classifier confusion matrix (test set)")
    figure.tight_layout()
    png_path = results_dir / "confusion_matrix.png"
    figure.savefig(png_path, dpi=150)
    plt.close(figure)

    print(f"Saved confusion matrix CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved confusion matrix PNG: {png_path.relative_to(PROJECT_ROOT)}")


def save_metrics(metrics: dict[str, float], results_dir: Path) -> None:
    """Save metrics as JSON (named values) and CSV (one row)."""
    json_path = results_dir / "classification_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = results_dir / "classification_metrics.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    print(f"Saved metrics JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics CSV: {csv_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(DATA_PATH)
    verify_dataset(dataframe)

    # Support Q&A records are a retrieval corpus, not classifier training data.
    amazon_reviews = dataframe[dataframe["source"] == "amazon_review"].copy()
    features = amazon_reviews["query_text"]
    labels = amazon_reviews["category"]

    # Stratified split keeps each category at about 80% train / 20% test.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    print("=== Split sizes ===")
    print(f"Train examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")
    print("Train category counts:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test category counts:")
    print(y_test.value_counts().sort_index().to_string())
    print()

    pipeline = build_pipeline()
    print("Fitting TF-IDF and Logistic Regression on training data only...")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    ordered_labels = AMAZON_CATEGORIES
    metrics = evaluate_predictions(y_test, y_pred, ordered_labels)

    print("=== Test-set metrics ===")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=ordered_labels,
            digits=4,
            zero_division=0,
        )
    )

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    save_metrics(metrics, RESULTS_DIR)
    save_confusion_matrix(y_test, y_pred, ordered_labels, RESULTS_DIR)

    print()
    print("Milestone 1 training and evaluation finished.")


if __name__ == "__main__":
    main()
