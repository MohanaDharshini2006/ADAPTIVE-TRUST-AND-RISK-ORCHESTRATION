"""
ATRO Technical Support — quick inference smoke test
Run from the project root in VS Code.

This tests whether the EXISTING six-category clothing/product classifier loads
and produces predictions. It does NOT calculate valid accuracy. Website/app
technical queries are flagged as outside the current six-label model.
"""
from pathlib import Path
import sys

try:
    import joblib
except ImportError:
    print("ERROR: joblib is missing in the selected Python environment.")
    print("In VS Code terminal, activate the project's .venv and install project requirements if approved.")
    sys.exit(1)

MODEL_PATH = Path("models/tfidf_logreg_issue_classifier.joblib")

PRODUCT_QUERIES = [
    "The shirt is too tight around the shoulders and does not fit.",
    "The fabric feels rough and is not the material described.",
    "The stitching is coming apart after one wash.",
    "The zipper is broken and the buttons are loose.",
    "The colour faded after washing.",
    "The item arrived with poor overall quality and visible defects.",
]

WEB_APP_QUERIES = [
    "I cannot log in to my account.",
    "The password reset link is not working.",
    "I never received the OTP verification code.",
    "The shopping app crashes when I open it.",
    "The cart fails when I try to check out.",
    "Product images are not loading on the website.",
    "The payment page shows an error before an order is placed.",
]

def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Run this script from the project root and check the model file path.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print("=" * 72)
    print("ATRO TECHNICAL SUPPORT — SMOKE TEST")
    print("Purpose: verify model loading and example predictions, not benchmark accuracy.")
    print("=" * 72)

    print("\nA) CLOTHING / PRODUCT QUERIES (supported by current six-label baseline)")
    for i, query in enumerate(PRODUCT_QUERIES, 1):
        try:
            predicted = model.predict([query])[0]
            line = f"{i}. Query: {query}\n   Predicted category: {predicted}"
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([query])[0]
                classes = getattr(model, "classes_", None)
                if classes is None and hasattr(model, "named_steps"):
                    final_step = list(model.named_steps.values())[-1]
                    classes = getattr(final_step, "classes_", None)
                if classes is not None:
                    best_index = int(probs.argmax())
                    line += f"\n   Model probability: {probs[best_index]:.3f}"
            print(line)
        except Exception as exc:
            print(f"{i}. Prediction failed: {exc}")

    print("\nB) WEBSITE / APP TECHNICAL QUERIES (outside current six-label classifier)")
    for i, query in enumerate(WEB_APP_QUERIES, 1):
        print(f"{i}. Query: {query}")
        print("   Expected handling: technical_support top-level route; subcategory model needed.")
        print("   Current status: NOT EVALUATED by the six-category product classifier.")

    print("\nSUMMARY")
    print("- If product predictions appear, the saved model loaded and inference ran.")
    print("- Do not treat these hand-picked examples as an accuracy test.")
    print("- Do not report probabilities as calibrated confidence without validation.")
    print("- To measure accuracy, use a separate labelled held-out test set with known expected labels.")

if __name__ == "__main__":
    main()
