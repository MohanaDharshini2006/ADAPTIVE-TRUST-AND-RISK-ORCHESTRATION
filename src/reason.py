import os
from dotenv import load_dotenv
load_dotenv()

REASONS = ["damaged_product", "wrong_item", "size_issue", "changed_mind", "other"]

KEYWORDS = {
    "damaged_product": ["damage", "torn", "tear", "broken", "defect", "stain", "hole", "ripped"],
    "wrong_item": ["wrong", "different", "not what i ordered", "incorrect"],
    "size_issue": ["size", "too small", "too big", "too large", "doesn't fit", "does not fit"],
    "changed_mind": ["changed my mind", "don't like", "do not like", "no longer"],
}


def keyword_reason(text):
    t = text.lower()
    for reason, words in KEYWORDS.items():
        if any(w in t for w in words):
            return reason
    return "other"


def classify_reason(text):
    """Returns (reason, method). Uses Groq if configured, else keywords."""
    key, model = os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_MODEL")
    if key and model:
        try:
            from groq import Groq
            resp = Groq(api_key=key).chat.completions.create(
                model=model, temperature=0,
                messages=[
                    {"role": "system", "content":
                     "Classify the customer's return reason. Reply with exactly one label from: "
                     + ", ".join(REASONS) + ". No other text."},
                    {"role": "user", "content": text},
                ])
            label = resp.choices[0].message.content.strip().lower()
            if label in REASONS:
                return label, "groq"
        except Exception as e:
            print("Groq failed, using keyword fallback:", e)
    return keyword_reason(text), "keyword"