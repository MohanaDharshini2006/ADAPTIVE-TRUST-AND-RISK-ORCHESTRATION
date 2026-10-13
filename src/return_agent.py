from database import get_order
from matcher import Matcher
from reason import classify_reason
from eligibility import check_eligibility

_matcher = None


def process_return(order_id, text, image_path, condition="unknown"):
    global _matcher
    _matcher = _matcher or Matcher()

    reason, method = classify_reason(text)
    order = get_order(order_id)

    score = None
    if order and image_path:
        score = _matcher.score(order["original_image"], image_path)

    result = check_eligibility(order_id, reason, score, condition)
    result["evidence"] = result["evidence"] + [
        f"reason classified by {method}",
        f"similarity={score:.2f}" if score is not None else "no image provided",
    ]
    return result