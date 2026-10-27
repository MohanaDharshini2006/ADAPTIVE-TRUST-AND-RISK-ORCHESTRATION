import os
from database import get_order
from matcher import Matcher
from reason import classify_reason
from eligibility import check_eligibility
from condition import check_condition

_matcher = None


def process_return(order_id, text, image_path, condition=None):
    global _matcher
    _matcher = _matcher or Matcher()

    order = get_order(order_id)

    note = None
    if condition is None and image_path:
        ref_image = order.get("original_image") if order else None
        if ref_image and os.path.isfile(ref_image):
            res_cond, res_note = check_condition(image_path, reference_path=order["original_image"])
        else:
            res_cond, res_note = check_condition(image_path)
        condition = res_cond if res_cond is not None else "unknown"
        note = res_note
    if condition is None:
        condition = "unknown"

    reason, method = classify_reason(text)

    score = None
    if order and image_path:
        score = _matcher.score(order["original_image"], image_path)

    result = check_eligibility(order_id, reason, score, condition)
    evidence_items = [
        f"reason classified by {method}",
        f"similarity={score:.2f}" if score is not None else "no image provided",
    ]
    if note is not None:
        evidence_items.append(f"condition ({condition}): {note}")
    result["evidence"] = result["evidence"] + evidence_items
    return result