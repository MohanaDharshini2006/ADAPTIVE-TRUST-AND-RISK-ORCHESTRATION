import json
from datetime import date, datetime
from database import get_order


def load_policy(path="policy.json"):
    with open(path) as f:
        return json.load(f)


def check_eligibility(order_id, reason, match_score, condition="unknown",
                      today=None, policy=None,order=None):
    policy = policy or load_policy()
    today = today or date.today()
    evidence, flags = [], []

    def result(status, refund=0.0, policy_match=False):
        return {"order_id": order_id, "return_reason": reason,
                "product_match_score": match_score, "condition": condition,
                "policy_match": policy_match, "status": status,
                "return_eligible": status == "eligible", "refund_amount": refund,
                "evidence": evidence, "risk_flags": flags}

    order = order or get_order(order_id)
    if order is None:
        flags.append("order_not_found")
        return result("not_eligible")
    if order["return_status"] != "none":
        flags.append("already_returned")
        return result("not_eligible")

    rule = policy["reasons"].get(reason)
    if rule is None:
        evidence.append(f"reason '{reason}' is not eligible under policy")
        return result("not_eligible")

    days = (today - datetime.strptime(order["order_date"], "%Y-%m-%d").date()).days
    evidence.append(f"{days} days since order (limit {policy['return_window_days']})")
    if days > policy["return_window_days"]:
        return result("not_eligible")

    # product match zone
    if match_score is None:
        zone = "unknown"
    elif match_score >= policy["match_threshold"]:
        zone = "match"
    elif match_score < policy["mismatch_threshold"]:
        zone = "mismatch"
    else:
        zone = "uncertain"
    evidence.append(f"match zone: {zone}")

    if rule["requires_product_match"]:
        if zone == "mismatch":
            flags.append("product_mismatch")
        elif zone == "uncertain":
            flags.append("low_match_confidence")
        elif zone == "unknown":
            flags.append("no_image_evidence")
    else:
        # "wrong item" style claims: the photo can support a claim but cannot prove it
        if zone == "match":
            flags.append("claim_conflicts_with_image")
        elif zone == "mismatch":
            flags.append("wrong_item_unverified")
        else:
            flags.append("wrong_item_needs_check")

    required = rule["required_condition"]
    if required != "any":
        if condition == "unknown":
            flags.append("condition_unknown")
        elif condition != required:
            flags.append("claim_condition_conflict")

    if flags:                                   # weak or conflicting evidence: never auto-approve
        return result("needs_verification", policy_match=True)
    return result("eligible", refund=order["price"], policy_match=True)