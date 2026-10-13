from database import get_order
from eligibility import load_policy

# prototype values: tune them by experiment, don't treat them as final
RISK_WEIGHTS = {
    "product_mismatch": 0.6,
    "claim_condition_conflict": 0.4,
    "claim_conflicts_with_image": 0.4,
    "already_returned": 0.7,
    "order_not_found": 0.5,
    "low_match_confidence": 0.2,
    "no_image_evidence": 0.25,
    "condition_unknown": 0.15,
    "wrong_item_unverified": 0.1,     # new
    "wrong_item_needs_check": 0.1,    # new
}
CONFLICT_FLAGS = {"product_mismatch", "claim_condition_conflict", "claim_conflicts_with_image"}


def evaluate(result, policy=None):
    policy = policy or load_policy()
    lo, hi = policy["mismatch_threshold"], policy["match_threshold"]
    flags, score = result["risk_flags"], result["product_match_score"]

    # TRUST: how complete and strong is the agent's own evidence?
    match_conf = 0.0 if score is None else min(1.0, max(0.0, (score - lo) / (hi - lo)))
    cond_known = 0.0 if result["condition"] == "unknown" else 1.0
    policy_ok = 1.0 if result["policy_match"] else 0.0
    trust = round(0.5 * match_conf + 0.3 * cond_known + 0.2 * policy_ok, 2)

    # RISK: fraud and error indicators
    risk = sum(RISK_WEIGHTS.get(f, 0) for f in flags)
    if result["refund_amount"] > 2000:
        risk += 0.1                                  # higher value, more care
    risk = round(min(1.0, risk), 2)

    # CONFLICT: do the claim and the evidence disagree?
    conflict = any(f in CONFLICT_FLAGS for f in flags)

    # DECISION
    if risk >= 0.5:
        decision = "ESCALATE"
    elif result["status"] == "not_eligible":
        decision = "ACCEPT"                          # a clear rule-based rejection
    elif result["status"] == "eligible" and trust >= 0.7 and risk < 0.3:
        decision = "ACCEPT"
    else:
        decision = "VERIFY"

    if decision == "ACCEPT":
        action = "REFUND" if result["return_eligible"] else "REJECT"
    elif decision == "VERIFY":
        action = "REQUEST_MORE_EVIDENCE"
    else:
        action = "HUMAN_REVIEW"

    return {"decision": decision, "next_action": action, "trust": trust,
            "risk": risk, "conflict": conflict, "risk_flags": flags}


def billing_handoff(result, atro):
    """Only an accepted, eligible return produces refund information."""
    if atro["decision"] == "ACCEPT" and result["return_eligible"]:
        order = get_order(result["order_id"])
        return {"order_id": result["order_id"],
                "customer_id": order["customer_id"],
                "refund_amount": result["refund_amount"]}
    return None