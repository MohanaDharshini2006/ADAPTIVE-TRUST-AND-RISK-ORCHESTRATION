import sys
from datetime import date, datetime, timedelta
sys.path.insert(0, "src")
from database import get_order
from eligibility import check_eligibility

OID = "ORD001"
order = get_order(OID)
order_day = datetime.strptime(order["order_date"], "%Y-%m-%d").date()

def status(**kw):
    args = dict(order_id=OID, reason="damaged_product", match_score=0.92, condition="damaged")
    args.update(kw)
    return check_eligibility(**args)["status"]

tests = {
    "same product + damaged + in window -> eligible":   lambda: status() == "eligible",
    "different product -> verification":                lambda: status(match_score=0.30) == "needs_verification",
    "low similarity -> verification":                   lambda: status(match_score=0.70) == "needs_verification",
    "return period exceeded -> not eligible":           lambda: status(today=order_day + timedelta(days=40)) == "not_eligible",
    "claims damage but image intact -> verification":   lambda: status(condition="intact") == "needs_verification",
    "no image / unknown condition -> verification":     lambda: status(match_score=None, condition="unknown") == "needs_verification",
    "wrong item but photo matches order -> verification": lambda: status(reason="wrong_item") == "needs_verification",
    "ineligible reason -> not eligible":                lambda: status(reason="changed_mind") == "not_eligible",
    "unknown order -> not eligible":                    lambda: status(order_id="ORD999") == "not_eligible",
    "refund equals order price":                        lambda: check_eligibility(OID, "damaged_product", 0.92, "damaged")["refund_amount"] == order["price"],
}

failed = 0
for name, t in tests.items():
    ok = t()
    failed += not ok
    print("PASS" if ok else "FAIL", "-", name)
print(f"\n{len(tests) - failed}/{len(tests)} passed")