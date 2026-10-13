import sys
from datetime import datetime, timedelta
sys.path.insert(0, "src")
from database import get_order
from eligibility import check_eligibility
from atro import evaluate, billing_handoff

OID = "ORD001"
order = get_order(OID)
order_day = datetime.strptime(order["order_date"], "%Y-%m-%d").date()


def run(**kw):
    args = dict(order_id=OID, reason="damaged_product", match_score=0.92, condition="damaged")
    args.update(kw)
    res = check_eligibility(**args)
    a = evaluate(res)
    return res, a, billing_handoff(res, a)


checks = []

res, a, h = run()
checks.append(("good case -> ACCEPT + refund handoff",
               a["decision"] == "ACCEPT" and h is not None and h["refund_amount"] == order["price"]))

res, a, h = run(match_score=0.70)
checks.append(("low similarity -> VERIFY, no handoff",
               a["decision"] == "VERIFY" and h is None))

res, a, h = run(match_score=0.30)
checks.append(("different product claimed damaged -> ESCALATE",
               a["decision"] == "ESCALATE" and h is None))

res, a, h = run(today=order_day + timedelta(days=40))
checks.append(("expired window -> ACCEPT rejection, no handoff",
               a["decision"] == "ACCEPT" and a["next_action"] == "REJECT" and h is None))

res, a, h = run(condition="intact")
checks.append(("damage claimed, image intact -> VERIFY with conflict",
               a["decision"] == "VERIFY" and a["conflict"]))

res, a, h = run(order_id="ORD999")
checks.append(("unknown order -> ESCALATE", a["decision"] == "ESCALATE"))

res, a, h = run(reason="wrong_item", match_score=0.30, condition="any")
checks.append(("wrong item, photo differs -> VERIFY (cannot be confirmed from image)",
               a["decision"] == "VERIFY" and h is None))

failed = 0
for name, ok in checks:
    failed += not ok
    print("PASS" if ok else "FAIL", "-", name)
print(f"\n{len(checks) - failed}/{len(checks)} passed")