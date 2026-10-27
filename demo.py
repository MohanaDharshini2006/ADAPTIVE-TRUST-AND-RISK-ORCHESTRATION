import sys, json
sys.path.insert(0, "src")
import pandas as pd
from database import get_order
from return_agent import process_return
from atro import evaluate, billing_handoff

meta = pd.read_csv("data/crops_meta.csv")
order = get_order("ORD001")
users = meta[meta.source == "user"]

same = users[users.key == order["product_key"]].iloc[0]
other = users[(users.key != order["product_key"]) &
              (users.category_name.str.lower() == order["product_name"].lower())].iloc[0]

cases = {
    "SAME product, customer says damaged": (same.item_id, "The shirt arrived damaged", None),
    "SAME product, no condition info":     (same.item_id, "The shirt arrived damaged", None),
    "DIFFERENT product (same category)":   (other.item_id, "I received the wrong item", None),
}
for name, (iid, text, cond) in cases.items():
    out = process_return("ORD001", text, f"data/crops/{iid}.jpg", condition=cond)
    print("\n==", name)
    print(json.dumps(out, indent=2))

    atro = evaluate(out)
    print("ATRO:", json.dumps(atro, indent=2))
    print("Billing handoff:", billing_handoff(out, atro))