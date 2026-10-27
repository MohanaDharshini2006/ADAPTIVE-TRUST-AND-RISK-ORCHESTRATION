"""Read-only access to the orders the Order Agent stored in MongoDB (database ATRO, collection orders).
One document = one ordered product: order_id, customer_id, product, category, size, quantity,
status, tracking_number, expected_delivery.  Edit CFG if the Order Agent's fields change.
"""
import datetime
import json
import os
import re
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from pymongo import MongoClient

CFG = {
    "orders": os.environ.get("MONGODB_ORDERS_COLLECTION", "orders"),
    "returns": os.environ.get("MONGODB_RETURNS_COLLECTION", "return_requests"),  # the Return Agent's own records
    "order_id": "order_id", "customer_id": "customer_id", "product": "product", "product_id": "product_id",
    "category": "category", "size": "size", "quantity": "quantity", "status": "status",
    "expected_delivery": "expected_delivery", "delivered_at": "delivered_at", "price": "price",
    "delivered_statuses": {"delivered"},
    "catalog_file": "data/catalog.json",   # optional: {"floral maxi dress": {"image": "...", "price": 999, "product_id": "..."}}
}
_SAFE = re.compile(r"^[\w@.+\-# ]{1,64}$")          # plain text only: blocks operator objects such as {"$ne": null}


@lru_cache(maxsize=1)
def _db():
    return MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=8000)[os.environ["MONGODB_DB"]]


def _text(value):
    if not isinstance(value, str) or not _SAFE.match(value.strip()):
        raise ValueError("invalid input")
    return value.strip()


def _when(value):
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime.datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value


def catalog_entry(name):
    try:
        with open(CFG["catalog_file"], encoding="utf-8") as f:
            return json.load(f).get(str(name).strip().lower(), {})
    except (OSError, ValueError):
        return {}


def verify_purchase(order_id, customer_id, product_ref, window_days=30):
    """Did this customer order this exact product, and can it still be returned?
    Returns {"verified", "reason", "order", "item", "delivery_date_approximate"}.
    A wrong order id and a wrong customer give the same answer, so nothing leaks about other customers."""
    def fail(why, doc=None):
        return {"verified": False, "reason": why, "order": doc, "item": doc, "delivery_date_approximate": False}
    try:
        order_id, customer_id, product_ref = _text(order_id), _text(customer_id).lower(), _text(product_ref).lower()
    except ValueError:
        return fail("invalid_input")

    docs = [d for d in _db()[CFG["orders"]].find({CFG["order_id"]: order_id})
            if str(d.get(CFG["customer_id"], "")).strip().lower() == customer_id]
    if not docs:
        return fail("order_not_found")
    item = next((d for d in docs if product_ref in (str(d.get(CFG["product"], "")).strip().lower(),
                                                    str(d.get(CFG["product_id"], "")).strip().lower())), None)
    if item is None:
        return fail("product_not_in_order", docs[0])
    if str(item.get(CFG["status"], "")).strip().lower() not in CFG["delivered_statuses"]:
        return fail("not_delivered", item)

    exact = item.get(CFG["delivered_at"])
    start = _when(exact or item.get(CFG["expected_delivery"]))
    approx = exact is None
    if start and (datetime.datetime.now(datetime.timezone.utc) - start).days > window_days:
        return {**fail("return_window_passed", item), "delivery_date_approximate": approx}

    done = sum(r.get("quantity", 1) for r in _db()[CFG["returns"]].find(
        {"order_id": order_id, "product": item.get(CFG["product"]), "status": {"$ne": "declined"}}))
    if done >= int(item.get(CFG["quantity"], 1)):
        return {**fail("already_returned", item), "delivery_date_approximate": approx}
    return {"verified": True, "reason": "ok", "order": item, "item": item, "delivery_date_approximate": approx}


def as_legacy_order(doc):
    """Same shape the existing pipeline (eligibility, ATRO) already expects from get_order."""
    name = doc.get(CFG["product"], "")
    cat = catalog_entry(name)
    return {"order_id": doc[CFG["order_id"]], "customer_id": str(doc.get(CFG["customer_id"], "")),
            "product_id": str(doc.get(CFG["product_id"]) or cat.get("product_id") or name),
            "product_key": str(doc.get(CFG["product_id"]) or cat.get("product_id") or name),
            "product_name": name, "color": "", "size": doc.get(CFG["size"], ""),
            "price": float(doc.get(CFG["price"]) or cat.get("price") or 0),
            "order_date": str(doc.get(CFG["delivered_at"]) or doc.get(CFG["expected_delivery"]) or "")[:10],
            "original_image": cat.get("image", ""), "return_status": "none"}


def order_items(order_id, customer_id):
    """All products on this order that belong to this customer (empty list if none)."""
    try:
        oid, cid = _text(order_id), _text(customer_id).lower()
    except ValueError:
        return []
    return [d for d in _db()[CFG["orders"]].find({CFG["order_id"]: oid})
            if str(d.get(CFG["customer_id"], "")).strip().lower() == cid]


def record_return(order_id, product, quantity, status, extra=None):
    """Save the Return Agent's own record. Needs write access to the return_requests collection only."""
    try:
        _db()[CFG["returns"]].insert_one({
            "order_id": order_id, "product": product, "quantity": quantity, "status": status,
            "created_at": datetime.datetime.now(datetime.timezone.utc), **(extra or {})})
        return True
    except Exception:
        return False