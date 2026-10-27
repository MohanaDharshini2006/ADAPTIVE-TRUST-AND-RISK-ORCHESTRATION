"""Read-only MongoDB schema check. Prints collection names and ONE sample document per collection
with personal data masked, so the output is safe to share. Never prints the connection string.
Set MONGODB_URI and MONGODB_DB (and optionally MONGODB_COLLECTIONS=orders,products).
"""
import datetime
import os
import re
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from bson import ObjectId
from pymongo import MongoClient

SENSITIVE = re.compile(r"mail|phone|mobile|name|address|street|city|zip|pin|token|password|secret|card|upi|ifsc", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def mask(value, key=""):
    if isinstance(value, dict):
        return {k: mask(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [mask(v, key) for v in value[:3]] + (["...more items"] if len(value) > 3 else [])
    if isinstance(value, ObjectId):
        return "<ObjectId>"
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, str):
        if key and SENSITIVE.search(key) and key.lower() not in ("product_name", "productname", "item_name", "itemname"):
            return "<masked>"
        return EMAIL.sub("<email>", value)[:80]
    return value


uri, dbname = os.environ.get("MONGODB_URI"), os.environ.get("MONGODB_DB")
if not uri or not dbname:
    sys.exit("Set MONGODB_URI and MONGODB_DB first.")
db = MongoClient(uri, serverSelectionTimeoutMS=8000)[dbname]
try:
    db.client.admin.command("ping")
    print("Connected and signed in: OK")
except Exception as e:
    sys.exit(f"Connection or sign-in failed: {type(e).__name__} (code {getattr(e, 'code', None)}): {str(e)[:150]}")
try:
    names = [c.strip() for c in os.environ.get("MONGODB_COLLECTIONS", "").split(",") if c.strip()] or db.list_collection_names()
except Exception as e:
    sys.exit(f"Could not list collections: {type(e).__name__} (code {getattr(e, 'code', None)}): {str(e)[:150]}\n"
             "Set MONGODB_COLLECTIONS=orders (your collection name) and retry.")
print("Collections:", names)
for name in names[:10]:
    doc = db[name].find_one()
    print(f"\n== {name}  (about {db[name].estimated_document_count()} documents)")
    import json
    print(json.dumps(mask(doc), indent=2, default=str) if doc else "(empty)")