"""Tests the MongoDB sign-in without quoting or encoding problems.
Asks for the host, username and password (the password is hidden), builds a safe connection string,
and can save it to .env. Run from the project root:  python mongo_login_test.py
"""
import getpass
import os
import sys
from urllib.parse import quote_plus

from pymongo import MongoClient

host = input("Cluster host, the part after the @ in Atlas's string (like cluster0.xxxxx.mongodb.net): ").strip()
host = host.replace("mongodb+srv://", "").split("@")[-1].split("/")[0].strip()
user = input("Database username: ").strip()
password = getpass.getpass("Database password (typing is hidden): ")
db_name = input("Database name [ATRO]: ").strip() or "ATRO"

uri = f"mongodb+srv://{quote_plus(user)}:{quote_plus(password)}@{host}/?appName=Cluster0"
try:
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
except Exception as e:
    msg = str(e)[:160]
    hint = ""
    if "DNS" in msg or "does not exist" in msg:
        hint = "\nThe host looks wrong. Copy it from Atlas: Connect, Drivers, Python (do not use an example code)."
    elif "bad auth" in msg or getattr(e, "code", None) == 8000:
        hint = "\nThe host is right, but this username/password pair is rejected. Re-check the username and reset the password."
    elif "timed out" in msg.lower() or "ServerSelection" in type(e).__name__:
        hint = "\nTimed out: add your current IP under Network Access in Atlas."
    sys.exit(f"Failed: {type(e).__name__} (code {getattr(e, 'code', None)}): {msg}{hint}")

print("Sign-in OK")
try:
    print(f"Can read {db_name}.orders, documents: {client[db_name]['orders'].estimated_document_count()}")
except Exception as e:
    print(f"Signed in, but reading {db_name}.orders failed: {type(e).__name__} (code {getattr(e, 'code', None)}). Check the user's role.")

if input("Save the connection to .env in this folder? (y/n) ").strip().lower() == "y":
    keep = []
    if os.path.exists(".env"):
        keep = [l for l in open(".env", encoding="utf-8").read().splitlines()
                if not l.startswith(("MONGODB_URI=", "MONGODB_DB=", "MONGODB_COLLECTIONS="))]
    keep += [f"MONGODB_URI='{uri}'", f"MONGODB_DB={db_name}", "MONGODB_COLLECTIONS=orders"]
    open(".env", "w", encoding="utf-8").write("\n".join(keep) + "\n")
    print("Saved to .env. Make sure .env is listed in .gitignore, and never share that file.")