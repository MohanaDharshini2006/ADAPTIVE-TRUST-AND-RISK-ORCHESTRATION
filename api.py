"""ATRO Return Agent - web API. Serves the front end in web/ and wraps the Python agent.
Run from the project root:  uvicorn api:app --host 127.0.0.1 --port 8000
"""
import io
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "src")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from atro import billing_handoff, evaluate
from database import DB, get_order
from eligibility import load_policy
from return_agent import process_return

app = FastAPI(title="ATRO Return Agent API")
MAX_BYTES = 8 * 1024 * 1024


@app.get("/api/orders")
def orders():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT order_id, product_name, color, size, price, order_date "
                       "FROM orders ORDER BY order_id").fetchall()
    con.close()
    keys = ("order_id", "product_name", "color", "size", "price", "order_date")
    return [dict(zip(keys, r)) for r in rows]


@app.get("/api/orders/{order_id}/image")
def order_image(order_id: str):
    order = get_order(order_id)
    if not order or not os.path.isfile(order["original_image"]):
        raise HTTPException(404, "No original image for this order.")
    return FileResponse(order["original_image"])


@app.get("/api/policy")
def policy():
    p = load_policy()
    return {k: p[k] for k in ("return_window_days", "match_threshold", "mismatch_threshold")}


@app.get("/api/status")
def status():
    return {"groq_key": bool(os.environ.get("GROQ_API_KEY")),
            "reason_model": os.environ.get("GROQ_MODEL") or None,
            "vision_model": os.environ.get("GROQ_VISION_MODEL") or None}


@app.post("/api/return")
def check_return(order_id: str = Form(...), message: str = Form(...),
                 condition_mode: str = Form("auto"), manual_condition: str = Form("damaged"),
                 photo: UploadFile = File(...)):
    if not get_order(order_id):
        raise HTTPException(404, "Unknown order.")
    if condition_mode == "auto":
        condition = None
    elif condition_mode == "skip":
        condition = "unknown"
    elif condition_mode == "manual" and manual_condition in ("damaged", "intact"):
        condition = manual_condition
    else:
        raise HTTPException(400, "Invalid condition option.")

    data = photo.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Photo is larger than 8 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "The file is not a readable image.")

    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        img.save(path)
        result = process_return(order_id, message[:2000], path, condition=condition)
        atro = evaluate(result)
        return {"result": result, "atro": atro, "billing": billing_handoff(result, atro)}
    except Exception as e:
        raise HTTPException(500, f"The agent failed: {e}")
    finally:
        os.remove(path)


app.mount("/", StaticFiles(directory="web", html=True), name="web")
