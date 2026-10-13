import base64, io, json, os, re
from PIL import Image

PROMPT = (
    "You are inspecting a photo of a clothing item that a customer wants to return. "
    "Decide whether the garment shows real damage: tears, holes, rips, burn marks, "
    "stains, broken seams, missing buttons or broken zips. "
    "Do NOT count intentional design features such as distressed or ripped denim, "
    "lace, cut-outs, embroidery, prints, creases, wrinkles or folds. "
    "Reply with JSON only, no other text: "
    '{"condition": "damaged" | "intact" | "unclear", "evidence": "one short sentence"}. '
    "Use 'unclear' if the garment is not clearly visible, the image is blurry, "
    "or you cannot tell whether a mark is damage."
)


def _encode(path, max_side=768):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def check_condition(image_path):
    """Returns (condition, note). condition is 'damaged', 'intact' or 'unknown'."""
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_VISION_MODEL")
    if not (key and model):
        return "unknown", "no vision model configured"
    try:
        from groq import Groq
        resp = Groq(api_key=key).chat.completions.create(
            model=model, temperature=0, max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_encode(image_path)}"}},
            ]}])
        text = resp.choices[0].message.content or ""
        found = [m for m in re.findall(r"\{[^{}]*\}", text, re.S) if '"condition"' in m]
        data = json.loads(found[-1])               # last JSON object in the reply
        cond = str(data.get("condition", "unclear")).lower()
        if cond not in ("damaged", "intact"):
            cond = "unknown"
        return cond, str(data.get("evidence", ""))
    except Exception as e:
        return "unknown", f"vision check failed: {e}"