import os
from PIL import Image

SRC = "data/cloth_damage_30_sets"
CROP_BOTTOM = 0.0       # the latest batch has no leftover strip; raise to ~0.1 only if one shows up

for lab in ("damaged", "intact"):
    os.makedirs(f"data/damage_real_test/{lab}", exist_ok=True)

saved = []
for f in sorted(os.listdir(SRC)):
    low = f.lower()
    if not low.endswith((".png", ".jpg", ".jpeg")):
        continue
    if "_damaged" in low:
        lab = "damaged"
    elif "_original" in low:
        lab = "intact"
    else:
        print("skipped (name has neither _damaged nor _original):", f)
        continue
    img = Image.open(f"{SRC}/{f}").convert("RGB")
    if CROP_BOTTOM > 0:
        img = img.crop((0, 0, img.width, int(img.height * (1 - CROP_BOTTOM))))
    out = f"data/damage_real_test/{lab}/{os.path.splitext(f)[0]}.jpg"
    img.save(out, quality=95)
    saved.append(out)

print("prepared", len(saved), "images; example size:", Image.open(saved[0]).size)