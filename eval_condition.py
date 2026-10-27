import os, sys, time, csv, re, argparse, random
from PIL import Image, ImageEnhance

sys.path.insert(0, "src")
from condition import check_condition

parser = argparse.ArgumentParser(description="Evaluate garment condition detection")
parser.add_argument("--mode", choices=["single", "reference"], default="single", help="Evaluation mode")
parser.add_argument("--split", choices=["dev", "holdout", "all"], default="all", help="Dataset split")
args = parser.parse_args()


def in_split(filename, split):
    m = re.search(r"set_(\d+)_", filename)
    if not m:
        return False
    num = int(m.group(1))
    if split == "dev":
        return 1 <= num <= 15
    elif split == "holdout":
        return 16 <= num <= 30
    elif split == "all":
        return 1 <= num <= 30
    return False


def get_reference_path(filename):
    m = re.search(r"set_(\d+)_", filename)
    if not m:
        return None
    xx = m.group(1)
    return f"data/damage_real_test/intact/set_{xx}_original.jpg"


def build_intact_returned():
    intact_dir = "data/damage_real_test/intact"
    out_dir = "data/damage_real_test/intact_returned"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(intact_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    random.seed(42)
    for f in files:
        img = Image.open(os.path.join(intact_dir, f)).convert("RGB")
        w, h = img.size

        # 1. Random rotation between -12 and 12 degrees with a white fill
        angle = random.uniform(-12, 12)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

        # 2. Random crop keeping 80-90% of the image then resized back to original size
        crop_factor = random.uniform(0.8, 0.9)
        crop_w = int(w * crop_factor)
        crop_h = int(h * crop_factor)
        x0 = random.randint(0, w - crop_w)
        y0 = random.randint(0, h - crop_h)
        img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((w, h), resample=Image.Resampling.BICUBIC)

        # 3. Brightness factor between 0.75 and 1.25
        bf = random.uniform(0.75, 1.25)
        img = ImageEnhance.Brightness(img).enhance(bf)

        # 4. Contrast factor between 0.8 and 1.2
        cf = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(cf)

        # 5. Horizontal flip with 50% probability
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        out_name = re.sub(r"_original\.", "_returned.", f)
        img.save(os.path.join(out_dir, out_name), quality=95)


if args.mode == "reference":
    build_intact_returned()
    CACHE = f"data/damage_real_test/results_reference2_{args.split}.csv"
else:
    CACHE = f"data/damage_real_test/results_{args.mode}_{args.split}.csv"

done = {}


def is_error(pred, note):
    if not note:
        return False
    if note.startswith("vision check failed"):
        return True
    if note.startswith("no vision model configured"):
        return True
    if "error" in note.lower() and pred == "unknown":
        return True
    return False


def is_rate_limit(note):
    n = note.lower()
    return "429" in n or "rate_limit" in n or "rate limit" in n


def save_cache():
    with open(CACHE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["truth", "file", "pred", "note"])
        for (t, f), (p, n) in sorted(done.items()):
            if not is_error(p, n):
                w.writerow([t, f, p, n])


if os.path.exists(CACHE):
    with open(CACHE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not is_error(r.get("pred", ""), r.get("note", "")):
                done[(r["truth"], r["file"])] = (r["pred"], r["note"])


def get_truth_folder(truth, mode):
    if truth == "intact" and mode == "reference":
        return "data/damage_real_test/intact_returned"
    return f"data/damage_real_test/{truth}"


stopped = False
for truth in ("damaged", "intact"):
    folder = get_truth_folder(truth, args.mode)
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")) or not in_split(f, args.split) or (truth, f) in done:
            continue
        ref_path = get_reference_path(f) if args.mode == "reference" else None
        try:
            pred, note = check_condition(f"{folder}/{f}", reference_path=ref_path)
        except Exception as e:
            pred, note = "unknown", f"vision check failed: {e}"

        if is_error(pred, note):
            print("ERROR on", f, "->", note[:140])
            if is_rate_limit(note):
                print("Rate limit reached. Rerun later.")
                save_cache()
                stopped = True
                break
            continue

        done[(truth, f)] = (pred, note)
        print(f"{truth:8s} -> {pred:8s} {f}  | {note[:80]}")
        save_cache()
        time.sleep(1)
    if stopped:
        break

save_cache()

rows = [(t, p, f, n) for (t, f), (p, n) in done.items()]
n = len(rows)
total = sum(len([x for x in os.listdir(get_truth_folder(t, args.mode))
                 if x.lower().endswith((".jpg", ".jpeg", ".png")) and in_split(x, args.split)]) for t in ("damaged", "intact"))
print(f"\nanswered: {n} of {total} images" + ("  (INCOMPLETE)" if n < total else ""))
if n:
    tp = sum(t == "damaged" and p == "damaged" for t, p, *_ in rows)
    fp = sum(t == "intact" and p == "damaged" for t, p, *_ in rows)
    fn = sum(t == "damaged" and p == "intact" for t, p, *_ in rows)
    unk = sum(p == "unknown" for _, p, *_ in rows)
    dmg = sum(t == "damaged" for t, *_ in rows)
    decided = [r for r in rows if r[1] != "unknown"]
    acc = sum(t == p for t, p, *_ in decided) / max(len(decided), 1)
    print(f"unknown: {unk}   accuracy on decided: {acc:.2f}")
    print(f"damaged found: {tp}/{dmg}   wrongly flagged intact: {fp}   damaged missed as intact: {fn}")
    print("\nMISSED DAMAGE:")
    for t, p, f, note in rows:
        if t == "damaged" and p != "damaged":
            print(f"  {f} -> {p} | {note}")
    print("\nWRONG ON INTACT / UNKNOWN:")
    for t, p, f, note in rows:
        if t == "intact" and p != "intact":
            print(f"  {f} -> {p} | {note}")