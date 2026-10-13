import os, sys, time, csv
sys.path.insert(0, "src")
from condition import check_condition

CACHE = "data/damage_real_test/results.csv"
done = {}
if os.path.exists(CACHE):
    with open(CACHE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            done[(r["truth"], r["file"])] = (r["pred"], r["note"])

stopped = False
for truth in ("damaged", "intact"):
    folder = f"data/damage_real_test/{truth}"
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")) or (truth, f) in done:
            continue
        pred, note = check_condition(f"{folder}/{f}")
        if note.startswith("vision check failed"):          # an error, not an answer
            print("ERROR on", f, "->", note[:140])
            if "429" in note or "rate_limit" in note:
                print("Rate limit reached. Stopping. Rerun later; finished images are saved.")
                stopped = True
                break
            continue
        done[(truth, f)] = (pred, note)
        print(f"{truth:8s} -> {pred:8s} {f}  | {note[:80]}")
        time.sleep(1)
    if stopped:
        break

with open(CACHE, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["truth", "file", "pred", "note"])
    for (t, f), (p, n) in sorted(done.items()):
        w.writerow([t, f, p, n])

rows = [(t, p, f, n) for (t, f), (p, n) in done.items()]
n = len(rows)
total = sum(len([x for x in os.listdir(f"data/damage_real_test/{t}")
                 if x.lower().endswith((".jpg", ".jpeg", ".png"))]) for t in ("damaged", "intact"))
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