import sqlite3, random
from datetime import date, timedelta
import pandas as pd
from PIL import Image

DB = "data/orders.db"

PALETTE = {"black": (25, 25, 25), "white": (240, 240, 240), "grey": (128, 128, 128),
           "red": (190, 30, 40), "blue": (40, 70, 160), "navy": (20, 30, 80),
           "green": (50, 130, 60), "yellow": (230, 200, 50), "pink": (230, 130, 160),
           "brown": (110, 70, 40), "beige": (210, 190, 150), "purple": (110, 50, 140)}


def dominant_colour(path):
    """Rough colour name from the centre of the crop (ignores white padding)."""
    im = Image.open(path).convert("RGB").resize((64, 64))
    px = [p for p in im.crop((16, 16, 48, 48)).getdata() if min(p) < 235]
    if not px:
        return "white"
    mean = tuple(sum(c) / len(px) for c in zip(*px))
    return min(PALETTE, key=lambda k: sum((a - b) ** 2 for a, b in zip(mean, PALETTE[k])))


def init_db(n=30, seed=42):
    random.seed(seed)
    meta = pd.read_csv("data/crops_meta.csv")
    shop = meta[(meta.source == "shop") & (meta.fold == "test")].drop_duplicates("key").sample(n, random_state=seed)

    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS orders")
    con.execute("""CREATE TABLE orders (
        order_id TEXT PRIMARY KEY, customer_id TEXT, product_id TEXT, product_key TEXT,
        product_name TEXT, color TEXT, size TEXT, price REAL, order_date TEXT,
        original_image TEXT, return_status TEXT)""")

    for i, r in enumerate(shop.itertuples(), start=1):
        img = f"data/crops/{r.item_id}.jpg"
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            f"ORD{i:03d}", f"CUST{random.randint(1, 10):03d}", f"P{r.item_id}", r.key,
            r.category_name.title(), dominant_colour(img), random.choice(["S", "M", "L", "XL"]),
            random.randint(499, 2999), str(date.today() - timedelta(days=random.randint(3, 25))),
            img, "none"))
    con.commit(); con.close()
    print(f"Created {n} orders in {DB}")


def get_order(order_id):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    return dict(row) if row else None


if __name__ == "__main__":
    init_db()