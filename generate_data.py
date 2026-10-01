"""Script to generate realistic synthetic 12-month sales dataset with a deliberate anomaly."""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

# Deterministic seed for reproducible evaluation
random.seed(42)

OUT_DIR = Path("sample_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "sales.csv"

# Catalog
PRODUCTS = {
    "Electronics": [
        ("Laptop Pro", 1200.0, 0.05),
        ("Wireless Headphones", 150.0, 0.10),
        ("Smart Watch", 300.0, 0.08),
    ],
    "Furniture": [
        ("Ergonomic Chair", 350.0, 0.12),
        ("Standing Desk", 600.0, 0.10),
        ("Bookshelf", 200.0, 0.15),
    ],
    "Office Supplies": [
        ("Notebook Pack", 20.0, 0.05),
        ("Gel Pens", 15.0, 0.05),
        ("Desk Mat", 30.0, 0.08),
    ],
}

REGIONS = ["North", "South", "East", "West"]

# Generate data for 2024 (12 months: Jan 1 to Dec 31)
start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)

rows = []
cur_date = start_date

while cur_date <= end_date:
    month = cur_date.month
    # Daily transactions across regions
    for region in REGIONS:
        # 3 transactions per region per day on average
        num_tx = 3
        for _ in range(num_tx):
            category = random.choice(list(PRODUCTS.keys()))
            prod_name, base_price, default_disc = random.choice(PRODUCTS[category])
            
            # Anomaly injection:
            # In March (month == 3), in the 'North' region, for 'Electronics':
            # A regional warehouse fire/stockout occurred. Units plummeted to near zero.
            if region == "North" and category == "Electronics" and month == 3:
                # 90% chance no transaction
                if random.random() < 0.90:
                    continue
                units = 1
                discount = 0.0
            else:
                if category == "Electronics":
                    units = random.randint(2, 4)
                elif category == "Furniture":
                    units = random.randint(1, 3)
                else:  # Office Supplies
                    units = random.randint(10, 20)
                discount = round(default_disc + random.uniform(-0.02, 0.02), 2)
                discount = max(0.0, min(0.30, discount))

            raw_rev = units * base_price
            rev = round(raw_rev * (1.0 - discount), 2)

            rows.append({
                "date": cur_date.isoformat(),
                "region": region,
                "product": prod_name,
                "category": category,
                "units": units,
                "revenue": rev,
                "discount": discount,
            })

    cur_date += timedelta(days=1)

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "region", "product", "category", "units", "revenue", "discount"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} records into {OUT_FILE}")
