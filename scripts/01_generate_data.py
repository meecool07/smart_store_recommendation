"""
Phase 1: Generate synthetic retail dataset (mimics UCI Online Retail Dataset)
Run this if you don't have the Kaggle dataset yet.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── Product catalog ───────────────────────────────────────────────────────────
PRODUCTS = {
    "P001": "Bread", "P002": "Milk", "P003": "Butter", "P004": "Eggs",
    "P005": "Cheese", "P006": "Jam", "P007": "Coffee", "P008": "Tea",
    "P009": "Sugar", "P010": "Salt", "P011": "Flour", "P012": "Rice",
    "P013": "Pasta", "P014": "Tomato Sauce", "P015": "Olive Oil",
    "P016": "Chicken", "P017": "Beef", "P018": "Fish", "P019": "Vegetables",
    "P020": "Fruits", "P021": "Yogurt", "P022": "Cream", "P023": "Ice Cream",
    "P024": "Biscuits", "P025": "Chocolate", "P026": "Chips", "P027": "Juice",
    "P028": "Water", "P029": "Soda", "P030": "Beer",
    "P031": "Shampoo", "P032": "Soap", "P033": "Toothpaste", "P034": "Detergent",
    "P035": "Tissue", "P036": "Bin Bags", "P037": "Cling Film",
}

# ─── Realistic basket patterns (association rules baked in) ───────────────────
BASKET_PATTERNS = [
    ["Bread", "Milk", "Butter"],
    ["Bread", "Eggs", "Butter"],
    ["Bread", "Jam", "Butter"],
    ["Coffee", "Milk", "Sugar"],
    ["Tea", "Milk", "Sugar", "Biscuits"],
    ["Pasta", "Tomato Sauce", "Cheese"],
    ["Pasta", "Olive Oil", "Cheese"],
    ["Rice", "Chicken", "Vegetables"],
    ["Rice", "Fish", "Vegetables"],
    ["Chicken", "Vegetables", "Olive Oil"],
    ["Beef", "Tomato Sauce", "Pasta"],
    ["Yogurt", "Fruits", "Juice"],
    ["Chips", "Soda", "Chocolate"],
    ["Beer", "Chips"],
    ["Shampoo", "Soap", "Toothpaste"],
    ["Shampoo", "Toothpaste", "Detergent"],
    ["Detergent", "Tissue", "Bin Bags"],
    ["Milk", "Yogurt", "Cream"],
    ["Ice Cream", "Chocolate", "Biscuits"],
    ["Flour", "Eggs", "Butter", "Sugar", "Milk"],  # baking basket
    ["Bread", "Cheese", "Tomato Sauce"],
    ["Coffee", "Biscuits", "Chocolate"],
    ["Juice", "Fruits", "Yogurt"],
    ["Water", "Chips", "Soda"],
    ["Eggs", "Milk", "Cheese"],
]

def generate_invoice_id(i):
    return f"INV{str(100000 + i).zfill(6)}"

def random_date():
    start = datetime(2023, 1, 1)
    return start + timedelta(days=random.randint(0, 364))

rows = []
n_transactions = 3000

for i in range(n_transactions):
    invoice = generate_invoice_id(i)
    date = random_date()

    # Pick a base pattern, sometimes combine two
    pattern = random.choice(BASKET_PATTERNS).copy()
    if random.random() < 0.3:
        pattern += random.choice(BASKET_PATTERNS)
    pattern = list(set(pattern))

    # Add 1-2 random products occasionally
    if random.random() < 0.2:
        all_products = list(PRODUCTS.values())
        pattern.append(random.choice(all_products))

    for product in pattern:
        rows.append({
            "InvoiceNo": invoice,
            "Description": product,
            "Quantity": random.randint(1, 5),
            "InvoiceDate": date.strftime("%Y-%m-%d"),
            "UnitPrice": round(random.uniform(0.5, 15.0), 2),
            "CustomerID": random.randint(10000, 19999),
            "Country": random.choice(["United Kingdom", "Germany", "France", "India", "USA"])
        })

df = pd.DataFrame(rows)

# Inject some noise to clean later
cancel_idx = random.sample(range(len(df)), 50)
df.loc[cancel_idx, "InvoiceNo"] = df.loc[cancel_idx, "InvoiceNo"].apply(lambda x: "C" + x)

neg_idx = random.sample(range(len(df)), 30)
df.loc[neg_idx, "Quantity"] = -df.loc[neg_idx, "Quantity"]

null_idx = random.sample(range(len(df)), 40)
df.loc[null_idx, "Description"] = None

df.to_csv("data/raw_retail.csv", index=False)
print(f"✅ Generated {len(df)} rows → data/raw_retail.csv")
print(f"   Transactions: {df['InvoiceNo'].nunique()}")
print(f"   Products:     {df['Description'].nunique()}")
