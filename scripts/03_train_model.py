"""
Phase 2 – Steps 4, 5, 6: Basket encoding → FP-Growth → Association Rules
"""

import pandas as pd
import sqlite3
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fp_growth_engine import fpgrowth, association_rules

DB_PATH    = "data/retail_store.db"
RULES_CSV  = "models/association_rules.csv"
RULES_PKL  = "models/association_rules.pkl"
ITEMS_PKL  = "models/all_items.pkl"
os.makedirs("models", exist_ok=True)

# ─── Step 4: Load from SQL + build basket format ───────────────────────────────
print("📊 Loading transactions from SQLite...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT InvoiceID, Product FROM Transactions", conn)
conn.close()
print(f"   {len(df):,} rows | {df['InvoiceID'].nunique():,} invoices | {df['Product'].nunique():,} products")

# Group by invoice → list of products (our format)
print("\n🛒 Building basket lists...")
baskets = df.groupby("InvoiceID")["Product"].apply(list).tolist()
print(f"   Total baskets: {len(baskets):,}")

# Save item list
all_items = sorted(df["Product"].unique().tolist())
with open(ITEMS_PKL, "wb") as f:
    pickle.dump(all_items, f)
print(f"   Saved {len(all_items)} unique items → {ITEMS_PKL}")

# ─── Step 5: FP-Growth ─────────────────────────────────────────────────────────
print("\n🌳 Running FP-Growth (min_support=0.02)...")
frequent_itemsets = fpgrowth(
    baskets,
    min_support=0.02,
    max_len=4
)
print(f"   Found {len(frequent_itemsets):,} frequent itemsets")

# ─── Step 6: Association Rules ─────────────────────────────────────────────────
print("\n📐 Generating association rules (confidence≥0.4, lift≥1.0)...")
rules_list = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.4
)

# Filter for quality rules
rules_list = [r for r in rules_list if r["lift"] >= 1.0 and r["confidence"] >= 0.4]

# Convert to DataFrame
import pandas as pd
rules = pd.DataFrame(rules_list)

# Add readable string columns
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

# Round metrics
for col in ["support", "confidence", "lift", "leverage", "conviction"]:
    if col in rules.columns:
        rules[col] = rules[col].round(4)

rules.sort_values("lift", ascending=False, inplace=True)
rules.reset_index(drop=True, inplace=True)

print(f"   ✅ {len(rules):,} quality rules generated")
print(f"\n   Top 5 rules by lift:")
top = rules[["antecedents_str", "consequents_str", "confidence", "lift"]].head(5)
for _, row in top.iterrows():
    print(f"   [{row['antecedents_str']}] → [{row['consequents_str']}]  "
          f"conf={row['confidence']:.2f}  lift={row['lift']:.2f}")

# Save
rules.to_csv(RULES_CSV, index=False)
print(f"\n💾 Saved rules CSV → {RULES_CSV}")

with open(RULES_PKL, "wb") as f:
    pickle.dump(rules, f)
print(f"💾 Saved rules pickle → {RULES_PKL}")
print("\n🏁 Phase 2 complete! Model ready.")
