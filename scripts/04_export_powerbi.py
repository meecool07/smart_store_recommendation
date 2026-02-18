"""
Phase 5: Export data for Power BI Dashboard
Run this to generate Excel/CSV files that Power BI can connect to.
"""

import pandas as pd
import sqlite3
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fp_growth_engine import fpgrowth, association_rules

DB_PATH   = Path(__file__).parent.parent / "data"   / "retail_store.db"
RULES_PKL = Path(__file__).parent.parent / "models" / "association_rules.pkl"
OUT_DIR   = Path(__file__).parent.parent / "data"   / "powerbi_exports"
OUT_DIR.mkdir(exist_ok=True, parents=True)

print("📊 Preparing Power BI exports...\n")

# ─── 1. Transactions summary ───────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

tx = pd.read_sql("""
    SELECT
        Product,
        SUM(Quantity)                    AS TotalQuantity,
        COUNT(*)                         AS TransactionCount,
        COUNT(DISTINCT InvoiceID)        AS UniqueInvoices,
        SUM(Quantity * COALESCE(UnitPrice,0)) AS Revenue
    FROM Transactions
    GROUP BY Product
    ORDER BY TotalQuantity DESC
""", conn)
tx.to_csv(OUT_DIR / "product_summary.csv", index=False)
print(f"✅ product_summary.csv      ({len(tx)} rows)")

# ─── 2. Monthly trend ─────────────────────────────────────────────────────────
monthly = pd.read_sql("""
    SELECT
        substr(Date,1,7) AS Month,
        COUNT(DISTINCT InvoiceID) AS Transactions,
        SUM(Quantity)             AS ItemsSold
    FROM Transactions
    GROUP BY Month ORDER BY Month
""", conn)
monthly.to_csv(OUT_DIR / "monthly_trend.csv", index=False)
print(f"✅ monthly_trend.csv        ({len(monthly)} rows)")

# ─── 3. Country breakdown ─────────────────────────────────────────────────────
country = pd.read_sql("""
    SELECT Country,
           COUNT(DISTINCT InvoiceID) AS Transactions,
           SUM(Quantity)             AS ItemsSold
    FROM Transactions
    WHERE Country IS NOT NULL AND Country != 'Unknown'
    GROUP BY Country ORDER BY Transactions DESC
""", conn)
country.to_csv(OUT_DIR / "country_breakdown.csv", index=False)
print(f"✅ country_breakdown.csv    ({len(country)} rows)")

conn.close()

# ─── 4. Association rules ─────────────────────────────────────────────────────
with open(RULES_PKL, "rb") as f:
    rules = pickle.load(f)

pbi_rules = rules[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
pbi_rules.columns = ["Antecedents","Consequents","Support","Confidence","Lift"]
pbi_rules.to_csv(OUT_DIR / "association_rules.csv", index=False)
print(f"✅ association_rules.csv    ({len(pbi_rules)} rules)")

# ─── 5. Top bundles ───────────────────────────────────────────────────────────
top_bundles = pbi_rules.nlargest(30, "Lift").copy()
top_bundles["Bundle"] = top_bundles["Antecedents"] + " → " + top_bundles["Consequents"]
top_bundles.to_csv(OUT_DIR / "top_bundles.csv", index=False)
print(f"✅ top_bundles.csv          ({len(top_bundles)} bundles)")

print(f"\n📁 All exports → {OUT_DIR}")
print("\n Power BI Setup:")
print("  1. Open Power BI Desktop")
print("  2. Get Data → Text/CSV")
print("  3. Import each CSV file")
print("  4. Create relationships on 'Product' column")
print("  5. Build visuals:")
print("     • Bar chart: product_summary → TotalQuantity")
print("     • Line chart: monthly_trend → Transactions")
print("     • Map: country_breakdown → Transactions")
print("     • Table: association_rules (filtered by Lift > 2)")
print("     • Scatter: Support vs Confidence, sized by Lift")
