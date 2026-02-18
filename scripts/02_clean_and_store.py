"""
Phase 1 – Step 2 & 3: Clean data + Store in SQLite
Handles both the generated dataset AND the real Kaggle Online Retail dataset.
"""

import pandas as pd
import sqlite3
import os

# ─── Config ────────────────────────────────────────────────────────────────────
RAW_CSV   = "data/raw_retail.csv"    # change to your Kaggle CSV path if needed
CLEAN_CSV = "data/clean_retail.csv"
DB_PATH   = "data/retail_store.db"

# ─── Load ───────────────────────────────────────────────────────────────────────
print("📂 Loading raw data...")
df = pd.read_csv(RAW_CSV)
print(f"   Raw rows: {len(df):,}")

# ─── Clean ─────────────────────────────────────────────────────────────────────
print("\n🧹 Cleaning data...")

# 1. Drop null descriptions / invoice numbers
before = len(df)
df.dropna(subset=["Description", "InvoiceNo"], inplace=True)
print(f"   Removed {before - len(df):,} rows with null Description/Invoice")

# 2. Remove cancelled invoices (start with 'C')
before = len(df)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
print(f"   Removed {before - len(df):,} cancelled invoices")

# 3. Remove negative quantities
before = len(df)
df = df[df["Quantity"] > 0]
print(f"   Removed {before - len(df):,} rows with non-positive quantity")

# 4. Remove zero/negative prices (if column exists)
if "UnitPrice" in df.columns:
    before = len(df)
    df = df[df["UnitPrice"] > 0]
    print(f"   Removed {before - len(df):,} rows with invalid price")

# 5. Standardise column names (handles Kaggle naming)
rename_map = {}
col_lower = {c.lower(): c for c in df.columns}
if "invoiceno"  in col_lower: rename_map[col_lower["invoiceno"]]  = "InvoiceID"
if "invoicedate" in col_lower: rename_map[col_lower["invoicedate"]] = "Date"
if "description" in col_lower: rename_map[col_lower["description"]] = "Product"
if "quantity"    in col_lower: rename_map[col_lower["quantity"]]    = "Quantity"
df.rename(columns=rename_map, inplace=True)

# Ensure we have the four core columns
required = ["InvoiceID", "Product", "Quantity", "Date"]
for col in required:
    if col not in df.columns:
        df[col] = "Unknown"

# Keep only required + extra useful columns
keep = required.copy()
for extra in ["UnitPrice", "CustomerID", "Country"]:
    if extra in df.columns:
        keep.append(extra)
df = df[keep]

# 6. Strip whitespace from product names
df["Product"] = df["Product"].str.strip().str.title()

# 7. Remove duplicates (same invoice + product)
before = len(df)
df.drop_duplicates(subset=["InvoiceID", "Product"], inplace=True)
print(f"   Removed {before - len(df):,} duplicate invoice-product pairs")

print(f"\n✅ Clean rows: {len(df):,}")
print(f"   Unique transactions: {df['InvoiceID'].nunique():,}")
print(f"   Unique products:     {df['Product'].nunique():,}")

# ─── Save clean CSV ────────────────────────────────────────────────────────────
df.to_csv(CLEAN_CSV, index=False)
print(f"\n💾 Saved → {CLEAN_CSV}")

# ─── Store in SQLite ───────────────────────────────────────────────────────────
print(f"\n🗄️  Writing to SQLite: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS Transactions")
cursor.execute("""
    CREATE TABLE Transactions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        InvoiceID   TEXT    NOT NULL,
        Product     TEXT    NOT NULL,
        Quantity    INTEGER NOT NULL,
        Date        TEXT    NOT NULL,
        UnitPrice   REAL,
        CustomerID  TEXT,
        Country     TEXT
    )
""")

df.to_sql("Transactions", conn, if_exists="append", index=False)

# Verify
row_count = cursor.execute("SELECT COUNT(*) FROM Transactions").fetchone()[0]
prod_count = cursor.execute("SELECT COUNT(DISTINCT Product) FROM Transactions").fetchone()[0]
inv_count  = cursor.execute("SELECT COUNT(DISTINCT InvoiceID) FROM Transactions").fetchone()[0]

conn.commit()
conn.close()

print(f"   ✅ Rows inserted:       {row_count:,}")
print(f"   ✅ Unique transactions: {inv_count:,}")
print(f"   ✅ Unique products:     {prod_count:,}")
print(f"\n🏁 Phase 1 complete! Database ready at: {DB_PATH}")
