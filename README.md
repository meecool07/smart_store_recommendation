# 🛒 Smart Store Recommendation System

> **End-to-end retail recommendation engine using FP-Growth association rule mining**  
> Industry-level · Resume-ready · Deployable on Streamlit Cloud

---

## 📐 Architecture

```
SQL Database (SQLite)
      ↓
Python Pipeline (FP-Growth)
      ↓
Saved Rules (.pkl / .csv)
      ↓
Streamlit App ──→ User Gets Recommendations
      
SQL → Power BI Dashboard (business insights)
```

---

## 🗂️ Project Structure

```
smart_store_recommender/
│
├── app.py                    ← Streamlit UI (Phase 4)
├── recommender.py            ← Recommendation engine (Phase 3)
├── fp_growth_engine.py       ← Pure Python FP-Growth (no mlxtend needed)
├── requirements.txt
│
├── scripts/
│   ├── 01_generate_data.py   ← Synthetic data (or use Kaggle dataset)
│   ├── 02_clean_and_store.py ← Clean + store in SQLite (Phase 1)
│   ├── 03_train_model.py     ← FP-Growth training (Phase 2)
│   └── 04_export_powerbi.py  ← Power BI CSV exports (Phase 5)
│
├── data/
│   ├── raw_retail.csv
│   ├── clean_retail.csv
│   ├── retail_store.db       ← SQLite database
│   └── powerbi_exports/      ← CSV files for Power BI
│
└── models/
    ├── association_rules.pkl ← Trained rules (pickle)
    ├── association_rules.csv ← Trained rules (CSV)
    └── all_items.pkl         ← Product catalogue
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Option A — Use synthetic data
```bash
python scripts/01_generate_data.py
```

### 2. Option B — Use real Kaggle dataset
1. Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
2. Save as `data/raw_retail.csv`
3. The cleaning script handles both formats

### 3. Clean + store in SQLite
```bash
python scripts/02_clean_and_store.py
```

### 4. Train FP-Growth model
```bash
python scripts/03_train_model.py
```

### 5. Launch Streamlit app
```bash
streamlit run app.py
```

### 6. (Optional) Export for Power BI
```bash
python scripts/04_export_powerbi.py
```

---

## 🧠 How FP-Growth Works

```
All Transactions
      ↓
Count item frequencies → Filter by min_support
      ↓
Build FP-Tree (compressed structure)
      ↓
Mine conditional pattern bases recursively
      ↓
Frequent Itemsets (e.g. {Bread, Butter, Milk}: support=0.08)
      ↓
Generate Association Rules:
  [Bread, Butter] → [Milk]   confidence=0.72  lift=3.1
```

### Why FP-Growth over Apriori?
| Feature | Apriori | FP-Growth |
|---------|---------|-----------|
| Database scans | Once per itemset size | Twice (build + mine) |
| Memory | Exponential candidate sets | Compact FP-tree |
| Speed on large data | Slow | 10–100× faster |

---

## 📊 Association Rule Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Support** | P(A ∪ B) | How often the combination appears |
| **Confidence** | P(B\|A) = P(A∪B)/P(A) | How often rule is correct |
| **Lift** | Conf / P(B) | How much better than random (>1 = positive association) |

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub (public or private)
2. Go to https://share.streamlit.io
3. Connect your repo → set **Main file path** = `app.py`
4. Add this to `.streamlit/config.toml`:
   ```toml
   [server]
   headless = true
   ```
5. In Streamlit Cloud, add to `packages.txt`:
   ```
   python3-dev
   ```
6. Your app will be live at `https://your-app.streamlit.app` 🎉

> **Note**: Pre-train the model locally and commit the `models/` and `data/retail_store.db` files to your repo, or run the setup scripts in `startup.py`.

---

## 📋 Power BI Dashboard Setup

After running `scripts/04_export_powerbi.py`:

1. Open **Power BI Desktop**
2. **Get Data → Text/CSV** → import each file from `data/powerbi_exports/`
3. Build these visuals:
   - **Bar chart**: `product_summary.csv` → Top products by quantity
   - **Line chart**: `monthly_trend.csv` → Transaction volume over time
   - **Map**: `country_breakdown.csv` → Sales by country
   - **Table**: `association_rules.csv` → Filter Lift > 2.0
   - **Scatter plot**: Support vs Confidence, bubble size = Lift

---

## 🏆 Resume One-Liner

> *Designed and deployed an end-to-end retail recommendation engine using FP-Growth association rule mining. Integrated SQLite database for transaction storage, implemented custom FP-Growth algorithm from scratch, developed dynamic recommendation logic handling multi-item baskets, and deployed an interactive Streamlit application with real-time business insights dashboard.*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Database | SQLite (dev) / MySQL or PostgreSQL (prod) |
| ML Algorithm | FP-Growth (custom implementation) |
| Backend | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Frontend | Streamlit |
| Charts | Plotly Express |
| BI Dashboard | Power BI |
| Deployment | Streamlit Cloud |
