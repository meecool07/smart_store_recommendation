"""
Phase 4 – Streamlit UI: Smart Store Recommendation System
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recommender import RecommendationEngine

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Store Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #1a1d27; }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid #2a4a7f;
    }
    .main-header h1 { color: #4fc3f7; margin: 0; font-size: 2rem; }
    .main-header p  { color: #90caf9; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d27, #212431);
        border: 1px solid #2a4a7f; border-radius: 10px;
        padding: 1.2rem; text-align: center;
    }
    .metric-card h3 { color: #4fc3f7; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #aab4c8; margin: 0.2rem 0 0; font-size: 0.85rem; }
    .rec-card {
        background: #1a1d27; border-left: 4px solid #4fc3f7;
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    .rec-card h4 { color: #e8f4fd; margin: 0 0 0.4rem; }
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 0.78rem; margin-right: 6px;
    }
    .badge-blue   { background: #0d47a1; color: #90caf9; }
    .badge-green  { background: #1b5e20; color: #a5d6a7; }
    .badge-orange { background: #e65100; color: #ffcc80; }
    .section-header {
        color: #4fc3f7; font-size: 1.15rem; font-weight: 600;
        border-bottom: 1px solid #2a4a7f; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem;
    }
    .stMultiSelect > div > div { background: #1a1d27 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Load engine (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading recommendation model…")
def load_engine():
    return RecommendationEngine()

@st.cache_data(ttl=300)
def get_product_freq():
    return RecommendationEngine.get_product_frequency(30)

@st.cache_data(ttl=300)
def get_monthly_sales():
    return RecommendationEngine.get_monthly_sales()

@st.cache_data(ttl=300)
def get_country_sales():
    return RecommendationEngine.get_category_sales()

try:
    engine = load_engine()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛒 Smart Store")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🎯 Recommendations", "📊 Business Dashboard", "📋 Association Rules", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model Settings**")
    top_n        = st.slider("Max recommendations", 3, 10, 5)
    min_conf     = st.slider("Min Confidence", 0.0, 1.0, 0.4, 0.05)
    min_lift     = st.slider("Min Lift", 1.0, 5.0, 1.0, 0.1)
    st.markdown("---")
    if model_loaded:
        st.success(f"✅ Model: {len(engine.rules):,} rules")
        st.info(f"📦 Products: {len(engine.all_items):,}")
    else:
        st.error("⚠️ Model not loaded. Run training scripts first.")

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛒 Smart Store Recommendation System</h1>
    <p>FP-Growth Association Rule Mining · SQL Backend · Real-time Recommendations</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
if "Recommendations" in page:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">🧺 Build Your Basket</div>', unsafe_allow_html=True)

        if model_loaded:
            all_items = engine.get_all_items()
            selected = st.multiselect(
                "Select items already in your basket:",
                options=all_items,
                placeholder="Start typing a product name…",
            )

            # Text input as alternative
            st.markdown("**Or type custom items:**")
            custom_input = st.text_input("Comma-separated", placeholder="e.g. Bread, Milk")
            if custom_input:
                custom_items = [i.strip().title() for i in custom_input.split(",") if i.strip()]
                selected = list(set(selected + custom_items))

            if selected:
                st.markdown("**Current basket:**")
                st.markdown(" ".join([f"`{item}`" for item in selected]))

            get_recs = st.button("🔍 Get Recommendations", use_container_width=True)

            if get_recs and selected:
                recs = engine.recommend(selected, top_n=top_n,
                                        min_confidence=min_conf, min_lift=min_lift)

                if recs.empty:
                    st.warning("No recommendations found. Try adding more items or adjusting filters.")
                else:
                    st.markdown(f'<div class="section-header">✨ Recommended for You ({len(recs)})</div>',
                                unsafe_allow_html=True)
                    for i, row in recs.iterrows():
                        conf_pct = int(row["confidence"] * 100)
                        lift_val = row["lift"]
                        st.markdown(f"""
                        <div class="rec-card">
                            <h4>#{i+1} &nbsp; {row['product']}</h4>
                            <span class="badge badge-blue">Confidence: {conf_pct}%</span>
                            <span class="badge badge-green">Lift: {lift_val:.2f}x</span>
                            <span class="badge badge-orange">Support: {row['support']:.3f}</span>
                            <br><small style="color:#6a7c99">Based on: {row['based_on']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            elif get_recs and not selected:
                st.info("Please select at least one product.")
        else:
            st.error("Run setup scripts first (see About page).")

    with col2:
        st.markdown('<div class="section-header">📈 Top Products</div>', unsafe_allow_html=True)
        try:
            freq_df = get_product_freq()
            if not freq_df.empty:
                fig = px.bar(
                    freq_df.head(15),
                    x="Frequency", y="Product", orientation="h",
                    color="Frequency",
                    color_continuous_scale=["#0d47a1", "#4fc3f7", "#00e5ff"],
                    template="plotly_dark",
                    labels={"Frequency": "Transaction Count"},
                )
                fig.update_layout(
                    plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                    height=450, showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(categoryorder="total ascending"),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart not available: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Business Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown('<div class="section-header">📊 Business Intelligence Dashboard</div>',
                unsafe_allow_html=True)

    # KPI cards
    try:
        DB = Path(__file__).parent / "data" / "retail_store.db"
        conn = sqlite3.connect(DB)
        total_tx   = pd.read_sql("SELECT COUNT(DISTINCT InvoiceID) as n FROM Transactions", conn).iloc[0,0]
        total_prod = pd.read_sql("SELECT COUNT(DISTINCT Product) as n FROM Transactions", conn).iloc[0,0]
        total_qty  = pd.read_sql("SELECT SUM(Quantity) as n FROM Transactions", conn).iloc[0,0]
        conn.close()
        n_rules = len(engine.rules) if model_loaded else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{total_tx:,}",   "Total Transactions"),
            (c2, f"{total_prod:,}", "Unique Products"),
            (c3, f"{total_qty:,}",  "Items Sold"),
            (c4, f"{n_rules:,}",    "Association Rules"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <h3>{val}</h3><p>{label}</p>
            </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"KPIs unavailable: {e}")

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**🗓️ Monthly Transaction Volume**")
        try:
            monthly = get_monthly_sales()
            fig = px.area(monthly, x="Month", y="Transactions",
                          template="plotly_dark",
                          color_discrete_sequence=["#4fc3f7"])
            fig.update_layout(plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                              margin=dict(l=0,r=0,t=10,b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

    with col2:
        st.markdown("**🌍 Sales by Country**")
        try:
            country = get_country_sales()
            fig = px.pie(country.head(8), names="Country", values="Transactions",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                              margin=dict(l=0,r=0,t=10,b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

    if model_loaded:
        st.markdown("**🔗 Top Product Bundles (by Lift)**")
        try:
            top_rules = engine.top_rules(20)
            if not top_rules.empty:
                fig = px.scatter(
                    top_rules,
                    x="confidence", y="lift", size="support",
                    hover_data=["antecedents_str","consequents_str"],
                    template="plotly_dark",
                    color="lift",
                    color_continuous_scale=["#0d47a1","#4fc3f7","#00e5ff"],
                    labels={"confidence":"Confidence","lift":"Lift"},
                )
                fig.update_layout(plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                                  height=350, margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Association Rules Table
# ═══════════════════════════════════════════════════════════════════════════════
elif "Rules" in page:
    st.markdown('<div class="section-header">📋 All Association Rules</div>', unsafe_allow_html=True)

    if model_loaded:
        rules_df = engine.rules.copy()
        display_cols = ["antecedents_str","consequents_str","support","confidence","lift"]
        display_cols = [c for c in display_cols if c in rules_df.columns]
        disp = rules_df[display_cols].rename(columns={
            "antecedents_str":"IF (Antecedent)",
            "consequents_str":"THEN (Consequent)",
            "support":"Support",
            "confidence":"Confidence",
            "lift":"Lift",
        })

        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Search product", placeholder="e.g. Bread")
        with col2:
            sort_by = st.selectbox("Sort by", ["Lift", "Confidence", "Support"], index=0)

        if search:
            mask = (
                disp["IF (Antecedent)"].str.contains(search, case=False, na=False) |
                disp["THEN (Consequent)"].str.contains(search, case=False, na=False)
            )
            disp = disp[mask]

        disp = disp.sort_values(sort_by, ascending=False)

        st.dataframe(
            disp,
            use_container_width=True,
            height=500,
            column_config={
                "Support":    st.column_config.ProgressColumn("Support",    min_value=0, max_value=1, format="%.3f"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.3f"),
                "Lift":       st.column_config.NumberColumn("Lift", format="%.2f"),
            }
        )
        st.caption(f"Showing {len(disp):,} rules")

        # Download
        st.download_button(
            "⬇️ Download Rules CSV",
            data=disp.to_csv(index=False),
            file_name="association_rules.csv",
            mime="text/csv",
        )
    else:
        st.error("Model not loaded.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    ## About This Project

    **Smart Store Recommendation System** is an end-to-end retail intelligence platform.

    ### Architecture
    ```
    SQL Database (SQLite)
          ↓
    FP-Growth Model (mlxtend)
          ↓
    Association Rules (.pkl)
          ↓
    Streamlit App ← you are here
          ↓
    User Gets Recommendations
    ```

    ### How to Run (First Time Setup)

    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Generate synthetic data (or place your Kaggle CSV in data/)
    python scripts/01_generate_data.py

    # 3. Clean data + store in SQLite
    python scripts/02_clean_and_store.py

    # 4. Train FP-Growth model
    python scripts/03_train_model.py

    # 5. Launch app
    streamlit run app.py
    ```

    ### Technology Stack
    | Component | Technology |
    |-----------|------------|
    | Database  | SQLite / MySQL |
    | ML Model  | FP-Growth (mlxtend) |
    | Backend   | Python 3.10+ |
    | Frontend  | Streamlit |
    | Charts    | Plotly Express |

    ### Why FP-Growth over Apriori?
    - **Faster**: No repeated database scans
    - **Memory efficient**: Compact FP-tree structure
    - **Scalable**: Handles large retail datasets
    """)
