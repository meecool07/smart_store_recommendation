"""
Phase 3 – Steps 7 & 8: Production-level recommendation engine.
Import this module in Streamlit and anywhere else you need recommendations.
"""

import pickle
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Optional

BASE_DIR   = Path(__file__).parent
RULES_PKL  = BASE_DIR / "models" / "association_rules.pkl"
ITEMS_PKL  = BASE_DIR / "models" / "all_items.pkl"
DB_PATH    = BASE_DIR / "data"    / "retail_store.db"


# ═══════════════════════════════════════════════════════════════════════════════
class RecommendationEngine:
    """
    FP-Growth based product recommendation engine.

    Usage:
        engine = RecommendationEngine()
        recs   = engine.recommend(["Bread", "Milk"])
    """

    def __init__(self):
        self.rules: Optional[pd.DataFrame] = None
        self.all_items: Optional[List[str]] = None
        self._load()

    # ── Load model ─────────────────────────────────────────────────────────────
    def _load(self):
        if not RULES_PKL.exists():
            raise FileNotFoundError(
                f"Model not found at {RULES_PKL}. "
                "Run scripts/03_train_model.py first."
            )
        with open(RULES_PKL, "rb") as f:
            self.rules = pickle.load(f)

        # Ensure antecedents/consequents are frozensets
        if "antecedents" in self.rules.columns:
            self.rules["antecedents"] = self.rules["antecedents"].apply(
                lambda x: frozenset(x) if not isinstance(x, frozenset) else x
            )
            self.rules["consequents"] = self.rules["consequents"].apply(
                lambda x: frozenset(x) if not isinstance(x, frozenset) else x
            )

        if ITEMS_PKL.exists():
            with open(ITEMS_PKL, "rb") as f:
                self.all_items = pickle.load(f)
        else:
            self.all_items = []

    # ── Core recommendation function ───────────────────────────────────────────
    def recommend(
        self,
        input_items: List[str],
        top_n: int = 5,
        min_confidence: float = 0.0,
        min_lift: float = 1.0,
    ) -> pd.DataFrame:
        """
        Given a list of items in the current basket, return recommended items.

        Returns DataFrame with columns:
            product, confidence, lift, support, based_on
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()

        input_set = frozenset([i.strip().title() for i in input_items])
        results   = []

        for _, rule in self.rules.iterrows():
            antecedents = rule["antecedents"]
            consequents = rule["consequents"]

            # Rule fires if ALL antecedent items are in the basket
            if antecedents.issubset(input_set):
                confidence = rule["confidence"]
                lift       = rule["lift"]
                support    = rule["support"]

                if confidence < min_confidence or lift < min_lift:
                    continue

                for product in consequents:
                    if product not in input_set:   # don't recommend what's in basket
                        results.append({
                            "product":    product,
                            "confidence": round(confidence, 4),
                            "lift":       round(lift, 4),
                            "support":    round(support, 4),
                            "based_on":   ", ".join(sorted(antecedents)),
                        })

        if not results:
            return pd.DataFrame(columns=["product","confidence","lift","support","based_on"])

        # Deduplicate: keep best lift per product
        rec_df = pd.DataFrame(results)
        rec_df = (
            rec_df.sort_values("lift", ascending=False)
                  .drop_duplicates(subset=["product"])
                  .head(top_n)
                  .reset_index(drop=True)
        )
        return rec_df

    # ── Fuzzy product search ───────────────────────────────────────────────────
    def search_items(self, query: str, max_results: int = 10) -> List[str]:
        """Find items in the catalogue that contain the query string."""
        q = query.strip().lower()
        if not self.all_items:
            return []
        return [item for item in self.all_items if q in item.lower()][:max_results]

    # ── Stats helpers (for Power BI / dashboard) ───────────────────────────────
    def top_rules(self, n: int = 20) -> pd.DataFrame:
        if self.rules is None:
            return pd.DataFrame()
        cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        available = [c for c in cols if c in self.rules.columns]
        return self.rules[available].head(n)

    def top_products(self, n: int = 20) -> pd.DataFrame:
        """Return most frequently appearing products in rule consequents."""
        if self.rules is None:
            return pd.DataFrame()
        prod_series = self.rules["consequents"].explode()
        counts = prod_series.value_counts().head(n).reset_index()
        counts.columns = ["product", "rule_appearances"]
        return counts

    def get_all_items(self) -> List[str]:
        return self.all_items or []

    # ── SQL helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def get_product_frequency(top_n: int = 30) -> pd.DataFrame:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            f"""
            SELECT Product, SUM(Quantity) as TotalQty, COUNT(*) as Frequency
            FROM Transactions
            GROUP BY Product
            ORDER BY Frequency DESC
            LIMIT {top_n}
            """,
            conn
        )
        conn.close()
        return df

    @staticmethod
    def get_monthly_sales() -> pd.DataFrame:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            """
            SELECT substr(Date,1,7) as Month,
                   COUNT(DISTINCT InvoiceID) as Transactions,
                   SUM(Quantity) as TotalItems
            FROM Transactions
            GROUP BY Month
            ORDER BY Month
            """,
            conn
        )
        conn.close()
        return df

    @staticmethod
    def get_category_sales() -> pd.DataFrame:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            """
            SELECT Country, COUNT(DISTINCT InvoiceID) as Transactions
            FROM Transactions
            WHERE Country IS NOT NULL AND Country != 'Unknown'
            GROUP BY Country
            ORDER BY Transactions DESC
            """,
            conn
        )
        conn.close()
        return df


# ─── CLI Quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RecommendationEngine()
    print(f"Loaded {len(engine.rules)} rules, {len(engine.all_items)} products\n")

    test_baskets = [
        ["Bread"],
        ["Bread", "Butter"],
        ["Coffee", "Milk"],
        ["Pasta", "Cheese"],
        ["Tea"],
    ]
    for basket in test_baskets:
        recs = engine.recommend(basket, top_n=3)
        print(f"Input: {basket}")
        if recs.empty:
            print("  → No recommendations found")
        else:
            for _, r in recs.iterrows():
                print(f"  → {r['product']:20s}  conf={r['confidence']:.2f}  lift={r['lift']:.2f}")
        print()
