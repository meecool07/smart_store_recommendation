"""
fp_growth_engine.py
Pure-Python FP-Growth + Association Rules
No mlxtend required — works offline.
"""

from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# FP-Tree node
# ═══════════════════════════════════════════════════════════════════════════════
class FPNode:
    def __init__(self, item, parent=None):
        self.item    = item
        self.count   = 0
        self.parent  = parent
        self.children: Dict[str, FPNode] = {}
        self.link: Optional[FPNode] = None   # header table link

    def increment(self, count: int = 1):
        self.count += count


# ═══════════════════════════════════════════════════════════════════════════════
# FP-Tree
# ═══════════════════════════════════════════════════════════════════════════════
class FPTree:
    def __init__(self, transactions, min_support: int, root_value=None, root_count=0):
        self.root = FPNode(root_value)
        self.root.count = root_count
        self.header: Dict[str, FPNode] = {}   # item → first node in chain

        self._build(transactions, min_support)

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self, transactions, min_support: int):
        # 1. Count frequencies
        freq: Dict[str, int] = defaultdict(int)
        for t in transactions:
            for item in t:
                freq[item] += 1

        # 2. Filter & sort each transaction by frequency desc
        self.freq = {k: v for k, v in freq.items() if v >= min_support}

        for transaction in transactions:
            filtered = sorted(
                [item for item in transaction if item in self.freq],
                key=lambda x: (-self.freq[x], x)
            )
            if filtered:
                self._insert(filtered)

    def _insert(self, items: List[str]):
        node = self.root
        for item in items:
            if item in node.children:
                node.children[item].increment()
            else:
                new_node = FPNode(item, parent=node)
                new_node.count = 1
                node.children[item] = new_node
                # update header chain
                if item not in self.header:
                    self.header[item] = new_node
                else:
                    current = self.header[item]
                    while current.link:
                        current = current.link
                    current.link = new_node
            node = node.children[item]

    def is_empty(self) -> bool:
        return len(self.root.children) == 0

    # ── Conditional pattern base ───────────────────────────────────────────────
    def conditional_pattern_base(self, item: str) -> List[Tuple[List[str], int]]:
        patterns = []
        node = self.header.get(item)
        while node:
            prefix, count = [], node.count
            parent = node.parent
            while parent.item is not None:
                prefix.append(parent.item)
                parent = parent.parent
            if prefix:
                patterns.append((prefix[::-1], count))
            node = node.link
        return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# FP-Growth mining
# ═══════════════════════════════════════════════════════════════════════════════
def fpgrowth_mine(tree: FPTree, min_support: int, prefix: frozenset) -> List[Tuple[frozenset, int]]:
    """Recursively mine frequent itemsets from the FP-tree."""
    itemsets = []

    for item, node in sorted(tree.header.items(), key=lambda x: tree.freq.get(x[0], 0)):
        new_itemset = prefix | {item}
        support = 0
        n = node
        while n:
            support += n.count
            n = n.link

        if support < min_support:
            continue

        itemsets.append((new_itemset, support))

        # Build conditional tree
        cond_patterns = tree.conditional_pattern_base(item)
        cond_transactions = []
        for pattern, count in cond_patterns:
            cond_transactions.extend([pattern] * count)

        if cond_transactions:
            cond_tree = FPTree(cond_transactions, min_support)
            if not cond_tree.is_empty():
                sub_itemsets = fpgrowth_mine(cond_tree, min_support, new_itemset)
                itemsets.extend(sub_itemsets)

    return itemsets


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════
def fpgrowth(transactions: List[List[str]], min_support: float = 0.02, max_len: int = 4):
    """
    Mine frequent itemsets.

    Parameters
    ----------
    transactions  : list of lists (each inner list = one basket)
    min_support   : float in [0, 1]
    max_len       : max itemset size

    Returns
    -------
    list of dicts: [{"support": float, "itemsets": frozenset}, ...]
    """
    n = len(transactions)
    min_count = max(1, int(min_support * n))

    tree = FPTree(transactions, min_count)
    raw  = fpgrowth_mine(tree, min_count, frozenset())

    result = []
    for itemset, count in raw:
        if len(itemset) <= max_len:
            result.append({
                "support":  round(count / n, 6),
                "itemsets": itemset,
            })

    return result


def association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4):
    """
    Generate association rules from frequent itemsets.

    Parameters
    ----------
    frequent_itemsets : list of dicts (output of fpgrowth())
    metric            : "confidence" or "lift"
    min_threshold     : minimum value of the chosen metric

    Returns
    -------
    list of dicts with keys:
        antecedents, consequents, support, confidence, lift, leverage, conviction
    """
    # Build support lookup
    support_map = {fs["itemsets"]: fs["support"] for fs in frequent_itemsets}

    rules = []
    for fs in frequent_itemsets:
        itemset = fs["itemsets"]
        if len(itemset) < 2:
            continue

        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent  = frozenset(antecedent)
                consequent  = itemset - antecedent

                ant_support  = support_map.get(antecedent, 0)
                cons_support = support_map.get(consequent, 0)
                rule_support = fs["support"]

                if ant_support == 0 or cons_support == 0:
                    continue

                confidence  = rule_support / ant_support
                lift        = confidence / cons_support
                leverage    = rule_support - ant_support * cons_support
                conviction  = (1 - cons_support) / (1 - confidence + 1e-10) if confidence < 1 else float("inf")

                value = confidence if metric == "confidence" else lift
                if value >= min_threshold:
                    rules.append({
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "support":     round(rule_support, 6),
                        "confidence":  round(confidence, 6),
                        "lift":        round(lift, 6),
                        "leverage":    round(leverage, 6),
                        "conviction":  round(min(conviction, 999.0), 4),
                    })

    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules
