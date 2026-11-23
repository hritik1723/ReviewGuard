"""Association rule mining (Apriori & FP-Growth) for ReviewGuard.

Classical techniques:
1. Feature discretization into binary transaction itemsets
2. Frequent itemset mining via Apriori (mlxtend)
3. Frequent itemset mining via FP-Growth (mlxtend)
4. Association rule extraction with Support, Confidence, and Lift metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from .load_data import has_behavioral_metadata, load_dataset


def discretize_review_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Transform continuous and categorical review attributes into boolean transaction flags."""
    transactions = pd.DataFrame(index=df.index)

    # 1. Rating
    rating = pd.to_numeric(df.get("rating", 3.0), errors="coerce").fillna(3.0)
    transactions["rating_5"] = rating == 5.0
    transactions["rating_1"] = rating == 1.0

    # 2. Text properties
    text_len = df["text"].astype(str).str.len()
    word_count = df["text"].astype(str).str.split().str.len()
    excl_count = df["text"].astype(str).str.count("!")

    transactions["short_review"] = word_count < 15
    transactions["long_review"] = word_count > 60
    transactions["high_exclamations"] = excl_count >= 2

    # 3. Behavioral signals
    if "verified_purchase" in df.columns:
        ver = (
            df["verified_purchase"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "verified"])
        )
        transactions["unverified_buyer"] = ~ver
        transactions["verified_buyer"] = ver
    else:
        transactions["unverified_buyer"] = False
        transactions["verified_buyer"] = True

    if "account_age_days" in df.columns:
        age = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(365.0)
        transactions["new_account"] = age < 30
        transactions["mature_account"] = age >= 180
    else:
        transactions["new_account"] = False
        transactions["mature_account"] = True

    if "user_id" in df.columns:
        user_counts = df.groupby("user_id")["rating"].transform("count")
        transactions["burst_posting"] = user_counts >= 3
    else:
        transactions["burst_posting"] = False

    # 4. Target class
    if "label" in df.columns:
        transactions["label_fake"] = df["label"] == 1
        transactions["label_genuine"] = df["label"] == 0

    return transactions


def mine_association_rules(
    dataset_name: str = "sample",
    algorithm: str = "fpgrowth",
    min_support: float = 0.05,
    min_confidence: float = 0.4,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Mine frequent itemsets and extract association rules using Apriori or FP-Growth."""
    if output_dir is None:
        output_dir = Path.cwd() / "reviewguard" / "models"
        if not output_dir.exists():
            output_dir = Path.cwd() / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_dataset(dataset_name)

    if not meta["has_metadata"]:
        return {
            "requires_metadata": True,
            "message": (
                "Pattern Mining requires a dataset with behavioral metadata (user_id, verified_purchase, account_age_days). "
                "The current dataset does not include reviewer signals. Please switch to the "
                "Synthetic Sample Reviews dataset to mine behavioral association rules."
            ),
        }

    transactions = discretize_review_transactions(df)

    # Mine frequent itemsets
    algo_clean = algorithm.lower().strip()
    if algo_clean == "apriori":
        frequent_itemsets = apriori(
            transactions,
            min_support=min_support,
            use_colnames=True,
        )
    else:
        algo_clean = "fpgrowth"
        frequent_itemsets = fpgrowth(
            transactions,
            min_support=min_support,
            use_colnames=True,
        )

    if frequent_itemsets.empty:
        return {
            "requires_metadata": False,
            "algorithm": algo_clean,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "total_rules": 0,
            "rules": [],
            "top_itemsets": [],
        }

    # Extract association rules
    rules_df = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    # Sort by lift descending
    rules_df = rules_df.sort_values("lift", ascending=False)

    rules_list = []
    for _, row in rules_df.head(50).iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])
        is_fake = "label_fake" in con
        rules_list.append(
            {
                "antecedents": ant,
                "consequents": con,
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
                "is_fake_target": is_fake,
            }
        )

    # Top frequent itemsets
    top_itemsets = []
    for _, row in (
        frequent_itemsets.sort_values("support", ascending=False)
        .head(15)
        .iterrows()
    ):
        top_itemsets.append(
            {
                "itemset": list(row["itemsets"]),
                "support": round(float(row["support"]), 4),
            }
        )

    result = {
        "requires_metadata": False,
        "algorithm": algo_clean,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "total_rules": len(rules_list),
        "rules": rules_list,
        "top_itemsets": top_itemsets,
    }

    with open(output_dir / "association_rules.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
