"""Feature selection methods for ReviewGuard.

Classical techniques:
1. Chi-Square Test (select_k_best with chi2)
2. Information Gain / Mutual Information (mutual_info_classif)
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import issparse


def select_features_chi2(
    X,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> List[Dict]:
    """Compute Chi-Square feature importance scores and return top-N ranked features."""
    # Ensure non-negative inputs for Chi2
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)

    # Shift minimum to 0 if negative values exist
    min_val = np.min(X_dense)
    if min_val < 0:
        X_dense = X_dense - min_val

    scores, pvalues = chi2(X_dense, y)

    # Replace NaNs/Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)

    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[: min(top_n, len(feature_names))]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        clean_name = feat_name.replace("tfidf_", "")
        results.append(
            {
                "rank": rank,
                "feature": clean_name,
                "raw_feature_name": feat_name,
                "score": round(float(scores[idx]), 4),
                "p_value": float(pvalues[idx]),
                "method": "chi_square",
                "category": "Behavioral"
                if clean_name
                in [
                    "user_avg_rating",
                    "rating_deviation",
                    "reviews_per_day",
                    "time_gap_days",
                    "verified_flag",
                    "account_age_days",
                ]
                else (
                    "Linguistic"
                    if clean_name
                    in [
                        "review_length",
                        "word_count",
                        "punctuation_count",
                        "capital_ratio",
                        "exclamation_count",
                        "first_person_ratio",
                        "sentiment_compound",
                        "sentiment_pos",
                        "sentiment_neg",
                    ]
                    else "TF-IDF Lexical"
                ),
            }
        )

    return results


def select_features_mutual_info(
    X,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    random_state: int = 42,
) -> List[Dict]:
    """Compute Information Gain (Mutual Information) scores and return top-N features."""
    y_disc = np.asarray(y, dtype=int)
    scores = mutual_info_classif(X, y_disc, random_state=random_state)
    scores = np.nan_to_num(scores, nan=0.0)

    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[: min(top_n, len(feature_names))]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        clean_name = feat_name.replace("tfidf_", "")
        results.append(
            {
                "rank": rank,
                "feature": clean_name,
                "raw_feature_name": feat_name,
                "score": round(float(scores[idx]), 4),
                "method": "mutual_info",
                "category": "Behavioral"
                if clean_name
                in [
                    "user_avg_rating",
                    "rating_deviation",
                    "reviews_per_day",
                    "time_gap_days",
                    "verified_flag",
                    "account_age_days",
                ]
                else (
                    "Linguistic"
                    if clean_name
                    in [
                        "review_length",
                        "word_count",
                        "punctuation_count",
                        "capital_ratio",
                        "exclamation_count",
                        "first_person_ratio",
                        "sentiment_compound",
                        "sentiment_pos",
                        "sentiment_neg",
                    ]
                    else "TF-IDF Lexical"
                ),
            }
        )

    return results
