"""Feature engineering pipeline for ReviewGuard.

Extracts:
1. Text Linguistic Features:
   - TF-IDF (unigram + bigram)
   - Review character length
   - Word count
   - Punctuation count
   - Capitalization ratio (uppercase / total chars)
   - Exclamation mark count
   - Sentiment analysis (VADER compound, positive, negative scores)
   - First-person pronoun ratio (I, me, my, we, us, our...)

2. Behavioral Features (Metadata dataset only):
   - User average rating
   - Rating deviation from product mean
   - Reviewer posting frequency (reviews per day / burst posting)
   - Time gap between user's consecutive reviews
   - Verified purchase binary flag
   - Account age in days
"""

import string
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .load_data import has_behavioral_metadata
from .preprocess import clean_text_basic, preprocess_text


# Initialize VADER sentiment analyzer
try:
    VADER = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon", quiet=True)
    VADER = SentimentIntensityAnalyzer()

FIRST_PERSON_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}
PUNCT_SET = set(string.punctuation)


def extract_linguistic_features_single(text: str) -> Dict[str, float]:
    """Extract statistical and linguistic scalar features from a single raw review string."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    char_len = len(text)
    words = text.split()
    word_count = len(words)

    # Punctuation and exclamation count
    punct_count = sum(1 for c in text if c in PUNCT_SET)
    exclamation_count = text.count("!")

    # Capital ratio
    upper_count = sum(1 for c in text if c.isupper())
    capital_ratio = (upper_count / char_len) if char_len > 0 else 0.0

    # First-person pronoun ratio
    words_lower = [w.lower().strip(string.punctuation) for w in words]
    fp_count = sum(1 for w in words_lower if w in FIRST_PERSON_PRONOUNS)
    first_person_ratio = (fp_count / word_count) if word_count > 0 else 0.0

    # Sentiment with VADER
    try:
        sent = VADER.polarity_scores(text)
        sentiment_compound = float(sent.get("compound", 0.0))
        sentiment_pos = float(sent.get("pos", 0.0))
        sentiment_neg = float(sent.get("neg", 0.0))
    except Exception:
        sentiment_compound, sentiment_pos, sentiment_neg = 0.0, 0.0, 0.0

    return {
        "review_length": float(char_len),
        "word_count": float(word_count),
        "punctuation_count": float(punct_count),
        "capital_ratio": float(capital_ratio),
        "exclamation_count": float(exclamation_count),
        "first_person_ratio": float(first_person_ratio),
        "sentiment_compound": float(sentiment_compound),
        "sentiment_pos": float(sentiment_pos),
        "sentiment_neg": float(sentiment_neg),
    }


def extract_linguistic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Batch extract linguistic scalar features for a DataFrame."""
    records = []
    for text in df["text"]:
        records.append(extract_linguistic_features_single(text))
    return pd.DataFrame(records, index=df.index)


def compute_behavioral_features(
    df: pd.DataFrame,
    product_mean_ratings: Optional[Dict[str, float]] = None,
    user_avg_ratings: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Compute behavioral features from metadata columns.

    Returns:
    - behavioral_df: DataFrame with computed behavioral columns
    - product_mean_ratings: reference dictionary of product ID -> mean rating
    - user_avg_ratings: reference dictionary of user ID -> average rating
    """
    if not has_behavioral_metadata(df):
        raise ValueError("DataFrame lacks behavioral metadata columns.")

    data = df.copy()

    # Ensure rating is numeric
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(3.0)

    # 1. Product mean rating and rating deviation
    if "product_id" in data.columns:
        if product_mean_ratings is None:
            product_mean_ratings = (
                data.groupby("product_id")["rating"].mean().to_dict()
            )
        data["product_avg_rating"] = (
            data["product_id"].map(product_mean_ratings).fillna(data["rating"].mean())
        )
        data["rating_deviation"] = (
            data["rating"] - data["product_avg_rating"]
        ).abs()
    else:
        product_mean_ratings = {}
        data["rating_deviation"] = 0.0

    # 2. User average rating
    if user_avg_ratings is None:
        user_avg_ratings = data.groupby("user_id")["rating"].mean().to_dict()
    data["user_avg_rating"] = (
        data["user_id"].map(user_avg_ratings).fillna(data["rating"])
    )

    # 3. Reviews per day and time gap
    if "review_date" in data.columns:
        data["parsed_date"] = pd.to_datetime(data["review_date"], errors="coerce")
        # Sort by user and date to compute inter-review time gap
        sorted_df = data.sort_values(["user_id", "parsed_date"])
        time_diffs = (
            sorted_df.groupby("user_id")["parsed_date"].diff().dt.total_seconds()
            / 86400.0
        )
        data["time_gap_days"] = (
            time_diffs.reindex(data.index).fillna(30.0).clip(lower=0.0)
        )

        user_counts = data.groupby("user_id")["rating"].transform("count")
        user_span = data.groupby("user_id")["parsed_date"].transform(
            lambda s: max((s.max() - s.min()).total_seconds() / 86400.0, 1.0)
        )
        data["reviews_per_day"] = (user_counts / user_span).clip(
            lower=0.01, upper=50.0
        )
    else:
        data["time_gap_days"] = 30.0
        data["reviews_per_day"] = 0.1

    # 4. Verified purchase flag
    if "verified_purchase" in data.columns:
        data["verified_flag"] = (
            data["verified_purchase"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "verified"])
            .astype(float)
        )
    else:
        data["verified_flag"] = 1.0

    # 5. Account age
    if "account_age_days" in data.columns:
        data["account_age_days"] = (
            pd.to_numeric(data["account_age_days"], errors="coerce")
            .fillna(365.0)
            .clip(lower=1.0)
        )
    else:
        data["account_age_days"] = 365.0

    behavioral_cols = [
        "user_avg_rating",
        "rating_deviation",
        "reviews_per_day",
        "time_gap_days",
        "verified_flag",
        "account_age_days",
    ]
    return (
        data[behavioral_cols].astype(float),
        product_mean_ratings,
        user_avg_ratings,
    )


class FeaturePipeline:
    """End-to-end feature extraction and transformation pipeline."""

    def __init__(
        self,
        max_tfidf_features: int = 2000,
        include_behavioral: bool = False,
        random_state: int = 42,
    ):
        self.max_tfidf_features = max_tfidf_features
        self.include_behavioral = include_behavioral
        self.random_state = random_state

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_tfidf_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )
        self.scaler = StandardScaler(with_mean=False)
        self.feature_names: List[str] = []
        self.product_mean_ratings: Dict[str, float] = {}
        self.user_avg_ratings: Dict[str, float] = {}
        self.linguistic_col_names: List[str] = [
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
        self.behavioral_col_names: List[str] = [
            "user_avg_rating",
            "rating_deviation",
            "reviews_per_day",
            "time_gap_days",
            "verified_flag",
            "account_age_days",
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit TF-IDF and Scaler on dataset."""
        cleaned_text = df["text"].map(preprocess_text)
        tfidf_mat = self.tfidf.fit_transform(cleaned_text)

        # Linguistic features
        ling_df = extract_linguistic_dataframe(df)

        if self.include_behavioral and has_behavioral_metadata(df):
            behav_df, prod_dict, user_dict = compute_behavioral_features(df)
            self.product_mean_ratings = prod_dict
            self.user_avg_ratings = user_dict
            dense_feats = pd.concat([ling_df, behav_df], axis=1).values
            active_dense_names = (
                self.linguistic_col_names + self.behavioral_col_names
            )
        else:
            dense_feats = ling_df.values
            active_dense_names = self.linguistic_col_names

        self.scaler.fit(dense_feats)
        tfidf_names = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        self.feature_names = tfidf_names + active_dense_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        """Transform dataset to combined feature matrix."""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline is not fitted yet.")

        cleaned_text = df["text"].map(preprocess_text)
        tfidf_mat = self.tfidf.transform(cleaned_text)
        ling_df = extract_linguistic_dataframe(df)

        if self.include_behavioral and has_behavioral_metadata(df):
            behav_df, _, _ = compute_behavioral_features(
                df,
                product_mean_ratings=self.product_mean_ratings,
                user_avg_ratings=self.user_avg_ratings,
            )
            dense_feats = pd.concat([ling_df, behav_df], axis=1).values
        else:
            dense_feats = ling_df.values

        dense_scaled = self.scaler.transform(dense_feats)
        dense_sparse = csr_matrix(dense_scaled)
        combined = hstack([tfidf_mat, dense_sparse]).tocsr()
        return combined

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def transform_single(
        self,
        text: str,
        rating: Optional[float] = 5.0,
        account_age_days: Optional[float] = 365.0,
        verified_purchase: Optional[bool] = True,
    ):
        """Transform a single review for live inference."""
        cleaned_text = preprocess_text(text)
        tfidf_vec = self.tfidf.transform([cleaned_text])
        ling_dict = extract_linguistic_features_single(text)
        ling_vals = [ling_dict[col] for col in self.linguistic_col_names]

        if self.include_behavioral:
            # Approximate behavioral metrics for single live test review
            user_avg_rating = float(rating if rating is not None else 5.0)
            rating_dev = 0.0
            reviews_per_day = 0.2
            time_gap_days = 15.0
            verified_flag = 1.0 if verified_purchase else 0.0
            acc_age = (
                float(account_age_days) if account_age_days is not None else 365.0
            )

            behav_vals = [
                user_avg_rating,
                rating_dev,
                reviews_per_day,
                time_gap_days,
                verified_flag,
                acc_age,
            ]
            dense_arr = np.array([ling_vals + behav_vals], dtype=float)
        else:
            dense_arr = np.array([ling_vals], dtype=float)

        dense_scaled = self.scaler.transform(dense_arr)
        combined = hstack([tfidf_vec, csr_matrix(dense_scaled)]).tocsr()
        return combined, ling_dict

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeaturePipeline":
        return joblib.load(filepath)
