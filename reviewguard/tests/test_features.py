"""Pytest test suite for feature extraction pipeline."""

import pytest
from reviewguard.src.features import (
    FeaturePipeline,
    extract_linguistic_features_single,
    compute_behavioral_features,
)
from reviewguard.src.load_data import load_dataset


def test_extract_linguistic_features_single():
    text = "I LOVE THIS PRODUCT SO MUCH!!!! BEST EVER IN MY LIFE."
    ling = extract_linguistic_features_single(text)
    assert ling["exclamation_count"] == 4
    assert ling["capital_ratio"] > 0.5
    assert ling["first_person_ratio"] > 0
    assert "sentiment_compound" in ling
    assert ling["word_count"] > 5


def test_feature_pipeline_fit_transform():
    df, meta = load_dataset("sample", sample_size=40)
    pipeline = FeaturePipeline(max_tfidf_features=50, include_behavioral=True)
    X = pipeline.fit_transform(df)
    assert X.shape[0] == 40
    assert X.shape[1] > 50

    # Single transformation
    X_single, ling = pipeline.transform_single(
        "Great quality and prompt delivery. Highly recommended!",
        rating=5.0,
        account_age_days=100,
        verified_purchase=True,
    )
    assert X_single.shape[0] == 1
    assert X_single.shape[1] == X.shape[1]
