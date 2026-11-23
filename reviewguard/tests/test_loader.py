"""Pytest test suite for ReviewGuard data loader."""

import pytest
from reviewguard.src.load_data import (
    find_dataset_path,
    has_behavioral_metadata,
    list_available_datasets,
    load_dataset,
    normalize_labels,
)
import pandas as pd


def test_find_dataset_path():
    p_kaggle = find_dataset_path("kaggle")
    assert p_kaggle is not None, "Kaggle dataset path should be auto-detected"
    assert p_kaggle.exists()

    p_sample = find_dataset_path("sample")
    assert p_sample is not None, "Sample dataset path should be auto-detected"
    assert p_sample.exists()


def test_normalize_labels():
    s = pd.Series(["CG", "OR", "fake", "genuine", 1, 0, "Fake"])
    norm = normalize_labels(s)
    assert norm.tolist() == [1, 0, 1, 0, 1, 0, 1]


def test_load_sample_dataset():
    df, meta = load_dataset("sample", sample_size=50)
    assert len(df) == 50
    assert "text" in df.columns
    assert "label" in df.columns
    assert meta["has_metadata"] is True
    assert meta["is_synthetic"] is True
    assert set(df["label"].unique()).issubset({0, 1})


def test_load_kaggle_dataset():
    df, meta = load_dataset("kaggle", sample_size=60)
    assert len(df) == 60
    assert "text" in df.columns
    assert "label" in df.columns
    assert meta["has_metadata"] is False
    assert meta["is_synthetic"] is False


def test_list_available_datasets():
    datasets = list_available_datasets()
    assert len(datasets) >= 2
    ids = [d["id"] for d in datasets]
    assert "kaggle" in ids
    assert "sample" in ids
