"""Pytest test suite for preprocessing module."""

import pytest
from reviewguard.src.preprocess import (
    clean_text_basic,
    preprocess_text,
    tokenize_and_lemmatize,
)


def test_clean_text_basic():
    raw = "Check out this link: https://example.com/item and <br /> HTML tags."
    cleaned = clean_text_basic(raw)
    assert "https://" not in cleaned
    assert "<br />" not in cleaned
    assert "HTML tags" in cleaned


def test_tokenize_and_lemmatize():
    text = "The quick brown foxes were jumping over the lazy dogs!"
    tokens = tokenize_and_lemmatize(text)
    # Lemmatizer changes foxes -> fox
    assert "fox" in tokens
    # Stopwords like 'the', 'were', 'over' are removed
    assert "the" not in tokens
    assert "were" not in tokens


def test_preprocess_text():
    text = "Amazing purchase! LOVE this coffee maker so much!"
    res = preprocess_text(text)
    assert isinstance(res, str)
    assert len(res) > 0
    assert "!" not in res
