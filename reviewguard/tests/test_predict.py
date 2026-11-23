"""Pytest test suite for ReviewGuard live predictor."""

import pytest
from reviewguard.src.predict import ReviewGuardPredictor


def test_predict_suspicious_review():
    predictor = ReviewGuardPredictor()
    text = "MUST BUY THIS RIGHT NOW!! BEST PRODUCT EVER CREATED IN THE ENTIRE WORLD AMAZING MIRACLE!!!"
    res = predictor.predict(text, rating=5.0, account_age_days=5, verified_purchase=False)

    assert "verdict" in res
    assert "fake_probability" in res
    assert "per_model_predictions" in res
    assert "suspicious_highlights" in res
    assert res["fake_probability"] > 0.5
    assert len(res["suspicious_highlights"]) > 0


def test_predict_genuine_review():
    predictor = ReviewGuardPredictor()
    text = "The kettle has worked well over the last six months. Moderate noise level and boil time is normal."
    res = predictor.predict(text, rating=4.0, account_age_days=300, verified_purchase=True)

    assert "verdict" in res
    assert res["verdict"] in ("Genuine", "Suspicious")
    assert res["fake_probability"] < 0.6
