"""Pytest test suite for ReviewGuard REST API handlers."""

import pytest
from reviewguard.app.main import app, healthcheck
from reviewguard.app.routes import (
    association,
    clustering,
    datasets,
    ensemble,
    evaluation,
    features,
    hmm,
    overview,
    predict,
)
from reviewguard.app.schemas import ImbalanceRequest, PredictRequest


def test_api_health():
    res = healthcheck()
    assert res["status"] == "healthy"
    assert res["service"] == "ReviewGuard"


def test_api_overview():
    data = overview.get_overview_telemetry()
    assert "metrics" in data
    assert data["metrics"]["best_f1_score"] > 0
    assert "radar_comparison" in data
    assert "recent_detections" in data


def test_api_datasets():
    data = datasets.get_datasets()
    assert "datasets" in data
    assert len(data["datasets"]) >= 2
    ids = [d["id"] for d in data["datasets"]]
    assert "kaggle" in ids
    assert "sample" in ids


def test_api_dataset_explore():
    data = datasets.explore_dataset(dataset="kaggle", page=1, page_size=5)
    assert "records" in data
    assert len(data["records"]) == 5
    assert "class_balance" in data
    assert data["class_balance"]["fake_cg"] > 0


def test_api_features_importance():
    data = features.get_features_importance(dataset="sample", top_n=5)
    assert "chi_square_features" in data
    assert len(data["chi_square_features"]) == 5
    assert "mutual_information_features" in data


def test_api_imbalance():
    req = ImbalanceRequest(dataset_name="sample", method="smote")
    data = features.test_imbalance_handling(req)
    assert "before" in data
    assert "after" in data
    assert data["after"]["total"] >= data["before"]["total"]


def test_api_evaluation():
    data = evaluation.get_evaluation_metrics()
    assert "ranked_models" in data
    assert len(data["ranked_models"]) > 0


def test_api_ensemble():
    data = ensemble.get_ensemble_metrics()
    assert "ensembles" in data
    assert "random_forest" in data["ensembles"]


def test_api_clustering():
    data = clustering.get_clustering_results(dataset="sample", n_clusters=3, linkage="ward")
    assert "cluster_summaries" in data
    assert len(data["cluster_summaries"]) == 3
    assert "elbow_curve" in data


def test_api_association():
    data = association.get_association_rules(dataset="sample", algorithm="fpgrowth")
    assert "rules" in data
    assert len(data["rules"]) > 0


def test_api_hmm_summary():
    data = hmm.get_hmm_summary(dataset="sample")
    assert "transition_matrix" in data
    assert "user_summaries" in data
    assert len(data["user_summaries"]) > 0


def test_api_predict():
    payload = PredictRequest(
        text="This is an absolute miraculous product, best quality ever in the world! Must buy now!",
        rating=5.0,
        account_age_days=10,
        verified_purchase=False,
    )
    data = predict.predict_review(payload)
    assert "verdict" in data
    assert data["verdict"] in ("Fake", "Suspicious")
    assert data["fake_probability"] > 0.5
    assert "per_model_predictions" in data


def test_api_history():
    data = predict.get_history(limit=10)
    assert "history" in data
    assert isinstance(data["history"], list)
