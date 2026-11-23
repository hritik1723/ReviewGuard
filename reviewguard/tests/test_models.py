"""Pytest test suite for classical ML models and evaluation."""

import pytest
import numpy as np
from reviewguard.src.evaluate import evaluate_predictions
from reviewguard.src.train import train_and_evaluate_models


def test_evaluate_predictions():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "auc" in metrics
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 3


def test_train_and_evaluate_models(tmp_path):
    # Run rapid training test with 80 samples into temporary directory
    res = train_and_evaluate_models(
        dataset_name="sample",
        model_keys=["naive_bayes", "decision_tree"],
        max_samples=80,
        cv_folds=3,
        test_size=0.25,
        output_dir=tmp_path,
    )
    assert "best_model" in res
    assert "models" in res
    assert "naive_bayes" in res["models"]
    assert "decision_tree" in res["models"]
    assert res["models"]["naive_bayes"]["f1"] >= 0.0
