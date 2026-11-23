"""Pytest test suite for Clustering, Association Rules, and HMM modules."""

import pytest
from reviewguard.src.association import mine_association_rules
from reviewguard.src.clustering import run_reviewer_clustering
from reviewguard.src.hmm_model import (
    ClassicalHMM,
    discretize_observation,
    train_and_save_hmm,
    get_user_hmm_analysis,
)


def test_clustering_sample():
    res = run_reviewer_clustering(dataset_name="sample", n_clusters=3)
    assert res["requires_metadata"] is False
    assert "elbow_curve" in res
    assert "pca_points" in res
    assert len(res["cluster_summaries"]) == 3


def test_clustering_kaggle_requires_metadata():
    res = run_reviewer_clustering(dataset_name="kaggle")
    assert res["requires_metadata"] is True
    assert "requires user metadata" in res["message"]


def test_association_rules():
    res = mine_association_rules(dataset_name="sample", min_support=0.05, min_confidence=0.3)
    assert res["requires_metadata"] is False
    assert "rules" in res
    assert "algorithm" in res


def test_hmm_viterbi_and_forward():
    hmm = ClassicalHMM()
    # Test observation sequence: extreme rating burst [3, 3, 3]
    obs = [3, 3, 3]
    log_lik, alpha = hmm.forward(obs)
    path, score = hmm.viterbi(obs)

    assert len(path) == 3
    # Burst spam sequence should decode into state 1 (spammer)
    assert 1 in path
    assert isinstance(log_lik, float)


def test_user_hmm_analysis():
    summary = train_and_save_hmm(dataset_name="sample")
    assert summary["requires_metadata"] is False
    first_user = summary["user_summaries"][0]["user_id"]

    analysis = get_user_hmm_analysis(first_user, dataset_name="sample")
    assert analysis["user_id"] == first_user
    assert "timeline" in analysis
    assert len(analysis["timeline"]) > 0
