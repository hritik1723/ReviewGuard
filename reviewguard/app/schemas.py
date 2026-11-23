"""Pydantic schemas for ReviewGuard REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., description="Review text to classify")
    rating: Optional[float] = Field(5.0, description="Star rating assigned (1.0 to 5.0)")
    account_age_days: Optional[float] = Field(365.0, description="Reviewer account age in days")
    verified_purchase: Optional[bool] = Field(True, description="Whether purchase is verified")


class TrainRequest(BaseModel):
    dataset_name: Optional[str] = Field("kaggle", description="Dataset identifier: 'kaggle' or 'sample'")
    models: Optional[List[str]] = Field(
        default=["naive_bayes", "logistic_regression", "decision_tree", "svm", "knn"],
        description="Classical model keys to train"
    )
    test_size: Optional[float] = Field(0.2, description="Test split ratio (0.1 to 0.4)")
    cv_folds: Optional[int] = Field(5, description="Cross validation folds (5 or 10)")
    imbalance_method: Optional[str] = Field("none", description="'none', 'smote', 'oversample', 'undersample', 'class_weight'")
    max_samples: Optional[int] = Field(6000, description="Max training rows for laptop speed")
    max_tfidf_features: Optional[int] = Field(2000, description="TF-IDF vocabulary dimension")


class ImbalanceRequest(BaseModel):
    dataset_name: Optional[str] = "kaggle"
    method: Optional[str] = "smote"


class ClusteringRequest(BaseModel):
    dataset_name: Optional[str] = "sample"
    n_clusters: Optional[int] = 4
    linkage_method: Optional[str] = "ward"


class AssociationRequest(BaseModel):
    dataset_name: Optional[str] = "sample"
    algorithm: Optional[str] = "fpgrowth"
    min_support: Optional[float] = 0.05
    min_confidence: Optional[float] = 0.4
