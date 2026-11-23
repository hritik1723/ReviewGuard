"""Feature engineering, feature selection, and class imbalance routes."""

from typing import Optional
from fastapi import APIRouter, Query
from reviewguard.app.schemas import ImbalanceRequest
from reviewguard.src.feature_selection import (
    select_features_chi2,
    select_features_mutual_info,
)
from reviewguard.src.features import FeaturePipeline
from reviewguard.src.imbalance import handle_imbalance
from reviewguard.src.load_data import load_dataset

router = APIRouter()


@router.get("/features/importance")
def get_features_importance(
    dataset: str = Query("kaggle", description="Dataset identifier: 'kaggle' or 'sample'"),
    top_n: int = Query(20, ge=5, le=50),
):
    """Return top feature scores via Chi-Square and Mutual Information feature selection."""
    dataset_val = str(dataset) if isinstance(dataset, str) else "kaggle"
    top_n_val = int(top_n) if isinstance(top_n, int) else 20

    df, meta = load_dataset(dataset_val, sample_size=3000)
    has_meta = meta["has_metadata"]

    pipe = FeaturePipeline(
        max_tfidf_features=1000,
        include_behavioral=has_meta,
        random_state=42,
    )
    X = pipe.fit_transform(df)
    y = df["label"].astype(int).values

    chi2_ranks = select_features_chi2(X, y, pipe.feature_names, top_n=top_n_val)
    mi_ranks = select_features_mutual_info(X, y, pipe.feature_names, top_n=top_n_val)

    # Signal type breakdown
    categories = {"TF-IDF Lexical": 0, "Linguistic": 0, "Behavioral": 0}
    for item in chi2_ranks:
        cat = item.get("category", "TF-IDF Lexical")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "dataset_name": dataset,
        "has_metadata": has_meta,
        "total_features": len(pipe.feature_names),
        "chi_square_features": chi2_ranks,
        "mutual_information_features": mi_ranks,
        "category_distribution": categories,
    }


@router.post("/imbalance")
def test_imbalance_handling(req: ImbalanceRequest):
    """Simulate resampling on dataset and return before/after class distributions."""
    df, meta = load_dataset(req.dataset_name or "sample", sample_size=2000)
    y = df["label"].values

    # Build dummy or real features for resampling demo
    pipe = FeaturePipeline(max_tfidf_features=100, include_behavioral=meta["has_metadata"])
    X = pipe.fit_transform(df)

    _, _, stats = handle_imbalance(X, y, method=req.method or "smote")
    return stats
