"""Reviewer clustering API routes."""

import json
from pathlib import Path
from fastapi import APIRouter, Query
from reviewguard.src.clustering import run_reviewer_clustering
from reviewguard.src.predict import get_models_directory

router = APIRouter()


@router.get("/clustering")
def get_clustering_results(
    dataset: str = Query("sample", description="Dataset identifier: 'sample' or 'kaggle'"),
    n_clusters: int = Query(4, ge=2, le=8),
    linkage: str = Query("ward", description="'ward', 'complete', 'average', 'single'"),
):
    """Retrieve K-Means elbow data, 2D PCA scatter points, dendrogram, and cluster summaries."""
    dataset_val = str(dataset) if isinstance(dataset, str) else "sample"
    k_val = int(n_clusters) if isinstance(n_clusters, int) else 4
    link_val = str(linkage) if isinstance(linkage, str) else "ward"

    models_dir = get_models_directory()
    cache_file = models_dir / "clustering_results.json"

    # If default sample with 4 clusters and ward linkage, serve cached if available
    if dataset_val == "sample" and k_val == 4 and link_val == "ward" and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Execute on demand
    return run_reviewer_clustering(
        dataset_name=dataset_val,
        n_clusters=k_val,
        linkage_method=link_val,
    )
