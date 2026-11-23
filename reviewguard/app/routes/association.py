"""Association rules and pattern mining API routes."""

import json
from pathlib import Path
from fastapi import APIRouter, Query
from reviewguard.src.association import mine_association_rules
from reviewguard.src.predict import get_models_directory

router = APIRouter()


@router.get("/association")
def get_association_rules(
    dataset: str = Query("sample", description="Dataset identifier: 'sample' or 'kaggle'"),
    algorithm: str = Query("fpgrowth", description="'fpgrowth' or 'apriori'"),
    min_support: float = Query(0.05, ge=0.01, le=0.5),
    min_confidence: float = Query(0.4, ge=0.1, le=1.0),
):
    """Retrieve frequent behavioral patterns and association rules."""
    dataset_val = str(dataset) if isinstance(dataset, str) else "sample"
    algo_val = str(algorithm) if isinstance(algorithm, str) else "fpgrowth"
    supp_val = float(min_support) if isinstance(min_support, (int, float)) else 0.05
    conf_val = float(min_confidence) if isinstance(min_confidence, (int, float)) else 0.4

    models_dir = get_models_directory()
    cache_file = models_dir / "association_rules.json"

    # Return cached if matching default parameters
    if (
        dataset_val == "sample"
        and algo_val == "fpgrowth"
        and supp_val == 0.05
        and conf_val == 0.4
        and cache_file.exists()
    ):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return mine_association_rules(
        dataset_name=dataset_val,
        algorithm=algo_val,
        min_support=supp_val,
        min_confidence=conf_val,
    )
