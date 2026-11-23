"""Ensemble learning API routes."""

import json
from pathlib import Path
from fastapi import APIRouter
from reviewguard.src.predict import get_models_directory

router = APIRouter()


@router.get("/ensemble")
def get_ensemble_metrics():
    """Retrieve Bagging, Boosting, Soft Voting performance and feature importances."""
    models_dir = get_models_directory()
    ens_file = models_dir / "ensemble_metrics.json"

    if not ens_file.exists():
        return {
            "is_trained": False,
            "message": "Ensemble models have not been trained yet.",
            "ensembles": {},
            "feature_importances": [],
            "comparison": [],
        }

    try:
        with open(ens_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["is_trained"] = True
        return data
    except Exception as e:
        return {"error": f"Failed to load ensemble metrics: {str(e)}"}
