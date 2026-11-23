"""Model evaluation and metrics inspection routes."""

import json
from pathlib import Path
from fastapi import APIRouter
from reviewguard.src.predict import get_models_directory

router = APIRouter()


@router.get("/evaluation")
def get_evaluation_metrics():
    """Retrieve detailed evaluation metrics, confusion matrices, ROC/PR curves, and CV scores."""
    models_dir = get_models_directory()
    metrics_file = models_dir / "metrics.json"
    summary_file = models_dir / "summary.json"

    if not metrics_file.exists():
        return {
            "is_trained": False,
            "message": "Models have not been trained yet. Please initiate training via /api/train.",
            "models": {},
        }

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load metrics: {str(e)}", "models": {}}

    summary_info = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_info = json.load(f)
        except Exception:
            pass

    # Build model comparison leaderboards
    ranked_models = []
    for k, v in metrics_data.items():
        ranked_models.append({
            "model_key": k,
            "name": v.get("display_name", k),
            "accuracy": v.get("accuracy", 0.0),
            "precision": v.get("precision", 0.0),
            "recall": v.get("recall", 0.0),
            "f1": v.get("f1", 0.0),
            "auc": v.get("auc", 0.0),
            "cv_mean": v.get("cross_validation", {}).get("cv_mean", 0.0),
            "cv_std": v.get("cross_validation", {}).get("cv_std", 0.0),
        })

    # Sort descending by F1
    ranked_models.sort(key=lambda m: m["f1"], reverse=True)

    return {
        "is_trained": True,
        "dataset_name": summary_info.get("dataset_name", "Kaggle Benchmark"),
        "best_model": summary_info.get("best_model"),
        "best_f1": summary_info.get("best_f1"),
        "ranked_models": ranked_models,
        "models": metrics_data,
    }
