"""Hidden Markov Model (HMM) sequence analysis API routes."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from reviewguard.src.hmm_model import (
    get_user_hmm_analysis,
    train_and_save_hmm,
)
from reviewguard.src.predict import get_models_directory

router = APIRouter()


@router.get("/hmm/summary")
def get_hmm_summary(
    dataset: str = Query("sample", description="Dataset identifier: 'sample' or 'kaggle'")
):
    """Retrieve global HMM transition matrix, emission matrix, and reviewer summaries."""
    dataset_val = str(dataset) if isinstance(dataset, str) else "sample"
    models_dir = get_models_directory()
    cache_file = models_dir / "hmm_model.json"

    if dataset_val == "sample" and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return train_and_save_hmm(dataset_name=dataset_val)


@router.get("/hmm/{user_id}")
def get_user_sequence(
    user_id: str,
    dataset: str = Query("sample", description="Dataset identifier: 'sample' or 'kaggle'"),
):
    """Retrieve Viterbi hidden state trajectory and log-likelihood for a specific user sequence."""
    dataset_val = str(dataset) if isinstance(dataset, str) else "sample"
    result = get_user_hmm_analysis(user_id=str(user_id), dataset_name=dataset_val)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
