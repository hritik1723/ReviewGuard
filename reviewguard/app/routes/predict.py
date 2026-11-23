"""Live inference and history API routes."""

from fastapi import APIRouter, HTTPException
from reviewguard.app.schemas import PredictRequest
from reviewguard.src.predict import ReviewGuardPredictor, get_prediction_history

router = APIRouter()

# Instantiate global predictor instance
_predictor = None


def get_predictor() -> ReviewGuardPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ReviewGuardPredictor()
    return _predictor


@router.post("/predict")
def predict_review(req: PredictRequest):
    """Run real-time multi-model classification and linguistic explainability on a single review."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    predictor = get_predictor()
    try:
        result = predictor.predict(
            text=req.text,
            rating=req.rating,
            account_age_days=req.account_age_days,
            verified_purchase=req.verified_purchase,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_history(limit: int = 20):
    """Retrieve recent inference audit logs."""
    history = get_prediction_history(limit=limit)
    return {"history": history}
