"""Model training API routes with non-blocking background task execution."""

import threading
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks
from reviewguard.app.schemas import TrainRequest
from reviewguard.src.ensemble import train_ensemble_models
from reviewguard.src.train import train_and_evaluate_models

router = APIRouter()


class TrainingManager:
    """Manages asynchronous background training jobs and logs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.status = "idle"
        self.progress = 0
        self.message = "System idle. Ready for model training."
        self.logs: List[Dict[str, str]] = []
        self.result: Optional[Dict] = None

    def log(self, text: str, progress: Optional[int] = None):
        with self.lock:
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": text,
            }
            self.logs.append(entry)
            self.message = text
            if progress is not None:
                self.progress = progress

    def start_training(self, req: TrainRequest):
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.status = "running"
            self.progress = 5
            self.logs = []
            self.result = None

        def _worker():
            try:
                self.log(f"Starting training run on '{req.dataset_name}' dataset...", 10)
                self.log(
                    f"Configuration: Models={req.models}, CV Folds={req.cv_folds}, "
                    f"Imbalance={req.imbalance_method}, Test Size={req.test_size}",
                    15,
                )

                # 1. Classical Models Training
                summary = train_and_evaluate_models(
                    dataset_name=req.dataset_name or "kaggle",
                    model_keys=req.models,
                    test_size=req.test_size or 0.2,
                    cv_folds=req.cv_folds or 5,
                    imbalance_method=req.imbalance_method or "none",
                    max_samples=req.max_samples or 6000,
                    max_tfidf_features=req.max_tfidf_features or 2000,
                    progress_callback=lambda msg, pct: self.log(msg, pct),
                )

                # 2. Ensemble Models Training
                self.log("Training Ensemble Models (Bagging, Boosting, Soft Voting)...", 85)
                ens_summary = train_ensemble_models(
                    dataset_name=req.dataset_name or "kaggle",
                    max_samples=min(req.max_samples or 6000, 6000),
                    test_size=req.test_size or 0.2,
                )
                self.log("Ensemble models trained and serialized.", 95)

                with self.lock:
                    self.is_running = False
                    self.status = "completed"
                    self.progress = 100
                    self.result = {
                        "single_models": summary,
                        "ensembles": ens_summary,
                    }
                    self.message = f"Training completed successfully! Best model: {summary.get('best_model')}"
                self.log("Artifacts persisted to disk.", 100)

            except Exception as e:
                with self.lock:
                    self.is_running = False
                    self.status = "error"
                    self.message = f"Training failed: {str(e)}"
                self.log(f"ERROR: {str(e)}")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return True


training_manager = TrainingManager()


@router.post("/train")
def trigger_training(req: TrainRequest):
    """Trigger background model training job."""
    started = training_manager.start_training(req)
    if not started:
        return {
            "status": "running",
            "message": "A training run is already in progress. Please wait for it to complete.",
        }
    return {
        "status": "started",
        "message": "Model training pipeline triggered in background.",
    }


@router.get("/train/status")
def get_training_status():
    """Poll real-time training progress and terminal logs."""
    return {
        "is_running": training_manager.is_running,
        "status": training_manager.status,
        "progress": training_manager.progress,
        "message": training_manager.message,
        "logs": training_manager.logs[-30:],  # return last 30 log lines
        "has_result": training_manager.result is not None,
    }
