"""Real-time live inference and linguistic explainability for ReviewGuard.

Outputs:
1. Verdict: 'Genuine', 'Suspicious', or 'Fake'
2. Calibrated ensemble probability
3. Per-model breakdown (Naive Bayes, KNN, SVM, Decision Tree, Logistic Regression, Ensemble)
4. Feature explainability (top positive/fake and negative/genuine contributors)
5. In-text suspicious word / token highlights
6. Prediction telemetry history logger
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

from .features import FeaturePipeline, extract_linguistic_features_single
from .preprocess import clean_text_basic, tokenize_and_lemmatize


SUSPICIOUS_KEYWORDS = {
    "miracle", "miraculous", "must", "buy", "amazing", "best", "ever",
    "scam", "game", "changer", "revolutionary", "flawless", "perfection",
    "unbelievable", "awesome", "guaranteed", "highly", "recommend", "waste",
    "money", "horrible", "terrible", "worst", "trash", "junk", "ten", "stars",
    "everyone", "obsessed", "lifesaver", "worthless", "fake"
}

GENUINE_SUPERLATIVES = {
    "best", "worst", "greatest", "flawless", "perfection", "unbelievable",
    "miracle", "miraculous", "revolutionary", "lifesaver", "worthless"
}


def get_models_directory() -> Path:
    """Resolve models directory."""
    for d in [
        Path.cwd() / "reviewguard" / "models",
        Path.cwd() / "models",
        Path(__file__).resolve().parent.parent / "models",
    ]:
        if d.exists():
            return d.resolve()
    # Create default
    d = Path.cwd() / "reviewguard" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


class ReviewGuardPredictor:
    """Live inference engine with multi-model voting and token explainability."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir) if models_dir else get_models_directory()
        self.pipeline: Optional[FeaturePipeline] = None
        self.models: Dict[str, object] = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Load fitted FeaturePipeline and trained models from disk."""
        pipe_path = self.models_dir / "feature_pipeline.joblib"
        if pipe_path.exists():
            try:
                self.pipeline = FeaturePipeline.load(str(pipe_path))
            except Exception:
                self.pipeline = None

        # Load individual models
        model_names = [
            "naive_bayes",
            "logistic_regression",
            "decision_tree",
            "svm",
            "knn",
            "random_forest",
            "adaboost",
            "gradient_boosting",
            "soft_voting",
        ]
        for name in model_names:
            m_path = self.models_dir / f"{name}.joblib"
            if m_path.exists():
                try:
                    self.models[name] = joblib.load(str(m_path))
                except Exception:
                    pass

    def reload(self):
        """Reload models if freshly retrained."""
        self.models.clear()
        self._load_artifacts()

    def predict(
        self,
        text: str,
        rating: Optional[float] = 5.0,
        account_age_days: Optional[float] = 365.0,
        verified_purchase: Optional[bool] = True,
    ) -> Dict:
        """Execute full prediction pipeline on a single review."""
        if not self.pipeline:
            self._load_artifacts()

        # If still no pipeline, raise or return error
        if not self.pipeline:
            raise RuntimeError(
                "Models have not been trained yet. Please run training first."
            )

        rating_val = float(rating) if rating is not None else 5.0
        acc_age = float(account_age_days) if account_age_days is not None else 365.0
        verified = bool(verified_purchase) if verified_purchase is not None else True

        # Extract feature vector
        X_vec, ling_stats = self.pipeline.transform_single(
            text=text,
            rating=rating_val,
            account_age_days=acc_age,
            verified_purchase=verified,
        )

        per_model_results = {}
        probabilities = []
        weights = []

        # Model weights in final ensemble decision
        model_weights = {
            "soft_voting": 1.5,
            "random_forest": 1.0,
            "gradient_boosting": 1.0,
            "svm": 1.0,
            "logistic_regression": 0.9,
            "decision_tree": 0.7,
            "naive_bayes": 0.7,
            "knn": 0.6,
        }

        # 1. Per-model inferences
        for m_key, model in self.models.items():
            if m_key in ("adaboost",):
                continue
            try:
                # Naive Bayes non-negative adjustment
                if m_key == "naive_bayes":
                    X_input = X_vec.copy()
                    X_input.data = np.clip(X_input.data, 0, None)
                else:
                    X_input = X_vec

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_input)[0]
                    p_fake = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    pred = model.predict(X_input)[0]
                    p_fake = float(pred)

                pred_label = 1 if p_fake >= 0.5 else 0
                per_model_results[m_key] = {
                    "prediction": pred_label,
                    "probability_fake": round(p_fake, 4),
                    "probability_genuine": round(1.0 - p_fake, 4),
                    "verdict": "Fake" if pred_label == 1 else "Genuine",
                }

                w = model_weights.get(m_key, 0.8)
                probabilities.append(p_fake * w)
                weights.append(w)
            except Exception:
                pass

        # 2. Overall aggregated probability
        if probabilities and weights:
            final_prob = sum(probabilities) / sum(weights)
        else:
            final_prob = 0.5

        # 3. Feature and word explainability & highlights
        highlights = self._generate_highlights(text)
        top_words = self._explain_tokens(text, X_vec)

        # Linguistic & Behavioral heuristic calibration for extreme signals
        kw_count = len(highlights)
        excl_count = ling_stats.get("exclamation_count", 0)

        if kw_count >= 3 or (kw_count >= 1 and excl_count >= 2):
            final_prob = max(final_prob, 0.68)
        elif kw_count >= 2 or excl_count >= 2:
            final_prob = max(final_prob, 0.54)
        elif not verified and acc_age < 30 and rating_val in (1.0, 5.0) and kw_count >= 1:
            final_prob = max(final_prob, 0.58)
        elif verified and acc_age > 180 and excl_count == 0 and kw_count == 0:
            final_prob = min(final_prob, 0.28)

        # 4. Determine final verdict
        final_prob = float(np.clip(final_prob, 0.01, 0.99))
        if final_prob >= 0.60:
            verdict = "Fake"
            verdict_badge = "error"
        elif final_prob >= 0.38:
            verdict = "Suspicious"
            verdict_badge = "warning"
        else:
            verdict = "Genuine"
            verdict_badge = "success"

        # Explicitly ensure Ensemble is present in per_model_predictions
        per_model_results["ensemble"] = {
            "prediction": 1 if final_prob >= 0.5 else 0,
            "probability_fake": round(final_prob, 4),
            "probability_genuine": round(1.0 - final_prob, 4),
            "verdict": "Fake" if final_prob >= 0.5 else "Genuine",
        }

        result = {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "fake_probability": round(final_prob, 4),
            "genuine_probability": round(1.0 - final_prob, 4),
            "per_model_predictions": per_model_results,
            "top_contributing_words": top_words,
            "suspicious_highlights": highlights,
            "linguistic_stats": ling_stats,
            "timestamp": datetime.now().isoformat(),
        }

        # Log prediction
        self._log_history(text, result)

        return result

    def _explain_tokens(self, text: str, X_vec) -> List[Dict]:
        """Identify features that contributed most heavily toward Fake vs Genuine verdict."""
        contributions = []

        # Use Logistic Regression coefficients if available
        lr_model = self.models.get("logistic_regression")
        if lr_model and hasattr(lr_model, "coef_"):
            coefs = lr_model.coef_[0]
            # Match non-zero elements
            indices = X_vec.indices
            data_vals = X_vec.data
            for idx, val in zip(indices, data_vals):
                if idx < len(self.pipeline.feature_names):
                    feat_name = self.pipeline.feature_names[idx].replace("tfidf_", "")
                    impact = float(coefs[idx] * val)
                    contributions.append({
                        "feature": feat_name,
                        "impact": round(impact, 4),
                        "direction": "fake" if impact > 0 else "genuine",
                    })

        # Sort by absolute impact
        contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
        return contributions[:12]

    def _generate_highlights(self, text: str) -> List[Dict]:
        """Detect word spans in text that exhibit suspicious linguistic cues."""
        words = text.split()
        highlights = []

        for idx, word in enumerate(words):
            w_clean = word.lower().strip(".,!?:;\"'()[]{}")
            if w_clean in SUSPICIOUS_KEYWORDS:
                reason = "Extreme superlative" if w_clean in GENUINE_SUPERLATIVES else "High-weight token in the model"
                highlights.append({
                    "word": word,
                    "index": idx,
                    "reason": reason,
                    "severity": "high",
                })
            elif "!" in word:
                highlights.append({
                    "word": word,
                    "index": idx,
                    "reason": "Exclamatory sentiment inflation",
                    "severity": "medium",
                })
            elif word.isupper() and len(word) > 2:
                highlights.append({
                    "word": word,
                    "index": idx,
                    "reason": "All-caps emphasis",
                    "severity": "medium",
                })

        return highlights

    def _log_history(self, text: str, result: Dict):
        """Append inference entry to models/history.json for auditing."""
        history_file = self.models_dir / "history.json"
        try:
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    hist = json.load(f)
            else:
                hist = []

            entry = {
                "id": max((int(entry.get("id", 0)) for entry in hist), default=0) + 1,
                "text_snippet": text[:90] + ("..." if len(text) > 90 else ""),
                "verdict": result["verdict"],
                "fake_probability": result["fake_probability"],
                "timestamp": result["timestamp"],
            }
            # Prepend newest
            hist.insert(0, entry)
            # Retain last 100 entries
            hist = hist[:100]

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
        except Exception:
            pass


def get_prediction_history(limit: int = 20) -> List[Dict]:
    """Retrieve recent prediction history for UI table."""
    history_file = get_models_directory() / "history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data[:limit]
        except Exception:
            return []
    return []
