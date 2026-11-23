"""Ensemble learning pipeline for ReviewGuard.

Classical techniques:
1. Bagging: Random Forest Classifier
2. Boosting: AdaBoost Classifier
3. Boosting: Gradient Boosting Classifier
4. Model Averaging / Soft Voting Classifier with configurable weights
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import train_test_split

from .evaluate import evaluate_predictions
from .features import FeaturePipeline
from .load_data import load_dataset


class SoftVotingModel:
    """Configurable soft voting ensemble with custom model weights."""

    def __init__(self, models: Dict[str, object], weights: Optional[Dict[str, float]] = None):
        self.models = models
        if weights is None:
            # Default equal weights
            self.weights = {k: 1.0 / len(models) for k in models}
        else:
            total = sum(weights.values())
            self.weights = {k: weights.get(k, 0.0) / total for k in models}

    def predict_proba(self, X) -> np.ndarray:
        """Compute weighted average predicted probabilities."""
        avg_probs = np.zeros((X.shape[0], 2), dtype=float)
        for key, model in self.models.items():
            w = self.weights.get(key, 0.0)
            if w <= 0:
                continue
            try:
                probs = model.predict_proba(X)
                if probs.shape[1] == 1:
                    probs = np.hstack([1.0 - probs, probs])
            except Exception:
                preds = model.predict(X)
                probs = np.zeros((X.shape[0], 2))
                probs[:, 1] = preds
                probs[:, 0] = 1.0 - preds

            avg_probs += w * probs

        return avg_probs

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def train_ensemble_models(
    dataset_name: str = "kaggle",
    max_samples: int = 6000,
    test_size: float = 0.2,
    custom_weights: Optional[Dict[str, float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
) -> Dict:
    """Train Bagging (Random Forest), Boosting (AdaBoost, GradientBoosting), and Soft Voting.

    Returns comparison with single classical models and feature importances.
    """
    if output_dir is None:
        output_dir = Path.cwd() / "reviewguard" / "models"
        if not output_dir.exists():
            output_dir = Path.cwd() / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df, meta = load_dataset(dataset_name, sample_size=max_samples, random_state=random_state)
    has_meta = meta["has_metadata"]

    # 2. Features - use 2000 features matching FeaturePipeline default
    pipe_path = output_dir / "feature_pipeline.joblib"
    if pipe_path.exists():
        try:
            pipeline = FeaturePipeline.load(str(pipe_path))
            X = pipeline.transform(df)
        except Exception:
            pipeline = FeaturePipeline(
                max_tfidf_features=2000,
                include_behavioral=has_meta,
                random_state=random_state,
            )
            X = pipeline.fit_transform(df)
    else:
        pipeline = FeaturePipeline(
            max_tfidf_features=2000,
            include_behavioral=has_meta,
            random_state=random_state,
        )
        X = pipeline.fit_transform(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    ensemble_results = {}

    # 3. Random Forest (Bagging)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=6,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    ensemble_results["random_forest"] = {
        **evaluate_predictions(y_test, rf_pred, rf_prob),
        "type": "Bagging",
        "display_name": "Random Forest (Bagging)",
        "description": "Bootstrap aggregating with 100 decorrelated decision trees.",
    }
    joblib.dump(rf, output_dir / "random_forest.joblib")

    # 4. AdaBoost (Boosting)
    adaboost = AdaBoostClassifier(
        n_estimators=80,
        learning_rate=0.4,
        random_state=random_state,
    )
    adaboost.fit(X_train, y_train)
    ada_pred = adaboost.predict(X_test)
    ada_prob = adaboost.predict_proba(X_test)[:, 1]
    ensemble_results["adaboost"] = {
        **evaluate_predictions(y_test, ada_pred, ada_prob),
        "type": "Boosting",
        "display_name": "AdaBoost (Adaptive Boosting)",
        "description": "Sequential boosting adjusting sample weights on misclassifications.",
    }
    joblib.dump(adaboost, output_dir / "adaboost.joblib")

    # 5. Gradient Boosting (Boosting)
    # Using subsample of features or max_features='sqrt' for rapid training
    gbdt = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        max_features="sqrt",
        random_state=random_state,
    )
    gbdt.fit(X_train, y_train)
    gbdt_pred = gbdt.predict(X_test)
    gbdt_prob = gbdt.predict_proba(X_test)[:, 1]
    ensemble_results["gradient_boosting"] = {
        **evaluate_predictions(y_test, gbdt_pred, gbdt_prob),
        "type": "Boosting",
        "display_name": "Gradient Boosting (GBDT)",
        "description": "Stage-wise additive expansion optimizing binomial deviance loss.",
    }
    joblib.dump(gbdt, output_dir / "gradient_boosting.joblib")

    # 6. Soft Voting / Model Averaging
    # Load or instantiate base models if available
    voting_components = {
        "random_forest": rf,
        "gradient_boosting": gbdt,
        "adaboost": adaboost,
    }

    # Add single base models if present in models directory
    for key in ["svm", "logistic_regression", "naive_bayes"]:
        path = output_dir / f"{key}.joblib"
        if path.exists():
            try:
                voting_components[key] = joblib.load(path)
            except Exception:
                pass

    if custom_weights is None:
        custom_weights = {
            "random_forest": 0.40,
            "gradient_boosting": 0.35,
            "adaboost": 0.25,
        }

    voting_model = SoftVotingModel(voting_components, custom_weights)
    vote_probs = voting_model.predict_proba(X_test)
    vote_pred = voting_model.predict(X_test)

    ensemble_results["soft_voting"] = {
        **evaluate_predictions(y_test, vote_pred, vote_probs[:, 1]),
        "type": "Model Averaging",
        "display_name": "Soft Voting Ensemble",
        "description": "Weighted probability consensus across Bagging, Boosting, and Regularized models.",
        "weights": custom_weights,
    }
    joblib.dump(voting_model, output_dir / "soft_voting.joblib")

    # 7. Extract Feature Importances from Random Forest
    rf_importances = rf.feature_importances_
    ranked_indices = np.argsort(rf_importances)[::-1][:20]
    top_features = []
    for rank, idx in enumerate(ranked_indices, 1):
        raw_name = (
            pipeline.feature_names[idx]
            if idx < len(pipeline.feature_names)
            else f"feat_{idx}"
        )
        clean_name = raw_name.replace("tfidf_", "")
        top_features.append(
            {
                "rank": rank,
                "feature": clean_name,
                "importance": round(float(rf_importances[idx]), 4),
                "category": "Behavioral"
                if clean_name
                in [
                    "user_avg_rating",
                    "rating_deviation",
                    "reviews_per_day",
                    "time_gap_days",
                    "verified_flag",
                    "account_age_days",
                ]
                else (
                    "Linguistic"
                    if clean_name
                    in [
                        "review_length",
                        "word_count",
                        "punctuation_count",
                        "capital_ratio",
                        "exclamation_count",
                        "first_person_ratio",
                        "sentiment_compound",
                        "sentiment_pos",
                        "sentiment_neg",
                    ]
                    else "TF-IDF Lexical"
                ),
            }
        )

    # 8. Single Model vs Ensemble Comparison Table
    # Load single model metrics from summary.json if present
    single_vs_ensemble = []
    single_metrics_path = output_dir / "metrics.json"
    if single_metrics_path.exists():
        try:
            with open(single_metrics_path, "r", encoding="utf-8") as f:
                single_metrics = json.load(f)
            baseline_f1 = max(
                (m.get("f1", 0.0) for m in single_metrics.values()), default=0.85
            )
            for k, m in single_metrics.items():
                single_vs_ensemble.append(
                    {
                        "model": m.get("display_name", k),
                        "type": "Single Model",
                        "f1": m.get("f1", 0.0),
                        "accuracy": m.get("accuracy", 0.0),
                        "delta_f1": round(m.get("f1", 0.0) - baseline_f1, 4),
                    }
                )
        except Exception:
            baseline_f1 = 0.85
    else:
        baseline_f1 = 0.85

    for k, m in ensemble_results.items():
        single_vs_ensemble.append(
            {
                "model": m.get("display_name", k),
                "type": m.get("type", "Ensemble"),
                "f1": m.get("f1", 0.0),
                "accuracy": m.get("accuracy", 0.0),
                "delta_f1": round(m.get("f1", 0.0) - baseline_f1, 4),
            }
        )

    output = {
        "dataset_name": dataset_name,
        "ensembles": ensemble_results,
        "feature_importances": top_features,
        "comparison": single_vs_ensemble,
        "best_ensemble": max(
            ensemble_results.keys(), key=lambda k: ensemble_results[k]["f1"]
        ),
    }

    with open(output_dir / "ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
