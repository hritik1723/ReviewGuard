"""Training pipeline for classical machine learning models in ReviewGuard.

Supported models:
1. Multinomial Naive Bayes (MultinomialNB)
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (LinearSVC with Probability Calibration)
4. Decision Tree (CART)
5. Logistic Regression with L1 Regularization (LASSO)
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .evaluate import evaluate_model_cv, evaluate_predictions
from .features import FeaturePipeline
from .imbalance import handle_imbalance
from .load_data import load_dataset


DEFAULT_MODEL_KEYS = ["naive_bayes", "logistic_regression", "decision_tree", "svm", "knn"]


def get_base_classifier(model_key: str, class_weight: Optional[str] = None, random_state: int = 42):
    """Instantiate a classical classifier with reproducible parameters."""
    key = model_key.lower().strip()

    if key in ("naive_bayes", "multinomial_nb", "nb"):
        return MultinomialNB(alpha=1.0)

    elif key in ("logistic_regression", "lasso", "lr"):
        # L1 LASSO regularization using liblinear solver
        return LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
        )

    elif key in ("decision_tree", "dt"):
        return DecisionTreeClassifier(
            max_depth=16,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif key in ("svm", "linear_svc"):
        base_svc = LinearSVC(
            C=1.0,
            penalty="l2",
            class_weight=class_weight,
            max_iter=2000,
            random_state=random_state,
            dual="auto",
        )
        # CalibratedClassifierCV gives well-calibrated probabilities
        return CalibratedClassifierCV(estimator=base_svc, cv=3)

    elif key in ("knn", "k_nearest_neighbors"):
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine",
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model key: {model_key}")


def train_and_evaluate_models(
    dataset_name: str = "kaggle",
    model_keys: Optional[List[str]] = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    imbalance_method: str = "none",
    max_samples: Optional[int] = 10000,
    max_tfidf_features: int = 2000,
    random_state: int = 42,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """End-to-end training and evaluation pipeline for classical ML models.

    Parameters:
    - dataset_name: 'kaggle', 'sample', or custom file path
    - model_keys: list of model keys to train
    - test_size: test split fraction (e.g. 0.2)
    - cv_folds: 5 or 10 fold cross validation
    - imbalance_method: 'none', 'smote', 'oversample', 'undersample', 'class_weight'
    - max_samples: max dataset rows to use (for laptop speed, defaults to 10k)
    - max_tfidf_features: TF-IDF feature vocabulary size
    - progress_callback: optional progress reporting fn(status_text, percent)
    """
    if model_keys is None:
        model_keys = DEFAULT_MODEL_KEYS

    if output_dir is None:
        # Default to reviewguard/models/
        output_dir = Path.cwd() / "reviewguard" / "models"
        if not output_dir.exists():
            output_dir = Path.cwd() / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading dataset...", 10)

    # 1. Load data
    df, meta = load_dataset(dataset_name, sample_size=max_samples, random_state=random_state)
    has_meta = meta["has_metadata"]

    if progress_callback:
        progress_callback("Extracting features...", 25)

    # 2. Extract features
    pipeline = FeaturePipeline(
        max_tfidf_features=max_tfidf_features,
        include_behavioral=has_meta,
        random_state=random_state,
    )
    X = pipeline.fit_transform(df)
    y = df["label"].values

    # Save feature pipeline
    pipeline.save(str(output_dir / "feature_pipeline.joblib"))

    if progress_callback:
        progress_callback("Splitting and balancing data...", 35)

    # 3. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4. Handle Imbalance
    use_class_weight = "balanced" if imbalance_method == "class_weight" else None
    X_train_res, y_train_res, imb_stats = handle_imbalance(
        X_train, y_train, method=imbalance_method, random_state=random_state
    )

    # 5. Train each model
    results = {}
    best_model_name = None
    best_f1 = -1.0

    step_increment = 50 / max(len(model_keys), 1)
    current_progress = 40.0

    for idx, key in enumerate(model_keys):
        model_display_name = {
            "naive_bayes": "Multinomial Naive Bayes",
            "logistic_regression": "Logistic Regression (L1 LASSO)",
            "decision_tree": "Decision Tree (CART)",
            "svm": "Linear SVM (Calibrated)",
            "knn": "K-Nearest Neighbors",
        }.get(key, key)

        if progress_callback:
            progress_callback(f"Training {model_display_name}...", int(current_progress))

        clf = get_base_classifier(key, class_weight=use_class_weight, random_state=random_state)

        # Initialize training and test matrices
        X_tr = X_train_res
        y_tr = y_train_res
        X_te = X_test

        # For Naive Bayes, ensure non-negative inputs
        if key in ("naive_bayes", "multinomial_nb"):
            X_tr = X_train_res.copy()
            X_te = X_test.copy()
            X_tr.data = np.clip(X_tr.data, 0, None)
            X_te.data = np.clip(X_te.data, 0, None)
        elif key == "knn" and X_train_res.shape[0] > 6000:
            # Subsample for KNN to keep training under a few seconds
            sub_idx = np.random.RandomState(random_state).choice(
                X_train_res.shape[0], size=5000, replace=False
            )
            X_tr = X_train_res[sub_idx]
            y_tr = y_train_res[sub_idx]

        # Fit
        clf.fit(X_tr, y_tr)

        # Predict
        y_pred = clf.predict(X_te)
        try:
            y_prob = clf.predict_proba(X_te)[:, 1]
        except Exception:
            try:
                dec = clf.decision_function(X_te)
                y_prob = 1.0 / (1.0 + np.exp(-dec))
            except Exception:
                y_prob = y_pred.astype(float)

        # Evaluation metrics
        metrics = evaluate_predictions(y_test, y_pred, y_prob)

        # K-Fold CV
        cv_eval = evaluate_model_cv(clf, X_tr, y_tr, cv_folds=cv_folds, random_state=random_state)
        metrics["cross_validation"] = cv_eval

        # Model metadata
        metrics["model_key"] = key
        metrics["display_name"] = model_display_name
        metrics["hyperparameters"] = {
            k: str(v) for k, v in clf.get_params().items() if not k.startswith("_")
        }

        # Save individual model
        joblib.dump(clf, output_dir / f"{key}.joblib")

        results[key] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = key

        current_progress += step_increment

    if progress_callback:
        progress_callback("Finalizing training summary...", 95)

    summary = {
        "dataset_name": dataset_name,
        "total_samples": len(df),
        "train_samples": int(X_train_res.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features_count": int(X.shape[1]),
        "has_metadata": has_meta,
        "imbalance_handling": imb_stats,
        "best_model": best_model_name,
        "best_f1": best_f1,
        "models": results,
    }

    # Save summary and metrics
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if progress_callback:
        progress_callback("Training completed successfully!", 100)

    return summary
