"""Model evaluation and metrics calculation for ReviewGuard.

Classical metrics:
- Accuracy, Precision, Recall, F1 Score
- ROC AUC & ROC Curve coordinates
- Precision-Recall Curve coordinates
- Confusion Matrix (TN, FP, FN, TP)
- K-Fold Cross-Validation scores (mean + std)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def downsample_curve(
    x_vals: np.ndarray, y_vals: np.ndarray, max_points: int = 30
) -> List[Dict[str, float]]:
    """Downsample ROC/PR curve points for compact JSON serialization and smooth charts."""
    if len(x_vals) <= max_points:
        return [
            {"x": round(float(x), 4), "y": round(float(y), 4)}
            for x, y in zip(x_vals, y_vals)
        ]

    indices = np.linspace(0, len(x_vals) - 1, max_points, dtype=int)
    return [
        {"x": round(float(x_vals[i]), 4), "y": round(float(y_vals[i]), 4)}
        for i in indices
    ]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    """Compute comprehensive classical classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (
        cm.ravel()
        if cm.size == 4
        else (int(cm[0, 0]), 0, 0, 0)
    )

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)

            metrics["auc"] = round(auc, 4)
            metrics["roc_curve"] = downsample_curve(fpr, tpr)
            metrics["pr_curve"] = downsample_curve(recalls, precisions)
        except Exception:
            metrics["auc"] = round(acc, 4)
            metrics["roc_curve"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
            metrics["pr_curve"] = [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 0.0}]
    else:
        metrics["auc"] = round(acc, 4)
        metrics["roc_curve"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
        metrics["pr_curve"] = [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 0.0}]

    return metrics


def evaluate_model_cv(
    model,
    X,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict:
    """Run Stratified K-Fold cross-validation and return fold scores, mean, and std."""
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
    return {
        "cv_folds": cv_folds,
        "fold_scores": [round(float(s), 4) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 4),
        "cv_std": round(float(np.std(scores)), 4),
    }
