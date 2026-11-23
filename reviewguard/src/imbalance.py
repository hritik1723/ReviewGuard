"""Class imbalance handling and outlier detection for ReviewGuard.

Classical techniques:
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. Random Oversampling
3. Random Undersampling
4. Class weights calculation ('balanced')
5. Outlier detection (Isolation Forest / Interquartile Range)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from collections import Counter
from scipy.sparse import issparse, csr_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def handle_imbalance(
    X,
    y: np.ndarray,
    method: str = "none",
    random_state: int = 42,
) -> Tuple[object, np.ndarray, Dict]:
    """Handle class imbalance on training data.

    Parameters:
    - X: feature matrix (sparse or dense)
    - y: labels
    - method: 'none', 'smote', 'oversample', 'undersample', 'class_weight'
    - random_state: integer seed

    Returns:
    - X_res: resampled feature matrix
    - y_res: resampled labels
    - stats: dictionary with before/after counts and ratio
    """
    initial_counts = dict(Counter(y))
    before_counts = {
        "genuine_0": int(initial_counts.get(0, 0)),
        "fake_1": int(initial_counts.get(1, 0)),
        "total": int(len(y)),
        "ratio_fake": float(
            initial_counts.get(1, 0) / len(y) if len(y) > 0 else 0.0
        ),
    }

    method_clean = method.lower().strip()

    if method_clean == "none" or method_clean == "class_weight":
        return X, y, {
            "method": method_clean,
            "before": before_counts,
            "after": before_counts,
            "class_weights": get_class_weights(y) if method_clean == "class_weight" else None,
        }

    if not HAS_IMBLEARN:
        # Graceful fallback if imblearn not available
        return X, y, {
            "method": "none (imblearn not available)",
            "before": before_counts,
            "after": before_counts,
            "class_weights": None,
        }

    # Only resample if multiple classes exist
    if len(np.unique(y)) < 2:
        return X, y, {
            "method": method_clean,
            "before": before_counts,
            "after": before_counts,
            "class_weights": None,
        }

    if method_clean == "smote":
        k_neighbors = min(5, min(initial_counts.values()) - 1)
        if k_neighbors < 1:
            sampler = RandomOverSampler(random_state=random_state)
        else:
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    elif method_clean == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif method_clean == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        sampler = None

    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    after_counts_raw = dict(Counter(y_res))
    after_counts = {
        "genuine_0": int(after_counts_raw.get(0, 0)),
        "fake_1": int(after_counts_raw.get(1, 0)),
        "total": int(len(y_res)),
        "ratio_fake": float(
            after_counts_raw.get(1, 0) / len(y_res) if len(y_res) > 0 else 0.0
        ),
    }

    return X_res, y_res, {
        "method": method_clean,
        "before": before_counts,
        "after": after_counts,
        "class_weights": None,
    }


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for classical classifiers."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(round(w, 4)) for cls, w in zip(classes, weights)}


def detect_outliers(X, contamination: float = 0.05, random_state: int = 42) -> np.ndarray:
    """Detect anomalous review vectors using classical Isolation Forest."""
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    # -1 for outliers, 1 for inliers
    preds = iso.fit_predict(X)
    return preds == -1
