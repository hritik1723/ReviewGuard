"""Data loading and schema normalization for ReviewGuard.

Supports:
1. Kaggle "Fake Reviews Dataset" (CG=fake, OR=genuine, text_, rating, category)
2. Synthetic Sample Reviews Dataset (adds review_id, user_id, product_id, review_date, verified_purchase, account_age_days)
3. Custom uploaded CSV datasets
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "user_id",
    "product_id",
    "review_date",
    "verified_purchase",
    "account_age_days",
]


def find_dataset_path(identifier: str = "kaggle") -> Optional[Path]:
    """Auto-detect path for dataset based on identifier or search candidate directories."""
    # Possible base directories to search
    base_dirs = [
        Path.cwd(),
        Path.cwd() / "reviewguard",
        Path.cwd() / "reviewguard" / "data" / "raw",
        Path.cwd() / "data" / "raw",
        Path(__file__).resolve().parent.parent / "data" / "raw",
        Path(__file__).resolve().parent.parent.parent,
    ]

    identifier_lower = identifier.lower().strip()

    if identifier_lower in ("kaggle", "main", "fake_reviews"):
        candidates = [
            "fake reviews dataset.csv",
            "fake_reviews_dataset.csv",
            "fake_reviews.csv",
            "Fake_Reviews_Dataset.csv",
        ]
        for d in base_dirs:
            for c in candidates:
                p = d / c
                if p.exists() and p.is_file():
                    return p.resolve()

    elif identifier_lower in ("sample", "synthetic", "sample_reviews"):
        candidates = [
            "sample_reviews_dataset.csv",
            "sample_reviews.csv",
            "sample_dataset.csv",
        ]
        for d in base_dirs:
            for c in candidates:
                p = d / c
                if p.exists() and p.is_file():
                    return p.resolve()

    # Try exact path or relative path
    direct_path = Path(identifier)
    if direct_path.exists() and direct_path.is_file():
        return direct_path.resolve()

    for d in base_dirs:
        p = d / identifier
        if p.exists() and p.is_file():
            return p.resolve()

    return None


def has_behavioral_metadata(df: pd.DataFrame) -> bool:
    """Check if the dataframe has required behavioral metadata columns."""
    needed = {"user_id", "review_date"}
    cols = {c.lower() for c in df.columns}
    return needed.issubset(cols)


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize label column to binary 0 (Genuine/OR) and 1 (Fake/CG)."""
    mapping = {
        "CG": 1,
        "cg": 1,
        "FAKE": 1,
        "fake": 1,
        "Fake": 1,
        "1": 1,
        1: 1,
        1.0: 1,
        "OR": 0,
        "or": 0,
        "GENUINE": 0,
        "genuine": 0,
        "Genuine": 0,
        "REAL": 0,
        "real": 0,
        "0": 0,
        0: 0,
        0.0: 0,
    }
    return series.map(mapping).fillna(0).astype(int)


def load_dataset(
    identifier: str = "kaggle",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Load dataset, clean nulls/duplicates, normalize labels, and return df + metadata info.

    Parameters
    ----------
    identifier : str
        'kaggle', 'sample', or a specific file path / name.
    sample_size : Optional[int]
        If set, stratify/sample to this number of rows for faster processing.
    random_state : int
        Seed for reproducibility.
    """
    path = find_dataset_path(identifier)
    if path is None:
        raise FileNotFoundError(f"Could not locate dataset for '{identifier}'.")

    # Read CSV
    df = pd.read_csv(path)

    # Standardize column names
    col_rename = {}
    for col in df.columns:
        c_clean = col.strip()
        if c_clean.lower() in ("text_", "text", "review_text", "review"):
            col_rename[col] = "text"
        elif c_clean.lower() in ("label", "target", "class"):
            col_rename[col] = "label"
        elif c_clean.lower() in ("rating", "score", "stars"):
            col_rename[col] = "rating"
        elif c_clean.lower() in ("category", "product_category"):
            col_rename[col] = "category"
        elif c_clean.lower() in ("verified_purchase", "verified"):
            col_rename[col] = "verified_purchase"
        elif c_clean.lower() in ("account_age_days", "account_age"):
            col_rename[col] = "account_age_days"

    df = df.rename(columns=col_rename)

    if "text" not in df.columns:
        raise ValueError(f"Dataset at {path} lacks a review text column.")

    # Drop null text and rating
    initial_len = len(df)
    df = df.dropna(subset=["text"])
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)

    # Drop duplicates on text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Normalize labels if present
    if "label" in df.columns:
        df["label"] = normalize_labels(df["label"])
    else:
        df["label"] = 0

    has_meta = has_behavioral_metadata(df)
    is_synthetic = "sample" in path.name.lower() or "synthetic" in path.name.lower()

    # If sample size requested and dataframe is larger, subsample stratified
    if sample_size and len(df) > sample_size:
        if "label" in df.columns and df["label"].nunique() > 1:
            from sklearn.model_selection import train_test_split
            _, df = train_test_split(
                df, test_size=sample_size, stratify=df["label"], random_state=random_state
            )
            df = df.reset_index(drop=True)
        else:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    meta_info = {
        "dataset_name": path.stem,
        "filename": path.name,
        "filepath": str(path),
        "total_rows": int(len(df)),
        "initial_rows": int(initial_len),
        "has_metadata": bool(has_meta),
        "is_synthetic": bool(is_synthetic),
        "columns": list(df.columns),
        "fake_count": int((df["label"] == 1).sum()) if "label" in df.columns else 0,
        "genuine_count": int((df["label"] == 0).sum()) if "label" in df.columns else 0,
        "categories": df["category"].dropna().unique().tolist()
        if "category" in df.columns
        else [],
    }

    return df, meta_info


def list_available_datasets() -> List[Dict]:
    """Scan and return details for all available datasets."""
    datasets = []
    seen_paths = set()

    for ident in ["kaggle", "sample"]:
        path = find_dataset_path(ident)
        if path and str(path) not in seen_paths:
            seen_paths.add(str(path))
            try:
                # Read just header & row count efficiently
                df_head = pd.read_csv(path, nrows=50)
                has_meta = has_behavioral_metadata(df_head)
                is_synthetic = (
                    "sample" in path.name.lower() or "synthetic" in path.name.lower()
                )
                # Count lines quickly
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    row_count = sum(1 for _ in f) - 1

                datasets.append(
                    {
                        "id": "kaggle" if "fake" in path.name.lower() else "sample",
                        "name": "Kaggle Fake Reviews Dataset"
                        if "fake" in path.name.lower()
                        else "Synthetic Sample Reviews (with Metadata)",
                        "filename": path.name,
                        "rows": max(row_count, 0),
                        "has_metadata": has_meta,
                        "is_synthetic": is_synthetic,
                        "description": "Kaggle benchmark (40k reviews, CG vs OR, no user metadata)"
                        if not has_meta
                        else "Synthetic behavioral dataset (1,000 reviews with reviewer IDs, timestamps, verified flags)",
                    }
                )
            except Exception as e:
                pass

    return datasets
