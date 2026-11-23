"""Dataset Explorer and corpus management routes."""

import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import numpy as np
import pandas as pd

from reviewguard.src.load_data import (
    find_dataset_path,
    has_behavioral_metadata,
    list_available_datasets,
    load_dataset,
)

router = APIRouter()


@router.get("/datasets")
def get_datasets():
    """List all available datasets in workspace."""
    return {"datasets": list_available_datasets()}


@router.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload custom CSV dataset, validate columns, and make available in selector."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are currently supported.")

    upload_dir = Path.cwd() / "reviewguard" / "data" / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df, meta = load_dataset(str(dest_path), sample_size=100)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded dataset: {str(e)}")

    return {
        "message": f"Successfully uploaded {file.filename}",
        "filename": file.filename,
        "rows": meta["total_rows"],
        "has_metadata": meta["has_metadata"],
        "columns": meta["columns"],
    }


@router.get("/dataset/explore")
def explore_dataset(
    dataset: str = Query("kaggle", description="Dataset identifier: 'kaggle' or 'sample'"),
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    label_filter: Optional[str] = Query(None),
):
    """Corpus distributions, rating histograms, class balance, and paginated review records."""
    dataset_val = str(dataset) if isinstance(dataset, str) else "kaggle"
    page_val = int(page) if isinstance(page, int) else 1
    page_size_val = int(page_size) if isinstance(page_size, int) else 15
    search_val = str(search) if isinstance(search, str) else None
    cat_val = str(category) if isinstance(category, str) else None
    label_val = str(label_filter) if isinstance(label_filter, str) else None

    try:
        # Load up to 10k rows for exploration speed
        df, meta = load_dataset(dataset_val, sample_size=10000)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Basic stats
    total_count = len(df)
    fake_count = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    genuine_count = int((df["label"] == 0).sum()) if "label" in df.columns else 0

    # Rating distribution
    rating_counts = df["rating"].value_counts().sort_index().to_dict()
    rating_dist = [
        {"rating": float(r), "count": int(c)}
        for r, c in rating_counts.items()
    ]

    # Word counts
    word_lengths = df["text"].astype(str).str.split().str.len()
    word_stats = {
        "mean": round(float(word_lengths.mean()), 1),
        "median": int(word_lengths.median()),
        "min": int(word_lengths.min()),
        "max": int(word_lengths.max()),
    }

    # Categories
    categories = df["category"].dropna().unique().tolist() if "category" in df.columns else []

    # Filtering for paginated table
    filtered_df = df.copy()
    if search_val:
        search_lower = search_val.lower()
        filtered_df = filtered_df[
            filtered_df["text"].astype(str).str.lower().str.contains(search_lower)
        ]
    if cat_val and cat_val != "all" and "category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["category"] == cat_val]
    if label_val and label_val != "all" and "label" in filtered_df.columns:
        target_val = 1 if label_val.lower() in ("fake", "1", "cg") else 0
        filtered_df = filtered_df[filtered_df["label"] == target_val]

    total_filtered = len(filtered_df)
    total_pages = max(1, int(np.ceil(total_filtered / page_size_val)))
    start_idx = (page_val - 1) * page_size_val
    end_idx = start_idx + page_size_val

    page_rows = filtered_df.iloc[start_idx:end_idx]
    records = []
    for idx, row in page_rows.iterrows():
        raw_text = str(row["text"])
        snippet = raw_text[:120] + ("..." if len(raw_text) > 120 else "")
        rec = {
            "index": int(idx),
            "review_id": str(row.get("review_id", f"R{idx:05d}")),
            "category": str(row.get("category", "General")),
            "rating": float(row.get("rating", 5.0)),
            "label": "Fake (CG)" if row.get("label") == 1 else "Genuine (OR)",
            "label_int": int(row.get("label", 0)),
            "text": raw_text,
            "snippet": snippet,
            "word_count": len(raw_text.split()),
        }
        # Add metadata fields if present
        if meta["has_metadata"]:
            rec["user_id"] = str(row.get("user_id", "N/A"))
            rec["verified_purchase"] = bool(row.get("verified_purchase", True))
            rec["account_age_days"] = int(row.get("account_age_days", 365))
            rec["review_date"] = str(row.get("review_date", "N/A"))

        records.append(rec)

    return {
        "dataset_name": meta["dataset_name"],
        "filename": meta["filename"],
        "has_metadata": meta["has_metadata"],
        "is_synthetic": meta["is_synthetic"],
        "total_records": total_count,
        "class_balance": {
            "fake_cg": fake_count,
            "genuine_or": genuine_count,
            "fake_pct": round(100.0 * fake_count / max(total_count, 1), 1),
            "genuine_pct": round(100.0 * genuine_count / max(total_count, 1), 1),
        },
        "rating_distribution": rating_dist,
        "word_stats": word_stats,
        "categories": categories,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_filtered": total_filtered,
            "total_pages": total_pages,
        },
        "records": records,
    }
