"""Reviewer clustering pipeline for ReviewGuard.

Classical techniques:
1. K-Means Clustering with Elbow Method (k=2..10) and Silhouette evaluation
2. 2D PCA dimensionality reduction for visual cluster projection
3. Agglomerative Hierarchical Clustering with linkage selector (Ward, Complete, Average, Single) and Dendrogram data
4. Minimum Spanning Tree (MST) graph clustering concept for syndicate detection
5. Behavioral cluster profiling & anomaly tagging
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .load_data import has_behavioral_metadata, load_dataset


def extract_reviewer_profiles(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate review-level rows into distinct reviewer behavioral profiles."""
    if not has_behavioral_metadata(df):
        raise ValueError("Dataset does not contain reviewer metadata columns.")

    data = df.copy()
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(3.0)
    data["review_len"] = data["text"].astype(str).str.len()
    data["exclamations"] = data["text"].astype(str).str.count("!")

    if "verified_purchase" in data.columns:
        data["verified_num"] = (
            data["verified_purchase"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "verified"])
            .astype(float)
        )
    else:
        data["verified_num"] = 1.0

    if "review_date" in data.columns:
        data["parsed_date"] = pd.to_datetime(data["review_date"], errors="coerce")
    else:
        data["parsed_date"] = pd.to_datetime("2024-01-01")

    # Group by user_id
    grouped = data.groupby("user_id")

    profiles = []
    for user_id, group in grouped:
        count = len(group)
        avg_rating = float(group["rating"].mean())
        rating_std = float(group["rating"].std(ddof=0)) if count > 1 else 0.0
        verified_pct = float(group["verified_num"].mean())
        avg_len = float(group["review_len"].mean())
        avg_excl = float(group["exclamations"].mean())
        fake_ratio = (
            float((group["label"] == 1).mean()) if "label" in group.columns else 0.0
        )

        # Posting frequency (reviews per day span)
        date_span = max(
            (group["parsed_date"].max() - group["parsed_date"].min()).total_seconds()
            / 86400.0,
            1.0,
        )
        reviews_per_day = float(count / date_span)

        if "account_age_days" in group.columns:
            acc_age = float(
                pd.to_numeric(group["account_age_days"], errors="coerce")
                .dropna()
                .mean()
            )
        else:
            acc_age = 365.0

        profiles.append(
            {
                "user_id": str(user_id),
                "review_count": count,
                "avg_rating": round(avg_rating, 2),
                "rating_std": round(rating_std, 2),
                "reviews_per_day": round(reviews_per_day, 3),
                "verified_ratio": round(verified_pct, 2),
                "avg_review_length": round(avg_len, 1),
                "exclamation_avg": round(avg_excl, 2),
                "account_age_days": round(acc_age, 0),
                "actual_fake_ratio": round(fake_ratio, 2),
            }
        )

    profile_df = pd.DataFrame(profiles)
    return profile_df, data


def run_reviewer_clustering(
    dataset_name: str = "sample",
    n_clusters: int = 4,
    linkage_method: str = "ward",
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
) -> Dict:
    """Execute complete clustering suite on reviewer behavioral profiles."""
    if output_dir is None:
        output_dir = Path.cwd() / "reviewguard" / "models"
        if not output_dir.exists():
            output_dir = Path.cwd() / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_dataset(dataset_name)

    if not meta["has_metadata"]:
        return {
            "requires_metadata": True,
            "message": (
                "Reviewer Clustering requires user metadata (user_id, review_date, verified_purchase). "
                "The current dataset does not include reviewer identities. Please switch to the "
                "Synthetic Sample Reviews dataset to explore reviewer clusters."
            ),
        }

    profile_df, _ = extract_reviewer_profiles(df)

    feature_cols = [
        "avg_rating",
        "rating_std",
        "reviews_per_day",
        "verified_ratio",
        "avg_review_length",
        "exclamation_avg",
        "account_age_days",
    ]

    X_raw = profile_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. Elbow Method & Silhouette Scores (k = 2 to 8)
    elbow_data = []
    max_k = min(8, len(profile_df) - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        sil = float(silhouette_score(X_scaled, km.labels_))
        elbow_data.append(
            {
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette": round(sil, 4),
            }
        )

    # 2. Final K-Means Model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    profile_df["cluster"] = cluster_labels

    # 3. 2D PCA Projection
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    profile_df["pca_x"] = np.round(pca_coords[:, 0], 3)
    profile_df["pca_y"] = np.round(pca_coords[:, 1], 3)

    pca_explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # Sample points for 2D chart if user count is large
    sample_size = min(200, len(profile_df))
    plot_df = profile_df.sample(n=sample_size, random_state=random_state)
    pca_points = []
    for _, r in plot_df.iterrows():
        pca_points.append(
            {
                "x": float(r["pca_x"]),
                "y": float(r["pca_y"]),
                "cluster": int(r["cluster"]),
                "user_id": str(r["user_id"]),
                "avg_rating": float(r["avg_rating"]),
                "reviews_per_day": float(r["reviews_per_day"]),
                "verified_ratio": float(r["verified_ratio"]),
                "actual_fake_ratio": float(r["actual_fake_ratio"]),
            }
        )

    # 4. Cluster Summaries & Suspicious Tagging
    cluster_summaries = []
    for c_id in range(n_clusters):
        c_subset = profile_df[profile_df["cluster"] == c_id]
        c_size = len(c_subset)
        if c_size == 0:
            continue

        c_avg_rating = float(c_subset["avg_rating"].mean())
        c_rpd = float(c_subset["reviews_per_day"].mean())
        c_ver = float(c_subset["verified_ratio"].mean())
        c_len = float(c_subset["avg_review_length"].mean())
        c_excl = float(c_subset["exclamation_avg"].mean())
        c_fake = float(c_subset["actual_fake_ratio"].mean())

        # Anomaly heuristic tagging
        if c_rpd > 1.5 or (c_avg_rating > 4.7 and c_ver < 0.3):
            tag = "Suspicious Ring / Bot Farm"
            severity = "high"
        elif c_avg_rating < 1.8 and c_rpd > 0.8:
            tag = "Review Bombing Cluster"
            severity = "high"
        elif c_fake > 0.45 or c_excl > 2.0:
            tag = "Elevated Anomalies"
            severity = "medium"
        else:
            tag = "Normal / Organic Behavior"
            severity = "low"

        cluster_summaries.append(
            {
                "cluster_id": c_id,
                "size": c_size,
                "percentage": round(100.0 * c_size / len(profile_df), 1),
                "avg_rating": round(c_avg_rating, 2),
                "reviews_per_day": round(c_rpd, 3),
                "verified_ratio": round(c_ver, 2),
                "avg_length": round(c_len, 1),
                "fake_ratio": round(c_fake, 2),
                "suspicious_tag": tag,
                "severity": severity,
            }
        )

    # 5. Hierarchical Clustering & Dendrogram Data
    # Subsample 25 representative users for clear dendrogram rendering
    sub_hier = profile_df.sample(n=min(25, len(profile_df)), random_state=random_state)
    sub_X = scaler.transform(sub_hier[feature_cols].values)

    linkage_clean = linkage_method.lower().strip()
    if linkage_clean not in ("ward", "complete", "average", "single"):
        linkage_clean = "ward"

    Z = linkage(sub_X, method=linkage_clean)
    dendro_dict = dendrogram(Z, labels=sub_hier["user_id"].tolist(), no_plot=True)

    dendrogram_data = {
        "linkage_method": linkage_clean,
        "icoord": [[round(float(v), 2) for v in row] for row in dendro_dict["icoord"][:20]],
        "dcoord": [[round(float(v), 2) for v in row] for row in dendro_dict["dcoord"][:20]],
        "labels": dendro_dict["ivl"][:20],
    }

    # 6. Minimum Spanning Tree (MST) Concept Graph
    # Compute pairwise distances between cluster centroids
    centroids = kmeans.cluster_centers_
    dist_mat = squareform(pdist(centroids, metric="euclidean"))
    mst_mat = minimum_spanning_tree(dist_mat).toarray()

    mst_edges = []
    threshold = float(np.mean(dist_mat[dist_mat > 0])) if np.any(dist_mat > 0) else 1.0
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if mst_mat[i, j] > 0:
                mst_edges.append(
                    {
                        "source": int(i),
                        "target": int(j),
                        "weight": round(float(mst_mat[i, j]), 3),
                        "is_anomaly_cut": bool(mst_mat[i, j] > threshold),
                    }
                )

    result = {
        "requires_metadata": False,
        "total_reviewers": len(profile_df),
        "n_clusters": n_clusters,
        "elbow_curve": elbow_data,
        "pca_explained_variance": pca_explained,
        "pca_points": pca_points,
        "cluster_summaries": cluster_summaries,
        "dendrogram": dendrogram_data,
        "mst_edges": mst_edges,
    }

    with open(output_dir / "clustering_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
