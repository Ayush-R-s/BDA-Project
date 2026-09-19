"""
Spatial and Demographic Clustering Models for Indian Districts (NCRB 2024).

Implements Robust Scaling, PCA Dimensionality Reduction, K-Means Clustering,
Silhouette Score Evaluation, and DBSCAN Anomaly Detection.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

FEATURE_COLUMNS = [
    "violent_crime_ratio",
    "property_crime_ratio",
    "women_vulnerability_share",
    "child_vulnerability_share",
    "caste_atrocity_share",
    "juvenile_delinquency_ratio",
    "cyber_crime_intensity_per_1k",
    "trafficking_vulnerability_proxy",
]


def prepare_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Prepare scaled feature matrix with 98th percentile clipping to handle specialized police cells."""
    X_df = df[FEATURE_COLUMNS].copy()

    # Add log-transformed volume features to capture scale without skew
    X_df["log_total_crime"] = np.log1p(df["total_crime_burden"])
    X_df["log_contraband"] = np.log1p(df["contraband_enforcement_total"])

    # Clip extreme ratio outliers at 98th percentile (prevents division-by-zero artifacts from specialized cells)
    for col in FEATURE_COLUMNS:
        p98 = X_df[col].quantile(0.98)
        X_df[col] = np.clip(X_df[col], 0, p98)

    # Standard scaling across all dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return X_df, X_scaled, scaler


def run_pca(X_scaled: np.ndarray, n_components: int = 4) -> Tuple[PCA, np.ndarray, pd.DataFrame]:
    """Execute Principal Component Analysis and compute factor loadings."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    all_cols = FEATURE_COLUMNS + ["log_total_crime", "log_contraband"]
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=all_cols,
    )
    return pca, X_pca, loadings


def evaluate_kmeans_range(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
    """Evaluate K-Means across k values using Inertia, Silhouette Score, and Davies-Bouldin Index."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        results.append({
            "k": k,
            "inertia": round(km.inertia_, 2),
            "silhouette_score": round(sil, 4),
            "davies_bouldin": round(db, 4),
        })
    return pd.DataFrame(results)


def fit_optimal_kmeans(
    df: pd.DataFrame, X_scaled: np.ndarray, k: int = 5
) -> Tuple[pd.DataFrame, KMeans, pd.DataFrame]:
    """Fit optimal K-Means model, assign cluster IDs, name typologies, and compute centroids."""
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster_id"] = labels

    # Compute centroid profiles
    cols_to_profile = FEATURE_COLUMNS + [
        "total_ipc_crimes",
        "total_crime_burden",
        "contraband_enforcement_total",
    ]
    centroids = df_clustered.groupby("cluster_id")[cols_to_profile].mean()

    # Dynamic typology assignment based on centroid signature
    typology_map = {}
    for c_id in range(k):
        c_row = centroids.loc[c_id]
        if c_row["cyber_crime_intensity_per_1k"] > 500 or c_row["total_ipc_crimes"] < 500:
            typology_map[c_id] = "Specialized Investigation & Cyber Wings"
        elif c_row["women_vulnerability_share"] > 0.20 or c_row["violent_crime_ratio"] > 0.035:
            typology_map[c_id] = "High Violent Crime & Women Vulnerability Belts"
        elif c_row["total_crime_burden"] > 30000:
            typology_map[c_id] = "Major Commercial & High Crime Burden Hubs"
        elif c_row["caste_atrocity_share"] > 0.020:
            typology_map[c_id] = "Agrarian Belts with Elevated Atrocity Reporting"
        else:
            typology_map[c_id] = "Low-Intensity Stable Administrative Districts"

    df_clustered["typology_label"] = df_clustered["cluster_id"].map(typology_map)
    return df_clustered, km, centroids


def run_dbscan_anomalies(
    df: pd.DataFrame, X_scaled: np.ndarray, eps: float = 2.2, min_samples: int = 4
) -> pd.DataFrame:
    """Identify acute multivariate crime anomaly districts using DBSCAN."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    df_out = df.copy()
    df_out["dbscan_cluster"] = labels
    df_out["is_anomaly"] = labels == -1
    return df_out
