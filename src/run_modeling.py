"""
Master Modeling Pipeline Runner for NCRB 2024 Crime Analytics Suite.

Executes PCA Dimensionality Reduction, K-Means Spatial Clustering, DBSCAN
Anomaly Detection, and Metropolitan Policing Quadrant Analysis. Generates
visualizations and exports results to outputs/ and data/processed/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.modeling.cluster_models import (
    evaluate_kmeans_range,
    fit_optimal_kmeans,
    prepare_clustering_features,
    run_dbscan_anomalies,
    run_pca,
)
from src.modeling.metro_analytics import (
    compute_policing_quadrants,
    extract_dominant_motives,
)


def run_modeling_pipeline():
    print("=" * 80)
    print("NCRB 2024 MACHINE LEARNING & SPATIAL CLUSTERING PIPELINE")
    print("=" * 80)

    proc_dir = PROJECT_ROOT / "data" / "processed"
    outputs_dir = PROJECT_ROOT / "outputs"
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 150

    # ==========================================
    # 1. LOAD PREPROCESSED DATASETS
    # ==========================================
    print("\n[Step 1/5] Loading Processed District & Metropolitan Feature Matrices...")
    df_dist = pd.read_parquet(proc_dir / "district_engineered_features.parquet")
    df_metro_trends = pd.read_parquet(proc_dir / "metro_master_trends.parquet")
    df_metro_master = pd.read_parquet(proc_dir / "metro_master_feature_matrix.parquet")
    print(f"  District Dataset Shape: {df_dist.shape}")
    print(f"  Metropolitan Trends Shape: {df_metro_trends.shape}")

    # ==========================================
    # 2. FEATURE SCALING & PCA
    # ==========================================
    print("\n[Step 2/5] Running Robust Scaling & PCA Dimensionality Reduction...")
    X_df, X_scaled, scaler = prepare_clustering_features(df_dist)
    pca, X_pca, loadings = run_pca(X_scaled, n_components=4)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    print(f"  PCA Explained Variance Ratio: {np.round(var_ratio, 4)}")
    print(f"  Cumulative Explained Variance (4 PCs): {cum_var[-1]:.4f}")

    # Save PCA loadings
    loadings.to_csv(outputs_dir / "pca_factor_loadings.csv")
    print(f"  Saved factor loadings -> {outputs_dir.relative_to(PROJECT_ROOT)}/pca_factor_loadings.csv")

    # ==========================================
    # 3. K-MEANS OPTIMAL K EVALUATION & FITTING
    # ==========================================
    print("\n[Step 3/5] Evaluating Optimal K and Fitting K-Means Model...")
    eval_df = evaluate_kmeans_range(X_scaled, k_range=range(2, 9))
    eval_df.to_csv(outputs_dir / "kmeans_evaluation_metrics.csv", index=False)
    print("  K-Means Evaluation Metrics:\n", eval_df.to_string(index=False))

    # Plot Elbow & Silhouette curve
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(eval_df["k"], eval_df["inertia"], "o-", color="#2980b9", lw=2, label="Inertia (Elbow)")
    ax2.plot(eval_df["k"], eval_df["silhouette_score"], "s--", color="#e74c3c", lw=2, label="Silhouette Score")
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia", color="#2980b9", fontweight="bold")
    ax2.set_ylabel("Silhouette Score", color="#e74c3c", fontweight="bold")
    plt.title("Optimal Cluster Count Analysis: Elbow & Silhouette Score", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "elbow_silhouette_analysis.png")
    plt.close()
    print("  Saved: elbow_silhouette_analysis.png")

    # Fit k=5
    optimal_k = 5
    df_clustered, km, centroids = fit_optimal_kmeans(df_dist, X_scaled, k=optimal_k)
    centroids.to_csv(outputs_dir / "cluster_profiles.csv")
    print(f"\n  Fitted K-Means with k={optimal_k}. Typology Distribution:")
    print(df_clustered["typology_label"].value_counts().to_string())

    # DBSCAN Anomaly Detection
    df_clustered = run_dbscan_anomalies(df_clustered, X_scaled, eps=2.2, min_samples=4)
    n_anomalies = df_clustered["is_anomaly"].sum()
    print(f"  DBSCAN Anomaly Districts Identified: {n_anomalies} out of {len(df_clustered)}")

    # Export clustered dataset
    df_clustered.to_parquet(proc_dir / "district_clusters.parquet", index=False)
    df_clustered.to_csv(proc_dir / "district_clusters.csv", index=False)
    print(f"  Saved clustered districts -> {proc_dir.relative_to(PROJECT_ROOT)}/district_clusters.parquet")

    # ==========================================
    # 4. CLUSTER VISUALIZATIONS
    # ==========================================
    print("\n[Step 4/5] Generating High-Resolution Cluster Visualizations...")

    # A. PCA Biplot with Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_clustered["typology_label"],
        palette="tab10",
        alpha=0.8,
        s=50,
    )
    plt.title(f"PCA 2D Projection of Indian Districts (k={optimal_k} Clusters)", fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Typology")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_cluster_biplot.png")
    plt.close()
    print("  Saved: pca_cluster_biplot.png")

    # B. Centroid Profile Heatmap
    plt.figure(figsize=(11, 5))
    centroid_norm = (centroids - centroids.mean()) / centroids.std()
    sns.heatmap(
        centroid_norm.T,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Standardized Z-Score Deviation"},
    )
    plt.title("Crime Typology Centroid Profiles (Z-Score Deviation)", fontweight="bold")
    plt.xlabel("Cluster ID")
    plt.ylabel("Crime Dimension")
    plt.tight_layout()
    plt.savefig(figures_dir / "district_cluster_profiles_heatmap.png")
    plt.close()
    print("  Saved: district_cluster_profiles_heatmap.png")

    # ==========================================
    # 5. METROPOLITAN POLICY QUADRANT & MOTIVES
    # ==========================================
    print("\n[Step 5/5] Generating Metropolitan Policy Quadrant & Motive Analytics...")

    df_quadrants = compute_policing_quadrants(df_metro_trends)
    df_quadrants.to_parquet(proc_dir / "metro_policing_quadrants.parquet", index=False)
    df_quadrants.to_csv(proc_dir / "metro_policing_quadrants.csv", index=False)

    # Plot Quadrant
    plt.figure(figsize=(11, 6.5))
    med_x = df_quadrants["med_crime_rate"].iloc[0]
    med_y = df_quadrants["med_chargesheet_rate"].iloc[0]

    sns.scatterplot(
        data=df_quadrants,
        x="crime_rate_per_lakh",
        y="chargesheeting_rate",
        hue="policing_quadrant",
        size="population_lakhs",
        sizes=(70, 450),
        palette="Set1",
        alpha=0.85,
    )
    plt.axvline(med_x, color="black", linestyle="--", alpha=0.5, label=f"Median Crime Rate ({med_x:.1f})")
    plt.axhline(med_y, color="black", linestyle=":", alpha=0.5, label=f"Median Chargesheeting ({med_y:.1f}%)")

    # Label key benchmark metros
    key_cities = ["Delhi", "Mumbai", "Bengaluru City", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad", "Patna", "Jaipur"]
    for _, r in df_quadrants.iterrows():
        if any(k.lower() in r["city"].lower() for k in key_cities):
            plt.text(r["crime_rate_per_lakh"] + 6, r["chargesheeting_rate"] + 0.8, r["city"], fontsize=8.5, weight="semibold")

    plt.title("Metropolitan Policing Efficiency Quadrant Analysis (NCRB 2024)", fontweight="bold")
    plt.xlabel("Cognizable Crime Rate per Lakh Population (2024)")
    plt.ylabel("Chargesheeting Rate (%) (2024)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Quadrant Classification")
    plt.tight_layout()
    plt.savefig(figures_dir / "metro_policing_efficiency_quadrant.png")
    plt.close()
    print("  Saved: metro_policing_efficiency_quadrant.png")

    # Dominant Murder and Cyber Motives Plot
    murder_motives, cyber_motives = extract_dominant_motives(df_metro_master)
    murder_motives.to_csv(outputs_dir / "murder_motives_ranking.csv", index=False)
    cyber_motives.to_csv(outputs_dir / "cyber_motives_ranking.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    top_m = murder_motives.head(6)
    sns.barplot(data=top_m, y="motive", x="total_incidents", ax=axes[0], palette="Reds_r")
    axes[0].set_title("Top Causes of Murder in Indian Metros (2024)", fontweight="bold")
    axes[0].set_xlabel("Reported Incidents")
    axes[0].set_ylabel("")

    top_c = cyber_motives.head(6)
    sns.barplot(data=top_c, y="motive", x="total_incidents", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Top Cybercrime Motives in Indian Metros (2024)", fontweight="bold")
    axes[1].set_xlabel("Reported Incidents")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(figures_dir / "top_murder_and_cyber_motives.png")
    plt.close()
    print("  Saved: top_murder_and_cyber_motives.png")

    print("\n" + "=" * 80)
    print("MODELING PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_modeling_pipeline()
