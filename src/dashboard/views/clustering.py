"""
Spatial Clustering Analytics Page — PCA biplot, Elbow/Silhouette curves, and centroid profiles.
Refactored with glassmorphism and calm, non-neon aesthetics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.dashboard.theme import (
    BG_CARD,
    BG_DARK,
    BORDER_GLASS,
    CLUSTER_COLORS,
    DANGER,
    GRID_COLOR,
    PRIMARY,
    PRIMARY_MUTED,
    SECONDARY,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    make_kpi_html,
    style_figure,
)


def render(
    df_clusters: pd.DataFrame,
    df_profiles: pd.DataFrame,
    df_pca_loadings: pd.DataFrame,
    df_kmeans_metrics: pd.DataFrame,
):
    """Render the Spatial Clustering Analytics page."""

    # ── Page Header ──
    st.markdown(
        f"""
        <div style="padding: 10px 0 24px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:24px;">🪩</span>
                <h1 style="font-size:30px; font-weight:800; color:{TEXT_PRIMARY}; margin:0; letter-spacing:-0.5px;">
                Spatial Crime Clustering & Latent Structure</h1>
            </div>
            <p style="color:{TEXT_SECONDARY}; font-size:14px; margin:0; line-height:1.5;">
            Unsupervised machine learning diagnostics: PCA projection, K-Means convergence, and typological centroids
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Row 1: PCA Biplot ──
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        🔬 Latent Projection Space — 964 Districts Projected onto PC1 × PC2
        </h3>
        """,
        unsafe_allow_html=True,
    )

    feature_cols = [
        "violent_crime_ratio", "property_crime_ratio", "women_vulnerability_share",
        "child_vulnerability_share", "caste_atrocity_share", "juvenile_delinquency_ratio",
        "cyber_crime_intensity_per_1k", "trafficking_vulnerability_proxy",
    ]

    X = df_clusters[feature_cols].copy()
    X["log_total_crime"] = np.log1p(df_clusters["total_crime_burden"])
    X["log_contraband"] = np.log1p(df_clusters["contraband_enforcement_total"])

    for col in feature_cols:
        p98 = X[col].quantile(0.98)
        X[col] = np.clip(X[col], 0, p98)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_pca_plot = df_clusters[["state", "district", "cluster_id", "typology_label", "is_anomaly", "total_crime_burden"]].copy()
    df_pca_plot["PC1"] = X_pca[:, 0]
    df_pca_plot["PC2"] = X_pca[:, 1]

    cluster_names = df_clusters.groupby("cluster_id")["typology_label"].first().to_dict()

    fig_biplot = go.Figure()
    for cid in sorted(df_pca_plot["cluster_id"].unique()):
        subset = df_pca_plot[df_pca_plot["cluster_id"] == cid]
        fig_biplot.add_trace(go.Scatter(
            x=subset["PC1"],
            y=subset["PC2"],
            mode="markers",
            name=f"C{cid}: {cluster_names.get(cid, '')[:32]}",
            marker=dict(
                color=CLUSTER_COLORS.get(cid, PRIMARY),
                size=6,
                opacity=0.75,
                line=dict(width=0.5, color="#0B0F17"),
            ),
            text=subset.apply(
                lambda r: f"{r['district']}, {r['state']}<br>Cluster: {r['typology_label']}<br>Crime Burden: {int(r['total_crime_burden']):,}",
                axis=1,
            ),
            hoverinfo="text",
        ))

    # DBSCAN Anomalies ring
    anomalies = df_pca_plot[df_pca_plot["is_anomaly"]]
    if not anomalies.empty:
        fig_biplot.add_trace(go.Scatter(
            x=anomalies["PC1"],
            y=anomalies["PC2"],
            mode="markers",
            name="⚠ DBSCAN Anomaly",
            marker=dict(
                color="rgba(0,0,0,0)",
                size=12,
                line=dict(width=1.8, color=DANGER),
                symbol="circle-open",
            ),
            hoverinfo="skip",
        ))

    ev = pca.explained_variance_ratio_
    style_figure(
        fig_biplot,
        title="",
        height=500,
        xaxis=dict(title=f"Principal Component 1 ({ev[0]*100:.1f}% Variance)", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title=f"Principal Component 2 ({ev[1]*100:.1f}% Variance)", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.24,
            xanchor="center", x=0.5, font=dict(size=10, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig_biplot, use_container_width=True)

    # ── KPI Cards for PCA ──
    pca_cols = st.columns(4)
    pca_kpis = [
        ("PC1 Variance Explained", f"{ev[0]*100:.1f}%", PRIMARY),
        ("PC2 Variance Explained", f"{ev[1]*100:.1f}%", PRIMARY_MUTED),
        ("PC3 Variance Explained", f"{ev[2]*100:.1f}%", SECONDARY),
        ("Cumulative (4 PCs)", f"{sum(ev)*100:.1f}%", SUCCESS),
    ]
    for col, (label, value, color) in zip(pca_cols, pca_kpis):
        col.markdown(make_kpi_html(label, value, color=color), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Elbow + Silhouette ──
    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            📉 Elbow Optimization Curve (SSE Inertia)
            </h3>
            """,
            unsafe_allow_html=True,
        )
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=df_kmeans_metrics["k"],
            y=df_kmeans_metrics["inertia"],
            mode="lines+markers",
            line=dict(color=PRIMARY, width=2),
            marker=dict(size=8, color=PRIMARY, line=dict(width=1.5, color=TEXT_PRIMARY)),
            hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>",
        ))
        if 5 in df_kmeans_metrics["k"].values:
            optimal = df_kmeans_metrics[df_kmeans_metrics["k"] == 5].iloc[0]
            fig_elbow.add_trace(go.Scatter(
                x=[5], y=[optimal["inertia"]],
                mode="markers",
                marker=dict(size=13, color="#E5C07B", symbol="diamond", line=dict(width=1.5, color=TEXT_PRIMARY)),
                name="Optimal k=5",
                hovertemplate="Optimal k=5<br>Inertia: %{y:,.0f}<extra></extra>",
            ))
        style_figure(
            fig_elbow,
            title="",
            height=340,
            xaxis=dict(title="Clusters (k)", dtick=1, showgrid=True, gridcolor=GRID_COLOR),
            yaxis=dict(title="Within-Cluster SSE", showgrid=True, gridcolor=GRID_COLOR),
            showlegend=False,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_sil:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            📈 Silhouette Coefficient Evaluation
            </h3>
            """,
            unsafe_allow_html=True,
        )
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=df_kmeans_metrics["k"],
            y=df_kmeans_metrics["silhouette_score"],
            mode="lines+markers",
            line=dict(color=SUCCESS, width=2),
            marker=dict(size=8, color=SUCCESS, line=dict(width=1.5, color=TEXT_PRIMARY)),
            hovertemplate="k=%{x}<br>Silhouette: %{y:.4f}<extra></extra>",
        ))
        if 5 in df_kmeans_metrics["k"].values:
            optimal = df_kmeans_metrics[df_kmeans_metrics["k"] == 5].iloc[0]
            fig_sil.add_trace(go.Scatter(
                x=[5], y=[optimal["silhouette_score"]],
                mode="markers",
                marker=dict(size=13, color="#E5C07B", symbol="diamond", line=dict(width=1.5, color=TEXT_PRIMARY)),
                name="Optimal k=5",
            ))
        style_figure(
            fig_sil,
            title="",
            height=340,
            xaxis=dict(title="Clusters (k)", dtick=1, showgrid=True, gridcolor=GRID_COLOR),
            yaxis=dict(title="Silhouette Score", showgrid=True, gridcolor=GRID_COLOR),
            showlegend=False,
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    # ── Row 3: Cluster Centroid Radar ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        🕸️ Typological Centroid Profiles (Normalized Feature Radar)
        </h3>
        """,
        unsafe_allow_html=True,
    )

    radar_cols = [
        "violent_crime_ratio", "property_crime_ratio", "women_vulnerability_share",
        "child_vulnerability_share", "caste_atrocity_share", "juvenile_delinquency_ratio",
        "cyber_crime_intensity_per_1k", "trafficking_vulnerability_proxy",
    ]
    radar_labels = [
        "Violent Crime", "Property Crime", "Women Vuln.", "Child Vuln.",
        "Caste Atrocity", "Juvenile Del.", "Cyber Intensity", "Trafficking",
    ]

    available_radar_cols = [c for c in radar_cols if c in df_profiles.columns]
    available_radar_labels = [radar_labels[i] for i, c in enumerate(radar_cols) if c in df_profiles.columns]

    profile_values = df_profiles[available_radar_cols].copy()
    maxvals = profile_values.max().replace(0, 1)
    profile_normalized = profile_values.div(maxvals)

    fig_radar = go.Figure()
    for idx, r in profile_normalized.iterrows():
        cid = int(df_profiles.loc[idx, "cluster_id"])
        vals = r.tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=available_radar_labels + [available_radar_labels[0]],
            fill="toself",
            fillcolor=f"{CLUSTER_COLORS.get(cid, PRIMARY)}18",
            line=dict(color=CLUSTER_COLORS.get(cid, PRIMARY), width=1.8),
            name=f"C{cid}: {cluster_names.get(cid, '')[:28]}",
        ))

    style_figure(
        fig_radar,
        title="",
        height=450,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor=GRID_COLOR, tickfont=dict(size=9, color=TEXT_MUTED), range=[0, 1.1]),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10, color=TEXT_SECONDARY)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5, font=dict(size=10, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Row 4: PCA Factor Loadings Heatmap ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        🧬 Principal Component Factor Loadings Matrix
        </h3>
        """,
        unsafe_allow_html=True,
    )

    loading_cols = [c for c in df_pca_loadings.columns if c.startswith("PC")]
    row_labels = df_pca_loadings.iloc[:, 0].tolist() if df_pca_loadings.columns[0] not in loading_cols else df_pca_loadings.index.tolist()

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=df_pca_loadings[loading_cols].values,
            x=loading_cols,
            y=row_labels,
            colorscale=[[0.0, "#2563EB"], [0.5, "#111726"], [1.0, "#E06C75"]],
            zmid=0,
            text=[[f"{v:.3f}" for v in row] for row in df_pca_loadings[loading_cols].values],
            texttemplate="%{text}",
            textfont=dict(size=11, color=TEXT_PRIMARY),
            hovertemplate="Feature: %{y}<br>Component: %{x}<br>Loading: %{z:.4f}<extra></extra>",
        )
    )
    style_figure(
        fig_heatmap,
        title="",
        height=420,
        xaxis=dict(title="Principal Component", side="top"),
        yaxis=dict(title="", autorange="reversed"),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
