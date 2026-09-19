"""
Anomaly & Deep Dive Page — DBSCAN anomaly analysis, filterable tables, and CSV export.
Refactored with glassmorphism and calm, non-neon aesthetics.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import (
    BG_CARD,
    BG_DARK,
    BG_GLASS,
    BORDER_GLASS,
    CLUSTER_COLORS,
    DANGER,
    GRID_COLOR,
    PRIMARY,
    PRIMARY_MUTED,
    SECONDARY,
    SHADOW_GLASS,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    make_kpi_html,
    style_figure,
)


def render(df_clusters: pd.DataFrame):
    """Render the Anomaly & Deep Dive page."""

    # ── Page Header ──
    st.markdown(
        f"""
        <div style="padding: 10px 0 24px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:24px;">🔍</span>
                <h1 style="font-size:30px; font-weight:800; color:{TEXT_PRIMARY}; margin:0; letter-spacing:-0.5px;">
                Outlier & Anomaly Diagnostics</h1>
            </div>
            <p style="color:{TEXT_SECONDARY}; font-size:14px; margin:0; line-height:1.5;">
            DBSCAN density-based anomaly discovery across spatial crime feature distributions · NCRB 2024
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_anomaly = df_clusters[df_clusters["is_anomaly"]].copy()
    df_normal = df_clusters[~df_clusters["is_anomaly"]]

    # ── KPI Cards (Glassmorphic) ──
    kpi_cols = st.columns(5)
    anomaly_kpis = [
        ("Outlier Districts", str(len(df_anomaly)), DANGER),
        ("Standard Cohort", str(len(df_normal)), SUCCESS),
        ("Affected States", str(df_anomaly["state"].nunique()), SECONDARY),
        ("Outlier Avg Crime", f"{df_anomaly['total_crime_burden'].mean():,.0f}" if not df_anomaly.empty else "0", PRIMARY_MUTED),
        ("Cohort Avg Crime", f"{df_normal['total_crime_burden'].mean():,.0f}", PRIMARY),
    ]
    for col, (label, value, color) in zip(kpi_cols, anomaly_kpis):
        col.markdown(make_kpi_html(label, value, color=color), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Row 1: Anomaly vs Normal Comparison ──
    col_chart, col_statewise = st.columns([1, 1])

    with col_chart:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            ⚡ Outlier Footprint vs Standard Cohort Mean
            </h3>
            """,
            unsafe_allow_html=True,
        )

        compare_cols = [
            ("total_crime_burden", "Total Crime"),
            ("violent_crime_ratio", "Violent Ratio"),
            ("women_vulnerability_share", "Women Vuln."),
            ("cyber_crime_intensity_per_1k", "Cyber Intensity"),
            ("trafficking_vulnerability_proxy", "Trafficking"),
        ]

        categories = [c[1] for c in compare_cols]
        anomaly_means = [df_anomaly[c[0]].mean() if not df_anomaly.empty else 0 for c in compare_cols]
        normal_means = [df_normal[c[0]].mean() for c in compare_cols]

        max_vals = [max(a, n, 1) for a, n in zip(anomaly_means, normal_means)]
        anomaly_norm = [a / m for a, m in zip(anomaly_means, max_vals)]
        normal_norm = [n / m for n, m in zip(normal_means, max_vals)]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=categories, y=anomaly_norm,
            name="Outlier Districts",
            marker_color=DANGER,
            text=[f"{v:.2f}" for v in anomaly_means],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_MUTED),
            hovertemplate="<b>%{x}</b><br>Outlier Mean: %{text}<extra></extra>",
        ))
        fig_compare.add_trace(go.Bar(
            x=categories, y=normal_norm,
            name="Standard Cohort",
            marker_color=PRIMARY,
            text=[f"{v:.2f}" for v in normal_means],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_MUTED),
            hovertemplate="<b>%{x}</b><br>Standard Mean: %{text}<extra></extra>",
        ))
        style_figure(
            fig_compare,
            title="",
            height=380,
            barmode="group",
            xaxis=dict(title="", tickfont=dict(size=11), showgrid=False),
            yaxis=dict(title="Normalized Value", range=[0, 1.25], showgrid=True, gridcolor=GRID_COLOR),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=10, color=TEXT_SECONDARY),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with col_statewise:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            🗺️ Geographic Concentration of Outliers
            </h3>
            """,
            unsafe_allow_html=True,
        )

        if not df_anomaly.empty:
            state_anomalies = df_anomaly["state"].value_counts().sort_values(ascending=True).reset_index()
            state_anomalies.columns = ["State", "Anomaly Count"]

            fig_state = go.Figure(data=[
                go.Bar(
                    y=state_anomalies["State"],
                    x=state_anomalies["Anomaly Count"],
                    orientation="h",
                    marker=dict(
                        color=state_anomalies["Anomaly Count"],
                        colorscale=[[0.0, "#1E3A8A"], [1.0, "#E06C75"]],
                        cornerradius=4,
                    ),
                    text=state_anomalies["Anomaly Count"].astype(str),
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_MUTED),
                    hovertemplate="<b>%{y}</b><br>Outliers: %{x}<extra></extra>",
                )
            ])
            style_figure(
                fig_state,
                title="",
                height=380,
                xaxis=dict(title="Number of Outlier Districts", showgrid=True, gridcolor=GRID_COLOR),
                yaxis=dict(title=""),
                showlegend=False,
            )
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            st.info("No anomaly districts detected by DBSCAN.")

    # ── Row 2: Filterable Anomaly Table ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        📋 Outlier Registry & Feature Matrix
        </h3>
        """,
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([1, 1, 1.2])
    with filter_cols[0]:
        if not df_anomaly.empty:
            states_with_anomalies = sorted(df_anomaly["state"].unique())
            state_filter = st.multiselect("Filter by State", states_with_anomalies, default=[])
        else:
            state_filter = []

    with filter_cols[1]:
        if not df_anomaly.empty:
            typologies = sorted(df_anomaly["typology_label"].unique())
            typology_filter = st.multiselect("Filter by Typology", typologies, default=[])
        else:
            typology_filter = []

    df_filtered = df_anomaly.copy()
    if state_filter:
        df_filtered = df_filtered[df_filtered["state"].isin(state_filter)]
    if typology_filter:
        df_filtered = df_filtered[df_filtered["typology_label"].isin(typology_filter)]

    table_cols = [
        "state", "district", "typology_label", "total_crime_burden",
        "violent_crime_ratio", "women_vulnerability_share",
        "cyber_crime_intensity_per_1k", "trafficking_vulnerability_proxy",
    ]
    available_table_cols = [c for c in table_cols if c in df_filtered.columns]
    df_table = df_filtered[available_table_cols].sort_values("total_crime_burden", ascending=False)

    rename_map = {
        "state": "State",
        "district": "District",
        "typology_label": "Typology",
        "total_crime_burden": "Total Crime",
        "violent_crime_ratio": "Violent Ratio",
        "women_vulnerability_share": "Women Vuln.",
        "cyber_crime_intensity_per_1k": "Cyber Intensity",
        "trafficking_vulnerability_proxy": "Trafficking Proxy",
    }
    df_table = df_table.rename(columns={k: v for k, v in rename_map.items() if k in df_table.columns})

    st.dataframe(df_table, use_container_width=True, hide_index=True)

    with filter_cols[2]:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Outlier Register (CSV)",
            data=csv_data,
            file_name="ncrb_2024_anomaly_districts.csv",
            mime="text/csv",
        )
