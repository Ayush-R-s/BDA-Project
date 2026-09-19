"""
District Crime Explorer Page — Drill into any individual district's crime profile.
Refactored with glassmorphism and calm, non-neon aesthetics.
"""

import numpy as np
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

# Radar dimensions for district profiling
RADAR_FEATURES = [
    ("violent_crime_ratio", "Violent Crime"),
    ("property_crime_ratio", "Property Crime"),
    ("women_vulnerability_share", "Women Vulnerability"),
    ("child_vulnerability_share", "Child Vulnerability"),
    ("caste_atrocity_share", "Caste Atrocity"),
    ("juvenile_delinquency_ratio", "Juvenile Delinquency"),
    ("cyber_crime_intensity_per_1k", "Cyber Intensity"),
    ("trafficking_vulnerability_proxy", "Trafficking Proxy"),
]


def render(df_clusters: pd.DataFrame):
    """Render the District Crime Explorer page."""

    # ── Page Header ──
    st.markdown(
        f"""
        <div style="padding: 10px 0 24px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:24px;">📍</span>
                <h1 style="font-size:30px; font-weight:800; color:{TEXT_PRIMARY}; margin:0; letter-spacing:-0.5px;">
                District Crime Profile Explorer</h1>
            </div>
            <p style="color:{TEXT_SECONDARY}; font-size:14px; margin:0; line-height:1.5;">
            Individual administrative district diagnostics, vulnerability radar footprints, and cohort comparisons
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Filters ──
    col_state, col_district = st.columns(2)
    with col_state:
        states = sorted(df_clusters["state"].unique())
        selected_state = st.selectbox("Select State / UT", states, index=states.index("Delhi") if "Delhi" in states else 0)

    districts_in_state = sorted(df_clusters[df_clusters["state"] == selected_state]["district"].unique())
    with col_district:
        selected_district = st.selectbox("Select District", districts_in_state)

    # ── Get district data ──
    row = df_clusters[(df_clusters["state"] == selected_state) & (df_clusters["district"] == selected_district)]
    if row.empty:
        st.error("No data found for this district.")
        return
    row = row.iloc[0]

    cluster_id = int(row["cluster_id"])
    typology = row["typology_label"]
    is_anomaly = bool(row["is_anomaly"])
    cluster_color = CLUSTER_COLORS.get(cluster_id, PRIMARY)

    # ── District Identity Card (Glassmorphic) ──
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    anomaly_badge = (
        f'<span style="background:rgba(239,68,68,0.15);color:{DANGER};border:1px solid rgba(239,68,68,0.3);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">⚠ STATISTICAL OUTLIER</span>'
        if is_anomaly
        else f'<span style="background:rgba(16,185,129,0.12);color:{SUCCESS};border:1px solid rgba(16,185,129,0.25);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">✓ STANDARD PROFILE</span>'
    )

    st.markdown(
        f"""
        <div style="
            background: {BG_GLASS};
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid {BORDER_GLASS};
            border-radius: 16px;
            padding: 22px 28px;
            margin-bottom: 20px;
            box-shadow: {SHADOW_GLASS};
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                <div>
                    <div style="display:flex; align-items:baseline; gap:12px;">
                        <h2 style="margin:0; font-size:26px; font-weight:700; color:{TEXT_PRIMARY}; letter-spacing:-0.4px;">{selected_district}</h2>
                        <span style="color:{TEXT_MUTED}; font-size:14px; font-weight:500;">{selected_state}</span>
                    </div>
                    <div style="color:{TEXT_SECONDARY}; font-size:13px; margin-top:6px;">
                        Assigned Cohort: <span style="color:{cluster_color}; font-weight:600;">Cluster {cluster_id} — {typology}</span>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    {anomaly_badge}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards (Glassmorphic) ──
    kpi_cols = st.columns(5)
    district_kpis = [
        ("Total Crime Burden", f"{int(row['total_crime_burden']):,}", PRIMARY),
        ("IPC Crimes", f"{int(row['total_ipc_crimes']):,}", PRIMARY_MUTED),
        ("Crimes Against Women", f"{int(row['crimes_against_women_total']):,}", WARNING),
        ("Cybercrimes Recorded", f"{int(row['cybercrimes_total']):,}", SECONDARY),
        ("Missing Persons", f"{int(row['missing_persons_total']):,}", DANGER),
    ]
    for col, (label, value, color) in zip(kpi_cols, district_kpis):
        col.markdown(make_kpi_html(label, value, color=color), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Radar Chart + Comparison Table ──
    col_radar, col_table = st.columns([1.2, 1])

    with col_radar:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            🕸️ Multi-Dimensional Vulnerability Footprint
            </h3>
            """,
            unsafe_allow_html=True,
        )

        cluster_data = df_clusters[df_clusters["cluster_id"] == cluster_id]
        national_median = df_clusters[[f[0] for f in RADAR_FEATURES]].median()

        feature_cols = [f[0] for f in RADAR_FEATURES]
        feature_labels = [f[1] for f in RADAR_FEATURES]

        national_max = df_clusters[feature_cols].quantile(0.95)
        national_max = national_max.replace(0, 1)

        district_vals = [min(row[c] / national_max[c], 1.5) for c in feature_cols]
        cluster_means = [min(cluster_data[c].mean() / national_max[c], 1.5) for c in feature_cols]
        national_meds = [min(national_median[c] / national_max[c], 1.5) for c in feature_cols]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=district_vals + [district_vals[0]],
            theta=feature_labels + [feature_labels[0]],
            fill="toself",
            fillcolor=f"rgba(59, 130, 246, 0.18)",
            line=dict(color=PRIMARY, width=2),
            name=f"{selected_district}",
            hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_means + [cluster_means[0]],
            theta=feature_labels + [feature_labels[0]],
            fill="toself",
            fillcolor=f"rgba(229, 192, 123, 0.10)",
            line=dict(color="#E5C07B", width=1.5, dash="dash"),
            name=f"Cluster {cluster_id} Avg",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=national_meds + [national_meds[0]],
            theta=feature_labels + [feature_labels[0]],
            line=dict(color=TEXT_MUTED, width=1, dash="dot"),
            name="National Median",
        ))

        style_figure(
            fig_radar,
            title="",
            height=420,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor=GRID_COLOR, tickfont=dict(size=9, color=TEXT_MUTED)),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10, color=TEXT_SECONDARY)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_table:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            📊 Cohort Benchmarking Breakdown
            </h3>
            """,
            unsafe_allow_html=True,
        )

        comparison_data = []
        for col_name, label in RADAR_FEATURES:
            comparison_data.append({
                "Indicator": label,
                f"{selected_district}": f"{row[col_name]:.4f}",
                f"Cluster {cluster_id} Avg": f"{cluster_data[col_name].mean():.4f}",
                "National Median": f"{national_median[col_name]:.4f}",
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True, height=210)

        st.markdown(
            f"""
            <h4 style="color:{TEXT_PRIMARY}; font-size:14px; font-weight:600; margin: 16px 0 8px 0;">
            📋 Absolute Head Counts
            </h4>
            """,
            unsafe_allow_html=True,
        )
        abs_data = {
            "Category": [
                "Violent Crimes", "Property Crimes", "SC/ST Atrocities",
                "Juvenile IPC Crimes", "NDPS / Narcotics", "Arms Act Cases",
                "Missing Children",
            ],
            "Count": [
                f"{int(row['violent_crime_total']):,}",
                f"{int(row['property_crime_total']):,}",
                f"{int(row['sc_st_atrocities_total']):,}",
                f"{int(row['juvenile_ipc_crimes_total']):,}",
                f"{int(row['ndps_narcotics_cases']):,}",
                f"{int(row['arms_act_cases']):,}",
                f"{int(row['missing_children_total']):,}",
            ],
        }
        st.dataframe(pd.DataFrame(abs_data), use_container_width=True, hide_index=True, height=180)

    # ── Row 3: Cohort Peers ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        Cohort Peers in Cluster {cluster_id} ({typology})
        </h3>
        """,
        unsafe_allow_html=True,
    )
    same_cluster = (
        cluster_data[["state", "district", "total_crime_burden", "violent_crime_ratio", "women_vulnerability_share", "is_anomaly"]]
        .sort_values("total_crime_burden", ascending=False)
        .head(15)
    )
    same_cluster = same_cluster.rename(columns={
        "state": "State",
        "district": "District",
        "total_crime_burden": "Total Crime",
        "violent_crime_ratio": "Violent Ratio",
        "women_vulnerability_share": "Women Vuln.",
        "is_anomaly": "Anomaly",
    })
    st.dataframe(same_cluster, use_container_width=True, hide_index=True)
