"""
National Overview Page — High-level KPIs, typology distribution, and top crime states.
Refactored with glassmorphism and calm, non-neon aesthetics.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import (
    BG_CARD,
    BG_DARK,
    BORDER_GLASS,
    CLUSTER_COLOR_LIST,
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


def render(df_clusters: pd.DataFrame, df_metro: pd.DataFrame):
    """Render the National Overview page."""

    # ── Page Header (Clean, refined, non-neon) ──
    st.markdown(
        f"""
        <div style="padding: 10px 0 26px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:24px;">🇮🇳</span>
                <h1 style="font-size:30px; font-weight:800; color:{TEXT_PRIMARY}; margin:0; letter-spacing:-0.5px;">
                National Crime Landscape & Spatial Typologies</h1>
            </div>
            <p style="color:{TEXT_SECONDARY}; font-size:14px; margin:0; line-height:1.5;">
            High-level policy analytics across 964 administrative districts and 37 metropolitan urban centers · NCRB 2024
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards (Frosted Glass) ──
    total_districts = len(df_clusters)
    total_states = df_clusters["state"].nunique()
    total_crimes = int(df_clusters["total_crime_burden"].sum())
    total_ipc = int(df_clusters["total_ipc_crimes"].sum())
    anomaly_count = int(df_clusters["is_anomaly"].sum())
    metro_cities = len(df_metro)

    cols = st.columns(6)
    kpis = [
        ("Districts Analyzed", f"{total_districts:,}", "", PRIMARY),
        ("States & UTs", str(total_states), "", SECONDARY),
        ("Total Crime Burden", f"{total_crimes:,.0f}", "", PRIMARY_MUTED),
        ("Total IPC Crimes", f"{total_ipc:,.0f}", "", SECONDARY),
        ("Metro Centers", str(metro_cities), "", SUCCESS),
        ("Statistical Outliers", str(anomaly_count), "", DANGER),
    ]
    for col, (label, value, delta, color) in zip(cols, kpis):
        col.markdown(make_kpi_html(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Typology Distribution + Top States ──
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin:0;">
                National Crime Typology Share
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        typology_counts = df_clusters["typology_label"].value_counts().reset_index()
        typology_counts.columns = ["Typology", "Count"]

        # Assign colors based on cluster order
        typology_color_map = {}
        cluster_typology = df_clusters.groupby("cluster_id")["typology_label"].first().to_dict()
        for cid, tname in cluster_typology.items():
            typology_color_map[tname] = CLUSTER_COLORS.get(cid, PRIMARY)

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=typology_counts["Typology"],
                    values=typology_counts["Count"],
                    hole=0.62,
                    marker=dict(
                        colors=[typology_color_map.get(t, PRIMARY) for t in typology_counts["Typology"]],
                        line=dict(color="#0B0F17", width=2),
                    ),
                    textinfo="percent",
                    textfont=dict(size=12, color=TEXT_PRIMARY),
                    hovertemplate="<b>%{label}</b><br>Districts: %{value}<br>Share: %{percent}<extra></extra>",
                )
            ]
        )
        style_figure(
            fig_donut,
            title="",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(size=10, color=TEXT_SECONDARY),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        # Center annotation with clean typography
        fig_donut.add_annotation(
            text=f"<b style='font-size:26px;'>{total_districts}</b><br><span style='font-size:11px;color:{TEXT_MUTED}'>DISTRICTS</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin:0;">
                Top 15 States by Cumulative Crime Volume
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        state_crimes = (
            df_clusters.groupby("state")["total_crime_burden"]
            .sum()
            .sort_values(ascending=True)
            .tail(15)
            .reset_index()
        )
        state_crimes.columns = ["State", "Total Crimes"]

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    y=state_crimes["State"],
                    x=state_crimes["Total Crimes"],
                    orientation="h",
                    marker=dict(
                        color=state_crimes["Total Crimes"],
                        colorscale=[[0.0, "#1E3A8A"], [0.5, "#2563EB"], [1.0, "#60A5FA"]],
                        line=dict(width=0),
                    ),
                    text=[f"{v:,.0f}" for v in state_crimes["Total Crimes"]],
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_MUTED),
                    hovertemplate="<b>%{y}</b><br>Total Crimes: %{x:,.0f}<extra></extra>",
                )
            ]
        )
        style_figure(
            fig_bar,
            title="",
            height=400,
            xaxis=dict(title="Total IPC + SLL Incidents", showgrid=True, gridcolor=GRID_COLOR),
            yaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 3: Typology Composition by State (Stacked Bar) ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        State-wise Typology Distribution (Top 20 States by District Count)
        </h3>
        """,
        unsafe_allow_html=True,
    )

    top_states = df_clusters["state"].value_counts().head(20).index.tolist()
    df_top = df_clusters[df_clusters["state"].isin(top_states)]
    state_typology = df_top.groupby(["state", "typology_label"]).size().reset_index(name="count")

    fig_stacked = go.Figure()
    for typology in typology_counts["Typology"]:
        subset = state_typology[state_typology["typology_label"] == typology]
        fig_stacked.add_trace(
            go.Bar(
                x=subset["state"],
                y=subset["count"],
                name=typology[:35] + "…" if len(typology) > 35 else typology,
                marker_color=typology_color_map.get(typology, PRIMARY),
                hovertemplate="<b>%{x}</b><br>%{fullData.name}<br>Districts: %{y}<extra></extra>",
            )
        )

    style_figure(
        fig_stacked,
        title="",
        height=420,
        barmode="stack",
        xaxis=dict(title="", tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="Number of Districts", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # ── Row 4: Typology Summary Statistics ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        Cluster Profile Summary
        </h3>
        """,
        unsafe_allow_html=True,
    )

    summary_stats = df_clusters.groupby("typology_label").agg(
        districts=("district", "count"),
        avg_total_crime=("total_crime_burden", "mean"),
        avg_violent_ratio=("violent_crime_ratio", "mean"),
        avg_women_share=("women_vulnerability_share", "mean"),
        avg_cyber_intensity=("cyber_crime_intensity_per_1k", "mean"),
        anomalies=("is_anomaly", "sum"),
    ).reset_index()

    summary_stats.columns = [
        "Typology", "Districts", "Avg Crime Burden", "Avg Violent Ratio",
        "Avg Women Vuln. Share", "Avg Cyber Intensity", "Anomalies"
    ]
    summary_stats["Avg Crime Burden"] = summary_stats["Avg Crime Burden"].map(lambda x: f"{x:,.0f}")
    summary_stats["Avg Violent Ratio"] = summary_stats["Avg Violent Ratio"].map(lambda x: f"{x:.4f}")
    summary_stats["Avg Women Vuln. Share"] = summary_stats["Avg Women Vuln. Share"].map(lambda x: f"{x:.4f}")
    summary_stats["Avg Cyber Intensity"] = summary_stats["Avg Cyber Intensity"].map(lambda x: f"{x:.1f}")
    summary_stats["Anomalies"] = summary_stats["Anomalies"].astype(int)

    st.dataframe(
        summary_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Typology": st.column_config.TextColumn("Crime Typology", width="large"),
        },
    )
