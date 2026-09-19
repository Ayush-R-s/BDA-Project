"""
Metropolitan Policy Quadrants Page — City policing efficiency scatter, motive breakdowns.
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
    DANGER,
    GRID_COLOR,
    PRIMARY,
    PRIMARY_MUTED,
    QUADRANT_COLOR_MAP,
    QUADRANT_COLORS,
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


def render(
    df_metro: pd.DataFrame,
    df_murder_motives: pd.DataFrame,
    df_cyber_motives: pd.DataFrame,
):
    """Render the Metropolitan Policy Quadrants page."""

    # ── Page Header ──
    st.markdown(
        f"""
        <div style="padding: 10px 0 24px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:24px;">🏙️</span>
                <h1 style="font-size:30px; font-weight:800; color:{TEXT_PRIMARY}; margin:0; letter-spacing:-0.5px;">
                Metropolitan Policing Efficiency & Policy Quadrants</h1>
            </div>
            <p style="color:{TEXT_SECONDARY}; font-size:14px; margin:0; line-height:1.5;">
            Operational capacity evaluation across 37 million-plus urban agglomerations · NCRB 2024
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards (Glassmorphic) ──
    total_cities = len(df_metro)
    avg_crime_rate = df_metro["crime_rate_per_lakh"].mean()
    avg_chargesheet = df_metro["chargesheeting_rate"].mean()
    avg_cagr = df_metro["ipc_cagr_2022_2024"].mean()

    kpi_cols = st.columns(4)
    metro_kpis = [
        ("Monitored Metros", str(total_cities), PRIMARY),
        ("Mean Crime Rate", f"{avg_crime_rate:.1f} / lakh", SECONDARY),
        ("Mean Chargesheet Rate", f"{avg_chargesheet:.1f}%", SUCCESS),
        ("3-Year IPC CAGR", f"{avg_cagr:+.1f}%", WARNING),
    ]
    for col, (label, value, color) in zip(kpi_cols, metro_kpis):
        col.markdown(make_kpi_html(label, value, color=color), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Quadrant Scatter Plot ──
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        🎯 Metropolitan Policing Quadrant Matrix
        </h3>
        """,
        unsafe_allow_html=True,
    )

    colors = []
    for _, r in df_metro.iterrows():
        q = r["policing_quadrant"]
        matched = False
        for key, color in QUADRANT_COLOR_MAP.items():
            if key[:2] in q[:2]:
                colors.append(color)
                matched = True
                break
        if not matched:
            colors.append(PRIMARY)

    med_crime = df_metro["med_crime_rate"].iloc[0] if "med_crime_rate" in df_metro.columns else df_metro["crime_rate_per_lakh"].median()
    med_cs = df_metro["med_chargesheet_rate"].iloc[0] if "med_chargesheet_rate" in df_metro.columns else df_metro["chargesheeting_rate"].median()

    fig_quad = go.Figure()

    x_min, x_max = 0, df_metro["crime_rate_per_lakh"].max() * 1.15
    y_min, y_max = 0, min(df_metro["chargesheeting_rate"].max() * 1.15, 105)

    quad_labels = [
        (x_min + (med_crime - x_min) / 2, y_max - 5, "Q3: Effective Containment", "#98C379"),
        (med_crime + (x_max - med_crime) / 2, y_max - 5, "Q2: Active Enforcement", "#E5C07B"),
        (x_min + (med_crime - x_min) / 2, y_min + 5, "Q4: Latent Risk / Low Reporting", "#64748B"),
        (med_crime + (x_max - med_crime) / 2, y_min + 5, "Q1: Critical Bottleneck", "#E06C75"),
    ]

    for qx, qy, qlabel, qcolor in quad_labels:
        fig_quad.add_annotation(
            x=qx, y=qy, text=qlabel,
            showarrow=False,
            font=dict(size=11, color=qcolor, family="Inter, sans-serif"),
            opacity=0.7,
        )

    fig_quad.add_trace(go.Scatter(
        x=df_metro["crime_rate_per_lakh"],
        y=df_metro["chargesheeting_rate"],
        mode="markers+text",
        text=df_metro["city"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_SECONDARY),
        marker=dict(
            size=df_metro["population_lakhs"].clip(5, 60) * 1.1,
            color=colors,
            opacity=0.85,
            line=dict(width=1, color="#0B0F17"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Crime Rate: %{x:.1f} per lakh<br>"
            "Chargesheeting: %{y:.1f}%<br>"
            "Population: %{customdata:.1f} lakh<extra></extra>"
        ),
        customdata=df_metro["population_lakhs"],
    ))

    fig_quad.add_hline(y=med_cs, line_dash="dash", line_color=TEXT_MUTED, line_width=1, opacity=0.4)
    fig_quad.add_vline(x=med_crime, line_dash="dash", line_color=TEXT_MUTED, line_width=1, opacity=0.4)

    style_figure(
        fig_quad,
        title="",
        height=520,
        xaxis=dict(title="Crime Incidence Rate (per Lakh Population)", range=[x_min, x_max], showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="Chargesheeting Rate (%)", range=[y_min, y_max], showgrid=True, gridcolor=GRID_COLOR),
        showlegend=False,
    )
    st.plotly_chart(fig_quad, use_container_width=True)

    # ── Quadrant Cards (Frosted Glass) ──
    quad_counts = df_metro["policing_quadrant"].value_counts()
    legend_cols = st.columns(4)
    quad_items = [
        ("Q1: Critical Bottleneck", "#E06C75", "High Crime, Low Chargesheet"),
        ("Q2: Active Enforcement", "#E5C07B", "High Crime, High Chargesheet"),
        ("Q3: Effective Containment", "#98C379", "Low Crime, High Chargesheet"),
        ("Q4: Latent Risk", "#64748B", "Low Crime, Low Chargesheet"),
    ]
    for col, (qname, qcolor, qdesc) in zip(legend_cols, quad_items):
        count = sum(1 for q in quad_counts.index if qname[:2] in q[:2])
        col.markdown(
            f"""
            <div style="
                background: {BG_GLASS};
                backdrop-filter: blur(12px);
                border: 1px solid {BORDER_GLASS};
                border-radius: 12px;
                padding: 14px 16px;
                text-align: center;
                box-shadow: {SHADOW_GLASS};
            ">
                <div style="font-size:22px; font-weight:700; color:{qcolor};">{count}</div>
                <div style="font-size:11px; font-weight:600; color:{TEXT_PRIMARY}; margin-top:2px;">{qname}</div>
                <div style="font-size:11px; color:{TEXT_MUTED}; margin-top:3px;">{qdesc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Motive Breakdowns ──
    col_murder, col_cyber = st.columns(2)

    with col_murder:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            🔪 Primary Homicide Motives
            </h3>
            """,
            unsafe_allow_html=True,
        )
        df_m = df_murder_motives.copy()
        motive_col = df_m.columns[0]
        incident_col = df_m.columns[1]
        df_m = df_m[df_m[incident_col] > 0]
        df_m = df_m[~df_m[motive_col].str.contains("City", case=False, na=False)]
        df_m = df_m.sort_values(incident_col, ascending=True).tail(10)

        fig_murder = go.Figure(data=[
            go.Bar(
                y=df_m[motive_col],
                x=df_m[incident_col],
                orientation="h",
                marker=dict(
                    color=df_m[incident_col],
                    colorscale=[[0.0, "#1E3A8A"], [1.0, "#E06C75"]],
                    cornerradius=4,
                ),
                text=[f"{int(v):,}" for v in df_m[incident_col]],
                textposition="outside",
                textfont=dict(size=10, color=TEXT_MUTED),
                hovertemplate="<b>%{y}</b><br>Incidents: %{x:,.0f}<extra></extra>",
            )
        ])
        style_figure(fig_murder, title="", height=380, xaxis=dict(title="Total Incidents", showgrid=True, gridcolor=GRID_COLOR), yaxis=dict(title=""), showlegend=False)
        st.plotly_chart(fig_murder, use_container_width=True)

    with col_cyber:
        st.markdown(
            f"""
            <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
            💻 Primary Cybercrime Motives
            </h3>
            """,
            unsafe_allow_html=True,
        )
        df_c = df_cyber_motives.copy()
        motive_col_c = df_c.columns[0]
        incident_col_c = df_c.columns[1]
        df_c = df_c[df_c[incident_col_c] > 0]
        df_c = df_c[~df_c[motive_col_c].str.contains("City", case=False, na=False)]
        df_c = df_c.sort_values(incident_col_c, ascending=True).tail(10)

        fig_cyber = go.Figure(data=[
            go.Bar(
                y=df_c[motive_col_c],
                x=df_c[incident_col_c],
                orientation="h",
                marker=dict(
                    color=df_c[incident_col_c],
                    colorscale=[[0.0, "#1E293B"], [1.0, "#56B6C2"]],
                    cornerradius=4,
                ),
                text=[f"{int(v):,}" for v in df_c[incident_col_c]],
                textposition="outside",
                textfont=dict(size=10, color=TEXT_MUTED),
                hovertemplate="<b>%{y}</b><br>Incidents: %{x:,.0f}<extra></extra>",
            )
        ])
        style_figure(fig_cyber, title="", height=380, xaxis=dict(title="Total Incidents", showgrid=True, gridcolor=GRID_COLOR), yaxis=dict(title=""), showlegend=False)
        st.plotly_chart(fig_cyber, use_container_width=True)

    # ── Row 3: City Data Table ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:600; margin-bottom:8px;">
        📋 Metropolitan Performance Register
        </h3>
        """,
        unsafe_allow_html=True,
    )

    display_cols = [
        "city", "crime_rate_per_lakh", "chargesheeting_rate",
        "ipc_cagr_2022_2024", "population_lakhs", "policing_quadrant",
    ]
    available_display = [c for c in display_cols if c in df_metro.columns]
    df_display = df_metro[available_display].copy()
    df_display = df_display.sort_values("crime_rate_per_lakh", ascending=False)

    rename_map = {
        "city": "City",
        "crime_rate_per_lakh": "Crime Rate (/Lakh)",
        "chargesheeting_rate": "Chargesheet %",
        "ipc_cagr_2022_2024": "3Y CAGR %",
        "population_lakhs": "Pop. (Lakh)",
        "policing_quadrant": "Quadrant",
    }
    df_display = df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns})

    st.dataframe(df_display, use_container_width=True, hide_index=True)
