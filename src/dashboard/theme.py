"""
Centralized Theme & Color Configuration for NCRB 2024 Crime Analytics Dashboard.

Provides a sophisticated, glassmorphic dark-theme design system with calm,
non-neon palettes, translucent cards with backdrop blurs, and transparent chart backgrounds.
"""

import plotly.graph_objects as go
import plotly.io as pio

# ─────────────────────────────────────────────
# COLOR PALETTES (Plain, Refined & Non-Neon)
# ─────────────────────────────────────────────

# Primary & Functional Accents
PRIMARY = "#3B82F6"        # Classic Slate Blue
PRIMARY_MUTED = "#60A5FA"  # Soft Blue
PRIMARY_LIGHT = "#60A5FA"  # Alias
SECONDARY = "#64748B"      # Steel Slate
ACCENT = "#64748B"         # Alias
SUCCESS = "#10B981"        # Soft Emerald
WARNING = "#F59E0B"        # Warm Ochre / Amber
DANGER = "#EF4444"         # Muted Crimson

# Background & Glass Surfaces
BG_DARK = "#0B0F17"        # Deep slate canvas
BG_CARD = "#111726"        # Fallback card background
BG_SURFACE = "#182234"     # Card surface highlight
BG_GLASS = "rgba(22, 28, 42, 0.65)"
BORDER_GLASS = "rgba(255, 255, 255, 0.08)"
SHADOW_GLASS = "0 8px 32px 0 rgba(0, 0, 0, 0.35)"

# Typography
TEXT_PRIMARY = "#F8FAFC"   # Crisp Off-White
TEXT_SECONDARY = "#94A3B8" # Muted Slate
TEXT_MUTED = "#64748B"     # Dim Slate
GRID_COLOR = "rgba(255, 255, 255, 0.05)"

# Cluster typology colors (Subdued, Harmonious & Plain)
CLUSTER_COLORS = {
    0: "#E06C75",   # Muted Terracotta — High Violent Crime & Women Vulnerability
    1: "#61AFEF",   # Steel Blue — Agrarian Belts
    2: "#56B6C2",   # Muted Cyan / Sage — Low-Intensity Stable
    3: "#E5C07B",   # Warm Sand / Amber — Major Commercial Hubs
    4: "#98C379",   # Soft Olive / Green — Specialized Cyber Wings
}

CLUSTER_COLOR_LIST = ["#E06C75", "#61AFEF", "#56B6C2", "#E5C07B", "#98C379"]

# Policing quadrant colors
QUADRANT_COLORS = {
    "Q1": "#E06C75",   # Critical Bottleneck — Muted Crimson
    "Q2": "#E5C07B",   # Active Enforcement — Warm Sand
    "Q3": "#98C379",   # High Containment — Soft Sage
    "Q4": "#64748B",   # Latent Risk — Steel Gray
}

QUADRANT_COLOR_MAP = {
    "Q1: Critical Bottleneck (High Crime, Low Chargesheet)": "#E06C75",
    "Q2: High Incident / Active Enforcement (High Crime, High Chargesheet)": "#E5C07B",
    "Q3: Effective Containment (Low Crime, High Chargesheet)": "#98C379",
    "Q4: Low Reporting / Latent Risk (Low Crime, Low Chargesheet)": "#64748B",
}

# Diverging / Sequential palette for heatmaps
HEATMAP_COLORSCALE = [
    [0.0, "#0B0F17"],
    [0.2, "#182234"],
    [0.4, "#2563EB"],
    [0.6, "#60A5FA"],
    [0.8, "#E5C07B"],
    [1.0, "#E06C75"],
]

# Gradient palette for bar charts
BAR_GRADIENT = [
    "#3B82F6", "#4F90F7", "#649FF9", "#7AAEFB", "#8FBCFD",
    "#94A3B8", "#A3B1C4", "#B2BFD0", "#C1CDDC", "#D1DCE8",
]


# ─────────────────────────────────────────────
# CHART LAYOUT HELPERS
# ─────────────────────────────────────────────

def get_base_layout(**overrides) -> dict:
    """Return a base Plotly layout dict for transparent glassmorphic styling."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=TEXT_PRIMARY,
            size=12,
        ),
        title_font=dict(size=18, color=TEXT_PRIMARY, family="Inter, sans-serif"),
        margin=dict(l=50, r=25, t=50, b=45),
        legend=dict(
            bgcolor="rgba(17, 23, 38, 0.75)",
            bordercolor=BORDER_GLASS,
            borderwidth=1,
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            title_font=dict(color=TEXT_SECONDARY, size=12),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            title_font=dict(color=TEXT_SECONDARY, size=12),
        ),
        hoverlabel=dict(
            bgcolor="#161C2A",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="rgba(255, 255, 255, 0.15)",
        ),
    )
    layout.update(overrides)
    return layout


def style_figure(fig: go.Figure, **layout_overrides) -> go.Figure:
    """Apply the glassmorphic dark theme to any Plotly figure."""
    fig.update_layout(**get_base_layout(**layout_overrides))
    return fig


def make_kpi_html(label: str, value: str, delta: str = "", color: str = PRIMARY) -> str:
    """Generate a clean glassmorphic KPI card."""
    delta_html = ""
    if delta:
        delta_color = SUCCESS if delta.startswith("+") or delta.startswith("▲") else DANGER
        delta_html = f'<div style="font-size:12px;color:{delta_color};margin-top:4px;font-weight:500;">{delta}</div>'

    return f"""
    <div style="
        background: {BG_GLASS};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {BORDER_GLASS};
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: {SHADOW_GLASS};
        transition: transform 0.2s ease, border-color 0.2s ease;
    ">
        <div style="font-size:11px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1.2px;font-weight:600;">{label}</div>
        <div style="font-size:26px;font-weight:700;color:{TEXT_PRIMARY};margin-top:6px;letter-spacing:-0.5px;">{value}</div>
        {delta_html}
    </div>
    """


def make_glass_panel(content_html: str, padding: str = "24px 28px") -> str:
    """Wrap content in a glassmorphic card container."""
    return f"""
    <div style="
        background: {BG_GLASS};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {BORDER_GLASS};
        border-radius: 16px;
        padding: {padding};
        box-shadow: {SHADOW_GLASS};
        margin-bottom: 20px;
    ">
        {content_html}
    </div>
    """
