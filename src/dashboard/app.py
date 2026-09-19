"""
NCRB 2024 Crime Analytics Dashboard — Main Streamlit Application.

Interactive dashboard for exploring spatial crime clustering, metropolitan policing
efficiency, and anomaly detection across 964 Indian administrative districts.
Styled with a modern glassmorphic interface and a refined, non-neon color palette.

Launch: .\.venv\Scripts\streamlit.exe run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from src.dashboard.theme import (
    BG_CARD,
    BG_DARK,
    BG_GLASS,
    BG_SURFACE,
    BORDER_GLASS,
    GRID_COLOR,
    PRIMARY,
    SHADOW_GLASS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NCRB 2024 Crime Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM GLASSMORPHIC CSS
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global Canvas with subtle atmospheric ambient depth */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: 
                radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.05) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(100, 116, 139, 0.04) 0%, transparent 45%),
                #0B0F17 !important;
        }}

        /* Main container */
        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }}

        /* Glassmorphic Sidebar styling */
        section[data-testid="stSidebar"] {{
            background: rgba(12, 17, 26, 0.85) !important;
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }}

        section[data-testid="stSidebar"] .stMarkdown h1 {{
            font-size: 18px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
        }}

        /* Frosted Glass Dataframe container */
        .stDataFrame {{
            background: {BG_GLASS};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {BORDER_GLASS};
            border-radius: 12px;
            overflow: hidden;
            box-shadow: {SHADOW_GLASS};
        }}

        /* Button styling — Frosted Glass */
        .stDownloadButton > button {{
            background: rgba(59, 130, 246, 0.15) !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid rgba(59, 130, 246, 0.35) !important;
            backdrop-filter: blur(8px);
            border-radius: 10px;
            font-weight: 600;
            padding: 10px 20px;
            transition: all 0.25s ease;
        }}
        .stDownloadButton > button:hover {{
            background: rgba(59, 130, 246, 0.3) !important;
            border-color: rgba(59, 130, 246, 0.6) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.25);
        }}

        /* Selectbox & Input styling */
        .stSelectbox > div > div {{
            background: rgba(17, 23, 36, 0.7) !important;
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid {BORDER_GLASS} !important;
            color: {TEXT_PRIMARY} !important;
        }}
        .stSelectbox > div > div:hover {{
            border-color: rgba(255, 255, 255, 0.18) !important;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid {BORDER_GLASS};
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px 8px 0 0;
            padding: 8px 18px;
            color: {TEXT_SECONDARY};
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            color: {TEXT_PRIMARY} !important;
            border-bottom: 2px solid {PRIMARY} !important;
        }}

        /* Hide Streamlit default chrome */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* Plotly chart container rounded edges */
        .js-plotly-plot {{
            border-radius: 14px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# DATA LOADING (Cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_district_clusters():
    path = PROJECT_ROOT / "data" / "processed" / "district_clusters.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metro_quadrants():
    path = PROJECT_ROOT / "data" / "processed" / "metro_policing_quadrants.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_cluster_profiles():
    path = PROJECT_ROOT / "outputs" / "cluster_profiles.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_pca_loadings():
    path = PROJECT_ROOT / "outputs" / "pca_factor_loadings.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_kmeans_metrics():
    path = PROJECT_ROOT / "outputs" / "kmeans_evaluation_metrics.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_murder_motives():
    path = PROJECT_ROOT / "outputs" / "murder_motives_ranking.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_cyber_motives():
    path = PROJECT_ROOT / "outputs" / "cyber_motives_ranking.csv"
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 12px 6px 20px 6px; border-bottom: 1px solid {BORDER_GLASS}; margin-bottom: 14px;">
            <div style="display:flex; align-items:center; gap: 10px;">
                <div style="
                    width: 32px; height: 32px; border-radius: 8px; 
                    background: rgba(59, 130, 246, 0.15); 
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    display:flex; align-items:center; justify-content:center;
                    font-size: 16px;">🛡️</div>
                <div>
                    <div style="font-size:16px; font-weight:700; color:{TEXT_PRIMARY}; letter-spacing:-0.3px;">NCRB Analytics</div>
                    <div style="color:{TEXT_MUTED}; font-size:11px; margin-top:1px;">Crime Intelligence v1.0</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = option_menu(
        menu_title=None,
        options=[
            "National Overview",
            "District Explorer",
            "Spatial Clustering",
            "Metro Quadrants",
            "Anomaly Deep Dive",
            "AI Assistant",
        ],
        icons=[
            "house-fill",
            "geo-alt-fill",
            "diagram-3-fill",
            "building",
            "exclamation-triangle-fill",
            "robot",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": PRIMARY, "font-size": "15px"},
            "nav-link": {
                "font-size": "13px",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "9px 14px",
                "border-radius": "10px",
                "color": TEXT_SECONDARY,
                "font-weight": "500",
                "--hover-color": "rgba(255, 255, 255, 0.04)",
            },
            "nav-link-selected": {
                "background": "rgba(59, 130, 246, 0.12)",
                "color": TEXT_PRIMARY,
                "font-weight": "600",
                "border": "1px solid rgba(59, 130, 246, 0.28)",
            },
        },
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="border-top: 1px solid {BORDER_GLASS}; padding-top: 16px;">
            <p style="color:{TEXT_MUTED}; font-size:11px; text-align:center; line-height:1.6;">
            Data Source: <b style="color:{TEXT_SECONDARY};">NCRB India 2024</b><br>
            964 Districts · 37 Metro Cities<br>
            Glassmorphic Analytics Interface
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PAGE ROUTING (Views)
# ─────────────────────────────────────────────
df_clusters = load_district_clusters()
df_metro = load_metro_quadrants()

if selected_page == "National Overview":
    from src.dashboard.views.overview import render
    render(df_clusters, df_metro)

elif selected_page == "District Explorer":
    from src.dashboard.views.district_explorer import render
    render(df_clusters)

elif selected_page == "Spatial Clustering":
    df_profiles = load_cluster_profiles()
    df_pca_loadings = load_pca_loadings()
    df_kmeans_metrics = load_kmeans_metrics()
    from src.dashboard.views.clustering import render
    render(df_clusters, df_profiles, df_pca_loadings, df_kmeans_metrics)

elif selected_page == "Metro Quadrants":
    df_murder = load_murder_motives()
    df_cyber = load_cyber_motives()
    from src.dashboard.views.metropolitan import render
    render(df_metro, df_murder, df_cyber)

elif selected_page == "Anomaly Deep Dive":
    from src.dashboard.views.anomalies import render
    render(df_clusters)

elif selected_page == "AI Assistant":
    df_profiles = load_cluster_profiles()
    df_murder = load_murder_motives()
    df_cyber = load_cyber_motives()
    from src.dashboard.views.ai_chat import render
    render(df_clusters, df_metro, df_profiles, df_murder, df_cyber)
