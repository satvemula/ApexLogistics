import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION & DESIGN SYSTEM CONSTANTS
# ==============================================================================

# --- BRAND PALETTE ---
COLORS = {
    "NAVY": "#012A5C",
    "GOLD": "#DAA520",
    "MUTED_GOLD_BG": "#FFF8DC", 
    "LIGHT_GREY": "#F5F6F8",
    "MEDIUM_GREY": "#D9D9D9",
    "GREEN": "#4BB543",
    "RED": "#D9534F",
    "WHITE": "#FFFFFF",
    "BLACK": "#000000"
}

# --- FEATURE CONFIGURATION ---
FEATURE_LABELS = {
    "i1_dep_1_place": "Origin Airport (Leg 1)",
    "i1_rcs_p": "Planned Check-In Time",
    "o_dep_1_place": "Outbound Airport",
    "i1_hops": "Inbound Hops",
    "o_hops": "Outbound Hops",
    "legs": "Total Journey Legs"
}

FEATURE_IMPORTANCE = {
    "Feature": ["i1_dep_1_place", "i1_rcs_p", "o_dep_1_place", "i1_hops", "o_hops", "legs"],
    "Importance": [0.211, 0.187, 0.142, 0.095, 0.078, 0.055]
}

MOCKED_PORT_IDS = [100, 240, 560, 815]

# ==============================================================================
# 2. SETUP & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Apex Logistics Executive Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* VARIABLES */
    :root {
        --navy: #012A5C;
        --gold: #DAA520;
        --bg-grey: #F5F6F8;
        --border: #D9D9D9;
        --success: #4BB543;
        --danger: #D9534F;
        --muted-gold: #FFF8DC;
        --white: #FFFFFF;
        --black: #000000;
    }

    /* GLOBAL TYPOGRAPHY & RESET */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--muted-gold) !important;
        color: var(--navy) !important;
    }

    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
    }

    /* Hide default Streamlit header/menu */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* MAIN CONTAINER BOX STYLING */
    .block-container {
        background-color: var(--navy);
        padding: 3rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        margin-top: 2rem;
        margin-bottom: 2rem;
        max-width: 95% !important;
        border: 2px solid var(--gold);
    }

    /* Text Color Overrides for Dark Background (Navy Container) */
    .block-container p,
    .block-container li,
    .block-container span,
    .block-container label,
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container h5,
    .block-container h6,
    .block-container .stMarkdown,
    [data-testid="stMetricLabel"] {
        color: var(--white) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--gold) !important;
    }

    /* EXECUTIVE DECISION BAR (White Box inside Navy Container) */
    .exec-bar {
        background-color: var(--white);
        border: 2px solid var(--gold);
        border-radius: 8px;
        padding: 20px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .exec-bar h3, .exec-bar [data-testid="stMetricValue"] {
        color: var(--navy) !important;
    }

    .exec-bar [data-testid="stMetricLabel"] {
        color: var(--gold) !important;
    }

    /* BUTTON STYLING */
    .stButton > button {
        background-color: var(--gold);
        color: var(--navy) !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: var(--white);
        color: var(--navy) !important;
    }

    /* CARD STYLE - For charts and structural columns */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
        background-color: var(--white);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid var(--gold);
    }

    /* Fix text color inside the white cards (back to Navy) */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div p,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div h1,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div h2,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div h3,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div span,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div li {
        color: var(--navy) !important;
    }

    /* Remove background from individual markdown elements */
    .stMarkdown {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* Fix tabs text color */
    button[data-baseweb="tab"] {
        color: var(--white) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--gold) !important;
        border-color: var(--gold) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. BACKEND: DATABRICKS CONNECTION
# ==============================================================================
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
MODEL_URI = "models:/workspace.default.apexlogistics/1"

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    import mlflow.sklearn
    mlflow.set_tracking_uri("databricks")

@st.cache_resource
def get_model():
    if not (DATABRICKS_HOST and DATABRICKS_TOKEN):
        return None
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        return model
    except:
        return None

active_model = get_model()

# ==============================================================================
# 4. COMPONENT FUNCTIONS
# ==============================================================================

def kpi_header_section():
    # Using f-strings here for simple variable insertion, which is safe in HTML
    html_code = f"""
    <div class="exec-bar">
        <h3 style="margin-top: 0; margin-bottom: 20px; font-weight: 700;">
            Executive Decision Bar
        </h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between;">
            <div style="flex: 1; min-width: 140px; text-align: center;">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 4px; color: {COLORS['GOLD']} !important;">On-Time Rate (QTD)</div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 28px; font-weight: 700; color: {COLORS['NAVY']};">88.6%</div>
                <div style="color: {COLORS['GREEN']}; font-weight: 600; font-size: 14px;">▲
