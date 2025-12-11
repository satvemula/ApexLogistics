"""
Apex Logistics - Enterprise ML Operations Dashboard
===================================================

Production-grade web application for logistics shipment delay prediction
and operational intelligence using CatBoost machine learning models.

Enterprise Features:
- Real-time operational metrics and KPIs
- Advanced risk analytics and predictive insights
- Executive-level reporting and visualization
- Secure model inference via Databricks MLflow
- Comprehensive audit logging and monitoring

Author: Apex Logistics Data Science & Engineering
Version: 7.0.0 Enterprise
Last Updated: 2025-02-14
"""

import os
import sys
import logging
import traceback
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MLflow and model imports
MLFLOW_AVAILABLE = False
_mlflow_module = None
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    _mlflow_module = mlflow
except ImportError as e:
    MLFLOW_AVAILABLE = False
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.warning(f"MLflow import failed: {e}")

# Visualization Imports
import plotly.graph_objects as go
try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False
    logger = logging.getLogger("ApexLogistics")
    logger.warning("PyDeck not available - falling back to Plotly for maps")

# ==============================================================================
# ENTERPRISE CONFIGURATION
# ==============================================================================

# Environment Configuration
# Load from environment variables (can also use Streamlit secrets in .streamlit/secrets.toml)
def get_config():
    """Load configuration from environment variables or Streamlit secrets."""
    try:
        # Try Streamlit secrets first
        secrets = st.secrets.get("DATABRICKS", {})
        host = secrets.get("host", "") or os.environ.get("DATABRICKS_HOST", "")
        token = secrets.get("token", "") or os.environ.get("DATABRICKS_TOKEN", "")
        model_uri = secrets.get("model_uri", "") or os.environ.get("MODEL_URI", "models:/workspace.default.apexlogistics/1")
        return host, token, model_uri
    except:
        # Fallback to environment variables
        return (
            os.environ.get("DATABRICKS_HOST", ""),
            os.environ.get("DATABRICKS_TOKEN", ""),
            os.environ.get("MODEL_URI", "models:/workspace.default.apexlogistics/1")
        )

# Initialize with defaults (will be updated in main() if Streamlit secrets available)
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
MODEL_URI = os.environ.get("MODEL_URI", "models:/workspace.default.apexlogistics/1")
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"

# Application Configuration
APP_NAME = "Apex Logistics Operations Intelligence"
APP_VERSION = "7.0.0"
MODEL_NAME = "CatBoost_Delay_Classifier_v2"
MODEL_TRAINING_DATE = "2025-02-14"

# Enterprise Color Palette (Refined Gold & Navy)
# Enterprise Color Palettes
THEMES = {
    "day": {
        "NAVY": "#0A1F3D",
        "NAVY_LIGHT": "#1A3A5C",
        "GOLD": "#C9A961",
        "GOLD_DARK": "#A68B3D",
        "GOLD_LIGHT": "#E5D4A3",
        "WHITE": "#FFFFFF",
        "GRAY_LIGHT": "#E2E8F0", # Darkened for better contrast against white cards
        "GRAY_MEDIUM": "#E9ECEF",
        "GRAY_DARK": "#6C757D",
        "SUCCESS": "#28A745",
        "WARNING": "#FFC107",
        "DANGER": "#DC3545",
        "INFO": "#17A2B8",
        "BLACK": "#212529",
        "GLASS_BG": "rgba(255, 255, 255, 0.95)", # Increased opacity for better legibility
        "GLASS_BORDER": "rgba(255, 255, 255, 0.5)",
        "TEXT_PRIMARY": "#0A1F3D", # Navy for primary text in day
        "TEXT_SECONDARY": "#6C757D"
    },
    "night": {
        "NAVY": "#0F172A",      # Slate 900
        "NAVY_LIGHT": "#1E293B", # Slate 800
        "GOLD": "#38BDF8",       # Sky 400
        "GOLD_DARK": "#0EA5E9",  # Sky 500
        "GOLD_LIGHT": "#7DD3FC", # Sky 300
        "WHITE": "#1E293B",      # Slate 800 (Card Bg)
        "GRAY_LIGHT": "#020617", # Slate 950 (Deep background)
        "GRAY_MEDIUM": "#334155",# Slate 700
        "GRAY_DARK": "#CBD5E1",  # Slate 300 (Lightened for readability)
        "SUCCESS": "#4ADE80",    # Green 400
        "WARNING": "#FACC15",    # Yellow 400
        "DANGER": "#F87171",     # Red 400
        "INFO": "#22D3EE",       # Cyan 400
        "BLACK": "#F8FAFC",      # Slate 50 (Text)
        "GLASS_BG": "rgba(30, 41, 59, 0.8)",
        "GLASS_BORDER": "rgba(255, 255, 255, 0.1)",
        "TEXT_PRIMARY": "#F8FAFC", # White/Slate 50 for primary text in night
        "TEXT_SECONDARY": "#CBD5E1" # Slate 300
    }
}

# Initialize default colors (will be updated by session state)
COLORS = THEMES["day"]

# Model Configuration
EXPECTED_FEATURE_COUNT = 252
TARGET_CLASSES = {0: "Single Leg", 1: "Two Leg", 2: "Multi-Leg"}

# Input Validation Ranges
INPUT_RANGES = {
    "legs": (1, 3),
    "i1_rcs_p": (0, 5000),
    "i1_hops": (1, 4),
    "o_hops": (1, 4),
    "i1_dep_1_place": (100, 1000),
    "o_dep_1_place": (100, 1000)
}

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def setup_logging() -> logging.Logger:
    """Configure enterprise logging system."""
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler("apex_logistics.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("ApexLogistics")
    logger.info(f"Enterprise application initialized - Version {APP_VERSION}")
    return logger

logger = setup_logging()

# ==============================================================================
# MODEL LOADING & MANAGEMENT
# ==============================================================================

@st.cache_resource(show_spinner="Connecting to Databricks and loading model...")
def load_model(host: str, token: str, model_uri: str) -> Optional[Any]:
    """Load trained model from Databricks MLflow with enterprise error handling."""
    # Try importing MLflow at runtime (in case it was installed after app start)
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as e:
        logger.error(f"MLflow import failed at runtime: {e}")
        logger.error(f"Python path: {sys.executable}")
        return None
    
    if not host or not token:
        logger.warning(f"Databricks credentials missing - host: {bool(host)}, token: {bool(token)}")
        return None
    
    try:
        # Set credentials in environment
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        
        logger.info(f"Connecting to Databricks: {host}")
        logger.info(f"Loading model from: {model_uri}")
        
        mlflow.set_tracking_uri("databricks")
        
        # Try loading the model
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully from Databricks")
            return model
        except Exception as model_error:
            # If model URI format is wrong, provide helpful error
            error_msg = str(model_error)
            logger.error(f"Model loading failed: {error_msg}")
            
            if "Invalid model id" in error_msg or "must start with 'm-'" in error_msg:
                st.error(f"""
                **Model URI Format Error**
                
                The model URI `{model_uri}` is incorrect.
                
                **Correct formats:**
                - `models:/<model_name>/<version>` (e.g., `models:/apexlogistics/1`)
                - `models:/<model_name>/Staging` or `models:/<model_name>/Production`
                - Full run URI: `runs:/<run_id>/model`
                
                Please check your Databricks MLflow Model Registry and update the `model_uri` in `.streamlit/secrets.toml`
                """)
            else:
                st.error(f"Failed to load model: {error_msg}")
            
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Failed to connect to Databricks: {str(e)}")
        return None

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

def validate_inputs(
    legs: int,
    rcs_p: int,
    i_hops: int,
    o_hops: int,
    dep1: int,
    dep2: int
) -> Tuple[bool, Optional[str]]:
    """Enterprise-grade input validation."""
    validations = [
        (legs, "Total Journey Legs", INPUT_RANGES["legs"]),
        (rcs_p, "Planned Check-In Time", INPUT_RANGES["i1_rcs_p"]),
        (i_hops, "Inbound Hops", INPUT_RANGES["i1_hops"]),
        (o_hops, "Outbound Hops", INPUT_RANGES["o_hops"]),
        (dep1, "Origin Airport ID", INPUT_RANGES["i1_dep_1_place"]),
        (dep2, "Outbound Airport ID", INPUT_RANGES["o_dep_1_place"])
    ]
    
    for value, name, (min_val, max_val) in validations:
        if not (min_val <= value <= max_val):
            return False, f"{name} must be between {min_val} and {max_val}"
    
    return True, None

def preprocess_inputs(
    legs: int,
    rcs_p: int,
    i_hops: int,
    o_hops: int,
    dep1: int,
    dep2: int
) -> pd.DataFrame:
    """Preprocess inputs to match model feature requirements."""
    base_features = [legs, rcs_p, i_hops, o_hops, dep1, dep2]
    padded_features = base_features + [0.0] * (EXPECTED_FEATURE_COUNT - len(base_features))
    feature_df = pd.DataFrame([padded_features])
    feature_df.columns = [f"f_{i}" for i in range(EXPECTED_FEATURE_COUNT)]
    return feature_df

def predict_delay_risk(
    model: Any,
    feature_df: pd.DataFrame
) -> Tuple[float, int, Dict[str, float]]:
    """Generate prediction with enterprise error handling."""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_df)[0]
            predicted_class = int(np.argmax(probabilities))
            prob_dict = {
                TARGET_CLASSES[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            delay_risk = float(probabilities[1] + probabilities[2]) if len(probabilities) > 2 else float(probabilities[1])
            return delay_risk, predicted_class, prob_dict
        else:
            prediction = model.predict(feature_df)[0]
            return 0.5, int(prediction), {TARGET_CLASSES[int(prediction)]: 1.0}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

# ==============================================================================
# ENTERPRISE UI COMPONENTS
# ==============================================================================

def apply_enterprise_styling():
    """Apply enterprise-grade styling - professional, clean, executive-focused."""
    st.markdown(f"""
    <style>
    /* Enterprise Color System */
    :root {{
        --navy: {COLORS['NAVY']};
        --navy-light: {COLORS['NAVY_LIGHT']};
        --gold: {COLORS['GOLD']};
        --gold-dark: {COLORS['GOLD_DARK']};
        --gold-light: {COLORS['GOLD_LIGHT']};
        --white: {COLORS['WHITE']};
        --gray-light: {COLORS['GRAY_LIGHT']};
        --gray-medium: {COLORS['GRAY_MEDIUM']};
        --gray-dark: {COLORS['GRAY_DARK']};
        --success: {COLORS['SUCCESS']};
        --warning: {COLORS['WARNING']};
        --danger: {COLORS['DANGER']};
        --info: {COLORS['INFO']};
    }}

    /* Global Reset - Professional Typography */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        background-color: {COLORS['GRAY_LIGHT']};
        color: {COLORS['TEXT_PRIMARY']};
        line-height: 1.6;
    }}

    /* Remove Streamlit Defaults */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }}

    header[data-testid="stHeader"] {{
        display: none;
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: {COLORS['GLASS_BG']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {COLORS['GLASS_BORDER']};
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }}

    .glass-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }}

    /* Enterprise Header Bar - COMPACT SINGLE LINE */
    .enterprise-header {{
        /* Use the base navy color as requested */
        background-color: {COLORS['NAVY']}; 
        padding: 0.5rem 1.5rem; /* Highly compacted padding */
        margin: -1rem -1rem 1rem -1rem; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        
        /* New: Flexbox to manage internal layout cleanly */
        display: flex;
        align-items: center; /* Vertically center the content */
        justify-content: space-between; /* Push content to edges */
        
        /* Added the gold border back to the bottom of the content container */
        border-bottom: 2px solid {COLORS['GOLD']}; 
    }}

    .enterprise-header h1 {{
        color: {COLORS['WHITE']};
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.5px;
    }}

    .enterprise-header p {{
        color: {COLORS['GOLD_LIGHT']};
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
        font-weight: 300;
    }}

    /* KPI Dashboard - Executive Style */
    .kpi-dashboard {{
        background: {COLORS['WHITE']};
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid {COLORS['GRAY_MEDIUM']};
    }}

    .kpi-card {{
        background: {COLORS['WHITE']};
        border-left: 4px solid {COLORS['GOLD']};
        padding: 1.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}

    .kpi-label {{
        color: {COLORS['TEXT_SECONDARY']};
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }}

    .kpi-value {{
        color: {COLORS['TEXT_PRIMARY']};
        font-size: 2rem;
        font-weight: 700;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        margin: 0.5rem 0;
    }}

    .kpi-change {{
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }}

    /* Professional Cards */
    .enterprise-card {{
        background: {COLORS['WHITE']};
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid {COLORS['GRAY_MEDIUM']};
    }}

    .card-title {{
        color: {COLORS['TEXT_PRIMARY']};
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid {COLORS['GOLD']};
    }}

    /* Professional Buttons */
    .stButton > button {{
        background-color: {COLORS['NAVY']};
        color: {COLORS['WHITE']} !important;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .stButton > button:hover {{
        background-color: {COLORS['NAVY_LIGHT']};
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }}

    /* Form Styling */
    .stNumberInput > div > div > input {{
        border: 1px solid {COLORS['GRAY_MEDIUM']};
        border-radius: 4px;
        color: {COLORS['TEXT_PRIMARY']};
    }}

    .stSelectbox > div > div > select {{
        border: 1px solid {COLORS['GRAY_MEDIUM']};
        border-radius: 4px;
        color: {COLORS['TEXT_PRIMARY']};
    }}

    /* Metrics Styling */
    [data-testid="stMetricValue"] {{
        color: {COLORS['TEXT_PRIMARY']} !important;
        font-weight: 700;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS['TEXT_SECONDARY']} !important;
        font-weight: 500;
    }}

    /* Sidebar Professional Styling */
    /* Sidebar Professional Styling */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['NAVY']}; /* Company Blue */
        border-right: 1px solid {COLORS['GOLD']}; /* Company Gold Border */
    }}

    /* Sidebar Text & Headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {{
        color: {COLORS['WHITE']} !important;
    }}

    /* Specific fix for Radio Buttons in Sidebar */
    section[data-testid="stSidebar"] .stRadio label {{
        color: {COLORS['WHITE']} !important;
    }}
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
        color: {COLORS['WHITE']} !important;
    }}

    /* Remove emoji styling issues */
    .stMarkdown {{
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    /* Ensure form labels and titles are visible */
    .stMarkdown h4 {{
        color: {COLORS['TEXT_PRIMARY']} !important;
        background-color: transparent !important;
    }}

    /* Ensure all Streamlit labels are visible */
    label {{
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    /* Input field labels - make them black and visible */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stTextArea label {{
        color: {COLORS['TEXT_PRIMARY']} !important;
        font-weight: 500 !important;
    }}

    /* Streamlit widget labels */
    [data-testid="stWidgetLabel"] {{
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    /* All p elements in forms */
    .element-container p {{
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    /* Plotly chart text visibility */
    .js-plotly-plot {{
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    /* Professional Info Boxes - High Contrast */
    .stInfo {{
        background-color: {COLORS['GRAY_LIGHT']};
        border-left: 4px solid {COLORS['INFO']};
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    .stInfo p, .stInfo div {{
        color: {COLORS['TEXT_PRIMARY']} !important;
    }}

    .stSuccess {{
        background-color: #d4edda;
        border-left: 4px solid {COLORS['SUCCESS']};
        color: #155724 !important;
    }}

    .stSuccess p, .stSuccess div, .stSuccess strong {{
        color: #155724 !important;
    }}

    .stWarning {{
        background-color: #fff3cd;
        border-left: 4px solid {COLORS['WARNING']};
        color: #856404 !important;
    }}

    .stWarning p, .stWarning div, .stWarning strong {{
        color: #856404 !important;
    }}

    .stError {{
        background-color: #f8d7da;
        border-left: 4px solid {COLORS['DANGER']};
        color: #721c24 !important;
    }}

    .stError p, .stError div, .stError strong {{
        color: #721c24 !important;
    }}

    /* Professional Tables */
    .stDataFrame {{
        border: 1px solid {COLORS['GRAY_MEDIUM']};
        border-radius: 4px;
    }}

    /* Hide Streamlit Menu */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def render_enterprise_header():
    """Render compact single-line header."""
    current_time = datetime.now().strftime("%B %d, %Y | %I:%M %p")
    
    st.markdown(f"""<div class="enterprise-header">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <h1 style="color: {COLORS['WHITE']}; font-size: 1.25rem; font-weight: 700; margin: 0; white-space: nowrap;">
            APEX LOGISTICS
        </h1>
        <div style="width: 1px; height: 20px; background: {COLORS['GOLD']}; opacity: 0.5;"></div>
        <p style="color: {COLORS['GOLD_LIGHT']}; font-size: 0.9rem; margin: 0; font-weight: 300; white-space: nowrap;">
            Operations Intelligence Platform
        </p>
    </div>
    <div style="text-align: right;">
        <div style="color: {COLORS['WHITE']}; font-size: 0.85rem; font-weight: 400;">{current_time}</div>
        <div style="color: {COLORS['GOLD_LIGHT']}; font-size: 0.75rem; opacity: 0.8;">v{APP_VERSION}</div>
    </div>
</div>""", unsafe_allow_html=True)

def render_sidebar(model: Optional[Any], current_theme: str) -> str:
    """Render professional sidebar with system information and theme toggle."""
    with st.sidebar:
        # Theme Toggle
        st.markdown(f"""
        <div style="padding: 1rem 0; border-bottom: 1px solid {COLORS['GOLD_DARK']}; margin-bottom: 1.5rem;">
            <h3 style="color: {COLORS['WHITE']}; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                Interface Settings
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme toggle using radio button for simplicity (can be styled better later)
        theme_options = ["Day Mode", "Night Mode"]
        default_index = 0 if current_theme == "day" else 1
        
        selected_theme_label = st.radio(
            "Color Theme",
            options=theme_options,
            index=default_index,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        new_theme = "day" if selected_theme_label == "Day Mode" else "night"
        
        if new_theme != current_theme:
            st.session_state["theme"] = new_theme
            st.rerun()

        # View Mode Toggle
        st.markdown(f"""
        <div style="margin-top: 1.5rem;">
            <label style="color: {COLORS['GOLD']}; font-weight: 600; font-size: 0.9rem;">View Mode</label>
        </div>
        """, unsafe_allow_html=True)
        
        view_mode = st.radio(
            "View Mode Details",
            options=["Executive", "Analyst"],
            index=0 if st.session_state.get("role", "Executive") == "Executive" else 1,
            label_visibility="collapsed",
            key="role_selector"
        )
        
        role = view_mode.lower()
        if role != st.session_state.get("role", "executive"):
            st.session_state["role"] = role
            st.rerun()
            
        st.markdown(f"""
        <div style="padding: 1rem 0; border-bottom: 1px solid {COLORS['GOLD_DARK']}; margin-bottom: 1.5rem;">
            <h3 style="color: {COLORS['WHITE']}; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                System Status
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Status
        if model:
            st.markdown(f"""
            <div style="background: #d4edda; padding: 1rem; border-radius: 4px; border-left: 4px solid {COLORS['SUCCESS']}; margin-bottom: 1rem;">
                <div style="color: #155724; font-weight: 600; margin-bottom: 0.25rem;">Operational</div>
                <div style="color: #155724; font-size: 0.85rem;">Model connected to Databricks</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 4px; border-left: 4px solid {COLORS['WARNING']}; margin-bottom: 1rem;">
                <div style="color: #856404; font-weight: 600; margin-bottom: 0.25rem;">Demonstration Mode</div>
                <div style="color: #856404; font-size: 0.85rem;">Using simulated predictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        # System Information
        st.markdown(f"""
        <div style="padding: 1rem 0; border-bottom: 1px solid {COLORS['GRAY_MEDIUM']}; margin-bottom: 1rem;">
            <div style="color: {COLORS['GRAY_DARK']}; font-size: 0.85rem; margin-bottom: 0.5rem;">
                <strong>Model:</strong> {MODEL_NAME}
            </div>
            <div style="color: {COLORS['GRAY_DARK']}; font-size: 0.85rem; margin-bottom: 0.5rem;">
                <strong>Last Training:</strong> {MODEL_TRAINING_DATE}
            </div>
            <div style="color: {COLORS['GRAY_DARK']}; font-size: 0.85rem;">
                <strong>Environment:</strong> Production
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown(f"""
        <div style="padding: 1rem 0;">
            <h4 style="color: {COLORS['NAVY']}; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.75rem;">
                Quick Actions
            </h4>
            <div style="color: {COLORS['GRAY_DARK']}; font-size: 0.85rem; line-height: 1.8;">
                • Enter shipment details below<br>
                • Review risk assessment results<br>
                • Export reports as needed<br>
                • Monitor operational metrics
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_kpi_dashboard(role: str = "executive"):
    """Render executive KPI dashboard with professional metrics."""
    # Generate realistic enterprise metrics
    base_date = datetime.now()
    
    # Calculate realistic KPIs
    on_time_rate = 91.4  # More realistic
    shipments_at_risk = 87  # More realistic
    avg_planned_time = 1247  # minutes
    active_shipments = 3847
    
    # Calculate changes (realistic variations)
    on_time_change = 1.8
    risk_change = -12
    time_change = -18
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title" style="color: {COLORS["TEXT_PRIMARY"]}; border-bottom-color: {COLORS["GOLD"]};">Operational Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div>
            <div class="kpi-label" style="color: {COLORS['TEXT_SECONDARY']};">On-Time Performance</div>
            <div class="kpi-value" style="color: {COLORS['NAVY']};">{on_time_rate:.1f}%</div>
            <div class="kpi-change" style="color: {COLORS['SUCCESS']};">
                ▲ {on_time_change:.1f}% vs previous period
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div>
            <div class="kpi-label" style="color: {COLORS['TEXT_SECONDARY']};">High-Risk Shipments</div>
            <div class="kpi-value" style="color: {COLORS['NAVY']};">{shipments_at_risk}</div>
            <div class="kpi-change" style="color: {COLORS['NAVY']};">
                ▼ {abs(risk_change)} fewer than last week
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div>
            <div class="kpi-label" style="color: {COLORS['TEXT_SECONDARY']};">Average Transit Time</div>
            <div class="kpi-value" style="color: {COLORS['NAVY']};">{avg_planned_time:,} min</div>
            <div class="kpi-change" style="color: {COLORS['NAVY']};">
                ▼ {abs(time_change)} min improvement
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div>
            <div class="kpi-label" style="color: {COLORS['TEXT_SECONDARY']};">Active Shipments</div>
            <div class="kpi-value" style="color: {COLORS['NAVY']};">{active_shipments:,}</div>
            <div class="kpi-change" style="color: {COLORS['GOLD_DARK']};">
                {active_shipments - 3723} new today
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_prediction_interface(model: Optional[Any], role: str = "executive"):
    """Render professional prediction interface."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title" style="color: {COLORS["TEXT_PRIMARY"]}; border-bottom-color: {COLORS["GOLD"]};">Shipment Risk Assessment</div>', unsafe_allow_html=True)
    
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<h4 style='color: {COLORS['NAVY']}; font-size: 1rem; font-weight: 600; margin-bottom: 1rem; background-color: {COLORS['WHITE']}; padding: 0.5rem; border-left: 3px solid {COLORS['GOLD']};'>Journey Configuration</h4>", unsafe_allow_html=True)
            legs = st.number_input(
                "Total Journey Legs",
                min_value=INPUT_RANGES["legs"][0],
                max_value=INPUT_RANGES["legs"][1],
                value=2,
                help="Number of legs in the shipment journey"
            )
            rcs_p = st.number_input(
                "Planned Check-In Time (minutes)",
                min_value=INPUT_RANGES["i1_rcs_p"][0],
                max_value=INPUT_RANGES["i1_rcs_p"][1],
                value=1500,
                help="Planned check-in time in minutes"
            )
            i_hops = st.number_input(
                "Inbound Transfer Points",
                min_value=INPUT_RANGES["i1_hops"][0],
                max_value=INPUT_RANGES["i1_hops"][1],
                value=2,
                help="Number of inbound transfer points"
            )
        
        with col2:
            st.markdown(f"<h4 style='color: {COLORS['NAVY']}; font-size: 1rem; font-weight: 600; margin-bottom: 1rem; background-color: {COLORS['WHITE']}; padding: 0.5rem; border-left: 3px solid {COLORS['GOLD']};'>Route Details</h4>", unsafe_allow_html=True)
            o_hops = st.number_input(
                "Outbound Transfer Points",
                min_value=INPUT_RANGES["o_hops"][0],
                max_value=INPUT_RANGES["o_hops"][1],
                value=2,
                help="Number of outbound transfer points"
            )
            dep1 = st.selectbox(
                "Origin Facility",
                options=[100, 240, 540, 560, 700, 815],
                index=1,
                format_func=lambda x: f"Facility {x}",
                help="Origin facility identifier"
            )
            dep2 = st.selectbox(
                "Destination Facility",
                options=[100, 240, 540, 560, 700, 815],
                index=2,
                format_func=lambda x: f"Facility {x}",
                help="Destination facility identifier"
            )
        
        submit_button = st.form_submit_button(
            "Generate Risk Assessment",
            use_container_width=True
        )
    
    if submit_button:
        with st.spinner("Analyzing shipment configuration..."):
            is_valid, error_msg = validate_inputs(legs, rcs_p, i_hops, o_hops, dep1, dep2)
            
            if not is_valid:
                st.error(f"Validation Error: {error_msg}")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            try:
                feature_df = preprocess_inputs(legs, rcs_p, i_hops, o_hops, dep1, dep2)
                
                if model:
                    logger.info("Using real model for prediction")
                    delay_risk, predicted_class, class_probs = predict_delay_risk(model, feature_df)
                    st.success("✓ Prediction generated using ML model")
                else:
                    logger.warning("Model is None - using mock prediction")
                    # Realistic mock prediction
                    delay_risk = 0.68
                    predicted_class = 1
                    class_probs = {"Single Leg": 0.22, "Two Leg": 0.58, "Multi-Leg": 0.20}
                    st.warning("⚠️ Using simulated prediction (model not loaded)")
                
                render_prediction_results(delay_risk, predicted_class, class_probs, legs, rcs_p, role)
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                st.error(f"Analysis Error: {str(e)}")
                if DEBUG_MODE or role == "analyst":
                    st.exception(e)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_prediction_results(
    delay_risk: float,
    predicted_class: int,
    class_probs: Dict[str, float],
    legs: int,
    rcs_p: int,
    role: str = "executive"
):
    """Render professional prediction results."""
    st.markdown("---")
    
    # Risk Assessment
    if delay_risk > 0.7:
        risk_level = "HIGH RISK"
        risk_color = COLORS['DANGER']
        risk_bg = "#f8d7da"
        risk_text_color = "#721c24"
    elif delay_risk > 0.5:
        risk_level = "ELEVATED RISK"
        risk_color = COLORS['WARNING']
        risk_bg = "#fff3cd"
        risk_text_color = "#856404"
    else:
        risk_level = "STANDARD RISK"
        risk_color = COLORS['SUCCESS']
        risk_bg = "#d4edda"
        risk_text_color = "#155724"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="background: {risk_bg}; border-left: 4px solid {risk_color};
                    padding: 2rem; border-radius: 4px; text-align: center;">
            <div style="color: {risk_text_color}; font-size: 0.85rem; font-weight: 600; 
                        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">
                Risk Classification
            </div>
            <div style="color: {risk_text_color}; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {risk_level}
            </div>
            <div style="color: {risk_text_color}; font-size: 1.5rem; font-weight: 600;">
                {delay_risk:.1%} Probability
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h4 style='color: {COLORS['NAVY']}; margin-bottom: 1rem;'>Prediction Confidence Distribution</h4>", unsafe_allow_html=True)
        
        prob_df = pd.DataFrame({
            "Journey Type": list(class_probs.keys()),
            "Probability": list(class_probs.values())
        })
        
        fig = px.bar(
            prob_df,
            x="Journey Type",
            y="Probability",
            color="Probability",
            color_continuous_scale=[COLORS['GOLD_LIGHT'], COLORS['GOLD'], COLORS['GOLD_DARK']],
            text="Probability"
        )
        fig.update_traces(
            texttemplate='%{text:.1%}',
            textposition='outside',
            marker_line_color=COLORS['NAVY'],
            marker_line_width=1.5
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=COLORS['BLACK'], size=12),
            showlegend=False,
            yaxis=dict(
                tickformat=".0%", 
                gridcolor=COLORS['GRAY_MEDIUM'], 
                title="Probability",
                title_font=dict(color=COLORS['BLACK'], size=12),
                tickfont=dict(color=COLORS['BLACK'], size=10),
            ),
            xaxis=dict(
                title="Predicted Shipment Journey Type",
                title_font=dict(color=COLORS['BLACK'], size=12),
                tickfont=dict(color=COLORS['BLACK'], size=10),
            ),
            margin=dict(t=20, l=40, r=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# ADVANCED VISUALIZATIONS
# ==============================================================================

def render_geospatial_map(dep1: int, dep2: int):
    """Render 3D Arc Layer map using PyDeck."""
    st.markdown(f'<div class="card-title" style="color: {COLORS["TEXT_PRIMARY"]}; border-bottom-color: {COLORS["GOLD"]};">Global Logistics Network</div>', unsafe_allow_html=True)
    
    # Facility Coordinates (Demo Data)
    FACILITIES = {
        100: {"name": "JFK Hub", "lat": 40.6413, "lon": -73.7781},
        240: {"name": "LAX Gateway", "lat": 33.9416, "lon": -118.4085},
        540: {"name": "LHR Center", "lat": 51.4700, "lon": -0.4543},
        560: {"name": "DXB Transit", "lat": 25.2532, "lon": 55.3657},
        700: {"name": "HKG Port", "lat": 22.3080, "lon": 113.9185},
        815: {"name": "SYD Hub", "lat": -33.9399, "lon": 151.1753}
    }
    
    # Get coordinates for selected route
    origin = FACILITIES.get(dep1, FACILITIES[100])
    dest = FACILITIES.get(dep2, FACILITIES[540])
    
    # Generate arcs for all facilities to show network
    arcs_data = [
        {"source": [FACILITIES[100]["lon"], FACILITIES[100]["lat"]], "target": [FACILITIES[540]["lon"], FACILITIES[540]["lat"]], "value": 10},
        {"source": [FACILITIES[240]["lon"], FACILITIES[240]["lat"]], "target": [FACILITIES[700]["lon"], FACILITIES[700]["lat"]], "value": 5},
        {"source": [FACILITIES[560]["lon"], FACILITIES[560]["lat"]], "target": [FACILITIES[815]["lon"], FACILITIES[815]["lat"]], "value": 3},
        # Active Route
        {"source": [origin["lon"], origin["lat"]], "target": [dest["lon"], dest["lat"]], "value": 20, "color": [201, 169, 97, 255]} # Gold
    ]
    
    if PYDECK_AVAILABLE:
        view_state = pdk.ViewState(
            latitude=20.0,
            longitude=0.0,
            zoom=1,
            pitch=45,
        )
        
        layer = pdk.Layer(
            "ArcLayer",
            data=arcs_data,
            get_source_position="source",
            get_target_position="target",
            get_width="value",
            get_tilt=15,
            get_source_color=[10, 31, 61, 150],  # Navy transparent
            get_target_color=[255, 0, 0, 255] if "color" not in arcs_data[-1] else [201, 169, 97, 255],
            pickable=True,
            auto_highlight=True,
        )
        
        r = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9" if st.session_state["theme"] == "day" else "mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=True
        )
        
        st.pydeck_chart(r)
    else:
        # Plotly scattergeo fallback
        fig = go.Figure()
        
        # Add facility markers
        lats = [f["lat"] for f in FACILITIES.values()]
        lons = [f["lon"] for f in FACILITIES.values()]
        names = [f["name"] for f in FACILITIES.values()]
        
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            text=names,
            mode='markers+text',
            marker=dict(size=12, color=COLORS['GOLD'], line=dict(width=2, color=COLORS['WHITE'])),
            textposition="top center",
            textfont=dict(size=9, color=COLORS['TEXT_PRIMARY']),
            name='Hubs',
            hoverinfo='text'
        ))
        
        # Add route lines
        for arc in arcs_data:
            is_active = "color" in arc
            fig.add_trace(go.Scattergeo(
                lon=[arc["source"][0], arc["target"][0]],
                lat=[arc["source"][1], arc["target"][1]],
                mode='lines',
                line=dict(
                    width=3 if is_active else 1.5,
                    color=COLORS['GOLD'] if is_active else COLORS['NAVY_LIGHT']
                ),
                opacity=1.0 if is_active else 0.5,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Configure the map
        is_day_mode = st.session_state.get("theme", "day") == "day"
        
        fig.update_layout(
            geo=dict(
                projection_type='orthographic',
                showland=True,
                landcolor='rgb(243, 243, 243)' if is_day_mode else 'rgb(30, 30, 30)',
                coastlinecolor='rgb(204, 204, 204)' if is_day_mode else 'rgb(100, 100, 100)',
                coastlinewidth=1,
                showocean=True,
                oceancolor='rgb(230, 245, 255)' if is_day_mode else 'rgb(15, 23, 42)',
                showcountries=True,
                countrycolor='rgb(204, 204, 204)' if is_day_mode else 'rgb(71, 85, 105)',
                countrywidth=0.5,
                showlakes=True,
                lakecolor='rgb(230, 245, 255)' if is_day_mode else 'rgb(15, 23, 42)',
                bgcolor='rgba(0,0,0,0)',
                center=dict(lat=20, lon=0),
                projection_scale=1.0
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
def render_sankey_diagram():
    """Render Sankey diagram of Shipment Flows."""
    st.markdown(f'<div class="card-title" style="color: {COLORS["TEXT_PRIMARY"]}; border-bottom-color: {COLORS["GOLD"]};">Shipment Flow Distribution</div>', unsafe_allow_html=True)
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = ["Origin: JFK", "Origin: LAX", "Hub: London", "Hub: Dubai", "Dest: Hong Kong", "Dest: Sydney"],
          color = [COLORS['NAVY_LIGHT']] * 6
        ),
        link = dict(
          source = [0, 1, 0, 2, 3, 3], # indices correspond to labels
          target = [2, 3, 3, 4, 4, 5],
          value =  [8, 4, 2, 8, 4, 2],
          color = [COLORS['GOLD_LIGHT']] * 6
      ))])
    
    fig.update_layout(
        font=dict(size=12, color=COLORS['TEXT_PRIMARY']),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_advanced_visualizations(role: str):
    """Render advanced visualizations based on role."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    if role == "executive":
        render_sankey_diagram()
    else:
        # Analysts see the map
        col1, col2 = st.columns([2, 1])
        with col1:
            render_geospatial_map(100, 540) # Default demo route
        with col2:
             st.markdown(f"""
             <div style="background: {COLORS['GRAY_LIGHT']}; padding: 1.5rem; border-radius: 8px;">
                 <h4 style="color: {COLORS['NAVY']}; margin-top: 0;">Network Status</h4>
                 <p style="font-size: 0.9rem;">Global hub connectivity is operating at 98% efficiency. 
                 Major congestion reported in EU sector.</p>
                 <div style="margin-top: 1rem; border-top: 1px solid {COLORS['GRAY_MEDIUM']}; padding-top: 1rem;">
                    <strong>Active Alerts:</strong><br>
                    • LHR: Weather Delay<br>
                    • DXB: Capacity Warning
                 </div>
             </div>
             """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize Session State for Theme
    if "theme" not in st.session_state:
        st.session_state["theme"] = "day"
    
    # Initialize Session State for Role
    if "role" not in st.session_state:
        st.session_state["role"] = "executive"

    # Update Global Colors based on Theme
    global COLORS
    COLORS = THEMES[st.session_state["theme"]]
    
    # Set page config once
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 1. Apply Enterprise Styling (CSS) - Uses updated COLORS
    apply_enterprise_styling()
    
    # 2. Get Configuration
    host, token, model_uri = get_config()
    
    # 3. Load Model
    model = load_model(host, token, model_uri)
    
    # 4. Render Sidebar
    render_sidebar(model, st.session_state["theme"])
    
    # 5. Render Header
    render_enterprise_header()

    # 6. Render Main Content
    render_kpi_dashboard(st.session_state["role"])
    
    # 7. Render Advanced Visualizations
    render_advanced_visualizations(st.session_state["role"])
    
    # 8. Render Prediction Interface
    render_prediction_interface(model, st.session_state["role"])
    
    # Render footer information if needed (currently hidden by CSS)
    if DEBUG_MODE:
        st.sidebar.caption(f"Host: {host[:10]}...")
        st.sidebar.caption(f"Model URI: {model_uri}")
        st.sidebar.caption(f"Model Loaded: {bool(model)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal application error: {str(e)}")
        st.error(f"A fatal application error occurred. Check logs for details. Error: {str(e)}")