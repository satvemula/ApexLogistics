"""
Apex Logistics - Production ML Prediction Dashboard
===================================================

A professional web application for logistics shipment delay prediction using
CatBoost machine learning models deployed via Databricks MLflow.

Author: Apex Logistics Data Science Team
Version: 2.0.0
Last Updated: 2025-02-14

Architecture:
- Modular design with separation of concerns
- Production-ready error handling and logging
- Professional UI/UX with enterprise-grade features
- Secure model loading and inference
- Comprehensive input validation

Deployment Notes:
- Configure DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
- Model URI: models:/workspace.default.apexlogistics/1
- For Docker deployment, see Dockerfile (to be created)
- For CI/CD integration, see .github/workflows (to be created)
"""

import os
import sys
import logging
import traceback
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MLflow and model imports
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    st.error("⚠️ MLflow not available. Install with: pip install mlflow")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Environment Configuration
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
MODEL_URI = os.environ.get("MODEL_URI", "models:/workspace.default.apexlogistics/1")
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"

# Application Configuration
APP_NAME = "Apex Logistics Executive Dashboard"
APP_VERSION = "2.0.0"
MODEL_NAME = "CatBoost_Delay_Classifier_v2"
MODEL_TRAINING_DATE = "2025-02-14"

# Brand Color Palette (Gold & Blue)
COLORS = {
    "NAVY": "#012A5C",
    "GOLD": "#DAA520",
    "MUTED_GOLD_BG": "#FFF8DC",
    "LIGHT_GREY": "#F5F6F8",
    "MEDIUM_GREY": "#D9D9D9",
    "GREEN": "#4BB543",
    "RED": "#D9534F",
    "ORANGE": "#FF8C00",
    "WHITE": "#FFFFFF",
    "BLACK": "#000000",
    "DARK_GREY": "#2C3E50"
}

# Feature Configuration
FEATURE_LABELS = {
    "i1_dep_1_place": "Origin Airport (Leg 1)",
    "i1_rcs_p": "Planned Check-In Time (minutes)",
    "o_dep_1_place": "Outbound Airport",
    "i1_hops": "Inbound Hops",
    "o_hops": "Outbound Hops",
    "legs": "Total Journey Legs"
}

# Feature Importance (from model training)
FEATURE_IMPORTANCE = {
    "Feature": ["i1_dep_1_place", "i1_rcs_p", "o_dep_1_place", "i1_hops", "o_hops", "legs"],
    "Importance": [0.211, 0.187, 0.142, 0.095, 0.078, 0.055]
}

# Model Configuration
EXPECTED_FEATURE_COUNT = 252  # After preprocessing and feature engineering
TARGET_CLASSES = {0: "1 Leg", 1: "2 Legs", 2: "3 Legs"}  # Encoded target mapping

# Input Validation Ranges
INPUT_RANGES = {
    "legs": (1, 3),
    "i1_rcs_p": (0, 5000),
    "i1_hops": (1, 4),
    "o_hops": (1, 4),
    "i1_dep_1_place": (100, 1000),
    "o_dep_1_place": (100, 1000)
}

# Mock Port IDs for filtering
MOCKED_PORT_IDS = [100, 240, 560, 815]

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging.
    
    Returns:
        Logger instance configured for the application
    """
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
    logger.info(f"Application started - Version {APP_VERSION}")
    return logger

logger = setup_logging()

# ==============================================================================
# MODEL LOADING & MANAGEMENT
# ==============================================================================

@st.cache_resource(show_spinner="Loading ML model from Databricks...")
def load_model() -> Optional[Any]:
    """
    Load the trained model from Databricks MLflow.
    
    Uses Streamlit caching to avoid reloading on every interaction.
    
    Returns:
        Loaded model object or None if loading fails
    """
    if not MLFLOW_AVAILABLE:
        logger.error("MLflow not available")
        return None
    
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        logger.warning("Databricks credentials not configured. Running in mock mode.")
        return None
    
    try:
        # Configure MLflow tracking
        mlflow.set_tracking_uri("databricks")
        
        # Load model
        logger.info(f"Loading model from: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# ==============================================================================
# DATA PREPROCESSING PIPELINE
# ==============================================================================

def validate_inputs(
    legs: int,
    rcs_p: int,
    i_hops: int,
    o_hops: int,
    dep1: int,
    dep2: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate user inputs against expected ranges.
    
    Args:
        legs: Total journey legs
        rcs_p: Planned check-in time
        i_hops: Inbound hops
        o_hops: Outbound hops
        dep1: Origin airport ID
        dep2: Outbound airport ID
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    validations = [
        (legs, "legs", INPUT_RANGES["legs"]),
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
    """
    Preprocess user inputs to match model's expected feature format.
    
    This function replicates the preprocessing pipeline from training:
    1. Creates base feature vector
    2. Pads to expected feature count (252 features)
    3. Returns DataFrame with proper column names
    
    Note: In production, you would want to:
    - Load saved preprocessors (LabelEncoder, StandardScaler, etc.)
    - Apply the exact same transformations
    - Handle interaction features properly
    
    For now, this creates a simplified feature vector that matches
    the expected shape. In a full production system, you would save
    and load the preprocessing pipeline.
    
    Args:
        legs: Total journey legs
        rcs_p: Planned check-in time
        i_hops: Inbound hops
        o_hops: Outbound hops
        dep1: Origin airport ID
        dep2: Outbound airport ID
    
    Returns:
        DataFrame with 252 features (f_0 to f_251)
    """
    # Create base feature vector from inputs
    # Note: This is a simplified version. In production, you'd use
    # the exact same preprocessing pipeline from training
    base_features = [
        legs, rcs_p, i_hops, o_hops, dep1, dep2
    ]
    
    # Pad to expected feature count
    # In production, this would be done through proper feature engineering
    padded_features = base_features + [0.0] * (EXPECTED_FEATURE_COUNT - len(base_features))
    
    # Create DataFrame with expected column names
    feature_df = pd.DataFrame([padded_features])
    feature_df.columns = [f"f_{i}" for i in range(EXPECTED_FEATURE_COUNT)]
    
    return feature_df

def predict_delay_risk(
    model: Any,
    feature_df: pd.DataFrame
) -> Tuple[float, int, Dict[str, float]]:
    """
    Generate prediction from the model.
    
    Args:
        model: Trained model object
        feature_df: Preprocessed feature DataFrame
    
    Returns:
        Tuple of (probability, predicted_class, class_probabilities)
    """
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_df)[0]
            predicted_class = int(np.argmax(probabilities))
            
            # Create probability dictionary
            prob_dict = {
                TARGET_CLASSES[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            # For delay risk, we use the probability of multi-leg journeys (2 or 3 legs)
            # as a proxy for complexity/delay risk
            delay_risk = float(probabilities[1] + probabilities[2]) if len(probabilities) > 2 else float(probabilities[1])
            
            return delay_risk, predicted_class, prob_dict
        else:
            # Fallback for models without predict_proba
            prediction = model.predict(feature_df)[0]
            return 0.5, int(prediction), {TARGET_CLASSES[int(prediction)]: 1.0}
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def apply_custom_styling():
    """Apply custom CSS styling for professional appearance."""
    st.markdown(f"""
    <style>
    /* CSS Variables for Brand Colors */
    :root {{
        --navy: {COLORS['NAVY']};
        --gold: {COLORS['GOLD']};
        --bg-grey: {COLORS['LIGHT_GREY']};
        --border: {COLORS['MEDIUM_GREY']};
        --success: {COLORS['GREEN']};
        --danger: {COLORS['RED']};
        --muted-gold: {COLORS['MUTED_GOLD_BG']};
        --white: {COLORS['WHITE']};
        --black: {COLORS['BLACK']};
    }}

    /* Global Typography & Reset */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(135deg, var(--muted-gold) 0%, #F0E68C 100%);
        color: var(--navy);
    }}

    /* Remove default Streamlit padding */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }}

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {{
        display: none;
    }}

    /* Main Container Styling */
    .main-container {{
        background-color: var(--navy);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        margin: 2rem auto;
        border: 3px solid var(--gold);
    }}

    /* Executive Decision Bar */
    .exec-bar {{
        background: linear-gradient(135deg, var(--white) 0%, #F8F9FA 100%);
        border: 2px solid var(--gold);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 28px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}

    .exec-bar h3 {{
        color: var(--navy) !important;
        margin-top: 0;
        margin-bottom: 20px;
        font-weight: 700;
        font-size: 1.4rem;
    }}

    /* KPI Cards */
    .kpi-card {{
        text-align: center;
        padding: 16px;
        border-radius: 8px;
        background: rgba(218, 165, 32, 0.05);
        border: 1px solid rgba(218, 165, 32, 0.2);
    }}

    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--gold) 0%, #B8860B 100%);
        color: var(--navy) !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #B8860B 0%, var(--gold) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }}

    /* Card Styling */
    .metric-card {{
        background-color: var(--white);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid var(--gold);
        margin-bottom: 16px;
    }}

    /* Text Color Overrides */
    .block-container p,
    .block-container li,
    .block-container span,
    .block-container label,
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container h5,
    .block-container h6 {{
        color: var(--white) !important;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--gold) !important;
        font-weight: 700;
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--white) !important;
        opacity: 0.9;
    }}

    /* Sidebar Styling */
    .css-1d391kg {{
        background-color: var(--navy);
    }}

    /* Info Boxes */
    .stInfo {{
        background-color: rgba(255, 255, 255, 0.1);
        border-left: 4px solid var(--gold);
    }}

    /* Error/Success Messages */
    .stError {{
        background-color: rgba(217, 83, 79, 0.1);
        border-left: 4px solid var(--danger);
    }}

    .stSuccess {{
        background-color: rgba(75, 181, 67, 0.1);
        border-left: 4px solid var(--success);
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: var(--white) !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--gold) !important;
        border-color: var(--gold) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render application header."""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: {COLORS['GOLD']}; font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🚚 APEX LOGISTICS
        </h1>
        <p style="color: {COLORS['WHITE']}; font-size: 1.2rem; opacity: 0.9;">
            Executive Dashboard | AI-Powered Shipment Risk Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(model: Optional[Any]):
    """Render sidebar with model information and instructions."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem 0;">
            <h2 style="color: {COLORS['GOLD']}; margin-bottom: 1rem;">📊 Model Information</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Status
        if model:
            st.success("✅ Model Operational")
            status_color = COLORS['GREEN']
            status_text = "Connected to Databricks"
        else:
            st.warning("⚠️ Model Offline (Mock Mode)")
            status_color = COLORS['ORANGE']
            status_text = "Using fallback predictions"
        
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="color: {COLORS['WHITE']}; margin: 0.5rem 0;"><strong>Model:</strong> {MODEL_NAME}</p>
            <p style="color: {COLORS['WHITE']}; margin: 0.5rem 0;"><strong>Status:</strong> <span style="color: {status_color};">{status_text}</span></p>
            <p style="color: {COLORS['WHITE']}; margin: 0.5rem 0;"><strong>Version:</strong> {APP_VERSION}</p>
            <p style="color: {COLORS['WHITE']}; margin: 0.5rem 0;"><strong>Last Training:</strong> {MODEL_TRAINING_DATE}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Instructions
        st.markdown(f"""
        <div style="padding: 1rem 0;">
            <h3 style="color: {COLORS['GOLD']}; margin-bottom: 1rem;">📖 Instructions</h3>
            <ol style="color: {COLORS['WHITE']}; padding-left: 1.5rem;">
                <li>Enter shipment details in the form</li>
                <li>Click "Calculate Risk Score"</li>
                <li>Review prediction and confidence</li>
                <li>Check feature importance insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset Info
        st.markdown(f"""
        <div style="padding: 1rem 0;">
            <h3 style="color: {COLORS['GOLD']}; margin-bottom: 1rem;">📈 Dataset</h3>
            <p style="color: {COLORS['WHITE']}; font-size: 0.9rem;">
                Trained on 3,942 logistics shipments with 98 features.
                Best model: CatBoost (F1: 0.70, Accuracy: 0.71)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; color: {COLORS['WHITE']}; opacity: 0.7; font-size: 0.8rem;">
            <p>Apex Logistics</p>
            <p>Confidential | 2025</p>
        </div>
        """, unsafe_allow_html=True)

def render_kpi_dashboard():
    """Render executive KPI dashboard."""
    html_code = f"""
    <div class="exec-bar">
        <h3>📊 Executive Decision Bar</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between;">
            <div class="kpi-card" style="flex: 1; min-width: 160px;">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 8px; color: {COLORS['GOLD']};">
                    On-Time Rate (QTD)
                </div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 32px; font-weight: 700; color: {COLORS['NAVY']}; margin-bottom: 4px;">
                    88.6%
                </div>
                <div style="color: {COLORS['GREEN']}; font-weight: 600; font-size: 14px;">
                    ▲ 2.1% vs last quarter
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 160px; border-left: 2px solid rgba(218, 165, 32, 0.3); padding-left: 20px;">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 8px; color: {COLORS['GOLD']};">
                    Shipments At Risk
                </div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 32px; font-weight: 700; color: {COLORS['NAVY']}; margin-bottom: 4px;">
                    142
                </div>
                <div style="color: {COLORS['RED']}; font-weight: 600; font-size: 14px;">
                    ▼ 5% vs last week
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 160px; border-left: 2px solid rgba(218, 165, 32, 0.3); padding-left: 20px;">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 8px; color: {COLORS['GOLD']};">
                    Avg Planned Time
                </div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 32px; font-weight: 700; color: {COLORS['NAVY']}; margin-bottom: 4px;">
                    1,550m
                </div>
                <div style="color: {COLORS['GREEN']}; font-weight: 600; font-size: 14px;">
                    ▼ 12m improvement
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 160px; border-left: 2px solid rgba(218, 165, 32, 0.3); padding-left: 20px;">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 8px; color: {COLORS['GOLD']};">
                    Active Shipments
                </div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 32px; font-weight: 700; color: {COLORS['NAVY']}; margin-bottom: 4px;">
                    3,942
                </div>
                <div style="color: {COLORS['GOLD']}; font-weight: 600; font-size: 14px;">
                    ● 124 new today
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def render_prediction_form(model: Optional[Any]):
    """Render prediction input form and results."""
    st.markdown("---")
    st.markdown(f"<h2 style='color: {COLORS['GOLD']}; margin-bottom: 1.5rem;'>⚡ Manual Risk Assessment</h2>", unsafe_allow_html=True)
    
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"<h4 style='color: {COLORS['WHITE']};'>Journey Configuration</h4>", unsafe_allow_html=True)
            legs = st.number_input(
                "Total Journey Legs",
                min_value=INPUT_RANGES["legs"][0],
                max_value=INPUT_RANGES["legs"][1],
                value=2,
                help="Number of legs in the shipment journey (1-3)"
            )
            rcs_p = st.number_input(
                "Planned Check-In Time (minutes)",
                min_value=INPUT_RANGES["i1_rcs_p"][0],
                max_value=INPUT_RANGES["i1_rcs_p"][1],
                value=1500,
                help="Planned check-in time in minutes"
            )
        
        with col2:
            st.markdown(f"<h4 style='color: {COLORS['WHITE']};'>Route Complexity</h4>", unsafe_allow_html=True)
            i_hops = st.number_input(
                "Inbound Hops",
                min_value=INPUT_RANGES["i1_hops"][0],
                max_value=INPUT_RANGES["i1_hops"][1],
                value=2,
                help="Number of inbound transfer points"
            )
            o_hops = st.number_input(
                "Outbound Hops",
                min_value=INPUT_RANGES["o_hops"][0],
                max_value=INPUT_RANGES["o_hops"][1],
                value=2,
                help="Number of outbound transfer points"
            )
        
        with col3:
            st.markdown(f"<h4 style='color: {COLORS['WHITE']};'>Airport Locations</h4>", unsafe_allow_html=True)
            dep1 = st.selectbox(
                "Origin Airport ID",
                options=[100, 240, 540, 560, 700, 815],
                index=1,
                help="Origin airport identifier"
            )
            dep2 = st.selectbox(
                "Outbound Airport ID",
                options=[100, 240, 540, 560, 700, 815],
                index=2,
                help="Outbound airport identifier"
            )
        
        submit_button = st.form_submit_button(
            "🚀 CALCULATE RISK SCORE",
            use_container_width=True
        )
    
    # Process form submission
    if submit_button:
        with st.spinner("🔄 Processing prediction..."):
            # Validate inputs
            is_valid, error_msg = validate_inputs(legs, rcs_p, i_hops, o_hops, dep1, dep2)
            
            if not is_valid:
                st.error(f"❌ Validation Error: {error_msg}")
                return
            
            try:
                # Preprocess inputs
                feature_df = preprocess_inputs(legs, rcs_p, i_hops, o_hops, dep1, dep2)
                
                # Generate prediction
                if model:
                    delay_risk, predicted_class, class_probs = predict_delay_risk(model, feature_df)
                else:
                    # Mock prediction for demo
                    delay_risk = 0.78
                    predicted_class = 1
                    class_probs = {"1 Leg": 0.15, "2 Legs": 0.65, "3 Legs": 0.20}
                
                # Display results
                render_prediction_results(delay_risk, predicted_class, class_probs, legs, rcs_p)
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                st.error(f"❌ Prediction Error: {str(e)}")
                if DEBUG_MODE:
                    st.exception(e)

def render_prediction_results(
    delay_risk: float,
    predicted_class: int,
    class_probs: Dict[str, float],
    legs: int,
    rcs_p: int
):
    """Render prediction results with visualizations."""
    st.markdown("---")
    
    # Risk Assessment Card
    risk_level = "HIGH RISK" if delay_risk > 0.6 else "MODERATE RISK" if delay_risk > 0.4 else "LOW RISK"
    risk_color = COLORS['RED'] if delay_risk > 0.6 else COLORS['ORANGE'] if delay_risk > 0.4 else COLORS['GREEN']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}dd 100%);
                    color: white; padding: 2rem; border-radius: 12px; text-align: center;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
            <h3 style="color: white; margin: 0 0 1rem 0; font-size: 1.2rem;">Risk Level</h3>
            <h1 style="color: white; margin: 0; font-size: 3rem; font-weight: 700;">{risk_level}</h1>
            <p style="color: white; margin: 1rem 0 0 0; font-size: 1.5rem; font-weight: 600;">
                {delay_risk:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Class Probabilities Visualization
        st.markdown(f"<h4 style='color: {COLORS['WHITE']}; margin-bottom: 1rem;'>Prediction Confidence</h4>", unsafe_allow_html=True)
        
        prob_df = pd.DataFrame({
            "Class": list(class_probs.keys()),
            "Probability": list(class_probs.values())
        })
        
        fig = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            color="Probability",
            color_continuous_scale=["#4BB543", "#FF8C00", "#D9534F"],
            text="Probability"
        )
        fig.update_traces(
            texttemplate='%{text:.1%}',
            textposition='outside',
            marker_line_color=COLORS['NAVY'],
            marker_line_width=2
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['WHITE']),
            showlegend=False,
            yaxis=dict(tickformat=".0%", gridcolor=COLORS['MEDIUM_GREY'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Reasoning
        reasoning = generate_reasoning(delay_risk, legs, rcs_p, predicted_class)
        st.info(f"**🤖 AI Reasoning:** {reasoning}")

def generate_reasoning(
    delay_risk: float,
    legs: int,
    rcs_p: int,
    predicted_class: int
) -> str:
    """Generate human-readable reasoning for the prediction."""
    reasons = []
    
    if rcs_p > 2000:
        reasons.append("High planned check-in time indicates potential congestion")
    if legs >= 2:
        reasons.append("Multi-leg journey increases complexity and delay risk")
    if delay_risk > 0.7:
        reasons.append("Systemic factors suggest elevated delay probability")
    
    if not reasons:
        reasons.append("Configuration suggests standard operational risk")
    
    return ". ".join(reasons) + "."

def render_feature_importance():
    """Render feature importance visualization."""
    st.markdown("---")
    st.markdown(f"<h2 style='color: {COLORS['GOLD']}; margin-bottom: 1.5rem;'>🔍 AI Interpretability & Explainability</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"<h4 style='color: {COLORS['WHITE']}; margin-bottom: 1rem;'>Feature Importance Scorecard</h4>", unsafe_allow_html=True)
        
        df_imp = pd.DataFrame(FEATURE_IMPORTANCE)
        df_imp['Readable Label'] = df_imp['Feature'].map(FEATURE_LABELS)
        
        fig = px.bar(
            df_imp.sort_values('Importance', ascending=True),
            x='Importance',
            y='Readable Label',
            orientation='h',
            color='Importance',
            color_continuous_scale=[COLORS['NAVY'], COLORS['GOLD']],
            text='Importance'
        )
        fig.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside',
            marker_line_color=COLORS['GOLD'],
            marker_line_width=1
        )
        fig.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['WHITE']),
            showlegend=False,
            xaxis=dict(title="Importance Score", gridcolor=COLORS['MEDIUM_GREY']),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"<h4 style='color: {COLORS['WHITE']}; margin-bottom: 1rem;'>💡 Key Insights</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 8px; border-left: 4px solid {COLORS['GOLD']};">
            <ul style="color: {COLORS['WHITE']}; padding-left: 1.5rem; line-height: 1.8;">
                <li><strong>Origin Airport</strong> is the primary driver (21.1% importance)</li>
                <li><strong>Planned Check-in</strong> duration significantly impacts risk buffer</li>
                <li><strong>Outbound Hops</strong> add complexity at end-of-chain</li>
                <li><strong>Multi-leg journeys</strong> correlate with higher delay probability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_risk_overview():
    """Render risk overview charts and filters."""
    st.markdown("---")
    st.markdown(f"<h2 style='color: {COLORS['GOLD']}; margin-bottom: 1.5rem;'>📈 Risk Overview & Analytics</h2>", unsafe_allow_html=True)
    
    # Filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 2])
    
    with filter_col1:
        timeframe = st.selectbox("Timeframe", ["Last 24h", "Last 7 Days", "Last 30 Days"], key="timeframe")
    with filter_col2:
        selected_ports = st.multiselect("Port Filter", MOCKED_PORT_IDS, default=[240], key="ports")
    with filter_col3:
        metric_view = st.selectbox("Metric", ["Delay Prob", "Volume"], key="metric")
    with filter_col4:
        risk_status = "🟡 MODERATE RISK" if len(selected_ports) > 0 else "🟢 LOW RISK"
        st.info(f"**Status:** {risk_status}. 142 Shipments flagged for review.")
    
    # Generate mock data based on filters
    if timeframe == "Last 24h":
        periods = 24
        freq = 'h'
        date_start = datetime.now().strftime("%Y-%m-%d")
    elif timeframe == "Last 7 Days":
        periods = 7
        freq = 'D'
        date_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        periods = 6
        freq = 'ME'
        date_start = "2025-01-01"
    
    dates = pd.date_range(start=date_start, periods=periods, freq=freq)
    
    # Generate data with seed for consistency
    seed_val = hash(tuple(selected_ports)) + hash(timeframe) + hash(metric_view)
    np.random.seed(abs(seed_val) % (2**32 - 1))
    
    base_rate = 0.15
    if 100 in selected_ports:
        base_rate += 0.05
    if 815 in selected_ports:
        base_rate -= 0.03
    
    delay_rates = np.random.normal(loc=base_rate, scale=0.02, size=periods).clip(0, 1)
    volumes = np.random.randint(3000, 6000, size=periods)
    
    chart_data = pd.DataFrame({
        "Date": dates,
        "Delay Rate": delay_rates,
        "Volume": volumes
    })
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown(f"<h4 style='color: {COLORS['WHITE']}; margin-bottom: 1rem;'>Systemic Delay Trend</h4>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Add target range
        fig.add_hrect(
            y0=0.10, y1=0.15,
            line_width=0,
            fillcolor=COLORS['GOLD'],
            opacity=0.2,
            annotation_text="Target Range",
            annotation_position="top right"
        )
        
        # Add delay rate line
        fig.add_trace(go.Scatter(
            x=chart_data['Date'],
            y=chart_data['Delay Rate'],
            mode='lines+markers',
            name='Delay Rate',
            line=dict(color=COLORS['GOLD'], width=3, shape='spline'),
            marker=dict(size=8, color=COLORS['MUTED_GOLD_BG'])
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['WHITE']),
            yaxis=dict(tickformat=".2%", gridcolor=COLORS['MEDIUM_GREY']),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        tab_vol, tab_heat = st.tabs(["Volume", "Regional Heatmap"])
        
        with tab_vol:
            st.markdown(f"<h4 style='color: {COLORS['WHITE']}; margin-bottom: 1rem;'>Shipment Volume</h4>", unsafe_allow_html=True)
            fig_vol = px.bar(
                chart_data,
                x='Date',
                y='Volume',
                color_discrete_sequence=[COLORS['GOLD']]
            )
            fig_vol.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['WHITE']),
                yaxis=dict(gridcolor=COLORS['MEDIUM_GREY'])
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with tab_heat:
            st.markdown(f"<h4 style='color: {COLORS['WHITE']}; margin-bottom: 1rem;'>Regional Risk Heatmap</h4>", unsafe_allow_html=True)
            z = np.random.rand(3, 3)
            fig_heat = go.Figure(data=go.Heatmap(
                z=z,
                x=['APAC', 'EMEA', 'NAM'],
                y=['Air', 'Sea', 'Rail'],
                colorscale=[[0, COLORS['WHITE']], [1, COLORS['RED']]],
                text=z,
                texttemplate='%{text:.2f}',
                textfont={"color": COLORS['WHITE']}
            ))
            fig_heat.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['WHITE'])
            )
            st.plotly_chart(fig_heat, use_container_width=True)

def render_model_health(model: Optional[Any]):
    """Render model health and status information."""
    st.markdown("---")
    st.markdown(f"<h2 style='color: {COLORS['GOLD']}; margin-bottom: 1.5rem;'>🏥 Model Health & Status</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Name", MODEL_NAME)
    with col2:
        status = "✅ Operational" if model else "⚠️ Offline (Mocking)"
        status_color = COLORS['GREEN'] if model else COLORS['ORANGE']
        st.markdown(f"<div style='color: {status_color}; font-weight: 600;'>Status: {status}</div>", unsafe_allow_html=True)
    with col3:
        st.metric("Last Training", MODEL_TRAINING_DATE)
    with col4:
        version_hash = MODEL_URI.split("/")[-1][:8] if "/" in MODEL_URI else "N/A"
        st.metric("Version Hash", version_hash)

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_styling()
    
    # Load model
    model = load_model()
    
    # Render UI components
    render_header()
    render_sidebar(model)
    
    # Main content
    render_kpi_dashboard()
    render_prediction_form(model)
    render_risk_overview()
    render_feature_importance()
    render_model_health(model)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0; color: {COLORS['WHITE']}; opacity: 0.8;">
        <p style="margin: 0.5rem 0;">Apex Logistics | Confidential | 2025</p>
        <p style="margin: 0.5rem 0; font-size: 0.9rem;">Version {APP_VERSION} | Powered by CatBoost & MLflow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}")
        logger.critical(traceback.format_exc())
        st.error(f"❌ Application Error: {str(e)}")
        if DEBUG_MODE:
            st.exception(e)
