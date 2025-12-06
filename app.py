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
    "MUTED_GOLD_BG": "#FFF8DC", # New Muted Gold for Header/Card Background
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

# Actual Feature Importance (From your model)
FEATURE_IMPORTANCE = {
    "Feature": ["i1_dep_1_place", "i1_rcs_p", "o_dep_1_place", "i1_hops", "o_hops", "legs"],
    "Importance": [0.211, 0.187, 0.142, 0.095, 0.078, 0.055]
}

MOCKED_PORT_IDS = [100, 240, 560, 815]

# ==============================================================================
# 2. SETUP & STYLING
# ==============================================================================
st.set_page_config(
    page_title="LogiChain Executive Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Custom CSS for Strict Design Adherence
st.markdown(f"""
    <style>
    /* VARIABLES */
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

    /* GLOBAL TYPOGRAPHY & RESET */
    /* Set the main background to NAVY as requested */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', sans-serif;
        background-color: var(--muted-gold) !important;
        color: var(--navy) !important;
    }}

    /* Remove default Streamlit top padding to fix "the top" issue */
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
    }}

    /* Hide default Streamlit header/menu for cleaner look */
    header[data-testid="stHeader"] {{
        display: none;
    }}

    /* MAIN CONTAINER BOX */
    /* Set the main container (the app card) to NAVY */
    .block-container {{
        background-color: var(--navy);
        padding: 3rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        margin-top: 2rem;
        margin-bottom: 2rem;
        max-width: 95% !important;
        border: 2px solid var(--gold);
    }}

    /* Text Color Overrides for Dark Background (Navy Container) */
    /* Force text to White/Gold inside the Navy card */
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
    [data-testid="stMetricLabel"] {{
        color: var(--white) !important;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--gold) !important;
    }}

    /* EXECUTIVE DECISION BAR (White Box inside Navy Container) */
    .exec-bar {{
        background-color: var(--white);
        border: 2px solid var(--gold);
        border-radius: 8px;
        padding: 20px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}

    /* Text inside Executive Bar (White Background -> Navy Text) */
    .exec-bar h3 {{
        color: var(--navy) !important; /* Header inside white box is Navy */
    }}

    /* Ensure metrics inside the exec bar look correct */
    .exec-bar [data-testid="stMetricValue"] {{
        font-size: 32px !important;
        color: var(--navy) !important; /* Metric value is Navy */
    }}

    .exec-bar [data-testid="stMetricLabel"] {{
        color: var(--gold) !important; /* Label is Gold */
    }}

    /* COMPONENT OVERRIDES */
    .stButton > button {{
        background-color: var(--gold);
        color: var(--navy) !important; /* Force Navy text on Gold buttons */
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: var(--white);
        color: var(--navy) !important;
    }}

    /* CARD STYLE - Applied ONLY to structural columns inside the main container */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {{
        background-color: var(--white);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid var(--gold);
    }}

    /* Fix text color inside the white cards (back to Navy) */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div p,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div h1,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div h2,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div h3,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div span,
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div li {{
        color: var(--navy) !important;
    }}

    /* Remove background from individual markdown elements */
    .stMarkdown {{
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}

    /* Fix tabs text color */
    button[data-baseweb="tab"] {{
        color: var(--white) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--gold) !important;
        border-color: var(--gold) !important;
    }}
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
    """1) Executive Decision Bar (White Box inside Navy Container)"""

    # Using HTML/Flexbox allows us to wrap the entire row in the background color
    html_code = f"""
    <div class="exec-bar">
        <h3 style="margin-top: 0; margin-bottom: 20px; font-weight: 700;">
            Executive Decision Bar
        </h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between;">
            <!-- Metric 1 -->
            <div style="flex: 1; min-width: 140px; text-align: center;">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 4px; color: {COLORS['GOLD']} !important;">On-Time Rate (QTD)</div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 28px; font-weight: 700; color: {COLORS['NAVY']};">88.6%</div>
                <div style="color: {COLORS['GREEN']}; font-weight: 600; font-size: 14px;">▲ 2.1%</div>
            </div>
            <!-- Metric 2 -->
            <div style="flex: 1; min-width: 140px; text-align: center; border-left: 1px solid rgba(218, 165, 32, 0.3);">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 4px; color: {COLORS['GOLD']} !important;">Shipments At Risk</div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 28px; font-weight: 700; color: {COLORS['NAVY']};">142</div>
                <div style="color: {COLORS['RED']}; font-weight: 600; font-size: 14px;">▼ -5%</div>
            </div>
            <!-- Metric 3 -->
            <div style="flex: 1; min-width: 140px; text-align: center; border-left: 1px solid rgba(218, 165, 32, 0.3);">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 4px; color: {COLORS['GOLD']} !important;">Avg Planned Time</div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 28px; font-weight: 700; color: {COLORS['NAVY']};">1550m</div>
                <div style="color: {COLORS['GREEN']}; font-weight: 600; font-size: 14px;">▼ -12m</div>
            </div>
            <!-- Metric 4 -->
            <div style="flex: 1; min-width: 140px; text-align: center; border-left: 1px solid rgba(218, 165, 32, 0.3);">
                <div style="font-size: 14px; font-weight: 600; opacity: 0.8; margin-bottom: 4px; color: {COLORS['GOLD']} !important;">Active Shipments</div>
                <div style="font-family: 'Roboto Mono', monospace; font-size: 28px; font-weight: 700; color: {COLORS['NAVY']};">3,942</div>
                <div style="color: {COLORS['GOLD']}; font-weight: 600; font-size: 14px;">● 124 new</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def risk_filters_and_charts():
    """2) Risk Overview Section & Dynamic Charts"""
    st.markdown("#### Risk Overview & Filtration")

    # --- FILTERS ---
    with st.container():
        f_col1, f_col2, f_col3, f_summary = st.columns([1, 1, 1, 2])

        with f_col1:
            timeframe = st.selectbox("Timeframe", ["Last 24h", "Last 7 Days", "Last 30 Days"])
        with f_col2:
            # Multi-select returns a list
            selected_ports = st.multiselect("Port Filter", MOCKED_PORT_IDS, default=[240])
        with f_col3:
            metric_view = st.selectbox("Metric", ["Delay Prob", "Volume"])

        with f_summary:
            st.info("🟡 **Status:** MODERATE RISK. 142 Shipments flagged for review.")

    # --- DYNAMIC DATA GENERATION BASED ON FILTERS ---
    # Determine number of data points based on timeframe
    if timeframe == "Last 24h":
        periods = 24
        freq = 'h'
        date_start = datetime.now().strftime("%Y-%m-%d")
        x_label = "Hour"
    elif timeframe == "Last 7 Days":
        periods = 7
        freq = 'D'
        date_start = "2025-01-01"
        x_label = "Date"
    else: # Last 30 Days / Monthly default
        periods = 6
        freq = 'ME'
        date_start = "2025-01-01"
        x_label = "Date"

    dates = pd.date_range(start=date_start, periods=periods, freq=freq)

    # Generate random data that "reacts" to filters
    # FIX: Use specific hashing of values so changes trigger visual updates
    seed_val = hash(tuple(selected_ports)) + hash(timeframe) + hash(metric_view)
    # Ensure seed is positive and fits within 32-bit integer range
    np.random.seed(abs(seed_val) % (2**32 - 1))

    base_rate = 0.15
    if 100 in selected_ports: base_rate += 0.05 # Mock logic: Port 100 is slower
    if 815 in selected_ports: base_rate -= 0.03 # Mock logic: Port 815 is faster

    delay_rates = np.random.normal(loc=base_rate, scale=0.02, size=periods).clip(0, 1)
    volumes = np.random.randint(3000, 6000, size=periods)

    chart_data = pd.DataFrame({
        "Date": dates,
        "Delay Rate": delay_rates,
        "Volume": volumes
    })

    # --- CHARTS ROW ---
    c1, c2 = st.columns(2)

    # 3) Delay Trend Chart (Plotly)
    with c1:
        st.markdown("#### Systemic Delay Trend")
        fig = go.Figure()

        # Gold Acceptable Range Band
        fig.add_hrect(
            y0=0.10, y1=0.15,
            line_width=0, fillcolor=COLORS['GOLD'], opacity=0.2,
            annotation_text="Target Range", annotation_position="top right"
        )

        # CHANGED: Navy Line, Muted Gold Points (using standard Gold #DAA520 for visibility)
        fig.add_trace(go.Scatter(
            x=chart_data['Date'],
            y=chart_data['Delay Rate'],
            mode='lines+markers',
            name='Delay Rate',
            line=dict(color=COLORS['GOLD'], width=3, shape='spline'),
            marker=dict(size=8, color=COLORS['MUTED_GOLD_BG'])
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat=".2%", gridcolor=COLORS['MEDIUM_GREY']),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 4) Shipment Volume & Heatmap
    with c2:
        # Using Tabs here to save visual space within the column
        tab_vol, tab_heat = st.tabs(["Volume", "Regional Heatmap"])

        with tab_vol:
            fig_vol = px.bar(
                chart_data, x='Date', y='Volume',
                color_discrete_sequence=[COLORS['NAVY']]
            )
            fig_vol.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab_heat:
            # Mock Heatmap Data that changes slightly with filters
            z = np.random.rand(3, 3)
            fig_heat = go.Figure(data=go.Heatmap(
                z=z, x=['APAC', 'EMEA', 'NAM'], y=['Air', 'Sea', 'Rail'],
                colorscale=[[0, 'white'], [1, COLORS['RED']]]
            ))
            fig_heat.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)

def feature_importance_section():
    """5) AI Interpretability"""
    st.markdown("---")
    st.markdown("### AI Interpretability & Explainability")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("#### Feature Importance Scorecard")
        df_imp = pd.DataFrame(FEATURE_IMPORTANCE)
        df_imp['Readable Label'] = df_imp['Feature'].map(FEATURE_LABELS)

        fig = px.bar(
            df_imp.sort_values('Importance', ascending=True),
            x='Importance', y='Readable Label', orientation='h',
            # Gold bars for high contrast
            color_discrete_sequence=[COLORS['GOLD']]
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Insight Summary")
        # CHANGED: Text color explicitly set to NAVY for visibility on white card
        st.markdown(f"""
        <div style="background-color: 'NAVY'; border: 1px solid {COLORS['MEDIUM_GREY']}; padding: 20px; border-radius: 8px;">
            <ul style="padding-left: 20px; color: {COLORS['BLACK']} !important;">
                <li><b>Origin Airport</b> is the primary driver (21.1%).</li>
                <li><b>Planned Check-in</b> duration dictates risk buffer.</li>
                <li><b>Outbound Hops</b> add complexity at end-of-chain.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def model_health():
    """6) Model Health Section"""
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    # Use standard text color variables for high contrast (white/gold on navy)
    with c1:
        st.markdown(f"**Model Name:**")
        st.markdown(f'<p style="color: {COLORS["BLACK"]}; font-weight: 500;">CatBoost_Delay_Classifier_v2</p>', unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Status:**")
        status_color = COLORS['GREEN'] if active_model else COLORS['RED']
        status = "Operational" if active_model else "Offline (Mocking)"
        st.markdown(f'<p style="color: {status_color}; font-weight: 500;">{status}</p>', unsafe_allow_html=True)
    with c3:
        st.markdown(f"**Last Training:**")
        st.markdown(f'<p style="color: {COLORS["BLACK"]}; font-weight: 500;">2025-02-14</p>', unsafe_allow_html=True)
    with c4:
        st.markdown(f"**Version Hash:**")
        st.markdown(f'<p style="color: {COLORS["BLACK"]}; font-weight: 500;">{MODEL_URI.split("/")[-1][:8]}</p>', unsafe_allow_html=True)

def prediction_module():
    """Executive Risk Assessment Module (Manual Prediction)"""
    st.markdown("---")
    st.subheader("⚡ Manual Risk Assessment")

    # We use a container with a border to visually group this tool
    with st.container():
        with st.form("exec_pred_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                legs = st.number_input("Total Journey Legs", 1, 3, 1)
                rcs = st.number_input("Planned Check-In (min)", 0, 5000, 1500)
            with c2:
                i_hops = st.number_input("Inbound Hops", 1, 4, 2)
                o_hops = st.number_input("Outbound Hops", 1, 4, 2)
            with c3:
                dep1 = st.selectbox("Origin Airport ID", [100, 540, 815], index=1)
                dep2 = st.selectbox("Outbound Airport ID", [100, 540, 815], index=2)

            submit = st.form_submit_button("CALCULATE RISK SCORE")

    if submit:
        raw_inputs = [legs, rcs, i_hops, o_hops, dep1, dep2]
        padded_inputs = raw_inputs + [0.0] * 246

        input_data = pd.DataFrame([padded_inputs])
        input_data.columns = [f"f_{i}" for i in range(252)]

        pred_prob = 0.78
        is_delayed = True

        if active_model:
            try:
                probs = active_model.predict_proba(input_data)[0]
                pred_prob = probs[1]
                is_delayed = pred_prob > 0.5
            except Exception as e:
                st.error(f"Model Error: {e}")

        r1, r2 = st.columns([1, 2])
        with r1:
            color = COLORS['RED'] if is_delayed else COLORS['GREEN']
            label = "HIGH RISK" if is_delayed else "LOW RISK"
            st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                    <h2 style="color: white; margin:0;">{label}</h2>
                    <h1 style="color: white; margin:0;">{pred_prob:.1%}</h1>
                </div>
            """, unsafe_allow_html=True)
        with r2:
            st.info("**AI Reasoning:** High planned check-in time combined with specific Origin Airport ID indicates systemic congestion.")

# ==============================================================================
# 5. MAIN APP LAYOUT
# ==============================================================================
def main():
    kpi_header_section()

    # Combined Filters and Charts into one reactive function
    risk_filters_and_charts()

    prediction_module()
    feature_importance_section()
    model_health()

    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: {COLORS['WHITE']};'>LogiChain Analytics | Confidential | 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
