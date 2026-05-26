import streamlit as st
import subprocess
import sys
import json
import threading
import queue
import time
import pandas as pd
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ADR / MRMT Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"   # 👈 IMPORTANTE
)

# =========================================================
# MODERN CSS FIX (RESPONSIVE + NO TOP GAP)
# =========================================================

st.markdown("""
<style>

/* =====================================================
   REMOVE TOP SPACING (RENDER FIX)
===================================================== */

header {
    visibility: visible !important;
    height: auto !important;
}


.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
}

/* =====================================================
   MAIN LAYOUT
===================================================== */

.main .block-container{
    max-width: 1600px;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* =====================================================
   TITLE FIX (reduce ugly margin)
===================================================== */

h1 {
    margin-top: 0rem !important;
    margin-bottom: 0.2rem !important;
    font-size: 2.1rem !important;
}

/* =====================================================
   LABELS
===================================================== */

label {
    font-size: 0.90rem !important;
    font-weight: 500 !important;
}

/* =====================================================
   INPUTS
===================================================== */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}

/* =====================================================
   BUTTONS
===================================================== */

.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
    white-space: nowrap;   /* 👈 evita salto de línea */
    overflow: hidden;
    text-overflow: ellipsis;
}
}

/* =====================================================
   SIDEBAR FIX
===================================================== */

section[data-testid="stSidebar"] {
    width: 320px !important;
}

/* Sidebar content spacing */
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* =====================================================
   METRICS
===================================================== */

[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 0.8rem;
}

/* =====================================================
   REDUCE VERTICAL SPACING
===================================================== */

.element-container {
    margin-bottom: 0.3rem !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION DEFAULTS
# =========================================================

DEFAULTS = {
    "spacing": 30.0,
    "dt": 36000.0,
    "T": 2592000.0,
    "phi_const": 0.1,
    "a_l": 10.0,
    "a_t": 1.0,
    "D_d": 1e-19,
    "eps": 1e-16,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# TITLE
# =========================================================

st.title("🌊 ADR / MRMT Transport Simulator")
st.caption("Reactive transport simulation platform")

# =========================================================
# SIDEBAR (ALL CLOSED BY DEFAULT)
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    with st.expander("General", expanded=False):   # 👈 CLOSED
        metodo = st.selectbox("Numerical Method", ["rbf", "fenicsx"])
        model = st.selectbox("Transport Model", ["adr", "mrmt_semi", "mrmt_block"])
        domain = st.selectbox("Domain Type", ["real", "benchmark"])
        run_type = st.selectbox("Run Type", ["standard", "optimization"])

    with st.expander("Execution", expanded=False):  # 👈 CLOSED
        pre = st.checkbox("Preprocessing")
        postproc = st.checkbox("Postprocessing")
        plot_grid = st.checkbox("Plot Grid")

    with st.expander("Physical Processes", expanded=False):  # 👈 CLOSED
        activate_fuente = st.checkbox("Enable Source", value=True)
        activate_ext = st.checkbox("Enable Extraction", value=True)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧮 Numerical",
    "🌊 Physics",
    "🧩 MRMT",
    "💾 Output",
    "🚀 Run"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Numerical Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        spacing = st.number_input("Spacing", min_value=1e-12, value=st.session_state.spacing)

    with col2:
        dt = st.number_input("dt", min_value=1e-12, value=st.session_state.dt)

    with col3:
        T = st.number_input("Total Time", min_value=1e-12, value=st.session_state.T)

    with col4:
        eps = st.number_input("Tolerance", min_value=1e-30, value=st.session_state.eps, format="%.2e")

    Nt = int(T / dt)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("dt", f"{dt:.2e}")
    c2.metric("T", f"{T:.2e}")
    c3.metric("Spacing", f"{spacing:.2f}")
    c4.metric("Timesteps", f"{Nt:,}")

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("Physical Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        phi_const = st.number_input("Porosity", 0.0, 1.0, st.session_state.phi_const)

    with col2:
        a_l = st.number_input("a_l", 0.0, st.session_state.a_l)

    with col3:
        a_t = st.number_input("a_t", 0.0, st.session_state.a_t)

    with col4:
        D_d = st.number_input("Molecular Diffusion", 0.0, st.session_state.D_d, format="%.2e")

# =========================================================
# TAB 3
# =========================================================

with tab3:

    if "mrmt" in model:
        st.subheader("MRMT Parameters")

        Nr = st.number_input("Number of Immobile Regions", min_value=1, value=3)

        default_df = pd.DataFrame({
            "Deff": [1e-9, 5e-10, 1e-10],
            "beta": [0.15, 0.10, 0.05],
            "phi_im": [0.1, 0.05, 0.02]
        })

        mrmt_df = st.data_editor(default_df, num_rows="dynamic", height=260)

    else:
        Nr = 0
        mrmt_df = pd.DataFrame()
        st.info("MRMT disabled for ADR model.")

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader("Output Settings")

    col1, col2 = st.columns(2)

    with col1:
        save_dat = st.checkbox("Save Simulation Data", value=True)

    with col2:
        animate = st.checkbox("Generate Animation", value=True)

    uploaded_file = st.file_uploader("Upload Custom Domain", type=["vtk", "msh", "csv"])

# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.subheader("Simulation Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model)
    c2.metric("Method", metodo)
    c3.metric("Domain", domain)
    c4.metric("Run", run_type)

    st.divider()

    left, center, right = st.columns([3, 2, 1])

    with right:
        run_button = st.button("🚀 Run Simulation")

# =========================================================
# EXECUTION (simplificado)
# =========================================================

if run_button:

    output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)

    config = {
        "method": metodo,
        "model": model,
        "domain": domain,
        "run_type": run_type,
        "spacing": spacing,
        "dt": dt,
        "T": T,
        "phi_const": phi_const,
        "a_l": a_l,
        "a_t": a_t,
        "D_d": D_d,
        "eps": eps,
        "save_dat": save_dat,
        "animate": animate
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    st.success("Config generated (execution pipeline ready)")