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
    initial_sidebar_state="collapsed"  # ✅ FIX: sidebar colapsada por defecto
)

# =========================================================
# CSS (RESPONSIVE + SPACING FIX)
# =========================================================

st.markdown("""
<style>

/* ================= GLOBAL SPACING FIX ================= */

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
}

/* ================= TITLE ================= */

h1 {
    margin-top: 0rem !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.2 !important;
}

/* ================= HEADER ================= */

header {
    padding-top: 0rem !important;
}

/* ================= RESPONSIVE ================= */

@media (max-width: 768px){
    .main .block-container{
        max-width: 100% !important;
        padding: 0.8rem !important;
    }

    .stButton > button{
        width: 100% !important;
        min-width: unset !important;
    }
}

/* ================= BUTTON FIX ================= */

.stButton > button {
    width: auto !important;
    min-width: 180px;
    height: 44px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

/* ================= INPUTS ================= */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

div[data-baseweb="input"] input {
    font-size: 0.92rem;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    width: 320px !important;
}

/* FIX: toggle visible siempre */
button[kind="header"] {
    z-index: 999999 !important;
}

/* ================= METRICS ================= */

[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 0.8rem;
}

/* ================= SPACING ================= */

.element-container {
    margin-bottom: 0.4rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE DEFAULTS
# =========================================================

st.markdown("""
<script>
const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
const btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');

if (sidebar && btn) {
    setTimeout(() => {
        btn.click();
    }, 50);
}
</script>
""", unsafe_allow_html=True)

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
# INIT STATE
# =========================================================

run_button = False

# =========================================================
# TITLE
# =========================================================

st.title("🌊 ADR / MRMT Transport Simulator")
st.caption("Reactive transport simulation platform")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    with st.expander("General", expanded=False):
        metodo = st.selectbox("Numerical Method", ["rbf", "fenicsx"])
        model = st.selectbox("Transport Model", ["adr", "mrmt_semi", "mrmt_block"])
        domain = st.selectbox("Domain Type", ["real", "benchmark"])
        run_type = st.selectbox("Run Type", ["standard", "optimization"])

    with st.expander("Execution", expanded=False):
        pre = st.checkbox("Preprocessing")
        postproc = st.checkbox("Postprocessing")
        plot_grid = st.checkbox("Plot Grid")

    with st.expander("Physical Processes", expanded=False):
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

    spacing = col1.number_input("Spacing", value=st.session_state.spacing)
    dt = col2.number_input("dt", value=st.session_state.dt)
    T = col3.number_input("Total Time", value=st.session_state.T)
    eps = col4.number_input("Tolerance", value=st.session_state.eps, format="%.2e")

    Nt = int(T / dt)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("dt", f"{dt:.2e}")
    m2.metric("T", f"{T:.2e}")
    m3.metric("Spacing", spacing)
    m4.metric("Timesteps", f"{Nt:,}")

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("Physical Parameters")

    col1, col2, col3, col4 = st.columns(4)

    phi_const = col1.number_input("Porosity", value=st.session_state.phi_const)
    a_l = col2.number_input("a_l", value=st.session_state.a_l)
    a_t = col3.number_input("a_t", value=st.session_state.a_t)
    D_d = col4.number_input("Diffusion", value=st.session_state.D_d, format="%.2e")

# =========================================================
# TAB 3
# =========================================================

with tab3:

    if "mrmt" in model:

        Nr = st.number_input("Nr regions", value=3)

        mrmt_df = st.data_editor(pd.DataFrame({
            "Deff": [1e-9, 5e-10, 1e-10],
            "beta": [0.15, 0.10, 0.05],
            "phi_im": [0.1, 0.05, 0.02]
        }))

    else:
        Nr = 0
        mrmt_df = pd.DataFrame()
        st.info("MRMT disabled")

# =========================================================
# TAB 4
# =========================================================

with tab4:

    save_dat = st.checkbox("Save Data", value=True)
    animate = st.checkbox("Animate", value=True)

    uploaded_file = st.file_uploader("Upload domain", type=["vtk", "msh", "csv"])

# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model)
    c2.metric("Method", metodo)
    c3.metric("Domain", domain)
    c4.metric("Run", run_type)

    errors = []

    if T <= dt:
        errors.append("T must be > dt")

    if spacing <= 0:
        errors.append("Spacing invalid")

    if not errors:
        st.success("Configuration valid")

        _, _, right = st.columns([1,2,1])
        with right:
            run_button = st.button("🚀 Run Simulation")

    else:
        for e in errors:
            st.error(e)
        run_button = False

# =========================================================
# EXECUTION
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

    progress_bar = st.progress(0)
    status_box = st.empty()
    log_container = st.empty()

    status_box.info("Running simulation...")

    process = subprocess.Popen(
        [sys.executable, "run.py", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    logs = ""

    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            logs += line
            log_container.code(logs)

        time.sleep(0.05)

    stderr = process.stderr.read()

    if stderr:
        status_box.error("Failed")
        st.code(stderr)
    else:
        status_box.success("Done")
        st.balloons()