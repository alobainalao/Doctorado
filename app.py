import streamlit as st
import subprocess
import sys
import os
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
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   MAIN LAYOUT
===================================================== */

.main .block-container{
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
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

div[data-baseweb="select"] > div {
    min-height: 42px;
    border-radius: 10px;
}

div[data-baseweb="input"] > div {
    border-radius: 10px;
}

div[data-baseweb="input"] input {
    font-size: 0.92rem;
}

/* =====================================================
   BUTTONS
===================================================== */

.stButton > button {
    width: auto !important;
    min-width: 200px;
    border-radius: 10px;
    height: 44px;
    font-weight: 600;
    font-size: 0.95rem;
}

/* =====================================================
   CHECKBOX
===================================================== */

.stCheckbox {
    margin-bottom: 0.2rem;
}

/* =====================================================
   TABS
===================================================== */

button[data-baseweb="tab"] {
    font-size: 0.95rem;
    padding-top: 0.6rem;
    padding-bottom: 0.6rem;
}

/* =====================================================
   EXPANDERS
===================================================== */

.streamlit-expanderHeader {
    font-size: 0.95rem;
    font-weight: 600;
}

/* =====================================================
   CODE
===================================================== */

pre {
    border-radius: 12px !important;
}

/* =====================================================
   DATA EDITOR
===================================================== */

[data-testid="stDataEditor"] {
    border-radius: 12px;
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
    margin-bottom: 0.35rem;
}

/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    width: 320px !important;
}

/* =====================================================
   HEADERS
===================================================== */

h1 {
    margin-bottom: 0.3rem;
}

h2, h3 {
    margin-top: 0.3rem;
}

/* =====================================================
   INFO BOXES
===================================================== */

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE DEFAULTS
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    with st.expander("General", expanded=True):

        metodo = st.selectbox(
            "Numerical Method",
            ["rbf", "fenicsx"]
        )

        model = st.selectbox(
            "Transport Model",
            ["adr", "mrmt_semi", "mrmt_block"]
        )

        domain = st.selectbox(
            "Domain Type",
            ["real", "benchmark"]
        )

        run_type = st.selectbox(
            "Run Type",
            ["standard", "optimization"]
        )

    with st.expander("Execution", expanded=True):

        pre = st.checkbox("Preprocessing")
        postproc = st.checkbox("Postprocessing")
        plot_grid = st.checkbox("Plot Grid")

    with st.expander("Physical Processes", expanded=True):

        activate_fuente = st.checkbox(
            "Enable Source",
            value=True
        )

        activate_ext = st.checkbox(
            "Enable Extraction",
            value=True
        )

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧮 Numerical",
    "🌊 Physics",
    "🧩 MRMT",
    "💾 Output",
    "🚀 Run"
])

# =========================================================
# TAB 1 - NUMERICAL
# =========================================================

with tab1:

    st.subheader("Numerical Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        spacing = st.number_input(
            "Spacing",
            min_value=1e-12,
            value=st.session_state.spacing,
            step=1.0
        )

    with col2:

        dt = st.number_input(
            "dt",
            min_value=1e-12,
            value=st.session_state.dt
        )

    with col3:

        T = st.number_input(
            "Total Time",
            min_value=1e-12,
            value=st.session_state.T
        )

    with col4:

        eps = st.number_input(
            "Tolerance",
            min_value=1e-30,
            value=st.session_state.eps,
            format="%.2e"
        )

    Nt = int(T / dt)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("dt", f"{dt:.2e}")
    m2.metric("T", f"{T:.2e}")
    m3.metric("Spacing", spacing)
    m4.metric("Timesteps", f"{Nt:,}")

# =========================================================
# TAB 2 - PHYSICS
# =========================================================

with tab2:

    st.subheader("Physical Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        phi_const = st.number_input(
            "Porosity",
            min_value=1e-8,
            max_value=1.0,
            value=st.session_state.phi_const
        )

    with col2:

        a_l = st.number_input(
            "a_l",
            min_value=0.0,
            value=st.session_state.a_l
        )

    with col3:

        a_t = st.number_input(
            "a_t",
            min_value=0.0,
            value=st.session_state.a_t
        )

    with col4:

        D_d = st.number_input(
            "Molecular Diffusion",
            min_value=0.0,
            value=st.session_state.D_d,
            format="%.2e"
        )

    with st.expander("Advanced Physics"):

        st.write(
            "Additional physical models can be added here."
        )

# =========================================================
# TAB 3 - MRMT
# =========================================================

with tab3:

    if "mrmt" in model:

        st.subheader("MRMT Parameters")

        Nr = st.number_input(
            "Number of Immobile Regions",
            min_value=1,
            value=3
        )

        default_df = pd.DataFrame({
            "Deff": [1e-9, 5e-10, 1e-10],
            "beta": [0.15, 0.10, 0.05],
            "phi_im": [0.1, 0.05, 0.02]
        })

        mrmt_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            height=220,
            use_container_width=False
        )

    else:

        Nr = 0

        mrmt_df = pd.DataFrame()

        st.info(
            "MRMT parameters disabled for ADR model."
        )

# =========================================================
# TAB 4 - OUTPUT
# =========================================================

with tab4:

    st.subheader("Output Settings")

    col1, col2 = st.columns(2)

    with col1:

        save_dat = st.checkbox(
            "Save Simulation Data",
            value=True
        )

    with col2:

        animate = st.checkbox(
            "Generate Animation",
            value=True
        )

    uploaded_file = st.file_uploader(
        "Upload Custom Domain",
        type=["vtk", "msh", "csv"]
    )

# =========================================================
# TAB 5 - RUN
# =========================================================

with tab5:

    st.subheader("Simulation Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", model)
    c2.metric("Method", metodo)
    c3.metric("Domain", domain)
    c4.metric("Run", run_type)

    st.divider()

    errors = []

    if T <= dt:
        errors.append("T must be greater than dt.")

    if spacing <= 0:
        errors.append("Spacing must be positive.")

    if phi_const <= 0 or phi_const > 1:
        errors.append("Porosity must be in (0,1].")

    if "mrmt" in model:

        if len(mrmt_df) != Nr:

            errors.append(
                "MRMT rows must equal Nr."
            )

    if errors:

        for err in errors:
            st.error(err)

        run_button = False

    else:

        st.success("Configuration valid.")

        _, center_col, _ = st.columns([1, 2, 1])

        with center_col:

            run_button = st.button(
                "🚀 Run Simulation"
            )

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

        "pre": pre,
        "postproc": postproc,
        "plot_grid": plot_grid,

        "activate_fuente": activate_fuente,
        "activate_ext": activate_ext,

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

    if "mrmt" in model:

        config["Nr"] = Nr
        config["Deff"] = mrmt_df["Deff"].tolist()
        config["beta"] = mrmt_df["beta"].tolist()
        config["phi_im"] = mrmt_df["phi_im"].tolist()

    config_path = output_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # =====================================================
    # SAVE UPLOADED FILE
    # =====================================================

    if uploaded_file is not None:

        save_path = output_dir / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

    st.divider()

    progress_bar = st.progress(0)

    status_box = st.empty()

    log_container = st.empty()

    logs = ""

    status_box.info("Simulation running...")

    process = subprocess.Popen(
        [
            sys.executable,
            "run.py",
            str(config_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    q = queue.Queue()

    # =====================================================
    # THREAD
    # =====================================================

    def enqueue_output(pipe, q):

        for line in iter(pipe.readline, ''):
            q.put(line)

        pipe.close()

    t = threading.Thread(
        target=enqueue_output,
        args=(process.stdout, q)
    )

    t.daemon = True
    t.start()

    # =====================================================
    # LIVE LOGGING
    # =====================================================

    while process.poll() is None:

        while not q.empty():

            line = q.get()

            logs += line

            if "PROGRESS:" in line:

                try:

                    value = int(
                        line.split("PROGRESS:")[1]
                    )

                    progress_bar.progress(value)

                except:
                    pass

            log_container.code(
                logs,
                language="bash"
            )

        time.sleep(0.05)

    # =====================================================
    # FINALIZATION
    # =====================================================

    stderr = process.stderr.read()

    if stderr:

        status_box.error(
            "Simulation failed."
        )

        st.code(
            stderr,
            language="bash"
        )

    else:

        progress_bar.progress(100)

        status_box.success(
            "Simulation completed successfully."
        )

        st.balloons()

        st.divider()

        st.subheader("Downloads")

        log_path = output_dir / "simulation.log"

        with open(log_path, "w") as f:
            f.write(logs)

        col1, col2 = st.columns(2)

        with col1:

            with open(config_path, "rb") as f:

                st.download_button(
                    "⬇️ Download Config",
                    data=f,
                    file_name="config.json",
                    mime="application/json"
                )

        with col2:

            with open(log_path, "rb") as f:

                st.download_button(
                    "⬇️ Download Logs",
                    data=f,
                    file_name="simulation.log",
                    mime="text/plain"
                )