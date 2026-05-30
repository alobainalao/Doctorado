import streamlit as st
import subprocess
import sys
import os
import json
import time
import pandas as pd
from pathlib import Path

# =========================================================
# QUERY PARAMS
# =========================================================

params = st.query_params
theme = params.get("theme", "dark")

# =========================================================
# STREAMLIT THEME
# =========================================================

st._config.set_option(
    "theme.base",
    "dark" if theme == "dark" else "light"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ADR / MRMT Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# =========================================================
# CSS CLEAN + COMPACTO
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
}

h1 {
    margin-top: 0rem !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.2 !important;
}

header {
    padding-top: 0rem !important;
}

/* BOTONES */
.stButton > button {
    min-width: 180px;
    height: 44px;
    border-radius: 10px;
    font-weight: 600;
}

/* INPUTS */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    width: 320px !important;
}

[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 0.8rem;
}

.element-container {
    margin-bottom: 0.4rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# STATE DEFAULTS
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
    st.session_state.setdefault(k, v)

# =========================================================
# TITLE
# =========================================================

st.title("🌊 ADR / MRMT Transport Simulator")
st.caption("Reactive transport simulation platform")

# =========================================================
# SIDEBAR (CONFIG)
# =========================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    metodo = st.selectbox("Numerical Method", ["bfr", "fenicsx"])
    model = st.selectbox("Transport Model", ["adr", "mrmt_semi", "mrmt_block"])
    domain = st.selectbox("Domain Type", ["real", "benchmark"])
    run_type = st.selectbox("Run Type", ["standard", "optimization"])

    st.divider()

    pre = st.checkbox("Preprocessing")
    postproc = st.checkbox("Postprocessing")
    plot_grid = st.checkbox("Plot Grid")

    activate_fuente = st.checkbox("Enable Source", value=True)
    activate_ext = st.checkbox("Enable Extraction", value=True)

# =========================================================
# TABS UI
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧮 Numerical",
    "🌊 Physics",
    "🧩 MRMT",
    "💾 Output",
    "⚙️ Optimization",
    "🚀 Run"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:
    st.subheader("Numerical Parameters")

    c1, c2, c3, c4 = st.columns(4)

    spacing = c1.number_input("Spacing", value=st.session_state.spacing)
    dt = c2.number_input("dt", value=st.session_state.dt)
    T = c3.number_input("Total Time", value=st.session_state.T)
    eps = c4.number_input("Tolerance", value=st.session_state.eps, format="%.2e")

    Nt = int(T / dt)

    st.divider()
    st.write(f"Timesteps: **{Nt:,}**")

# =========================================================
# TAB 2
# =========================================================

with tab2:
    st.subheader("Physical Parameters")

    c1, c2, c3, c4 = st.columns(4)

    phi_const = c1.number_input("Porosity", value=st.session_state.phi_const)
    a_l = c2.number_input("a_l", value=st.session_state.a_l)
    a_t = c3.number_input("a_t", value=st.session_state.a_t)
    D_d = c4.number_input("Diffusion", value=st.session_state.D_d, format="%.2e")

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


with tab5:

    st.subheader("Optimization parameters")

    optimize = st.checkbox("Enable optimization", value=False)

    if optimize:

        col1, col2, col3 = st.columns(3)

        with col1:
            gamma = st.number_input("gamma", value=1.0, format="%.4f")

        with col2:
            koppa = st.number_input("koppa", value=1.0, format="%.4f")

        with col3:
            z0 = st.number_input("z0", value=0.0, format="%.4f")

    else:
        gamma = 1.0
        koppa = 1.0
        z0 = 0.0

        st.info("Optimization disabled")

# =========================================================
# TAB 5 - RUN
# =========================================================

with tab6:

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

    submit = False

    if not errors:
        st.success("Configuration valid")
        _, _, right = st.columns([1,2,1])
        with right:
            submit = st.button("🚀 Run Simulation")

    else:
        for e in errors:
            st.error(e)

# =========================================================
# EXECUTION (TU FORMATO EXACTO + ENV FIX)
# =========================================================

if submit:

    st.write("### ⏳ Ejecutando simulación...")

    env = os.environ.copy()

    env.update({
        "model": model,
        "domain": domain,
        "metodo": metodo,

        "pre": str(pre),
        "postproc": str(postproc),
        "plot_grid": str(plot_grid),

        "activate_fuente": str(activate_fuente),
        "activate_ext": str(activate_ext),

        "spacing": str(spacing),
        "dt": str(dt),
        "T": str(T),

        "phi_const": str(phi_const),
        "a_l": str(a_l),
        "a_t": str(a_t),
        "D_d": str(D_d),
        "eps": str(eps),

        "Nr": str(Nr),

        "save_dat": str(save_dat),
        "animate": str(animate),

        "run_type": run_type,

        "gamma": str(gamma),
        "koppa": str(koppa),
        "z0": str(z0),
    })

    process = subprocess.Popen(
        [sys.executable, "run.py"],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    log_area = st.empty()
    logs = ""

    for line in process.stdout:
        logs += line
        log_area.text(logs)

    errors = process.stderr.read()

    if errors:
        st.error("❌ Se detectaron errores")
        st.text(errors)
    else:
        st.success("✅ Simulación terminada correctamente")