import streamlit as st
import subprocess
import sys
import os

st.set_page_config(layout="wide")
st.title("Simulador de Transporte (ADR / MRMT)")

# =========================
# FORMULARIO
# =========================
with st.form("sim_form"):

    col1, col2 = st.columns(2)

    # =========================
    # CONFIGURACIÓN GENERAL
    # =========================
    with col1:
        st.subheader("⚙️ Configuración general")

        metodo = st.selectbox(
            "Método numérico",
            ["bfr", "fenicx"],
            help="Selecciona el método de discretización espacial"
        )

        model = st.selectbox(
            "Modelo de transporte",
            ["adr", "mrmt_semi", "mrmt_block"],
            help="ADR: clásico | MRMT: incluye transferencia a zonas inmóviles"
        )

        domain = st.selectbox(
            "Tipo de dominio",
            ["real", "benchmark"],
            help="Dominio físico real o caso de prueba"
        )

        st.markdown("#### Opciones de ejecución")

        pre = st.checkbox("Ejecutar preprocesamiento")
        postproc = st.checkbox("Ejecutar postproceso", value=True)
        plot_grid = st.checkbox("Visualizar malla")

    # =========================
    # PARÁMETROS NUMÉRICOS
    # =========================
    with col2:
        st.subheader("🧮 Parámetros numéricos")

        spacing = st.number_input(
            "Resolución espacial (spacing)",
            value=30.0,
            help="Distancia entre nodos de la malla"
        )

        dt = st.number_input(
            "Paso de tiempo (dt)",
            value=3600.0,
            help="Intervalo temporal de simulación"
        )

        T = st.number_input(
            "Tiempo total de simulación",
            value=2592000.0
        )

        st.markdown("#### Activación de procesos físicos")

        activate_fuente = st.checkbox("Incluir fuente", value=True)
        activate_ext = st.checkbox("Incluir extracción", value=True)

    # =========================
    # PARÁMETROS FÍSICOS
    # =========================
    st.subheader("🌊 Parámetros físicos")

    col3, col4 = st.columns(2)

    with col3:
        phi_const = st.number_input(
            "Porosidad del medio",
            value=0.1
        )

        a_l = st.number_input(
            "Dispersividad longitudinal (a_l)",
            value=10.0
        )

    with col4:
        a_t = st.number_input(
            "Dispersividad transversal (a_t)",
            value=1.0
        )

        D_d = st.number_input(
            "Coeficiente de difusión molecular",
            value=1e-19,
            format="%.2e"
        )

        eps = st.number_input(
            "Tolerancia numérica",
            value=1e-16,
            format="%.2e"
        )

    # =========================
    # MRMT (DINÁMICO)
    # =========================
    if "mrmt" in model:

        st.subheader("🧩 Parámetros MRMT")

        Nr = st.number_input(
            "Número de regiones inmóviles",
            value=3,
            help="Cantidad de zonas de intercambio"
        )

        Deff = st.text_input(
            "Difusividades efectivas (Deff)",
            "1e-9,5e-10,1e-10",
            help="Separadas por coma"
        )

        beta = st.text_input(
            "Coeficientes de intercambio (beta)",
            "0.15,0.1,0.05"
        )

        phi_im = st.text_input(
            "Porosidad inmóvil",
            "0.1,0.05,0.02"
        )

    else:
        # valores por defecto silenciosos
        Nr = 3
        Deff = "1e-9,5e-10,1e-10"
        beta = "0.15,0.1,0.05"
        phi_im = "0.1,0.05,0.02"

    submit = st.form_submit_button("🚀 Ejecutar simulación")

# =========================
# EJECUCIÓN
# =========================
if submit:

    st.write("### ⏳ Ejecutando simulación...")

    env = os.environ.copy()
    env.update({
        "MODEL": model,
        "DOMAIN": domain,
        "METODO": metodo,
        "PRE": str(pre),
        "POST": str(postproc),
        "PLOT": str(plot_grid),
        "FUENTE": str(activate_fuente),
        "EXT": str(activate_ext),

        "SPACING": str(spacing),
        "DT": str(dt),
        "T": str(T),

        "PHI_CONST": str(phi_const),
        "A_L": str(a_l),
        "A_T": str(a_t),
        "D_D": str(D_d),
        "EPS": str(eps),

        "NR": str(Nr),
        "DEFF": Deff,
        "BETA": beta,
        "PHI_IM": phi_im
    })

    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=os.getcwd(),  # 🔥 importante
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
