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

    col1, col2, col3 = st.columns(3)

    # =========================
    # COLUMNA 1: CONFIG GENERAL
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

        st.markdown("#### 🧠 Parámetros avanzados")

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
    # COLUMNA 2: NUMÉRICOS + FÍSICOS
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
            value=36000,
            help="Intervalo temporal de simulación"
        )

        T = st.number_input(
            "Tiempo total de simulación",
            value=2592000
        )

        st.markdown("#### 🌊 Parámetros físicos")

        phi_const = st.number_input(
            "Porosidad del medio",
            value=0.1
        )

        a_l = st.number_input(
            "Dispersividad longitudinal (a_l)",
            value=10.0
        )

        

    # =========================
    # COLUMNA 3: CONTROL + PROCESOS
    # =========================
    with col3:
        st.subheader("💾 Salida y control")

        save_dat = st.checkbox("Guardar datos", value=True)
        animate = st.checkbox("Generar animación", value=True)

        st.markdown("#### ⚡ Activación de procesos físicos")

        activate_fuente = st.checkbox("Incluir fuente", value=True)
        activate_ext = st.checkbox("Incluir extracción", value=True)


        st.markdown("#### ▶️ Opciones de ejecución")

        pre = st.checkbox("Ejecutar preprocesamiento")
        postproc = st.checkbox("Ejecutar postproceso")
        plot_grid = st.checkbox("Visualizar malla")

        a_t = st.number_input(
            "Dispersividad transversal (a_t)",
            value=1.0
        )

    # =========================
    # MRMT (DINÁMICO)
    # =========================
    if "mrmt" in model:

        st.markdown("---")
        st.subheader("🧩 Parámetros MRMT")

        col4, col5, col6, col7 = st.columns(4)

        with col4:
            Nr = st.number_input(
                "Número de regiones inmóviles",
                value=3,
                help="Cantidad de zonas de intercambio"
            )

        with col5:
            Deff = st.text_input(
                "Difusividades efectivas (Deff)",
                "1e-9,5e-10,1e-10"
            )

        with col6:
            beta = st.text_input(
                "Coeficientes de intercambio (beta)",
                "0.15,0.1,0.05"
            )

        with col7:
            phi_im = st.text_input(
                "Porosidad inmóvil",
                "0.1,0.05,0.02"
            )

    else:
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
        "Deff": Deff,
        "phi_im": phi_im,
        "beta": beta,

        "save_dat": str(save_dat),
        "animate": str(animate)
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

    


