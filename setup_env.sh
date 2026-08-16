#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Instala Miniconda (si falta), crea el env conda `bfr_env`
# e instala todas las dependencias para correr el simulador ADR/MRMT.
#
# Uso:
#   bash setup_env.sh
#
# Idempotente: se puede volver a correr sin romper nada.
# Al terminar:
#   conda activate bfr_env
#   python run.py                 # simulación / optimización (config/default.json)
#   streamlit run app.py          # interfaz web
# =============================================================================
set -euo pipefail

# ---- Configuración -----------------------------------------------------------
ENV_NAME="bfr_env"
PY_VERSION="3.10"                 # coincide con runtime.txt (python-3.10.18)
CONDA_HOME="${HOME}/miniconda3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

info()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; }

# ---- 1) Localizar o instalar conda ------------------------------------------
find_conda() {
    if command -v conda >/dev/null 2>&1; then
        # conda ya en PATH: derivar su carpeta base
        CONDA_HOME="$(conda info --base)"
        return 0
    fi
    for base in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda"; do
        if [ -f "${base}/etc/profile.d/conda.sh" ]; then
            CONDA_HOME="${base}"
            return 0
        fi
    done
    return 1
}

install_miniconda() {
    info "conda no encontrado. Instalando Miniconda en ${CONDA_HOME} ..."

    local arch installer url tmp
    arch="$(uname -m)"
    case "${arch}" in
        x86_64)  installer="Miniconda3-latest-Linux-x86_64.sh"  ;;
        aarch64) installer="Miniconda3-latest-Linux-aarch64.sh" ;;
        *) error "Arquitectura no soportada: ${arch}"; exit 1    ;;
    esac
    url="https://repo.anaconda.com/miniconda/${installer}"

    tmp="$(mktemp -d)"
    info "Descargando ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${tmp}/miniconda.sh"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "${url}" -O "${tmp}/miniconda.sh"
    else
        error "Se necesita curl o wget para descargar Miniconda."; exit 1
    fi

    bash "${tmp}/miniconda.sh" -b -p "${CONDA_HOME}"
    rm -rf "${tmp}"
    info "Miniconda instalado."
}

if ! find_conda; then
    install_miniconda
fi

# Cargar la función `conda` en este shell no interactivo
# shellcheck disable=SC1091
source "${CONDA_HOME}/etc/profile.d/conda.sh"
info "Usando conda en: ${CONDA_HOME}"

# ---- 2) Crear el entorno -----------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    info "El env '${ENV_NAME}' ya existe; se reutiliza."
else
    info "Creando env '${ENV_NAME}' (python=${PY_VERSION}) ..."
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"

# ---- 3) Dependencias de sistema vía conda-forge ------------------------------
# ffmpeg: requerido por matplotlib.FFMpegWriter (videos H/V/C y adjuntos).
# Las ruedas pip de rtree/shapely ya traen sus libs nativas (libspatialindex/geos),
# así que solo hace falta ffmpeg desde conda.
# Nota: conda 26.3 tiene un lock intermitente del caché de "shards"
# (sqlite3.OperationalError: database is locked); reintentamos y, si persiste,
# caemos al binario que trae imageio-ffmpeg vía pip.
info "Instalando ffmpeg (conda-forge) ..."
ffmpeg_ok=0
for attempt in 1 2 3; do
    if conda install -y -c conda-forge ffmpeg; then ffmpeg_ok=1; break; fi
    warn "intento ${attempt} de ffmpeg falló (lock del caché de conda); reintentando en 5s ..."
    sleep 5
done
if [ "${ffmpeg_ok}" -ne 1 ]; then
    warn "conda no pudo instalar ffmpeg; usando fallback pip (imageio-ffmpeg)."
    python -m pip install imageio-ffmpeg
fi

# ---- 4) Dependencias Python vía pip -----------------------------------------
info "Actualizando pip / setuptools / wheel ..."
python -m pip install --upgrade pip setuptools wheel

if [ -f "${REQ_FILE}" ]; then
    info "Instalando requirements.txt ..."
    python -m pip install -r "${REQ_FILE}"
else
    error "No se encontró ${REQ_FILE}"; exit 1
fi

# ---- 5) Verificación ---------------------------------------------------------
info "Verificando imports clave ..."
python - <<'PY'
import importlib, sys

mods = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "shapely": "shapely",
    "rtree": "rtree",
    "streamlit": "streamlit",
    "rbf.pde.fd": "treverhines-rbf",   # el paquete se importa como `rbf`
}
fail = []
for mod, pkg in mods.items():
    try:
        importlib.import_module(mod)
        print(f"  ok  {mod:16s} ({pkg})")
    except Exception as e:
        fail.append((mod, pkg, e))
        print(f"  FAIL {mod:16s} ({pkg}): {e}")

import shutil
print("  ok  ffmpeg" if shutil.which("ffmpeg") else "  FAIL ffmpeg no encontrado en PATH")

if fail:
    print("\nFaltan dependencias:", ", ".join(m for m, _, _ in fail))
    sys.exit(1)
print("\nEntorno verificado correctamente.")
PY

info "Listo. Para usar el proyecto:"
echo ""
echo "    conda activate ${ENV_NAME}"
echo "    cd ${SCRIPT_DIR}"
echo "    python run.py            # simulación / optimización"
echo "    streamlit run app.py     # interfaz web"
echo ""
