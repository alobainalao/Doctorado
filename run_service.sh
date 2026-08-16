#!/usr/bin/env bash
# =============================================================================
# run_service.sh — Levanta los servicios del simulador ADR/MRMT en el env conda.
#
# Dos variantes:
#   bash run_service.sh cli   [args...]   -> python run.py        (simulación/optim.)
#   bash run_service.sh web   [args...]   -> streamlit run app.py (interfaz web)
#
# Alias aceptados:  cli = sim | run    ·    web = streamlit | app
#
# Ejemplos:
#   bash run_service.sh cli
#   bash run_service.sh web --server.port 8502
#
# Requiere el env creado por setup_env.sh.
# =============================================================================
set -euo pipefail

ENV_NAME="bfr_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info()  { printf '\033[1;34m[run]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[run]\033[0m %s\n' "$*" >&2; }

usage() {
    cat >&2 <<EOF
Uso: bash run_service.sh <variante> [args...]

Variantes:
  cli   (sim|run)        python run.py        — simulación / optimización
  web   (streamlit|app)  streamlit run app.py — interfaz web

Ejemplos:
  bash run_service.sh cli
  bash run_service.sh web --server.port 8502
EOF
    exit 2
}

[ $# -ge 1 ] || usage
variant="$1"; shift || true

# ---- Localizar y cargar conda ------------------------------------------------
CONDA_HOME=""
if command -v conda >/dev/null 2>&1; then
    CONDA_HOME="$(conda info --base)"
else
    for base in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda"; do
        [ -f "${base}/etc/profile.d/conda.sh" ] && { CONDA_HOME="${base}"; break; }
    done
fi
if [ -z "${CONDA_HOME}" ]; then
    error "No se encontró conda. Corre primero: bash setup_env.sh"; exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_HOME}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    error "El env '${ENV_NAME}' no existe. Corre primero: bash setup_env.sh"; exit 1
fi
conda activate "${ENV_NAME}"

cd "${SCRIPT_DIR}"

# ---- Despachar variante ------------------------------------------------------
case "${variant}" in
    cli|sim|run)
        info "Servicio CLI  ->  python run.py ${*}"
        exec python run.py "$@"
        ;;
    web|streamlit|app)
        info "Servicio WEB  ->  streamlit run app.py ${*}"
        exec streamlit run app.py "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        error "Variante desconocida: '${variant}'"
        usage
        ;;
esac
