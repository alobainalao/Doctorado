# Imagen de desarrollo para el backend fenicsx — DOLFINx oficial +
# dependencias de funtions/get_phi.py y funtions/utils.py que bfr_env no
# necesita pero este backend sí (pandas, matplotlib, scikit-learn, h5py).
#
# Build:
#   docker build -t fenicsx-dev -f fenicsx/Dockerfile .
# Run (monta codigo/ como /workspace):
#   docker run --rm -it -v "$(pwd):/workspace" -w /workspace fenicsx-dev bash
FROM dolfinx/dolfinx:stable

RUN pip install --no-cache-dir pandas matplotlib scikit-learn h5py
