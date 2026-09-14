# Imagen de desarrollo para el backend fenicsx — DOLFINx oficial +
# dependencias de funtions/get_phi.py y funtions/utils.py que bfr_env no
# necesita pero este backend sí (pandas, matplotlib, scikit-learn, h5py).
#
# Build:
#   docker build -t fenicsx-dev -f fenicsx/Dockerfile .
# Run (monta codigo/ como /workspace):
#   docker run --rm -it -v "$(pwd):/workspace" -w /workspace fenicsx-dev bash
FROM dolfinx/dolfinx:stable

# ffmpeg: para exportar las animaciones en mp4 (como el backend bfr). Si falta,
# fenicsx/io.py cae a gif automáticamente.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# streamlit: para correr la app (app.py) DENTRO de este contenedor cuando se
# usa el backend fenicsx (que requiere dolfinx, ausente en bfr_env). La app
# lanza run.py como subproceso, que despacha a fenicsx.adjoint:optimize_fenicsx.
RUN pip install --no-cache-dir pandas matplotlib scikit-learn h5py gmsh streamlit
