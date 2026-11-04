"""
Campos del modelo sin W- y con almacenamiento en el último paso, 3 modelos (filas) × 4 campos (columnas):
h, |u|, C_m y C_im (media de las regiones inmóviles; ADR no tiene). Tres figuras: BFR, FEniCSx y BFR − FEniCSx.

    python scripts/figura_campos_6variantes.py [grupo]

Lee <grupo>/N_<modelo>/<backend>/simulation_results.npz de data/output/experimentos (grupo por defecto
N_modelo_nuevo) y escribe data/output/experimentos/<grupo>/campos_{bfr,fenicsx,diferencia}.png.
Sólo se dibuja lo que está en disco; los campos se llevan a los nodos de la malla común (experimentos.comparar).
"""
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

from experimentos.comparar import malla_comun

GRUPO = sys.argv[1] if len(sys.argv) > 1 else "N_modelo_nuevo"
PREF = "N_" if GRUPO == "N_modelo_nuevo" else "NA_"
RAIZ = os.path.join(CODIGO, "data", "output", "experimentos", GRUPO)
MODELOS = ("adr", "mrmt_semi", "mrmt_block")
NOMBRE = {"adr": "ADR", "mrmt_semi": "MRMT semi", "mrmt_block": "MRMT bloque"}
CAMPOS = ("h [m]", "|u| [m/s]", "C_m", "C_im (media)")


def _vertices(nodos_origen, M):
    d, idx = cKDTree(nodos_origen).query(M["nodes"])
    assert d.max() < 1e-6, "nodos de la malla común no coinciden con los del resultado"
    return idx


def cargar(modelo, backend, M):
    z = np.load(os.path.join(RAIZ, f"{PREF}{modelo}", backend, "simulation_results.npz"), allow_pickle=True)
    nodos = np.asarray(z["nodes"], float)
    idx = _vertices(nodos, M)
    h = np.asarray(z["H"], float).reshape(len(z["H"]), -1)[-1][idx]
    U = np.asarray(z["U"], float)
    U = U[-1, 0] if U.ndim == 4 else U[-1]
    u = np.linalg.norm(U, axis=-1)[idx]
    c = np.asarray(z["C"], float).reshape(len(z["C"]), -1)[-1][idx]
    ci = None
    if "Ci" in z.files:                      # BFR: (Nr, Nt, 1, N)
        ci = np.asarray(z["Ci"], float)[:, -1, 0].mean(axis=0)[idx]
    elif "C_im" in z.files:                  # FEniCSx: (Nt, Nr, N_Q) en nodos P2
        nq = np.asarray(z["nodes_Q"], float)
        ci = np.asarray(z["C_im"], float)[-1].mean(axis=0)[_vertices(nq, M)]
    return [h, u, c, ci]


def dibujar(filas, titulo, ruta, M, sim=False):
    tri = mtri.Triangulation(M["nodes"][:, 0], M["nodes"][:, 1], M["tris"])
    fig, ax = plt.subplots(3, 4, figsize=(20, 9), constrained_layout=True)
    for j in range(4):
        vals = [f[j] for f in filas if f[j] is not None]
        if not vals:
            continue
        if sim:
            m = max(np.abs(v).max() for v in vals) or 1.0
            kw = dict(cmap="RdBu_r", vmin=-m, vmax=m)
        else:
            lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
            kw = dict(cmap="viridis", vmin=lo, vmax=hi)
        for i, f in enumerate(filas):
            a = ax[i, j]
            if f[j] is None:
                a.text(0.5, 0.5, "sin región inmóvil", ha="center", va="center", transform=a.transAxes)
                a.set_axis_off()
                continue
            im = a.tripcolor(tri, f[j], shading="gouraud", **kw)
            a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(CAMPOS[j])
            if j == 0:
                a.set_ylabel(NOMBRE[MODELOS[i]])
        fig.colorbar(im, ax=ax[:, j], shrink=0.7, location="bottom", pad=0.02)
    fig.suptitle(titulo)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    print("escrito", ruta)


def main():
    M = malla_comun()
    datos = {b: [cargar(m, b, M) for m in MODELOS] for b in ("bfr", "fenicsx")}
    t = "último paso, sin W- y con almacenamiento"
    dibujar(datos["bfr"], f"BFR (RBF-FD), {t}", os.path.join(RAIZ, "campos_bfr.png"), M)
    dibujar(datos["fenicsx"], f"FEniCSx, {t}", os.path.join(RAIZ, "campos_fenicsx.png"), M)
    dif = [[None if (a is None or b is None) else a - b for a, b in zip(fb, ff)]
           for fb, ff in zip(datos["bfr"], datos["fenicsx"])]
    dibujar(dif, f"BFR − FEniCSx, {t}", os.path.join(RAIZ, "campos_diferencia.png"), M, sim=True)


if __name__ == "__main__":
    main()
