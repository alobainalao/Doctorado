"""
Adjuntos ψ_C y ψ_h del modelo sin W- y con almacenamiento en t = 0 (fin del barrido retrógrado),
3 modelos (filas) × [ψ_C BFR, ψ_C FEniCSx, ψ_h BFR, ψ_h FEniCSx]. Escala de color propia por panel
(las magnitudes de BFR y FEniCSx difieren), simétrica respecto de 0.

    python scripts/figura_adjuntos_6variantes.py [indice_de_paso]

Lee data/output/experimentos/N_adjunto/NA_<modelo>/<backend>/adjoint_results.npz (grupo N_adjunto).
Escribe data/output/experimentos/N_adjunto/adjuntos.png. El adjunto BFR no está validado.
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

RAIZ = os.path.join(CODIGO, "data", "output", "experimentos", "N_adjunto")
MODELOS = ("adr", "mrmt_semi", "mrmt_block")
NOMBRE = {"adr": "ADR", "mrmt_semi": "MRMT semi", "mrmt_block": "MRMT bloque"}
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def _vertices(origen, M):
    d, idx = cKDTree(origen).query(M["nodes"])
    assert d.max() < 1e-6
    return idx


def cargar(modelo, backend, M):
    z = np.load(os.path.join(RAIZ, f"NA_{modelo}", backend, "adjoint_results.npz"), allow_pickle=True)
    if backend == "bfr":
        n = np.asarray(z["nodes"], float)
        i = _vertices(n, M)
        return (np.asarray(z["psi_C"], float).reshape(len(z["psi_C"]), -1)[K][i],
                np.asarray(z["psi_H"], float).reshape(len(z["psi_H"]), -1)[K][i])
    return (np.asarray(z["psi_C"], float)[K][_vertices(np.asarray(z["nodes_Q"], float), M)],
            np.asarray(z["psi_H"], float)[K][_vertices(np.asarray(z["nodes_G"], float), M)])


def main():
    M = malla_comun()
    tri = mtri.Triangulation(M["nodes"][:, 0], M["nodes"][:, 1], M["tris"])
    D = {(m, b): cargar(m, b, M) for m in MODELOS for b in ("bfr", "fenicsx")}
    cols = [("ψ_C", 0, "bfr", "BFR"), ("ψ_C", 0, "fenicsx", "FEniCSx"),
            ("ψ_h", 1, "bfr", "BFR"), ("ψ_h", 1, "fenicsx", "FEniCSx")]
    fig, ax = plt.subplots(3, 4, figsize=(20, 9), constrained_layout=True)
    for i, m in enumerate(MODELOS):
        for j, (nom, k, b, et) in enumerate(cols):
            v = D[(m, b)][k]
            a = ax[i, j]
            lim = np.abs(v).max() or 1.0
            im = a.tripcolor(tri, v, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, shrink=0.8, location="bottom", pad=0.02)
            if i == 0:
                a.set_title(f"{nom}, {et}" + (" (no validado)" if b == "bfr" else ""))
            if j == 0:
                a.set_ylabel(NOMBRE[m])
    fig.suptitle(f"Adjuntos en el paso {K} del barrido (t = {K}Δt), sin W- y con almacenamiento")
    ruta = os.path.join(RAIZ, "adjuntos.png")
    fig.savefig(ruta, dpi=110)
    print("escrito", ruta)


if __name__ == "__main__":
    main()
