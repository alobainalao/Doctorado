"""
Grids comparativos 3 × 2 (ADR / MRMT semi / MRMT bloque) × (BFR / FEniCSx) por campo, más una
tercera columna BFR − FEniCSx en los campos del estado. Un PNG por campo:
h, u, C_m, C_im (último paso) y psi_C, psi_h (t = 0 del barrido retrógrado).

    python scripts/grids_comparativos.py [indice_de_paso_adjunto]

Lee data/output/experimentos/N_adjunto/NA_<modelo>/<backend>/{simulation,adjoint}_results.npz y escribe
data/output/experimentos/grids/grid_<campo>.png. Los adjuntos de BFR no están validados; el adjunto no
lleva diferencia (las magnitudes de los dos backends difieren en órdenes de magnitud).
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

RAIZ = os.path.join(CODIGO, "data", "output", "experimentos")
MODELOS = ("adr", "mrmt_semi", "mrmt_block")
NOMBRE = {"adr": "ADR", "mrmt_semi": "MRMT semi", "mrmt_block": "MRMT bloque"}
BACK = (("bfr", "BFR (RBF-FD)"), ("fenicsx", "FEniCSx (FEM)"))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CAMPOS = {"h": "h [m]", "u": "|u| [m/s]", "C_m": "C_m", "C_im": "C_im (media de regiones)",
          "psi_C": "ψ_C", "psi_h": "ψ_h"}
ADJ = ("psi_C", "psi_h")


def _vert(origen, M):
    d, idx = cKDTree(np.asarray(origen, float)).query(M["nodes"])
    assert d.max() < 1e-6
    return idx


def estado(m, b, M):
    z = np.load(os.path.join(RAIZ, "N_adjunto", f"NA_{m}", b, "simulation_results.npz"), allow_pickle=True)
    i = _vert(z["nodes"], M)
    ult = lambda a: np.asarray(a, float).reshape(len(a), -1)[-1][i]
    U = np.asarray(z["U"], float)
    U = U[-1, 0] if U.ndim == 4 else U[-1]
    ci = None
    if "Ci" in z.files:
        ci = np.asarray(z["Ci"], float)[:, -1, 0].mean(axis=0)[i]
    elif "C_im" in z.files:
        ci = np.asarray(z["C_im"], float)[-1].mean(axis=0)[_vert(z["nodes_Q"], M)]
    return {"h": ult(z["H"]), "u": np.linalg.norm(U, axis=-1)[i], "C_m": ult(z["C"]), "C_im": ci}


def adjunto(m, b, M):
    z = np.load(os.path.join(RAIZ, "N_adjunto", f"NA_{m}", b, "adjoint_results.npz"), allow_pickle=True)
    if b == "bfr":
        i = _vert(z["nodes"], M)
        f = lambda a: np.asarray(a, float).reshape(len(a), -1)[K][i]
        return {"psi_C": f(z["psi_C"]), "psi_h": f(z["psi_H"])}
    return {"psi_C": np.asarray(z["psi_C"], float)[K][_vert(z["nodes_Q"], M)],
            "psi_h": np.asarray(z["psi_H"], float)[K][_vert(z["nodes_G"], M)]}


def main():
    M = malla_comun()
    tri = mtri.Triangulation(M["nodes"][:, 0], M["nodes"][:, 1], M["tris"])
    D = {(m, b): {**estado(m, b, M), **adjunto(m, b, M)} for m in MODELOS for b, _ in BACK}
    out = os.path.join(RAIZ, "grids")
    os.makedirs(out, exist_ok=True)
    for c, etq in CAMPOS.items():
        adj = c in ADJ
        ncol = 2 if adj else 3
        fig, ax = plt.subplots(3, ncol, figsize=(7 * ncol, 8.5), constrained_layout=True)
        # escala común a los dos backends (estado) o propia por panel (adjunto)
        vals = [D[(m, b)][c] for m in MODELOS for b, _ in BACK if D[(m, b)][c] is not None]
        for i, m in enumerate(MODELOS):
            for j, (b, nb) in enumerate(BACK):
                a, v = ax[i, j], D[(m, b)][c]
                if v is None:
                    a.text(0.5, 0.5, "sin región inmóvil", ha="center", va="center", transform=a.transAxes)
                    a.set_axis_off()
                    continue
                if adj:
                    lim = np.abs(v).max() or 1.0
                    kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
                else:
                    kw = dict(cmap="viridis", vmin=min(x.min() for x in vals), vmax=max(x.max() for x in vals))
                im = a.tripcolor(tri, v, shading="gouraud", **kw)
                a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
                if adj:
                    fig.colorbar(im, ax=a, shrink=0.8, location="bottom", pad=0.02)
                if i == 0:
                    a.set_title(nb + (" (no validado)" if adj and b == "bfr" else ""))
                if j == 0:
                    a.set_ylabel(NOMBRE[m])
            if not adj:
                a = ax[i, 2]
                vb, vf = D[(m, "bfr")][c], D[(m, "fenicsx")][c]
                if vb is None or vf is None:
                    a.set_axis_off()
                    continue
                d = vb - vf
                lim = max(np.abs(D[(mm, "bfr")][c] - D[(mm, "fenicsx")][c]).max()
                          for mm in MODELOS if D[(mm, "bfr")][c] is not None) or 1.0
                imd = a.tripcolor(tri, d, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
                a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    a.set_title("BFR − FEniCSx")
        if not adj:
            fig.colorbar(im, ax=ax[:, :2], shrink=0.6, location="bottom", pad=0.02, label=etq)
            fig.colorbar(imd, ax=ax[:, 2], shrink=0.6, location="bottom", pad=0.02, label="BFR − FEniCSx")
        fig.suptitle(f"{etq}: " + (f"paso {K} del barrido retrógrado" if adj else "último paso")
                     + ", sin W- y con almacenamiento")
        fig.savefig(os.path.join(out, f"grid_{c}.png"), dpi=100)
        plt.close(fig)
        print("escrito", os.path.join(out, f"grid_{c}.png"))


if __name__ == "__main__":
    main()
