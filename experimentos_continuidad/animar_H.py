"""Video de H: BFR concatenada | BFR desarrollada | FEM | BFR desarrollada − FEM.

    python experimentos_continuidad/animar_H.py ID_COMP ID_EXP ID_FEM_CORRIDA SALIDA.mp4

H en el rango de FEM (turbo, view.plot); lo que se sale del rango satura y el título
indica max|H| del cuadro (la concatenada diverge: la saturación es la información).
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)
sys.path.insert(0, CODIGO)
from view.plot import get_colormaps


def main(comp, exp, fem, out):
    R = lambda *a: os.path.join(AQUI, "resultados", *a)
    m = pickle.load(open(os.path.join(CODIGO, "data/input/malla_compartida/malla.pkl"), "rb"))
    nodes = np.asarray(m["nodes"]); x, z = nodes.T; pz = np.asarray(m["pozo_cor"])
    tri = mtri.Triangulation(x, z, np.asarray(m["triangles"]))
    Hc = np.load(R(comp, "bfr_H", "H.npz"))["H"]; He = np.load(R(exp, "bfr_H", "H.npz"))["H"]
    f = np.load(R(fem, "fenicsx", "simulation_results.npz"), allow_pickle=True)
    j = cKDTree(nodes).query(np.asarray(f["nodes"]))[1]
    Hf = np.zeros_like(He); Hf[:, j] = np.asarray(f["H"]).reshape(len(He), -1)
    Hf[:, np.setdiff1d(np.arange(len(nodes)), j)] = np.nan   # fantasmas: sin FEM
    dt = float(f["dt"]); t = (np.arange(len(He)) + 1) * dt / 86400
    cmap, _ = get_colormaps()
    nh = Normalize(np.nanmin(Hf), np.nanmax(Hf))
    D = He - Hf; lim = max(float(np.nanpercentile(np.abs(D), 99.5)), 1e-3); nd = TwoSlopeNorm(0, -lim, lim)
    fig, ax = plt.subplots(4, 1, figsize=(11, 13), constrained_layout=True)
    pan = [("BFR concatenada (1.er orden compuesto)", Hc, nh, cmap), ("BFR desarrollada", He, nh, cmap),
           ("FEniCS", Hf, nh, cmap), ("BFR desarrollada − FEM  [m]", D, nd, "RdBu_r")]
    arts = []
    for a, (tt, A, nm, cm) in zip(ax, pan):
        a.set_aspect("equal"); a.set_title(tt, fontsize=10); a.plot(*pz, "k+", ms=10)
        im = a.tripcolor(tri, np.nan_to_num(A[0], nan=nm.vmin if hasattr(nm, "vmin") else 0), norm=nm, cmap=cm,
                         shading="gouraud", rasterized=True)
        fig.colorbar(im, ax=a, fraction=0.02, extend="both" if cm is cmap else "neither"); arts.append((a, tt, im, A))
    ttl = fig.suptitle("")
    wr = FFMpegWriter(fps=12)
    with wr.saving(fig, out, 110):
        for k in range(len(He)):
            for a, tt, im, A in arts:
                im.set_array(np.nan_to_num(A[k], nan=0.0))
            arts[0][0].set_title(f"BFR concatenada (1.er orden compuesto)   máx|H| = {np.abs(Hc[k]).max():.3g} m", fontsize=10)
            ttl.set_text(f"H, salida libre — día {t[k]:.1f}")
            wr.grab_frame()


if __name__ == "__main__":
    main(*sys.argv[1:5])
