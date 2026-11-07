"""Video de C: referencia A/C vs variante, y su diferencia (misma escala de color).

    python experimentos_continuidad/animar_comparacion.py ID_REF ID_VAR SALIDA.mp4 [zoom_m] [--backend=bfr|fenicsx]

zoom_m: semiancho (m) de una ventana centrada en el pozo; sin él, dominio completo.
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view.plot import get_colormaps  # paletas de los videos de las demás variantes

AQUI = os.path.dirname(os.path.abspath(__file__))
CODIGO = os.path.dirname(AQUI)


def cargar(i, backend, nodes):
    """C(t) sobre TODOS los nodos de la malla compartida (FEM: sin fantasmas, se
    empareja por coordenadas y los fantasmas quedan en 0; no están en los triángulos)."""
    d = np.load(os.path.join(AQUI, "resultados", i, backend, "simulation_results.npz"), allow_pickle=True)
    C = np.asarray(d["C"]).reshape(len(d["C"]), -1)
    if backend == "fenicsx":
        from scipy.spatial import cKDTree
        dist, j = cKDTree(nodes).query(np.asarray(d["nodes"]))
        assert dist.max() < 1e-6, "nodos FEM ≠ malla compartida"
        full = np.zeros((C.shape[0], len(nodes))); full[:, j] = C; C = full
    return C, float(d["dt"])


def main(ref, var, out, zoom=None, backend="bfr"):
    m = pickle.load(open(os.path.join(CODIGO, "data/input/malla_compartida/malla.pkl"), "rb"))
    x, z = np.asarray(m["nodes"]).T
    tri = mtri.Triangulation(x, z, np.asarray(m["triangles"]))
    pz = np.asarray(m["pozo_cor"])
    A, dt = cargar(ref, backend, np.c_[x, z]); B, _ = cargar(var, backend, np.c_[x, z])
    _, cmap_c = get_colormaps()
    cmax = max(float(A.max()), float(B.max()), 1e-30)
    lim = max(float(np.percentile(np.abs(B - A), 99.5)), cmax * 1e-3)
    nc, nd = Normalize(0.0, cmax), TwoSlopeNorm(0.0, -lim, lim)
    if zoom:   # ventana cuadrada: tres columnas
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    else:      # dominio 10×3 km: tres filas
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    pan = [("C [kg/m³]  " + backend + " " + ref, A, nc, cmap_c), ("C [kg/m³]  " + backend + " " + var, B, nc, cmap_c),
           ("diferencia  " + var + " − " + ref, B - A, nd, "RdBu_r")]
    arts = []
    for a, (t, F, nm, cm) in zip(ax, pan):
        a.set_aspect("equal"); a.set_title(t, fontsize=10)
        if zoom:
            a.set_xlim(pz[0] - zoom, pz[0] + zoom); a.set_ylim(pz[1] - zoom, pz[1] + zoom)
        a.plot(*pz, "k+", ms=10)
        im = a.tripcolor(tri, F[0], norm=nm, cmap=cm, shading="gouraud", rasterized=True)
        fig.colorbar(im, ax=a, fraction=0.03 if zoom else 0.02); arts.append((im, F))
    ttl = fig.suptitle("")
    w = FFMpegWriter(fps=12)
    with w.saving(fig, out, 110):
        for k in range(A.shape[0]):
            for im, F in arts:
                im.set_array(F[k])
            ttl.set_text(f"día {(k + 1) * dt / 86400:.1f}")
            w.grab_frame()


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--backend=")]
    bk = next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--backend=")), "bfr")
    main(a[0], a[1], a[2], float(a[3]) if len(a) > 3 else None, bk)
