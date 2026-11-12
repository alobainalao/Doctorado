"""
Animaciones de los experimentos BFR (RBF-FD) vs FEniCS (FEM).

Por caso genera:
    <caso>/<backend>/figures/{H,C,V}.mp4    un video por campo y backend

La comparación BFR vs FEM (grilla con diferencia y curvas) NO está aquí: la
hace experimentos/comparar.py junto con las métricas, en una sola pasada.

Rangos de color fijos por caso y comunes a ambos backends (los mismos en los
videos individuales y en la grilla), para que BFR y FEM se lean igual. En los
casos que sólo corren BFR (grupos B y F) la columna FEM es la de `ref_fem`.

    python -m experimentos.animar --caso M_modelo_nuevo/M0_adr
    python -m experimentos.animar --grupo M_modelo_nuevo MC_malla_conos
    python -m experimentos.animar --todos [--jobs 4] [--force]
"""
import argparse
import copy
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.catalogo import GRUPOS, iter_casos  # noqa: E402
from view.plot import get_colormaps  # noqa: E402

OUT_ROOT = os.path.join(CODIGO_DIR, "data", "output", "experimentos")
LABEL_B = {"bfr": "BFR (RBF-FD)", "fenicsx": "FEniCS (FEM)"}

# ── Dominio físico (idéntico a scripts/animate_*.py) ─────────────────────────
XMIN, XMAX = 816.0, 10755.0
ZMIN, ZMAX = -2690.5, 250.5
DOMINIO = np.array([[816, -2690.5], [4354, -1757.5], [10755, -768.5],
                    [10755, 250.5], [816, 250.5]])
DOM_PATH = MplPath(DOMINIO)

NX = 500
NZ = round(NX * (ZMAX - ZMIN) / (XMAX - XMIN))
_XI, _ZI = np.meshgrid(np.linspace(XMIN, XMAX, NX), np.linspace(ZMIN, ZMAX, NZ))
GRID_PTS = np.column_stack([_XI.ravel(), _ZI.ravel()])
_QX, _QZ = np.meshgrid(np.linspace(XMIN, XMAX, 40), np.linspace(ZMIN, ZMAX, 13))
_QALL = np.column_stack([_QX.ravel(), _QZ.ravel()])
Q_PTS = _QALL[DOM_PATH.contains_points(_QALL)]

FPS = 12
WRITER_KW = dict(fps=FPS, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
CMAP_MAIN, CMAP_C = get_colormaps()
CAMPOS = (  # clave, etiqueta, colormap
    ("H", "H (m)", CMAP_MAIN),
    ("C", "C (kg/m³)", CMAP_C),
    ("V", "|U| (m/s)", CMAP_MAIN),
)


def _interp_matrix(nodes, pts):
    """Matriz dispersa (len(pts) × N) de interpolación lineal en la triangulación
    de `nodes`, más la máscara de puntos fuera del casco. Se arma una vez y cada
    cuadro es un producto matriz-vector (en vez de un LinearNDInterpolator)."""
    tri = Delaunay(nodes)
    s = tri.find_simplex(pts)
    fuera = s < 0
    s = np.where(fuera, 0, s)
    T = tri.transform[s]
    b = np.einsum("ijk,ik->ij", T[:, :2], pts - T[:, 2])
    w = np.column_stack([b, 1 - b.sum(axis=1)])
    w[fuera] = 0
    filas = np.repeat(np.arange(len(pts)), 3)
    M = sparse.csr_matrix((w.ravel(), (filas, tri.simplices[s].ravel())),
                          shape=(len(pts), len(nodes)))
    return M, fuera


def _cargar(path):
    """Campos en los nodos interiores (sin fantasmas BFR) + operadores de interpolación."""
    z = np.load(path, allow_pickle=True)
    nodes = np.asarray(z["nodes"], float)
    U = np.asarray(z["U"], float)
    if U.ndim == 4:                       # BFR: (Nt, K, N, 2)
        U = U[:, 0]
    dentro = DOM_PATH.contains_points(nodes, radius=1e-3)
    nodes = nodes[dentro]
    d = dict(H=np.asarray(z["H"], float)[:, 0, dentro],
             C=np.asarray(z["C"], float)[:, 0, dentro],
             U=U[:, dentro], dt=float(z["dt"]))
    d["V"] = np.linalg.norm(d["U"], axis=-1)
    M, fuera = _interp_matrix(nodes, GRID_PTS)
    d["Mg"], d["fuera"] = M, fuera.reshape(NZ, NX)
    d["Mq"], _ = _interp_matrix(nodes, Q_PTS)
    return d


def _img(d, campo, k):
    im = (d["Mg"] @ d[campo][k]).reshape(NZ, NX)
    return np.ma.masked_where(d["fuera"], im)


def _flechas(d, k):
    ux, uz = d["Mq"] @ d["U"][k, :, 0], d["Mq"] @ d["U"][k, :, 1]
    mag = np.maximum(np.hypot(ux, uz), 1e-30)
    return ux / mag, uz / mag


def _rangos(datos):
    """(vmin, vmax) por campo, comunes a todos los backends del caso.
    H y C en su rango completo (recortar por percentiles escondía el cono del
    pozo); |U| lineal como en scripts/animate_forward.py, saturado en el
    percentil 99 (el pico del pozo está en unos pocos nodos)."""
    r = {}
    todo = {c: np.concatenate([d[c].ravel() for d in datos]) for c, _, _ in CAMPOS}
    r["H"] = (float(todo["H"].min()), float(todo["H"].max()))
    r["C"] = (0.0, max(float(todo["C"].max()), 1e-30))
    r["V"] = (0.0, max(float(np.percentile(todo["V"], 99.0)), 1e-30))
    return r


def _panel(ax, d, campo, vmin, vmax, cmap):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.axis("off")
    clip = MplPolygon(DOMINIO, transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    cm = copy.copy(cmap)
    cm.set_bad("white")
    im = ax.imshow(_img(d, campo, 0), origin="lower", extent=[XMIN, XMAX, ZMIN, ZMAX],
                   norm=Normalize(vmin, vmax),
                   cmap=cm, aspect="equal", interpolation="antialiased")
    im.set_clip_path(clip)
    ax.add_patch(MplPolygon(DOMINIO, fc="none", ec="#999999", lw=0.6))
    q = None
    if campo == "V":
        q = ax.quiver(Q_PTS[:, 0], Q_PTS[:, 1], *_flechas(d, 0), scale=50,
                      scale_units="width", width=0.0015, headwidth=4, headlength=4,
                      color="black", alpha=0.7, zorder=5)
    return im, q


def _actualizar(im, q, d, campo, k):
    im.set_data(_img(d, campo, k))
    if q is not None:
        q.set_UVC(*_flechas(d, k))


def _t_dias(d, k):
    return (k + 1) * d["dt"] / 86400


def video_backend(d, backend, titulo, rangos, out_dir):
    """H.mp4, C.mp4 y V.mp4 de un backend."""
    os.makedirs(out_dir, exist_ok=True)
    for campo, etiqueta, cmap in CAMPOS:
        fig = plt.figure(figsize=(14, 5.2))
        ax = fig.add_axes([0.01, 0.02, 0.98, 0.86])
        im, q = _panel(ax, d, campo, *rangos[campo], cmap)
        cax = ax.inset_axes([0.62, 0.05, 0.33, 0.06])
        fig.colorbar(im, cax=cax, orientation="horizontal", label=etiqueta,
                     extend="max" if campo == "V" else "neither")
        sup = fig.suptitle("", fontsize=14)
        writer = FFMpegWriter(**WRITER_KW)
        with writer.saving(fig, os.path.join(out_dir, f"{campo}.mp4"), dpi=110):
            for k in range(d[campo].shape[0]):
                _actualizar(im, q, d, campo, k)
                sup.set_text(f"{etiqueta}  ·  {LABEL_B[backend]}  ·  {titulo}"
                             f"   |   t = {_t_dias(d, k):.2f} días")
                writer.grab_frame()
        plt.close(fig)


def animar_caso(g, c, spec, force=False):
    caso_dir = os.path.join(OUT_ROOT, g, c)
    ref = spec.get("ref_fem", f"{g}/{c}")
    fuentes = {"bfr": os.path.join(caso_dir, "bfr"),
               "fenicsx": os.path.join(OUT_ROOT, ref, "fenicsx")}
    fuentes = {b: p for b, p in fuentes.items()
               if os.path.exists(os.path.join(p, "simulation_results.npz"))}
    tag = f"{g}/{c}"
    if not fuentes:
        print(f"[animar] ⏭  {tag}: sin resultados", flush=True)
        return False
    hecho = os.path.join(fuentes.get("bfr", caso_dir), "figures", "C.mp4")
    # Un video sólo vale si es más nuevo que TODOS los resultados que muestra (si no, quedó viejo).
    npz_mtime = max(os.path.getmtime(os.path.join(p, "simulation_results.npz")) for p in fuentes.values())
    if os.path.exists(hecho) and not force and os.path.getmtime(hecho) >= npz_mtime:
        print(f"[animar] ⏭  {tag}: ya animado y vigente (usar --force)", flush=True)
        return True

    datos = {b: _cargar(os.path.join(p, "simulation_results.npz")) for b, p in fuentes.items()}
    rangos = _rangos(list(datos.values()))
    # Videos individuales sólo de los backends corridos en este caso (no de ref_fem).
    for b, d in datos.items():
        if fuentes[b].startswith(caso_dir):
            video_backend(d, b, c, rangos, os.path.join(fuentes[b], "figures"))
    # La grilla BFR|FEM ya no se hace aquí: la comparación (con diferencia y
    # curvas) la genera comparar.py en la misma pasada que las métricas.
    print(f"[animar] ✓  {tag}", flush=True)
    return True


def _trabajo(args):
    g, c, force = args
    spec = GRUPOS[g]["casos"][c]
    try:
        return animar_caso(g, c, spec, force)
    except Exception as e:  # un caso roto no detiene al resto
        print(f"[animar] ✗  {g}/{c}: {e!r}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grupo", nargs="+")
    ap.add_argument("--caso", nargs="+")
    ap.add_argument("--todos", action="store_true", help="todos los grupos de comparación (A–F)")
    ap.add_argument("--force", action="store_true", help="rehacer aunque ya existan los videos")
    ap.add_argument("--jobs", type=int, default=4, help="casos en paralelo")
    args = ap.parse_args()
    if not (args.grupo or args.caso or args.todos):
        ap.error("indica --grupo, --caso o --todos")

    tareas = [(g, c, args.force) for g, c, _ in iter_casos(args.grupo, args.caso)
              if GRUPOS[g].get("comparar", True)]
    with Pool(min(args.jobs, len(tareas)) or 1) as pool:
        ok = pool.map(_trabajo, tareas, chunksize=1)
    if not all(ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
