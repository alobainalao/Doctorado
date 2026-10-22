#!/usr/bin/env python
"""
animate_forward.py  <model>  <metodo>  [run_type]

Genera H.mp4, C.mp4 y V.mp4 leyendo simulation_results.npz.

Diseño para comparación justa entre 6 casos (3 modelos × 2 backends):
  - extent fijo al dominio geométrico real (816,10755) × (-2690.5,250.5)
  - imagen clipeada al polígono pentagonal con im.set_clip_path()
    → sin NaN en datos → sin halos; el fondo blanco del axes cubre el exterior
  - colormaps y rangos fijos idénticos para todos los casos
  - quiver solo dentro del polígono pentagonal del dominio
  - figsize y layout idénticos entre backends

Uso:
    python scripts/animate_forward.py adr bfr
    python scripts/animate_forward.py adr bfr comparison
"""
import sys, os
_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.path import Path as _MplPath
from matplotlib.patches import Polygon as _MplPolygon

# ─── Argumentos ───────────────────────────────────────────────────────────────
model    = sys.argv[1] if len(sys.argv) > 1 else "adr"
metodo   = sys.argv[2] if len(sys.argv) > 2 else "bfr"
run_type = sys.argv[3] if len(sys.argv) > 3 else ""

LABEL_M = {"adr": "ADR", "mrmt_semi": "MRMT Semi", "mrmt_block": "MRMT Bloque"}
LABEL_B = {"bfr": "BFR (RBF-FD)", "fenicsx": "FEniCS (FEM)"}

_suffix   = f"/{run_type}" if run_type else ""
# EXP_ROOT (experimentos/): datos en <EXP_ROOT>/<model>/<metodo>/ (ver experimentos/README.md)
_exp_root = os.environ.get("EXP_ROOT")
def _data_dir(b, m):
    return f"{_exp_root}/{m}/{b}" if _exp_root else f"./data/output/{b}/data/real/{m}{_suffix}"
root_data = _data_dir(metodo, model)
root_fig  = f"{root_data}/figures" if _exp_root else f"./data/output/{metodo}/figures/real/{model}{_suffix}"

sim_path = f"{root_data}/simulation_results.npz"
if not os.path.exists(sim_path):
    print(f"[animate_forward] NO encontrado: {sim_path}"); sys.exit(1)

os.makedirs(root_fig, exist_ok=True)

DPI = 150

# ─── Dominio físico — idéntico para BFR y FEniCS ────────────────────────────
_XMIN, _XMAX =  816.0, 10755.0
_ZMIN, _ZMAX = -2690.5, 250.5

_DOMAIN_VERTS = np.array([
    [816,   -2690.5],
    [4354,  -1757.5],
    [10755,  -768.5],
    [10755,   250.5],
    [816,     250.5],
])

# ─── figsize y axes rect ──────────────────────────────────────────────────────
_FIG_W, _FIG_H = 18.0, 7.0
_AX_RECT = [0.01, 0.01, 0.98, 0.90]

# ─── Rejilla de interpolación compartida ─────────────────────────────────────
_NX_G = 500
_NZ_G = max(1, round(_NX_G * (_ZMAX - _ZMIN) / (_XMAX - _XMIN)))
_XI, _ZI = np.meshgrid(np.linspace(_XMIN, _XMAX, _NX_G),
                       np.linspace(_ZMIN, _ZMAX, _NZ_G))

# ─── Rejilla quiver — solo puntos DENTRO del dominio ─────────────────────────
_DOMAIN_PATH = _MplPath(_DOMAIN_VERTS)
_QNX, _QNZ = 40, 15
_QXZ = np.meshgrid(np.linspace(_XMIN, _XMAX, _QNX),
                   np.linspace(_ZMIN, _ZMAX, _QNZ))
_Q_ALL = np.column_stack([_QXZ[0].ravel(), _QXZ[1].ravel()])
_Q_PTS = _Q_ALL[_DOMAIN_PATH.contains_points(_Q_ALL)]

# ─── Cargar datos ─────────────────────────────────────────────────────────────
sim = np.load(sim_path, allow_pickle=True)
has_U = "U" in sim

if metodo == "bfr":
    H_arr   = sim["H"][:, 0, :]
    C_arr   = sim["C"][:, 0, :]
    U_arr   = sim["U"][:, 0, :, :] if has_U else None
    nodes_H = sim["nodes"]
    nodes_C = sim["nodes"]
else:
    H_arr   = sim["H"][:, 0, :]
    C_arr   = sim["C"][:, 0, :]
    nodes_H = sim["nodes"]
    U_arr   = sim["U"] if has_U else None
    N_G = nodes_H.shape[0]
    if C_arr.shape[-1] == N_G:
        nodes_C = nodes_H
    elif "nodes_Q" in sim:
        nodes_C = sim["nodes_Q"]
    else:
        adj_path = f"{root_data}/adjoint_results.npz"
        nodes_C = None
        if os.path.exists(adj_path):
            adj_d   = np.load(adj_path, allow_pickle=True)
            nodes_C = adj_d.get("nodes_Q", None)
    if nodes_C is None:
        print("[animate_forward] FEniCS: nodes_Q no disponible, se omite C.mp4")

dt  = float(sim["dt"])
Nt  = H_arr.shape[0]
Q0  = float(sim["Q"][0]) if "Q" in sim else None

print(f"[animate_forward] {metodo}/{model}: Nt={Nt}  dt={dt:.0f}s  has_U={has_U}")
print(f"[animate_forward] quiver: {len(_Q_PTS)} pts dentro del dominio (de {_QNX*_QNZ})")

tag = f"{LABEL_M.get(model, model)} · {LABEL_B.get(metodo, metodo)}"

# ─── Colormaps y rangos fijos ─────────────────────────────────────────────────
from view.plot import get_colormaps
cmap_main, cmap_h = get_colormaps()

VMIN_H, VMAX_H = 305.0, 309.0
VMIN_V, VMAX_V = 4e-12, 4.5e-4
VMIN_C, VMAX_C = 0.0, 7.6e-11

_WRITER_KW = dict(fps=20, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])


def _make_fig_ax():
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    ax  = fig.add_axes(_AX_RECT)
    ax.axis("off")
    ax.set_facecolor("white")
    return fig, ax


def _domain_clip(ax):
    """Clip invisible del dominio pentagonal para set_clip_path()."""
    patch = _MplPolygon(_DOMAIN_VERTS, transform=ax.transData,
                        fc="none", ec="none", zorder=0)
    ax.add_patch(patch)
    return patch


def _render(out_path, nodes, frames, vmin, vmax, cmap, suptitle_prefix, Q0=None):
    """Panel único — H.mp4 o C.mp4."""
    import copy
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay as ScipyDelaunay

    _del = ScipyDelaunay(nodes)
    cmap_i = copy.copy(cmap)
    cmap_i.set_bad(color="white")

    F0 = LinearNDInterpolator(_del, frames[0])(_XI, _ZI)

    fig, ax = _make_fig_ax()
    clip = _domain_clip(ax)

    im = ax.imshow(F0, origin="lower",
                   extent=[_XMIN, _XMAX, _ZMIN, _ZMAX],
                   vmin=vmin, vmax=vmax, cmap=cmap_i,
                   aspect="equal", interpolation="antialiased")
    im.set_clip_path(clip)

    cax = ax.inset_axes([0.62, 0.04, 0.33, 0.07])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=8)
    if Q0 is not None:
        ax.set_title(f"Q = {Q0:.3e} m³/s", fontsize=10, pad=2)

    writer = FFMpegWriter(**_WRITER_KW)
    with writer.saving(fig, out_path, dpi=DPI):
        for k in range(frames.shape[0]):
            im.set_data(LinearNDInterpolator(_del, frames[k])(_XI, _ZI))
            t_h = k * dt / 3600.0
            fig.suptitle(f"{suptitle_prefix}   |   t = {t_h:.1f} h",
                         fontsize=18, color="#0000FF", y=0.995)
            writer.grab_frame()
    plt.close(fig)
    print(f"[animate_forward] ✓ {out_path}")


def _norm_uv(ux, uz):
    mag = np.maximum(np.sqrt(ux**2 + uz**2), 1e-30)
    return ux / mag, uz / mag


def _render_V(out_path, nodes, U, vmin, vmax, cmap, suptitle_prefix, Q0=None):
    """V.mp4 — magnitud |U| clipeada al dominio + quiver solo en interior."""
    import copy
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay as ScipyDelaunay

    Umag = np.linalg.norm(U, axis=2)
    Ux, Uz = U[:, :, 0], U[:, :, 1]
    _del = ScipyDelaunay(nodes)
    cmap_v = copy.copy(cmap)
    cmap_v.set_bad(color="white")

    def _iq(arr_t):
        return np.nan_to_num(LinearNDInterpolator(_del, arr_t)(_Q_PTS))

    V0 = LinearNDInterpolator(_del, Umag[0])(_XI, _ZI)
    ux0_n, uz0_n = _norm_uv(_iq(Ux[0]), _iq(Uz[0]))

    fig, ax = _make_fig_ax()
    clip = _domain_clip(ax)

    im = ax.imshow(V0, origin="lower",
                   extent=[_XMIN, _XMAX, _ZMIN, _ZMAX],
                   vmin=vmin, vmax=vmax, cmap=cmap_v,
                   aspect="equal", interpolation="antialiased")
    im.set_clip_path(clip)

    cax = ax.inset_axes([0.62, 0.04, 0.33, 0.07])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=8)
    if Q0 is not None:
        ax.set_title(f"Q = {Q0:.3e} m³/s", fontsize=10, pad=2)

    _qu = ax.quiver(_Q_PTS[:, 0], _Q_PTS[:, 1], ux0_n, uz0_n,
                    scale=50, scale_units="width",
                    width=0.0015, headwidth=4, headlength=4,
                    color="black", alpha=0.75, zorder=5)

    writer = FFMpegWriter(**_WRITER_KW)
    with writer.saving(fig, out_path, dpi=DPI):
        for k in range(U.shape[0]):
            im.set_data(LinearNDInterpolator(_del, Umag[k])(_XI, _ZI))
            _qu.set_UVC(*_norm_uv(_iq(Ux[k]), _iq(Uz[k])))
            t_h = k * dt / 3600.0
            fig.suptitle(f"{suptitle_prefix}   |   t = {t_h:.1f} h",
                         fontsize=18, color="#0000FF", y=0.995)
            writer.grab_frame()
    plt.close(fig)
    print(f"[animate_forward] ✓ {out_path}")


# ─── Generar videos ───────────────────────────────────────────────────────────
_render(f"{root_fig}/H.mp4", nodes_H, H_arr, VMIN_H, VMAX_H, cmap_main,
        f"Hydraulic Head H (m)  ·  {tag}", Q0=Q0)

if nodes_C is not None:
    _render(f"{root_fig}/C.mp4", nodes_C, C_arr, VMIN_C, VMAX_C, cmap_h,
            f"Concentration C (kg/m³)  ·  {tag}", Q0=Q0)
else:
    print("[animate_forward] C.mp4 omitido (sin nodes_C)")

if U_arr is not None:
    _render_V(f"{root_fig}/V.mp4", nodes_H, U_arr, VMIN_V, VMAX_V, cmap_main,
              f"Velocity |U| (m/s)  ·  {tag}", Q0=Q0)
else:
    print("[animate_forward] V.mp4 omitido (sin datos de velocidad)")
