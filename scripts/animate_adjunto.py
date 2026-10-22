#!/usr/bin/env python
"""
animate_adjunto.py  <model>  <metodo>  [run_type]

Genera adjunto_psi_H.mp4, adjunto_psi_C.mp4 y adjunto_panel.mp4.

Diseño para comparación justa entre 6 casos:
  - extent fijo al dominio geométrico real (816,10755) × (-2690.5,250.5)
  - exterior cubierto con PathPatch blanco (sin NaN → sin halos por bilinear)
  - escala de color por backend (bfr / fenicsx) — misma para los 3 modelos
  - figsize consistente, axes ocupa casi toda la figura
  - animación T → 0 (adjunto retrocede en tiempo)

Uso:
    python scripts/animate_adjunto.py adr bfr
    python scripts/animate_adjunto.py adr bfr comparison
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

for p in [f"{root_data}/adjoint_results.npz", f"{root_data}/simulation_results.npz"]:
    if not os.path.exists(p):
        print(f"[animate] NO encontrado: {p}"); sys.exit(1)

os.makedirs(root_fig, exist_ok=True)

DPI = 150

# ─── Dominio físico fijo — idéntico para BFR y FEniCS ────────────────────────
_XMIN, _XMAX =  816.0, 10755.0
_ZMIN, _ZMAX = -2690.5, 250.5

_DOMAIN_VERTS = np.array([
    [816,   -2690.5],
    [4354,  -1757.5],
    [10755,  -768.5],
    [10755,   250.5],
    [816,     250.5],
])

def _domain_clip(ax):
    """Clip invisible del dominio pentagonal para set_clip_path()."""
    patch = _MplPolygon(_DOMAIN_VERTS, transform=ax.transData,
                        fc="none", ec="none", zorder=0)
    ax.add_patch(patch)
    return patch

# ─── figsize y axes rect ──────────────────────────────────────────────────────
_FIG_W, _FIG_H  = 18.0, 7.0
_AX_RECT        = [0.01, 0.01, 0.98, 0.90]

# ─── Rejilla de interpolación compartida ─────────────────────────────────────
_NX_G = 500
_NZ_G = max(1, round(_NX_G * (_ZMAX - _ZMIN) / (_XMAX - _XMIN)))
_XI, _ZI = np.meshgrid(np.linspace(_XMIN, _XMAX, _NX_G),
                       np.linspace(_ZMIN, _ZMAX, _NZ_G))

# ─── Cargar datos ─────────────────────────────────────────────────────────────
adj = np.load(f"{root_data}/adjoint_results.npz", allow_pickle=True)
sim = np.load(f"{root_data}/simulation_results.npz", allow_pickle=True)

is_bfr = "nodes" in adj and "nodes_G" not in adj

if is_bfr:
    psi_H    = adj["psi_H"][:, 0, :]
    psi_C    = adj["psi_C"][:, 0, :]
    nodes_H  = nodes_C = adj["nodes"]
else:
    psi_H    = adj["psi_H"]
    psi_C    = adj["psi_C"]
    nodes_H  = adj["nodes_G"]
    nodes_C  = adj["nodes_Q"]

dt  = float(adj["dt"])
Nt  = psi_C.shape[0]
Q0  = float(sim["Q"][0]) if "Q" in sim else None
print(f"[animate] {metodo}/{model}: Nt={Nt}  dt={dt:.0f}s")

# ─── Colormap adjunto ─────────────────────────────────────────────────────────
import copy as _copy
cmap_adj = matplotlib.colormaps["YlOrRd"]

# ─── Escala por backend (misma para los 3 modelos de cada backend) ────────────
_MODELS_ALL = ["adr", "mrmt_semi", "mrmt_block"]
_vmH_b = {"bfr": [], "fenicsx": []}
_vmC_b = {"bfr": [], "fenicsx": []}
for _m in _MODELS_ALL:
    for _b in ["bfr", "fenicsx"]:
        _p = f"{_data_dir(_b, _m)}/adjoint_results.npz"
        if not os.path.exists(_p):
            continue
        _d    = np.load(_p, allow_pickle=True)
        _ibfr = "nodes" in _d and "nodes_G" not in _d
        _pH   = _d["psi_H"][:, 0, :] if _ibfr else _d["psi_H"]
        _pC   = _d["psi_C"][:, 0, :] if _ibfr else _d["psi_C"]
        _vmH_b[_b].append(float(np.nanmax(np.abs(_pH))))
        _vmC_b[_b].append(float(np.nanmax(np.abs(_pC))))

vmax_H = max(_vmH_b[metodo]) if _vmH_b[metodo] else 1.0
vmax_C = max(_vmC_b[metodo]) if _vmC_b[metodo] else 1.0
print(f"[animate] Escala {metodo}: vmax_H={vmax_H:.3e}  vmax_C={vmax_C:.3e}")

tag = f"{LABEL_M.get(model, model)} · {LABEL_B.get(metodo, metodo)}"

_WRITER_KW = dict(fps=20, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])


def _render_single(out_path, nodes, frames, vmin, vmax, cmap, suptitle_base, Q0=None):
    """Panel único — adjunto_psi_H.mp4 o adjunto_psi_C.mp4. Corre T → 0."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay as ScipyDelaunay

    _del  = ScipyDelaunay(nodes)
    cmap_i = _copy.copy(cmap)
    cmap_i.set_bad(color="white")

    F0 = LinearNDInterpolator(_del, frames[Nt - 1])(_XI, _ZI)

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    ax  = fig.add_axes(_AX_RECT)
    ax.axis("off")
    ax.set_facecolor("white")
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
        for k in range(Nt - 1, -1, -1):
            im.set_data(LinearNDInterpolator(_del, frames[k])(_XI, _ZI))
            t_adj_h = (Nt - 1 - k) * dt / 3600.0
            fig.suptitle(f"{suptitle_base}   |   t_adj = {t_adj_h:.1f} h desde T",
                         fontsize=18, color="#0000FF", y=0.995)
            writer.grab_frame()
    plt.close(fig)
    print(f"[animate] ✓ {out_path}")


# ─── adjunto_psi_H.mp4 ────────────────────────────────────────────────────────
_render_single(
    f"{root_fig}/adjunto_psi_H.mp4",
    nodes_H, psi_H, 0.0, vmax_H, cmap_adj,
    f"Adjoint ψ_H  ·  {tag}", Q0=Q0,
)

# ─── adjunto_psi_C.mp4 ────────────────────────────────────────────────────────
_render_single(
    f"{root_fig}/adjunto_psi_C.mp4",
    nodes_C, psi_C, 0.0, vmax_C, cmap_adj,
    f"Adjoint ψ_C  ·  {tag}", Q0=Q0,
)

# ─── adjunto_panel.mp4 (ψ_H arriba, ψ_C abajo) ───────────────────────────────
from scipy.interpolate import LinearNDInterpolator as _LNDI
from scipy.spatial import Delaunay as _ScipyDel

_del_H = _ScipyDel(nodes_H)
_del_C = _ScipyDel(nodes_C)

cmap_p = _copy.copy(cmap_adj)
cmap_p.set_bad(color="white")

FH0 = _LNDI(_del_H, psi_H[Nt - 1])(_XI, _ZI)
FC0 = _LNDI(_del_C, psi_C[Nt - 1])(_XI, _ZI)

fig2 = plt.figure(figsize=(_FIG_W, _FIG_H * 2))
ax_H = fig2.add_axes([0.01, 0.52, 0.98, 0.44])
ax_C = fig2.add_axes([0.01, 0.03, 0.98, 0.44])
ax_H.axis("off"); ax_H.set_facecolor("white")
ax_C.axis("off"); ax_C.set_facecolor("white")
clip_H = _domain_clip(ax_H)
clip_C = _domain_clip(ax_C)

_CI = [0.62, 0.04, 0.33, 0.07]

im_H2 = ax_H.imshow(FH0, origin="lower", extent=[_XMIN, _XMAX, _ZMIN, _ZMAX],
                     vmin=0.0, vmax=vmax_H, cmap=cmap_p,
                     aspect="equal", interpolation="antialiased")
im_H2.set_clip_path(clip_H)
ax_H.set_title(r"$\psi_H$ — adjunto de flujo", fontsize=13, fontweight="bold", loc="left")
fig2.colorbar(im_H2, cax=ax_H.inset_axes(_CI),
              orientation="horizontal").ax.tick_params(labelsize=8)

im_C2 = ax_C.imshow(FC0, origin="lower", extent=[_XMIN, _XMAX, _ZMIN, _ZMAX],
                     vmin=0.0, vmax=vmax_C, cmap=cmap_p,
                     aspect="equal", interpolation="antialiased")
im_C2.set_clip_path(clip_C)
ax_C.set_title(r"$\psi_C$ — adjunto de transporte", fontsize=13, fontweight="bold", loc="left")
fig2.colorbar(im_C2, cax=ax_C.inset_axes(_CI),
              orientation="horizontal").ax.tick_params(labelsize=8)

suptitle2 = fig2.suptitle("", fontsize=16, fontweight="bold", color="#0000FF", y=0.995)

writer2 = FFMpegWriter(**_WRITER_KW)
out_panel = f"{root_fig}/adjunto_panel.mp4"
with writer2.saving(fig2, out_panel, dpi=DPI):
    for k in range(Nt - 1, -1, -1):
        im_H2.set_data(_LNDI(_del_H, psi_H[k])(_XI, _ZI))
        im_C2.set_data(_LNDI(_del_C, psi_C[k])(_XI, _ZI))
        t_adj_h = (Nt - 1 - k) * dt / 3600.0
        suptitle2.set_text(f"{tag}   |   t_adj = {t_adj_h:.1f} h desde T")
        writer2.grab_frame()
plt.close(fig2)
print(f"[animate] ✓ {out_panel}")
