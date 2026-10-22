#!/usr/bin/env python
"""
animate_grid.py  <field>  [run_type]

field ∈ {H, C, V, adj_H, adj_C}

Genera grid_<field>.mp4: grilla 3 × 2  (ADR / MRMT Semi / MRMT Bloque) × (BFR / FEniCS)
con colorbar global etiquetada, timer y dominio pentagonal clipado.
Sin título en el video — el título viene del SlideLayout de React.

Salida: ./data/output/grid/grid_<field>.mp4
"""
import sys, os
_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import PowerNorm, Normalize, LogNorm
from matplotlib.patches import Polygon as _MplPolygon
from matplotlib.path import Path as _MplPath
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import copy as _copy

# ── Argumentos ─────────────────────────────────────────────────────────────
field    = sys.argv[1] if len(sys.argv) > 1 else "H"
run_type = sys.argv[2] if len(sys.argv) > 2 else "comparison"

MODELS   = ["adr", "mrmt_semi", "mrmt_block"]
BACKENDS = ["bfr", "fenicsx"]
LABELS_M = {"adr": "ADR", "mrmt_semi": "MRMT Semi", "mrmt_block": "MRMT Bloque"}
LABELS_B = {"bfr": "BFR — RBF-FD", "fenicsx": "FEniCS — FEM"}

is_adj  = field.startswith("adj_")
_suffix = f"/{run_type}" if run_type else ""

# ── Dominio físico ──────────────────────────────────────────────────────────
_XMIN, _XMAX = 816.0, 10755.0
_ZMIN, _ZMAX = -2690.5, 250.5
_DA          = (_XMAX - _XMIN) / (_ZMAX - _ZMIN)   # 3.3795 (aspect ratio físico)

_DOMAIN_VERTS = np.array([
    [ 816,  -2690.5],
    [4354,  -1757.5],
    [10755,  -768.5],
    [10755,   250.5],
    [ 816,    250.5],
])
_DOMAIN_PATH = _MplPath(_DOMAIN_VERTS)

# ── Rejilla de interpolación ────────────────────────────────────────────────
_NX_G = 400
_NZ_G = max(1, round(_NX_G * (_ZMAX - _ZMIN) / (_XMAX - _XMIN)))
_XI, _ZI = np.meshgrid(np.linspace(_XMIN, _XMAX, _NX_G),
                       np.linspace(_ZMIN, _ZMAX, _NZ_G))

# ── Colormaps ───────────────────────────────────────────────────────────────
from view.plot import get_colormaps
cmap_main, cmap_h = get_colormaps()
_cmap_adj = matplotlib.colormaps["YlOrRd"]

_FIELD_INFO = {
    "H":     dict(vmin=305.0, vmax=309.0,   cmap=cmap_main,  cb_label="H  (m)",
                  cb_fmt="%.0f",     timer_pfx="t"),
    "C":     dict(vmin=0.0,   vmax=7.6e-11, cmap=cmap_h,     cb_label="C  (kg/m³)",
                  cb_fmt="%.1e",     timer_pfx="t"),
    "V":     dict(vmin=4e-12, vmax=4.5e-4,  cmap=cmap_main,  cb_label="|U|  (m/s)",
                  cb_fmt="%.1e",     timer_pfx="t"),
    "adj_H": dict(vmin=0.0,   vmax=None,    cmap=_cmap_adj,  cb_label="ψ_H",
                  cb_fmt="%.2e",     timer_pfx="τ",  norm_gamma=0.35),
    "adj_C": dict(vmin=0.0,   vmax=None,    cmap=_cmap_adj,  cb_label="ψ_C",
                  cb_fmt="%.2e",     timer_pfx="τ",  norm_gamma=0.35),
}
_fi = _FIELD_INFO[field]

# ── Cargar datos ────────────────────────────────────────────────────────────
_exp_root = os.environ.get("EXP_ROOT")   # experimentos/: <EXP_ROOT>/<model>/<backend>/

def _load_case(model, backend):
    root_data = (f"{_exp_root}/{model}/{backend}" if _exp_root
                 else f"./data/output/{backend}/data/real/{model}{_suffix}")
    if is_adj:
        path = f"{root_data}/adjoint_results.npz"
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        _ibfr = "nodes_G" not in d
        f_key = "psi_H" if field == "adj_H" else "psi_C"
        if _ibfr:
            frames = d[f_key][:, 0, :]
            nodes  = d["nodes"]
        else:
            frames = d[f_key]
            nodes  = d["nodes_G"] if field == "adj_H" else d["nodes_Q"]
        dt = float(d["dt"])
        return {"frames": frames, "nodes": nodes, "dt": dt, "Nt": frames.shape[0]}
    else:
        path = f"{root_data}/simulation_results.npz"
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        nodes = d["nodes"]
        dt    = float(d["dt"])
        if field == "V":
            if "U" not in d:
                return None
            U = d["U"]
            if U.ndim == 4:   # BFR: (Nt, 1, N, 2)
                U = U[:, 0, :, :]
            return {"frames": U, "nodes": nodes, "dt": dt, "Nt": U.shape[0],
                    "is_velocity": True}
        fkey = "H" if field == "H" else "C"
        frames = d[fkey][:, 0, :]
        return {"frames": frames, "nodes": nodes, "dt": dt, "Nt": frames.shape[0]}

print(f"[grid] field={field}  run_type={run_type}", flush=True)
cases = {}
for m in MODELS:
    for b in BACKENDS:
        c = _load_case(m, b)
        if c is None:
            print(f"[grid]   SKIP {m}/{b} — no encontrado", flush=True)
        else:
            cases[(m, b)] = c
            print(f"[grid]   {m}/{b}: Nt={c['Nt']} dt={c['dt']:.0f}s", flush=True)

if not cases:
    print("[grid] Sin datos, abortando."); sys.exit(1)

# ── Escala global ───────────────────────────────────────────────────────────
vmin = _fi["vmin"]
vmax = _fi["vmax"]
if vmax is None:
    vmax_vals = [float(np.nanmax(np.abs(c["frames"]))) for c in cases.values()]
    vmax = max(vmax_vals)
    print(f"[grid] vmax adjunto global = {vmax:.3e}", flush=True)

cmap_used = _copy.copy(_fi["cmap"])
cmap_used.set_bad("white")

_gamma   = _fi.get("norm_gamma")
_use_log = _fi.get("use_log", False)
if _gamma:
    norm_obj = PowerNorm(_gamma, vmin=vmin, vmax=vmax)
elif _use_log:
    norm_obj = LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
else:
    norm_obj = Normalize(vmin=vmin, vmax=vmax)

Nt = min(c["Nt"] for c in cases.values())
dt = next(iter(cases.values()))["dt"]

# ── Dimensiones figura ──────────────────────────────────────────────────────
LEFT_LBL  = 0.028   # etiquetas verticales → margen mínimo
RIGHT_MGN = 0.006
TOP_HDR   = 0.175   # solo espacio para headers de columna sobre las celdas
BOT_MGN   = 0.008   # la colorbar/timer flotan en posición fija CB_BOTTOM=0.840
HGAP      = 0.015
VGAP      = 0.012

AVAIL_W = 1.0 - LEFT_LBL - RIGHT_MGN - HGAP
AVAIL_H = 1.0 - TOP_HDR  - BOT_MGN   - VGAP * 2
COL_W   = AVAIL_W / 2
ROW_H   = AVAIL_H / 3

FIG_H = 13.0
FIG_W = _DA * (ROW_H * FIG_H) / COL_W
DPI   = 180   # mayor resolución

print(f"[grid] figsize=({FIG_W:.2f}, {FIG_H:.2f}) DPI={DPI} → "
      f"{int(FIG_W*DPI)}×{int(FIG_H*DPI)} px", flush=True)

# ── Figura ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")

# ── Axes de datos ───────────────────────────────────────────────────────────
axes = {}
for ri, model in enumerate(MODELS):
    y0 = 1.0 - TOP_HDR - (ri + 1) * ROW_H - ri * VGAP
    for ci, backend in enumerate(BACKENDS):
        x0 = LEFT_LBL + ci * (COL_W + HGAP)
        ax = fig.add_axes([x0, y0, COL_W, ROW_H])
        ax.set_xlim(_XMIN, _XMAX)
        ax.set_ylim(_ZMIN, _ZMAX)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_edgecolor("#bbbbbb")
        axes[(model, backend)] = ax

# ── Colorbar global (arriba, entre headers y primera fila) ──────────────────
pos_tl = axes[(MODELS[0], BACKENDS[0])].get_position()
pos_tr = axes[(MODELS[0], BACKENDS[1])].get_position()

pos_tl = axes[(MODELS[0], BACKENDS[0])].get_position()
pos_tr = axes[(MODELS[0], BACKENDS[1])].get_position()
cbar_x0 = pos_tl.x0
cbar_x1 = pos_tr.x1

# Orden vertical: TIMER → COLORBAR → GAP → GRID
# El GAP es el espacio en blanco entre la colorbar y el tope del grid.
# pos_tl.y1 = tope de la primera fila de celdas.
# Colorbar fija en la banda superior, lejos del grid.
CB_BOTTOM = 0.882   # fijo: gap visible entre colorbar y headers del grid
CB_HEIGHT = 0.036

cax = fig.add_axes([cbar_x0, CB_BOTTOM, cbar_x1 - cbar_x0, CB_HEIGHT])
sm  = plt.cm.ScalarMappable(cmap=cmap_used, norm=norm_obj)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.set_label(_fi["cb_label"], fontsize=19, labelpad=8,
             color="#cc0000", fontweight="bold")
cb.ax.tick_params(labelsize=18, top=True, bottom=False, labelcolor="#cc0000")
if _fi.get("use_log"):
    cb.ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    cb.ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
elif field in ("adj_H", "adj_C"):
    cb.set_ticks(np.linspace(0.0, vmax, 5))
    cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
elif field == "H":
    cb.set_ticks([305, 306, 307, 308, 309])
    cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
elif field == "C":
    cb.set_ticks([0, 1.9e-11, 3.8e-11, 5.7e-11, 7.6e-11])
    cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
else:
    cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(_fi["cb_fmt"]))

# ── Timer (encima de la colorbar, esquina superior derecha) ────────────────
timer_txt = fig.text(
    cbar_x1, 0.992,
    f"{_fi['timer_pfx']} = 0.0 h",
    ha="right", va="top",
    fontsize=22, fontweight="bold", fontfamily="monospace",
    color="#000080",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaaacc", lw=1.5, alpha=0.92),
)

# ── Encabezados de columna (pegados al tope del grid) ──────────────────────
for ci, backend in enumerate(BACKENDS):
    pos = axes[(MODELS[0], backend)].get_position()
    fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.006,
             LABELS_B[backend], ha="center", va="bottom",
             fontsize=20, fontweight="bold", color="#cc0000")

# ── Etiquetas de fila (verticales, pegadas a la izquierda del grid) ─────────
for ri, model in enumerate(MODELS):
    pos = axes[(model, BACKENDS[0])].get_position()
    fig.text(pos.x0 - 0.004, pos.y0 + pos.height / 2,
             LABELS_M[model], ha="center", va="center",
             fontsize=17, fontweight="bold", color="#cc0000",
             rotation=90)

# ── Inicializar imshow + clip por panel ─────────────────────────────────────
imgs = {}
for m in MODELS:
    for b in BACKENDS:
        if (m, b) not in cases:
            continue
        c   = cases[(m, b)]
        ax  = axes[(m, b)]
        _dl = Delaunay(c["nodes"])

        clip = _MplPolygon(_DOMAIN_VERTS, transform=ax.transData,
                           fc="none", ec="none", zorder=0)
        ax.add_patch(clip)

        if c.get("is_velocity"):
            F0 = LinearNDInterpolator(_dl, np.linalg.norm(c["frames"][0], axis=-1))(_XI, _ZI)
        elif is_adj:
            F0 = LinearNDInterpolator(_dl, c["frames"][Nt - 1])(_XI, _ZI)
        else:
            F0 = LinearNDInterpolator(_dl, c["frames"][0])(_XI, _ZI)

        im = ax.imshow(F0, origin="lower",
                       extent=[_XMIN, _XMAX, _ZMIN, _ZMAX],
                       norm=norm_obj, cmap=cmap_used,
                       aspect="auto", interpolation="antialiased")
        im.set_clip_path(clip)
        imgs[(m, b)] = {"im": im, "dl": _dl, "frames": c["frames"],
                        "is_vel": c.get("is_velocity", False)}

# ── Quiver para V ───────────────────────────────────────────────────────────
qvrs = {}
if field == "V":
    _QNX, _QNZ = 45, 14
    _QX, _QZ   = np.meshgrid(np.linspace(_XMIN, _XMAX, _QNX),
                              np.linspace(_ZMIN, _ZMAX, _QNZ))
    _QALL = np.column_stack([_QX.ravel(), _QZ.ravel()])
    _QPTS = _QALL[_DOMAIN_PATH.contains_points(_QALL)]

    for m in MODELS:
        for b in BACKENDS:
            if (m, b) not in imgs:
                continue
            info = imgs[(m, b)]
            U0   = info["frames"][0]
            ux   = np.nan_to_num(LinearNDInterpolator(info["dl"], U0[:, 0])(_QPTS))
            uz   = np.nan_to_num(LinearNDInterpolator(info["dl"], U0[:, 1])(_QPTS))
            mag  = np.maximum(np.sqrt(ux**2 + uz**2), 1e-30)
            qvrs[(m, b)] = axes[(m, b)].quiver(
                _QPTS[:, 0], _QPTS[:, 1], ux / mag, uz / mag,
                scale=42, scale_units="width", width=0.0022,
                headwidth=4, headlength=4, color="black", alpha=0.65, zorder=5,
            )


# ── Animación ───────────────────────────────────────────────────────────────
out_dir  = f"{_exp_root}/grid" if _exp_root else "./data/output/grid"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/grid_{field}.mp4"

_WRITER_KW = dict(fps=20, codec="h264",
                  extra_args=["-pix_fmt", "yuv420p", "-crf", "14"])

writer      = FFMpegWriter(**_WRITER_KW)
frame_iter  = range(Nt - 1, -1, -1) if is_adj else range(Nt)

print(f"[grid] Generando {Nt} frames → {out_path}", flush=True)
with writer.saving(fig, out_path, dpi=DPI):
    for k in frame_iter:
        t_h = ((Nt - 1 - k) if is_adj else k) * dt / 3600.0
        timer_txt.set_text(f"{_fi['timer_pfx']} = {t_h:.1f} h")

        for m in MODELS:
            for b in BACKENDS:
                if (m, b) not in imgs:
                    continue
                info = imgs[(m, b)]
                if info["is_vel"]:
                    Umag = np.linalg.norm(info["frames"][k], axis=-1)
                    info["im"].set_data(LinearNDInterpolator(info["dl"], Umag)(_XI, _ZI))
                    if (m, b) in qvrs:
                        Uk  = info["frames"][k]
                        ux  = np.nan_to_num(LinearNDInterpolator(info["dl"], Uk[:, 0])(_QPTS))
                        uz  = np.nan_to_num(LinearNDInterpolator(info["dl"], Uk[:, 1])(_QPTS))
                        mag = np.maximum(np.sqrt(ux**2 + uz**2), 1e-30)
                        qvrs[(m, b)].set_UVC(ux / mag, uz / mag)
                else:
                    info["im"].set_data(
                        LinearNDInterpolator(info["dl"], info["frames"][k])(_XI, _ZI)
                    )

        writer.grab_frame()

plt.close(fig)
print(f"[grid] ✓ {out_path}", flush=True)
