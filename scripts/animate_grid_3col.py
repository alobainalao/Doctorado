#!/usr/bin/env python
"""
animate_grid_3col.py  <campo>

campo ∈ {H, dH, C, V, C_im, adj_H, adj_C}   (dH = h − h del primer paso guardado)

Grilla 3 × 3 como la de animate_grid.py (filas ADR / MRMT semi / MRMT bloque; columnas BFR / FEniCS) con una
tercera columna BFR − FEniCS (escala simétrica propia, mapa RdBu_r). Mismos mapas de color y rangos de la
presentación; si los datos salen del rango habitual, el rango se amplía y se avisa por consola.
Lee <EXP_ROOT>/NA_<modelo>/<backend>/{simulation,adjoint}_results.npz (EXP_ROOT por defecto
data/output/experimentos/N_adjunto) y escribe <EXP_ROOT>/grid3/grid3_<campo>.mp4.
Los adjuntos de BFR no están validados. V: |u| con flechas (dirección de u). C_im: media de las regiones inmóviles.
Adjuntos en la convención del código (L = J − ψ·R), como la presentación; el documento usa ψ = −ψ_código/Δt.
"""
import copy
import os
import sys

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Polygon as _Poly
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from view.plot import get_colormaps

campo = sys.argv[1] if len(sys.argv) > 1 else "C"
MODELOS = ["adr", "mrmt_semi", "mrmt_block"]
BACK = ["bfr", "fenicsx"]
LM = {"adr": "ADR", "mrmt_semi": "MRMT Semi", "mrmt_block": "MRMT Bloque"}
LB = {"bfr": "BFR — RBF-FD", "fenicsx": "FEniCS — FEM"}
PREFIJO = os.environ.get("EXP_PREFIJO", "NA_")      # carpeta de cada caso: <PREFIJO><modelo> (grupo M_adjunto: MA_)
ROOT = os.environ.get("EXP_ROOT", os.path.join(_CODIGO, "data", "output", "experimentos", "N_adjunto"))
is_adj = campo.startswith("adj_")

_XMIN, _XMAX, _ZMIN, _ZMAX = 816.0, 10755.0, -2690.5, 250.5
_VERTS = np.array([[816, -2690.5], [4354, -1757.5], [10755, -768.5], [10755, 250.5], [816, 250.5]])
_NX = 400
_NZ = round(_NX * (_ZMAX - _ZMIN) / (_XMAX - _XMIN))
_XI, _ZI = np.meshgrid(np.linspace(_XMIN, _XMAX, _NX), np.linspace(_ZMIN, _ZMAX, _NZ))

cmap_main, cmap_h = get_colormaps()
_adj = matplotlib.colormaps["YlOrRd"]
INFO = {
    "H": dict(lo=305.0, hi=309.0, cmap=cmap_main, lab="H  (m)", fmt="%.1f", pfx="t"),
    "dH": dict(lo=-0.6, hi=0.0, cmap=cmap_main, lab="h − h(t₁)  (m)", fmt="%.2f", pfx="t"),
    "C": dict(lo=0.0, hi=7.6e-11, cmap=cmap_h, lab="C  (kg/m³)", fmt="%.1e", pfx="t"),
    "C_im": dict(lo=0.0, hi=7.6e-11, cmap=cmap_h, lab="C_im  (kg/m³)", fmt="%.1e", pfx="t"),
    "V": dict(lo=4e-12, hi=4.5e-4, cmap=cmap_main, lab="|U|  (m/s)", fmt="%.1e", pfx="t"),
    "adj_H": dict(lo=0.0, hi=None, cmap=_adj, lab="ψ_H", fmt="%.2e", pfx="τ", gamma=0.35),
    "adj_C": dict(lo=0.0, hi=None, cmap=_adj, lab="ψ_C", fmt="%.2e", pfx="τ", gamma=0.35),
}[campo]
# H_RANGE="lo,hi": rango propio de h (el de la presentación, 305–309, apenas deja ver el cono); lo que baja de lo se pinta en gris
_h_rango = os.environ.get("H_RANGE") if campo == "H" else None
if _h_rango:
    INFO["lo"], INFO["hi"] = (float(x) for x in _h_rango.split(","))


def cargar(m, b):
    d0 = os.path.join(ROOT, f"{PREFIJO}{m}", b)
    if is_adj:
        z = np.load(os.path.join(d0, "adjoint_results.npz"), allow_pickle=True)
        fk = "psi_H" if campo == "adj_H" else "psi_C"
        if b == "bfr":
            return z["nodes"], np.asarray(z[fk], float)[:, 0], float(z["dt"])
        return (z["nodes_G"] if campo == "adj_H" else z["nodes_Q"]), np.asarray(z[fk], float), float(z["dt"])
    z = np.load(os.path.join(d0, "simulation_results.npz"), allow_pickle=True)
    dt = float(z["dt"])
    if campo == "V":
        U = np.asarray(z["U"], float)
        U = U[:, 0] if U.ndim == 4 else U
        return z["nodes"], np.linalg.norm(U, axis=-1), dt, U
    if campo == "C_im":
        if b == "bfr":
            return z["nodes"], np.asarray(z["Ci"], float)[:, :, 0].mean(axis=0), dt
        if "C_im" not in z.files:
            return None
        return z["nodes_Q"], np.asarray(z["C_im"], float).mean(axis=1), dt
    F = np.asarray(z["H" if campo in ("H", "dH") else "C"], float)[:, 0]
    return z["nodes"], (F - F[0] if campo == "dH" else F), dt


datos = {}
for m in MODELOS:
    for b in BACK:
        try:
            r = cargar(m, b)
        except (KeyError, FileNotFoundError):
            r = None
        if r is not None:
            n, f, dt = r[:3]
            datos[(m, b)] = dict(f=f, dl=Delaunay(np.asarray(n, float)), dt=dt, U=r[3] if len(r) > 3 else None)
if not datos:
    sys.exit("sin datos")
Nt = min(c["f"].shape[0] for c in datos.values())
dt = next(iter(datos.values()))["dt"]


def grilla(c, k):
    return LinearNDInterpolator(c["dl"], c["f"][k])(_XI, _ZI)


# Rangos: los habituales, ampliados si los datos los superan (adjunto: máximo global).
mx = max(float(np.nanmax(np.abs(c["f"][:Nt]))) for c in datos.values())
mn = min(float(np.nanmin(c["f"][:Nt])) for c in datos.values())
lo, hi = INFO["lo"], INFO["hi"] if INFO["hi"] is not None else mx
if INFO["hi"] is not None and campo != "dH" and (mx > hi or mn < lo):
    if os.environ.get("GRID3_RANGO_DATOS") and campo != "V":
        # GRID3_RANGO_DATOS=1: escala con el rango de los datos (otra malla o configuración). En V no: |u| es
        # singular en el pozo y con su máximo el resto del dominio sale uniforme; se satura solo alrededor del pozo.
        print(f"[grid3] datos [{mn:.3e}, {mx:.3e}] exceden [{lo:.3e}, {hi:.3e}]: se usa el rango de los datos", flush=True)
        lo, hi = mn, mx
    else:
        # rango de la presentación sin ampliar: lo que lo excede satura (como en los videos de la presentación)
        print(f"[grid3] datos [{mn:.3e}, {mx:.3e}] exceden [{lo:.3e}, {hi:.3e}]: se satura, rango de la presentación", flush=True)
if campo == "dH":
    lo, hi = mn, max(mx, 0.0)
    print(f"[grid3] dH rango propio [{lo:.3f}, {hi:.3f}] m", flush=True)
if is_adj:
    # como en la presentación: YlOrRd con PowerNorm(0.35) desde 0, una escala por backend (BFR y FEniCS difieren en
    # órdenes de magnitud); lo negativo se muestra en blanco. Convención del código (ψ_doc = −ψ/Δt).
    esc = {b: max(float(np.nanmax(c["f"][:Nt])) for (mm, bb), c in datos.items() if bb == b) for b in BACK}
    cm = copy.copy(INFO["cmap"])
    norm = {b: PowerNorm(INFO["gamma"], vmin=0.0, vmax=esc[b]) for b in BACK}
else:
    cm = copy.copy(INFO["cmap"]); norm = Normalize(vmin=lo, vmax=hi)
cm.set_bad("white")
if _h_rango:
    cm.set_under("0.85")

# Escala simétrica de la diferencia: máximo sobre una muestra de instantes.
ks = np.unique(np.linspace(0, Nt - 1, 12).astype(int))
dmax = 0.0
sb, sf = (esc["bfr"], esc["fenicsx"]) if is_adj else (1.0, 1.0)   # adjuntos: diferencia de campos normalizados
for m in MODELOS:
    if (m, "bfr") in datos and (m, "fenicsx") in datos:
        for k in ks:
            dd = np.abs(grilla(datos[(m, "bfr")], k) / sb - grilla(datos[(m, "fenicsx")], k) / sf)
            dmax = max(dmax, float(np.nanpercentile(dd, 99.5)))   # robusto: un pico en el pozo no fija la escala
dmax = dmax or 1.0
dnorm = Normalize(vmin=-dmax, vmax=dmax)

# ── Figura ───────────────────────────────────────────────────────────────────
DA = (_XMAX - _XMIN) / (_ZMAX - _ZMIN)
L, R, TOP, BOT, HG, VG = 0.03, 0.006, 0.17, 0.008, 0.012, 0.012
CW = (1 - L - R - 2 * HG) / 3
RH = (1 - TOP - BOT - 2 * VG) / 3
FH = 13.0
FW = DA * RH * FH / CW
fig = plt.figure(figsize=(FW, FH)); fig.patch.set_facecolor("white")
ax, im = {}, {}
for i, m in enumerate(MODELOS):
    y0 = 1 - TOP - (i + 1) * RH - i * VG
    for j in range(3):
        a = fig.add_axes([L + j * (CW + HG), y0, CW, RH])
        a.set_xlim(_XMIN, _XMAX); a.set_ylim(_ZMIN, _ZMAX); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.6); sp.set_edgecolor("#bbbbbb")
        ax[(i, j)] = a
        clip = _Poly(_VERTS, transform=a.transData, fc="none", ec="none"); a.add_patch(clip)
        if j < 2 and (m, BACK[j]) not in datos or j == 2 and not all((m, b) in datos for b in BACK):
            a.text(0.5, 0.5, "sin región inmóvil" if campo == "C_im" else "sin datos", ha="center",
                   va="center", transform=a.transAxes, fontsize=16)
            continue
        g = np.full(_XI.shape, np.nan)
        im[(i, j)] = a.imshow(g, origin="lower", extent=[_XMIN, _XMAX, _ZMIN, _ZMAX], aspect="auto",
                              norm=dnorm if j == 2 else (norm[BACK[j]] if is_adj else norm), cmap="RdBu_r" if j == 2 else cm,
                              interpolation="antialiased")
        im[(i, j)].set_clip_path(clip)


# Flechas de u (sólo campo V): dirección normalizada en puntos de una malla gruesa dentro del dominio.
from matplotlib.path import Path as _P
qv = {}
if campo == "V":
    _QX, _QZ = np.meshgrid(np.linspace(_XMIN, _XMAX, 45), np.linspace(_ZMIN, _ZMAX, 14))
    _QA = np.column_stack([_QX.ravel(), _QZ.ravel()])
    _QP = _QA[_P(_VERTS).contains_points(_QA)]


def flechas(c, k):
    ux = np.nan_to_num(LinearNDInterpolator(c["dl"], c["U"][k][:, 0])(_QP))
    uz = np.nan_to_num(LinearNDInterpolator(c["dl"], c["U"][k][:, 1])(_QP))
    mag = np.maximum(np.hypot(ux, uz), 1e-30)
    return ux / mag, uz / mag


if campo == "V":
    for i, m in enumerate(MODELOS):
        for j, b in enumerate(BACK):
            if (i, j) in im:
                u0, w0 = flechas(datos[(m, b)], 0)
                qv[(i, j)] = ax[(i, j)].quiver(_QP[:, 0], _QP[:, 1], u0, w0, scale=42, scale_units="width",
                                               width=0.0022, headwidth=4, headlength=4, color="black",
                                               alpha=0.65, zorder=5)


def barra(x0, x1, nrm, cmp_, etq, fmt, ticks=None):
    cax = fig.add_axes([x0, 0.882, x1 - x0, 0.036])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmp_, norm=nrm), cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top"); cb.ax.xaxis.set_label_position("top")
    cb.set_label(etq, fontsize=19, labelpad=8, color="#cc0000", fontweight="bold")
    cb.ax.tick_params(labelsize=16, top=True, bottom=False, labelcolor="#cc0000")
    cb.set_ticks(ticks if ticks is not None else np.linspace(nrm.vmin, nrm.vmax, 5))
    cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))


p0, p1, p2 = ax[(0, 0)].get_position(), ax[(0, 1)].get_position(), ax[(0, 2)].get_position()
if is_adj:
    barra(p0.x0 + 0.02, p0.x1 - 0.03, norm["bfr"], cm, INFO["lab"] + " BFR", INFO["fmt"], ticks=np.linspace(0, esc["bfr"], 5))
    barra(p1.x0 + 0.02, p1.x1 - 0.03, norm["fenicsx"], cm, INFO["lab"] + " FEniCS", INFO["fmt"], ticks=np.linspace(0, esc["fenicsx"], 5))
    barra(p2.x0 + 0.03, p2.x1 - 0.03, dnorm, matplotlib.colormaps["RdBu_r"], "BFR/máx − FEniCS/máx", "%.2f")
else:
    barra(p0.x0 + 0.02, p1.x1 - 0.03, norm, cm, INFO["lab"], INFO["fmt"])
    barra(p2.x0 + 0.03, p2.x1 - 0.03, dnorm, matplotlib.colormaps["RdBu_r"], "BFR − FEniCS", "%.1e")
timer = fig.text(p2.x1, 0.992, f"{INFO['pfx']} = 0.0 h", ha="right", va="top", fontsize=22,
                 fontweight="bold", fontfamily="monospace", color="#000080",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaaacc", lw=1.5, alpha=0.92))
for j, tt in enumerate([LB["bfr"], LB["fenicsx"],
                        "BFR/máx − FEniCS/máx" if is_adj else "BFR − FEniCS"]):
    p = ax[(0, j)].get_position()
    fig.text(p.x0 + p.width / 2, p.y1 + 0.006, tt, ha="center", va="bottom", fontsize=20,
             fontweight="bold", color="#cc0000")
for i, m in enumerate(MODELOS):
    p = ax[(i, 0)].get_position()
    fig.text(p.x0 - 0.004, p.y0 + p.height / 2, LM[m], ha="center", va="center", fontsize=17,
             fontweight="bold", color="#cc0000", rotation=90)

out_dir = os.path.join(ROOT, "grid3"); os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f"grid3_{campo}{'_cono' if _h_rango else ''}.mp4")
orden = range(Nt - 1, -1, -1) if is_adj else range(Nt)
print(f"[grid3] {campo}: Nt={Nt} → {out}", flush=True)
w = FFMpegWriter(fps=20, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "16"])
with w.saving(fig, out, dpi=110):
    for n, k in enumerate(orden):
        timer.set_text(f"{INFO['pfx']} = {n * dt / 3600.0:.1f} h")
        for i, m in enumerate(MODELOS):
            gs = {b: grilla(datos[(m, b)], k) for b in BACK if (m, b) in datos}
            for j, b in enumerate(BACK):
                if (i, j) in im:
                    im[(i, j)].set_data(gs[b])
                if (i, j) in qv:
                    qv[(i, j)].set_UVC(*flechas(datos[(m, b)], k))
            if (i, 2) in im:
                im[(i, 2)].set_data(gs["bfr"] / sb - gs["fenicsx"] / sf)
        w.grab_frame()
plt.close(fig)
print("[grid3] ✓", out, flush=True)
