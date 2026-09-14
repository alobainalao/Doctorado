"""
Animación de la trayectoria de optimización adjunta.

Produce una animación con 4 paneles que muestran la "huella" del optimizador
L-BFGS-B a lo largo de las iteraciones:

  ┌─────────────┬───────────────┬──────────────┐
  │ J(iter)     │  z_p(iter)    │  Q̄(iter)     │
  │ convergencia│  profundidad  │  caudal med  │
  ├─────────────┴───────────────┴──────────────┤
  │      Q(t) — perfil de extracción            │
  │      huella: iteraciones pasadas en gris    │
  └─────────────────────────────────────────────┘

Cada panel crece con las iteraciones: las pasadas quedan como rastro gris
desvanecido y el estado actual se resalta en color.

Entrada : data/output/{metodo}/data/{domain}/{model}/optimization_results.npz
Salida  : data/output/{metodo}/figures/{domain}/{model}/opt_animation.gif
          data/output/{metodo}/figures/{domain}/{model}/opt_animation.mp4

Uso:
    conda activate bfr_env
    python -m scripts.opt_animation               # bfr, real, adr (defaults)
    python -m scripts.opt_animation fenicsx       # backend fenicsx
    python -m scripts.opt_animation bfr real mrmt # modelo MRMT
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ─── Argumentos ──────────────────────────────────────────────────────────────
metodo = sys.argv[1] if len(sys.argv) > 1 else "bfr"
domain = sys.argv[2] if len(sys.argv) > 2 else "real"
model  = sys.argv[3] if len(sys.argv) > 3 else "adr"

data_path = f"./data/output/{metodo}/data/{domain}/{model}/optimization_results.npz"
fig_dir   = f"./data/output/{metodo}/figures/{domain}/{model}"
os.makedirs(fig_dir, exist_ok=True)

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"No encontrado: {data_path}\n"
        f"Corre primero: python main.py  (con run_type='optimization')"
    )

# ─── Parámetros de tiempo ─────────────────────────────────────────────────
from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters()
p = RUNTIME.get()
dt_days = float(p.dt) / 86400.0

# ─── Datos ───────────────────────────────────────────────────────────────────
raw    = np.load(data_path, allow_pickle=True)
Q_all  = raw["Q"]       # (n_iter, Nt)
zp_all = raw["zp"]      # (n_iter,)
J_all  = raw["J"]       # (n_iter,)

n_iter, Nt = Q_all.shape
iters      = np.arange(1, n_iter + 1)
t_days     = np.arange(Nt) * dt_days
Qmean_all  = Q_all.mean(axis=1)

print(f"[opt_animation] {n_iter} iteraciones, Nt={Nt}, backend={metodo}")
print(f"  J:  {J_all[0]:.4g} → {J_all[-1]:.4g}")
print(f"  zp: {zp_all[0]:.1f} → {zp_all[-1]:.1f} m")
print(f"  Q̄:  {Qmean_all[0]:.3e} → {Qmean_all[-1]:.3e} m³/s")

# ─── Estilo ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.25,
    "grid.linestyle"   : "--",
    "figure.facecolor" : "#fafafa",
    "axes.facecolor"   : "#fafafa",
})

C_TRAIL   = "#bbbbbb"   # color del rastro pasado
C_CURRENT = "#e63946"   # color del punto actual (rojo)
C_LINE    = "#457b9d"   # color de la línea de trail (azul apagado)
CMAP_Q    = plt.cm.plasma   # colormap para los perfiles Q(t)

# ─── Límites ─────────────────────────────────────────────────────────────────
def _lims(arr, pad=0.10):
    lo, hi = arr.min(), arr.max()
    rng = max(hi - lo, abs(hi) * 0.005, abs(lo) * 0.005, 1e-30)
    return lo - pad * rng, hi + pad * rng

J_lim   = _lims(J_all)
zp_lim  = _lims(zp_all)
Qm_lim  = _lims(Qmean_all)
Qt_lim  = _lims(Q_all)
x_lim   = (0.5, n_iter + 0.5)

# ─── Figura y subplots ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8.5))
fig.patch.set_facecolor("#fafafa")

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    height_ratios=[1, 1.4],
    hspace=0.42, wspace=0.32,
    top=0.91, bottom=0.09, left=0.07, right=0.97,
)
ax_J   = fig.add_subplot(gs[0, 0])
ax_zp  = fig.add_subplot(gs[0, 1])
ax_Qmn = fig.add_subplot(gs[0, 2])
ax_Qt  = fig.add_subplot(gs[1, :])

# Configuración estática de ejes
def _setup(ax, title, xlabel, ylabel, xlim, ylim):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8)

_setup(ax_J,   "Convergencia de $J$",      "Iteración", "$J$",
       x_lim, J_lim)
_setup(ax_zp,  "Profundidad del pozo",      "Iteración", "$z_p$ (m)",
       x_lim, zp_lim)
_setup(ax_Qmn, "Caudal medio $\\bar{Q}$",  "Iteración", "$\\bar{Q}$ (m³/s)",
       x_lim, Qm_lim)
_setup(ax_Qt,  "Perfil de extracción $Q(t)$",
       "Tiempo (días)", "$Q(t)$ (m³/s)",
       (t_days[0], t_days[-1]), Qt_lim)

# Línea de referencia: J final
ax_J.axhline(J_all[-1], color=C_CURRENT, lw=0.7, ls="--", alpha=0.5)

# Título global
fig.suptitle(
    f"Optimización adjunta · {metodo.upper()} · {domain}/{model}",
    fontsize=13, fontweight="bold", y=0.97,
)

# ─── Artistas dinámicos ───────────────────────────────────────────────────────

# Trail (línea gris) + punto actual para los 3 paneles superiores
line_J,   = ax_J.plot([],   [], color=C_LINE,  lw=1.5, alpha=0.6, zorder=2)
line_zp,  = ax_zp.plot([],  [], color=C_LINE,  lw=1.5, alpha=0.6, zorder=2)
line_Qmn, = ax_Qmn.plot([], [], color=C_LINE,  lw=1.5, alpha=0.6, zorder=2)

dot_J   = ax_J.scatter([],   [], s=80, color=C_CURRENT, zorder=5, edgecolors="white", linewidths=1)
dot_zp  = ax_zp.scatter([],  [], s=80, color=C_CURRENT, zorder=5, edgecolors="white", linewidths=1)
dot_Qmn = ax_Qmn.scatter([], [], s=80, color=C_CURRENT, zorder=5, edgecolors="white", linewidths=1)

# Anotación del valor actual (se actualiza cada frame)
ann_J   = ax_J.annotate("",   xy=(0, 0), fontsize=8, color=C_CURRENT,
                          xytext=(6, 4), textcoords="offset points")
ann_zp  = ax_zp.annotate("",  xy=(0, 0), fontsize=8, color=C_CURRENT,
                          xytext=(6, 4), textcoords="offset points")
ann_Qmn = ax_Qmn.annotate("", xy=(0, 0), fontsize=8, color=C_CURRENT,
                          xytext=(6, 4), textcoords="offset points")

# Lista de líneas Q(t) (se crean/destruyen en update)
qt_lines = []

# Contador de iteración (texto en ax_Qt)
iter_text = ax_Qt.text(
    0.01, 0.96, "", transform=ax_Qt.transAxes,
    fontsize=10, va="top", ha="left",
    color="#333333", fontweight="bold",
)

# ─── Animación ───────────────────────────────────────────────────────────────

MAX_TRAIL = 6   # máximo de perfiles Q(t) visibles con huella de color

def init():
    line_J.set_data([], [])
    line_zp.set_data([], [])
    line_Qmn.set_data([], [])
    dot_J.set_offsets(np.empty((0, 2)))
    dot_zp.set_offsets(np.empty((0, 2)))
    dot_Qmn.set_offsets(np.empty((0, 2)))
    ann_J.set_text("")
    ann_zp.set_text("")
    ann_Qmn.set_text("")
    iter_text.set_text("")
    return [line_J, line_zp, line_Qmn, dot_J, dot_zp, dot_Qmn,
            ann_J, ann_zp, ann_Qmn, iter_text]


def _gradient_trail(ax, x_all, y_all, k):
    """Dibuja el trail con degradado de color: viejo=gris, nuevo=azul."""
    if k < 1:
        return []
    x  = np.asarray(x_all[:k+1], float)
    y  = np.asarray(y_all[:k+1], float)
    pts = np.c_[x, y].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # color: 0=gris claro, n=azul oscuro
    colors = plt.cm.Blues(np.linspace(0.25, 0.85, len(segs)))
    lc = LineCollection(segs, colors=colors, linewidths=2, zorder=2)
    ax.add_collection(lc)
    return [lc]


# Guardamos las LineCollection de trail para eliminarlas en el siguiente frame
_trail_lcs = {"J": None, "zp": None, "Qmn": None}


def update(k):
    # ─── Paneles superiores ───────────────────────────────────────────────

    # Punto actual
    for dot, arr in [(dot_J, J_all), (dot_zp, zp_all), (dot_Qmn, Qmean_all)]:
        dot.set_offsets([[iters[k], arr[k]]])

    # Anotaciones
    ann_J.set_text(f"{J_all[k]:.4g}")
    ann_J.xy = (iters[k], J_all[k])

    ann_zp.set_text(f"{zp_all[k]:.1f} m")
    ann_zp.xy = (iters[k], zp_all[k])

    ann_Qmn.set_text(f"{Qmean_all[k]:.2e}")
    ann_Qmn.xy = (iters[k], Qmean_all[k])

    # Eliminar trails anteriores
    for key, lc in _trail_lcs.items():
        if lc is not None:
            lc.remove()

    # Crear trails con degradado
    lc_J  = _gradient_trail(ax_J,   iters, J_all,      k)
    lc_zp = _gradient_trail(ax_zp,  iters, zp_all,     k)
    lc_Qm = _gradient_trail(ax_Qmn, iters, Qmean_all,  k)

    _trail_lcs["J"]   = lc_J[0]  if lc_J  else None
    _trail_lcs["zp"]  = lc_zp[0] if lc_zp else None
    _trail_lcs["Qmn"] = lc_Qm[0] if lc_Qm else None

    # Puntos de historia (pequeños, grises)
    if k > 0:
        line_J.set_data(iters[:k+1],   J_all[:k+1])
        line_zp.set_data(iters[:k+1],  zp_all[:k+1])
        line_Qmn.set_data(iters[:k+1], Qmean_all[:k+1])
    else:
        line_J.set_data([iters[0]],   [J_all[0]])
        line_zp.set_data([iters[0]],  [zp_all[0]])
        line_Qmn.set_data([iters[0]], [Qmean_all[0]])

    # ─── Panel Q(t): huella ───────────────────────────────────────────────
    for ln in qt_lines:
        try:
            ln.remove()
        except Exception:
            pass
    qt_lines.clear()

    norm_iter = Normalize(vmin=0, vmax=max(n_iter - 1, 1))

    # Cuántos perfiles pasados mostrar con huella visible
    i_start = max(0, k - MAX_TRAIL)

    for i in range(i_start, k + 1):
        age   = k - i          # 0 = current
        frac  = 1.0 - age / max(MAX_TRAIL, 1)
        alpha = max(0.08, frac ** 1.5)
        lw    = 0.7 + 2.3 * frac
        color = CMAP_Q(norm_iter(i)) if age == 0 else C_TRAIL

        ln, = ax_Qt.plot(t_days, Q_all[i], color=color, lw=lw,
                         alpha=alpha, zorder=2 + frac)
        qt_lines.append(ln)

    # Si hay iteraciones muy viejas: mostrarlas todas muy tenues en gris
    if i_start > 0:
        for i in range(i_start):
            ln, = ax_Qt.plot(t_days, Q_all[i], color=C_TRAIL,
                             lw=0.4, alpha=0.06, zorder=1)
            qt_lines.append(ln)

    # Texto de estado
    dJ = J_all[k] - J_all[0]
    dJ_str = f"ΔJ = {dJ:+.3g}"
    iter_text.set_text(
        f"iter {k+1}/{n_iter}  ·  z_p = {zp_all[k]:.1f} m  ·  "
        f"J = {J_all[k]:.4g}  ·  {dJ_str}"
    )

    return ([line_J, line_zp, line_Qmn, dot_J, dot_zp, dot_Qmn,
             ann_J, ann_zp, ann_Qmn, iter_text]
            + qt_lines
            + [lc for lc in _trail_lcs.values() if lc is not None])


# Frames: repetir el primer y último frame para dar pausas de entrada/salida
HOLD_START = 2
HOLD_END   = 4
frames = (
    [0] * HOLD_START
    + list(range(n_iter))
    + [n_iter - 1] * HOLD_END
)

ani = FuncAnimation(
    fig, update, frames=frames,
    init_func=init, interval=500, blit=False,
)

# ─── Guardar ─────────────────────────────────────────────────────────────────
gif_path = os.path.join(fig_dir, "opt_animation.gif")
mp4_path = os.path.join(fig_dir, "opt_animation.mp4")

print(f"[opt_animation] Guardando GIF ({len(frames)} frames) → {gif_path}", flush=True)
ani.save(gif_path, writer=PillowWriter(fps=2), dpi=120)
print(f"  ✓ GIF guardado")

try:
    print(f"[opt_animation] Guardando MP4 → {mp4_path}", flush=True)
    ani.save(mp4_path, writer=FFMpegWriter(fps=2, bitrate=2000), dpi=120)
    print(f"  ✓ MP4 guardado")
except Exception as e:
    print(f"  [aviso] MP4 no disponible: {e}")

plt.close(fig)
print("[opt_animation] Listo.")
