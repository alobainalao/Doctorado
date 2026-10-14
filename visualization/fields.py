"""
Visualización de campos escalares y vectoriales sobre nubes de puntos.

Funciones principales:
  plot_scalar_field(nodes, values, ax, ...)
  plot_vector_field(nodes, U, ax, ...)
  plot_comparison_2x2(nodes, u_ref, u_approx, ...)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from visualization.style import COLORMAPS


# ---------------------------------------------------------------------------
# Campo escalar
# ---------------------------------------------------------------------------

def plot_scalar_field(nodes, values, ax, cmap=None, vmin=None, vmax=None,
                      colorbar=True, cb_label="", title="", point_size=0.5,
                      **scatter_kw):
    """
    Dibuja un campo escalar sobre una nube de puntos 2D (x,z).

    Parámetros
    ----------
    nodes   : (N, 2)  coordenadas (x, z) en metros
    values  : (N,)    valores del campo
    ax      : Axes de matplotlib
    cmap    : colormap (por defecto "viridis")
    colorbar: añadir barra de color
    cb_label: etiqueta de la barra de color (incluir unidades)
    title   : título del panel

    Retorna
    -------
    scatter : PathCollection (para añadir colorbar externa)
    """
    if cmap is None:
        cmap = COLORMAPS["scalar"]

    sc = ax.scatter(
        nodes[:, 0], nodes[:, 1],
        c=values, cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=point_size, rasterized=True,
        linewidths=0,
        **scatter_kw
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    if title:
        ax.set_title(title)

    if colorbar:
        cb = plt.colorbar(sc, ax=ax, label=cb_label,
                          fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize="small")

    return sc


# ---------------------------------------------------------------------------
# Campo vectorial
# ---------------------------------------------------------------------------

def plot_vector_field(nodes, U, ax, subsample=200, scale=None,
                      color="#333333", title="", bg_scalar=None,
                      bg_cmap=None, bg_vmin=None, bg_vmax=None,
                      bg_label="", point_size=0.3, seed=42):
    """
    Dibuja flechas quiver subsampled sobre una nube de puntos.

    nodes     : (N, 2)
    U         : (N, 2)  campo vectorial
    subsample : número de flechas a mostrar (aleatorio uniforme)
    bg_scalar : (N,) campo escalar de fondo (opcional)
    seed      : semilla para reproducibilidad del subsampling
    """
    N = len(nodes)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(subsample, N), replace=False)

    # Fondo escalar si se da
    sc = None
    if bg_scalar is not None:
        sc = ax.scatter(
            nodes[:, 0], nodes[:, 1],
            c=bg_scalar, cmap=bg_cmap or COLORMAPS["scalar"],
            vmin=bg_vmin, vmax=bg_vmax,
            s=point_size, rasterized=True, linewidths=0,
            zorder=1
        )
        plt.colorbar(sc, ax=ax, label=bg_label, fraction=0.046, pad=0.04)

    # Flechas
    ax.quiver(
        nodes[idx, 0], nodes[idx, 1],
        U[idx, 0], U[idx, 1],
        color=color, scale=scale, angles="xy", scale_units="xy",
        zorder=2, linewidth=0.5
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    if title:
        ax.set_title(title)

    return sc


# ---------------------------------------------------------------------------
# Panel de comparación 2×2
# ---------------------------------------------------------------------------

def plot_comparison_2x2(nodes, u_ref, u_approx, field_label, units,
                        label_ref="FEM", label_approx="BFR",
                        model="", t_days=None, profile="THESIS",
                        point_size=0.4):
    """
    Genera un panel 2×2 de comparación:
      (a) Referencia     (b) Aproximación
      (c) Diferencia     (d) Error relativo

    Parámetros
    ----------
    nodes       : (N, 2)  coordenadas del grid de referencia
    u_ref       : (N,)    campo de referencia
    u_approx    : (N,)    campo a comparar (interpolado al mismo grid)
    field_label : str     nombre del campo ("Concentración C", etc.)
    units       : str     unidades ("[kg/m³]")
    label_ref   : str     etiqueta de referencia
    label_approx: str     etiqueta del método a comparar
    model       : str     nombre del modelo para el título
    t_days      : float   tiempo en días (para el título)
    profile     : str     perfil de estilo

    Retorna
    -------
    fig, axes
    """
    from visualization.style import apply_style

    diff = u_approx - u_ref
    rel  = np.abs(diff) / (np.abs(u_ref) + 1e-30)
    rel  = np.clip(rel, 0, 5)

    vmin = min(u_ref.min(), u_approx.min())
    vmax = max(u_ref.max(), u_approx.max())
    diff_max = np.abs(diff).max()

    title_fig = f"{field_label} {units}"
    if model:
        title_fig = f"{title_fig} — {model.upper()}"
    if t_days is not None:
        title_fig += f"  (t = {t_days:.1f} días)"

    with apply_style(profile):
        from visualization.style import PROFILES
        w, h = PROFILES[profile]["figure.figsize"]
        fig, axes = plt.subplots(2, 2, figsize=(w * 2, h * 2))
        fig.suptitle(title_fig, fontsize=plt.rcParams["axes.titlesize"] + 1)

        panels = [
            (axes[0, 0], u_ref,    f"(a) {label_ref}",   COLORMAPS["scalar"],  vmin, vmax,  units),
            (axes[0, 1], u_approx, f"(b) {label_approx}", COLORMAPS["scalar"],  vmin, vmax,  units),
            (axes[1, 0], diff,     f"(c) Diferencia ({label_approx} − {label_ref})",
             COLORMAPS["divergent"], -diff_max, diff_max, units),
            (axes[1, 1], rel,      "(d) Error relativo |diff|/|ref|",
             COLORMAPS["error"], 0, min(rel.max(), 2.0), ""),
        ]

        for ax, vals, title, cmap, vlo, vhi, cb_lbl in panels:
            norm = None
            if cmap == COLORMAPS["divergent"] and vhi > 0 and vlo < 0:
                # TwoSlopeNorm requiere vmin < vcenter < vmax estricto
                norm = TwoSlopeNorm(vmin=vlo, vcenter=0, vmax=vhi)
            sc = ax.scatter(
                nodes[:, 0], nodes[:, 1],
                c=vals, cmap=cmap,
                norm=norm,
                vmin=vlo if norm is None else None,
                vmax=vhi if norm is None else None,
                s=point_size, rasterized=True, linewidths=0
            )
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]", fontsize="small")
            ax.set_ylabel("z [m]", fontsize="small")
            ax.set_title(title)
            plt.colorbar(sc, ax=ax, label=cb_lbl, fraction=0.046, pad=0.04)

    return fig, axes


# ---------------------------------------------------------------------------
# Perfil temporal (breakthrough curve)
# ---------------------------------------------------------------------------

def plot_timeseries(t_arr, series_dict, ax, xlabel="Tiempo [días]",
                    ylabel="C [kg/m³]", title="", log_y=False):
    """
    Dibuja series temporales múltiples en un eje dado.

    series_dict : {"nombre": (t, valores)} o {"nombre": valores}  (usa t_arr)
    """
    from visualization.style import COLORS, MARKERS

    for i, (name, data) in enumerate(series_dict.items()):
        if isinstance(data, tuple):
            t_plot, y = data
        else:
            t_plot, y = t_arr, data

        color = list(COLORS.values())[i % len(COLORS)]
        marker = list(MARKERS.values())[i % len(MARKERS)]

        ax.plot(t_plot, y, label=name, color=color, marker=marker,
                markevery=max(1, len(t_plot) // 10), linewidth=1.5,
                markersize=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
