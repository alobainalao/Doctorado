import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from contextlib import ExitStack
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from funtions.runtime import RUNTIME
p = RUNTIME.params

def plot_mesh(nodes, groups, simplices, title='Malla', figsize=(10, 3)):

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(nodes[:, 0], nodes[:, 1], s=0.5, color='gray')

    for name, idxs in groups.items():
        ax.scatter(nodes[idxs, 0], nodes[idxs, 1], s=1, label=name)

    ax.set_title(title)
    ax.legend(fontsize='x-small')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def update_supertitles(
        fig_H,
        fig_V,
        fig_C,
        fig_C_im,
        t0
):
    """
    Actualiza títulos para H, V, C y C_im(r)

    Parameters
    ----------
    fig_H, fig_V, fig_C : matplotlib.figure.Figure

    fig_C_im : list
        Lista de figuras MRMT

    t0 : float
        Tiempo en segundos
    """

    t_hours = t0 / 3600.0

    title_H = fig_H.suptitle(
        f"Spatial Distribution of Hydraulic Head H (m) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )

    title_V = fig_V.suptitle(
        f"Velocity Magnitude Field |U| (m/s) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )

    title_C = fig_C.suptitle(
        f"Solute Concentration Field C (kg/m³) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )

    titles_C_im = []

    for r, fig in enumerate(fig_C_im):

        title = fig.suptitle(

            f"Immobile Concentration C_im(r={r}) "
            f"(kg/m³) at t = {t_hours:.2f} h",

            fontsize=20,

            color="#DCCB7F",

            y=0.95

        )

        titles_C_im.append(title)

    return (
        title_H,
        title_V,
        title_C,
        titles_C_im
    )

def get_colormaps():

    base = plt.cm.get_cmap("turbo")
    colors = base(np.linspace(0.08, 1.0, 256))

    cmap_main = LinearSegmentedColormap.from_list(
        "turbo_paraview",
        colors
    )

    base = plt.cm.get_cmap("inferno_r")
    colors_main = base(np.linspace(0, 1.0, 240))
    gray_zero = np.array([[0.92, 0.92, 0.92, 1.0]])

    colors = np.vstack((gray_zero, colors_main))

    cmap_h = LinearSegmentedColormap.from_list(
        "turbo_paraview_h",
        colors,
        256
    )

    return cmap_main, cmap_h

def create_figures():

    fig_H, axes_H = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))
    fig_V, axes_V = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))
    fig_C, axes_C = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))


    fig_C_im = []
    axes_C_im = []

    for r in range(p.Nr):
        fig, axes = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))
        fig_C_im.append(fig)
        axes_C_im.append(np.atleast_2d(axes))

    from matplotlib.colors import LinearSegmentedColormap
    base = plt.cm.get_cmap("turbo")

    # Recorte para eliminar el azul oscuro inicial
    colors = base(np.linspace(0.08, 1.0, 256))

    cmap_main = LinearSegmentedColormap.from_list(
        "turbo_paraview",
        colors
    )

    base = plt.cm.get_cmap("inferno_r")
    colors_main = base(np.linspace(0, 1.0, 240))
    gray_zero = np.array([[0.92, 0.92, 0.92, 1.0]]) 

    # Recorte para eliminar el azul oscuro inicial
    colors = np.vstack((gray_zero, colors_main))


    cmap_h = LinearSegmentedColormap.from_list(
        "turbo_paraview",
        colors,
        256
    )

    return [fig_H, fig_V, fig_C, fig_C_im], [np.atleast_2d(axes_H), np.atleast_2d(axes_V),
                     np.atleast_2d(axes_C), axes_C_im], [cmap_main, cmap_h]

def export_videos(figs, sim, frames, mask_h):

    writer_kwargs = dict(
        fps=20,
        codec="libx264",
        bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p"]
    )

    writers = [FFMpegWriter(**writer_kwargs) for _ in figs]

    context = [
        writers[i].saving(figs[i], f"output_{i}.mp4", dpi=150)
        for i in range(len(figs))
    ]

    with ExitStack() as stack:

        for c in context:
            stack.enter_context(c)

        for n in range(frames):

            sim.update(n, mask_h)

            for w in writers:
                w.grab_frame()

def setup_figures():
    (fig_H, fig_V, fig_C, fig_C_im,
     axes_H, axes_V, axes_C, axes_C_im,
     cmap_main, cmap_h) = create_figures(p.Nr)

    figs = [fig_H, fig_V, fig_C] + fig_C_im

    for fig in figs:
        fig.subplots_adjust(
            left=0.05, right=0.97,
            top=0.83, bottom=0.05,
            wspace=0.12, hspace=0.12
        )

    return (fig_H, fig_V, fig_C, fig_C_im,
            axes_H, axes_V, axes_C, axes_C_im,
            cmap_main, cmap_h)

def create_visual_arrays():
    im_H = [[None for _ in range(p.ncols)] for _ in range(p.nrows)]
    im_V = [[None for _ in range(p.ncols)] for _ in range(p.nrows)]
    im_C = [[None for _ in range(p.ncols)] for _ in range(p.nrows)]

    im_C_im = [
        [[None for _ in range(p.ncols)] for _ in range(p.nrows)]
        for _ in range(p.Nr)
    ]

    quiv_V = [[None for _ in range(p.ncols)] for _ in range(p.nrows)]

    return im_H, im_V, im_C, im_C_im, quiv_V

def initialize_plots(d, Qout, axes, cmaps, figs):

    im_H, im_V, im_C, im_C_im, quiv_V = create_visual_arrays()

    X = d.xy_grid[0].T
    Y = d.xy_grid[1].T

    for k in range(p.K):
        i, j = divmod(k, p.ncols)

        # H
        H0 = d.I.dot(d.H[k])
        H0[~d.mask] = np.nan
        H0 = H0.reshape(d.xy_grid.shape[1:]).T

        im_H[i][j] = axes[0][i,j].imshow(
            H0, vmin=305, vmax=309,
            cmap=cmaps[0], origin="lower",
            extent=[p.xmin, p.xmax, p.ymin, p.ymax]
        )
        axes[0][i,j].set_title(
                    f"Caudal de extracción Q = {Qout[k]:.3e} m³/s",
                    fontsize=12
                )

        # V
        speed0 = np.sqrt(d.U[k][:,0]**2 + d.U[k][:,1]**2)
        S0 = d.I.dot(speed0)
        S0[~d.mask] = np.nan
        S0 = S0.reshape(d.xy_grid.shape[1:]).T

        im_V[i][j] = axes[1][i,j].imshow(
            S0, vmin=4e-12, vmax=4.5e-4,
            cmap=cmaps[0], origin="lower",
            extent=[p.xmin, p.xmax, p.ymin, p.ymax]
        )

        # quiver
        Ux = d.I.dot(d.U[k][:,0] / speed0)
        Uy = d.I.dot(d.U[k][:,1] / speed0)

        Ux[~d.mask] = np.nan
        Uy[~d.mask] = np.nan

        Ux = Ux.reshape(d.xy_grid.shape[1:]).T
        Uy = Uy.reshape(d.xy_grid.shape[1:]).T

        quiv_V[i][j] = axes[1][i,j].quiver(
            X[::p.sp_q, ::p.sp_q],
            Y[::p.sp_q, ::p.sp_q],
            Ux[::p.sp_q, ::p.sp_q],
            Uy[::p.sp_q, ::p.sp_q],
            scale=50, width=0.001, color="k"
        )

        axes[1][i,j].set_title(
                    f"Caudal de extracción Q = {Qout[k]:.3e} m³/s",
                    fontsize=12
                )

        # C
        C0 = d.I.dot(d.C[k])
        C0[~d.mask] = np.nan
        C0 = C0.reshape(d.xy_grid.shape[1:]).T

        im_C[i][j] = axes[2][i,j].imshow(
            C0, vmin=0, vmax=7.6e-11,
            cmap=cmaps[1], origin="lower",
            extent=[p.xmin, p.xmax, p.ymin, p.ymax]
        )
        axes[2][i,j].set_title(
                    f"Caudal de extracción Q = {Qout[k]:.3e} m³/s",
                    fontsize=12
                )

        # MRMT
        for r in range(p.Nr):
            Cim = d.I.dot(d.C_im[k][r])
            Cim[~d.mask] = np.nan
            Cim = Cim.reshape(d.xy_grid.shape[1:]).T

            im_C_im[r][i][j] = axes[3][r][i,j].imshow(
                Cim, vmin=0, vmax=7.6e-11,
                cmap=cmaps[1], origin="lower",
                extent=[p.xmin, p.xmax, p.ymin, p.ymax]
            )

            axes[3][r][i][j].set_title(
                    f"Caudal de extracción Q = {Qout[k]:.3e} m³/s",
                    fontsize=12
                )

        
        ims = [
            im_H[0][0],
            im_V[0][0],
            im_C[0][0]
            ]
        
        for r in range(p.Nr):

            ims.append(
                im_C_im[r][0][0]
            )


        figs_flat = [f for item in figs for f in (item if isinstance(item, list) else [item])]
        y_pos  = 0.94   # misma altura para las 3 colorbars (puedes cambiarlo)

        for fig, im in zip(figs_flat, ims):

            # Crear el eje para la colorbar (manual)
            cax = fig.add_axes([0.15, y_pos, 0.70, 0.02])   # [x, y, width, height]

            # Crear la colorbar
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

            # Números en rojo
            cbar.ax.tick_params(colors='red')

    return im_H, im_V, im_C, im_C_im, quiv_V
