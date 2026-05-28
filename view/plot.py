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

def update_supertitles(figs, t):
    """
    figs: list
        [fig_H, fig_V, fig_C, figs_C_im]

    t: float
        tiempo en segundos

    mrmt: bool
        activa títulos C_im
    """

    t_hours = t / 3600.0

    titles = []

    # ---- H ----
    title_H = figs[0].suptitle(
        f"Spatial Distribution of Hydraulic Head H (m) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )
    titles.append(title_H)

    # ---- V ----
    title_V = figs[1].suptitle(
        f"Velocity Magnitude Field |U| (m/s) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )
    titles.append(title_V)

    # ---- C ----
    title_C = figs[2].suptitle(
        f"Solute Concentration Field C (kg/m³) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )
    titles.append(title_C)

    # ---- MRMT ----
    titles_C_im = []

    if len(figs) > 3 and figs[3] is not None:
        for r, fig in enumerate(figs[3]):
            title = fig.suptitle(
                f"Immobile Concentration C_im(r={r}) "
                f"(kg/m³) at t = {t_hours:.2f} h",
                fontsize=20,
                color="#DCCB7F",
                y=0.95
            )
            titles_C_im.append(title)

    titles.append(titles_C_im)

    return titles

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

# =========================================================
# FIGURAS ADJUNTO
# =========================================================
def create_adj_figures():

    p = RUNTIME.get()

    # -----------------------------------------------------
    # ψ_H
    # -----------------------------------------------------
    fig_psiH, axes_psiH = plt.subplots(
        p.nrows, p.ncols,
        figsize=(18, 7)
    )

    # -----------------------------------------------------
    # ψ_C
    # -----------------------------------------------------
    fig_psiC, axes_psiC = plt.subplots(
        p.nrows, p.ncols,
        figsize=(18, 7)
    )

    # -----------------------------------------------------
    # MRMT adjunto (si aplica)
    # -----------------------------------------------------
    fig_psiC_im = []
    axes_psiC_im = []

    if "mrmt" in p.model:
        for r in range(p.Nr):
            fig, axes = plt.subplots(
                p.nrows, p.ncols,
                figsize=(18, 7)
            )
            fig_psiC_im.append(fig)
            axes_psiC_im.append(np.atleast_2d(axes))

    # =====================================================
    # COLORMAPS (ADJUNTO)
    # =====================================================

    from matplotlib.colors import LinearSegmentedColormap

    # -----------------------------------------------------
    # colormap principal (psi)
    # -----------------------------------------------------
    base = plt.cm.get_cmap("viridis")
    colors = base(np.linspace(0, 1.0, 256))

    cmap_psi = LinearSegmentedColormap.from_list(
        "psi_cmap",
        colors
    )

    # -----------------------------------------------------
    # colormap alterno (sensibilidades / flujo adjunto)
    # -----------------------------------------------------
    base = plt.cm.get_cmap("coolwarm")
    colors = base(np.linspace(0, 1.0, 256))

    cmap_flux = LinearSegmentedColormap.from_list(
        "flux_cmap",
        colors
    )

    return (
        [fig_psiH, fig_psiC, fig_psiC_im],
        [np.atleast_2d(axes_psiH), np.atleast_2d(axes_psiC), axes_psiC_im],
        [cmap_psi, cmap_flux]
    )

def create_figures():

    fig_H, axes_H = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))
    fig_V, axes_V = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))
    fig_C, axes_C = plt.subplots(p.nrows, p.ncols, figsize=(18, 7))


    fig_C_im = []
    axes_C_im = []
    if "mrmt" in p.model:
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

    if "mrmt" in p.model:
        im_C_im = [
            [[None for _ in range(p.ncols)] for _ in range(p.nrows)]
            for _ in range(p.Nr)
        ]
    else:
        im_C_im = []

    quiv_V = [[None for _ in range(p.ncols)] for _ in range(p.nrows)]

    return im_H, im_V, im_C, im_C_im, quiv_V

def create_adjoint_visual_arrays():

    # =====================================================
    # ψ_h
    # =====================================================
    im_psiH = [
        [None for _ in range(p.ncols)]
        for _ in range(p.nrows)
    ]

    # =====================================================
    # ψ_C
    # =====================================================
    im_psiC = [
        [None for _ in range(p.ncols)]
        for _ in range(p.nrows)
    ]

    return im_psiH, im_psiC

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
                    f"Caudal de extracción Q = {Qout[k][0]:.3e} m³/s",
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
                    f"Caudal de extracción Q = {Qout[k][0]:.3e} m³/s",
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
                    f"Caudal de extracción Q = {Qout[k][0]:.3e} m³/s",
                    fontsize=12
                )

        # MRMT
        if "mrmt" in p.model:
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
                    f"Caudal de extracción Q = {Qout[k][0]:.3e} m³/s",
                    fontsize=12
                )

        
        ims = [
            im_H[0][0],
            im_V[0][0],
            im_C[0][0]
            ]
        if "mrmt" in p.model:
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

# =========================================================
# INITIALIZE ADJOINT PLOTS
# =========================================================
def initialize_adj_plots(d, axes, cmaps, figs):

    im_psiH, im_psiC  = create_adjoint_visual_arrays()

    p = RUNTIME.get()

    X = d.xy_grid[0].T
    Y = d.xy_grid[1].T

    for k in range(p.K):
        i, j = divmod(k, p.ncols)

        # =================================================
        # ψ_H
        # =================================================
        psiH0 = d.I.dot(d.psi_H[k])
        psiH0[~d.mask] = np.nan
        psiH0 = psiH0.reshape(d.xy_grid.shape[1:]).T

        im_psiH[i][j] = axes[0][i, j].imshow(
            psiH0,
            vmin=-0.0006, vmax=0.0006,   # adjuntos suelen ser pequeños y con signo
            cmap=cmaps[0],
            origin="lower",
            extent=[p.xmin, p.xmax, p.ymin, p.ymax]
        )

        axes[0][i, j].set_title(r"$\psi_h$")

        # =================================================
        # ψ_C
        # =================================================
        psiC0 = d.I.dot(d.psi_C[k])
        psiC0[~d.mask] = np.nan
        psiC0 = psiC0.reshape(d.xy_grid.shape[1:]).T

        im_psiC[i][j] = axes[1][i, j].imshow(
            psiC0,
            vmin=-0.0006, vmax=0.0006,
            cmap=cmaps[0],
            origin="lower",
            extent=[p.xmin, p.xmax, p.ymin, p.ymax]
        )

        axes[1][i, j].set_title(r"$\psi_C$")

        # =================================================
        # MRMT adjunto (si aplica)
        # =================================================
        # if "mrmt" in p.model:
        #     for r in range(p.Nr):

        #         psiCim = d.I.dot(d.psiC_im[k][r])
        #         psiCim[~d.mask] = np.nan
        #         psiCim = psiCim.reshape(d.xy_grid.shape[1:]).T

        #         im_psiC_im[r][i][j] = axes[2][r][i, j].imshow(
        #             psiCim,
        #             vmin=-1e-6, vmax=1e-6,
        #             cmap=cmaps[0],
        #             origin="lower",
        #             extent=[p.xmin, p.xmax, p.ymin, p.ymax]
        #         )

        #     axes[2][r][i][j].set_title(r"$\psi_C^{im}$")

        # =================================================
        # flatten images for colorbars
        # =================================================
        ims = [
            im_psiH[0][0],
            im_psiC[0][0],
        ]

        # if "mrmt" in p.model:
        #     for r in range(p.Nr):
        #         ims.append(im_psiC_im[r][0][0])

        # =================================================
        # COLORBARS
        # =================================================
        figs_flat = [
            f for item in figs
            for f in (item if isinstance(item, list) else [item])
        ]

        y_pos = 0.94

        for fig, im in zip(figs_flat, ims):

            cax = fig.add_axes([0.15, y_pos, 0.70, 0.02])

            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

            cbar.ax.tick_params(colors='black')
    # , im_psiC_im, quiv
    return im_psiH, im_psiC




def create_optimization_figures():

    fig = plt.figure(figsize=(14, 10))

    # =====================================================
    # GRID
    # =====================================================

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,

        # alturas
        height_ratios=[0.3, 0.6],

        # anchos
        width_ratios=[0.3, 0.7],

        hspace=0.25,
        wspace=0.20
    )

    # =====================================================
    # AXES
    # =====================================================

    # arriba izquierda -> J
    ax_J = fig.add_subplot(gs[0, 0])

    # arriba derecha -> Q(t)
    ax_Q = fig.add_subplot(gs[0, 1])

    # abajo -> xp/zp
    ax_zp = fig.add_subplot(gs[1, :])

    axes = {
        "J": ax_J,
        "Q": ax_Q,
        "zp": ax_zp
    }

    return fig, axes

def create_optimization_visuals():

    visuals = {
        "Q": None,
        "zp": None,
        "J": None
    }

    return visuals

def initialize_optimization_plots(history, fig, axes):

    visuals = create_optimization_visuals()

    Nt = len(history["Q"][0])

    t = np.arange(Nt)

    # -----------------------------------------------------
    # Q(t)
    # -----------------------------------------------------

    visuals["Q"], = axes[1].plot(
        t,
        history["Q"][0],
        lw=2
    )

    qmin = np.min([np.min(q) for q in history["Q"]])
    qmax = np.max([np.max(q) for q in history["Q"]])

    axes[1].set_xlim(0, Nt-1)
    axes[1].set_ylim(qmin, qmax)

    axes[1].set_title("Control Q(t)")
    axes[1].set_xlabel("Tiempo")
    axes[1].set_ylabel("Q")

    # -----------------------------------------------------
    # zp
    # -----------------------------------------------------

    visuals["zp"], = axes[2].plot([], [], lw=2)

    axes[2].set_xlim(0, len(history["zp"]))
    axes[2].set_ylim(
        np.min(history["zp"]),
        np.max(history["zp"])
    )

    axes[2].set_title("Posición del pozo z_p")
    axes[2].set_xlabel("Iteración")
    axes[2].set_ylabel("z_p")

    # -----------------------------------------------------
    # funcional
    # -----------------------------------------------------

    visuals["J"], = axes[0].plot([], [], lw=2)

    axes[0].set_xlim(0, len(history["J"]))
    axes[0].set_ylim(
        np.min(history["J"]),
        np.max(history["J"])
    )

    axes[0].set_title("Funcional J")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("J")

    plt.tight_layout()

    return visuals

def update_optimization_frame(frame, history, visuals, fig):

    Nt = len(history["Q"][0])

    t = np.arange(Nt)

    # -----------------------------------------------------
    # Q
    # -----------------------------------------------------

    visuals["Q"].set_data(
        t,
        history["Q"][frame]
    )

    # -----------------------------------------------------
    # zp
    # -----------------------------------------------------

    visuals["zp"].set_data(
        np.arange(frame + 1),
        history["zp"][:frame + 1]
    )

    # -----------------------------------------------------
    # J
    # -----------------------------------------------------

    visuals["J"].set_data(
        np.arange(frame + 1),
        history["J"][:frame + 1]
    )

    fig.suptitle(
        f"Iteración de optimización {frame}",
        fontsize=14
    )

    return list(visuals.values())

def animate_optimization(history, save_path="optimization.mp4"):

    fig, axes = create_optimization_figures()

    visuals = initialize_optimization_plots(
        history,
        fig,
        axes
    )

    anim = FuncAnimation(
        fig,
        lambda frame: update_optimization_frame(
            frame,
            history,
            visuals,
            fig
        ),
        frames=len(history["Q"]),
        interval=300,
        blit=False
    )

    anim.save(save_path)

    plt.close(fig)
