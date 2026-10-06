"""
Salida del backend fenicsx: guardado de datos y animación, análogos a lo que
hace el backend bfr en view/animation.py (finalize_outputs / create_writers).

Se mantiene el MISMO formato de `simulation_results.npz` que bfr (H, C, U,
nodes, dt, con eje K) para que las herramientas de postproceso lo lean igual.
La animación usa mp4 si hay ffmpeg (como bfr) o cae a gif (PillowWriter) si no,
para no depender del contenedor.
"""
import os

import numpy as np


def save_results(save_dir, nodes, H_hist, C_hist, U_hist, dt, C_im_hist=None, nodes_Q=None):
    """
    Guarda la historia temporal en `{save_dir}/simulation_results.npz`, mismo
    formato que el backend bfr (finalize_outputs): H y C con eje de realización
    K=1 (fenicsx corre un solo run), U con sus dos componentes. Todos los
    campos van sobre los nodos de G (Lagrange-1 = vértices de malla); C y U se
    interpolan a G en el forward para compartir el mismo `nodes` que H.
    Para MRMT, C_im_hist es opcional: shape (Nt, Nr, N_dofs_Q) sobre DOFs de Q
    (espacio distinto de G) — `nodes_Q` son sus coordenadas propias, necesarias
    para poder graficarlo (no coinciden con `nodes`).
    """
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/simulation_results.npz"
    arrays = dict(
        H=np.asarray(H_hist)[:, None, :],   # (Nt, 1, N) — eje K como bfr
        C=np.asarray(C_hist)[:, None, :],   # (Nt, 1, N)
        U=np.asarray(U_hist),               # (Nt, N, 2)
        nodes=np.asarray(nodes),            # (N, 2)
        dt=dt,
    )
    if C_im_hist is not None:
        arrays["C_im"] = np.asarray(C_im_hist)  # (Nt, Nr, N_dofs_Q)
        if nodes_Q is not None:
            arrays["nodes_Q"] = np.asarray(nodes_Q)  # (N_dofs_Q, 2)
    np.savez(path, **arrays)
    print(f"[fenicsx] datos guardados ✔  {path}")


def save_adjoint_results(save_dir, nodes_G, nodes_Q, psiH_hist, psiC_hist, dt):
    """Guarda el historial adjunto (psiH en DOFs de G, psiC en DOFs de Q) en
    `{save_dir}/adjoint_results.npz`, análogo a finalize_outputs_adj de bfr."""
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/adjoint_results.npz"
    np.savez(
        path,
        psi_H=np.asarray(psiH_hist),   # (Nt, N_G)
        psi_C=np.asarray(psiC_hist),   # (Nt, N_Q)
        nodes_G=np.asarray(nodes_G),   # (N_G, 2)  DOFs de G (Lagrange-1)
        nodes_Q=np.asarray(nodes_Q),   # (N_Q, 2)  DOFs de Q (Lagrange-2)
        dt=dt,
    )
    print(f"[fenicsx] adjunto guardado ✔  {path}")


# Mismos rangos fijos y textos que view/plot.py:initialize_plots /
# update_supertitles usan para H/V/C en bfr — para que las animaciones de
# ambos backends sean directamente comparables (no auto-escaladas).
_FIXED_RANGES = {
    "H": (305.0, 309.0),
    "V": (4e-12, 4.5e-4),
    "C": (0.0, 7.6e-11),
    # Estados adjuntos (view/plot.py:initialize_adj_plots): pequeños, con signo.
    "PSI": (-0.0006, 0.0006),
}
_SUPTITLES = {
    "H": "Spatial Distribution of Hydraulic Head H (m)",
    "V": "Velocity Magnitude Field |U| (m/s)",
    "C": "Solute Concentration Field C (kg/m³)",
}


def _pick_writer(fps):
    """mp4 (FFMpegWriter, como bfr) si ffmpeg está disponible; si no, gif."""
    from matplotlib.animation import FFMpegWriter, PillowWriter
    try:
        if FFMpegWriter.isAvailable():
            return FFMpegWriter(fps=fps), "mp4"
    except Exception:
        pass
    return PillowWriter(fps=fps), "gif"


def _triangulation(G):
    """Triangulación de matplotlib alineada con el orden de DOF de G (P1), vía
    dolfinx.plot.vtk_mesh — sus puntos siguen el mismo orden que `f.x.array`."""
    import dolfinx.plot
    import matplotlib.tri as mtri
    topo, _cell_types, geom = dolfinx.plot.vtk_mesh(G)
    tris = topo.reshape(-1, 4)[:, 1:]   # [3, i, j, k, 3, ...] -> (Ncell, 3)
    return mtri.Triangulation(geom[:, 0], geom[:, 1], tris)


def _triangulation_from_points(coords):
    """Triangulación por Delaunay (matplotlib.tri la calcula sola si no se
    pasan `triangles`) a partir de coordenadas sueltas. C_im vive en los DOFs
    de Q (Lagrange-2): vtk_mesh(Q) no da una malla P1 directa, así que en vez
    de reconstruir la topología de Q, se retriangula por posición — igual de
    válido para tripcolor (no depende de qué elementos usó el solver)."""
    import matplotlib.tri as mtri
    coords = np.asarray(coords)
    return mtri.Triangulation(coords[:, 0], coords[:, 1])


# Mismo tamaño de figura que bfr (view/plot.py:create_figures, K=1 -> un solo
# panel ocupa el lienzo completo) — con (8,4) los títulos a fontsize=20 se
# recortaban contra el borde de la figura.
_FIGSIZE = (18, 7)

# Malla de quiver para V: densidad comparable a bfr (grid 400x80 con
# stride sp_q=5 -> ~80x16 flechas); aquí se interpola desde la triangulación
# FEM a esta malla regular más gruesa (bfr también interpola sobre grid).
_QUIVER_NX, _QUIVER_NY = 40, 12


def _quiver_grid(tri):
    import numpy as np
    xs = np.linspace(tri.x.min(), tri.x.max(), _QUIVER_NX)
    ys = np.linspace(tri.y.min(), tri.y.max(), _QUIVER_NY)
    X, Y = np.meshgrid(xs, ys)
    finder = tri.get_trifinder()
    inside = finder(X.ravel(), Y.ravel()) >= 0
    return X, Y, inside.reshape(X.shape)


def _render_field(save_dir, name, tri, frames, dt, Q0=None, suptitle=None, out_name=None,
                   range_key=None, cmap_key=None):
    import matplotlib
    matplotlib.use("Agg")
    from view.plot import get_colormaps  # misma paleta que bfr, una sola fuente

    range_key = range_key or name
    cmap_key = cmap_key or name
    out_name = out_name or name
    suptitle = suptitle or _SUPTITLES.get(name, name)

    frames = np.asarray(frames)
    vmin, vmax = _FIXED_RANGES.get(range_key, (float(frames.min()), float(frames.max())))
    if vmin == vmax:
        vmax = vmin + 1e-12

    cmap_main, cmap_h = get_colormaps()
    cmap = cmap_h if cmap_key == "C" else cmap_main

    import matplotlib.pyplot as plt
    # Mismo fps que bfr (view/animation.py: writer_kwargs de H/V/C) — misma
    # cantidad de frames (uno por paso, Nt) al mismo ritmo de reproducción.
    writer, ext = _pick_writer(fps=20)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    tpc = ax.tripcolor(tri, frames[0], shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_aspect("equal")
    fig.colorbar(tpc, cax=ax.inset_axes([0.62, 0.10, 0.33, 0.06]),
                 orientation="horizontal").ax.tick_params(labelsize=8)
    if Q0 is not None:
        # Mismo subtítulo que initialize_plots (bfr) en cada panel.
        ax.set_title(f"Caudal de extracción Q = {Q0:.3e} m³/s", fontsize=12)
    path = f"{save_dir}/{out_name}.{ext}"
    with writer.saving(fig, path, dpi=120):
        for k, fr in enumerate(frames):
            tpc.set_array(fr)
            # Mismo formato de supertítulo que update_supertitles (bfr).
            t_hours = k * dt / 3600.0
            fig.suptitle(f"{suptitle} at t = {t_hours:.2f} h",
                         fontsize=20, color="#0000FF", y=0.99)
            writer.grab_frame()
    plt.close(fig)
    print(f"[fenicsx] animación guardada ✔  {path}")


def _render_velocity(save_dir, tri, U_hist, dt, Q0=None):
    """Igual que _render_field para |U|, pero agrega quiver de dirección
    normalizada (Ux/|U|, Uy/|U|) sobre una malla regular interpolada desde la
    triangulación FEM — mismo tratamiento que initialize_plots/Simulation.update
    en bfr (quiver de dirección, no de magnitud, actualizado cada frame)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import LinearTriInterpolator
    from view.plot import get_colormaps

    U = np.asarray(U_hist)                      # (Nt, N, 2)
    Umag = np.linalg.norm(U, axis=2)             # (Nt, N)
    vmin, vmax = _FIXED_RANGES["V"]

    cmap_main, _ = get_colormaps()

    Xg, Yg, inside = _quiver_grid(tri)

    def direction_at(frame_idx):
        ux = np.ma.filled(LinearTriInterpolator(tri, U[frame_idx, :, 0])(Xg, Yg), np.nan)
        uy = np.ma.filled(LinearTriInterpolator(tri, U[frame_idx, :, 1])(Xg, Yg), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            speed = np.hypot(ux, uy)
            speed = np.where(speed > 0, speed, np.nan)
            ux, uy = ux / speed, uy / speed
        ux = np.where(inside, ux, np.nan)
        uy = np.where(inside, uy, np.nan)
        return ux, uy

    writer, ext = _pick_writer(fps=20)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    tpc = ax.tripcolor(tri, Umag[0], shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap_main)
    ax.set_aspect("equal")
    fig.colorbar(tpc, cax=ax.inset_axes([0.62, 0.10, 0.33, 0.06]),
                 orientation="horizontal").ax.tick_params(labelsize=8)
    ux0, uy0 = direction_at(0)
    quiv = ax.quiver(Xg, Yg, ux0, uy0, scale=50, width=0.001, color="k")
    if Q0 is not None:
        ax.set_title(f"Caudal de extracción Q = {Q0:.3e} m³/s", fontsize=12)
    path = f"{save_dir}/V.{ext}"
    with writer.saving(fig, path, dpi=120):
        for k in range(len(Umag)):
            tpc.set_array(Umag[k])
            ux, uy = direction_at(k)
            quiv.set_UVC(ux, uy)
            t_hours = k * dt / 3600.0
            fig.suptitle(f"{_SUPTITLES['V']} at t = {t_hours:.2f} h",
                         fontsize=20, color="#0000FF", y=0.99)
            writer.grab_frame()
    plt.close(fig)
    print(f"[fenicsx] animación guardada ✔  {path}")


def animate_results(save_dir, G, H_hist, C_hist, U_hist, dt, Q0=None,
                     C_im_hist=None, nodes_Q=None):
    """
    Renderiza H, |U| (con quiver de dirección) y C sobre la malla a lo largo
    del tiempo (análogo a los H.mp4/V.mp4/C.mp4 de bfr). Si el modelo es MRMT
    (C_im_hist/nodes_Q dados), también renderiza C_im_r{r}.mp4 por región —
    análogo a los C_im_r{r}.mp4 de bfr (mismo nombre de archivo, misma paleta
    y rango que C móvil). Robusto: si algo falla (p.ej. sin backend de
    render), avisa y no tumba la corrida.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        tri = _triangulation(G)
        _render_field(save_dir, "H", tri, H_hist, dt, Q0=Q0)
        _render_velocity(save_dir, tri, U_hist, dt, Q0=Q0)
        _render_field(save_dir, "C", tri, C_hist, dt, Q0=Q0)

        if C_im_hist is not None and nodes_Q is not None:
            C_im = np.asarray(C_im_hist)  # (Nt, Nr, N_dofs_Q)
            tri_Q = _triangulation_from_points(nodes_Q)
            for r in range(C_im.shape[1]):
                _render_field(
                    save_dir, "C", tri_Q, C_im[:, r, :], dt, Q0=Q0,
                    suptitle=f"Immobile Concentration C_im(r={r}) (kg/m³)",
                    out_name=f"C_im_r{r}",
                )
    except Exception as e:  # pragma: no cover - depende del entorno de render
        print(f"[fenicsx] aviso: no se pudo generar la animación ({e}); "
              f"se omite (los datos .npz sí se guardaron si save_dat=True).")


def animate_forward_fields(save_dir, G, nodes_Q, h_hist, C_hist, dt, Q0=None):
    """Renderiza H.mp4 (DOFs de G) y C.mp4 (DOFs de Q, retriangulado por
    posición — C vive en Lagrange-2, no en G) — sin V: el forward de
    optimización no lleva historial de velocidad, ver
    fenicsx/adjoint.py:_solve_forward. Usado para animar cada iteración del
    optimizador, análogo a lo que bfr hace en optimize_bfr
    (gradient(..., animate=True)). Robusto ante fallos de render."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        tri_G = _triangulation(G)
        tri_Q = _triangulation_from_points(nodes_Q)
        _render_field(save_dir, "H", tri_G, h_hist, dt, Q0=Q0)
        _render_field(save_dir, "C", tri_Q, C_hist, dt, Q0=Q0)
    except Exception as e:  # pragma: no cover
        print(f"[fenicsx] aviso: no se pudo generar la animación forward ({e}); se omite.")


def animate_adjoint_results(save_dir, G, nodes_Q, psiH_hist, psiC_hist, dt):
    """Renderiza psi_H.mp4 (DOFs de G) y psi_C.mp4 (DOFs de Q, retriangulado
    por posición) — análogo a los psi_H.mp4/psi_C.mp4 de bfr (mismo rango fijo
    y paleta que initialize_adj_plots). Robusto ante fallos de render."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        tri_G = _triangulation(G)
        tri_Q = _triangulation_from_points(nodes_Q)
        _render_field(save_dir, "psiH", tri_G, psiH_hist, dt,
                      suptitle="Adjoint State ψ_h", out_name="psi_H",
                      range_key="PSI", cmap_key="H")
        _render_field(save_dir, "psiC", tri_Q, psiC_hist, dt,
                      suptitle="Adjoint State ψ_C", out_name="psi_C",
                      range_key="PSI", cmap_key="H")
    except Exception as e:  # pragma: no cover
        print(f"[fenicsx] aviso: no se pudo generar la animación adjunta ({e}); se omite.")
