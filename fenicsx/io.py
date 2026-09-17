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


def save_results(save_dir, nodes, H_hist, C_hist, U_hist, dt, C_im_hist=None):
    """
    Guarda la historia temporal en `{save_dir}/simulation_results.npz`, mismo
    formato que el backend bfr (finalize_outputs): H y C con eje de realización
    K=1 (fenicsx corre un solo run), U con sus dos componentes. Todos los
    campos van sobre los nodos de G (Lagrange-1 = vértices de malla); C y U se
    interpolan a G en el forward para compartir el mismo `nodes` que H.
    Para MRMT, C_im_hist es opcional: shape (Nt, Nr, N_dofs_Q) sobre DOFs de Q.
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
    np.savez(path, **arrays)
    print(f"[fenicsx] datos guardados ✔  {path}")


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


def _render_field(save_dir, name, tri, frames, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.asarray(frames)
    vmin, vmax = float(frames.min()), float(frames.max())
    if vmin == vmax:
        vmax = vmin + 1e-12

    writer, ext = _pick_writer(fps=8)
    fig, ax = plt.subplots(figsize=(8, 4))
    tpc = ax.tripcolor(tri, frames[0], shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    fig.colorbar(tpc, ax=ax)
    ax.set_aspect("equal")
    path = f"{save_dir}/{name}.{ext}"
    with writer.saving(fig, path, dpi=120):
        for k, fr in enumerate(frames):
            tpc.set_array(fr)
            ax.set_title(f"{name}   t = {k * dt:.3g} s")
            writer.grab_frame()
    plt.close(fig)
    print(f"[fenicsx] animación guardada ✔  {path}")


def animate_results(save_dir, G, H_hist, C_hist, U_hist, dt):
    """
    Renderiza H, |U| y C sobre la malla a lo largo del tiempo (análogo a los
    H.mp4/V.mp4/C.mp4 de bfr). Robusto: si algo falla (p.ej. sin backend de
    render), avisa y no tumba la corrida.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        tri = _triangulation(G)
        U = np.asarray(U_hist)                       # (Nt, N, 2)
        Umag = np.linalg.norm(U, axis=2)             # (Nt, N)
        _render_field(save_dir, "H", tri, H_hist, dt)
        _render_field(save_dir, "V", tri, Umag, dt)
        _render_field(save_dir, "C", tri, C_hist, dt)
    except Exception as e:  # pragma: no cover - depende del entorno de render
        print(f"[fenicsx] aviso: no se pudo generar la animación ({e}); "
              f"se omite (los datos .npz sí se guardaron si save_dat=True).")
