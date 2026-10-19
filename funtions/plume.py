"""
Streamline de la pluma del contaminante.

Integra la trayectoria real del contaminante desde fnte hasta la frontera
izquierda (outlet) usando el campo de velocidades Darcy U del pkl de
preproceso BFR con integración RK4.

La streamline cubre fnte → pozo → outlet (frontera izquierda), de modo que
la función de espaciado pueda refinar toda la zona de contaminación.
El resultado se cachea en data/input/plume_streamline.npy.
"""
import os
import numpy as np

_CACHE_PATH = "data/input/plume_streamline.npy"


def _rk4_step(pt, ds, interp_u, interp_v):
    def unit_vel(p):
        ux = float(interp_u([p])[0])
        uz = float(interp_v([p])[0])
        spd = np.hypot(ux, uz)
        if spd < 1e-14:
            return None
        return np.array([ux / spd, uz / spd])

    k1 = unit_vel(pt)
    if k1 is None:
        return None
    k2 = unit_vel(pt + 0.5 * ds * k1)
    if k2 is None:
        k2 = k1
    k3 = unit_vel(pt + 0.5 * ds * k2)
    if k3 is None:
        k3 = k2
    k4 = unit_vel(pt + ds * k3)
    if k4 is None:
        k4 = k3
    return pt + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def compute_plume_streamline(pkl_path, ds=30.0, n_steps=5000):
    """
    Integra la streamline desde fnte hasta la frontera izquierda (outlet)
    con RK4 sobre el campo de velocidades Darcy U[0] del pkl.

    La trayectoria pasa a través de la vecindad del pozo y continúa hasta
    que sale por la frontera izquierda del dominio.

    Devuelve array (M, 2) o None si el pkl no existe.
    """
    from scipy.interpolate import LinearNDInterpolator
    from funtions.runtime import RUNTIME
    import pickle

    if not os.path.exists(pkl_path):
        return None

    p = RUNTIME.get()
    fnte = np.asarray(p.fnte, float)
    vert = np.asarray(p.vert, float)
    x_outlet = vert[:, 0].min()   # frontera izquierda del dominio

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    nodes = data["nodes"]
    U0 = data["U"][0]

    interp_u = LinearNDInterpolator(nodes, U0[:, 0], fill_value=0.0)
    interp_v = LinearNDInterpolator(nodes, U0[:, 1], fill_value=0.0)

    pts = [fnte.copy()]
    pt = fnte.copy()

    for _ in range(n_steps):
        new_pt = _rk4_step(pt, ds, interp_u, interp_v)
        if new_pt is None:
            break
        pts.append(new_pt.copy())
        pt = new_pt
        # Parar al alcanzar la frontera izquierda
        if pt[0] <= x_outlet + ds:
            break

    streamline = np.array(pts)

    os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
    np.save(_CACHE_PATH, streamline)
    print(f"[plume] streamline calculada: {len(streamline)} puntos "
          f"({streamline[0]} → {streamline[-1]}) → {_CACHE_PATH}")
    return streamline


def plume_cone_dist(pts, streamline, w0=80.0, w1=600.0, depth_below=150.0):
    """
    Distancia efectiva de cada punto a la pluma con ancho de cono variable.

    El cono se ensancha linealmente desde w0 (en fnte, s=0) hasta w1
    (en el outlet, s=L). Para un punto dentro del cono (d < w_local), la
    distancia efectiva se escala a [0, w0] para que el espaciado mínimo
    se aplique en todo el interior del cono.

        d_eff = d * w0 / w_local   si d < w_local
              = d                  si d >= w_local

    Args:
        pts       : (N, 2) puntos de consulta
        streamline: (M, 2) eje de la pluma
        w0        : semi-ancho mínimo del cono en fnte [m]
        w1        : semi-ancho máximo del cono en outlet [m]

    Devuelve:
        d_eff : (N,) distancia efectiva al eje de la pluma
    """
    pts = np.atleast_2d(pts)
    seg_vecs = np.diff(streamline, axis=0)          # (M-1, 2)
    seg_len2 = np.einsum('ij,ij->i', seg_vecs, seg_vecs)  # (M-1,)

    # Longitudes acumuladas (para parametrizar el cono)
    seg_len = np.sqrt(np.maximum(seg_len2, 1e-20))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])  # (M,)
    L = arc[-1]

    n_pts = len(pts)
    d_eff = np.full(n_pts, np.inf)

    for k, (a, ab, l2, l) in enumerate(zip(streamline[:-1], seg_vecs, seg_len2, seg_len)):
        if l2 < 1e-20:
            continue
        t = np.clip(((pts - a) @ ab) / l2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d_lat = np.linalg.norm(pts - proj, axis=1)   # distancia lateral al eje

        # Penalizar puntos que quedan más de depth_below metros POR DEBAJO
        # de la proyección sobre el eje (z_pts < z_proj - depth_below).
        z_proj = proj[:, 1]
        dz_below = np.maximum(0.0, z_proj - depth_below - pts[:, 1])
        d = np.sqrt(d_lat**2 + dz_below**2)

        # Posición normalizada a lo largo del eje (0 = fnte, 1 = outlet)
        s_norm = np.clip((arc[k] + t * l) / L, 0.0, 1.0)
        w_local = w0 + (w1 - w0) * s_norm

        # Dentro del cono lateral: escalar para que w_local → w0
        inside = d_lat < w_local
        d_scaled = np.where(inside, d_lat * w0 / w_local + dz_below, d)

        d_eff = np.minimum(d_eff, d_scaled)

    return d_eff


def get_plume_streamline(pkl_path=None):
    """Devuelve la streamline cacheada o la calcula."""
    if os.path.exists(_CACHE_PATH):
        return np.load(_CACHE_PATH)

    if pkl_path is None:
        from funtions.runtime import RUNTIME
        p = RUNTIME.get()
        pkl_path = f"data/input/bfr/{p.domain}/adr/preproceso.pkl"

    return compute_plume_streamline(pkl_path)
