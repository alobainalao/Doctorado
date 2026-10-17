"""
Streamline de la pluma del contaminante.

Traza la trayectoria real desde fnte hasta pozo integrando el campo de
velocidades Darcy U (estado estacionario) del pkl de preproceso BFR.
Usa RK4 con paso normalizado en dirección para seguir la curva de flujo.

La streamline se cachea en data/input/plume_streamline.npy. Si el pkl aún
no existe (primera generación de malla), devuelve None y el caller usa el
fallback (distancia a fnte+pozo directamente).
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


def compute_plume_streamline(pkl_path, ds=30.0, n_steps=3000):
    """
    Integra la streamline desde fnte hasta pozo con RK4.

    pkl_path : preproceso.pkl del backend BFR (contiene nodes y U[0])
    ds       : paso de integración [m]; ~spacing es razonable
    n_steps  : máximo de pasos antes de cortar

    Devuelve array (M, 2) de puntos o None si el pkl no existe.
    Guarda el resultado en _CACHE_PATH para reutilización.
    """
    from scipy.interpolate import LinearNDInterpolator
    from funtions.runtime import RUNTIME
    import pickle

    if not os.path.exists(pkl_path):
        return None

    p = RUNTIME.get()
    fnte = np.asarray(p.fnte, float)
    pozo = np.asarray(p.pozo, float)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    nodes = data["nodes"]   # (N, 2)
    U0 = data["U"][0]       # (N, 2) velocidad Darcy en t=0

    interp_u = LinearNDInterpolator(nodes, U0[:, 0], fill_value=0.0)
    interp_v = LinearNDInterpolator(nodes, U0[:, 1], fill_value=0.0)

    pts = [fnte.copy()]
    pt = fnte.copy()
    prev_dist = np.linalg.norm(pt - pozo)

    for _ in range(n_steps):
        new_pt = _rk4_step(pt, ds, interp_u, interp_v)
        if new_pt is None:
            break
        pts.append(new_pt.copy())
        pt = new_pt
        dist = np.linalg.norm(pt - pozo)
        # Parar si llegamos cerca del pozo
        if dist < ds * 1.5:
            break
        # Parar si la distancia empieza a crecer de nuevo después de acercarse
        # bastante (el punto de mínimo acercamiento fue en el paso anterior)
        if dist > prev_dist and prev_dist < 800:
            break
        # Parar si pasamos el pozo en x (no tiene sentido seguir hacia la pared)
        if pt[0] < pozo[0] - 400:
            break
        prev_dist = dist

    # Truncar en el punto más cercano al pozo y unir con él
    arr = np.array(pts)
    idx_closest = int(np.argmin(np.linalg.norm(arr - pozo, axis=1)))
    streamline = np.vstack([arr[: idx_closest + 1], pozo])

    os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
    np.save(_CACHE_PATH, streamline)
    print(f"[plume] streamline calculada: {len(streamline)} puntos → {_CACHE_PATH}")
    return streamline


def get_plume_streamline(pkl_path=None):
    """
    Devuelve la streamline de la pluma (cacheada o calculada).

    Si el pkl no existe todavía (primera generación de malla), devuelve None;
    el caller debe usar fallback a distancia directa a fnte y pozo.
    """
    if os.path.exists(_CACHE_PATH):
        return np.load(_CACHE_PATH)

    if pkl_path is None:
        from funtions.runtime import RUNTIME
        p = RUNTIME.get()
        pkl_path = f"data/input/bfr/{p.domain}/adr/preproceso.pkl"

    return compute_plume_streamline(pkl_path)
