"""
Lectura de las corridas del simulador que usa el notebook del cuadrado:

  Z_cuadrado/Z{0,1}_*      BFR, cuadrado de 600 m con pozo central (rama exp/sumidero-W)
  S_sumidero/S{2,3}_*      FEniCS, dominio real, fuente 30 d + 90 d de limpieza (balance por paso)
  S_sumidero/S{0,1}_*      BFR y FEniCS, dominio real, 30 d (configuración de referencia)

Sólo lee resultados: no corre nada. Las cantidades se calculan igual que en
experimentos/analisis_sumidero.py (áreas nodales de la triangulación del
dominio sin fantasmas; C captada = ∫ g_ε C con la gaussiana del pozo).
"""
import json

import numpy as np
from matplotlib.tri import Triangulation

import rutas

DIA = 86400.0


def _areas_nodales(n, dentro):
    tri = Triangulation(n[dentro, 0], n[dentro, 1])
    w = np.zeros(len(n))
    idx = np.nonzero(dentro)[0]
    for t3 in tri.triangles:
        q = n[idx[t3]]
        a = 0.5 * abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1]))
        w[idx[t3]] += a / 3
    return tri, w


def cargar_Z(caso):
    """Z0_sin_sumidero o Z1_con_sumidero (BFR)."""
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from funtions.utils import discrete_delta, normalizar_delta_area, gaussian_2d
    from funtions.operators import rbf_integration_weights

    d_caso = rutas.grupo("Z_cuadrado") / caso / "bfr"
    cfg = json.load(open(d_caso / "config.json"))
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.params
    d = np.load(d_caso / "simulation_results.npz")
    n, C, dt = d["nodes"], d["C"][:, 0], float(d["dt"])
    L = float(p.lado_cuadrado)
    tol = 1e-9
    dentro = (n[:, 0] >= -tol) & (n[:, 0] <= L + tol) & (n[:, 1] >= -L - tol) & (n[:, 1] <= tol)
    tri, w = _areas_nodales(n, dentro)
    pozo = np.asarray(p.pozo, float)
    delta = discrete_delta(n, pozo, p.radio_delta_pozo)
    if p.norm_delta_pozo == "area":
        delta = normalizar_delta_area(delta, rbf_integration_weights(n))
    g = gaussian_2d(n, pozo, 1, p.epsilon_pozo)
    C0 = float(p.C_init_const)
    masa = (C * p.phi_const * p.R) @ w
    k_pozo = int(np.argmin(np.linalg.norm(n - pozo, axis=1)))
    return dict(
        t=np.arange(1, len(C) + 1) * dt / DIA, dt=dt, C0=C0, Q=float(p.Q_std),
        masa=masa / masa[0], captada=C @ (g * w) / C0, c_nodo_pozo=C[:, k_pozo] / C0,
        nodos=n[dentro] - [L / 2, -L / 2], tri=tri, C=C[:, dentro] / C0,
        r_pozo=float(p.r_pozo), factor=float(cfg.get("factor_sumidero_C", 1.0 if cfg.get("extract_C") else 0.0)),
    )


def cargar_balance_fem(caso):
    """Balance por paso de FEniCS (S2_limpieza_sin / S3_limpieza_con)."""
    d = rutas.grupo("S_sumidero") / caso / "fenicsx"
    b = dict(np.load(d / "balance_masa.npz"))
    dt = float(b["dt"])
    m0 = float(b["masa0"])
    acum = lambda v: np.cumsum(v) * dt / m0
    return dict(
        t=np.arange(1, len(b["masa"]) + 1) * dt / DIA,
        masa=b["masa"] / m0, captada=b["C_pozo"],
        agua=acum(b["pozo_agua"] * b["Q"]), W=acum(b["sumidero_W"]),
        frontera=acum(b["frontera_sale"]), fuente=acum(b["fuente"]),
    )


def cargar_S30(caso, backend):
    """S0_sin_termino / S1_con_termino a 30 d: masa y C captada del simulador."""
    d = rutas.grupo("S_sumidero") / caso / backend
    r = np.load(d / "simulation_results.npz")
    return r
