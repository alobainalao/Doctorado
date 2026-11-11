"""
Malla compartida con conos: 4 niveles de espaciado en los que el nivel intermedio son los dos conos de
oct 2026 (commits 87f4654 y 5c2d9a8) en lugar del anillo radial de la malla compartida.

    python -m preprocessing.malla_conos                       # → data/input/malla_conos/malla.pkl
    python -m preprocessing.malla_conos --figura <prefijo>    # + PNG de la malla y del espaciado

Corre en bfr_env.  No sustituye a data/input/malla_compartida/malla.pkl: para usarla, un config aparte
con "mesh_pkl": "data/input/malla_conos/malla.pkl".

Niveles (valores del config: min_sp, s_cone, min_sp_boundary, spacing):
  * min_sp (4 m)            dos elipses:
        - pozo: eje mayor vertical, con focos en la posición actual del pozo y en la cota superior de la
          optimización (x_p, zp_max), y semieje menor r_circle: cubre la zona que recorre z_p al optimizar;
        - fuente: la misma construcción, horizontal y más chica: focos en la fuente y DX_FUENTE aguas
          abajo (hacia la salida), semieje menor B_FUENTE;
  * s_cone (16 m)           dentro de los dos conos:
        - pluma: alrededor de la línea de corriente de Darcy desde la fuente hasta la salida (RK4 con la
          velocidad inicial del preproceso actual), semiancho de PLUME_W0 en la fuente a PLUME_W1 en la
          salida, y como mucho PLUME_DEPTH_BELOW por debajo del eje;
        - pozo: columna vertical desde WELL_R0 bajo el pozo hasta el eje de la pluma, semiancho de
          WELL_R0 en el pozo a WELL_R1 en el eje;
  * min_sp_boundary (30 m)  sobre la frontera; en la frontera superior es además una cota inferior:
          ninguna zona (en particular el cono de la pluma, que llega al techo) la baja de 30 m allí;
  * spacing (60 m)          en el resto.
Fuera de cada zona el espaciado crece linealmente con la distancia, GRADO metros por metro, hasta
`spacing`; el espaciado final es el mínimo de las cuatro zonas y del refinamiento por K de la malla
compartida (preprocessing/malla_compartida.py:_espaciado_K), acotado por debajo cerca del techo por
min_sp_boundary − GRADO·d_techo.  Se precalcula en una rejilla de PASO_REJILLA m.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from preprocessing import malla_compartida as mc   # noqa: E402

SALIDA = os.path.join("data", "input", "malla_conos", "malla.pkl")
EJE = os.path.join("data", "input", "malla_conos", "eje_pluma.npy")
PREPROCESO = os.path.join("data", "input", "bfr", "real", "adr", "preproceso.pkl")
GRADO = 0.2          # [m/m] crecimiento del espaciado fuera de cada zona (vecinos con razón ≈ 1.2)
DX_FUENTE = 250.0    # [m] distancia entre los focos de la elipse de la fuente (la del pozo: |z_p − zp_max|)
B_FUENTE = 50.0      # [m] semieje menor de la elipse de la fuente
PASO_REJILLA = 10.0  # [m] rejilla en la que se precalcula h (el generador lo evalúa millones de veces)


def exceso_pluma(xy, eje, w0, w1, abajo):
    """Distancia de cada punto al cono de la pluma (0 dentro): lateral más allá del semiancho local
    w(s) = w0 + (w1 − w0)·s/L y vertical más allá de `abajo` por debajo del eje."""
    seg = np.diff(eje, axis=0)
    l2 = np.einsum("ij,ij->i", seg, seg)
    arco = np.r_[0.0, np.cumsum(np.sqrt(l2))]
    e = np.full(len(xy), np.inf)
    for k in range(len(seg)):
        if l2[k] < 1e-20:
            continue
        t = np.clip(((xy - eje[k]) @ seg[k]) / l2[k], 0.0, 1.0)
        proy = eje[k] + t[:, None] * seg[k]
        w = w0 + (w1 - w0) * (arco[k] + t * np.sqrt(l2[k])) / arco[-1]
        lat = np.linalg.norm(xy - proy, axis=1)
        bajo = np.maximum(0.0, proy[:, 1] - abajo - xy[:, 1])
        e = np.minimum(e, np.hypot(np.maximum(0.0, lat - w), bajo))
    return e


def exceso_pozo(xy, pozo, z_eje, r0, r1):
    """Distancia al cono vertical del pozo (0 dentro): de z_p − r0 a z_eje, semiancho r0 → r1."""
    xp, zp = pozo
    z = xy[:, 1]
    w = r0 + (r1 - r0) * np.clip((z - zp) / (z_eje - zp), 0.0, 1.0)
    return np.hypot(np.hypot(np.maximum(0.0, np.abs(xy[:, 0] - xp) - w), np.maximum(0.0, zp - r0 - z)),
                    np.maximum(0.0, z - z_eje))


def exceso_elipse(xy, f1, f2, b):
    """Distancia aproximada a la elipse de focos f1, f2 y semieje menor b (0 dentro):
    (|x − f1| + |x − f2|)/2 − a, con a = √(c² + b²); exacta sobre el eje mayor."""
    a = np.hypot(0.5 * np.linalg.norm(np.subtract(f2, f1)), b)
    s = np.linalg.norm(xy - f1, axis=1) + np.linalg.norm(xy - f2, axis=1)
    return np.maximum(0.0, 0.5 * s - a)


def elipses(p, eje):
    """(f1, f2, b) de las elipses del pozo y de la fuente."""
    pozo, fnte = np.asarray(p.pozo, float), np.asarray(p.fnte, float)
    zp_max = float(getattr(p, "zp_max", -200.0))
    sentido = np.sign(eje[min(5, len(eje) - 1), 0] - eje[0, 0]) or -1.0   # hacia la salida
    return {"pozo": (pozo, np.array([pozo[0], zp_max]), float(p.r_circle)),
            "fuente": (fnte, fnte + np.array([sentido * DX_FUENTE, 0.0]), B_FUENTE)}


def espaciado(p, eje, vert):
    """h(xy) de 4 niveles con los conos como nivel intermedio."""
    from funtions.mesh_test import PLUME_W0, PLUME_W1, PLUME_DEPTH_BELOW, WELL_R0, WELL_R1
    pozo, fnte = np.asarray(p.pozo, float), np.asarray(p.fnte, float)
    s_min, s_cono = float(p.min_sp), float(p.s_cone)
    s_bnd, s_max = float(p.min_sp_boundary), float(p.spacing)
    z_eje = float(np.interp(pozo[0], eje[::-1, 0], eje[::-1, 1]))   # eje de la pluma sobre el pozo

    elip = elipses(p, eje)
    z_techo = float(vert[:, 1].max())

    def h_exacto(xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        e_elip = np.minimum.reduce([exceso_elipse(xy, *elip[k]) for k in elip])
        e_cono = np.minimum(exceso_pluma(xy, eje, PLUME_W0, PLUME_W1, PLUME_DEPTH_BELOW),
                            exceso_pozo(xy, pozo, z_eje, WELL_R0, WELL_R1))
        return np.minimum.reduce([
            s_min + GRADO * e_elip,
            s_cono + GRADO * e_cono,
            s_bnd + GRADO * mc._dist_borde(xy, vert),
            np.full(len(xy), s_max),
        ])

    from scipy.interpolate import RegularGridInterpolator
    h_k = mc._espaciado_K(h_exacto, vert, mc.REFINO_K, mc.H_MIN_K) if mc.REFINO_K else h_exacto
    x = np.arange(vert[:, 0].min() - PASO_REJILLA, vert[:, 0].max() + 2 * PASO_REJILLA, PASO_REJILLA)
    z = np.arange(vert[:, 1].min() - PASO_REJILLA, vert[:, 1].max() + 2 * PASO_REJILLA, PASO_REJILLA)
    X, Z = np.meshgrid(x, z, indexing="ij")
    P = np.c_[X.ravel(), Z.ravel()]
    techo = s_bnd - GRADO * (z_techo - P[:, 1])               # 30 m en el techo, sin efecto a 70 m de él
    tabla = np.maximum(h_k(P), techo).reshape(X.shape)
    f = RegularGridInterpolator((x, z), tabla, bounds_error=False, fill_value=None)
    return lambda xy: f(np.atleast_2d(np.asarray(xy, float)))


def figura_espaciado(m, eje, ruta):
    """Distancia al vecino más cercano de cada nodo, con el eje de la pluma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    n_dom = m["meta"]["n_dominio"]
    x = m["nodes"][:n_dom]
    d = cKDTree(x).query(x, 2)[0][:, 1]
    fig, ax = plt.subplots(figsize=(16, 5))
    sc = ax.scatter(*x.T, c=d, s=1.0, cmap="viridis_r", vmin=0, vmax=60)
    ax.plot(*eje.T, "r-", lw=0.8, label="eje de la pluma")
    ax.set_aspect("equal")
    ax.set_title(f"Malla con conos: {n_dom} nodos del dominio; color = distancia al vecino más cercano [m]")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    print(f"[malla] figura → {ruta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=mc.BASE_JSON)
    ap.add_argument("--salida", default=SALIDA)
    ap.add_argument("--preproceso", default=PREPROCESO, help="pkl con la velocidad inicial U[0]")
    ap.add_argument("--figura", help="prefijo de los PNG (malla y espaciado)")
    args = ap.parse_args()
    os.chdir(CODIGO_DIR)
    if os.path.exists(args.salida):
        raise SystemExit(f"[malla] {args.salida} ya existe: muévela antes (la malla congelada no se pisa)")

    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from funtions.plume import compute_plume_streamline
    with open(args.config) as f:
        cfg = json.load(f)
    RUNTIME.params = Parameters(config={**cfg, "metodo": "bfr"})
    p = RUNTIME.params
    vert = np.asarray(p.vert, float)

    eje = compute_plume_streamline(args.preproceso, ruta=EJE)
    if eje is None:
        raise SystemExit(f"[malla] no existe {args.preproceso}")
    nota = (f"; 4 niveles con conos: {p.min_sp:g} m en las elipses de pozo (vertical, focos z_p y zp_max) y fuente "
            f"(horizontal), {p.s_cone:g} m en los conos de pluma y pozo, {p.min_sp_boundary:g} m en la frontera "
            f"(cota inferior en el techo), {p.spacing:g} m lejos (eje {EJE})")
    m = mc.generar(cfg, h_fijo=espaciado(p, eje, vert), refino_k=None, nota_extra=nota)   # K ya va en h
    m["meta"]["origen"] = (f"preprocessing/malla_conos.py (min_sp={p.min_sp}, s_cone={p.s_cone} en los conos, "
                           f"min_sp_boundary={p.min_sp_boundary}, spacing={p.spacing}, r_circle={p.r_circle}, "
                           f"grado={GRADO})")
    m["eje_pluma"] = eje

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, "wb") as f:
        pickle.dump(m, f)
    print(f"[malla] {m['meta']['n_nodos']} nodos ({m['meta']['n_dominio']} del dominio) → {args.salida}")
    print(json.dumps(mc.diagnostico(m), indent=2, ensure_ascii=False))
    if args.figura:
        mc.figura(m, args.figura + "_malla.png")
        figura_espaciado(m, eje, args.figura + "_espaciado.png")


if __name__ == "__main__":
    main()
