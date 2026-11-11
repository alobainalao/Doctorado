"""
Variante de la malla compartida con los conos de refinamiento de oct 2026
(commits 87f4654 y 5c2d9a8, retirados en ed9151a): un cono horizontal que
sigue la pluma y un cono vertical de succión sobre el pozo, superpuestos a
los 4 niveles de la malla compartida.

    python -m preprocessing.malla_conos                       # → data/input/malla_conos/malla.pkl
    python -m preprocessing.malla_conos --s-cono 20 --figura malla.png

Corre en bfr_env.  No sustituye a data/input/malla_compartida/malla.pkl: para
usarla, un config aparte con "mesh_pkl": "data/input/malla_conos/malla.pkl".

Conos (mismas funciones y constantes que entonces, funtions/plume.py y
funtions/mesh_test.py):
  * Pluma: eje = línea de corriente de la velocidad de Darcy inicial desde
    la fuente hasta la salida (RK4, recalculada con el preproceso actual);
    semiancho de PLUME_W0 en la fuente a PLUME_W1 en la salida; hacia abajo
    del eje sólo se extiende PLUME_DEPTH_BELOW.
  * Pozo: eje vertical x = x_pozo desde WELL_R0 por debajo del pozo hasta
    la cota de la fuente + WELL_Z_OFFSET; semiancho de WELL_R0 a WELL_R1.
Dentro de cada cono el espaciado parte de s_cono y crece como √d hasta
`spacing`; el espaciado final es el mínimo con el de la malla compartida
(círculos de min_sp en pozo y fuente, frontera, refinamiento por K).
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


def espaciado_conos(p, eje, s_cono):
    """h(xy) de los dos conos, con s_min = s_cono y s_max = spacing."""
    from funtions.plume import plume_cone_dist
    from funtions.mesh_test import (PLUME_W0, PLUME_W1, PLUME_DEPTH_BELOW, PLUME_D_FAR,
                                    WELL_R0, WELL_R1, WELL_Z_OFFSET, _s, _well_vertical_cone)
    pozo, fnte, s_max = np.asarray(p.pozo, float), np.asarray(p.fnte, float), float(p.spacing)

    def h(xy):
        xy = np.atleast_2d(xy)
        d = plume_cone_dist(xy, eje, w0=PLUME_W0, w1=PLUME_W1, depth_below=PLUME_DEPTH_BELOW)
        s_pluma = _s(d, s_cono, s_max, PLUME_D_FAR)
        s_pozo = _well_vertical_cone(xy, pozo, s_cono, s_max, r0=WELL_R0, r1=WELL_R1,
                                     z_top=float(fnte[1]) + WELL_Z_OFFSET)
        return np.minimum(s_pluma, s_pozo)
    return h


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
    ap.add_argument("--s-cono", type=float, help="espaciado en el eje de los conos (por defecto s_cone)")
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
    s_cono = float(args.s_cono or p.s_cone)

    eje = compute_plume_streamline(args.preproceso, ruta=EJE)
    if eje is None:
        raise SystemExit(f"[malla] no existe {args.preproceso}")
    nota = (f"; conos de pluma y pozo con s_cono={s_cono:g} m (eje {EJE}, "
            f"velocidad inicial de {args.preproceso})")
    m = mc.generar(cfg, h_extra=espaciado_conos(p, eje, s_cono), nota_extra=nota)
    m["meta"]["origen"] = m["meta"]["origen"].replace("malla_compartida.py", "malla_conos.py") + f", s_cono={s_cono:g}"
    m["eje_pluma"] = eje

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, "wb") as f:
        pickle.dump(m, f)
    print(f"[malla] {m['meta']['n_nodos']} nodos → {args.salida}")
    print(json.dumps(mc.diagnostico(m), indent=2, ensure_ascii=False))
    if args.figura:
        mc.figura(m, args.figura + "_malla.png")
        figura_espaciado(m, eje, args.figura + "_espaciado.png")


if __name__ == "__main__":
    main()
