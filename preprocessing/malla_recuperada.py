"""
Recupera como malla compartida (pkl con frontera exacta, triángulos y fantasmas) la nube de nodos de una
corrida anterior.  Sirve para las mallas de oct 2026 con el cono horizontal de la pluma y el cono vertical
del pozo (commits 5c2d9a8…2146e19), que no se guardaron como pkl pero sí dentro de sus resultados:

    data/output/bfr/data/real/adr/malla_sp60/simulation_results.npz        13 918 nodos (19 oct, mesh.png)
    data/output/bfr/data/real/adr/malla_10_30_60/simulation_results.npz    12 112 nodos (19 oct)
    data/output/bfr/data/real/adr/comparison_sp60/simulation_results.npz    9 289 nodos (20 oct, BFR y FEniCSx)

    python -m preprocessing.malla_recuperada --npz <simulation_results.npz> --salida data/input/malla_recuperada/<nombre>.pkl

Corre en bfr_env.  Se conservan los nodos interiores tal cual (los de frontera y los fantasmas se rehacen
con el saneamiento de preprocessing/malla_compartida.py); el espaciado local es la distancia al vecino
más cercano en la nube original.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from preprocessing import malla_compartida as mc   # noqa: E402

FRAC_INTERIOR = 0.25   # nodos a < FRAC_INTERIOR·h del borde se tratan como de frontera (se rehacen)


def recuperar(nodos_npz, vert):
    """Nodos interiores de la nube original y espaciado h(xy) = distancia al vecino más cercano."""
    x = np.asarray(nodos_npz, float)
    hn = cKDTree(x).query(x, 2)[0][:, 1]
    dentro = mc._dentro(x, vert) & (mc._dist_borde(x, vert) > FRAC_INTERIOR * hn)
    xi, hi = x[dentro], hn[dentro]
    arbol = cKDTree(xi)

    def h(xy):
        d, i = arbol.query(np.atleast_2d(xy), 4)
        w = 1.0 / np.maximum(d, 1e-9)
        return (hi[i] * w).sum(1) / w.sum(1)
    return xi, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="simulation_results.npz con la clave 'nodes'")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--config", default=mc.BASE_JSON)
    ap.add_argument("--figura", help="PNG con la malla y zooms de esquinas y pozo")
    args = ap.parse_args()
    os.chdir(CODIGO_DIR)
    if os.path.exists(args.salida):
        raise SystemExit(f"[malla] {args.salida} ya existe: muévela antes (la malla congelada no se pisa)")

    with open(args.config) as f:
        cfg = json.load(f)
    from config.parameters import Parameters
    vert = np.asarray(Parameters(config={**cfg, "metodo": "bfr"}).vert, float)
    xi, h = recuperar(np.load(args.npz)["nodes"], vert)
    m = mc.generar(cfg, interiores=xi, h_fijo=h, n_suave=0,
                   nota_extra=f"; nodos interiores recuperados de {args.npz}")
    m["meta"]["origen"] = f"preprocessing/malla_recuperada.py desde {args.npz}"

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, "wb") as f:
        pickle.dump(m, f)
    print(f"[malla] {m['meta']['n_nodos']} nodos ({m['meta']['n_dominio']} del dominio) → {args.salida}")
    print(json.dumps(mc.diagnostico(m), indent=2, ensure_ascii=False))
    if args.figura:
        mc.figura(m, args.figura)


if __name__ == "__main__":
    main()
