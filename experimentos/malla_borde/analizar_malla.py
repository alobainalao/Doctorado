"""
Compara BFR y FEniCSx en una corrida de experimentos/malla_borde/correr_malla.py (directo y ψ_H) y la pone junto a la
corrida oficial (malla compartida).  ψ_H de BFR con la salida en 0 (convención del cierre libre) en ambas.

    python -m experimentos.malla_borde.analizar_malla <etiqueta> [<modelo>]
"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(CODIGO_DIR, "data", "output")
etiqueta = sys.argv[1]
modelo = sys.argv[2] if len(sys.argv) > 2 else "adr"


def cargar(bfr_dir, fem_dir, malla_pkl):
    L = lambda d, f: np.load(os.path.join(d, f + "_results.npz"), allow_pickle=True)  # noqa: E731
    sb, sf, ab, af = L(bfr_dir, "simulation"), L(fem_dir, "simulation"), L(bfr_dir, "adjoint"), L(fem_dir, "adjoint")
    M = cKDTree(sb["nodes"]).query(sf["nodes"])[1]            # nodo FEM k ↔ índice BFR M[k]
    salida = np.isin(M, np.asarray(pickle.load(open(malla_pkl, "rb"))["groups"]["boundary:outlet"], int))
    return sb, sf, ab, af, M, salida


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def analizar(nombre, bfr_dir, fem_dir, malla_pkl):
    sb, sf, ab, af, M, salida = cargar(bfr_dir, fem_dir, malla_pkl)
    x, z = sf["nodes"][:, 0], sf["nodes"][:, 1]
    nn = cKDTree(sf["nodes"]).query(sf["nodes"], k=7)[1][:, 1:]
    print(f"\n=== {nombre}: {len(sf['nodes'])} nodos de dominio")
    Hb, Hf = sb["H"][-1, 0][M], sf["H"][-1, 0]
    Cb, Cf = sb["C"][-1, 0][M], sf["C"][-1, 0]
    Ub, Uf = np.linalg.norm(sb["U"][-1, 0][M], axis=1), np.linalg.norm(sf["U"][-1], axis=1)
    print(f"directo (último paso):  ΔH_rms = {np.sqrt(np.mean((Hb - Hf) ** 2)):.4f} m   C L2 rel = {rel(Cb, Cf):.4f}   |U| L2 rel = {rel(Ub, Uf):.4f}")
    zonas = {"superficie z>230": z > 230, "franja 150–230": (z > 150) & (z <= 230), "z<−100": z < -100, "todo z<230": z < 230}
    print(f"{'ψ_H (t_BFR ↔ t_FEM)':22s} " + " ".join(f"{k:>34s}" for k in zonas))
    print(f"{'':22s} " + " ".join(f"{'corr   rug_BFR  rug_FEM  rms/FEM':>34s}" for _ in zonas))
    for l, t in ((20, 16), (60, 57), (84, 82), (120, 119)):
        pb = ab["psi_H"][l - 1, 0][M].copy()
        pb[salida] = 0.0
        pf = af["psi_H"][t]
        fila = []
        for m in zonas.values():
            m = m & ~salida
            r = lambda v: np.sqrt(np.mean((v - v[nn].mean(1))[m] ** 2)) / np.std(v[m])  # noqa: E731
            fila.append(f"{np.corrcoef(pb[m], pf[m])[0, 1]:6.3f} {r(pb):8.3f} {r(pf):8.3f} {np.sqrt(np.mean(pb[m] ** 2)) / np.sqrt(np.mean(pf[m] ** 2)):8.2f}")
        print(f"l={l:3d} (t={t:3d})          " + " ".join(f"{c:>34s}" for c in fila))


base = os.path.join(OUT, "experimentos", "malla_borde", etiqueta, modelo)
analizar(f"malla {etiqueta}", os.path.join(base, "bfr"), os.path.join(base, "fenicsx"),
         os.path.join(OUT, "experimentos", "malla_borde", "mallas", f"m_{etiqueta}.pkl"))
analizar("malla oficial", os.path.join(OUT, "bfr", "data", "real", modelo, "comparison"),
         os.path.join(OUT, "fenicsx", "data", "real", modelo, "comparison"),
         os.path.join(CODIGO_DIR, "data", "input", "malla_compartida", "malla.pkl"))
