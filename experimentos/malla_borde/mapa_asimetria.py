"""
Dónde está la asimetría de W·A_left: reparto de ‖WA−(WA)ᵀ‖_F² por lado del polígono (el más cercano) y banda de distancia.

    python -m experimentos.malla_borde.mapa_asimetria <etiqueta> [<etiqueta> ...]     # lee el preproceso de cada malla
"""
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERT = np.array([[816, -2690.5], [4354, -1757.5], [10755, -768.5], [10755, 250.5], [816, 250.5]])
LADOS = ["pared inferior inclinada", "entrada (derecha, x=10755)", "pared superior", "salida (izquierda, x=816)"]
# vértices en orden: 0→1 inferior1, 1→2 inferior2, 2→3 entrada, 3→4 superior, 4→0 salida
NOMBRES = ["inferior 1", "inferior 2", "entrada", "superior", "salida"]


def dist_lado(xy):
    out = np.empty((len(xy), 5))
    for i in range(5):
        p0, p1 = VERT[i], VERT[(i + 1) % 5]
        e = p1 - p0
        t = np.clip(((xy - p0) @ e) / (e @ e), 0, 1)
        out[:, i] = np.linalg.norm(xy - (p0 + t[:, None] * e), axis=1)
    return out


for tag in sys.argv[1:]:
    ruta = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "malla_borde", tag, "bfr", "preproceso.pkl") \
        if tag != "oficial" else None
    if tag == "oficial":
        ruta = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "malla_borde", "oficial", "bfr", "preproceso.pkl")
    raw = pickle.load(open(ruta, "rb"))
    A = sp.csr_matrix(raw["A_left"])
    nodes, wi, g = np.asarray(raw["nodes"]), np.asarray(raw["wi"], float), raw["groups"]
    dom = np.hstack([g["interior"], g["boundary:inlet"], g["boundary:outlet"], g["boundary:wall"]]).astype(int)
    Md = (sp.diags(wi) @ A)[dom][:, dom].tocsr()
    D = (Md - Md.T).tocsr()
    fd = np.asarray(D.multiply(D).sum(1)).ravel()          # contribución de cada fila a ‖D‖_F²
    fm = np.asarray(Md.multiply(Md).sum(1)).ravel()
    a = np.asarray(abs(D).sum(1)).ravel() / np.maximum(np.asarray((abs(Md) + abs(Md.T)).sum(1)).ravel(), 1e-300)
    xy = nodes[dom]
    dl = dist_lado(xy)
    lado, dmin = dl.argmin(1), dl.min(1)
    print(f"\n=== {tag}: N_dominio={len(dom)}   ‖D‖_F/‖M‖_F = {np.sqrt(fd.sum() / fm.sum()):.3f}")
    print(f"{'lado más cercano':14s} " + " ".join(f"{b:>14s}" for b in ("<30 m", "30-100 m", "100-300 m", ">300 m")))
    for i in range(5):
        fila = []
        for lo, hi in ((-1, 30), (30, 100), (100, 300), (300, 1e9)):
            m = (lado == i) & (dmin > lo) & (dmin <= hi)
            fila.append(f"{100 * fd[m].sum() / fd.sum():5.1f}% (a={np.median(a[m]) if m.sum() else float('nan'):.2f})")
        print(f"{NOMBRES[i]:14s} " + " ".join(f"{c:>14s}" for c in fila))
    k = np.argsort(fd)[::-1][:8]
    print("filas que más aportan a ‖D‖_F² (x, z, lado, dist, aporte, a_i):")
    for j in k:
        print(f"   ({xy[j,0]:8.1f}, {xy[j,1]:8.1f})  {NOMBRES[lado[j]]:10s} d={dmin[j]:6.1f}  {100*fd[j]/fd.sum():5.2f}%  a={a[j]:.2f}")
