"""
Variantes de la malla compartida para estudiar el ruido de ψ_H cerca de la pared (docs/adjunto_sin_sumidero.tex).
No modifica preprocessing/malla_compartida.py: reutiliza su `generar` cambiando tres pasos por reemplazo de texto.

    python -m experimentos.malla_borde.gen_malla <delta> <capas> <modo> <salida.pkl>

  delta  : DELTA_FANTASMA (la oficial es 1.0)
  capas  : número de capas de nodos paralelas a cada pared (0 = sin capas)
  modo   : "mezcla"     paso uniforme junto a la pared y gradual hacia h(x) a partir de la 3.ª capa (D_MEZCLA m)
           "fijas"      capas a d_k = k·(√3/2)·s del lado, con el paso tangencial s del lado (frontera uniforme)
           "graduadas"  cada capa avanza (√3/2)·h(x) local y reparte sus nodos con ∫ds/h (frontera uniforme)
           "ninguna"    solo la frontera con espaciado uniforme por lado (capas = 0)
Las capas van escalonadas medio paso (triángulos equiláteros) y se fijan en el suavizado de Laplace.
"""
import inspect
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODIGO_DIR)
import preprocessing.malla_compartida as mc  # noqa: E402

RAIZ3_2 = 0.8660254037844386
D_MEZCLA = float(os.environ.get("D_MEZCLA", 200.0))     # [m] distancia de la mezcla uniforme → h(x)
FRAC_SOLAPE = float(os.environ.get("FRAC_SOLAPE", 0.7))   # radio de exclusión entre nodos de capas, en unidades de h local


def _paso_lado(a, b, h):
    L = np.linalg.norm(b - a)
    s = np.linspace(0.0, L, 4001)
    m = max(1, int(round(L / float(np.mean(h(a + np.outer(s / L, b - a)))))))
    return L, m, L / m


def _frontera_uniforme(vert, grupo_lado, h):
    """Como mc._frontera, pero con paso uniforme en cada lado (h = media de h(s) en el lado)."""
    pts, grp, nor, lado = [], [], [], []
    lados = mc._lados(vert)
    for i, (a, b, n) in enumerate(lados):
        L, m, _ = _paso_lado(a, b, h)
        for k, sv in enumerate(np.linspace(0.0, L, m + 1)[:-1]):
            pts.append(a + sv / L * (b - a))
            if k == 0:
                nb = n + lados[i - 1][2]
                nor.append(nb / np.linalg.norm(nb))
                grp.append("wall")
            else:
                nor.append(n)
                grp.append(grupo_lado[i])
            lado.append(i)
    return np.array(pts), np.array(grp), np.array(nor), np.array(lado)


def _quitar_duplicados(xy, dmin):
    keep = np.ones(len(xy), bool)
    for i, j in sorted(cKDTree(xy).query_pairs(dmin)):
        if keep[i] and keep[j]:
            keep[j] = False
    return xy[keep]


def _quitar_solapes(xy, dk, radio):
    """Donde se cruzan las capas de dos lados (esquinas): recorrido de la capa más cercana a la pared a la más lejana;
    cada nodo conservado elimina a los posteriores a menos de radio(x) = FRAC_SOLAPE·h(x)."""
    tree = cKDTree(xy)
    keep = np.ones(len(xy), bool)
    r = radio(xy)
    for i in np.argsort(dk, kind="stable"):
        if keep[i]:
            for j in tree.query_ball_point(xy[i], r[i]):
                if j != i and keep[j] and dk[j] >= dk[i]:
                    keep[j] = False
    return xy[keep]


def _capas(vert, h, K):
    # "mezcla": paso uniforme sp del lado hasta ~2 capas y después pasa gradualmente (en D_MEZCLA m) al campo h(x)
    def hloc(P, dd, sp):
        w = np.clip((dd - 2 * RAIZ3_2 * sp) / D_MEZCLA, 0.0, 1.0)
        return sp + w * (h(P) - sp)
    cand, dk = [], []
    lados = mc._lados(vert)
    for (a, b, n) in lados:
        L, m, sp = _paso_lado(a, b, h)
        t = (b - a) / L
        s = np.linspace(0.0, L, 4001)
        d = np.zeros_like(s)
        for k in range(1, K + 1):
            off = 0.5 if k % 2 == 1 else 1.0
            if MODO in ("graduadas", "mezcla"):
                hl = hloc if MODO == "mezcla" else (lambda P, dd, sp_: h(P))
                d = d + RAIZ3_2 * hl(a + np.outer(s, t) - np.outer(d, n), d, sp)
                hs = hl(a + np.outer(s, t) - np.outer(d, n), d, sp)
                F = np.concatenate([[0.0], np.cumsum(0.5 * (1 / hs[1:] + 1 / hs[:-1]) * np.diff(s))])
                sv = np.interp(np.arange(off, F[-1] - 1e-9, 1.0), F, s)
            else:
                d = np.full_like(s, k * RAIZ3_2 * sp)
                sv = np.arange(0.5 * sp if k % 2 == 1 else sp, L - 1e-9, sp)
            dv = np.interp(sv, s, d)
            cand.append(a + np.outer(sv, t) - np.outer(dv, n))
            dk.append(dv)
            if k == K:
                mc._ULTIMA.append((cand[-1], dv))
    xy, dk = np.vstack(cand), np.concatenate(dk)
    ok = mc._dentro(xy, vert) & (mc._dist_borde(xy, vert) >= 0.9 * dk)   # fuera del solape de esquinas
    if MODO in ("graduadas", "mezcla"):
        xy = _quitar_solapes(xy[ok], dk[ok], lambda P: FRAC_SOLAPE * h(P))
    else:
        xy = _quitar_duplicados(xy[ok], 0.6 * min(_paso_lado(a, b, h)[2] for a, b, _ in lados))
    print(f"[capas] {K} capas ({MODO}): {len(xy)} nodos")
    return xy


def _lejos(interior, vert, hi):
    """Interiores que se conservan: fuera de la última capa (+0.5 h)."""
    pout = np.vstack([u[0] for u in mc._ULTIMA])
    dout = np.concatenate([u[1] for u in mc._ULTIMA])
    j = cKDTree(pout).query(interior)[1]
    return mc._dist_borde(interior, vert) >= dout[j] + 0.5 * hi


def main():
    global MODO
    delta, K, MODO, salida = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    mc.DELTA_FANTASMA = delta
    mc._frontera = _frontera_uniforme
    mc._ULTIMA = []
    mc._capas, mc._lejos, mc.N_CAPAS = _capas, _lejos, K
    src = inspect.getsource(mc.generar)
    rep = [("lejos = (_dist_borde(interior, vert) >= FRAC_BORDE * hi) & _dentro(interior, vert)",
            "capas = _capas(vert, h, N_CAPAS)\n    lejos = _lejos(interior, vert, hi) & _dentro(interior, vert)"
            if K > 0 else
            "capas = np.zeros((0, 2))\n    lejos = (_dist_borde(interior, vert) >= FRAC_BORDE * hi) & _dentro(interior, vert)"),
           ("nodos = np.vstack([bnd, interior[lejos], pozo[None]])",
            "nodos = np.vstack([bnd, capas, interior[lejos], pozo[None]])"),
           ("np.r_[np.arange(n_bnd), i_pozo]", "np.r_[np.arange(n_bnd + len(capas)), i_pozo]")]
    for a, b in rep:
        assert a in src, a
        src = src.replace(a, b)
    exec(src, mc.__dict__)
    sys.argv = ["malla_compartida", "--salida", salida]
    mc.main()


MODO = "ninguna"
if __name__ == "__main__":
    main()
