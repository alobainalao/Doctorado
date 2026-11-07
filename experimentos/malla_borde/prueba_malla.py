"""
Mide, para una malla dada, cuánto se parece la transpuesta del operador de flujo a su directo (docs/adjunto_sin_sumidero.tex):

    python -m experimentos.malla_borde.prueba_malla <etiqueta> <malla.pkl>      # corre en bfr_env

Con una fuente gaussiana lisa b: x = A⁻¹ b y y = A⁻ᵀ(W b); y/W debería parecerse a x (A ≈ W⁻¹·simétrica).
Imprime una línea RESULT con JSON: asimetría de WA (global y por banda de distancia a la superficie),
error ‖y/W−x‖ y rugosidad local de y/W (zonas y absoluta), con tres fuentes (profunda, media, superficial).
Solo toca data/output/experimentos/malla_borde/<etiqueta>/ (el preproceso de la malla).
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(CODIGO_DIR)
sys.path.insert(0, CODIGO_DIR)
tag, meshpkl = sys.argv[1], sys.argv[2]

cfg = json.load(open("data/output/_cfg/bfr_adr_comparison.json"))
cfg.update({"galerkin_banda": 0.0, "mesh_pkl": meshpkl,
            "exp_dir": os.path.join(CODIGO_DIR, "data", "output", "experimentos", "malla_borde", tag)})
from funtions.runtime import RUNTIME  # noqa: E402
from config.parameters import Parameters  # noqa: E402
RUNTIME.params = Parameters(config=cfg)
p = RUNTIME.get()
from preprocessing.preprocess import load_data  # noqa: E402
from funtions.operators import build_H_matrix  # noqa: E402

d = load_data()
nodes, g, wi = d.nodes, d.groups, d.wi
N = len(nodes)
gi, inn, out, wall = (np.asarray(g[k], int) for k in ("interior", "boundary:inlet", "boundary:outlet", "boundary:wall"))
dom = np.hstack([gi, inn, out, wall])
nd = len(dom)
A = sp.csr_matrix(build_H_matrix(nodes, g, d.normals, p.theta, d.pho, d.K, d.div_K, d.eps_M, -1))
M = sp.diags(wi) @ A
Md = M[dom][:, dom].tocsr()
D = (Md - Md.T).tocsr()
a = np.asarray(abs(D).sum(1)).ravel() / np.maximum(np.asarray((abs(Md) + abs(Md.T)).sum(1)).ravel(), 1e-300)
top = nodes[dom, 1].max()
dist = top - nodes[dom, 1]
tg = np.full(nd, "i", dtype=object)
for k, idx in (("in", inn), ("out", out), ("w", wall)):
    tg[np.isin(dom, idx)] = k
mediana = lambda m: float(np.median(a[m])) if m.sum() > 5 else None  # noqa: E731
res = dict(tag=tag, N=int(N), nd=int(nd), asim_global=float(sp.linalg.norm(D) / sp.linalg.norm(Md)))
res["bandas"] = {f"{lo:g}-{hi:g}": mediana((tg == "i") & (dist > lo) & (dist <= hi))
                 for lo, hi in ((1e-6, 50), (50, 100), (100, 150), (150, 250), (250, 400), (400, 1e9))}
res["bandas"]["fila"] = mediana((tg == "w") & (dist < 1e-6))
tsup = np.sort(nodes[wall][nodes[wall, 1] > top - 1e-6][:, 0])
res["top_h_media"], res["top_h_cv"] = float(np.diff(tsup).mean()), float(np.diff(tsup).std() / np.diff(tsup).mean())

lu = spla.splu(sp.csc_matrix(A))
Wd = np.zeros(N)
Wd[dom] = wi[dom]
nn = cKDTree(nodes[dom]).query(nodes[dom], k=7)[1][:, 1:]
zd = nodes[dom, 1]
topn = wall[nodes[wall, 1] > top - 1e-6]
for nom, (x0, z0, s) in (("prof", (3700, -700, 150)), ("med", (3700, -300, 250)), ("sup", (3700, 100, 250))):
    bf = np.zeros(N)
    r2 = (nodes[:, 0] - x0) ** 2 + (nodes[:, 1] - z0) ** 2
    bf[dom] = np.exp(-r2[dom] / (2 * s * s))
    x = lu.solve(bf)
    y = lu.solve(Wd * bf, trans="T")
    yw = np.zeros(N)
    yw[dom] = y[dom] / wi[dom]
    sc = np.sqrt(np.mean(x[dom] ** 2))
    res["err_" + nom] = float(np.sqrt(np.mean((yw[dom] - x[dom]) ** 2)) / sc)
    res["err_tope_" + nom] = float(np.sqrt(np.mean((yw[topn] - x[topn]) ** 2)) / sc)
    if nom == "sup":
        r = yw[dom] - yw[dom][nn].mean(1)
        for k, m in (("sup", zd > top - 20), ("cerca", (zd > top - 100) & (zd <= top - 20)), ("prof", zd < -100)):
            res["rug_" + k] = float(np.sqrt(np.mean(r[m] ** 2)) / np.std(yw[dom][m]))
        for k, m in (("fila", zd > top - 1e-6), ("0-100m", zd > top - 100),
                     ("100-250m", (zd <= top - 100) & (zd > top - 250)), ("prof", zd < -100)):
            res["rugabs_" + k] = float(np.sqrt(np.mean(r[m] ** 2)) / sc)
print("RESULT", json.dumps(res), flush=True)
