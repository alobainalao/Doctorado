"""Prototipo (b): flujo en los puntos medios de las aristas, divergencia de vuelta a los nodos.

    L_esc = Gmn_x · diag(Kx_m) · Gnm_x + Gmn_z · diag(Kz_m) · Gnm_z
  Gnm: derivada RBF-FD evaluada en los puntos medios con fuentes = nodos
  Gmn: derivada RBF-FD evaluada en los nodos con fuentes = puntos medios   (operadores entre mallas)
  Kx_m, Kz_m: media armónica de los extremos de la arista.
Prueba: (1) solución fabricada (verdad analítica) y (2) autovalores positivos de L sola.
"""
import os, sys, json
import numpy as np, sympy as sp
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)
os.chdir(CODIGO); sys.path.insert(0, CODIGO)
cfg = json.load(open("experimentos/base.json"))
cfg.update(metodo="bfr", exp_dir="experimentos_continuidad/resultados/_tmp_esc/bfr", animate=False, save_dat=False)
from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(config=cfg); p = RUNTIME.get()
from funtions.operators import rbf_op
from preprocessing.preprocess import load_data
import pickle

d = load_data(); nodes = d.nodes; N = len(nodes); eps_M = d.eps_M; g = d.groups
Gx, Gz = d.grad
D2x = rbf_op([2, 0], nodes, nodes, eps_M); D2z = rbf_op([0, 2], nodes, nodes, eps_M)
tris = np.asarray(pickle.load(open(cfg["mesh_pkl"], "rb"))["triangles"])
ed = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]); ed.sort(axis=1)
ed = np.unique(ed, axis=0); Nm = len(ed)
mid = 0.5 * (nodes[ed[:, 0]] + nodes[ed[:, 1]])
eps_mid = 0.5 * (eps_M[ed[:, 0]] + eps_M[ed[:, 1]])
Gnm_x = rbf_op([1, 0], mid, nodes, eps_mid); Gnm_z = rbf_op([0, 1], mid, nodes, eps_mid)
Gmn_x = rbf_op([1, 0], nodes, mid, eps_M); Gmn_z = rbf_op([0, 1], nodes, mid, eps_M)
print(f"puntos medios: {Nm}  (nodos {N})")


def harm(a, b): return 2 * a * b / (a + b)


def L_esc(Kx_v, Kz_v):
    Kxm = harm(Kx_v[ed[:, 0]], Kx_v[ed[:, 1]]); Kzm = harm(Kz_v[ed[:, 0]], Kz_v[ed[:, 1]])
    return Gmn_x @ diags(Kxm) @ Gnm_x + Gmn_z @ diags(Kzm) @ Gnm_z


def L_con(Kx_v, Kz_v): return Gx @ diags(Kx_v) @ Gx + Gz @ diags(Kz_v) @ Gz
def L_des(Kx_v, Kz_v): return diags(Kx_v) @ D2x + diags(Kz_v) @ D2z + diags(Gx @ Kx_v) @ Gx + diags(Gz @ Kz_v) @ Gz


# ---------- (1) solución fabricada
x, z = nodes.T; hloc = cKDTree(nodes).query(nodes, k=2)[0][:, 1]
inter = np.zeros(N, bool); inter[g['interior']] = True
zonas = {"fina h≤8": hloc <= 8, "media 8<h≤25": (hloc > 8) & (hloc <= 25), "gruesa h>25": hloc > 25}
X, Z = sp.symbols("x z"); h_s = sp.sin(2 * sp.pi * X / 3000) * sp.cos(2 * sp.pi * Z / 1500)
for contraste in [10.0, 1e6]:
    a = sp.log(contraste)
    Kx_s = sp.exp(a / 2 * sp.sin(2 * sp.pi * X / 4000) * sp.cos(2 * sp.pi * Z / 2500)); Kz_s = Kx_s / 3
    ex_s = sp.diff(Kx_s * sp.diff(h_s, X), X) + sp.diff(Kz_s * sp.diff(h_s, Z), Z)
    f = lambda e: sp.lambdify((X, Z), e, "numpy")(x, z) * np.ones(N)
    h, Kx_v, Kz_v, ex = f(h_s), f(Kx_s), f(Kz_s), f(ex_s)
    print(f"\n[solución fabricada] contraste K ≈ {Kx_v.max() / Kx_v.min():.3g}   error rms / rms exacto")
    print(f"  {'zona':14s} {'desarrollada':>13s} {'concatenada':>12s} {'escalonada':>11s}")
    Ls = {"des": L_des(Kx_v, Kz_v), "con": L_con(Kx_v, Kz_v), "esc": L_esc(Kx_v, Kz_v)}
    for zn, m in zonas.items():
        mk = inter & m
        e = {k: np.sqrt(np.mean(((L @ h) - ex)[mk] ** 2)) / np.sqrt(np.mean(ex[mk] ** 2)) for k, L in Ls.items()}
        print(f"  {zn:14s} {e['des']:13.3g} {e['con']:12.3g} {e['esc']:11.3g}")

# ---------- (2) autovalores positivos de L sola (fantasmas = 0)
dt = p.dt; S = np.asarray(d.pho)
dom = np.hstack([g['interior'], g['boundary:inlet'], g['boundary:outlet'], g['boundary:wall']])
Kreal = d.K.copy(); Kcte = np.tile(np.median(d.K, axis=0), (N, 1))
print(f"\n[espectro de S⁻¹L, Δt={dt}] z = Δt·λ ; z>0 crece")
for kn, K in [("K real", Kreal), ("K constante", Kcte)]:
    for on, fL in [("desarrollada", L_des), ("concatenada", L_con), ("escalonada", L_esc)]:
        M = (diags(1.0 / S[dom]) @ fL(K[:, 0], K[:, 1]).tocsr()[dom][:, dom]).tocsc()
        zs = []
        for z0 in (0.05, 0.3, 0.9, 1.5):
            try:
                w = eigs(M, k=6, sigma=z0 / dt, which="LM", tol=1e-6, maxiter=5000, return_eigenvectors=False)
            except Exception as e:
                w = getattr(e, "eigenvalues", np.array([]))
            zs += list(dt * w.real[np.abs(dt * w.imag) < 1e-3 + 1e-3 * np.abs(dt * w.real)])
        zs = np.array(sorted(set(np.round(zs, 5)))); pos = zs[zs > 0]
        print(f"{kn:12s} {on:13s} reales halladas {len(zs):2d}; z>0: {len(pos):2d} "
              f"{str(np.round(pos[-3:], 4).tolist()) if len(pos) else '(ninguno)'}   z_max={zs.max() if len(zs) else float('nan'):+.4g}")
import shutil; shutil.rmtree("experimentos_continuidad/resultados/_tmp_esc", ignore_errors=True)
