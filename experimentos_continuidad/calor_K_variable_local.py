"""∇·(K∇u) con K variable: directa/desarrollada vs global Gx·K·Gx vs composición local en forma de flujo.

    python experimentos_continuidad/calor_K_variable_local.py

Local, forma de flujo (extiende calor_composicion_local.py):  fila i = Σ_{j∈N_i} c_ij · K_j · A^(i)_j
  (K en los nodos del vecindario, donde se evalúa el flujo K∂u; c y A como allá). Sin ∇K explícito.
Pruebas: (1) solución fabricada con K suave de contraste 10, 1e3, 1e6; (2) autovalores positivos con el K real
(S = φ, fantasmas = 0, Δt = 18000 s, mismos desplazamientos que espectro_L_solo.py).
"""
import os, sys, json
import numpy as np, sympy as sp
from scipy.sparse import diags, coo_matrix
from scipy.sparse.linalg import eigs
from scipy.spatial import cKDTree
import rbf.pde.fd as fd

AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)
os.chdir(CODIGO); sys.path.insert(0, CODIGO)
cfg = json.load(open("experimentos/base.json"))
cfg.update(metodo="bfr", exp_dir="experimentos_continuidad/resultados/_tmp_calorK/bfr", animate=False, save_dat=False)
from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(config=cfg); p = RUNTIME.get()
from funtions.operators import rbf_op
from preprocessing.preprocess import load_data

d = load_data(); nodes = d.nodes; N = len(nodes); g = d.groups
eps_node = np.asarray(d.eps_M)[:, 0]
Gx, Gz = d.grad
D2x = rbf_op([2, 0], nodes, nodes, d.eps_M); D2z = rbf_op([0, 2], nodes, nodes, d.eps_M)
n, m = p.n_stencil, 12
idx = cKDTree(nodes).query(nodes, k=n)[1]


def pesos_locales(diff, chunk=1500):
    """c (N,m): ∂ en el centro con N_i;  A (N,m,n): ∂ en cada j∈N_i con P_i."""
    C = np.zeros((N, m)); A = np.zeros((N, m, n))
    for a in range(0, N, chunk):
        b = min(N, a + chunk); B = b - a
        P = nodes[idx[a:b]]; Nb = nodes[idx[a:b, :m]]; ep = eps_node[a:b]
        C[a:b] = fd.weights(nodes[a:b], Nb, diff, phi=p.phi, order=p.order, eps=np.repeat(ep[:, None], m, axis=1))
        A[a:b] = fd.weights(Nb.reshape(B * m, 2), np.repeat(P, m, axis=0), diff, phi=p.phi, order=p.order,
                            eps=np.repeat(np.repeat(ep, m)[:, None], n, axis=1)).reshape(B, m, n)
    return C, A


Cx, Ax = pesos_locales([1, 0]); Cz, Az = pesos_locales([0, 1])
rows = np.repeat(np.arange(N), n); cols = idx.ravel()


def L_local(Kx, Kz):
    Wx = np.einsum("bj,bjk->bk", Cx * Kx[idx[:, :m]], Ax)
    Wz = np.einsum("bj,bjk->bk", Cz * Kz[idx[:, :m]], Az)
    return coo_matrix(((Wx + Wz).ravel(), (rows, cols)), shape=(N, N)).tocsr()


def L_global(Kx, Kz): return Gx @ diags(Kx) @ Gx + Gz @ diags(Kz) @ Gz
def L_desarr(Kx, Kz): return diags(Kx) @ D2x + diags(Kz) @ D2z + diags(Gx @ Kx) @ Gx + diags(Gz @ Kz) @ Gz
OPS = [("desarrollada", L_desarr), ("global", L_global), ("local", L_local)]

x, z = nodes.T; hloc = cKDTree(nodes).query(nodes, k=2)[0][:, 1]
inter = np.zeros(N, bool); inter[g['interior']] = True
zonas = {"fina h≤8": hloc <= 8, "media 8<h≤25": (hloc > 8) & (hloc <= 25), "gruesa h>25": hloc > 25}
X, Z = sp.symbols("x z"); h_s = sp.sin(2 * sp.pi * X / 3000) * sp.cos(2 * sp.pi * Z / 1500)
for contraste in [10.0, 1e3, 1e6]:
    a = sp.log(contraste)
    Kx_s = sp.exp(a / 2 * sp.sin(2 * sp.pi * X / 4000) * sp.cos(2 * sp.pi * Z / 2500)); Kz_s = Kx_s / 3
    ex_s = sp.diff(Kx_s * sp.diff(h_s, X), X) + sp.diff(Kz_s * sp.diff(h_s, Z), Z)
    f = lambda e: sp.lambdify((X, Z), e, "numpy")(x, z) * np.ones(N)
    h, Kx_v, Kz_v, ex = f(h_s), f(Kx_s), f(Kz_s), f(ex_s)
    print(f"\n[solución fabricada] contraste de K ≈ {Kx_v.max() / Kx_v.min():.3g};  rms(error)/rms(exacto)")
    print(f"  {'zona':14s} {'desarrollada':>13s} {'global':>9s} {'local':>9s}")
    Ls = [fL(Kx_v, Kz_v) for _, fL in OPS]
    for zn, mk0 in zonas.items():
        mk = inter & mk0
        e = [np.sqrt(np.mean(((L @ h) - ex)[mk] ** 2)) / np.sqrt(np.mean(ex[mk] ** 2)) for L in Ls]
        print(f"  {zn:14s} {e[0]:13.3g} {e[1]:9.3g} {e[2]:9.3g}")

dt = p.dt; S = np.asarray(d.pho)
dom = np.hstack([g['interior'], g['boundary:inlet'], g['boundary:outlet'], g['boundary:wall']])
print(f"\n[espectro de S⁻¹L, K REAL, Δt={dt}]  z = Δt·λ ; z>0 ⇒ crece")
for nm, fL in OPS:
    M = (diags(1.0 / S[dom]) @ fL(d.K[:, 0], d.K[:, 1]).tocsr()[dom][:, dom]).tocsc()
    zs = []
    for z0 in (0.05, 0.3, 0.9, 1.5):
        try:
            w = eigs(M, k=6, sigma=z0 / dt, which="LM", tol=1e-6, maxiter=5000, return_eigenvectors=False)
        except Exception as ex_:
            w = getattr(ex_, "eigenvalues", np.array([]))
        zs += list(dt * w.real[np.abs(dt * w.imag) < 1e-3 + 1e-3 * np.abs(dt * w.real)])
    zs = np.array(sorted(set(np.round(zs, 5)))); pos = zs[zs > 0]
    print(f"  {nm:12s} reales halladas {len(zs):2d}; z>0: {len(pos):2d} {str(np.round(pos[-4:], 4).tolist()) if len(pos) else '(ninguno)'}   z_max = {zs.max() if len(zs) else float('nan'):+.4g}")
import shutil; shutil.rmtree("experimentos_continuidad/resultados/_tmp_calorK", ignore_errors=True)
