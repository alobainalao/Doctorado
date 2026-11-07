"""Ecuación del calor  u_t = ∇·(K∇u)  con K constante: tres construcciones del Laplaciano RBF-FD.

    python experimentos_continuidad/calor_composicion_local.py

  directa      K·(D2x + D2z)                                (stencil de 2.º orden, lo del código)
  global G·G   K·(Gx·Gx + Gz·Gz)                            (producto de matrices completas)
  local        composición dentro del stencil (Shankar-Wright-Narayan 2020, ec. 10):
               fila i = Σ_{j∈N_i} c_ij · A^(i)_j,  A^(i)_j = pesos de ∂ en el nodo j usando SOLO los n nodos de P_i,
               c_ij = pesos de ∂ en el nodo i usando los m nodos más cercanos N_i.
Mide (1) exactitud en un campo suave (verdad analítica) y (2) autovalores positivos de S⁻¹L con ceros en los
fantasmas, igual que espectro_L_solo.py (misma malla, mismos Δt, S, desplazamientos) para poder comparar.
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
cfg.update(metodo="bfr", exp_dir="experimentos_continuidad/resultados/_tmp_calor/bfr", animate=False, save_dat=False)
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
idx = cKDTree(nodes).query(nodes, k=n)[1]                      # (N,n), primera columna = el propio nodo


def local_composition(diff, chunk=1500):
    """Matriz N×N: ∂_diff ∘ ∂_diff compuesta dentro de cada stencil."""
    rows, cols, vals = [], [], []
    e = np.zeros(2); 
    for a in range(0, N, chunk):
        b = min(N, a + chunk); B = b - a
        P = nodes[idx[a:b]]                                     # (B,n,2) stencil completo
        Nb = nodes[idx[a:b, :m]]                                # (B,m,2) vecindario pequeño
        ep = eps_node[a:b]
        # c_ij: pesos de ∂ en el centro i con los m nodos de N_i
        c = fd.weights(nodes[a:b], Nb, diff, phi=p.phi, order=p.order, eps=np.repeat(ep[:, None], m, axis=1))
        # A^(i)_j: pesos de ∂ en cada nodo j de N_i con los n nodos de P_i
        xj = Nb.reshape(B * m, 2)
        Pj = np.repeat(P, m, axis=0)                            # (B*m,n,2)
        epj = np.repeat(ep, m)
        A = fd.weights(xj, Pj, diff, phi=p.phi, order=p.order, eps=np.repeat(epj[:, None], n, axis=1)).reshape(B, m, n)
        W = np.einsum("bj,bjk->bk", c, A)                       # (B,n)
        rows.append(np.repeat(np.arange(a, b), n)); cols.append(idx[a:b].ravel()); vals.append(W.ravel())
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()


Lx_loc = local_composition([1, 0]); Lz_loc = local_composition([0, 1])
L_loc = Lx_loc + Lz_loc
L_dir = D2x + D2z
L_glo = Gx @ Gx + Gz @ Gz

# autoprueba: reproducción exacta de polinomios de grado 2 (∇² de x²+z² = 4)
x, z = nodes.T
f = x ** 2 + z ** 2
inter = np.zeros(N, bool); inter[g['interior']] = True
for nm, L in [("directa", L_dir), ("global G·G", L_glo), ("local", L_loc)]:
    print(f"autoprueba ∇²(x²+z²)=4:  {nm:11s} error máx relativo = {np.abs((L @ f)[inter] - 4).max() / 4:.2e}")

# (1) exactitud en campo suave con ∇²h exacto
X, Z = sp.symbols("x z"); h_s = sp.sin(2 * sp.pi * X / 3000) * sp.cos(2 * sp.pi * Z / 1500)
ex_s = sp.diff(h_s, X, 2) + sp.diff(h_s, Z, 2)
h = sp.lambdify((X, Z), h_s, "numpy")(x, z); ex = sp.lambdify((X, Z), ex_s, "numpy")(x, z)
hloc = cKDTree(nodes).query(nodes, k=2)[0][:, 1]
zonas = {"fina h≤8": hloc <= 8, "media 8<h≤25": (hloc > 8) & (hloc <= 25), "gruesa h>25": hloc > 25}
print(f"\n[exactitud, ∇²h con h=sin·cos]  rms(error)/rms(exacto)\n  {'zona':14s} {'directa':>9s} {'global G·G':>11s} {'local':>9s}")
for zn, mk0 in zonas.items():
    mk = inter & mk0
    e = [np.sqrt(np.mean(((L @ h) - ex)[mk] ** 2)) / np.sqrt(np.mean(ex[mk] ** 2)) for L in (L_dir, L_glo, L_loc)]
    print(f"  {zn:14s} {e[0]:9.3g} {e[1]:11.3g} {e[2]:9.3g}")

# (2) autovalores positivos de S⁻¹·K·L (K constante, fantasmas = 0)
dt = p.dt; S = np.asarray(d.pho); Kc = float(np.median(d.K[:, 0]))
dom = np.hstack([g['interior'], g['boundary:inlet'], g['boundary:outlet'], g['boundary:wall']])
print(f"\n[espectro de S⁻¹·K·L, K = mediana = {Kc:.4g}, Δt = {dt}]   z = Δt·λ ;  z>0 ⇒ crece (polo de θ=1 en z=1)")
for nm, L in [("directa", L_dir), ("global G·G", L_glo), ("local", L_loc)]:
    M = (diags(Kc / S[dom]) @ L.tocsr()[dom][:, dom]).tocsc()
    zs = []
    for z0 in (0.05, 0.3, 0.9, 1.5):
        try:
            w = eigs(M, k=6, sigma=z0 / dt, which="LM", tol=1e-6, maxiter=5000, return_eigenvectors=False)
        except Exception as ex_:
            w = getattr(ex_, "eigenvalues", np.array([]))
        zs += list(dt * w.real[np.abs(dt * w.imag) < 1e-3 + 1e-3 * np.abs(dt * w.real)])
    zs = np.array(sorted(set(np.round(zs, 5)))); pos = zs[zs > 0]
    print(f"  {nm:11s} reales halladas {len(zs):2d}; z>0: {len(pos):2d} {str(np.round(pos[-3:], 4).tolist()) if len(pos) else '(ninguno)'}   z_max = {zs.max() if len(zs) else float('nan'):+.4g}")
import shutil; shutil.rmtree("experimentos_continuidad/resultados/_tmp_calor", ignore_errors=True)
