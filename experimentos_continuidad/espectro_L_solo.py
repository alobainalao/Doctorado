"""Espectro de L sola (sin esquema temporal ni filas de frontera): ¿tiene autovalores positivos?

    python experimentos_continuidad/espectro_L_solo.py

Resuelve  S⁻¹·L_dd  con L_dd = L[dominio, dominio] (valores de los fantasmas = 0) y reporta
z_max = Δt·máx Re(λ)  (>0 ⇒ modo que crece). Variantes: L desarrollada / concatenada × K real / K constante.
"""
import os, sys, json
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)
os.chdir(CODIGO); sys.path.insert(0, CODIGO)
cfg = json.load(open("experimentos/base.json"))
cfg.update(metodo="bfr", exp_dir="experimentos_continuidad/resultados/_tmp_esp/bfr", animate=False, save_dat=False)
from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(config=cfg); p = RUNTIME.get()
from funtions.operators import rbf_op
from preprocessing.preprocess import load_data

d = load_data(); nodes = d.nodes; N = len(nodes); eps_M = d.eps_M
Gx, Gz = d.grad
D2x = rbf_op([2, 0], nodes, nodes, eps_M); D2z = rbf_op([0, 2], nodes, nodes, eps_M)
g = d.groups
dom = np.hstack([g['interior'], g['boundary:inlet'], g['boundary:outlet'], g['boundary:wall']])
S = np.asarray(d.pho)                                    # Ss = φ
dt = p.dt


def operadores(K):
    Kx, Kz = diags(K[:, 0]), diags(K[:, 1])
    dKx, dKz = Gx @ K[:, 0], Gz @ K[:, 1]
    des = Kx @ D2x + Kz @ D2z + diags(dKx) @ Gx + diags(dKz) @ Gz
    con = Gx @ Kx @ Gx + Gz @ Kz @ Gz
    return {"desarrollada": des, "concatenada": con}


Kreal = d.K.copy(); Kcte = np.tile(np.median(d.K, axis=0), (N, 1))
print(f"Δt = {dt} s;  z = Δt·λ(S⁻¹L)  (z>0 ⇒ crece; polo del esquema θ=1 en z=1)")
for kn, K in [("K real", Kreal), ("K constante", Kcte)]:
    for on, L in operadores(K).items():
        M = (diags(1.0 / S[dom]) @ L.tocsr()[dom][:, dom]).tocsc()
        # inversión con desplazamiento en z = 0.05, 0.3, 0.9, 1.5 (λ = z/Δt): autovalores de L cercanos a esos puntos
        zs = []
        for z0 in (0.05, 0.3, 0.9, 1.5):
            try:
                w = eigs(M, k=6, sigma=z0 / dt, which="LM", tol=1e-6, maxiter=5000, return_eigenvectors=False)
            except Exception as e:                      # ArpackNoConvergence trae los que sí convergieron
                w = getattr(e, "eigenvalues", np.array([]))
            zs += list(dt * w.real[np.abs(dt * w.imag) < 1e-3 + 1e-3 * np.abs(dt * w.real)])
        zs = np.array(sorted(set(np.round(zs, 5))))
        pos = zs[zs > 0]
        print(f"{kn:12s} {on:13s} autovalores reales hallados: {len(zs)};  con z>0: {len(pos)}  "
              f"{'z_pos = ' + str(np.round(pos[-4:], 4).tolist()) if len(pos) else '(ninguno)'};  z_max = {zs.max() if len(zs) else float('nan'):+.4g}")
import shutil; shutil.rmtree("experimentos_continuidad/resultados/_tmp_esp", ignore_errors=True)
