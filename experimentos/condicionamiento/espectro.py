"""
Espectro y no normalidad de los operadores de avance de BFR (los mismos que usa el simulador y su adjunto exacto).

Para una corrida ya hecha (su config.json y simulation_results.npz):
  * flujo:      G_h = A_left⁻¹ A_right                          (preproceso, esquema θ del flujo)
  * transporte: G_C = A⁻¹ B en el paso k (adjunto_exacto._coefs + OperadoresTransporte, como step_time)
De cada G: los autovalores de mayor módulo (ρ(G) > 1 ⇒ inestable), la norma ‖G‖₂ (valor singular máximo; ‖G‖ ≫ ρ ⇒
operador muy no normal: crecimiento transitorio y adjunto sensible) y cond₁ estimado de la matriz a factorizar.
El adjunto resuelve con Aᵀ: mismo espectro y misma norma que G.

    python -m experimentos.condicionamiento.espectro <dir_corrida_bfr> <salida.json> [paso=72]     # bfr_env
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, onenormest, splu, svds

CODIGO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODIGO)


def _avance(A, B):
    """LinearOperator de G = A⁻¹B (y Gᵀ = Bᵀ A⁻ᵀ) con una sola factorización."""
    lu = splu(A.tocsc())
    n = A.shape[0]
    return LinearOperator((n, n), matvec=lambda v: lu.solve(B @ v), rmatvec=lambda v: B.T @ lu.solve(v, trans="T"),
                          dtype=float), lu


def _cond1(A, lu):
    n = A.shape[0]
    Ainv = LinearOperator((n, n), matvec=lambda v: lu.solve(v), rmatvec=lambda v: lu.solve(v, trans="T"), dtype=float)
    return float(onenormest(A.tocsc()) * onenormest(Ainv))


def analizar(nombre, A, B, nodos, pozo, k_eig=6):
    G, lu = _avance(A, B)
    from scipy.sparse.linalg import ArpackNoConvergence
    try:
        val, vec = eigs(G, k=k_eig, which="LM", tol=1e-8, maxiter=5000, ncv=max(4 * k_eig, 40))
    except ArpackNoConvergence as e:      # autovalores agrupados cerca de 1: se usan los que convergieron
        val, vec = e.eigenvalues, e.eigenvectors
        print(f"    (ARPACK: {len(val)}/{k_eig} autovalores convergidos)")
    orden = np.argsort(-np.abs(val))
    val, vec = val[orden], vec[:, orden]
    sv = svds(G, k=1, which="LM", return_singular_vectors=False, tol=1e-6)[0]
    rho = float(np.abs(val[0]))
    # dónde vive el modo dominante: distancia al pozo del nodo con mayor |v|
    loc = [float(np.linalg.norm(nodos[np.argmax(np.abs(vec[:, j]))] - pozo)) for j in range(len(val))]
    out = dict(rho=rho, norma2=float(sv), no_normalidad=float(sv / rho), cond1=_cond1(A, lu),
               autovalores=[[float(v.real), float(v.imag)] for v in val], dist_pozo_modo=loc)
    print(f"  {nombre}: ρ(G)={rho:.6f}  ‖G‖₂={sv:.3e}  ‖G‖/ρ={sv / rho:.2e}  cond₁(A)={out['cond1']:.2e}  "
          f"modo dominante a {loc[0]:.0f} m del pozo")
    return out


def main(dir_run, salida, paso=72):
    from config.parameters import Parameters
    from funtions.runtime import RUNTIME
    cfg = json.load(open(os.path.join(dir_run, "config.json")))
    cfg["exp_dir"] = os.path.join(os.path.dirname(salida), "pre_" + os.path.basename(os.path.dirname(dir_run.rstrip("/"))))
    cfg["run_type"] = "comparison"
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    from preprocessing.preprocess import load_data
    d = load_data()
    raw = pickle.load(open(os.path.join(p.save_preprocess, "preproceso.pkl"), "rb"))
    nodos = np.asarray(d.nodes, float)
    pozo = np.asarray(d.pozo_cor, float)
    res = {"corrida": dir_run, "N": int(len(nodos)), "paso": paso}
    print(f"{dir_run}: N={len(nodos)}")
    res["flujo"] = analizar("flujo     ", raw["A_left"], raw["A_right"], nodos, pozo)

    from funtions.adjunto_exacto import OperadoresTransporte, D_total_, _coefs
    z = np.load(os.path.join(dir_run, "simulation_results.npz"))
    H, U = z["H"][:, 0], z["U"][:, 0]
    ops = OperadoresTransporte(d)
    Un, Uo = U[paso], U[paso - 1]
    Dn, Do = D_total_(d, p, Un), D_total_(d, p, Uo)
    h_t = (H[paso] - H[paso - 1]) / p.dt
    cA, cB = _coefs(d, p, d.pho, Un, Dn, Do, h_t)
    A = ops.matriz(cA, Dn, True)
    B = ops.matriz(cB, Do, False)
    res["transporte"] = analizar("transporte", A, B, nodos, pozo)
    json.dump(res, open(salida, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 72)
