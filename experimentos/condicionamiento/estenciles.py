"""
Línea de investigación: condicionamiento de los estenciles RBF-FD de BFR como posible causa de las oscilaciones.

Para cada nodo del dominio, con la misma construcción que el simulador (MQ, ε_i = 1/h_i con h_i = distancia media a
los 8 vecinos, n_stencil vecinos incluidos los fantasmas, polinomios de grado `order`):
  * cond₂ del sistema local aumentado [Φ P; Pᵀ 0] (coordenadas centradas y escaladas por h_i);
  * pesos del laplaciano L = ∂xx + ∂zz (rbf.pde.fd.weight_matrix): signo del peso central (debe ser < 0),
    ‖w‖₁·h_i² (≈ 8–20 en un estencil sano) y mayor peso fuera del centro con signo positivo relativo;
  * distancia al pozo, a la fuente y a la frontera.

    python -m experimentos.condicionamiento.estenciles <mesh_pkl> <salida.npz> [n_stencil=25]     # bfr_env
"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

CODIGO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODIGO)


def _mq(r, eps):
    return np.sqrt(1.0 + (eps * r) ** 2)


def _monomios(x, orden):
    cols = [np.ones(len(x))]
    if orden >= 1:
        cols += [x[:, 0], x[:, 1]]
    if orden >= 2:
        cols += [x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2]
    return np.column_stack(cols)


def analizar(nodos_todos, dominio, n, orden=2):
    from rbf.pde.fd import weight_matrix
    arbol = cKDTree(nodos_todos)
    d8 = cKDTree(nodos_todos).query(nodos_todos, 9)[0][:, 1:].mean(1)      # = local_lengthscale(k=8)
    eps = 1.0 / d8
    _, vec = arbol.query(nodos_todos[dominio], n)
    cond = np.empty(len(dominio))
    for k, (i, idx) in enumerate(zip(dominio, vec)):
        x = (nodos_todos[idx] - nodos_todos[i]) / d8[i]                     # centrado y escalado
        r = np.linalg.norm(x[:, None] - x[None], axis=2)
        P = _monomios(x, orden)
        A = np.block([[_mq(r, eps[i] * d8[i]), P], [P.T, np.zeros((P.shape[1], P.shape[1]))]])
        cond[k] = np.linalg.cond(A)
    W = (weight_matrix(x=nodos_todos[dominio], p=nodos_todos, n=n, diffs=[[2, 0], [0, 2]], phi="mq", order=orden,
                       eps=np.repeat(eps[dominio][:, None], n, axis=1), chunk_size=None)).tocsr()   # = eps_M del simulador
    centro = np.array([W[k, i] for k, i in enumerate(dominio)])
    h2 = d8[dominio] ** 2
    norma1 = np.asarray(abs(W).sum(1)).ravel() * h2
    Wc = W.copy().tolil()
    for k, i in enumerate(dominio):
        Wc[k, i] = 0.0
    Wc = Wc.tocsr()
    fuera_pos = np.asarray(Wc.max(1).todense()).ravel() * h2                # vecino positivo mayor (normal)
    fuera_neg = -np.asarray(Wc.min(1).todense()).ravel() * h2               # vecino negativo (no M-matriz)
    return dict(cond=cond, centro=centro * h2, norma1=norma1, fuera_pos=fuera_pos, fuera_neg=fuera_neg, h=d8[dominio])


def main(pkl, salida, n=25):
    from config.parameters import Parameters
    from funtions.runtime import RUNTIME
    import json
    RUNTIME.params = Parameters(config={**json.load(open(os.path.join(CODIGO, "experimentos", "base.json"))), "metodo": "bfr"})
    m = pickle.load(open(pkl, "rb"))
    todos = np.asarray(m["nodes"], float); g = m["groups"]
    fant = np.concatenate([np.asarray(v, int) for k, v in g.items() if k.startswith("ghost")])
    dominio = np.setdiff1d(np.arange(len(todos)), fant)
    r = analizar(todos, dominio, n)
    pozo = np.asarray(m["pozo_cor"], float)
    from preprocessing.malla_compartida import _dist_borde
    vert = np.array([[816, -2690.5], [4354, -1757.5], [10755, -768.5], [10755, 250.5], [816, 250.5]])
    np.savez(salida, nodos=todos[dominio], d_pozo=np.linalg.norm(todos[dominio] - pozo, axis=1),
             d_borde=_dist_borde(todos[dominio], vert), n_stencil=n, **r)
    c = r["cond"]
    print(f"{os.path.basename(os.path.dirname(pkl))}: N={len(dominio)} n={n}  cond₂ mediana {np.median(c):.2e}  p99 {np.percentile(c, 99):.2e}  máx {c.max():.2e}")
    print(f"  laplaciano: peso central ≥ 0 en {int((r['centro'] >= 0).sum())} nodos; ‖w‖₁h² mediana {np.median(r['norma1']):.1f} p99 {np.percentile(r['norma1'], 99):.1f} máx {r['norma1'].max():.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25)
