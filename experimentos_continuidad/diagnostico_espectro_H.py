"""Radio espectral de la matriz de avance del flujo, G = A_left⁻¹·A_right (sin forzamientos).

    python experimentos_continuidad/diagnostico_espectro_H.py ID [clave=valor ...]

ρ(G) > 1 ⇒ el esquema amplifica algún modo en cada paso (inestabilidad del operador, no del
forzamiento). Imprime los mayores |λ| y su parte real/imag. No toca resultados de corridas.
"""
import os, sys, json
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)


def main(argv):
    ident, *over = argv
    cfg = json.load(open(os.path.join(CODIGO, "experimentos", "base.json")))
    cfg.update(metodo="bfr", exp_dir=os.path.join("experimentos_continuidad", "resultados", ident, "bfr_H"),
               animate=False, save_dat=False)
    for kv in over:
        k, v = kv.split("=", 1); cfg[k] = json.loads(v)
    os.chdir(CODIGO); sys.path.insert(0, CODIGO)
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    RUNTIME.params = Parameters(config=cfg)
    from preprocessing.preprocess import load_data
    from scipy.sparse.linalg import LinearOperator, eigs
    d = load_data(); N = len(d.nodes)
    G = LinearOperator((N, N), dtype=float, matvec=lambda x: d.A_left.solve(np.asarray(d.A_right @ x).ravel()))
    w = eigs(G, k=12, which="LM", tol=1e-4, return_eigenvectors=False, maxiter=5000)
    w = w[np.argsort(-np.abs(w))]
    print(f"[{ident}] ρ(G)={np.abs(w[0]):.6g}")
    for x in w[:12]:
        print(f"   λ = {x.real:+.6g} {x.imag:+.3g}i   |λ|={abs(x):.6g}")
    np.save(os.path.join(os.environ.get("DIAG_OUT", "/tmp"), f"espectro_{ident}.npy"), w)
    pkl = os.path.join(CODIGO, cfg["exp_dir"], "preproceso.pkl")
    if os.path.exists(pkl):
        os.remove(pkl)


if __name__ == "__main__":
    main(sys.argv[1:])
