"""Resuelve SOLO el flujo H en BFR (sin transporte) con la misma malla/config que las corridas.

    python experimentos_continuidad/resolver_H.py ID [clave=valor ...]

Salida: resultados/<ID>/bfr_H/H.npz (H[t, nodo], dt, nodes) + config.json. Replica el
paso de flujo de funtions/step_time.py (H_vector + A_left.solve) con la misma indexación
de Qout que view/animation.py. El flujo no depende de C, así que vale para toda forma de transporte.
"""
import os, sys, json, subprocess, datetime
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CODIGO = os.path.dirname(AQUI)


def main(argv):
    ident, *over = argv
    cfg = json.load(open(os.path.join(CODIGO, "experimentos", "base.json")))
    out = os.path.join(AQUI, "resultados", ident, "bfr_H")
    cfg.update(metodo="bfr", exp_dir=os.path.relpath(out, CODIGO), animate=False, save_dat=False)
    for kv in over:
        k, v = kv.split("=", 1); cfg[k] = json.loads(v)
    os.makedirs(out, exist_ok=True)
    cfg["_etiqueta"] = dict(id=ident, backend="bfr", solo="H",
                            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODIGO, text=True).strip(),
                            fecha=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2, ensure_ascii=False)

    os.chdir(CODIGO); sys.path.insert(0, CODIGO)
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    from preprocessing.preprocess import load_data
    from funtions.operators import H_vector
    d = load_data()
    N = len(d.nodes)
    Q = p.Qout[0]
    H = d.H[0].copy()
    hist = np.zeros((p.Nt, N))
    for n in range(p.Nt):
        rhs = H_vector(H, d.A_right, d.delta_p, d.groups, d.nodes, N, None, Q[n - 1], Q[n], gauss_p=d.gauss_p)
        H = d.A_left.solve(rhs)
        hist[n] = H
    np.savez(os.path.join(out, "H.npz"), H=hist, dt=p.dt, nodes=d.nodes)
    pkl = os.path.join(out, "preproceso.pkl")
    if os.path.exists(pkl):
        os.remove(pkl)
    print("H listo:", hist.shape, "rango", hist.min(), hist.max())


if __name__ == "__main__":
    main(sys.argv[1:])
