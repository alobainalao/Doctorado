"""
Optimización adjunta (FEniCSx) del modelo sin W- y con almacenamiento, para ADR / MRMT semi / MRMT bloque.

    python -m experimentos.optimizar_modelo_nuevo <adr|mrmt_semi|mrmt_block> [--nt N] [--iter K] [--modulada]

Corre `optimize_fenicsx` (scipy trust-constr sobre x = [Q(t), z_p] con el gradiente adjunto de
docs/reanalisis_adr_mrmt.tex) en el dominio real con horizonte corto. La CI de la pluma de funcion.h5 no
llega al pozo en horizontes cortos (J_m = 0), así que, como en tests/test_gradiente_modelo_nuevo.py,
se usa C uniforme (o modulada con --modulada) para que J_m ≠ 0 y el adjunto del transporte trabaje.

Entorno: fenics_env. Salida: data/output/optimizacion_modelo_nuevo/<modelo>.json
"""
import json
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters

MRMT = {"Nr": 2, "Deff": [1e-9, 5e-10], "beta": [0.15, 0.10], "phi_im": [0.1, 0.05]}


def _arg(prefijo, defecto):
    for a in sys.argv:
        if a.startswith(prefijo):
            return type(defecto)(a[len(prefijo):])
    return defecto


def main(modelo):
    nt = _arg("--nt=", 6)
    maxiter = _arg("--iter=", 8)
    env = {"metodo": "fenicsx", "run_type": "optimization", "domain": "real", "spacing": 120,
           "dt": 36000, "T": nt * 36000, "save_dat": False, "animate": False, "postproc": False,
           "pre": True, "model": modelo, "extract_C": False, "factor_sumidero_C": 0.0,
           "factor_almacenamiento_C": 1.0, "opt_maxiter": maxiter}
    if modelo.startswith("mrmt"):
        env.update(MRMT)
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()
    p.C_init_override = 1.0
    if "--modulada" in sys.argv:
        p.C_init_modulacion = (0.5, 2000.0, 800.0)
    p.gamma = _arg("--gamma=", 1.0)
    p.koppa = _arg("--koppa=", 0.0)
    p.factor_almacenamiento_C = 1.0
    p.factor_sumidero_C = 0.0
    p.extract_C = False

    from fenicsx.mesh import get_mesh
    from fenicsx.adjoint import build_context, functional_fenicsx, optimize_fenicsx

    mesh_cache = get_mesh(list(p.pozo), p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mesh_cache)
    x0 = np.concatenate([np.asarray(p.Qout[0], float), [float(p.pozo[1])]])
    J0 = functional_fenicsx(x0, ctx)

    res = optimize_fenicsx(p, mesh_cache=mesh_cache)
    out = {"modelo": modelo, "nt": nt, "J0": J0, "x0": x0.tolist(),
           "x_opt": np.asarray(res.x).tolist(), "J_escalada_opt": float(res.fun),
           "nit": int(res.nit), "success": bool(res.success), "message": str(res.message)}
    d = os.path.join(CODIGO, "data", "output", "optimizacion_modelo_nuevo")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{modelo}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("[opt] guardado", os.path.join(d, f"{modelo}.json"))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")][0])
