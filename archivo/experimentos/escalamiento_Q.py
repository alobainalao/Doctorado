"""
Escalamiento de J1 y de C en el pozo con Q constante y z_p variable (ADR, caso nitrato_demanda_alta).

    python -m experimentos.escalamiento_Q

Responde cuánto Q hace falta para que el bombeo mueva C en el pozo a escala de J2. Entorno: fenics_env.
Salida: data/output/optimizacion_nitrato_demanda_alta/escalamiento_Q.json
"""
import json
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
os.chdir(CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters

DIR = os.path.join(CODIGO, "data", "output", "optimizacion_nitrato_demanda_alta")
QS = (0.0, 0.01, 0.05, 0.2, 1.0, 5.0)
ZPS = (-200.0, -300.0, -400.0, -500.0, -600.0, -700.0)


def main():
    ref = json.load(open(os.path.join(DIR, "adr.json")))
    cfg = json.load(open(os.path.join(CODIGO, "config", "nitrato_demanda_alta.json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": "adr", "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    p.C_ref = ref["C_ref_modelo"]
    from fenicsx.mesh import get_mesh
    from fenicsx.adjoint import build_context, _solve_forward
    mc = get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)
    f = ref["f"]
    nodes = ctx.Q.tabulate_dof_coordinates()[:, :2]
    out = []
    for q in QS:
        for zp in ZPS:
            try:
                r = _solve_forward(ctx, np.full(p.Nt, q), zp)
                C = np.asarray(r["C"]) * f
                m = (nodes[:, 0] - p.pozo[0]) ** 2 + (nodes[:, 1] - zp) ** 2 <= 30.0 ** 2
                reg = dict(Q=q, zp=zp, J1=float(r["J1"]), C_pozo_max=float(C[:, m].max()),
                           C_pozo_fin=float(C[-1, m].max()), C_glob_max=float(C.max()))
            except Exception as e:                      # Q grande puede romper el directo
                reg = dict(Q=q, zp=zp, error=str(e)[:120])
            out.append(reg)
            print(reg, flush=True)
            json.dump(out, open(os.path.join(DIR, "escalamiento_Q.json"), "w"), indent=1)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
