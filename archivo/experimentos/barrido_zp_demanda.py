"""
Barrido de J1/J2/J3 en z_p con Q = Q_opt fijo (ADR; VARIANTE=<config> en el entorno, p.ej. nitrato_vscale_malla).

    python -m experimentos.barrido_zp_demanda

Entorno: fenics_env. Lee data/output/optimizacion_nitrato_demanda/adr.json; escribe barrido_zp.json ahí mismo.
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

VARIANTE = os.environ.get("VARIANTE", "nitrato_demanda")     # config/<VARIANTE>.json -> optimizacion_<VARIANTE>
DIR = os.path.join(CODIGO, "data", "output", "optimizacion_" + VARIANTE)
ZPS = [-200.0 - 20.0 * i for i in range(11)] + [-450.0, -500.0, -550.0, -600.0, -650.0, -700.0]


def main():
    ref = json.load(open(os.path.join(DIR, "adr.json")))
    cfg = json.load(open(os.path.join(CODIGO, "config", VARIANTE + ".json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": "adr", "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    p.C_ref = ref["C_ref_modelo"]
    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context, _solve_forward, _pesos_J
    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)
    Q = np.asarray(ref["x_opt"][:-1], float)
    w1, w2, w3 = _pesos_J(p, len(Q) * p.dt)
    out = []
    for zp in ZPS:
        r = _solve_forward(ctx, Q, zp)
        J2 = w2 * p.koppa * abs(zp - p.z0) ** 2
        J3 = w3 * np.sum(Q ** 2) * p.dt * abs(zp - p.z0)
        out.append(dict(zp=zp, J=float(r["J"]), J1=float(r["J1"]), J2=float(J2), J3=float(J3)))
        print(out[-1], flush=True)
    json.dump(out, open(os.path.join(DIR, "barrido_zp.json"), "w"), indent=1)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
