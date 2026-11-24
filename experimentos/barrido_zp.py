"""
Barrido de Ĵ(z_p) con el caudal fijo de un iterado guardado: J y sus términos J1 (calidad), J2 (instalación) y J3 (bombeo)
en una lista de profundidades. Da la curva calidad–costo (J_m frente a J_inst + J_op) de la lámina del frente de Pareto:
para cada peso γ, el mínimo de γ J_m + J_inst + J_op sobre la curva (aproximación con el caudal fijo).

    python -m experimentos.barrido_zp <modelo> <variante> <iterado.json> <salida.json> z1,z2,...   # fenics_env

Salida: JSON con zp, J, J1, J2, J3 (normalizados como en optimizar_nitrato, γ = 1) y el iterado de origen.
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

C_LIMITE = 0.05                    # kg/m³ (50 mg/L como nitrato, OMS), igual que optimizar_nitrato


def main(modelo, variante, ruta_x, salida, zs):
    cfg = json.load(open(os.path.join(CODIGO, "config", f"{variante}.json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": modelo, "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()

    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context, _solve_forward, _pesos_J

    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)
    dir_out = os.path.join(CODIGO, "data", "output", f"optimizacion_{variante}")
    f = json.load(open(os.path.join(dir_out, "f_fuente.json")))["f"]
    p.C_ref = C_LIMITE / f
    d = json.load(open(ruta_x))
    Q = np.asarray(d.get("x", d.get("x_opt")), float)[:-1]
    w1, w2, w3 = _pesos_J(p, len(Q) * p.dt)

    out = dict(modelo=modelo, iterado=os.path.relpath(ruta_x, CODIGO), nit=d.get("nit"), zp=[], J=[], J1=[], J2=[], J3=[])
    for zp in (float(z) for z in zs.split(",")):
        r = _solve_forward(ctx, Q, zp)
        J2 = w2 * p.koppa * abs(zp - p.z0) ** 2
        J3 = w3 * float(np.sum(Q ** 2)) * p.dt * abs(zp - p.z0)
        for k, v in (("zp", zp), ("J", float(r["J"])), ("J1", float(r["J1"])), ("J2", float(J2)), ("J3", float(J3))):
            out[k].append(v)
        print(f"z_p={zp:.0f}  J={float(r['J']):.6f}  J1={float(r['J1']):.6f}", flush=True)
        with open(salida, "w") as fh:          # se reescribe en cada punto: un corte no pierde lo calculado
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])
