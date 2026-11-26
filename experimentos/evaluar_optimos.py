"""
Evaluación cruzada de los óptimos de experimentos/optimizar_nitrato.py: Ĵ y sus términos de un modelo en el
iterado guardado de cada modelo (<modelo>_progreso.json, o <modelo>.json si la corrida terminó). Si el
óptimo de otro modelo da menor Ĵ que el propio, el optimizador paró antes de tiempo.

    python -m experimentos.evaluar_optimos <adr|mrmt_semi|mrmt_block> <variante> [x_de ...] [--nit=N]

x_de: modelos cuyo iterado se evalúa (por defecto los otros dos). --nit: el iterado N de <x_de>_progreso.jsonl en lugar
del final (comparar con el mismo número de iteraciones). Entorno: fenics_env.
Salida: data/output/optimizacion_<variante>/cruce/<modelo>_en_<x_de>.json
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
MODELOS = ("adr", "mrmt_semi", "mrmt_block")


def iterado(dir_out, m, nit=None):
    """x guardado del modelo m: el iterado nit del registro si se pide; si no, el óptimo final si existe, y si no el
    último iterado aceptado."""
    if nit:
        for linea in open(os.path.join(dir_out, f"{m}_progreso.jsonl")):
            r = json.loads(linea)
            if r["nit"] == nit:
                return np.asarray(r["x"]), f"progreso it {nit}"
        raise SystemExit(f"{m}: el registro no tiene la iteración {nit}")
    ruta = os.path.join(dir_out, f"{m}.json")
    if os.path.exists(ruta):
        return np.asarray(json.load(open(ruta))["x_opt"]), "final"
    d = json.load(open(os.path.join(dir_out, f"{m}_progreso.json")))
    return np.asarray(d["x"]), f"progreso it {d['nit']}"


def main(modelo, variante, *args):
    nit = next((int(a.split("=")[1]) for a in args if a.startswith("--nit=")), None)
    x_de = tuple(a for a in args if not a.startswith("--"))
    dir_out = os.path.join(CODIGO, "data", "output", f"optimizacion_{variante}")
    x_de = x_de or tuple(m for m in MODELOS if m != modelo)
    cfg = json.load(open(os.path.join(CODIGO, "config", f"{variante}.json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": modelo, "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()

    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context, _solve_forward, _pesos_J

    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)
    # misma fuente en los tres modelos: f de ADR (f_fuente.json de la optimización)
    f = json.load(open(os.path.join(dir_out, "f_fuente.json")))["f"]
    p.C_ref = C_LIMITE / f
    nQ = len(np.asarray(p.Qout[0]))
    w1, w2, w3 = _pesos_J(p, nQ * p.dt)

    os.makedirs(os.path.join(dir_out, "cruce"), exist_ok=True)
    for m in x_de:
        x, origen = iterado(dir_out, m, nit)
        Q, zp = x[:-1], float(x[-1])
        r = _solve_forward(ctx, Q, zp)
        J2 = w2 * p.koppa * abs(zp - p.z0) ** 2
        J3 = w3 * np.sum(Q ** 2) * p.dt * abs(zp - p.z0)
        out = dict(modelo=modelo, x_de=m, origen=origen, zp=zp,
                   J=float(r["J"]), J1=float(r["J1"]), J2=float(J2), J3=float(J3))
        with open(os.path.join(dir_out, "cruce", f"{modelo}_en_{m}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
