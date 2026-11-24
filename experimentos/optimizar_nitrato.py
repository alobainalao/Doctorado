"""
Optimización adjunta (FEniCSx) con config/nitrato.json: nitrato, J1/J2/J3 normalizados (normalizar_J).

    python -m experimentos.optimizar_nitrato <adr|mrmt_semi|mrmt_block> [variante c_limite] [--x0=RUTA --etiqueta=E
                                             --barrera0=B --radio0=R --factible=1]

--x0: reanuda desde un iterado guardado (<modelo>_progreso.json o <modelo>.json, de cualquier modelo); --etiqueta:
las salidas pasan a <modelo>_<etiqueta>[_progreso].json para no pisar las de la corrida original; --barrera0 y
--radio0: barrera y radio de confianza iniciales de trust-constr (por defecto los de scipy, 0.1 y 1); --factible:
todos los iterados admisibles (keep_feasible; x0 se proyecta al interior si hace falta). Si ya hay salidas con
el mismo nombre la corrida no arranca: nunca sobrescribe resultados. --f=<variante>: usa la escala f de la fuente de otra
variante (necesario si la fuente está apagada: el pico sin bombeo es 0).

La fuente se escala por f tal que el pico de C sin bombeo (ADR) sea 42 mg/L como N = 0.1859 kg/m³ de
nitrato. El modelo es lineal en C, así que equivale a expresar C_ref = 0.05 kg/m³ (valor guía OMS)
en unidades del modelo: C_ref_modelo = 0.05 / f. La fuente es la misma en los tres modelos: f se
calcula con ADR y se reusa en MRMT (data/output/optimizacion_nitrato/f_fuente.json).

Entorno: fenics_env. Salida: data/output/optimizacion_nitrato/<modelo>.json
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

C_FUENTE = 42e-3 * 62.0 / 14.0     # 42 mg/L como N -> kg/m³ de nitrato
C_LIMITE = 0.05                    # kg/m³ (50 mg/L como nitrato, OMS)
DIR_OUT = os.path.join(CODIGO, "data", "output", "optimizacion_nitrato")


def main(modelo, variante=None, c_limite=None, x0=None, etiqueta=None, barrera0=None, radio0=None, factible=None, f=None):
    """variante: nombre de config/<variante>.json (p.ej. nitrato_estricto) y c_limite [kg/m³]; salida
    aparte en data/output/optimizacion_<variante>/ para no pisar los resultados estándar. x0: ruta de un iterado
    guardado desde el que se reanuda; etiqueta: sufijo de las salidas."""
    global DIR_OUT, C_LIMITE
    if variante:
        DIR_OUT = os.path.join(CODIGO, "data", "output", f"optimizacion_{variante}")
    if c_limite:
        C_LIMITE = float(c_limite)
    nombre = f"{modelo}_{etiqueta}" if etiqueta else modelo
    previas = [r for r in (f"{nombre}.json", f"{nombre}_progreso.json", f"{nombre}_progreso.jsonl")
               if os.path.exists(os.path.join(DIR_OUT, r))]
    if previas:
        sys.exit(f"ya existen {previas} en {DIR_OUT}: usa otra --etiqueta para no sobrescribirlos")
    cfg = json.load(open(os.path.join(CODIGO, "config", f"{variante or 'nitrato'}.json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": modelo, "pre": True,
                "save_dat": False, "opt_maxiter": 100, "animate": False, "postproc": False,
                "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()

    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context, _solve_forward, optimize_fenicsx, _pesos_J

    # Igual que main.py: con `mesh_pkl` se usa la malla compartida congelada; si no, la malla gmsh gruesa.
    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)
    nQ = len(np.asarray(p.Qout[0]))
    zp0 = float(p.pozo[1])

    os.makedirs(DIR_OUT, exist_ok=True)
    ruta_f = os.path.join(DIR_OUT, "f_fuente.json")
    r0 = _solve_forward(ctx, np.zeros(nQ), zp0)
    pico = max(float(np.max(c)) for c in r0["C"])
    if f:                                   # escala de otra variante
        f = json.load(open(os.path.join(CODIGO, "data", "output", f"optimizacion_{f}", "f_fuente.json")))["f"]
        if not os.path.exists(ruta_f):
            json.dump({"f": f, "pico_adr": pico}, open(ruta_f, "w"))
    elif modelo == "adr" or not os.path.exists(ruta_f):
        f = C_FUENTE / pico
        if modelo == "adr":
            json.dump({"f": f, "pico_adr": pico}, open(ruta_f, "w"))
    else:
        f = json.load(open(ruta_f))["f"]
    p.C_ref = C_LIMITE / f
    print(f"MODELO={modelo} pico_modelo={pico:.4e} f={f:.4e} C_ref_modelo={p.C_ref:.4e}", flush=True)

    if x0:
        d = json.load(open(x0))
        p.opt_x0, x0 = x0, np.asarray(d.get("x", d.get("x_opt")), float)
    else:
        x0 = np.concatenate([np.asarray(p.Qout[0], float), [zp0]])
    p.opt_barrera0, p.opt_radio0, p.opt_factible = barrera0, radio0, bool(factible)
    w1, w2, w3 = _pesos_J(p, nQ * p.dt)

    def comp(Q, zp):
        r = _solve_forward(ctx, Q, zp)
        J2 = w2 * p.koppa * abs(zp - p.z0) ** 2
        J3 = w3 * np.sum(np.asarray(Q) ** 2) * p.dt * abs(zp - p.z0)
        return dict(J=float(r["J"]), J1=float(r["J1"]), J2=float(J2), J3=float(J3))

    tabla = {"x0": comp(x0[:-1], x0[-1])}
    if not getattr(p, "opt_x0", None):     # referencias del arranque estándar (al reanudar ya están en la corrida original)
        tabla.update({"x0_recortado_Qmax": comp(np.full(nQ, p.Q_max), zp0), "Q=0": comp(np.zeros(nQ), zp0)})
    # iterado aceptado en cada iteración y paro limpio: touch <DIR_OUT>/<nombre>_progreso.stop
    p.opt_progreso = os.path.join(DIR_OUT, f"{nombre}_progreso")
    if os.path.exists(p.opt_progreso + ".stop"):
        os.remove(p.opt_progreso + ".stop")
    res = optimize_fenicsx(p, mesh_cache=mc)
    xo = np.asarray(res.x)
    tabla["optimo"] = comp(xo[:-1], xo[-1])
    for k, v in tabla.items():
        print(f"{k:20s}", {a: f"{b:.4e}" for a, b in v.items()}, flush=True)

    out = dict(modelo=modelo, pico_modelo=pico, f=f, C_ref_modelo=p.C_ref, x0=x0.tolist(), x0_de=getattr(p, "opt_x0", None),
               x_opt=xo.tolist(), nit=int(res.nit), success=bool(res.success),
               msg=str(res.message), tabla=tabla)
    with open(os.path.join(DIR_OUT, f"{nombre}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("FIN", modelo, out["nit"], out["success"], out["msg"], flush=True)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opc = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    main(*pos[:3], **opc)
