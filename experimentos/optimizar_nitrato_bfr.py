"""
Optimización adjunta en BFR con la misma configuración que experimentos/optimizar_nitrato.py (FEniCSx).

    python -m experimentos.optimizar_nitrato_bfr <adr|mrmt_semi|mrmt_block> [variante c_limite]

El factor f de la fuente se lee de <DIR_OUT>/f_fuente.json, que escribe la optimización ADR de FEniCSx, así
que C_ref_modelo = C_LIMITE / f es el mismo en los dos backends (mismo funcional). El mismo x0, la misma
malla (`mesh_pkl`) y el mismo paro por criticidad χ (funtions/optimizador.py).

Entorno: bfr_env. Salida: <DIR_OUT>/bfr_<modelo>.json y la carpeta de la corrida <DIR_OUT>/<modelo>/bfr/
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

C_LIMITE = 0.05                    # kg/m³ (50 mg/L como nitrato, OMS), igual que optimizar_nitrato.py
DIR_OUT = os.path.join(CODIGO, "data", "output", "optimizacion_nitrato")


def main(modelo, variante=None, c_limite=None):
    global DIR_OUT, C_LIMITE
    if variante:
        DIR_OUT = os.path.join(CODIGO, "data", "output", f"optimizacion_{variante}")
    if c_limite:
        C_LIMITE = float(c_limite)
    ruta_f = os.path.join(DIR_OUT, "f_fuente.json")
    if not os.path.exists(ruta_f):
        raise FileNotFoundError(f"{ruta_f}: corre antes la optimización ADR de FEniCSx (optimizar_nitrato.py)")
    f = json.load(open(ruta_f))["f"]

    cfg = json.load(open(os.path.join(CODIGO, "config", f"{variante or 'nitrato'}.json")))
    cfg.update({"metodo": "bfr", "run_type": "optimization", "model": modelo, "pre": True,
                "save_dat": False, "opt_maxiter": 100, "animate": False, "postproc": False,
                "extract_C": False, "exp_dir": os.path.join(DIR_OUT, modelo)})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    p.C_ref = C_LIMITE / f

    from preprocessing.preprocess import load_data
    from funtions.optimizacion_bfr import functional, _directo, optimize_bfr

    d = load_data()
    nQ = len(np.asarray(p.Qout[0]))
    zp0 = float(p.pozo[1])
    x0 = np.concatenate([np.asarray(p.Qout[0], float), [zp0]])
    pico = max(float(np.max(c)) for c in _directo(np.concatenate([np.zeros(nQ), [zp0]]), d, p)["C"])
    print(f"MODELO={modelo} pico_modelo_bfr={pico:.4e} f(FEniCSx)={f:.4e} C_ref_modelo={p.C_ref:.4e}", flush=True)

    tabla = {"x0": functional(x0, d, p),
             "x0_recortado_Qmax": functional(np.concatenate([np.full(nQ, p.Q_max), [zp0]]), d, p),
             "Q=0": functional(np.concatenate([np.zeros(nQ), [zp0]]), d, p)}
    # iterado aceptado en cada iteración y paro limpio: touch <DIR_OUT>/bfr_<modelo>_progreso.stop
    p.opt_progreso = os.path.join(DIR_OUT, f"bfr_{modelo}_progreso")
    if os.path.exists(p.opt_progreso + ".stop"):
        os.remove(p.opt_progreso + ".stop")
    res = optimize_bfr(d, p)
    xo = np.asarray(res.x)
    tabla["optimo"] = functional(xo, d, p)
    for k, v in tabla.items():
        print(f"{k:20s} J={v:.6e}", flush=True)

    out = dict(modelo=modelo, backend="bfr", pico_modelo=pico, f=f, C_ref_modelo=p.C_ref, x0=x0.tolist(),
               x_opt=xo.tolist(), nit=int(res.nit), success=bool(res.success),
               msg=str(res.message), tabla=tabla)
    with open(os.path.join(DIR_OUT, f"bfr_{modelo}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("FIN", modelo, out["nit"], out["success"], out["msg"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
