"""
Exporta los resultados de la optimización adjunta (experimentos/optimizar_nitrato.py, variante
nitrato_vscale_malla) como módulo JS para la diapositiva de control óptimo de la presentación
(avance3, A3N33): caudal inicial y óptimo, cota del ciclo de la bomba, volumen del tanque y
términos normalizados del funcional de cada modelo optimizado.

    python3 scripts/exportar_optimizacion_presentacion.py [ruta_de_la_presentacion]      # bfr_env

Escribe <presentacion>/src/slides-avance3/optimoNitrato.js
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc-sin-sumidero")
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
VARIANTE = "nitrato_vscale_malla"
DIR_OPT = os.path.join(CODIGO, "data", "output", f"optimizacion_{VARIANTE}")
MODELOS = (("adr", "ADR"), ("mrmt_semi", "MRMT semi"), ("mrmt_block", "MRMT bloque"))


def main():
    from funtions import water_supply as ws
    cfg = json.load(open(os.path.join(CODIGO, "config", f"{VARIANTE}.json")))
    p = SimpleNamespace(**cfg)
    nt = int(round(p.T / p.dt))
    t = (np.arange(nt + 1) * p.dt / 86400).round(4).tolist()
    cota = ws.pump_upper_bounds(p, nt, p.Q_max)
    demanda = ws.demand_vector(p, nt)
    tanque = lambda q: [p.S0] + ws.storage_trajectory(np.asarray(q), p).round(1).tolist()
    out = dict(t_dias=t, cota=np.round(cota, 4).tolist(), S_max=p.S_max, Q_max=p.Q_max,
               demanda_media=float(demanda.sum() / (p.T)), V_s=p.Vscale, gamma=p.gamma, modelos=[])
    for m, nombre in MODELOS:
        ruta = os.path.join(DIR_OPT, f"{m}.json")
        if not os.path.exists(ruta):
            continue
        r = json.load(open(ruta))
        q0, qo = np.asarray(r["x0"][:-1]), np.asarray(r["x_opt"][:-1])
        if not out["modelos"]:
            out.update(Q0=np.round(q0, 4).tolist(), S0=tanque(q0), zp0=r["x0"][-1])
        out["modelos"].append(dict(id=m, nombre=nombre, nit=r["nit"], zp=round(r["x_opt"][-1], 1),
                                   Q=np.round(qo, 4).tolist(), S=tanque(qo),
                                   J0=r["tabla"]["x0"], J=r["tabla"]["optimo"]))
    js = ("// Generado por codigo/scripts/exportar_optimizacion_presentacion.py desde\n"
          f"// data/output/optimizacion_{VARIANTE}/<modelo>.json. No editar a mano.\n"
          f"export const OPT = {json.dumps(out, ensure_ascii=False)};\n")
    ruta = os.path.join(PRES, "src", "slides-avance3", "optimoNitrato.js")
    with open(ruta, "w") as f:
        f.write(js)
    print("escrito", ruta, [m["id"] for m in out["modelos"]])


if __name__ == "__main__":
    main()
