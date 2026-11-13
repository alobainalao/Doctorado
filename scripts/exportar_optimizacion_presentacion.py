"""
Exporta los resultados de la optimización adjunta (experimentos/optimizar_nitrato.py, variante
nitrato_vscale_malla_g1, γ = 1, malla con conos) como módulo JS para la presentación (avance3): caudal
inicial y óptimo, cota del ciclo de la bomba, demanda por paso, volumen del tanque, términos normalizados
del funcional, trayectoria de J y z_p por evaluación (leída del registro de la optimización) y los perfiles
horarios de consumo y de disponibilidad de la bomba (funtions/water_supply.py).  Además, la verificación del
gradiente adjunto de los 6 backends (data/output/verificacion_adjunto/*_alm1.json) en verificacion.js.

    python3 scripts/exportar_optimizacion_presentacion.py [ruta_de_la_presentacion]      # bfr_env

Escribe <presentacion>/src/slides-avance3/optimoNitrato.js
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc")
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
VARIANTE = "nitrato_vscale_malla_g1"
LOGS = os.path.join(CODIGO, "data", "output", "_logs_malla_conos")
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
               demanda_media=float(demanda.sum() / (p.T)), V_s=p.Vscale, gamma=p.gamma, dt=p.dt,
               demanda=np.round(demanda, 2).tolist(), t0_hora=p.t0_hour,
               perfiles=dict(domestico=np.round(ws._DOMESTIC_HOURLY, 4).tolist(),
                             riego=np.round(ws._AGRI_HOURLY, 4).tolist(),
                             bomba=ws._PUMP_AVAIL.tolist(), D_dom=p.demand_day, D_agri=p.demand_agri),
               modelos=[])
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
                                   J0=r["tabla"]["x0"], J=r["tabla"]["optimo"], historia=historia(m)))
    js = ("// Generado por codigo/scripts/exportar_optimizacion_presentacion.py desde\n"
          f"// data/output/optimizacion_{VARIANTE}/<modelo>.json. No editar a mano.\n"
          f"export const OPT = {json.dumps(out, ensure_ascii=False)};\n")
    ruta = os.path.join(PRES, "src", "slides-avance3", "optimoNitrato.js")
    with open(ruta, "w") as f:
        f.write(js)
    print("escrito", ruta, [m["id"] for m in out["modelos"]])
    verificacion()


def historia(m):
    """J y z_p de cada evaluación del funcional, en el orden del optimizador (registro 4_opt_g1_<m>.log)."""
    import re
    ruta = os.path.join(LOGS, f"4_opt_g1_{m}.log")
    if not os.path.exists(ruta):
        return None
    pat = re.compile(r"\[fenicsx-opt\] J=([0-9.e+-]+) .*z_p=([0-9.e+-]+)")
    ev = [(float(a), float(b)) for a, b in pat.findall(open(ruta).read())]
    return dict(J=[round(j, 6) for j, _ in ev], zp=[round(z, 1) for _, z in ev])


def verificacion():
    """Error relativo adjunto vs diferencias finitas centradas, por backend, modelo, dirección y paso ε."""
    filas = []
    for b in ("bfr", "fenicsx"):
        for m, nombre in MODELOS:
            ruta = os.path.join(CODIGO, "data", "output", "verificacion_adjunto", f"{b}_{m}_alm1.json")
            if os.path.exists(ruta):
                for r in json.load(open(ruta)):
                    filas.append(dict(backend=b, modelo=m, direccion=r["direccion"], eps=r["eps"],
                                      adjunto=r["adjunto"], fd=r["fd"], err=r["err_rel"]))
    ruta = os.path.join(PRES, "src", "slides-avance3", "verificacion.js")
    with open(ruta, "w") as f:
        f.write("// Generado por codigo/scripts/exportar_optimizacion_presentacion.py desde\n"
                "// data/output/verificacion_adjunto/*_alm1.json. No editar a mano.\n"
                f"export const VERIF = {json.dumps(filas, ensure_ascii=False)};\n")
    print("escrito", ruta, len(filas), "filas")


if __name__ == "__main__":
    main()
