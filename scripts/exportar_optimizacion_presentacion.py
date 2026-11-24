"""
Exporta los resultados de la optimización adjunta (experimentos/optimizar_nitrato.py, variante
nitrato_vscale_malla_eje_g1, γ = 1, malla con conos y eje de 1 m) como módulo JS para la presentación (avance3):
caudal inicial y óptimo, cota del ciclo de la bomba, demanda por paso, volumen del tanque, términos normalizados
del funcional, trayectoria de J y z_p por evaluación (leída del registro de la optimización) y los perfiles
horarios de consumo y de disponibilidad de la bomba (funtions/water_supply.py).  Si la corrida no terminó
(<modelo>.json ausente) se usa el último iterado aceptado (<modelo>_progreso.json) y los términos J2, J3 se
calculan de los controles (J1 = Ĵ − J2 − J3).  Además: Ĵ(z_p) con paso de 1 m en las dos mallas (ondulación),
el barrido de J1 con el caudal de ADR, C en el pozo (analisis_zp/), la evaluación cruzada de los óptimos
(cruce/, experimentos/evaluar_optimos.py) y la verificación del gradiente adjunto de los 6 backends
(data/output/verificacion_adjunto/*_alm1.json) en verificacion.js.

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
VARIANTE = "nitrato_vscale_malla_eje_g1"
LOGS = os.path.join(CODIGO, "data", "output", "_logs_malla_conos")
LOG_OPT = "21_opt_eje_{}.log"
DIR_OPT = os.path.join(CODIGO, "data", "output", f"optimizacion_{VARIANTE}")
# Ĵ(z_p) con el caudal óptimo ADR de la malla con conos, paso de 1 m: sin y con el eje refinado
LOGS_ONDULACION = dict(conos="13_fino_r20.log", eje="15_fino_eje.log")
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
    nQ = len(cota)
    for m, nombre in MODELOS:
        r = resultado(m, p, nQ)
        if r is None:
            continue
        q0, qo = np.asarray(r["x0"][:-1]), np.asarray(r["x_opt"][:-1])
        if not out["modelos"]:
            out.update(Q0=np.round(q0, 4).tolist(), S0=tanque(q0), zp0=r["x0"][-1])
        out["modelos"].append(dict(id=m, nombre=nombre, nit=r["nit"], zp=round(r["x_opt"][-1], 1),
                                   chi=r.get("chi"), estado=r["estado"],
                                   Q=np.round(qo, 4).tolist(), S=tanque(qo),
                                   J0=r["tabla"]["x0"], J=r["tabla"]["optimo"], historia=historia(m)))
    # f: unidades del modelo → kg/m³ de nitrato (misma fuente en los tres modelos)
    out.update(f=json.load(open(os.path.join(DIR_OPT, "f_fuente.json")))["f"],
               ondulacion=ondulacion(), analisis=analisis(), cruce=cruce(), pareto=pareto())
    js = ("// Generado por codigo/scripts/exportar_optimizacion_presentacion.py desde\n"
          f"// data/output/optimizacion_{VARIANTE}/ (<modelo>.json o <modelo>_progreso.json). No editar a mano.\n"
          f"export const OPT = {json.dumps(out, ensure_ascii=False)};\n")
    ruta = os.path.join(PRES, "src", "slides-avance3", "optimoNitrato.js")
    with open(ruta, "w") as f:
        f.write(js)
    print("escrito", ruta, [m["id"] for m in out["modelos"]])
    verificacion()


def terminos(p, J, Q, zp, nQ):
    """Términos normalizados de Ĵ: J2 y J3 de los controles (pesos de funtions/optimizador.py:pesos_J), J1 = Ĵ − J2 − J3."""
    from funtions.optimizador import pesos_J
    _, w2, w3 = pesos_J(SimpleNamespace(**{**vars(p), "C_ref": 1.0}), nQ * p.dt)     # w1 no se usa: J1 sale por diferencia
    J2 = w2 * p.koppa * abs(zp - p.z0) ** 2
    J3 = w3 * float(np.sum(np.asarray(Q) ** 2)) * p.dt * abs(zp - p.z0)
    return dict(J=J, J1=J - J2 - J3, J2=J2, J3=J3)


def resultado(m, p, nQ):
    """x0, x_opt, nit y tabla de términos: de <m>.json si la corrida terminó; si no, del último iterado aceptado
    (<m>_progreso.json) con Ĵ(x0) de la primera evaluación del registro."""
    ruta = os.path.join(DIR_OPT, f"{m}.json")
    if os.path.exists(ruta):
        r = json.load(open(ruta))
        return dict(r, estado="final")
    ruta = os.path.join(DIR_OPT, f"{m}_progreso.json")
    h = historia(m)
    if not os.path.exists(ruta) or h is None:
        return None
    d = json.load(open(ruta))
    x0 = [float(p.Q_std)] * nQ + [h["zp"][0]]
    xo = d["x"]
    tabla = dict(x0=terminos(p, h["J"][0], x0[:-1], x0[-1], nQ),
                 optimo=terminos(p, d["J"], xo[:-1], xo[-1], nQ))
    return dict(x0=x0, x_opt=xo, nit=d["nit"], chi=d["chi"], tabla=tabla, estado="progreso")


def ondulacion():
    """Ĵ(z_p) con paso de 1 m y el mismo caudal, en la malla con conos y en la de eje refinado."""
    import re
    out = {}
    for k, f in LOGS_ONDULACION.items():
        ruta = os.path.join(LOGS, f)
        if os.path.exists(ruta):
            ev = re.findall(r"z_p=([0-9.e+-]+) J=([0-9.e+-]+)", open(ruta).read())
            out[k] = dict(zp=[float(a) for a, _ in ev], J=[float(b) for _, b in ev])
    return out


def analisis():
    """Barrido de Ĵ y J1 en z_p con el caudal óptimo ADR, y C en el pozo / masa móvil en z_p = −383 m (analisis_zp/)."""
    d = os.path.join(DIR_OPT, "analisis_zp")
    out = dict(barrido={}, diag={})
    for m in ("adr", "mrmt_semi"):
        ruta = os.path.join(d, f"barrido_J1_{m}_Qadr.json")
        if os.path.exists(ruta):
            b = json.load(open(ruta))["barrido"]
            out["barrido"][m] = dict(zp=[r["zp"] for r in b], J=[r["J"] for r in b], J1=[r["J1"] for r in b])
        ruta = os.path.join(d, f"diag_{m}_zp383_Qadr.json")
        if os.path.exists(ruta):
            g = json.load(open(ruta))
            out["diag"][m] = dict(zp=g["zp"], J=g["J"], J1=g["J1"], Cpozo=[float(c) for c in g["Cpozo"]],
                                  masa=[float(c) for c in g["masa"]])
    return out


def cruce():
    """Ĵ de cada modelo en el iterado guardado de los otros (cruce/<modelo>_en_<x_de>.json)."""
    d = os.path.join(DIR_OPT, "cruce")
    if not os.path.isdir(d):
        return []
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]


def pareto():
    """Curva calidad–costo de ADR con el caudal fijo de su mejor iterado: J1, J2 y J3 en cada z_p del barrido
    (pareto/adr_parte*.json, experimentos/barrido_zp.py), ordenados por z_p."""
    d = os.path.join(DIR_OPT, "pareto")
    partes = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))
              if f.startswith("adr_parte")] if os.path.isdir(d) else []
    if not partes:
        return None
    filas = sorted(zip(*[sum((q[k] for q in partes), []) for k in ("zp", "J1", "J2", "J3")]))
    return dict(iterado=partes[0]["iterado"], nit=partes[0]["nit"],
                **{k: [round(float(r[i]), 6) for r in filas] for i, k in enumerate(("zp", "J1", "J2", "J3"))})


def historia(m):
    """J y z_p de cada evaluación del funcional, en el orden del optimizador (registro LOG_OPT)."""
    import re
    ruta = os.path.join(LOGS, LOG_OPT.format(m))
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
