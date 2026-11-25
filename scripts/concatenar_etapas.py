"""
Concatena las etapas de cada optimización (corrida original, reanudaciones y relanzamiento con la demanda promediada
por paso) en un solo historial con la iteración acumulada, leído de los registros de experimentos/optimizar_nitrato.py.

Cada cadena sigue el iterado que de verdad se usó como x0 en la etapa siguiente; las ramas que no llevan al iterado
final quedan fuera (MRMT semi reanudada/reanudada2, bloque original y desde_semi2, ADR original como rama aparte).
MRMT bloque arranca del iterado 64 de MRMT semi, así que su cadena empieza con esas 64 iteraciones. Las etapas
con demanda "instantanea" usan d_n, a_n a mitad del paso; las "promediada", el promedio del paso (profile_step_mean):
en la frontera cambian las restricciones del tanque y x0 se proyecta, por eso Ĵ salta. Tope: 110 iteraciones
acumuladas (scratchpad tope_110.sh pide el paro).

    python3 scripts/concatenar_etapas.py

Escribe data/output/optimizacion_nitrato_vscale_malla_eje_g1_prom/desde_iterados_viejos/cadenas.json (serie 31-32,
sustituida el 24-nov por la serie 33 desde cero: _logs_malla_conos/cola_33.py)
"""
import json
import os
import re

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(CODIGO, "data", "output", "_logs_malla_conos")
SALIDA = os.path.join(CODIGO, "data", "output", "optimizacion_nitrato_vscale_malla_eje_g1_prom", "desde_iterados_viejos",
                      "cadenas.json")
TOPE = 110

# (registro, modelo de la etapa, demanda)
CADENAS = {
    "adr": [("21_opt_eje_adr.log", "adr", "instantanea"),
            ("27_reanudar_adr.log", "adr", "instantanea"),
            ("28_reanudar2_adr.log", "adr", "instantanea"),
            ("31_prom_adr.log", "adr", "promediada")],
    "mrmt_semi": [("21_opt_eje_mrmt_semi.log", "mrmt_semi", "instantanea"),
                  ("31_prom_mrmt_semi.log", "mrmt_semi", "promediada")],
    "mrmt_block": [("21_opt_eje_mrmt_semi.log", "mrmt_semi", "instantanea"),
                   ("27_reanudar_mrmt_block.log", "mrmt_block", "instantanea"),
                   ("31_prom_mrmt_block.log", "mrmt_block", "promediada")],
    "validacion_sin_nitrato": [("30_validacion_sin_nitrato.log", "adr", "instantanea"),
                               ("32_prom_validacion_sin_nitrato_desde36.log", "adr", "promediada")],
}

IT = re.compile(r"\[fenicsx-opt it (\d+)\] χ=(\S+) \(tol \S+\)\s+Ĵ=(\S+)\s+violación=(\S+)")
EV = re.compile(r"\[fenicsx-opt\] J=([0-9.e+-]+) .*z_p=([0-9.e+-]+)")


def etapa(log):
    """Iteraciones (χ, Ĵ, violación) y evaluaciones (Ĵ, z_p) de un registro, en orden."""
    txt = open(os.path.join(LOGS, log)).read() if os.path.exists(os.path.join(LOGS, log)) else ""
    its = [dict(nit=int(n), chi=float(c), J=float(j), violacion=float(v)) for n, c, j, v in IT.findall(txt)]
    evs = [dict(J=float(j), zp=float(z)) for j, z in EV.findall(txt)]
    return its, evs, "FIN" in txt or "terminado:" in txt


def cadena(partes):
    iteraciones, evaluaciones, etapas, previas = [], [], [], 0
    for k, (log, modelo, demanda) in enumerate(partes):
        its, evs, fin = etapa(log)
        for r in its:
            if previas + r["nit"] > TOPE:
                break
            iteraciones.append(dict(r, it=previas + r["nit"], etapa=k))
        evaluaciones += [dict(e, etapa=k) for e in evs]
        etapas.append(dict(registro=log, modelo=modelo, demanda=demanda, desde=previas, nit=len(its), terminada=fin,
                           J0=evs[0]["J"] if evs else None, J_fin=its[-1]["J"] if its else None))
        previas += len(its)
    return dict(etapas=etapas, iteraciones=iteraciones, evaluaciones=evaluaciones, total=min(previas, TOPE))


if __name__ == "__main__":
    out = dict(tope=TOPE, cadenas={m: cadena(p) for m, p in CADENAS.items()})
    with open(SALIDA, "w") as fh:
        json.dump(out, fh, indent=1)
    for m, c in out["cadenas"].items():
        print(f"{m:24s} {c['total']:3d}/{TOPE} it  " + "  →  ".join(
            f"{e['modelo']}[{e['desde'] + 1}–{e['desde'] + e['nit']}] Ĵ {e['J0']:.5f}→{e['J_fin']:.5f}"
            if e["nit"] else f"{e['modelo']}[pendiente]" for e in c["etapas"]))
    print("→", SALIDA)
