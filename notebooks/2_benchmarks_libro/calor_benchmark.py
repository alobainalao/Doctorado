"""Benchmark de referencia: control distribuido de la ecuación del calor (Manzoni–Quarteroni–Salsa 2021, §8.3).

Uso (desde codigo/notebooks):
    python3 calor_benchmark.py                # nivel rápido (n = 20): RBF-FD y FEM contra la referencia, OK/FALLA
    python3 calor_benchmark.py --libro        # Tablas 8.2–8.3 del libro frente a nuestros valores
    python3 calor_benchmark.py --referencia   # recalcula calor_referencia.json (FEM n = 20, 40, 80 + Richardson)
    python3 calor_benchmark.py --congelar     # guarda la instantánea de regresión del nivel rápido

Referencia: FEM P1 con el blanco integrado exactamente (no interpolado), Crank–Nicolson; caso 1 con Δt uniforme
(Δt = 1/(5n)), caso 2 con Δt graduado geométricamente hacia T (capa límite del control). Refinamiento conjunto
h, Δt por 2 y Richardson con el orden observado.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from calor_nucleo import BOOK, FEM2, RBF2, HeatOCP, pcg, time_grid

REF = Path(__file__).with_name("calor_referencia.json")
BETAS = (1e-2, 1e-3, 1e-4)
N_REF = (20, 40, 80)
TOL_CG = 1e-8


def tgrid(case, n):
    """Malla temporal ligada a la espacial: Δt0 = 1/(5n); caso 2: graduada hasta Δt_min ≲ h^2/10."""
    nt = 5 * n
    if case == 1:
        return time_grid(nt)
    K = int(np.ceil(np.log2((1 / nt) / (0.1 / n**2))))
    return time_grid(nt, K)


def solve(disc, case, beta, t, tol=TOL_CG, exact=True):
    o = HeatOCP(disc, case, beta, None, t=t, exact=exact)
    U, it, _ = pcg(o, tol=tol)
    return o, U, it


def richardson(v):
    v0, v1, v2 = v
    p = np.log2(abs((v1 - v0) / (v2 - v1)))
    lim = v2 + (v2 - v1) / (2**p - 1)
    return float(lim), float(p), float(abs(lim - v2))


def calcular_referencia():
    ref = {"fuente": "FEM P1, blanco exacto, CN, n = " + ", ".join(map(str, N_REF)) + " con Δt0 = 1/(5n)"
                     " (caso 2 graduado hacia T), Richardson con orden observado", "casos": {}}
    for case in (1, 2):
        niv = {b: [] for b in BETAS}
        its = {b: [] for b in BETAS}
        for n in N_REF:
            tt = time.time()
            d = FEM2(n)
            t = tgrid(case, n)
            for b in BETAS:
                o, U, it = solve(d, case, b, t)
                niv[b].append(float(o.J(U))); its[b].append(it)
            print(f"  caso {case} n={n} ({len(t) - 1} pasos): {time.time() - tt:.0f}s "
                  + " ".join(f"{niv[b][-1]:.7g}" for b in BETAS), flush=True)
        ref["casos"][str(case)] = {}
        for b in BETAS:
            lim, p, err = richardson(niv[b])
            ref["casos"][str(case)][f"{b:g}"] = {"niveles": niv[b], "it_cg": its[b], "valor": lim, "orden": p,
                                                  "error_estimado_n80": err}
    REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print(f"referencia escrita en {REF}")
    return ref


def nivel(n, ref, verbose=True):
    """RBF-FD y FEM en la malla n (Δt ligado), con evaluación cruzada del control RBF-FD en el FEM."""
    out = {}
    g, f = RBF2(n), FEM2(n)  # misma retícula: el control nodal pasa de uno a otro sin interpolar
    if verbose:
        print(f"\nn = {n}: {g.N} nodos")
        print(f"{'caso':>4s} {'β':>6s} | {'J*_ref':>10s} | {'J*_RBF':>10s} {'err%':>6s} {'it':>3s} | "
              f"{'J*_FEM':>10s} {'err%':>6s} | {'J_FEM(û_RBF)':>12s} {'Δ%':>6s} | {'s':>4s}")
    for case in (1, 2):
        t = tgrid(case, n)
        for b in BETAS:
            tt = time.time()
            og, ug, itg = solve(g, case, b, t, tol=1e-6)
            of, uf, itf = solve(f, case, b, t, tol=1e-6)
            Jx = of.J(ug)
            Jr = ref["casos"][str(case)][f"{b:g}"]["valor"]
            Jg, Jf = og.J(ug), of.J(uf)
            out[(case, b)] = dict(J_rbf=Jg, J_fem=Jf, J_cruz=Jx, it=itg)
            if verbose:
                print(f"{case:4d} {b:6.0e} | {Jr:10.6g} | {Jg:10.6g} {100 * (Jg / Jr - 1):6.2f} {itg:3d} | "
                      f"{Jf:10.6g} {100 * (Jf / Jr - 1):6.2f} | {Jx:12.6g} {100 * (Jx / Jf - 1):6.2f} | "
                      f"{time.time() - tt:4.1f}")
    return out


# Criterios del nivel rápido (n = 20), con margen sobre lo medido el 2026-10-28. En el caso 2 con β pequeño n = 20 es
# preasintótica (blanco final discontinuo: el FEM tiene 6 % y 42 % de error), así que ahí sólo se exige el orden de
# magnitud; la verificación fina es la regresión contra la instantánea congelada.
TOL_J = {(1, 1e-2): 0.01, (1, 1e-3): 0.02, (1, 1e-4): 0.04, (2, 1e-2): 0.04, (2, 1e-3): 0.25, (2, 1e-4): 1.5}
CRITERIOS = {"cruz": 0.15, "it_cg": 60, "regresion": 1e-8}


def verificar(r, ref):
    fallos = []
    snap = ref.get("regresion_n20")
    for (case, b), x in r.items():
        Jr = ref["casos"][str(case)][f"{b:g}"]["valor"]
        tag = f"caso {case} β={b:g}"
        if abs(x["J_rbf"] / Jr - 1) > TOL_J[(case, b)]: fallos.append(f"{tag} J*_RBF")
        if abs(x["J_fem"] / Jr - 1) > TOL_J[(case, b)]: fallos.append(f"{tag} J*_FEM")
        if not (-1e-9 <= x["J_cruz"] / x["J_fem"] - 1 <= CRITERIOS["cruz"]): fallos.append(f"{tag} cruzada")
        if x["it"] > CRITERIOS["it_cg"]: fallos.append(f"{tag} iteraciones")
        if snap is None:
            continue
        s = snap[f"{case}_{b:g}"]
        if abs(x["J_rbf"] / s - 1) > CRITERIOS["regresion"]: fallos.append(f"{tag} regresión")
    if snap is None:
        fallos.append("sin instantánea de regresión (correr --congelar)")
    return fallos


def tabla_libro(ref):
    """Nuestros J* con la lectura literal de la ec. 8.12 (blanco interpolado en los nodos) en mallas comparables a
    las del libro, frente a las Tablas 8.2–8.3 y a la referencia (blanco exacto, malla y Δt convergidos)."""
    print("\nLibro (Tablas 8.2–8.3) vs nuestro FEM con blanco nodal (ec. 8.12 literal) vs referencia")
    print("malla (i): 712 vértices, Δt = 0.05 -> nuestra n = 25 (676);  (ii): 1921 vértices, Δt = 0.01 -> n = 45 (2116)")
    print(f"{'caso':>4s} {'β':>6s} | {'libro (i)':>10s} {'nodal n25':>10s} | {'libro (ii)':>10s} {'nodal n45':>10s} | "
          f"{'exacto n45':>10s} {'referencia':>10s}")
    for case in (1, 2):
        for b in BETAS:
            row = []
            for n, nt, exact in ((25, 20, False), (45, 100, False), (45, 100, True)):
                o, U, _ = solve(FEM2(n), case, b, time_grid(nt), exact=exact)
                row.append(o.J(U))
            Jr = ref["casos"][str(case)][f"{b:g}"]["valor"]
            print(f"{case:4d} {b:6.0e} | {BOOK[case]['i'][b][0]:10.5g} {row[0]:10.5g} | "
                  f"{BOOK[case]['ii'][b][0]:10.5g} {row[1]:10.5g} | {row[2]:10.5g} {Jr:10.5g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--referencia", action="store_true")
    ap.add_argument("--libro", action="store_true")
    ap.add_argument("--congelar", action="store_true")
    a = ap.parse_args()
    if a.referencia or not REF.exists():
        calcular_referencia()
    ref = json.loads(REF.read_text())
    print("Referencia (" + ref["fuente"] + "):")
    for case, e in ref["casos"].items():
        print(f"  caso {case}: " + "   ".join(f"β={b}: {v['valor']:.7g} ± {v['error_estimado_n80']:.1e} (orden {v['orden']:.2f})"
                                       for b, v in e.items()))
    tt = time.time()
    if a.libro:
        tabla_libro(ref)
    else:
        r = nivel(20, ref)
        if a.congelar:
            ref["regresion_n20"] = {f"{c}_{b:g}": float(x["J_rbf"]) for (c, b), x in r.items()}
            REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
            print("instantánea de regresión (n = 20) guardada")
        fallos = verificar(r, ref)
        print("\nRESULTADO:", "OK" if not fallos else "FALLA en " + ", ".join(fallos))
    print(f"tiempo total: {time.time() - tt:.1f}s")


if __name__ == "__main__":
    main()
