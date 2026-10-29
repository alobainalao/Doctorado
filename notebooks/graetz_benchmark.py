"""Benchmark de referencia: control óptimo de Graetz (Manzoni–Quarteroni–Salsa 2021, §6.5.1) + pozos.

Uso (desde codigo/notebooks):
    python3 graetz_benchmark.py                 # nivel rápido (h = 1/40): verificación en segundos
    python3 graetz_benchmark.py --convergencia  # h = 1/20, 1/40, 1/80 contra la referencia extrapolada
    python3 graetz_benchmark.py --referencia    # recalcula graetz_referencia.json (FEM fino + Richardson)

La referencia son valores FEM P1 en mallas graduadas (q = 2) hacia las uniones Dirichlet–Neumann (1, 0) y (1, 1),
n = 80, 160, 320, extrapolados por Richardson con el orden observado; se contrasta con q = 3 en n = 320. En retícula
uniforme J* converge sólo con orden 1 (singularidad r^(1/2) del estado controlado en esas esquinas); graduada, ~2.
El libro no sirve como referencia numérica (Tabla 6.2 incoherente, ver el notebook §1.1).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from graetz_nucleo import CASES, FEM, OCP, RBFFD, Wells, l2_gamma, ncg_proj, pcg, transfer_control

REF = Path(__file__).with_name("graetz_referencia.json")
N_REF = (80, 160, 320)
Q_REF = 2.0     # graduación de la malla FEM de referencia (y de la comparación rápida)
TOL_CG = 1e-8   # tolerancia del CG para la referencia (error de optimización despreciable)
TOL_BENCH = 1e-6  # tolerancia del CG en el benchmark: con 1e-3, J* queda ~0.5 % por encima del óptimo discreto


def richardson(v):
    """Extrapolación con tres niveles de razón 2: orden observado p y valor límite."""
    v0, v1, v2 = v
    p = np.log2(abs((v1 - v0) / (v2 - v1)))
    lim = v2 + (v2 - v1) / (2**p - 1)
    return float(lim), float(p), float(abs(lim - v2))


def graetz_case(disc, c, tol=TOL_BENCH):
    o, z = CASES[c]
    ocp = OCP(disc, o, z)
    J0 = ocp.J(np.zeros(len(disc.ic)))
    ocp.nsolves = 0
    u, h, _ = pcg(ocp, tol=tol)
    return dict(J0=J0, J=h[-1], it=len(h) - 1, solves=ocp.nsolves, u=u, ocp=ocp)


def calcular_referencia():
    """FEM P1 graduado en N_REF, CG a tol 1e-8, pozos (retícula uniforme: sin esquinas singulares) con CG
    proyectado; guarda los niveles, el límite de Richardson y el contraste con q = 3."""
    niv = {c: {"J0": [], "J": [], "it": []} for c in CASES}
    pozos = {"J": [], "u": []}
    for n in N_REF:
        t = time.time()
        f = FEM(n, q=Q_REF)
        for c in CASES:
            r = graetz_case(f, c, tol=TOL_CG)
            for k in ("J0", "J", "it"):
                niv[c][k].append(float(r[k]))
        del f
        wf = Wells(FEM(n, kind="canal", bc="libre"))
        u, h = ncg_proj(wf, np.zeros(4), tol=1e-8)
        pozos["J"].append(float(h[-1])); pozos["u"].append(u.tolist())
        del wf
        print(f"  FEM n={n}: {time.time() - t:.1f}s", flush=True)
    f = FEM(N_REF[-1], q=3.0)
    q3 = {c: float(graetz_case(f, c, tol=TOL_CG)["J"]) for c in CASES}
    del f
    ref = {"fuente": f"FEM P1 graduado q={Q_REF:g}, n = " + ", ".join(map(str, N_REF))
                     + ", Richardson con orden observado",
           "graetz": {}, "pozos": {}}
    for c in CASES:
        e = {"niveles": niv[c], "J_q3_n320": q3[c]}
        for k in ("J0", "J"):
            lim, p, err = richardson(niv[c][k])
            e[k] = {"valor": lim, "orden": p, "error_estimado_n320": err}
        ref["graetz"][c] = e
    lim, p, err = richardson(pozos["J"])
    U = np.array(pozos["u"])
    ref["pozos"] = {"niveles": pozos, "J": {"valor": lim, "orden": p, "error_estimado_n320": err},
                    "u": [richardson(U[:, i])[0] for i in range(4)]}
    REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print(f"referencia escrita en {REF}")
    return ref


def nivel(h_inv, ref, verbose=True):
    """Corre RBF-FD (fantasmas) y FEM en h = 1/h_inv, con evaluación cruzada y pozos."""
    out = {}
    t0 = time.time()
    g = RBFFD(1 / h_inv); tg = time.time() - t0
    t0 = time.time()
    f = FEM(h_inv, q=Q_REF); tf = time.time() - t0
    if verbose:
        print(f"\nh = 1/{h_inv}: RBF-FD {g.N} nodos (+{g.Nt - g.N} fantasmas) en {tg:.2f}s |"
              f" FEM graduado q={Q_REF:g}: {f.N} vértices en {tf:.2f}s")
        print(f"{'caso':4s} {'J*_ref':>10s} | {'J*_RBF':>10s} {'err%':>6s} {'it':>3s} {'solv':>4s} | {'J*_FEM':>10s} {'err%':>6s} {'it':>3s}"
              f" | {'J_FEM(û_RBF)':>12s} {'Δ%':>6s} {'‖û_RBF-û_FEM‖%':>15s}")
    for c in CASES:
        Jr = ref["graetz"][c]["J"]["valor"]
        t = time.time()
        rg = graetz_case(g, c); trg = time.time() - t
        rf = graetz_case(f, c)
        ux = transfer_control(g, rg["u"], f)
        Jx = rf["ocp"].J(ux)
        du = l2_gamma(f, ux - rf["u"]) / l2_gamma(f, rf["u"])
        out[c] = dict(J_rbf=rg["J"], J_fem=rf["J"], J0_rbf=rg["J0"], J0_fem=rf["J0"], it_rbf=rg["it"], it_fem=rf["it"],
                      solves=rg["solves"], J_cruz=Jx, du=du, t_opt=trg)
        if verbose:
            print(f"({c})  {Jr:10.6f} | {rg['J']:10.6f} {100 * (rg['J'] / Jr - 1):6.2f} {rg['it']:3d} {rg['solves']:4d} |"
                  f" {rf['J']:10.6f} {100 * (rf['J'] / Jr - 1):6.2f} {rf['it']:3d} | {Jx:12.6f}"
                  f" {100 * (Jx / rf['J'] - 1):6.2f} {100 * du:15.1f}")
    t = time.time()
    wp = Wells(RBFFD(1 / h_inv, kind="canal", bc="libre"))
    u, hh = ncg_proj(wp, np.zeros(4))
    out["pozos"] = dict(J=hh[-1], u=u, it=len(hh) - 1, solves=wp.nsolves, t=time.time() - t)
    if verbose:
        rp = ref["pozos"]
        print(f"pozos: J* = {hh[-1]:.6e} (ref {rp['J']['valor']:.6e}, err {100 * (hh[-1] / rp['J']['valor'] - 1):.2f} %)"
              f"  u = {np.round(u, 4)} (ref {np.round(rp['u'], 4)})  it = {len(hh) - 1}  solves = {wp.nsolves}"
              f"  {out['pozos']['t']:.2f}s")
    return out


# Criterios de aceptación del nivel rápido (h = 1/40), en dos capas:
# * exactitud contra la referencia continua (con margen sobre lo medido el 2026-10-27; el error del RBF-FD uniforme
#   está dominado por la singularidad de esquina, ~7 % en el caso a);
# * regresión: J* debe coincidir con la instantánea congelada (`--congelar`) salvo redondeo. Cualquier cambio en la
#   discretización o en el optimizador la rompe, a propósito.
CRITERIOS = {
    "err_J_rbf": 0.10,     # |J*_RBF/J*_ref - 1|
    "err_J_fem": 0.02,     # |J*_FEM/J*_ref - 1|   (FEM graduado q=2)
    "cruz": 0.03,          # J_FEM(û_RBF)/J*_FEM - 1   (el óptimo RBF-FD es bueno para el FEM; óptimo espurio => >>)
    "it_cg": 30,           # iteraciones de CG (tol 1e-6)
    "err_J_pozos": 0.02,   # |J*_pozos/J_ref - 1|
    "err_u_pozos": 0.02,   # max_i |u_i - u_i,ref| / max u_ref
    "regresion": 1e-8,     # |J*/J*_congelado - 1|
}


def verificar(r, ref):
    fallos = []
    for c in CASES:
        Jr = ref["graetz"][c]["J"]["valor"]
        x = r[c]
        if abs(x["J_rbf"] / Jr - 1) > CRITERIOS["err_J_rbf"]: fallos.append(f"({c}) J*_RBF")
        if abs(x["J_fem"] / Jr - 1) > CRITERIOS["err_J_fem"]: fallos.append(f"({c}) J*_FEM")
        if not (-1e-12 <= x["J_cruz"] / x["J_fem"] - 1 <= CRITERIOS["cruz"]): fallos.append(f"({c}) evaluación cruzada")
        if x["it_rbf"] > CRITERIOS["it_cg"]: fallos.append(f"({c}) iteraciones CG")
    rp, p = ref["pozos"], r["pozos"]
    if abs(p["J"] / rp["J"]["valor"] - 1) > CRITERIOS["err_J_pozos"]: fallos.append("pozos J*")
    if np.abs(p["u"] - np.array(rp["u"])).max() / max(rp["u"]) > CRITERIOS["err_u_pozos"]: fallos.append("pozos u*")
    snap = ref.get("regresion_h40")
    if snap is None:
        fallos.append("sin instantánea de regresión (correr --congelar)")
    else:
        for c in CASES:
            for k in ("J_rbf", "J_fem"):
                if abs(r[c][k] / snap[c][k] - 1) > CRITERIOS["regresion"]: fallos.append(f"({c}) regresión {k}")
        if abs(p["J"] / snap["pozos"] - 1) > CRITERIOS["regresion"]: fallos.append("pozos regresión")
    return fallos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--referencia", action="store_true", help="recalcular la referencia FEM (minutos)")
    ap.add_argument("--convergencia", action="store_true", help="niveles h = 1/20, 1/40, 1/80")
    ap.add_argument("--congelar", action="store_true", help="guardar la instantánea de regresión de h = 1/40")
    a = ap.parse_args()
    if a.referencia or not REF.exists():
        calcular_referencia()
    ref = json.loads(REF.read_text())
    if a.congelar:
        r = nivel(40, ref, verbose=False)
        ref["regresion_h40"] = {c: {k: float(r[c][k]) for k in ("J_rbf", "J_fem")} for c in CASES}
        ref["regresion_h40"]["pozos"] = float(r["pozos"]["J"])
        REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
        print("instantánea de regresión (h = 1/40) guardada")
    print("Referencia (" + ref["fuente"] + "):")
    for c, e in ref["graetz"].items():
        print(f"  ({c}) J(u0) = {e['J0']['valor']:.6f}  J* = {e['J']['valor']:.6f} ± {e['J']['error_estimado_n320']:.1e}"
              f"  (orden observado {e['J']['orden']:.2f}; q=3, n=320: {e['J_q3_n320']:.6f})")
    t = time.time()
    if a.convergencia:
        for n in (20, 40, 80):
            nivel(n, ref)
    else:
        r = nivel(40, ref)
        fallos = verificar(r, ref)
        print("\nRESULTADO:", "OK" if not fallos else "FALLA en " + ", ".join(fallos))
    print(f"tiempo total: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
