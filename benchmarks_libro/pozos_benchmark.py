"""Benchmark de referencia del peldaño 4: canal de Graetz con 4 pozos y tasas de extracción u_i(t) (transitorio).

Uso (desde codigo/benchmarks_libro):
    python3 pozos_benchmark.py                # nivel rápido (h = 1/20): RBF-FD y FEM contra la referencia, OK/FALLA
    python3 pozos_benchmark.py --referencia   # recalcula pozos_referencia.json (FEM n = 20, 40, 80 + Richardson)
    python3 pozos_benchmark.py --congelar     # guarda la instantánea de regresión del nivel rápido

Referencia: FEM P1, Crank–Nicolson con Δt = 1/n, u_i constante a trozos en M = 20 intervalos; refinamiento conjunto
h, Δt por 2 y Richardson con el orden observado. Dos escenarios: libre (u_max = 5, ninguna cota superior activa) y
con cotas (u_max = 0.3, activas en el paso del penacho). El control es un vector de R^{M×K} que no depende de la
malla, así que cada malla arranca del óptimo de la anterior (y el RBF-FD del óptimo FEM). Optimizador: L-BFGS-B con
parada absoluta ‖P(u − ∇J) − u‖_∞ ≤ TOL_PG ‖∇J(0)‖_∞, igual para cualquier arranque. (El CG-PR+ proyectado de los
peldaños 2–3 llega al mismo óptimo pero cuesta 1.8× (libre) a 7× (con cotas) más evaluaciones: ver el notebook, §4.)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pozos_nucleo import FEM, RBFFD, T_FIN, ConstantWells, TransientWells

REF = Path(__file__).with_name("pozos_referencia.json")
N_REF = (20, 40, 80)
M_CTRL = 20
ESCENARIOS = {"libre": 5.0, "cotas": 0.3}
TOL_PG = 1e-8


def problema(disc, n, umax, M=M_CTRL, theta=0.5):
    return TransientWells(disc, T=T_FIN, nt=int(round(T_FIN * n)), M=M, theta=theta, umax=umax)


def fem(n):
    return FEM(n, kind="canal", bc="libre", factor=False)


def rbf(n):
    return RBFFD(1 / n, kind="canal", bc="libre")


def optimo(tw, u0=None, tol=TOL_PG):
    """L-BFGS-B con cotas [0, u_max]; devuelve (u, historia de J por evaluación, nº de evaluaciones)."""
    g0 = getattr(tw, "_g0", None)
    if g0 is None:
        g0 = tw._g0 = np.abs(tw.Jg(tw.zeros())[1]).max()
    tw.nsolves = 0
    hist = []

    def fg(v):
        J, G, _ = tw.Jg(v); hist.append(J); return J, G
    r = minimize(fg, tw.zeros() if u0 is None else np.clip(u0, 0, tw.umax), jac=True, method="L-BFGS-B",
                 bounds=[(0, tw.umax)] * (tw.M * tw.K), options={"gtol": tol * g0, "ftol": 1e-15, "maxiter": 500})
    hist.append(r.fun)
    return r.x, np.array(hist), tw.nsolves


def gproj(tw, u):
    """‖P(u − ∇J) − u‖_∞ relativo a ‖∇J(0)‖_∞: medida de optimalidad independiente del arranque."""
    _, g, _ = tw.Jg(u)
    _, g0, _ = tw.Jg(tw.zeros())
    return float(np.abs(np.clip(u - g, 0, tw.umax) - u).max() / np.abs(g0).max())


def richardson(v):
    v0, v1, v2 = v
    p = np.log2(abs((v1 - v0) / (v2 - v1)))
    lim = v2 + (v2 - v1) / (2**p - 1)
    return float(lim), float(p), float(abs(lim - v2))


def calcular_referencia():
    ref = {"fuente": f"FEM P1, CN, Δt = 1/n, n = {', '.join(map(str, N_REF))}, M = {M_CTRL} intervalos de control, "
                     "Richardson con orden observado", "escenarios": {}}
    for esc, umax in ESCENARIOS.items():
        niv = {"J0": [], "J": [], "Jc": []}
        u = c = None
        e = {}
        for n in N_REF:
            tt = time.time()
            tw = problema(fem(n), n, umax)
            niv["J0"].append(float(tw.J(tw.zeros())))
            u, h, ns = optimo(tw, u)
            niv["J"].append(float(h[-1]))
            c, hc, _ = optimo(ConstantWells(tw), c)
            niv["Jc"].append(float(hc[-1]))
            e[f"u_n{n}"] = u.tolist()
            print(f"  {esc} n={n}: J0={niv['J0'][-1]:.7e} J*={h[-1]:.7e} ({ns} evals) "
                  f"J*_const={hc[-1]:.7e}  {time.time() - tt:.0f}s", flush=True)
            del tw
        for k, v in niv.items():
            lim, p, err = richardson(v)
            e[k] = {"niveles": v, "valor": lim, "orden": p, "error_estimado_n80": err}
        e["u_const_n80"] = c.tolist()
        ua, ub = np.array(e["u_n40"]), np.array(e["u_n80"])
        e["dif_u_n40_n80"] = float(np.linalg.norm(ua - ub) / np.linalg.norm(ub))
        ref["escenarios"][esc] = e
    REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print(f"referencia escrita en {REF}")
    return ref


def nivel(n, ref, verbose=True):
    """FEM desde cero y RBF-FD desde el óptimo FEM en la retícula n, con evaluación cruzada J_FEM(û_RBF)."""
    out = {}
    f, g = fem(n), rbf(n)
    if verbose:
        print(f"\nn = {n}: RBF-FD {g.Nt} incógnitas, FEM {f.N}; {int(T_FIN * n)} pasos, M = {M_CTRL}")
        print(f"{'esc':>5s} | {'J*_ref':>10s} | {'J*_FEM':>10s} {'err%':>5s} {'ev':>3s} | {'J*_RBF':>10s} {'err%':>5s} "
              f"{'ev':>3s} {'gproj':>7s} | {'J_FEM(û_RBF)':>12s} {'Δ%':>5s} | {'‖Δu‖%':>6s} | {'s':>4s}")
    for esc, umax in ESCENARIOS.items():
        tt = time.time()
        r = ref["escenarios"][esc]
        tf, tg = problema(f, n, umax), problema(g, n, umax)
        uf, hf, nf = optimo(tf)
        ug, hg, ng = optimo(tg, uf)
        Jx = tf.J(ug)
        uref = np.array(r["u_n80"])
        du = float(np.linalg.norm(ug - uref) / np.linalg.norm(uref))
        gp = gproj(tg, ug)
        Jr = r["J"]["valor"]
        out[esc] = dict(J_fem=float(hf[-1]), J_rbf=float(hg[-1]), J_cruz=float(Jx), it=nf, du=du, gproj=gp)
        if verbose:
            print(f"{esc:>5s} | {Jr:10.5e} | {hf[-1]:10.5e} {100 * (hf[-1] / Jr - 1):5.2f} {nf:3d} | "
                  f"{hg[-1]:10.5e} {100 * (hg[-1] / Jr - 1):5.2f} {ng:3d} {gp:7.1e} | {Jx:12.5e} "
                  f"{100 * (Jx / hf[-1] - 1):5.2f} | {100 * du:6.2f} | {time.time() - tt:4.0f}")
    return out


# Criterios del nivel rápido (n = 20). err_J: J* contra la referencia (el error de discretización medido es <1 %);
# cruz: J_FEM(û_RBF) no puede bajar del óptimo FEM y debe quedar cerca; du: control contra el de n = 80;
# gproj: optimalidad del óptimo RBF-FD (gradiente proyectado relativo); evals: evaluaciones de (J, ∇J) del FEM.
CRITERIOS = {"err_J": 0.03, "cruz": 0.01, "du": 0.05, "evals": 60, "gproj": 1e-6, "regresion": 1e-6}


def verificar(r, ref):
    fallos = []
    snap = ref.get("regresion_n20")
    for esc, x in r.items():
        Jr = ref["escenarios"][esc]["J"]["valor"]
        if abs(x["J_fem"] / Jr - 1) > CRITERIOS["err_J"]: fallos.append(f"{esc} J*_FEM")
        if abs(x["J_rbf"] / Jr - 1) > CRITERIOS["err_J"]: fallos.append(f"{esc} J*_RBF")
        if not (-1e-6 <= x["J_cruz"] / x["J_fem"] - 1 <= CRITERIOS["cruz"]): fallos.append(f"{esc} cruzada")
        if x["du"] > CRITERIOS["du"]: fallos.append(f"{esc} control")
        if x["it"] > CRITERIOS["evals"]: fallos.append(f"{esc} evaluaciones")
        if x["gproj"] > CRITERIOS["gproj"]: fallos.append(f"{esc} optimalidad RBF-FD")
        if snap is not None and abs(x["J_rbf"] / snap[esc] - 1) > CRITERIOS["regresion"]:
            fallos.append(f"{esc} regresión")
    if snap is None:
        fallos.append("sin instantánea de regresión (correr --congelar)")
    return fallos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--referencia", action="store_true")
    ap.add_argument("--congelar", action="store_true")
    a = ap.parse_args()
    if a.referencia or not REF.exists():
        calcular_referencia()
    ref = json.loads(REF.read_text())
    print("Referencia (" + ref["fuente"] + "):")
    for esc, e in ref["escenarios"].items():
        print(f"  {esc}: " + "   ".join(f"{k}={e[k]['valor']:.6e} ± {e[k]['error_estimado_n80']:.0e} (orden {e[k]['orden']:.2f})"
                                     for k in ("J0", "J", "Jc")))
    tt = time.time()
    r = nivel(20, ref)
    if a.congelar:
        ref["regresion_n20"] = {esc: x["J_rbf"] for esc, x in r.items()}
        REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
        print("instantánea de regresión (n = 20) guardada")
    fallos = verificar(r, ref)
    print("\nRESULTADO:", "OK" if not fallos else "FALLA en " + ", ".join(fallos))
    print(f"tiempo total: {time.time() - tt:.1f}s")


if __name__ == "__main__":
    main()
