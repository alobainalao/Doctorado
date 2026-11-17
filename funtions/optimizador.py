"""
Optimizador común de BFR y FEniCSx sobre x = [Q_1..Q_Nt, z_p] (tesis, Cap. III, Algoritmo y criterio de paro).

    res, chi = minimizar(fun, jac, x0, p, zp_lo, zp_hi, log=print, progreso=None)

Mismo algoritmo en los dos backends:
  * conjunto admisible U_ad: 0 ≤ Q_n ≤ Q_max·a(τ_n) (pump_upper_bounds), z_lo ≤ z_p ≤ z_hi, tanque 0 ≤ S_n ≤ S_max
    (supply_constraint) y ciclo de bomba (pump_duty_constraint);
  * variables escaladas ξ = x/s, s = [Q_max, …, Q_max, z_hi − z_lo]: todas de orden 1, sin depender del gradiente;
  * trust-constr (punto interior, BFGS) con el gradiente adjunto;
  * paro por la medida de criticidad χ(ξ) = ‖ξ − P_U(ξ − ∇Ĵ(ξ))‖∞ < opt_chi_tol con el iterado admisible. El gtol de
    trust-constr usa multiplicadores de mínimos cuadrados sin signo y, con más filas de restricción que variables, se
    cumple en puntos no estacionarios: se deja en 1e-12;
  * progreso (opcional): el iterado aceptado se escribe en <progreso>.json en cada iteración, y si existe
    <progreso>.stop la optimización termina en la siguiente iteración con el resultado guardado como siempre.
"""
import json
import os

import numpy as np


def conjunto_admisible(p, nQ, zp_lo, zp_hi):
    """(lb, ub, [(A, l, u), …]) de U_ad en variables físicas."""
    from funtions.water_supply import supply_constraint, pump_upper_bounds, pump_duty_constraint
    n = nQ + 1
    Q_max = float(getattr(p, "Q_max", 1e-2))
    lb = np.concatenate([np.zeros(nQ), [zp_lo]])
    ub = np.concatenate([pump_upper_bounds(p, nQ, Q_max), [zp_hi]])
    lin = [supply_constraint(p, n)]
    duty = pump_duty_constraint(p, n, Q_max)
    if duty is not None:
        lin.append(duty)
    return lb, ub, lin


def minimizar(fun, jac, x0, p, zp_lo, zp_hi, log=print, progreso=None):
    """fun, jac en variables físicas. Devuelve (res, chi) con res.x físico y chi la historia de χ por iteración."""
    from scipy.optimize import minimize, Bounds, LinearConstraint

    x0 = np.asarray(x0, float)
    nQ = len(x0) - 1
    Q_max = float(getattr(p, "Q_max", 1e-2))
    lb, ub, lin_fis = conjunto_admisible(p, nQ, zp_lo, zp_hi)
    s = np.concatenate([np.full(nQ, Q_max), [zp_hi - zp_lo]])
    lin = [(A * s[None, :], l, u) for A, l, u in lin_fis]
    lb_s, ub_s = lb / s, ub / s
    chi_tol = float(getattr(p, "opt_chi_tol", 1e-4))

    def proyectar(y):
        """P_U(y): proyección euclídea sobre U_ad escalado (QP pequeño, sin simulaciones)."""
        r = minimize(lambda z: 0.5 * np.dot(z - y, z - y), np.clip(y, lb_s, ub_s), jac=lambda z: z - y,
                     method="SLSQP", bounds=list(zip(lb_s, ub_s)),
                     constraints=[c for A_, l_, u_ in lin for c in (
                         {"type": "ineq", "fun": lambda z, A_=A_, l_=l_: A_ @ z - l_, "jac": lambda z, A_=A_: A_},
                         {"type": "ineq", "fun": lambda z, A_=A_, u_=u_: u_ - A_ @ z, "jac": lambda z, A_=A_: -A_})],
                     options={"ftol": 1e-14, "maxiter": 500})
        return r.x

    chi_hist = []
    parado = [False]
    if progreso and os.path.exists(progreso + ".stop"):
        os.remove(progreso + ".stop")

    def criticidad(xk, state):
        chi = float(np.max(np.abs(xk - proyectar(xk - state.grad))))
        chi_hist.append(chi)
        log(f"[opt it {state.nit}] χ={chi:.3e} (tol {chi_tol:.0e})  Ĵ={state.fun:.6e}  "
            f"violación={state.constr_violation:.1e}")
        if progreso:
            with open(progreso + ".json", "w") as fh:
                json.dump(dict(nit=int(state.nit), x=(s * xk).tolist(), J=float(state.fun), chi=chi,
                               violacion=float(state.constr_violation)), fh)
            if os.path.exists(progreso + ".stop"):
                log(f"[opt] paro pedido ({progreso}.stop)")
                parado[0] = True
                return True
        return chi < chi_tol and state.constr_violation < 1e-8

    opts = {"maxiter": int(getattr(p, "opt_maxiter", 20)), "gtol": 1e-12, "xtol": 1e-8}
    res = minimize(lambda xi: fun(s * xi), x0 / s, jac=lambda xi: np.asarray(jac(s * xi)) * s,
                   method="trust-constr", bounds=Bounds(lb_s, ub_s),
                   constraints=[LinearConstraint(A_, l_, u_) for A_, l_, u_ in lin],
                   callback=criticidad, options=opts)
    res.x = s * res.x
    if parado[0]:
        res.success, res.message = False, "paro pedido por el usuario"
    elif res.status == 3:
        res.success, res.message = True, f"χ < {chi_tol:g} (KKT)"
    return res, np.asarray(chi_hist)
