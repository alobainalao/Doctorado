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
  * P_U por SLSQP certificada con su residuo KKT (proyectar): si la proyección no es exacta, χ no es fiable, se avisa
    y no se para;
  * progreso (opcional): el iterado aceptado se escribe en <progreso>.json en cada iteración, y si existe
    <progreso>.stop la optimización termina en la siguiente iteración con el resultado guardado como siempre.
"""
import json
import os

import numpy as np


def pesos_J(p, T_total):
    """Pesos (w1, w2, w3) de J1, J2, J3 (mismos que fenicsx/adjoint.py:_pesos_J). Sin `normalizar_J` valen 1.
    Con `normalizar_J` cada término se divide entre su máximo posible: J1 entre C_ref²·T (C_ref = límite de salud,
    en unidades de C), J2 entre max|z_p − z0|² sobre [zp_min, zp_max], J3 entre Q_max²·T·max|z_p − z0|."""
    if not getattr(p, "normalizar_J", False):
        return 1.0, 1.0, 1.0
    z0 = float(getattr(p, "z0", 0.0))
    dz = max(abs(float(p.zp_min) - z0), abs(float(p.zp_max) - z0))
    Q_max = float(getattr(p, "Q_max", 1e-2))
    return 1.0 / (float(p.C_ref) ** 2 * T_total), 1.0 / dz ** 2, 1.0 / (Q_max ** 2 * T_total * dz)


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


def residuo_proyeccion(y, z, lb, ub, lin):
    """Error KKT de z como proyección euclídea de y sobre U = {lb ≤ z ≤ ub, l ≤ A z ≤ u}: máximo entre la violación
    relativa de U y ‖(y − z) − Gᵀμ‖∞, con G z ≤ h las restricciones activas en z y μ ≥ 0 sus multiplicadores (NNLS).
    Con U convexo, z = P_U(y) ⇔ residuo 0."""
    from scipy.optimize import nnls
    n = len(z)
    G = np.vstack([-np.eye(n), np.eye(n)] + [M for A, l, u in lin for M in (-A, A)])
    h = np.concatenate([-lb, ub] + [v for A, l, u in lin for v in (-l, u)])
    fin = np.isfinite(h)
    G, h = G[fin], h[fin]
    holgura = (h - G @ z) / (1.0 + np.abs(h))
    act = holgura < 1e-9
    mu = nnls(G[act].T, y - z)[0] if act.any() else np.zeros(0)
    return max(0.0, -float(holgura.min()), float(np.max(np.abs(y - z - G[act].T @ mu))))


def proyectar(y, lb, ub, lin, m=0.0):
    """P_U(y): proyección euclídea sobre U = {lb ≤ z ≤ ub, l ≤ A z ≤ u} (QP pequeño, sin simulaciones); con m > 0, sobre
    U contraído una fracción m del ancho de cada cota (punto interior). Devuelve (z, residuo KKT de z). Se revisa el
    residuo y no r.success: SLSQP llega a declarar fallo (status 8) con la proyección exacta."""
    from scipy.optimize import minimize
    if m:
        lb, ub = lb + m * (ub - lb), ub - m * (ub - lb)
        lin = [(A, l + m * (u - l), u - m * (u - l)) for A, l, u in lin]
    r = minimize(lambda z: 0.5 * np.dot(z - y, z - y), np.clip(y, lb, ub), jac=lambda z: z - y,
                 method="SLSQP", bounds=list(zip(lb, ub)),
                 constraints=[c for A_, l_, u_ in lin for c in (
                     {"type": "ineq", "fun": lambda z, A_=A_, l_=l_: A_ @ z - l_, "jac": lambda z, A_=A_: A_},
                     {"type": "ineq", "fun": lambda z, A_=A_, u_=u_: u_ - A_ @ z, "jac": lambda z, A_=A_: -A_})],
                 options={"ftol": 1e-14, "maxiter": 500})
    return r.x, residuo_proyeccion(y, r.x, lb, ub, lin)


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
    proy_tol = 1e-2 * chi_tol           # error admitido de P_U: 1 % de la tolerancia de χ

    chi_hist = []
    parado = [False]
    if progreso and os.path.exists(progreso + ".stop"):
        os.remove(progreso + ".stop")

    def criticidad(xk, state):
        z, residuo = proyectar(xk - state.grad, lb_s, ub_s, lin)
        chi = float(np.max(np.abs(xk - z)))
        chi_hist.append(chi)
        log(f"[opt it {state.nit}] χ={chi:.3e} (tol {chi_tol:.0e})  Ĵ={state.fun:.6e}  "
            f"violación={state.constr_violation:.1e}")
        if residuo > proy_tol:
            log(f"[opt it {state.nit}] AVISO: proyección no certificada (residuo KKT {residuo:.1e}): χ no es fiable")
        if progreso:
            with open(progreso + ".json", "w") as fh:
                json.dump(dict(nit=int(state.nit), x=(s * xk).tolist(), J=float(state.fun), chi=chi,
                               violacion=float(state.constr_violation), residuo_proy=residuo), fh)
            if os.path.exists(progreso + ".stop"):
                log(f"[opt] paro pedido ({progreso}.stop)")
                parado[0] = True
                return True
        return chi < chi_tol and residuo <= proy_tol and state.constr_violation < 1e-8

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
