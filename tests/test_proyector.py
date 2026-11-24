"""
Pruebas del proyector sobre U_ad y del paro por χ (funtions/optimizador.py: proyectar, residuo_proyeccion, minimizar),
con el conjunto admisible de config/nitrato_vscale_malla_eje_g1_sin_nitrato.json en variables escaladas:

 1. certificado: el residuo KKT de P_U(y) es ~0 para y lejos de U, y crece si se perturba la proyección o se la
    saca de U; un punto admisible es su propia proyección;
 2. proyección al interior (m > 0): el resultado queda dentro de U contraído y su residuo es ~0;
 3. (--completo, ~3 min) minimizar con Ĵ = J_inst + J_op explícito (caso sin nitrato, J_m ≡ 0, sin EDP): para por
    χ < tol y llega al óptimo exacto, z_p* = z_max y Q* = argmin Σ Q_n², que es P_U(0, z_max).

    python -m tests.test_proyector [--completo]
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

from funtions.optimizador import conjunto_admisible, minimizar, pesos_J, proyectar, residuo_proyeccion


def caso():
    p = SimpleNamespace(**json.load(open(os.path.join(_CODIGO, "config", "nitrato_vscale_malla_eje_g1_sin_nitrato.json"))))
    p.C_ref = 1.0                                     # J_m ≡ 0: w1 no interviene
    nt = int(round(p.T / p.dt))
    lb, ub, lin = conjunto_admisible(p, nt, p.zp_min, p.zp_max)
    s = np.concatenate([np.full(nt, p.Q_max), [p.zp_max - p.zp_min]])
    return p, nt, s, lb / s, ub / s, [(A * s[None, :], l, u) for A, l, u in lin]


def prueba_certificado(lb, ub, lin):
    rng = np.random.default_rng(0)
    peor = 0.0
    for _ in range(5):
        y = rng.uniform(lb - 0.5, ub + 0.5)
        z, r = proyectar(y, lb, ub, lin)
        peor = max(peor, r)
        assert residuo_proyeccion(y, z + 1e-3 * rng.standard_normal(len(z)), lb, ub, lin) > 1e-4
    assert peor < 1e-8, peor
    assert residuo_proyeccion(y, np.clip(y, lb, ub), lb, ub, lin) > 1e-4      # sólo cotas: viola el tanque
    z2, r2 = proyectar(z, lb, ub, lin)
    assert r2 < 1e-8 and np.max(np.abs(z2 - z)) < 1e-8
    print(f"1. certificado: residuo de P_U(y) ≤ {peor:.1e}; perturbada o sin tanque > 1e-4; P_U(z) = z  OK")


def prueba_interior(lb, ub, lin, m=1e-3):
    z, r = proyectar(np.r_[np.zeros(len(lb) - 1), lb[-1] - 1.0], lb, ub, lin, m)
    margen = min(np.min(np.r_[z - lb, ub - z] / np.r_[ub - lb, ub - lb]),
                 *[np.min(np.r_[A @ z - l, u - A @ z] / np.r_[u - l, u - l]) for A, l, u in lin])
    assert r < 1e-8 and margen > m * (1 - 1e-6), (r, margen)
    print(f"2. proyección al interior (m = {m:g}): margen relativo {margen:.2e}, residuo {r:.1e}  OK")


def prueba_completa(p, nt, s, lb, ub, lin):
    _, w2, w3 = pesos_J(p, nt * p.dt)

    def fun(x):
        d = abs(x[-1] - p.z0)
        return w2 * p.koppa * d ** 2 + w3 * x[:-1] @ x[:-1] * p.dt * d

    def jac(x):
        Q, d = x[:-1], abs(x[-1] - p.z0)
        return np.r_[2 * w3 * Q * p.dt * d, 2 * w2 * p.koppa * (x[-1] - p.z0) + w3 * Q @ Q * p.dt * np.sign(x[-1] - p.z0)]

    z, r = proyectar(np.r_[np.zeros(nt), ub[-1]], lb, ub, lin)          # óptimo exacto: Q* = P_U(0) con z_p = z_max
    assert r < 1e-8, r
    J_ex = fun(s * z)
    p.opt_maxiter = 300
    x0 = np.r_[np.minimum(ub[:-1] * s[:-1], 0.5 * p.Q_max), -700.0]
    res, chi = minimizar(fun, jac, x0, p, p.zp_min, p.zp_max, log=lambda m: None)
    assert res.success and "KKT" in res.message, res.message
    assert abs(res.x[-1] - p.zp_max) < 0.5 and abs(res.fun - J_ex) / J_ex < 1e-4, (res.x[-1], res.fun, J_ex)
    print(f"3. minimizar: {res.message} en {res.nit} it, Ĵ = {res.fun:.6f} (exacto {J_ex:.6f}), z_p = {res.x[-1]:.2f}  OK")


if __name__ == "__main__":
    p, nt, s, lb, ub, lin = caso()
    prueba_certificado(lb, ub, lin)
    prueba_interior(lb, ub, lin)
    if "--completo" in sys.argv:
        prueba_completa(p, nt, s, lb, ub, lin)
