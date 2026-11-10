"""
Optimización adjunta (FEniCSx) con variables adimensionales y reinicios, para que arranques distintos
lleguen al mismo óptimo.

    python -m experimentos.optimizar_escalado <modelo> <variante> <zp0> <Q0/Q_max> <etiqueta>

Diferencias con fenicsx.adjoint.optimize_fenicsx (que no se modifica):
  - y = [Q/Q_max, z_p/1000]: Q (~0.1) y z_p (~300 m) quedan en la misma escala, así `gtol` mide ambos;
  - J y ∇J se calculan juntos (un directo + un adjunto por punto) y se reusan entre fun y jac;
  - reinicios desde el punto final mientras J baje más de 1e-4 relativo (hasta 4);
  - al final se informa el gradiente proyectado sobre las restricciones activas y las cotas.
Mismas cotas y restricciones (suministro, ciclo de bomba) que optimize_fenicsx. C_ref y f salen de
data/output/optimizacion_<variante>/adr.json (misma normalización que optimizar_nitrato).

Entorno: fenics_env. Salida: data/output/optimizacion_<variante>/escalado_<modelo>_<etiqueta>.json
"""
import json
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
os.chdir(CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters

ESC_Z = 1000.0


def main(modelo, variante, zp0, q0, etiqueta):
    from scipy.optimize import minimize, Bounds, LinearConstraint
    dir_out = os.path.join(CODIGO, "data", "output", "optimizacion_" + variante)
    ref = json.load(open(os.path.join(dir_out, "adr.json")))
    cfg = json.load(open(os.path.join(CODIGO, "config", variante + ".json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": modelo, "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    p.C_ref = ref["C_ref_modelo"]
    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context, _solve_forward, _solve_adjoint_grad
    from funtions.water_supply import supply_constraint, pump_duty_constraint, pump_upper_bounds
    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)

    nQ = int(p.Nt)
    Q_max = float(p.Q_max)
    n = nQ + 1
    esc = np.r_[np.full(nQ, Q_max), ESC_Z]                     # x = esc * y
    zlo, zhi = float(p.zp_min), float(p.zp_max)
    bounds = Bounds(np.r_[np.zeros(nQ), zlo / ESC_Z], np.r_[pump_upper_bounds(p, nQ, Q_max) / Q_max, zhi / ESC_Z])
    A, clb, cub = supply_constraint(p, n)
    cons = [LinearConstraint(A * esc[None, :], clb, cub)]
    duty = pump_duty_constraint(p, n, Q_max)
    if duty is not None:
        cons.append(LinearConstraint(duty[0] * esc[None, :], duty[1], duty[2]))

    cache = {}

    def evalua(y):
        k = y.tobytes()
        if cache.get("k") != k:
            x = y * esc
            st = _solve_forward(ctx, x[:-1], float(x[-1]))
            g, _ = _solve_adjoint_grad(ctx, st)
            cache.update(k=k, J=float(st["J"]), g=np.asarray(g, float) * esc)
            hist.append(cache["J"])
        return cache["J"], cache["g"]

    hist = []
    y0 = np.r_[np.full(nQ, q0), zp0 / ESC_Z]
    y = y0.copy()
    rondas = []
    for r in range(4):
        res = minimize(lambda v: evalua(v)[0], y, jac=lambda v: evalua(v)[1], method="trust-constr",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "gtol": 1e-7, "xtol": 1e-10})
        rondas.append(dict(J=float(res.fun), zp=float(res.x[-1] * ESC_Z), nit=int(res.nit), msg=str(res.message)))
        print(f"[escalado] ronda {r}: J={res.fun:.6e} z_p={res.x[-1]*ESC_Z:.1f} nit={res.nit} {res.message}", flush=True)
        mejora = (evalua(y)[0] - res.fun) / max(abs(res.fun), 1e-30)
        y = np.asarray(res.x, float)
        if r > 0 and mejora < 1e-4:
            break

    J, g = evalua(y)
    x = y * esc
    # gradiente proyectado: variables libres (fuera de cotas) y filas activas de las restricciones
    libres = np.r_[(y[:-1] > 1e-9) & (y[:-1] < bounds.ub[:-1] - 1e-9), True]
    filas = []
    for c in cons:
        v = c.A @ y
        for i in range(len(v)):
            if min(v[i] - c.lb[i], c.ub[i] - v[i]) < 1e-9:
                filas.append(np.asarray(c.A[i]).ravel()[libres])
    gl = g[libres]
    if filas:
        M = np.array(filas)
        coef, *_ = np.linalg.lstsq(M.T, gl, rcond=None)
        gp = gl - M.T @ coef
    else:
        gp = gl
    out = dict(modelo=modelo, variante=variante, zp0=zp0, q0=q0, x_opt=x.tolist(), J=J, rondas=rondas,
               norma_g_proyectado=float(np.linalg.norm(gp)), norma_g=float(np.linalg.norm(g)),
               n_filas_activas=len(filas), n_libres=int(libres.sum()), evaluaciones=len(hist))
    json.dump(out, open(os.path.join(dir_out, f"escalado_{modelo}_{etiqueta}.json"), "w"), indent=1)
    print(f"FIN {etiqueta}: J={J:.6e} z_p={x[-1]:.1f} ‖g_proy‖={out['norma_g_proyectado']:.3e} evals={len(hist)}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]), float(a[3]), a[4])
