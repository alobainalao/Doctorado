"""
Solución esperada de los casos de validación de la optimización (experimentos/optimizar_nitrato.py), sin resolver EDP:

- sin_nitrato (fuente apagada, C inicial 0): J_m ≡ 0, así que Ĵ = J_inst + J_op y las dos crecen con d_p = z_0 − z_p:
  z_p* = z_max y Q* minimiza Σ Q_n² con las restricciones lineales del tanque, la bomba y su ciclo (problema cuadrático).
- sin_demanda (d_n = 0): Q* = 0 (bombear solo cuesta y el tanque queda en S_0, dentro de [0, S_max]); z_p* depende de J_m.

    python -m experimentos.validacion_esperada <variante_base>        # bfr_env o fenics_env

Salida: data/output/optimizacion_<variante_base>_sin_nitrato/esperado.json (z_p*, Q*, J2, J3, S*).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)

from funtions import water_supply as ws


def main(variante):
    p = SimpleNamespace(**json.load(open(os.path.join(CODIGO, "config", f"{variante}_sin_nitrato.json"))))
    nt = int(round(p.T / p.dt))
    n = nt + 1
    zp = float(p.zp_max)                                          # J_inst y J_op crecen con la profundidad
    dz = max(abs(p.zp_min - p.z0), abs(p.zp_max - p.z0))           # pesos de normalizar_J (fenicsx/adjoint.py, pesos_J)
    w2, w3 = 1.0 / dz ** 2, 1.0 / (p.Q_max ** 2 * nt * p.dt * dz)
    ub = ws.pump_upper_bounds(p, nt, p.Q_max)
    A, lb, ubs = ws.supply_constraint(p, n)
    cons = [LinearConstraint(A[:, :-1], lb - A[:, -1] * zp, ubs - A[:, -1] * zp)]
    duty = ws.pump_duty_constraint(p, n, p.Q_max)
    if duty is not None:
        Ad, ld, ud = duty
        cons.append(LinearConstraint(Ad[:, :-1], ld - Ad[:, -1] * zp, ud - Ad[:, -1] * zp))
    r = minimize(lambda q: float(q @ q), np.minimum(ub, np.full(nt, 0.5 * p.Q_max)), jac=lambda q: 2 * q,
                 method="trust-constr", bounds=Bounds(np.zeros(nt), ub), constraints=cons,
                 options={"maxiter": 2000, "gtol": 1e-12, "xtol": 1e-14})
    Q = r.x
    d = abs(zp - p.z0)
    out = dict(caso="sin_nitrato", zp=zp, Q=np.round(Q, 6).tolist(), J1=0.0, J2=w2 * p.koppa * d ** 2,
               J3=w3 * float(Q @ Q) * p.dt * d, S=[p.S0] + ws.storage_trajectory(Q, p).round(1).tolist(),
               violacion=float(r.constr_violation), estado=r.message)
    out["J"] = out["J2"] + out["J3"]
    dir_out = os.path.join(CODIGO, "data", "output", f"optimizacion_{variante}_sin_nitrato")
    os.makedirs(dir_out, exist_ok=True)
    with open(os.path.join(dir_out, "esperado.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"z_p* = {zp} m  J2 = {out['J2']:.4f}  J3 = {out['J3']:.4f}  J = {out['J']:.4f}  "
          f"Q medio = {Q.mean():.4f}  S en [{min(out['S']):.1f}, {max(out['S']):.1f}]  violación {r.constr_violation:.1e}")


if __name__ == "__main__":
    main(sys.argv[1])
