"""
Verificación del óptimo de experimentos/optimizar_nitrato.py (FEniCSx).

    python -m experimentos.verificar_optimo <modelo> kkt            # FD en x_opt, restricciones activas, barrido en z_p
    python -m experimentos.verificar_optimo <modelo> arranque <zp0> <Q0> <etiqueta>

Entorno: fenics_env. Salida: data/output/optimizacion_nitrato/verificacion_<modelo>_<...>.json
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

VARIANTE = os.environ.get("VARIANTE", "nitrato")      # config/<VARIANTE>.json; carpeta optimizacion_<VARIANTE>
DIR = os.path.join(CODIGO, "data", "output", "optimizacion_" + VARIANTE)


def preparar(modelo):
    cfg = json.load(open(os.path.join(CODIGO, "config", VARIANTE + ".json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": modelo, "pre": True,
                "save_dat": False, "opt_maxiter": 100, "animate": False, "postproc": False,
                "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    ref = json.load(open(os.path.join(DIR, f"{modelo}.json")))
    p.C_ref = ref["C_ref_modelo"]          # misma normalización que la corrida original
    return p, ref


def main(modelo, modo, args):
    p, ref = preparar(modelo)
    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import (build_context, functional_fenicsx, gradient_fenicsx,
                                 optimize_fenicsx)
    from funtions.water_supply import supply_constraint, pump_duty_constraint, storage_trajectory
    # Igual que main.py: con `mesh_pkl` se usa la malla compartida congelada; si no, la malla gmsh gruesa.
    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)

    if modo == "reinicio":                  # <etiqueta_origen> <etiqueta>: parte del x_opt de otra corrida
        xi = json.load(open(os.path.join(DIR, f"verificacion_{modelo}_arranque_{args[0]}.json")))["x_opt"]
        p.pozo = [p.pozo[0], float(xi[-1])]
        p.Qout = [np.asarray(xi[:-1], float)] + list(p.Qout[1:])
        modo, args = "arranque", [xi[-1], 0.0, args[1]]
    if modo == "arranque":
        zp0, Q0, et = float(args[0]), float(args[1]), args[2]
        if not (len(args) == 3 and args[1] == 0.0 and isinstance(args[0], float) and p.pozo[1] == zp0):
            p.pozo = [p.pozo[0], zp0]
            p.Qout = [np.full(len(np.asarray(p.Qout[0])), Q0)] + list(p.Qout[1:])
        res = optimize_fenicsx(p, mesh_cache=mc)
        out = dict(zp0=zp0, Q0=Q0, x_opt=np.asarray(res.x).tolist(), J=float(res.fun), nit=int(res.nit),
                   success=bool(res.success), msg=str(res.message))
        json.dump(out, open(os.path.join(DIR, f"verificacion_{modelo}_arranque_{et}.json"), "w"), indent=1)
        print("FIN", et, out["J"], out["x_opt"][-1], out["nit"], out["msg"], flush=True)
        return

    ctx = build_context(list(p.pozo), p, mc)
    if modo == "barrido":
        xo = np.asarray(ref["x_opt"], float)
        out = []
        for z in np.arange(-320, -199, 10.0).tolist():
            x = xo.copy(); x[-1] = z
            J = float(functional_fenicsx(x, ctx)); gz = float(gradient_fenicsx(x, ctx)[-1])
            out.append((z, J, gz)); print("zp", z, J, "dJ/dzp", gz, flush=True)
        json.dump(out, open(os.path.join(DIR, f"verificacion_{modelo}_barrido_fino.json"), "w"))
        print("FIN barrido", flush=True)
        return
    xo = np.asarray(ref["x_opt"], float)
    n = len(xo)
    J0 = functional_fenicsx(xo, ctx)
    g = gradient_fenicsx(xo, ctx)
    print(f"J(x_opt)={J0:.6e} ‖g‖={np.linalg.norm(g):.4e} g_zp={g[-1]:.4e}", flush=True)

    # (1) adjunto vs diferencias finitas centradas en x_opt
    rng = np.random.default_rng(0)
    dirs = {"gradiente": g / np.linalg.norm(g), "zp": np.eye(n)[-1],
            "Q_unif": np.r_[np.ones(n - 1) / np.sqrt(n - 1), 0.0],
            "aleatoria": rng.standard_normal(n)}
    dirs["aleatoria"] /= np.linalg.norm(dirs["aleatoria"])
    fd = {}
    for nom, d in dirs.items():
        # paso relativo a la escala de cada variable (Q ~ 4e-4, z_p ~ 1e2)
        esc = np.r_[np.full(n - 1, 1e-4), 1.0]
        for h in (1e-2, 1e-3):
            dd = d * esc / np.linalg.norm(d * esc)
            dx = h * dd
            Jp, Jm = functional_fenicsx(xo + dx * np.r_[np.full(n - 1, 1e-4), 1.0] / 1.0, ctx), None
            Jm = functional_fenicsx(xo - dx * np.r_[np.full(n - 1, 1e-4), 1.0] / 1.0, ctx)
            paso = dx * np.r_[np.full(n - 1, 1e-4), 1.0]
            dJfd = (Jp - Jm) / 2.0
            dJad = float(g @ paso)
            fd[f"{nom}_h{h}"] = dict(fd=dJfd, adj=dJad, rel=abs(dJfd - dJad) / max(abs(dJad), 1e-30))
            print(nom, h, fd[f"{nom}_h{h}"], flush=True)

    # (2) restricciones activas y gradiente proyectado
    A, clb, cub = supply_constraint(p, n)
    val = A @ xo
    act = [("sum_lb", i) for i in range(len(val)) if val[i] - clb[i] < 1e-6 * max(1, abs(clb[i]))] + \
          [("sum_ub", i) for i in range(len(val)) if cub[i] - val[i] < 1e-6 * max(1, abs(cub[i]))]
    duty = pump_duty_constraint(p, n, float(p.Q_max))
    info = dict(n_suministro_activas=len(act), S_min=float(storage_trajectory(xo[:-1], p).min()),
                S_max=float(storage_trajectory(xo[:-1], p).max()))
    Qmax = float(p.Q_max)
    info["Q_en_cota_inf"] = int(np.sum(xo[:-1] <= 1e-9))
    info["Q_en_cota_sup"] = int(np.sum(xo[:-1] >= Qmax - 1e-9))
    if duty is not None:
        Ad, lbd, ubd = duty
        vd = Ad @ xo
        info["duty_activas"] = int(np.sum(np.minimum(vd - lbd, ubd - vd) < 1e-9))
        info["duty_n"] = int(len(vd))
    print("restricciones:", info, flush=True)

    # gradiente proyectado sobre variables libres (Q > cotas) con duty/suministro como igualdad si activas
    libres = np.r_[(xo[:-1] > 1e-9) & (xo[:-1] < Qmax - 1e-9), True]
    G = [np.eye(n)[i] for i in []]
    filas = []
    if act:
        filas += [A[i] for _, i in act]
    if duty is not None:
        for i in range(len(vd)):
            if min(vd[i] - lbd[i], ubd[i] - vd[i]) < 1e-9:
                filas.append(Ad[i])
    gl = g[libres]
    if filas:
        M = np.array([np.asarray(f).ravel()[libres] for f in filas])
        coef, *_ = np.linalg.lstsq(M.T, gl, rcond=None)
        gproj = gl - M.T @ coef
    else:
        gproj = gl
    info["‖g‖"] = float(np.linalg.norm(g))
    info["‖g_libres‖"] = float(np.linalg.norm(gl))
    info["‖g_proyectado‖"] = float(np.linalg.norm(gproj))
    info["n_filas_activas"] = len(filas)
    print("proyectado:", info, flush=True)

    # (3) barrido de J en z_p con Q = Q_opt fijo
    zps = np.arange(-1400, -199, 100.0).tolist() + [float(xo[-1])]
    barrido = []
    for z in sorted(zps):
        x = xo.copy(); x[-1] = z
        barrido.append((z, float(functional_fenicsx(x, ctx))))
        print("zp", z, barrido[-1][1], flush=True)

    json.dump(dict(J_opt=J0, fd=fd, restricciones=info, barrido_zp=barrido),
              open(os.path.join(DIR, f"verificacion_{modelo}_kkt.json"), "w"), indent=1, default=float)
    print("FIN kkt", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
