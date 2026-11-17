"""
Funcional, gradiente y optimización de BFR sobre x = [Q_1..Q_Nt, z_p].

    functional(x, d, p)    Ĵ (funtions/utils.py:compute_functional) del directo con el control x
    gradient(x, d, p)      ∇Ĵ por el adjunto discreto exacto (funtions/adjunto_exacto.py)
    optimize_bfr(d, p)     optimizador común con FEniCSx (funtions/optimizador.py)

El directo es view.animation.solve_forward; functional y gradient en el mismo x comparten un solo directo.
"""
import os
import time

import numpy as np

from funtions.utils import update_pozo, compute_functional


def solve_forward(d, Qout, animate, save_data):
    from view.animation import solve_forward as _sf   # import perezoso: view.animation reexporta este módulo
    return _sf(d, Qout, animate, save_data)


def pack_controls(Q, zp):

    return np.concatenate([
        Q.ravel(),
        np.array([zp])
    ])

def unpack_controls(x):

    Q = x[:-1]

    zp = x[-1]

    return Q, zp

# Historias del último directo por control: functional y gradient en el mismo x (trust-constr los pide así)
# comparten un solo directo, sin escribir ni releer el npz.
_ULTIMO = {"clave": None, "hist": None}


def _directo(x, d, p, animate=False, save_data=False):
    """Directo con el control x = [Q, z_p]; devuelve las historias {H, U, C[, Ci, Csol]} (niveles 1..N_t)."""
    Q, zp = unpack_controls(np.asarray(x, float))
    clave = (np.asarray(x, float).tobytes(), p.model, id(d))
    if _ULTIMO["clave"] == clave and not animate and not save_data:
        return _ULTIMO["hist"]
    pozo_cor = np.array(p.pozo, dtype=float)   # p.pozo puede ser entero: sin esto z_p se truncaba a metros enteros
    pozo_cor[1] = zp
    d = update_pozo(d, pozo_cor)
    hist = solve_forward(d, [Q.copy()], animate, save_data=save_data)
    _ULTIMO.update(clave=clave, hist=hist)
    return hist


def functional(x, d, p, animate=False, save_data=False):
    Q, zp = unpack_controls(np.asarray(x, float))
    hist = _directo(x, d, p, animate, save_data)
    return compute_functional(Q, zp, hist["C"], d)


def gradient(x, d, p, animate=False, save_data=False):
    """∇Ĵ por el adjunto discreto exacto (funtions/adjunto_exacto.py): derivada exacta del esquema programado."""
    if getattr(p, "upwind_transport", False):
        raise NotImplementedError("gradiente BFR: el adjunto exacto no cubre el transporte upwind")
    from funtions.adjunto_exacto import gradiente_exacto
    Q, zp = unpack_controls(np.asarray(x, float))
    hist = _directo(x, d, p, animate, save_data)
    gQ, gzp = gradiente_exacto(d, p, Q, zp, hist["H"], hist["U"], hist["C"], Ci=hist.get("Ci"),
                               Csol=hist.get("Csol"))
    return pack_controls(gQ, gzp)


def optimize_bfr(d, p, max_iter=None):
    """Optimización adjunta sobre x = [Q(t), z_p] con el optimizador común (funtions/optimizador.py): variables
    escaladas, trust-constr y paro por la medida de criticidad χ; mismo algoritmo que optimize_fenicsx."""
    import json
    from funtions.optimizador import minimizar
    from funtions.water_supply import storage_trajectory

    Q0 = np.asarray(p.Qout[0], float)
    x0 = pack_controls(Q0, float(p.pozo[1]))
    zp_lo = float(getattr(p, "zp_min", float(d.nodes[:, 1].min()) + float(p.spacing)))
    zp_hi = float(getattr(p, "zp_max", float(d.nodes[:, 1].max()) - float(p.spacing)))
    if max_iter:
        p.opt_maxiter = int(max_iter)

    opt_dir = p.save_data
    os.makedirs(opt_dir, exist_ok=True)
    prog_path = os.path.join(opt_dir, "progress.jsonl")
    open(prog_path, "w").close()
    history = {"Q": [], "zp": [], "J": [], "grad_norm": []}
    t0 = time.time()

    def _log(msg):
        print(msg, flush=True)

    def fun(x):
        t_s = time.time()
        J = functional(x, d, p)
        history["Q"].append(x[:-1].copy()); history["zp"].append(float(x[-1])); history["J"].append(float(J))
        with open(prog_path, "a") as fh:
            fh.write(json.dumps({"eval": len(history["J"]), "J": float(J), "zp": float(x[-1]),
                                 "Q_mean": float(x[:-1].mean()), "iter_s": round(time.time() - t_s, 1),
                                 "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        _log(f"[bfr-opt] J={J:.6e}  z_p={x[-1]:.1f}m  Q̄={x[:-1].mean():.3e}  ({time.time() - t_s:.0f}s)")
        return J

    def jac(x):
        g = gradient(x, d, p)
        history["grad_norm"].append(float(np.linalg.norm(g)))
        _log(f"[bfr-opt grad #{len(history['grad_norm'])}] z_p={x[-1]:.1f}m  ‖∇Ĵ‖={history['grad_norm'][-1]:.3e}")
        return g

    res, chi = minimizar(fun, jac, x0, p, zp_lo, zp_hi, log=_log,
                         progreso=getattr(p, "opt_progreso", os.path.join(opt_dir, "progreso")))
    x_opt = res.x
    S_opt = storage_trajectory(x_opt[:-1], p)
    _log(f"[bfr-opt] terminado: J={res.fun:.6e}  success={res.success}  nit={res.nit}  msg={res.message}")
    _log(f"[bfr-opt] z_p*={x_opt[-1]:.2f}m  S∈[{S_opt.min():.2f}, {S_opt.max():.2f}] m³")
    np.savez(f"{opt_dir}/optimization_results.npz",
             Q=np.asarray(history["Q"]), zp=np.asarray(history["zp"]), J=np.asarray(history["J"]),
             grad_norm=np.asarray(history["grad_norm"]), chi=chi,
             x_opt=x_opt, J_opt=res.fun, S_opt=S_opt)
    _log(f"[bfr-opt] historial guardado ✔  {opt_dir}/optimization_results.npz")
    try:
        from scripts.opt_animation import render_opt_animation
        render_opt_animation(np.asarray(history["Q"]), np.asarray(history["zp"]), np.asarray(history["J"]),
                             dt=p.dt, fig_dir=opt_dir, title=f"BFR · {p.domain}/{p.model}")
    except Exception as e:
        _log(f"[bfr-opt] aviso: no se pudo generar opt_animation ({e})")
    return res
