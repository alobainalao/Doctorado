"""
Gradiente adjunto vs diferencias finitas del modelo sin W- y con almacenamiento
(docs/reanalisis_adr_mrmt.tex), en las 6 variantes: {adr, mrmt_semi, mrmt_block} × {bfr, fenicsx}.

A diferencia de tests/test_taylor_adjoint.py, la CI es C = 1 uniforme (`C_init_override`):
en el dominio real la pluma de funcion.h5 no llega al pozo en horizontes cortos, J_m = 0 y el
test sólo ejercita la parte económica del gradiente. Aquí J_m ≠ 0 y se ejercitan el adjunto del
transporte, el del flujo y la fuente de almacenamiento -S_s ∂ₜ(C ψ_C).

Para cada dirección d se compara  <∇J, d>  con  [J(x+εd) − J(x−εd)] / 2ε  (error relativo).
Con `--sin-almacenamiento` se repite con factor_almacenamiento_C = 0 (el adjunto sin la fuente).

Uso (un modelo por proceso; ver NOTA de tests/test_taylor_adjoint.py):
    python -m tests.test_gradiente_modelo_nuevo bfr adr            # bfr_env
    python -m tests.test_gradiente_modelo_nuevo fenicsx mrmt_semi  # fenics_env
Salida: una línea JSON por dirección en data/output/verificacion_adjunto/<backend>_<modelo>.json
"""
import json
import os
import sys

import numpy as np

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters

BASE = {
    "run_type": "optimization", "domain": "real", "dt": 36000, "T": 3 * 36000,
    "save_dat": False, "animate": False, "pre": True,
    "extract_C": False, "factor_sumidero_C": 0.0,
}
BASE_BFR = {**BASE, "metodo": "bfr", "spacing": 30, "save_dat": True}
BASE_FEM = {**BASE, "metodo": "fenicsx", "spacing": 120, "postproc": False}
MRMT = {"Nr": 2, "Deff": [1e-9, 5e-10], "beta": [0.15, 0.10], "phi_im": [0.1, 0.05]}
EPS_FRAC = [1e-2, 3e-3]          # pasos de las diferencias centradas (relativos a la escala)


def _direcciones(x0, grad):
    rng = np.random.default_rng(42)
    nQ = len(x0) - 1
    uni = np.zeros_like(x0); uni[:nQ] = 1.0
    r1 = np.zeros_like(x0); r1[:nQ] = rng.standard_normal(nQ)
    r2 = rng.standard_normal(len(x0))
    out = {"-grad": -grad, "Q uniforme": uni, "Q aleatoria": r1, "(Q,zp) aleatoria": r2}
    return {k: v / np.linalg.norm(v) for k, v in out.items()}


def main(backend, model, factor_alm):
    env = {**(BASE_BFR if backend == "bfr" else BASE_FEM), "model": model,
           "factor_almacenamiento_C": factor_alm}
    if model.startswith("mrmt"):
        env.update(MRMT)
    for a in sys.argv:                  # --nt=N: número de pasos (por defecto 3)
        if a.startswith("--q="):          # caudal de referencia Q_std (por defecto el de la config)
            env["Q_std"] = float(a[4:])
        if a.startswith("--nt="):
            env["T"] = int(a[5:]) * env["dt"]
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()
    p.C_init_override = 1.0
    p.clip_C = "--sin-recorte" not in sys.argv         # el recorte C ≥ 0 no es diferenciable
    p.adj_disp_u = "--sin-dterm" not in sys.argv       # BFR: término ∂_u D en el adjunto del flujo
    p.adj_C_bc_exacta = "--bc-exacta" in sys.argv      # BFR: transpuesta exacta en inlet/outlet
    if "--modulada" in sys.argv:        # C no uniforme: ∇C ≠ 0 y el término advectivo del adjunto trabaja
        p.C_init_modulacion = (0.5, 2000.0, 800.0)
    p.factor_almacenamiento_C = factor_alm
    p.factor_sumidero_C = 0.0
    p.extract_C = False

    Q0 = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0 = np.concatenate([Q0, [zp0]])

    def set_gamma(g):
        p.gamma = float(g)

    if backend == "bfr":
        from preprocessing.preprocess import load_data
        from view.animation import functional, gradient
        d = load_data()
        p.koppa = 0.0
        J = lambda x: functional(x, d, p, animate=False, save_data=True)
        G = lambda x: gradient(x, d, p)
    else:
        from fenicsx.mesh import get_mesh
        from fenicsx.adjoint import build_context, functional_fenicsx, gradient_fenicsx
        p.koppa = 0.0
        ctx = build_context(list(p.pozo), p, get_mesh(list(p.pozo), p.spacing, regenerate=True))
        J = lambda x: functional_fenicsx(x, ctx)
        G = lambda x: gradient_fenicsx(x, ctx)

    # γ: el gradiente del costo económico (z_p, Q) domina a γ J_m con γ = 1 y el test dejaría de
    # ver el adjunto del transporte. Se elige γ para que γ J_m sea 100 veces el costo económico.
    set_gamma(0.0)
    J_eco = J(x0)
    set_gamma(1.0)
    J_m_raw = J(x0) - J_eco
    set_gamma(100.0 * max(abs(J_eco), 1e-30) / max(abs(J_m_raw), 1e-300))
    print(f"[gamma] J_eco={J_eco:.3e}  J_m(gamma=1)={J_m_raw:.3e}  gamma={p.gamma:.3e}", flush=True)

    z0 = float(getattr(p, "z0", 0.0))
    dt = float(p.dt)

    def eco(x):                        # J_op = |z_p - z_0| Σ Q_n² Δt   (koppa = 0)
        return abs(x[-1] - z0) * float(np.sum(x[:-1] ** 2)) * dt

    def eco_grad(x):
        g = np.zeros_like(x)
        g[:-1] = 2.0 * abs(x[-1] - z0) * x[:-1] * dt
        g[-1] = np.sign(x[-1] - z0) * float(np.sum(x[:-1] ** 2)) * dt
        return g

    # Se prueba sólo γ J_m: se resta la parte económica (analítica) del funcional y del gradiente.
    # (si J(γ=0) ≠ eco(x0) el costo del código no es el supuesto y el test se detiene)
    set_gamma(0.0)
    assert abs(J(x0) - eco(x0)) <= 1e-9 * max(abs(eco(x0)), 1.0), (J(x0), eco(x0))
    set_gamma(100.0 * max(abs(J_eco), 1e-30) / max(abs(J_m_raw), 1e-300))
    Jm = lambda x: J(x) - eco(x)
    J0 = Jm(x0)
    g0 = np.asarray(G(x0), float) - eco_grad(x0)
    # paso de las diferencias finitas: relativo a Q0 (y 1 m en z_p)
    scale = max(float(np.abs(Q0).max()), 1e-12)
    filas = []
    eps_frac = EPS_FRAC[:1] if "--rapido" in sys.argv else EPS_FRAC   # --rapido: un paso, sin -grad
    for nombre, dvec in _direcciones(x0, g0).items():
        if "--rapido" in sys.argv and nombre == "-grad":
            continue
        # z_p se mide en metros: se perturba con paso propio para no mover el pozo cientos de metros
        paso = np.where(np.arange(len(x0)) < len(x0) - 1, scale, 1.0)
        for ef in eps_frac:
            eps = ef
            xp, xm = x0 + eps * paso * dvec, x0 - eps * paso * dvec
            fd = (Jm(xp) - Jm(xm)) / (2 * eps)
            ad = float(g0 @ (paso * dvec))
            err = abs(ad - fd) / max(abs(fd), 1e-300)
            filas.append(dict(backend=backend, model=model, factor_alm=factor_alm, direccion=nombre,
                              eps=eps, adjunto=ad, fd=fd, err_rel=err, J0=J0))
            print(f"{backend}/{model} alm={factor_alm} {nombre:18s} eps={eps:.0e}  "
                  f"adj={ad:+.5e}  fd={fd:+.5e}  err={err:.2e}", flush=True)

    out = os.path.join(_CODIGO, "data", "output", "verificacion_adjunto")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{backend}_{model}_alm{int(factor_alm)}{'_mod' if '--modulada' in sys.argv else ''}{'_bcx' if '--bc-exacta' in sys.argv else ''}{'_sindterm' if '--sin-dterm' in sys.argv else ''}.json"), "w") as f:
        json.dump(filas, f, indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], 0.0 if "--sin-almacenamiento" in sys.argv else 1.0)
