"""
Checkpoint del gradiente adjunto para modelos MRMT Semi y MRMT Block en el
backend FEniCS (DOLFINx). Mismo protocolo que fenicsx/test_gradient_checkpoint.py
(ADR): compara grad_Q obtenido por el adjunto contra diferencias finitas centradas.

Para correr (requiere el entorno `fenics_env`):
    /home/alex/miniconda3/envs/fenics_env/bin/python -m fenicsx.test_gradient_checkpoint_mrmt_fenicsx [semi|block]
"""
import sys
import numpy as np

from funtions.runtime import RUNTIME
from config.parameters import Parameters

DELTA_FRACS = [1e-1, 1e-2, 1e-3, 1e-4]
TOL = 5e-3

BASE_ENV = {
    "run_type": "optimization",
    "metodo": "fenicsx",
    "domain": "real",
    "spacing": 100,
    "dt": 36000,
    "T": 3 * 36000,
    "save_dat": False,
    "animate": False,
    "postproc": False,
    "pre": False,
    "Nr": 2,
    "Deff": [1e-9, 5e-10],
    "beta": [0.15, 0.10],
    "phi_im": [0.1, 0.05],
}


def run_checkpoint(model_name):
    env = dict(BASE_ENV, model=model_name)
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()

    from fenicsx.mesh import get_mesh
    from fenicsx.adjoint import build_context, functional_fenicsx, gradient_fenicsx

    pozo = list(p.pozo)
    mesh_cache = get_mesh(pozo, p.spacing, regenerate=False)
    ctx = build_context(pozo, p, mesh_cache)

    Q0 = np.asarray(p.Qout[0], float)
    zp0 = float(pozo[1])
    x0 = np.concatenate([Q0, [zp0]])
    Nt = len(Q0)

    g = gradient_fenicsx(x0, ctx)
    indices = sorted({0, Nt // 2, Nt - 1})
    print(f"\n=== Checkpoint MRMT-FEniCS {model_name.upper()} | Nt={Nt} ===")

    all_pass = True
    for i in indices:
        label = f"grad_Q[{i}]"
        g_adj = g[i]
        print(f"\n--- {label} ---  adjunto={g_adj:.6e}")
        scale = abs(x0[i]) if abs(x0[i]) > 1e-14 else 1.0
        errs = []
        for frac in DELTA_FRACS:
            d = frac * scale
            xp = x0.copy(); xp[i] += d
            xm = x0.copy(); xm[i] -= d
            fd = (functional_fenicsx(xp, ctx) - functional_fenicsx(xm, ctx)) / (2 * d)
            rel = abs(fd - g_adj) / max(abs(g_adj), 1e-14)
            errs.append(rel)
            print(f"  delta={d:.2e}  FD={fd:.6e}  err_rel={rel:.3e}")
        if min(errs) >= TOL:
            all_pass = False
            print(f"  FALLA (min_err={min(errs):.3e} >= {TOL})")

    # grad_zp
    label = "grad_zp"
    g_adj = g[-1]
    print(f"\n--- {label} ---  adjunto={g_adj:.6e}")
    scale = abs(x0[-1]) if abs(x0[-1]) > 1e-14 else 1.0
    errs = []
    for frac in DELTA_FRACS:
        d = frac * scale
        xp = x0.copy(); xp[-1] += d
        xm = x0.copy(); xm[-1] -= d
        fd = (functional_fenicsx(xp, ctx) - functional_fenicsx(xm, ctx)) / (2 * d)
        rel = abs(fd - g_adj) / max(abs(g_adj), 1e-14)
        errs.append(rel)
        print(f"  delta={d:.2e}  FD={fd:.6e}  err_rel={rel:.3e}")
    if min(errs) >= TOL:
        all_pass = False
        print(f"  FALLA (min_err={min(errs):.3e} >= {TOL})")

    print(f"\n{'OK' if all_pass else 'FALLÓ'}: Checkpoint MRMT-FEniCS {model_name}")
    return all_pass


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "semi"
    if model not in ("semi", "block"):
        print("Uso: python -m fenicsx.test_gradient_checkpoint_mrmt_fenicsx [semi|block]")
        sys.exit(1)
    ok = run_checkpoint(f"mrmt_{model}")
    sys.exit(0 if ok else 1)
