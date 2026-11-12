"""
guardar_gradiente.py  <model>  <metodo>

Computa el gradiente adjunto ∂J/∂Q_n y ∂J/∂z_p para la configuración
de checkpoint (malla gruesa, horizonte corto) y guarda el resultado en:

    data/output/{metodo}/data/real/{model}/gradient_checkpoint.npz

Uso:
    # bfr_env
    python scripts/guardar_gradiente.py adr bfr
    python scripts/guardar_gradiente.py mrmt_semi bfr
    python scripts/guardar_gradiente.py mrmt_block bfr

    # fenics_env
    python scripts/guardar_gradiente.py adr fenicsx
    python scripts/guardar_gradiente.py mrmt_semi fenicsx
    python scripts/guardar_gradiente.py mrmt_block fenicsx
"""

import sys, os
import numpy as np

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

model  = sys.argv[1] if len(sys.argv) > 1 else "adr"
metodo = sys.argv[2] if len(sys.argv) > 2 else "bfr"

BASE_ENV = {
    "run_type": "optimization",
    "domain": "real",
    "dt": 36000,
    "T": 4 * 36000,
    "save_dat": False,
    "animate": False,
    "pre": True,
}

MRMT_PARAMS = {
    "Nr": 2,
    "Deff":   [1e-9, 5e-10],
    "beta":   [0.15, 0.10],
    "phi_im": [0.1,  0.05],
}

# Malla gruesa pero diferente por backend para equilibrar tiempo de cómputo.
SPACING = {"bfr": 60, "fenicsx": 120}

env = {
    **BASE_ENV,
    "metodo": metodo,
    "model":  model,
    "spacing": SPACING[metodo],
    "pre": True,
}
if model.startswith("mrmt"):
    env.update(MRMT_PARAMS)

from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(env=env)
p = RUNTIME.get()

out_dir = f"./data/output/{metodo}/data/real/{model}"
os.makedirs(out_dir, exist_ok=True)

print(f"[grad] model={model}  metodo={metodo}  Nt={int(p.T/p.dt)}  spacing={env['spacing']}")

# ─── BFR ─────────────────────────────────────────────────────────────────────
if metodo == "bfr":
    from preprocessing.preprocess import load_data
    from view.animation import gradient, functional

    d = load_data()
    Q0 = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0 = np.concatenate([Q0, [zp0]])

    # Auto-escala γ: con gamma=1, kappa=0 → J0 ≈ J_obs; escalar para J1 ≈ 1e12
    p.gamma = 1.0
    p.koppa = 0.0
    J0_raw = functional(x0, d, p, animate=False, save_data=True)
    TARGET_J1 = 1e12
    p.gamma = TARGET_J1 / max(abs(J0_raw), 1e-14) if abs(J0_raw) > 0 else 1.0
    print(f"[grad] J0_raw={J0_raw:.3e}  γ auto={p.gamma:.3e}")

    g = gradient(x0, d, p)
    grad_Q  = g[:-1]
    grad_zp = float(g[-1])

# ─── FEniCS ──────────────────────────────────────────────────────────────────
else:
    from fenicsx.mesh import get_mesh
    from fenicsx.adjoint import (
        build_context, gradient_fenicsx, _solve_forward, _unpack,
    )

    mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mesh_cache)

    Q0 = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0 = np.concatenate([Q0, [zp0]])

    obs2_dt = _solve_forward(ctx, *_unpack(x0))["obs2_dt"]
    TARGET_J1 = 1e12
    p.gamma = TARGET_J1 / obs2_dt if obs2_dt > 0 else 1.0
    p.koppa = 0.0
    print(f"[grad] obs2_dt={obs2_dt:.3e}  γ auto={p.gamma:.3e}")

    if ctx.comm.rank != 0:
        sys.exit(0)

    g = gradient_fenicsx(x0, ctx)
    grad_Q  = g[:-1]
    grad_zp = float(g[-1])

# ─── Guardar ─────────────────────────────────────────────────────────────────
Nt  = len(grad_Q)
tvec = np.arange(Nt) * p.dt

out_path = f"{out_dir}/gradient_checkpoint.npz"
np.savez(out_path,
         grad_Q=grad_Q,
         grad_zp=np.array([grad_zp]),
         tvec=tvec,
         model=model,
         metodo=metodo,
         dt=p.dt,
         gamma=p.gamma,
         spacing=env["spacing"])

print(f"[grad] ✓  {out_path}")
print(f"[grad] grad_Q  rango [{grad_Q.min():.3e}, {grad_Q.max():.3e}]")
print(f"[grad] grad_zp = {grad_zp:.6e}")
