#!/usr/bin/env python
"""
Re-corre el forward FEniCS con Q=Q_std (estándar) y guarda en la ruta
top-level (misma que usa animate_forward.py).

Uso (desde codigo/ con fenics_env):
    python scripts/rerun_fenicsx_standard.py mrmt_semi
    python scripts/rerun_fenicsx_standard.py mrmt_block
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

model = sys.argv[1] if len(sys.argv) > 1 else "mrmt_semi"
assert model in ("adr", "mrmt_semi", "mrmt_block"), f"modelo no reconocido: {model}"

# Usar JSON mode para heredar todos los parámetros físicos de default.json,
# luego sobreescribir model/metodo/run_type antes de _build_derived
from config.parameters import Parameters

p = Parameters(env=None)   # lee default.json (dt=3600, T=2592000, etc.)
p.model    = model
p.metodo   = "fenicsx"
p.run_type = "standard"
p.pre      = False
p.save_dat = True
p.animate  = False
p._build_derived()         # reconstruir Nt, Qout, etc. con nuevos valores

Nt    = p.Nt
Q_std = float(getattr(p, "Q_std", 1e-3))
print(f"[rerun] {model}  Nt={Nt}  dt={p.dt}  Q_std={Q_std:.4e}", flush=True)

# Inicializar RUNTIME (requerido por get_init_values y otros helpers)
from funtions.runtime import RUNTIME
RUNTIME.params = p

# Destino top-level (misma ruta que lee animate_forward.py)
top_dir = f"data/output/fenicsx/data/real/{model}"
os.makedirs(top_dir, exist_ok=True)

# Parche: redirigir save al top-level Y añadir Q al npz
import numpy as np
import fenicsx.io as fx_io

_orig_save = fx_io.save_results

def _patched_save(save_dir, nodes, H_hist, C_hist, U_hist, dt,
                  C_im_hist=None, nodes_Q=None):
    # Guardar en top-level (no en standard/ subdirectorio)
    _orig_save(top_dir, nodes, H_hist, C_hist, U_hist, dt, C_im_hist, nodes_Q)
    # Añadir Q (constante Q_std) y zp al npz para compatibilidad
    path = f"{top_dir}/simulation_results.npz"
    d = dict(np.load(path, allow_pickle=True))
    d["Q"]  = np.full(len(H_hist), Q_std)
    d["zp"] = np.float64(p.pozo)
    np.savez(path, **d)
    print(f"[rerun] Q={Q_std:.4e} añadido ✔  {path}", flush=True)

fx_io.save_results = _patched_save

# Malla cacheada (pre=False → reusar malla existente)
from fenicsx.mesh import get_mesh
mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=False)

# Correr forward estándar
from fenicsx.forward import solve_forward_fenicsx
solve_forward_fenicsx(p.Qout[0], p.pozo, p, mesh_cache=mesh_cache)

print(f"[rerun] {model} LISTO", flush=True)
