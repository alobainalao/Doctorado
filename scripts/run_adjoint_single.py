#!/usr/bin/env python
"""
run_adjoint_single.py  <model>  <metodo>

Corre un único forward + un único adjunto para la combinación dada y guarda:
  data/output/{metodo}/data/real/{model}/simulation_results.npz
  data/output/{metodo}/data/real/{model}/adjoint_results.npz

Usar con el entorno correcto:
  BFR:     /home/alex/miniconda3/envs/bfr_env/bin/python scripts/run_adjoint_single.py adr bfr
  FEniCS:  /home/alex/miniconda3/envs/fenics_env/bin/python scripts/run_adjoint_single.py adr fenicsx
"""
import sys, json, os

# Asegurar que codigo/ esté en sys.path (el script vive en codigo/scripts/)
_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

model  = sys.argv[1]   # adr | mrmt_semi | mrmt_block
metodo = sys.argv[2]   # bfr | fenicsx

print(f"\n{'='*60}", flush=True)
print(f"[adj] model={model}  metodo={metodo}", flush=True)
print(f"{'='*60}", flush=True)

# Config derivada de default.json sin modificarlo
with open("config/default.json") as f:
    cfg = json.load(f)

cfg["model"]    = model
cfg["metodo"]   = metodo
# run_type=optimization: el preproceso BFR incluye A_H/B_H que necesita step_adjoint.
# FEniCS ignora este campo en su adjunto (usa UFL autodiff directamente).
cfg["run_type"] = "optimization"
cfg["animate"]  = False
cfg["save_dat"] = True
cfg["pre"]      = True   # regenerar preproceso para asegurar A_H/B_H en BFR

tmp = f"/tmp/adj_{model}_{metodo}.json"
with open(tmp, "w") as f:
    json.dump(cfg, f)

# RUNTIME debe estar listo ANTES de cualquier import de animation/preprocess
# (esos módulos hacen `p = RUNTIME.params` al importarse).
from funtions.runtime import RUNTIME
from config.parameters import Parameters

RUNTIME.params = Parameters(json_path=tmp)
p = RUNTIME.params

# Guardar en raíz (sin subcarpeta optimization/) para acceso uniforme
root_data  = f"./data/output/{metodo}/data/real/{model}"
root_video = f"./data/output/{metodo}/figures/real/{model}"
os.makedirs(root_data,  exist_ok=True)
os.makedirs(root_video, exist_ok=True)
p.save_data  = root_data
p.save_video = root_video

print(f"[adj] destino: {root_data}", flush=True)

# =====================================================================
# BFR  (RBF-FD, bfr_env)
# =====================================================================
if metodo == "bfr":
    from preprocessing.preprocess import load_data
    from view.animation import solve_forward, solve_adjoint
    from funtions.utils import get_solucion

    print("[adj] BFR preprocesando...", flush=True)
    d = load_data()

    print("[adj] BFR forward...", flush=True)
    solve_forward(d, p.Qout, animate=False, save_data=True)

    h, U, C = get_solucion(["H", "U", "C"],
                            f"{p.save_data}/simulation_results.npz")

    print("[adj] BFR adjunto...", flush=True)
    solve_adjoint(h, U, C, p.Qout, d, animate=False, save_data=True)

    print(f"[adj] ✓ BFR  {model}  →  {root_data}/adjoint_results.npz", flush=True)

# =====================================================================
# FEniCS  (DOLFINx, fenics_env)
# =====================================================================
else:
    import numpy as np
    from fenicsx.mesh import get_mesh
    from fenicsx.adjoint import build_context, _solve_forward, _solve_adjoint_grad
    from fenicsx import io as fx_io

    print("[adj] FEniCS cargando malla...", flush=True)
    mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=False)

    print("[adj] FEniCS construyendo contexto...", flush=True)
    ctx = build_context(list(p.pozo), p, mesh_cache)

    Q0  = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])

    print("[adj] FEniCS forward...", flush=True)
    states = _solve_forward(ctx, Q0, zp0)

    print("[adj] FEniCS adjunto (UFL autodiff)...", flush=True)
    _, adj_states = _solve_adjoint_grad(ctx, states)

    if ctx.comm.rank == 0:
        nodes_G = ctx.G.tabulate_dof_coordinates()[:, :2]
        nodes_Q = ctx.Q.tabulate_dof_coordinates()[:, :2]

        save_kw = dict(
            H     = np.asarray(states["h"])[:, None, :],
            C     = np.asarray(states["C"])[:, None, :],
            nodes = nodes_G,
            dt    = p.dt,
            Q     = states["Q"],
            zp    = states["zp"],
        )
        if states.get("C_im"):
            save_kw["C_im"]    = np.asarray(states["C_im"])
            save_kw["nodes_Q"] = nodes_Q

        np.savez(f"{root_data}/simulation_results.npz", **save_kw)
        print(f"[adj] fenicsx simulation_results.npz guardado", flush=True)

        fx_io.save_adjoint_results(
            root_data, nodes_G, nodes_Q,
            adj_states["psiH"], adj_states["psiC"], p.dt,
        )
        print(f"[adj] ✓ FEniCS {model}  →  {root_data}/adjoint_results.npz", flush=True)
