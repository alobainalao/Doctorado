import sys, os, numpy as np, json

METODO = sys.argv[1]   # 'bfr' o 'fenicsx'
MODEL  = sys.argv[2]   # 'adr', 'mrmt_semi', 'mrmt_block'

os.chdir('/home/alex/dev/doctorado/codigo')
sys.path.insert(0, '.')

with open('config/default.json') as f:
    cfg = json.load(f)
with open('config/malla_sp60.json') as f:
    cfg.update(json.load(f))

cfg['model']    = MODEL
cfg['metodo']   = METODO
cfg['run_type'] = 'comparison'

from funtions.runtime import RUNTIME
from config.parameters import Parameters

p = Parameters(env=cfg)
p.run_type   = 'comparison'
p.save_data  = f'./data/output/{METODO}/data/real/{MODEL}/comparison_sp60'
p.save_dat   = p.save_data
p.save_video = f'./data/output/{METODO}/figures/real/{MODEL}/comparison_sp60'
os.makedirs(p.save_data,  exist_ok=True)
os.makedirs(p.save_video, exist_ok=True)
RUNTIME.params = p

print(f'[{METODO}/{MODEL}] Nt={p.Nt}  dt={p.dt}  min_sp={p.min_sp}  min_sp_bnd={getattr(p,"min_sp_boundary",0)}  spacing={p.spacing}', flush=True)

if METODO == 'fenicsx':
    from fenicsx.adjoint import adjoint_fenicsx
    from fenicsx.mesh import build_mesh_from_bfr_pkl
    pkl_path = f'data/input/bfr/real/{MODEL}/preproceso.pkl'
    mesh_cache = build_mesh_from_bfr_pkl(pkl_path)
    adjoint_fenicsx(p, mesh_cache=mesh_cache)
else:
    from preprocessing.preprocess import load_data
    from view.animation import solve_forward, solve_adjoint
    from funtions.utils import get_solucion
    RUNTIME.params = p
    d = load_data()
    RUNTIME.params = p
    solve_forward(d, p.Qout, True, True)
    h, U, C = get_solucion(['H','U','C'], f'{p.save_data}/simulation_results.npz')
    solve_adjoint(h, U, C, p.Qout, d, True, True)

print(f'[{METODO}/{MODEL}] DONE', flush=True)
