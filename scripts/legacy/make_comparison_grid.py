"""
Post-proceso: genera 5 videos grid 3x2 combinando los 6 backends.

Grid layout:
         BFR          FEniCS
ADR       [0,0]        [0,1]
MRMT Semi [1,0]        [1,1]
MRMT Block[2,0]        [2,1]

Videos: H.mp4, V.mp4, C.mp4, C_im_r0.mp4, C_im_r1.mp4  (C_im solo MRMT)
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

os.chdir('/home/alex/dev/doctorado/codigo')
sys.path.insert(0, '.')

# ── config ──────────────────────────────────────────────────────────────────
with open('config/default.json') as f:
    cfg = json.load(f)
with open('config/malla_sp60.json') as f:
    cfg.update(json.load(f))

from funtions.runtime import RUNTIME
from config.parameters import Parameters

p = Parameters(env={**cfg, 'model': 'adr', 'metodo': 'bfr', 'run_type': 'comparison'})
RUNTIME.params = p

# ── grilla regular y máscara ──────────────────────────────────────────────────
from preprocessing.preprocess import load_data
from scipy.interpolate import LinearNDInterpolator

cfg_bfr = {**cfg, 'model': 'adr', 'metodo': 'bfr', 'run_type': 'comparison'}
p_bfr = Parameters(env=cfg_bfr)
RUNTIME.params = p_bfr
d_bfr = load_data()
RUNTIME.params = p_bfr

xy_grid = d_bfr.xy_grid          # (2, Nx, Ny)
mask    = d_bfr.mask
NX, NY  = xy_grid.shape[1], xy_grid.shape[2]
X       = xy_grid[0].ravel()
Y       = xy_grid[1].ravel()
xy_flat = np.column_stack([X, Y])  # (NX*NY, 2)

# Cache de interpoladores por conjunto de nodos (hashados por nbytes+shape)
_interp_cache = {}

def get_interp(nodes):
    key = nodes.shape[0]
    if key not in _interp_cache:
        print(f"  Construyendo interpolador para {nodes.shape[0]} nodos...", flush=True)
        dummy = np.zeros(len(nodes))
        interp = LinearNDInterpolator(nodes, dummy)
        interp.values = None  # placeholder
        _interp_cache[key] = (interp, nodes)
    return _interp_cache[key][0]

def field2img(vals_n, nodes):
    interp = get_interp(nodes)
    interp.values = vals_n.reshape(-1, 1).astype(float)
    img = interp(xy_flat).ravel()
    img[~mask] = np.nan
    return img.reshape(NX, NY).T

# ── paths ────────────────────────────────────────────────────────────────────
BASE = 'data/output/{metodo}/data/real/{model}/comparison_sp60/simulation_results.npz'
GRID_OUT = 'data/output/comparison_grid_sp60'
os.makedirs(GRID_OUT, exist_ok=True)

BACKENDS = [
    ('bfr',     'adr'),
    ('fenicsx', 'adr'),
    ('bfr',     'mrmt_semi'),
    ('fenicsx', 'mrmt_semi'),
    ('bfr',     'mrmt_block'),
    ('fenicsx', 'mrmt_block'),
]
LABELS = [
    'BFR · ADR',     'FEniCS · ADR',
    'BFR · MRMT Semi','FEniCS · MRMT Semi',
    'BFR · MRMT Block','FEniCS · MRMT Block',
]

# ── cargar npz ───────────────────────────────────────────────────────────────
print("Cargando datos...", flush=True)
data = {}
for metodo, model in BACKENDS:
    path = BASE.format(metodo=metodo, model=model)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta: {path}")
    d_npz = np.load(path, allow_pickle=True)
    data[(metodo, model)] = d_npz
    print(f"  {metodo}/{model}: H{d_npz['H'].shape}", flush=True)

# Nt mínimo común
Nt = min(d['H'].shape[0] for d in data.values())
print(f"Nt común: {Nt}", flush=True)

# ── colormaps ────────────────────────────────────────────────────────────────
base = matplotlib.colormaps["turbo"]
cmap_C = LinearSegmentedColormap.from_list("c_cmap", base(np.linspace(0.08, 1.0, 256)))

base2 = matplotlib.colormaps["inferno_r"]
cols = np.vstack([[[0.92,0.92,0.92,1.0]], base2(np.linspace(0,1,240))])
cmap_H = LinearSegmentedColormap.from_list("h_cmap", cols, 256)

# ── rangos globales ───────────────────────────────────────────────────────────
print("Calculando rangos...", flush=True)
def get_H(d, n): return d['H'][n,0,:] if d['H'].ndim==3 else d['H'][n,:]
def get_C(d, n): return d['C'][n,0,:] if d['C'].ndim==3 else d['C'][n,:]
def get_U(d, n):
    U = d['U']
    u = U[n,0,:,:] if U.ndim==4 else U[n,:,:]
    return np.sqrt(u[:,0]**2 + u[:,1]**2)
def get_Ci(d, r, n):
    if 'Ci' not in d: return np.zeros(d['H'].shape[-1])
    Ci = d['Ci']
    return Ci[r,n,0,:] if Ci.ndim==4 else Ci[r,n,:]

H_all = np.concatenate([data[k]['H'][:,0,:] if data[k]['H'].ndim==3 else data[k]['H'] for k in data], axis=1)
C_all = np.concatenate([data[k]['C'][:,0,:] if data[k]['C'].ndim==3 else data[k]['C'] for k in data], axis=1)
U_all = np.concatenate([
    np.array([get_U(data[k], n) for n in range(data[k]['U'].shape[0])])
    for k in data], axis=1)

H_vmin, H_vmax = np.nanpercentile(H_all, 1), np.nanpercentile(H_all, 99)
C_vmin, C_vmax = 0.0, np.nanpercentile(C_all, 99)
V_vmin, V_vmax = 0.0, np.nanpercentile(U_all, 99)

# C_im para MRMT
has_mrmt = any('mrmt' in model for _, model in BACKENDS)
Nr = int(cfg.get('Nr', 3))
if has_mrmt:
    mrmt_keys = [(m,mo) for m,mo in BACKENDS if 'mrmt' in mo]
    Cim_all = np.concatenate([
        np.array([[get_Ci(data[k], r, n) for n in range(Nt)] for r in range(Nr)]).reshape(Nr,-1)
        for k in mrmt_keys], axis=1)
    Cim_vmin, Cim_vmax = 0.0, np.nanpercentile(Cim_all, 99)

# ── función para hacer un video ──────────────────────────────────────────────
def make_video(field_fn, vmin, vmax, cmap, fname, title):
    """field_fn(n, metodo, model, nodes) → 1D array de nodos"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    ims = []
    for idx, ((metodo, model), label) in enumerate(zip(BACKENDS, LABELS)):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        nodes = data[(metodo,model)]['nodes']
        img0 = field2img(field_fn(0, metodo, model, nodes), nodes)
        im = ax.imshow(img0, vmin=vmin, vmax=vmax, cmap=cmap,
                       origin='lower', extent=[p.xmin, p.xmax, p.ymin, p.ymax],
                       aspect='auto')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    fig.suptitle(f'{title}  t=0 h', fontsize=12)
    plt.colorbar(ims[0], ax=axes, shrink=0.6)
    plt.tight_layout()

    dt_s = int(data[('bfr','adr')]['dt'])
    writer = FFMpegWriter(fps=20, codec='libx264', bitrate=3000,
                          extra_args=['-pix_fmt','yuv420p'])
    out_path = f'{GRID_OUT}/{fname}'
    with writer.saving(fig, out_path, dpi=120):
        for n in range(Nt):
            t_h = n * dt_s / 3600
            for idx, (metodo, model) in enumerate(BACKENDS):
                nodes = data[(metodo,model)]['nodes']
                img = field2img(field_fn(n, metodo, model, nodes), nodes)
                ims[idx].set_data(img)
            fig.suptitle(f'{title}  t={t_h:.1f} h', fontsize=12)
            writer.grab_frame()
            if n % 20 == 0:
                print(f'  {fname} frame {n}/{Nt}', flush=True)
    plt.close(fig)
    print(f'  → {out_path}', flush=True)

# ── H ────────────────────────────────────────────────────────────────────────
print("Video H...", flush=True)
make_video(lambda n,m,mo,nd: get_H(data[(m,mo)], n),
           H_vmin, H_vmax, cmap_H, 'H.mp4', 'Hydraulic Head H (m)')

# ── V ────────────────────────────────────────────────────────────────────────
print("Video V...", flush=True)
make_video(lambda n,m,mo,nd: get_U(data[(m,mo)], n),
           V_vmin, V_vmax, cmap_C, 'V.mp4', 'Velocity |U| (m/s)')

# ── C ────────────────────────────────────────────────────────────────────────
print("Video C...", flush=True)
make_video(lambda n,m,mo,nd: get_C(data[(m,mo)], n),
           C_vmin, C_vmax, cmap_C, 'C.mp4', 'Concentration C (kg/m³)')

# ── C_im ────────────────────────────────────────────────────────────────────
if has_mrmt:
    for r in range(Nr):
        print(f"Video C_im r{r}...", flush=True)
        def _cim(n, m, mo, nd, _r=r):
            return get_Ci(data[(m,mo)], _r, n)
        make_video(_cim, Cim_vmin, Cim_vmax, cmap_C,
                   f'C_im_r{r}.mp4', f'Immobile Conc C_im(r={r}) (kg/m³)')

print("DONE — videos en:", GRID_OUT, flush=True)
