#!/usr/bin/env python
"""
run_all_6_variants.py

Corre las 6 variantes (ADR / MRMT Semi / MRMT Bloque) × (BFR / FEniCS) con la
configuración nueva (malla 4 niveles, dt=18000s) y luego genera los 5 videos
de grilla 3×2 para la presentación:  H, C, V, adj_H, adj_C.

Uso:
    conda activate bfr_env   && python scripts/run_all_6_variants.py --backend bfr
    conda activate fenics_env && python scripts/run_all_6_variants.py --backend fenicsx
    conda activate bfr_env   && python scripts/run_all_6_variants.py --videos-only

    # Para correr todo BFR + generar videos de una vez:
    conda activate bfr_env && python scripts/run_all_6_variants.py --backend bfr --videos

Flags:
    --backend  bfr | fenicsx   corre solo ese backend
    --model    adr | mrmt_semi | mrmt_block  (opcional, corre solo ese modelo)
    --pre      regenerar preproceso (default: True la primera vez; False si ya existe pkl)
    --videos   después de correr, genera los 5 videos de grilla
    --videos-only  solo genera los videos (no corre simulaciones)
    --dt       paso de tiempo en segundos (default: 18000 = 5 h)
"""

import argparse, os, sys, subprocess, time

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODIGO)

# ── Argumentos ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--backend",    choices=["bfr", "fenicsx"], default=None)
parser.add_argument("--model",      choices=["adr", "mrmt_semi", "mrmt_block"], default=None)
parser.add_argument("--pre",        action="store_true", default=False,
                    help="Forzar regeneración del preproceso")
parser.add_argument("--videos",     action="store_true", default=False,
                    help="Generar videos de grilla después de correr")
parser.add_argument("--videos-only",action="store_true", default=False,
                    help="Solo generar videos (sin correr simulaciones)")
parser.add_argument("--dt",         type=float, default=18000.0,
                    help="Paso de tiempo en segundos (default: 18000 = 5 h)")
args = parser.parse_args()

MODELS   = ["adr", "mrmt_semi", "mrmt_block"]
BACKENDS = ["bfr", "fenicsx"]

if args.model:
    MODELS = [args.model]
if args.backend:
    BACKENDS = [args.backend]

# ── Configuración común ──────────────────────────────────────────────────────
BASE_CFG = {
    "domain":    "real",
    "run_type":  "standard",
    "animate":   False,
    "postproc":  False,
    "save_dat":  True,
    "dt":        args.dt,
    "T":         2592000,
    # Malla 4 niveles (ya en default.json; aquí explícito para no depender del JSON)
    "spacing":         60,
    "min_sp":           4,
    "s_cone":          16,
    "min_sp_boundary": 30,
}


def _pkl_exists(backend, model):
    path = f"./data/input/{backend}/real/{model}/preproceso.pkl"
    return os.path.exists(path)


def run_variant(backend, model, pre=False):
    cfg = dict(BASE_CFG)
    cfg["metodo"] = backend
    cfg["model"]  = model
    # pre=True la primera vez o si se fuerza; False si el pkl ya existe y no se fuerza
    cfg["pre"] = pre or not _pkl_exists(backend, model)

    print(f"\n{'='*60}", flush=True)
    print(f"  Corriendo: {model.upper()} / {backend.upper()}  (pre={cfg['pre']})", flush=True)
    print(f"{'='*60}", flush=True)

    from config.parameters import Parameters
    from funtions.runtime import RUNTIME

    # Castear a str para que cast() en Parameters los procese correctamente
    env = {k: str(v) for k, v in cfg.items()}
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()

    t0 = time.perf_counter()

    if backend == "fenicsx":
        from fenicsx.forward import solve_forward_fenicsx
        from fenicsx.mesh import get_mesh
        mesh_cache = get_mesh(p.pozo, p.spacing,
                              regenerate=bool(cfg["pre"]))
        solve_forward_fenicsx(p.Qout[0], p.pozo, p, mesh_cache=mesh_cache)
    else:
        from view.animation import solve_forward
        from preprocessing.preprocess import load_data
        d = load_data()
        solve_forward(d, p.Qout, animate=False, save_data=True)

    dt_s = time.perf_counter() - t0
    print(f"  ✓ {model}/{backend}  ({dt_s:.1f} s)", flush=True)


def generate_videos():
    fields = ["H", "C", "V", "adj_H", "adj_C"]
    # adj_H y adj_C requieren datos de run_type=comparison; los demás standard
    for field in fields:
        run_type = "comparison" if field.startswith("adj_") else "standard"
        print(f"\n[videos] Generando grid_{field}.mp4  (run_type={run_type})", flush=True)
        cmd = [sys.executable,
               os.path.join(_CODIGO, "scripts", "animate_grid.py"),
               field, run_type]
        ret = subprocess.run(cmd, cwd=_CODIGO)
        if ret.returncode != 0:
            print(f"[videos] ⚠ grid_{field}.mp4 falló (código {ret.returncode})", flush=True)
        else:
            print(f"[videos] ✓ grid_{field}.mp4", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if not args.videos_only:
        total_t0 = time.perf_counter()
        errors = []

        for model in MODELS:
            for backend in BACKENDS:
                try:
                    run_variant(backend, model, pre=args.pre)
                except Exception as e:
                    msg = f"{model}/{backend}: {e}"
                    print(f"\n  ✗ ERROR  {msg}", flush=True)
                    errors.append(msg)

        elapsed = time.perf_counter() - total_t0
        print(f"\n{'='*60}", flush=True)
        print(f"  Simulaciones completadas en {elapsed/60:.1f} min", flush=True)
        if errors:
            print(f"  ⚠ Errores ({len(errors)}):", flush=True)
            for e in errors:
                print(f"    - {e}", flush=True)
        print(f"{'='*60}", flush=True)

    if args.videos or args.videos_only:
        generate_videos()
