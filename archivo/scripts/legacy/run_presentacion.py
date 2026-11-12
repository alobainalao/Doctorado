#!/usr/bin/env python
"""
run_presentacion.py
Una sola llamada: corre las 6 variantes (BFR en bfr_env, FEniCS en fenics_env)
y genera los 5 videos de grilla para la presentación.

Uso:
    python scripts/run_presentacion.py            # todo
    python scripts/run_presentacion.py --pre      # forzar regeneración de malla
    python scripts/run_presentacion.py --videos-only
"""

import argparse, os, subprocess, sys, time

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_SCRIPT = os.path.join(_CODIGO, "scripts", "run_all_6_variants.py")

parser = argparse.ArgumentParser()
parser.add_argument("--pre",         action="store_true")
parser.add_argument("--videos-only", action="store_true", dest="videos_only")
args = parser.parse_args()

def run(env, extra_args, label):
    print(f"\n{'━'*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'━'*60}", flush=True)
    cmd = ["conda", "run", "-n", env, "python", PY_SCRIPT] + extra_args
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_CODIGO)
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0:
        print(f"  ✗ {label} falló (código {ret.returncode})", flush=True)
        sys.exit(ret.returncode)
    print(f"  ✓ {label}  ({elapsed/60:.1f} min)", flush=True)

base = ["--dt", "18000"]
if args.pre:
    base += ["--pre"]

if not args.videos_only:
    run("bfr_env",    ["--backend", "bfr"]     + base, "BFR  — bfr_env")
    run("fenics_env", ["--backend", "fenicsx"] + base, "FEniCS — fenics_env")

run("bfr_env", ["--videos-only"], "Videos grilla  — bfr_env")

print(f"\n{'━'*60}", flush=True)
print("  ✓ Todo listo.  Videos en: data/output/grid/", flush=True)
print(f"{'━'*60}", flush=True)
