"""
Corre el modelo nuevo (comparison) en BFR y FEniCSx sobre una malla alternativa de experimentos/malla_borde/.

    python -m experimentos.malla_borde.correr_malla --malla data/output/experimentos/malla_borde/mallas/m_g10s7.pkl \
        --etiqueta g10s7 --modelo adr

Config = la de experimentos/correr_main.py + mesh_pkl + exp_dir propio.  No toca default.json, base.json ni las
salidas de main.  Salidas: data/output/experimentos/malla_borde/<etiqueta>/<modelo>/{bfr,fenicsx}/.
"""
import argparse
import json
import os
import subprocess
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODIGO_DIR)
from experimentos.correr_main import config, MODELOS  # noqa: E402
from experimentos.catalogo import ENVS  # noqa: E402


def correr(malla, etiqueta, modelo, backends, run_type="comparison"):
    exp_dir = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "malla_borde", etiqueta, modelo)
    for backend in backends:
        cfg = dict(config(modelo, backend, run_type), mesh_pkl=os.path.relpath(malla, CODIGO_DIR), exp_dir=exp_dir)
        ruta = os.path.join(CODIGO_DIR, "data", "output", "_cfg", f"{backend}_{modelo}_malla_{etiqueta}.json")
        with open(ruta, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[correr_malla] {backend}/{modelo} malla={etiqueta} → {exp_dir}/{backend}", flush=True)
        rc = subprocess.call(["conda", "run", "-n", ENVS[backend], "--no-capture-output", "python", "-m",
                              "experimentos._worker", ruta], cwd=CODIGO_DIR)
        print(f"[correr_malla] {backend} terminó con código {rc}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--malla", required=True)
    ap.add_argument("--etiqueta", required=True)
    ap.add_argument("--modelo", default="adr", choices=MODELOS)
    ap.add_argument("--backend", nargs="+", default=["bfr", "fenicsx"], choices=list(ENVS))
    a = ap.parse_args()
    correr(os.path.abspath(a.malla), a.etiqueta, a.modelo, a.backend)
