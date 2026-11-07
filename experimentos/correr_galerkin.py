"""
Variante opt-in del modelo nuevo: filas del flujo Galerkin P1 en una franja bajo la superficie (galerkin_banda, m).
Solo BFR; FEniCSx no cambia (se enlaza su corrida oficial para poder armar los grids).

    python -m experimentos.correr_galerkin --modelo adr --ancho 200

Config = la de correr_main + galerkin_banda + exp_dir propio, así que no toca default.json, base.json ni las
salidas de main.  Salidas: data/output/experimentos/galerkin_banda/<modelo>/{bfr,fenicsx}/.
"""
import argparse
import json
import os
import subprocess
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)
from experimentos.correr_main import config, MODELOS  # noqa: E402
from experimentos.catalogo import ENVS  # noqa: E402

RAIZ = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "galerkin_banda")


def correr(modelo, ancho, run_type="comparison"):
    exp_dir = os.path.join(RAIZ, modelo)
    cfg = dict(config(modelo, "bfr", run_type), galerkin_banda=float(ancho), exp_dir=exp_dir)
    ruta = os.path.join(CODIGO_DIR, "data", "output", "_cfg", f"bfr_{modelo}_galerkin{int(ancho)}.json")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[correr_galerkin] bfr/{modelo} banda={ancho} m → {exp_dir}", flush=True)
    rc = subprocess.call(["conda", "run", "-n", ENVS["bfr"], "--no-capture-output", "python", "-m",
                          "experimentos._worker", ruta], cwd=CODIGO_DIR)
    # enlaza la corrida oficial de FEniCSx (sin copiarla) para scripts/animate_grid.py (EXP_ROOT)
    ofi = os.path.join(CODIGO_DIR, "data", "output", "fenicsx", "data", "real", modelo, run_type)
    dst = os.path.join(exp_dir, "fenicsx")
    if os.path.isdir(ofi) and not os.path.lexists(dst):
        os.symlink(ofi, dst)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", nargs="+", default=["adr"], choices=MODELOS)
    ap.add_argument("--ancho", type=float, default=200.0)
    a = ap.parse_args()
    for m in a.modelo:
        correr(m, a.ancho)
