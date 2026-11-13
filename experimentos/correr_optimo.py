"""
Directo + adjunto (run_type=comparison) con el control óptimo de experimentos/optimizar_nitrato.py, en BFR y
FEniCSx, y grids 3x3 (BFR | FEniCSx | diferencia) de las salidas.

    python3 -m experimentos.correr_optimo [modelo ...] [--variante nitrato_vscale_malla_g1] [--grids]

Control: data/output/optimizacion_<variante>/<modelo>.json (x_opt = [Q_1..Q_Nt, z_p]).  Config: config/<variante>.json
con Q_control y zp_control.  Salidas: data/output/experimentos/OPT_<variante>/<modelo>/<backend>; grids en
data/output/experimentos/OPT_<variante>/grid3 (scripts/animate_grid_3col.py, como experimentos/correr_main.py).
"""
import argparse
import json
import os
import subprocess
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)
from experimentos.catalogo import ENVS  # noqa: E402

MODELOS = ("adr", "mrmt_semi", "mrmt_block")
CAMPOS = ("H", "C", "V", "adj_H", "adj_C")


def correr(modelo, backend, variante, raiz):
    with open(os.path.join(CODIGO_DIR, "config", variante + ".json")) as f:
        cfg = json.load(f)
    with open(os.path.join(CODIGO_DIR, "data", "output", "optimizacion_" + variante, modelo + ".json")) as f:
        x = json.load(f)["x_opt"]
    cfg.update({"model": modelo, "metodo": backend, "run_type": "comparison", "save_dat": True, "animate": False,
                "postproc": False, "extract_C": False, "pre": True,
                "Q_control": [float(q) for q in x[:-1]], "zp_control": float(x[-1]),
                "exp_dir": os.path.join(raiz, modelo)})
    os.makedirs(os.path.join(CODIGO_DIR, "data", "output", "_cfg"), exist_ok=True)
    ruta = os.path.join(CODIGO_DIR, "data", "output", "_cfg", f"{backend}_{modelo}_optimo_{variante}.json")
    with open(ruta, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[correr_optimo] {backend}/{modelo}  z_p={x[-1]:.1f} m", flush=True)
    return subprocess.call(["conda", "run", "-n", ENVS[backend], "--no-capture-output", "python", "-m",
                            "experimentos._worker", ruta], cwd=CODIGO_DIR)


def grids(raiz):
    env = {**os.environ, "EXP_ROOT": os.path.join(CODIGO_DIR, raiz), "EXP_PREFIJO": ""}
    for campo in CAMPOS:
        subprocess.call(["conda", "run", "-n", "bfr_env", "--no-capture-output", "python",
                         "scripts/animate_grid_3col.py", campo], cwd=CODIGO_DIR, env=env)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelo", nargs="*", default=list(MODELOS))
    ap.add_argument("--variante", default="nitrato_vscale_malla_g1")
    ap.add_argument("--grids", action="store_true", help="sólo generar los grids 3x3")
    a = ap.parse_args()
    raiz = os.path.join("data", "output", "experimentos", "OPT_" + a.variante)
    if not a.grids:
        for m in a.modelo:
            for b in ("bfr", "fenicsx"):
                correr(m, b, a.variante, raiz)
    grids(raiz)
