"""
Corre el modelo nuevo (sin W-, con almacenamiento, salida libre) con la configuración de main y deja las
salidas donde las deja main.py: data/output/<backend>/{data,figures}/real/<modelo>/<run_type>/.

    python -m experimentos.correr_main                     # 3 modelos x 2 backends, run_type=comparison
    python -m experimentos.correr_main --backend bfr --modelo adr
    python -m experimentos.correr_main --grids              # + grids 3x2 (data/output/grid/grid_<campo>.mp4)

Config = experimentos/base.json (default.json de main con malla compartida) + n_stencil 25 + salida libre +
fuente común (eps_z 30 m, 5e-12).  No se fija exp_dir, así que las rutas son las de main.
"""
import argparse
import json
import os
import subprocess
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)
from experimentos.catalogo import ENVS, MAIN_FUENTE  # noqa: E402

MODELOS = ("adr", "mrmt_semi", "mrmt_block")
CAMPOS = ("H", "C", "V", "adj_H", "adj_C")


def config(modelo, backend, run_type):
    with open(os.path.join(CODIGO_DIR, "experimentos", "base.json")) as f:
        cfg = json.load(f)
    cfg.update({"model": modelo, "metodo": backend, "run_type": run_type, "extract_C": False,
                "Q_std": 0.3, "factor_sumidero_C": 0.0, "factor_almacenamiento_C": 1.0,
                "bc_outlet_mode": "libre", **MAIN_FUENTE})
    return cfg


def correr(modelo, backend, run_type):
    cfg = config(modelo, backend, run_type)
    os.makedirs(os.path.join(CODIGO_DIR, "data", "output", "_cfg"), exist_ok=True)
    ruta = os.path.join(CODIGO_DIR, "data", "output", "_cfg", f"{backend}_{modelo}_{run_type}.json")
    with open(ruta, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[correr_main] {backend}/{modelo} {run_type}", flush=True)
    return subprocess.call(["conda", "run", "-n", ENVS[backend], "--no-capture-output", "python", "-m",
                            "experimentos._worker", ruta], cwd=CODIGO_DIR)


def comparar(modelos, run_type):
    """Métricas y video BFR | FEniCSx por modelo -> data/output/comparacion/<modelo>/ (se lanza en bfr_env)."""
    from experimentos.comparar import comparar_caso
    for m in modelos:
        d = [os.path.join(CODIGO_DIR, "data", "output", b, "data", "real", m, run_type) for b in ("bfr", "fenicsx")]
        r = comparar_caso("main", m, {"desc": f"{m}, modelo nuevo, Q_std=0.3"}, True, tuple(d),
                          os.path.join(CODIGO_DIR, "data", "output", "comparacion", m))
        print(f"[comparar] {m}: ΔH_rms={r['H_rms_final_m']:.2e} m  C_L2={r['C_L2_rel_final']:.2e}  "
              f"|U|_L2={r['V_L2_rel_final']:.2e}  masa B/F={r['masa_ratio_final']:.3f}", flush=True)


def grids(run_type):
    for campo in CAMPOS:
        subprocess.call(["conda", "run", "-n", "bfr_env", "--no-capture-output", "python",
                         "scripts/animate_grid.py", campo, run_type], cwd=CODIGO_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", nargs="+", default=["bfr", "fenicsx"], choices=list(ENVS))
    ap.add_argument("--modelo", nargs="+", default=list(MODELOS), choices=MODELOS)
    ap.add_argument("--run-type", default="comparison")
    ap.add_argument("--grids", action="store_true", help="sólo generar los grids 3x2")
    ap.add_argument("--comparar", action="store_true", help="sólo métricas y video de comparación (en bfr_env)")
    a = ap.parse_args()
    if a.comparar:
        comparar(a.modelo, a.run_type)
    elif a.grids:
        grids(a.run_type)
    else:
        for b in a.backend:
            for m in a.modelo:
                correr(m, b, a.run_type)
