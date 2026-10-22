"""
Corre UN caso de experimentos/ dentro del entorno conda del backend.

    python -m experimentos._worker <ruta/config.json>

Lo invoca experimentos/correr.py; no se usa a mano.  La config llega completa
(Parameters(config=...)), así que no hay claves descartadas ni defaults ocultos.
"""
import json
import os
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)

    os.chdir(CODIGO_DIR)
    sys.path.insert(0, CODIGO_DIR)

    from funtions.runtime import RUNTIME
    from config.parameters import Parameters

    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.params
    print(f"[worker] {p.metodo}/{p.model}  run_type={p.run_type}  Nt={p.Nt}  dt={p.dt}  "
          f"fuente={p.activate_fuente}  pozo={p.activate_ext}  "
          f"bc_outlet={getattr(p, 'bc_outlet_mode', 'pfs')}  → {p.save_data}", flush=True)

    from main import main as run_main
    run_main()

    # El preproceso BFR del caso (~50 MB) es reconstruible desde la malla
    # congelada; se borra salvo que se pida conservarlo.
    pkl = os.path.join(p.save_preprocess, "preproceso.pkl")
    if p.metodo == "bfr" and not cfg.get("keep_pre", False) and os.path.exists(pkl):
        os.remove(pkl)


if __name__ == "__main__":
    main(sys.argv[1])
