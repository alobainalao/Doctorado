"""Lanza UNA corrida BFR de la rama de continuidad: base.json + overrides.

    python experimentos_continuidad/correr_forma.py ID forma [clave=valor ...] [--smoke] [--backend=bfr|fenicsx]

Salida en experimentos_continuidad/resultados/<ID>/bfr/ (no toca data/output ni
los benchmarks oficiales). Guarda config.json con la forma, commit y fecha.
"""
import json, os, subprocess, sys, datetime

AQUI = os.path.dirname(os.path.abspath(__file__))
CODIGO = os.path.dirname(AQUI)
ENVS = {"bfr": "bfr_env", "fenicsx": "fenics_env"}


def main(argv):
    smoke = "--smoke" in argv
    backend = next((a.split("=")[1] for a in argv if a.startswith("--backend=")), "bfr")
    ident, forma, *over = [a for a in argv if a != "--smoke" and not a.startswith("--backend=")]
    cfg = json.load(open(os.path.join(CODIGO, "experimentos", "base.json")))
    cfg.update(metodo=backend, transport_form=forma,
               exp_dir=os.path.join("experimentos_continuidad", "resultados", ident))
    for kv in over:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    if smoke:
        cfg["T"] = 2 * cfg["dt"]
    out = os.path.join(CODIGO, cfg["exp_dir"], backend)
    os.makedirs(out, exist_ok=True)
    cfg["_etiqueta"] = dict(
        id=ident, backend=backend, formulacion=forma,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODIGO, text=True).strip(),
        fecha=datetime.datetime.now().isoformat(timespec="seconds"))
    cp = os.path.join(out, "config.json")
    json.dump(cfg, open(cp, "w"), indent=2, ensure_ascii=False)
    return subprocess.call([f"/home/alex/miniconda3/envs/{ENVS[backend]}/bin/python", "-m", "experimentos._worker", cp], cwd=CODIGO)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
