"""
Informe HTML general de los experimentos BFR vs FEniCS.

Lee sólo lo que produjo experimentos/comparar.py en su pasada única
(<caso>/comparacion/metricas.json + comparacion.mp4): las gráficas del
informe se dibujan con esas mismas series, y su cursor sigue al cuadro del
video que se está viendo. No recalcula nada.

    python -m experimentos.informe

Salida: data/output/experimentos/informe.html (los videos se enlazan con
rutas relativas: abrir el HTML desde esa carpeta). Las conclusiones escritas
a mano viven en experimentos/conclusiones.html y se insertan tal cual.
"""
import json
import os
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.catalogo import GRUPOS, iter_casos  # noqa: E402

OUT_ROOT = os.path.join(CODIGO_DIR, "data", "output", "experimentos")
CONCLUSIONES = os.path.join(CODIGO_DIR, "experimentos", "conclusiones.html")
BASE_JSON = os.path.join(CODIGO_DIR, "experimentos", "base.json")


def recoger():
    grupos = []
    for g, gspec in GRUPOS.items():
        if not gspec.get("comparar", True):
            continue
        casos = []
        for _, c, spec in iter_casos([g]):
            f = os.path.join(OUT_ROOT, g, c, "comparacion", "metricas.json")
            if not os.path.exists(f):
                casos.append({"caso": c, "desc": spec["desc"], "falta": True})
                continue
            with open(f) as fh:
                m = json.load(fh)
            video = f"{g}/{c}/comparacion/comparacion.mp4"
            m["video"] = video if os.path.exists(os.path.join(OUT_ROOT, video)) else None
            m["caso"] = c
            casos.append(m)
        grupos.append({"grupo": g, "desc": gspec["desc"], "casos": casos})
    return grupos


def main():
    grupos = recoger()
    with open(BASE_JSON) as fh:
        base = json.load(fh)
    concl = ""
    if os.path.exists(CONCLUSIONES):
        with open(CONCLUSIONES) as fh:
            concl = fh.read()
    datos = json.dumps({"grupos": grupos, "base": {k: base[k] for k in (
        "Q_std", "dt", "T", "theta", "theta_C", "r_pozo", "norm_delta_pozo", "sigma_fuente",
        "n_stencil", "bc_outlet_mode")}}, separators=(",", ":"))
    with open(os.path.join(os.path.dirname(__file__), "informe_plantilla.html")) as fh:
        plantilla = fh.read()
    out = os.path.join(OUT_ROOT, "informe.html")
    with open(out, "w") as fh:
        fh.write(plantilla.replace("/*__DATOS__*/null", datos.replace("</", "<\\/"))
                 .replace("<!--__CONCLUSIONES__-->", concl))
    n = sum(1 for g in grupos for c in g["casos"] if not c.get("falta"))
    faltan = [f"{g['grupo']}/{c['caso']}" for g in grupos for c in g["casos"] if c.get("falta")]
    print(f"[informe] {n} casos → {os.path.relpath(out, CODIGO_DIR)}")
    if faltan:
        print("[informe] sin comparación: " + ", ".join(faltan))


if __name__ == "__main__":
    main()
