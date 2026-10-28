"""
Informe HTML general de los experimentos BFR vs FEniCS.

Lee sólo lo que produjo experimentos/comparar.py en su pasada única
(<caso>/comparacion/metricas.json + comparacion_{H,C,V}.{mp4,webm}): las
gráficas del informe se dibujan con esas mismas series, y su cursor sigue al
cuadro del video que se está viendo. No recalcula nada.

    python -m experimentos.informe

Salida: data/output/experimentos/informe.html (los videos se enlazan con
rutas relativas: abrir el HTML desde esa carpeta).  Textos escritos a mano:
  experimentos/conclusiones.html   revisión y conclusiones (se insertan tal cual)
  experimentos/analisis_casos.json análisis de cada caso; cada nota guarda las
                                   métricas con que se escribió y el informe la
                                   marca como desactualizada si ya no coinciden.
"""
import json
import os
import pickle
import shutil
import sys

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.catalogo import GRUPOS, iter_casos  # noqa: E402

OUT_ROOT = os.path.join(CODIGO_DIR, "data", "output", "experimentos")
CONCLUSIONES = os.path.join(CODIGO_DIR, "experimentos", "conclusiones.html")
ANALISIS = os.path.join(CODIGO_DIR, "experimentos", "analisis_casos.json")
BASE_JSON = os.path.join(CODIGO_DIR, "experimentos", "base.json")
CAMPOS = ("H", "C", "V")
TOL_NOTA = 0.02      # diferencia relativa a partir de la cual una nota está desactualizada


def _nota(clave, r, notas):
    """Nota escrita a mano del caso, marcada si sus métricas de referencia ya no son las actuales."""
    n = notas.get(clave)
    if not n:
        return None
    viejas = [k for k, v in n.get("ref", {}).items()
              if k in r and abs(r[k] - v) > TOL_NOTA * max(abs(v), 1e-12)]
    return {"html": n["html"], "desactualizada": viejas}


def recoger(notas):
    grupos = []
    for g, gspec in GRUPOS.items():
        if not gspec.get("comparar", True):
            continue
        casos = []
        for _, c, spec in iter_casos([g]):
            cambios = {**spec.get("cfg", {}), **{f"{k} (sólo BFR)": v for k, v in spec.get("bfr", {}).items()}}
            f = os.path.join(OUT_ROOT, g, c, "comparacion", "metricas.json")
            if not os.path.exists(f):
                casos.append({"caso": c, "desc": spec["desc"], "cambios": cambios, "falta": True})
                continue
            with open(f) as fh:
                m = json.load(fh)
            videos = {}
            for campo in CAMPOS:
                rel = f"{g}/{c}/comparacion/comparacion_{campo}"
                if os.path.exists(os.path.join(OUT_ROOT, rel + ".mp4")):
                    videos[campo] = {fmt: rel + "." + fmt for fmt in ("webm", "mp4")
                                     if os.path.exists(os.path.join(OUT_ROOT, rel + "." + fmt))}
                    videos[campo]["poster"] = rel + "_final.png"
            m.update(caso=c, cambios=cambios, videos=videos,
                     nota=_nota(f"{g}/{c}", m["resumen"], notas))
            casos.append(m)
        grupos.append({"grupo": g, "desc": gspec["desc"], "pregunta": gspec.get("pregunta", ""),
                       "casos": casos})
    return grupos


def malla(base):
    """Metadatos de la malla compartida + su figura copiada junto al informe."""
    with open(os.path.join(CODIGO_DIR, base["mesh_pkl"]), "rb") as f:
        meta = pickle.load(f).get("meta", {})
    png = os.path.join(os.path.dirname(os.path.join(CODIGO_DIR, base["mesh_pkl"])), "malla.png")
    if os.path.exists(png):
        shutil.copy2(png, os.path.join(OUT_ROOT, "malla.png"))
        meta["figura"] = "malla.png"
    return meta


def main():
    with open(BASE_JSON) as fh:
        base = json.load(fh)
    notas = {}
    if os.path.exists(ANALISIS):
        with open(ANALISIS) as fh:
            notas = json.load(fh)
    grupos = recoger(notas)
    concl = ""
    if os.path.exists(CONCLUSIONES):
        with open(CONCLUSIONES) as fh:
            concl = fh.read()
    datos = json.dumps({"grupos": grupos, "malla": malla(base), "base": {k: base[k] for k in (
        "Q_std", "dt", "T", "theta", "theta_C", "r_pozo", "norm_delta_pozo", "sigma_fuente",
        "n_stencil", "bc_outlet_mode", "spacing", "min_sp", "s_cone", "min_sp_boundary")}},
        separators=(",", ":"))
    with open(os.path.join(os.path.dirname(__file__), "informe_plantilla.html")) as fh:
        plantilla = fh.read()
    out = os.path.join(OUT_ROOT, "informe.html")
    with open(out, "w") as fh:
        fh.write(plantilla.replace("/*__DATOS__*/null", datos.replace("</", "<\\/"))
                 .replace("<!--__CONCLUSIONES__-->", concl))
    n = sum(1 for g in grupos for c in g["casos"] if not c.get("falta"))
    faltan = [f"{g['grupo']}/{c['caso']}" for g in grupos for c in g["casos"] if c.get("falta")]
    viejas = [f"{g['grupo']}/{c['caso']}" for g in grupos for c in g["casos"]
              if c.get("nota") and c["nota"]["desactualizada"]]
    print(f"[informe] {n} casos → {os.path.relpath(out, CODIGO_DIR)}")
    if faltan:
        print("[informe] sin comparación: " + ", ".join(faltan))
    if viejas:
        print("[informe] notas de análisis desactualizadas: " + ", ".join(viejas))


if __name__ == "__main__":
    main()
