"""
Exporta la malla compartida (data/input/malla_compartida/malla.pkl, la que usan BFR y FEniCSx en la comparación) como
módulo JS para la diapositiva de malla de la presentación (avance3, A3N32): aristas de la triangulación ya proyectadas
al panorama, frontera por grupos y tres ventanas (pozo y dos esquinas, como en malla.png) con sus fantasmas.

    python3 scripts/exportar_malla_presentacion.py [ruta_de_la_presentacion]      # bfr_env

Escribe <presentacion>/src/slides-avance3/mallaCompartida.js
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc-sin-sumidero")
MALLA = os.path.join(CODIGO, "data", "input", "malla_compartida", "malla.pkl")

XMIN, XMAX, ZMIN, ZMAX = 816.0, 10755.0, -2690.5, 250.5
W_PAN = 1000                                        # ancho del panorama en unidades SVG
H_PAN = round(W_PAN * (ZMAX - ZMIN) / (XMAX - XMIN))
S_ZOOM = 400                                        # lado de cada ventana en unidades SVG
VENTANAS = [                                        # (clave, título, x0, x1, z0, z1) — las de malla.png
    ("pozo", "Pozo (3700, −700)", 3550.0, 3850.0, -800.0, -600.0),
    ("salida", "Esquina salida/pared (816, −2690)", 700.0, 1150.0, -2750.0, -2410.0),
    ("entrada", "Esquina pared/entrada (10755, −768)", 10450.0, 10800.0, -900.0, -600.0),
]


def _aristas(t):
    e = np.sort(np.r_[t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=1)
    return np.unique(e, axis=0)


def main():
    d = pickle.load(open(MALLA, "rb"))
    nodos = np.asarray(d["nodes"], float)
    t = np.asarray(d["triangles"], int)
    nd = t.max() + 1
    g = d["groups"]
    E = _aristas(t)

    px = lambda x: (x - XMIN) / (XMAX - XMIN) * W_PAN
    pz = lambda z: (ZMAX - z) / (ZMAX - ZMIN) * H_PAN
    P = np.c_[px(nodos[:, 0]), pz(nodos[:, 1])]
    pan = np.round(np.c_[P[E[:, 0]], P[E[:, 1]]], 1).ravel().tolist()

    frontera = {}
    for k in ("inlet", "outlet", "wall"):
        idx = np.asarray(g.get(f"boundary:{k}", []), int)
        frontera[k] = np.round(P[idx], 1).ravel().tolist()

    ventanas = []
    for clave, titulo, x0, x1, z0, z1 in VENTANAS:
        esc = S_ZOOM / max(x1 - x0, z1 - z0)
        wx, wz = (x1 - x0) * esc, (z1 - z0) * esc
        lx = lambda x: (x - x0) * esc
        lz = lambda z: (z1 - z) * esc
        cen = nodos[t].mean(1)
        m = 0.15 * max(x1 - x0, z1 - z0)
        sel = (cen[:, 0] > x0 - m) & (cen[:, 0] < x1 + m) & (cen[:, 1] > z0 - m) & (cen[:, 1] < z1 + m)
        Ez = _aristas(t[sel])
        L = np.c_[lx(nodos[:, 0]), lz(nodos[:, 1])]
        dentro = lambda i: (nodos[i, 0] >= x0) & (nodos[i, 0] <= x1) & (nodos[i, 1] >= z0) & (nodos[i, 1] <= z1)
        v = dict(clave=clave, titulo=titulo, w=round(wx, 1), h=round(wz, 1),
                 aristas=np.round(np.c_[L[Ez[:, 0]], L[Ez[:, 1]]], 1).ravel().tolist(), frontera={}, fantasmas=[])
        for k in ("inlet", "outlet", "wall"):
            idx = np.asarray(g.get(f"boundary:{k}", []), int)
            idx = idx[dentro(idx)]
            v["frontera"][k] = np.round(L[idx], 1).ravel().tolist()
        gh = np.concatenate([np.asarray(g.get(f"ghosts:{k}", []), int) for k in ("inlet", "outlet", "wall")])
        gh = gh[dentro(gh)]
        v["fantasmas"] = np.round(L[gh], 1).ravel().tolist()
        if clave == "pozo":
            v["pozo"] = [round(float(lx(d["pozo_cor"][0])), 1), round(float(lz(d["pozo_cor"][1])), 1)]
        ventanas.append(v)

    meta = d["meta"]
    info = dict(sha=hashlib.sha256(nodos.tobytes()).hexdigest()[:12], fecha=meta.get("fecha"), n_dominio=int(nd),
                n_fantasmas=int(len(nodos) - nd), n_triangulos=int(len(t)), ang_min=round(meta["calidad"]["ang_min"], 1),
                ang_max=round(meta["calidad"]["ang_max"], 0), W=W_PAN, H=H_PAN)
    js = ("// Generado por codigo/scripts/exportar_malla_presentacion.py desde data/input/malla_compartida/malla.pkl\n"
          "// (la malla de la comparación BFR vs FEniCSx). No editar a mano.\n"
          f"export const INFO = {json.dumps(info)};\n"
          f"export const PANORAMA = {json.dumps(pan)};\n"
          f"export const FRONTERA = {json.dumps(frontera)};\n"
          f"export const VENTANAS = {json.dumps(ventanas, ensure_ascii=False)};\n")
    ruta = os.path.join(PRES, "src", "slides-avance3", "mallaCompartida.js")
    with open(ruta, "w") as f:
        f.write(js)
    print("escrito", ruta, f"{len(js) // 1024} KB", info)


if __name__ == "__main__":
    main()
