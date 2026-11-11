"""
Exporta la malla compartida (data/input/malla_compartida/malla.pkl, la que usan BFR y FEniCSx en la comparación) como
módulo JS para la diapositiva de malla de la presentación (avance3, A3N32): aristas de la triangulación ya proyectadas
al panorama, frontera por grupos y tres ventanas (pozo y dos esquinas, como en malla.png) con sus fantasmas.

    python3 scripts/exportar_malla_presentacion.py [ruta_de_la_presentacion]      # bfr_env
    python3 scripts/exportar_malla_presentacion.py --conos [--malla=<pkl>] [ruta] # variante con conos

Escribe <presentacion>/src/slides-avance3/mallaCompartida.js; con --conos exporta la variante
data/input/malla_conos/malla.pkl (preprocessing/malla_conos.py) a mallaConos.js, con el eje de la pluma, el
contorno del cono vertical del pozo y ventanas sobre el pozo y la pluma (diapositiva A3N34).
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONOS = "--conos" in sys.argv[1:]
_MALLA = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--malla=")]
_ARGS = [a for a in sys.argv[1:] if a != "--conos" and not a.startswith("--malla=")]
PRES = _ARGS[0] if _ARGS else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc-sin-sumidero")
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
MALLA = (os.path.join(CODIGO, _MALLA[0]) if _MALLA else
         os.path.join(CODIGO, "data", "input", "malla_conos" if CONOS else "malla_compartida", "malla.pkl"))

XMIN, XMAX, ZMIN, ZMAX = 816.0, 10755.0, -2690.5, 250.5
W_PAN = 1000                                        # ancho del panorama en unidades SVG
H_PAN = round(W_PAN * (ZMAX - ZMIN) / (XMAX - XMIN))
S_ZOOM = 400                                        # lado de cada ventana en unidades SVG
VENTANAS = [                                        # (clave, título, x0, x1, z0, z1) — las de malla.png
    ("pozo", "Pozo (3700, −700)", 3550.0, 3850.0, -800.0, -600.0),
    ("salida", "Esquina salida/pared (816, −2690)", 700.0, 1150.0, -2750.0, -2410.0),
    ("entrada", "Esquina pared/entrada (10755, −768)", 10450.0, 10800.0, -900.0, -600.0),
]
if CONOS:
    VENTANAS = [
        ("pozo", "Pozo (3700, −700)", 3550.0, 3850.0, -800.0, -600.0),
        ("columna", "Del pozo a la pluma", 3150.0, 4250.0, -850.0, 250.0),
        ("pluma", "Pluma (x ≈ 6000)", 5400.0, 6600.0, -650.0, 250.0),
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

    # Conos del campo de espaciado (funtions/mesh_test.py): círculo r_circle y cono r_cone alrededor de pozo y fuente.
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    RUNTIME.params = Parameters(json_path=os.path.join(CODIGO, "config", "malla_sp60.json"))
    fnte = np.asarray(RUNTIME.params.fnte, float)
    esc = W_PAN / (XMAX - XMIN)
    conos = [dict(nombre=n, x=round(float(px(c[0])), 1), z=round(float(pz(c[1])), 1),
                  r_circulo=round(100.0 * esc, 2), r_cono=round(500.0 * esc, 2))
             for n, c in (("pozo", np.asarray(d["pozo_cor"], float)), ("fuente", fnte))]

    extra = ""
    if CONOS:
        # Eje de la pluma y contorno del cono vertical (preprocessing/malla_conos.py:exceso_pozo: semiancho
        # WELL_R0 desde WELL_R0 bajo el pozo, crece hasta WELL_R1 en el eje de la pluma sobre el pozo).
        from funtions.mesh_test import WELL_R0, WELL_R1
        # Malla recuperada de oct 2026: su eje es el que se usó entonces (data/input/plume_streamline.npy).
        eje = np.asarray(d["eje_pluma"] if "eje_pluma" in d else
                         np.load(os.path.join(CODIGO, "data", "input", "plume_streamline.npy")), float)
        xp, zp = (float(v) for v in d["pozo_cor"])
        z_eje = float(np.interp(xp, eje[::-1, 0], eje[::-1, 1]))
        ancho = lambda z: WELL_R0 + (WELL_R1 - WELL_R0) * np.clip((z - zp) / (z_eje - zp), 0, 1)
        zs = np.array([zp - WELL_R0, zp, z_eje])
        cont = np.r_[np.c_[xp - ancho(zs), zs], np.c_[xp + ancho(zs), zs][::-1]]
        # Elipses de 4 m (preprocessing/malla_conos.py:elipses), como polígonos de 72 lados.
        from preprocessing.malla_conos import elipses
        elip, focos = [], []
        for f1, f2, sb in elipses(RUNTIME.params, eje).values():
            c = 0.5 * np.linalg.norm(f2 - f1)
            sa, u = np.hypot(c, sb), (f2 - f1) / max(2 * c, 1e-12)
            ang = np.linspace(0, 2 * np.pi, 73)[:-1]
            q = 0.5 * (f1 + f2) + np.outer(sa * np.cos(ang), u) + np.outer(sb * np.sin(ang), [-u[1], u[0]])
            elip.append(np.round(np.c_[px(q[:, 0]), pz(q[:, 1])], 1).ravel().tolist())
            focos += [[round(float(px(f[0])), 1), round(float(pz(f[1])), 1)] for f in (f1, f2)]
        extra = (f"export const EJE = {json.dumps(np.round(np.c_[px(eje[:, 0]), pz(eje[:, 1])], 1).ravel().tolist())};\n"
                 f"export const CONO_POZO = {json.dumps(np.round(np.c_[px(cont[:, 0]), pz(cont[:, 1])], 1).ravel().tolist())};\n"
                 f"export const ELIPSES = {json.dumps(elip)};\n"
                 f"export const FOCOS = {json.dumps(focos)};\n")
        for v, (_, _, x0, x1, z0, z1) in zip(ventanas, VENTANAS):
            e = S_ZOOM / max(x1 - x0, z1 - z0)
            v["eje"] = np.round(np.c_[(eje[:, 0] - x0) * e, (z1 - eje[:, 1]) * e], 1).ravel().tolist()

    meta = d["meta"]
    info = dict(sha=hashlib.sha256(nodos.tobytes()).hexdigest()[:12], fecha=meta.get("fecha"), n_dominio=int(nd),
                n_fantasmas=int(len(nodos) - nd), n_triangulos=int(len(t)), ang_min=round(meta["calidad"]["ang_min"], 1),
                ang_max=round(meta["calidad"]["ang_max"], 0), W=W_PAN, H=H_PAN)
    js = (f"// Generado por codigo/scripts/exportar_malla_presentacion.py desde {os.path.relpath(MALLA, CODIGO)}\n"
          + ("// (variante con conos de pluma y pozo). No editar a mano.\n" if CONOS else
             "// (la malla de la comparación BFR vs FEniCSx). No editar a mano.\n") +
          f"export const INFO = {json.dumps(info)};\n"
          f"export const PANORAMA = {json.dumps(pan)};\n"
          f"export const FRONTERA = {json.dumps(frontera)};\n"
          f"export const VENTANAS = {json.dumps(ventanas, ensure_ascii=False)};\n"
          f"export const CONOS = {json.dumps(conos)};\n" + extra)
    ruta = os.path.join(PRES, "src", "slides-avance3", "mallaConos.js" if CONOS else "mallaCompartida.js")
    with open(ruta, "w") as f:
        f.write(js)
    print("escrito", ruta, f"{len(js) // 1024} KB", info)


if __name__ == "__main__":
    main()
