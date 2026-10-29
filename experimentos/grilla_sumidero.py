"""
Grilla 2×2 del grupo S_sumidero: C en filas BFR / FEniCS y columnas con / sin
el sumidero W- del pozo (factor_sumidero_C = 1 / 0).

    python -m experimentos.grilla_sumidero [--campo C|H|V]

Salida en data/output/experimentos/S_sumidero/:
    grilla_<campo>.mp4         video 2×2 (mismo rango de color en los 4 paneles)
    grilla_<campo>_final.png   último cuadro
    resumen.txt                masa ∫φC final / inicial y C captada en el pozo
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.animar import (  # noqa: E402
    CAMPOS, LABEL_B, OUT_ROOT, WRITER_KW, _actualizar, _cargar, _panel, _rangos, _t_dias,
)

GRUPO = "S_sumidero"
COLUMNAS = (("S1_con_termino", "con W₋"), ("S0_sin_termino", "sin W₋"))
FILAS = ("bfr", "fenicsx")


def _area_nodal(nodes):
    """Área nodal (1/3 de cada triángulo de Delaunay) para integrar en el dominio."""
    from scipy.spatial import Delaunay
    tri = Delaunay(nodes)
    p = nodes[tri.simplices]
    a = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                     - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    w = np.zeros(len(nodes))
    np.add.at(w, tri.simplices.ravel(), np.repeat(a / 3, 3))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campo", default="C", choices=[c for c, _, _ in CAMPOS])
    campo = ap.parse_args().campo
    etiqueta, cmap = {c: (e, m) for c, e, m in CAMPOS}[campo]

    raiz = os.path.join(OUT_ROOT, GRUPO)
    datos = {(b, c): _cargar(os.path.join(raiz, c, b, "simulation_results.npz"))
             for b in FILAS for c, _ in COLUMNAS}
    rangos = _rangos(list(datos.values()))
    nt = min(d[campo].shape[0] for d in datos.values())

    fig, axs = plt.subplots(2, 2, figsize=(16, 7.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.1, wspace=0.03, hspace=0.18)
    paneles = {}
    for i, b in enumerate(FILAS):
        for j, (c, nom) in enumerate(COLUMNAS):
            ax = axs[i, j]
            paneles[(b, c)] = _panel(ax, datos[(b, c)], campo, *rangos[campo], cmap)
            ax.set_title(f"{LABEL_B[b]}  ·  {nom}", fontsize=12)
    im = paneles[(FILAS[0], COLUMNAS[0][0])][0]
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    fig.colorbar(im, cax=cax, orientation="horizontal", label=etiqueta)
    sup = fig.suptitle("", fontsize=14)

    out = os.path.join(raiz, f"grilla_{campo}.mp4")
    writer = FFMpegWriter(**WRITER_KW)
    with writer.saving(fig, out, dpi=100):
        for k in range(nt):
            for clave, (im_, q) in paneles.items():
                _actualizar(im_, q, datos[clave], campo, k)
            sup.set_text(f"{etiqueta}: sumidero W₋ del pozo, con y sin el término"
                         f"   |   t = {_t_dias(datos[(FILAS[0], COLUMNAS[0][0])], k):.2f} días")
            writer.grab_frame()
    fig.savefig(os.path.join(raiz, f"grilla_{campo}_final.png"), dpi=100)
    plt.close(fig)
    print("video:", out)

    # Resumen: ∫C (φ del caso no está en el npz) y C captada con la gaussiana del pozo
    import json
    lineas = []
    for b in FILAS:
        for c, nom in COLUMNAS:
            z = np.load(os.path.join(raiz, c, b, "simulation_results.npz"))
            d = datos[(b, c)]
            cfg = json.load(open(os.path.join(raiz, c, b, "config.json")))
            nodes = np.asarray(z["nodes"], float)
            from experimentos.animar import DOM_PATH
            nodes = nodes[DOM_PATH.contains_points(nodes, radius=1e-3)]
            w = _area_nodal(nodes)
            m = d["C"] @ w
            from config.parameters import Parameters
            p = Parameters(config=cfg)                    # pozo y ε del kernel derivados
            r = np.linalg.norm(nodes - np.asarray(p.pozo, float), axis=1)
            eps = float(p.epsilon_pozo)
            g = np.exp(-r ** 2 / (2 * eps ** 2)); g /= g @ w
            cap = d["C"] @ (g * w)
            lineas.append(f"{LABEL_B[b]:14s} {nom:7s}  ∫C final/máx = {m[-1] / m.max():.4f}   "
                          f"C captada final = {cap[-1]:.4e}   máx = {cap.max():.4e}")
    with open(os.path.join(raiz, "resumen.txt"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
