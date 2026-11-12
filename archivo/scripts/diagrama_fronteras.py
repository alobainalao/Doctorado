"""
Diagrama de las condiciones de frontera de h y C en FEniCSx y BFR (configuración estándar:
bc_outlet_mode = "pfs"). Cada condición está leída de:
  FEniCSx: fenicsx/forward.py (_boundaries_elements: bc_C = []; forma débil con h_inlet en ds_in y h_outlet en ds_out)
  BFR:     funtions/operators.py (make_H_bc, make_transport_bc, make_adj_H_bc, make_adj_C_bc)
Salida: docs/diagrama_condiciones_frontera.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = [(816, -2690.5), (4354, -1757.5), (10755, -768.5), (10755, 250.5), (816, 250.5)]
ROJO, AZUL, VERDE = "#b30000", "#1f4e9c", "#2e7d32"

PANELES = {
    ("FEniCSx (elementos finitos)", "Flujo  h"): dict(
        arriba="$(K\\nabla h)\\cdot n=0$  (natural)",
        izq="Γout\nflujo PRESCRITO:\n$(K\\nabla h)\\cdot n=h_{out}(z)$\n(natural)",
        der="Γin\n$-(K\\nabla h)\\cdot n=h_{in}(z)$\n$=-f_{in}$  (natural)\n$u\\cdot n=-f_{in}\\leq0$",
        abajo="$(K\\nabla h)\\cdot n=0$  (natural)"),
    ("BFR (RBF-FD)", "Flujo  h"): dict(
        arriba="$(K\\nabla h)\\cdot n=0$  (fila fantasma)",
        izq="Γout  (modo «pfs»)\nLIBRE, 2.º orden:\n$\\nabla(K\\nabla h)\\cdot n=0$\n(no prescribe flujo)",
        der="Γin\n$-(K\\nabla h)\\cdot n=h_{in}(z)$\n$=-f_{in}$  (fila fantasma)\n$u\\cdot n=-f_{in}\\leq0$",
        abajo="$(K\\nabla h)\\cdot n=0$  (fila fantasma)"),
    ("FEniCSx (elementos finitos)", "Transporte  C"): dict(
        arriba="$(D\\nabla C)\\cdot n=0$  (natural)",
        izq="Γout\n$(D\\nabla C)\\cdot n=0$\n(natural; la advección sale\nlibre, sin término de borde)",
        der="Γin\n$(D\\nabla C)\\cdot n=0$\n(natural; sin Dirichlet,\n`bc_C = []`)",
        abajo="$(D\\nabla C)\\cdot n=0$  (natural)"),
    ("BFR (RBF-FD)", "Transporte  C"): dict(
        arriba="$(D\\nabla C)\\cdot n=0$  (fila fantasma)",
        izq="Γout\n$(D\\nabla C)\\cdot n=0$\n(fila fantasma)",
        der="Γin\n$(D\\nabla C)\\cdot n=0$\n(fila fantasma)",
        abajo="$(D\\nabla C)\\cdot n=0$  (fila fantasma)"),
}


def panel(ax, tit, d):
    ax.add_patch(Polygon(V, closed=True, fc="#eef3fb", ec="black", lw=1.6))
    xs = [v[0] for v in V]; zs = [v[1] for v in V]
    ax.set_xlim(min(xs) - 5200, max(xs) + 5200)
    ax.set_ylim(min(zs) - 1400, max(zs) + 1100)
    ax.set_aspect("auto"); ax.axis("off")
    ax.set_title(f"{tit[0]} — {tit[1]}", fontsize=12, fontweight="bold", color=AZUL if "Flujo" in tit[1] else VERDE)
    ax.text(5800, 700, d["arriba"], ha="center", va="bottom", fontsize=9.5, color=ROJO)
    ax.text(7300, -2350, d["abajo"] + "\n(fondo, 2 tramos)", ha="center", va="top", fontsize=9.5, color=ROJO)
    ax.text(10755 + 250, -250, d["der"], ha="left", va="center", fontsize=9.5, color=ROJO)
    ax.text(816 - 250, -250, d["izq"], ha="right", va="center", fontsize=9.5, color=ROJO)
    ax.annotate("", xy=(2300, -1200), xytext=(9000, -1200),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.text(5650, -1050, "flujo (derecha → izquierda)", ha="center", va="bottom", fontsize=8.5, color="gray")


def main():
    fig, axs = plt.subplots(2, 2, figsize=(17, 9))
    for (i, rec), ax in zip(enumerate([("FEniCSx (elementos finitos)", "Flujo  h"), ("BFR (RBF-FD)", "Flujo  h"),
                                       ("FEniCSx (elementos finitos)", "Transporte  C"), ("BFR (RBF-FD)", "Transporte  C")]),
                            axs.ravel()):
        panel(ax, rec, PANELES[rec])
    fig.suptitle("Condiciones de frontera del estado (configuración estándar, n = normal exterior; Γin = derecha, Γout = izquierda)",
                 fontsize=13)
    fig.text(0.5, 0.012,
             "BFR implementa estas filas por componentes con K y D diagonales (p. ej. en Γin: $-K_x\\partial_x h$, pues $n=e_x$).\n"
             "Adjunto — FEniCSx: todas naturales (derivadas del residuo).  BFR: $\\psi_C=0$ y $\\psi_h=0$ en Γin y Γout "
             "(filas de Dirichlet); fondo: Neumann $(D\\nabla\\psi_C)\\cdot n=0$, $(K\\nabla\\psi_h)\\cdot n=0$.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    out = os.path.join(CODIGO, "docs", "diagrama_condiciones_frontera.png")
    fig.savefig(out, dpi=130)
    print("escrito", out)


if __name__ == "__main__":
    main()
