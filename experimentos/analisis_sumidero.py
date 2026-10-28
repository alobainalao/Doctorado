"""
Análisis del grupo W_sumidero (decisión 1): con y sin W- = χ(Q) g_ε C.

    python -m experimentos.analisis_sumidero              # grupo W (dominio real)
    python -m experimentos.analisis_sumidero --cuadrado   # grupo Z (cuadrado)

Lee balance_masa.npz y simulation_results.npz de W1_con_sumidero y
W0_sin_sumidero y genera en data/output/experimentos/W_sumidero/:
    balance.png   masa, salidas acumuladas, concentración captada, cierre
    campos.png    C en varios instantes, ambas versiones
    resumen.txt   cifras del balance
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "W_sumidero")
CASOS = {"con $W_-$ (implementado)": "W1_con_sumidero",
         "sin $W_-$ (forma NC correcta)": "W0_sin_sumidero"}
DIA = 86400.0


def cargar(caso):
    d = os.path.join(RAIZ, caso, "fenicsx")
    b = dict(np.load(os.path.join(d, "balance_masa.npz")))
    r = np.load(os.path.join(d, "simulation_results.npz"))
    return b, r


def acumulado(b, clave, factor=1.0):
    return np.cumsum(factor * b[clave]) * float(b["dt"])


def main():
    datos = {et: cargar(c) for et, c in CASOS.items()}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    lineas = []
    for et, (b, _) in datos.items():
        dt = float(b["dt"])
        t = np.arange(1, len(b["masa"]) + 1) * dt / DIA
        m0 = float(b["masa0"])
        agua = acumulado(b, "pozo_agua", b["Q"])          # Q ∫δ C  (sale con el agua)
        sumW = acumulado(b, "sumidero_W")                 # ∫ χ g C (sumidero añadido)
        front = acumulado(b, "frontera_sale")              # ∮ (u·n)⁺ C
        fuente = acumulado(b, "fuente")
        # Cierre del modelo resuelto: d/dt ∫φRC = -∫u·∇C + fuente - W - decaimiento
        rhs = -b["u_grad_C"] + b["fuente"] - b["sumidero_W"] - b["decaimiento"]
        masa_pred = m0 + np.cumsum(rhs) * dt
        cierre = np.abs(masa_pred - b["masa"]).max() / max(b["masa"].max(), m0)

        l, = ax[0, 0].plot(t, b["masa"] / m0, label=et)
        c = l.get_color()
        ax[0, 1].plot(t, agua / m0, color=c, label=f"{et}: con el agua del pozo")
        ax[0, 1].plot(t, sumW / m0, color=c, ls="--", label=f"{et}: sumidero $W_-$")
        ax[0, 1].plot(t, front / m0, color=c, ls=":", label=f"{et}: frontera de salida")
        ax[1, 0].plot(t, b["C_pozo"], color=c, label=et)
        ax[1, 1].plot(t, masa_pred / m0, color=c, ls="--", lw=1)
        ax[1, 1].plot(t, b["masa"] / m0, color=c, lw=0.8, label=f"{et} (cierre {cierre:.1%})")
        lineas.append(
            f"{et}\n"
            f"  masa inicial            {m0:.4e}\n"
            f"  fuente acumulada        {fuente[-1]:.4e}  ({fuente[-1] / m0:.2%} de m0)\n"
            f"  masa final              {b['masa'][-1]:.4e}  ({b['masa'][-1] / m0:.2%} de m0)\n"
            f"  salida con el agua      {agua[-1]:.4e}  ({agua[-1] / m0:.2%} de m0)\n"
            f"  sumidero W-             {sumW[-1]:.4e}  ({sumW[-1] / m0:.2%} de m0)\n"
            f"  frontera de salida      {front[-1]:.4e}  ({front[-1] / m0:.2%} de m0)\n"
            f"  ∫C∇·u acumulado         {acumulado(b, 'C_div_u')[-1]:.4e}\n"
            f"  C captada máx ∫gC       {b['C_pozo'].max():.4e}\n"
            f"  cierre del balance      {cierre:.2%}\n")

    ax[0, 0].set_title("Masa en el dominio $\\int\\phi R C$ / masa inicial")
    ax[0, 1].set_title("Masa que sale (acumulada) / masa inicial")
    ax[1, 0].set_title("Concentración captada por el pozo $\\int g_\\varepsilon C$")
    ax[1, 1].set_title("Cierre: masa (—) frente a la integrada del balance (--)")
    for a in ax.flat:
        a.set_xlabel("t (días)")
        a.axvline(30, color="gray", lw=0.6)
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "balance.png"), dpi=120)

    # Campos de C en varios instantes
    _, r0 = next(iter(datos.values()))
    nodes = r0["nodes"]
    tri = Triangulation(nodes[:, 0], nodes[:, 1])
    nt = r0["C"].shape[0]
    instantes = [0, nt // 12, nt // 4, nt // 2, nt - 1]
    vmax = max(r["C"][0, 0].max() for _, r in datos.values())
    fig, ax = plt.subplots(len(datos), len(instantes), figsize=(20, 5.5), sharex=True, sharey=True)
    for i, (et, (b, r)) in enumerate(datos.items()):
        dt = float(b["dt"])
        for j, k in enumerate(instantes):
            cs = ax[i, j].tripcolor(tri, r["C"][k, 0], vmin=0, vmax=vmax, cmap="viridis", shading="gouraud")
            ax[i, j].set_title(f"{et}\nt = {(k + 1) * dt / DIA:.0f} d", fontsize=8)
            ax[i, j].set_aspect("equal")
    fig.colorbar(cs, ax=ax, shrink=0.6)
    plt.savefig(os.path.join(RAIZ, "campos.png"), dpi=110)

    with open(os.path.join(RAIZ, "resumen.txt"), "w") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))




def cuadrado():
    """Grupo Z_cuadrado (BFR): C = 1 inicial, pozo central, pfs en todo el borde."""
    raiz = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "Z_cuadrado")
    casos = {"con $W_-$ (implementado)": "Z1_con_sumidero",
             "sin $W_-$ (forma NC correcta)": "Z0_sin_sumidero"}
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    a_t, a_x = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    for i, (et, caso) in enumerate(casos.items()):
        d = np.load(os.path.join(raiz, caso, "bfr", "simulation_results.npz"))
        n, C, H = d["nodes"], d["C"][:, 0], d["H"][:, 0]
        dt = float(d["dt"])
        dentro = (n[:, 0] >= 0) & (n[:, 0] <= 1000) & (n[:, 1] >= -1000) & (n[:, 1] <= 0)
        r = np.linalg.norm(n - [500, -500], axis=1)
        g = np.exp(-r ** 2 / (2 * (20 / 3) ** 2)) * dentro
        g /= g.sum()
        t = np.arange(1, len(C) + 1) * dt / DIA
        # Último paso antes de que la carga en los fantasmas del borde diverja
        k = int(np.argmax(H.min(axis=1) < 2 * np.median(H.min(axis=1)))) - 1
        k = len(C) - 1 if k < 0 else k
        a_t.plot(t, C @ g, label=et)
        tri = Triangulation(n[dentro, 0], n[dentro, 1])
        for j, (paso, nom) in enumerate(((0, "1 paso"), (k, f"t = {t[k]:.0f} d"))):
            ax = fig.add_subplot(gs[i, j])
            cs = ax.tripcolor(tri, C[paso][dentro], vmin=0, vmax=1, cmap="viridis", shading="gouraud")
            ax.set_xlim(300, 700); ax.set_ylim(-700, -300); ax.set_aspect("equal")
            ax.set_title(f"{et}, {nom}", fontsize=9)
        fila = dentro & (np.abs(n[:, 1] + 500) < 6)
        o = np.argsort(n[fila, 0])
        a_x.plot(n[fila, 0][o], C[k][fila][o], ".-", ms=3, label=et)
    campos = [a for a in fig.axes if a not in (a_t, a_x)]
    fig.colorbar(cs, ax=campos, shrink=0.6, location="right", label="C")
    a_t.set_title("Concentración captada $\\int g_\\varepsilon C$"); a_t.set_xlabel("t (días)"); a_t.set_ylim(-0.05, 1.05); a_t.legend(fontsize=8)
    a_x.set_title("C en la horizontal del pozo (z = −500)"); a_x.set_xlabel("x (m)"); a_x.set_xlim(300, 700); a_x.legend(fontsize=8)
    plt.savefig(os.path.join(raiz, "cuadrado.png"), dpi=110, bbox_inches="tight")
    print("figura:", os.path.join(raiz, "cuadrado.png"))


if __name__ == "__main__":
    sys.exit(cuadrado() if "--cuadrado" in sys.argv else main())
