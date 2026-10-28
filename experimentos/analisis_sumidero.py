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
sys.path.insert(0, CODIGO_DIR)
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
    """Grupo Z_cuadrado (BFR): cuadrado contaminado, pozo central, flujo con la
    condición libre en todo el borde (estacionario en horas). Balance de soluto:
        d/dt ∫φC = −Q·∫δC (sale con el agua) − ∫χ g C (W-, sólo Z1),
    pues en el borde sólo entra agua y su C es nula."""
    import json
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from funtions.utils import discrete_delta, normalizar_delta_area, gaussian_2d
    from funtions.operators import rbf_integration_weights

    raiz = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "Z_cuadrado")
    casos = {"con $W_-$ (implementado)": "Z1_con_sumidero",
             "sin $W_-$ (forma NC correcta)": "Z0_sin_sumidero"}
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.1])
    a_m, a_s, a_c = fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]), fig.add_subplot(gs[2, 4])
    lineas = []
    for fila, (et, caso) in enumerate(casos.items()):
        d_caso = os.path.join(raiz, caso, "bfr")
        cfg = json.load(open(os.path.join(d_caso, "config.json")))
        RUNTIME.params = Parameters(config=cfg)
        p = RUNTIME.params
        d = np.load(os.path.join(d_caso, "simulation_results.npz"))
        n, C, dt = d["nodes"], d["C"][:, 0], float(d["dt"])
        L = float(p.lado_cuadrado)
        dentro = (n[:, 0] >= -1e-9) & (n[:, 0] <= L + 1e-9) & (n[:, 1] >= -L - 1e-9) & (n[:, 1] <= 1e-9)
        tri = Triangulation(n[dentro, 0], n[dentro, 1])
        w = np.zeros(len(n))                       # áreas nodales del dominio (sin fantasmas)
        idx = np.nonzero(dentro)[0]
        for t3 in tri.triangles:
            q = n[idx[t3]]
            w[idx[t3]] += 0.5 * abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1])) / 3
        pozo = np.asarray(p.pozo, float)
        delta = discrete_delta(n, pozo, p.radio_delta_pozo)
        if p.norm_delta_pozo == "area":
            delta = normalizar_delta_area(delta, rbf_integration_weights(n))
        g = gaussian_2d(n, pozo, 1, p.epsilon_pozo)
        C0 = float(p.C_init_const)
        t = np.arange(1, len(C) + 1) * dt / DIA
        masa = (C * p.phi_const * p.R) @ w
        m0 = float(C[0] * p.phi_const * p.R @ w)
        agua = np.cumsum(p.Q_std * (C @ (delta * w))) * dt
        sumW = np.cumsum(C @ (g * w)) * dt if p.extract_C else np.zeros_like(t)
        c_pozo = C @ (g * w)
        borde = dentro & (np.minimum.reduce([n[:, 0], L - n[:, 0], n[:, 1] + L, -n[:, 1]]) < 1e-6)
        l, = a_m.plot(t, masa / m0, label=et)
        col = l.get_color()
        a_s.plot(t, agua / m0, color=col, label=f"{et}: con el agua")
        if p.extract_C:
            a_s.plot(t, sumW / m0, color=col, ls="--", label=f"{et}: sumidero $W_-$")
        a_c.plot(t, c_pozo / C0, color=col, label=et)
        for j, dia in enumerate((0, 10, 20, 35, 60)):
            k = min(int(np.argmin(np.abs(t - max(dia, t[0])))), len(t) - 1)
            ax = fig.add_subplot(gs[fila, j])
            cs = ax.tripcolor(tri, C[k][dentro] / C0, vmin=0, vmax=1, cmap="viridis", shading="gouraud")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{et}, t = {t[k]:.0f} d", fontsize=8)
        k_l = np.argmax(masa / m0 < 0.01) if (masa / m0 < 0.01).any() else None
        lineas.append(
            f"{et}\n"
            f"  masa inicial ∫φRC        {m0:.4e}\n"
            f"  masa final               {masa[-1] / m0:.3%} de m0\n"
            f"  salida con el agua       {agua[-1] / m0:.3%} de m0\n"
            f"  sumidero W-              {sumW[-1] / m0:.3%} de m0\n"
            f"  cierre del balance       {abs(m0 - masa[-1] - agua[-1] - sumW[-1]) / m0:.3%} de m0\n"
            f"  masa < 1 % de m0 en      {'%.1f d' % t[k_l] if k_l is not None else 'no se alcanza'}\n"
            f"  C máx en el borde / C0   {np.abs(C[:, borde]).max() / C0:.2e}\n"
            f"  C mín / C0 (oscilación)  {C[:, dentro].min() / C0:.2e}\n")
    fig.colorbar(cs, ax=fig.axes[3:], shrink=0.5, location="right", label="$C/C_0$")
    a_m.set_title("Masa en el dominio $\\int\\phi R C$ / masa inicial"); a_m.set_yscale("log"); a_m.set_ylim(1e-4, 1.5)
    a_s.set_title("Masa que sale (acumulada) / masa inicial")
    a_c.set_title("Concentración captada $\\int g_\\varepsilon C$ / $C_0$")
    for a in (a_m, a_s, a_c):
        a.set_xlabel("t (días)"); a.legend(fontsize=7)
    plt.savefig(os.path.join(raiz, "cuadrado.png"), dpi=110, bbox_inches="tight")
    with open(os.path.join(raiz, "resumen.txt"), "w") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))
    print("figura:", os.path.join(raiz, "cuadrado.png"))


if __name__ == "__main__":
    sys.exit(cuadrado() if "--cuadrado" in sys.argv else main())
