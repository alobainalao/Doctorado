"""
¿Sale el contaminante por el pozo sin el sumidero W-? Balance de soluto de los
casos de limpieza del grupo S (fuente 30 d, después 90 d sin fuente):
S3_limpieza_con (factor_sumidero_C = 1) y S2_limpieza_sin (= 0), BFR y FEniCS.

    python -m experimentos.limpieza_sumidero

Términos (por paso, integrados en el tiempo):
    masa        ∫φRC
    agua        Q ∫δC        soluto que sale con el agua bombeada
    W           ∫χ g_ε C     sumidero W- (sólo con el término)
    frontera    ∮(u·n)⁺C     sale por la frontera (u·n > 0)
    fuente      ∫S
En la forma no conservativa la salida con el agua NO es un término de la
ecuación: la produce −∫u·∇C = −∮(u·n)C + ∫C∇·u, con ∇·u = −Qδ − S_s∂_t h.
FEniCS guarda los términos en balance_masa.npz (forward con balance_masa);
en BFR se reconstruyen de C, U y la malla compartida (grupos y normales).

Salida en data/output/experimentos/S_sumidero/: limpieza.png, limpieza.txt
"""
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)
RAIZ = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "S_sumidero")
CASOS = (("S3_limpieza_con", "con W₋"), ("S2_limpieza_sin", "sin W₋"))
DIA = 86400.0


def _fem(caso):
    d = os.path.join(RAIZ, caso, "fenicsx")
    b = dict(np.load(os.path.join(d, "balance_masa.npz")))
    dt = float(b["dt"])
    return dict(t=np.arange(1, len(b["masa"]) + 1) * dt / DIA, dt=dt, masa=b["masa"],
                agua=b["Q"] * b["pozo_agua"], W=b["sumidero_W"], frontera=b["frontera_sale"],
                entra=b["frontera_entra"], fuente=b["fuente"], decae=b["decaimiento"],
                u_grad_C=b["u_grad_C"], C_pozo=b["C_pozo"])


def _bfr(caso):
    """Balance reconstruido de simulation_results.npz y la malla compartida."""
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from funtions.utils import discrete_delta, normalizar_delta_area, gaussian_2d, fuente_C
    from funtions.operators import rbf_integration_weights
    import funtions.get_phi as gph

    d = os.path.join(RAIZ, caso, "bfr")
    cfg = json.load(open(os.path.join(d, "config.json")))
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.params
    z = np.load(os.path.join(d, "simulation_results.npz"))
    n, C, U, dt = z["nodes"], z["C"][:, 0], z["U"][:, 0], float(z["dt"])
    m = pickle.load(open(os.path.join(CODIGO_DIR, cfg["mesh_pkl"]), "rb"))
    if not np.allclose(m["nodes"], n):
        raise RuntimeError("los nodos BFR no son los de la malla compartida")
    g = m["groups"]
    dom = np.hstack([g["interior"], g["boundary:inlet"], g["boundary:outlet"], g["boundary:wall"]]).astype(int)
    # Áreas nodales con los triángulos de la malla compartida (sin fantasmas)
    tri = np.asarray(m["triangles"])
    q = n[tri]
    a = 0.5 * np.abs((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                     - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))
    w = np.zeros(len(n)); np.add.at(w, tri.ravel(), np.repeat(a / 3, 3))
    phi = gph.gen_malla(n)
    pozo = np.asarray(m["pozo_cor"], float)
    delta = discrete_delta(n, pozo, p.radio_delta_pozo)
    if p.norm_delta_pozo == "area":
        delta = normalizar_delta_area(delta, rbf_integration_weights(n))
    gp = gaussian_2d(n, pozo, 1, p.epsilon_pozo)
    gf = gaussian_2d(n, np.asarray(p.fnte, float), 1, p.epsilon_x, p.epsilon_y)
    # Longitud de frontera asociada a cada nodo de entrada/salida (media
    # distancia a sus dos vecinos del mismo tramo)
    lon, idx_b = [], []
    for grupo in ("boundary:inlet", "boundary:outlet"):
        ib = np.asarray(g[grupo], int)
        tang = np.c_[-m["normals"][ib, 1], m["normals"][ib, 0]]
        s = n[ib] @ tang[0]
        o = np.argsort(s); s = s[o]; ib = ib[o]
        ds = np.diff(s)
        lon.append(0.5 * (np.r_[ds[0], ds] + np.r_[ds, ds[-1]])); idx_b.append(ib)
    lon, idx_b = np.concatenate(lon), np.concatenate(idx_b)
    un = np.einsum("kij,ij->ki", U[:, idx_b], m["normals"][idx_b])
    t = np.arange(1, len(C) + 1) * dt / DIA
    t_off = getattr(p, "t_fuente_off", None)
    Tf = float(getattr(p, "T_fuente", None) or p.T)
    amp = np.array([0.0 if (t_off is not None and ti * DIA > float(t_off)) else fuente_C(ti * DIA, Tf) for ti in t])
    Q = float(p.Q_std)
    return dict(t=t, dt=dt, masa=(C[:, dom] * p.R * phi[dom]) @ w[dom],
                agua=Q * (C[:, dom] @ (delta[dom] * w[dom])),
                W=p.factor_sumidero_C * (C[:, dom] @ (gp[dom] * w[dom])),
                frontera=(np.maximum(un, 0) * C[:, idx_b]) @ lon,
                entra=(np.minimum(un, 0) * C[:, idx_b]) @ lon,
                fuente=amp * float(gf[dom] @ w[dom]) if p.activate_fuente else 0 * t,
                decae=p.landa * p.R * ((C[:, dom] * phi[dom]) @ w[dom]),
                C_pozo=C[:, dom] @ (gp[dom] * w[dom]))


def main():
    datos = {(b, c): f(c) for b, f in (("BFR (RBF-FD)", _bfr), ("FEniCS (FEM)", _fem)) for c, _ in CASOS}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    lineas = []
    for i, b in enumerate(("BFR (RBF-FD)", "FEniCS (FEM)")):
        for j, (c, nom) in enumerate(CASOS):
            x = datos[(b, c)]
            t, dt = x["t"], x["dt"]
            k30 = int(np.argmin(np.abs(t - 30)))
            m30 = x["masa"][k30]
            acum = {k: np.cumsum(x[k]) * dt for k in ("agua", "W", "frontera", "entra", "fuente", "decae")}
            col = f"C{j}"
            ax[i, 0].plot(t, x["masa"] / m30, color=col, label=nom)
            for k, ls in (("agua", "-"), ("W", "--"), ("frontera", ":")):
                if k == "W" and not np.any(x["W"]):
                    continue
                ax[i, 1].plot(t, (acum[k] - acum[k][k30]) / m30, color=col, ls=ls,
                              label=f"{nom}: {dict(agua='con el agua', W='sumidero W₋', frontera='frontera')[k]}")
            ax[i, 2].plot(t, x["C_pozo"], color=col, label=nom)
            sale = {k: (acum[k][-1] - acum[k][k30]) / m30 for k in ("agua", "W", "frontera")}
            perdida = (m30 - x["masa"][-1]) / m30
            lineas.append(
                f"{b:14s} {nom:7s}  masa a 120 d = {x['masa'][-1] / m30:6.1%} de la de 30 d  "
                f"(bajó {perdida:6.1%})  |  de 30 a 120 d sale: con el agua {sale['agua']:6.1%}, "
                f"W₋ {sale['W']:6.1%}, frontera {sale['frontera']:6.1%}  |  C captada 120 d = {x['C_pozo'][-1]:.3e}")
        ax[i, 0].set_title(f"{b}: masa ∫φRC / masa a 30 d")
        ax[i, 1].set_title(f"{b}: soluto que sale desde 30 d / masa a 30 d")
        ax[i, 2].set_title(f"{b}: C captada ∫g_ε C")
        for a in ax[i]:
            a.axvline(30, color="gray", lw=0.6); a.set_xlabel("t (días)"); a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "limpieza.png"), dpi=110)
    with open(os.path.join(RAIZ, "limpieza.txt"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))
    print("figura:", os.path.join(RAIZ, "limpieza.png"))


if __name__ == "__main__":
    main()
