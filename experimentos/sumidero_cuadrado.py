"""
Cuadrado con pozo central (grupo Z, decisión 1): páginas animadas del flujo y
del transporte con y sin el sumidero W- = χ(Q) g_ε C.

    python -m experimentos.sumidero_cuadrado flujo        # corre el flujo con 3 BC + página
    python -m experimentos.sumidero_cuadrado transporte   # página desde Z0/Z1 ya corridos

Flujo: transitorio de carga en el cuadrado del grupo Z (dt = 300 s el primer
día y dt = 18000 s hasta 90 d), sobre la misma malla, con tres tratamientos de
la condición libre ∂_n(u·n) = 0 en todo el borde:
    libre     fantasma + ecuación de borde S_s ∂_t H = ∂_τ(K_τ ∂_τ H)
    pfs_nn    sólo en el fantasma
    pfs       condición programada de la maestría (término cruzado)
Transporte: lee Z1_con_sumidero y Z0_sin_sumidero (correr --grupo Z_cuadrado)
y anima H, C con W- y C sin W-, con la masa ∫φC y la concentración captada.

Salida en data/output/experimentos/Z_cuadrado/:
    flujo/{libre,pfs_nn,pfs}.npz, flujo.html, transporte.html
"""
import base64
import io
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation, LinearTriInterpolator  # noqa: E402
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402
from PIL import Image  # noqa: E402

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)
RAIZ = os.path.join(CODIGO_DIR, "data", "output", "experimentos", "Z_cuadrado")
PLANTILLAS = os.path.join(CODIGO_DIR, "experimentos", "plantillas")
DIA = 86400.0
MODOS = {"continuo": "libre", "fantasma_nn": "pfs_nn", "programada": "pfs"}   # clave de la página → BC


# ── utilidades de fotogramas ───────────────────────────────────────────────
def _png(fig):
    """Figura → PNG base64 con paleta de 96 colores (≈ 4× más pequeño)."""
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=80, transparent=True); plt.close(fig)
    im = Image.open(io.BytesIO(b.getvalue())).convert("RGBA").quantize(colors=96, method=Image.Quantize.FASTOCTREE)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


class Rejilla:
    """Interpolación de un campo nodal a una rejilla regular que cubre todo el
    cuadrado (lineal en la triangulación + vecino más cercano fuera del casco)."""

    def __init__(self, nodes, dom, L):
        self.n, self.dom, self.L = nodes, dom, L
        self.tri = Triangulation(nodes[dom, 0], nodes[dom, 1])
        self.g1 = np.linspace(0, L, 151)
        self.GX, self.GZ = np.meshgrid(self.g1, self.g1 - L)
        a = np.linspace(L / 24, L - L / 24, 12)
        self.AX, self.AZ = np.meshgrid(a, a - L)
        self.lejos = np.hypot(self.AX - L / 2, self.AZ + L / 2) > L / 15

    def campo(self, v):
        I = np.ma.filled(LinearTriInterpolator(self.tri, v)(self.GX, self.GZ), np.nan)
        hueco = ~np.isfinite(I)
        if hueco.any():
            I[hueco] = NearestNDInterpolator(self.n[self.dom], v)(self.GX[hueco], self.GZ[hueco])
        return I

    def lienzo(self):
        fig = plt.figure(figsize=(3.6, 3.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.set_xlim(0, self.L); ax.set_ylim(-self.L, 0); ax.set_aspect("equal")
        return fig, ax

    def carga(self, h, niveles):
        """Carga con isolíneas y dirección de −∇h."""
        I = self.campo(h)
        fig, ax = self.lienzo()
        ax.contourf(self.GX, self.GZ, I, levels=niveles, cmap="viridis", extend="both")
        if h.max() - h.min() > 1e-6:
            ax.contour(self.GX, self.GZ, I, levels=np.linspace(niveles[0], niveles[-1], 9)[1:-1],
                       colors="w", linewidths=0.6, alpha=0.8)
            dz, dx = np.gradient(I, self.g1, self.g1)
            ux = RegularGridInterpolator((self.g1 - self.L, self.g1), -dx)((self.AZ, self.AX))
            uz = RegularGridInterpolator((self.g1 - self.L, self.g1), -dz)((self.AZ, self.AX))
            m = np.hypot(ux, uz); m[m == 0] = 1
            s = self.lejos
            ax.quiver(self.AX[s], self.AZ[s], (ux / m)[s], (uz / m)[s], color="k", alpha=0.7,
                      pivot="mid", scale=22, width=0.006, headwidth=4, headlength=4.5, headaxislength=4)
        return _png(fig)


def _areas(nodes, dom, tri):
    """Área nodal (1/3 de cada triángulo) de los nodos del dominio."""
    w = np.zeros(len(dom))
    for t3 in tri.triangles:
        q = nodes[dom[t3]]
        w[t3] += 0.5 * abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1])
                           - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1])) / 3
    return w


def _pagina(plantilla, datos, salida):
    s = open(os.path.join(PLANTILLAS, plantilla)).read()
    with open(salida, "w") as f:
        f.write(s.replace("__DATOS__", json.dumps(datos, separators=(",", ":"))))
    print("página:", salida)


def _config_z():
    """Config BFR completa del caso Z0 (base.json + catálogo), sin correrlo."""
    from experimentos.catalogo import GRUPOS, config_caso
    base = json.load(open(os.path.join(CODIGO_DIR, "experimentos", "base.json")))
    spec = GRUPOS["Z_cuadrado"]["casos"]["Z0_sin_sumidero"]
    return config_caso(base, "Z_cuadrado", "Z0_sin_sumidero", spec, "bfr", os.path.dirname(RAIZ))


# ── flujo ──────────────────────────────────────────────────────────────────
def flujo():
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from preprocessing.preprocess import run_preprocess

    os.chdir(CODIGO_DIR)
    d_out = os.path.join(RAIZ, "flujo"); os.makedirs(d_out, exist_ok=True)
    base = dict(_config_z(), exp_dir=d_out)
    Q, S = float(base["Q_std"]), float(base["S_s_const"])
    # Malla del caso Z congelada una vez: los tres modos y los dos dt la comparten
    pkl = os.path.join(d_out, "malla.pkl")
    if not os.path.exists(pkl):
        RUNTIME.params = Parameters(config=dict(base, mesh_pkl=None))
        d = run_preprocess()
        pickle.dump({k: d[k] for k in ("nodes", "groups", "normals")}
                    | {"simplices": None, "pozo_cor": np.asarray(RUNTIME.params.pozo, float)}, open(pkl, "wb"))
    fases = [(300.0, DIA), (18000.0, 90 * DIA)]
    res = {}
    for clave, modo in MODOS.items():
        t, H, ts, Hs = 0.0, None, [0.0], []
        for dt, t_fin in fases:
            RUNTIME.params = Parameters(config=dict(base, dt=dt, bc_outlet_mode=modo, mesh_pkl=pkl))
            d = run_preprocess(); n, g = d["nodes"], d["groups"]
            dom = np.hstack([g["interior"], g["boundary:inlet"], g["boundary:outlet"], g["boundary:wall"]]).astype(int)
            fan = np.setdiff1d(np.arange(len(n)), dom)
            lu = spl.splu(d["A_left"].tocsc())
            b = np.zeros(len(n)); b[dom] = -Q * d["delta_p"][dom]
            if H is None:
                H = np.full(len(n), float(base["h_init_const"])); Hs.append(H.copy())
            while t < t_fin - 1e-6:
                r = d["A_right"] @ H; r[fan] = 0         # como H_vector: filas fantasma = dato (0)
                H = lu.solve(r + b); t += dt
                ts.append(t); Hs.append(H.copy())
        res[clave] = dict(t=np.array(ts), H=np.array(Hs), nodes=n, dom=dom, borde=g["boundary:outlet"])
        np.savez_compressed(os.path.join(d_out, f"{modo}.npz"), **res[clave])
        print(f"[flujo] {modo}: borde a 90 d [{H[g['boundary:outlet']].min():.4g}, "
              f"{H[g['boundary:outlet']].max():.4g}] m, h mín {H[dom].min():.4g} m")

    # Fotogramas: 30 instantes logarítmicos el primer día y 45 lineales hasta 90 d
    t = res["continuo"]["t"]
    objetivo = np.r_[0, np.logspace(np.log10(300), np.log10(DIA), 30), np.linspace(1.5 * DIA, 90 * DIA, 45)]
    idx = sorted(set(int(np.argmin(np.abs(t - x))) for x in objetivo))
    c0 = np.array([float(base["lado_cuadrado"]) / 2, -float(base["lado_cuadrado"]) / 2])
    out = {"t": [round(float(t[k]) / DIA, 4) for k in idx]}
    for clave, c in res.items():
        n, H, dom = c["nodes"], c["H"], c["dom"]
        rej = Rejilla(n, dom, float(base["lado_cuadrado"]))
        area = _areas(n, dom, rej.tri)
        img, rango = [], []
        for k in idx:
            h = H[k][dom]
            lv = (np.linspace(268, 300.001, 33) if clave == "continuo"
                  else np.linspace(h.min(), h.max() + max(1e-6, 1e-9 * abs(h.max())), 33))
            img.append(rej.carga(h, lv)); rango.append([round(float(h.min()), 2), round(float(h.max()), 2)])
        # Agua liberada por el almacenamiento / Q (el resto entra por el borde)
        alm = np.r_[np.nan, S * ((H[:-1, dom] - H[1:, dom]) @ area) / np.diff(t)] / Q
        rr = np.linalg.norm(n[dom] - c0, axis=1)
        out[clave] = {"img": img, "rango": rango,
                      "borde": [round(float(H[k, c["borde"]].mean()), 3) for k in idx],
                      "alm": [None if not np.isfinite(alm[k]) else round(float(alm[k]), 5) for k in idx],
                      "pozo": [round(float(base["h_init_const"] - H[k][dom][np.argmin(rr)]), 3) for k in idx]}
        if clave == "continuo":
            sub = np.random.default_rng(0).choice(len(dom), min(500, len(dom)), replace=False)
            out["r"] = [round(float(v), 2) for v in rr[sub]]
            out[clave]["s"] = [[round(float(base["h_init_const"] - H[k][dom][i]), 2) for i in sub] for k in idx]
    _pagina("sumidero_flujo.html", out, os.path.join(RAIZ, "flujo.html"))


# ── transporte ─────────────────────────────────────────────────────────────
def transporte():
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from funtions.utils import gaussian_2d

    os.chdir(CODIGO_DIR)
    def cargar(caso):
        d = os.path.join(RAIZ, caso, "bfr")
        return np.load(os.path.join(d, "simulation_results.npz")), json.load(open(os.path.join(d, "config.json")))
    d1, cfg = cargar("Z1_con_sumidero"); d0, _ = cargar("Z0_sin_sumidero")
    RUNTIME.params = Parameters(config=cfg); p = RUNTIME.params
    n, dt, C0, L = d0["nodes"], float(d0["dt"]), float(cfg["C_init_const"]), float(cfg["lado_cuadrado"])
    if not (np.allclose(d0["nodes"], d1["nodes"]) and np.abs(d1["H"] - d0["H"]).max() == 0):
        raise RuntimeError("Z0 y Z1 no comparten malla y campo de carga")
    dom = np.nonzero((n[:, 0] >= -1e-9) & (n[:, 0] <= L + 1e-9) & (n[:, 1] >= -L - 1e-9) & (n[:, 1] <= 1e-9))[0]
    rej = Rejilla(n, dom, L)
    w = _areas(n, dom, rej.tri)
    g = gaussian_2d(n[dom], np.asarray(p.pozo, float), 1, p.epsilon_pozo)
    nt = d0["C"].shape[0]
    t = np.arange(1, nt + 1) * dt / DIA
    pasos = [k for k in sorted(set([0] + [int(round(j * DIA / dt)) - 1 for j in range(1, int(t[-1]) + 1)])) if 0 <= k < nt]
    H = d0["H"][:, 0]
    out = {"t": [round(float(t[k]), 3) for k in pasos], "img_h": [], "img_c1": [], "img_c0": []}
    for k in pasos:
        out["img_h"].append(rej.carga(H[k][dom], np.linspace(268, 300.001, 33)))
        for clave, dd in (("img_c1", d1), ("img_c0", d0)):
            fig, ax = rej.lienzo()
            ax.contourf(rej.GX, rej.GZ, rej.campo(dd["C"][k, 0][dom] / C0), levels=np.linspace(0, 1, 41),
                        cmap="viridis", extend="both")
            ax.plot(L / 2, -L / 2, "o", ms=4, mfc="none", mec="w", mew=1)
            out[clave].append(_png(fig))
    for clave, dd in (("c1", d1), ("c0", d0)):
        C = dd["C"][:, 0][:, dom]
        masa = C @ w
        out["masa_" + clave] = [round(float(v), 5) for v in masa / masa[0]]
        out["cap_" + clave] = [round(float(v), 5) for v in (C @ (g * w)) / C0]
    rr = np.linalg.norm(n[dom] - [L / 2, -L / 2], axis=1)
    out["h_pozo"] = [round(float(v), 3) for v in H[:, dom][:, np.argmin(rr)]]
    out["dt_dias"] = dt / DIA
    print(f"[transporte] masa final con W- {out['masa_c1'][-1]:.4f}, sin W- {out['masa_c0'][-1]:.4f}")
    _pagina("sumidero_transporte.html", out, os.path.join(RAIZ, "transporte.html"))


if __name__ == "__main__":
    que = sys.argv[1:] or ["flujo", "transporte"]
    for q in que:
        {"flujo": flujo, "transporte": transporte}[q]()
