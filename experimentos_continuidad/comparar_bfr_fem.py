"""Comparación BFR vs FEniCS de una misma corrida (mismo ID) sobre la malla compartida.

    python experimentos_continuidad/comparar_bfr_fem.py ID [zoom_m]

Salida en graficas/ y tablas/: video completo y de zoom (BFR | FEM | BFR−FEM, paletas
de view.plot), curvas temporales y metricas_<ID>.json. Normas y masa con área nodal
(1/3 del área de los triángulos incidentes): la malla es mucho más fina en pozo y fuente.
"""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm

AQUI = os.path.dirname(os.path.abspath(__file__))
CODIGO = os.path.dirname(AQUI)
sys.path.insert(0, CODIGO); sys.path.insert(0, AQUI)
from view.plot import get_colormaps
from animar_comparacion import cargar


def area_nodal(nodes, tris):
    P = nodes[tris]
    a = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                     - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    w = np.zeros(len(nodes))
    for k in range(3):
        np.add.at(w, tris[:, k], a / 3.0)
    return w


def main(ident, zoom=600.0):
    m = pickle.load(open(os.path.join(CODIGO, "data/input/malla_compartida/malla.pkl"), "rb"))
    nodes = np.asarray(m["nodes"]); tris = np.asarray(m["triangles"]); pz = np.asarray(m["pozo_cor"])
    x, z = nodes.T
    tri = mtri.Triangulation(x, z, tris)
    w = area_nodal(nodes, tris)
    B, dt = cargar(ident, "bfr", nodes); F, _ = cargar(ident, "fenicsx", nodes)
    dom = w > 0                                  # nodos que están en algún triángulo
    kp = int(np.argmin(np.linalg.norm(nodes - pz, axis=1)))
    D = B - F
    t = (np.arange(B.shape[0]) + 1) * dt / 86400.0

    l2 = np.sqrt((w * D ** 2).sum(1)) / np.maximum(np.sqrt((w * F ** 2).sum(1)), 1e-300)
    mB, mF = (w * B).sum(1), (w * F).sum(1)
    met = dict(id=ident, formulacion="continuity", dt_s=dt,
               L2_rel_final=float(l2[-1]), L2_rel_max=float(l2.max()),
               Linf_final_sobre_Cmax_FEM=float(np.abs(D[-1]).max() / F[-1].max()),
               Cmax_final=dict(bfr=float(B[-1].max()), fem=float(F[-1].max())),
               C_pozo_final=dict(bfr=float(B[-1, kp]), fem=float(F[-1, kp])),
               masa_int_C_final=dict(bfr=float(mB[-1]), fem=float(mF[-1]), rel=float((mB[-1] - mF[-1]) / mF[-1])),
               C_min=dict(bfr=float(B.min()), fem=float(F.min())),
               cuadros=int(B.shape[0]), pozo_nodo=kp)
    os.makedirs(os.path.join(AQUI, "tablas"), exist_ok=True)
    json.dump(met, open(os.path.join(AQUI, "tablas", f"metricas_bfr_vs_fem_{ident}.json"), "w"), indent=2)

    # ---- curvas
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), constrained_layout=True)
    for a, (nm, fb, ff) in zip(ax, [("C máx", B.max(1), F.max(1)), ("C en el pozo", B[:, kp], F[:, kp]),
                                     ("∫C dΩ (área nodal)", mB, mF)]):
        a.plot(t, fb, "r", label="BFR"); a.plot(t, ff, "b", label="FEM"); a.set_title(nm); a.set_xlabel("día")
    ax[0].legend(); ax[3].plot(t, 100 * l2, "k"); ax[3].set_title("‖BFR−FEM‖₂/‖FEM‖₂ [%]"); ax[3].set_xlabel("día")
    fig.suptitle(f"BFR vs FEM — {ident} (transport_form=continuity)")
    fig.savefig(os.path.join(AQUI, "graficas", f"curvas_bfr_vs_fem_{ident}.png"), dpi=120); plt.close(fig)

    # ---- videos
    _, cmap_c = get_colormaps()
    cmax = max(float(B.max()), float(F.max()), 1e-30)
    lim = max(float(np.percentile(np.abs(D), 99.5)), cmax * 1e-3)
    nc, nd = Normalize(0.0, cmax), TwoSlopeNorm(0.0, -lim, lim)
    for suf, zm in [("completo", None), ("zoom_pozo", zoom)]:
        fig, ax = (plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True) if zm
                   else plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True))
        pan = [("BFR (RBF-FD)  C [kg/m³]", B, nc, cmap_c), ("FEniCS (FEM)  C [kg/m³]", F, nc, cmap_c),
               ("BFR − FEM", D, nd, "RdBu_r")]
        arts = []
        for a, (tt, A, nm, cm) in zip(ax, pan):
            a.set_aspect("equal"); a.set_title(tt, fontsize=10); a.plot(*pz, "k+", ms=10)
            if zm:
                a.set_xlim(pz[0] - zm, pz[0] + zm); a.set_ylim(pz[1] - zm, pz[1] + zm)
            im = a.tripcolor(tri, A[0], norm=nm, cmap=cm, shading="gouraud", rasterized=True)
            fig.colorbar(im, ax=a, fraction=0.03 if zm else 0.02); arts.append((im, A))
        ttl = fig.suptitle("")
        wr = FFMpegWriter(fps=12)
        with wr.saving(fig, os.path.join(AQUI, "graficas", f"BFR_vs_FEM_{ident}_{suf}.mp4"), 110):
            for k in range(B.shape[0]):
                for im, A in arts:
                    im.set_array(A[k])
                ttl.set_text(f"{ident}  —  día {t[k]:.1f}")
                wr.grab_frame()
        plt.close(fig)
    return met


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 600.0), indent=2))
