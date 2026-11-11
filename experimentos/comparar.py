"""
Comparación BFR (RBF-FD) vs FEniCS (FEM) en UNA sola pasada: las métricas y
la animación salen de los mismos arreglos en memoria, cuadro a cuadro.

No hay figuras de postproceso aparte: cada cuadro del video muestra los
campos de ambos backends, su diferencia y las curvas de comparación con un
cursor en el instante del cuadro; el último cuadro se guarda como PNG y las
mismas series se escriben en metricas.json (que lee informe.py).

Sesgos que se evitan
--------------------
* Malla: el resultado BFR debe tener exactamente los nodos de la malla
  congelada y el FEM exactamente sus nodos sin fantasmas; si no, se rechaza
  (resultados de otra malla mezclados en silencio).
* Config: las config.json de ambos backends deben coincidir salvo `metodo`,
  `exp_dir` y los cambios propios de cada backend declarados en el catálogo.
* Densidad de nodos: normas y masa con pesos de área nodal (P1 concentrada)
  sobre la triangulación que usa FEniCS, no sumas nodales (la malla es
  15× más fina cerca del pozo y la fuente).
* H: error en metros (RMS y máx.), no relativo a ‖H‖ (≈ 200 m de nivel
  base hacen que el relativo sea ~1e-4 aunque el cono difiera).
* Pozo: concentración extraída = media de C con la gaussiana del sumidero
  (ε = r_pozo/3, la misma de ambos backends), no el valor de un nodo.
* Dibujo: tripcolor Gouraud sobre la misma triangulación = el campo P1
  exacto en ambos backends, sin remuestreo a una rejilla.

    python -m experimentos.comparar --caso M_modelo_nuevo/M0_adr
    python -m experimentos.comparar --grupo M_modelo_nuevo D_caudal_pozo
    python -m experimentos.comparar --todos [--jobs 4] [--sin-video]

Salidas: <caso>/comparacion/{metricas.json, comparacion_{H,C,V}.{mp4,webm},
                             comparacion_{H,C,V}_final.png}
         data/output/experimentos/resumen.csv (una fila por caso)
"""
import argparse
import csv
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from config.geometry import build_geometry  # noqa: E402
from experimentos.catalogo import GRUPOS, iter_casos  # noqa: E402

OUT_ROOT = os.path.join(CODIGO_DIR, "data", "output", "experimentos")
MALLA_PKL = os.path.join(CODIGO_DIR, "data", "input", "malla_compartida", "malla.pkl")
TOL_NODO = 1e-6          # [m] emparejamiento BFR ↔ vértices FEM
MAX_CUADROS = 144        # el video se submuestrea si Nt es mayor (las métricas no)
IGNORAR_CFG = {"metodo", "exp_dir", "keep_pre"}


class ErrorCaso(Exception):
    """El caso no se puede comparar sin sesgo (malla/config/datos)."""


# ── Malla común ──────────────────────────────────────────────────────────────

def malla_comun(ruta=MALLA_PKL):
    """Nodos sin fantasmas, triángulos (los mismos que usa FEniCS, ver
    fenicsx.mesh.build_mesh_from_bfr_pkl), pesos de área nodal y nodo del pozo.
    `ruta`: pkl de la malla (por defecto la compartida; los casos con otra `mesh_pkl` pasan la suya)."""
    with open(ruta, "rb") as f:
        m = pickle.load(f)
    todos, groups = np.asarray(m["nodes"], float), m["groups"]
    fant = np.zeros(len(todos), bool)
    for k, idx in groups.items():
        if k.startswith("ghost"):
            fant[np.asarray(idx, int)] = True
    keep = np.nonzero(~fant)[0]
    old2new = np.full(len(todos), -1)
    old2new[keep] = np.arange(len(keep))
    nodes = todos[keep]

    if "triangles" in m:
        # Malla saneada (preprocessing/malla_compartida.py): triángulos guardados.
        tris = old2new[np.asarray(m["triangles"], np.int64)]
    else:
        import triangle as tr
        bnd = np.concatenate([groups[k] for k in groups if k.startswith("boundary")])
        pts = todos[bnd]
        c0 = pts.mean(axis=0)
        orden = old2new[bnd[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]]
        segs = np.column_stack([orden, np.roll(orden, -1)]).astype(np.int32)
        cdt = tr.triangulate({"vertices": nodes, "segments": segs}, "pq20S0")
        if len(cdt["vertices"]) != len(nodes):
            raise ErrorCaso("la CDT añadió nodos Steiner: no coincide con la malla FEM")
        tris = cdt["triangles"].astype(np.int64)
    if (tris < 0).any():
        raise ErrorCaso("hay triángulos con nodos fantasma")

    p = nodes[tris]
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    w = np.bincount(tris.ravel(), weights=np.repeat(area / 3, 3), minlength=len(nodes))
    return dict(todos=todos, keep=keep, nodes=nodes, tris=tris, w=w,
                area=float(area.sum()), pozo=np.asarray(m["pozo_cor"], float),
                n_fantasmas=int(fant.sum()))


# ── Carga y validación ───────────────────────────────────────────────────────

def _leer(path):
    z = np.load(path, allow_pickle=True)
    U = np.asarray(z["U"], float)
    if U.ndim == 4:                       # BFR: (Nt, K, N, 2)
        U = U[:, 0]
    return dict(nodes=np.asarray(z["nodes"], float), H=np.asarray(z["H"], float)[:, 0],
                C=np.asarray(z["C"], float)[:, 0], U=U, dt=float(z["dt"]))


def _cfg(dir_):
    ruta = os.path.join(dir_, "config.json")
    if not os.path.exists(ruta):    # diseño de main: la config del lanzador está en data/output/_cfg/
        partes = os.path.normpath(dir_).split(os.sep)
        ruta = os.path.join(CODIGO_DIR, "data", "output", "_cfg", f"{partes[-5]}_{partes[-2]}_{partes[-1]}.json")
    with open(ruta) as f:
        return json.load(f)


def cargar_caso(g, c, spec, M, dirs=None):
    """Campos de ambos backends sobre los nodos comunes (orden de M['nodes']).
    dirs=(d_bfr, d_fem) lee las salidas con el diseño de main (data/output/<backend>/data/real/<modelo>/<run_type>)."""
    ref = spec.get("ref_fem", f"{g}/{c}")
    d_bfr = dirs[0] if dirs else os.path.join(OUT_ROOT, g, c, "bfr")
    d_fem = dirs[1] if dirs else os.path.join(OUT_ROOT, ref, "fenicsx")
    for d in (d_bfr, d_fem):
        if not os.path.exists(os.path.join(d, "simulation_results.npz")):
            raise ErrorCaso(f"falta {os.path.relpath(d, CODIGO_DIR)}/simulation_results.npz")

    # Config: idénticas salvo lo declarado como propio de cada backend.
    cb, cf = _cfg(d_bfr), _cfg(d_fem)
    propias = IGNORAR_CFG | set(spec.get("bfr", {})) | set(spec.get("fenicsx", {}))
    dif = sorted(k for k in set(cb) | set(cf) if k not in propias and cb.get(k) != cf.get(k))
    if dif:
        raise ErrorCaso("config BFR ≠ FEM en " + ", ".join(f"{k}: {cb.get(k)} vs {cf.get(k)}" for k in dif))

    bfr, fem = _leer(os.path.join(d_bfr, "simulation_results.npz")), _leer(os.path.join(d_fem, "simulation_results.npz"))
    if bfr["nodes"].shape != M["todos"].shape or not np.allclose(bfr["nodes"], M["todos"], atol=TOL_NODO):
        raise ErrorCaso(f"BFR con {len(bfr['nodes'])} nodos ≠ malla del caso ({len(M['todos'])}): resultado de otra malla")
    dist, idx = cKDTree(fem["nodes"]).query(M["nodes"])
    if len(fem["nodes"]) != len(M["nodes"]) or dist.max() > TOL_NODO or len(set(idx)) != len(idx):
        raise ErrorCaso(f"FEM con {len(fem['nodes'])} nodos no coincide con los {len(M['nodes'])} de la malla")
    if bfr["H"].shape[0] != fem["H"].shape[0] or bfr["dt"] != fem["dt"]:
        raise ErrorCaso("Nt/dt distintos entre backends")

    datos = {}
    for b, d, sel in (("bfr", bfr, M["keep"]), ("fenicsx", fem, idx)):
        U = d["U"][:, sel]
        datos[b] = dict(H=d["H"][:, sel], C=d["C"][:, sel], U=U, V=np.linalg.norm(U, axis=-1))
        for campo in ("H", "C", "V"):
            if not np.isfinite(datos[b][campo]).all():
                raise ErrorCaso(f"{b}: valores no finitos en {campo}")
    return datos, bfr["dt"], cb, ref


# ── Métricas (las mismas que dibuja el video) ────────────────────────────────

def metricas(datos, dt, cfg, M):
    w, W = M["w"], M["w"].sum()
    b, f = datos["bfr"], datos["fenicsx"]
    Nt = b["H"].shape[0]
    t = (np.arange(Nt) + 1) * dt / 86400       # ambos guardan tras cada paso

    def l2(x):
        return np.sqrt((w * x ** 2).sum(axis=1))

    s = {"t_dias": t}
    dH = b["H"] - f["H"]
    s["H_rms_m"] = l2(dH) / np.sqrt(W)
    s["H_max_m"] = np.abs(dH).max(axis=1)
    for campo in ("C", "V"):
        d = b[campo] - f[campo]
        s[f"{campo}_L2_rel"] = l2(d) / np.maximum(l2(f[campo]), 1e-300)
        s[f"{campo}_Linf_rel"] = np.abs(d).max(axis=1) / np.maximum(np.abs(f[campo]).max(axis=1), 1e-300)

    # Gaussiana del sumidero del pozo (misma ε en ambos backends con r_pozo).
    eps = float(cfg["r_pozo"]) / 3 if cfg.get("r_pozo") else 60.0
    r2 = ((M["nodes"] - M["pozo"]) ** 2).sum(axis=1)
    gw = w * np.exp(-r2 / (2 * eps ** 2))
    gw /= gw.sum()
    for bk, d in (("bfr", b), ("fem", f)):
        s[f"masa_{bk}"] = d["C"] @ w                      # ∫C dΩ  [kg/m³·m²]
        s[f"Cpozo_{bk}"] = d["C"] @ gw                    # C extraída
        s[f"Hpozo_{bk}"] = d["H"] @ gw                    # nivel en el pozo
        s[f"Cmax_{bk}"] = d["C"].max(axis=1)
        s[f"Vmedia_{bk}"] = d["V"] @ w / W                # |U| media en Ω

    rango_H = float(max(np.ptp(f["H"][-1]), 1e-30))
    r = dict(
        Nt=Nt, dt=dt, n_nodos=len(w), area_m2=float(W), eps_pozo_m=eps,
        H_rms_final_m=float(s["H_rms_m"][-1]), H_max_final_m=float(s["H_max_m"][-1]),
        H_rms_sobre_rango=float(s["H_rms_m"][-1] / rango_H), H_rango_fem_m=rango_H,
        C_L2_rel_final=float(s["C_L2_rel"][-1]), C_L2_rel_max=float(s["C_L2_rel"].max()),
        C_Linf_rel_final=float(s["C_Linf_rel"][-1]),
        V_L2_rel_final=float(s["V_L2_rel"][-1]), V_L2_rel_max=float(s["V_L2_rel"].max()),
        masa_bfr_final=float(s["masa_bfr"][-1]), masa_fem_final=float(s["masa_fem"][-1]),
        masa_ratio_final=float(s["masa_bfr"][-1] / max(s["masa_fem"][-1], 1e-300)),
        Cpozo_bfr_final=float(s["Cpozo_bfr"][-1]), Cpozo_fem_final=float(s["Cpozo_fem"][-1]),
        Hpozo_bfr_final=float(s["Hpozo_bfr"][-1]), Hpozo_fem_final=float(s["Hpozo_fem"][-1]),
        Cmin_bfr=float(b["C"].min()), Cmin_fem=float(f["C"].min()),
    )
    return s, r


# ── Video: un video por campo (BFR / FEM / BFR−FEM + curvas con cursor) ─────

COLOR_BFR = "#dc2626"     # rojo: BFR, como en la presentación (avance1-doc)
COLOR_FEM = "#2563eb"     # azul: FEniCS
COLOR_ERR = "#374151"     # gris: errores BFR−FEM
CAMPOS_VIDEO = ("H", "C", "V")
QNX, QNZ = 44, 14         # rejilla de flechas (como scripts/animate_forward.py)


def _bary(tri, pts):
    """Matriz (len(pts) × N) de interpolación P1 sobre la triangulación común
    (el mismo campo que dibuja tripcolor, sin remuestreo de Delaunay)."""
    from scipy import sparse
    t = tri.get_trifinder()(pts[:, 0], pts[:, 1])
    ok = t >= 0
    pts, t = pts[ok], t[ok]
    v = tri.triangles[t]
    x, y = tri.x[v], tri.y[v]
    det = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    l1 = ((y[:, 1] - y[:, 2]) * (pts[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (pts[:, 1] - y[:, 2])) / det
    l2 = ((y[:, 2] - y[:, 0]) * (pts[:, 0] - x[:, 2]) + (x[:, 0] - x[:, 2]) * (pts[:, 1] - y[:, 2])) / det
    w = np.column_stack([l1, l2, 1 - l1 - l2])
    filas = np.repeat(np.arange(len(pts)), 3)
    return pts, sparse.csr_matrix((w.ravel(), (filas, v.ravel())), shape=(len(pts), len(tri.x)))


def video(datos, s, titulo, M, out_dir, fuente=None):
    """comparacion_{H,C,V}.mp4: cada uno con BFR, FEM y BFR−FEM a todo el ancho
    y dos o tres curvas con un cursor en el instante del cuadro.  Paletas de
    view.plot.get_colormaps() (las de los videos de las demás variantes):
    H y |U| con la turbo, C con la inferno; |U| lineal saturado en el pozo y
    con flechas de dirección (normalizadas) como en animate_forward.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.tri import Triangulation
    from view.plot import get_colormaps

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
    cmap_main, cmap_c = get_colormaps()
    tri = Triangulation(M["nodes"][:, 0], M["nodes"][:, 1], M["tris"])
    b, f = datos["bfr"], datos["fenicsx"]
    Nt = b["H"].shape[0]
    paso = int(np.ceil(Nt / MAX_CUADROS))
    cuadros = list(range(paso - 1, Nt, paso))
    if cuadros[-1] != Nt - 1:
        cuadros.append(Nt - 1)
    t = s["t_dias"]

    x0, x1 = M["nodes"][:, 0].min(), M["nodes"][:, 0].max()
    z0, z1 = M["nodes"][:, 1].min(), M["nodes"][:, 1].max()
    qx, qz = np.meshgrid(np.linspace(x0, x1, QNX + 2)[1:-1], np.linspace(z0, z1, QNZ + 2)[1:-1])
    q_pts, Bq = _bary(tri, np.column_stack([qx.ravel(), qz.ravel()]))

    def todo(campo):
        return np.concatenate([b[campo].ravel(), f[campo].ravel()])

    v_sat = float(np.percentile(todo("V"), 99.0))
    conf = {
        "H": dict(nombre="Carga hidráulica H", unidad="m", cmap=cmap_main,
                  norm=Normalize(float(todo("H").min()), float(todo("H").max())), ext="neither",
                  curvas=(("Error de H: BFR − FEM (m)", [("H_rms_m", "RMS en Ω", "-", COLOR_ERR),
                                                         ("H_max_m", "máximo", "--", COLOR_ERR)], "log"),
                          ("H en el pozo (m)", [("Hpozo_bfr", "BFR", "-", COLOR_BFR),
                                                ("Hpozo_fem", "FEM", "--", COLOR_FEM)], "linear"))),
        "C": dict(nombre="Concentración C", unidad="kg/m³", cmap=cmap_c,
                  norm=Normalize(0.0, max(float(todo("C").max()), 1e-30)), ext="neither",
                  curvas=(("Error relativo L2 de C", [("C_L2_rel", "‖ΔC‖/‖C_FEM‖", "-", COLOR_ERR)], "log"),
                          ("Masa ∫C dΩ", [("masa_bfr", "BFR", "-", COLOR_BFR),
                                          ("masa_fem", "FEM", "--", COLOR_FEM)], "linear"),
                          ("C extraída en el pozo", [("Cpozo_bfr", "BFR", "-", COLOR_BFR),
                                                     ("Cpozo_fem", "FEM", "--", COLOR_FEM)], "linear"))),
        "V": dict(nombre="Velocidad de Darcy |U|", unidad="m/s", cmap=cmap_main,
                  norm=Normalize(0.0, max(v_sat, 1e-30)), ext="max",
                  curvas=(("Error relativo L2 de |U|", [("V_L2_rel", "‖Δ|U|‖/‖|U|_FEM‖", "-", COLOR_ERR)], "log"),
                          ("|U| media en Ω (m/s)", [("Vmedia_bfr", "BFR", "-", COLOR_BFR),
                                                    ("Vmedia_fem", "FEM", "--", COLOR_FEM)], "linear"))),
    }

    os.makedirs(out_dir, exist_ok=True)
    for campo in CAMPOS_VIDEO:
        cf = conf[campo]
        dif = b[campo] - f[campo]
        # Diferencia saturada en el percentil 99.5: con el máximo, unos pocos
        # nodos del pozo dejaban el resto del panel en blanco.
        lim_max = max(float(np.abs(dif).max()), 1e-30)
        lim = max(float(np.percentile(np.abs(dif), 99.5)), lim_max * 1e-3)
        fig = plt.figure(figsize=(13, 14.2))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.02], left=0.01, right=0.92,
                              top=0.945, bottom=0.27, hspace=0.06, wspace=0.015)
        filas = ((b[campo], cf["cmap"], cf["norm"], "BFR (RBF-FD)", COLOR_BFR, b),
                 (f[campo], cf["cmap"], cf["norm"], "FEniCS (FEM)", COLOR_FEM, f),
                 (dif, "RdBu_r", TwoSlopeNorm(0.0, -lim, lim), "BFR − FEM", COLOR_ERR, None))
        paneles = []
        for i, (arr, cm, nm, etiqueta, color, fuente_U) in enumerate(filas):
            ax = fig.add_subplot(gs[i, 0])
            tp = ax.tripcolor(tri, arr[0], shading="gouraud", cmap=cm, norm=nm, rasterized=True)
            ax.plot(*M["pozo"], marker="v", ms=11, mec="k", mfc="w", lw=0, zorder=6)
            if fuente is not None:
                ax.plot(*fuente, marker="*", ms=15, mec="k", mfc="w", lw=0, zorder=6)
            ax.set_aspect("equal")
            ax.set_xlim(x0, x1)
            ax.set_ylim(z0, z1)
            ax.axis("off")
            ax.text(0.006, 0.97, etiqueta, transform=ax.transAxes, ha="left", va="top",
                    fontsize=18, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"))
            q = None
            if campo == "V" and fuente_U is not None:
                q = ax.quiver(q_pts[:, 0], q_pts[:, 1], *_dir(Bq, fuente_U["U"][0]), scale=60,
                              scale_units="width", width=0.0016, headwidth=4, headlength=4,
                              color="black", alpha=0.75, zorder=5)
            # Una sola barra de color para BFR y FEM (misma norma, abarca las dos filas);
            # la diferencia es otra magnitud y lleva la suya.
            if i == 1:
                paneles.append((tp, arr, q, fuente_U))
                continue
            cax = fig.add_subplot(gs[0:2, 1] if i == 0 else gs[2, 1])
            cb = fig.colorbar(tp, cax=cax, extend=cf["ext"] if i < 2 else
                              ("both" if lim < lim_max else "neither"))
            cb.set_label(("Δ" if i == 2 else "") + f"{campo if campo != 'V' else '|U|'} ({cf['unidad']})")
            if campo == "C" or campo == "V" or i == 2:
                cb.formatter.set_powerlimits((-2, 3))
            paneles.append((tp, arr, q, fuente_U))

        ga = fig.add_gridspec(1, len(cf["curvas"]), left=0.085, right=0.985, top=0.225,
                              bottom=0.045, wspace=0.3)
        cursores, puntos = [], []
        for j, (tit, series, escala) in enumerate(cf["curvas"]):
            ax = fig.add_subplot(ga[0, j])
            for clave, lab, ls, color in series:
                y = np.asarray(s[clave])
                ax.plot(t, y, ls, lw=2.4, color=color, label=lab)
                pt, = ax.plot(t[0], y[0], "o", ms=8, color=color)
                puntos.append((pt, y))
            ys = np.concatenate([np.asarray(s[k]) for k, *_ in series])
            if escala == "log" and (ys > 0).all() and ys.max() > 10 * ys.min():
                ax.set_yscale("log")
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            cursores.append(ax.axvline(t[0], color="k", lw=1.2, alpha=0.6))
            ax.set_title(tit)
            ax.set_xlabel("t (días)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", framealpha=0.85)
        sup = fig.suptitle("", fontsize=21, fontweight="bold")

        writer = FFMpegWriter(fps=12, codec="h264",
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"])
        with writer.saving(fig, os.path.join(out_dir, f"comparacion_{campo}.mp4"), dpi=90):
            for k in cuadros:
                for tp, arr, q, fuente_U in paneles:
                    tp.set_array(arr[k])
                    if q is not None:
                        q.set_UVC(*_dir(Bq, fuente_U["U"][k]))
                for cur in cursores:
                    cur.set_xdata([t[k], t[k]])
                for pt, y in puntos:
                    pt.set_data([t[k]], [y[k]])
                texto = f"{cf['nombre']}  ·  {titulo}  ·  t = {t[k]:.1f} días"
                sup.set_text(texto)
                sup.set_fontsize(min(21, 21 * 62 / len(texto)))     # títulos largos (B, F)
                writer.grab_frame()
        # El PNG es el último cuadro del video, no una figura aparte.
        fig.savefig(os.path.join(out_dir, f"comparacion_{campo}_final.png"), dpi=60)
        plt.close(fig)
        _webm(os.path.join(out_dir, f"comparacion_{campo}.mp4"))
    return [float(t[k]) for k in cuadros]


def _webm(mp4):
    """Copia VP9 del mismo video: los Chromium sin códecs propietarios (p. ej.
    los de Playwright) no reproducen H.264; el informe ofrece ambos."""
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "36", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
                    mp4[:-4] + ".webm"], check=True)


def _dir(B, U):
    """Dirección de U (vector unitario) en los puntos de las flechas."""
    ux, uz = B @ U[:, 0], B @ U[:, 1]
    mag = np.maximum(np.hypot(ux, uz), 1e-30)
    return ux / mag, uz / mag


# ── Orquestación ─────────────────────────────────────────────────────────────

_MALLAS = {}


def comparar_caso(g, c, spec, con_video=True, dirs=None, out=None):
    ruta = os.path.join(CODIGO_DIR, spec.get("cfg", {}).get("mesh_pkl", os.path.relpath(MALLA_PKL, CODIGO_DIR)))
    if ruta not in _MALLAS:
        _MALLAS[ruta] = malla_comun(ruta)
    _M = _MALLAS[ruta]
    out = out or os.path.join(OUT_ROOT, g, c, "comparacion")
    datos, dt, cfg, ref = cargar_caso(g, c, spec, _M, dirs)
    s, r = metricas(datos, dt, cfg, _M)
    titulo = c if ref == f"{g}/{c}" else f"{c}  (FEM: {ref.split('/')[-1]})"
    if con_video:
        t_cuadros = video(datos, s, titulo, _M, out,
                          fuente=build_geometry(cfg["domain"])["fnte"] if cfg.get("activate_fuente") else None)
        r["cuadros_video"] = len(t_cuadros)
        s["t_cuadros"] = t_cuadros          # t de cada cuadro: el informe sincroniza el cursor
    r = {"grupo": g, "caso": c, "ref_fem": ref, "desc": spec["desc"], **r}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metricas.json"), "w") as fh:
        json.dump({"resumen": r, "series": {k: np.asarray(v).tolist() for k, v in s.items()},
                   "malla": {"area_m2": _M["area"], "n_nodos": len(_M["nodes"]),
                             "n_fantasmas": _M["n_fantasmas"]}}, fh)
    # metrics.json/series.png del comparador anterior quedarían desfasados.
    for viejo in ("metrics.json", "series.png", "grid.mp4", "grid_final.png",
                  "comparacion.mp4", "comparacion_final.png"):
        if os.path.exists(os.path.join(out, viejo)):
            os.remove(os.path.join(out, viejo))
    return r


def _trabajo(args):
    g, c, con_video = args
    try:
        r = comparar_caso(g, c, GRUPOS[g]["casos"][c], con_video)
        print(f"[comparar] ✓  {g}/{c}:  ΔH_rms={r['H_rms_final_m']:.2e} m  "
              f"C_L2={r['C_L2_rel_final']:.2e}  |U|_L2={r['V_L2_rel_final']:.2e}  "
              f"masa B/F={r['masa_ratio_final']:.3f}", flush=True)
        return r
    except ErrorCaso as e:
        print(f"[comparar] ✗  {g}/{c}: {e}", flush=True)
        return None


def _resumen_csv(filas):
    path = os.path.join(OUT_ROOT, "resumen.csv")
    previas = {}
    if os.path.exists(path):
        with open(path) as fh:
            previas = {(r["grupo"], r["caso"]): r for r in csv.DictReader(fh)}
    for r in filas:
        previas[(r["grupo"], r["caso"])] = r
    if not previas:
        return
    campos = list(dict.fromkeys(k for r in previas.values() for k in r))
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        for k in sorted(previas):
            wr.writerow(previas[k])
    print(f"[comparar] resumen → {os.path.relpath(path, CODIGO_DIR)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grupo", nargs="+")
    ap.add_argument("--caso", nargs="+")
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--sin-video", action="store_true", help="sólo métricas")
    ap.add_argument("--jobs", type=int, default=3, help="casos en paralelo")
    ap.add_argument("--smoke", action="store_true", help="leer de data/output/experimentos_smoke/")
    args = ap.parse_args()
    if args.smoke:
        global OUT_ROOT
        OUT_ROOT += "_smoke"
    if not (args.grupo or args.caso or args.todos):
        ap.error("indica --grupo, --caso o --todos")

    tareas = [(g, c, not args.sin_video) for g, c, _ in iter_casos(args.grupo, args.caso)
              if GRUPOS[g].get("comparar", True)]
    with Pool(min(args.jobs, len(tareas)) or 1) as pool:
        filas = pool.map(_trabajo, tareas, chunksize=1)
    _resumen_csv([r for r in filas if r])
    if not all(filas):
        sys.exit(1)


if __name__ == "__main__":
    main()
