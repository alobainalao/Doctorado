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
  sobre la triangulación CDT que usa FEniCS, no sumas nodales (la malla es
  15× más fina cerca del pozo y la fuente).
* H: error en metros (RMS y máx.), no relativo a ‖H‖ (≈ 200 m de nivel
  base hacen que el relativo sea ~1e-4 aunque el cono difiera).
* Pozo: concentración extraída = media de C con la gaussiana del sumidero
  (ε = r_pozo/3, la misma de ambos backends), no el valor de un nodo.
* Dibujo: tripcolor Gouraud sobre la misma triangulación = el campo P1
  exacto en ambos backends, sin remuestreo a una rejilla.

    python -m experimentos.comparar --caso A_terminos/A3_fuente_pozo
    python -m experimentos.comparar --grupo A_terminos D_caudal_pozo
    python -m experimentos.comparar --todos [--jobs 4] [--sin-video]

Salidas: <caso>/comparacion/{metricas.json, comparacion.mp4, comparacion_final.png}
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

from experimentos.catalogo import GRUPOS, iter_casos  # noqa: E402

OUT_ROOT = os.path.join(CODIGO_DIR, "data", "output", "experimentos")
MALLA_PKL = os.path.join(CODIGO_DIR, "data", "input", "malla_compartida", "malla.pkl")
TOL_NODO = 1e-6          # [m] emparejamiento BFR ↔ vértices FEM
MAX_CUADROS = 144        # el video se submuestrea si Nt es mayor (las métricas no)
IGNORAR_CFG = {"metodo", "exp_dir", "keep_pre"}


class ErrorCaso(Exception):
    """El caso no se puede comparar sin sesgo (malla/config/datos)."""


# ── Malla común ──────────────────────────────────────────────────────────────

def malla_comun():
    """Nodos sin fantasmas, triángulos CDT (los mismos que construye
    fenicsx.mesh.build_mesh_from_bfr_pkl), pesos de área nodal y nodo del pozo."""
    import triangle as tr
    with open(MALLA_PKL, "rb") as f:
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

    bnd = np.concatenate([groups[k] for k in groups if k.startswith("boundary")])
    pts = todos[bnd]
    c0 = pts.mean(axis=0)
    orden = old2new[bnd[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]]
    segs = np.column_stack([orden, np.roll(orden, -1)]).astype(np.int32)
    cdt = tr.triangulate({"vertices": nodes, "segments": segs}, "pq20S0")
    if len(cdt["vertices"]) != len(nodes):
        raise ErrorCaso("la CDT añadió nodos Steiner: no coincide con la malla FEM")
    tris = cdt["triangles"].astype(np.int64)

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
    with open(os.path.join(dir_, "config.json")) as f:
        return json.load(f)


def cargar_caso(g, c, spec, M):
    """Campos de ambos backends sobre los nodos comunes (orden de M['nodes'])."""
    ref = spec.get("ref_fem", f"{g}/{c}")
    d_bfr = os.path.join(OUT_ROOT, g, c, "bfr")
    d_fem = os.path.join(OUT_ROOT, ref, "fenicsx")
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
        raise ErrorCaso(f"BFR con {len(bfr['nodes'])} nodos ≠ malla congelada ({len(M['todos'])}): resultado de otra malla")
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


# ── Video: campos + diferencia + curvas con cursor, cuadro a cuadro ─────────

def video(datos, s, titulo, M, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    from matplotlib.tri import Triangulation
    from view.plot import get_colormaps

    cmap_main, cmap_c = get_colormaps()
    tri = Triangulation(M["nodes"][:, 0], M["nodes"][:, 1], M["tris"])
    b, f = datos["bfr"], datos["fenicsx"]
    Nt = b["H"].shape[0]
    paso = int(np.ceil(Nt / MAX_CUADROS))
    cuadros = list(range(paso - 1, Nt, paso))
    if cuadros[-1] != Nt - 1:
        cuadros.append(Nt - 1)

    def todo(campo):
        return np.concatenate([b[campo].ravel(), f[campo].ravel()])

    vmax_v = max(float(todo("V").max()), 1e-30)
    vmin_v = min(max(float(np.percentile(todo("V"), 1)), 1e-30), vmax_v / 10)
    filas = (
        ("H", "H (m)", cmap_main, Normalize(float(todo("H").min()), float(todo("H").max()))),
        ("C", "C (kg/m³)", cmap_c, Normalize(0.0, max(float(todo("C").max()), 1e-30))),
        ("V", "|U| (m/s)", cmap_main, LogNorm(vmin_v, vmax_v)),
    )

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.05], width_ratios=[1, 1, 1, 0.001],
                          left=0.035, right=0.965, top=0.93, bottom=0.06, hspace=0.18, wspace=0.12)
    paneles = []
    for i, (campo, etiqueta, cmap, norm) in enumerate(filas):
        dif = b[campo] - f[campo]
        lim = max(float(np.abs(dif).max()), 1e-30)
        cols = ((b[campo], cmap, norm, "BFR (RBF-FD)"),
                (f[campo], cmap, norm, "FEniCS (FEM)"),
                (dif, "RdBu_r", TwoSlopeNorm(0.0, -lim, lim), "BFR − FEM"))
        for j, (arr, cm, nm, col) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            vals = np.maximum(arr[0], vmin_v) if (campo == "V" and j < 2) else arr[0]
            tp = ax.tripcolor(tri, vals, shading="gouraud", cmap=cm, norm=nm, rasterized=True)
            ax.plot(*M["pozo"], marker="v", ms=5, mec="k", mfc="w", lw=0)
            ax.set_aspect("equal")
            ax.axis("off")
            if i == 0:
                ax.set_title(col, fontsize=13, fontweight="bold")
            if j == 0:
                ax.text(-0.02, 0.5, etiqueta, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=12, fontweight="bold")
            cax = ax.inset_axes([0.60, 0.04, 0.36, 0.05])
            cb = fig.colorbar(tp, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=7)
            paneles.append((tp, arr, campo == "V" and j < 2))

    t = s["t_dias"]
    ga = gs[3, :3].subgridspec(1, 4, wspace=0.32)
    curvas = (
        ("Error de H (m)", [("H_rms_m", "RMS", "-"), ("H_max_m", "máx", "--")], "log"),
        ("Error relativo L2", [("C_L2_rel", "C", "-"), ("V_L2_rel", "|U|", "-")], "log"),
        ("Masa ∫C dΩ", [("masa_bfr", "BFR", "-"), ("masa_fem", "FEM", "--")], "linear"),
        ("C extraída en el pozo", [("Cpozo_bfr", "BFR", "-"), ("Cpozo_fem", "FEM", "--")], "linear"),
    )
    cursores, puntos = [], []
    for j, (tit, series, escala) in enumerate(curvas):
        ax = fig.add_subplot(ga[0, j])
        for clave, lab, ls in series:
            y = np.asarray(s[clave])
            ax.plot(t, y, ls, lw=1.6, label=lab)
            pt, = ax.plot(t[0], y[0], "o", ms=5, color=ax.lines[-1].get_color())
            puntos.append((pt, y))
        if escala == "log" and all((np.asarray(s[k]) > 0).all() for k, _, _ in series):
            ax.set_yscale("log")
        cursores.append(ax.axvline(t[0], color="k", lw=0.8, alpha=0.6))
        ax.set_title(tit, fontsize=11)
        ax.set_xlabel("t (días)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    sup = fig.suptitle("", fontsize=15)

    os.makedirs(out_dir, exist_ok=True)
    writer = FFMpegWriter(fps=12, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    with writer.saving(fig, os.path.join(out_dir, "comparacion.mp4"), dpi=90):
        for k in cuadros:
            for tp, arr, piso in paneles:
                tp.set_array(np.maximum(arr[k], vmin_v) if piso else arr[k])
            for cur in cursores:
                cur.set_xdata([t[k], t[k]])
            for pt, y in puntos:
                pt.set_data([t[k]], [y[k]])
            sup.set_text(f"{titulo}   |   t = {t[k]:.2f} días")
            writer.grab_frame()
    # El PNG es el último cuadro del video, no una figura aparte.
    fig.savefig(os.path.join(out_dir, "comparacion_final.png"), dpi=90)
    plt.close(fig)
    return [float(t[k]) for k in cuadros]


# ── Orquestación ─────────────────────────────────────────────────────────────

_M = None


def comparar_caso(g, c, spec, con_video=True):
    global _M
    if _M is None:
        _M = malla_comun()
    out = os.path.join(OUT_ROOT, g, c, "comparacion")
    datos, dt, cfg, ref = cargar_caso(g, c, spec, _M)
    s, r = metricas(datos, dt, cfg, _M)
    titulo = c if ref == f"{g}/{c}" else f"{c}  (FEM: {ref.split('/')[-1]})"
    if con_video:
        t_cuadros = video(datos, s, titulo, _M, out)
        r["cuadros_video"] = len(t_cuadros)
        s["t_cuadros"] = t_cuadros          # t de cada cuadro: el informe sincroniza el cursor
    r = {"grupo": g, "caso": c, "ref_fem": ref, "desc": spec["desc"], **r}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metricas.json"), "w") as fh:
        json.dump({"resumen": r, "series": {k: np.asarray(v).tolist() for k, v in s.items()},
                   "malla": {"area_m2": _M["area"], "n_nodos": len(_M["nodes"]),
                             "n_fantasmas": _M["n_fantasmas"]}}, fh)
    # metrics.json/series.png del comparador anterior quedarían desfasados.
    for viejo in ("metrics.json", "series.png", "grid.mp4", "grid_final.png"):
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
