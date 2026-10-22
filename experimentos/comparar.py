"""
Comparación nodo a nodo BFR (RBF-FD) vs FEniCS (FEM) sobre la malla compartida.

FEniCS guarda sus campos en los vértices P1, que son los nodos BFR sin los
fantasmas; se emparejan por coordenadas (sin interpolación).

Por paso de tiempo y para H, C y |U|:
    L2_rel  = ‖u_bfr − u_fem‖₂ / ‖u_fem‖₂
    Linf_rel = max|u_bfr − u_fem| / max|u_fem|
y la curva de llegada C(t) en el nodo del pozo para ambos backends.

    python -m experimentos.comparar --caso A_terminos/A0_sin_fuente_sin_pozo
    python -m experimentos.comparar --grupo A_terminos B_bc_salida_fem
    python -m experimentos.comparar --todos

Salidas: <caso>/comparacion/{metrics.json, series.png} y
         data/output/experimentos/resumen.csv (una fila por caso).
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.catalogo import GRUPOS, iter_casos  # noqa: E402

OUT_ROOT = os.path.join(CODIGO_DIR, "data", "output", "experimentos")
TOL_NODO = 1e-6   # [m] tolerancia para emparejar nodos BFR ↔ vértices FEM
CAMPOS = ("H", "C", "U")


def _cargar(path):
    z = np.load(path, allow_pickle=True)
    H = np.asarray(z["H"], float)[:, 0, :]
    C = np.asarray(z["C"], float)[:, 0, :]
    U = np.asarray(z["U"], float)
    if U.ndim == 4:                       # BFR: (Nt, K, N, 2)
        U = U[:, 0]
    return dict(nodes=np.asarray(z["nodes"], float), H=H, C=C,
                U=np.linalg.norm(U, axis=-1), dt=float(z["dt"]))


def _normas(a, b):
    """a = BFR, b = FEM (referencia); arrays (Nt, N)."""
    d = a - b
    l2 = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)
    linf = np.abs(d).max(axis=1) / np.maximum(np.abs(b).max(axis=1), 1e-300)
    return l2, linf


def comparar_caso(g, c, spec):
    caso_dir = os.path.join(OUT_ROOT, g, c)
    ref = spec.get("ref_fem", f"{g}/{c}")
    f_bfr = os.path.join(caso_dir, "bfr", "simulation_results.npz")
    f_fem = os.path.join(OUT_ROOT, ref, "fenicsx", "simulation_results.npz")
    for f in (f_bfr, f_fem):
        if not os.path.exists(f):
            print(f"[comparar] ⏭  {g}/{c}: falta {os.path.relpath(f, CODIGO_DIR)}")
            return None

    bfr, fem = _cargar(f_bfr), _cargar(f_fem)
    if bfr["H"].shape[0] != fem["H"].shape[0] or bfr["dt"] != fem["dt"]:
        print(f"[comparar] ✗  {g}/{c}: Nt/dt distintos entre backends")
        return None

    dist, idx = cKDTree(bfr["nodes"]).query(fem["nodes"])
    ok = dist < TOL_NODO
    if ok.sum() < 0.99 * len(fem["nodes"]):
        print(f"[comparar] ✗  {g}/{c}: sólo {ok.sum()}/{len(ok)} nodos coinciden — ¿malla no compartida?")
        return None
    ib, ifem = idx[ok], np.nonzero(ok)[0]

    # Nodo del pozo: el de la malla congelada más cercano a pozo_cor.
    with open(os.path.join(CODIGO_DIR, "data", "input", "malla_compartida", "malla.pkl"), "rb") as f:
        pozo = np.asarray(pickle.load(f)["pozo_cor"], float)
    k_pozo = int(np.argmin(np.linalg.norm(bfr["nodes"][ib] - pozo, axis=1)))

    Nt, dt = bfr["H"].shape[0], bfr["dt"]
    t_dias = (np.arange(Nt) + 1) * dt / 86400
    series = {"t_dias": t_dias.tolist()}
    resumen = {"grupo": g, "caso": c, "ref_fem": ref, "n_nodos": int(ok.sum()), "Nt": Nt}
    for campo in CAMPOS:
        l2, linf = _normas(bfr[campo][:, ib], fem[campo][:, ifem])
        series[f"{campo}_L2_rel"] = l2.tolist()
        series[f"{campo}_Linf_rel"] = linf.tolist()
        resumen[f"{campo}_L2_rel_final"] = float(l2[-1])
        resumen[f"{campo}_L2_rel_max"] = float(l2.max())
        resumen[f"{campo}_Linf_rel_final"] = float(linf[-1])
    c_pozo_bfr = bfr["C"][:, ib][:, k_pozo]
    c_pozo_fem = fem["C"][:, ifem][:, k_pozo]
    series["C_pozo_bfr"] = c_pozo_bfr.tolist()
    series["C_pozo_fem"] = c_pozo_fem.tolist()
    resumen["C_pozo_bfr_final"] = float(c_pozo_bfr[-1])
    resumen["C_pozo_fem_final"] = float(c_pozo_fem[-1])
    # Indicador de masa: suma nodal de C (no es la integral; sirve para el cociente).
    resumen["C_suma_ratio_final"] = float(bfr["C"][-1, ib].sum() / max(fem["C"][-1, ifem].sum(), 1e-300))

    out = os.path.join(caso_dir, "comparacion")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"resumen": resumen, "series": series}, f, indent=1)
    _figura(series, f"{g}/{c}", os.path.join(out, "series.png"))
    print(f"[comparar] ✓  {g}/{c}:  L2_rel final  H={resumen['H_L2_rel_final']:.2e}  "
          f"C={resumen['C_L2_rel_final']:.2e}  |U|={resumen['U_L2_rel_final']:.2e}")
    return resumen


def _figura(s, titulo, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = s["t_dias"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for campo, lab in (("H", "H"), ("C", "C"), ("U", "|U|")):
        ax[0].semilogy(t, s[f"{campo}_L2_rel"], label=lab)
    ax[0].set(xlabel="t [días]", ylabel="L2 relativo BFR vs FEM", title="Error relativo")
    ax[0].legend()
    ax[1].plot(t, s["C_pozo_bfr"], label="BFR")
    ax[1].plot(t, s["C_pozo_fem"], "--", label="FEM")
    ax[1].set(xlabel="t [días]", ylabel="C en el pozo", title="Curva de llegada")
    ax[1].legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _actualizar_resumen(filas):
    path = os.path.join(OUT_ROOT, "resumen.csv")
    previas = {}
    if os.path.exists(path):
        with open(path) as f:
            previas = {(r["grupo"], r["caso"]): r for r in csv.DictReader(f)}
    for r in filas:
        previas[(r["grupo"], r["caso"])] = r
    if not previas:
        return
    campos = list(next(iter(previas.values())).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for k in sorted(previas):
            w.writerow(previas[k])
    print(f"[comparar] resumen → {os.path.relpath(path, CODIGO_DIR)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", nargs="+")
    ap.add_argument("--caso", nargs="+")
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="leer de data/output/experimentos_smoke/")
    args = ap.parse_args()
    if args.smoke:
        global OUT_ROOT
        OUT_ROOT += "_smoke"
    if not (args.grupo or args.caso or args.todos):
        ap.error("indica --grupo, --caso o --todos")

    filas = []
    for g, c, spec in iter_casos(args.grupo, args.caso):
        if not GRUPOS[g].get("comparar", True):
            continue
        r = comparar_caso(g, c, spec)
        if r:
            filas.append(r)
    _actualizar_resumen(filas)


if __name__ == "__main__":
    main()
