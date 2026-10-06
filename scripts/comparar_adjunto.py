#!/usr/bin/env python
"""
comparar_adjunto.py

Figura comparativa BFR (RBF-FD) vs FEniCS (FEM) de las soluciones adjuntas
(psi_H, psi_C) para los tres modelos (adr, mrmt_semi, mrmt_block).

Muestra un snapshot del instante t = T/2 (mitad del tiempo forward) y otro
en t ≈ 0 (donde el adjunto tiene su mayor amplitud).

Uso:
    python scripts/comparar_adjunto.py

Genera:
    data/output/comparison_adjunto_bfr_fenicsx.png
"""
import sys, os
_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

MODELS  = ["adr", "mrmt_semi", "mrmt_block"]
LABELS  = {"adr": "ADR", "mrmt_semi": "MRMT Semi", "mrmt_block": "MRMT Bloque"}
FIELDS  = [("psi_H", r"$\psi_H$"), ("psi_C", r"$\psi_C$")]

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de carga
# ──────────────────────────────────────────────────────────────────────────────

def load_adj(metodo, model):
    path = f"./data/output/{metodo}/data/real/{model}/adjoint_results.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    is_bfr = "nodes" in d
    if is_bfr:
        psi_H = d["psi_H"][:, 0, :]   # (Nt, N)
        psi_C = d["psi_C"][:, 0, :]
        nH = nC = d["nodes"]           # (N, 2)
    else:
        psi_H = d["psi_H"]             # (Nt, N_G)
        psi_C = d["psi_C"]             # (Nt, N_Q)
        nH    = d["nodes_G"]
        nC    = d["nodes_Q"]
    dt = float(d["dt"])
    return dict(psi_H=psi_H, psi_C=psi_C, nH=nH, nC=nC, dt=dt, Nt=psi_H.shape[0])


def grid_extents(data_list):
    xs, zs = [], []
    for d in data_list:
        if d is None:
            continue
        xs.extend([d["nH"][:, 0], d["nC"][:, 0]])
        zs.extend([d["nH"][:, 1], d["nC"][:, 1]])
    xall = np.concatenate(xs); zall = np.concatenate(zs)
    return xall.min(), xall.max(), zall.min(), zall.max()


def interp(psi, nodes, Xi, Zi):
    itp = LinearNDInterpolator(nodes, psi, fill_value=np.nan)
    return itp(Xi, Zi)


# ──────────────────────────────────────────────────────────────────────────────
# Cargar todos los datos
# ──────────────────────────────────────────────────────────────────────────────

data = {m: {b: load_adj(b, m) for b in ("bfr", "fenicsx")} for m in MODELS}

available = {m: {b: data[m][b] is not None for b in ("bfr", "fenicsx")} for m in MODELS}
print("Datos disponibles:")
for m in MODELS:
    print(f"  {LABELS[m]:14s}  BFR={available[m]['bfr']}  FEniCS={available[m]['fenicsx']}")

all_data = [data[m][b] for m in MODELS for b in ("bfr", "fenicsx") if data[m][b]]
xmin, xmax, zmin, zmax = grid_extents(all_data)
xi = np.linspace(xmin, xmax, 250)
zi = np.linspace(zmin, zmax, 70)
Xi, Zi = np.meshgrid(xi, zi)

# ──────────────────────────────────────────────────────────────────────────────
# Figura: 3 modelos × 2 backends × 2 campos = 12 paneles
# Dos columnas de comparación (BFR | FEniCS) por campo, 3 filas por modelo.
# Layout: cols = [psi_H BFR | psi_H FEniCS | psi_C BFR | psi_C FEniCS]
# ──────────────────────────────────────────────────────────────────────────────

NROW = len(MODELS)
NCOL = 4   # psi_H BFR | psi_H FEniCS | psi_C BFR | psi_C FEniCS
aspect = (xmax - xmin) / (zmax - zmin)
fig_w = NCOL * 3.5
fig_h = NROW * 2.2

fig, axes = plt.subplots(NROW, NCOL,
                         figsize=(fig_w, fig_h),
                         constrained_layout=True)
if NROW == 1:
    axes = axes[np.newaxis, :]

CMAPS = {"psi_H": "RdBu_r", "psi_C": "PuOr_r"}
col_titles = [
    r"$\psi_H$ — BFR", r"$\psi_H$ — FEniCS",
    r"$\psi_C$ — BFR", r"$\psi_C$ — FEniCS",
]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, fontsize=9, fontweight="bold")

backends   = ["bfr", "fenicsx"]
field_keys = [("psi_H", "nH"), ("psi_H", "nH"), ("psi_C", "nC"), ("psi_C", "nC")]
backend_ci = ["bfr", "fenicsx", "bfr", "fenicsx"]   # qué backend va en cada columna

for ri, model in enumerate(MODELS):
    vH = vC = 1e-30
    for b in ("bfr", "fenicsx"):
        dd = data[model][b]
        if dd is None:
            continue
        Nt = dd["Nt"]
        n0 = 0       # t ≈ 0 → máximo del adjunto
        vH = max(vH, abs(dd["psi_H"][n0]).max())
        vC = max(vC, abs(dd["psi_C"][n0]).max())
    vH = max(vH, 1e-30);  vC = max(vC, 1e-30)

    for ci, (fk, nk) in enumerate(field_keys):
        b  = backend_ci[ci]
        ax = axes[ri, ci]
        dd = data[model][b]
        cmap = CMAPS[fk]
        vmax = vH if fk == "psi_H" else vC

        if dd is None:
            ax.text(0.5, 0.5, "no disponible", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_axis_off()
            continue

        n0 = 0   # t = 0 (máximo adjunto)
        img = interp(dd[fk][n0], dd[nk], Xi, Zi)

        im = ax.imshow(img, origin="lower", aspect="auto",
                       extent=[xmin, xmax, zmin, zmax],
                       cmap=cmap, vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.ax.tick_params(labelsize=6)

        if ci == 0:
            ax.set_ylabel(f"{LABELS[model]}\nz (m)", fontsize=8)
        if ri == NROW - 1:
            ax.set_xlabel("x (m)", fontsize=8)
        ax.tick_params(labelsize=7)

fig.suptitle(
    "Soluciones adjuntas  BFR (RBF-FD) vs FEniCS — t = 0 (amplitud máxima)",
    fontsize=11, fontweight="bold", y=1.01
)

out = "./data/output/comparison_adjunto_bfr_fenicsx.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[comparar] ✓ {out}")

# ──────────────────────────────────────────────────────────────────────────────
# Segunda figura: diferencia absoluta BFR − FEniCS
# (solo si hay ambos backends para un modelo dado)
# ──────────────────────────────────────────────────────────────────────────────

has_both = [m for m in MODELS
            if data[m]["bfr"] is not None and data[m]["fenicsx"] is not None]

if has_both:
    NROW2 = len(has_both)
    fig2, axes2 = plt.subplots(NROW2, 2, figsize=(7, NROW2 * 2.2),
                               constrained_layout=True)
    if NROW2 == 1:
        axes2 = axes2[np.newaxis, :]

    axes2[0, 0].set_title(r"$\Delta\psi_H$ (BFR − FEniCS)", fontsize=9, fontweight="bold")
    axes2[0, 1].set_title(r"$\Delta\psi_C$ (BFR − FEniCS)", fontsize=9, fontweight="bold")

    for ri, model in enumerate(has_both):
        db = data[model]["bfr"]
        df = data[model]["fenicsx"]
        n0 = 0

        for ci, (fk, nk_b, nk_f) in enumerate([
            ("psi_H", "nH", "nH"),
            ("psi_C", "nC", "nC"),
        ]):
            img_b = interp(db[fk][n0], db[nk_b], Xi, Zi)
            img_f = interp(df[fk][n0], df[nk_f], Xi, Zi)
            diff  = img_b - img_f

            vmax = max(abs(np.nanmax(np.abs(diff))), 1e-30)

            im = axes2[ri, ci].imshow(
                diff, origin="lower", aspect="auto",
                extent=[xmin, xmax, zmin, zmax],
                cmap="coolwarm", vmin=-vmax, vmax=vmax
            )
            cb = fig2.colorbar(im, ax=axes2[ri, ci], shrink=0.85, pad=0.01)
            cb.ax.tick_params(labelsize=6)
            if ci == 0:
                axes2[ri, ci].set_ylabel(f"{LABELS[model]}\nz (m)", fontsize=8)
            axes2[ri, ci].tick_params(labelsize=7)

    fig2.suptitle(
        "Diferencia adjunta BFR − FEniCS  (t = 0)", fontsize=11, fontweight="bold"
    )
    out2 = "./data/output/comparison_adjunto_diferencia.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[comparar] ✓ {out2}")
else:
    print("[comparar] no hay modelos con ambos backends disponibles aún")
