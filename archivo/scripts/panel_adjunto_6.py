"""
panel_adjunto_6.py

Genera un panel 6×3 con snapshots del adjunto de concentración ψ_C
para las 6 combinaciones (3 modelos × 2 backends) a t = T, T/2, T/4.

Uso:
    python scripts/panel_adjunto_6.py
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

MODELS   = ["adr", "mrmt_semi", "mrmt_block"]
BACKENDS = ["bfr", "fenicsx"]
LABELS_M = {"adr": "ADR", "mrmt_semi": "MRMT Semi", "mrmt_block": "MRMT Bloque"}
LABELS_B = {"bfr": "BFR (RBF-FD)", "fenicsx": "FEniCS (FEM)"}

GRID_NX, GRID_NZ = 200, 150


def load(metodo, model):
    path = f"./data/output/{metodo}/data/real/{model}/adjoint_results.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    is_bfr = "nodes" in d
    if is_bfr:
        psi_C  = d["psi_C"][:, 0, :]  # (Nt, N)
        nodes  = d["nodes"]            # (N, 2)
    else:
        psi_C  = d["psi_C"]           # (Nt, N_Q)
        nodes  = d["nodes_Q"]
    dt   = float(d["dt"])
    return dict(psi_C=psi_C, nodes=nodes, dt=dt, Nt=psi_C.shape[0])


def snap(psi_C, nodes, t_idx, xi, zi):
    itp = LinearNDInterpolator(nodes, psi_C[t_idx], fill_value=np.nan)
    return itp(xi, zi)


# ─── Cargar datos ─────────────────────────────────────────────────────────────
data = {}
for m in MODELS:
    for b in BACKENDS:
        data[(m, b)] = load(b, m)

# Dominio común (usar todos los nodos disponibles)
all_x, all_z = [], []
for d in data.values():
    if d:
        all_x.append(d["nodes"][:, 0])
        all_z.append(d["nodes"][:, 1])
xmin, xmax = np.concatenate(all_x).min(), np.concatenate(all_x).max()
zmin, zmax = np.concatenate(all_z).min(), np.concatenate(all_z).max()

xi_1d = np.linspace(xmin, xmax, GRID_NX)
zi_1d = np.linspace(zmin, zmax, GRID_NZ)
Xi, Zi = np.meshgrid(xi_1d, zi_1d)

# ─── Figura ───────────────────────────────────────────────────────────────────
NROW = len(MODELS) * len(BACKENDS)   # 6
NCOL = 3                              # 3 instantes: T (t_idx=-1), T/2, T/4≈t_idx≈0

fig, axes = plt.subplots(
    NROW, NCOL,
    figsize=(14, 3.0 * NROW),
    constrained_layout=True
)

for ri, (m, b) in enumerate([(m, b) for m in MODELS for b in BACKENDS]):
    d = data[(m, b)]
    if d is None:
        for ci in range(NCOL):
            axes[ri, ci].text(0.5, 0.5, "no disponible",
                              ha="center", va="center",
                              transform=axes[ri, ci].transAxes)
        axes[ri, 0].set_ylabel(f"{LABELS_M[m]}\n{LABELS_B[b]}", fontsize=7)
        continue

    Nt = d["Nt"]
    # instantes backward: Nt-1 = t≈T (primer paso adjunto), Nt//2 = T/2, 0 = t≈0
    t_indices = [Nt - 1, Nt // 2, 0]
    t_labels  = [r"$t \approx T$ (inicio adjunto)", r"$t \approx T/2$", r"$t \approx 0$ (máximo)"]

    # escala común: máx absoluto de los 3 snapshots
    vals = [snap(d["psi_C"], d["nodes"], ti, Xi, Zi) for ti in t_indices]
    vmax = max(np.nanmax(np.abs(v)) for v in vals)
    vmax = vmax if vmax > 0 else 1.0

    for ci, (t_idx, t_lbl, val) in enumerate(zip(t_indices, t_labels, vals)):
        ax = axes[ri, ci]
        im = ax.pcolormesh(Xi, Zi, val, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, rasterized=True)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        if ri == 0:
            ax.set_title(t_lbl, fontsize=8, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{LABELS_M[m]}\n{LABELS_B[b]}", fontsize=7)
        ax.set_xlabel("x (m)", fontsize=6)

fig.suptitle(r"Adjunto de concentración $\psi_C$ — 6 combinaciones (3 modelos × 2 backends)",
             fontsize=11, fontweight="bold")

out = "./data/output/panel_adjunto_6_combinaciones.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"[panel] ✓ {out}")
