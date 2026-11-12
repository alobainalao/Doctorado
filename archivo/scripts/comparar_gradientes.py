"""
comparar_gradientes.py

Carga los gradientes adjuntos guardados por `guardar_gradiente.py` y genera
una figura comparativa BFR (RBF-FD) vs FEniCS (FEM) de ∂J/∂Q(t) y ∂J/∂z_p
para los tres modelos.

Prerequisito:
    # bfr_env
    python scripts/guardar_gradiente.py adr        bfr
    python scripts/guardar_gradiente.py mrmt_semi  bfr
    python scripts/guardar_gradiente.py mrmt_block bfr

    # fenics_env
    python scripts/guardar_gradiente.py adr        fenicsx
    python scripts/guardar_gradiente.py mrmt_semi  fenicsx
    python scripts/guardar_gradiente.py mrmt_block fenicsx

Uso (cualquier env con matplotlib):
    python scripts/comparar_gradientes.py

Genera:
    data/output/comparison_gradientes_bfr_fenicsx.png
"""

import sys, os
_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["adr", "mrmt_semi", "mrmt_block"]
LABELS = {"adr": "ADR", "mrmt_semi": "MRMT Semi", "mrmt_block": "MRMT Bloque"}
COLORS = {"bfr": "#1f77b4", "fenicsx": "#d62728"}
LINESTYLES = {"bfr": "-", "fenicsx": "--"}

# ─── Cargar datos ─────────────────────────────────────────────────────────────

def load_grad(metodo, model):
    path = f"./data/output/{metodo}/data/real/{model}/gradient_checkpoint.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return dict(
        grad_Q  = d["grad_Q"],
        grad_zp = float(np.asarray(d["grad_zp"]).ravel()[0]),
        tvec    = d["tvec"],
        gamma   = float(d["gamma"]),
        spacing = int(d["spacing"]),
    )

data = {m: {b: load_grad(b, m) for b in ("bfr", "fenicsx")} for m in MODELS}

print("Datos disponibles:")
for m in MODELS:
    bfr_ok  = data[m]["bfr"]  is not None
    fen_ok  = data[m]["fenicsx"] is not None
    print(f"  {LABELS[m]:14s}  BFR={bfr_ok}  FEniCS={fen_ok}")

# ─── Figura ───────────────────────────────────────────────────────────────────
# Filas = modelos,  cols = [∂J/∂Q  |  normalizado  |  ∂J/∂z_p]
NROW = len(MODELS)
fig, axes = plt.subplots(NROW, 3, figsize=(14, NROW * 3.2), constrained_layout=True)

for ri, model in enumerate(MODELS):
    ax_raw  = axes[ri, 0]
    ax_norm = axes[ri, 1]
    ax_zp   = axes[ri, 2]

    zp_vals = {}
    plotted = False
    for b in ("bfr", "fenicsx"):
        d = data[model][b]
        if d is None:
            continue
        t_h = d["tvec"] / 3600.0
        gQ  = d["grad_Q"]
        gzp = d["grad_zp"]
        sp  = d["spacing"]
        lbl = f"{b.upper()} (spacing={sp}m)"

        ax_raw.plot(t_h, gQ, color=COLORS[b], ls=LINESTYLES[b],
                    lw=1.5, label=lbl)

        norm = np.max(np.abs(gQ)) or 1.0
        ax_norm.plot(t_h, gQ / norm, color=COLORS[b], ls=LINESTYLES[b],
                     lw=1.5, label=lbl)

        zp_vals[b] = gzp
        plotted = True

    if not plotted:
        for ax in (ax_raw, ax_norm, ax_zp):
            ax.text(0.5, 0.5, "no disponible", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
        axes[ri, 0].set_ylabel(LABELS[model], fontsize=9)
        continue

    # Panel ∂J/∂z_p: barras por backend
    bs_avail = [b for b in ("bfr", "fenicsx") if b in zp_vals]
    ax_zp.bar(bs_avail,
              [zp_vals[b] for b in bs_avail],
              color=[COLORS[b] for b in bs_avail], alpha=0.8, width=0.5)
    ax_zp.axhline(0, color="k", linewidth=0.6)
    ax_zp.set_xticks(range(len(bs_avail)))
    ax_zp.set_xticklabels([b.upper() for b in bs_avail], fontsize=9)
    ax_zp.set_ylabel(r"$\partial J/\partial z_p$", fontsize=9)

    # Métricas de similitud cuando hay ambos
    if data[model]["bfr"] is not None and data[model]["fenicsx"] is not None:
        db = data[model]["bfr"]
        df = data[model]["fenicsx"]
        # interpolar al menor Nt
        Nt_b, Nt_f = len(db["grad_Q"]), len(df["grad_Q"])
        if Nt_b != Nt_f:
            t_common = np.linspace(0, min(db["tvec"][-1], df["tvec"][-1]), min(Nt_b, Nt_f))
            gQ_b = np.interp(t_common, db["tvec"], db["grad_Q"])
            gQ_f = np.interp(t_common, df["tvec"], df["grad_Q"])
        else:
            gQ_b = db["grad_Q"];  gQ_f = df["grad_Q"]

        # correlación de Pearson
        if gQ_b.std() > 0 and gQ_f.std() > 0:
            rho = np.corrcoef(gQ_b, gQ_f)[0, 1]
        else:
            rho = float("nan")

        # diferencia relativa en norma-2
        diff_norm = np.linalg.norm(gQ_b - gQ_f)
        avg_norm  = 0.5 * (np.linalg.norm(gQ_b) + np.linalg.norm(gQ_f))
        rel_diff  = diff_norm / max(avg_norm, 1e-14)

        ax_norm.set_title(f"ρ={rho:.3f}  |Δg|/|g|={rel_diff:.2e}", fontsize=8)
    else:
        ax_norm.set_title("(un solo backend)", fontsize=8)

    axes[ri, 0].set_ylabel(f"{LABELS[model]}\n∂J/∂Q", fontsize=9)
    for ax in (ax_raw, ax_norm):
        ax.set_xlabel("t (h)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    ax_zp.tick_params(labelsize=7)
    ax_zp.grid(True, axis="y", alpha=0.3)

axes[0, 0].set_title(r"$\partial J / \partial Q(t)$  — bruto", fontsize=9, fontweight="bold")
axes[0, 1].set_title(r"$\partial J / \partial Q(t)$ — normalizado", fontsize=9, fontweight="bold")
axes[0, 2].set_title(r"$\partial J / \partial z_p$", fontsize=9, fontweight="bold")

fig.suptitle(
    "Comparación de gradientes adjuntos: BFR (RBF-FD) vs FEniCS (FEM)\n"
    "(configuración checkpoint: malla gruesa, horizonte corto)",
    fontsize=11, fontweight="bold"
)

out = "./data/output/comparison_gradientes_bfr_fenicsx.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[comparar] ✓ {out}")
