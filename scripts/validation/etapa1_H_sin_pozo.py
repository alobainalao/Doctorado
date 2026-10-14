"""
scripts/validation/etapa1_H_sin_pozo.py
========================================
ETAPA 1 — Validación de H sin pozo, modelo ADR.

Ejecuta ambos backends con activate_ext=False y compara el campo de carga
hidráulica H mediante normas L2/L∞ y figuras.

Uso:
    conda run -n bfr_env    python -m scripts.validation.etapa1_H_sin_pozo bfr
    conda run -n fenics_env python -m scripts.validation.etapa1_H_sin_pozo fenicsx
    conda run -n bfr_env    python -m scripts.validation.etapa1_H_sin_pozo compare
"""

import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # codigo/
sys.path.insert(0, str(ROOT))

SAVE_BFR    = ROOT / "data/output/bfr/data/real/adr/validation_nowell"
SAVE_FEM    = ROOT / "data/output/fenicsx/data/real/adr/validation_nowell"
REPORT_DIR  = ROOT / "reports/validation"
FIGURES_DIR = ROOT / "figures/comparison"
RESULTS_DIR = ROOT / "results/comparison"

for d in [SAVE_BFR, SAVE_FEM, REPORT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Runner BFR ───────────────────────────────────────────────────────────────

def run_bfr():
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters

    p = Parameters()   # carga default.json
    p.model           = "adr"
    p.metodo          = "bfr"
    p.activate_ext    = False
    p.activate_fuente = False
    p.animate         = False
    p.save_dat        = True
    p.save_data       = str(SAVE_BFR)
    p.Qout            = [np.zeros(p.Nt)]
    RUNTIME.params    = p   # ANTES de importar view.animation (lee RUNTIME al importarse)

    from view.animation import solve_forward
    from preprocessing.preprocess import load_data

    d = load_data()
    solve_forward(d, p.Qout, animate=False, save_data=True)
    print(f"[BFR] OK → {SAVE_BFR}/simulation_results.npz")


# ── Runner FEniCS ────────────────────────────────────────────────────────────

def run_fenicsx():
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from fenicsx.forward import solve_forward_fenicsx
    from fenicsx.mesh import get_mesh

    p = Parameters()
    p.model           = "adr"
    p.metodo          = "fenicsx"
    p.activate_ext    = False
    p.activate_fuente = False
    p.animate         = False
    p.save_dat        = True
    p.save_data       = str(SAVE_FEM)
    RUNTIME.params    = p

    mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=False)
    Qout = np.zeros(p.Nt)   # sin extracción

    solve_forward_fenicsx(Qout, p.pozo, p,
                          mesh_cache=mesh_cache,
                          save_dir=str(SAVE_FEM))
    print(f"[FEniCS] OK → {SAVE_FEM}/simulation_results.npz")


# ── Comparación ──────────────────────────────────────────────────────────────

def compare():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator
    import json

    bfr_npz = SAVE_BFR / "simulation_results.npz"
    fem_npz = SAVE_FEM / "simulation_results.npz"

    if not bfr_npz.exists():
        print(f"ERROR: falta {bfr_npz}"); return
    if not fem_npz.exists():
        print(f"ERROR: falta {fem_npz}"); return

    bfr = np.load(bfr_npz)
    fem = np.load(fem_npz)

    nodes_bfr = bfr["nodes"]   # (N_bfr, 2)
    nodes_fem = fem["nodes"]   # (N_fem, 2)
    H_bfr = bfr["H"]
    H_fem = fem["H"]

    # Normalizar a (Nt, N)
    H_bfr = H_bfr[:, 0, :] if H_bfr.ndim == 3 else H_bfr
    H_fem = H_fem[:, 0, :] if H_fem.ndim == 3 else H_fem

    Nt = min(H_bfr.shape[0], H_fem.shape[0])
    H_bfr = H_bfr[:Nt]
    H_fem = H_fem[:Nt]

    print(f"BFR:    {nodes_bfr.shape[0]} nodos, {Nt} pasos")
    print(f"FEniCS: {nodes_fem.shape[0]} nodos, {Nt} pasos")
    print(f"H_BFR   rango: [{H_bfr.min():.4f}, {H_bfr.max():.4f}] m")
    print(f"H_FEM   rango: [{H_fem.min():.4f}, {H_fem.max():.4f}] m")

    # ── Interpolación FEM → nodos BFR ───────────────────────────────────────
    metrics = []
    t_steps = [0, Nt // 4, Nt // 2, Nt - 1]
    H_diff_all = []
    H_fem_on_bfr_all = []

    for t in range(Nt):
        interp = LinearNDInterpolator(nodes_fem, H_fem[t])
        H_f = interp(nodes_bfr)
        mask = ~np.isnan(H_f)

        H_b = H_bfr[t][mask]
        H_fi = H_f[mask]
        diff = H_b - H_fi

        ref_l2  = np.linalg.norm(H_b)
        ref_inf = np.max(np.abs(H_b))
        L2_abs  = np.linalg.norm(diff) / np.sqrt(mask.sum())
        L2_rel  = np.linalg.norm(diff) / ref_l2 if ref_l2 > 0 else np.nan
        Li_abs  = np.max(np.abs(diff))
        Li_rel  = Li_abs / ref_inf if ref_inf > 0 else np.nan

        metrics.append({
            "t_step": t, "n_valid": int(mask.sum()),
            "H_bfr_mean": float(H_b.mean()),
            "H_bfr_range": float(H_b.max() - H_b.min()),
            "H_fem_mean": float(H_fi.mean()),
            "H_fem_range": float(H_fi.max() - H_fi.min()),
            "L2_abs_m": float(L2_abs),
            "L2_rel_pct": float(L2_rel * 100),
            "Linf_abs_m": float(Li_abs),
            "Linf_rel_pct": float(Li_rel * 100),
        })

        diff_full = np.full(len(nodes_bfr), np.nan)
        diff_full[mask] = diff
        H_diff_all.append(diff_full)

        H_f_full = np.full(len(nodes_bfr), np.nan)
        H_f_full[mask] = H_fi
        H_fem_on_bfr_all.append(H_f_full)

    H_diff_all = np.array(H_diff_all)
    H_fem_on_bfr_all = np.array(H_fem_on_bfr_all)

    # ── Guardar ─────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "etapa1_H_nowell_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        RESULTS_DIR / "etapa1_H_nowell_diff.npz",
        H_diff=H_diff_all,
        H_bfr=H_bfr,
        H_fem_on_bfr=H_fem_on_bfr_all,
        nodes_bfr=nodes_bfr,
    )

    # ── Tabla de resumen ────────────────────────────────────────────────────
    header = (f"{'t':>5} | {'N_val':>7} | {'H_BFR_mean':>11} | {'H_FEM_mean':>11} | "
              f"{'L2_abs[m]':>10} | {'L2_rel[%]':>10} | {'Linf[m]':>8} | {'Linf[%]':>8}")
    print()
    print("=" * len(header))
    print("ETAPA 1 — H sin pozo, ADR: BFR vs FEniCS (nodos BFR como referencia)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in [metrics[t] for t in t_steps]:
        print(f"{m['t_step']:>5} | {m['n_valid']:>7} | "
              f"{m['H_bfr_mean']:>11.4f} | {m['H_fem_mean']:>11.4f} | "
              f"{m['L2_abs_m']:>10.4f} | {m['L2_rel_pct']:>10.3f} | "
              f"{m['Linf_abs_m']:>8.4f} | {m['Linf_rel_pct']:>8.3f}")
    print("=" * len(header))
    print(f"(métricas completas: {out_json})")

    # ── Figuras ──────────────────────────────────────────────────────────────
    t_f = Nt - 1
    vmin = min(H_bfr[t_f].min(), np.nanmin(H_fem_on_bfr_all[t_f]))
    vmax = max(H_bfr[t_f].max(), np.nanmax(H_fem_on_bfr_all[t_f]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("ETAPA 1: H sin pozo (ADR) — BFR vs FEniCS", fontsize=13)

    ax = axes[0, 0]
    sc = ax.scatter(nodes_bfr[:, 0], nodes_bfr[:, 1], c=H_bfr[t_f],
                    s=0.5, cmap="plasma", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="H [m]")
    ax.set_title(f"H BFR — t_f = {t_f+1}·dt")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = axes[0, 1]
    sc = ax.scatter(nodes_bfr[:, 0], nodes_bfr[:, 1], c=H_fem_on_bfr_all[t_f],
                    s=0.5, cmap="plasma", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="H [m]")
    ax.set_title(f"H FEniCS (→nodos BFR) — t_f = {t_f+1}·dt")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = axes[1, 0]
    diff_f = H_diff_all[t_f]
    lim = np.nanpercentile(np.abs(diff_f[~np.isnan(diff_f)]), 99)
    sc = ax.scatter(nodes_bfr[:, 0], nodes_bfr[:, 1], c=diff_f,
                    s=0.5, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(sc, ax=ax, label="ΔH [m]")
    ax.set_title(f"H_BFR − H_FEM — t_f = {t_f+1}·dt")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = axes[1, 1]
    t_ax = np.arange(Nt)
    ax.plot(t_ax, [m["L2_rel_pct"] for m in metrics],   label="L2_rel [%]",   lw=1.5)
    ax.plot(t_ax, [m["Linf_rel_pct"] for m in metrics], label="Linf_rel [%]", lw=1.5)
    ax.axhline(5,  color="orange", ls="--", lw=0.8, label="umbral 5%")
    ax.axhline(20, color="red",    ls="--", lw=0.8, label="umbral 20%")
    ax.set_xlabel("paso de tiempo")
    ax.set_ylabel("error relativo [%]")
    ax.set_title("Evolución temporal del error en H")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "etapa1_H_nowell_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figura: {fig_path}")

    # ── Dictamen ────────────────────────────────────────────────────────────
    m_f = metrics[Nt - 1]
    L2r = m_f["L2_rel_pct"]
    Lir = m_f["Linf_rel_pct"]
    diff_range = abs(m_f["H_bfr_range"] - m_f["H_fem_range"])

    print()
    print("─" * 60)
    print("DICTAMEN ETAPA 1 (H sin pozo, ADR):")
    print(f"  L2_rel  (t_f) = {L2r:.2f}%")
    print(f"  Linf_rel(t_f) = {Lir:.2f}%")
    print(f"  Rango BFR     = {m_f['H_bfr_range']:.4f} m")
    print(f"  Rango FEM     = {m_f['H_fem_range']:.4f} m")
    if L2r < 5.0:
        print("  ✅ APROBADA — L2_rel < 5%")
    elif L2r < 20.0:
        print(f"  ⚠️  APROBADA CON RESERVAS — diferencias atribuibles a BC/discretización")
    else:
        print(f"  ❌ NO APROBADA — L2_rel ≥ 20%. Revisar BCs antes de continuar.")
    print("─" * 60)


# ── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "bfr":
        run_bfr()
    elif mode == "fenicsx":
        run_fenicsx()
    elif mode == "compare":
        compare()
    else:
        print(f"Modo desconocido: {mode}. Usar: bfr | fenicsx | compare")
        sys.exit(1)
