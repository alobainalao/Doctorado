#!/usr/bin/env python
"""
test_kernel_fixes.py
--------------------
Prueba dos correcciones al kernel del pozo BFR y compara H cerca del pozo
contra la línea base BFR y FEniCS.

Fix 1: centrar discrete_delta en p.pozo (posición real del pozo) en lugar de
       pozo_cor (nodo más cercano, r≈16.4 m).
Fix 2: Fix 1 + forzar un nodo en p.pozo (análogo a gmsh.embed de FEniCS) vía
       force_well_node=true.

Uso:
    python scripts/test_kernel_fixes.py
"""

import os
import sys
import shutil
import subprocess
import numpy as np

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODIGO_DIR, "data", "output", "bfr", "data", "real", "adr", "comparison")
FENICS_FILE = os.path.join(CODIGO_DIR, "data", "output", "fenicsx", "data", "real", "adr", "comparison", "simulation_results.npz")
BASELINE_FILE = os.path.join(DATA_DIR, "simulation_results_baseline.npz")
FIX1_FILE     = os.path.join(DATA_DIR, "simulation_results_fix1.npz")
FIX2_FILE     = os.path.join(DATA_DIR, "simulation_results_fix2.npz")
MAIN_FILE     = os.path.join(DATA_DIR, "simulation_results.npz")

POZO = np.array([3700.0, -700.0])


def run_bfr(label, extra_env):
    env = {**os.environ, **extra_env}
    cmd = ["conda", "run", "-n", "bfr_env", "--no-capture-output",
           "python", "run.py"]
    print(f"\n[fix] ▶  {label}", flush=True)
    r = subprocess.run(cmd, cwd=CODIGO_DIR, env=env)
    if r.returncode != 0:
        print(f"[fix] ✗  FALLO: {label}  (rc={r.returncode})", flush=True)
        sys.exit(r.returncode)
    print(f"[fix] ✓  {label}", flush=True)


def h_min_near_well(npz_file, label, node_file=None):
    """Devuelve H en el nodo más cercano al pozo para cada timestep."""
    data = np.load(npz_file)
    H = data["H"]  # (Nt, K, N)
    # Si hay archivo de nodos (pkl) se usa; si no, buscamos en coords guardadas
    nodes_key = "nodes" if "nodes" in data else None

    # Intentar cargar nodos del pkl BFR o usar la aproximación de índice
    if node_file and os.path.exists(node_file):
        import pickle
        with open(node_file, "rb") as f:
            d = pickle.load(f)
        nodes = d["nodes"]
        idx = np.argmin(np.linalg.norm(nodes - POZO, axis=1))
        r = np.linalg.norm(nodes[idx] - POZO)
    else:
        # Fallback: asumimos índice 0 de H (K=1) y el primer nodo (raro)
        print(f"  [{label}] AVISO: sin archivo de nodos, no se puede encontrar índice del pozo")
        return None, None

    H_well = H[:, 0, idx]  # (Nt,)
    print(f"  [{label}] nodo más cercano al pozo: r={r:.2f} m  (idx={idx})")
    print(f"  [{label}] H_well t=0:    {H_well[0]:.4f} m")
    print(f"  [{label}] H_well t=final:{H_well[-1]:.4f} m")
    print(f"  [{label}] ΔH (drawdown): {H_well[-1] - H_well[0]:.4f} m")
    return H_well, idx


def h_min_fenicsx():
    """Carga H de FEniCS cerca del pozo."""
    if not os.path.exists(FENICS_FILE):
        print(f"  [FEniCS] archivo no encontrado: {FENICS_FILE}")
        return None
    data = np.load(FENICS_FILE)
    H = data["H"]  # (Nt, 1, N_fem)
    # FEniCS guarda coordenadas de nodos si están disponibles
    # Sin nodos guardados en npz, buscamos en el pkl de fenicsx si existe
    fenics_pkl = os.path.join(CODIGO_DIR, "data", "input", "fenicsx", "real", "adr", "preproceso.pkl")
    # FEniCS no usa pkl; buscamos en las coords de la malla si se guardaron
    # Usa heurística: el nodo con mayor drawdown en el timestep final
    H_last = H[-1, 0, :]
    idx = np.argmin(H_last)
    H_well = H[:, 0, idx]
    print(f"  [FEniCS] nodo con mayor drawdown (idx={idx})")
    print(f"  [FEniCS] H_well t=0:    {H_well[0]:.4f} m")
    print(f"  [FEniCS] H_well t=final:{H_well[-1]:.4f} m")
    print(f"  [FEniCS] ΔH (drawdown): {H_well[-1] - H_well[0]:.4f} m")
    return H_well


def print_summary(results):
    print("\n" + "═"*60)
    print("RESUMEN: H_well en t=final (drawdown respecto a t=0)")
    print("═"*60)
    print(f"  {'Variante':<22} {'H_final (m)':>12} {'Drawdown (m)':>14}")
    print("─"*60)
    for name, (H_well,) in results.items():
        if H_well is not None:
            dw = H_well[-1] - H_well[0]
            print(f"  {name:<22} {H_well[-1]:>12.4f} {dw:>14.4f}")
    print("═"*60)


# ── BFR pickle path ──────────────────────────────────────────────────────────
BFR_PKL    = os.path.join(CODIGO_DIR, "data", "input", "bfr", "real", "adr", "preproceso.pkl")
BFR_PKL_BK = BFR_PKL + ".baseline_backup"

if __name__ == "__main__":

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 0. Backup de resultados base ──────────────────────────────────────────
    if os.path.exists(MAIN_FILE) and not os.path.exists(BASELINE_FILE):
        shutil.copy2(MAIN_FILE, BASELINE_FILE)
        print(f"[fix] Backup baseline → {BASELINE_FILE}")

    if os.path.exists(BFR_PKL) and not os.path.exists(BFR_PKL_BK):
        shutil.copy2(BFR_PKL, BFR_PKL_BK)
        print(f"[fix] Backup pkl → {BFR_PKL_BK}")

    base_env = {
        "run_type":  "comparison",
        "model":     "adr",
        "metodo":    "bfr",
        "save_dat":  "true",
        "animate":   "false",
        "pre":       "true",
    }

    # ── 1. Fix 1: kernel centrado en p.pozo, sin nodo forzado ────────────────
    print("\n[fix] ═══  FIX 1: kernel center = p.pozo, sin nodo forzado  ═══")
    run_bfr("Fix 1 (kernel center)", {**base_env, "force_well_node": "false"})
    if os.path.exists(MAIN_FILE):
        shutil.copy2(MAIN_FILE, FIX1_FILE)
        print(f"[fix] Fix 1 guardado → {FIX1_FILE}")

    # ── 2. Fix 2: kernel centrado en p.pozo + nodo forzado en el pozo ────────
    print("\n[fix] ═══  FIX 2: kernel center = p.pozo + nodo forzado  ═══")
    run_bfr("Fix 2 (kernel center + forced node)", {**base_env, "force_well_node": "true"})
    if os.path.exists(MAIN_FILE):
        shutil.copy2(MAIN_FILE, FIX2_FILE)
        print(f"[fix] Fix 2 guardado → {FIX2_FILE}")

    # ── 3. Análisis comparativo ───────────────────────────────────────────────
    print("\n[fix] ═══  ANÁLISIS H cerca del pozo  ═══")

    results = {}

    print("\n── Baseline BFR (kernel centrado en pozo_cor, r≈16.4 m) ──")
    if os.path.exists(BASELINE_FILE):
        H_base, _ = h_min_near_well(BASELINE_FILE, "baseline", BFR_PKL_BK)
        results["BFR baseline"] = (H_base,)
    else:
        print("  AVISO: no hay baseline guardado")

    print("\n── Fix 1 BFR (kernel centrado en p.pozo, sin nodo) ──")
    if os.path.exists(FIX1_FILE):
        # Fix 1 pkl aún compatible con baseline pkl (mismo nodo set)
        H_fix1, _ = h_min_near_well(FIX1_FILE, "fix1", BFR_PKL_BK)
        results["BFR fix1 (centro)"] = (H_fix1,)

    print("\n── Fix 2 BFR (kernel centrado en p.pozo + nodo forzado) ──")
    if os.path.exists(FIX2_FILE):
        # Fix 2 tiene nuevos nodos → usar pkl actual (tiene el nodo forzado)
        H_fix2, _ = h_min_near_well(FIX2_FILE, "fix2", BFR_PKL)
        results["BFR fix2 (nodo)"] = (H_fix2,)

    print("\n── FEniCS (referencia) ──")
    H_fem = h_min_fenicsx()
    results["FEniCS"] = (H_fem,)

    print_summary(results)

    # ── 4. Guardar numpy con todos los H_well ────────────────────────────────
    out = {}
    for k, (v,) in results.items():
        if v is not None:
            out[k.replace(" ", "_")] = v
    np_out = os.path.join(DATA_DIR, "kernel_fix_comparison.npz")
    np.savez(np_out, **out)
    print(f"\n[fix] Comparación guardada en: {np_out}")
