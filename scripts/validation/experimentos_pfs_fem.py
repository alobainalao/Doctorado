"""
scripts/validation/experimentos_pfs_fem.py
==========================================
Comparación BFR (modo "fem") vs FEniCS en dos escenarios:

  Experimento A — sin extracción (activate_ext=False)
  Experimento B — con extracción  (activate_ext=True)

MALLA COMPARTIDA (IMPORTANTE):
  Ambos backends usan EXACTAMENTE la misma malla de referencia:
  - 16,552 nodos generados por BFR con parámetros estándar (default.json)
  - Ubicada en: data/input/bfr/real/adr/preproceso.pkl
  - NUNCA se regenera — es la malla de referencia para todas las comparaciones
  - BFR siempre corre con pre=False (reutiliza pkl existente)
  - FEniCS construye su malla DOLFINx desde ese mismo pkl

Uso (desde codigo/):
  conda run -n bfr_env    python -m scripts.validation.experimentos_pfs_fem bfr_A
  conda run -n fenics_env python -m scripts.validation.experimentos_pfs_fem fem_A
  conda run -n bfr_env    python -m scripts.validation.experimentos_pfs_fem compare_A

  conda run -n bfr_env    python -m scripts.validation.experimentos_pfs_fem bfr_B
  conda run -n fenics_env python -m scripts.validation.experimentos_pfs_fem fem_B
  conda run -n bfr_env    python -m scripts.validation.experimentos_pfs_fem compare_B
"""

import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # codigo/
sys.path.insert(0, str(ROOT))

# Directorios de salida
SAVE_BFR_A  = ROOT / "data/output/bfr/data/real/adr/exp_A_nowell"
SAVE_FEM_A  = ROOT / "data/output/fenicsx/data/real/adr/exp_A_nowell"
SAVE_BFR_B  = ROOT / "data/output/bfr/data/real/adr/exp_B_well"
SAVE_FEM_B  = ROOT / "data/output/fenicsx/data/real/adr/exp_B_well"
REPORT_DIR  = ROOT / "reports/validation"
FIGURES_DIR = ROOT / "figures/comparison"
RESULTS_DIR = ROOT / "results/comparison"

for d in [SAVE_BFR_A, SAVE_FEM_A, SAVE_BFR_B, SAVE_FEM_B,
          REPORT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Verificación de malla compartida de referencia
# ---------------------------------------------------------------------------

def get_shared_mesh_path():
    """
    Devuelve la ruta de la malla COMPARTIDA y CENTRALIZADA para ambos backends.

    REGLA FUNDAMENTAL:
    - Ubicación única: data/input/bfr/real/adr/preproceso.pkl
    - 16,552 nodos exactos (generados con default.json)
    - NUNCA se regenera, se usa SIEMPRE en:
      * Todos los experimentos
      * Todas las ramas de git
      * Todas las configuraciones de parámetros (flujo/transporte)
    - BFR y FEniCS usan exactamente la MISMA malla
    - Parámetros de física pueden cambiar, pero NO la malla

    Llamar desde la app o scripts de comparación. Nunca modificar.
    """
    return ROOT / "data/input/bfr/real/adr/preproceso.pkl"


def get_shared_initial_conditions_path():
    """
    Devuelve la ruta del archivo HDF5 COMPARTIDO con las condiciones iniciales (CI).

    REGLA FUNDAMENTAL:
    - Ubicación única: data/input/funcion.h5 (dominio real)
    - Contiene H0 y C0 iniciales IDÉNTICOS para ambos backends
    - NUNCA se regenera — es la CI de referencia para todas las comparaciones
    - BFR y FEniCS interpolan desde los mismos nodos y valores

    Parámetros de dinámica (theta, dispersión, etc) pueden cambiar,
    pero las CI (H0, C0) SON FIJAS.
    """
    domain = "real"  # Los experimentos siempre usan dominio real
    if domain == "real":
        return ROOT / "data/input/funcion.h5"
    elif domain == "benchmark":
        return ROOT / "data/input/benchmark.h5"
    else:
        raise ValueError(f"Dominio inválido: {domain}")


def verify_reference_mesh():
    """
    Verifica que la malla COMPARTIDA existe y es válida.

    Esta malla es ÚNICA — la usan BFR y FEniCS siempre,
    sin importar rama, parámetros o contexto.
    """
    import pickle

    pkl_path = get_shared_mesh_path()

    if not pkl_path.exists():
        print("\n" + "="*70)
        print("❌ ERROR: Malla COMPARTIDA no existe")
        print("="*70)
        print(f"Ubicación: {pkl_path}")
        print("\n ESTA MALLA SE CREA UNA SOLA VEZ y se usa en TODO (app, scripts, ramas):")
        print("  1. Asegúrate que default.json está SIN CAMBIOS")
        print("  2. Ejecuta: conda run -n bfr_env python -m scripts.validation.etapa1_H_sin_pozo bfr")
        print("  3. Generará automáticamente: {pkl_path}")
        print("  4. ← ESA malla se usa en TODAS las comparaciones y ambos backends")
        print("="*70)
        raise FileNotFoundError(str(pkl_path))

    # Verificar número de nodos
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    n_nodes = len(data["nodes"])

    if n_nodes != 16552:
        print(f"\n⚠️  ADVERTENCIA: Malla tiene {n_nodes} nodos (esperado: 16,552)")
        print("   → NO modificar default.json ni pre=True en experimentos")
    else:
        print(f"✓ Malla COMPARTIDA verificada: {n_nodes} nodos (fija para BFR + FEniCS)")

    return pkl_path


def verify_reference_initial_conditions():
    """
    Verifica que las condiciones iniciales COMPARTIDAS (H0, C0) existen y son válidas.

    Estas CI son ÚNICAS — las usan BFR y FEniCS siempre,
    sin importar rama, parámetros o contexto.
    """
    import h5py

    h5_path = get_shared_initial_conditions_path()

    if not h5_path.exists():
        print("\n" + "="*70)
        print("❌ ERROR: Condiciones iniciales COMPARTIDAS no existen")
        print("="*70)
        print(f"Ubicación: {h5_path}")
        print("\nESTAS CI SE CREAN UNA SOLA VEZ y se usan en TODO (app, scripts, ramas):")
        print("  1. Asegúrate que default.json está SIN CAMBIOS")
        print("  2. Ejecuta: conda run -n bfr_env python -m scripts.validation.etapa1_H_sin_pozo bfr")
        print("  3. Generará automáticamente: {h5_path}")
        print("  4. ← ESAS CI se usan en TODAS las comparaciones y ambos backends")
        print("="*70)
        raise FileNotFoundError(str(h5_path))

    # Verificar contenido del archivo
    try:
        with h5py.File(h5_path, "r") as h5f:
            has_h = "h" in h5f
            has_c = "c" in h5f
            has_nodes_g = "nodes_G" in h5f
            has_nodes_q = "nodes_Q" in h5f

            if has_h:
                n_h = len(h5f["h"])
            if has_c:
                n_c = len(h5f["c"])

        if not (has_h and has_c and has_nodes_g and has_nodes_q):
            print(f"\n⚠️  ADVERTENCIA: Archivo HDF5 incompleto")
            print(f"   h={has_h}, c={has_c}, nodes_G={has_nodes_g}, nodes_Q={has_nodes_q}")
        else:
            print(f"✓ CI COMPARTIDAS verificadas: H({n_h} nodos), C({n_c} nodos)")

    except Exception as e:
        print(f"\n⚠️  ERROR al leer CI: {e}")
        raise

    return h5_path


# ---------------------------------------------------------------------------
# Helpers comunes
# ---------------------------------------------------------------------------

def _base_params(activate_ext: bool):
    """Parámetros compartidos entre BFR y FEM para los dos experimentos."""
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters

    p = Parameters()
    p.model           = "adr"
    p.activate_fuente = True
    p.activate_ext    = activate_ext
    p.animate         = False
    p.save_dat        = True
    return p


def _print_metadata(label: str, p):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  bc_outlet_mode = {getattr(p, 'bc_outlet_mode', 'pfs')}")
    print(f"  activate_ext   = {p.activate_ext}")
    print(f"  model          = {p.model}")
    print(f"  spacing        = {p.spacing}")
    print(f"  Nt             = {p.Nt}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Runners BFR
# ---------------------------------------------------------------------------

def run_bfr(activate_ext: bool, save_dir: Path, exp_label: str):
    from funtions.runtime import RUNTIME

    p = _base_params(activate_ext)
    p.metodo         = "bfr"
    p.bc_outlet_mode = "fem"   # modo de compatibilidad con FEniCS
    p.pre            = False   # NUNCA regenerar: reutilizar pkl estándar de referencia
    p.save_data      = str(save_dir)
    p.run_type       = "standard"

    Qout = [np.full(p.Nt, p.Q_std)] if activate_ext else [np.zeros(p.Nt)]
    p.Qout = Qout

    # Fijar RUNTIME antes de importar módulos que capturan p a nivel de módulo
    RUNTIME.params = p

    from view.animation import solve_forward
    from preprocessing.preprocess import load_data
    _print_metadata(f"BFR — {exp_label}", p)

    # Verificar que la malla COMPARTIDA existe
    pkl_shared = get_shared_mesh_path()
    if not pkl_shared.exists():
        raise FileNotFoundError(
            f"Malla COMPARTIDA no existe: {pkl_shared}\n"
            f"Crea primero con: conda run -n bfr_env python -m scripts.validation.etapa1_H_sin_pozo bfr"
        )

    d = load_data()
    solve_forward(d, p.Qout, animate=False, save_data=True)
    print(f"[BFR/{exp_label}] OK → {save_dir}/simulation_results.npz")


def run_bfr_A():
    run_bfr(activate_ext=False, save_dir=SAVE_BFR_A, exp_label="Exp-A (sin extracción)")

def run_bfr_B():
    run_bfr(activate_ext=True,  save_dir=SAVE_BFR_B, exp_label="Exp-B (con extracción)")


# ---------------------------------------------------------------------------
# Runners FEniCS
# ---------------------------------------------------------------------------

def run_fenicsx(activate_ext: bool, save_dir: Path, exp_label: str):
    from funtions.runtime import RUNTIME

    p = _base_params(activate_ext)
    p.metodo    = "fenicsx"
    p.save_data = str(save_dir)
    p.run_type  = "standard"

    RUNTIME.params = p

    from fenicsx.forward import solve_forward_fenicsx
    from fenicsx.mesh import build_mesh_from_bfr_pkl
    _print_metadata(f"FEM — {exp_label}", p)

    # Construir malla FEniCS desde la malla COMPARTIDA (centralizada, la misma para TODO)
    pkl_path = get_shared_mesh_path()
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"PKL estándar (malla compartida) no existe: {pkl_path}\n"
            f"Genera primero una simulación BFR con default.json para crear la malla de referencia."
        )

    print(f"[mesh] Construyendo malla FEniCS desde pkl estándar (16,552 nodos): {pkl_path}")
    mesh_cache = build_mesh_from_bfr_pkl(str(pkl_path))

    Qout = np.full(p.Nt, p.Q_std) if activate_ext else np.zeros(p.Nt)
    solve_forward_fenicsx(Qout, p.pozo, p,
                          mesh_cache=mesh_cache,
                          save_dir=str(save_dir))
    print(f"[FEM/{exp_label}] OK → {save_dir}/simulation_results.npz")


def run_fenicsx_A():
    run_fenicsx(activate_ext=False, save_dir=SAVE_FEM_A, exp_label="Exp-A (sin extracción)")

def run_fenicsx_B():
    run_fenicsx(activate_ext=True,  save_dir=SAVE_FEM_B, exp_label="Exp-B (con extracción)")


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------

def compare(exp_label: str, bfr_dir: Path, fem_dir: Path,
            activate_ext: bool, out_suffix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator
    import json

    bfr_npz = bfr_dir / "simulation_results.npz"
    fem_npz = fem_dir / "simulation_results.npz"

    if not bfr_npz.exists():
        print(f"ERROR: falta {bfr_npz}"); return
    if not fem_npz.exists():
        print(f"ERROR: falta {fem_npz}"); return

    bfr = np.load(bfr_npz)
    fem = np.load(fem_npz)

    nodes_bfr = bfr["nodes"]
    nodes_fem = fem["nodes"]
    H_bfr = bfr["H"]
    H_fem = fem["H"]
    C_bfr = bfr.get("C", None)
    C_fem = fem.get("C", None)

    # Normalizar shape (Nt, N)
    def _norm2d(arr):
        return arr[:, 0, :] if arr.ndim == 3 else arr

    H_bfr = _norm2d(H_bfr)
    H_fem = _norm2d(H_fem)
    if C_bfr is not None: C_bfr = _norm2d(C_bfr)
    if C_fem is not None: C_fem = _norm2d(C_fem)

    Nt = min(H_bfr.shape[0], H_fem.shape[0])
    H_bfr, H_fem = H_bfr[:Nt], H_fem[:Nt]
    if C_bfr is not None and C_fem is not None:
        C_bfr = C_bfr[:Nt]
        C_fem = C_fem[:Nt]

    print(f"\n{'='*65}")
    print(f"  {exp_label}: BFR(fem-mode) vs FEniCS")
    print(f"  activate_ext = {activate_ext}")
    print(f"  BFR nodos: {nodes_bfr.shape[0]}   FEM nodos: {nodes_fem.shape[0]}   Nt: {Nt}")

    # Verificar que las mallas sean idénticas
    if nodes_bfr.shape[0] != nodes_fem.shape[0]:
        print(f"\n⚠️  ADVERTENCIA: Número de nodos distinto!")
        print(f"   BFR: {nodes_bfr.shape[0]},  FEM: {nodes_fem.shape[0]}")
        print(f"   → Se usará interpolación FEM→BFR (NO es comparación nodo-a-nodo ideal)")
    else:
        max_dist = np.max(np.linalg.norm(nodes_bfr - nodes_fem, axis=1))
        if max_dist < 1e-6:
            print(f"  ✓ Mallas idénticas (distancia máxima: {max_dist:.2e})")
        else:
            print(f"\n⚠️  ADVERTENCIA: Mallas no coinciden (distancia máxima: {max_dist:.4f} m)")

    print(f"  H_BFR rango: [{H_bfr.min():.4f}, {H_bfr.max():.4f}]")
    print(f"  H_FEM rango: [{H_fem.min():.4f}, {H_fem.max():.4f}]")

    # ── Métricas H ──────────────────────────────────────────────────────────
    metrics_H = []
    H_diff_all = []
    H_fem_on_bfr = []

    t_steps_report = [0, Nt // 4, Nt // 2, Nt - 1]

    for t in range(Nt):
        interp = LinearNDInterpolator(nodes_fem, H_fem[t])
        H_f = interp(nodes_bfr)
        mask = ~np.isnan(H_f)

        H_b = H_bfr[t][mask]
        H_fi = H_f[mask]
        diff = H_b - H_fi
        ref_l2 = np.linalg.norm(H_b)
        ref_inf = np.max(np.abs(H_b)) if len(H_b) > 0 else 1.0

        metrics_H.append({
            "t_step": t,
            "n_valid": int(mask.sum()),
            "H_bfr_mean":  float(H_b.mean())  if len(H_b) > 0 else float("nan"),
            "H_fem_mean":  float(H_fi.mean()) if len(H_fi) > 0 else float("nan"),
            "H_bfr_range": float(H_b.max() - H_b.min()) if len(H_b) > 0 else float("nan"),
            "H_fem_range": float(H_fi.max() - H_fi.min()) if len(H_fi) > 0 else float("nan"),
            "L2_abs_m":    float(np.linalg.norm(diff) / np.sqrt(mask.sum())) if mask.sum() > 0 else float("nan"),
            "L2_rel_pct":  float(np.linalg.norm(diff) / ref_l2 * 100) if ref_l2 > 0 else float("nan"),
            "Linf_abs_m":  float(np.max(np.abs(diff))) if len(diff) > 0 else float("nan"),
            "Linf_rel_pct":float(np.max(np.abs(diff)) / ref_inf * 100) if ref_inf > 0 else float("nan"),
        })

        d_full = np.full(len(nodes_bfr), np.nan)
        d_full[mask] = diff
        H_diff_all.append(d_full)

        f_full = np.full(len(nodes_bfr), np.nan)
        f_full[mask] = H_fi
        H_fem_on_bfr.append(f_full)

    H_diff_all  = np.array(H_diff_all)
    H_fem_on_bfr = np.array(H_fem_on_bfr)

    # ── Métricas C (si disponible) ───────────────────────────────────────────
    metrics_C = []
    if C_bfr is not None and C_fem is not None:
        for t in range(Nt):
            interp_C = LinearNDInterpolator(nodes_fem, C_fem[t])
            C_f = interp_C(nodes_bfr)
            mask_c = ~np.isnan(C_f)
            C_b = C_bfr[t][mask_c]
            C_fi = C_f[mask_c]
            diff_c = C_b - C_fi
            ref_c = np.linalg.norm(C_b)
            metrics_C.append({
                "t_step": t,
                "L2_rel_pct": float(np.linalg.norm(diff_c) / ref_c * 100) if ref_c > 0 else float("nan"),
                "Linf_abs":   float(np.max(np.abs(diff_c))) if len(diff_c) > 0 else float("nan"),
                "C_bfr_max":  float(C_b.max())  if len(C_b)  > 0 else float("nan"),
                "C_fem_max":  float(C_fi.max()) if len(C_fi) > 0 else float("nan"),
            })

    # ── Tabla H ─────────────────────────────────────────────────────────────
    hdr = (f"{'t':>5} | {'N_val':>7} | {'H_BFR_mean':>11} | {'H_FEM_mean':>11} | "
           f"{'L2_abs[m]':>10} | {'L2_rel[%]':>10} | {'Linf[m]':>8} | {'Linf[%]':>8}")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for t in t_steps_report:
        if t < len(metrics_H):
            m = metrics_H[t]
            print(f"{m['t_step']:>5} | {m['n_valid']:>7} | "
                  f"{m['H_bfr_mean']:>11.4f} | {m['H_fem_mean']:>11.4f} | "
                  f"{m['L2_abs_m']:>10.4f} | {m['L2_rel_pct']:>10.3f} | "
                  f"{m['Linf_abs_m']:>8.4f} | {m['Linf_rel_pct']:>8.3f}")

    # ── Tabla C ─────────────────────────────────────────────────────────────
    if metrics_C:
        print(f"\nConcentración C — pasos seleccionados:")
        print(f"{'t':>5} | {'L2_rel[%]':>10} | {'Linf_abs':>10} | {'C_BFR_max':>10} | {'C_FEM_max':>10}")
        for t in t_steps_report:
            if t < len(metrics_C):
                m = metrics_C[t]
                print(f"{m['t_step']:>5} | {m['L2_rel_pct']:>10.3f} | "
                      f"{m['Linf_abs']:>10.3g} | {m['C_bfr_max']:>10.3g} | {m['C_fem_max']:>10.3g}")

    # ── Caudal de extracción (experimento B) ─────────────────────────────────
    if activate_ext:
        print("\nCaudal de extracción (integrado del pozo):")
        if "C_out" in bfr:
            print(f"  BFR C_out total = {bfr['C_out'].sum():.4g}")
        if "C_out" in fem:
            print(f"  FEM C_out total = {fem['C_out'].sum():.4g}")

    # ── Guardar ──────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / f"experimentos_{out_suffix}_metrics.json"
    with open(out_json, "w") as f:
        import json as _json
        _json.dump({
            "experiment": exp_label,
            "activate_ext": activate_ext,
            "bc_outlet_mode_bfr": "fem",
            "n_bfr": int(nodes_bfr.shape[0]),
            "n_fem": int(nodes_fem.shape[0]),
            "Nt": Nt,
            "metrics_H": metrics_H,
            "metrics_C": metrics_C,
        }, f, indent=2)

    np.savez_compressed(
        RESULTS_DIR / f"experimentos_{out_suffix}_diff.npz",
        H_diff=H_diff_all,
        H_bfr=H_bfr,
        H_fem_on_bfr=H_fem_on_bfr,
        nodes_bfr=nodes_bfr,
    )

    # ── Figuras ──────────────────────────────────────────────────────────────
    t_f = Nt - 1
    vmin = min(H_bfr[t_f].min(), np.nanmin(H_fem_on_bfr[t_f]))
    vmax = max(H_bfr[t_f].max(), np.nanmax(H_fem_on_bfr[t_f]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{exp_label}: H — BFR(fem-mode) vs FEniCS", fontsize=12)

    ax = axes[0, 0]
    sc = ax.scatter(nodes_bfr[:, 0], nodes_bfr[:, 1], c=H_bfr[t_f],
                    s=0.5, cmap="plasma", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="H [m]")
    ax.set_title(f"H BFR(fem-mode) — t={t_f+1}·dt")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = axes[0, 1]
    sc = ax.scatter(nodes_bfr[:, 0], nodes_bfr[:, 1], c=H_fem_on_bfr[t_f],
                    s=0.5, cmap="plasma", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="H [m]")
    ax.set_title(f"H FEniCS (→nodos BFR) — t={t_f+1}·dt")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = axes[1, 0]
    diff_f = H_diff_all[t_f]
    lim = np.nanpercentile(np.abs(diff_f[~np.isnan(diff_f)]), 99) if (~np.isnan(diff_f)).sum() > 0 else 1.0
    sc = ax.scatter(nodes_bfr[:, 0], nodes_bfr[:, 1], c=diff_f,
                    s=0.5, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(sc, ax=ax, label="ΔH [m]")
    ax.set_title(f"H_BFR − H_FEM — t={t_f+1}·dt")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = axes[1, 1]
    t_ax = np.arange(Nt)
    ax.plot(t_ax, [m["L2_rel_pct"] for m in metrics_H],   label="L2_rel [%]",   lw=1.5)
    ax.plot(t_ax, [m["Linf_rel_pct"] for m in metrics_H], label="Linf_rel [%]", lw=1.5)
    ax.axhline(5,  color="orange", ls="--", lw=0.8, label="5 %")
    ax.axhline(20, color="red",    ls="--", lw=0.8, label="20 %")
    ax.set_xlabel("paso de tiempo")
    ax.set_ylabel("error relativo [%]")
    ax.set_title("Evolución temporal del error en H")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"experimentos_{out_suffix}_H_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigura: {fig_path}")
    print(f"Métricas: {out_json}")

    # ── Dictamen ─────────────────────────────────────────────────────────────
    m_f = metrics_H[Nt - 1]
    L2r = m_f["L2_rel_pct"]
    Lir = m_f["Linf_rel_pct"]
    print(f"\n{'─'*55}")
    print(f"DICTAMEN {exp_label}:")
    print(f"  L2_rel  (t_f) = {L2r:.2f}%")
    print(f"  Linf_rel(t_f) = {Lir:.2f}%")
    if L2r < 5.0:
        print("  APROBADA — L2_rel < 5%")
    elif L2r < 20.0:
        print("  APROBADA CON RESERVAS — diferencias atribuibles a discretización")
    else:
        print("  NO APROBADA — L2_rel >= 20%. Revisar BCs antes de continuar.")
    print(f"{'─'*55}")


def compare_A():
    compare("Experimento A (sin extracción)",
            SAVE_BFR_A, SAVE_FEM_A,
            activate_ext=False, out_suffix="A_nowell")

def compare_B():
    compare("Experimento B (con extracción)",
            SAVE_BFR_B, SAVE_FEM_B,
            activate_ext=True, out_suffix="B_well")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

MODES = {
    "bfr_A":    run_bfr_A,
    "fem_A":    run_fenicsx_A,
    "compare_A":compare_A,
    "bfr_B":    run_bfr_B,
    "fem_B":    run_fenicsx_B,
    "compare_B":compare_B,
}

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPARACIÓN BFR(fem-mode) vs FEniCS")
    print("Malla + Condiciones Iniciales COMPARTIDAS (centralizadas)")
    print("="*70)

    # Verificar que la malla Y las CI de referencia existen ANTES de cualquier experimento
    verify_reference_mesh()
    verify_reference_initial_conditions()
    print()

    mode = sys.argv[1] if len(sys.argv) > 1 else "compare_A"
    if mode not in MODES:
        print(f"\n❌ Modo desconocido: {mode!r}")
        print(f"Modos disponibles: {', '.join(MODES)}")
        sys.exit(1)

    MODES[mode]()
