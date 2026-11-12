"""
Comparación cuantitativa BFR/RBF-FD vs FEM/FEniCS.

Calcula normas L2, L∞ y error relativo para los campos H, C, |U|, U_x, U_z
en los tres modelos (adr, mrmt_semi, mrmt_block), interpolando al grid común
(FEniCS como referencia por tener mayor densidad de nodos).

Salida:
  - scripts/comparison_report.txt   — tabla ASCII de resultados
  - scripts/comparison_report.csv   — tabla CSV
  - scripts/figures/comparison_*.png — paneles 2×2 por campo y modelo

Uso:
  python -m scripts.quantitative_comparison
  python -m scripts.quantitative_comparison --model adr
  python -m scripts.quantitative_comparison --model mrmt_semi --timestep -1
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "output"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

MODELS = ["adr", "mrmt_semi", "mrmt_block"]


# ---------------------------------------------------------------------------
# Interpolación a grid FEniCS (referencia)
# ---------------------------------------------------------------------------

def interpolate_to_target(values, src_nodes, tgt_nodes):
    """
    Interpola `values` (N_src,) definido en `src_nodes` (N_src,2)
    a los puntos `tgt_nodes` (N_tgt,2).

    Usa LinearNDInterpolator (baricéntrico) con fallback a vecino más cercano
    fuera de la envolvente convexa de src_nodes.
    """
    interp_lin = LinearNDInterpolator(src_nodes, values, fill_value=np.nan)
    result = interp_lin(tgt_nodes)

    # Rellenar NaN con vecino más cercano
    mask_nan = np.isnan(result)
    if mask_nan.any():
        interp_nn = NearestNDInterpolator(src_nodes, values)
        result[mask_nan] = interp_nn(tgt_nodes[mask_nan])

    return result


# ---------------------------------------------------------------------------
# Normas
# ---------------------------------------------------------------------------

def l2_norm(u, weights=None):
    """||u||_{L2} discreta (pesos uniformes si weights=None)."""
    if weights is None:
        return np.sqrt(np.mean(u**2))
    return np.sqrt(np.sum(u**2 * weights) / np.sum(weights))


def linf_norm(u):
    """||u||_{L∞} discreta."""
    return np.max(np.abs(u))


def relative_error(u_ref, u_approx, weights=None, eps=1e-30):
    """Error relativo en norma L2."""
    num = l2_norm(u_approx - u_ref, weights)
    den = l2_norm(u_ref, weights) + eps
    return num / den


def angular_error_deg(u_ref, u_approx):
    """
    Error angular medio (grados) entre dos campos vectoriales 2D.
    u_ref, u_approx: (N, 2)
    """
    # Magnitudes
    mag_ref = np.linalg.norm(u_ref, axis=1)
    mag_app = np.linalg.norm(u_approx, axis=1)

    # Producto punto normalizado
    valid = (mag_ref > 1e-15) & (mag_app > 1e-15)
    cos_theta = np.clip(
        np.sum(u_ref[valid] * u_approx[valid], axis=1)
        / (mag_ref[valid] * mag_app[valid]),
        -1.0, 1.0
    )
    angles = np.degrees(np.arccos(cos_theta))
    return np.mean(angles), np.max(angles)


# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------

def load_npz(backend, model, time_idx):
    """
    Carga simulation_results.npz del directorio comparison/.

    Retorna dict con:
      nodes   : (N, 2)
      H       : (N,)
      C       : (N,)
      U       : (N, 2)
      C_im    : (Nr, N) o None
      dt      : float
    """
    path = DATA / backend / "data" / "real" / model / "comparison" / "simulation_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}")

    with np.load(path, allow_pickle=True) as d:
        nodes = d["nodes"]                          # (N, 2)
        dt = float(d["dt"])

        # H: shape puede ser (Nt, 1, N) o (Nt, N)
        H_all = d["H"].squeeze()                    # → (Nt, N)
        C_all = d["C"].squeeze()                    # → (Nt, N)

        # U: BFR = (Nt, 1, N, 2), FEniCS = (Nt, N, 2)
        U_all = d["U"].reshape(d["U"].shape[0], -1, 2)   # → (Nt, N, 2)

        # C_im si existe
        C_im_all = None
        if "Ci" in d:
            C_im_all = d["Ci"]                      # BFR: (Nr, Nt, 1, N)
            C_im_all = C_im_all[:, :, 0, :]         # → (Nr, Nt, N)
        elif "C_im" in d:
            C_im_all = d["C_im"]                    # FEniCS: (Nt, Nr, N_Q) o similar
            # FEniCS C_im usa nodos_Q distintos; omitir por ahora
            C_im_all = None

        Nt = H_all.shape[0]
        tidx = time_idx % Nt

    return {
        "nodes": nodes,
        "H": H_all[tidx],
        "C": C_all[tidx],
        "U": U_all[tidx],
        "C_im": C_im_all[:, tidx, :] if C_im_all is not None else None,
        "dt": dt,
        "Nt": Nt,
        "tidx": tidx,
    }


# ---------------------------------------------------------------------------
# Análisis de un modelo a un instante
# ---------------------------------------------------------------------------

def compare_model(model, time_idx, verbose=True):
    """
    Compara BFR vs FEniCS para `model` en el instante `time_idx`.

    Retorna dict con todas las métricas.
    """
    bfr = load_npz("bfr", model, time_idx)
    fen = load_npz("fenicsx", model, time_idx)

    tgt = fen["nodes"]  # Grid de referencia = FEniCS

    # Interpolar BFR → grid FEniCS
    H_bfr_i = interpolate_to_target(bfr["H"], bfr["nodes"], tgt)
    C_bfr_i = interpolate_to_target(bfr["C"], bfr["nodes"], tgt)
    Ux_bfr_i = interpolate_to_target(bfr["U"][:, 0], bfr["nodes"], tgt)
    Uz_bfr_i = interpolate_to_target(bfr["U"][:, 1], bfr["nodes"], tgt)

    H_fen = fen["H"]
    C_fen = fen["C"]
    Ux_fen = fen["U"][:, 0]
    Uz_fen = fen["U"][:, 1]

    Umag_bfr = np.sqrt(Ux_bfr_i**2 + Uz_bfr_i**2)
    Umag_fen = np.sqrt(Ux_fen**2 + Uz_fen**2)

    U_bfr_2d = np.column_stack([Ux_bfr_i, Uz_bfr_i])
    U_fen_2d = np.column_stack([Ux_fen, Uz_fen])

    ang_mean, ang_max = angular_error_deg(U_fen_2d, U_bfr_2d)

    metrics = {
        "model": model,
        "tidx": bfr["tidx"],
        "t_days": bfr["tidx"] * bfr["dt"] / 86400,
        "N_bfr": len(bfr["nodes"]),
        "N_fen": len(fen["nodes"]),

        # H
        "H_l2_rel":   relative_error(H_fen, H_bfr_i),
        "H_linf_abs": linf_norm(H_bfr_i - H_fen),
        "H_range_bfr": float(bfr["H"].max() - bfr["H"].min()),
        "H_range_fen": float(H_fen.max() - H_fen.min()),

        # C
        "C_l2_rel":   relative_error(C_fen, C_bfr_i),
        "C_linf_abs": linf_norm(C_bfr_i - C_fen),
        "C_max_bfr": float(bfr["C"].max()),
        "C_max_fen": float(C_fen.max()),
        "C_neg_bfr": int((bfr["C"] < 0).sum()),  # valores negativos (artefactos)

        # |U|
        "U_l2_rel":   relative_error(Umag_fen, Umag_bfr),
        "U_linf_abs": linf_norm(Umag_bfr - Umag_fen),
        "U_max_bfr": float(Umag_bfr.max()),
        "U_max_fen": float(Umag_fen.max()),

        # Ux, Uz
        "Ux_l2_rel":  relative_error(Ux_fen, Ux_bfr_i),
        "Uz_l2_rel":  relative_error(Uz_fen, Uz_bfr_i),

        # Ángulo
        "U_ang_mean_deg": ang_mean,
        "U_ang_max_deg": ang_max,

        # Datos para figuras
        "_tgt": tgt,
        "_H_bfr_i": H_bfr_i,  "_H_fen": H_fen,
        "_C_bfr_i": C_bfr_i,  "_C_fen": C_fen,
        "_Umag_bfr": Umag_bfr, "_Umag_fen": Umag_fen,
        "_U_bfr_2d": U_bfr_2d, "_U_fen_2d": U_fen_2d,
    }

    if verbose:
        _print_metrics(metrics)

    return metrics


def _print_metrics(m):
    print(f"\n{'='*60}")
    print(f"  Modelo: {m['model'].upper()}   t = {m['t_days']:.1f} días  (paso {m['tidx']})")
    print(f"  Nodos BFR: {m['N_bfr']}   Nodos FEniCS: {m['N_fen']}")
    print(f"{'='*60}")
    print(f"  {'Variable':<12} {'err L2 rel':>12} {'err L∞ abs':>14} {'notas'}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*20}")
    print(f"  {'H (m)':<12} {m['H_l2_rel']:>12.4f} {m['H_linf_abs']:>14.4f}  range BFR={m['H_range_bfr']:.2f} FEM={m['H_range_fen']:.2f}")
    print(f"  {'C (kg/m³)':<12} {m['C_l2_rel']:>12.4f} {m['C_linf_abs']:>14.2e}  max BFR={m['C_max_bfr']:.2e} FEM={m['C_max_fen']:.2e}")
    if m["C_neg_bfr"] > 0:
        print(f"  {'':12}   *** {m['C_neg_bfr']} nodos BFR con C<0 (artefacto numérico)")
    print(f"  {'|U| (m/s)':<12} {m['U_l2_rel']:>12.4f} {m['U_linf_abs']:>14.2e}  max BFR={m['U_max_bfr']:.2e} FEM={m['U_max_fen']:.2e}")
    print(f"  {'Ux':<12} {m['Ux_l2_rel']:>12.4f}")
    print(f"  {'Uz':<12} {m['Uz_l2_rel']:>12.4f}")
    print(f"  {'U ángulo':<12} {'':>12} {'':>14}  medio={m['U_ang_mean_deg']:.2f}°  máx={m['U_ang_max_deg']:.2f}°")
    print()


# ---------------------------------------------------------------------------
# Figuras de comparación
# ---------------------------------------------------------------------------

def plot_comparison(metrics, field="C", save=True):
    """
    Genera panel 2×2: (a) BFR, (b) FEniCS, (c) diferencia absoluta, (d) error relativo.
    """
    model = metrics["model"]
    tgt = metrics["_tgt"]
    x, z = tgt[:, 0], tgt[:, 1]

    if field == "H":
        u_bfr = metrics["_H_bfr_i"]
        u_fen = metrics["_H_fen"]
        label = "h [m]"
        title_field = "Carga hidráulica h"
    elif field == "C":
        u_bfr = metrics["_C_bfr_i"]
        u_fen = metrics["_C_fen"]
        label = "C [kg/m³]"
        title_field = "Concentración C"
    elif field == "U":
        u_bfr = metrics["_Umag_bfr"]
        u_fen = metrics["_Umag_fen"]
        label = "|U| [m/s]"
        title_field = "Velocidad Darcy |U|"
    else:
        return

    diff = u_bfr - u_fen
    rel = np.abs(diff) / (np.abs(u_fen) + 1e-30)
    rel = np.clip(rel, 0, 10)  # evitar artefactos extremos

    # Escala común para BFR y FEniCS
    vmin = min(u_bfr.min(), u_fen.min())
    vmax = max(u_bfr.max(), u_fen.max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(f"{title_field} — {model.upper()}  "
                 f"(t = {metrics['t_days']:.1f} días)", fontsize=12)

    panels = [
        (axes[0, 0], u_bfr, "BFR/RBF-FD", "viridis", vmin, vmax, label),
        (axes[0, 1], u_fen, "FEM/FEniCS",  "viridis", vmin, vmax, label),
        (axes[1, 0], diff,  "Diferencia (BFR − FEM)", "RdBu_r",
         -np.abs(diff).max(), np.abs(diff).max(), label),
        (axes[1, 1], rel,   "Error relativo |diff|/|FEM|", "Reds",
         0, min(rel.max(), 2.0), ""),
    ]

    for ax, vals, title, cmap, vlo, vhi, cb_label in panels:
        sc = ax.scatter(x, z, c=vals, cmap=cmap, vmin=vlo, vmax=vhi,
                        s=0.3, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("z [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, label=cb_label, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        fname = FIGS / f"comparison_{model}_{field}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Figura guardada: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Informe CSV y TXT
# ---------------------------------------------------------------------------

def write_report(all_metrics, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).parent

    # Cabecera
    header = (
        "model,tidx,t_days,N_bfr,N_fen,"
        "H_l2_rel,H_linf_abs,H_range_bfr,H_range_fen,"
        "C_l2_rel,C_linf_abs,C_max_bfr,C_max_fen,C_neg_bfr,"
        "U_l2_rel,U_linf_abs,U_max_bfr,U_max_fen,"
        "Ux_l2_rel,Uz_l2_rel,"
        "U_ang_mean_deg,U_ang_max_deg\n"
    )

    csv_lines = [header]
    for m in all_metrics:
        csv_lines.append(
            f"{m['model']},{m['tidx']},{m['t_days']:.2f},"
            f"{m['N_bfr']},{m['N_fen']},"
            f"{m['H_l2_rel']:.6f},{m['H_linf_abs']:.6f},"
            f"{m['H_range_bfr']:.4f},{m['H_range_fen']:.4f},"
            f"{m['C_l2_rel']:.6f},{m['C_linf_abs']:.2e},"
            f"{m['C_max_bfr']:.2e},{m['C_max_fen']:.2e},{m['C_neg_bfr']},"
            f"{m['U_l2_rel']:.6f},{m['U_linf_abs']:.2e},"
            f"{m['U_max_bfr']:.2e},{m['U_max_fen']:.2e},"
            f"{m['Ux_l2_rel']:.6f},{m['Uz_l2_rel']:.6f},"
            f"{m['U_ang_mean_deg']:.4f},{m['U_ang_max_deg']:.4f}\n"
        )

    csv_path = out_dir / "comparison_report.csv"
    with open(csv_path, "w") as f:
        f.writelines(csv_lines)
    print(f"\nCSV guardado: {csv_path}")

    # Tabla ASCII
    txt_path = out_dir / "comparison_report.txt"
    with open(txt_path, "w") as f:
        f.write("COMPARACIÓN CUANTITATIVA BFR/RBF-FD vs FEM/FEniCS\n")
        f.write("="*75 + "\n")
        f.write(f"{'Modelo':<12} {'Campo':<8} {'Err L2 rel':>12} {'Err L∞ abs':>14} {'Notas'}\n")
        f.write("-"*75 + "\n")
        for m in all_metrics:
            f.write(f"{m['model']:<12} {'H':<8} {m['H_l2_rel']:>12.4f} {m['H_linf_abs']:>14.4f}  range BFR={m['H_range_bfr']:.2f} FEM={m['H_range_fen']:.2f}\n")
            f.write(f"{'':12} {'C':<8} {m['C_l2_rel']:>12.4f} {m['C_linf_abs']:>14.2e}  max BFR={m['C_max_bfr']:.2e} FEM={m['C_max_fen']:.2e}\n")
            neg_note = f"  *** {m['C_neg_bfr']} nodos con C<0" if m['C_neg_bfr'] > 0 else ""
            f.write(f"{'':12} {'|U|':<8} {m['U_l2_rel']:>12.4f} {m['U_linf_abs']:>14.2e}  max BFR={m['U_max_bfr']:.2e} FEM={m['U_max_fen']:.2e}{neg_note}\n")
            f.write(f"{'':12} {'U_ang':<8} {'':>12} {'':>14}  medio={m['U_ang_mean_deg']:.2f}°  máx={m['U_ang_max_deg']:.2f}°\n")
            f.write("-"*75 + "\n")

        # Interpretación
        f.write("\nINTERPRETACIÓN DE LAS MÉTRICAS\n")
        f.write("-"*75 + "\n")
        f.write(textwrap.dedent("""
        Error relativo L2:
          < 0.05  : Excelente acuerdo cuantitativo
          0.05–0.20: Acuerdo moderado (diferencias esperables por malla/BCs)
          > 0.20  : Diferencia significativa — investigar causa

        Error L∞:
          Diferencia máxima puntual (puede estar localizada en el pozo o frontera)

        Error angular del campo U:
          < 5°  : Campo direccional prácticamente idéntico
          5–15° : Diferencias moderadas en la dirección del flujo
          > 15° : Discrepancias direccionales significativas

        H range BFR vs FEM:
          Si range_BFR >> range_FEM → BCs de flujo diferentes (ver AUDIT_INVENTORY §3.1)
          La razón conocida: BFR usa Dirichlet; FEniCS usa condición natural (Neumann).
        """))

    print(f"Informe texto guardado: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparación cuantitativa BFR vs FEniCS")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--timestep", type=int, default=-1,
                        help="Índice de paso temporal (−1 = último)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]

    all_metrics = []
    for model in models:
        print(f"\n>>> Comparando modelo: {model.upper()}")
        try:
            m = compare_model(model, args.timestep, verbose=True)
            all_metrics.append(m)

            if not args.no_figures:
                for field in ["H", "C", "U"]:
                    plot_comparison(m, field=field)

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

    if all_metrics:
        write_report(all_metrics, out_dir=Path(__file__).parent)

    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN EJECUTIVO")
    print("="*60)
    for m in all_metrics:
        flag_H = "⚠️" if m["H_l2_rel"] > 0.2 else ("✓" if m["H_l2_rel"] < 0.05 else "○")
        flag_C = "⚠️" if m["C_l2_rel"] > 0.2 else ("✓" if m["C_l2_rel"] < 0.05 else "○")
        flag_U = "⚠️" if m["U_l2_rel"] > 0.2 else ("✓" if m["U_l2_rel"] < 0.05 else "○")
        print(f"  {m['model']:<12}  H:{flag_H} {m['H_l2_rel']:.3f}  "
              f"C:{flag_C} {m['C_l2_rel']:.3f}  "
              f"|U|:{flag_U} {m['U_l2_rel']:.3f}  "
              f"∠:{m['U_ang_mean_deg']:.1f}°")
    print()
    print("Leyenda: ✓ <5%  ○ 5-20%  ⚠️ >20%  error relativo L2")


if __name__ == "__main__":
    main()
