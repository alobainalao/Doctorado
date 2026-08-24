"""
Checkpoint del gradiente adjunto (Pista A, semestre 3 — ver ROADMAP.md).

Compara grad_Q / grad_zp (funtions/utils.py, obtenidos vía el estado adjunto
psi_H/psi_C) contra diferencias finitas centradas sobre el funcional J(Q,z_p)
(compute_functional / functional, view/animation.py).

Es la única forma confiable de verificar la implementación del método
adjunto (funtions/operators.py, funtions/step_time.py, view/animation.py):
la revisión estática del código no puede resolver por sí sola las
convenciones de signo del esquema trapezoidal implícito (matrices A/B con
sig=∓1 en step_adjoint) frente a la derivación continua de
articulos/optimizacion-adjunto/optimizacion+adjunto.tex Sección 8.

Uso:
    conda activate bfr_env
    python -m tests.test_gradient_checkpoint

Criterio de aceptación: el error relativo debe decrecer ~O(delta^2) al
reducir delta y luego estabilizarse/crecer por error de redondeo (forma de
"V" en log-log). Si en cambio el error se mantiene ~O(1) para todo delta,
el gradiente adjunto es incorrecto.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funtions.runtime import RUNTIME
from config.parameters import Parameters
from preprocessing.preprocess import load_data
from view.animation import gradient, functional, pack_controls, unpack_controls


# =========================================================
# CONFIGURACIÓN: malla/horizonte reducidos para que el
# checkpoint corra en minutos, no en horas. No usar estos
# parámetros para las corridas de optimización "reales".
# =========================================================
CHECKPOINT_ENV = {
    "run_type": "optimization",
    "domain": "real",
    "model": "adr",
    "spacing": 60,      # malla más gruesa que default.json (30)
    "dt": 36000,
    "T": 5 * 36000,     # horizonte corto: 5 pasos de tiempo
    "save_dat": True,
    "animate": False,
    "postproc": False,
    "pre": True,
}

# Fracciones relativas del valor de cada control, no deltas absolutos: Q (~1e-3
# a 1e-5, ver Q_base en config/parameters.py) y z_p (~O(1-10) m) difieren en
# varios órdenes de magnitud, así que un delta absoluto fijo dejaría a uno de
# los dos fuera del régimen local donde la diferencia finita centrada aproxima
# bien la derivada.
DELTA_FRACTIONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
TOL_RELATIVE_ERROR = 1e-3
TIME_INDICES_TO_CHECK = None  # se definen tras conocer Nt (inicio/medio/fin)


def central_difference(x0, i, delta, d, p):
    x_plus = x0.copy()
    x_plus[i] += delta
    J_plus = functional(x_plus, d, p, animate=False, save_data=True)

    x_minus = x0.copy()
    x_minus[i] -= delta
    J_minus = functional(x_minus, d, p, animate=False, save_data=True)

    return (J_plus - J_minus) / (2.0 * delta)


def checkpoint_component(label, x0, i, g_adjoint_i, d, p):
    print(f"\n--- {label} (índice {i}) ---")
    print(f"grad adjunto = {g_adjoint_i: .6e}")

    scale = abs(x0[i]) if abs(x0[i]) > 1e-14 else 1.0

    errors = []
    for frac in DELTA_FRACTIONS:
        delta = frac * scale
        fd = central_difference(x0, i, delta, d, p)
        rel_err = (
            abs(fd - g_adjoint_i) / max(abs(g_adjoint_i), 1e-14)
        )
        errors.append(rel_err)
        print(f"  delta={delta:.3e} (frac={frac:.0e})  FD={fd: .6e}  err_rel={rel_err:.3e}")

    return errors


def main():
    RUNTIME.params = Parameters(env=CHECKPOINT_ENV)
    p = RUNTIME.get()

    d = load_data()

    Nt = len(p.Qout[0])
    global TIME_INDICES_TO_CHECK
    TIME_INDICES_TO_CHECK = sorted(set([0, Nt // 2, Nt - 1]))

    Q0 = p.Qout[0].copy()
    zp0 = float(p.pozo[1])
    x0 = pack_controls(Q0, zp0)

    print(f"Nt={Nt}, índices de Q a chequear: {TIME_INDICES_TO_CHECK}")

    # -----------------------------------------------------
    # gradiente adjunto (una sola resolución forward+adjunta)
    # -----------------------------------------------------
    g = gradient(x0, d, p, animate=False, save_data=True)

    results = {}

    for n in TIME_INDICES_TO_CHECK:
        results[f"grad_Q[{n}]"] = checkpoint_component(
            f"grad_Q[t_{n}]", x0, n, g[n], d, p
        )

    idx_zp = len(x0) - 1
    results["grad_zp"] = checkpoint_component(
        "grad_zp", x0, idx_zp, g[idx_zp], d, p
    )

    # -----------------------------------------------------
    # gráfica error relativo vs. delta (log-log)
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, errors in results.items():
        ax.loglog(DELTA_FRACTIONS, errors, marker="o", label=label)

    ax.set_xlabel(r"$\delta$ relativo ($\delta / |x_i|$)")
    ax.set_ylabel("error relativo")
    ax.set_title("Checkpoint del gradiente adjunto")
    ax.axhline(TOL_RELATIVE_ERROR, color="k", linestyle="--",
               label=f"tolerancia ({TOL_RELATIVE_ERROR:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("tests/gradient_checkpoint.png", dpi=150)
    print("\nFigura guardada en tests/gradient_checkpoint.png")

    # -----------------------------------------------------
    # veredicto
    # -----------------------------------------------------
    min_errors = {label: min(errs) for label, errs in results.items()}
    print("\n=== Resumen ===")
    all_pass = True
    for label, min_err in min_errors.items():
        status = "OK" if min_err < TOL_RELATIVE_ERROR else "FALLA"
        if min_err >= TOL_RELATIVE_ERROR:
            all_pass = False
        print(f"{label}: error relativo mínimo = {min_err:.3e}  [{status}]")

    if all_pass:
        print("\nCheckpoint OK: el gradiente adjunto es consistente con "
              "diferencias finitas.")
    else:
        print("\nCheckpoint FALLÓ: revisar make_adj_C_operator/"
              "make_adj_H_operator (funtions/operators.py) y su "
              "correspondencia con optimizacion+adjunto.tex Sección 8 "
              "antes de confiar en corridas de optimización.")

    return all_pass


if __name__ == "__main__":
    main()
