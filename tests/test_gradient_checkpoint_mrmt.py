"""
Checkpoint del gradiente adjunto para modelos MRMT Semi y MRMT Block.

Mismo protocolo que test_gradient_checkpoint.py (ADR): compara grad_Q
obtenido por el adjunto contra diferencias finitas centradas sobre J.

Para MRMT Semi el adjunto backward recorre el operador splitting en orden
inverso (r = Nr-1..0); para MRMT Block usa las matrices de transporte
modificadas (con alpha_sum diagonal) y la fuente 4R·pho·alpha_r·C_im.

Uso:
    conda activate bfr_env
    python -m tests.test_gradient_checkpoint_mrmt [semi|block]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funtions.runtime import RUNTIME
from config.parameters import Parameters

DELTA_FRACTIONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
TOL_RELATIVE_ERROR = 5e-3   # más holgado que ADR: el split introduce O(dt)

BASE_ENV = {
    "run_type": "optimization",
    "domain": "real",
    "spacing": 60,
    "dt": 36000,
    "T": 3 * 36000,     # 3 pasos: rápido pero suficiente para gradiente
    "save_dat": True,
    "animate": False,
    "postproc": False,
    "pre": True,
    # parámetros MRMT mínimos (Nr=2 para cubrir el loop multi-región)
    "Nr": 2,
    "Deff": [1e-9, 5e-10],
    "beta": [0.15, 0.10],
    "phi_im": [0.1, 0.05],
}


def central_difference(x0, i, delta, d, p):
    from view.animation import functional
    x_plus  = x0.copy(); x_plus[i]  += delta
    x_minus = x0.copy(); x_minus[i] -= delta
    return (
        functional(x_plus,  d, p, animate=False, save_data=True)
        - functional(x_minus, d, p, animate=False, save_data=True)
    ) / (2.0 * delta)


def checkpoint_component(label, x0, i, g_adj, d, p):
    print(f"\n--- {label} (índice {i}) ---")
    print(f"grad adjunto = {g_adj: .6e}")
    scale = abs(x0[i]) if abs(x0[i]) > 1e-14 else 1.0
    errors = []
    for frac in DELTA_FRACTIONS:
        delta = frac * scale
        fd = central_difference(x0, i, delta, d, p)
        rel_err = abs(fd - g_adj) / max(abs(g_adj), 1e-14)
        errors.append(rel_err)
        print(f"  delta={delta:.2e}  FD={fd: .6e}  err_rel={rel_err:.3e}")
    return errors


def run_checkpoint(model_name):
    env = dict(BASE_ENV, model=model_name)
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()

    from preprocessing.preprocess import load_data
    from view.animation import gradient, functional, pack_controls, unpack_controls

    d = load_data()

    Nt = len(p.Qout[0])
    indices = sorted({0, Nt // 2, Nt - 1})
    print(f"\n=== Checkpoint MRMT {model_name.upper()} | Nt={Nt} ===")

    Q0  = p.Qout[0].copy()
    zp0 = float(p.pozo[1])
    x0  = pack_controls(Q0, zp0)

    g = gradient(x0, d, p, animate=False, save_data=True)

    results = {}
    for n in indices:
        results[f"grad_Q[{n}]"] = checkpoint_component(
            f"grad_Q[t_{n}]", x0, n, g[n], d, p
        )

    results["grad_zp"] = checkpoint_component(
        "grad_zp", x0, len(x0) - 1, g[-1], d, p
    )

    # --- gráfica ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, errors in results.items():
        ax.loglog(DELTA_FRACTIONS, errors, marker="o", label=label)
    ax.set_xlabel(r"$\delta$ relativo")
    ax.set_ylabel("error relativo")
    ax.set_title(f"Checkpoint gradiente adjunto — MRMT {model_name}")
    ax.axhline(TOL_RELATIVE_ERROR, color="k", ls="--",
               label=f"tol={TOL_RELATIVE_ERROR:.0e}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = f"tests/gradient_checkpoint_mrmt_{model_name}.png"
    fig.savefig(out, dpi=150)
    print(f"Figura: {out}")

    # --- veredicto ---
    print("\n=== Resumen ===")
    all_pass = True
    for label, errs in results.items():
        min_e = min(errs)
        ok = min_e < TOL_RELATIVE_ERROR
        if not ok:
            all_pass = False
        print(f"{label}: min err_rel={min_e:.3e}  {'OK' if ok else 'FALLA'}")

    if all_pass:
        print(f"\nCheckpoint MRMT {model_name} OK ✔")
    else:
        print(f"\nCheckpoint MRMT {model_name} FALLÓ — revisar step_adjoint MRMT.")
    return all_pass


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "semi"
    if model not in ("semi", "block"):
        print("Uso: python -m tests.test_gradient_checkpoint_mrmt [semi|block]")
        sys.exit(1)
    ok = run_checkpoint(f"mrmt_{model}")
    sys.exit(0 if ok else 1)
