"""
Checkpoint del gradiente adjunto del backend FEM (DOLFINx) — espejo de
`tests/test_gradient_checkpoint.py` (backend bfr), pero llamando a
`fenicsx.adjoint.functional_fenicsx` / `gradient_fenicsx`.

Compara el gradiente adjunto (∂J/∂Q_n, ∂J/∂z_p, obtenido vía ψ_C/ψ_h y AD de
UFL) contra diferencias finitas centradas del MISMO funcional J(Q, z_p). Valida
la CONSISTENCIA interna del adjunto discreto (no la comparación con bfr, que
difiere por la discretización — ver fenicsx/README.md).

Sólo corre en el entorno DOLFINx (Docker `dolfinx/dolfinx:stable`):

    docker run --rm -v "$(pwd):/workspace" -w /workspace dolfinx/dolfinx:stable \
      bash -c "pip install -q pandas matplotlib scikit-learn h5py gmsh; \
               python -m fenicsx.test_gradient_checkpoint"

Criterio: el error relativo debe decrecer ~O(δ²) y luego crecer por redondeo
(forma de "V" en log-log). Error ~O(1) para todo δ ⇒ gradiente incorrecto.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funtions.runtime import RUNTIME
from config.parameters import Parameters
from fenicsx.mesh import get_mesh
from fenicsx.adjoint import (
    build_context, functional_fenicsx, gradient_fenicsx, _solve_forward, _unpack,
)


# Malla/horizonte reducidos para que corra en minutos (igual criterio que bfr).
CHECKPOINT_ENV = {
    "metodo": "fenicsx",
    "run_type": "optimization",
    "domain": "real",
    "model": "adr",
    "spacing": 120,     # malla gruesa (fenicsx es más caro que bfr)
    "dt": 36000,
    "T": 4 * 36000,     # horizonte corto: 4 pasos de tiempo
    "save_dat": False,
    "animate": False,
    "postproc": False,
    "pre": True,
}
# Toggles de diagnóstico por variable de entorno (para aislar términos z_p).
for _k in ("activate_fuente", "activate_ext"):
    if _k in os.environ:
        CHECKPOINT_ENV[_k] = os.environ[_k].lower() == "true"

DELTA_FRACTIONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
TOL_RELATIVE_ERROR = 1e-3


def central_difference(x0, i, delta, ctx):
    x_plus = x0.copy(); x_plus[i] += delta
    x_minus = x0.copy(); x_minus[i] -= delta
    return (functional_fenicsx(x_plus, ctx) - functional_fenicsx(x_minus, ctx)) / (2.0 * delta)


def checkpoint_component(label, x0, i, g_adjoint_i, ctx):
    print(f"\n--- {label} (índice {i}) ---")
    print(f"grad adjunto = {g_adjoint_i: .6e}")
    scale = abs(x0[i]) if abs(x0[i]) > 1e-14 else 1.0
    errors = []
    for frac in DELTA_FRACTIONS:
        delta = frac * scale
        fd = central_difference(x0, i, delta, ctx)
        rel_err = abs(fd - g_adjoint_i) / max(abs(g_adjoint_i), 1e-14)
        errors.append(rel_err)
        print(f"  delta={delta:.3e} (frac={frac:.0e})  FD={fd: .6e}  err_rel={rel_err:.3e}")
    return errors


def main():
    RUNTIME.params = Parameters(env=CHECKPOINT_ENV)
    p = RUNTIME.get()

    mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mesh_cache)

    Q0 = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0 = np.concatenate([Q0, [zp0]])
    Nt = len(Q0)

    # -----------------------------------------------------------------
    # Auto-escala γ para que la observación en el pozo (J1, y por tanto
    # ψ_C/ψ_h) DOMINE sobre los términos económicos analíticos. Sin esto,
    # con γ=1 la concentración (~1e-11) hace J1 despreciable y el checkpoint
    # sólo validaría los términos económicos, no la maquinaria adjunta.
    # También κ=0 para aislar la física en grad_zp.
    # -----------------------------------------------------------------
    obs2_dt = _solve_forward(ctx, *_unpack(x0))["obs2_dt"]
    TARGET_J1 = 1e12
    p.gamma = TARGET_J1 / obs2_dt if obs2_dt > 0 else 1.0
    p.koppa = 0.0
    print(f"obs2_dt={obs2_dt:.3e}  → γ auto={p.gamma:.3e} (J1≈{TARGET_J1:.0e}); κ=0")
    idx_to_check = sorted(set([0, Nt // 2, Nt - 1]))
    print(f"Nt={Nt}, índices de Q a chequear: {idx_to_check}")

    # gradiente adjunto (una sola resolución forward+adjunta)
    g = gradient_fenicsx(x0, ctx)

    results = {}
    for n in idx_to_check:
        results[f"grad_Q[{n}]"] = checkpoint_component(f"grad_Q[t_{n}]", x0, n, g[n], ctx)
    idx_zp = len(x0) - 1
    results["grad_zp"] = checkpoint_component("grad_zp", x0, idx_zp, g[idx_zp], ctx)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, errors in results.items():
        ax.loglog(DELTA_FRACTIONS, errors, marker="o", label=label)
    ax.set_xlabel(r"$\delta$ relativo ($\delta/|x_i|$)")
    ax.set_ylabel("error relativo")
    ax.set_title("Checkpoint del gradiente adjunto (fenicsx)")
    ax.axhline(TOL_RELATIVE_ERROR, color="k", linestyle="--",
               label=f"tolerancia ({TOL_RELATIVE_ERROR:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fenicsx/gradient_checkpoint.png", dpi=150)
    print("\nFigura guardada en fenicsx/gradient_checkpoint.png")

    min_errors = {label: min(errs) for label, errs in results.items()}
    print("\n=== Resumen ===")
    all_pass = True
    for label, min_err in min_errors.items():
        status = "OK" if min_err < TOL_RELATIVE_ERROR else "FALLA"
        if min_err >= TOL_RELATIVE_ERROR:
            all_pass = False
        print(f"{label}: error relativo mínimo = {min_err:.3e}  [{status}]")

    print("\nCheckpoint OK." if all_pass else
          "\nCheckpoint FALLÓ: revisar fenicsx/adjoint.py "
          "(residuales/derivadas UFL, signos de la recursión, kernels z_p).")
    return all_pass


if __name__ == "__main__":
    main()
