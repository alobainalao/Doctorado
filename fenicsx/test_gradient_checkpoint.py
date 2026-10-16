"""
Checkpoint del gradiente adjunto del backend FEM (DOLFINx).

Compara el gradiente adjunto (∂J/∂Q_n, ∂J/∂z_p) contra diferencias finitas
centradas del MISMO funcional J(Q, z_p). Valida la consistencia interna del
adjunto discreto para los tres modelos: adr, mrmt_semi, mrmt_block.

Requiere el entorno `fenics_env` (dolfinx 0.11.0 conda-forge):

    /home/alex/miniconda3/envs/fenics_env/bin/python -m fenicsx.test_gradient_checkpoint [adr|mrmt_semi|mrmt_block]

Criterio: error relativo ~O(δ²) con "V" en log-log. O(1) ⇒ gradiente incorrecto.
"""
import os, sys
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
BASE_CHECKPOINT_ENV = {
    "metodo": "fenicsx",
    "run_type": "optimization",
    "domain": "real",
    "spacing": 120,     # malla gruesa (fenicsx es más caro que bfr)
    "dt": 36000,
    "T": 4 * 36000,     # horizonte corto: 4 pasos de tiempo
    "save_dat": False,
    "animate": False,
    "postproc": False,
    "pre": True,
}

# Parámetros MRMT mínimos (Nr=2 cubre el loop multi-región; igual que bfr)
MRMT_PARAMS = {
    "Nr": 2,
    "Deff":   [1e-9, 5e-10],
    "beta":   [0.15, 0.10],
    "phi_im": [0.1,  0.05],
}

def make_env(model: str) -> dict:
    env = {**BASE_CHECKPOINT_ENV, "model": model}
    if model.startswith("mrmt"):
        env.update(MRMT_PARAMS)
    return env

# Toggles de diagnóstico por variable de entorno (para aislar términos z_p).
CHECKPOINT_ENV = make_env("adr")
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


def run_model(model: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Checkpoint FEniCS — model={model}")
    print(f"{'='*60}")

    env = make_env(model)
    for _k in ("activate_fuente", "activate_ext"):
        if _k in os.environ:
            env[_k] = os.environ[_k].lower() == "true"

    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()

    mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mesh_cache)

    Q0 = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0 = np.concatenate([Q0, [zp0]])
    Nt = len(Q0)

    # Auto-escala γ: fuerza J1 ≫ términos económicos para validar el adjunto
    # físico. κ=0 aísla grad_zp de la economía.
    obs2_dt = _solve_forward(ctx, *_unpack(x0))["obs2_dt"]
    TARGET_J1 = 1e12
    p.gamma = TARGET_J1 / obs2_dt if obs2_dt > 0 else 1.0
    p.koppa = 0.0
    print(f"obs2_dt={obs2_dt:.3e}  → γ auto={p.gamma:.3e} (J1≈{TARGET_J1:.0e}); κ=0")
    idx_to_check = sorted(set([0, Nt // 2, Nt - 1]))
    print(f"Nt={Nt}, índices de Q a chequear: {idx_to_check}")

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
    ax.set_title(f"Checkpoint gradiente adjunto — fenicsx/{model}")
    ax.axhline(TOL_RELATIVE_ERROR, color="k", linestyle="--",
               label=f"tolerancia ({TOL_RELATIVE_ERROR:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = f"gradient_checkpoint_fenicsx_{model}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nFigura: {fname}")

    min_errors = {label: min(errs) for label, errs in results.items()}
    print("\n=== Resumen ===")
    all_pass = True
    for label, min_err in min_errors.items():
        status = "OK" if min_err < TOL_RELATIVE_ERROR else "FALLA"
        if min_err >= TOL_RELATIVE_ERROR:
            all_pass = False
        print(f"{label}: error relativo mínimo = {min_err:.3e}  [{status}]")

    print("Checkpoint OK." if all_pass else
          "Checkpoint FALLÓ: revisar fenicsx/adjoint.py.")
    return all_pass


def main():
    # Si se pasa un modelo por CLI se corre solo ese; si no, los tres.
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = ["adr", "mrmt_semi", "mrmt_block"]

    summary = {}
    for model in models:
        summary[model] = run_model(model)

    print(f"\n{'='*60}")
    print("RESUMEN GLOBAL")
    for m, ok in summary.items():
        print(f"  {m:15s}  {'OK' if ok else 'FALLA'}")
    print(f"{'='*60}")
    return all(summary.values())


if __name__ == "__main__":
    main()
