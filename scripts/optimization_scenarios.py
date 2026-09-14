"""
Corridas de optimización adjunta — 3 escenarios base (Pista A, sem3 mes 2).
Ver ROADMAP.md §"Mes 2: corridas de optimización".

Cada escenario varía Q_max / cotas de z_p; el optimizador es optimize_bfr
(scipy L-BFGS-B con restricción de suministro), que usa gradient()/functional()
validados por tests/test_gradient_checkpoint.py.

Uso:
    conda activate bfr_env
    python -m scripts.optimization_scenarios

Salidas por escenario (en p.save_data/):
    optimization_results.npz  — historial Q(iter), zp(iter), J(iter), x_opt
    convergence_<escenario>.png — ||g|| y J vs. iteración

Salida global:
    scripts/optimization_summary.png — comparación de los 3 escenarios
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from funtions.runtime import RUNTIME
from config.parameters import Parameters
# animation.py usa `p = RUNTIME.params` a nivel de módulo; debe importarse
# DESPUÉS de fijar RUNTIME.params (mismo patrón que run.py). Se importa
# dentro de run_scenario() antes de cada escenario y se actualiza
# view.animation.p explícitamente para que Simulation.dt sea correcto.

# =========================================================
# ESCENARIOS
# =========================================================
# Cada dict es el env que se pasa a Parameters; los campos no listados usan
# los defaults de _build_derived() (dominio real, modelo ADR, horizonte
# completo T=2592000, spacing=30).
SCENARIOS = [
    {
        "name": "conservador",
        "env": {
            "run_type": "optimization",
            "domain": "real",
            "model": "adr",
            "spacing": 30,
            "dt": 36000,
            "T": 2592000,
            "pre": True,
            "save_dat": True,
            "animate": False,
            "postproc": False,
            "gamma": 1.0,
            "koppa": 1.0,
            "Q_max": 5e-3,
            "opt_maxiter": 30,
        },
    },
    {
        "name": "agresivo",
        "env": {
            "run_type": "optimization",
            "domain": "real",
            "model": "adr",
            "spacing": 30,
            "dt": 36000,
            "T": 2592000,
            "pre": True,
            "save_dat": True,
            "animate": False,
            "postproc": False,
            "gamma": 1.0,
            "koppa": 1.0,
            "Q_max": 1e-2,
            "opt_maxiter": 30,
        },
    },
    {
        "name": "profundo",
        "env": {
            "run_type": "optimization",
            "domain": "real",
            "model": "adr",
            "spacing": 30,
            "dt": 36000,
            "T": 2592000,
            "pre": True,
            "save_dat": True,
            "animate": False,
            "postproc": False,
            "gamma": 1.0,
            "koppa": 2.0,   # mayor penalización en z_p para forzar pozo profundo
            "Q_max": 5e-3,
            # zp_min/zp_max se fijan post-build usando la geometría del dominio
            "opt_maxiter": 30,
        },
    },
]

# Punto inicial alternativo para verificar sensibilidad: Q0 = Q_max/2
SENSITIVITY_CHECK = True


def _set_params_and_sync(env):
    """Fija RUNTIME.params y sincroniza el p de nivel de módulo en animation.py."""
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()
    import view.animation as _anim
    _anim.p = p          # actualiza la variable de módulo que usa Simulation
    return p


def run_scenario(scenario):
    name = scenario["name"]
    env = scenario["env"].copy()

    print(f"\n{'='*60}")
    print(f"ESCENARIO: {name}")
    print(f"{'='*60}")

    p = _set_params_and_sync(env)

    # Para "profundo": forzar z_p al cuarto inferior del dominio
    if name == "profundo":
        from config.geometry import build_geometry
        geom = build_geometry(p.domain, p.spacing)
        z_all = geom.get("nodes_z", None)
        if z_all is None:
            z_lo = float(getattr(p, "zp_min", p.spacing))
            z_hi = float(getattr(p, "zp_max", p.spacing * 5))
        else:
            z_lo = float(np.percentile(z_all, 0))
            z_hi = float(np.percentile(z_all, 50))
        env["zp_min"] = z_lo
        env["zp_max"] = z_hi
        p = _set_params_and_sync(env)

    from preprocessing.preprocess import load_data
    from view.animation import optimize_bfr, pack_controls, unpack_controls, functional, gradient

    d = load_data()

    results = {}

    # -- corrida desde punto inicial base --
    print(f"\n[{name}] corrida desde Q0 base...")
    res_base = optimize_bfr(d, p)
    results["base"] = res_base

    if SENSITIVITY_CHECK:
        # -- corrida desde Q0 = Q_max/2 --
        Q_max = float(getattr(p, "Q_max", 1e-2))
        Nt = len(p.Qout[0])
        env2 = env.copy()
        p2 = _set_params_and_sync(env2)
        p2.Qout[0] = np.full(Nt, Q_max / 2.0)
        print(f"\n[{name}] corrida desde Q0 = Q_max/2 = {Q_max/2:.3e} (sensibilidad)...")
        res_alt = optimize_bfr(d, p2)
        results["alt"] = res_alt

        J_base = float(res_base.fun)
        J_alt  = float(res_alt.fun)
        print(f"\n[{name}] J* base={J_base:.6e}  J* alt={J_alt:.6e}  "
              f"dif_rel={abs(J_base-J_alt)/max(abs(J_base),1e-14):.3e}")
        if abs(J_base - J_alt) / max(abs(J_base), 1e-14) < 0.01:
            print(f"[{name}] → misma solución desde ambos arranques (óptimo único)")
        else:
            print(f"[{name}] → soluciones distintas (posibles óptimos locales o "
                  "sensibilidad al arranque — revisar)")

    _plot_convergence(name, results, p.save_data)
    return results


def _plot_convergence(name, results, save_dir):
    npz_path = os.path.join(save_dir, "optimization_results.npz")
    if not os.path.exists(npz_path):
        print(f"[{name}] optimization_results.npz no encontrado, saltando figura.")
        return

    data = np.load(npz_path, allow_pickle=True)
    J_hist = data["J"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(J_hist)), J_hist, marker="o", markersize=3)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("J (escala log)")
    ax.set_title(f"Convergencia — escenario '{name}'")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    os.makedirs("scripts", exist_ok=True)
    fig_path = f"scripts/convergence_{name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[{name}] figura guardada: {fig_path}")


def plot_summary(all_results):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 4),
                             sharey=False)
    if len(SCENARIOS) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, SCENARIOS):
        name = scenario["name"]
        npz_path = _find_npz(name)
        if npz_path is None:
            ax.set_title(f"{name}\n(sin datos)")
            continue
        data = np.load(npz_path, allow_pickle=True)
        J_hist = data["J"]
        ax.semilogy(J_hist, marker="o", markersize=3)
        ax.set_title(f"{name}")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("J")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Resumen convergencia — 3 escenarios de optimización (sem3)")
    fig.tight_layout()
    fig.savefig("scripts/optimization_summary.png", dpi=150)
    plt.close(fig)
    print("Resumen guardado: scripts/optimization_summary.png")


def _find_npz(name):
    # Busca optimization_results.npz en las rutas que genera Parameters
    # para el escenario dado (sin re-ejecutar Parameters completo).
    candidate = f"./data/output/bfr/data/real/adr/optimization_results.npz"
    if os.path.exists(candidate):
        return candidate
    return None


if __name__ == "__main__":
    all_results = {}
    for scenario in SCENARIOS:
        all_results[scenario["name"]] = run_scenario(scenario)

    plot_summary(all_results)
    print("\nTodos los escenarios completados.")
    print("Siguiente paso: revisar scripts/optimization_summary.png y los")
    print("  convergence_*.png antes de proceder al barrido de Pareto.")
