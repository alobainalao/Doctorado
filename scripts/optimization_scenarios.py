"""
Corridas de optimización adjunta — 6 escenarios (barrido Pareto sobre gamma).

Cada escenario recorre la frontera Pareto J_M (contaminación en el pozo) vs
J_E (costo económico: profundidad + energía de bombeo), variando el peso gamma
del término de calidad.  Los demás parámetros son idénticos en todos los
escenarios para que las diferencias sean únicamente del balance calidad/costo.

Condiciones impuestas en todos los escenarios
──────────────────────────────────────────────
• Q0 = Q_std = 1e-3 m²/s  (constante, orden estándar)
  — garantiza subsidencia de contaminante desde la iteración 0: el bombeo
    arrastra el contaminante de z = +109 m (fuente) hacia z = −700 m (pozo).
• z0 = 0 m  (referencia de costo en superficie)
  — con zp_inicial = −700 m, JE|t=0 = koppa·700² + Q_std²·Nt·dt·700 ≠ 0.
• La restricción de suministro (supply_constraint) se aplica durante la
  optimización; Q0 puede ser infactible y SLSQP proyecta al interior.

Parámetros fijos
─────────────────
  domain  = real   (pozo [3700,−700], fuente [9116,109])
  model   = adr    (advección-difusión-reacción)
  spacing = 30 m   (Vscale = 30)
  dt      = 36 000 s  →  Nt = 72 pasos (T = 30 días)
  model   = mrmt_semi  (semi-analítico BFR)
  koppa   = 1e-2  (J2|t=0 ≈ 4900, orden JE ≈ 5159)
  Q_max   = 5e−3 m²/s
  Q_std   = 1e−3 m²/s  (Q0 y referencia)
  gamma   = [1e19, 5e19, 1e20, 2e20, 5e20, 1e21]  (absorbe escala C~1e-11)
  gamma_balance ≈ 1.48e20  (J1≈JE en estado inicial)

Uso:
    conda activate bfr_env
    python -m scripts.optimization_scenarios

Salidas por escenario (en p.save_data/):
    optimization_results.npz   — historial Q(iter), zp(iter), J(iter), x_opt
    convergence_<esc>.png      — ||g|| y J vs. iteración

Salida global:
    scripts/optimization_summary.png — comparación de los 6 escenarios (Pareto)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from funtions.runtime import RUNTIME
from config.parameters import Parameters

# =========================================================
# ESCENARIOS — barrido Pareto sobre gamma
# =========================================================
# gamma pequeño → el optimizador prioriza bajar JE (costo), acepta más
#                 contaminación en el pozo.
# gamma grande  → prioriza eliminar contaminación sin importar el costo.
#
# z0 = 0.0 en todos: referencia de costo en superficie → JE ≠ 0 en t=0
# porque zp_inicial = p.pozo[1] = −700 m ≠ 0.

_BASE = dict(
    run_type="optimization",
    domain="real",
    model="mrmt_semi",
    metodo="bfr",
    spacing=30,
    dt=36000,
    T=2592000,
    pre=True,
    save_dat=True,
    animate=False,
    postproc=False,
    # koppa=1e-2: J2|t=0 = 1e-2*700² = 4900 (orden JE ~ 5159)
    # gamma calibrado para modelo mrmt_semi (C ~ 1e-11):
    #   J1_pure (opt scale) ≈ 3.5e-17
    #   gamma_balance = JE / J1_pure ≈ 1.48e20
    # Barrido: gamma=1e19 (ECO) … 1e21 (AMB), cruce ~1.5e20
    koppa=1e-2,
    Q_max=5e-3,
    z0=0.0,         # referencia en superficie → JE|t=0 ≠ 0
    opt_maxiter=50,
    Nr=3,
    Deff=[1e-9, 5e-10, 1e-10],
    phi_im=[0.1, 0.05, 0.02],
    beta=[0.15, 0.1, 0.05],
)

SCENARIOS = [
    {"name": "pareto_e19",  "env": {**_BASE, "gamma": 1e19}},   # J1/JE≈0.07  ECO
    {"name": "pareto_e195", "env": {**_BASE, "gamma": 5e19}},   # J1/JE≈0.34  ECO
    {"name": "pareto_e20",  "env": {**_BASE, "gamma": 1e20}},   # J1/JE≈0.68  ~BAL
    {"name": "pareto_e205", "env": {**_BASE, "gamma": 2e20}},   # J1/JE≈1.36  ~BAL
    {"name": "pareto_e205h","env": {**_BASE, "gamma": 5e20}},   # J1/JE≈3.38  AMB
    {"name": "pareto_e21",  "env": {**_BASE, "gamma": 1e21}},   # J1/JE≈6.77  AMB
]

# Verificación de sensibilidad al punto inicial (Q0 = Q_max/2 vs Q_std)
SENSITIVITY_CHECK = False


def _set_params_and_sync(env):
    """Fija RUNTIME.params y sincroniza el p de nivel de módulo en animation.py."""
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()
    import view.animation as _anim
    _anim.p = p
    return p


def run_scenario(scenario):
    name = scenario["name"]
    env  = scenario["env"].copy()

    print(f"\n{'='*60}")
    print(f"ESCENARIO: {name}  (gamma={env['gamma']:.2g})")
    print(f"{'='*60}")

    p = _set_params_and_sync(env)

    # Verificar condición inicial: JE ≠ 0
    zp0 = float(p.pozo[1])
    z0  = float(p.z0)
    Q0  = float(p.Q_std)
    JE0 = p.koppa * (zp0 - z0)**2 + Q0**2 * p.Nt * p.dt * abs(zp0 - z0)
    print(f"  Q0       = {Q0:.3e} m²/s  (orden Q_std)")
    print(f"  zp0      = {zp0:.1f} m   z0 = {z0:.1f} m")
    print(f"  JE(t=0)  = {JE0:.4e}  {'✓ ≠ 0' if abs(JE0) > 1e-12 else '✗ = 0 — ERROR'}")

    from preprocessing.preprocess import load_data
    from view.animation import optimize_bfr

    d = load_data()

    results = {}

    # -- corrida desde Q0 = Q_std (base) --
    print(f"\n[{name}] corrida base (Q0 = Q_std = {Q0:.3e})...")
    res_base = optimize_bfr(d, p)
    results["base"] = res_base

    if SENSITIVITY_CHECK:
        # -- corrida desde Q0 = Q_max/2 --
        Q_max = float(getattr(p, "Q_max", 5e-3))
        Nt    = int(p.Nt)
        p2 = _set_params_and_sync(env.copy())
        p2.Qout[0] = np.full(Nt, Q_max / 2.0)
        print(f"\n[{name}] sensibilidad: Q0 = Q_max/2 = {Q_max/2:.3e}...")
        res_alt = optimize_bfr(d, p2)
        results["alt"] = res_alt

        J_base = float(res_base.fun)
        J_alt  = float(res_alt.fun)
        dif_rel = abs(J_base - J_alt) / max(abs(J_base), 1e-14)
        print(f"\n[{name}] J* base={J_base:.6e}  J* alt={J_alt:.6e}  dif_rel={dif_rel:.3e}")
        if dif_rel < 0.01:
            print(f"[{name}] → misma solución (óptimo único)")
        else:
            print(f"[{name}] → soluciones distintas (revisar óptimos locales)")

    _plot_convergence(name, results, p.save_data)
    return results


def _plot_convergence(name, results, save_dir):
    npz_path = os.path.join(save_dir, "optimization_results.npz")
    if not os.path.exists(npz_path):
        print(f"[{name}] optimization_results.npz no encontrado, saltando figura.")
        return

    data   = np.load(npz_path, allow_pickle=True)
    J_hist = data["J"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(J_hist)), J_hist, marker="o", markersize=3)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("J (escala log)")
    ax.set_title(f"Convergencia — {name}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    os.makedirs("scripts", exist_ok=True)
    fig_path = f"scripts/convergence_{name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[{name}] figura guardada: {fig_path}")


def plot_summary(all_results):
    """Gráfico de frontera Pareto J_M vs J_E para los 6 escenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gammas, J_finals = [], []
    for scenario in SCENARIOS:
        name = scenario["name"]
        npz  = _find_npz(name)
        if npz is None:
            continue
        data = np.load(npz, allow_pickle=True)
        gammas.append(scenario["env"]["gamma"])
        J_finals.append(float(data["J"][-1]))

    # Panel 1: convergencia por escenario
    ax0 = axes[0]
    for scenario in SCENARIOS:
        name = scenario["name"]
        npz  = _find_npz(name)
        if npz is None:
            continue
        data   = np.load(npz, allow_pickle=True)
        J_hist = data["J"]
        ax0.semilogy(J_hist, label=f"γ={scenario['env']['gamma']:.2g}", marker=".", markersize=2)
    ax0.set_title("Convergencia J (6 escenarios Pareto)")
    ax0.set_xlabel("Iteración")
    ax0.set_ylabel("J")
    ax0.legend(fontsize=8)
    ax0.grid(True, which="both", alpha=0.3)

    # Panel 2: J* vs gamma (posición sobre la frontera Pareto)
    ax1 = axes[1]
    if gammas:
        ax1.semilogx(gammas, J_finals, marker="o", linewidth=1.5, markersize=6)
        ax1.set_title("J* vs γ (frontera Pareto)")
        ax1.set_xlabel("γ (peso calidad)")
        ax1.set_ylabel("J* óptimo")
        ax1.grid(True, alpha=0.3)

    fig.suptitle("Barrido Pareto — 6 escenarios de optimización adjunta")
    fig.tight_layout()
    fig.savefig("scripts/optimization_summary.png", dpi=150)
    plt.close(fig)
    print("Resumen guardado: scripts/optimization_summary.png")


def _find_npz(name):
    candidate = f"./data/output/bfr/data/real/adr/optimization/optimization_results.npz"
    if os.path.exists(candidate):
        return candidate
    # fallback ruta anterior
    candidate2 = f"./data/output/bfr/data/real/adr/optimization_results.npz"
    if os.path.exists(candidate2):
        return candidate2
    return None


if __name__ == "__main__":
    all_results = {}
    for scenario in SCENARIOS:
        all_results[scenario["name"]] = run_scenario(scenario)

    plot_summary(all_results)
    print("\nTodos los escenarios completados.")
    print("Revisar scripts/optimization_summary.png y convergence_*.png")
    print("antes de proceder al análisis final del barrido Pareto.")
