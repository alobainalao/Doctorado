"""
Barrido de γ para el frente de Pareto (Pista A, sem3 mes 2-3).
Ver ROADMAP.md §"Mes 2-3: frente de Pareto".

El funcional escalarizado es J = γ·J_m + J_e, donde:
  J_m = Σ_n (C_p(t_n))² · wi · dt      (contaminación en el pozo)
  J_e = κ·|z_p - z_0|² + Σ Q² · dt · |z_p - z_0|  (esfuerzo/energía)

Para cada γ en la malla log-espaciada se corre optimize_bfr() y se
descomponen los objetivos del óptimo. El conjunto de pares (J_m*, J_e*)
traza el frente de Pareto.

Uso:
    conda activate bfr_env
    python -m scripts.pareto_sweep

Salidas:
    scripts/pareto_results.npz  — arrays gamma, Jm, Je, J_total, x_opt
    scripts/pareto_front.png    — figura del frente de Pareto
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from funtions.runtime import RUNTIME
from config.parameters import Parameters


def _set_params_and_sync(env):
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()
    import view.animation as _anim
    _anim.p = p
    return p

# =========================================================
# CONFIGURACIÓN DEL BARRIDO
# =========================================================
GAMMA_VALUES = np.logspace(-3, 3, 8)   # 8 puntos log-espaciados [1e-3 ... 1e3]

BASE_ENV = {
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
    "koppa": 1.0,
    "Q_max": 5e-3,
    "opt_maxiter": 30,
}


def decompose_J(x_opt, d, env_base):
    """
    Calcula (J_m, J_e) del óptimo x_opt aislando cada objetivo.
    J_m = J con gamma=1, koppa=0 (solo contaminación)
    J_e = J con gamma=0, koppa=1 (solo esfuerzo)
    """
    from view.animation import functional

    env_jm = env_base.copy()
    env_jm["gamma"] = 1.0
    env_jm["koppa"] = 0.0
    p_jm = _set_params_and_sync(env_jm)
    Jm = float(functional(x_opt, d, p_jm, animate=False, save_data=False))

    env_je = env_base.copy()
    env_je["gamma"] = 0.0
    env_je["koppa"] = 1.0
    p_je = _set_params_and_sync(env_je)
    Je = float(functional(x_opt, d, p_je, animate=False, save_data=False))

    return Jm, Je


def run_sweep():
    gammas = []
    Jm_list = []
    Je_list = []
    Jtot_list = []
    x_opts = []

    for i, gamma in enumerate(GAMMA_VALUES):
        print(f"\n{'='*60}")
        print(f"gamma = {gamma:.4e}  ({i+1}/{len(GAMMA_VALUES)})")
        print(f"{'='*60}")

        env = BASE_ENV.copy()
        env["gamma"] = float(gamma)

        p = _set_params_and_sync(env)

        from preprocessing.preprocess import load_data
        from view.animation import optimize_bfr

        d = load_data()
        res = optimize_bfr(d, p)

        x_opt = res.x
        J_tot = float(res.fun)

        # Descomponer objetivos sin el peso γ del barrido
        Jm, Je = decompose_J(x_opt, d, BASE_ENV)

        print(f"  J_total={J_tot:.6e}  J_m={Jm:.6e}  J_e={Je:.6e}")

        gammas.append(gamma)
        Jm_list.append(Jm)
        Je_list.append(Je)
        Jtot_list.append(J_tot)
        x_opts.append(x_opt)

    return (
        np.array(gammas),
        np.array(Jm_list),
        np.array(Je_list),
        np.array(Jtot_list),
        np.array(x_opts),
    )


def check_nondominance(Jm, Je):
    """Devuelve True si ningún punto domina a otro (no-dominancia del frente)."""
    n = len(Jm)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if Jm[j] <= Jm[i] and Je[j] <= Je[i] and (Jm[j] < Jm[i] or Je[j] < Je[i]):
                dominated[i] = True
                break
    n_dom = dominated.sum()
    if n_dom == 0:
        print("No-dominancia verificada: ningún punto Pareto es dominado.")
    else:
        print(f"Advertencia: {n_dom} punto(s) dominado(s) — revisar convergencia "
              "de esas corridas (posible mínimo local).")
    return dominated


def plot_pareto(gammas, Jm, Je):
    # Ordenar por J_m creciente para trazar la curva
    order = np.argsort(Jm)
    Jm_s, Je_s, g_s = Jm[order], Je[order], gammas[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Jm_s, Je_s, c=np.log10(g_s), cmap="viridis",
                    s=80, zorder=5)
    ax.plot(Jm_s, Je_s, "k--", alpha=0.4, linewidth=0.8)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}\,\gamma$")

    ax.set_xlabel(r"$J_m$ — contaminación en el pozo")
    ax.set_ylabel(r"$J_e$ — esfuerzo de extracción")
    ax.set_title("Frente de Pareto  —  $J = \\gamma J_m + J_e$  (sem 3)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs("scripts", exist_ok=True)
    fig.savefig("scripts/pareto_front.png", dpi=150)
    plt.close(fig)
    print("Figura guardada: scripts/pareto_front.png")


if __name__ == "__main__":
    gammas, Jm, Je, Jtot, x_opts = run_sweep()

    # Verificar no-dominancia
    dominated = check_nondominance(Jm, Je)

    # Guardar resultados
    os.makedirs("scripts", exist_ok=True)
    np.savez(
        "scripts/pareto_results.npz",
        gamma=gammas,
        Jm=Jm,
        Je=Je,
        J_total=Jtot,
        x_opts=x_opts,
        dominated=dominated,
    )
    print("Resultados guardados: scripts/pareto_results.npz")

    plot_pareto(gammas, Jm, Je)

    print("\nResumen del frente de Pareto:")
    print(f"  {'γ':>10}  {'J_m':>12}  {'J_e':>12}  {'dominado':>9}")
    for g, jm, je, dom in zip(gammas, Jm, Je, dominated):
        print(f"  {g:10.4e}  {jm:12.5e}  {je:12.5e}  {'sí' if dom else 'no':>9}")

    print("\nSiguiente paso: revisar scripts/pareto_front.png y comparar con el")
    print("  esquema conceptual en optimizacion+adjunto.tex §frente de Pareto.")
