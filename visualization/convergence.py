"""
Visualización de curvas de convergencia:
  - Error vs paso de malla (estudio de convergencia espacial)
  - Error vs ε (validación del gradiente por diferencias finitas)
  - Historia de convergencia del optimizador
"""

import numpy as np
import matplotlib.pyplot as plt
from visualization.style import COLORS, MARKERS, apply_style


def plot_gradient_fd_check(eps_arr, err_arr, method_label="BFR",
                           profile="THESIS", ax=None):
    """
    Curva log-log de error relativo del gradiente vs tamaño de perturbación ε.

    Se espera:
      - Régimen de truncamiento: pendiente ≈ +2 (curva decreciente a la derecha)
      - Régimen de redondeo: pendiente ≈ -1 (curva creciente a la izquierda)
      - Mínimo < tolerancia: gradiente validado

    Parámetros
    ----------
    eps_arr   : array de perturbaciones relativas
    err_arr   : array de errores relativos correspondientes
    """
    with apply_style(profile):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            fig = ax.figure
            own_fig = False

        ax.loglog(eps_arr, err_arr, "o-",
                  color=COLORS["bfr"], label=f"Error relativo ({method_label})",
                  linewidth=1.5, markersize=5)

        # Líneas de referencia O(ε) y O(ε²)
        eps_ref = np.array([eps_arr.min(), eps_arr.max()])
        e_mid = np.min(err_arr)
        eps_mid = eps_arr[np.argmin(err_arr)]

        ax.loglog(eps_ref, e_mid * (eps_ref / eps_mid)**2,
                  "--", color="gray", alpha=0.5, label="O(ε²)")
        ax.loglog(eps_ref, e_mid * (eps_ref / eps_mid)**(-1),
                  ":",  color="gray", alpha=0.5, label="O(1/ε)")

        ax.axhline(1e-3, color="red", linestyle="--", alpha=0.4, label="Tolerancia 1e-3")

        ax.set_xlabel("Perturbación relativa ε")
        ax.set_ylabel("Error relativo del gradiente")
        ax.set_title(f"Validación del gradiente adjunto — {method_label}")
        ax.legend(fontsize="small")
        ax.grid(True, which="both", alpha=0.2)

    return (fig, ax) if own_fig else ax


def plot_convergence_history(iterations, J_values, grad_norms=None,
                              label="", profile="THESIS"):
    """
    Historia de convergencia del optimizador.

    iterations : array de índices de iteración
    J_values   : valores del funcional J en cada iteración
    grad_norms : ||∇J|| en cada iteración (opcional)
    """
    with apply_style(profile):
        nplots = 2 if grad_norms is not None else 1
        fig, axes = plt.subplots(1, nplots, figsize=plt.rcParams["figure.figsize"])

        if nplots == 1:
            axes = [axes]

        # J(k)
        axes[0].semilogy(iterations, J_values, "o-",
                          color=COLORS["bfr"], linewidth=1.5, markersize=4,
                          label=label or "J(k)")
        axes[0].set_xlabel("Iteración k")
        axes[0].set_ylabel("J(Q, z_p)")
        axes[0].set_title("Historia de convergencia — funcional J")
        axes[0].grid(True, alpha=0.3)

        if grad_norms is not None:
            axes[1].semilogy(iterations, grad_norms, "s-",
                              color=COLORS["adjoint"], linewidth=1.5, markersize=4,
                              label="||∇J||")
            axes[1].set_xlabel("Iteración k")
            axes[1].set_ylabel("||∇J||")
            axes[1].set_title("Norma del gradiente")
            axes[1].grid(True, alpha=0.3)

    return fig, axes


def plot_pareto_front(J1_vals, J3_vals, gamma_vals=None,
                       label="Pareto BFR", profile="THESIS"):
    """
    Frente de Pareto (J_m, J_e) del problema multiobjetivo.

    J1_vals   : valores de J1 (monitoreo contaminación)
    J3_vals   : valores de J3 (costo de bombeo)
    gamma_vals: valores de γ usados para la escalarización (para colorear)
    """
    with apply_style(profile):
        fig, ax = plt.subplots()

        if gamma_vals is not None:
            sc = ax.scatter(J1_vals, J3_vals, c=gamma_vals, cmap="viridis",
                             s=40, zorder=3, label=label)
            plt.colorbar(sc, ax=ax, label="γ (peso J₁)")
        else:
            ax.plot(J1_vals, J3_vals, "o-", color=COLORS["bfr"],
                     linewidth=1.5, markersize=6, label=label)

        ax.set_xlabel("J₁ — concentración en pozo [kg²·m/s]")
        ax.set_ylabel("J₃ — costo de bombeo [m³/s·m]")
        ax.set_title("Frente de Pareto")
        ax.legend()
        ax.grid(True, alpha=0.3)

    return fig, ax
