"""
Cruce de resultados de la optimización adjunta entre los backends `bfr`
(RBF-FD) y `fenicsx` (FEM/DOLFINx).

Ambos backends usan el MISMO optimizador (scipy L-BFGS-B) alimentado por su
gradiente adjunto (validado en tests/test_gradient_checkpoint.py y
fenicsx/test_gradient_checkpoint.py) y guardan el historial en el MISMO formato
`optimization_results.npz` (Q, zp, J por iteración + x_opt/J_opt). Este script
los carga y compara: trayectoria de J, de z_p y el control Q(t) óptimo.

Uso (matplotlib basta, no requiere rbf ni dolfinx):
    python -m scripts.cross_opt_bfr_fenicsx

Requisito: haber corrido ANTES la optimización en ambos backends con el mismo
escenario (mismo spacing/dt/T/gamma/koppa/z0), p.ej. vía run.py o la app:
    bfr     -> data/output/bfr/data/real/adr/optimization_results.npz
    fenicsx -> data/output/fenicsx/data/real/adr/optimization_results.npz
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(metodo, domain="real", model="adr"):
    path = f"./data/output/{metodo}/data/{domain}/{model}/optimization_results.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No hay resultados de optimización para '{metodo}' en {path}. "
            f"Corre antes la optimización de ese backend (run.py / app)."
        )
    d = np.load(path, allow_pickle=True)
    return dict(Q=d["Q"], zp=d["zp"], J=d["J"], x_opt=d["x_opt"], J_opt=float(d["J_opt"]))


def main(domain="real", model="adr"):
    bfr = _load("bfr", domain, model)
    fen = _load("fenicsx", domain, model)

    # -------- resumen numérico --------
    print(f"{'':12} {'J_inicial':>14} {'J_final':>14} {'z_p*':>10} {'evals':>7}")
    for name, r in [("bfr", bfr), ("fenicsx", fen)]:
        print(f"{name:12} {r['J'][0]:14.6e} {r['J_opt']:14.6e} "
              f"{float(r['x_opt'][-1]):10.3f} {len(r['J']):7d}")

    # -------- figura comparativa --------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) J vs evaluación (semilog)
    for name, r, c in [("bfr", bfr, "tab:blue"), ("fenicsx", fen, "tab:red")]:
        J = np.maximum(np.asarray(r["J"], float), 1e-30)
        ax[0].semilogy(J, marker="o", ms=4, color=c, label=name)
    ax[0].set_xlabel("evaluación del funcional")
    ax[0].set_ylabel("J")
    ax[0].set_title("Descenso del funcional")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend()

    # (2) z_p vs evaluación
    for name, r, c in [("bfr", bfr, "tab:blue"), ("fenicsx", fen, "tab:red")]:
        ax[1].plot(r["zp"], marker="s", ms=4, color=c, label=name)
    ax[1].set_xlabel("evaluación del funcional")
    ax[1].set_ylabel(r"$z_p$ (m)")
    ax[1].set_title("Profundidad del pozo")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    # (3) Q(t) óptimo
    for name, r, c in [("bfr", bfr, "tab:blue"), ("fenicsx", fen, "tab:red")]:
        Qopt = np.asarray(r["x_opt"][:-1], float)
        ax[2].plot(np.arange(len(Qopt)), Qopt, marker="^", ms=5, color=c, label=name)
    ax[2].set_xlabel("paso de tiempo n")
    ax[2].set_ylabel(r"$Q^*(t_n)$")
    ax[2].set_title("Tasa de extracción óptima")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    fig.suptitle("Cruce optimización adjunta: bfr (RBF-FD) vs fenicsx (FEM)")
    fig.tight_layout()
    out = "./data/output/comparison_opt_bfr_fenicsx.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigura guardada en {out}")


if __name__ == "__main__":
    main()
