"""
Postproceso del backend fenicsx: curvas de llegada (BTC) en puntos de control,
análogo a postprocessing/create_btc.py del backend bfr.

A diferencia de bfr —cuyo create_btc compara dos corridas (adr vs mrmt_block)
leídas de disco— aquí se generan las BTC de la propia corrida fenicsx (un solo
modelo), a partir de la historia de C ya en memoria.
"""
import os

import numpy as np


# Mismos puntos de control por defecto que postprocessing/create_btc.py (bfr).
DEFAULT_CONTROL_POINTS = np.array([
    [10, -5],
    [30, -10],
    [60, -15],
    [90, -20],
])


def create_btc(save_dir, nodes, C_hist, dt, control_points=None):
    """
    Extrae C(t) en los nodos más cercanos a los puntos de control y guarda
    `{save_dir}/btc.png`. `C_hist` es (Nt, N) sobre los nodos de G.
    """
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if control_points is None:
        control_points = DEFAULT_CONTROL_POINTS

    C = np.asarray(C_hist)                       # (Nt, N)
    nodes = np.asarray(nodes)
    _, idx = cKDTree(nodes).query(control_points)

    t = np.arange(C.shape[0]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, node_i in enumerate(idx):
        cp, nd = control_points[i], nodes[node_i]
        axes[i].plot(t, C[:, node_i])
        axes[i].set_title(f"P{i+1}  pedido {tuple(cp)} → nodo ({nd[0]:.0f}, {nd[1]:.0f})")
        axes[i].set_xlabel("Tiempo [s]")
        axes[i].set_ylabel("C")
        axes[i].grid(True)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/btc.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[fenicsx] BTC guardada ✔  {path}")
