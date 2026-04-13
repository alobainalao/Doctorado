import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from funtions.runtime import RUNTIME
p = RUNTIME.params



# ==========================================================
# 1. Leer datos
# ==========================================================
def load_simulation_data(path):
    data = np.load(path)
    return data["C"], data["nodes"]


# ==========================================================
# 2. Encontrar nodos más cercanos a puntos de control
# ==========================================================
def get_nearest_nodes(nodes, control_points):
    tree = cKDTree(nodes)
    _, idx = tree.query(control_points)

    return idx, nodes[idx]


# ==========================================================
# 3. Extraer BTCs (4 x Nt)
# ==========================================================
def extract_btc_matrix(C_hist, node_indices):
    """
    C_hist: (Nt, Nnodes)
    node_indices: lista de 4 índices
    """
    btc = np.array([C_hist[:,0, i] for i in node_indices])
    return btc  # shape (4, Nt)


# ==========================================================
# 4. Graficar comparación
# ==========================================================
def plot_btc_comparison(btc_adr, btc_mrmt, dt, labels=None):
    p = RUNTIME.get()
    Nt = btc_adr.shape[1]
    t = np.arange(Nt) * dt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t, btc_adr[i], label="ADR")
        axes[i].plot(t, btc_mrmt[i], "--", label="MRMT")
        
        if labels:
            axes[i].set_title(labels[i])
        else:
            axes[i].set_title(f"Punto {i+1}")

        axes[i].set_xlabel("Tiempo")
        axes[i].set_ylabel("C")
        axes[i].grid()
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"{p.save_video}/btc.png")


# ==========================================================
# 5. Pipeline completo
# ==========================================================
def run_btc_analysis(path_adr, path_mrmt, control_points):

    # --- cargar ---
    adr, nodes_adr = load_simulation_data(path_adr)
    mrmt, nodes_mrmt = load_simulation_data(path_mrmt)

    # --- puntos originales (NO se modifican) ---
    control_points_ref = control_points.copy()

    # --- proyección en ADR ---
    idx_adr, cp_adr = get_nearest_nodes(nodes_adr, control_points_ref)

    # --- proyección en MRMT ---
    idx_mrmt, cp_mrmt = get_nearest_nodes(nodes_mrmt, control_points_ref)

    # --- validar coherencia ---
    tol = 1e-2  # ajusta según tu malla

    import matplotlib.pyplot as plt

    plt.scatter(nodes_adr[:,0], nodes_adr[:,1], c='b', s=1)
    plt.scatter(nodes_mrmt[:,0], nodes_mrmt[:,1], c='c', s=1)
    plt.scatter(cp_adr[:,0], cp_adr[:,1], c='r')
    plt.scatter(cp_adr[:,0], cp_adr[:,1], c='k')

    if not np.allclose(cp_adr, cp_mrmt, atol=tol):
        print("⚠️ Advertencia: los puntos no coinciden entre mallas")
        print("ADR:\n", cp_adr)
        print("MRMT:\n", cp_mrmt)
    else:
        print("✔ Puntos consistentes entre mallas")

    # --- extraer BTCs ---
    btc_adr = extract_btc_matrix(adr, idx_adr)
    btc_mrmt = extract_btc_matrix(mrmt, idx_mrmt)


    # --- plot ---
    plot_btc_comparison(
        btc_adr,
        btc_mrmt,
        dt= p.dt,
        labels=[f"P{i+1}" for i in range(4)]
    )


def create_btc():
    # === puntos de control (elige estratégicamente) ===
    control_points = np.array([
        [10, -5],
        [30, -10],
        [60, -15],
        [90, -20]
    ])

    # === rutas ===
    import os

    base_path = os.path.dirname(p.save_data)  # quita el model

    path_adr = os.path.join(base_path, "adr", "simulation_results.npz")
    path_mrmt = os.path.join(base_path, "mrmt_block", "simulation_results.npz")

    # === ejecutar ===
    run_btc_analysis(
        path_adr,
        path_mrmt,
        control_points
    )



