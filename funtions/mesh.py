import numpy as np
from shapely.geometry import Polygon, Point
from rbf.pde.nodes import poisson_disc_nodes

from config import vert, fnte, pozo, spacing, min_sp, plot_grid, domain
from view.plot import plot_mesh

# ---------------------------
# Espaciado adaptativo
# ---------------------------
def Lspacing(x, s_M=100, min_s=spacing):
    poly = Polygon(vert)

    x = np.atleast_2d(x)

    dist_borde = np.array([poly.boundary.distance(Point(px, py)) for px, py in x])
    dist_fnte = np.linalg.norm(x - fnte, axis=1)
    dist_pozo = np.linalg.norm(x - pozo, axis=1)

    d = np.minimum.reduce([dist_borde, dist_pozo, dist_fnte])

    Ly = np.ptp(vert[:, 1])

    spacing_local = (s_M - min_s) * np.sqrt(np.clip(d / Ly, 0, 1)) + min_s

    return np.clip(spacing_local, min_s, s_M)


def spacing_func(xy):
    return Lspacing(np.atleast_2d(xy), s_M=spacing, min_s=min_sp)


# ---------------------------
# Generador de nodos
# ---------------------------
def _boundary_config():
    smp = [[i, (i + 1) % len(vert)] for i in range(len(vert))]

    BOUNDARY_GROUPS = {
        "real": {
            'inlet': [2],
            'outlet': [4],
            'wall': [0, 1, 3]
        },
        "benchmark": {
            'inlet': [1],
            'outlet': [3],
            'wall': [0, 2]
        }
    }

    try:
        boundary_groups = BOUNDARY_GROUPS[domain]
    except KeyError:
        raise ValueError(f"Dominio inválido: {domain}")


    boundary_groups_with_ghosts = ['inlet', 'outlet', 'wall']

    return smp, boundary_groups, boundary_groups_with_ghosts


def build_nodes(sp_u=spacing):
    smp, boundary_groups, boundary_groups_with_ghosts = _boundary_config()

    nodes, groups, normals = poisson_disc_nodes(
        sp_u,
        (vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    idx_p = np.argmin(np.linalg.norm(nodes - pozo, axis=1))

    return nodes, groups, normals, smp, nodes[idx_p]


def build_nodes_var():
    smp, boundary_groups, boundary_groups_with_ghosts = _boundary_config()

    nodes, groups, normals = poisson_disc_nodes(
        spacing_func,
        (vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    idx_p = np.argmin(np.linalg.norm(nodes - pozo, axis=1))

    if plot_grid:
        plot_mesh(nodes, groups, smp, title='Discretización del dominio')

    return nodes, groups, normals, smp, nodes[idx_p]

