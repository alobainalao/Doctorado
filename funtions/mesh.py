import numpy as np
from shapely.geometry import Polygon, Point
from rbf.pde.nodes import poisson_disc_nodes
from funtions.runtime import RUNTIME
from view.plot import plot_mesh

# ---------------------------
# Espaciado adaptativo
# ---------------------------
def Lspacing(x, s_M=100, min_s=None):

    p = RUNTIME.get()

    if min_s is None:
        min_s = p.spacing

    poly = Polygon(p.vert)

    x = np.atleast_2d(x)

    dist_borde = np.array([poly.boundary.distance(Point(px, py)) for px, py in x])
    dist_fnte = np.linalg.norm(x - p.fnte, axis=1)
    dist_pozo = np.linalg.norm(x - p.pozo, axis=1)

    d = np.minimum.reduce([dist_borde, dist_pozo, dist_fnte])

    Ly = np.ptp(p.vert[:, 1])

    spacing_local = (s_M - min_s) * np.sqrt(np.clip(d / Ly, 0, 1)) + min_s

    return np.clip(spacing_local, min_s, s_M)


def spacing_func(xy):
    p = RUNTIME.get()
    return Lspacing(np.atleast_2d(xy), s_M=p.spacing, min_s=p.min_sp)


# ---------------------------
# Generador de nodos
# ---------------------------
def _boundary_config():
    p = RUNTIME.get()
    smp = [[i, (i + 1) % len(p.vert)] for i in range(len(p.vert))]

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
        boundary_groups = BOUNDARY_GROUPS[p.domain]
    except KeyError:
        raise ValueError(f"Dominio inválido: {p.domain}")


    boundary_groups_with_ghosts = ['inlet', 'outlet', 'wall']

    return smp, boundary_groups, boundary_groups_with_ghosts


def build_nodes(sp_u=None):
    p = RUNTIME.get()
    if min_s is None:
        min_s = p.spacing

    smp, boundary_groups, boundary_groups_with_ghosts = _boundary_config()

    nodes, groups, normals = poisson_disc_nodes(
        sp_u,
        (p.vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    idx_p = np.argmin(np.linalg.norm(nodes - p.pozo, axis=1))

    return nodes, groups, normals, smp, nodes[idx_p]


def build_nodes_var():
    p = RUNTIME.get()
    smp, boundary_groups, boundary_groups_with_ghosts = _boundary_config()

    nodes, groups, normals = poisson_disc_nodes(
        spacing_func,
        (p.vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    idx_p = np.argmin(np.linalg.norm(nodes - p.pozo, axis=1))

    if p.plot_grid:
        plot_mesh(nodes, groups, smp, title='Discretización del dominio')

    print(nodes.shape)
    return nodes, groups, normals, smp, nodes[idx_p]

