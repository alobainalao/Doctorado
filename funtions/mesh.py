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
    """
    Espaciado variable para la nube de nodos BFR.

    Refina cerca del pozo (p.pozo) usando p.min_sp como espaciado mínimo.
    Opcionalmente refina también cerca de la fuente de contaminante (p.fnte)
    usando p.min_sp_fnte (si no está definido o es ≥ p.spacing, sin efecto).

    La transición es suave: s = (s_max - s_min) * sqrt(clip(d/Ly, 0, 1)) + s_min
    """
    p = RUNTIME.get()

    min_sp_pozo  = getattr(p, "min_sp",       p.spacing)
    min_sp_fnte  = getattr(p, "min_sp_fnte",  p.spacing)

    if min_sp_pozo >= p.spacing and min_sp_fnte >= p.spacing:
        return np.full(np.atleast_2d(xy).shape[0], p.spacing)

    x  = np.atleast_2d(xy)
    Ly = np.ptp(np.asarray(p.vert)[:, 1])

    # Espaciado desde la distancia al pozo
    s_pozo = p.spacing
    if min_sp_pozo < p.spacing:
        dist_pozo = np.linalg.norm(x - np.asarray(p.pozo), axis=1)
        s_pozo = (p.spacing - min_sp_pozo) * np.sqrt(np.clip(dist_pozo / Ly, 0, 1)) + min_sp_pozo

    # Espaciado desde la distancia a la fuente de contaminante
    s_fnte = p.spacing
    if min_sp_fnte < p.spacing:
        dist_fnte = np.linalg.norm(x - np.asarray(p.fnte), axis=1)
        s_fnte = (p.spacing - min_sp_fnte) * np.sqrt(np.clip(dist_fnte / Ly, 0, 1)) + min_sp_fnte

    # El espaciado final es el mínimo de ambas contribuciones
    s_local = np.minimum(s_pozo, s_fnte)
    return np.clip(s_local, min(min_sp_pozo, min_sp_fnte), p.spacing)


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

    # Forzar nodo en la posición exacta del pozo (análogo a gmsh.embed en FEniCS)
    # para que discrete_delta quede centrado en r=0.
    well = np.asarray(p.pozo, dtype=float)
    if np.linalg.norm(nodes - well, axis=1).min() > 0.5:
        new_idx = len(nodes)
        nodes = np.vstack([nodes, well])
        normals = np.vstack([normals, [[0.0, 0.0]]])
        groups['interior'] = np.append(groups['interior'], new_idx)

    idx_p = np.argmin(np.linalg.norm(nodes - p.pozo, axis=1))

    if p.plot_grid:
        plot_mesh(nodes, groups, smp, title='Discretización del dominio')

    print(nodes.shape)
    return nodes, groups, normals, smp, nodes[idx_p]

