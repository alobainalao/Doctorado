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


def _dist_to_polyline(pts, polyline):
    """
    Distancia de cada punto en `pts` (N,2) a la polilínea más cercana.
    La polilínea es un array (M,2) de vértices consecutivos.
    """
    if len(polyline) < 2:
        return np.linalg.norm(pts - polyline[0], axis=1)
    ab_vecs = np.diff(polyline, axis=0)          # (M-1, 2)
    ab2 = np.einsum('ij,ij->i', ab_vecs, ab_vecs)  # (M-1,)
    # Para cada punto, distancia a cada segmento
    pts = np.atleast_2d(pts)
    dists = np.empty((len(pts), len(ab_vecs)))
    for k, (a, ab, l2) in enumerate(zip(polyline[:-1], ab_vecs, ab2)):
        if l2 < 1e-20:
            dists[:, k] = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(((pts - a) @ ab) / l2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            dists[:, k] = np.linalg.norm(pts - proj, axis=1)
    return dists.min(axis=1)


def spacing_func(xy):
    p = RUNTIME.get()
    if p.min_sp >= p.spacing:
        return np.full(np.atleast_2d(xy).shape[0], p.spacing)
    x = np.atleast_2d(xy)
    pozo = np.asarray(p.pozo, float)
    fnte = np.asarray(p.fnte, float)
    Ly = np.ptp(np.asarray(p.vert)[:, 1])

    # Intentar usar la streamline real de la pluma (calculada con Darcy).
    # Si el pkl aún no existe (primera generación), usa solo pozo y fuente.
    from funtions.plume import get_plume_streamline
    plume_pts = get_plume_streamline()

    dist_pozo = np.linalg.norm(x - pozo, axis=1)
    dist_fnte = np.linalg.norm(x - fnte, axis=1)

    if plume_pts is not None and len(plume_pts) >= 2:
        dist_pluma = _dist_to_polyline(x, plume_pts)
        d = np.minimum.reduce([dist_pozo, dist_fnte, dist_pluma])
    else:
        d = np.minimum(dist_pozo, dist_fnte)

    s_local = (p.spacing - p.min_sp) * np.sqrt(np.clip(d / Ly, 0, 1)) + p.min_sp
    return np.clip(s_local, p.min_sp, p.spacing)


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

