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


def _s(d, s_min, s_max, d_far):
    """Espaciado monotonamente creciente con la distancia d."""
    return s_min + (s_max - s_min) * np.sqrt(np.clip(d / d_far, 0.0, 1.0))


def _well_vertical_cone(pts, pozo, s_min, s_max, r0=150.0, r1=500.0, z_top=200.0):
    """
    Cono de succión vertical: máx densidad en el eje x=x_pozo para z>z_pozo,
    expandiéndose lateralmente al subir. El cono conecta con la pluma en z_top.

    Distancia efectiva:
      - Eje (dx=0, dz>0): 0  → s_min
      - Dentro del cono: dx escalado a [0, r0]
      - Fuera del cono:  dx - w + r0  (exceso sobre el borde)
      - Por debajo del pozo: distancia radial al pozo
    """
    pts = np.atleast_2d(pts)
    xp, zp = float(pozo[0]), float(pozo[1])
    dz = pts[:, 1] - zp
    dx = np.abs(pts[:, 0] - xp)

    # Ancho del cono a altura dz sobre el pozo
    dz_norm = np.clip(dz / (z_top - zp), 0.0, 1.0)
    w = r0 + (r1 - r0) * dz_norm

    d_radial = np.linalg.norm(pts - np.array([xp, zp]), axis=1)
    d_inside = dx * r0 / np.maximum(w, 1e-10)
    d_outside = dx - w + r0

    d_cone = np.where(dz <= 0, d_radial,
                      np.where(dx < w, d_inside, d_outside))

    return _s(d_cone, s_min, s_max, r1 + 400.0)


def spacing_func(xy):
    """
    Función de espaciado estratégico de la malla con 4 contribuciones:

    1. Cono horizontal de la pluma  — eje = streamline Darcy, ancho creciente.
    2. Vecindad radial del pozo     — mínimo en el pozo, aumenta al alejarse.
    3. Cono vertical de succión     — desde el pozo hacia arriba, anisótropo.
    4. Refinamiento en fronteras    — dentro de ~200 m de cualquier frontera.

    En cada punto prevalece el espaciado más fino de los cuatro.
    Si min_sp >= spacing, la malla es uniforme (comportamiento conservado).
    """
    p = RUNTIME.get()
    if p.min_sp >= p.spacing:
        return np.full(np.atleast_2d(xy).shape[0], p.spacing)

    x    = np.atleast_2d(xy)
    pozo = np.asarray(p.pozo, float)
    fnte = np.asarray(p.fnte, float)
    vert = np.asarray(p.vert, float)
    s_min, s_max = float(p.min_sp), float(p.spacing)

    # ── 1. Cono horizontal de la pluma ────────────────────────────────────
    from funtions.plume import get_plume_streamline, plume_cone_dist
    plume_pts = get_plume_streamline()
    if plume_pts is not None and len(plume_pts) >= 2:
        d_pluma = plume_cone_dist(x, plume_pts, w0=60.0, w1=500.0)
    else:
        d_pluma = np.minimum(np.linalg.norm(x - fnte, axis=1),
                             np.linalg.norm(x - pozo, axis=1))
    s_plume = _s(d_pluma, s_min, s_max, 500.0)

    # ── 2. Vecindad radial del pozo ───────────────────────────────────────
    d_pozo = np.linalg.norm(x - pozo, axis=1)
    s_pozo_r = _s(d_pozo, s_min, s_max, 600.0)

    # ── 3. Cono vertical de succión (desde el pozo hacia arriba) ──────────
    # z_top: nivel de la pluma (z de fnte); el cono conecta ambas zonas.
    s_pozo_v = _well_vertical_cone(x, pozo, s_min, s_max,
                                   r0=150.0, r1=500.0,
                                   z_top=float(fnte[1]) + 50.0)

    # ── 4. Fronteras (Shapely, igual que Lspacing) ────────────────────────
    d_borde = np.array([
        Polygon(vert).boundary.distance(Point(px, pz)) for px, pz in x
    ])
    s_borde = _s(d_borde, s_min, s_max, 200.0)

    return np.minimum.reduce([s_plume, s_pozo_r, s_pozo_v, s_borde])


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

