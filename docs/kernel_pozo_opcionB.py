"""
Opción B: Pre-malla paramétrica para optimización del pozo.

En lugar de forzar UN nodo en p.pozo (z_p = -700m fijo), se insertan nodos
en toda la franja vertical x = pozo[0], z ∈ [zp_min, zp_max] con paso min_sp.
Esto garantiza que para cualquier z_p en el rango de búsqueda siempre haya
un nodo BFR a distancia < min_sp/2 ≈ 11 m del pozo.

Ventaja: Fix 2 aproximado sin remallado durante la optimización.
Riesgo: nodos alineados en x = 3700 m → estenciles RBF-FD mal condicionados
        si todos los vecinos tienen la misma abscisa. Verificar con
        diag_velocity_frames.py o con la norma del residual del sistema lineal.

Número de nodos extra: (zp_max - zp_min) / min_sp ≈ (1200) / 22 ≈ 55 nodos
(solo los que quedan dentro del dominio).

Implementación propuesta para build_nodes_var() en funtions/mesh.py:
"""

import numpy as np
from shapely.geometry import Polygon, Point


def _well_column_nodes(p):
    """
    Devuelve la cuadrícula vertical de nodos potenciales del pozo.
    Solo incluye puntos que caen dentro del polígono del dominio.
    """
    poly = Polygon(p.vert)
    x_p = float(p.pozo[0])
    z_grid = np.arange(p.zp_min, p.zp_max + p.min_sp, p.min_sp)
    candidates = np.column_stack([np.full_like(z_grid, x_p), z_grid])
    inside = np.array([poly.contains(Point(x, z)) for x, z in candidates])
    return candidates[inside]


def build_nodes_var_optim():
    """
    Variante de build_nodes_var para run_type=optimization:
    inserta la columna vertical de nodos del pozo en lugar de solo el punto
    z_p = -700 m.

    Cambios respecto a build_nodes_var actual:
    - Reemplaza el bloque de nodo único por _well_column_nodes
    - Agrega todos los nodos de la columna que no tengan un vecino < min_sp/2
    """
    from funtions.runtime import RUNTIME
    from funtions.mesh import _boundary_config
    from rbf.pde.nodes import poisson_disc_nodes
    from funtions.mesh import spacing_func

    p = RUNTIME.get()
    smp, boundary_groups, boundary_groups_with_ghosts = _boundary_config()

    nodes, groups, normals = poisson_disc_nodes(
        spacing_func,
        (p.vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    # Insertar columna vertical de nodos para toda la trayectoria z_p
    well_col = _well_column_nodes(p)
    added = 0
    for wpt in well_col:
        dists = np.linalg.norm(nodes - wpt, axis=1)
        if dists.min() > p.min_sp / 2:
            new_idx = len(nodes)
            nodes = np.vstack([nodes, wpt])
            normals = np.vstack([normals, [[0.0, 0.0]]])
            groups['interior'] = np.append(groups['interior'], new_idx)
            added += 1

    print(f"  [well_column] {added} nodos extra insertados en x={p.pozo[0]}")

    idx_p = np.argmin(np.linalg.norm(nodes - np.asarray(p.pozo), axis=1))
    return nodes, groups, normals, smp, nodes[idx_p]


# ── Integración propuesta ────────────────────────────────────────────────────
#
# En funtions/mesh.py, modificar build_nodes_var:
#
#   if getattr(p, 'run_type', 'standard') == 'optimization':
#       # columna de nodos cubre todo el rango z_p
#       well_pts = _well_column_nodes(p)
#       threshold = p.min_sp / 2
#   else:
#       # solo el punto inicial del pozo (comportamiento actual Fix 2)
#       well_pts = np.asarray(p.pozo, dtype=float).reshape(1, 2)
#       threshold = 0.5
#
#   for wpt in well_pts:
#       if np.linalg.norm(nodes - wpt, axis=1).min() > threshold:
#           new_idx = len(nodes)
#           nodes = np.vstack([nodes, wpt])
#           normals = np.vstack([normals, [[0.0, 0.0]]])
#           groups['interior'] = np.append(groups['interior'], new_idx)
#
# ── Validación recomendada antes de activar ──────────────────────────────────
#
# 1. Verificar que los nodos en x=3700 no degraden el número de condición:
#       python scripts/diag_velocity_frames.py  (con p.run_type=optimization)
#
# 2. Comparar H en el pozo para z_p = -700 m (debería ser idéntico a Fix 2)
#    y para z_p = -900 m (debería mostrar mejora respecto a Fix 1 solo).
#
# 3. Verificar que el gradiente ∂J/∂z_p sea suave (no saltos al pasar por
#    un nodo de la columna) usando test_gradient_checkpoint.py.
