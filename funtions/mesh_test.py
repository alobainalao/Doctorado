
"""
Generación de nodos adaptativos RBF-FD.

Caché persistente:
    spacing_field_cache.npz

El campo normalizado q(x,y) pertenece a [0, 1]:
    spacing(x,y) = min_sp + q(x,y) * (spacing - min_sp)

Cambiar min_sp o spacing NO obliga a reconstruir q.
Cambiar la geometría, la streamline o grid_step invalida la caché.

Nota:
    diagnose_triangulation() detecta triángulos alargados.
    No garantiza su eliminación automática.
"""

from pathlib import Path
import hashlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
from rbf.pde.nodes import poisson_disc_nodes

from funtions.runtime import RUNTIME
from view.plot import plot_mesh


# ============================================================
# CONFIGURACIÓN
# ============================================================

CACHE_FILE = Path(__file__).with_name("spacing_field_cache.npz")

# Paso de la cuadrícula usada para precalcular q(x,y).
# No se llama h para evitar conflictos con variables del modelo.
DEFAULT_GRID_STEP = 20.0

PLUME_W0 = 60.0
PLUME_W1 = 500.0
PLUME_DEPTH_BELOW = 150.0
PLUME_D_FAR = 500.0

WELL_R0 = 150.0
WELL_R1 = 500.0
WELL_D_FAR = 900.0
WELL_Z_OFFSET = 50.0

BOUNDARY_D_FAR = 200.0

# Radios de los 4 niveles de refinamiento.
# Nivel 1 (min_sp):        círculo de radio CIRCLE_R centrado en pozo y fuente.
# Nivel 2 (s_cone):        transición desde CIRCLE_R hasta CONE_R.
# Nivel 3 (min_sp_bnd):    franja de frontera (gestionada por spacing_func_cached).
# Nivel 4 (spacing):       campo lejano.
CIRCLE_R = 100.0
CONE_R   = 500.0

# Tasa máxima de cambio del campo normalizado q por metro.
# Limita la gradación entre zonas finas y gruesas para evitar
# triángulos elongados en la transición.  Rango útil: 0.003–0.015.
MAX_GRADATION = 0.005

# Caché en memoria para no abrir el archivo en cada evaluación.
_SPACING_CACHE = {}


# ============================================================
# FUNCIONES DE ESPACIAMIENTO
# ============================================================

def Lspacing(x, s_M=100, min_s=None):
    """Función de espaciamiento original, conservada por compatibilidad."""
    p = RUNTIME.get()

    if min_s is None:
        min_s = p.spacing

    polygon = Polygon(p.vert)
    x = np.atleast_2d(np.asarray(x, dtype=float))

    d_boundary = np.array([
        polygon.boundary.distance(Point(px, py))
        for px, py in x
    ])

    d_source = np.linalg.norm(x - p.fnte, axis=1)
    d_well = np.linalg.norm(x - p.pozo, axis=1)

    d = np.minimum.reduce([d_boundary, d_well, d_source])

    Ly = np.ptp(np.asarray(p.vert)[:, 1])
    if Ly <= 0:
        raise ValueError("La extensión vertical del dominio debe ser positiva.")

    result = (s_M - min_s) * np.sqrt(np.clip(d / Ly, 0, 1)) + min_s

    return np.clip(result, min_s, s_M)


def _enforce_gradation(q, x_grid, y_grid, max_grad=MAX_GRADATION):
    """
    Limita la tasa de cambio de q(x,y) ∈ [0,1] propagando desde las celdas
    más finas (q mínimo) hacia afuera mediante Dijkstra.

    Garantiza: q[b] ≤ q[a] + max_grad * |a-b|  ∀ vecinos a,b

    Con max_grad = 0.005 /m y grid_step = 20 m, la transición de q=0 a q=1
    requiere al menos 200 m, lo que produce triángulos bien formados en zonas
    de cambio entre refinado fino y grueso.
    """
    import heapq

    nx, ny = q.shape
    result = q.copy().astype(np.float64)

    dx = np.diff(x_grid)  # (nx-1,)
    dy = np.diff(y_grid)  # (ny-1,)

    heap = []
    for i in range(nx):
        for j in range(ny):
            heapq.heappush(heap, (result[i, j], i, j))

    visited = np.zeros((nx, ny), dtype=bool)

    OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        q_cur, i, j = heapq.heappop(heap)
        if visited[i, j]:
            continue
        visited[i, j] = True

        for di, dj in OFFSETS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or visited[ni, nj]:
                continue

            ci = i if di >= 0 else ni
            cj = j if dj >= 0 else nj
            step_x = dx[ci] if 0 <= ci < len(dx) else 0.0
            step_y = dy[cj] if 0 <= cj < len(dy) else 0.0
            dist = np.sqrt(
                (di != 0) * step_x ** 2 + (dj != 0) * step_y ** 2
            )

            q_allowed = q_cur + max_grad * dist
            if q_allowed < result[ni, nj]:
                result[ni, nj] = q_allowed
                heapq.heappush(heap, (q_allowed, ni, nj))

    return np.clip(result, 0.0, 1.0)


def _s(d, s_min, s_max, d_far):
    """Espaciamiento creciente con la distancia."""
    return s_min + (s_max - s_min) * np.sqrt(
        np.clip(np.asarray(d, dtype=float) / d_far, 0.0, 1.0)
    )


def _well_vertical_cone(
    pts,
    pozo,
    s_min,
    s_max,
    r0=WELL_R0,
    r1=WELL_R1,
    z_top=200.0,
):
    """Cono de refinamiento por encima del pozo."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))

    xp, zp = float(pozo[0]), float(pozo[1])
    dz = pts[:, 1] - zp
    dx = np.abs(pts[:, 0] - xp)

    denominator = z_top - zp

    if abs(denominator) < 1e-12:
        dz_norm = np.zeros_like(dz)
    else:
        dz_norm = np.clip(dz / denominator, 0.0, 1.0)

    width = r0 + (r1 - r0) * dz_norm

    d_inside = dx * r0 / np.maximum(width, 1e-10)
    d_outside = dx - width + r0
    d_cone = np.where(dx < width, d_inside, d_outside)

    # El cono aplica desde r0 por debajo del pozo hasta z=-150 m.
    active = (dz >= -r0) & (pts[:, 1] <= -150.0)
    d_eff = np.where(active, d_cone, r1 + 400.0)

    return _s(d_eff, s_min, s_max, WELL_D_FAR)


def _get_plume_geometry():
    """Obtiene la streamline actual, si existe."""
    from funtions.plume import get_plume_streamline

    plume_pts = get_plume_streamline()

    if plume_pts is None:
        return None

    plume_pts = np.asarray(plume_pts, dtype=float)

    if (
        plume_pts.ndim != 2
        or plume_pts.shape[0] < 2
        or plume_pts.shape[1] != 2
    ):
        return None

    return plume_pts


def _spacing_field_raw(xy, s_min, s_max, plume_pts=None):
    """
    Campo de espaciamiento con 4 niveles de refinamiento.

    Se llama con s_min=0, s_max=1 para construir el campo normalizado q(x,y).
    Los niveles intermedios (s_cone, min_sp_boundary) se expresan como fracciones
    derivadas de los parámetros del modelo, de modo que el caché debe invalidarse
    si cambian min_sp, s_cone, min_sp_boundary o spacing.

    Niveles:
      q = 0          → min_sp          (círculo CIRCLE_R entorno a pozo y fuente)
      q = q_cone     → s_cone          (cono hasta CONE_R)
      q = q_bnd      → min_sp_boundary (franja de fronteras)
      q = 1          → spacing         (campo lejano)
    """
    p = RUNTIME.get()

    x = np.atleast_2d(np.asarray(xy, dtype=float))
    if x.shape[1] != 2:
        raise ValueError("Las coordenadas deben tener forma (N, 2).")

    pozo = np.asarray(p.pozo, dtype=float)
    fnte = np.asarray(p.fnte, dtype=float)

    # Niveles absolutos del modelo (para calcular fracciones en espacio q)
    p_min  = float(getattr(p, 'min_sp', 4.0))
    p_cone = float(getattr(p, 's_cone', 16.0))
    p_bnd  = float(getattr(p, 'min_sp_boundary', 30.0))
    p_max  = float(getattr(p, 'spacing', 60.0))
    denom  = max(p_max - p_min, 1e-12)

    R_circle = float(getattr(p, 'r_circle', CIRCLE_R))
    R_cone   = float(getattr(p, 'r_cone', CONE_R))

    q_cone = (p_cone - p_min) / denom   # ≈ 0.214 con (4, 16, 60)
    q_bnd  = (p_bnd  - p_min) / denom   # ≈ 0.464 con (4, 30, 60)

    # Distancia al centro más cercano (pozo o fuente)
    d_well = np.linalg.norm(x - pozo, axis=1)
    d_fnte = np.linalg.norm(x - fnte, axis=1)
    d_ctr  = np.minimum(d_well, d_fnte)

    # Campo de zona central
    in_circle = d_ctr < R_circle
    in_cone   = (d_ctr >= R_circle) & (d_ctr < R_cone)

    t_cone = np.clip((d_ctr - R_circle) / max(R_cone - R_circle, 1.0), 0.0, 1.0)
    t_far  = np.clip((d_ctr - R_cone)   / max(PLUME_D_FAR, 1.0),       0.0, 1.0)

    q_center = np.where(
        in_circle, 0.0,
        np.where(
            in_cone,
            q_cone * np.sqrt(t_cone),
            q_cone + (1.0 - q_cone) * np.sqrt(t_far),
        )
    )

    # Campo de frontera: q_bnd en la franja, 1.0 lejos
    d_boundary = _dist_to_boundary(x, p.vert)
    t_bnd      = np.clip(d_boundary / BOUNDARY_D_FAR, 0.0, 1.0)
    q_boundary = q_bnd + (1.0 - q_bnd) * np.sqrt(t_bnd)

    result = np.minimum(q_center, q_boundary)

    return np.clip(result, 0.0, 1.0)


def spacing_func(xy):
    """Versión directa, sin caché, para comparar o depurar."""
    p = RUNTIME.get()

    s_min = float(p.min_sp)
    s_max = float(p.spacing)

    x = np.atleast_2d(np.asarray(xy, dtype=float))

    if s_min >= s_max:
        return np.full(x.shape[0], s_max, dtype=float)

    return _spacing_field_raw(x, s_min, s_max)


# ============================================================
# CAMPO NORMALIZADO Y CACHÉ PERSISTENTE
# ============================================================

def _configuration_signature(grid_step, plume_pts):
    """
    Firma de la geometría que genera el campo.

    min_sp y spacing se excluyen intencionalmente.
    """
    p = RUNTIME.get()
    digest = hashlib.sha256()

    def add_array(name, value):
        arr = np.ascontiguousarray(
            np.asarray(value, dtype=np.float64)
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())

    add_array("vertices", p.vert)
    add_array("well", p.pozo)
    add_array("source", p.fnte)
    add_array("grid_step", [grid_step])

    digest.update(str(getattr(p, "domain", "")).encode("utf-8"))

    if plume_pts is None:
        digest.update(b"no_plume_streamline")
    else:
        add_array("plume_streamline", plume_pts)

    add_array(
        "constants",
        [
            PLUME_W0,
            PLUME_W1,
            PLUME_DEPTH_BELOW,
            PLUME_D_FAR,
            WELL_R0,
            WELL_R1,
            WELL_D_FAR,
            WELL_Z_OFFSET,
            BOUNDARY_D_FAR,
            MAX_GRADATION,
            CIRCLE_R,
            CONE_R,
        ],
    )

    # Los niveles de espaciado forman parte del campo normalizado q(x,y):
    # cambiar min_sp, s_cone, min_sp_boundary o spacing invalida el caché.
    add_array("levels", [
        float(getattr(p, 'min_sp', 4.0)),
        float(getattr(p, 's_cone', 16.0)),
        float(getattr(p, 'min_sp_boundary', 30.0)),
        float(getattr(p, 'spacing', 60.0)),
        float(getattr(p, 'r_circle', CIRCLE_R)),
        float(getattr(p, 'r_cone', CONE_R)),
    ])

    return digest.hexdigest()


def _make_regular_grid(vertices, grid_step):
    """Construye la cuadrícula que almacena el campo normalizado."""
    if grid_step <= 0:
        raise ValueError("grid_step debe ser mayor que cero.")

    vertices = np.asarray(vertices, dtype=float)

    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)

    x_grid = np.arange(xmin, xmax + grid_step, grid_step)
    y_grid = np.arange(ymin, ymax + grid_step, grid_step)

    if x_grid[-1] < xmax:
        x_grid = np.append(x_grid, xmax)

    if y_grid[-1] < ymax:
        y_grid = np.append(y_grid, ymax)

    return x_grid, y_grid


def build_spacing_cache(
    grid_step=DEFAULT_GRID_STEP,
    cache_file=CACHE_FILE,
):
    """Calcula q(x,y) una vez y guarda el resultado en disco."""
    p = RUNTIME.get()

    cache_file = Path(cache_file)
    plume_pts = _get_plume_geometry()

    signature = _configuration_signature(grid_step, plume_pts)

    x_grid, y_grid = _make_regular_grid(p.vert, grid_step)

    xx, yy = np.meshgrid(
        x_grid,
        y_grid,
        indexing="ij",
    )

    points = np.column_stack((xx.ravel(), yy.ravel()))

    # Campo normalizado q(x,y): no depende de min_sp ni de spacing.
    q = _spacing_field_raw(
        points,
        0.0,
        1.0,
        plume_pts=plume_pts,
    )

    q = np.clip(q.reshape(xx.shape), 0.0, 1.0)

    # Control de gradación: suaviza la transición entre zonas para evitar
    # triángulos elongados en poisson_disc_nodes.
    q = _enforce_gradation(q, x_grid, y_grid)

    # Campo de distancia a la frontera d_bnd(x,y): geométrico, independiente
    # de los valores de spacing. Se usa para aplicar min_sp_boundary en runtime.
    d_bnd = _dist_to_boundary(points, p.vert).reshape(xx.shape)

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_file,
        x_grid=x_grid,
        y_grid=y_grid,
        q=q,
        d_bnd=d_bnd,
        signature=np.asarray(signature),
        grid_step=np.asarray(float(grid_step)),
    )

    print(f"Campo normalizado calculado y guardado: {cache_file}")

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "q": q,
        "d_bnd": d_bnd,
        "signature": signature,
    }


def load_or_build_spacing_cache(
    grid_step=DEFAULT_GRID_STEP,
    cache_file=CACHE_FILE,
    force_rebuild=False,
):
    """
    Carga el campo existente o lo calcula si no existe/no es compatible.
    """
    cache_file = Path(cache_file)
    plume_pts = _get_plume_geometry()

    expected_signature = _configuration_signature(
        grid_step,
        plume_pts,
    )

    if cache_file.exists() and not force_rebuild:
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                stored_signature = str(data["signature"].item())
                stored_step = float(data["grid_step"].item())

                if (
                    stored_signature == expected_signature
                    and np.isclose(stored_step, grid_step)
                ):
                    print(f"Cargando campo desde caché: {cache_file}")

                    result = {
                        "x_grid": data["x_grid"].copy(),
                        "y_grid": data["y_grid"].copy(),
                        "q": data["q"].copy(),
                        "signature": stored_signature,
                    }
                    if "d_bnd" in data:
                        result["d_bnd"] = data["d_bnd"].copy()
                    return result

                print(
                    "La geometría o grid_step cambió. "
                    "Se reconstruirá el campo."
                )

        except (OSError, ValueError, KeyError) as exc:
            print(f"Caché inválida ({exc}); se reconstruirá.")

    return build_spacing_cache(
        grid_step=grid_step,
        cache_file=cache_file,
    )


def prepare_spacing_cache(
    grid_step=DEFAULT_GRID_STEP,
    cache_file=CACHE_FILE,
    force_rebuild=False,
):
    """Carga o genera la caché y prepara su interpolador."""
    cache = load_or_build_spacing_cache(
        grid_step=grid_step,
        cache_file=cache_file,
        force_rebuild=force_rebuild,
    )

    interpolator = RegularGridInterpolator(
        (cache["x_grid"], cache["y_grid"]),
        cache["q"],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    _SPACING_CACHE["cache"] = cache
    _SPACING_CACHE["interpolator"] = interpolator
    _SPACING_CACHE["grid_step"] = float(grid_step)
    _SPACING_CACHE["cache_file"] = str(Path(cache_file).resolve())

    if "d_bnd" in cache:
        _SPACING_CACHE["d_bnd_interpolator"] = RegularGridInterpolator(
            (cache["x_grid"], cache["y_grid"]),
            cache["d_bnd"],
            method="linear",
            bounds_error=False,
            fill_value=0.0,  # fuera del grid → asumir en frontera (conservador)
        )

    return cache


def clear_spacing_cache_memory():
    """Vacía la caché en memoria, pero conserva el archivo del disco."""
    _SPACING_CACHE.clear()


def _dist_to_boundary(xy, vert):
    """Distancia mínima de cada punto a los segmentos del polígono de dominio."""
    xy = np.atleast_2d(xy)
    vert = np.asarray(vert, dtype=float)
    n = len(vert)
    dist = np.full(len(xy), np.inf)
    for i in range(n):
        a = vert[i]
        b = vert[(i + 1) % n]
        ab = b - a
        l2 = np.dot(ab, ab)
        if l2 < 1e-14:
            d = np.linalg.norm(xy - a, axis=1)
        else:
            t = np.clip(((xy - a) @ ab) / l2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(xy - proj, axis=1)
        dist = np.minimum(dist, d)
    return dist


def spacing_func_cached(
    xy,
    grid_step=DEFAULT_GRID_STEP,
    cache_file=CACHE_FILE,
):
    """
    Evalúa el campo precalculado y aplica los valores actuales de
    p.min_sp y p.spacing sin reconstruir q(x,y).

    Si p define min_sp_boundary, aplica ese mínimo en una franja de
    2*min_sp_boundary metros desde la frontera del dominio.
    """
    p = RUNTIME.get()

    x = np.atleast_2d(np.asarray(xy, dtype=float))

    cache_path = str(Path(cache_file).resolve())

    cache_is_ready = (
        "interpolator" in _SPACING_CACHE
        and _SPACING_CACHE.get("grid_step") == float(grid_step)
        and _SPACING_CACHE.get("cache_file") == cache_path
    )

    if not cache_is_ready:
        prepare_spacing_cache(
            grid_step=grid_step,
            cache_file=cache_file,
        )

    s_min = float(p.min_sp)
    s_max = float(p.spacing)

    if s_min >= s_max:
        return np.full(x.shape[0], s_max, dtype=float)

    q = _SPACING_CACHE["interpolator"](x)
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)

    s = s_min + q * (s_max - s_min)

    # Mínimo de frontera: si p.min_sp_boundary > p.min_sp, los nodos dentro
    # de una franja de 2*min_sp_boundary metros desde la frontera no pueden
    # tener espaciado menor que min_sp_boundary.
    # d_bnd se obtiene del interpolador precalculado (caché); si no existe
    # (caché antigua sin d_bnd), se calcula directamente.
    s_min_bnd = float(getattr(p, "min_sp_boundary", 0.0))
    if s_min_bnd > s_min:
        buffer = 2.0 * s_min_bnd
        if "d_bnd_interpolator" in _SPACING_CACHE:
            dist_bnd = np.asarray(_SPACING_CACHE["d_bnd_interpolator"](x), dtype=float)
        else:
            dist_bnd = _dist_to_boundary(x, p.vert)
        s_floor = np.where(dist_bnd < buffer, s_min_bnd, s_min)
        s = np.maximum(s, s_floor)

    return s


def compare_spacing_functions(
    points,
    grid_step=DEFAULT_GRID_STEP,
):
    """Compara la función original con la aproximación precalculada."""
    points = np.atleast_2d(np.asarray(points, dtype=float))

    exact = spacing_func(points)
    cached = spacing_func_cached(points, grid_step=grid_step)

    error = cached - exact

    report = {
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "exact": exact,
        "cached": cached,
        "error": error,
    }

    print(
        "Comparación de espaciamiento: "
        f"error máximo={report['max_abs_error']:.6g}; "
        f"error medio={report['mean_abs_error']:.6g}"
    )

    return report


# ============================================================
# CONFIGURACIÓN DE FRONTERAS
# ============================================================

def _boundary_config():
    p = RUNTIME.get()

    smp = [
        [i, (i + 1) % len(p.vert)]
        for i in range(len(p.vert))
    ]

    boundary_groups_by_domain = {
        "real": {
            "inlet": [2],
            "outlet": [4],
            "wall": [0, 1, 3],
        },
        "benchmark": {
            "inlet": [1],
            "outlet": [3],
            "wall": [0, 2],
        },
        # Todo el borde con la condición de salida (pfs): frontera libre.
        "cuadrado": {
            "inlet": [],
            "outlet": [0, 1, 2, 3],
            "wall": [],
        },
    }

    try:
        boundary_groups = boundary_groups_by_domain[p.domain]
    except KeyError as exc:
        raise ValueError(
            f"Dominio inválido: {p.domain}"
        ) from exc

    boundary_groups_with_ghosts = [
        "inlet",
        "outlet",
        "wall",
    ]

    return smp, boundary_groups, boundary_groups_with_ghosts


# ============================================================
# SUAVIZADO DE LAPLACE
# ============================================================

def _insert_steiner_points(nodes, groups, normals, domain_poly, max_aspect=8.0):
    """
    Inserta nodos en el circuncentro de triángulos con aspect ratio > max_aspect.

    Equivalente a la corrección q20 de CDT pero sobre la nube BFR.
    Solo inserta si el circuncentro cae dentro del dominio y no está demasiado
    cerca de un nodo existente (distancia mínima = spacing_local/2).
    """
    from scipy.spatial import Delaunay, KDTree

    tri = Delaunay(nodes)
    pts = nodes[tri.simplices]  # (T, 3, 2)

    side_a = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    side_b = np.linalg.norm(pts[:, 0] - pts[:, 2], axis=1)
    side_c = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    sides = np.column_stack([side_a, side_b, side_c])
    longest = np.max(sides, axis=1)
    shortest = np.maximum(np.min(sides, axis=1), 1e-12)
    ar = longest / shortest

    bad = np.where(ar > max_aspect)[0]
    if len(bad) == 0:
        return nodes, groups

    kd = KDTree(nodes)
    new_pts = []

    for t in bad:
        A, B, C = pts[t]
        # Intentar circuncentro; si cae fuera del dominio, usar centroide.
        ax, ay = A; bx, by = B; cx, cy = C
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) > 1e-14:
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
                  (cx**2 + cy**2) * (ay - by)) / D
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
                  (cx**2 + cy**2) * (bx - ax)) / D
            cc = np.array([ux, uy])
            if not domain_poly.contains(Point(cc)):
                cc = np.array([(ax + bx + cx) / 3, (ay + by + cy) / 3])
        else:
            cc = np.array([(ax + bx + cx) / 3, (ay + by + cy) / 3])

        if not domain_poly.contains(Point(cc)):
            continue

        dist, _ = kd.query(cc.reshape(1, -1))
        # Distancia mínima: la mitad del lado corto del triángulo
        if dist[0] < shortest[t] * 0.5:
            continue

        new_pts.append(cc)

    if not new_pts:
        return nodes, groups, normals

    new_arr = np.array(new_pts)
    start = len(nodes)
    nodes = np.vstack([nodes, new_arr])
    normals = np.vstack([normals, np.zeros((len(new_arr), 2))])

    if "interior" not in groups:
        groups["interior"] = np.array([], dtype=int)
    groups["interior"] = np.append(
        groups["interior"],
        np.arange(start, start + len(new_arr), dtype=int),
    )

    print(f"Steiner: {len(new_arr)} nodos insertados (triángulos malos: {len(bad)})")
    return nodes, groups, normals


def _laplacian_smooth(nodes, groups, domain_poly, n_iter=5, omega=0.35):
    """
    Suavizado de Laplace para los nodos interiores.

    Mueve cada nodo interior hacia el promedio de sus vecinos en la
    triangulación de Delaunay.  Los nodos de frontera y ghost no se mueven.
    Cada paso se limita a omega ∈ (0,1] y se rechaza si el candidato cae
    fuera del dominio.

    Reduce la razón de aspecto de los triángulos elongados que aparecen en
    las zonas de transición del campo de espaciamiento, mejorando la
    isotopía de los stencils RBF-FD.
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import Point

    # Nodos fijos: frontera + ghost
    fixed = set()
    for key, idx in groups.items():
        # Las claves de frontera son 'boundary:<grupo>' (antes se comparaba con
        # '<grupo>' y los nodos de frontera se movían hacia dentro).
        if 'ghost' in key or key.startswith('boundary:'):
            fixed.update(idx.tolist())

    for _ in range(n_iter):
        tri = Delaunay(nodes)

        # Construir lista de vecinos
        nbrs = [[] for _ in range(len(nodes))]
        for s in tri.simplices:
            for a in range(3):
                for b in range(3):
                    if a != b:
                        nbrs[s[a]].append(s[b])
        # Deduplicar
        nbrs = [list(set(v)) for v in nbrs]

        new_nodes = nodes.copy()
        for i in range(len(nodes)):
            if i in fixed or not nbrs[i]:
                continue
            centroid = nodes[nbrs[i]].mean(axis=0)
            candidate = nodes[i] + omega * (centroid - nodes[i])
            if domain_poly.contains(Point(candidate)):
                new_nodes[i] = candidate
        nodes = new_nodes

    return nodes


# ============================================================
# GENERACIÓN DE NODOS
# ============================================================

def build_nodes(sp_u=None):
    """Genera nodos con espaciamiento constante o callback suministrado."""
    p = RUNTIME.get()

    smp, boundary_groups, boundary_groups_with_ghosts = (
        _boundary_config()
    )

    if sp_u is None:
        sp_u = float(p.spacing)

    nodes, groups, normals = poisson_disc_nodes(
        sp_u,
        (p.vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts,
    )

    well = np.asarray(p.pozo, dtype=float)
    idx_p = int(np.argmin(np.linalg.norm(nodes - well, axis=1)))

    return nodes, groups, normals, smp, nodes[idx_p]


def build_nodes_var(
    use_cache=True,
    grid_step=DEFAULT_GRID_STEP,
    cache_file=CACHE_FILE,
    force_rebuild=False,
    n_smooth=5,
):
    """
    Genera nodos adaptativos.

    use_cache=True:
        Carga el campo del disco o lo calcula si no existe.

    force_rebuild=True:
        Fuerza el recálculo del campo y sobrescribe la caché.
    """
    p = RUNTIME.get()

    smp, boundary_groups, boundary_groups_with_ghosts = (
        _boundary_config()
    )

    if use_cache:
        prepare_spacing_cache(
            grid_step=grid_step,
            cache_file=cache_file,
            force_rebuild=force_rebuild,
        )

        def spacing_callback(xy):
            return spacing_func_cached(
                xy,
                grid_step=grid_step,
                cache_file=cache_file,
            )

    else:
        spacing_callback = spacing_func

    nodes, groups, normals = poisson_disc_nodes(
        spacing_callback,
        (p.vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts,
    )

    # Suavizado de Laplace + inserción de Steiner para mejorar isotopía.
    domain_poly = Polygon(np.asarray(p.vert, dtype=float))
    if n_smooth > 0:
        nodes = _laplacian_smooth(nodes, groups, domain_poly, n_iter=n_smooth)
        print(f"Suavizado Laplace ({n_smooth} iter) aplicado.")
    nodes, groups, normals = _insert_steiner_points(nodes, groups, normals, domain_poly)

    # Asegurar un nodo en la posición exacta del pozo.
    well = np.asarray(p.pozo, dtype=float)

    if np.linalg.norm(nodes - well, axis=1).min() > 0.5:
        new_idx = len(nodes)

        nodes = np.vstack([nodes, well])
        normals = np.vstack([normals, [[0.0, 0.0]]])

        if "interior" not in groups:
            groups["interior"] = np.array([], dtype=int)

        groups["interior"] = np.append(
            groups["interior"],
            new_idx,
        )

    idx_p = int(np.argmin(np.linalg.norm(nodes - well, axis=1)))

    if getattr(p, "plot_grid", False):
        plot_mesh(
            nodes,
            groups,
            smp,
            title="Discretización del dominio",
        )

    print(f"Nodos generados: {len(nodes)}")
    print(f"Nodo del pozo: {nodes[idx_p]}")

    return nodes, groups, normals, smp, nodes[idx_p]


# ============================================================
# DIAGNÓSTICO DE TRIÁNGULOS
# ============================================================

def diagnose_triangulation(
    nodes,
    min_angle_degrees=20.0,
    max_aspect_ratio=8.0,
    domain_vertices=None,
):
    """
    Diagnostica triángulos de Delaunay dentro del dominio.

    Un triángulo se marca si:
      - su ángulo mínimo es inferior a min_angle_degrees, o
      - su razón entre lado mayor y lado menor supera max_aspect_ratio.

    domain_vertices: vértices del dominio (p.vert) para filtrar triángulos
      del convex hull que caen fuera del dominio (en dominios no convexos).
      Si no se pasa, intenta leerlo de RUNTIME.
    """
    from shapely.geometry import Polygon as _Poly, Point as _Pt

    points = np.asarray(nodes, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("nodes debe tener forma (N, 2), con N >= 3.")

    if domain_vertices is None:
        try:
            domain_vertices = RUNTIME.get().vert
        except Exception:
            pass

    triangulation = Delaunay(points)
    triangles = triangulation.simplices

    # Filtrar triángulos cuyo centroide cae fuera del dominio.
    if domain_vertices is not None:
        domain_poly = _Poly(np.asarray(domain_vertices, float))
        centroids = points[triangles].mean(axis=1)
        in_domain = np.array(
            [domain_poly.contains(_Pt(c)) for c in centroids]
        )
        triangles = triangles[in_domain]
    xyz = points[triangles]

    side_a = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
    side_b = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
    side_c = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)

    sides = np.column_stack([side_a, side_b, side_c])

    area = 0.5 * np.abs(
        (xyz[:, 1, 0] - xyz[:, 0, 0])
        * (xyz[:, 2, 1] - xyz[:, 0, 1])
        - (xyz[:, 1, 1] - xyz[:, 0, 1])
        * (xyz[:, 2, 0] - xyz[:, 0, 0])
    )

    longest = np.max(sides, axis=1)
    shortest = np.maximum(
        np.min(sides, axis=1),
        np.finfo(float).eps,
    )

    aspect_ratio = longest / shortest

    angles = np.empty((len(triangles), 3), dtype=float)

    for i in range(3):
        a = sides[:, i]
        b = sides[:, (i + 1) % 3]
        c = sides[:, (i + 2) % 3]

        cosine = np.clip(
            (b**2 + c**2 - a**2)
            / np.maximum(2.0 * b * c, 1e-30),
            -1.0,
            1.0,
        )

        angles[:, i] = np.degrees(np.arccos(cosine))

    min_angles = np.min(angles, axis=1)

    bad_mask = (
        (min_angles < min_angle_degrees)
        | (aspect_ratio > max_aspect_ratio)
    )

    report = {
        "triangles": triangles,
        "area": area,
        "min_angle_degrees": min_angles,
        "aspect_ratio": aspect_ratio,
        "bad_mask": bad_mask,
        "bad_triangles": triangles[bad_mask],
        "bad_count": int(np.count_nonzero(bad_mask)),
        "triangle_count": int(len(triangles)),
    }

    print(
        f"Triángulos marcados: {report['bad_count']} / "
        f"{report['triangle_count']}"
    )

    return report


