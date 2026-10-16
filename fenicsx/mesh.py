"""
Generación de malla FEM (gmsh + DOLFINx) del dominio real del acuífero.

build_mesh_from_bfr_pkl() permite usar exactamente la nube de nodos BFR
(con su triangulación Delaunay) como malla DOLFINx, para comparaciones
nodo a nodo entre ambos backends.

El contorno del dominio es literalmente `config/geometry.py:build_geometry`
(`p.vert`, el mismo pentágono que usa bfr para generar sus nodos RBF-FD) y
la clasificación de fronteras inlet/outlet/wall replica
`funtions/mesh.py:_boundary_config` — así ambos backends resuelven sobre el
MISMO dominio. No se importa `funtions.mesh` directamente porque arrastra
`rbf`/`shapely`, que no están instalados en `fenics_env` (ver
fenicsx/README.md); el mapeo de bordes se duplica aquí, minimal y comentado.

El pozo (x_p, z_p) no está hardcodeado a [3700, -700]: se recibe como
parámetro `pozo`, para que z_p pueda ser un control optimizable (igual que
p.pozo en el backend bfr).

Reutiliza funtions.get_phi.well_layer_data() sólo para la porosidad por
capa (no para el contorno) — misma fuente de datos reales que ya usa bfr.

Requiere un entorno con dolfinx + gmsh + mpi4py (no están en
requirements.txt / bfr_env). Ver fenicsx/README.md.
"""
import os
import pickle

import gmsh
import numpy as np
from mpi4py import MPI
import basix.ufl
import dolfinx
import dolfinx.mesh as dmesh
# En DOLFINx >=0.11 el submódulo se renombró de dolfinx.io.gmshio a
# dolfinx.io.gmsh (colisiona de nombre con el paquete `gmsh` importado
# arriba, de ahí el alias).
from dolfinx.io import gmsh as gmshio

from funtions.get_phi import well_layer_data, interpolar_bidimensional
from funtions.runtime import RUNTIME

# Mismo mapeo edge-index -> grupo físico que funtions/mesh.py:_boundary_config
# (edge i conecta vert[i] con vert[(i+1) % len(vert)], mismo orden que p.vert).
# Duplicado aquí porque ese módulo no se puede importar en fenics_env (ver
# docstring del archivo).
_BOUNDARY_GROUPS = {
    "real": {"inlet": [2], "outlet": [4], "wall": [0, 1, 3]},
    "benchmark": {"inlet": [1], "outlet": [3], "wall": [0, 2]},
}


def generar_malla(vertices, pozo, boundary_groups, lc=80, lc_near=None,
                  r_near=100.0, r_far=350.0,
                  output_file="data/fenicsx/malla_generada.msh"):
    """
    Genera la malla 2D sobre el polígono `vertices`.

    lc       : tamaño global de elemento [m]
    lc_near  : tamaño de elemento cerca del pozo; si es None o >= lc, malla uniforme.
    r_near   : radio [m] dentro del cual se aplica lc_near (100 m ≈ Gaussian support 2.5*30m*1.3)
    r_far    : radio [m] donde la transición a lc global termina
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    mesh_comm = MPI.COMM_WORLD

    gmsh.initialize()
    try:
        point_ids = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in vertices]
        line_ids = [
            gmsh.model.geo.addLine(point_ids[i], point_ids[(i + 1) % len(vertices)])
            for i in range(len(vertices))
        ]

        gmsh.model.geo.addCurveLoop(line_ids, 1)
        surface_id = gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()

        _lc_pozo = lc_near if (lc_near is not None and lc_near < lc) else lc
        pid_pozo = gmsh.model.geo.addPoint(float(pozo[0]), float(pozo[1]), 0, _lc_pozo)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [pid_pozo], 2, surface_id)

        # Refinamiento local cerca del pozo: campo Distance + Threshold de gmsh.
        # Esto replica para FEniCS el mismo efecto que min_sp hace en BFR:
        # más nodos cerca del pozo → mejor resolución del cono de velocidades.
        if lc_near is not None and lc_near < lc:
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [pid_pozo])
            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField",   f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin",   lc_near)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax",   lc)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin",   r_near)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax",   r_far)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

        # Orden inlet -> wall -> outlet: addPhysicalGroup asigna tags
        # secuenciales 1,2,3, y fenicsx/forward.py asume inlet_marker=1,
        # outlet_marker=3 (ver _boundaries_elements) — mismo orden que
        # funtions/mesh.py:_boundary_config para que ambos backends manden
        # los mismos flujos por los mismos bordes físicos.
        for name in ("inlet", "wall", "outlet"):
            entities = [line_ids[i] for i in boundary_groups[name]]
            tag = gmsh.model.addPhysicalGroup(1, entities)
            gmsh.model.setPhysicalName(1, tag, name)

        gmsh.model.addPhysicalGroup(2, [surface_id])
        gmsh.model.setPhysicalName(2, surface_id, "surface")
        gmsh.model.geo.synchronize()
        gmsh.write(output_file)

        gmsh.model.mesh.generate(2)
        gmsh.write(output_file)

        # DOLFINx >=0.11: model_to_mesh devuelve un MeshData (antes era una
        # tupla (domain, cell_tags, facet_tags)).
        mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0, gdim=2)
        domain, facet_tags = mesh_data.mesh, mesh_data.facet_tags
    finally:
        gmsh.finalize()

    return domain, facet_tags


def build_mesh(pozo, spacing=80.0, output_file="data/fenicsx/malla_generada.msh"):
    """
    Construye la malla FEM y la porosidad interpolada sobre sus nodos.

    pozo: [x_p, z_p] — posición del pozo (z_p es el control optimizable).
    Devuelve (domain, facet_tags, porosidad_array) — porosidad_array es un
    numpy array plano (no un dolfinx.fem.Function todavía; eso se arma en
    fenicsx/forward.py una vez definido el espacio de funciones G).
    """
    p = RUNTIME.get()
    try:
        boundary_groups = _BOUNDARY_GROUPS[p.domain]
    except KeyError:
        raise ValueError(f"Dominio inválido: {p.domain}")

    lc_near = getattr(p, "fenicsx_lc_near", None)
    domain, facet_tags = generar_malla(
        p.vert, pozo, boundary_groups, lc=spacing, lc_near=lc_near,
        output_file=output_file,
    )

    puntos_dados, porosidad_n, _, _ = well_layer_data()
    malla_points = domain.geometry.x[:, :2]
    porosidad = interpolar_bidimensional(malla_points, puntos_dados, porosidad_n)

    return domain, facet_tags, porosidad


def load_cached_mesh(output_file="data/fenicsx/malla_generada.msh"):
    """
    Carga una malla ya generada (.msh) sin volver a mallar — etapa de
    preprocesamiento con `pre=False`, análoga a load_data() de bfr cuando no
    reejecuta el preproceso. La porosidad se recomputa sobre los nodos (barato).
    Asume que el .msh se generó con el mismo `pozo` embebido.
    """
    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"No hay malla cacheada en '{output_file}'. Corre al menos una vez "
            f"con pre=True (o activa 'Preprocessing' en la app) para generarla."
        )

    result = gmshio.read_from_msh(output_file, MPI.COMM_WORLD, 0, gdim=2)
    # DOLFINx >=0.11 devuelve MeshData; versiones previas, una tupla.
    if hasattr(result, "mesh"):
        domain, facet_tags = result.mesh, result.facet_tags
    else:
        domain, _cell_tags, facet_tags = result

    puntos_dados, porosidad_n, _, _ = well_layer_data()
    porosidad = interpolar_bidimensional(domain.geometry.x[:, :2], puntos_dados, porosidad_n)

    return domain, facet_tags, porosidad


def build_mesh_from_bfr_pkl(pkl_path="data/input/bfr/real/adr/preproceso.pkl"):
    """
    Construye una malla DOLFINx usando exactamente los nodos y la triangulación
    Delaunay del PKL de BFR.  Permite comparar FEniCS y BFR sobre el MISMO
    conjunto de nodos (misma geometría discreta, distintos esquemas numéricos).

    La etiqueta de facetas (facet_tags) se reconstruye a partir de los grupos
    de frontera guardados en el PKL ('groups': dict edge_name -> índices globales).
    Marcadores: inlet=1, wall=2, outlet=3  (igual que generar_malla).

    Devuelve (domain, facet_tags, porosidad_array).
    """
    from scipy.spatial import Delaunay

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    all_nodes = data["nodes"]   # (N, 2) — incluye nodos ghost fuera del dominio
    groups    = data["groups"]

    # Excluir nodos ghost (fuera del dominio) — solo interior + boundary
    ghost_idxs = set()
    for key in groups:
        if key.startswith("ghost"):
            ghost_idxs.update(int(i) for i in groups[key])

    keep_mask = np.ones(len(all_nodes), dtype=bool)
    keep_mask[list(ghost_idxs)] = False
    keep_idxs = np.where(keep_mask)[0]

    # Reasignar índices: old_idx -> new_idx
    old_to_new = np.full(len(all_nodes), -1, dtype=np.int64)
    old_to_new[keep_idxs] = np.arange(len(keep_idxs))

    nodes = all_nodes[keep_idxs]   # (N_clean, 2)

    # Extraer frontera como polígono cerrado ordenado por ángulo respecto al
    # centroide — necesario para CDT constrained que respete el borde.
    import triangle as _tr
    bnd_keys = [k for k in groups if k.startswith("boundary")]
    bnd_all_old = np.concatenate([groups[k] for k in bnd_keys])
    bnd_pts = all_nodes[bnd_all_old]
    centroid = bnd_pts.mean(axis=0)
    angle_order = np.argsort(
        np.arctan2(bnd_pts[:, 1] - centroid[1], bnd_pts[:, 0] - centroid[0])
    )
    bnd_ordered = old_to_new[bnd_all_old[angle_order]]
    n_bnd = len(bnd_ordered)
    segs = np.column_stack([bnd_ordered, np.roll(bnd_ordered, -1)]).astype(np.int32)

    # CDT con calidad q20 (ángulo mínimo ≥ 20°), S0 = sin Steiner en frontera.
    # Usando los nodos BFR como puntos fijos: 0 nodos Steiner necesarios.
    cdt = _tr.triangulate({"vertices": nodes, "segments": segs}, "pq20S0")
    triangles = cdt["triangles"].astype(np.int64)   # (M, 3)
    nodes = cdt["vertices"]   # puede incluir Steiner si q20 lo requiere

    # --- malla de celdas ---
    coord_elem = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = dmesh.create_mesh(
        MPI.COMM_WORLD,
        triangles,
        coord_elem,
        nodes.astype(np.float64),
    )

    # --- facet_tags desde grupos BFR ---
    # DOLFINx reordena los vértices internamente, así que mapeamos por
    # coordenadas (KDTree): buscamos cada DOF de la malla en nuestra nube
    # original y asignamos su marker de frontera.
    from scipy.spatial import cKDTree

    dof_coords = domain.geometry.x[:, :2]   # coords en orden DOLFINx
    tree = cKDTree(nodes)
    _, dof_to_node = tree.query(dof_coords, k=1)  # dof i → nodo original

    group_to_marker = {"boundary:inlet": 1, "boundary:wall": 2, "boundary:outlet": 3}
    node_to_marker_orig = np.zeros(len(nodes), dtype=np.int32)
    for key, marker in group_to_marker.items():
        idxs = groups.get(key)
        for idx in (idxs if idxs is not None else []):
            new_idx = old_to_new[int(idx)]
            if new_idx >= 0:
                node_to_marker_orig[new_idx] = marker

    # dof_marker[i] = marker del DOF i según DOLFINx
    dof_marker = node_to_marker_orig[dof_to_node]

    domain.topology.create_entities(1)
    domain.topology.create_connectivity(1, 0)
    domain.topology.create_connectivity(2, 1)
    domain.topology.create_connectivity(1, 2)
    e2v = domain.topology.connectivity(1, 0)
    n_edges = e2v.num_nodes

    edge_markers = np.zeros(n_edges, dtype=np.int32)
    for e in range(n_edges):
        verts = e2v.links(e)
        m0, m1 = dof_marker[verts[0]], dof_marker[verts[1]]
        if m0 > 0 and m0 == m1:
            edge_markers[e] = m0

    from dolfinx.mesh import meshtags
    tagged = np.sort(np.where(edge_markers > 0)[0]).astype(np.int32)
    facet_tags = meshtags(domain, 1, tagged, edge_markers[tagged])

    # --- porosidad sobre nodos DOLFINx (puede incluir Steiner) ---
    from funtions.get_phi import well_layer_data, interpolar_bidimensional
    puntos_dados, porosidad_n, _, _ = well_layer_data()
    porosidad = interpolar_bidimensional(domain.geometry.x[:, :2], puntos_dados, porosidad_n)

    n_steiner = len(nodes) - len(keep_idxs)
    print(f"[mesh] BFR pkl → DOLFINx CDT: N={len(nodes)} nodos (+{n_steiner} Steiner), {len(triangles)} triángulos")
    return domain, facet_tags, porosidad


def get_mesh(pozo, spacing, regenerate, output_file="data/fenicsx/malla_generada.msh"):
    """
    Etapa de preprocesamiento configurable (flag `pre`): si `regenerate` (o no
    existe cache), genera la malla desde cero; si no, reusa la .msh cacheada.
    """
    if regenerate or not os.path.exists(output_file):
        print(">>> [fenicsx] preprocesamiento: generando malla...")
        return build_mesh(pozo, spacing=spacing, output_file=output_file)
    print(">>> [fenicsx] preprocesamiento: reusando malla cacheada", output_file)
    return load_cached_mesh(output_file=output_file)
