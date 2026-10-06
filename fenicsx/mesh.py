"""
Generación de malla FEM (gmsh + DOLFINx) del dominio real del acuífero.

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

import gmsh
from mpi4py import MPI
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


def generar_malla(vertices, pozo, boundary_groups, lc=80, output_file="data/fenicsx/malla_generada.msh"):
    """
    Genera la malla 2D sobre el polígono `vertices` (p.vert de bfr),
    embebiendo `pozo` como nodo de malla (refinamiento local; la física de
    fuente/sumidero se evalúa vía interpolación en `pozo`, no depende de qué
    punto exacto quedó embebido — ver fenicsx/forward.py).
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

        pid_pozo = gmsh.model.geo.addPoint(float(pozo[0]), float(pozo[1]), 0, lc)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [pid_pozo], 2, surface_id)

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

    domain, facet_tags = generar_malla(
        p.vert, pozo, boundary_groups, lc=spacing, output_file=output_file
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
