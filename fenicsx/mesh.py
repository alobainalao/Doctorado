"""
Generación de malla FEM (gmsh + DOLFINx) del dominio real del acuífero.

Adaptado de TesisCode/new_grid.py:generar_malla — la diferencia respecto al
original es que el pozo (x_p, z_p) ya no está hardcodeado a [3700, -700]:
se recibe como parámetro `pozo`, para que z_p pueda ser un control
optimizable (igual que p.pozo en el backend bfr, ver config/geometry.py).

Reutiliza funtions.get_phi.well_layer_data() para la geometría de capas y
porosidad — misma fuente de datos reales que ya usa el backend bfr, no se
duplica aquí.

Requiere un entorno con dolfinx + gmsh + mpi4py (no están en
requirements.txt / bfr_env). Ver fenicsx/README.md.

NO EJECUTADO/VERIFICADO: no hay entorno dolfinx disponible en este sandbox.
"""
import os

import gmsh
from mpi4py import MPI
# En DOLFINx >=0.11 el submódulo se renombró de dolfinx.io.gmshio a
# dolfinx.io.gmsh (colisiona de nombre con el paquete `gmsh` importado
# arriba, de ahí el alias).
from dolfinx.io import gmsh as gmshio

from funtions.get_phi import well_layer_data, split_point, interpolar_bidimensional


def generar_malla(vertices, puntos_internos, pozo, lc=80, output_file="data/fenicsx/malla_generada.msh"):
    """
    Genera la malla 2D del dominio real, embebiendo `pozo` como nodo de
    malla (refinamiento local; la física de fuente/sumidero se evalúa vía
    interpolación en `pozo`, no depende de qué punto exacto quedó
    embebido — ver fenicsx/forward.py).
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

        embedded_points = []
        for punto in puntos_internos:
            pid = gmsh.model.geo.addPoint(punto[0], punto[1], 0, lc)
            embedded_points.append(pid)

        pid_pozo = gmsh.model.geo.addPoint(float(pozo[0]), float(pozo[1]), 0, lc)
        embedded_points.append(pid_pozo)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, embedded_points, 2, surface_id)

        # Fronteras físicas — mismos índices de línea que TesisCode/new_grid.py,
        # dependen del orden de `vertices` (12 puntos externos, ver
        # funtions/get_phi.py:split_point).
        borders = {
            "inlet": line_ids[2:6],
            "walllet": line_ids[:2] + line_ids[6:9],
            "outlet": line_ids[9:],
        }
        for name, entities in borders.items():
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
    puntos_dados, porosidad_n, _, _ = well_layer_data()
    ext_points, int_points = split_point(puntos_dados)

    domain, facet_tags = generar_malla(
        ext_points, int_points, pozo, lc=spacing, output_file=output_file
    )

    malla_points = domain.geometry.x[:, :2]
    porosidad = interpolar_bidimensional(malla_points, puntos_dados, porosidad_n)

    return domain, facet_tags, porosidad
