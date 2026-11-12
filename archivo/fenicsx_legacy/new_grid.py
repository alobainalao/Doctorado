import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from basix.ufl import element
import pandas as pd
from dolfinx.io import gmshio
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def graficar_condiciones_de_frontera(vertices, sumidero, fuente):
    """
    Grafica las condiciones de frontera en un solo gráfico combinando agua y contaminante.
    """
    # Crear la figura
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.1, 0.3, 0.85, 0.6])
    plt.title("Condiciones de frontera para agua y contaminante")

    # --- Agua ---
    # Graficar inflow para el agua
    plt.plot(
        [10700, 10700],
        [-768.5, 308.5],
        'b-', linewidth=2, label="Entrada de agua (Neumann no homogenea)"
    )

    # Graficar outflow para el agua
    plt.plot(
        [770, 770],
        [-2690.5, 250.5],
        'r-', linewidth=2, label="Salida de agua (Neumann no homogenea)"
    )

    # Graficar outflow para el agua
    plt.plot(
        [850, 850],
        [-2690.5, 250.5],
        'c-', linewidth=2, label="Salida de contaminante (nothing)"
    )

    # Graficar walls para el agua
    plt.plot(
        [816, 4354, 10755],
        [-2690.5, -1757.5, -768.5],
        'g-', linewidth=2, label="Fondo y superficie (no-slip)"
    )

    plt.plot(
        [816, 4354, 6717, 10755],
        [250.5, 194.5, 334.75, 308.8],
        'g-', linewidth=2
    )

    # --- Contaminante ---
    # Graficar inflow para el contaminante (entrada en un pequeño espesor cercano a la superficie)
    espesor_superficie = 200  # Proporción del segmento inflow correspondiente a la superficie
    inflow_x_cont = [10800, 10800]
    inflow_z_cont = [308.5, 308.5 - espesor_superficie]
    plt.plot(inflow_x_cont, inflow_z_cont, 'c-', linewidth=2)

    # Graficar el resto de inflow como cero
    plt.plot(
        inflow_x_cont,
        [inflow_z_cont[1], -768.5],
        'c-', linewidth=2
    )

    # Graficar el punto sumidero
    plt.scatter(
        sumidero[0], sumidero[1],
        color='m', s=30, label="Sumidero", zorder=5
    )
    plt.text(
        sumidero[0], sumidero[1], 'Sumidero', color='m',
        fontsize=10, ha='left', va='bottom'
    )

    plt.scatter(
        fuente[0], fuente[1],
        color='c', s=30, label="Fuente", zorder=5
    )
    plt.text(
        fuente[0], fuente[1], ' Fuente', color='c',
        fontsize=10, ha='left', va='bottom'
    )

    # Configuración del gráfico
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47), ncol=2)
    plt.grid(True)


    plt.savefig("boundary.png")
    print("fig saved")

def generar_malla(vertices, puntos, output_file=None):

    # Inicializar la comunicación MPI
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    try:
        # Inicializar GMSH
        gmsh.initialize()

        # Tamaño de los elementos
        lc = 80

        # Definir puntos y líneas del cuadrado
        point_ids = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in vertices]
        line_ids = [gmsh.model.geo.addLine(point_ids[i], point_ids[(i + 1) % len(vertices)]) for i in range(len(vertices))]

        # Crear la superficie
        gmsh.model.geo.addCurveLoop(line_ids, 1)
        surface_id = gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()


        # Agregar puntos internos asegurando que sean nodos de la malla
        embedded_points = []
        for punto in puntos:
            pid = gmsh.model.geo.addPoint(punto[0], punto[1], 0, lc)
            embedded_points.append(pid)

        # Agregar punto adicional 1200
        punto = [3700, -700]
        punto1 = [9116, 109]
        pid = gmsh.model.geo.addPoint(punto[0], punto[1], 0, lc)
        embedded_points.append(pid)
        pid = gmsh.model.geo.addPoint(punto1[0], punto1[1], 0, lc)
        embedded_points.append(pid)

        gmsh.model.geo.synchronize()

        # Asegurar que los puntos sean parte de la malla (Embedded Points)
        gmsh.model.mesh.embed(0, embedded_points, 2, surface_id)

        # Definir fronteras con etiquetas físicas
        borders = {
            "inlet": line_ids[2:6],
            "walllet": line_ids[:2] + line_ids[6:9],
            "outlet": line_ids[9:]
        }
        for name, entities in borders.items():
            tag = gmsh.model.addPhysicalGroup(1, entities)
            gmsh.model.setPhysicalName(1, tag, name)

        # Añadir grupo físico para la superficie (dim=2)
        gmsh.model.addPhysicalGroup(2, [surface_id])  # Dim=2 para la superficie
        gmsh.model.setPhysicalName(2, surface_id, "surface")
        gmsh.model.geo.synchronize()
        gmsh.write(output_file)

        # Generar la malla
        gmsh.model.mesh.generate(2)  # Algoritmo de mallado
        gmsh.write(output_file)
        domain, _, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

        from dolfinx.io import VTKFile
        with VTKFile(domain.comm, "malla.pvd", "w") as vtk:
            vtk.write_mesh(domain)

        # Graficar las condiciones de frontera
        #graficar_condiciones_de_frontera(puntos, punto, punto1)
    except Exception as e:
        print(f"Error al generar la malla: {str(e)}")
    finally:
        gmsh.finalize()

    return domain, facet_tags, punto



def filtrar_puntos_cercanos(puntos, tolerancia=5):
    puntos_unicos = []
    for p in puntos:
        if all(abs(p[0] - q[0]) > tolerancia or abs(p[1] - q[1]) > tolerancia for q in puntos_unicos):
            puntos_unicos.append(p)
    return puntos_unicos


def interpolar_bidimensional(malla, puntos, valores, metodo="linear"):
    """
    Interpola valores bidimensionales en toda la malla.

    :param malla: Malla generada.
    :param puntos: Lista de puntos específicos [[x1, y1], [x2, y2], ...].
    :param valores: Lista de valores correspondientes a los puntos [v1, v2, ...].
    :param metodo: Método de interpolación ("linear", "nearest", "cubic").
    :return: Función escalar interpolada en la malla.
    """
    if len(puntos) != len(valores):
        raise ValueError("El número de puntos debe coincidir con el número de valores.")

    # Convertir puntos y valores a arrays de NumPy
    puntos = np.array(puntos)
    valores = np.array(valores)

    # Obtener las coordenadas geométricas de los nodos de la malla
    puntos_malla = malla.geometry.x[:, :2]

    # Interpolar valores en los nodos de la malla
    valores_interpolados = griddata(puntos, valores, puntos_malla, method=metodo, fill_value=0.0)

    # Crear espacio de función escalar de primer orden
    s_cg1 = element("Lagrange", malla.topology.cell_name(), 1)
    espacio_funcion = fem.functionspace(malla, s_cg1)

    # Crear una función escalar sobre la malla
    funcion_interpolada = fem.Function(espacio_funcion)

    # Asignar los valores interpolados a la función
    with funcion_interpolada.vector.localForm() as local_form:
        local_form[:] = valores_interpolados

    return funcion_interpolada


def generar_puntos(lista_x, lista_y):
    # Verificar que ambas listas tengan la misma longitud
    if len(lista_x) != len(lista_y):
        raise ValueError("Las listas deben tener la misma longitud.")

    # Crear los puntos como tuplas de (x, y)
    puntos = [(x, y) for x, y in zip(lista_x, lista_y)]
    return puntos


def split_point(puntos):
    puntos = np.array(puntos)
    int_id = [6, 7, 8, 10]
    ext_id = [15, 11, 4, 3, 2, 1, 0, 5, 9, 12, 13, 14]
    return puntos[ext_id], puntos[int_id]


def plotear_campo_interpolado(malla, campo, nombre_archivo):
    """
    Plotea un campo interpolado sobre una malla y guarda la imagen.

    :param malla: Malla generada (DOLFINx Mesh).
    :param campo: Valores escalares interpolados (array 1D).
    :param nombre_archivo: Nombre del archivo para guardar la imagen.
    """
    # Extraer las coordenadas de la malla (solo x e y)
    puntos = malla.geometry.x[:, :2]

    # Agregar una coordenada z (por defecto 0) para que tenga 3 dimensiones
    puntos_3d = np.hstack([puntos, np.zeros((puntos.shape[0], 1))])

    # Crear conectividad si no existe (se asegura que la conectividad esté disponible)


# Bloque principal
def gen_malla():
    pozos = [1, 2, 3, 4]
    capas = ['A', 'B', 'C', 'D', 'E']

    # Crear el Mesh Grid
    mesh_grid = [(pozo, capa) for pozo in pozos for capa in capas]

    # Crear una constante para la columna X
    x = [10755, 10755, 10755, 10755, 10755, 6716, 6716, 6716, 6716, 6716, 4354, 4354, 4354, 4354, 4354, 816, 816, 816,
         816, 816]
    z0 = [362, 255, 28, -180, -607, 347.5, 322, -78, -457, np.nan, 346, 43, -390, np.nan, np.nan, 355, 146, -447, -2256,
          np.nan]
    z1 = [255, 28, -180, -607, -930, 322, -78, -457, -725, np.nan, 43, -390, -3125, np.nan, np.nan, 146, -447, -2256,
          -3125, np.nan]
    z_m = (np.array(z0) + np.array(z1)) / 2.
    porosidad_n = [0.87, 0.95, 0.89, 0.56, 0.03, 0.25, 0.56, 0.28, 0.08, np.nan, 0.83, 0.9, 0.03, np.nan, np.nan, 0.56,
                   0.87, 0.28, 0.03, np.nan]
    porosidad_x = [0.95, 0.95, 0.96, 0.86, 0.07, 0.62, 0.86, 0.66, 0.30, np.nan, 0.92, 0.95, 0.07, np.nan, np.nan, 0.86,
                   0.95, 0.66, 0.07, np.nan]
    porosidad_m = (np.array(porosidad_n) + np.array(porosidad_x)) / 2.
    # Crear el DataFrame

    data = {
        "Pozo": [pozo for pozo, _ in mesh_grid],
        "Capa": [capa for _, capa in mesh_grid],
        "X": x,
        "Z0": z0,
        "Z1": z1,
        "ZM": z_m,
        "Porosidad_n": porosidad_n,
        "Porosidad_x": porosidad_x,
        "Porosidad_m": porosidad_m
    }
    df = pd.DataFrame(data)
    df = df.dropna()
    # Guardar en un archivo CSV
    df.to_csv("datos_pozos.csv", index=False)
    print("Archivo 'datos_pozos.csv' generado con éxito.")

    # Extraer columnas del DataFrame
    x = df["X"].tolist()
    z_m = df["ZM"].tolist()
    porosidad_n = df["Porosidad_n"].tolist()
    porosidad_m = df["Porosidad_m"].tolist()
    porosidad_x = df["Porosidad_x"].tolist()

    puntos_dados = generar_puntos(x, z_m)
    ext_points, int_points = split_point(puntos_dados)
 
    # Generar malla
    malla, facet_tags, x_p = generar_malla(ext_points, int_points, "malla_generada.msh")

    # Interpolar valores sobre la malla
    por_min = interpolar_bidimensional(malla, puntos_dados, porosidad_n)
    
    # plotear_campo_interpolado(malla, por_min, "campo_interpolado.png")
    por_mean = interpolar_bidimensional(malla, puntos_dados, porosidad_m)
    por_max = interpolar_bidimensional(malla, puntos_dados, porosidad_x)

    # Guardar la malla y la función interpolada
    with io.XDMFFile(MPI.COMM_WORLD, "malla.xdmf", "w") as archivo:
        archivo.write_mesh(malla)
        archivo.write_function(por_min)

    print("malla generada")

    #return  malla, [por_min, por_mean, por_max], facet_tags, x_p
    return  malla, por_min, facet_tags, x_p
