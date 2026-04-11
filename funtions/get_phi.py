import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import pandas as pd
import io
import matplotlib.pyplot as plt

def filtrar_puntos_cercanos(puntos, tolerancia=5):
    puntos_unicos = []
    for p in puntos:
        if all(abs(p[0] - q[0]) > tolerancia or abs(p[1] - q[1]) > tolerancia for q in puntos_unicos):
            puntos_unicos.append(p)
    return puntos_unicos


def interpolar_bidimensional(puntos_malla, puntos, valores):
    puntos = np.asarray(puntos)
    valores = np.asarray(valores)

    # Limpieza de datos
    mask = np.all(np.isfinite(puntos), axis=1) & np.isfinite(valores)
    puntos = puntos[mask]
    valores = valores[mask]

    # Convertir meshgrid
    if isinstance(puntos_malla, tuple):
        X, Y = np.meshgrid(puntos_malla[0], puntos_malla[1])
        puntos_malla = np.column_stack((X.ravel(), Y.ravel()))
    else:
        puntos_malla = np.asarray(puntos_malla)

    # Interpolador lineal
    interp_lin = LinearNDInterpolator(puntos, valores, fill_value=valores.min())

    # Extrapolación con nearest neighbor
    interp_nn = NearestNDInterpolator(puntos, valores)

    # Calculamos
    vals = interp_lin(puntos_malla)

    return vals


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
def gen_malla(malla = None):
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
    

    # Interpolar valores sobre la malla
    por_min = interpolar_bidimensional(malla, puntos_dados, porosidad_n)
    
    # plotear_campo_interpolado(malla, por_min, "campo_interpolado.png")
    por_mean = interpolar_bidimensional(malla, puntos_dados, porosidad_m)
    por_max = interpolar_bidimensional(malla, puntos_dados, porosidad_x)


    #return  malla, [por_min, por_mean, por_max], facet_tags, x_p
    return  por_min