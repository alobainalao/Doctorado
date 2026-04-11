import numpy as np


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def dv(x0, xb):
    """
    Perfil de velocidad deformado (usado en condiciones de frontera)
    """
    return x0 - xb/2 + (x0 - xb) / (4 * np.pi) * np.sin(2 * x0 * np.pi / (x0 - xb))


# ==========================================================
# GEOMETRÍA PRINCIPAL
# ==========================================================

def build_geometry(domain="real", spacing=30):
    """
    Construye la geometría del dominio dependiendo del escenario.

    Parámetros:
    ----------
    domain : str
        "real" o "benchmark"
    spacing : float
        usado para definir epsilon

    Retorna:
    -------
    dict con todos los parámetros geométricos
    """

    if domain == "real":
        vert = np.array([
            [816,   -2690.5],
            [4354,  -1757.5],
            [10755,  -768.5],
            [10755,   250.5],
            [816,     250.5]
        ])

        pozo = np.array([3700, -700])
        fnte = np.array([9116, 109])

        # Parámetros hidráulicos reales
        zi_max, zo_max, z_min = 250, 250, -2691
        inlet_z_t, zo_min = -760, -2690.5
        inlet_a = -2e-4

    elif domain == "benchmark":

        # Dominio rectangular simple (ideal para validación)
        vert = np.array([
            [0, -30],
            [100, -30],
            [100, 0],
            [0, 0]
        ])

        pozo = np.array([20, -15])
        fnte = np.array([80, -10])

        # Parámetros simplificados
        zi_max, zo_max, z_min = 0, 0, -30
        inlet_z_t, zo_min = -15, -30
        inlet_a = -1e-4

    else:
        raise ValueError(f"Dominio inválido: {domain}")

    # ------------------------------------------------------
    # PROPIEDADES DERIVADAS
    # ------------------------------------------------------

    xmin, xmax = vert[:, 0].min(), vert[:, 0].max()
    ymin, ymax = vert[:, 1].min(), vert[:, 1].max()

    Lx = xmax - xmin
    Ly = ymax - ymin

    epsilon_x = 0.5 * spacing
    epsilon_y = spacing

    # Perfil de velocidades
    d_vel = inlet_a / 15
    outlet_a = inlet_a * dv(zi_max, inlet_z_t) / dv(zo_max, zo_min)

    return {
        "domain": domain,

        # Geometría
        "vert": vert,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "Lx": Lx,
        "Ly": Ly,

        # Puntos especiales
        "pozo": pozo,
        "fnte": fnte,

        # Parámetros espaciales
        "epsilon_x": epsilon_x,
        "epsilon_y": epsilon_y,

        # Flujo
        "zi_max": zi_max,
        "zo_max": zo_max,
        "z_min": z_min,
        "inlet_z_t": inlet_z_t,
        "zo_min": zo_min,
        "inlet_a": inlet_a,
        "d_vel": d_vel,
        "outlet_a": outlet_a
    }
