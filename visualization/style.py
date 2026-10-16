"""
Configuración centralizada de estilo para figuras científicas.

Tres perfiles:
  THESIS       : tesis LaTeX (15×8 cm, 300 DPI, fuente 10pt)
  PAPER        : artículo SIAM (columna simple 8.5×6 cm, 600 DPI, 9pt)
  PAPER_WIDE   : artículo SIAM doble columna (17×6 cm)
  PRESENTATION : presentación React (proporcional 16:9, 150 DPI, 14pt)

Uso:
  from visualization.style import apply_style, COLORMAPS
  with apply_style("PAPER"):
      fig, ax = plt.subplots()
      ...
"""

import contextlib
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paletas y colormaps
# ---------------------------------------------------------------------------

COLORMAPS = {
    # Campos escalares secuenciales (concentración, velocidad)
    "scalar":     "viridis",
    "scalar_alt": "plasma",

    # Carga hidráulica (varía alrededor de un valor medio ~ 310 m)
    "hydraulic":  "Blues",

    # Diferencias y errores (divergente con centro en 0)
    "divergent":  "RdBu_r",

    # Error relativo (siempre positivo)
    "error":      "Reds",

    # Convergencia (log scale, positivo)
    "convergence": "viridis_r",

    # Gradiente adjunto (puede ser positivo o negativo)
    "adjoint":    "seismic",
}

# Colores categóricos (accesibles, sin rojo/verde como única diferencia)
COLORS = {
    "bfr":     "#1f77b4",  # azul
    "fenicsx": "#ff7f0e",  # naranja
    "error":   "#d62728",  # rojo
    "ref":     "#2ca02c",  # verde (solo junto a otro color, no como única distinción)
    "neutral": "#7f7f7f",  # gris
    "adjoint": "#9467bd",  # morado
}

# Marcadores para series (accesibles)
MARKERS = {
    "bfr":     "o",
    "fenicsx": "s",
    "ref":     "^",
    "extra":   "D",
}

# ---------------------------------------------------------------------------
# Perfiles
# ---------------------------------------------------------------------------

PROFILES = {
    "THESIS": {
        "figure.figsize":    (15 / 2.54, 8 / 2.54),   # cm → pulgadas
        "savefig.dpi":       300,
        "figure.dpi":        100,
        "font.size":         10,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "axes.linewidth":    0.8,
        "font.family":       "serif",
        "text.usetex":       False,  # True si se quiere LaTeX (requiere instalación)
        "mathtext.fontset":  "stix",
        "figure.constrained_layout.use": True,
    },
    "PAPER": {
        "figure.figsize":    (8.5 / 2.54, 6.0 / 2.54),
        "savefig.dpi":       600,
        "figure.dpi":        150,
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.2,
        "lines.markersize":  4,
        "axes.linewidth":    0.6,
        "font.family":       "serif",
        "text.usetex":       False,
        "mathtext.fontset":  "stix",
        "figure.constrained_layout.use": True,
    },
    "PAPER_WIDE": {
        "figure.figsize":    (17.0 / 2.54, 6.0 / 2.54),
        "savefig.dpi":       600,
        "figure.dpi":        150,
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.2,
        "lines.markersize":  4,
        "axes.linewidth":    0.6,
        "font.family":       "serif",
        "text.usetex":       False,
        "mathtext.fontset":  "stix",
        "figure.constrained_layout.use": True,
    },
    "PRESENTATION": {
        "figure.figsize":    (16 * 0.5, 9 * 0.5),     # mitad del 16:9 en pulgadas
        "savefig.dpi":       150,
        "figure.dpi":        100,
        "font.size":         14,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   11,
        "lines.linewidth":   2.5,
        "lines.markersize":  8,
        "axes.linewidth":    1.2,
        "font.family":       "sans-serif",
        "text.usetex":       False,
        "mathtext.fontset":  "stixsans",
        "figure.constrained_layout.use": True,
    },
}


@contextlib.contextmanager
def apply_style(profile: str):
    """
    Context manager que aplica el perfil dado y restaura el estilo anterior.

    Uso:
        with apply_style("PAPER"):
            fig, ax = plt.subplots()
            ...
    """
    if profile not in PROFILES:
        raise ValueError(f"Perfil '{profile}' no reconocido. Opciones: {list(PROFILES)}")

    with plt.rc_context(PROFILES[profile]):
        yield


def get_figsize(profile: str, width_fraction: float = 1.0, aspect: float = None):
    """
    Devuelve (width, height) en pulgadas según el perfil.

    width_fraction: fracción del ancho del perfil (ej. 0.5 para media columna)
    aspect:        razón height/width (None = usar la del perfil)
    """
    w, h = PROFILES[profile]["figure.figsize"]
    w = w * width_fraction
    if aspect is not None:
        h = w * aspect
    return (w, h)
