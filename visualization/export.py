"""
Exportación centralizada de figuras.

Uso:
    from visualization.export import save_figure
    save_figure(fig, "comparison_adr_C", profile="PAPER")
    save_figure(fig, "comparison_adr_C", profile="THESIS")
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt

from visualization.style import PROFILES

# Directorio de salida raíz (configurable vía variable de entorno)
_OUTPUT_ROOT = Path(os.environ.get("VIZ_OUTPUT", "figures/"))


def save_figure(fig, name: str, profile: str, output_dir: Path = None,
                fmt: str = "png", tight: bool = True) -> Path:
    """
    Guarda `fig` con el DPI del perfil dado.

    Parámetros
    ----------
    fig        : matplotlib.figure.Figure
    name       : nombre base del archivo (sin extensión)
    profile    : "THESIS" | "PAPER" | "PAPER_WIDE" | "PRESENTATION"
    output_dir : directorio de salida (por defecto figures/<profile>/)
    fmt        : "png" | "pdf" | "svg" — o "pdf+png" para guardar ambos
    tight      : usar bbox_inches="tight"

    Retorna
    -------
    Path del archivo guardado (el primero si fmt="pdf+png")
    """
    if profile not in PROFILES:
        raise ValueError(f"Perfil desconocido: {profile}")

    if fmt == "pdf+png":
        p1 = save_figure(fig, name, profile, output_dir, "pdf", tight)
        save_figure(fig, name, profile, output_dir, "png", tight)
        return p1

    dpi = PROFILES[profile]["savefig.dpi"]

    if output_dir is None:
        output_dir = _OUTPUT_ROOT / profile.lower()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{name}.{fmt}"
    kwargs = {"dpi": dpi}
    if tight:
        kwargs["bbox_inches"] = "tight"

    fig.savefig(path, **kwargs)
    return path


def save_all_profiles(fig, name: str, profiles=None, output_dir: Path = None,
                      fmt: str = "pdf+png") -> dict:
    """
    Guarda `fig` en todos los perfiles indicados.

    Retorna dict {profile: Path}.
    """
    if profiles is None:
        profiles = ["THESIS", "PAPER", "PRESENTATION"]

    return {p: save_figure(fig, name, p, output_dir=output_dir, fmt=fmt)
            for p in profiles}
