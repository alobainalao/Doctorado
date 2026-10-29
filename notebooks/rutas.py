"""
Rutas comunes de los notebooks: raíz del repo `codigo/` y carpetas de
resultados de los experimentos.

Los resultados (`data/output/experimentos/<grupo>/`) no están en git y cada
checkout tiene los suyos: los grupos A–F y S se corrieron en el checkout
principal y W y Z en el worktree de la rama exp/sumidero-W. `grupo(nombre)`
busca el grupo en todos los worktrees del repo (primero en el actual), así
que el notebook funciona desde cualquiera de ellos.
"""
import os
import subprocess
import sys
from pathlib import Path

CODIGO = Path(__file__).resolve().parents[1]
if str(CODIGO) not in sys.path:
    sys.path.insert(0, str(CODIGO))


def _worktrees():
    try:
        out = subprocess.run(["git", "-C", str(CODIGO), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        raices = [Path(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("worktree ")]
    except (OSError, subprocess.CalledProcessError):
        raices = []
    return [CODIGO] + [r for r in raices if r.resolve() != CODIGO]


RAICES_EXP = [Path(os.environ["EXP_ROOT"])] if "EXP_ROOT" in os.environ else \
    [r / "data" / "output" / "experimentos" for r in _worktrees()]


def grupo(nombre):
    """Carpeta del grupo `nombre` (la primera que exista) o None."""
    for r in RAICES_EXP:
        if (r / nombre).is_dir():
            return r / nombre
    return None


def archivo(rel):
    """Primer `<raíz>/<rel>` que exista (p. ej. 'resumen.csv') o None."""
    for r in RAICES_EXP:
        if (r / rel).exists():
            return r / rel
    return None
