"""
Convierte un notebook escrito como script (celdas `# %%` y `# %% [markdown]`,
formato «percent» de jupytext) en .ipynb y, opcionalmente, lo ejecuta.

    python notebooks/construir.py notebooks/1_resultados/1_resultados_experimentos.py --ejecutar
    python notebooks/construir.py <archivo.py> [...]          # sólo convierte (sin salidas)

La fuente de verdad es el .py (se revisa bien en git); el .ipynb se regenera.
Ejecutar requiere nbformat + nbclient + ipykernel (entorno conda `doctorado`).
El kernel corre con el directorio del notebook como directorio de trabajo.
"""
import argparse
import re
from pathlib import Path

import nbformat


def leer_celdas(texto):
    celdas, tipo, buf = [], None, []

    def cerrar():
        if tipo is None:
            return
        lineas = buf[:]
        while lineas and not lineas[-1].strip():
            lineas.pop()
        while lineas and not lineas[0].strip():
            lineas.pop(0)
        if tipo == "markdown":
            lineas = [re.sub(r"^# ?", "", l) for l in lineas]
            celdas.append(nbformat.v4.new_markdown_cell("\n".join(lineas)))
        elif lineas:
            celdas.append(nbformat.v4.new_code_cell("\n".join(lineas)))

    for linea in texto.splitlines():
        if linea.startswith("# %%"):
            cerrar()
            tipo, buf = ("markdown" if "[markdown]" in linea else "code"), []
        else:
            buf.append(linea)
    cerrar()
    return celdas


def construir(py, ejecutar=False, timeout=1800):
    py = Path(py).resolve()
    nb = nbformat.v4.new_notebook()
    nb.cells = leer_celdas(py.read_text())
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    if ejecutar:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(py.parent)}}).execute()
    destino = py.with_suffix(".ipynb")
    nbformat.write(nb, destino)
    print(f"{destino}  ({len(nb.cells)} celdas{', ejecutado' if ejecutar else ''})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--ejecutar", action="store_true")
    a = ap.parse_args()
    for f in a.archivos:
        construir(f, a.ejecutar)
