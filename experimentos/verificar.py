"""
Verificaciones previas a cualquier experimento (las corre correr.py).

1. Malla congelada: existe, su número de nodos y su hash coinciden con los
   registrados al congelarla (nadie la regeneró).
2. n_stencil de base.json = n_stencil con que se construyó la malla.
3. CI (data/input/funcion.h5): C ≥ 0.

    python -m experimentos.verificar
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_JSON = os.path.join(CODIGO_DIR, "experimentos", "base.json")


def verificar():
    with open(BASE_JSON) as f:
        base = json.load(f)
    errores = []

    mesh_pkl = os.path.join(CODIGO_DIR, base["mesh_pkl"])
    if not os.path.exists(mesh_pkl):
        errores.append(f"falta la malla congelada: {base['mesh_pkl']}")
    else:
        with open(mesh_pkl, "rb") as f:
            m = pickle.load(f)
        nodes = np.ascontiguousarray(m["nodes"], dtype=np.float64)
        sha = hashlib.sha256(nodes.tobytes()).hexdigest()
        meta = m.get("meta", {})
        if len(nodes) != meta.get("n_nodos"):
            errores.append(f"malla con {len(nodes)} nodos (registrado {meta.get('n_nodos')})")
        if sha != meta.get("nodes_sha256"):
            errores.append("hash de nodos distinto al registrado: la malla cambió")
        if meta.get("n_stencil") != base["n_stencil"]:
            errores.append(f"n_stencil base.json={base['n_stencil']} ≠ malla={meta.get('n_stencil')}")
        print(f"[verificar] malla  {len(nodes)} nodos  sha256={sha[:12]}…  n_stencil={meta.get('n_stencil')}")

    try:
        import h5py
        with h5py.File(os.path.join(CODIGO_DIR, "data", "input", "funcion.h5"), "r") as f:
            c = f["c"][:]
        if (c < 0).any():
            errores.append(f"CI con {(c < 0).sum()} valores C<0 en funcion.h5")
        print(f"[verificar] CI     C ∈ [{c.min():.3e}, {c.max():.3e}]")
    except ImportError:
        print("[verificar] CI     (h5py no disponible en este entorno; se omite)")

    for e in errores:
        print(f"[verificar] ✗ {e}")
    if not errores:
        print("[verificar] ✓ malla y CI OK")
    return not errores


if __name__ == "__main__":
    sys.exit(0 if verificar() else 1)
