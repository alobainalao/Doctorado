"""
Prueba de humo del forward FEM (fenicsx).

Objetivo: solo confirmar que solve_forward_fenicsx corre sin crashear, con
malla gruesa, pocos pasos de tiempo y Qout=0 (sin extracción). No verifica
resultados físicos todavía — eso es un paso posterior (ver fenicsx/README.md,
"Lo que falta": verificación cruzada bfr<->fenicsx).

Uso (dentro del contenedor dolfinx, ver fenicsx/README.md "Cómo correrlo"):
    cd /workspace
    python -m fenicsx.test_forward_smoke
"""
import numpy as np

from funtions.runtime import RUNTIME
from config.parameters import Parameters
from fenicsx.forward import solve_forward_fenicsx

SMOKE_ENV = {
    "metodo": "fenicsx",
    "run_type": "standard",
    "domain": "real",
    "spacing": "200",
    "dt": "1e5",
    "T": "3e5",
    "save_dat": "False",
    "animate": "False",
}


def main():
    RUNTIME.params = Parameters(env=SMOKE_ENV)
    p = RUNTIME.get()

    Qout = np.zeros(p.Nt)
    pozo = p.pozo

    print(f"Nt={p.Nt}, dt={p.dt}, pozo={pozo}, spacing={p.spacing}")
    print("Corriendo solve_forward_fenicsx (sin extracción, malla gruesa)...")

    h_n, C_n, C_out, C_total = solve_forward_fenicsx(Qout, pozo, p)

    print("OK — no crasheó.")
    print(f"h_n: min={h_n.x.array.min():.4g} max={h_n.x.array.max():.4g}")
    print(f"C_n: min={C_n.x.array.min():.4g} max={C_n.x.array.max():.4g}")
    print(f"C_total por paso: {C_total}")
    print(f"C_out por paso: {C_out}")


if __name__ == "__main__":
    main()
