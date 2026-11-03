"""
Grilla de videos de CUALQUIER conjunto de corridas, con UNA sola paleta de color para toda la grilla.

Lee los `simulation_results.npz` ya guardados (la animación se genera a partir de lo que se obtuvo, no de
una corrida aparte), calcula (vmin, vmax) del campo sobre TODOS los paneles y TODOS los instantes y dibuja
una única barra de color. Exige el mismo `dt` en todos los paneles; si el número de pasos difiere usa el
menor y lo avisa. Sale una tabla con los rangos y los pasos mostrados para poder comprobar la correspondencia.

    python -m experimentos.grilla --campo C --cols 2 \
        S_sumidero/S1_con_termino:bfr S_sumidero/S1_con_termino:fenicsx \
        S_sumidero/S0_sin_termino:bfr S_sumidero/S0_sin_termino:fenicsx \
        --salida data/output/experimentos/grids/S_C.mp4

Cada panel es `grupo/caso:backend` (backend = bfr | fenicsx). Campos: H, C, V.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.animar import (  # noqa: E402
    CAMPOS, LABEL_B, OUT_ROOT, WRITER_KW, _actualizar, _cargar, _panel, _rangos, _t_dias,
)


def _resolver(spec):
    """`grupo/caso:backend` -> (ruta del npz, etiqueta)."""
    caso, _, backend = spec.rpartition(":")
    if backend not in LABEL_B or not caso:
        raise SystemExit(f"panel mal formado: {spec!r} (use grupo/caso:backend)")
    ruta = os.path.join(OUT_ROOT, caso, backend, "simulation_results.npz")
    if not os.path.exists(ruta):
        raise SystemExit(f"no hay resultados: {ruta}")
    return ruta, f"{LABEL_B[backend]}  ·  {caso.split('/')[-1]}"


def grilla(paneles, campo, cols, salida):
    etiqueta, cmap = {c: (e, m) for c, e, m in CAMPOS}[campo]
    rutas, titulos = zip(*[_resolver(s) for s in paneles])
    datos = [_cargar(r) for r in rutas]

    dts = {d["dt"] for d in datos}
    if len(dts) != 1:
        raise SystemExit(f"los paneles tienen dt distintos {sorted(dts)}: no se pueden sincronizar")
    nts = [d[campo].shape[0] for d in datos]
    nt = min(nts)
    if len(set(nts)) != 1:
        print(f"[grilla] AVISO: pasos distintos {nts}; se muestran los primeros {nt}", flush=True)

    # Rango único: todos los paneles y todos los instantes (los que se muestran).
    rec = [dict(d, **{campo: d[campo][:nt]}) for d in datos]
    vmin, vmax = _rangos(rec)[campo]
    print(f"[grilla] {campo}: paleta única [{vmin:.4e}, {vmax:.4e}] sobre {len(datos)} paneles × {nt} pasos", flush=True)
    for t, n, (a, b) in zip(titulos, nts, [(d[campo].min(), d[campo].max()) for d in datos]):
        print(f"           {t:55s} pasos={n:4d}  min={a:.4e}  max={b:.4e}", flush=True)

    filas = int(np.ceil(len(datos) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(8 * cols, 3.8 * filas + 1.2), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.22, wspace=0.03, hspace=0.18)
    pan = []
    for k, (d, t) in enumerate(zip(datos, titulos)):
        ax = axs[divmod(k, cols)]
        pan.append(_panel(ax, d, campo, vmin, vmax, cmap))
        ax.set_title(t, fontsize=11)
    for k in range(len(datos), filas * cols):
        axs[divmod(k, cols)].axis("off")
    cax = fig.add_axes([0.3, 0.12, 0.4, 0.025])
    fig.colorbar(pan[0][0], cax=cax, orientation="horizontal", label=etiqueta,
                 extend="max" if campo == "V" else "neither")
    sup = fig.suptitle("", fontsize=14)

    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    writer = FFMpegWriter(**WRITER_KW)
    with writer.saving(fig, salida, dpi=100):
        for k in range(nt):
            for (im, q), d in zip(pan, datos):
                _actualizar(im, q, d, campo, k)
            sup.set_text(f"{etiqueta}   |   t = {_t_dias(datos[0], k):.2f} días   (paso {k + 1}/{nt})")
            writer.grab_frame()
    fig.savefig(os.path.splitext(salida)[0] + "_final.png", dpi=100)
    plt.close(fig)
    print("[grilla] video:", salida, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paneles", nargs="+", help="grupo/caso:backend (en orden de lectura)")
    ap.add_argument("--campo", default="C", choices=[c for c, _, _ in CAMPOS])
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    grilla(a.paneles, a.campo, a.cols, a.salida)


if __name__ == "__main__":
    main()
