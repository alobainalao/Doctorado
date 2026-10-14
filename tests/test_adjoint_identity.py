"""
Test de consistencia adjunta por paso de tiempo (BFR / RBF-FD)

Para métodos de colocación RBF-FD el adjunto discreto NO tiene una relación
algebraica simple con la transpuesta de la matriz forward (al contrario que
en FEM Galerkin donde A_adj = A_fwd^T exactamente). El adjunto RBF-FD es la
discretización INDEPENDIENTE de la EDP adjunta continua.

La identidad válida es la LAGRANGIANA DISCRETA:
    ∂J/∂Q_n = ψ_C^n · ∂f_C^n/∂Q_n  (contribución del paso n al gradiente)

Esto puede verificarse STEP-BY-STEP perturbando solo Q_n y comparando con FD
centralizada. Es un superset del gradient check global: si cada componente
pasa, el gradiente adjunto es correcto paso a paso.

Ventaja sobre el gradient check global: localiza si la discrepancia está
concentrada en ciertos pasos de tiempo (inicio, final, etc.).

Uso (bfr_env):
    python -m tests.test_adjoint_identity [adr|mrmt_semi|mrmt_block]
    python -m tests.test_adjoint_identity          # corre los tres modelos

Criterio: error relativo de cada componente ∂J/∂Q_n < TOL para algún δ.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters

# ───────────────────────────────────────────────────────────────────────────
# Configuración: horizonte MUY corto (Nt=3) para que el test corra en < 2 min
# ───────────────────────────────────────────────────────────────────────────
BASE_ENV = {
    "metodo": "bfr",
    "run_type": "optimization",
    "domain": "real",
    "spacing": 80,
    "dt": 36000,
    "T": 3 * 36000,      # Nt = 3 pasos: suficiente para inicio/medio/fin
    "save_dat": True,
    "animate": False,
    "pre": True,
}

MRMT_PARAMS = {
    "Nr": 2,
    "Deff":   [1e-9, 5e-10],
    "beta":   [0.15, 0.10],
    "phi_im": [0.1,  0.05],
}

DELTA_FRACS  = [1e-2, 1e-3, 1e-4, 1e-5]
TOL          = 5e-3   # igual que el MRMT gradient check
TARGET_J1    = 1e12


def make_env(model: str) -> dict:
    env = {**BASE_ENV, "model": model}
    if model.startswith("mrmt"):
        env.update(MRMT_PARAMS)
    return env


def run_model(model: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Consistencia adjunta paso-a-paso — BFR / model={model}")
    print(f"{'='*60}")

    RUNTIME.params = Parameters(env=make_env(model))
    p = RUNTIME.get()

    from preprocessing.preprocess import load_data
    from view.animation import gradient, functional

    d = load_data()
    Q0  = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0  = np.concatenate([Q0, [zp0]])
    Nt  = len(Q0)

    # Auto-escala γ: dominan los términos observacionales
    p.gamma = 1.0
    p.koppa = 0.0
    J0_raw  = functional(x0, d, p, animate=False, save_data=True)
    p.gamma = TARGET_J1 / max(abs(J0_raw), 1e-14) if abs(J0_raw) > 0 else 1.0
    print(f"  J0_raw={J0_raw:.3e}  γ auto={p.gamma:.3e}  Nt={Nt}")

    # Gradiente adjunto (una sola resolución forward + adjunto)
    g_adj = gradient(x0, d, p)
    grad_Q = g_adj[:-1]

    # ── Test paso a paso: comprobar ∂J/∂Q_n para n ∈ {0, Nt//2, Nt-1} ──
    indices = sorted(set([0, Nt // 2, Nt - 1]))
    all_pass = True
    results  = {}

    for n in indices:
        g_n = float(grad_Q[n])
        scale = abs(x0[n]) if abs(x0[n]) > 1e-14 else 1.0
        errs  = []
        print(f"\n  --- paso n={n}  g_adj={g_n:.4e} ---")
        for frac in DELTA_FRACS:
            delta = frac * scale
            xp = x0.copy(); xp[n] += delta
            xm = x0.copy(); xm[n] -= delta
            fd = (functional(xp, d, p, animate=False, save_data=True)
                  - functional(xm, d, p, animate=False, save_data=True)) / (2 * delta)
            rel = abs(fd - g_n) / max(abs(g_n), 1e-14)
            errs.append(rel)
            print(f"    δ={delta:.2e}  FD={fd:.4e}  err_rel={rel:.2e}")
        results[f"n={n}"] = errs
        if min(errs) >= TOL:
            all_pass = False

    # ── Figura ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl, errs in results.items():
        ax.loglog(DELTA_FRACS, errs, marker="o", label=lbl)
    ax.axhline(TOL, color="k", ls="--", label=f"tol={TOL:.0e}")
    ax.set_xlabel(r"$\delta / |x_n|$")
    ax.set_ylabel("error relativo")
    ax.set_title(f"Consistencia adjunta paso-a-paso — bfr/{model}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = f"adjoint_identity_bfr_{model}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\n  Figura: {fname}")

    # ── Resumen ───────────────────────────────────────────────────────────
    print("\n  Resumen:")
    for lbl, errs in results.items():
        me = min(errs)
        st = "OK" if me < TOL else "FALLA"
        print(f"    {lbl}:  error_min={me:.2e}  [{st}]")
    print("  Identidad OK." if all_pass else "  FALLA — revisar step_adjoint / gradient en funtions/.")
    return all_pass


def main():
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = ["adr", "mrmt_semi", "mrmt_block"]

    summary = {}
    for model in models:
        summary[model] = run_model(model)

    print(f"\n{'='*60}")
    print("RESUMEN GLOBAL — Consistencia adjunta paso-a-paso")
    for m, ok in summary.items():
        print(f"  {m:15s}  {'OK' if ok else 'FALLA'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
