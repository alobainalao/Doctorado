"""
Test de Taylor para verificar el gradiente adjunto — 6 versiones del modelo.

El test de Taylor es la verificación canónica de gradientes en control óptimo:
dada una dirección dm, se comprueba que las tasas de convergencia sean:

    r0(h) = |J(m + h·dm) − J(m)|                       → O(h),  ratio ≈ 2
    r1(h) = |J(m + h·dm) − J(m) − h·⟨∇J(m), dm⟩|      → O(h²), ratio ≈ 4

Si r1 no muestra convergencia cuadrática, el gradiente adjunto es incorrecto.

Versiones cubiertas (2 backends × 3 modelos):

    backend  model         entorno
    -------  ------------  ----------
    bfr      adr           bfr_env
    bfr      mrmt_semi     bfr_env
    bfr      mrmt_block    bfr_env
    fenicsx  adr           fenics_env
    fenicsx  mrmt_semi     fenics_env
    fenicsx  mrmt_block    fenics_env

Uso:
    # Solo BFR (bfr_env):
    conda activate bfr_env
    python -m tests.test_taylor_adjoint [bfr] [adr|mrmt_semi|mrmt_block]

    # Solo FEniCS (fenics_env):
    /home/alex/miniconda3/envs/fenics_env/bin/python -m tests.test_taylor_adjoint fenicsx [adr|mrmt_semi|mrmt_block]

    # Todos los BFR (bfr_env):
    python -m tests.test_taylor_adjoint bfr

    # Todos los FEniCS:
    /home/alex/miniconda3/envs/fenics_env/bin/python -m tests.test_taylor_adjoint fenicsx

Criterio: ratio r1 ≥ 3.5 en al menos 2 refinamientos consecutivos (O(h²)).
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

# ─────────────────────────── Configuración común ────────────────────────────
# Horizonte corto para que cada variante corra en minutos.
BASE_BFR = {
    "metodo": "bfr",
    "run_type": "optimization",
    "domain": "real",
    "spacing": 80,
    "dt": 36000,
    "T": 3 * 36000,
    "save_dat": True,
    "animate": False,
    "pre": True,
}

BASE_FENICSX = {
    "metodo": "fenicsx",
    "run_type": "optimization",
    "domain": "real",
    "spacing": 120,     # malla más gruesa: fenicsx es más caro
    "dt": 36000,
    "T": 3 * 36000,
    "save_dat": False,
    "animate": False,
    "postproc": False,
    "pre": True,
}

MRMT_PARAMS = {
    "Nr": 2,
    "Deff":   [1e-9, 5e-10],
    "beta":   [0.15, 0.10],
    "phi_im": [0.1,  0.05],
}

# Pasos de refinamiento: h, h/2, h/4, h/8, h/16  (h0 relativo al control)
H_FRACS   = [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3]
TOL_RATIO = 3.5    # ratio mínimo de r1 para considerar convergencia O(h²)
TARGET_J  = 1e12   # escala de referencia para γ auto

# ─────────────────────────── Helpers ────────────────────────────────────────

def make_env(backend: str, model: str) -> dict:
    base = BASE_BFR if backend == "bfr" else BASE_FENICSX
    env = {**base, "model": model}
    if model.startswith("mrmt"):
        env.update(MRMT_PARAMS)
    return env


def _rng_direction(x0: np.ndarray, seed: int = 42) -> np.ndarray:
    """Dirección aleatoria normalizada (reproducible)."""
    rng = np.random.default_rng(seed)
    dm = rng.standard_normal(x0.shape)
    return dm / np.linalg.norm(dm)


# ─────────────────────────── Backend BFR ────────────────────────────────────

def taylor_bfr(model: str) -> bool:
    print(f"\n{'='*62}")
    print(f"  Test de Taylor — BFR / model={model}")
    print(f"{'='*62}")

    RUNTIME.params = Parameters(env=make_env("bfr", model))
    p = RUNTIME.get()

    from preprocessing.preprocess import load_data
    from view.animation import functional, gradient, pack_controls, unpack_controls

    d = load_data()
    Q0  = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0  = pack_controls(Q0, zp0)

    # Auto-escala γ
    p.gamma = 1.0
    p.koppa = 0.0
    J0_raw = functional(x0, d, p, animate=False, save_data=True)
    p.gamma = TARGET_J / max(abs(J0_raw), 1e-14) if abs(J0_raw) > 0 else 1.0
    print(f"  J0_raw={J0_raw:.3e}  γ={p.gamma:.3e}")

    J0    = functional(x0, d, p, animate=False, save_data=True)
    g0    = gradient(x0, d, p)
    dm    = _rng_direction(x0)
    slope = float(g0 @ dm)

    print(f"  ⟨∇J, dm⟩ = {slope:.6e}")
    print(f"\n  {'h':>10}  {'r0':>12}  {'r1':>12}  {'ratio_r0':>10}  {'ratio_r1':>10}")

    scale = max(abs(x0).max(), 1.0)
    r0_prev = r1_prev = None
    r0s, r1s = [], []
    ok = False

    for frac in H_FRACS:
        h  = frac * scale
        xh = x0 + h * dm
        Jh = functional(xh, d, p, animate=False, save_data=True)
        r0 = abs(Jh - J0)
        r1 = abs(Jh - J0 - h * slope)
        r0s.append(r0); r1s.append(r1)
        ratio_r0 = r0_prev / r0 if r0_prev and r0 > 0 else float("nan")
        ratio_r1 = r1_prev / r1 if r1_prev and r1 > 0 else float("nan")
        print(f"  {h:>10.3e}  {r0:>12.3e}  {r1:>12.3e}  {ratio_r0:>10.2f}  {ratio_r1:>10.2f}")
        r0_prev, r1_prev = r0, r1

    # Criterio: ≥2 ratios de r1 ≥ TOL_RATIO
    ratios_r1 = [r0s[i] / r1s[i+1] if r1s[i+1] > 0 else 0
                 for i in range(len(r1s)-1)]
    # Recalcular ratios correctamente: r1[i]/r1[i+1]
    ratios_r1 = [r1s[i] / r1s[i+1] if r1s[i+1] > 0 else 0
                 for i in range(len(r1s)-1)]
    n_ok = sum(1 for r in ratios_r1 if r >= TOL_RATIO)
    ok = n_ok >= 2

    # Figura
    hs = [frac * scale for frac in H_FRACS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs, r0s, "o-", label=r"$r_0$ (O(h))")
    ax.loglog(hs, r1s, "s-", label=r"$r_1$ (O(h²))")
    h_ref = np.array(hs)
    ax.loglog(h_ref, r0s[0] / hs[0] * h_ref,   "k--", lw=0.8, label="O(h)")
    ax.loglog(h_ref, r1s[0] / hs[0]**2 * h_ref**2, "k:",  lw=0.8, label="O(h²)")
    ax.set_xlabel("h"); ax.set_ylabel("residual")
    ax.set_title(f"Test de Taylor — bfr/{model}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = f"taylor_bfr_{model}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Figura: {fname}")
    print(f"  {'OK ✓' if ok else 'FALLA ✗'}  (ratios r1 ≥ {TOL_RATIO}: {n_ok}/{ len(ratios_r1) })")
    return ok


# ─────────────────────────── Backend FEniCS ─────────────────────────────────

def taylor_fenicsx(model: str) -> bool:
    print(f"\n{'='*62}")
    print(f"  Test de Taylor — FEniCS / model={model}")
    print(f"{'='*62}")

    RUNTIME.params = Parameters(env=make_env("fenicsx", model))
    p = RUNTIME.get()

    from fenicsx.mesh import get_mesh
    from fenicsx.adjoint import (
        build_context, functional_fenicsx, gradient_fenicsx, _unpack
    )

    pozo = list(p.pozo)
    mesh_cache = get_mesh(pozo, p.spacing, regenerate=bool(getattr(p, "pre", False)))
    ctx = build_context(pozo, p, mesh_cache)

    Q0  = np.asarray(p.Qout[0], float)
    zp0 = float(p.pozo[1])
    x0  = np.concatenate([Q0, [zp0]])

    J0 = functional_fenicsx(x0, ctx)
    g0 = gradient_fenicsx(x0, ctx)
    dm = _rng_direction(x0)
    slope = float(g0 @ dm)

    print(f"  J0={J0:.3e}  ⟨∇J, dm⟩={slope:.6e}")
    print(f"\n  {'h':>10}  {'r0':>12}  {'r1':>12}  {'ratio_r0':>10}  {'ratio_r1':>10}")

    scale = max(abs(x0).max(), 1.0)
    r0_prev = r1_prev = None
    r0s, r1s = [], []

    for frac in H_FRACS:
        h  = frac * scale
        xh = x0 + h * dm
        Jh = functional_fenicsx(xh, ctx)
        r0 = abs(Jh - J0)
        r1 = abs(Jh - J0 - h * slope)
        r0s.append(r0); r1s.append(r1)
        ratio_r0 = r0_prev / r0 if r0_prev and r0 > 0 else float("nan")
        ratio_r1 = r1_prev / r1 if r1_prev and r1 > 0 else float("nan")
        print(f"  {h:>10.3e}  {r0:>12.3e}  {r1:>12.3e}  {ratio_r0:>10.2f}  {ratio_r1:>10.2f}")
        r0_prev, r1_prev = r0, r1

    ratios_r1 = [r1s[i] / r1s[i+1] if r1s[i+1] > 0 else 0
                 for i in range(len(r1s)-1)]
    n_ok = sum(1 for r in ratios_r1 if r >= TOL_RATIO)
    ok = n_ok >= 2

    hs = [frac * scale for frac in H_FRACS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs, r0s, "o-", label=r"$r_0$ (O(h))")
    ax.loglog(hs, r1s, "s-", label=r"$r_1$ (O(h²))")
    h_ref = np.array(hs)
    ax.loglog(h_ref, r0s[0] / hs[0] * h_ref,       "k--", lw=0.8, label="O(h)")
    ax.loglog(h_ref, r1s[0] / hs[0]**2 * h_ref**2, "k:",  lw=0.8, label="O(h²)")
    ax.set_xlabel("h"); ax.set_ylabel("residual")
    ax.set_title(f"Test de Taylor — fenicsx/{model}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = f"taylor_fenicsx_{model}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Figura: {fname}")
    print(f"  {'OK ✓' if ok else 'FALLA ✗'}  (ratios r1 ≥ {TOL_RATIO}: {n_ok}/{len(ratios_r1)})")
    return ok


# ─────────────────────────── main ───────────────────────────────────────────
# IMPORTANTE: view.animation y preprocessing.preprocess fijan `p = RUNTIME.params`
# a nivel de módulo al importarse. Si se corre más de un modelo en la misma
# sesión Python, el segundo modelo hereda el `p` del primero (bug conocido del
# diseño del simulador). La solución es lanzar cada modelo en un subproceso
# independiente cuando se piden varios modelos; para un solo modelo se corre
# directamente (modo "worker") para no añadir overhead.

import subprocess

_THIS_MODULE = "tests.test_taylor_adjoint"


def _run_single(backend: str, model: str) -> bool:
    """Lanza este mismo módulo como subproceso con un solo modelo."""
    if backend == "fenicsx":
        python = "/home/alex/miniconda3/envs/fenics_env/bin/python"
    else:
        python = "/home/alex/miniconda3/envs/bfr_env/bin/python"
    cmd = [python, "-m", _THIS_MODULE, "--worker", backend, model]
    result = subprocess.run(cmd, cwd=_CODIGO)
    return result.returncode == 0


def main():
    args = sys.argv[1:]

    # ── Modo worker: un solo modelo, corre directamente ──
    if args and args[0] == "--worker":
        backend, model = args[1], args[2]
        run_fn = taylor_fenicsx if backend == "fenicsx" else taylor_bfr
        ok = run_fn(model)
        sys.exit(0 if ok else 1)

    # ── Modo orquestador: parsear y lanzar subprocesos ──
    if not args:
        try:
            import dolfinx  # noqa: F401
            backend = "fenicsx"
        except ImportError:
            backend = "bfr"
        models = ["adr", "mrmt_semi", "mrmt_block"]
    elif args[0] in ("bfr", "fenicsx"):
        backend = args[0]
        models  = [args[1]] if len(args) > 1 else ["adr", "mrmt_semi", "mrmt_block"]
    else:
        backend = "bfr"
        models  = [args[0]]

    summary = {}
    for model in models:
        key = f"{backend}/{model}"
        print(f"\n{'─'*62}")
        print(f"  Lanzando subproceso: {key}")
        print(f"{'─'*62}")
        summary[key] = _run_single(backend, model)

    print(f"\n{'='*62}")
    print("RESUMEN — Test de Taylor (gradiente adjunto)")
    print(f"{'='*62}")
    for key, ok in summary.items():
        estado = "OK ✓" if ok else "FALLA ✗"
        print(f"  {key:25s}  {estado}")
    print(f"{'='*62}")
    all_ok = all(summary.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
