"""Script temporal para correr los 3 escenarios de optimización."""
import numpy as np
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funtions.runtime import RUNTIME
from config.parameters import Parameters


def set_p(env):
    RUNTIME.params = Parameters(env=env)
    p = RUNTIME.get()
    import view.animation as _a
    _a.p = p
    return p


BASE = dict(
    run_type="optimization", domain="real", model="adr",
    spacing=30, dt=36000, T=2592000,
    pre=True, save_dat=True, animate=False, postproc=False,
    gamma=1.0, koppa=1.0, opt_maxiter=8,
)

SCENARIOS = [
    ("conservador", {**BASE, "Q_max": 5e-3}),
    ("agresivo",    {**BASE, "Q_max": 1e-2}),
    ("profundo",    {**BASE, "Q_max": 5e-3, "koppa": 2.0}),
]

os.makedirs("scripts", exist_ok=True)
summary = []

# Primer set_p antes de importar animation
set_p(SCENARIOS[0][1])

from preprocessing.preprocess import load_data
from view.animation import optimize_bfr, unpack_controls

for name, env in SCENARIOS:
    print(f"\n{'='*55}")
    print(f"ESCENARIO: {name}")
    print(f"{'='*55}")

    p = set_p(env)
    d = load_data()

    t0 = time.time()
    res = optimize_bfr(d, p)
    t1 = time.time()

    Q_opt, zp_opt = unpack_controls(res.x)
    print(f"[{name}] J*={res.fun:.4e}  zp*={zp_opt:.3f}  "
          f"t={t1-t0:.0f}s  nit={res.nit}  ok={res.success}")

    # figura de convergencia
    npz = f"{p.save_data}/optimization_results.npz"
    if os.path.exists(npz):
        data = np.load(npz, allow_pickle=True)
        J_hist = data["J"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(J_hist, "o-", markersize=3)
        ax.set_xlabel("Iteracion")
        ax.set_ylabel("J")
        ax.set_title(f"Convergencia — {name}")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"scripts/convergencia_{name}.png", dpi=150)
        plt.close(fig)
        print(f"  figura: scripts/convergencia_{name}.png")

    summary.append((name, float(res.fun), float(zp_opt), res.nit, res.success))

print("\n" + "="*55)
print("RESUMEN FINAL")
print("="*55)
print(f"{'escenario':12s}  {'J*':>12}  {'zp*':>8}  {'iters':>5}  ok")
for name, J, zp, nit, ok in summary:
    print(f"{name:12s}  {J:12.4e}  {zp:8.3f}  {nit:5d}  {ok}")
