"""Script temporal para verificar J_m antes de la optimización."""
import numpy as np
from funtions.runtime import RUNTIME
from config.parameters import Parameters

RUNTIME.params = Parameters()
p = RUNTIME.get()
p.pre = False

from preprocessing.preprocess import load_data
d = load_data()

from view.animation import solve_forward, update_pozo, get_solucion

Qout_0 = p.Qout[0]
savepath = f"{p.save_data}/simulation_results.npz"

def compute_Jm_at_zp(zp_val, label):
    pozo_c = np.array([3700., zp_val])
    d2 = update_pozo(d, pozo_c)
    solve_forward(d2, [Qout_0], False, True)  # save_data=True
    _, _, C = get_solucion(["H", "U", "C"], savepath)
    Nt = len(Qout_0)
    J1 = 0.0
    Cp_vals = []
    for n in range(Nt):
        Cp = float(d2.delta_p @ np.asarray(C[n]).squeeze())
        Cp_vals.append(Cp)
        J1 += p.gamma * Cp**2 * float(d2.wi.sum()) * float(p.dt)
    J2 = p.koppa * abs(zp_val - d2.z0)**2
    J3 = float(np.sum(Qout_0**2) * p.dt * abs(zp_val - d2.z0))
    print(f"{label} (z_p={zp_val}m): J_m={J1:.4e}, J_e={J2+J3:.4e}, J_tot={J1+J2+J3:.4e}")
    print(f"  Cp: min={min(Cp_vals):.4e}, max={max(Cp_vals):.4e}")
    return J1, J2+J3

print("=" * 60)
Jm_deep, Je_deep = compute_Jm_at_zp(-700., "PROFUNDO")
Jm_surf, Je_surf = compute_Jm_at_zp(220.,  "SUPERFICIE")
print("=" * 60)
if Jm_surf > 0:
    print(f"Ratio J_m(surf)/J_m(deep) = {Jm_surf/max(Jm_deep,1e-30):.2e}")
    print(f"Ratio J_m(surf)/J_e(surf) = {Jm_surf/max(Je_surf,1e-30):.2e}")
    print(f"Pareto activo: {'SI' if Jm_surf > 1e-6*Je_deep else 'NO (J_m muy pequeno)'}")
else:
    print("FALLO: J_m=0 en superficie — la fuente no genera concentracion suficiente")
