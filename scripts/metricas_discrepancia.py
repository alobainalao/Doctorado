"""
Métricas de discrepancia RBF-FD vs elementos finitos del caso M0_adr en la malla con conos (lámina 9 del avance 3):
L² relativa en t = T (la de experimentos/comparar.py), máxima en el tiempo y en espacio-tiempo L²(0,T;L²(Ω)), L¹ y L∞
en t = T, y las cantidades del pozo que usa la optimización: c_w(t) = ∫ g_ε C dΩ y J_m = ∫∫ g_ε C² dΩ dt (ε = r_pozo/3).
Solo lectura de los simulation_results.npz.

    python scripts/metricas_discrepancia.py      # fenics_env

Escribe data/output/experimentos/M_modelo_nuevo/M0_adr/metricas_alternativas.json
"""
import sys, os, pickle, json
import numpy as np
from scipy.spatial import cKDTree
COD = "/home/alex/dev/doctorado/codigo"
sys.path.insert(0, COD); os.chdir(COD)
caso = "M_modelo_nuevo/M0_adr"
cfgb = json.load(open(f"data/output/experimentos/{caso}/bfr/config.json"))
m = pickle.load(open(cfgb["mesh_pkl"], "rb"))
todos = np.asarray(m["nodes"], float)
fant = np.zeros(len(todos), bool)
for k, idx in m["groups"].items():
    if k.startswith("ghost"): fant[np.asarray(idx, int)] = True
keep = np.nonzero(~fant)[0]; o2n = np.full(len(todos), -1); o2n[keep] = np.arange(len(keep))
nodes = todos[keep]; tris = o2n[np.asarray(m["triangles"], np.int64)]
p = nodes[tris]
area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
w = np.bincount(tris.ravel(), weights=np.repeat(area / 3, 3), minlength=len(nodes))
pozo = np.asarray(m["pozo_cor"], float)
zb = np.load(f"data/output/experimentos/{caso}/bfr/simulation_results.npz", allow_pickle=True)
zf = np.load(f"data/output/experimentos/{caso}/fenicsx/simulation_results.npz", allow_pickle=True)
Cb = zb["C"][:, 0][:, keep]; Ub = zb["U"]; Ub = (Ub[:, 0] if Ub.ndim == 4 else Ub)[:, keep]
dist, idx = cKDTree(zf["nodes"]).query(nodes); assert dist.max() < 1e-6
Cf = zf["C"][:, 0][:, idx]; Uf = zf["U"][:, idx]
Vb, Vf = np.linalg.norm(Ub, axis=-1), np.linalg.norm(Uf, axis=-1)
dt = float(zb["dt"]); Nt = Cb.shape[0]
print("Nt", Nt, "dt[h]", dt / 3600, "nodos", len(nodes), "eps", cfgb.get("r_pozo"), cfgb.get("epsilon_p"))
l2 = lambda x: np.sqrt((w * x ** 2).sum(axis=-1))
eps = float(cfgb["r_pozo"]) / 3      # ε = r_pozo/3 del funcional J_m (config/parameters.py)
g = np.exp(-np.sum((nodes - pozo) ** 2, 1) / (2 * eps ** 2)); g /= (g * w).sum()
out = {}
for nom, B, F in (("|u|", Vb, Vf), ("C", Cb, Cf)):
    e_t = l2(B - F) / l2(F)
    st = np.sqrt((l2(B - F) ** 2).sum() / (l2(F) ** 2).sum())
    linf = np.abs(B[-1] - F[-1]).max() / np.abs(F[-1]).max()
    l1 = (w * np.abs(B[-1] - F[-1])).sum() / (w * np.abs(F[-1])).sum()
    out[nom] = dict(final=e_t[-1], max_t=e_t.max(), t_max=(np.argmax(e_t) + 1) * dt / 86400, espacio_tiempo=st, linf_final=linf, l1_final=l1)
# cantidades de interés del pozo: c_w(t) = ∫ g_ε C dΩ y J_m = ∫∫ g_ε C² dΩ dt
cwb, cwf = (Cb * g * w).sum(1), (Cf * g * w).sum(1)
Jb, Jf = (Cb ** 2 * g * w).sum() * dt, (Cf ** 2 * g * w).sum() * dt
out["pozo"] = dict(cw_L2t=np.sqrt(((cwb - cwf) ** 2).sum() / (cwf ** 2).sum()), cw_final=abs(cwb[-1] / cwf[-1] - 1), Jm=abs(Jb / Jf - 1))
Mb, Mf = (Cb * w).sum(1), (Cf * w).sum(1)
out["intC"] = dict(final=abs(Mb[-1] / Mf[-1] - 1), max_t=np.max(np.abs(Mb / Mf - 1)))
for k, v in out.items():
    print(k, {a: (round(100 * b, 3) if a != "t_max" else round(b, 1)) for a, b in v.items()})
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in out.items()},
          open(f"data/output/experimentos/{caso}/metricas_alternativas.json", "w"), indent=1)
