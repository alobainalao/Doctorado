"""Compara, sobre las matrices reales, L_des = Kx D2x + Kz D2z + (∂xKx)Gx + (∂zKz)Gz
(término desarrollado) con L_conc = Gx Kx Gx + Gz Kz Gz (concatenado).

    python experimentos_continuidad/diferencia_expandido_vs_concatenado.py

Descompone (L_conc − L_des) h = A + B con A = Kx(GxGx − D2x)h + Kz(GzGz − D2z)h y
B = Σ_j g_ij (K_j − K_i)(s_j − s_i) (incrementos de K y de s = G h). Imprime errores
relativos por zona de malla, para campos suaves, polinómicos y de alta frecuencia.
"""
import os, sys, json, pickle
import numpy as np
from scipy.sparse import diags, csr_matrix

AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)
os.chdir(CODIGO); sys.path.insert(0, CODIGO)
cfg = json.load(open("experimentos/base.json"))
cfg.update(metodo="bfr", exp_dir="experimentos_continuidad/resultados/_tmp_dif/bfr", animate=False, save_dat=False)
from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(config=cfg)
from funtions.operators import rbf_op
from preprocessing.preprocess import load_data
from scipy.spatial import cKDTree

d = load_data(); nodes = d.nodes; N = len(nodes); eps_M = d.eps_M; K = d.K
Gx, Gz = d.grad
D2x = rbf_op([2, 0], nodes, nodes, eps_M); D2z = rbf_op([0, 2], nodes, nodes, eps_M)
Kx, Kz = diags(K[:, 0]), diags(K[:, 1])
dKx, dKz = Gx @ K[:, 0], Gz @ K[:, 1]
L_des = Kx @ D2x + Kz @ D2z + diags(dKx) @ Gx + diags(dKz) @ Gz
L_con = Gx @ Kx @ Gx + Gz @ Kz @ Gz
A_op = Kx @ (Gx @ Gx - D2x) + Kz @ (Gz @ Gz - D2z)           # parte A
B_op = (L_con - L_des) - A_op                                 # parte B (incrementos)
dom = np.hstack([d.groups['interior']])                       # filas interiores
h = cKDTree(nodes).query(nodes, k=2)[0][:, 1]                 # espaciado local
zonas = {"fina h≤8": h <= 8, "media 8<h≤25": (h > 8) & (h <= 25), "gruesa h>25": h > 25}
x, z = nodes.T

# campos de prueba
f = np.load("experimentos_continuidad/resultados/B_Sphi_30d/fenicsx/simulation_results.npz", allow_pickle=True)
Hf = np.full(N, np.nan); j = cKDTree(nodes).query(np.asarray(f["nodes"]))[1]; Hf[j] = np.asarray(f["H"]).reshape(144, -1)[-1]
Hf[np.isnan(Hf)] = np.nanmean(Hf)
rng = np.random.default_rng(0)
X, Z = (x - x.min()) / 1e4, (z - z.min()) / 3e3
campos = {
    "H de FEM (día 30, suave)": Hf,
    "cuadrático exacto": 1e-6 * (x - 4000) ** 2 + 2e-6 * (z + 700) ** 2 + 1e-6 * (x - 4000) * (z + 700),
    "sin lento (λ≈2 km)": np.sin(2 * np.pi * x / 2000) * np.cos(2 * np.pi * z / 1500),
    "ruido nodo a nodo (alta frec.)": rng.standard_normal(N),
}
def rel(M, v, mask):
    num = np.abs((M @ v)[mask]); den = np.abs((L_des @ v)[mask])
    return np.sqrt(np.mean(num ** 2)) / max(np.sqrt(np.mean(den ** 2)), 1e-300)

print(f"{'campo':34s} {'zona':14s} ‖Δ‖/‖L_des h‖   ‖A‖/‖L_des h‖  ‖B‖/‖L_des h‖")
for nm, v in campos.items():
    for zn, m in zonas.items():
        mk = np.zeros(N, bool); mk[dom] = True; mk &= m
        print(f"{nm:34s} {zn:14s} {rel(L_con - L_des, v, mk):12.3g} {rel(A_op, v, mk):14.3g} {rel(B_op, v, mk):14.3g}")
# signo del cociente de Rayleigh sobre ruido: <0 difusivo; >0 antidifusivo
v = campos["ruido nodo a nodo (alta frec.)"]; mk = np.zeros(N, bool); mk[dom] = True
for nm, M in [("L_des", L_des), ("L_con", L_con)]:
    print(f"Rayleigh sobre ruido, {nm}: v·Lv/v·v = {(v[mk] @ (M @ v)[mk]) / (v[mk] @ v[mk]):.4g}   (0 = sin amortiguar; <0 amortigua)")
# efecto de K: variación local relativa de K en el stencil
Kv = K[:, 0]; print("K varía: min/max =", Kv.min(), Kv.max(), " (cociente", Kv.max() / Kv.min(), ")")
import shutil; shutil.rmtree("experimentos_continuidad/resultados/_tmp_dif", ignore_errors=True)
