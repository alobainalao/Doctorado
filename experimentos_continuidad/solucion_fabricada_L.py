"""Solución fabricada: ¿cuál operador (desarrollado o concatenado) aproxima ∇·(K∇h)?

    python experimentos_continuidad/solucion_fabricada_L.py

h(x,z), Kx(x,z), Kz(x,z) analíticos y suaves; el valor exacto de ∂x(Kx hx)+∂z(Kz hz) sale de
sympy. Se evalúa el error de L_des y de L_con contra el exacto, en nodos interiores, por zona de malla
y para dos contrastes de K. Referencia = verdad analítica, no un operador contra el otro.
"""
import os, sys, json
import numpy as np, sympy as sp
from scipy.sparse import diags
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__)); CODIGO = os.path.dirname(AQUI)
os.chdir(CODIGO); sys.path.insert(0, CODIGO)
cfg = json.load(open("experimentos/base.json"))
cfg.update(metodo="bfr", exp_dir="experimentos_continuidad/resultados/_tmp_sf/bfr", animate=False, save_dat=False)
from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(config=cfg)
from funtions.operators import rbf_op
from preprocessing.preprocess import load_data

d = load_data(); nodes = d.nodes; N = len(nodes); eps_M = d.eps_M
Gx, Gz = d.grad
D2x = rbf_op([2, 0], nodes, nodes, eps_M); D2z = rbf_op([0, 2], nodes, nodes, eps_M)
x, z = nodes.T
hloc = cKDTree(nodes).query(nodes, k=2)[0][:, 1]
zonas = {"fina h≤8": hloc <= 8, "media 8<h≤25": (hloc > 8) & (hloc <= 25), "gruesa h>25": hloc > 25}
inter = np.zeros(N, bool); inter[d.groups['interior']] = True
# contraste real de K en el interior
print("K real: min/max =", d.K[inter, 0].min(), d.K[inter, 0].max())

X, Z = sp.symbols("x z")
h_s = sp.sin(2 * sp.pi * X / 3000) * sp.cos(2 * sp.pi * Z / 1500)
for contraste in [10.0, 1e3, 1e6]:
    a = sp.log(contraste)
    Kx_s = sp.exp(a * sp.Rational(1, 2) * sp.sin(2 * sp.pi * X / 4000) * sp.cos(2 * sp.pi * Z / 2500))
    Kz_s = Kx_s / 3
    ex_s = sp.diff(Kx_s * sp.diff(h_s, X), X) + sp.diff(Kz_s * sp.diff(h_s, Z), Z)
    f = lambda e: sp.lambdify((X, Z), e, "numpy")(x, z) * np.ones(N)
    h, Kx_v, Kz_v, ex = f(h_s), f(Kx_s), f(Kz_s), f(ex_s)
    dKx, dKz = Gx @ Kx_v, Gz @ Kz_v                           # ∇K como lo hace el código (numérico)
    L_des = diags(Kx_v) @ D2x + diags(Kz_v) @ D2z + diags(dKx) @ Gx + diags(dKz) @ Gz
    L_con = Gx @ diags(Kx_v) @ Gx + Gz @ diags(Kz_v) @ Gz
    esc = np.sqrt(np.mean(ex[inter] ** 2))
    print(f"\nContraste de K (máx/mín) ≈ {Kx_v.max()/Kx_v.min():.3g}    ‖exacto‖_rms = {esc:.3g}")
    print(f"  {'zona':14s} {'error L_des':>12s} {'error L_con':>12s}   (rms del error / rms del exacto)")
    for zn, m in zonas.items():
        mk = inter & m
        e1 = np.sqrt(np.mean(((L_des @ h) - ex)[mk] ** 2)) / np.sqrt(np.mean(ex[mk] ** 2))
        e2 = np.sqrt(np.mean(((L_con @ h) - ex)[mk] ** 2)) / np.sqrt(np.mean(ex[mk] ** 2))
        print(f"  {zn:14s} {e1:12.3g} {e2:12.3g}")
import shutil; shutil.rmtree("experimentos_continuidad/resultados/_tmp_sf", ignore_errors=True)
