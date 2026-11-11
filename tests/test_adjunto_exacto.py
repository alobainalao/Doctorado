"""
Pruebas del adjunto discreto exacto de BFR (funtions/adjunto_exacto.py).

 1. reconstrucción: las matrices A y B armadas con los estenciles O_k y los coeficientes puntuales coinciden con
    build_transport_matrix (precisión de máquina);
 2. VJP del transporte respecto de H^ℓ y H^{ℓ-1} contra diferencias finitas de los constructores reales;
 3. gradiente completo (Q y z_p) contra diferencias finitas del funcional, con C uniforme y modulada.

    python -m tests.test_adjunto_exacto [--model=adr|mrmt_block|mrmt_semi] [--theta-c=0.5] [--nt=4] [--alm=1] [--sin-dispersion] [--domain=real --q=0.3]
"""
import os
import sys

import numpy as np

_CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters


def _arg(nombre, defecto):
    for a in sys.argv:
        if a.startswith(f"--{nombre}="):
            return type(defecto)(a.split("=", 1)[1])
    return defecto


def entorno():
    nt = _arg("nt", 4)
    return {"metodo": "bfr", "run_type": "optimization", "domain": _arg("domain", "cuadrado"), "mesh_pkl": None,
            "activate_fuente": False, "extract_C": False, "bc_outlet_mode": "libre",
            "lado_cuadrado": 600.0, "C_banda_limpia": 60.0, "Q_std": _arg("q", 0.01), "spacing": _arg("spacing", 30), "min_sp": 4,
            "dt": 36000, "T": nt * 36000, "pre": True, "save_dat": True, "animate": False,
            "theta": _arg("theta", 1.0), "theta_C": _arg("theta-c", 1.0), "model": _arg("model", "adr"),
            "factor_almacenamiento_C": _arg("alm", 1.0), "norm_delta_pozo": _arg("norm", "area"),
            "r_pozo": 20.0,
            **({"Nr": 2, "Deff": [1e-9, 5e-10], "beta": [0.15, 0.10], "phi_im": [0.1, 0.05]}
               if _arg("model", "adr").startswith("mrmt") else {})}


def main():
    RUNTIME.params = Parameters(env=entorno())
    p = RUNTIME.get()
    p.C_init_override = 1.0
    if "--sin-dispersion" in sys.argv:     # D constante: D_disp(U) ∝ |U| no es derivable en U = 0 (nodos estancados)
        p.a_l = p.a_t = 0.0
    if "--modulada" in sys.argv:
        p.C_init_modulacion = (0.5, 2000.0, 800.0)
    from preprocessing.preprocess import load_data
    from view.animation import functional, gradient, pack_controls
    from funtions.operators import build_transport_matrix
    from funtions.utils import D_total, Div_KD
    d = load_data()
    N = len(d.nodes)
    # ---------------- 1. reconstrucción --------------------------------------------------------
    from funtions.adjunto_exacto import OperadoresTransporte, _coefs
    ops = OperadoresTransporte(d)
    from funtions.operators import U_vector
    rng = np.random.default_rng(0)
    H = np.asarray(d.H[0], float) + 0.3 * rng.standard_normal(N)
    Hv = H + 0.2 * rng.standard_normal(N)
    U1 = U_vector(H, d.K, d.grad); U0 = U_vector(Hv, d.K, d.grad)
    D1, D0 = D_total(d.D_f, U1), D_total(d.D_f, U0)
    ht = (H - Hv) / p.dt
    cA, cB = _coefs(d, p, d.pho, U1, D1, D0, ht)
    A = ops.matriz(cA, D1, True); B = ops.matriz(cB, D0, False)
    A0 = build_transport_matrix(U1, d.nodes, d.groups, d.normals, d.pho, D1, Div_KD(D1, d.grad), 0, d.eps_M, d.gauss_p, -1, h_t=ht)
    B0 = build_transport_matrix(U1, d.nodes, d.groups, d.normals, d.pho, D0, Div_KD(D0, d.grad), 0, d.eps_M, d.gauss_p, +1, h_t=ht)
    print(f"[1] reconstrucción  |A-A0|max={abs(A-A0).max():.2e} (|A0|max={abs(A0).max():.2e})   "
          f"|B-B0|max={abs(B-B0).max():.2e}", flush=True)

    # ---------------- 2. VJP del transporte ----------------------------------------------------
    from funtions.adjunto_exacto import _vjp_transporte
    lam = rng.standard_normal(N); Cn = np.abs(rng.standard_normal(N)); Co = np.abs(rng.standard_normal(N))

    def escalar(Hn, Ho):
        Un, Uo = U_vector(Hn, d.K, d.grad), U_vector(Ho, d.K, d.grad)
        Dn_, Do_ = D_total(d.D_f, Un), D_total(d.D_f, Uo)
        h_t = (Hn - Ho) / p.dt
        A_ = build_transport_matrix(Un, d.nodes, d.groups, d.normals, d.pho, Dn_, Div_KD(Dn_, d.grad), 0, d.eps_M, d.gauss_p, -1, h_t=h_t)
        B_ = build_transport_matrix(Un, d.nodes, d.groups, d.normals, d.pho, Do_, Div_KD(Do_, d.grad), 0, d.eps_M, d.gauss_p, +1, h_t=h_t)
        return lam @ (A_ @ Cn) - lam @ (B_ @ Co)

    Un, Uo = U_vector(H, d.K, d.grad), U_vector(Hv, d.K, d.grad)
    Dn_, Do_ = D_total(d.D_f, Un), D_total(d.D_f, Uo)
    gN, gO = _vjp_transporte(d, p, ops, d.pho, lam, Cn, Co, Un, Uo, Dn_, Do_, (H - Hv) / p.dt, nivel_viejo_es_dato=False)
    for nombre, dv, g in (("H^l", np.concatenate([rng.standard_normal(N)]), gN), ("H^{l-1}", rng.standard_normal(N), gO)):
        for e in (1e-3, 1e-4):
            if nombre == "H^l":
                fd = (escalar(H + e * dv, Hv) - escalar(H - e * dv, Hv)) / (2 * e)
            else:
                fd = (escalar(H, Hv + e * dv) - escalar(H, Hv - e * dv)) / (2 * e)
            print(f"[2] vjp {nombre:8s} e={e:.0e}  vjp={g @ dv:+.8e}  fd={fd:+.8e}  err={abs(g @ dv - fd) / abs(fd):.2e}", flush=True)

    # ---------------- 3. gradiente completo ----------------------------------------------------
    p.koppa = 0.0
    p.gamma = 1.0
    Q0 = np.asarray(p.Qout[0], float); zp0 = float(p.pozo[1])
    zp0 = zp0 + _arg("zp-shift", 0.0)    # J(z_p) es discontinua en la posición inicial del pozo (nodo forzado): se evalúa al lado
    x0 = pack_controls(Q0, zp0)
    z0 = float(p.z0)

    def eco(x):
        return abs(x[-1] - z0) * float(np.sum(x[:-1] ** 2)) * p.dt

    J = lambda x: functional(x, d, p, animate=False, save_data=True)
    p.gamma = 0.0; Jeco = J(x0)
    p.gamma = 1.0; Jm1 = J(x0) - Jeco
    p.gamma = 100.0 * max(abs(Jeco), 1e-30) / max(abs(Jm1), 1e-300)
    print(f"[3] gamma={p.gamma:.3e}  J_m(1)={Jm1:.3e}", flush=True)
    p.adjoint_exact = True
    g = gradient(x0, d, p)
    rng = np.random.default_rng(1)
    dirs = {"-g": -g}
    uni = np.zeros_like(x0); uni[:-1] = 1; dirs["Q unif"] = uni
    dirs["Q rnd"] = np.concatenate([rng.standard_normal(len(x0) - 1), [0.0]])
    dirs["(Q,zp) rnd"] = rng.standard_normal(len(x0))
    escala = np.where(np.arange(len(x0)) < len(x0) - 1, max(abs(Q0).max(), 1e-12), 1.0)
    for nombre, dv in dirs.items():
        dv = dv / np.linalg.norm(dv)
        for ef in (3e-3, 1e-3):
            xp, xm = x0 + ef * escala * dv, x0 - ef * escala * dv
            fd = (J(xp) - J(xm)) / (2 * ef)
            ad = float(g @ (escala * dv))
            print(f"[3] {nombre:11s} e={ef:.0e}  adj={ad:+.6e}  fd={fd:+.6e}  err={abs(ad - fd) / abs(fd):.2e}", flush=True)


if __name__ == "__main__":
    main()
