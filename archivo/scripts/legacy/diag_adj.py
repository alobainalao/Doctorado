"""Diagnóstico: estabilidad del adjunto - matriz A_C y B_C."""
import numpy as np
from funtions.runtime import RUNTIME
from config.parameters import Parameters

p = Parameters()
RUNTIME.params = p

from preprocessing.preprocess import load_data
d = load_data()

data = np.load("data/output/bfr/data/real/adr/simulation_results.npz", allow_pickle=True)
U_hist = data["U"]  # (72, K, N, 2)

from funtions.utils import D_total, Div_KD
from funtions.operators import build_adj_C_matrix, build_transport_matrix

k = 0

for n in [71, 40, 10]:
    V_N = U_hist[n][k]
    D_N = D_total(d.D_f, V_N)
    Div_D_N = Div_KD(D_N, d.grad)
    Div_V_N = d.grad[0] @ V_N[:, 0] + d.grad[1] @ V_N[:, 1]

    A_C = build_adj_C_matrix(
        V=V_N, Div_V=Div_V_N, nodes=d.nodes, groups=d.groups,
        normals=d.normals, pho=d.pho, D=D_N, Div_D=Div_D_N,
        Qout=p.Qout[k][n], gauss_p=d.gauss_p, eps_M=d.eps_M, sig=-1
    )
    B_C = build_adj_C_matrix(
        V=V_N, Div_V=Div_V_N, nodes=d.nodes, groups=d.groups,
        normals=d.normals, pho=d.pho, D=D_N, Div_D=Div_D_N,
        Qout=p.Qout[k][n], gauss_p=d.gauss_p, eps_M=d.eps_M, sig=+1
    )
    # Forward para comparar
    A_fwd = build_transport_matrix(
        V_N, d.nodes, d.groups, d.normals, d.pho, D_N, Div_D_N,
        p.Qout[k][n], d.eps_M, d.gauss_p, sig=-1
    )

    dA = A_C.diagonal()
    dB = B_C.diagonal()
    dF = A_fwd.diagonal()

    print(f"\n=== step n={n} ===")
    print(f"A_C diag: min={np.min(dA):.4e}  max={np.max(dA):.4e}  mean={np.mean(dA):.4e}")
    print(f"B_C diag: min={np.min(dB):.4e}  max={np.max(dB):.4e}  mean={np.mean(dB):.4e}")
    print(f"A_fwd diag: min={np.min(dF):.4e}  max={np.max(dF):.4e}  mean={np.mean(dF):.4e}")
    print(f"Div_V: min={np.min(Div_V_N):.4e}  max={np.max(Div_V_N):.4e}")
    print(f"2*pho/dt (base): {2*np.mean(d.pho)/p.dt:.4e}")

    # Amplificación estimada con adjunto construido a mano
    from scipy.sparse.linalg import splu
    lu = splu(A_C)
    x = np.random.randn(A_C.shape[0]); x /= np.linalg.norm(x)
    for _ in range(15):
        y = lu.solve(B_C @ x); lam = np.linalg.norm(y); x = y / lam
    print(f"||A_C^{{-1}} B_C|| (manual, 15 its): {lam:.4e}")

    # Amplificación con adjunto discreto exacto (A_fwd^T, B_fwd^T)
    D_n_loc = D_total(d.D_f, U_hist[n-1][k] if n > 0 else U_hist[n][k])
    Div_D_n_loc = Div_KD(D_n_loc, d.grad)
    V_n_loc = U_hist[n-1][k] if n > 0 else U_hist[n][k]

    A_fwd2 = build_transport_matrix(
        V_N, d.nodes, d.groups, d.normals, d.pho, D_N, Div_D_N,
        p.Qout[k][n], d.eps_M, d.gauss_p, sig=-1
    )
    B_fwd2 = build_transport_matrix(
        V_n_loc, d.nodes, d.groups, d.normals, d.pho, D_n_loc, Div_D_n_loc,
        p.Qout[k][n-1 if n > 0 else n], d.eps_M, d.gauss_p, sig=+1
    )
    AT = A_fwd2.T.tocsr()
    BT = B_fwd2.T.tocsr()
    # Dirichlet BC psi=0 en inlet/outlet
    bc_idx = np.hstack([d.groups['boundary:inlet'], d.groups['boundary:outlet']])
    AT = AT.tolil(); BT = BT.tolil()
    for i in bc_idx:
        AT[i, :] = 0; AT[i, i] = 1
        BT[i, :] = 0
    AT = AT.tocsr(); BT = BT.tocsr()
    lu2 = splu(AT)
    x = np.random.randn(AT.shape[0]); x /= np.linalg.norm(x)
    for _ in range(15):
        y = lu2.solve(BT @ x); lam2 = np.linalg.norm(y); x = y / lam2
    print(f"||A_fwd^T^{{-1}} B_fwd^T|| (exacto, 15 its): {lam2:.4e}")
