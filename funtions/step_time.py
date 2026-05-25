
import numpy as np
from scipy.sparse.linalg import splu

# Dependencias externas (ya las tienes en otros módulos)
from funtions.operators import H_vector, U_vector, build_transport_matrix, C_vector, build_adj_C_matrix, build_adj_H_matrix, build_adj_C_rhs, build_adj_H_rhs
from funtions.utils import D_total, Div_KD
from funtions.runtime import RUNTIME


def step_time(H, U, C, C_im, nodes, groups, normals, A_solver, eps_M, K, grad,
              pho, D_f, A_right, delta_p, gauss_p, gauss_f, Qout_n, Qout_N,
              t, T, exp_lam_dt):
    p = RUNTIME.get()
    
    N = len(nodes)
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))
    
    D_n = D_total(D_f, U)
    Div_D_n = Div_KD(D_n, grad)


    # --- FLOW ---
    rhs = H_vector(H, A_right, delta_p, groups, nodes, N, None, Qout_n, Qout_N)
    H = A_solver.solve(rhs)

    U = U_vector(H, K, grad)

    # --- TRANSPORT ---
    D_N = D_total(D_f, U)
    Div_D_N = Div_KD(D_N, grad)

    C_solver = build_transport_matrix(U, nodes, groups, normals, pho, D_N, Div_D_N, Qout_N, eps_M, gauss_p, -1)
    C_right  = build_transport_matrix(U, nodes, groups, normals, pho, D_n, Div_D_n, Qout_n, eps_M, gauss_p, 1)
    rhs = C_vector(C, C_right, gauss_f, groups, t, T, N, C_im, pho)

    C_new = splu(C_solver).solve(rhs)

    C_im_new = np.zeros_like(C_im)

    if p.model == "mrmt_semi":
        # --- MRMT ---
        C_new = C_new[None, :]

        for r in range(p.Nr):
            C_im_new = C_new + (C_im - C_new) * exp_lam_dt[r]

            C_new += np.sum(
                p.beta[r] * (C_im - C_im_new),
                axis=0
            )

        C_new = C_new.squeeze()        

    elif p.model == "mrmt_block":

        coef = p.dt * p.alpha_r / p.beta 

        C_im_new[:, idx] = (
            coef[:, None] * (C_new[idx] + C[idx])
            + (1 - 2*coef[:, None]) * C_im[:, idx]
        )


    return H, U, C_new, C_im_new


# =========================================================
# TIME STEP ADJOINT
# =========================================================
def step_adjoint(
    A_solver,
    B,

    psiH_N,
    psiC_N,

    H_n,
    H_N,
    V_n,
    V_N,
    C_n,
    C_N,

    S_s,
    K,
    Div_K,

    pho,

    D,

    Qout_n,
    Qout_N,

    gauss_p,
    gamma,
    delta_p,

    grad,

    nodes,
    groups,
    normals,
    eps_M
):

    p = RUNTIME.get()
    # =====================================================
    # ψ_C
    # =====================================================

    # -----------------------------------------------------
    # velocity fields
    # V = K∇h
    # -----------------------------------------------------
    D_n = D_total(D, V_n)
    Div_D_n = Div_KD(D_n, grad)

    D_N = D_total(D, V_N)
    Div_D_N = Div_KD(D_N, grad)

    Div_V_n = (
        grad[0] @ V_n[:, 0]
        +
        grad[1] @ V_n[:, 1]
    )

    Div_V_N = (
        grad[0] @ V_N[:, 0]
        +
        grad[1] @ V_N[:, 1]
    )

    # -----------------------------------------------------
    # A ψ_C^n
    # -----------------------------------------------------
    A_C = build_adj_C_matrix(
        V=V_n,
        Div_V=Div_V_n,

        nodes=nodes,
        groups=groups,
        normals=normals,

        pho=pho,
        D=D_n,
        Div_D=Div_D_n,

        Qout=Qout_n,
        gauss_p=gauss_p,

        eps_M=eps_M,

        sig=-1
    )

    # -----------------------------------------------------
    # B ψ_C^{n+1}
    # -----------------------------------------------------
    B_C = build_adj_C_matrix(
        V=V_N,
        Div_V=Div_V_N,

        nodes=nodes,
        groups=groups,
        normals=normals,

        pho=pho,
        D=D_N,
        Div_D=Div_D_N,

        Qout=Qout_N,
        gauss_p=gauss_p,
        
        eps_M=eps_M,

        sig=+1
    )

    # -----------------------------------------------------
    # RHS ψ_C
    # -----------------------------------------------------
    rhs_C = build_adj_C_rhs(
        B=B_C,
        psiC_N=psiC_N,

        C_n=C_n,
        C_N=C_N,

        gamma=gamma,
        delta_p=delta_p,

        nodes=nodes,
        groups=groups
    )

    # -----------------------------------------------------
    # solve ψ_C
    # -----------------------------------------------------
    psiC_n = splu(A_C).solve(rhs_C)

    # -----------------------------------------------------
    # RHS ψ_H
    # -----------------------------------------------------
    rhs_H = build_adj_H_rhs(
        B=B,
        psiH_N=psiH_N,

        psiC_n=psiC_n,
        psiC_N=psiC_N,

        C_n=C_n,
        C_N=C_N,

        K=K,

        grad=grad,

        nodes=nodes,
        groups=groups
    )

    # -----------------------------------------------------
    # solve ψ_H
    # -----------------------------------------------------
    psiH_n = A_solver.solve(rhs_H)


    return psiH_n, psiC_n

