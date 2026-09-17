
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
        # Operador splitting secuencial: cada región r actualiza C_im[r] con su
        # propia tasa exp_lam_dt[r] y corrige C_new acumulativamente.
        for r in range(p.Nr):
            e_r = exp_lam_dt[r]
            C_im_new_r = C_new + (C_im[r] - C_new) * e_r
            C_new = C_new + p.beta[r] * (C_im[r] - C_im_new_r)
            C_im_new[r] = C_im_new_r
        C_new = np.maximum(C_new, 0.0)

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
    eps_M,

    psiC_im_N=None,   # (Nr, N) adjunto de C_im en n+1; None para ADR
    exp_lam_dt=None,  # tasas precomputadas para MRMT Semi
):

    p = RUNTIME.get()
    # =====================================================
    # ψ_C  —  adjunto discreto exacto: A_fwd^T, B_fwd^T
    # =====================================================

    # dispersion tensors
    D_n = D_total(D, V_n)
    D_N = D_total(D, V_N)
    Div_D_N = Div_KD(D_N, grad)
    Div_D_n = Div_KD(D_n, grad)

    # reconstruir A_fwd y B_fwd exactamente como en step_time
    A_fwd = build_transport_matrix(
        V_N, nodes, groups, normals, pho, D_N, Div_D_N,
        Qout_N, eps_M, gauss_p, sig=-1
    )
    B_fwd = build_transport_matrix(
        V_N, nodes, groups, normals, pho, D_n, Div_D_n,
        Qout_n, eps_M, gauss_p, sig=+1
    )

    # transponer
    AT = A_fwd.T.tocsr()
    BT = B_fwd.T.tocsr()

    # Dirichlet psi_C = 0 en inlet y outlet
    bc_idx = np.hstack([groups['boundary:inlet'], groups['boundary:outlet']])
    AT = AT.tolil(); BT = BT.tolil()
    for i in bc_idx:
        AT[i, :] = 0; AT[i, i] = 1.0
        BT[i, :] = 0
    AT = AT.tocsr(); BT = BT.tocsr()

    idx_dom = np.hstack((
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall'],
    ))

    psiC_im_n = None

    # --------------------------------------------------
    # MRMT Semi: backward por el operador splitting ANTES
    # del solve de transporte. La transparencia discreta
    # del split da:
    #   ψ_C_arr_r = (1−β_r α_r)·ψ_C_arr_{r+1} + α_r·ψ_C_im_r^{n+1}
    #   ψ_C_im_r^n = β_r α_r·ψ_C_arr_{r+1} + e_r·ψ_C_im_r^{n+1}
    # donde α_r = 1 − exp(−λ_r Δt), e_r = exp(−λ_r Δt).
    # --------------------------------------------------
    if p.model == "mrmt_semi" and psiC_im_N is not None:
        Nr = p.Nr
        beta = np.asarray(p.beta, float)
        psiC_im_n = np.zeros_like(psiC_im_N)
        psiC_arr = psiC_N.copy()
        for r in range(Nr - 1, -1, -1):
            e_r = float(exp_lam_dt[r])
            alpha_r = 1.0 - e_r
            psi_im_r = psiC_im_N[r]
            psiC_arr_new = (1.0 - beta[r] * alpha_r) * psiC_arr + alpha_r * psi_im_r
            psiC_im_n[r] = beta[r] * alpha_r * psiC_arr + e_r * psi_im_r
            psiC_arr = psiC_arr_new
        psiC_N_transport = psiC_arr
    else:
        psiC_N_transport = psiC_N

    # RHS ψ_C
    rhs_C = BT @ psiC_N_transport
    rhs_C[idx_dom] += (
        2.0 * gamma
        * (C_N[idx_dom] + C_n[idx_dom])
        * delta_p[idx_dom]
    )

    # --------------------------------------------------
    # MRMT Block: agregar coef_r·ψ_C_im_r^{n+1} al RHS
    # (adjunto de ∂C_im_r^{n+1}/∂C^{n+1} = coef_r).
    # --------------------------------------------------
    if p.model == "mrmt_block" and psiC_im_N is not None:
        coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
        for r in range(p.Nr):
            rhs_C[idx_dom] += coef[r] * psiC_im_N[r][idx_dom]

    rhs_C[bc_idx] = 0.0   # Dirichlet: psi_C = 0

    # solve ψ_C
    psiC_n = splu(AT).solve(rhs_C)

    # --------------------------------------------------
    # MRMT Block: ψ_C_im_r^n tras el solve del transporte.
    #   ψ_C_im_r^n = (1−2·coef_r)·ψ_C_im_r^{n+1}
    #              + 4R·pho·α_r · ψ_C^n
    # El segundo término es el adjunto de la fuente
    # 4R·α_r·pho·C_im_r^n en el RHS del transporte.
    # --------------------------------------------------
    if p.model == "mrmt_block" and psiC_im_N is not None:
        coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
        alpha_r_arr = np.asarray(p.alpha_r, float)
        psiC_im_n = np.zeros_like(psiC_im_N)
        for r in range(p.Nr):
            psiC_im_n[r][idx_dom] = (
                (1.0 - 2.0 * coef[r]) * psiC_im_N[r][idx_dom]
                + 4.0 * p.R * pho[idx_dom] * alpha_r_arr[r] * psiC_n[idx_dom]
            )

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

    # solve ψ_H
    psiH_n = A_solver.solve(rhs_H)

    return psiH_n, psiC_n, psiC_im_n

