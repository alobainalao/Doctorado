
import numpy as np
from scipy.sparse.linalg import splu

# Dependencias externas (ya las tienes en otros módulos)
from funtions.operators import H_vector, U_vector, build_transport_matrix, C_vector, build_adj_C_matrix, build_adj_H_matrix, build_adj_C_rhs, build_adj_H_rhs
from funtions.utils import D_total, Div_KD, upwind_diffusion
from funtions.runtime import RUNTIME


def step_time(H, U, C, C_im, nodes, groups, normals, A_solver, eps_M, K, grad,
              pho, D_f, A_right, delta_p, gauss_p, gauss_f, Qout_n, Qout_N,
              t, T, exp_lam_dt):
    p = RUNTIME.get()

    N = len(nodes)
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    _upwind = getattr(p, 'upwind_transport', False)

    D_n = D_total(D_f, U)
    Div_D_n = Div_KD(D_n, grad)

    # --- FLOW ---
    H_old = H
    rhs = H_vector(H, A_right, delta_p, groups, nodes, N, None, Qout_n, Qout_N, gauss_p=gauss_p)
    H = A_solver.solve(rhs)

    U = U_vector(H, K, grad)

    # --- TRANSPORT ---
    D_N = D_total(D_f, U)
    Div_D_N = Div_KD(D_N, grad)
    h_t = (H - H_old) / p.dt   # ∂ₜh del paso (término de almacenamiento)

    if _upwind:
        from funtions.upwind_operators import build_upwind_transport_matrix
        C_solver = build_upwind_transport_matrix(U, nodes, groups, normals, pho, D_N, Div_D_N, Qout_N, eps_M, gauss_p, -1, p, h_t=h_t)
        C_right  = build_upwind_transport_matrix(U, nodes, groups, normals, pho, D_n, Div_D_n, Qout_n, eps_M, gauss_p,  1, p, h_t=h_t)
    else:
        C_solver = build_transport_matrix(U, nodes, groups, normals, pho, D_N, Div_D_N, Qout_N, eps_M, gauss_p, -1, h_t=h_t)
        C_right  = build_transport_matrix(U, nodes, groups, normals, pho, D_n, Div_D_n, Qout_n, eps_M, gauss_p, 1, h_t=h_t)
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

    elif p.model == "mrmt_block":

        coef = p.dt * p.alpha_r / p.beta 
        th = float(getattr(p, "theta_C", 0.5))

        # C_im^{n+1} = C_im^n + 2·coef·(θ_C·C^{n+1} + (1−θ_C)·C^n − C_im^n);
        # θ_C = 0.5 → coef·(C^{n+1} + C^n) + (1 − 2·coef)·C_im^n (CN original).
        C_im_new[:, idx] = (
            2 * coef[:, None] * (th * C_new[idx] + (1 - th) * C[idx])
            + (1 - 2*coef[:, None]) * C_im[:, idx]
        )

    # Positividad física: C ≥ 0 en cada paso (mismo criterio que fenicsx/forward.py)
    if getattr(p, "clip_C", True):
        C_new = np.maximum(C_new, 0.0)

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
    wi,

    grad,

    nodes,
    groups,
    normals,
    eps_M,

    psiC_im_N=None,   # (Nr, N) adjunto de C_im en n+1; None para ADR
    exp_lam_dt=None,  # tasas precomputadas para MRMT Semi
    forced_well_node=-1,   # índice del nodo forzado en p.pozo (-1 si no hay)
    forced_well_nbrs=None,  # vecinos Poisson-disc para IDW
    forced_well_idw=None,   # pesos IDW normalizados
    C_nm1=None,             # C^{n-1}: fuente de almacenamiento del adjunto del flujo (None en n=0)
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
    h_t = (H_N - H_n) / p.dt   # ∂ₜh del paso: término de almacenamiento -S_s ∂ₜh·C

    # reconstruir A_fwd y B_fwd exactamente como en step_time
    A_fwd = build_transport_matrix(
        V_N, nodes, groups, normals, pho, D_N, Div_D_N,
        Qout_N, eps_M, gauss_p, sig=-1, h_t=h_t
    )
    B_fwd = build_transport_matrix(
        V_N, nodes, groups, normals, pho, D_n, Div_D_n,
        Qout_n, eps_M, gauss_p, sig=+1, h_t=h_t
    )

    # transponer
    AT = A_fwd.T.tocsr()
    BT = B_fwd.T.tocsr()

    # Dirichlet psi_C = 0 en inlet y outlet. Con flujo dispersivo nulo en todo el borde (BC natural
    # del transporte) la transpuesta exacta no lo impone: adj_C_bc_exacta=True la deja intacta.
    bc_idx = np.hstack([groups['boundary:inlet'], groups['boundary:outlet']])
    if not getattr(p, "adj_C_bc_exacta", False):
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

    # RHS ψ_C  —  ∂J₁/∂C = 2γ · gauss_p · wi · C  (adjunto exacto de J₁ = γ∫gₚC²dΩ)
    rhs_C = BT @ psiC_N_transport
    rhs_C[idx_dom] += (
        2.0 * gamma
        * (C_N[idx_dom] + C_n[idx_dom])
        * gauss_p[idx_dom] * wi[idx_dom]
    )

    # --------------------------------------------------
    # MRMT Block: agregar coef_r·ψ_C_im_r^{n+1} al RHS
    # Transpuesto de ∂C_im_r^{n+1}/∂C^{n+1} = κ_r (= dt·α_r/β_r).
    # Demostración: en el bloque L^T del sistema adjunto, la fila C tiene
    # -κ_r·I acoplada a C_im_r, que entra como +κ_r·ψ_im_r^{n+1} en rhs_C
    # antes de resolver A_C^T·φ_C = rhs_C.
    # --------------------------------------------------
    if p.model == "mrmt_block" and psiC_im_N is not None:
        # ∂C_im^{n+1}/∂C^{n+1} = 2·θ_C·coef (= coef con θ_C = 0.5).
        coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
        coef = 2.0 * float(getattr(p, "theta_C", 0.5)) * coef
        for r in range(p.Nr):
            rhs_C[idx_dom] += coef[r] * psiC_im_N[r][idx_dom]

    if not getattr(p, "adj_C_bc_exacta", False):
        rhs_C[bc_idx] = 0.0   # Dirichlet: psi_C = 0

    # solve ψ_C
    psiC_n = splu(AT).solve(rhs_C)

    # Corrección post-solve: el nodo forzado (r=0) tiene estencil RBF-FD degradado
    # (todos sus vecinos están a r≥16m) y produce una oscilación espuria en ψ_C.
    # Se reemplaza su valor por la interpolación IDW de sus vecinos Poisson-disc,
    # lo que equivale a tratarlo como nodo de interpolación (no colocation) en el adjunto.
    if forced_well_node >= 0 and forced_well_nbrs is not None:
        psiC_n[forced_well_node] = np.dot(forced_well_idw, psiC_n[forced_well_nbrs])

    # --------------------------------------------------
    # MRMT Block: ψ_C_im_r^n tras el solve del transporte.
    #   ψ_C_im_r^n = (1−2·coef_r)·ψ_C_im_r^{n+1}
    #              + 2R·pho·α_r · ψ_C^n
    # El segundo término es el adjunto de la fuente
    # 2R·α_r·pho·C_im_r^n en el RHS del transporte.
    # --------------------------------------------------
    if p.model == "mrmt_block" and psiC_im_N is not None:
        coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
        alpha_r_arr = np.asarray(p.alpha_r, float)
        psiC_im_n = np.zeros_like(psiC_im_N)
        for r in range(p.Nr):
            psiC_im_n[r][idx_dom] = (
                (1.0 - 2.0 * coef[r]) * psiC_im_N[r][idx_dom]
                + 2.0 * p.R * pho[idx_dom] * alpha_r_arr[r] * psiC_n[idx_dom]
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
        groups=groups,
        V_n=V_n,
        V_N=V_N
    )

    # Fuente de almacenamiento del adjunto del flujo (doc. adjunto_sin_sumidero, ec. adj_flujo):
    # derivada de ψ_C^T(A C^N − B C^n) respecto a H^n a través de c_s = −S_s ∂ₜh (S_s = φ),
    #   σ^n ∝ ψ_C^N(w_N C^N + w_n C^n) − ψ_C^n(w_N C^n + w_n C^{n−1}),   rhs_H += (S_s/2) σ^n.
    # La normalización (S_s/2, sin 1/dt ni 1/wi) es la de ψ_C del código (fuente 2γ g wi (C_N+C_n)):
    # contrastada con diferencias finitas (docs/adjunto_sin_sumidero.tex, §código).
    f_alm = getattr(p, "factor_almacenamiento_C", 0.0)
    if f_alm != 0.0:
        from funtions.operators import peso_theta_C
        w_N = peso_theta_C(p, -1); w_n = peso_theta_C(p, +1)
        sig_alm = psiC_N * (w_N * C_N + w_n * C_n)
        if C_nm1 is not None:
            sig_alm = sig_alm - psiC_n * (w_N * C_n + w_n * C_nm1)
        else:
            sig_alm = sig_alm - psiC_n * (w_N + w_n) * C_n
        ii = groups['interior']
        rhs_H[ii] += getattr(p, "_alm_src_scale", 0.5) * f_alm * pho[ii] * sig_alm[ii]

    # solve ψ_H
    psiH_n = A_solver.solve(rhs_H)

    return psiH_n, psiC_n, psiC_im_n

