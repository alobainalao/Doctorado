
import numpy as np
from scipy.sparse.linalg import splu

# Dependencias externas (ya las tienes en otros módulos)
from funtions.operators import H_vector, U_vector, build_transport_matrix, C_vector
from funtions.utils import D_total, Div_KD
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
    C_sol = C_new.copy()   # antes del recorte: el residuo del transporte del adjunto exacto se evalúa con ella

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

    return H, U, C_new, C_im_new, C_sol
