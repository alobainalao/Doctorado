
import numpy as np

# Dependencias externas (ya las tienes en otros módulos)
from funtions.operators import H_vector, U_vector, build_transport_matrix, C_vector
from funtions.utils import D_total, Div_KD
from funtions.runtime import RUNTIME


def step_time(H, C, C_im, nodes, groups, normals, A_solver, eps_M, K, grad,
              pho, D_f, A_right, delta_p, gauss_p, gauss_f, Qout,
              t, T, mask_h, exp_lam_dt):
    p = RUNTIME.get()
    
    N = len(nodes)
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))
    

    # --- FLOW ---
    rhs = H_vector(H, A_right, delta_p, groups, nodes, N, None, Qout)
    H_ = A_solver.solve(rhs)
    H = np.where(mask_h, H, H_)

    U = U_vector(H, K, grad)

    # --- TRANSPORT ---
    D = D_total(D_f, U)
    Div_D = Div_KD(D, grad)

    C_solver = build_transport_matrix(U, nodes, groups, normals, pho, D, Div_D, eps_M, gauss_p, -1)
    C_right  = build_transport_matrix(U, nodes, groups, normals, pho, D, Div_D, eps_M, gauss_p, 1)
    rhs = C_vector(C, C_right, gauss_f, groups, t, T, N, C_im, pho)

    C_new = C_solver.solve(rhs)

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

