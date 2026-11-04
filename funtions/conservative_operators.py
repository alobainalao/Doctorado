import numpy as np
from scipy.sparse import diags


def kernel_pozo(p, delta_p, gauss_p):
    """Núcleo del sumidero de flujo w_R: el mismo que usa H_vector."""
    if gauss_p is not None and getattr(p, "smooth_well_h", False):
        return gauss_p
    return delta_p


def build_conservative_transport_matrix(V, nodes, groups, normals,
                                        pho, D, Qout, eps_M,
                                        kernel_p, sig, p, grad):
    """
    Matriz de transporte en forma conservativa (sin desarrollar la divergencia).

    Ecuación (ec. transporte_conservativo, W_+ = 0 en el operador):
        φR ∂_t C + λ φR C + ∇·(u C − D∇C) + Q w_R C = W_+

    Se discretiza como flujo y luego divergencia, por nodo:
        F = u C − D ∇C,   ∇·F ≈ Dx(F_x) + Dy(F_y)
    con los mismos operadores Dx, Dy (RBF-FD) de la malla (la parte advectiva
    puede ser upwind, ver abajo). No aparece ∇·D ni
    ∇·u: ambos quedan dentro de la divergencia. El término Q w_R C compensa el
    C∇·u = −Q w_R C que contiene ∇·(uC) (queda u·∇C si ∇·u = −Q w_R), así que
    se incluye siempre que haya extracción, también en optimización.

    Misma escala temporal que build_transport_matrix: identidad 2φR/Δt y peso
    w = 2θ_C (lado implícito, sig<0) o 2(1−θ_C) (explícito, sig>0).
    """
    from funtions.operators import (
        apply_boundary_conditions, make_transport_bc, peso_theta_C
    )

    N = len(nodes)
    w = peso_theta_C(p, sig)
    idx_dom = np.hstack([
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall'],
    ])

    Gx, Gy = grad

    # ∇·(u C) − ∇·(D∇C) como producto de matrices ralas (flujo → divergencia)
    # Con upwind_transport el flujo advectivo uC se deriva con los stencils
    # direccionales (p._Dx_upw, p._Dy_upw, preconstruidos en main.py).
    Ax, Ay = Gx, Gy
    if getattr(p, 'upwind_transport', False):
        Ax, Ay = p._Dx_upw, p._Dy_upw
    L = (Ax @ diags(V[:, 0]) + Ay @ diags(V[:, 1])
         - Gx @ diags(D[:, 0]) @ Gx - Gy @ diags(D[:, 1]) @ Gy)

    base = pho * p.R
    coef_identity = base * (2. / p.dt)
    if p.activate_ext and Qout:
        coef_sink = np.asarray(Qout * kernel_p, float)
    else:
        coef_sink = np.zeros(N)

    # Todos los términos locales en C entran con el signo de −w·sig, como en
    # build_transport_matrix (LHS: +w·L, RHS: −w·L).
    coef_loc = coef_sink.copy()
    if p.model == "adr":
        coef_loc += base * p.landa
    elif p.model == "mrmt_block":
        coef_loc += 2 * base * p.alpha_sum
    elif p.model != "mrmt_semi":
        raise ValueError(f"Modelo no válido: {p.model}")

    A_dom = (diags(coef_identity)
             - w * sig * (L + diags(coef_loc)))
    A_dom = A_dom.tocsr()[idx_dom]

    bc_definitions = None if sig > 0 else make_transport_bc(normals, D)
    return apply_boundary_conditions(
        [(A_dom, idx_dom)], bc_definitions, nodes, groups, p, eps_M, N
    )
