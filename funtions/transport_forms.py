"""Formulaciones alternativas del transporte en BFR (experiment/transport-continuity-
substitution). Ver experimentos_continuidad/informe/C_derivacion_y_unidades.md.

  A  "conservative"    φR∂ₜC = ∇·(D∇C) − ∇·(qC) − Q w_R C − λφRC (+W₊)
  B  "continuity"      φR∂ₜC = ∇·(D∇C) − q·∇C + Ss ∂ₜh C − λφRC (+W₊)
  C  "continuity_noSs" B con Ss ∂ₜh = 0
"""
import numpy as np
from scipy.sparse import diags


def specific_storage(p, pho):
    """Mismo criterio que preprocessing/preprocess.py: S_s_const o S_s = φ."""
    n = len(pho)
    return np.full(n, float(p.S_s_const)) if getattr(p, "S_s_const", None) else pho


def storage_rate(p, pho, H_old, H_new):
    """κ = Ss (H^{n+1} − H^n)/Δt  [1/T], coeficiente de C en la sustitución de continuidad."""
    return specific_storage(p, pho) * (H_new - H_old) / p.dt


def add_storage_term(A, kappa, groups, p, sig):
    """Suma el término fuente +κ·C a la matriz de transporte ya ensamblada (cualquier
    esquema, también upwind). Como el decaimiento −λφR C entra con coef_identity
    −= w·sig·(φRλ), una fuente +κ C entra con coef_identity += w·sig·κ, con el
    mismo κ en ambos lados (θ_C·C^{n+1} + (1−θ_C)·C^n)."""
    from funtions.operators import peso_theta_C
    idx = np.hstack([groups['interior'], groups['boundary:inlet'],
                     groups['boundary:outlet'], groups['boundary:wall']])
    d = np.zeros(A.shape[0])
    d[idx] = peso_theta_C(p, sig) * sig * kappa[idx]
    return (A + diags(d)).tocsc()


def well_kernel(p, delta_p, gauss_p):
    """w_R: el mismo núcleo que H_vector usa como sumidero del flujo."""
    if gauss_p is not None and getattr(p, "smooth_well_h", False):
        return gauss_p
    return delta_p


def build_conservative_transport_matrix(V, nodes, groups, normals, pho, D, Qout,
                                        eps_M, w_R, sig, p, grad):
    """Formulación A en flujo→divergencia: ∇·(uC − D∇C) con Dx, Dy RBF-FD y sumidero
    Q w_R C (mismo w_R que el flujo). D diagonal (Dxx, Dzz) como en make_transport_operator.
    Escala CN idéntica a build_transport_matrix (identidad 2φR/Δt, peso peso_theta_C)."""
    from funtions.operators import (apply_boundary_conditions, make_transport_bc,
                                    peso_theta_C)
    if getattr(p, "upwind_transport", False):
        raise NotImplementedError("transport_form='conservative' no soporta upwind_transport")
    N = len(nodes)
    w = peso_theta_C(p, sig)
    idx = np.hstack([groups['interior'], groups['boundary:inlet'],
                     groups['boundary:outlet'], groups['boundary:wall']])
    Gx, Gy = grad
    L = (Gx @ diags(V[:, 0]) + Gy @ diags(V[:, 1])
         - Gx @ diags(D[:, 0]) @ Gx - Gy @ diags(D[:, 1]) @ Gy)

    base = pho * p.R
    loc = np.zeros(N)
    if p.activate_ext and Qout:
        loc += np.asarray(Qout * w_R, float)
    if p.model == "adr":
        loc += base * p.landa
    else:
        raise NotImplementedError("transport_form alternativo solo para model='adr'")

    A = (diags(base * (2. / p.dt)) - w * sig * (L + diags(loc))).tocsr()[idx]
    bc = None if sig > 0 else make_transport_bc(normals, D)
    return apply_boundary_conditions([(A, idx)], bc, nodes, groups, p, eps_M, N)
