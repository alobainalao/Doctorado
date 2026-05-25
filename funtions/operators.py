import numpy as np
from scipy.sparse.linalg import splu
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows
from scipy.sparse import csr_matrix, csc_matrix
from funtions.runtime import RUNTIME
from funtions.utils import In_h, apply_boundary, fuente_C

# ---------------------------
# 3) Operadores RBF-FD entre mallas
# ---------------------------
def rbf_op(diffs, x_eval, x_src, eps_arr, n=None):
    """
    Construye matriz que evalúa derivadas `diffs` definidas en puntos x_src
    y las evalúa en x_eval. Devuelve CSR tamaño (len(x_eval), N_src).
    """
    p = RUNTIME.get()

    if n is None:
        n = p.n_stencil
    W = weight_matrix(x=x_eval, p=x_src, n=n, 
                      diffs=[diffs], phi=p.phi, order=p.order,
                      eps=eps_arr, chunk_size=None)
    return csr_matrix(W)

def build_operators(nodes, eps_arr):
    """
    Construye operadores RBF-FD sobre una sola malla de nodos.
    Devuelve los gradientes y laplacianos necesarios para resolver
    las ecuaciones de flujo y transporte acopladas en el mismo dominio.
    """
    p = RUNTIME.get()

    # Derivadas primeras (gradiente)
    Dx = rbf_op([1, 0], nodes, nodes, eps_arr)

    Dy = rbf_op([0, 1], nodes, nodes, eps_arr)

    # # Derivadas segundas (laplaciano)
    # L = rbf_op([2, 0], N, nodes, nodes, eps_arr) + rbf_op([0, 2], N, nodes, nodes, eps_arr)

    # Gradiente vectorial y laplaciano vectorial (para conveniencia)
    grad = [Dx, Dy]
    # L_vec = bmat([[L, None],
    #               [None, L]])

    return grad


# =========================================================
# 1. ENSAMBLADOR GLOBAL
# =========================================================
def apply_boundary_conditions(blocks, bc_definitions, nodes, groups, p, eps_M, N):

    if bc_definitions is not None:
        for bc in bc_definitions:
            blocks.append(
                bc_block(
                    nodes, groups, p, eps_M,
                    bc["name"],
                    coeffs_func=bc["coeffs_func"]
                )
            )

    A = csc_matrix((N, N))
    for block in blocks:
        if block is None:
            continue
        A_temp, idx = block
        A += expand_rows(A_temp, idx, N)
    return A.tocsc()

# =========================================================
# 2. OPERADOR BASE (DOMINIO)
# =========================================================
def build_operator_block(nodes, p, operator_def):
    idx = operator_def["idx"]
    eps = operator_def["eps"]
    diffs = operator_def["diffs"]
    coeffs = operator_def["coeffs_func"](idx)

    order = operator_def.get("order", p.order)
    n = operator_def.get("n", p.n_stencil)

    return weight_matrix(
        x=nodes[idx],
        p=nodes,
        n=n,
        diffs=diffs,
        coeffs=coeffs,
        phi=p.phi,
        order=order,
        eps=eps[idx],
        chunk_size=None
    ), idx

# =========================================================
# 3. BC GENERAL (FACTORY PURA)
# =========================================================
def bc_block(nodes, groups, p, eps, bc_name, coeffs_func):
    idx = groups.get(f'boundary:{bc_name}', None)
    ghosts = groups.get(f'ghosts:{bc_name}', idx)

    if idx is None or len(idx) == 0:
        return None

    diffs_local, coeffs_local = coeffs_func(idx)

    A_temp = weight_matrix(
        x=nodes[idx],
        p=nodes,
        n=p.n_stencil,
        diffs=diffs_local,
        coeffs=coeffs_local,
        phi=p.phi,
        order=p.order,
        eps=eps[idx],
        chunk_size=None
    )

    return A_temp, ghosts

def build_matrix(
    nodes,
    groups,
    p,
    op_def,
    bc_definitions,
    eps,
    N,
    sig
):
    # -----------------------------
    # OPERATOR
    # -----------------------------
    blocks = [build_operator_block(nodes, p, op_def)]

    # -----------------------------
    # BC (solo si sig < 0)
    # -----------------------------
    if sig > 0:
        bc_definitions = None

    return apply_boundary_conditions(
        blocks,
        bc_definitions,
        nodes,
        groups,
        p,
        eps,
        N
    )

# =========================================================
# 4. BUILD H MATRIX (UNIFICADO)
# =========================================================
def make_H_operator(nodes, groups, theta, S_s, K, Div_K,
                    eps_M, eps_arr, sig, p, operator):

    idx_dom = np.hstack([
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ])

    OPERATORS = {
        "identity": lambda: dict(
            idx=idx_dom,
            eps=eps_arr,
            diffs=[[0, 0]],
            coeffs_func=lambda i: [1],
            order=0,
            n=1
        ),

        "full": lambda: dict(
            idx=idx_dom,
            eps=eps_M,
            diffs=[
                [0, 0],
                [2, 0],
                [0, 2],
                [1, 0],
                [0, 1]
            ],
            coeffs_func=lambda i: [
                S_s[i] / p.dt,
                theta * sig * K[i, 0],
                theta * sig * K[i, 1],
                theta * sig * Div_K[i, 0],
                theta * sig * Div_K[i, 1],
            ]
        )
    }


    key = "identity" if operator == "Identity" else "full"


    return OPERATORS[key]()

def make_H_bc(nodes, groups, K, Div_K, normals):

    return [
        dict(
            name="inlet",
            coeffs_func=lambda i: (
                [[1, 0]],
                [-K[i, 0]]
            )
        ),

        dict(
            name="wall",
            coeffs_func=lambda i: (
                [[1, 0], [0, 1]],
                [
                    -K[i, 0] * normals[i, 0],
                    -K[i, 1] * normals[i, 1]
                ]
            )
        ),

        dict(
            name="outlet",
            diffs=[
                [1, 0],
                [0, 1],
                [2, 0],
                [0, 2],
                [1, 1]
            ],
            coeffs_func=lambda i: (
                [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]],
                [
                    -(Div_K[i, 0] * normals[i, 0] + Div_K[i, 2] * normals[i, 1]),
                    -(Div_K[i, 3] * normals[i, 0] + Div_K[i, 1] * normals[i, 1]),
                    -(K[i, 0] * normals[i, 0]),
                    -(K[i, 1] * normals[i, 1]),
                    -(K[i, 0] * normals[i, 1] - K[i, 1] * normals[i, 0])
                ]
            )
        )
    ]

def build_H_matrix(nodes, groups, normals, theta, S_s, K, Div_K,
                   eps_M, sig=1, eps_arr=None, operator=None):

    p = RUNTIME.get()
    N = len(nodes)

    op_def = make_H_operator(
        nodes, groups, theta, S_s, K, Div_K,
        eps_M, eps_arr, sig, p, operator
    )

    bc_definitions = None if sig > 0 else make_H_bc(
        nodes, groups, K, Div_K, normals
    )

    return build_matrix(
        nodes, groups, p,
        op_def,
        bc_definitions,
        eps_M,
        N,
        sig
    )


# =========================================================
# 5. BUILD TRANSPORT MATRIX (UNIFICADO)
# =========================================================
def make_transport_operator(V, nodes, groups, pho, D, Div_D, Qout, p, eps_M, sig, gauss_p):
    
    U_div_D = (V - Div_D[:, :2]) / 2.0

    idx_dom = np.hstack([
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ])

    base = pho * p.R
    coef_identity = base * (2. / p.dt)

    if p.activate_ext and Qout and p.run_type != "optimization":
        coef_identity -= sig * gauss_p

    if p.model == "adr":
        coef_identity -= base * sig * p.landa
    elif p.model == "mrmt_block":
        coef_identity -= 2 * base * sig * p.alpha_sum
    elif p.model != "mrmt_semi":
        raise ValueError(f"Modelo no válido: {p.model}")

    return dict(
        idx=idx_dom,
        eps=eps_M,
        diffs=[
            [0, 0],
            [2, 0],
            [0, 2],
            [1, 0],
            [0, 1]
        ],
        coeffs_func=lambda i: [
            coef_identity[i],
            sig * D[i, 0],
            sig * D[i, 1],
            -sig * U_div_D[i, 0],
            -sig * U_div_D[i, 1],
        ]
    )

def make_transport_bc(normals, D):

    return [
        dict(
            name=tag,
            coeffs_func=lambda i: (
                [[1, 0], [0, 1]],
                [
                    -D[i, 0] * normals[i, 0],
                    -D[i, 1] * normals[i, 1]
                ]
            )
        )
        for tag in ["inlet", "wall", "outlet"]
    ]

def build_transport_matrix(V, nodes, groups, normals,
                           pho, D, Div_D,
                           Qout, eps_M, gauss_p, sig):

    p = RUNTIME.get()
    N = len(nodes)

    op_def = make_transport_operator(
        V, nodes, groups,
        pho, D, Div_D, Qout,
        p, eps_M, sig, gauss_p
    )

    bc_definitions = None if sig > 0 else make_transport_bc(normals, D)

    return build_matrix(
        nodes, groups, p,
        op_def,
        bc_definitions,
        eps_M,
        N,
        sig
    )



# =========================================================
# ADJOINT ψ_C
# =========================================================
def make_adj_C_operator(
    V,                 # K∇h
    Div_V,            # ∇·(K∇h)
    D,
    Div_D,
    Qout,
    pho,
    R,
    gauss_p,
    p,
    eps_M,
    sig,
    groups
):

    idx_dom = np.hstack((
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ))

    base = -2.0 * pho * R / p.dt

    if p.activate_ext and Qout:
        base += sig * gauss_p

    return dict(

        idx=idx_dom,

        eps=eps_M,

        diffs=[
            [0, 0],   # identidad
            [1, 0],   # dx
            [0, 1],   # dy
            [2, 0],   # dxx
            [0, 2]    # dyy
        ],

        coeffs_func=lambda i: [

            # ------------------------------------------------
            # IDENTIDAD
            #
            # A:
            # -2φR/dt + div(K∇h) + reaction
            #
            # B:
            # -2φR/dt - div(K∇h) - reaction
            # ------------------------------------------------
            base[i] + sig * (
                Div_V[i]
                - p.landa * pho[i] * R 
            ),

            # ------------------------------------------------
            # GRADIENTE
            #
            # A:
            # +(K∇h) + ∇D
            #
            # B:
            # -(K∇h) - ∇D
            # ------------------------------------------------
            sig * (V[i, 0] - Div_D[i, 0]),

            sig * (V[i, 1] - Div_D[i, 1]),

            # ------------------------------------------------
            # DIFUSIÓN
            #
            # A: +D : Hess
            # B: -D : Hess
            # ------------------------------------------------
            -sig * D[i, 0],

            -sig * D[i, 1],
        ]
    )
def make_adj_C_bc(normals, D):

    return [

        # -------------------------------------------------
        # Dirichlet
        # ψ = 0
        # -------------------------------------------------
        *[
            dict(
                name=tag,

                coeffs_func=lambda i: (
                    [[0, 0]],
                    [np.ones(len(i))]
                )
            )

            for tag in ["inlet", "outlet"]
        ],

        # -------------------------------------------------
        # Neumann
        # (D∇ψ)·n = 0
        # -------------------------------------------------
        dict(
            name="wall",
            coeffs_func=lambda i: (
                [
                [1, 0],
                [0, 1]
                ],
                [

                    D[i, 0] * normals[i, 0],

                    D[i, 1] * normals[i, 1]
                ]
            )
        )
    ]
def build_adj_C_matrix(
    V,
    Div_V,
    nodes,
    groups,
    normals,
    pho,
    D,
    Div_D,
    Qout,
    gauss_p,
    eps_M,
    sig
):

    p = RUNTIME.get()
    N = len(nodes)

    op_def = make_adj_C_operator(
        V,
        Div_V,
        D,
        Div_D,
        Qout,
        pho,
        p.R,
        gauss_p,
        p,
        eps_M,
        sig,
        groups
    )

    bc_definitions = make_adj_C_bc(normals, D)

    return build_matrix(
        nodes,
        groups,
        p,
        op_def,
        bc_definitions,
        eps_M,
        N,
        sig
    )


# =========================================================
# ADJOINT ψ_h
# =========================================================
def make_adj_H_operator(
    theta,
    S_s,
    K,
    Div_K,
    p,
    eps_M,
    sig,
    groups
):

    idx_dom = np.hstack((
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ))

    return dict(

        idx=idx_dom,

        eps=eps_M,

        diffs=[
            [0, 0],
            [2, 0],
            [0, 2],
            [1, 0],
            [0, 1]
        ],

        coeffs_func=lambda i: [

            # --------------------------------------------
            # IDENTIDAD
            #
            # A = -2Ss/dt
            # B = -2Ss/dt
            # --------------------------------------------
            -2.0 * S_s[i] / p.dt,

            # --------------------------------------------
            # DIFUSIÓN
            #
            # A = +K∇²
            # B = -K∇²
            # --------------------------------------------
            -theta * sig * K[i, 0],
            -theta * sig * K[i, 1],
            -theta * sig * Div_K[i, 0],
            -theta * sig * Div_K[i, 1],
        ]
    )
def make_adj_H_bc(normals, K):

    return [

        # -------------------------------------------------
        # Dirichlet
        # ψ_h = 0
        # -------------------------------------------------
        *[
            dict(
                name=tag,
                coeffs_func=lambda i: (
                    [[0, 0]],
                    [np.ones(len(i))]
                )
            )

            for tag in ["inlet", "outlet"]
        ],

        # -------------------------------------------------
        # Neumann
        # (K∇ψ_h)·n = 0
        # -------------------------------------------------
        dict(

            name="wall",

            coeffs_func=lambda i: (
                [
                    [1, 0],
                    [0, 1]
                ],
                [

                    K[i, 0] * normals[i, 0],

                    K[i, 1] * normals[i, 1]
                ]
            )
        )
    ]
def build_adj_H_matrix(
    nodes,
    groups,
    normals,
    theta,
    S_s,
    K,
    Div_K,
    eps_M,
    sig
):

    p = RUNTIME.get()
    N = len(nodes)

    op_def = make_adj_H_operator(
        theta,
        S_s,
        K,
        Div_K,
        p,
        eps_M,
        sig,
        groups
    )

    bc_definitions = make_adj_H_bc(
        normals,
        K
    )

    return build_matrix(
        nodes,
        groups,
        p,
        op_def,
        bc_definitions,
        eps_M,
        N,
        sig
    )


# ---------------------------
# 5) RHS builders adapted for two meshes
# ---------------------------
def H_vector(H, M_right, delta_p, groups, nodes, N, operador= None, Qout_n=0, Qout_N=0):
    p = RUNTIME.get()
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    b = np.zeros(N)
    aux = H if operador == "Identity" else M_right @ H
    if p.activate_ext and operador != "Identity":
        aux -= ((1-p.theta)*Qout_n + p.theta*Qout_N) * delta_p

    b[idx] = aux[idx]
    apply_boundary(b, nodes, groups.get('ghosts:inlet', []),
                    In_h(p.zi_max, p.inlet_z_t, p.inlet_z_t, p.inlet_a))
    # apply_boundary(b, nodes, groups.get('ghosts:outlet', []),
    #                 In_h(zo_max, zo_min, 0.4*zo_max + 0.6*zo_min,
    #                       outlet_a, outlet_a/150))

    return b

def U_vector(H, K, grad):
    """
    Calcula U = -K * grad(H) usando Darcy,
cc    donde grad es un operador tipo matriz que actúa con @.
    """

    # Gradiente escalar en cada dirección
    dHdx = grad[0] @ H    # derivada en x
    dHdy = grad[1] @ H    # derivada en y

    # Darcy: U = -K * grad(H)
    Ux = -K[:, 0] * dHdx
    Uy = -K[:, 1] * dHdy

    return np.column_stack([Ux, Uy])
   
def C_vector(C, M_right, gauss_f, groups, t, T, N, C_im, pho):
    p = RUNTIME.get()
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    b = np.zeros(N)
     # 🔹 Base común
    aux = M_right @ C
    b[idx] = aux[idx]

    # 🔹 Fuente (ADR + MRMT semi)
    if p.model == "adr" or p.model == "mrmt_semi":
        fuente = (fuente_C(t, T) + fuente_C(t - p.dt, T)) * gauss_f
        b[idx] += fuente[idx]

    # 🔹 Acoplamiento MRMT block
    elif p.model == "mrmt_block":
        b[idx] += np.sum(
            4 * p.R * p.alpha_r[:, None] * pho[idx] * C_im[:, idx],
            axis=0
        )

    else:
        raise ValueError(f"Modelo no válido: {p.model}")



    return b

# =========================================================
# RHS ψ_C
# =========================================================
def build_adj_C_rhs(
    B, 
    psiC_N,
    C_n,
    C_N,
    gamma,
    delta_p,
    nodes,
    groups
):


    # -----------------------------------------------------
    # RHS completo
    #
    # b = B ψ^{n+1} + f
    # -----------------------------------------------------
    rhs = B @ psiC_N

    idx_dom = np.hstack((
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ))

    # -----------------------------------------------------
    # 2γ(C^{n+1}+C^n)δ
    # -----------------------------------------------------
    rhs[idx_dom] += (
        2.0
        * gamma
        * (C_N[idx_dom] + C_n[idx_dom])
        * delta_p[idx_dom]
    )

    
    return rhs


# =========================================================
# ∇·(K ψ ∇C)
# =========================================================
def div_Kpsi_gradC(
    psi,
    C,
    K,
    Dx,
    Dy,
    nodes,
    groups
):

    N = len(nodes)

    idx_dom = np.hstack((
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ))

    # =====================================================
    # 1. ∇C
    # =====================================================
    gradCx = Dx @ C
    gradCy = Dy @ C

    # =====================================================
    # 2. K ψ ∇C
    # =====================================================
    flux_x = K[:, 0] * psi * gradCx
    flux_y = K[:, 1] * psi * gradCy

    # =====================================================
    # 3. ∇·(flux)
    # =====================================================
    div_flux = Dx @ flux_x + Dy @ flux_y

    # =====================================================
    # salida
    # =====================================================
    out = np.zeros(N)

    out[idx_dom] = div_flux[idx_dom]

    return out

# =========================================================
# RHS ψ_h
# =========================================================
def build_adj_H_rhs(
    B,
    psiH_N,

    psiC_n,
    psiC_N,

    C_n,
    C_N,

    K,

    grad,

    nodes,
    groups
):

    p = RUNTIME.get()
    # -----------------------------------------------------
    # término temporal
    # B ψ_h^n
    # -----------------------------------------------------
    rhs = B @ psiH_N

    # -----------------------------------------------------
    # término fuente
    #
    # -∇·(K ψ_C^{n+1} ∇C^{n+1})
    # -∇·(K ψ_C^n     ∇C^n)
    # -----------------------------------------------------
    rhs -= (1-p.theta)*div_Kpsi_gradC(
        psiC_N,
        C_N,
        K,
        grad[0],
        grad[1],
        nodes,
        groups
    )

    rhs -= p.theta*div_Kpsi_gradC(
        psiC_n,
        C_n,
        K,
        grad[0],
        grad[1],
        nodes,
        groups
    )

    return rhs