import numpy as np
from scipy.sparse.linalg import splu
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows
from scipy.sparse import csr_matrix, csc_matrix
from scipy.linalg import solve
from funtions.runtime import RUNTIME
from funtions.utils import In_h, apply_boundary, fuente_C
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

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



def phi(r, eps):
    return np.exp(-(eps * r)**2)

def rbf_matrix(nodes, eps):
    D = cdist(nodes, nodes)
    return phi(D, eps)

def rbf_integration_weights(nodes, eps=None):
    """Pesos de cuadratura por áreas de Delaunay.

    Cada nodo recibe 1/3 del área de cada triángulo al que pertenece.
    Satisface ∑w_i = área_del_dominio y ∫f dΩ ≈ ∑w_i f(x_i) con error
    O(h²). El parámetro eps se conserva por compatibilidad pero se ignora.
    """
    tri = Delaunay(nodes)
    wi = np.zeros(len(nodes))
    for simplex in tri.simplices:
        v = nodes[simplex]
        area = 0.5 * abs(
            (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
            - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
        )
        wi[simplex] += area / 3.0
    return wi


# ============================================================
# Delta discreta RBF
# ============================================================

def rbf_delta(nodes, x_p, eps):
    """
    Construye delta discreta/interpolante cardinal.

    Resultado:
        u(x_p) ≈ Σ delta_i u_i
    """

    # --------------------------------------------------------
    # matriz RBF entre nodos
    # --------------------------------------------------------

    D = cdist(nodes, nodes)

    A = phi(D, eps)

    # regularización
    A += 1e-12 * np.eye(len(nodes))

    # --------------------------------------------------------
    # evaluación en x_p
    # --------------------------------------------------------

    r_p = np.linalg.norm(nodes - x_p, axis=1)

    b = phi(r_p, eps)

    # --------------------------------------------------------
    # pesos cardinales
    # --------------------------------------------------------

    delta = solve(A, b, assume_a='sym')

    return delta

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
    """
    Condiciones de frontera para H.

    bc_outlet_mode (leído desde RUNTIME p):
      "pfs"  — condición de 2.º orden original (n·∇(K∇H) = 0). DEFAULT.
      "fem"  — Neumann de 1.º orden: K∇H·n = h_outlet(z), idéntica a la
               condición natural de FEniCS. Permite comparación rigurosa
               con el backend FEM bajo condiciones de salida equivalentes.
    """
    p = RUNTIME.get()
    bc_outlet_mode = getattr(p, "bc_outlet_mode", "pfs")

    if bc_outlet_mode == "fem":
        # Neumann 1.º orden: -(K∇H·n) = b  con  b = -h_outlet(z)
        # (aplicado en H_vector con signo correcto al nodo fantasma de salida)
        outlet_bc = dict(
            name="outlet",
            coeffs_func=lambda i: (
                [[1, 0], [0, 1]],
                [
                    -K[i, 0] * normals[i, 0],
                    -K[i, 1] * normals[i, 1]
                ]
            )
        )
    elif bc_outlet_mode == "dirichlet":
        # Carga fija H = h_borde en los nodos de frontera de la salida
        # (exp/sumidero-W: cuadrado alimentado por un acuífero extenso).
        outlet_bc = dict(
            name="outlet",
            coeffs_func=lambda i: ([[0, 0]], [np.ones(len(i))])
        )
    elif bc_outlet_mode == "pfs_nn":
        # Variante invariante de la condición libre: ∂_n(u·n) = 0, es decir
        # Σ_ij n_i n_j ∂_j(K_i ∂_i H) = 0 (exp/sumidero-W, decisión 2).
        outlet_bc = dict(
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
                    -normals[i, 0] * (Div_K[i, 0] * normals[i, 0] + Div_K[i, 2] * normals[i, 1]),
                    -normals[i, 1] * (Div_K[i, 3] * normals[i, 0] + Div_K[i, 1] * normals[i, 1]),
                    -(K[i, 0] * normals[i, 0] ** 2),
                    -(K[i, 1] * normals[i, 1] ** 2),
                    -(K[i, 0] + K[i, 1]) * normals[i, 0] * normals[i, 1]
                ]
            )
        )
    else:
        # Condición de radiación de 2.º orden original (modo "pfs")
        outlet_bc = dict(
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

        outlet_bc,
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
def peso_theta_C(p, sig):
    """Peso temporal del operador de transporte en la escala de Crank–Nicolson
    (2/dt en la identidad): 2θ_C en el lado implícito (sig<0) y 2(1−θ_C) en el
    explícito (sig>0). θ_C = 0.5 (default) reproduce el CN original; θ_C = 1 es
    Euler implícito (incluidos sumidero del pozo, decaimiento y fuente)."""
    th = float(getattr(p, "theta_C", 0.5))
    return 2.0 * th if sig < 0 else 2.0 * (1.0 - th)


def make_transport_operator(V, nodes, groups, pho, D, Div_D, Qout, p, eps_M, sig, gauss_p):
    
    U_div_D = (V - Div_D[:, :2]) / 2.0
    w = peso_theta_C(p, sig)

    idx_dom = np.hstack([
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall']
    ])

    base = pho * p.R
    coef_identity = base * (2. / p.dt)

    # Sumidero de contaminante en el pozo: el agua bombeada arrastra C disuelto.
    # Análogo al término `delta_C * C` de FEniCS. Default True en default.json.
    if getattr(p, "extract_C", True) and p.activate_ext and Qout and p.run_type != "optimization":
        coef_identity -= w * sig * gauss_p

    if p.model == "adr":
        coef_identity -= w * base * sig * p.landa
    elif p.model == "mrmt_block":
        # Factor 2: complemento de Schur tras eliminar C_im analíticamente.
        # La diagonal del sistema Schur-reducido lleva ±2·base·Σ_r α_r en LHS/RHS,
        # ponderada por w (θ_C·C^{n+1} + (1−θ_C)·C^n en el intercambio).
        # El RHS acoplado lleva 4·base·α_r·C_im_r^n (ver C_vector), sin peso θ.
        # Derivación: alpha_r = κ_r·β_r/dt; ver config/parameters.py.
        coef_identity -= w * 2 * base * sig * p.alpha_sum
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
            w * sig * D[i, 0],
            w * sig * D[i, 1],
            -w * sig * U_div_D[i, 0],
            -w * sig * U_div_D[i, 1],
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
            # Adjunto de U_div_D·∇C con U_div_D=(V-∇D)/2:
            # la integración por partes da -∇·(U_div_D)ψ ≈ -Div_V/2
            # ------------------------------------------------
            base[i] + sig * (
                Div_V[i] * 0.5
                - p.landa * pho[i] * R
            ),

            # ------------------------------------------------
            # GRADIENTE
            #
            # Adjunto de U_div_D·∇C: -U_div_D·∇ψ
            # U_div_D = (V - ∇D) / 2  →  usar la misma escala
            # ------------------------------------------------
            sig * (V[i, 0] - Div_D[i, 0]) * 0.5,

            sig * (V[i, 1] - Div_D[i, 1]) * 0.5,

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

            # Adjunto discreto exacto: A_H = A_left (operador de flujo forward).
            # La ecuación de flujo es simétrica → A_left^T = A_left = A_H.
            # Mismos coeficientes que make_H_operator "full".
            S_s[i] / p.dt,

            theta * sig * K[i, 0],
            theta * sig * K[i, 1],
            theta * sig * Div_K[i, 0],
            theta * sig * Div_K[i, 1],
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
def H_vector(H, M_right, delta_p, groups, nodes, N, operador=None, Qout_n=0, Qout_N=0, gauss_p=None):
    p = RUNTIME.get()
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    b = np.zeros(N)
    aux = H if operador == "Identity" else M_right @ H
    if p.activate_ext and operador != "Identity":
        # smooth_well_h=True → Gaussiano (mismo kernel que C extraction, igual que FEniCS)
        # smooth_well_h=False → Wendland C2 compacto (default histórico)
        well_kernel = gauss_p if (gauss_p is not None and getattr(p, "smooth_well_h", False)) else delta_p
        aux -= ((1-p.theta)*Qout_n + p.theta*Qout_N) * well_kernel

    b[idx] = aux[idx]
    apply_boundary(b, nodes, groups.get('ghosts:inlet', []),
                    In_h(p.zi_max, p.inlet_z_t, p.inlet_z_t, p.inlet_a))

    # bc_outlet_mode="fem": imponer K∇H·n = h_outlet(z) en la salida.
    # El operador de la BC es -(K∇H·n) = b, así que b = -h_outlet(z).
    # Como outlet_a < 0, -outlet_a > 0 y el perfil resultante es ≥ 0,
    # coherente con ∂H/∂x > 0 (gradiente hidráulico de derecha a izquierda).
    if getattr(p, "bc_outlet_mode", "pfs") == "dirichlet":
        g_out = groups.get('ghosts:outlet', [])
        b[g_out] = float(getattr(p, "h_borde", getattr(p, "h_init_const", 300.0)))
    if getattr(p, "bc_outlet_mode", "pfs") == "fem":
        apply_boundary(
            b, nodes, groups.get('ghosts:outlet', []),
            In_h(p.zo_max, p.zo_min,
                 0.4 * p.zo_max + 0.6 * p.zo_min,
                 -p.outlet_a, -p.outlet_a / 150)
        )

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

    # 🔹 Fuente de contaminación (todos los modelos)
    if p.model not in ("adr", "mrmt_semi", "mrmt_block"):
        raise ValueError(f"Modelo no válido: {p.model}")
    if getattr(p, "activate_fuente", True):
        fuente = (peso_theta_C(p, -1) * fuente_C(t, T)
                  + peso_theta_C(p, +1) * fuente_C(t - p.dt, T)) * gauss_f
        b[idx] += fuente[idx]

    # 🔹 Acoplamiento MRMT block: C_im^n → C (complemento de Schur).
    # Factor 4 = 2×2: un factor 2 del RHS Schur y otro de la regla trapezoidal.
    # Exactamente el transpuesto del término 4·base·α_r que aparece en la
    # actualización adjunta de ψ_im_r (step_time.py:step_adjoint).
    if p.model == "mrmt_block":
        b[idx] += np.sum(
            4 * p.R * p.alpha_r[:, None] * pho[idx] * C_im[:, idx],
            axis=0
        )



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