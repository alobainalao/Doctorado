import numpy as np
from scipy.sparse.linalg import splu
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows
from scipy.sparse import csr_matrix
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

#----------------------------
# 4) U_matrix and p_matrix adapted to two meshes
# ---------------------------

def build_H_matrix(nodes, groups, normals, theta, S_s, K, Div_K, eps_M, 
                    sig=1, eps_arr= None, operator = None):
    """
    Construye matrices A_h, B_h para:
    S_s ∂h/∂t - ∇·(K∇h) = f
    con esquema trapezoidal en el tiempo.
    """
    p = RUNTIME.get()
    N = len(nodes)
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    if operator == "Identity":
        A_temp = weight_matrix(
            x=nodes[idx],
            p=nodes,
            diffs=[[0,0]],
            n=1,
            phi=p.phi,
            order=0,
            eps=eps_arr[idx],
            chunk_size=None
        )
    else:
        A_temp = weight_matrix(
            x=nodes[idx],
            p=nodes,
            n=p.n_stencil,
            diffs=[
                [0,0],   # identidad
                [2,0],   # ∂²/∂x²
                [0,2],   # ∂²/∂y²
                [1,0],   # ∂/∂x
                [0,1]    # ∂/∂y
            ],
            coeffs=[
                S_s[idx]/p.dt,   # identidad
                theta*sig*K[idx,0],        # ∂²/∂x²
                theta*sig*K[idx,1],        # ∂²/∂y²
                theta*sig*Div_K[idx,0],    # ∂/∂x
                theta*sig*Div_K[idx,1]     # ∂/∂y
            ],
            phi=p.phi,
            order=p.order,
            eps=eps_M[idx],
            chunk_size=None
        )

    # expand to vector system (2*Nu)
    A  = expand_rows(A_temp, idx, N)

    if sig <0:
        #idx_I = np.hstack((groups['boundary:inlet'], groups['boundary:outlet'])) 
        idx_I = groups['boundary:inlet']
        A_fija = weight_matrix(
            x=nodes[idx_I],
            p=nodes,
            n=p.n_stencil,
            diffs=[[1,0]],
            coeffs=[-K[idx_I,0]],
            phi=p.phi, order=p.order, 
            eps= eps_M[idx_I],
            chunk_size=None
        )

        A_wall = weight_matrix(
            x=nodes[groups['boundary:wall']],
            p=nodes,
            n=p.n_stencil,
            diffs=[[1,0],  # ∂/∂x
                [0,1]],    # ∂/∂y
            coeffs=[
                -K[groups['boundary:wall'], 0]*normals[groups['boundary:wall'], 0],  # ∂/∂x
                -K[groups['boundary:wall'], 1]*normals[groups['boundary:wall'], 1]   # ∂/∂y
            ],
            phi=p.phi, order=p.order, 
            eps= eps_M[groups['boundary:wall']],
            chunk_size=None
        )
        
        idx_f = groups['boundary:outlet']
        A_free = weight_matrix(
            x=nodes[idx_f],
            p=nodes,
            n=p.n_stencil,
            diffs=[
                    [1,0],  # hx
                    [0,1],  # hy
                    [2,0],  # hxx
                    [0,2],  # hyy
                    [1,1],  # hxy
                ],
            coeffs=[
                    -(Div_K[idx_f,0] *normals[idx_f,0] + Div_K[idx_f,2] *normals[idx_f,1]),      # hx
                    -(Div_K[idx_f,3] *normals[idx_f,0] + Div_K[idx_f,1] *normals[idx_f,1]),      # hy
                    -(K[idx_f,0]  *normals[idx_f,0]),                                        # hxx
                    -(K[idx_f,1]  *normals[idx_f,1]),                                        # hyy
                    -(K[idx_f,0]  *normals[idx_f,1]  - K[idx_f,1]  *normals[idx_f,0]),       # hxy
                ],
            phi=p.phi,
            order=p.order,
            eps=eps_M[idx_f],
            chunk_size=None
        )

        #A += expand_rows(A_fija,  np.hstack((groups['ghosts:inlet'], groups['ghosts:outlet'])), N) 
        A += expand_rows(A_fija, groups['ghosts:inlet'], N) 
        A += expand_rows(A_wall, groups['ghosts:wall'], N)

        A += expand_rows(A_free, groups['ghosts:outlet'], N)

        return A.tocsc()

    return A.tocsc()

def build_transport_matrix(V, nodes, groups, normals, pho, D, Div_D, Qout, eps_M, 
                            gauss_p, sig):
    """
    Construye matrices A_c, B_c para:
    ∂(φCR)/∂t - (u·∇C) - ∇·(D∇C) + λφCR = W_+ - W_-
    con esquema trapezoidal en el tiempo.
    """
    p = RUNTIME.get()
    N = len(nodes)

    U_div_D =(V - Div_D[:, :2])/2.

    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    # =====================================================
    # 🔹 COEFICIENTE DEPENDIENTE DEL MODELO
    # =====================================================
    base = pho * p.R
    coef_identity = base * (2. / p.dt) 
    
    if p.activate_ext and  Qout:
        coef_identity -= sig * gauss_p


    if p.model == "adr":
        coef_identity -= base * sig * p.landa

    elif p.model == "mrmt_block":
        coef_identity -= 2 * base * sig * p.alpha_sum

    elif p.model != "mrmt_semi":
        raise ValueError(f"Modelo no válido: {p.model}")

    A_temp = weight_matrix(
        x=nodes[idx],
        p=nodes,
        n=p.n_stencil,
        diffs=[
            [0,0],   # identidad
            [2,0],   # ∂²/∂x²
            [0,2],   # ∂²/∂y²
            [1,0],   # ∂/∂x
            [0,1]    # ∂/∂y
        ],
        coeffs=[
            coef_identity[idx],   # identidad
            sig*D[idx,0],        # ∂²/∂x²
            sig*D[idx,1],        # ∂²/∂y²
            -sig*U_div_D[idx,0],    # ∂/∂x
            -sig*U_div_D[idx,1]     # ∂/∂y
        ],
        phi=p.phi,
        order=p.order,
        eps=eps_M[idx],
        chunk_size=None
    )

    # expand to vector system
    A  = expand_rows(A_temp, idx, N)

    if sig <0:
        idx_I = np.hstack((groups['boundary:wall'], groups['boundary:inlet'], groups['boundary:outlet']))
        A_fija = weight_matrix(
            x=nodes[idx_I],
            p=nodes,
            n=p.n_stencil,
            diffs=[[1,0], 
                [0,1]],   
            coeffs=[
                -D[idx_I, 0]*normals[idx_I, 0],  
                -D[idx_I, 1]*normals[idx_I, 1]   
            ],
            phi=p.phi, order=p.order, 
            eps= eps_M[idx_I],
            chunk_size=None
        )
        

        A += expand_rows(A_fija, np.hstack((groups['ghosts:wall'], groups['ghosts:inlet'], groups['ghosts:outlet'])), N)

        return splu(A.tocsc())

    return A.tocsc()

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

