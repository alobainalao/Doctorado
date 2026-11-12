import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, bmat, coo_matrix, diags, eye
from scipy.sparse.linalg import splu
from matplotlib.animation import FuncAnimation, PillowWriter
from rbf.pde.nodes import poisson_disc_nodes
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows, expand_cols
from rbf.pde.geometry import contains
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import ellipe
import get_phi as gph 
from shapely.geometry import Polygon, Point
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
import h5py
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

t0 = time.perf_counter()

# ---------------------------
# 1) Parámetros
# ---------------------------
pre = False
pre = True

savefile = "./preproceso.pkl"

spacing = 30
min_sp = 30
nu, rho = 1e-3, 1.0
dt, T = 3.6e4, 2592000
n_stencil, phi, order = 35, 'phs7', 4

n_stencil, phi, order = 25, 'mq', 2
os.makedirs("ns/figures", exist_ok=True)
tol= 1e-5
theta = 0.9
sp_q = 5

pozo = [3700, -700]
fnte = [9116, 109]

epsilon_x = 0.5*spacing    # Ancho horizontal (x)
epsilon_y = spacing    # Ancho vertical (y)

# vert = np.array([
#     [816,   -2690.5],
#     [4354,  -1757.5],
#     [10755,  -768.5],
#     [10755,   308.5],
#     [6716,    335],
#     [4354,    194.5],
#     [816,     250.5]
# ])
vert = np.array([
    [816,   -2690.5],
    [4354,  -1757.5],
    [10755,  -768.5],
    [10755,   250.5],
    [816,     250.5]
])

xmin, xmax = vert[:,0].min(), vert[:,0].max()
ymin, ymax = vert[:,1].min(), vert[:,1].max()

Lx = xmax - xmin
Ly = ymax - ymin

def dv( x0, xb):
    return  x0 - xb/2 +(x0-xb)/(4*np.pi)*np.sin(2* x0* np.pi / (x0-xb))

zi_max, zo_max, z_min = 250, 250, -2691
inlet_z_t, zo_min, inlet_a = -760, -2690.5, -2e-4
d_vel = inlet_a/15
outlet_a = inlet_a * dv(zi_max, inlet_z_t) / dv(zo_max, zo_min)

# -------------------- PARÁMETROS FÍSICOS ----------------------
g, nu, d_z, alpha = 9.81, 1.055e-6, 1e-3, 2.0
tol = 1e-6
R, landa = 1, 1e-10

# -------------- PARÁMETROS DE DISPERSIÓN ---------------------
a_l, a_t, D_d, eps = 10, 1, 1.2e-9, 1e-16


# MRMT parameters

Nr = 3

Deff = np.array([1e-9, 5e-10, 1e-10])
L = 0.01

alpha_im = np.zeros((3,len(phi)))

phi_im = np.array([
    0.1,
    0.05,
    0.02
])

beta = np.array([
    0.15,
    0.1,
    0.05
])

R_im = np.ones(Nr)




def Lspacing(x, s_M=100, min_s=spacing):
    """
    Espaciado adaptativo:
    - Fino cerca del obstáculo, fuente y pozo
    - Más grueso lejos del dominio o bordes
    """

    poly = Polygon(vert)

    # Asegurar formato (N, 2)
    x = np.atleast_2d(x)

    # Distancia al borde del polígono
    dist_borde = np.array([poly.boundary.distance(Point(px, py)) for px, py in x])

    # Distancias a fuente y pozo
    dist_fnte = np.linalg.norm(x - fnte, axis=1)
    dist_pozo = np.linalg.norm(x - pozo, axis=1)

    # Distancia mínima combinada
    d = np.minimum.reduce([dist_borde, dist_pozo, dist_fnte])

    # Altura total del dominio
    Ly = np.ptp(vert[:, 1])

    # Espaciado adaptativo
    spacing_local = (s_M - min_s) * np.sqrt(np.clip(d / Ly, 0, 1)) + min_s

    return np.clip(spacing_local, min_s, s_M)


# Devuelve siempre un escalar para un punto (como espera poisson_disc_nodes)
def spacing_func(xy):
    return Lspacing(np.atleast_2d(xy), s_M=30, min_s=min_sp)

class In_h():
    def __init__(self, z_0, z_min, z_t2, amplitud, v_min= 0):
        self.amplitud = amplitud
        self.z_0 = z_0
        self.z_t = z_0 - 100
        self.z_t2 = z_t2
        self.z_min = z_min
        self.v_min = v_min

    def __call__(self, x):
        z = x[1]

        values = np.piecewise(
            z,
            [
                (z < self.z_min) | (z > self.z_0),
                (z <= self.z_0) & (z > self.z_t),
                (z <= self.z_t) & (z > self.z_t2),
                (z <= self.z_t2) & (z > self.z_min),
            ],
            [
                lambda z_: 0,
                lambda z_: self.amplitud * np.cos((z_ - self.z_t) * np.pi / (2 * (self.z_0 - self.z_t))) ** 2,
                lambda z_: (self.amplitud-self.v_min) * np.cos((z_ - self.z_t) * np.pi / (2 * (self.z_t - self.z_t2))) ** 2 +self.v_min, 
                lambda z_: self.v_min* np.cos((z_ - self.z_t2) * np.pi / (2 * (self.z_t2 - self.z_min))) ** 2, 
            ]
        )

        return values

def gaussian_2d(x, x_s, amplitude, epsilon_x, epsilon_y=None):
    x = np.asarray(x, float)
    x_s = np.asarray(x_s, float)

    # Si no se da epsilon_y, se usa epsilon_x
    if epsilon_y is None:
        epsilon_y = epsilon_x

    # Asegurar forma (N,2)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    dx = x[:, 0] - x_s[0]
    dy = x[:, 1] - x_s[1]

    normalization = 1.0 / (2 * np.pi * epsilon_x * epsilon_y)

    g = amplitude * normalization * np.exp(
        -(dx**2 / (2 * epsilon_x**2) + dy**2 / (2 * epsilon_y**2))
    )

    return g if len(g) > 1 else g[0]

def fuente_C(t, T):
    return 1e-11 * t/T * np.exp(-8*t/T) + 1e-12 * (1 - np.exp(-8*t/T))


# ---------------------------
# 2) Generador de nodos
# ---------------------------
def build_nodes(sp_u=spacing):
    smp = [[i, (i + 1) % len(vert)] for i in range(len(vert))]

    # boundary_groups = {
    #     'inlet': [2],               
    #     'outlet': [6],                
    #     'wall': [0, 1, 3, 4, 5]
    # }
    boundary_groups = {
        'inlet': [2],               
        'outlet': [4],                
        'wall': [0, 1, 3]
    }

    boundary_groups_with_ghosts = ['inlet', 'outlet', 'wall']
    # --- nodes for velocity mesh (finer) ---
    nodes, groups, normals = poisson_disc_nodes(
        sp_u,
        (vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    idx_p = np.argmin(np.linalg.norm(nodes - pozo, axis=1))

    return nodes, groups, normals, smp, nodes[idx_p]

def build_nodes_var(sp_u=spacing):
    smp = [[i, (i + 1) % len(vert)] for i in range(len(vert))]

    # boundary_groups = {
    #     'inlet': [2],               
    #     'outlet': [6],                
    #     'wall': [0, 1, 3, 4, 5]
    # }
    boundary_groups = {
        'inlet': [2],               
        'outlet': [4],                
        'wall': [0, 1, 3]
    }

    boundary_groups_with_ghosts = ['inlet', 'outlet', 'wall']
    # --- nodes for velocity mesh (finer) ---
    nodes, groups, normals = poisson_disc_nodes(
        spacing_func,
        (vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts
    )

    idx_p = np.argmin(np.linalg.norm(nodes - pozo, axis=1))

    return nodes, groups, normals, smp, nodes[idx_p]

# ====================================================
# Utilidades para calcular eps local
# ====================================================

def local_lengthscale(nodes, k=8):
    """
    Calcula h_i = distancia media a los k vecinos más cercanos (excluye el propio punto).
    Devuelve array (N,)
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(nodes)
    dists, idxs = nbrs.kneighbors(nodes)
    # ignorar la primera columna (distancia a sí mismo = 0)
    h = np.mean(dists[:, 1:], axis=1)
    return h

def eps_from_h(h, scale=1.0, min_factor=0.75, max_factor=1.25):

    h = np.asarray(h)
    eps = scale / (h + 1e-16)              # relación inversa: menor h → mayor ε
    eps = np.clip(eps,                     # acotar razonablemente
                  np.min(eps)*min_factor,
                  np.max(eps)*max_factor)
    return eps

def K_ope(pho):
    K = np.zeros((len(pho), 2))
    K[:, 1] = 8.3e-3 * g * d_z**2 * pho**3 / (nu * (1 - pho)**2)             # componente y
    K[:, 0] = K[:, 1] * alpha**2  # componente x
    
    return K

def D_total(Dif, V):
    """
    Dif : difusión molecular (float, (N,), o (N,2))
    V   : velocidades (N,2)
    a_l : dispersividad longitudinal
    a_t : dispersividad transversal
    """

    V = np.asarray(V)

    # Componentes al cuadrado
    Vx2 = V[:, 0]**2
    Vy2 = V[:, 1]**2

    # Norma |V| con protección contra cero
    Vnorm = np.sqrt(Vx2 + Vy2)
    Vnorm[Vnorm == 0] = 1e-12   # evita división cero

    # Difusión por dispersión (N,2)
    D_disp = np.column_stack([
        a_l * Vx2 + a_t * Vy2,      # componente x
        a_t * Vx2 + a_l * Vy2       # componente y
    ]) / Vnorm[:, None]             # escala cada fila → (N,1)

    # Asegurar que Dif tiene shape (N,2)
    Dif = np.asarray(Dif)

    if Dif.ndim == 0:
        # Difusión molecular escalar → expandir a (N,2)
        Dif = np.full_like(D_disp, Dif)

    elif Dif.ndim == 1:
        # vector (N,) → expandir a dos columnas (N,2)
        Dif = np.column_stack([Dif, Dif])

    # Aquí ya todo es (N,2)
    return Dif + D_disp

def delta_char(nodes, x_p, tol):
    dist = np.linalg.norm(nodes - x_p, axis=1)
    return np.where(dist < tol, 1.0, 0.0)

def Div_KD(Op_vec, grad, anisotropic=True):

    if anisotropic:
        Kx, Ky = Op_vec.T
    else:
        Kx = Ky = Op_vec

    dKx_dx = grad[0] @ Kx
    dKy_dy = grad[1] @ Ky

    return np.column_stack([dKx_dx, dKy_dy])

# Función para definir el operador de difusión
def diffusion_operator(pho):

    Diff = np.zeros((len(pho), 2))
    Diff[:,0] = 0.5 * D_d * pho* ellipe(1 - alpha**-2)
    Diff[:,1] = Diff[:,0]/ alpha
    
    return Diff


# ---------------------------
# 3) Operadores RBF-FD entre mallas
# ---------------------------
def rbf_op(diffs, x_eval, x_src, eps_arr, n=n_stencil):
    """
    Construye matriz que evalúa derivadas `diffs` definidas en puntos x_src
    y las evalúa en x_eval. Devuelve CSR tamaño (len(x_eval), N_src).
    """
    W = weight_matrix(x=x_eval, p=x_src, n=n, 
                      diffs=[diffs], phi=phi, order=order,
                      eps=eps_arr, chunk_size=None)
    return csr_matrix(W)

def build_operators(nodes, eps_arr):
    """
    Construye operadores RBF-FD sobre una sola malla de nodos.
    Devuelve los gradientes y laplacianos necesarios para resolver
    las ecuaciones de flujo y transporte acopladas en el mismo dominio.
    """


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
    N = len(nodes)
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    if operator == "Identity":
        A_temp = weight_matrix(
            x=nodes[idx],
            p=nodes,
            diffs=[[0,0]],
            n=1,
            phi=phi,
            order=0,
            eps=eps_arr[idx],
            chunk_size=None
        )
    else:
        A_temp = weight_matrix(
            x=nodes[idx],
            p=nodes,
            n=n_stencil,
            diffs=[
                [0,0],   # identidad
                [2,0],   # ∂²/∂x²
                [0,2],   # ∂²/∂y²
                [1,0],   # ∂/∂x
                [0,1]    # ∂/∂y
            ],
            coeffs=[
                S_s[idx]/dt,   # identidad
                theta*sig*K[idx,0],        # ∂²/∂x²
                theta*sig*K[idx,1],        # ∂²/∂y²
                theta*sig*Div_K[idx,0],    # ∂/∂x
                theta*sig*Div_K[idx,1]     # ∂/∂y
            ],
            phi=phi,
            order=order,
            eps=eps_M[idx],
            chunk_size=None
        )

    # expand to vector system (2*Nu)
    A  = expand_rows(A_temp, idx, N)

    if sig <0:
        idx_I = np.hstack((groups['boundary:inlet'], groups['boundary:outlet'])) 
        A_fija = weight_matrix(
            x=nodes[idx_I],
            p=nodes,
            n=n_stencil,
            diffs=[[1,0]],
            coeffs=[-K[idx_I,0]],
            phi=phi, order=order, 
            eps= eps_M[idx_I],
            chunk_size=None
        )

        A_wall = weight_matrix(
            x=nodes[groups['boundary:wall']],
            p=nodes,
            n=n_stencil,
            diffs=[[1,0],  # ∂/∂x
                [0,1]],    # ∂/∂y
            coeffs=[
                -K[groups['boundary:wall'], 0]*normals[groups['boundary:wall'], 0],  # ∂/∂x
                -K[groups['boundary:wall'], 1]*normals[groups['boundary:wall'], 1]   # ∂/∂y
            ],
            phi=phi, order=order, 
            eps= eps_M[groups['boundary:wall']],
            chunk_size=None
        )
        
        # idx_f = groups['boundary:outlet']
        # A_free = weight_matrix(
        #     x=nodes[idx_f],
        #     p=nodes,
        #     n=n_stencil,
        #     diffs=[
        #         [2,0],   # ∂²/∂x²
        #         [0,2],   # ∂²/∂y²
        #         [1,0],   # ∂/∂x
        #         [0,1]    # ∂/∂y
        #     ],
        #     coeffs=[
        #         -K[idx_f,0]*normals[idx_f, 0],        # ∂²/∂x²
        #         -K[idx_f,1]*normals[idx_f, 1],        # ∂²/∂y²
        #         -Div_K[idx_f,0]*normals[idx_f, 0],    # ∂/∂x
        #         -Div_K[idx_f,1]*normals[idx_f, 1]     # ∂/∂y
        #     ],
        #     phi=phi,
        #     order=order,
        #     eps=eps_M[idx_f],
        #     chunk_size=None
        # )

        A += expand_rows(A_fija,  np.hstack((groups['ghosts:inlet'], groups['ghosts:outlet'])), N) 
        A += expand_rows(A_wall, groups['ghosts:wall'], N)

        #A += expand_rows(A_free, groups.get('ghosts:outlet', []), N)

        return A.tocsc()

    return A.tocsc()

def build_transport_matrix(V, nodes, groups, normals, pho, D, Div_D, eps_M, 
                            gauss_p, sig):
    """
    Construye matrices A_c, B_c para:
    ∂(φCR)/∂t - (u·∇C) - ∇·(D∇C) + λφCR = W_+ - W_-
    con esquema trapezoidal en el tiempo.
    """
    N = len(nodes)

    U_div_D =(V - Div_D)/2.

    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    A_temp = weight_matrix(
        x=nodes[idx],
        p=nodes,
        n=n_stencil,
        diffs=[
            [0,0],   # identidad
            [2,0],   # ∂²/∂x²
            [0,2],   # ∂²/∂y²
            [1,0],   # ∂/∂x
            [0,1]    # ∂/∂y
        ],
        coeffs=[
            2*R*pho[idx]*[1./dt - sig*alpha_sum] -sig*gauss_p[idx],   # identidad
            sig*D[idx,0],        # ∂²/∂x²
            sig*D[idx,1],        # ∂²/∂y²
            -sig*U_div_D[idx,0],    # ∂/∂x
            -sig*U_div_D[idx,1]     # ∂/∂y
        ],
        phi=phi,
            order=order,
            eps=eps_M[idx],
            chunk_size=None
        )

    # expand to vector system
    A  = expand_rows(A_temp, idx, N) 
    

    if sig <0:
        idx_G =np.hstack((groups['ghosts:wall'], groups['ghosts:inlet'], groups['ghosts:outlet']))
        idx_I = np.hstack((groups['boundary:wall'], groups['boundary:inlet'], groups['boundary:outlet']))
        A_fija = weight_matrix(
            x=nodes[idx_I],
            p=nodes,
            n=n_stencil,
            diffs=[[1,0],
                [0,1]],  
            coeffs=[
                -D[idx_I, 0]*normals[idx_I, 0], 
                -D[idx_I, 1]*normals[idx_I, 1] 
            ],
            phi=phi, order=order, 
            eps= eps_M[idx_I],
            chunk_size=None
        )
        
        A += expand_rows(A_fija, idx_G, N)
        
        #diagnose_matrix(A)
        return splu(A.tocsc())

    return A.tocsc()

def build_mrmt_block(coeffs, idx=None):
    if idx is None:
        idx = np.arange(len(nodes))

    return weight_matrix(
        x = nodes[idx],
        p = nodes,
        n = n_stencil,
        diffs = [[0,0]],
        coeffs = [coeffs],
        phi = phi,
        order = order,
        eps = eps_M[idx],
        chunk_size=None
    )

# ---------------------------
# 5) RHS builders adapted for two meshes
# ---------------------------
def H_vector(H, M_right, gauss_p, groups, nodes, N, operador= None, Qout=0):
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))

    b = np.zeros(N)
    if operador == "Identity":
        aux = H
    else:
        aux = M_right @ H - Qout*gauss_p

    b[idx] = aux[idx]
    apply_boundary(b, nodes, groups.get('ghosts:inlet', []),
                    In_h(zi_max, inlet_z_t, inlet_z_t, inlet_a))
    apply_boundary(b, nodes, groups.get('ghosts:outlet', []),
                    In_h(zo_max, zo_min, 0.4*zo_max + 0.6*zo_min,
                          outlet_a, outlet_a/150))

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
   
def C_vector(C, C_im, M_right, gauss_f, groups, t, T, N):
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))
    
    b = np.zeros(N)
    aux = M_right @ C 
    b[idx] = aux[idx]

    # aporte MRMT
    b[idx] += np.sum(
        4 * R * alpha_r[:, None] * pho[idx] * C_im[:, idx],
        axis=0
    )



    # aporte fuente
    b[idx] += (fuente_C(t, T) + fuente_C(t-dt, T))*gauss_f[idx]
    return b

# ---------------------------
# 6) Condiciones de frontera (igual que antes, pero aplicadas en malla u)
# ---------------------------

def apply_boundary(U, nodes, dirich_nd, inlet_obj: In_h):
    """
    Aplica las condiciones de entrada usando un objeto tipo In_h.

    U : array        → campo de velocidad/variable a imponer
    nodes : array    → coordenadas de los nodos (N x 2)
    dirich_nd : list → índices de nodos en la frontera de entrada
    inlet_obj : In_h → objeto que define el perfil de entrada
    """

    if dirich_nd is None or len(dirich_nd) == 0:
        return

    # extraer z de los nodos en la frontera (segunda columna)
    z = nodes[dirich_nd, 1]

    # asignar valores a U en los nodos de frontera
    U[dirich_nd] = inlet_obj(z)

def get_init_values(new_nodes,
                    filename="/home/alex/funcion.h5"):

    # Leer datos del archivo HDF5
    with h5py.File(filename, "r") as h5f:
        h = np.array(h5f["h"][:], dtype=float)           # shape = (Nh,)
        c = np.array(h5f["c"][:], dtype=float)           # shape = (Nc,)

        nodes_h = np.array(h5f["nodes_G"][:], dtype=float)[:, :2]  # shape = (Nh,2)
        nodes_c = np.array(h5f["nodes_Q"][:], dtype=float)[:, :2]  # shape = (Nc,2)

    # --------------------------
    # VALIDACIONES IMPORTANTES
    # --------------------------

    if nodes_h.shape[1] != 2:
        raise ValueError(f"nodes_G debe tener forma (N,2) y es {nodes_h.shape}")

    if nodes_c.shape[1] != 2:
        raise ValueError(f"nodes_Q debe tener forma (N,2) y es {nodes_c.shape}")

    if nodes_h.shape[0] != h.shape[0]:
        raise ValueError("nodes_G y h no coinciden en número de nodos")

    if nodes_c.shape[0] != c.shape[0]:
        raise ValueError("nodes_Q y c no coinciden en número de nodos")

    # Si vienen desordenados:
    # (Clough-Tocher funciona mejor si los nodos están ordenados)
    order_h = np.lexsort((nodes_h[:,1], nodes_h[:,0]))
    order_c = np.lexsort((nodes_c[:,1], nodes_c[:,0]))

    nodes_h = nodes_h[order_h]
    nodes_c = nodes_c[order_c]
    h = h[order_h]
    c = c[order_c]

    # --------------------------
    # CREAR INTERPOLADORES
    # --------------------------

    # Interpolador principal (C1 suavizado)
    interp_h_ct = CloughTocher2DInterpolator(nodes_h, h, fill_value=np.nan)

    # Interpolador de respaldo basado en vecino más cercano
    interp_h_nn = NearestNDInterpolator(nodes_h, h)

    # Evaluar en puntos XY
    h_ct = interp_h_ct(new_nodes)         # puede generar NaN fuera del dominio
    mask_nan = np.isnan(h_ct)

    # Reemplazar NaN con vecino más cercano
    h_new = h_ct.copy()
    h_new[mask_nan] = interp_h_nn(new_nodes[mask_nan])

    interp_c = CloughTocher2DInterpolator(nodes_c, c, fill_value = 0)
    c_new = interp_c(new_nodes)

    return h_new, c_new

import scipy.sparse.linalg as sla
def diagnose_matrix(A):

    print("----- MATRIX DIAGNOSTICS -----")

    print("Shape:", A.shape)

    print("Nonzeros:", A.nnz)

    print("Density:", A.nnz/(A.shape[0]*A.shape[1]))

    # filas vacias
    zero_rows = np.where(A.getnnz(axis=1) == 0)[0]
    print("Zero rows:", len(zero_rows))

    # columnas vacias
    zero_cols = np.where(A.getnnz(axis=0) == 0)[0]
    print("Zero cols:", len(zero_cols))

    # diagonal
    diag = A.diagonal()
    print("Min diagonal:", diag.min())
    print("Max diagonal:", diag.max())

    # verificar NaN
    print("Has NaN:", np.isnan(A.data).any())

    # verificar inf
    print("Has Inf:", np.isinf(A.data).any())


    print("------------------------------")

# ---------------------------
# 7) Paso temporal (Chorin) con dos mallas
# ---------------------------
def step_time(H, C, C_im, nodes, groups, normals, A_solver, eps_M, K, grad,
                  pho, D_f, A_right, delta_p, gauss_p, gauss_f, Qout,
                  t, T, mask_h):
    """
    U: vector en malla u (2*Nu)
    ops: dict con operadores
    """
    idx = np.hstack((groups['interior'], groups['boundary:inlet'], groups['boundary:outlet'], groups['boundary:wall']))
    
    N = len(nodes)
    rhs = H_vector(H, A_right, delta_p, groups, nodes, N, None, Qout)
    H_ = A_solver.solve(rhs)
    H = np.where(mask_h, H, H_)

    U = U_vector(H, K, grad)

    D = D_total(D_f, U)
    Div_D = Div_KD(D, grad)

    C_solver = build_transport_matrix(U, nodes, groups, normals, pho, D, Div_D, eps_M, gauss_p, -1)
    C_right  = build_transport_matrix(U, nodes, groups, normals, pho, D, Div_D, eps_M, gauss_p, 1)
    rhs = C_vector(C, C_im, C_right, gauss_f, groups, t, T, N)
    # solución transporte puro

    C_new= C_solver.solve(rhs)

    C_im_new = np.zeros_like(C_im)

    coef = dt * alpha_r / beta 

    C_im_new[:, idx] = (
        coef[:, None] * (C_new[idx] + C[idx])
        + (1 - 2*coef[:, None]) * C_im[:, idx]
    )

    return H, U, C_new, C_im_new


def plot_mesh(nodes, groups, simplices, title='Malla', figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Nodos base
    ax.scatter(nodes[:, 0], nodes[:, 1], s=0.5, color='gray', label='nodos', zorder=2)
    
    # Grupos (entrada, salida, paredes, obstáculo, etc.)
    for name, idxs in groups.items():
        ax.scatter(nodes[idxs, 0], nodes[idxs, 1], s=1, label=name, zorder=3)
    
    # Conectividad (malla)
    # for sp in simplices:
    #     ax.plot(nodes[sp, 0], nodes[sp, 1], 'k-', linewidth=0.5, zorder=1)
    
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def set_limits(ax, delt=150):
    """
    Ajusta los límites del eje con un margen adicional `delt`.

    Parámetros
    ----------
    ax : axis de matplotlib
    xmin, xmax, ymin, ymax : float
        Límites reales del dominio.
    delt : float
        Margen extra alrededor del dominio.
    """
    ax.set_xlim(xmin - delt, xmax + delt)
    ax.set_ylim(ymin - delt, ymax + delt)

def new_H_init(H, M_right, gauss_p, groups, N, eps_arr,
                 nodes, normals, pho, K, div_K, eps_M):
    
    H_solver = splu(build_H_matrix(nodes, groups, normals, theta, pho, K, div_K, eps_M, -1,  eps_arr, "Identity"))

    vec = H_vector(H, M_right, gauss_p, groups, nodes, N, operador= "Identity")

    return H_solver.solve(vec)

# ==============================
#     CLASE SIMULATION
# ==============================
def update_supertitles(
        fig_H,
        fig_V,
        fig_C,
        fig_C_im,
        t0
):
    """
    Actualiza títulos para H, V, C y C_im(r)

    Parameters
    ----------
    fig_H, fig_V, fig_C : matplotlib.figure.Figure

    fig_C_im : list
        Lista de figuras MRMT

    t0 : float
        Tiempo en segundos
    """

    t_hours = t0 / 3600.0

    title_H = fig_H.suptitle(
        f"Spatial Distribution of Hydraulic Head H (m) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )

    title_V = fig_V.suptitle(
        f"Velocity Magnitude Field |U| (m/s) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )

    title_C = fig_C.suptitle(
        f"Solute Concentration Field C (kg/m³) at t = {t_hours:.2f} h",
        fontsize=20,
        color="#DCCB7F",
        y=0.95
    )

    titles_C_im = []

    for r, fig in enumerate(fig_C_im):

        title = fig.suptitle(

            f"Immobile Concentration C_im(r={r}) "
            f"(kg/m³) at t = {t_hours:.2f} h",

            fontsize=20,

            color="#DCCB7F",

            y=0.95

        )

        titles_C_im.append(title)

    return (
        title_H,
        title_V,
        title_C,
        titles_C_im
    )


class Simulation:
    def __init__(self, H_list, C_list, C_im_list, Qout):
        self.H = [h.copy() for h in H_list]   # 4 H independientes
        self.C = [c.copy() for c in C_list]   # 4 C independientes
        self.C_im = [m.copy() for m in C_im_list]   # 4 C independientes
        self.Qout = Qout.copy()
        self.t = 0.0

    def update(self, step, mask_h):
        self.t += dt 
        for k in range(4): 
            self.H[k], U, self.C[k], self.C_im[k] = step_time( H=self.H[k], C=self.C[k], C_im=self.C_im[k],
                                         nodes=nodes, groups=groups, normals=normals, A_solver=A_solver, 
                                         eps_M=eps_M, K=K, grad=grad, pho=pho, D_f=D_f, A_right=A_right,
                                         delta_p=delta_p, gauss_p=gauss_p, gauss_f=gauss_f, Qout=self.Qout[k],
                                         t=self.t, T=T, mask_h=mask_h) 
                                     
            i, j = divmod(k, 2)
                                          
            # ----- campo H ----- S
            SH = I.dot(self.H[k]); SH[~mask] = np.nan 
            SH = SH.reshape(xy_grid.shape[1:]).T 

            im_H[i][j].set_data(SH)
             
            # ----- velocidad ----- 
            speed = np.sqrt(U[:,0]**2 + U[:,1]**2) 
            Sv = I.dot(speed); Sv[~mask] = np.nan 
            Sv = Sv.reshape(xy_grid.shape[1:]).T 

            im_V[i][j].set_data(Sv)

            Ux = U[:, 0]/speed 
            Uy = U[:, 1]/speed 
            Ux0 = I.dot(Ux); Ux0[~mask] = np.nan 
            Uy0 = I.dot(Uy); Uy0[~mask] = np.nan 

            Ux0 = Ux0.reshape(xy_grid.shape[1:]).T 
            Uy0 = Uy0.reshape(xy_grid.shape[1:]).T 

            quiv_V[i][j].set_UVC(Ux0[::sp_q, ::sp_q], Uy0[::sp_q, ::sp_q])
            
            # ----- campo C ----- 
            SC = I.dot(self.C[k]); 
            SC[~mask] = np.nan 
            SC = SC.reshape(xy_grid.shape[1:]).T 
            
            im_C[i][j].set_data(SC) 

            # ----- campo C_im -----
            for r in range(Nr):
                SC_im = I.dot(self.C_im[k][r])
                SC_im[~mask] = np.nan
                SC_im = SC_im.reshape(
                        xy_grid.shape[1:]
                ).T

                im_C_im[r][i][j].set_data(SC_im)


        # update_supertitles(
        #     fig_H, fig_V, fig_C, self.t
        # )

        print(self.t/3600)


if pre:
    print(">>> Ejecutando preprocesamiento...")
    #nodes, groups, normals, simplices, pozo_cor = build_nodes()
    nodes, groups, normals, simplices, pozo_cor = build_nodes_var()


    hU = local_lengthscale(nodes, k= n_stencil-1)
    eps_arr_ = eps_from_h(hU, scale=1.0)
    eps_M = np.repeat(eps_arr_[:, None], n_stencil, axis=1)
    eps_arr = np.repeat(eps_arr_[:, None], 1, axis=1)

    pho= gph.gen_malla(nodes)

    alpha_r = np.zeros((Nr))
    alpha_sum = 0

    phi_eff = pho
    #phi_eff = np.maximum(pho, 1e-1)

    for r in range(Nr):
        alpha_r[r] = (beta[r] * Deff[r]) / (2*beta[r]*R*L**2 + dt*Deff[r])
        alpha_sum += alpha_r[r]



    K = K_ope(pho)

    D_f = diffusion_operator(pho)

    grad = build_operators(nodes, eps_M)

    # --- Malla de velocidad U ---
    x = nodes[:,0]
    y = nodes[:,1]

    # dx_x = grad[0] @ x
    # dy_y = grad[1] @ y

    # print("Media Dx_u(x_u) ~", np.nanmean(dx_x), " (debería ser ~1)")
    # print("Media Dy_u(y_u) ~", np.nanmean(dy_y), " (debería ser ~1)")

    # dx_y = grad[0] @ y
    # dy_x = grad[1] @ x

    # print("Media Dx_u(y_u) ~", np.nanmean(dx_y), " (debería ser ~0)")
    # print("Media Dy_u(x_u) ~", np.nanmean(dy_x), " (debería ser ~0)")

    div_K= Div_KD(K, grad)

    A_left = build_H_matrix(nodes, groups, normals, theta, pho, K, div_K, eps_M, -1)
    A_right = build_H_matrix(nodes, groups, normals, 1-theta, pho, K, div_K, eps_M, 1)

    delta_p = delta_char(nodes, pozo_cor, 1e-6)
    gauss_p = gaussian_2d(nodes, pozo_cor, 1, epsilon_y)
    gauss_f = gaussian_2d(nodes, fnte, 1, epsilon_x, epsilon_y)

    # fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # # --- Gauss del pozo ---
    # sc1 = ax[0].scatter(x, y, c=gauss_p, cmap="viridis")
    # ax[0].set_title("Gaussian Pozo")
    # plt.colorbar(sc1, ax=ax[0])

    # # --- Gauss de la fuente ---
    # sc2 = ax[1].scatter(x, y, c=gauss_f, cmap="viridis")
    # ax[1].set_title("Gaussian Fuente")
    # plt.colorbar(sc2, ax=ax[1])

    # plt.tight_layout()
    # plt.show()



    N = nodes.shape[0]

    H0, C0 = get_init_values(nodes)
    H0 = new_H_init(H0, None, gauss_p, groups, N, eps_arr, 
                    nodes, normals, pho, K, div_K, eps_M)
    U = U_vector(H0, K, grad)

    # Duplicar para las 4 simulaciones
    H = [H0.copy() for _ in range(4)]
    C = [C0.copy() for _ in range(4)]
    C_im = [np.zeros((Nr, len(nodes))) for _ in range(4)]



    xy_grid = np.mgrid[xmin:xmax:400j, ymin:ymax:80j]
    xy = xy_grid.reshape(2, -1).T

    _, stencils = KDTree(nodes).query(xy)  # stencils.shape = (N_xy, 1)
    eps_matrix = eps_arr_[stencils]

    I = weight_matrix(x=xy, p=nodes, n=1, diffs=[[0,0]], phi=phi,
                    eps= eps_matrix, chunk_size=None)

    # Crear los segmentos del polígono (pares de índices)
    segments = np.column_stack((
        np.arange(len(vert)),
        np.roll(np.arange(len(vert)), -1)
    ))

    # Mask usando el polígono en lugar del rectángulo
    mask = contains(xy, vert, segments)

    # ========================================
    #        GUARDAR TODO EN UN SOLO PKL
    # ========================================
    data = dict(
        nodes=nodes, groups=groups, normals=normals, simplices=simplices,
        pozo_cor=pozo_cor, hU=hU, eps_arr_=eps_arr_, eps_M=eps_M,
        eps_arr=eps_arr, pho=pho, K=K, D_f=D_f, grad=grad,
        div_K=div_K, A_left=A_left, A_right=A_right,
        delta_p=delta_p, gauss_p=gauss_p, gauss_f=gauss_f,
        H=H, C=C, U=U, C_im =C_im, alpha_r = alpha_r, alpha_sum =alpha_sum,
        xy_grid=xy_grid, xy=xy, I=I, mask=mask,
        stencils=stencils
    )

    with open(savefile, "wb") as f:
        pickle.dump(data, f)

    print(">>> Preprocesamiento completado y guardado en", savefile)

else:

    print(">>> Cargando datos desde preproceso.pkl...")
    with open(savefile, "rb") as f:
        data = pickle.load(f)

    globals().update(data)

    print(">>> Preprocesamiento cargado.")

# plot_mesh(nodes, groups, simplices, title='Malla de Velocidad (U)')
A_solver = splu(A_left)

# ==============================
#     FIGURAS 2×2
# ==============================

fig_H, axes_H = plt.subplots(2, 2, figsize=(18, 7))
fig_V, axes_V = plt.subplots(2, 2, figsize=(18, 7))
fig_C, axes_C = plt.subplots(2, 2, figsize=(18, 7))

fig_C_im = []
axes_C_im = []
im_C_im = []

for r in range(Nr):

    fig, axes = plt.subplots(2,2,figsize=(18,7))

    fig_C_im.append(fig)
    axes_C_im.append(axes)

    im_C_im.append([[None]*2 for _ in range(2)])


figs = [fig_H, fig_V, fig_C] + fig_C_im


def apply_subplot_layout(
    figs,
    left=0.05,
    right=0.97,
    top=0.83,
    bottom=0.05,
    wspace=0.12,
    hspace=0.12
):
    """
    Aplica el mismo layout de subplots a una lista de figuras.

    Parameters
    ----------
    figs : list[matplotlib.figure.Figure]
        Lista de figuras (ej. [fig_H, fig_V, fig_C])
    """
    for fig in figs:
        fig.subplots_adjust(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            wspace=wspace,
            hspace=hspace
        )

apply_subplot_layout(figs)

im_H = [[None]*2 for _ in range(2)]
im_V = [[None]*2 for _ in range(2)]
im_C = [[None]*2 for _ in range(2)]
im_C_im = [
    [[None]*2 for _ in range(2)]
    for _ in range(Nr)
]
quiv_V = [[None]*2 for _ in range(2)]

# Coordenadas del grid
X = xy_grid[0].T
Y = xy_grid[1].T

# ==============================
#     INICIALIZACIÓN SUBPLOTS
# ==============================

Qout = [0, 1e-5, 1e-4, 1e-3]

mask_h_cor = np.isclose(nodes[:,0], xmin) & (nodes[:,1] < -4000)


from matplotlib.colors import LinearSegmentedColormap
base = plt.cm.get_cmap("turbo")

# Recorte para eliminar el azul oscuro inicial
colors = base(np.linspace(0.08, 1.0, 256))

cmap_paraview_like = LinearSegmentedColormap.from_list(
    "turbo_paraview",
    colors
)

base = plt.cm.get_cmap("inferno_r")
colors_main = base(np.linspace(0, 1.0, 240))
gray_zero = np.array([[0.92, 0.92, 0.92, 1.0]]) 

# Recorte para eliminar el azul oscuro inicial
colors = np.vstack((gray_zero, colors_main))


cmap_paraview_h = LinearSegmentedColormap.from_list(
    "turbo_paraview",
    colors,
    256
)

for k in range(4):
    i, j = divmod(k, 2)

    # ----- CAMPO H -----
    H0 = I.dot(H[k]);  H0[~mask] = np.nan
    H0 = H0.reshape(xy_grid.shape[1:]).T
    
    im_H[i][j] = axes_H[i,j].imshow(H0, vmin=305, vmax=309,
                                    cmap= cmap_paraview_like, origin="lower",
                                    extent=[xmin, xmax, ymin, ymax])
    axes_H[i,j].set_title(f"(Q={Qout[k]})")

    # ----- VELOCIDAD -----
    speed0 = np.sqrt(U[:,0]**2 + U[:,1]**2)
    S0 = I.dot(speed0); S0[~mask] = np.nan
    S0 = S0.reshape(xy_grid.shape[1:]).T

    im_V[i][j] = axes_V[i,j].imshow(S0, vmin=4e-12, vmax=4.5e-4,
                                    cmap=cmap_paraview_like, origin="lower",
                                    extent=[xmin, xmax, ymin, ymax])

    Ux = I.dot(U[:,0] / speed0); Ux[~mask] = np.nan
    Uy = I.dot(U[:,1] / speed0); Uy[~mask] = np.nan

    Ux0 = Ux.reshape(xy_grid.shape[1:]).T
    Uy0 = Uy.reshape(xy_grid.shape[1:]).T

    Xs = X[::sp_q, ::sp_q]
    Ys = Y[::sp_q, ::sp_q]
    Ux_s = Ux0[::sp_q, ::sp_q]
    Uy_s = Uy0[::sp_q, ::sp_q]

    quiv_V[i][j] = axes_V[i,j].quiver(
        Xs, Ys, Ux_s, Uy_s,
        scale=50, width=0.001, color="k"
    )

    axes_V[i][j].set_title(f"(Q={Qout[k]})")

    # ----- CONCENTRACIÓN -----
    C0 = I.dot(C[k]);  C0[~mask] = np.nan
    C0 = C0.reshape(xy_grid.shape[1:]).T

    im_C[i][j] = axes_C[i,j].imshow(C0, vmin=0, vmax=7.6e-11,
                                    cmap=cmap_paraview_h, origin="lower",
                                    extent=[xmin, xmax, ymin, ymax])
    
    axes_C[i][j].set_title(
        f"Caudal de extracción Q = {Qout[k]:.3e} m³/s",
        fontsize=12
    )

    for r in range(Nr):

        C0_im = I.dot(C_im[k][r])
        C0_im[~mask] = np.nan

        C0_im = C0_im.reshape(
            xy_grid.shape[1:]
        ).T

        im_C_im[r][i][j] = axes_C_im[r][i,j].imshow(
            C0_im,
            vmin=0,
            vmax=7.6e-11,
            cmap=cmap_paraview_h,
            origin="lower",
            extent=[xmin,xmax,ymin,ymax]
        )

        axes_C_im[r][i,j].set_title(
            f"MRMT r={r} Q={Qout[k]:.3e}"
        )


# title_H, title_V, title_C = update_supertitles(
#     fig_H, fig_V, fig_C, t0
# )

# Datos necesarios para el loop
ims = [
    im_H[1][1],
    im_V[1][1],
    im_C[1][1]
]

for r in range(Nr):

    ims.append(
        im_C_im[r][1][1]
    )

y_pos  = [0.94, 0.94, 0.94, 0.94]   # misma altura para las 3 colorbars (puedes cambiarlo)

for fig, im, ypos in zip(figs, ims, y_pos):

    # Crear el eje para la colorbar (manual)
    cax = fig.add_axes([0.15, ypos, 0.70, 0.02])   # [x, y, width, height]

    # Crear la colorbar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

    # Números en rojo
    cbar.ax.tick_params(colors='red')


# ==============================
#     ANIMACIÓN SIN DUPLICAR UPDATE
# ==============================
# from matplotlib.animation import PillowWriter

# # Writers GIF para las 3 figuras
# writer_H = PillowWriter(fps=20)
# writer_V = PillowWriter(fps=20)
# writer_C = PillowWriter(fps=20)


# sim =Simulation(H.copy(), C.copy(), Qout) 

# frames = int(T/dt) 
# print("Generando animaciones...")

# with writer_H.saving(fig_H, "ns/figures/H.gif", dpi=150), \
#     writer_V.saving(fig_V, "ns/figures/V.gif", dpi=150), \
#     writer_C.saving(fig_C, "ns/figures/C.gif", dpi=150):

#     for n in range(frames):
#         # --- física + actualizar artistas (H, V, C) ---
#         sim.update(n)

#         # --- guardar frame de cada figura ---
#         writer_H.grab_frame()
#         writer_V.grab_frame()
#         writer_C.grab_frame()


# t1 = time.perf_counter()
# print(f"Tiempo de ejecución: {t1 - t0:.6f} segundos")from matplotlib.animation import PillowWriter


import time
from matplotlib.animation import FFMpegWriter

# ==============================
# CONFIGURACIÓN VIDEO MP4
# ==============================
fps = 20
writer_kwargs = dict(
    fps=fps,
    codec="libx264",
    bitrate=3000,
    extra_args=["-pix_fmt", "yuv420p"]  # máxima compatibilidad
)

writer_H = FFMpegWriter(**writer_kwargs)
writer_V = FFMpegWriter(**writer_kwargs)
writer_C = FFMpegWriter(**writer_kwargs)
writer_C_im = [
    FFMpegWriter(**writer_kwargs)
    for r in range(Nr)
]



# ==============================
# INICIALIZACIÓN
# ==============================

sim = Simulation(H.copy(), C.copy(), C_im.copy(), Qout)

frames = int(T / dt)
print("Generando animaciones MP4...")

t_start = time.perf_counter()

# ==============================
# EXPORTACIÓN MP4
# ==============================
context = [
    writer_H.saving(
        fig_H,
        "ns/figures/H.mp4",
        dpi=150
    ),

    writer_V.saving(
        fig_V,
        "ns/figures/V.mp4",
        dpi=150
    ),

    writer_C.saving(
        fig_C,
        "ns/figures/C.mp4",
        dpi=150
    )
]

for r in range(Nr):

    context.append(

        writer_C_im[r].saving(

            fig_C_im[r],

            f"ns/figures/C_im_r{r}.mp4",

            dpi=150
        )
    )

from contextlib import ExitStack

with ExitStack() as stack:

    for c in context:
        stack.enter_context(c)

    for n in range(frames):

        sim.update(n, mask_h_cor)

        writer_H.grab_frame()
        writer_V.grab_frame()
        writer_C.grab_frame()

        for r in range(Nr):

            writer_C_im[r].grab_frame()


t_end = time.perf_counter()
print(f"Tiempo total de ejecución: {t_end - t_start:.2f} s")
