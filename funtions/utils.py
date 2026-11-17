import numpy as np
from scipy.sparse.linalg import splu
from sklearn.neighbors import NearestNeighbors
from scipy.special import ellipe
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
import h5py
from funtions.runtime import RUNTIME


class In_h():
    def __init__(self, z_0, z_min, z_t2, amplitud, v_min= 0):
        self.amplitud = amplitud
        self.z_0 = z_0
        self.z_t = z_0 - 100
        self.z_t2 = z_t2
        self.z_min = z_min
        self.v_min = v_min

    def __call__(self, x):
        p = RUNTIME.get()    
        z = x[1]

        if p.domain == "real":
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
        elif p.domain in ("benchmark", "cuadrado"):
            Lz = self.z_0 - self.z_min
            z_norm = (z - self.z_min) / Lz

            values = 4 * self.amplitud * z_norm * (1 - z_norm)

        else:
            raise ValueError(f"Dominio inválido: {p.domain}")

        return values

def gaussian_2d(x, x_s, amplitude, epsilon_x, epsilon_y=None):
    p = RUNTIME.get()
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
    p = RUNTIME.get()

    A = getattr(p, "intensidad_fuente", None)
    pico, base = (1e-11, 1e-12) if A is None else (float(A), float(A) / 10.0)
    return pico * t/T * np.exp(-8*t/T) + base * (1 - np.exp(-8*t/T))

def normalizar_delta_area(delta, wi):
    """Reescala la delta discreta (Σδ_i = 1) para que ∫δ dΩ ≈ Σ δ_i·w_i = 1,
    con w_i los pesos de cuadratura (área nodal). Así el caudal extraído es Q
    y no Q·(área de los nodos cerca del pozo): no depende de la malla."""
    s = float(np.sum(delta * wi))
    return delta / s if s > 0 else delta


def discrete_delta(nodes, x_p, radius):
    """
    Delta discreta compacta sobre nube de puntos.

    Parameters
    ----------
    nodes : (N,2)
    x_p   : (2,)
    radius: radio de soporte

    Returns
    -------
    delta : (N,)
    """

    # --------------------------------------------
    # distancia al punto fuente
    # --------------------------------------------

    diff = nodes - x_p

    r = np.linalg.norm(diff, axis=1)

    q = r / radius

    # --------------------------------------------
    # kernel compacto Wendland C2
    # --------------------------------------------

    delta = np.zeros(len(nodes))

    mask = q < 1.0

    qm = q[mask]

    delta[mask] = ((1.0 - qm)**4) * (4.0*qm + 1.0)

    # --------------------------------------------
    # normalización
    # --------------------------------------------

    s = np.sum(delta)

    if s > 0:
        delta /= s

    return delta

def update_pozo(d, pozo_cor):

    p = RUNTIME.get()

    d.pozo_cor = pozo_cor

    d.delta_p = discrete_delta(d.nodes, pozo_cor, p.radio_delta_pozo)
    if p.norm_delta_pozo == "area":
        from funtions.operators import rbf_integration_weights
        wi = getattr(d, "wi", None)
        d.delta_p = normalizar_delta_area(
            d.delta_p, wi if wi is not None else rbf_integration_weights(d.nodes))

    d.gauss_p = gaussian_2d(
        d.nodes, pozo_cor, 1, p.epsilon_pozo
    )

    return d

def get_solucion(props, filepath):

    with np.load(filepath, allow_pickle=True) as data:

        return [data[prop] for prop in props]


# ============================================================
# χε(Q)
# ============================================================

def chi_eps(Q, eps=1e-6):

    return 0.5 * (
        1.0 + np.tanh(Q / eps)
    )


def compute_functional(Q, zp, C, d):
    """Ĵ = γ w1 J₁ + κ w2 J₂ + w3 J₃  (su gradiente: funtions/adjunto_exacto.py; pesos w: funtions/optimizador.py).

    J₁ = ∫₀ᵀ ∫_Ω g_p(x) C(x,t)² dΩ dt   (kernel gaussiano normalizado, ∫g_p dΩ = 1)
    J₂ = |z_p − z₀|²
    J₃ = |z_p − z₀| ∫₀ᵀ Q(t)² dt
    """
    from funtions.optimizador import pesos_J
    p = RUNTIME.get()
    Nt = len(Q)
    w1, w2, w3 = pesos_J(p, Nt * p.dt)

    J1 = 0.0
    for n in range(Nt):
        C_n = np.asarray(C[n]).squeeze()
        J1 += p.gamma * w1 * np.sum(d.gauss_p * C_n**2 * d.wi) * p.dt

    J2 = p.koppa * w2 * abs(zp - d.z0)**2
    J3 = w3 * np.sum(Q**2) * p.dt * abs(zp - d.z0)

    return J1 + J2 + J3

# ====================================================
# Utilidades para calcular eps local
# ====================================================

def local_lengthscale(nodes, k=8):
    """
    Calcula h_i = distancia media a los k vecinos más cercanos (excluye el propio punto).
    Devuelve array (N,)
    """
    p = RUNTIME.get()

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(nodes)
    dists, idxs = nbrs.kneighbors(nodes)
    # ignorar la primera columna (distancia a sí mismo = 0)
    h = np.mean(dists[:, 1:], axis=1)
    return h

def eps_from_h(h, scale=1.0, min_factor=0.75, max_factor=1.25):
    p = RUNTIME.get()

    h = np.asarray(h)
    eps = scale / (h + 1e-16)              # relación inversa: menor h → mayor ε
    eps = np.clip(eps,                     # acotar razonablemente
                  np.min(eps)*min_factor,
                  np.max(eps)*max_factor)
    return eps

def K_ope(pho):
    p = RUNTIME.get()

    K = np.zeros((len(pho), 2))
    K[:, 1] = 8.3e-3 * p.g * p.d_z**2 * pho**3 / (p.nu * (1 - pho)**2)             # componente y
    K[:, 0] = K[:, 1] * p.alpha**2  # componente x
    
    return K

def D_total(Dif, V):
    """
    Dif : difusión molecular (float, (N,), o (N,2))
    V   : velocidades (N,2)
    a_l : dispersividad longitudinal
    a_t : dispersividad transversal
    """
    p = RUNTIME.get()
    V = np.asarray(V)

    # Componentes al cuadrado
    Vx2 = V[:, 0]**2
    Vy2 = V[:, 1]**2

    # Norma |V| con protección contra cero
    Vnorm = np.sqrt(Vx2 + Vy2)
    Vnorm[Vnorm == 0] = 1e-12   # evita división cero

    # Difusión por dispersión (N,2)
    D_disp = np.column_stack([
        p.a_l * Vx2 + p.a_t * Vy2,      # componente x
        p.a_t * Vx2 + p.a_l * Vy2       # componente y
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


def Div_KD(Op_vec, grad, anisotropic=True):

    if anisotropic:
        Kx, Ky = Op_vec.T
    else:
        Kx = Ky = Op_vec

    dKx_dx = grad[0] @ Kx
    dKy_dy = grad[1] @ Ky

    dKx_dy = grad[1] @ Kx
    dKy_dx = grad[0] @ Ky

    return np.column_stack([dKx_dx, dKy_dy, dKx_dy, dKy_dx])

# Función para definir el operador de difusión
def diffusion_operator(pho):
    p = RUNTIME.get()

    Diff = np.zeros((len(pho), 2))
    Diff[:,0] = 0.5 * p.D_d * pho* ellipe(1 - p.alpha**-2)
    Diff[:,1] = Diff[:,0]/ p.alpha
    
    return Diff


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

    # In_h lee z como x[1] (convención de fenicsx: x de forma (gdim, n)), así
    # que se le pasan las coordenadas completas transpuestas. Pasarle sólo el
    # vector z hacía que x[1] fuera el z del 2.º nodo → flujo uniforme en
    # toda la frontera en vez del perfil.
    U[dirich_nd] = inlet_obj(nodes[dirich_nd].T)

def get_init_values(new_nodes):
    p = RUNTIME.get()

    if p.domain == "cuadrado":
        # CI: carga constante (u = 0) y dominio contaminado salvo una franja
        # limpia de ancho C_banda_limpia junto al borde, por donde entra agua
        # limpia hacia el pozo (transición suave de 15 m).
        n = len(new_nodes)
        x, z = new_nodes[:, 0], new_nodes[:, 1]
        d = np.minimum.reduce([x - p.xmin, p.xmax - x, z - p.ymin, p.ymax - z])
        banda = float(getattr(p, "C_banda_limpia", 0.0))
        perfil = 0.5 * (1 + np.tanh((d - banda) / 15.0)) if banda > 0 else np.ones(n)
        return (np.full(n, float(getattr(p, "h_init_const", 300.0))),
                float(getattr(p, "C_init_const", 1.0)) * perfil)

    if p.domain == "real":
        filename="./data/input/funcion.h5"
    elif p.domain == "benchmark":
        filename="./data/input/benchmark.h5"
    else:
        raise ValueError(f"Dominio inválido: {p.domain}")


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

    interp_c = CloughTocher2DInterpolator(nodes_c, c, fill_value=0)
    c_new = interp_c(new_nodes)
    c_new = np.maximum(c_new, 0.0)  # CI no puede ser negativa físicamente

    # Sólo para pruebas de gradiente: C uniforme (J_m ≠ 0 en el pozo). None = CI del archivo.
    c_override = getattr(p, "C_init_override", None)
    if c_override is not None:
        c_new = np.full_like(c_new, float(c_override))
        mod = getattr(p, "C_init_modulacion", None)   # (A, λx, λz): C0 (1 + A cos(2πx/λx) cos(2πz/λz))
        if mod is not None:
            A, lx, lz = mod
            c_new = c_new * (1.0 + A * np.cos(2 * np.pi * new_nodes[:, 0] / lx) * np.cos(2 * np.pi * new_nodes[:, 1] / lz))

    return h_new, c_new

def new_H_init(H, M_right, gauss_p, groups, N, eps_arr,
                 nodes, normals, pho, K, div_K, eps_M):
    p = RUNTIME.get()
    from funtions.operators import  build_H_matrix, H_vector
    
    H_solver = splu(build_H_matrix(nodes, groups, normals, p.theta, pho, K, div_K, eps_M, -1,  eps_arr, "Identity"))

    vec = H_vector(H, M_right, gauss_p, groups, nodes, N, operador= "Identity")

    return H_solver.solve(vec)

