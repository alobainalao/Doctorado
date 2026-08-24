from ufl import (
    VectorConstant,
    TrialFunction,
    TestFunction,
    avg,
    conditional,
    sqrt,
    div,
    dot,
    le,
    dS,
    ds,
    dx,
    extract_blocks,
    grad,
    nabla_grad,
    gt,
    inner,
    outer, 
    as_vector,
    lhs,
    rhs,
    FacetNormal, 
    Measure,
    SpatialCoordinate,
    exp
)
import h5py
import pyvista
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from petsc4py import PETSc
import new_grid as ggt 
from dolfinx.fem import assemble_scalar
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.io import XDMFFile, gmshio, VTXWriter
from mpi4py import MPI
from scipy.special import ellipe
import matplotlib.pyplot as plt

def plot_subplot(ax, times, values, colors):
    for i in range(4):
        ax.plot(times, values[i], color=colors[i])
    ax.grid(True)

    ax.set_xlabel('t (d)', fontsize=9)
    ax.set_ylabel(r'C ($m^3$)', fontsize=8)
    
    # Ajuste fino de posición de etiquetas
    ax.xaxis.set_label_coords(1.05, -0.03)
    ax.yaxis.set_label_coords(-0.05, 1.1)

def plot_list(times, value, name_graf, colors, name="grafica", xlabel = "time", ylabel = ""):

    plt.figure(figsize=(8, 5))
    for i in range(4):
        plt.plot(times, value[i], linestyle="-", color=colors[i], label = name_graf[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(name+".png")

def fuente_C(t, T):
    return 1e-11 * t/T * np.exp(-8*t/T) + 1e-12 * (1 - np.exp(-8*t/T))

def K_ope(a, b, K_z, alpha):
    expr_vector  = as_vector([K_z* grad(a)[0]*alpha**2, K_z* grad(a)[1]])
    return inner(expr_vector, grad(b))

def D_ope(a, b, D):
    expr_vector  = as_vector([D[0]* grad(a)[0], D[1]* grad(a)[1]])
    return inner(expr_vector, grad(b))

# Función para definir el operador de difusión
def diffusion_operator(V, D_d, phi, alpha):
    D = fem.Function(V)
    D_m = fem.Function(V)
    Dm_n = fem.Function(V)

    expr_D = 0.5*D_d * phi*ellipe(1-alpha**-2)*as_vector([1, alpha**-1])
    D.interpolate(fem.Expression(expr_D, V.element.interpolation_points()))
    
    return D, D_m, Dm_n



class Out_h():
    def __init__(self, z_0, z_min, amplitud, d_vel):
        self.amplitud = amplitud
        self.z_0 = z_0
        self.z_t1 = z_0 - 100
        self.z_t2 = 0.7*z_0 + 0.3*z_min
        self.z_min = z_min
        self.dv = d_vel

    def __call__(self, x):
        dz = 100
        z = x[1]

        values = np.piecewise(
            z,
            [
                (z < self.z_min) | (z > self.z_0),
                (z <= self.z_0) & (z > self.z_t1),
                (z <= self.z_t1) & (z > self.z_t2),
                (z <= self.z_t2) & (z > self.z_min + dz),
                (z <= self.z_min + dz) & (z >= self.z_min)
            ],
            [
                lambda z_: 0,
                lambda z_: self.amplitud * np.cos((z_ - self.z_t1) * np.pi / (2 * (self.z_0 - self.z_t1))) ** 2,
                lambda z_: self.amplitud,
                lambda z_: (self.amplitud - self.dv) *
                          np.cos((z_ - self.z_t2) * np.pi / (2 * (self.z_t2 - self.z_min - dz))) ** 2 + self.dv,
                lambda z_: self.dv *
                          np.cos((z_ - self.z_min - dz) * np.pi / (2 * dz)) ** 2
            ]
        )

        return values
    
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

def dv( x0, xb):
    return  x0 - xb/2 +(x0-xb)/(4*np.pi)*np.sin(2* x0* np.pi / (x0-xb))

def create_solver(A, mesh_):
    solver = PETSc.KSP().create(mesh_.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.BCGS)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    return solver

def solve_step(L, a, bc, solver, sol):
    b = assemble_vector(L)
    apply_lifting(b, [a], [bc])
    set_bc(b, bc)
    solver(b, sol.x.petsc_vec)
    b.destroy()
    return sol

def variational_form_h_C(Q, G, R, landa, S_s, h, h_n, dt, theta, theta_C, K_z, alpha, h_inlet, h_outlet, ds_in, ds_out, QOut, Src, Src_n, delta, delta_C, phi, C, C_n, D, D_n, h_, u, u_n):
    # Definir funciones de prueba
    g_t = TestFunction(G)
    z_t = TestFunction(Q)

    # Ecuación variacional para h
    F_h = (
        inner(S_s*(h - h_n) / dt +  QOut*delta, g_t ) * dx
        + theta * K_ope(h, g_t, K_z, alpha) * dx
        + inner(h_inlet, g_t) * ds_in
        - inner(h_outlet, g_t) * ds_out
        + (1 - theta) * K_ope(h_n, g_t, K_z, alpha) * dx
    )

    # Ecuación variacional para C
    F_C = (
        R*inner(phi * (C - C_n) / dt, z_t) * dx

        + theta_C * D_ope(C, z_t, D) * dx
        + (1 - theta_C) * D_ope(C_n, z_t, D_n) * dx     

        + theta_C * inner(u, nabla_grad(C))* z_t* dx
        + (1 - theta_C) * inner(u_n, nabla_grad(C_n))* z_t* dx

        + theta_C *landa*R*inner(phi *C, z_t) * dx 
        + (1 - theta_C) *landa*R*inner(phi *C_n, z_t) * dx

        - theta_C *inner(Src, z_t) * dx 
        - (1 - theta_C) *inner(Src_n, z_t) * dx

        + theta_C *inner(delta_C*C, z_t) * dx 
        + (1 - theta_C) *inner(delta_C*C_n, z_t) * dx
    )

    # Formulación de las matrices y vectores
    a_h = fem.form(lhs(F_h))
    L_h = fem.form(rhs(F_h))
    a_C = fem.form(lhs(F_C))
    L_C = fem.form(rhs(F_C))

    return a_h, L_h, a_C, L_C

# Función para inicializar elementos y funciones
def boundaries_elements(mesh_, inlet_z_t, outlet_z_t, inlet_a, outlet_a, d_vel, zi_max, zo_max, s, j_in, facet_tags, inlet_marker, outlet_marker):
    
    from basix.ufl import element
    v_l2 = element("Lagrange", mesh_.topology.cell_name(), 2, shape=(mesh_.geometry.dim,))
    s_l1 = element("Lagrange", mesh_.topology.cell_name(), 1)
    s_l2 = element("Lagrange", mesh_.topology.cell_name(), 2)

    V = fem.functionspace(mesh_, v_l2)
    G = fem.functionspace(mesh_, s_l1)
    Q = fem.functionspace(mesh_, s_l2)
        
    h_inlet = fem.Function(G)
    inlet_h = In_h(zi_max, inlet_z_t, inlet_z_t, inlet_a)


    h_outlet = fem.Function(G)
    outlet_h = In_h(zo_max, outlet_z_t, 0.4*zo_max + 0.6*outlet_z_t, outlet_a, outlet_a/150)   

    factor = 1 #np.trapezoid(x_values, y_inlet)/np.trapezoid(x_values, y_outlet)

    h_inlet.interpolate(inlet_h)
    h_outlet.interpolate(outlet_h)


    C_inlet = fem.Function(Q)
    #C_inlet.interpolate(lambda x: np.where(x[1] > s, j_in, 0.0))
    C_inlet.interpolate(lambda x: np.full(x.shape[1], 0.0))
    bc_C = [fem.dirichletbc(C_inlet, fem.locate_dofs_topological(V, mesh_.topology.dim - 1, facet_tags.find(inlet_marker)))]

    # Crear medidas de integración con ds
    ds_in = Measure("ds", domain=mesh_, subdomain_data=facet_tags, subdomain_id=inlet_marker)
    ds_out = Measure("ds", domain=mesh_, subdomain_data=facet_tags, subdomain_id=outlet_marker)
    
    return V, G, Q, h_inlet, factor*h_outlet, bc_C, ds_in, ds_out

def initialize_variables(Q, G, V, h_0, C_0, x_p):
    phi_0 = fem.Function(G)
    Src = fem.Function(Q)
    Src_n = fem.Function(Q)
    h_n = fem.Function(G)  # Altura hidráulica en el paso anterior
    C_n = fem.Function(Q)  # Concentración en el paso anterior
    u_n = fem.Function(V)  # Velocidad en el paso anterior

    phi_0.interpolate(lambda x: np.full(x.shape[1], 1))
    h_n.interpolate(lambda x: np.full(x.shape[1], h_0))
    C_n.interpolate(lambda x: np.full(x.shape[1], C_0))
    Src_n.interpolate(lambda x: np.full(x.shape[1],0.0)) 
    return h_n, C_n, u_n, phi_0, Src, Src_n

def approximate_dirac_delta(Q, G, x_p, tol, QOut=0):
    
    # Definir la delta en la posición x_p
    delta = fem.Function(G)
    delta.interpolate(lambda x: np.where(np.sqrt((x[0] - x_p[0])**2 + (x[1] - x_p[1])**2) < tol, 1.0, 0.0))

    # Definir la delta desplazada en x_p + 300
    delta_in = fem.Function(Q)
    delta_in.interpolate(lambda x: np.where(np.sqrt((x[0] - (x_p[0] - 300))**2 + (x[1] - x_p[1])**2) < tol, 1.0, 0.0))

    # Definir la delta en x = 816
    delta_C = fem.Function(Q)   
    delta_C_inv = fem.Function(Q)

    return delta, delta_in, delta_C, delta_C_inv

def read_initial(Q, G):
    h2 = fem.Function(G)
    c2 = fem.Function(Q)

    # Cargar datos desde archivo .h5
    with h5py.File("funcion.h5", "r") as h5f:
        h2.x.array[:] = h5f["h"][:]
        c2.x.array[:] = h5f["c"][:]
    return h2, c2

def gaussian_2d(x, amplitude):
    x_s = np.array([9116, 109])
    epsilon_x = 5    # Ancho horizontal (x)
    epsilon_y = 30   # Ancho vertical (y)
    normalization = 1.0 / (2 * np.pi * epsilon_x * epsilon_y)
    return amplitude * normalization * np.exp(
        -(((x[0] - x_s[0])**2) / (2 * epsilon_x**2) +
          ((x[1] - x_s[1])**2) / (2 * epsilon_y**2))
    )

def main(pre=False):

    iter = [0, 1, 2] if pre else [3, 4, 5, 6]
    # ------------------- PARÁMETROS DEL PROBLEMA -------------------
    T, dt =[1e5, 1e7, 1e8, 2592000, 2592000, 2592000, 2592000], [1e3, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5] #86400   [1e5, 1e7, 1e8, 259200, 259200, 259200, 259200], [1e3, 1e3, 1e3, 1e2, 1e2, 1e2, 1e2] 
    theta, theta_C = 0.9, 0.9
    zi_max, zo_max, z_min = 308, 250, -2691
    inlet_z_t, outlet_z_t, inlet_a = -760, -2690.5, -2e-4
    d_vel = inlet_a/15
    outlet_a = inlet_a * dv(zi_max, inlet_z_t) / dv(zo_max, outlet_z_t)
    s, j_in = -100.0, 1.0
    C_0, h_0 = 0.0, zi_max
    inlet_marker, outlet_marker = 1, 3

    # -------------------- PARÁMETROS FÍSICOS ----------------------
    g, nu, d_z, alpha = 9.81, 1.055e-6, 1e-3, 2.0
    tol, QOut = 1e-6, [0, 0, 0, 0, 1e-5, 1e-4, 1e-3]
    R, landa = 1, 1e-10

    # -------------- PARÁMETROS DE DISPERSIÓN ---------------------
    a_l, a_t, D_d, eps = 10, 1, 1.2e-5, 1e-16

    mesh_, phi, facet_tags, x_p = ggt.gen_malla()
    x_p_extended = np.array([x_p[0], x_p[1], 0.0])
    V, G, Q, h_inlet, h_outlet, bc_C, ds_in, ds_out = boundaries_elements(mesh_, inlet_z_t, outlet_z_t, inlet_a, outlet_a, d_vel, zi_max, zo_max, s, j_in, facet_tags, inlet_marker, outlet_marker)
    
    # ----------- PARÁMETROS DE CONDICIONES DE FRONTERA -----------
    S_s, bc_h = phi, []


    # Definir variables para el paso actual
    h = TrialFunction(G)
    C = TrialFunction(Q)
    h_ = fem.Function(G) 
    C_ = fem.Function(Q)
    u = fem.Function(V) 
    gradH = fem.Function(V)

    D, D_m, Dm_n = diffusion_operator(V, D_d, phi, alpha)

    delta, delta_in, delta_C, delta_C_inv = approximate_dirac_delta(Q, G, x_p, tol, QOut)

    # Interpolación de funciones K_z, h_inlet, h_outlet, C_inlet
    K_z = fem.Function(G)
    expr_Ki = 8.3e-3 * g * d_z**2 * phi**3 / (nu * (1 - phi)**2)

    h_n, C_n, u_n, phi_0, Src, Src_n = initialize_variables(Q, G, V, h_0, C_0, x_p)
    expr_K_z = [500*phi_0, expr_Ki, expr_Ki, expr_Ki, expr_Ki, expr_Ki, expr_Ki]
    phi_a = [phi_0, phi, phi, phi, phi, phi, phi]


    from pathlib import Path
    # Crear carpeta principal
    base_folder = Path("results_tesis")
    base_folder.mkdir(exist_ok=True, parents=True)

    # Nombres de subcarpetas
    subfolders = ["sin_extraccion", "extraccion_baja", "extraccion_moderada", "extraccion_extrema"]
    colors = ["y", "g", "b", "r"]

    # Crear cada subcarpeta
    for name in subfolders:
        (base_folder / name).mkdir(exist_ok=True)

    writers = {
        "u": u_n,
        "C": C_n,
        "h": h_n
    }

    if not pre:
        h_initial, C_initial = read_initial(Q, G)
        C_s_out= [[], [], [], []]
        int_c = [[], [], [], []]
        


    for i in iter: 

        t = 0
        """
        kk= fem.Function(V)
        kk.interpolate(fem.Expression(as_vector([expr_K_z[2] * alpha**2,expr_K_z[2]]), V.element.interpolation_points()))
        vtx_k = VTXWriter(mesh_.comm, "results_tesis/k.bp", [kk], engine="BP4")
        vtx_k.write(t)
        vtx_k.close()
        """
        
        if i>=3:
            h_n.x.array[:] = h_initial.x.array[:]
            C_n.x.array[:] = C_initial.x.array[:]

            int_c[i-3].append(fem.assemble_scalar(fem.form(inner(1, C_n) * dx)))
            C_s_out[i-3].append(fem.assemble_scalar(fem.form(inner(delta_C, C_n) * dx)))



        K_z.interpolate(fem.Expression(expr_K_z[i], G.element.interpolation_points()))
        expr_un = as_vector([-K_z * alpha**2 * grad(h_n)[0],-K_z *grad(h_n)[1]])
        u_n.interpolate(fem.Expression(expr_un, V.element.interpolation_points()))
        expr_u = as_vector([-K_z * alpha**2 * grad(h_)[0],-K_z *grad(h_)[1]])

        #gradH.interpolate(fem.Expression(as_vector([-K_z* grad(h_n)[0],-K_z *grad(h_n)[1]]), V.element.interpolation_points()))

        if i>=3:           
            father_dir= "results_tesis/"+subfolders[i-3]
            writers_vtx = {}
            for name, field in writers.items():
                path = f"{father_dir}/{name}.bp"
                writers_vtx[name] = VTXWriter(mesh_.comm, path, [field], engine="BP4")
                writers_vtx[name].write(t)

        a_h, L_h, a_C, L_C = variational_form_h_C(Q, G, R, landa, S_s, h, h_n, dt[i], theta, theta_C, K_z, alpha, h_inlet, h_outlet, ds_in, ds_out, QOut[i], Src, Src_n, delta, delta_C, phi_a[i], C, C_n, D + D_m, D + Dm_n, h_, u, u_n)
       
        # Assemble matrices
        A1 = assemble_matrix(a_h, bcs= bc_h)
        A1.assemble()   
        solver1 = create_solver(A1, mesh_)

        if i>=2:   
            D_m_expr = as_vector([
                (a_l * u[0]**2 + a_t * u[1]**2) / sqrt(u[0]**2 + u[1]**2 + eps),
                (a_l * u[1]**2 + a_t * u[0]**2) / sqrt(u[0]**2 + u[1]**2 + eps)
            ])

            D_mn_expr = as_vector([
                (a_l * u_n[0]**2 + a_t * u_n[1]**2) / sqrt(u_n[0]**2 + u_n[1]**2 + eps),
                (a_l * u_n[1]**2 + a_t * u_n[0]**2) / sqrt(u_n[0]**2 + u_n[1]**2 + eps)
            ])

            epsilon = 30
            Q_f = float(QOut[i] != 0)
            x_p_np = np.array(x_p_extended)
            exp_DC = lambda x: Q_f / (2 * np.pi * epsilon**2) * np.exp(-np.sum((x.T - x_p_np)**2, axis=1) / (2 * epsilon**2))
            delta_C0 = fem.Function(Q)
            delta_C.interpolate(exp_DC)
            
        while t < T[i]:
            t += dt[i]

            # Resolver para h
            h_ = solve_step(L_h, a_h, bc_h, solver1, h_)
            h_n.x.array[:] = h_.x.array[:]

            u.interpolate(fem.Expression(expr_u, V.element.interpolation_points()))
            #gradH.interpolate(fem.Expression(as_vector([-K_z* grad(h_n)[0],-K_z *grad(h_n)[1]]), V.element.interpolation_points()))
        
            if i>=2: 
                Src.interpolate(lambda x: gaussian_2d(x, fuente_C(t, T[i])))

                D_m.interpolate(fem.Expression(D_m_expr , V.element.interpolation_points()))
                Dm_n.interpolate(fem.Expression(D_mn_expr , V.element.interpolation_points()))
    
                # Resolver para C
                A2 = assemble_matrix(a_C, bcs= bc_C)
                A2.assemble()

                solver2 = create_solver(A2, mesh_)
                C_ =solve_step(L_C, a_C, bc_C, solver2, C_)
    
                # Actualizar las variables de solución
                C_n.x.array[:] = np.maximum(C_.x.array[:], 0)
                Src_n.x.array[:] = Src.x.array[:]           

                # Write solutions to file
                
            if i>=3 :
                int_c[i-3].append(fem.assemble_scalar(fem.form(inner(1, C_n) * dx)))
                C_s_out[i-3].append(fem.assemble_scalar(fem.form(inner(delta_C, C_n) * dx)))

                for writer in writers_vtx.values():
                    writer.write(t)

            u_n.x.array[:] = u.x.array[:]

            print("t =", t, "i =", i)
        
        if i==2:
            # Guardar función manualmente en archivo .h5
            with h5py.File("funcion.h5", "w") as h5f:
                h5f.create_dataset("h", data=h_n.x.array)
                h5f.create_dataset("c", data=C_n.x.array)



    #if i ==6:
    if False:
        times = np.arange(0, T[i]+dt[i], dt[i])/86400

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.13, wspace=0.2)

        plot_subplot(axes[0], times, int_c[:] - fuente_C(times, T[i]), colors)
        plot_subplot(axes[1], times, C_s_out, colors)

        #plot_list(times, int_c[:] - fuente_C(times, T[i]), subfolders, colors, name="indomain", xlabel = "time", ylabel = "C_in_domain")
        #plot_list(times, C_s_out, subfolders, colors, name="outdomain", xlabel = "time", ylabel = "C_extraido")

        # Agregar etiquetas de anisometría (columnas) en la parte superior
        for j, type in enumerate(["Contaminate total en el dominio", "Contaminate en el punto de extraccion"]):
            fig.text(0.25 + j * 0.5, 0.02, f'{type}', ha='center', fontsize=10, color = "blue")

        # Crear elementos personalizados para la leyenda
        legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=2, label=subfolders[i]) for i in range(len(subfolders))]

        # Agregar la leyenda global a la figura
        fig.legend(handles=legend_elements, loc='upper center', ncol=4)
        plt.savefig("contaminante.png")

           

if __name__ == "__main__":
    main(True)
    main()