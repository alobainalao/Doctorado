"""
Forward solver FEM (DOLFINx) — núcleo físico extraído de
TesisCode/maintesis.py, adaptado para exponer un único run controlable
`solve_forward_fenicsx(Qout, pozo, p)` con el mismo contrato que
`view.animation.solve_forward` del backend bfr (RBF-FD): recibe Q(t) como
serie de tiempo y pozo=[x_p, z_p] como controles externos.

Lo que NO se trajo del script original: el pipeline de 7 etapas
(spin-up + 4 escenarios de Q fijo, estado guardado en funcion.h5). Eso era
un estudio de caso de la maestría, no una interfaz reutilizable para
optimización — se preserva tal cual en fenicsx/legacy/maintesis.py.

Cambios deliberados respecto al original (no son bugs, son adaptaciones
necesarias para Q(t) variable — documentadas aquí para que se revisen):

1. QOut pasa de ser un float de Python (horneado en la forma UFL al
   compilarla) a un `dolfinx.fem.Constant` cuyo `.value` se actualiza cada
   paso de tiempo — necesario para que Q(t) varíe sin recompilar la forma
   en cada paso. La matriz de flujo `A1` no depende de QOut (el término
   QOut*delta es puramente del lado derecho, ver variational_form_h_C), así
   que se ensambla y factoriza una sola vez — igual que en el original.

2. El gateo de la extracción de contaminante usaba una bandera binaria
   (Q_f = float(QOut[i] != 0)), fija para todo el escenario. Aquí se
   reemplaza por chi_eps(Q(t)) (funtions/utils.py — la misma función suave
   que ya usa el backend bfr), re-interpolada en delta_C cada paso. Mismo
   criterio de gateo (on/off según si se extrae), pero continuo en vez de
   0/1 — necesario para que el gradiente adjunto (a derivar) sea suave.

LIMITACIÓN CONOCIDA, no resuelta aquí: el término fuente/sumidero de la
ecuación de FLUJO (`delta`, vía approximate_dirac_delta) es una
aproximación "dura" que solo da un valor distinto de cero si hay un nodo de
malla exactamente en `pozo` (tol=1e-6) — depende de que `pozo` haya sido
embebido al generar la malla (fenicsx/mesh.py:build_mesh). Mover z_p sin
remallar NO mueve este término. El sumidero de TRANSPORTE (delta_C) sí usa
un perfil gaussiano suave (epsilon=30) evaluado directamente en `pozo`, así
que ese sí responde a z_p sin remallar. Antes de optimizar z_p con este
backend hay que decidir: (a) remallar en cada evaluación de z_p (caro, y el
adjunto tendría que incluir derivada de forma), o (b) sustituir `delta` por
un gaussiano suave igual que delta_C/Src, igualando el criterio ya usado en
bfr. No se tomó esa decisión aquí — hace falta antes de intentar
gradiente adjunto respecto a z_p en este backend.

Requiere dolfinx + gmsh + mpi4py + petsc4py + h5py (no están en
requirements.txt / bfr_env). Ver fenicsx/README.md.

NO EJECUTADO/VERIFICADO: no hay entorno dolfinx disponible en este sandbox.
Falta correrlo en un entorno real antes de confiar en los resultados.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from ufl import (
    TrialFunction, TestFunction, grad, nabla_grad, inner, as_vector,
    sqrt, lhs, rhs, dx, Measure,
)

from funtions.utils import chi_eps, get_init_values
from fenicsx.mesh import build_mesh

# ---------------------------------------------------------------------------
# Parámetros ESPECÍFICOS del backend FEM (geometría de los perfiles de
# frontera, markers de la malla y anchos de los sumideros gaussianos), tomados
# de TesisCode/maintesis.py. No existen en la config de bfr, así que se quedan
# aquí. El resto de parámetros físicos/numéricos —los que sí comparte con bfr
# (theta, g, nu, d_z, alpha, R, landa, a_l, a_t, D_d, eps)— se leen de la
# config `p` en solve_forward_fenicsx (ver _config_phys), para que la app
# controle ambos backends por igual y no queden valores horneados divergentes.
# ---------------------------------------------------------------------------
FEM_DEFAULTS = dict(
    zi_max=308, zo_max=250, z_min=-2691,
    inlet_z_t=-760, outlet_z_t=-2690.5, inlet_a=-2e-4,
    C_0=0.0,
    inlet_marker=1, outlet_marker=3,
    tol=1e-6,           # tolerancia del nodo-pozo del sumidero de flujo (no p.tol)
    epsilon_sink=30,    # ancho del sumidero gaussiano de transporte
)

# Parámetros con el MISMO significado y fórmula que en bfr: se toman de la
# config `p` para que la app los controle. Ver funtions/utils.py (dispersión,
# difusión) y funtions/operators.py (flujo) en el backend bfr.
_SHARED_WITH_BFR = ("theta", "g", "nu", "d_z", "alpha",
                    "R", "landa", "a_l", "a_t", "D_d", "eps")


def _config_phys(p):
    """Mezcla los defaults propios del FEM con los parámetros compartidos que
    vienen de la config `p` (misma fuente que usa bfr)."""
    phys = dict(FEM_DEFAULTS)
    for k in _SHARED_WITH_BFR:
        phys[k] = getattr(p, k)
    phys["theta_C"] = phys["theta"]   # bfr usa un único theta para flujo y transporte
    return phys


class Out_h:
    """Perfil de carga hidráulica de salida — copiado tal cual de maintesis.py."""

    def __init__(self, z_0, z_min, amplitud, d_vel):
        self.amplitud = amplitud
        self.z_0 = z_0
        self.z_t1 = z_0 - 100
        self.z_t2 = 0.7 * z_0 + 0.3 * z_min
        self.z_min = z_min
        self.dv = d_vel

    def __call__(self, x):
        dz = 100
        z = x[1]
        return np.piecewise(
            z,
            [
                (z < self.z_min) | (z > self.z_0),
                (z <= self.z_0) & (z > self.z_t1),
                (z <= self.z_t1) & (z > self.z_t2),
                (z <= self.z_t2) & (z > self.z_min + dz),
                (z <= self.z_min + dz) & (z >= self.z_min),
            ],
            [
                lambda z_: 0,
                lambda z_: self.amplitud * np.cos((z_ - self.z_t1) * np.pi / (2 * (self.z_0 - self.z_t1))) ** 2,
                lambda z_: self.amplitud,
                lambda z_: (self.amplitud - self.dv) *
                           np.cos((z_ - self.z_t2) * np.pi / (2 * (self.z_t2 - self.z_min - dz))) ** 2 + self.dv,
                lambda z_: self.dv *
                           np.cos((z_ - self.z_min - dz) * np.pi / (2 * dz)) ** 2,
            ],
        )


class In_h:
    """Perfil de carga hidráulica de entrada — copiado tal cual de maintesis.py."""

    def __init__(self, z_0, z_min, z_t2, amplitud, v_min=0):
        self.amplitud = amplitud
        self.z_0 = z_0
        self.z_t = z_0 - 100
        self.z_t2 = z_t2
        self.z_min = z_min
        self.v_min = v_min

    def __call__(self, x):
        z = x[1]
        return np.piecewise(
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
                lambda z_: (self.amplitud - self.v_min) * np.cos((z_ - self.z_t) * np.pi / (2 * (self.z_t - self.z_t2))) ** 2 + self.v_min,
                lambda z_: self.v_min * np.cos((z_ - self.z_t2) * np.pi / (2 * (self.z_t2 - self.z_min))) ** 2,
            ],
        )


def _dv(x0, xb):
    return x0 - xb / 2 + (x0 - xb) / (4 * np.pi) * np.sin(2 * x0 * np.pi / (x0 - xb))


def _fuente_C(t, T):
    return 1e-11 * t / T * np.exp(-8 * t / T) + 1e-12 * (1 - np.exp(-8 * t / T))


def _gaussian_2d(x, pozo_ext, amplitude):
    epsilon_x, epsilon_y = 5, 30
    normalization = 1.0 / (2 * np.pi * epsilon_x * epsilon_y)
    return amplitude * normalization * np.exp(
        -(((x[0] - pozo_ext[0]) ** 2) / (2 * epsilon_x ** 2) +
          ((x[1] - pozo_ext[1]) ** 2) / (2 * epsilon_y ** 2))
    )


def _create_solver(A, comm):
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.BCGS)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    return solver


def _solve_step(L, a, bc, solver, sol):
    b = assemble_vector(L)
    apply_lifting(b, [a], [bc])
    set_bc(b, bc)
    solver(b, sol.x.petsc_vec)
    b.destroy()
    return sol


def _diffusion_operator(V, D_d, phi, alpha):
    # D/D_m/Dm_n son campos VECTORIALES (2 componentes, ver expr_D) — deben
    # vivir en V (espacio vectorial), no en phi.function_space (G, escalar).
    # Bug introducido en el port (usaba phi.function_space); maintesis.py
    # original los definía sobre V, que es lo correcto.
    from scipy.special import ellipe
    D = fem.Function(V)
    D_m = fem.Function(V)
    Dm_n = fem.Function(V)
    expr_D = 0.5 * D_d * phi * ellipe(1 - alpha ** -2) * as_vector([1, alpha ** -1])
    D.interpolate(fem.Expression(expr_D, V.element.interpolation_points))
    return D, D_m, Dm_n


def _K_ope(a, b, K_z, alpha):
    expr_vector = as_vector([K_z * grad(a)[0] * alpha ** 2, K_z * grad(a)[1]])
    return inner(expr_vector, grad(b))


def _D_ope(a, b, D):
    expr_vector = as_vector([D[0] * grad(a)[0], D[1] * grad(a)[1]])
    return inner(expr_vector, grad(b))


def _boundaries_elements(domain, facet_tags, phys):
    v_l2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    s_l1 = element("Lagrange", domain.topology.cell_name(), 1)
    s_l2 = element("Lagrange", domain.topology.cell_name(), 2)

    V = fem.functionspace(domain, v_l2)
    G = fem.functionspace(domain, s_l1)
    Q = fem.functionspace(domain, s_l2)

    d_vel = phys["inlet_a"] / 15
    outlet_a = phys["inlet_a"] * _dv(phys["zi_max"], phys["inlet_z_t"]) / _dv(phys["zo_max"], phys["outlet_z_t"])

    h_inlet = fem.Function(G)
    h_inlet.interpolate(In_h(phys["zi_max"], phys["inlet_z_t"], phys["inlet_z_t"], phys["inlet_a"]))

    h_outlet = fem.Function(G)
    h_outlet.interpolate(In_h(
        phys["zo_max"], phys["outlet_z_t"],
        0.4 * phys["zo_max"] + 0.6 * phys["outlet_z_t"], outlet_a, outlet_a / 150,
    ))

    C_inlet = fem.Function(Q)
    C_inlet.interpolate(lambda x: np.full(x.shape[1], 0.0))
    # Fix respecto a maintesis.py (ver fenicsx/README.md, "Ya corregido"): el original
    # usaba el mapa de dofs de V (espacio vectorial) para una condición de
    # frontera sobre C_inlet, que vive en Q (espacio escalar) — dofmaps de
    # espacios distintos. Aquí se usa Q, el espacio correcto de C_inlet.
    bc_C = [fem.dirichletbc(
        C_inlet,
        fem.locate_dofs_topological(Q, domain.topology.dim - 1, facet_tags.find(phys["inlet_marker"])),
    )]

    ds_in = Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=phys["inlet_marker"])
    ds_out = Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=phys["outlet_marker"])

    return V, G, Q, h_inlet, h_outlet, bc_C, ds_in, ds_out


def _flow_forms(G, S_s, h, h_n, dt, theta, K_z, alpha,
                h_inlet, h_outlet, ds_in, ds_out, QOut_const, delta):
    """Formas (bilineal, lineal) del flujo θ-implícito. Se reusan tal cual
    para el spin-up estacionario (con otro dt), única fuente de verdad."""
    g_t = TestFunction(G)
    F_h = (
        inner(S_s * (h - h_n) / dt + QOut_const * delta, g_t) * dx
        + theta * _K_ope(h, g_t, K_z, alpha) * dx
        + inner(h_inlet, g_t) * ds_in
        - inner(h_outlet, g_t) * ds_out
        + (1 - theta) * _K_ope(h_n, g_t, K_z, alpha) * dx
    )
    return fem.form(lhs(F_h)), fem.form(rhs(F_h))


def _variational_form_h_C(Q, G, R, landa, S_s, h, h_n, dt, theta, theta_C, K_z, alpha,
                           h_inlet, h_outlet, ds_in, ds_out, QOut_const, Src, Src_n,
                           delta, delta_C, phi, C, C_n, D, D_n, u, u_n):
    z_t = TestFunction(Q)

    a_h, L_h = _flow_forms(G, S_s, h, h_n, dt, theta, K_z, alpha,
                           h_inlet, h_outlet, ds_in, ds_out, QOut_const, delta)

    F_C = (
        R * inner(phi * (C - C_n) / dt, z_t) * dx
        + theta_C * _D_ope(C, z_t, D) * dx
        + (1 - theta_C) * _D_ope(C_n, z_t, D_n) * dx
        + theta_C * inner(u, nabla_grad(C)) * z_t * dx
        + (1 - theta_C) * inner(u_n, nabla_grad(C_n)) * z_t * dx
        + theta_C * landa * R * inner(phi * C, z_t) * dx
        + (1 - theta_C) * landa * R * inner(phi * C_n, z_t) * dx
        - theta_C * inner(Src, z_t) * dx
        - (1 - theta_C) * inner(Src_n, z_t) * dx
        + theta_C * inner(delta_C * C, z_t) * dx
        + (1 - theta_C) * inner(delta_C * C_n, z_t) * dx
    )

    a_C = fem.form(lhs(F_C))
    L_C = fem.form(rhs(F_C))
    return a_h, L_h, a_C, L_C


def solve_forward_fenicsx(Qout, pozo, p, mesh_cache=None, save_dir=None):
    """
    Corre un forward completo (t=0 a T=len(Qout)*p.dt) con Q(t)=Qout y
    pozo=[x_p, z_p] dados. Mismo contrato que view.animation.solve_forward
    del backend bfr.

    mesh_cache: (domain, facet_tags, porosidad_array) ya construidos por
    fenicsx.mesh.build_mesh — pásalo para reusar la malla entre llamadas
    (p.ej. dentro de un optimizador) en vez de remallar cada vez. Si es
    None, se genera aquí usando `pozo` como punto embebido.

    Devuelve (h_n, C_n, C_out, C_total): h_n/C_n son los campos finales
    (dolfinx.fem.Function); C_out/C_total son series de tiempo (arrays de
    longitud Nt) con la concentración extraída y la masa total en el
    dominio en cada paso — análogas a lo que guarda maintesis.py en
    int_c/C_s_out.
    """
    phys = _config_phys(p)
    Nt = len(Qout)
    dt = p.dt
    T_total = Nt * dt

    pozo_ext = np.array([pozo[0], pozo[1], 0.0])

    if mesh_cache is None:
        domain, facet_tags, porosidad_raw = build_mesh(pozo, spacing=p.spacing)
    else:
        domain, facet_tags, porosidad_raw = mesh_cache

    V, G, Q, h_inlet, h_outlet, bc_C, ds_in, ds_out = _boundaries_elements(domain, facet_tags, phys)

    phi = fem.Function(G)
    phi.x.array[:] = porosidad_raw

    D, D_m, Dm_n = _diffusion_operator(V, phys["D_d"], phi, phys["alpha"])

    # --- sumidero/fuente puntuales, ver limitación documentada arriba ---
    delta = fem.Function(G)
    delta.interpolate(lambda x: np.where(
        np.sqrt((x[0] - pozo[0]) ** 2 + (x[1] - pozo[1]) ** 2) < phys["tol"], 1.0, 0.0
    ))
    delta_C = fem.Function(Q)

    h = TrialFunction(G)
    C = TrialFunction(Q)
    h_ = fem.Function(G)
    C_ = fem.Function(Q)
    u = fem.Function(V)
    u_n = fem.Function(V)
    h_n = fem.Function(G)
    C_n = fem.Function(Q)
    Src = fem.Function(Q)
    Src_n = fem.Function(Q)

    # Condición inicial = MISMO estado precedente que carga bfr (funcion.h5,
    # generado por las etapas iniciales del FEM de la maestría), interpolado a
    # los DOF de la malla de fenicsx con el mismo get_init_values que usa bfr.
    # Así ambos backends arrancan de idéntico (H, C) y este forward es sólo una
    # etapa final aplicada con los datos de la interfaz — no un spin-up propio.
    # H vive en G (Lagrange-1) y C en Q (Lagrange-2): DOF en coords distintas,
    # se evalúa el interpolador sobre cada espacio por separado.
    h0, _ = get_init_values(G.tabulate_dof_coordinates()[:, :2])
    _, c0 = get_init_values(Q.tabulate_dof_coordinates()[:, :2])
    h_n.x.array[:] = h0
    C_n.x.array[:] = np.nan_to_num(c0, nan=phys["C_0"])
    Src_n.x.array[:] = 0.0

    K_z = fem.Function(G)
    expr_Ki = 8.3e-3 * phys["g"] * phys["d_z"] ** 2 * phi ** 3 / (phys["nu"] * (1 - phi) ** 2)
    K_z.interpolate(fem.Expression(expr_Ki, G.element.interpolation_points))

    expr_un = as_vector([-K_z * phys["alpha"] ** 2 * grad(h_n)[0], -K_z * grad(h_n)[1]])
    u_n.interpolate(fem.Expression(expr_un, V.element.interpolation_points))
    expr_u = as_vector([-K_z * phys["alpha"] ** 2 * grad(h_)[0], -K_z * grad(h_)[1]])

    QOut_const = fem.Constant(domain, PETSc.ScalarType(float(Qout[0])))

    a_h, L_h, a_C, L_C = _variational_form_h_C(
        Q, G, phys["R"], phys["landa"], phi, h, h_n, dt, phys["theta"], phys["theta_C"],
        K_z, phys["alpha"], h_inlet, h_outlet, ds_in, ds_out, QOut_const, Src, Src_n,
        delta, delta_C, phi, C, C_n, D + D_m, D + Dm_n, u, u_n,
    )

    # A1 (flujo) no depende de QOut (el término QOut*delta es del lado
    # derecho, no bilineal en h) — se ensambla y factoriza una sola vez.
    A1 = assemble_matrix(a_h, bcs=[])
    A1.assemble()
    solver1 = _create_solver(A1, domain.comm)

    if domain.comm.rank == 0:
        print(f"[fenicsx] CI desde funcion.h5: "
              f"H=[{h_n.x.array.min():.4g}, {h_n.x.array.max():.4g}], "
              f"C=[{C_n.x.array.min():.3g}, {C_n.x.array.max():.3g}]")

    C_out = np.zeros(Nt)
    C_total = np.zeros(Nt)

    # Flags de la config (mismo criterio que bfr): activate_ext gatea la
    # extracción del pozo (término QOut del flujo y sumidero de transporte);
    # activate_fuente gatea la fuente de contaminante.
    ext_on = bool(getattr(p, "activate_ext", True))
    fuente_on = bool(getattr(p, "activate_fuente", True))

    t = 0.0
    for step in range(Nt):
        t += dt
        QOut_const.value = float(Qout[step]) if ext_on else 0.0

        h_ = _solve_step(L_h, a_h, [], solver1, h_)
        h_n.x.array[:] = h_.x.array[:]

        u.interpolate(fem.Expression(expr_u, V.element.interpolation_points))

        src_amp = _fuente_C(t, T_total) if fuente_on else 0.0
        Src.interpolate(lambda x: _gaussian_2d(x, pozo_ext, src_amp))

        gate = float(chi_eps(np.array(Qout[step]))) if ext_on else 0.0
        delta_C.interpolate(lambda x: gate / (2 * np.pi * phys["epsilon_sink"] ** 2) * np.exp(
            -((x[0] - pozo_ext[0]) ** 2 + (x[1] - pozo_ext[1]) ** 2) / (2 * phys["epsilon_sink"] ** 2)
        ))

        D_m_expr = as_vector([
            (phys["a_l"] * u[0] ** 2 + phys["a_t"] * u[1] ** 2) / sqrt(u[0] ** 2 + u[1] ** 2 + phys["eps"]),
            (phys["a_l"] * u[1] ** 2 + phys["a_t"] * u[0] ** 2) / sqrt(u[0] ** 2 + u[1] ** 2 + phys["eps"]),
        ])
        Dm_n_expr = as_vector([
            (phys["a_l"] * u_n[0] ** 2 + phys["a_t"] * u_n[1] ** 2) / sqrt(u_n[0] ** 2 + u_n[1] ** 2 + phys["eps"]),
            (phys["a_l"] * u_n[1] ** 2 + phys["a_t"] * u_n[0] ** 2) / sqrt(u_n[0] ** 2 + u_n[1] ** 2 + phys["eps"]),
        ])
        D_m.interpolate(fem.Expression(D_m_expr, V.element.interpolation_points))
        Dm_n.interpolate(fem.Expression(Dm_n_expr, V.element.interpolation_points))

        A2 = assemble_matrix(a_C, bcs=bc_C)
        A2.assemble()
        solver2 = _create_solver(A2, domain.comm)
        C_ = _solve_step(L_C, a_C, bc_C, solver2, C_)

        C_n.x.array[:] = np.maximum(C_.x.array[:], 0)
        Src_n.x.array[:] = Src.x.array[:]
        u_n.x.array[:] = u.x.array[:]

        C_total[step] = fem.assemble_scalar(fem.form(inner(1, C_n) * dx))
        C_out[step] = fem.assemble_scalar(fem.form(inner(delta_C, C_n) * dx))

    return h_n, C_n, C_out, C_total
