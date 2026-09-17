"""
Optimización por método adjunto para el backend FEM (DOLFINx) — contraparte de
`view/animation.py` (adjunto + gradiente conjugado) del backend `bfr` (RBF-FD),
pero derivando el adjunto de las **mismas formas débiles FEM** vía diferenciación
automática de UFL (`ufl.derivative`/`ufl.adjoint`), en vez de a mano.

Enfoque (discretize-then-optimize), ver el plan y
`articulos/optimizacion-adjunto/optimizacion+adjunto.tex` §8:

- El forward resuelve, por paso n, dos sistemas lineales: flujo `R^h_n(h)=0`
  (matriz constante) y transporte `R^C_n(C)=0` (matriz que depende de u(h) y del
  gateo χ_ε(Q)). Se guarda el historial de estados NATIVOS (h∈G, C∈Q) por paso.
- El adjunto discreto es la transpuesta de esos operadores, resuelto hacia atrás:
    (∂R^C_n/∂C_n)^T ψ_C^n = ∂J/∂C_n − (∂R^C_{n+1}/∂C_n)^T ψ_C^{n+1}
    (∂R^h_n/∂h_n)^T ψ_h^n = −(∂R^C_n/∂h_n)^T ψ_C^n
                             − (∂R^h_{n+1}/∂h_n)^T ψ_h^{n+1}
                             − (∂R^C_{n+1}/∂h_n)^T ψ_C^{n+1}
  Todos los bloques Jacobianos salen de `ufl.derivative` del residual respecto a
  los estados; el acople h→C (advección u=−K∇h y dispersión D(u)) es EXACTO
  porque el residual se escribe con la velocidad simbólica en h (no interpolada).
- Gradiente: dJ/dθ = ∂J/∂θ − Σ_n [ψ_h^n·∂R^h_n/∂θ + ψ_C^n·∂R^C_n/∂θ], con
  θ ∈ {Q(t), z_p}. Q entra en el RHS de flujo (Q·gauss_flow) y en el gateo
  χ_ε(Q) del sumidero de transporte; z_p entra en los kernels gaussianos del pozo
  (sumideros y observación) y en la fuente de contaminante, todos escritos
  simbólicamente en `zp_const` para que su sensibilidad ∂/∂z_p sea analítica y
  exacta (kernel·(x_z−z_p)/ε²).

Desviación deliberada respecto a `fenicsx/forward.py` (documentada): el sumidero
de FLUJO del pozo usa aquí un **gaussiano** (como el de transporte) en vez del
Wendland C2 normalizado del forward estándar, para que TODA la dependencia en z_p
sea diferenciable en forma cerrada. El checkpoint valida la CONSISTENCIA interna
(gradiente adjunto vs diferencias finitas del MISMO funcional), no la comparación
con `bfr` (que además difiere por la discretización).

Controles: x = [Q_0, …, Q_{Nt−1}, z_p] (misma convención que
`view/animation.py:pack_controls`). Optimizador: `scipy.optimize.minimize`
(L-BFGS-B). Requiere el entorno `fenics_env` (dolfinx 0.11.0 conda-forge,
ver fenicsx/README.md). `app.py` despacha automáticamente a ese entorno.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc, apply_lifting
from scipy.special import ellipe
import ufl
from ufl import (
    TrialFunction, TestFunction, grad, nabla_grad, inner, as_vector,
    sqrt, SpatialCoordinate, exp, pi,
)

# Grado de cuadratura FIJO en todas las formas. Crítico para el adjunto: los
# kernels del pozo son gaussianos NO polinómicos y muy estrechos (ε∼5–30) frente
# a la malla (spacing∼100). Si se deja que DOLFINx estime el grado por forma,
# elige uno distinto para `src_shape` y para su derivada `src_shape·(x_z−z_p)/ε²`
# (con un factor polinómico extra), y entonces el gradiente adjunto no coincide
# con la diferencia finita del MISMO funcional. Fijándolo, J y ∂J/∂θ usan la
# misma cuadratura y son consistentes por construcción.
_QD = 8
dx = ufl.dx(metadata={"quadrature_degree": _QD})

from funtions.utils import chi_eps, dchi_eps, get_init_values
from fenicsx.forward import (
    _config_phys, _boundaries_elements, _K_ope, _D_ope, _fuente_C,
    _mrmt_exp_lam_dt,
)


# ---------------------------------------------------------------------------
# Contexto: malla, espacios, campos que NO dependen de los controles.
# Se construye una vez y se reutiliza en cada evaluación del funcional/gradiente
# (la malla es fija durante la optimización; z_p sólo mueve los kernels del pozo,
# que se evalúan analíticamente, sin remallar — igual criterio que forward.py).
# ---------------------------------------------------------------------------
class _Context:
    pass


def _lu_solver(A, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    return ksp


def build_context(pozo, p, mesh_cache):
    """Arma todo lo independiente de los controles (espacios, K_z, φ, D estático,
    fronteras, condición inicial). `mesh_cache=(domain, facet_tags, porosidad)`."""
    from fenicsx.mesh import build_mesh

    phys = _config_phys(p)
    domain, facet_tags, porosidad_raw = (
        mesh_cache if mesh_cache is not None else build_mesh(pozo, spacing=p.spacing)
    )

    V, G, Q, h_inlet, h_outlet, bc_C, ds_in, ds_out = _boundaries_elements(
        domain, facet_tags, phys
    )

    phi = fem.Function(G)
    phi.x.array[:] = porosidad_raw

    # Conductividad K_z (depende sólo de φ) y dispersión estática D_stat (vector).
    K_z = fem.Function(G)
    K_z.interpolate(fem.Expression(
        8.3e-3 * phys["g"] * phys["d_z"] ** 2 * phi ** 3
        / (phys["nu"] * (1 - phi) ** 2),
        G.element.interpolation_points,
    ))
    ell = float(ellipe(1 - phys["alpha"] ** -2))
    D_stat = 0.5 * phys["D_d"] * phi * ell * as_vector([1.0, phys["alpha"] ** -1])

    # Condición inicial = mismo estado precedente que carga bfr/forward.
    h0, _ = get_init_values(G.tabulate_dof_coordinates()[:, :2])
    _, c0 = get_init_values(Q.tabulate_dof_coordinates()[:, :2])

    ctx = _Context()
    ctx.p, ctx.phys = p, phys
    ctx.domain, ctx.comm = domain, domain.comm
    ctx.V, ctx.G, ctx.Q = V, G, Q
    ctx.phi, ctx.K_z, ctx.D_stat = phi, K_z, D_stat
    ctx.h_inlet, ctx.h_outlet, ctx.bc_C = h_inlet, h_outlet, bc_C
    ctx.ds_in, ctx.ds_out = ds_in, ds_out
    ctx.h0 = h0
    ctx.c0 = np.nan_to_num(c0, nan=phys["C_0"])
    ctx.xp = float(pozo[0])

    # Anchos de los kernels gaussianos del pozo (ver docstring del módulo).
    ctx.ef = float(p.spacing)                      # sumidero de flujo
    ctx.es = float(phys["epsilon_sink"])           # sumidero de transporte
    ctx.ex, ctx.ey = 5.0, 30.0                     # fuente de contaminante
    ctx.ep = float(getattr(p, "epsilon_y", 30.0))  # observación en el pozo (J1)
    return ctx


# ---------------------------------------------------------------------------
# Kernels gaussianos simbólicos del pozo, centrados en (x_p, z_p) con z_p = zp_c
# (un fem.Constant). Se devuelven junto con su derivada analítica ∂/∂z_p, que
# para un gaussiano g es g·(x_z − z_p)/ε².
# ---------------------------------------------------------------------------
def _gauss(x, xp, zp_c, eps):
    return 1.0 / (2 * pi * eps ** 2) * exp(
        -((x[0] - xp) ** 2 + (x[1] - zp_c) ** 2) / (2 * eps ** 2)
    )


def _gauss_aniso(x, xp, zp_c, ex, ey):
    return 1.0 / (2 * pi * ex * ey) * exp(
        -((x[0] - xp) ** 2 / (2 * ex ** 2) + (x[1] - zp_c) ** 2 / (2 * ey ** 2))
    )


def _u_expr(h, K_z, alpha):
    """Velocidad de Darcy simbólica u = −K·(α²∂x h, ∂z h) — en h para que el
    acople h→C del adjunto sea exacto vía ufl.derivative."""
    return as_vector([-K_z * alpha ** 2 * grad(h)[0], -K_z * grad(h)[1]])


def _Dm(u, a_l, a_t, eps):
    n = sqrt(u[0] ** 2 + u[1] ** 2 + eps)
    return as_vector([
        (a_l * u[0] ** 2 + a_t * u[1] ** 2) / n,
        (a_l * u[1] ** 2 + a_t * u[0] ** 2) / n,
    ])


# ---------------------------------------------------------------------------
# Residuales UFL del paso n. Se construyen con Functions/Constants de modo que
# `ufl.derivative(R, estado)` dé los bloques Jacobianos del adjunto, y las formas
# de sensibilidad ∂R/∂θ (respecto a Q y z_p) se escriban explícitas.
# ---------------------------------------------------------------------------
def _flow_residual(ctx, h_cur, h_prev, Qc, zp_c):
    """R^h_n(h_cur; h_prev, Q_n, z_p) = 0."""
    p, phys = ctx.p, ctx.phys
    x = SpatialCoordinate(ctx.domain)
    g = TestFunction(ctx.G)
    theta, K_z, alpha = phys["theta"], ctx.K_z, phys["alpha"]
    gauss_flow = _gauss(x, ctx.xp, zp_c, ctx.ef)
    return (
        inner(ctx.phi * (h_cur - h_prev) / p.dt + Qc * gauss_flow, g) * dx
        + theta * _K_ope(h_cur, g, K_z, alpha) * dx
        + (1 - theta) * _K_ope(h_prev, g, K_z, alpha) * dx
        + inner(ctx.h_inlet, g) * ctx.ds_in
        - inner(ctx.h_outlet, g) * ctx.ds_out
    )


def _transport_residual(ctx, C_cur, C_prev, h_cur, h_prev, chi_c, zp_c,
                        src_amp_c, src_amp_prev_c):
    """R^C_n(C_cur; C_prev, h_cur, h_prev, χ(Q_n), z_p) = 0. La velocidad y la
    dispersión son simbólicas en h (acople h→C exacto)."""
    p, phys = ctx.p, ctx.phys
    x = SpatialCoordinate(ctx.domain)
    z = TestFunction(ctx.Q)
    theta, R, landa = phys["theta_C"], phys["R"], phys["landa"]
    a_l, a_t, eps = phys["a_l"], phys["a_t"], phys["eps"]

    u_new = _u_expr(h_cur, ctx.K_z, phys["alpha"])
    u_old = _u_expr(h_prev, ctx.K_z, phys["alpha"])
    D_new = ctx.D_stat + _Dm(u_new, a_l, a_t, eps)
    D_old = ctx.D_stat + _Dm(u_old, a_l, a_t, eps)

    gauss_C = _gauss(x, ctx.xp, zp_c, ctx.es)
    src_shape = _gauss_aniso(x, ctx.xp, zp_c, ctx.ex, ctx.ey)

    return (
        R * inner(ctx.phi * (C_cur - C_prev) / p.dt, z) * dx
        + theta * _D_ope(C_cur, z, D_new) * dx
        + (1 - theta) * _D_ope(C_prev, z, D_old) * dx
        + theta * inner(u_new, nabla_grad(C_cur)) * z * dx
        + (1 - theta) * inner(u_old, nabla_grad(C_prev)) * z * dx
        + theta * landa * R * inner(ctx.phi * C_cur, z) * dx
        + (1 - theta) * landa * R * inner(ctx.phi * C_prev, z) * dx
        - theta * inner(src_amp_c * src_shape, z) * dx
        - (1 - theta) * inner(src_amp_prev_c * src_shape, z) * dx
        + theta * inner(chi_c * gauss_C * C_cur, z) * dx
        + (1 - theta) * inner(chi_c * gauss_C * C_prev, z) * dx
    )


# ---------------------------------------------------------------------------
# FORWARD: resuelve paso a paso, guarda estados nativos y evalúa J.
# ---------------------------------------------------------------------------
def _solve_forward(ctx, Q, zp):
    """Corre el forward con controles (Q array, zp float). Devuelve dict con el
    historial de arrays de estado (h, C por paso), J y las cantidades que el
    adjunto necesita (χ, amplitudes de fuente por paso)."""
    p, phys = ctx.p, ctx.phys
    G, Qs = ctx.G, ctx.Q
    Nt = len(Q)
    dt = p.dt
    T_total = Nt * dt
    ext_on = bool(getattr(p, "activate_ext", True))
    fuente_on = bool(getattr(p, "activate_fuente", True))
    gamma = float(getattr(p, "gamma", 1.0))
    z0 = float(getattr(p, "z0", 0.0))

    # Constants controlables.
    zp_c = fem.Constant(ctx.domain, PETSc.ScalarType(zp))
    Qc = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))
    chi_c = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))
    src_c = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))
    src_prev_c = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))

    # Estados.
    h_cur, h_prev = fem.Function(G), fem.Function(G)
    C_cur, C_prev = fem.Function(Qs), fem.Function(Qs)
    h_prev.x.array[:] = ctx.h0
    C_prev.x.array[:] = ctx.c0

    # Forma de flujo: bilineal constante (A1). Se factoriza una vez.
    dh = TrialFunction(G)
    Rh_trial = _flow_residual(
        ctx, dh, h_prev, Qc, zp_c
    )
    a_h = fem.form(ufl.lhs(Rh_trial))
    A1 = assemble_matrix(a_h, bcs=[])
    A1.assemble()
    solver_h = _lu_solver(A1, ctx.comm)

    # Kernel de observación δ_p (fijo dado z_p) para J1.
    x = SpatialCoordinate(ctx.domain)
    delta_p = _gauss(x, ctx.xp, zp_c, ctx.ep)

    h_hist, C_hist = [], []
    chi_hist = np.zeros(Nt)
    src_hist = np.zeros(Nt)
    J1 = 0.0
    obs2_dt = 0.0   # Σ_n (∫δ_p C_n)² dt, sin ponderar por γ (para auto-escalar tests)

    # MRMT: inicializar C_im y tasas de decaimiento.
    is_mrmt = p.model.startswith("mrmt")
    C_im_hist = []   # historial de C_im (None para ADR)
    C_im, exp_lam_dt_mrmt = None, None
    if is_mrmt:
        exp_lam_dt_mrmt = _mrmt_exp_lam_dt(p)
        n_dofs_C = C_prev.x.array.shape[0]
        C_im = [np.zeros(n_dofs_C) for _ in range(int(p.Nr))]

    src_prev_amp = 0.0
    t = 0.0
    for n in range(Nt):
        t += dt
        Qc.value = float(Q[n]) if ext_on else 0.0
        chi_hist[n] = float(chi_eps(np.array(Q[n]))) if ext_on else 0.0
        chi_c.value = chi_hist[n]
        src_hist[n] = _fuente_C(t, T_total) if fuente_on else 0.0
        src_c.value = src_hist[n]
        src_prev_c.value = src_prev_amp

        # --- flujo ---
        L_h = fem.form(ufl.rhs(_flow_residual(ctx, dh, h_prev, Qc, zp_c)))
        b = assemble_vector(L_h)
        solver_h.solve(b, h_cur.x.petsc_vec)
        h_cur.x.scatter_forward()
        b.destroy()

        # --- transporte ---
        dC = TrialFunction(Qs)
        Rc_trial = _transport_residual(
            ctx, dC, C_prev, h_cur, h_prev, chi_c, zp_c, src_c, src_prev_c
        )
        a_C = fem.form(ufl.lhs(Rc_trial))
        L_C = fem.form(ufl.rhs(Rc_trial))
        A2 = assemble_matrix(a_C, bcs=ctx.bc_C)
        A2.assemble()
        solver_C = _lu_solver(A2, ctx.comm)
        bC = assemble_vector(L_C)
        apply_lifting(bC, [a_C], [ctx.bc_C])
        bC.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(bC, ctx.bc_C)
        solver_C.solve(bC, C_cur.x.petsc_vec)
        C_cur.x.scatter_forward()
        bC.destroy()
        A2.destroy()

        # (el forward físico recorta C≥0; aquí NO se recorta para no romper la
        # diferenciabilidad del funcional — el checkpoint compara J diferenciable)

        # MRMT: post-procesamiento sobre los DOF de C_cur (sin recortar).
        # Guarda C_im ANTES del update (estado en n) para el adjunto.
        C_im_snap = None
        if is_mrmt:
            C_im_snap = [c.copy() for c in C_im]
            if p.model == "mrmt_semi":
                C_arr = C_cur.x.array[:]
                for r in range(int(p.Nr)):
                    e_r = float(exp_lam_dt_mrmt[r])
                    C_im_new_r = C_arr + (C_im[r] - C_arr) * e_r
                    C_arr = C_arr + float(p.beta[r]) * (C_im[r] - C_im_new_r)
                    C_im[r] = C_im_new_r
                C_cur.x.array[:] = C_arr
            elif p.model == "mrmt_block":
                C_old_n = C_prev.x.array[:]  # C del paso anterior (antes del solve)
                C_new_arr = C_cur.x.array[:]
                coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
                for r in range(int(p.Nr)):
                    C_im[r] = coef[r] * (C_new_arr + C_old_n) + (1.0 - 2.0 * coef[r]) * C_im[r]

        # --- funcional J1 (observación en el pozo) ---
        Cwell = ctx.comm.allreduce(
            fem.assemble_scalar(fem.form(inner(delta_p, C_cur) * dx)), op=MPI.SUM
        )
        J1 += gamma * Cwell ** 2 * dt
        obs2_dt += Cwell ** 2 * dt

        h_hist.append(h_cur.x.array.copy())
        C_hist.append(C_cur.x.array.copy())
        if is_mrmt:
            C_im_hist.append(C_im_snap)

        h_prev.x.array[:] = h_cur.x.array
        C_prev.x.array[:] = C_cur.x.array
        src_prev_amp = src_hist[n]

    A1.destroy()

    # Términos económicos del funcional (idénticos a compute_functional de bfr).
    koppa = float(getattr(p, "koppa", 1.0))
    J2 = koppa * abs(zp - z0) ** 2
    J3 = float(np.sum(np.asarray(Q) ** 2)) * dt * abs(zp - z0)
    J = J1 + J2 + J3

    return dict(h=h_hist, C=C_hist, chi=chi_hist, src=src_hist, J=J, J1=J1,
                obs2_dt=obs2_dt, zp=zp, Q=np.asarray(Q, float),
                C_im=C_im_hist if is_mrmt else None)


# ---------------------------------------------------------------------------
# ADJUNTO + GRADIENTE: recursión backward por AD de UFL.
# ---------------------------------------------------------------------------
def _solve_adjoint_grad(ctx, states):
    p, phys = ctx.p, ctx.phys
    G, Qs = ctx.G, ctx.Q
    Q = states["Q"]
    zp = states["zp"]
    Nt = len(Q)
    dt = p.dt
    theta = phys["theta_C"]
    gamma = float(getattr(p, "gamma", 1.0))
    koppa = float(getattr(p, "koppa", 1.0))
    z0 = float(getattr(p, "z0", 0.0))

    zp_c = fem.Constant(ctx.domain, PETSc.ScalarType(zp))
    Qc = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))
    chi_c = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))
    src_c = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))
    src_prev_c = fem.Constant(ctx.domain, PETSc.ScalarType(0.0))

    h_cur, h_prev = fem.Function(G), fem.Function(G)
    C_cur, C_prev = fem.Function(Qs), fem.Function(Qs)

    dh, gtest = TrialFunction(G), TestFunction(G)
    dC = TrialFunction(Qs)
    x = SpatialCoordinate(ctx.domain)

    # --- Jacobianos de flujo (constantes) ---
    Rh_cur = _flow_residual(ctx, h_cur, h_prev, Qc, zp_c)
    a_hh = fem.form(ufl.adjoint(ufl.derivative(Rh_cur, h_cur, dh)))
    Ahh_T = assemble_matrix(a_hh, bcs=[]); Ahh_T.assemble()
    solver_h = _lu_solver(Ahh_T, ctx.comm)
    Ahh_prev = assemble_matrix(
        fem.form(ufl.derivative(Rh_cur, h_prev, dh)), bcs=[]
    ); Ahh_prev.assemble()

    # kernel de observación y su derivada en z_p (para ∂J1/∂C y ∂J1/∂z_p)
    delta_p = _gauss(x, ctx.xp, zp_c, ctx.ep)
    ddelta_p = delta_p * (x[1] - zp_c) / ctx.ep ** 2
    gauss_flow = _gauss(x, ctx.xp, zp_c, ctx.ef)
    dgauss_flow = gauss_flow * (x[1] - zp_c) / ctx.ef ** 2
    gauss_C = _gauss(x, ctx.xp, zp_c, ctx.es)
    dgauss_C = gauss_C * (x[1] - zp_c) / ctx.es ** 2
    src_shape = _gauss_aniso(x, ctx.xp, zp_c, ctx.ex, ctx.ey)
    dsrc_shape = src_shape * (x[1] - zp_c) / ctx.ey ** 2

    grad_Q = np.zeros(Nt)
    grad_zp = 0.0

    def arr(fn_space, ufl_form):
        v = assemble_vector(fem.form(ufl_form))
        v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return v

    # carry-overs desde el paso n+1
    termC_next = None      # (∂R^C_{n+1}/∂C_n)^T ψ_C^{n+1}
    termH_next = None      # (∂R^h_{n+1}/∂h_n)^T ψ_h^{n+1} + (∂R^C_{n+1}/∂h_n)^T ψ_C^{n+1}

    psiC = fem.Function(Qs)
    psiH = fem.Function(G)

    # MRMT: carry-over de ψ_C_im desde el paso n+1 (backward).
    is_mrmt = p.model.startswith("mrmt")
    C_im_hist = states.get("C_im") or []
    psiC_im_N = None
    if is_mrmt:
        n_dofs_C = psiC.x.array.shape[0]
        psiC_im_N = [np.zeros(n_dofs_C) for _ in range(int(p.Nr))]
        exp_lam_dt_mrmt = _mrmt_exp_lam_dt(p)

    src_prev = np.concatenate([[0.0], states["src"][:-1]])  # amplitud previa por paso

    for n in range(Nt - 1, -1, -1):
        # cargar estados y controles del paso n
        h_cur.x.array[:] = states["h"][n]
        h_prev.x.array[:] = states["h"][n - 1] if n > 0 else ctx.h0
        C_cur.x.array[:] = states["C"][n]
        C_prev.x.array[:] = states["C"][n - 1] if n > 0 else ctx.c0
        Qc.value = float(Q[n]) * bool(getattr(p, "activate_ext", True))
        chi_c.value = float(states["chi"][n])
        src_c.value = float(states["src"][n])
        src_prev_c.value = float(src_prev[n])

        Rc = _transport_residual(
            ctx, C_cur, C_prev, h_cur, h_prev, chi_c, zp_c, src_c, src_prev_c
        )
        Rh = _flow_residual(ctx, h_cur, h_prev, Qc, zp_c)

        # ---- ψ_C^n:  (∂R^C_n/∂C_n)^T ψ_C = ∂J/∂C_n − termC_next ----
        a_cc_T = fem.form(ufl.adjoint(ufl.derivative(Rc, C_cur, dC)))
        Acc_T = assemble_matrix(a_cc_T, bcs=ctx.bc_C); Acc_T.assemble()
        solver_C = _lu_solver(Acc_T, ctx.comm)

        Cwell = ctx.comm.allreduce(
            fem.assemble_scalar(fem.form(inner(delta_p, C_cur) * dx)), op=MPI.SUM
        )
        rhsC = arr(Qs, 2.0 * gamma * Cwell * dt * inner(delta_p, TestFunction(Qs)) * dx)
        if termC_next is not None:
            rhsC.axpy(-1.0, termC_next)
            termC_next.destroy()

        # MRMT Block: agregar Σ_r coef_r·ψ_C_im_r^{n+1} al RHS del adjunto
        # (adjunto de la fuente 4R·α_r·φ·C_im en la forma de transporte).
        if p.model == "mrmt_block" and psiC_im_N is not None:
            coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
            for r in range(int(p.Nr)):
                rhsC.array[:] += coef[r] * psiC_im_N[r]

        set_bc(rhsC, ctx.bc_C)   # ψ_C = 0 en el inlet (BC homogénea del adjunto)
        solver_C.solve(rhsC, psiC.x.petsc_vec)
        psiC.x.scatter_forward()
        rhsC.destroy()

        # MRMT Block: actualizar ψ_C_im^{n-1} tras el solve de transporte adjunto.
        # Adjunto de: C_im^n[r] = coef_r*(C^n + C^{n-1}) + (1-2·coef_r)·C_im^{n-1}[r]
        # → ψ_C_im^{n-1}[r] = (1-2·coef_r)·ψ_C_im^n[r]
        # El término coef_r·ψ_C_im^n se agrega a rhsC (ya hecho arriba) y a termC_next.
        if p.model == "mrmt_block" and psiC_im_N is not None:
            coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
            psiC_im_N_prev = psiC_im_N    # guardar para termC_next
            psiC_im_n = [np.zeros_like(psiC_im_N[r]) for r in range(int(p.Nr))]
            for r in range(int(p.Nr)):
                psiC_im_n[r] = (1.0 - 2.0 * coef[r]) * psiC_im_N[r]
            psiC_im_N = psiC_im_n
        else:
            psiC_im_N_prev = None

        # ---- ψ_h^n:  (∂R^h_n/∂h_n)^T ψ_h = −(∂R^C_n/∂h_n)^T ψ_C − termH_next ----
        Ach = assemble_matrix(
            fem.form(ufl.derivative(Rc, h_cur, dh)), bcs=[]
        ); Ach.assemble()
        rhsH = psiH.x.petsc_vec.duplicate()
        Ach.multTranspose(psiC.x.petsc_vec, rhsH)
        rhsH.scale(-1.0)
        if termH_next is not None:
            rhsH.axpy(-1.0, termH_next)
            termH_next.destroy()
        solver_h.solve(rhsH, psiH.x.petsc_vec)
        psiH.x.scatter_forward()
        rhsH.destroy()
        Ach.destroy(); Acc_T.destroy()

        # ---- gradiente: dJ/dθ = ∂J/∂θ − ψ_h·∂R^h/∂θ − ψ_C·∂R^C/∂θ ----
        # Q_n:  J3 + flujo (Q·gauss_flow) + transporte (χ'(Q)·gauss_C·C)
        gQ = 2.0 * abs(zp - z0) * float(Q[n]) * dt
        dRh_dQ = arr(G, inner(gauss_flow, gtest) * dx)
        gQ -= psiH.x.petsc_vec.dot(dRh_dQ)
        dRh_dQ.destroy()
        dchi = float(dchi_eps(np.array(Q[n])))
        dRc_dchi = arr(Qs, (
            theta * inner(gauss_C * C_cur, TestFunction(Qs))
            + (1 - theta) * inner(gauss_C * C_prev, TestFunction(Qs))
        ) * dx)
        gQ -= psiC.x.petsc_vec.dot(dRc_dchi) * dchi
        dRc_dchi.destroy()
        grad_Q[n] = gQ

        # z_p (acumula sobre pasos): J1 + acople adjunto (flujo y transporte)
        grad_zp += gamma * 2.0 * Cwell * dt * ctx.comm.allreduce(
            fem.assemble_scalar(fem.form(inner(ddelta_p, C_cur) * dx)), op=MPI.SUM
        )
        dRh_dzp = arr(G, Qc * inner(dgauss_flow, gtest) * dx)
        grad_zp -= psiH.x.petsc_vec.dot(dRh_dzp)
        dRh_dzp.destroy()
        ztest = TestFunction(Qs)
        dRc_dzp = arr(Qs, (
            theta * chi_c * inner(dgauss_C * C_cur, ztest)
            + (1 - theta) * chi_c * inner(dgauss_C * C_prev, ztest)
            - theta * inner(src_c * dsrc_shape, ztest)
            - (1 - theta) * inner(src_prev_c * dsrc_shape, ztest)
        ) * dx)
        grad_zp -= psiC.x.petsc_vec.dot(dRc_dzp)
        dRc_dzp.destroy()

        # ---- couplings hacia el paso n−1 ----
        # MRMT Semi: aplicar backward del operador splitting al ψ_C^n recién
        # resuelto antes de calcular termC_next, de modo que el carry al paso
        # n−1 sea (∂R^C_n/∂C_{n-1})^T @ ψ_C_transport^n (adjunto a nivel de
        # transporte, antes del MRMT) — análogo al BFR BT @ psiC_N_transport.
        if p.model == "mrmt_semi" and psiC_im_N is not None:
            Nr_m = int(p.Nr)
            beta_m = np.asarray(p.beta, float)
            psiC_im_n_new = [np.zeros_like(psiC_im_N[r]) for r in range(Nr_m)]
            psiC_arr = psiC.x.array[:].copy()
            for r in range(Nr_m - 1, -1, -1):
                e_r = float(exp_lam_dt_mrmt[r])
                alpha_r = 1.0 - e_r
                psiC_arr_new = (1.0 - beta_m[r] * alpha_r) * psiC_arr + alpha_r * psiC_im_N[r]
                psiC_im_n_new[r] = beta_m[r] * alpha_r * psiC_arr + e_r * psiC_im_N[r]
                psiC_arr = psiC_arr_new
            psiC.x.array[:] = psiC_arr   # ψ_C_transport → usado en Acc_prev^T
            psiC_im_N = psiC_im_n_new

        Acc_prev = assemble_matrix(
            fem.form(ufl.derivative(Rc, C_prev, dC)), bcs=[]
        ); Acc_prev.assemble()
        termC_next = psiC.x.petsc_vec.duplicate()
        Acc_prev.multTranspose(psiC.x.petsc_vec, termC_next)
        Acc_prev.destroy()

        # MRMT Block: agregar coef_r·ψ_C_im^n al carry-over hacia C^{n-1}.
        # (adjunto de la dependencia C_im^n[r] = coef_r*(C^n + C^{n-1}) + ...)
        if psiC_im_N_prev is not None:
            coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
            for r in range(int(p.Nr)):
                termC_next.array[:] += coef[r] * psiC_im_N_prev[r]

        Ach_prev = assemble_matrix(
            fem.form(ufl.derivative(Rc, h_prev, dh)), bcs=[]
        ); Ach_prev.assemble()
        termH_next = psiH.x.petsc_vec.duplicate()
        Ahh_prev.multTranspose(psiH.x.petsc_vec, termH_next)
        tmp = psiH.x.petsc_vec.duplicate()
        Ach_prev.multTranspose(psiC.x.petsc_vec, tmp)
        termH_next.axpy(1.0, tmp)
        tmp.destroy(); Ach_prev.destroy()

    if termC_next is not None:
        termC_next.destroy()
    if termH_next is not None:
        termH_next.destroy()
    Ahh_T.destroy(); Ahh_prev.destroy()

    # Términos económicos independientes del adjunto.
    sgn = np.sign(zp - z0) if zp != z0 else 0.0
    grad_zp += 2.0 * koppa * (zp - z0)
    grad_zp += float(np.sum(np.asarray(Q) ** 2)) * dt * sgn

    return np.concatenate([grad_Q, [grad_zp]])


# ---------------------------------------------------------------------------
# Envoltorios estilo view/animation.py (functional / gradient / optimizer).
# ---------------------------------------------------------------------------
def _unpack(x):
    return np.asarray(x[:-1], float), float(x[-1])


def functional_fenicsx(x, ctx):
    Q, zp = _unpack(x)
    return _solve_forward(ctx, Q, zp)["J"]


def gradient_fenicsx(x, ctx):
    Q, zp = _unpack(x)
    states = _solve_forward(ctx, Q, zp)
    return _solve_adjoint_grad(ctx, states)


def optimize_fenicsx(p, mesh_cache=None, max_iter=None):
    """Corre la optimización adjunta con scipy trust-constr sobre x=[Q(t), z_p].
    Incluye restricción lineal de suministro 0 ≤ S_n ≤ S_max (balance del tanque
    con demanda diaria D), idéntica a optimize_bfr. Guarda historial y resultado."""
    from scipy.optimize import minimize, Bounds, LinearConstraint
    from funtions.water_supply import (supply_constraint, storage_trajectory,
                                       pump_upper_bounds, pump_duty_constraint)

    pozo = list(p.pozo)
    ctx = build_context(pozo, p, mesh_cache)

    Q0 = np.asarray(p.Qout[0], float)
    x0 = np.concatenate([Q0, [float(pozo[1])]])
    n = len(x0)

    history = {"Q": [], "zp": [], "J": []}

    # --- Normalización del funcional ---
    if ctx.comm.rank == 0:
        print("[fenicsx-opt] Evaluando J₀ para normalización...", flush=True)
    J0_raw  = functional_fenicsx(x0, ctx)
    J_scale = max(abs(J0_raw), 1.0)
    if ctx.comm.rank == 0:
        print(f"[fenicsx-opt] J₀={J0_raw:.4e}  J_scale={J_scale:.4e}", flush=True)

    def fun(x):
        J = functional_fenicsx(x, ctx)
        history["Q"].append(x[:-1].copy())
        history["zp"].append(float(x[-1]))
        history["J"].append(float(J))
        if ctx.comm.rank == 0:
            print(f"[fenicsx-opt] J={J:.6e} (×{J_scale:.1e})  z_p={x[-1]:.4g}")
        return J / J_scale

    def jac(x):
        return gradient_fenicsx(x, ctx) / J_scale

    # Cotas físicas: Q(t) ≥ 0 hasta Q_max; z_p dentro del rango real de z de la
    # malla (con margen de un spacing). Igual criterio que L-BFGS-B anterior pero
    # ahora como Bounds de trust-constr (compatible con LinearConstraint).
    z_nodes = ctx.domain.geometry.x[:, 1]
    Q_max = float(getattr(p, "Q_max", 1e-2))
    zp_lo = float(getattr(p, "zp_min", float(z_nodes.min()) + float(p.spacing)))
    zp_hi = float(getattr(p, "zp_max", float(z_nodes.max()) - float(p.spacing)))
    Q_ub   = pump_upper_bounds(p, len(Q0), Q_max)
    bounds = Bounds(np.concatenate([np.zeros(len(Q0)), [zp_lo]]),
                    np.concatenate([Q_ub,               [zp_hi]]))

    # Restricción lineal de suministro: 0 ≤ S_n ≤ S_max (water_supply.py).
    # Fuerza a extraer lo suficiente para cubrir la demanda diaria D sin rebosar.
    A, clb, cub = supply_constraint(p, n)
    cons_list = [LinearConstraint(A, clb, cub)]
    duty = pump_duty_constraint(p, n, Q_max)
    if duty is not None:
        Ad, lbd, ubd = duty
        cons_list.append(LinearConstraint(Ad, lbd, ubd))

    opts = {"maxiter": int(max_iter) if max_iter else int(getattr(p, "opt_maxiter", 20)),
            "gtol": 1e-5, "xtol": 1e-8}
    res = minimize(fun, x0, jac=jac, method="trust-constr",
                   bounds=bounds, constraints=cons_list, options=opts)

    S_opt  = storage_trajectory(res.x[:-1], p)
    J_phys = res.fun * J_scale
    if ctx.comm.rank == 0:
        print(f"[fenicsx-opt] terminado: J={J_phys:.6e} (normalizado={res.fun:.4e}), "
              f"success={res.success}, nit={res.nit}, msg={res.message}")
        print(f"[fenicsx-opt] S(t)=[{S_opt.min():.3g}, {S_opt.max():.3g}]  "
              f"(S_max={getattr(p,'S_max',500.0)})")
        if bool(getattr(p, "save_dat", False)):
            import os
            save_dir = p.save_data
            os.makedirs(save_dir, exist_ok=True)
            np.savez(
                f"{save_dir}/optimization_results.npz",
                Q=np.asarray(history["Q"]),
                zp=np.asarray(history["zp"]),
                J=np.asarray(history["J"]),
                x_opt=res.x, J_opt=J_phys, J_opt_scaled=res.fun, S_opt=S_opt,
            )
            print(f"[fenicsx-opt] historial guardado ✔  "
                  f"{save_dir}/optimization_results.npz")
    return res
