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
import os
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
    _r = getattr(p, "r_pozo", None)
    ctx.ef = float(p.epsilon_pozo if _r else p.spacing)   # sumidero de flujo
    ctx.es = float(phys["epsilon_sink"])           # sumidero de transporte
    ctx.ex = float(phys.get("epsilon_x", 0.5 * p.spacing))   # fuente de contaminante
    ctx.ey = float(phys.get("epsilon_y", p.spacing))
    ctx.ep = float(p.epsilon_pozo if _r else phys.get("epsilon_p", p.spacing))  # observación en el pozo (J1)
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
def _const_theta(ctx, theta):
    """θ como Constant: con θ = 1, UFL reduce (1−θ)·expr a Zero y la integral
    pierde su dominio ("missing an integration domain")."""
    return fem.Constant(ctx.domain, PETSc.ScalarType(theta))


def _flow_residual(ctx, h_cur, h_prev, Qc, zp_c):
    """R^h_n(h_cur; h_prev, Q_n, z_p) = 0."""
    p, phys = ctx.p, ctx.phys
    x = SpatialCoordinate(ctx.domain)
    g = TestFunction(ctx.G)
    theta, K_z, alpha = _const_theta(ctx, phys["theta"]), ctx.K_z, phys["alpha"]
    gauss_flow = _gauss(x, ctx.xp, zp_c, ctx.ef)
    return (
        inner(ctx.phi * (h_cur - h_prev) / p.dt + Qc * gauss_flow, g) * dx
        + theta * _K_ope(h_cur, g, K_z, alpha) * dx
        + (1 - theta) * _K_ope(h_prev, g, K_z, alpha) * dx
        + inner(ctx.h_inlet, g) * ctx.ds_in
        - inner(ctx.h_outlet, g) * ctx.ds_out
    )


def _transport_residual(ctx, C_cur, C_prev, h_cur, h_prev, chi_c, zp_c,
                        src_amp_c, src_amp_prev_c, C_im_sum=None, alpha_sum=0.0):
    """R^C_n(C_cur; C_prev, h_cur, h_prev, χ(Q_n), z_p) = 0. La velocidad y la
    dispersión son simbólicas en h (acople h→C exacto).
    Para mrmt_block: C_im_sum (fem.Function en Q) = Σ_r α_r·C_im_r y alpha_sum = Σ_r α_r."""
    p, phys = ctx.p, ctx.phys
    x = SpatialCoordinate(ctx.domain)
    z = TestFunction(ctx.Q)
    theta, R, landa = _const_theta(ctx, phys["theta_C"]), phys["R"], phys["landa"]
    a_l, a_t, eps = phys["a_l"], phys["a_t"], phys["eps"]

    u_new = _u_expr(h_cur, ctx.K_z, phys["alpha"])
    u_old = _u_expr(h_prev, ctx.K_z, phys["alpha"])
    D_new = ctx.D_stat + _Dm(u_new, a_l, a_t, eps)
    D_old = ctx.D_stat + _Dm(u_old, a_l, a_t, eps)

    # Sumidero W- escalado por factor_sumidero_C (0 por defecto: anulado). Como
    # Constant: un 0.0 de Python reduce el término a Zero y la integral pierde
    # su dominio.
    f_W = fem.Constant(ctx.domain, PETSc.ScalarType(float(getattr(p, "factor_sumidero_C", 0.0))))
    gauss_C = f_W * _gauss(x, ctx.xp, zp_c, ctx.es)
    src_shape = _gauss_aniso(x, ctx.xp, zp_c, ctx.ex, ctx.ey)

    F_C = (
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
    if C_im_sum is not None and alpha_sum:
        F_C += (
            2.0 * R * alpha_sum * inner(ctx.phi * (theta * C_cur + (1 - theta) * C_prev), z) * dx
            - 2.0 * R * inner(ctx.phi * C_im_sum, z) * dx
        )
    return F_C


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
    C_im_hist = []         # historial de C_im (None para ADR)
    C_pre_mrmt_hist = []   # C_t^n pre-MRMT (solo mrmt_semi, para grad en adjunto)
    C_im, exp_lam_dt_mrmt = None, None
    C_im_sum_func = None   # fem.Function para acoplamiento MRMT_Block
    _mrmt_alpha_r = None
    _mrmt_alpha_sum = 0.0
    if is_mrmt:
        exp_lam_dt_mrmt = _mrmt_exp_lam_dt(p)
        n_dofs_C = C_prev.x.array.shape[0]
        C_im = [np.zeros(n_dofs_C) for _ in range(int(p.Nr))]
        if p.model == "mrmt_block":
            C_im_sum_func = fem.Function(Qs)
            C_im_sum_func.x.array[:] = 0.0
            _mrmt_alpha_r = np.asarray(p.alpha_r, float)
            _mrmt_alpha_sum = float(p.alpha_sum)

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
            ctx, dC, C_prev, h_cur, h_prev, chi_c, zp_c, src_c, src_prev_c,
            C_im_sum=C_im_sum_func, alpha_sum=_mrmt_alpha_sum,
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

        # MRMT: post-procesamiento sobre los DOF de C_cur (antes del recorte).
        # Guarda C_im ANTES del update (estado en n) para el adjunto.
        C_im_snap = None
        C_pre_mrmt_snap = None   # C_t^n (pre-MRMT) para grad en adjunto
        if is_mrmt:
            C_im_snap = [c.copy() for c in C_im]
            if p.model == "mrmt_semi":
                C_pre_mrmt_snap = C_cur.x.array.copy()   # save C_t^n before MRMT
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
                th = float(phys["theta_C"])
                for r in range(int(p.Nr)):
                    C_im[r] = (2.0 * coef[r] * (th * C_new_arr + (1.0 - th) * C_old_n)
                               + (1.0 - 2.0 * coef[r]) * C_im[r])
                # Actualizar C_im_sum para el próximo paso
                acc = np.zeros(C_im_sum_func.x.array.shape[0])
                for r in range(int(p.Nr)):
                    acc += _mrmt_alpha_r[r] * C_im[r]
                C_im_sum_func.x.array[:] = acc

        # Positividad física: C ≥ 0 en cada paso, igual que el forward físico.
        # Nota: el recorte no es diferenciable donde C<0; el adjunto lo ignora.
        C_cur.x.array[:] = np.maximum(C_cur.x.array, 0.0)

        # --- funcional J1 = γ ∫δₚ C² dΩ  (consistente con BFR: J₁=γ∫gₚC²dΩ) ---
        C2well = ctx.comm.allreduce(
            fem.assemble_scalar(fem.form(delta_p * C_cur * C_cur * dx)), op=MPI.SUM
        )
        J1 += gamma * C2well * dt
        obs2_dt += C2well * dt

        h_hist.append(h_cur.x.array.copy())
        C_hist.append(C_cur.x.array.copy())
        if is_mrmt:
            C_im_hist.append(C_im_snap)
            if C_pre_mrmt_snap is not None:
                C_pre_mrmt_hist.append(C_pre_mrmt_snap)

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
                C_im=C_im_hist if is_mrmt else None,
                C_pre_mrmt=C_pre_mrmt_hist if C_pre_mrmt_hist else None)


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
    theta = _const_theta(ctx, phys["theta_C"])
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
    # Sumidero W- escalado por factor_sumidero_C: entra en dRc/dχ y dRc/dz_p
    gauss_C = fem.Constant(ctx.domain, PETSc.ScalarType(float(getattr(ctx.p, "factor_sumidero_C", 0.0)))) \
        * _gauss(x, ctx.xp, zp_c, ctx.es)
    dgauss_C = gauss_C * (x[1] - zp_c) / ctx.es ** 2
    src_shape = _gauss_aniso(x, ctx.xp, zp_c, ctx.ex, ctx.ey)
    dsrc_shape = src_shape * (x[1] - zp_c) / ctx.ey ** 2

    grad_Q = np.zeros(Nt)
    grad_zp = 0.0

    psiH_hist = []   # acumula en orden backward; se invierte al final
    psiC_hist = []

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
    C_im_sum_func_bwd = None   # fem.Function para acoplamiento MRMT_Block backward
    _bwd_alpha_r = None
    _bwd_alpha_sum = 0.0
    if is_mrmt:
        n_dofs_C = psiC.x.array.shape[0]
        psiC_im_N = [np.zeros(n_dofs_C) for _ in range(int(p.Nr))]
        exp_lam_dt_mrmt = _mrmt_exp_lam_dt(p)
        if p.model == "mrmt_block":
            C_im_sum_func_bwd = fem.Function(Qs)
            C_im_sum_func_bwd.x.array[:] = 0.0
            _bwd_alpha_r = np.asarray(p.alpha_r, float)
            _bwd_alpha_sum = float(p.alpha_sum)

    src_prev = np.concatenate([[0.0], states["src"][:-1]])  # amplitud previa por paso

    C_pre_mrmt_list = states.get("C_pre_mrmt") or []  # C_t^n (pre-MRMT), solo mrmt_semi

    for n in range(Nt - 1, -1, -1):
        # cargar estados y controles del paso n
        h_cur.x.array[:] = states["h"][n]
        h_prev.x.array[:] = states["h"][n - 1] if n > 0 else ctx.h0
        C_cur.x.array[:] = states["C"][n]
        C_prev.x.array[:] = states["C"][n - 1] if n > 0 else ctx.c0
        # Para mrmt_semi: cargar C_t^n (pre-MRMT) para usar en dRc_dchi / dRc_dzp
        if C_pre_mrmt_list:
            C_cur_transport = C_pre_mrmt_list[n]   # array numpy pre-MRMT
        else:
            C_cur_transport = states["C"][n]        # ADR / mrmt_block: igual que C^n
        Qc.value = float(Q[n]) * bool(getattr(p, "activate_ext", True))
        chi_c.value = float(states["chi"][n])
        src_c.value = float(states["src"][n])
        src_prev_c.value = float(src_prev[n])

        # MRMT Block: reconstruir C_im_sum del paso n desde el historial
        if C_im_sum_func_bwd is not None and n < len(C_im_hist) and C_im_hist[n] is not None:
            acc = np.zeros(n_dofs_C)
            for r in range(int(p.Nr)):
                acc += float(_bwd_alpha_r[r]) * C_im_hist[n][r]
            C_im_sum_func_bwd.x.array[:] = acc

        Rc = _transport_residual(
            ctx, C_cur, C_prev, h_cur, h_prev, chi_c, zp_c, src_c, src_prev_c,
            C_im_sum=C_im_sum_func_bwd, alpha_sum=_bwd_alpha_sum,
        )
        Rh = _flow_residual(ctx, h_cur, h_prev, Qc, zp_c)

        # ---- ψ_C^n:  (∂R^C_n/∂C_n)^T ψ_C = ∂J/∂C_n − termC_next ----
        a_cc_T = fem.form(ufl.adjoint(ufl.derivative(Rc, C_cur, dC)))
        Acc_T = assemble_matrix(a_cc_T, bcs=ctx.bc_C); Acc_T.assemble()
        solver_C = _lu_solver(Acc_T, ctx.comm)

        # ∂J₁/∂C · v = 2γ ∫ δₚ C v dΩ  (adjunto exacto de J₁ = γ∫δₚC²dΩ)
        rhsC = arr(Qs, 2.0 * gamma * dt * inner(delta_p * C_cur, TestFunction(Qs)) * dx)
        if termC_next is not None:
            rhsC.axpy(-1.0, termC_next)
            termC_next.destroy()

        # MRMT Block: agregar Σ_r coef_r·ψ_C_im_r^{n+1} al RHS del adjunto
        # (adjunto de la fuente 4R·α_r·φ·C_im en la forma de transporte).
        if p.model == "mrmt_block" and psiC_im_N is not None:
            # ∂C_im^n/∂C^n = 2·θ_C·coef (= coef con θ_C = 0.5)
            coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
            coef = 2.0 * float(phys["theta_C"]) * coef
            for r in range(int(p.Nr)):
                rhsC.array[:] += coef[r] * psiC_im_N[r]

        # MRMT Semi: aplicar backward del operador splitting AL RHS (antes del solve).
        # Orden correcto: ψ_mrmt^n (= ∂J/∂C^n + termC_next) → MRMT adjoint → ψ_t^n
        # El solve da directamente ψ_t^n = A_CC^{-T} ψ_t^n.
        # Contraste con el orden incorrecto (aplicar DESPUÉS del solve) que produce
        # A_CC^{-T}(ψ_mrmt^n) y luego MRMT·A_CC^{-T}(ψ_mrmt^n) ≠ A_CC^{-T}(MRMT·ψ_mrmt^n).
        # Referencia: BFR step_adjoint (funtions/step_time.py líneas 157–169).
        if p.model == "mrmt_semi" and psiC_im_N is not None:
            Nr_m = int(p.Nr)
            beta_m = np.asarray(p.beta, float)
            psiC_im_n_new = [np.zeros_like(psiC_im_N[r]) for r in range(Nr_m)]
            psiC_arr = rhsC.array[:].copy()
            for r in range(Nr_m - 1, -1, -1):
                e_r = float(exp_lam_dt_mrmt[r])
                alpha_r = 1.0 - e_r
                psiC_arr_new = (1.0 - beta_m[r] * alpha_r) * psiC_arr + alpha_r * psiC_im_N[r]
                psiC_im_n_new[r] = beta_m[r] * alpha_r * psiC_arr + e_r * psiC_im_N[r]
                psiC_arr = psiC_arr_new
            rhsC.array[:] = psiC_arr
            psiC_im_N = psiC_im_n_new

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
        # Ach = ∂R^C_n/∂h_cur contiene ∇C_cur (acople advectivo); debe evaluarse en
        # C_t^n (solución de R^C=0), no en C^n_post (que incluye el update de MRMT).
        # Para ADR/mrmt_block: C_t^n = C^n_post → sin efecto. Para mrmt_semi: distinto.
        _C_cur_saved_h = C_cur.x.array.copy()
        C_cur.x.array[:] = C_cur_transport
        Ach = assemble_matrix(
            fem.form(ufl.derivative(Rc, h_cur, dh)), bcs=[]
        ); Ach.assemble()
        C_cur.x.array[:] = _C_cur_saved_h
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

        # capturar estado adjunto del paso n (antes de los carry-overs)
        psiH_hist.append(psiH.x.array.copy())
        psiC_hist.append(psiC.x.array.copy())

        # ---- gradiente: dJ/dθ = ∂J/∂θ − ψ_h·∂R^h/∂θ − ψ_C·∂R^C/∂θ ----
        # Q_n:  J3 + flujo (Q·gauss_flow) + transporte (χ'(Q)·gauss_C·C)
        gQ = 2.0 * abs(zp - z0) * float(Q[n]) * dt
        dRh_dQ = arr(G, inner(gauss_flow, gtest) * dx)
        gQ -= psiH.x.petsc_vec.dot(dRh_dQ)
        dRh_dQ.destroy()
        dchi = float(dchi_eps(np.array(Q[n])))

        # ∂J₁/∂zp = γ ∫(∂δₚ/∂zp) C² dΩ  (J₁ = γ∫δₚC²dΩ, C fijo respecto a zp aquí)
        grad_zp += gamma * dt * ctx.comm.allreduce(
            fem.assemble_scalar(fem.form(ddelta_p * C_cur * C_cur * dx)), op=MPI.SUM
        )
        dRh_dzp = arr(G, Qc * inner(dgauss_flow, gtest) * dx)
        grad_zp -= psiH.x.petsc_vec.dot(dRh_dzp)
        dRh_dzp.destroy()

        # dRc/dchi y dRc/dzp usan C_t^n (pre-MRMT para mrmt_semi; igual a C^n para ADR/block).
        _C_cur_saved = C_cur.x.array.copy()
        C_cur.x.array[:] = C_cur_transport

        dRc_dchi = arr(Qs, (
            theta * inner(gauss_C * C_cur, TestFunction(Qs))
            + (1 - theta) * inner(gauss_C * C_prev, TestFunction(Qs))
        ) * dx)
        gQ -= psiC.x.petsc_vec.dot(dRc_dchi) * dchi
        dRc_dchi.destroy()
        grad_Q[n] = gQ

        ztest = TestFunction(Qs)
        dRc_dzp = arr(Qs, (
            theta * chi_c * inner(dgauss_C * C_cur, ztest)
            + (1 - theta) * chi_c * inner(dgauss_C * C_prev, ztest)
            - theta * inner(src_c * dsrc_shape, ztest)
            - (1 - theta) * inner(src_prev_c * dsrc_shape, ztest)
        ) * dx)
        grad_zp -= psiC.x.petsc_vec.dot(dRc_dzp)
        dRc_dzp.destroy()

        C_cur.x.array[:] = _C_cur_saved   # restaurar para Acc_prev / Acc_h a continuación

        # ---- couplings hacia el paso n−1 ----
        Acc_prev = assemble_matrix(
            fem.form(ufl.derivative(Rc, C_prev, dC)), bcs=[]
        ); Acc_prev.assemble()
        termC_next = psiC.x.petsc_vec.duplicate()
        Acc_prev.multTranspose(psiC.x.petsc_vec, termC_next)
        Acc_prev.destroy()

        # MRMT Block: agregar coef_r·ψ_C_im^n al carry-over hacia C^{n-1}.
        # (adjunto de la dependencia C_im^n[r] = coef_r*(C^n + C^{n-1}) + ...)
        if psiC_im_N_prev is not None:
            # ∂C_im^n/∂C^{n-1} = 2·(1−θ_C)·coef (= coef con θ_C = 0.5; 0 con θ_C = 1)
            coef = p.dt * np.asarray(p.alpha_r, float) / np.asarray(p.beta, float)
            coef = 2.0 * (1.0 - float(phys["theta_C"])) * coef
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

    # Invertir al orden forward (psiH_hist[0] = paso 0, ..., [Nt-1] = paso Nt-1)
    adj_states = {
        "psiH": np.asarray(psiH_hist[::-1]),   # (Nt, N_G)
        "psiC": np.asarray(psiC_hist[::-1]),   # (Nt, N_Q)
    }
    return np.concatenate([grad_Q, [grad_zp]]), adj_states


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
    grad, _ = _solve_adjoint_grad(ctx, states)
    return grad


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

    history = {"Q": [], "zp": [], "J": [], "grad_norm": []}

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

    # p.save_data ya vive bajo .../optimization (run_type="optimization" es la
    # única forma de llegar a optimize_fenicsx, ver main.py) — no anidar de nuevo.
    save_on = bool(getattr(p, "save_dat", False))
    animate_on = bool(getattr(p, "animate", False))
    opt_dir = p.save_data
    if save_on and ctx.comm.rank == 0:
        os.makedirs(opt_dir, exist_ok=True)

    n_jac = [0]

    def jac(x):
        Q_c, zp_c = _unpack(x)
        if ctx.comm.rank == 0:
            print(f"[fenicsx-opt grad #{n_jac[0]+1}] adjunto en z_p={zp_c:.1f}m ...",
                  flush=True)
        states = _solve_forward(ctx, Q_c, zp_c)
        grad, adj_states = _solve_adjoint_grad(ctx, states)
        n_jac[0] += 1
        if save_on and ctx.comm.rank == 0:
            iter_dir = os.path.join(opt_dir, f"iter_{n_jac[0]:03d}")
            os.makedirs(iter_dir, exist_ok=True)
            nodes_G = ctx.G.tabulate_dof_coordinates()[:, :2]
            nodes_Q = ctx.Q.tabulate_dof_coordinates()[:, :2]
            # forward
            np.savez(f"{iter_dir}/simulation_results.npz",
                     H=np.asarray(states["h"])[:, None, :],
                     C=np.asarray(states["C"])[:, None, :],
                     nodes=nodes_G, dt=p.dt, Q=states["Q"], zp=states["zp"])
            # adjunto
            from fenicsx import io as fx_io
            fx_io.save_adjoint_results(
                iter_dir, nodes_G, nodes_Q,
                adj_states["psiH"], adj_states["psiC"], p.dt
            )
            if animate_on:
                # Forward (H/C, sin V: este solver no lleva historial de
                # velocidad) + adjunto (psi_H/psi_C) de ESTA iteración.
                fx_io.animate_forward_fields(
                    iter_dir, ctx.G, nodes_Q, states["h"], states["C"], p.dt,
                    Q0=float(Q_c[0]),
                )
                fx_io.animate_adjoint_results(
                    iter_dir, ctx.G, nodes_Q,
                    adj_states["psiH"], adj_states["psiC"], p.dt,
                )
        g_scaled = grad / J_scale
        gn = float(np.linalg.norm(g_scaled))
        history["grad_norm"].append(gn)
        if ctx.comm.rank == 0:
            print(f"[fenicsx-opt grad #{n_jac[0]}] ‖∇J‖={gn:.3e} (escalado, gtol=1e-5)")
        return g_scaled

    # gtol: norma del gradiente proyectado en espacio escalado (∥∇J/J_scale∥ < gtol)
    # equivale a condición KKT (Cap. 3 Obs. Slater + Thm. existencia).
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
        if save_on:
            np.savez(
                f"{opt_dir}/optimization_results.npz",
                Q=np.asarray(history["Q"]),
                zp=np.asarray(history["zp"]),
                J=np.asarray(history["J"]),
                grad_norm=np.asarray(history["grad_norm"]),
                x_opt=res.x, J_opt=J_phys, J_opt_scaled=res.fun, S_opt=S_opt,
            )
            print(f"[fenicsx-opt] historial guardado ✔  "
                  f"{opt_dir}/optimization_results.npz")
            try:
                from scripts.opt_animation import render_opt_animation
                render_opt_animation(
                    np.asarray(history["Q"]), np.asarray(history["zp"]),
                    np.asarray(history["J"]), dt=p.dt, fig_dir=opt_dir,
                    title=f"FENICSX · {p.domain}/{p.model}",
                )
            except Exception as e:
                print(f"[fenicsx-opt] aviso: no se pudo generar opt_animation ({e})")
    return res


# ---------------------------------------------------------------------------
# RUN_TYPE=comparison: forward físico + adjunto discreto, ambos guardados.
# ---------------------------------------------------------------------------
def adjoint_fenicsx(p, mesh_cache=None):
    """Forward físico (Wendland C2 → simulation_results.npz) +
    adjunto discreto (Gaussiano consistente → adjoint_results.npz).
    Llamado desde main.py con run_type='comparison'."""
    from fenicsx.forward import solve_forward_fenicsx
    from fenicsx import io as fx_io

    pozo = list(p.pozo)

    # Forward físico: guarda simulation_results.npz en p.save_data
    _orig = getattr(p, "save_dat", False)
    p.save_dat = True
    solve_forward_fenicsx(p.Qout[0], pozo, p, mesh_cache=mesh_cache)
    p.save_dat = _orig

    # Forward adjunto-consistente (gaussiano) + adjunto discreto
    ctx = build_context(pozo, p, mesh_cache)
    Q_arr = np.asarray(p.Qout[0], float).ravel()
    zp    = float(pozo[1])
    states = _solve_forward(ctx, Q_arr, zp)
    _, adj_states = _solve_adjoint_grad(ctx, states)

    # Guardar adjoint_results.npz (solo rank 0 en runs MPI)
    if ctx.comm.rank == 0:
        nodes_G = ctx.G.tabulate_dof_coordinates()[:, :2]
        nodes_Q = ctx.Q.tabulate_dof_coordinates()[:, :2]
        fx_io.save_adjoint_results(
            p.save_data, nodes_G, nodes_Q,
            adj_states["psiH"], adj_states["psiC"], p.dt,
        )
