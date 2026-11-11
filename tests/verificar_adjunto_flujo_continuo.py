"""
Segunda verificación (ver tests/verificar_adjunto_continuo.py): ecuación adjunta del FLUJO.
En el último paso (sin acoplamiento temporal) el código resuelve  (∂R^h/∂h)ᵀψ_h = −(∂R^C/∂h)ᵀψ_C.
Aquí se descompone (∂R^C/∂h)ᵀψ_C en los términos que el documento escribe en el segundo miembro,

    ∫ K(ψ_C∇C)·∇v      (advección: −∇·(K ψ_C ∇C))
    ∫ K d·∇v            (dispersión:  −∇·(K d), d = ∂ₓC∂ₓψ ∇ᵤD_x + ∂_zC∂_zψ ∇ᵤD_z)
    ∫ φ C ψ_C v / Δt    (almacenamiento: S_s ∂ₜ(Cψ))

y se ajusta por mínimos cuadrados el coeficiente de cada uno (±1 si la ecuación del documento es la del código,
el residuo del ajuste debe ser ~0).  Salida en consola; en fenics_env.
"""
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)

from funtions.runtime import RUNTIME
from config.parameters import Parameters

NT = 3
ENV = {"metodo": "fenicsx", "run_type": "optimization", "domain": "real", "spacing": 120, "dt": 36000,
       "T": NT * 36000, "save_dat": False, "animate": False, "postproc": False, "pre": True, "model": "adr",
       "extract_C": False, "factor_almacenamiento_C": 1.0, "theta_C": 1.0}


def main():
    RUNTIME.params = Parameters(env=ENV)
    p = RUNTIME.get()
    p.C_init_override = 1.0
    p.C_init_modulacion = (0.5, 2000.0, 800.0)
    p.gamma = 1.0e12
    p.koppa = 0.0
    p.clip_C = False
    p.factor_almacenamiento_C = 1.0

    import ufl
    from ufl import grad, inner
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector, assemble_matrix
    from fenicsx.mesh import get_mesh
    import fenicsx.adjoint as A

    ctx = A.build_context(list(p.pozo), p, get_mesh(list(p.pozo), p.spacing, regenerate=True))
    Q = np.asarray(p.Qout[0], float)
    zp = float(p.pozo[1])
    st = A._solve_forward(ctx, Q, zp)
    _, ad = A._solve_adjoint_grad(ctx, st)
    nL = NT - 1
    dt, phys = float(p.dt), ctx.phys
    Qs, G = ctx.Q, ctx.G
    zero = lambda: fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0))
    zp_c = fem.Constant(ctx.domain, A.PETSc.ScalarType(zp))
    C_n, C_prev, psi = fem.Function(Qs), fem.Function(Qs), fem.Function(Qs)
    h_n, h_prev = fem.Function(G), fem.Function(G)
    C_n.x.array[:] = st["C"][nL]; C_prev.x.array[:] = st["C"][nL - 1]
    h_n.x.array[:] = st["h"][nL]; h_prev.x.array[:] = st["h"][nL - 1]
    psi.x.array[:] = np.asarray(ad["psiC"])[nL] * dt   # multiplicador discreto = Δt ψ_doc
    Rc = A._transport_residual(ctx, C_n, C_prev, h_n, h_prev, zero(), zp_c, zero(), zero())
    dh = ufl.TrialFunction(G)
    Ach = assemble_matrix(fem.form(ufl.derivative(Rc, h_n, dh)), bcs=[]); Ach.assemble()
    w = Ach.createVecRight(); Ach.multTranspose(psi.x.petsc_vec, w)    # (∂R^C/∂h)ᵀψ_C   (en G)

    v = ufl.TestFunction(G)
    alpha, Kz = phys["alpha"], ctx.K_z
    u = A._u_expr(h_n, Kz, alpha)
    uvar = ufl.variable(u)
    Dv = A._Dm(uvar, phys["a_l"], phys["a_t"], phys["eps"])
    dDx, dDz = ufl.diff(Dv[0], uvar), ufl.diff(Dv[1], uvar)
    Cx, Cz = grad(C_n)[0], grad(C_n)[1]
    px, pz = grad(psi)[0], grad(psi)[1]
    b0 = psi * grad(C_n)
    d = Cx * px * dDx + Cz * pz * dDz
    Kdot = lambda b: alpha ** 2 * Kz * b[0] * grad(v)[0] + Kz * b[1] * grad(v)[1]
    dx = A.dx
    Mg = assemble_matrix(fem.form(inner(ufl.TrialFunction(G), v) * ufl.dx)); Mg.assemble()
    ksp = A._lu_solver(Mg, ctx.comm)

    def vec(f):
        b = assemble_vector(fem.form(f))
        b.ghostUpdate(addv=A.PETSc.InsertMode.ADD, mode=A.PETSc.ScatterMode.REVERSE)
        return b

    cols = {"adveccion  ∫K(ψ∇C)·∇v": vec(Kdot(b0) * dx), "dispersion ∫K d·∇v": vec(Kdot(d) * dx),
            "almacenamiento ∫φCψ v/Δt": vec(ctx.phi * C_n * psi * v / dt * dx)}
    names = list(cols)
    Ys = []
    for k in names:
        y = cols[k].duplicate(); ksp.solve(cols[k], y); Ys.append(y)
    yw = w.duplicate(); ksp.solve(w, yw)
    Gm = np.array([[cols[a].dot(Ys[j]) for j in range(3)] for a in names])
    bv = np.array([w.dot(Ys[j]) for j in range(3)])
    c = np.linalg.solve(Gm, bv)
    res = w.copy()
    for k, ck in zip(names, c):
        res.axpy(-ck, cols[k])
    yres = res.duplicate(); ksp.solve(res, yres)
    rel = float(np.sqrt(abs(res.dot(yres) / w.dot(yw))))
    print("coeficientes ajustados de (∂R^C/∂h)ᵀψ_C sobre los términos del documento:")
    for k, ck in zip(names, c):
        print(f"   {k:32s} {ck:+.6f}")
    print(f"residuo relativo del ajuste = {rel:.2e}")
    nd = {k: float(np.sqrt(abs(cols[k].dot(Ys[i])))) for i, k in enumerate(names)}
    print("normas duales:", {k: f"{x:.2e}" for k, x in nd.items()})
    import json
    d_out = os.path.join(CODIGO, "data", "output", "verificacion_adjunto")
    os.makedirs(d_out, exist_ok=True)
    json.dump(dict(coeficientes=dict(zip(names, map(float, c))), residuo_rel=rel, normas=nd),
              open(os.path.join(d_out, "continuo_flujo_adr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
