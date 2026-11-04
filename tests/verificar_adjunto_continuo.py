"""
¿El adjunto FEniCSx del modelo nuevo (ADR) satisface las ecuaciones adjuntas derivadas en
docs/reanalisis_adr_mrmt.tex?  Se evalúa, con los estados guardados del código, el residuo débil de

  −φR ∂ₜψ_C − ∇·(D∇ψ_C) − ∇·(uψ_C) + λφR ψ_C − S_s ∂ₜh ψ_C = −2γ δ_p C      (con (D∇ψ_C+uψ_C)·n = 0)

escrito aquí de forma independiente (Euler implícito, θ_C = 1, S_s = φ) para cada paso n, con ψ_doc = −ψ_código/Δt
(el código usa L = J − ψ·R con R en forma de tasa y J = γΔt Σ∫δC²; el documento L = J + ∫⟨R,ψ⟩dt).
El residuo se mide en la norma dual (rᵀ M⁻¹ r)^½ relativa a la del término mayor. Un residuo ~1e-8 o menos
significa que la ecuación escrita en el documento ES la que resuelve el código (términos, signos, frontera).
Como contraste se repite omitiendo el término de almacenamiento y el de advección (deben empeorar mucho).

    python tests/verificar_adjunto_continuo.py      # en fenics_env
Salida: data/output/verificacion_adjunto/continuo_adr.json
"""
import json
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
       "extract_C": False, "factor_sumidero_C": 0.0, "factor_almacenamiento_C": 1.0, "theta_C": 1.0}


def main():
    RUNTIME.params = Parameters(env=ENV)
    p = RUNTIME.get()
    p.C_init_override = 1.0
    p.C_init_modulacion = (0.5, 2000.0, 800.0)
    p.gamma = 1.0e12                    # J_m ~ 1e-22: sólo escala el adjunto, no cambia las ecuaciones
    p.koppa = 0.0
    p.clip_C = False
    p.factor_almacenamiento_C = 1.0

    import ufl
    from ufl import grad, inner, as_vector
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector, assemble_matrix
    from fenicsx.mesh import get_mesh
    import fenicsx.adjoint as A

    ctx = A.build_context(list(p.pozo), p, get_mesh(list(p.pozo), p.spacing, regenerate=True))
    Q = np.asarray(p.Qout[0], float)
    zp = float(p.pozo[1])
    st = A._solve_forward(ctx, Q, zp)
    _, ad = A._solve_adjoint_grad(ctx, st)
    psiC = -np.asarray(ad["psiC"]) / p.dt          # convención del documento
    dt, phys = float(p.dt), ctx.phys
    R, lam, gamma = phys["R"], phys["landa"], float(p.gamma)

    Qs, G = ctx.Q, ctx.G
    x = ufl.SpatialCoordinate(ctx.domain)
    zp_c = fem.Constant(ctx.domain, A.PETSc.ScalarType(zp))
    delta_p = A._gauss(x, ctx.xp, zp_c, ctx.ep)
    psi, psi_next, C_n = fem.Function(Qs), fem.Function(Qs), fem.Function(Qs)
    h_n, h_prev = fem.Function(G), fem.Function(G)
    v = ufl.TestFunction(Qs)
    u = A._u_expr(h_n, ctx.K_z, phys["alpha"])
    D = ctx.D_stat + A._Dm(u, phys["a_l"], phys["a_t"], phys["eps"])
    dx = A.dx
    t_tiempo = R * inner(ctx.phi * (psi - psi_next) / dt, v) * dx
    t_difusion = (D[0] * grad(psi)[0] * grad(v)[0] + D[1] * grad(psi)[1] * grad(v)[1]) * dx
    t_adveccion = inner(u, grad(v)) * psi * dx                # −∇·(uψ) integrado por partes; frontera: ver BC natural
    t_reaccion = lam * R * inner(ctx.phi * psi, v) * dx
    t_almac = -inner(ctx.phi * (h_n - h_prev) / dt * psi, v) * dx
    t_fuente = 2.0 * gamma * inner(delta_p * C_n, v) * dx
    M = A.assemble_matrix(fem.form(inner(ufl.TrialFunction(Qs), v) * dx)); M.assemble()
    ksp = A._lu_solver(M, ctx.comm)

    def vec(form):
        b = assemble_vector(fem.form(form))
        b.ghostUpdate(addv=A.PETSc.InsertMode.ADD, mode=A.PETSc.ScatterMode.REVERSE)
        return b

    def dual(b):
        y = b.duplicate()
        ksp.solve(b, y)
        return float(np.sqrt(abs(b.dot(y))))

    # Diagnóstico en el último paso (sin acoplamiento temporal): ¿la matriz adjunta del código coincide con mis términos?
    nL = NT - 1
    psi.x.array[:] = psiC[nL]; psi_next.x.array[:] = 0.0
    C_n.x.array[:] = st["C"][nL]; h_n.x.array[:] = st["h"][nL]
    h_prev.x.array[:] = st["h"][nL - 1]
    Rc = A._transport_residual(ctx, C_n, fem.Function(Qs), h_n, h_prev, fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0)),
                               zp_c, fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0)),
                               fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0)))
    C_prev_f = fem.Function(Qs); C_prev_f.x.array[:] = st["C"][nL - 1]
    Rc = A._transport_residual(ctx, C_n, C_prev_f, h_n, h_prev, fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0)),
                               zp_c, fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0)),
                               fem.Constant(ctx.domain, A.PETSc.ScalarType(0.0)))
    dC = ufl.TrialFunction(Qs)
    AccT = assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(Rc, C_n, dC))), bcs=[]); AccT.assemble()
    psi_c = fem.Function(Qs); psi_c.x.array[:] = np.asarray(ad["psiC"])[nL]
    w = AccT.createVecLeft(); AccT.mult(psi_c.x.petsc_vec, w)
    rhs_code = vec(2.0 * gamma * dt * inner(delta_p * C_n, v) * dx)
    print("sanidad: ||AccT psi_codigo - rhs||/||rhs|| =", dual(w - rhs_code) / dual(rhs_code), flush=True)
    # mis términos en convención del código: psi_c = -dt psi_doc  =>  AccT psi_c = -dt (suma de términos con psi_doc)
    cols = {k: vec(f) for k, f in dict(tiempo=t_tiempo, difusion=t_difusion, adveccion=t_adveccion,
                                       reaccion=t_reaccion, almacenamiento=t_almac).items()}
    for k, b in cols.items():
        b.scale(-dt)
    resto = w.duplicate(); resto.set(0.0)
    names = list(cols)
    # el término temporal con psi_next=0
    mine = w.duplicate(); mine.set(0.0)
    for k in names:
        mine.axpy(1.0, cols[k])
    d = w - mine
    print("||AccT psi - mis términos|| / ||AccT psi|| =", dual(d) / dual(w), flush=True)
    for k in names:
        print(f"   ||{k}||/||AccT psi|| = {dual(cols[k]) / dual(w):.3e}", flush=True)
    # ¿qué término explica d? mínimos cuadrados d ≈ Σ c_k cols_k en el producto M^-1
    Mi = lambda a, b: float(a.dot(b))
    def mv(b):
        y = b.duplicate(); ksp.solve(b, y); return y
    Y = [mv(cols[k]) for k in names]
    G_ = np.array([[Mi(cols[a], Y[j]) for j, a in enumerate(names)] for _ in names])
    G_ = np.array([[cols[names[i]].dot(Y[j]) for j in range(len(names))] for i in range(len(names))])
    bvec = np.array([d.dot(Y[j]) for j in range(len(names))])
    coef = np.linalg.lstsq(G_, bvec, rcond=None)[0]
    print("coeficientes del residuo sobre cada término (0 = coincide):", dict(zip(names, np.round(coef, 4))), flush=True)
    filas = []
    for n in range(NT):
        psi.x.array[:] = psiC[n]
        psi_next.x.array[:] = psiC[n + 1] if n + 1 < NT else 0.0
        C_n.x.array[:] = st["C"][n]
        h_n.x.array[:] = st["h"][n]
        h_prev.x.array[:] = st["h"][n - 1] if n > 0 else ctx.h0
        T = {k: vec(f) for k, f in dict(tiempo=t_tiempo, difusion=t_difusion, adveccion=t_adveccion,
                                        reaccion=t_reaccion, almacenamiento=t_almac, fuente=t_fuente).items()}
        norms = {k: dual(b) for k, b in T.items()}

        def resid(omitir=()):
            r = T["tiempo"].duplicate(); r.set(0.0)
            for k, b in T.items():
                if k not in omitir:
                    r.axpy(1.0, b)
            return dual(r)

        ref = max(norms.values())
        fila = dict(paso=n, normas=norms, residuo_rel=resid() / ref,
                    sin_almacenamiento=resid(("almacenamiento",)) / ref,
                    sin_adveccion=resid(("adveccion",)) / ref,
                    sin_fuente=resid(("fuente",)) / ref)
        filas.append(fila)
        print(f"n={n}: residuo/ref = {fila['residuo_rel']:.2e}  | sin almacenamiento {fila['sin_almacenamiento']:.2e}"
              f" | sin advección {fila['sin_adveccion']:.2e} | sin fuente {fila['sin_fuente']:.2e}", flush=True)
    out = os.path.join(CODIGO, "data", "output", "verificacion_adjunto")
    os.makedirs(out, exist_ok=True)
    json.dump(filas, open(os.path.join(out, "continuo_adr.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
