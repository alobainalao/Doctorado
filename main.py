from funtions.runtime import RUNTIME


def main():

    p = RUNTIME.get()
    save_data = p.save_dat
    animate = p.animate

    if getattr(p, "metodo", "bfr") == "fenicsx":
        # Backend FEM (DOLFINx). Import perezoso: bfr_env importa `rbf` que no
        # existe en fenics_env, y viceversa. Corre bajo fenics_env (ver app.py).
        from fenicsx.forward import solve_forward_fenicsx

        if p.run_type not in ("standard", "optimization", "comparison"):
            raise ValueError(f"run_type desconocido: {p.run_type}")

        if p.domain != "real":
            raise NotImplementedError(
                "El backend fenicsx sólo soporta domain='real'."
            )

        if p.model not in ("adr", "mrmt_semi", "mrmt_block"):
            raise NotImplementedError(
                f"El backend fenicsx no soporta model='{p.model}'."
            )

        # Etapa de preprocesamiento configurable (flag `pre`): regenera la malla
        # o reusa la cacheada. Guardado/animación/postproceso los maneja el
        # forward según save_dat/animate/postproc (ver solve_forward_fenicsx).
        from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
        if getattr(p, "mesh_pkl", None):
            # Malla congelada compartida con BFR (experimentos/).
            mesh_cache = build_mesh_from_bfr_pkl(p.mesh_pkl)
        elif getattr(p, "bfr_mesh", False):
            pkl_path = f"data/input/bfr/real/{p.model}/preproceso.pkl"
            mesh_cache = build_mesh_from_bfr_pkl(pkl_path)
        else:
            mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=bool(getattr(p, "pre", False)))

        if p.run_type == "optimization":
            # Adjunto discreto (AD de UFL) + scipy L-BFGS-B sobre x=[Q(t), z_p].
            from fenicsx.adjoint import optimize_fenicsx
            return optimize_fenicsx(p, mesh_cache=mesh_cache)

        if p.run_type == "comparison":
            # Forward físico + adjunto discreto, ambos guardados para comparación.
            from fenicsx.adjoint import adjoint_fenicsx
            return adjoint_fenicsx(p, mesh_cache=mesh_cache)

        return solve_forward_fenicsx(p.Qout[0], p.pozo, p, mesh_cache=mesh_cache)

    from view.animation import solve_forward, solve_adjoint, optimize_bfr
    from preprocessing.preprocess import load_data
    from funtions.utils import get_solucion

    d = load_data()

    # Para upwinding direccional: preconstruir Dx_upw, Dy_upw con V inicial.
    # La dirección del flujo cambia poco en tiempo → se reusan en todos los pasos.
    if getattr(p, 'upwind_transport', False):
        import numpy as np
        from funtions.upwind_operators import build_directional_gradient
        print("Construyendo gradiente upwind direccional (paso único)...", flush=True)
        p._Dx_upw, p._Dy_upw = build_directional_gradient(
            d.nodes, d.U, d.eps_M,
            n=p.n_stencil, phi=p.phi, order=p.order,
        )
        print("Gradiente upwind listo.", flush=True)

    if p.run_type == "standard":
        Qout = p.Qout
        solve_forward(d, Qout, animate, save_data)

    elif p.run_type == "comparison":
        # Forward + adjunto discreto guardados para comparación animada.
        Qout = p.Qout
        solve_forward(d, Qout, False, True)
        h, U, C = get_solucion(
            ["H", "U", "C"],
            f"{p.save_data}/simulation_results.npz",
        )
        solve_adjoint(h, U, C, Qout, d, False, True)

    elif p.run_type == "optimization":
        # Adjunto (ψ_h/ψ_C validados por tests/test_gradient_checkpoint.py) +
        # scipy L-BFGS-B, mismo optimizador que el backend fenicsx.
        return optimize_bfr(d, p)

    else:
        raise ValueError(f"run_type desconocido: {p.run_type}")


if __name__ == "__main__":
    from config.parameters import Parameters
    if RUNTIME.params is None:
        RUNTIME.params = Parameters()
    main()
