from funtions.runtime import RUNTIME


def main():

    p = RUNTIME.get()
    save_data = p.save_dat
    animate = p.animate

    if getattr(p, "metodo", "bfr") == "fenicsx":
        # Backend FEM (DOLFINx). Import perezoso: bfr_env importa `rbf` que no
        # existe en fenics_env, y viceversa. Corre bajo fenics_env (ver app.py).
        from fenicsx.forward import solve_forward_fenicsx

        if p.run_type not in ("standard", "optimization"):
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
        from fenicsx.mesh import get_mesh
        mesh_cache = get_mesh(p.pozo, p.spacing, regenerate=bool(getattr(p, "pre", False)))

        if p.run_type == "optimization":
            # Adjunto discreto (AD de UFL) + scipy L-BFGS-B sobre x=[Q(t), z_p].
            from fenicsx.adjoint import optimize_fenicsx
            return optimize_fenicsx(p, mesh_cache=mesh_cache)

        return solve_forward_fenicsx(p.Qout[0], p.pozo, p, mesh_cache=mesh_cache)

    from view.animation import solve_forward, optimize_bfr
    from preprocessing.preprocess import load_data

    d = load_data()

    if p.run_type == "standard":
        Qout = p.Qout
        solve_forward(d, Qout, animate, save_data)

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
