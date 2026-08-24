from funtions.runtime import RUNTIME


def main():

    p = RUNTIME.get()
    save_data = p.save_dat
    animate = p.animate

    if getattr(p, "metodo", "bfr") == "fenicsx":
        # Backend FEM (DOLFINx) — ver fenicsx/README.md. Solo el forward de
        # un run está portado; sin optimización/adjunto todavía, y sin
        # verificar en un entorno real (dolfinx no está en bfr_env).
        #
        # Import perezoso (aquí, no al inicio del módulo): el stack de bfr
        # (view.animation -> funtions.step_time -> funtions.operators)
        # importa `rbf`, que no existe en el entorno de fenicsx (Docker
        # dolfinx/dolfinx) — y viceversa, bfr_env no tiene dolfinx. Si el
        # import de view.animation estuviera al nivel de módulo, main.py no
        # se podría ni importar en ninguno de los dos entornos por separado.
        from fenicsx.forward import solve_forward_fenicsx

        if p.run_type != "standard":
            raise NotImplementedError(
                "El backend fenicsx todavía no implementa optimización/adjunto "
                "(ver fenicsx/README.md, punto 2). Usa run_type='standard'."
            )

        if p.domain != "real":
            raise NotImplementedError(
                "El backend fenicsx sólo soporta domain='real' (la malla se "
                "genera con los datos de pozos reales, ver fenicsx/mesh.py)."
            )

        if p.model != "adr":
            raise NotImplementedError(
                f"El backend fenicsx sólo soporta model='adr', no '{p.model}' "
                "(MRMT no está portado al FEM)."
            )

        return solve_forward_fenicsx(p.Qout[0], p.pozo, p)

    from view.animation import solve_forward, conjugate_gradient
    from preprocessing.preprocess import load_data

    d = load_data()

    if p.run_type == "standard":
        Qout = p.Qout
        solve_forward(d, Qout, animate, save_data)

    elif p.run_type == "optimization":
        conjugate_gradient(d, animate, save_data, p)

    else:
        raise ValueError(f"run_type desconocido: {p.run_type}")


if __name__ == "__main__":
    main()
