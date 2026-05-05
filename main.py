import numpy as np
from funtions.runtime import RUNTIME
from postprocessing.create_btc  import create_btc
from view.plot  import create_figures, initialize_plots
from view.animation  import *
from funtions.utils import update_pozo, get_solucion, grad_Q, grad_zp
from view import *

from preprocessing.preprocess  import load_data

def solve_adjoint(C, h, Q, zp, p):
    pass

def solve_forward(d, Qout, animate, save_data):
    # ==============================
    # 🔹 SOLO SI HAY ANIMACIÓN
    # ==============================
    figs = None
    visuals = None

    if animate:
        figs, axes, cmaps = create_figures()
        visuals = initialize_plots(d, Qout, axes, cmaps, figs)


    sim = Simulation(d.H, d.C, d.C_im, d.U, Qout)

    outputs = setup_outputs(
            figs=figs,
            save_data=save_data,
            animate=animate,
            nodes=d.nodes
        )

    run_simulation(sim, p.Nt, outputs, d, figs, visuals)

    finalize_outputs(outputs, sim)

    if p.postproc:
        create_btc()

def conjugate_gradient(d, animate, save_data, p, max_iter=20):

    Qout = p.Qout
    pozo_cor = p.pozo

    g_prev = None
    dQ = None

    for k in range(max_iter):
        update_pozo(d, pozo_cor)

        # FORWARD
        solve_forward(d, Qout, animate, save_data)
        h, C = get_solucion("H", "C", f"{p.save_data}/simulation_results.npz")

        # ADJOINT
        solve_adjoint(C, h, Q, zp, p)
        psi_C, psi_h = get_solucion("psi_C", "psi_H", f"{p.save_data}/simulation_results.npz")

        # GRADIENTES
        gQ = grad_Q(Q, zp, psi_h, psi_C, C, p)
        gzp = grad_zp(Q, zp, psi_h, psi_C, C, p)

        # dirección conjugada
        if k == 0:
            dQ = -gQ
        else:
            beta = np.dot(gQ, gQ) / np.dot(g_prev, g_prev)
            dQ = -gQ + beta * dQ

        # step (puedes hacer line search después)
        alpha = 1e-3

        Q = Q + alpha * dQ
        zp = zp - alpha * gzp

        g_prev = gQ

    return Q, zp


def main():
    print("main")
    
    p = RUNTIME.get()
    save_data = p.save_dat
    animate = p.animate

    d = load_data()

    if p.run_type == "forward":
        Qout = p.Qout
        solve_forward(d, Qout, animate, save_data)

    elif p.run_type == "optimization":
        conjugate_gradient(d, animate, save_data, p)

    else:
        raise ValueError(f"run_type desconocido: {run_type}")


if __name__ == "__main__":
    main()
