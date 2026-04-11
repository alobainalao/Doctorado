import numpy as np
import time
from preprocessing.preprocess  import load_data
from postprocessing.create_btc  import create_btc
from view.plot  import create_figures, initialize_plots
from view.animation  import *
from config import dt, T, xmin
from view import *
from config import postproc


def main(save_data=False, animate=False):

    d = load_data()

    # ==============================
    # FIGURAS
    # ==============================
    Qout = [0, 1e-5, 1e-4, 1e-3]

    # ==============================
    # 🔹 SOLO SI HAY ANIMACIÓN
    # ==============================
    figs = None
    visuals = None

    if animate:
        figs, axes, cmaps = create_figures()
        visuals = initialize_plots(d, Qout, axes, cmaps, figs)



    sim = Simulation(d.H.copy(), d.C.copy(), d.C_im.copy(), d.U.copy(), Qout)

    frames = int(T / dt)

    mask_h_cor = np.isclose(d.nodes[:,0], xmin) & (d.nodes[:,1] < -4000)

    outputs = setup_outputs(
            figs=figs,
            save_data=save_data,
            animate=animate,
            nodes=d.nodes
        )

    run_simulation(sim, frames, outputs, d, mask_h_cor, visuals)

    finalize_outputs(outputs, sim)

    if postproc:
        create_btc()

if __name__ == "__main__":
    main(save_data=True, animate=True)
