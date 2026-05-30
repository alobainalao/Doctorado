from contextlib import ExitStack
from funtions.step_time import step_time, step_adjoint
from funtions.runtime import RUNTIME
from view.plot import update_supertitles, update_optimization_frame, create_optimization_figures, setup_optimization_outputs

from postprocessing.create_btc  import create_btc
from view.plot  import create_figures, initialize_plots, create_adj_figures, initialize_adj_plots, initialize_optimization_plots
from funtions.utils import update_pozo, get_solucion, grad_Q, grad_zp, chi_eps, compute_functional
import os
import numpy as np
import time
from matplotlib.animation import FFMpegWriter
p = RUNTIME.params


# =========================================================
# SOLVE ADJOINT
# =========================================================

def solve_adjoint(h, U, C, Qout, d, animate=False, save_data=False):

    # ==============================
    # 🔹 SOLO SI HAY ANIMACIÓN
    # ==============================
    figs = None
    visuals = None

    if animate:
        figs, axes, cmaps = create_adj_figures()
        visuals = initialize_adj_plots(d, axes, cmaps, figs)

    # =====================================================
    # SIMULACIÓN ADJUNTA
    # =====================================================
    sim = SimulationAdjoint(d.psi_H, d.psi_C)

    outputs = setup_outputs_adj(
        figs=figs,
        save_data=save_data,
        animate=animate,
        nodes=d.nodes
    )

    run_simulation_adj(sim, h, U, C, Qout, p.Nt, outputs, d, figs, visuals)

    finalize_outputs_adj(outputs, sim)

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

    fig = None
    axes = None
    visuals = None

    if animate:
        fig, axes = create_optimization_figures()

        visuals = initialize_optimization_plots(
            history={"Q": [], "zp": [], "J": []},
            fig=fig,
            axes=axes
        )

    outputs = setup_optimization_outputs(
        fig=fig,
        save_data=save_data,
        animate=animate
    )

    # =====================================================
    # VARIABLES
    # =====================================================
    Qout = p.Qout
    pozo_cor = p.pozo

    g_prev = None
    dQ = None

    d.psi_H = [np.zeros_like(h) for h in d.H]
    d.psi_C = [np.zeros_like(c) for c in d.C]

    # =====================================================
    
    # TIMER
    # =====================================================
    t_start = time.perf_counter()

    print("Iniciando optimización...")


    if outputs["animate"]:
        stack = ExitStack()
        stack.enter_context(outputs["context"])

    try:

        for k in range(max_iter):

            # =================================================
            # GEOMETRÍA
            # =================================================

            d = update_pozo(d, pozo_cor)

            # =================================================
            # FORWARD
            # =================================================

            solve_forward(d, Qout, animate, save_data)

            h, U, C = get_solucion(
                ["H", "U", "C"],
                f"{p.save_data}/simulation_results.npz"
            )

            # =================================================
            # ADJOINT
            # =================================================

            solve_adjoint(h, U, C, Qout, d, animate, save_data)

            psi_C, psi_h = get_solucion(
                ["psi_C", "psi_H"],
                f"{p.save_data}/adjoint_results.npz"
            )

            # =================================================
            # GRADIENTES
            # =================================================

            gQ = grad_Q(Qout[0], psi_h, psi_C, C, d)
            gzp = grad_zp(Qout[0], psi_h, psi_C, C, d)

            # =================================================
            # DIRECCIÓN CONJUGADA
            # =================================================

            if k == 0:
                dQ = -gQ
            else:
                beta = np.dot(gQ, gQ) / (np.dot(g_prev, g_prev) + 1e-14)
                dQ = -gQ + beta * dQ

            # =================================================
            # STEP
            # =================================================

            alpha = 1e-3

            Qout[0] += alpha * dQ
            pozo_cor[1] -= alpha * gzp

            # =================================================
            # FUNCIONAL
            # =================================================

            J = compute_functional(Qout[0], pozo_cor[1], C, d)

            # =================================================
            # HISTORIAL
            # =================================================

            if outputs["save_data"]:

                outputs["Q_hist"].append(Qout[0].copy())
                outputs["zp_hist"].append(pozo_cor[1].copy())
                outputs["J_hist"].append(J)

            # =================================================
            # ANIMACIÓN
            # =================================================

            if outputs["animate"]:

                update_optimization_frame(
                    k,
                    {
                        "Q": outputs["Q_hist"],
                        "zp": outputs["zp_hist"],
                        "J": outputs["J_hist"]
                    },
                    visuals,
                    fig
                )

                outputs["writer"].grab_frame()

            g_prev = gQ.copy()

    finally:

        if outputs["animate"]:
            stack.close()

    print(f"Tiempo total: {time.perf_counter() - t_start:.2f} s")

    return Qout, pozo_cor, outputs



def pack_controls(Q, zp):

    return np.concatenate([
        Q.ravel(),
        np.array([zp])
    ])

def unpack_controls(x):

    Q = x[:-1]

    zp = x[-1]

    return Q, zp

def functional(
    x,
    d,
    p,
    animate=False,
    save_data=True
):

    # -----------------------------------------------------
    # controles
    # -----------------------------------------------------

    Q, zp = unpack_controls(x)

    Qout = [Q.copy()]

    pozo_cor = p.pozo.copy()

    pozo_cor[1] = zp

    # -----------------------------------------------------
    # actualizar pozo
    # -----------------------------------------------------

    d = update_pozo(d, pozo_cor)

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------

    solve_forward(
        d,
        Qout,
        animate,
        save_data
    )

    h, U, C = get_solucion(
        ["H", "U", "C"],
        f"{p.save_data}/simulation_results.npz"
    )

    # -----------------------------------------------------
    # funcional
    # -----------------------------------------------------

    J = compute_functional(
        Q,
        zp,
        C,
        d
    )

    return J

def gradient(
    x,
    d,
    p,
    animate=False,
    save_data=True
):

    # -----------------------------------------------------
    # controles
    # -----------------------------------------------------

    Q, zp = unpack_controls(x)

    Qout = [Q.copy()]

    pozo_cor = p.pozo.copy()

    pozo_cor[1] = zp

    # -----------------------------------------------------
    # actualizar pozo
    # -----------------------------------------------------

    d = update_pozo(d, pozo_cor)

    # -----------------------------------------------------
    # FORWARD
    # -----------------------------------------------------

    solve_forward(
        d,
        Qout,
        animate,
        save_data
    )

    h, U, C = get_solucion(
        ["H", "U", "C"],
        f"{p.save_data}/simulation_results.npz"
    )

    # -----------------------------------------------------
    # ADJOINT
    # -----------------------------------------------------

    solve_adjoint(
        h,
        U,
        C,
        Qout,
        d,
        animate,
        save_data
    )

    psi_C, psi_h = get_solucion(
        ["psi_C", "psi_H"],
        f"{p.save_data}/adjoint_results.npz"
    )

    # -----------------------------------------------------
    # gradientes parciales
    # -----------------------------------------------------

    gQ = grad_Q(
        Q,
        psi_h,
        psi_C,
        C,
        d
    )

    gzp = grad_zp(
        Q,
        psi_h,
        psi_C,
        C,
        d
    )

    # -----------------------------------------------------
    # gradiente global
    # -----------------------------------------------------

    g = pack_controls(gQ, gzp)

    return g

def line_search(
    f,
    x,
    ddir,
    g,
    d,
    p,
    alpha0=1.0,
    c=1e-4,
    tau=0.5,
    max_ls=20
):

    fx = f(x, d, p)

    alpha = alpha0

    gd = np.dot(g, ddir)

    for _ in range(max_ls):

        x_trial = x + alpha * ddir

        f_trial = f(x_trial, d, p)

        # Armijo
        if f_trial <= fx + c * alpha * gd:

            return alpha

        alpha *= tau

    return alpha


# =========================================================
# NONLINEAR CONJUGATE GRADIENT
# =========================================================

def nonlinear_cg(
    d,
    p,
    maxiter=50,
    tol=1e-6,
    animate=False,
    save_data=True
):

    # -----------------------------------------------------
    # control inicial
    # -----------------------------------------------------

    Q0 = p.Qout[0]

    zp0 = p.pozo[1]

    x = pack_controls(Q0, zp0)

    # -----------------------------------------------------
    # gradiente inicial
    # -----------------------------------------------------

    g = gradient(
        x,
        d,
        p,
        animate=False,
        save_data=save_data
    )

    # dirección inicial
    ddir = -g

    # historial
    history = {
        "J": [],
        "grad_norm": [],
        "Q": [],
        "zp": []
    }

    # =====================================================
    # ITERACIONES
    # =====================================================

    for k in range(maxiter):

        # -------------------------------------------------
        # funcional actual
        # -------------------------------------------------

        J = functional(
            x,
            d,
            p,
            animate=False,
            save_data=save_data
        )

        # -------------------------------------------------
        # norma gradiente
        # -------------------------------------------------

        gnorm = np.linalg.norm(g)

        # guardar
        Qk, zpk = unpack_controls(x)

        history["J"].append(J)
        history["grad_norm"].append(gnorm)
        history["Q"].append(Qk.copy())
        history["zp"].append(zpk)

        # print
        print(
            f"Iter {k:03d} | "
            f"J={J:.6e} | "
            f"||g||={gnorm:.3e} | "
            f"zp={zpk:.4f}"
        )

        # -------------------------------------------------
        # stopping
        # -------------------------------------------------

        if gnorm < tol:

            print("Convergencia alcanzada ✔")

            break

        # -------------------------------------------------
        # line search
        # -------------------------------------------------

        alpha = line_search(
            functional,
            x,
            ddir,
            g,
            d,
            p
        )

        # -------------------------------------------------
        # update variables
        # -------------------------------------------------

        x_new = x + alpha * ddir

        # -------------------------------------------------
        # nuevo gradiente
        # -------------------------------------------------

        g_new = gradient(
            x_new,
            d,
            p,
            animate=False,
            save_data=save_data
        )

        # -------------------------------------------------
        # Polak-Ribiere
        # -------------------------------------------------

        y = g_new - g

        beta = np.dot(g_new, y) / (
            np.dot(g, g) + 1e-14
        )

        # restart automático
        beta = max(beta, 0.0)

        # -------------------------------------------------
        # nueva dirección conjugada
        # -------------------------------------------------

        ddir = -g_new + beta * ddir

        # -------------------------------------------------
        # update variables
        # -------------------------------------------------

        x = x_new
        g = g_new

    # =====================================================
    # RESULTADO FINAL
    # =====================================================

    Q_opt, zp_opt = unpack_controls(x)

    return Q_opt, zp_opt, history



class Simulation:

    def __init__(self, H_list, C_list, C_im_list, U_list, Qout):

        self.H = [h.copy() for h in H_list]
        self.U = [u.copy() for u in U_list]
        self.C = [c.copy() for c in C_list]
        self.C_im = [m for m in C_im_list]


        self.Qout = Qout
        self.t = 0.0
        self.dt = p.dt

    def update(self, step, d, figs, visuals):
        self.t += p.dt 

        if visuals is not None:
            im_H, im_V, im_C, im_C_im, quiv_V = visuals

        for k in range(p.K): 
            self.H[k], self.U[k], self.C[k], self.C_im[k] = step_time( H=self.H[k], U=self.U[k], C=self.C[k], C_im=self.C_im[k],
                                         nodes=d.nodes, groups=d.groups, normals=d.normals, A_solver=d.A_left, 
                                         eps_M=d.eps_M, K=d.K, grad=d.grad, pho=d.pho, D_f=d.D_f, A_right=d.A_right,
                                         delta_p=d.delta_p, gauss_p=d.gauss_p, gauss_f=d.gauss_f, Qout_n=self.Qout[k][step-1],
                                         Qout_N=self.Qout[k][step], t=self.t, T=p.T, exp_lam_dt=d.exp_lam_dt) 


            if visuals is not None:
                i, j = divmod(k, p.ncols)
                                            
                # ----- campo H ----- S
                SH = d.I.dot(self.H[k]); SH[~d.mask] = np.nan 
                SH = SH.reshape(d.xy_grid.shape[1:]).T 

                im_H[i][j].set_data(SH)
                
                # ----- velocidad ----- 
                speed = np.sqrt(self.U[k][:,0]**2 + self.U[k][:,1]**2) 
                Sv = d.I.dot(speed); Sv[~d.mask] = np.nan 
                Sv = Sv.reshape(d.xy_grid.shape[1:]).T 

                im_V[i][j].set_data(Sv)

                Ux = self.U[k][:, 0]/speed 
                Uy = self.U[k][:, 1]/speed 
                Ux0 = d.I.dot(Ux); Ux0[~d.mask] = np.nan 
                Uy0 = d.I.dot(Uy); Uy0[~d.mask] = np.nan 

                Ux0 = Ux0.reshape(d.xy_grid.shape[1:]).T 
                Uy0 = Uy0.reshape(d.xy_grid.shape[1:]).T 

                quiv_V[i][j].set_UVC(Ux0[::p.sp_q, ::p.sp_q], Uy0[::p.sp_q, ::p.sp_q])
                
                # ----- campo C ----- 
                SC = d.I.dot(self.C[k]); 
                SC[~d.mask] = np.nan 
                SC = SC.reshape(d.xy_grid.shape[1:]).T 
                
                im_C[i][j].set_data(SC) 
                
                if "mrmt" in p.model:
                    # ----- campo C_im -----
                    for r in range(p.Nr):
                        SC_im = d.I.dot(self.C_im[k][r])
                        SC_im[~d.mask] = np.nan
                        SC_im = SC_im.reshape(
                                d.xy_grid.shape[1:]
                        ).T

                        im_C_im[r][i][j].set_data(SC_im)

            # if p.activate_ext and self.Qout[k][step] and p.run_type == "optimization":
            #     self.C[k] -= d.gauss_p*self.C[k]

            if p.activate_ext:
                self.C[k] -= d.gauss_p*self.C[k]*chi_eps(self.Qout[k][step])

        if visuals is not None:
            update_supertitles(figs, self.t)

        # print(self.t/3600)

class SimulationAdjoint:

    def __init__(self, psi_H_list, psi_C_list):

        self.psi_H = [h.copy() for h in psi_H_list]
        self.psi_C = [c.copy() for c in psi_C_list]

        self.t = p.T  # 👈 adjunto inicia al final del tiempo
        self.dt = p.dt

    def update(self, step, h, U, C, Qout, d, figs=None, visuals=None):

        self.t -= p.dt

        if visuals is not None:
            im_H, im_C = visuals

        for k in range(p.K):
        
            # =====================================================
            # STEP ADJUNTO
            # =====================================================
            self.psi_H[k], self.psi_C[k] = step_adjoint(
                A_solver=d.A_H,
                B=d.B_H,
                # ---------------------------------------------
                # adjoint state at n+1
                # ---------------------------------------------
                psiH_N=self.psi_H[k],
                psiC_N=self.psi_C[k],
                

                # ---------------------------------------------
                # forward states
                # ---------------------------------------------
                H_N=h[step][k],
                H_n=h[step - 1][k],

                # ---------------------------------------------
                # adjoint state at n+1
                # ---------------------------------------------
                V_N=U[step][k],
                V_n=U[step-1][k],

                C_N=C[step][k],
                C_n=C[step - 1][k],

                # ---------------------------------------------
                # hydraulic
                # ---------------------------------------------
                S_s=d.pho,
                K=d.K,
                Div_K=d.div_K,

                pho=d.pho,

                # ---------------------------------------------
                # transport
                # ---------------------------------------------
                D=d.D_f,

                # ---------------------------------------------
                # source / pumping
                # ---------------------------------------------
                Qout_N=Qout[k][step],
                Qout_n=Qout[k][step - 1],

                gauss_p=d.gauss_p,
                gamma=d.gamma,
                delta_p=d.delta_p,

                # ---------------------------------------------
                # derivative operators
                # ---------------------------------------------
                grad=d.grad,
                # ---------------------------------------------
                # geometry
                # ---------------------------------------------
                nodes=d.nodes,
                groups=d.groups,
                normals=d.normals,
                eps_M=d.eps_M
            )

            # =====================================================
            # VISUALIZATION (optional)
            # =====================================================
            if visuals is not None:
                i, j = divmod(k, p.ncols)

                # ---- ψ_H ----
                SH = d.I.dot(self.psi_H[k])
                SH[~d.mask] = np.nan
                SH = SH.reshape(d.xy_grid.shape[1:]).T

                im_H[i][j].set_data(SH)

                # ---- ψ_C ----
                SC = d.I.dot(self.psi_C[k])
                SC[~d.mask] = np.nan
                SC = SC.reshape(d.xy_grid.shape[1:]).T

                im_C[i][j].set_data(SC)

            # if p.activate_ext and self.Qout[k][step]:
            #     self.C[k] -= d.gauss_p*self.C[k]

            if p.activate_ext:
                self.psi_C[k] -= d.gauss_p*self.psi_C[k]*chi_eps(Qout[k][step])

        # if visuals is not None:
        #     update_supertitles(figs, self.t)

def run_simulation(sim, frames, outputs, d, figs=None, visuals=None):

    t_start = time.perf_counter()
    print("Iniciando simulación...")

    if outputs["animate"]:
        stack = ExitStack()
        for c in outputs["context"]:
            stack.enter_context(c)

    try:
        for n in range(frames):

            sim.update(n, d, figs, visuals)

            # 🔹 DATA
            if outputs["save_data"]:
                outputs["H_hist"].append(sim.H.copy())
                outputs["C_hist"].append(sim.C.copy())
                outputs["U_hist"].append(sim.U.copy())
                if "mrmt" in p.model:
                    for r in range(p.Nr):
                        outputs["Ci_hist"][r].append(sim.C_im[r].copy())



            # 🔹 ANIMACIÓN
            if outputs["animate"]:
                writer_H, writer_V, writer_C, writer_C_im = outputs["writers"]

                writer_H.grab_frame()
                writer_V.grab_frame()
                writer_C.grab_frame()

                if "mrmt" in p.model:
                    for r in range(p.Nr):
                        writer_C_im[r].grab_frame()

    finally:
        if outputs["animate"]:
            stack.close()

    print(f"Tiempo total: {time.perf_counter() - t_start:.2f} s")

def run_simulation_adj(sim, h, U, C, Qout, frames, outputs, d, figs=None, visuals=None):

    import time
    from contextlib import ExitStack

    t_start = time.perf_counter()
    print("Iniciando simulación adjunta...")

    maxC=[]
    maxH=[]

    if outputs["animate"]:
        stack = ExitStack()
        for c in outputs["context"]:
            stack.enter_context(c)

    try:
        for n in range(frames - 1, -1, -1):

            # =====================================================
            # UPDATE ADJUNTO
            # =====================================================
            sim.update(n, h, U, C, Qout, d, figs, visuals)
            # =====================================================
            # DATA STORAGE
            # =====================================================

            maxC.append(np.max(sim.psi_C))
            maxH.append(np.max(sim.psi_H))

            if outputs["save_data"]:

                outputs["psi_H_hist"].appendleft([x.copy() for x in sim.psi_H])
                outputs["psi_C_hist"].appendleft([x.copy() for x in sim.psi_C])

                if "mrmt" in p.model:
                    for r in range(p.Nr):
                        outputs["psi_Ci_hist"][r].append(
                            sim.psi_C_im[r].copy()
                        )

            # =====================================================
            # ANIMATION
            # =====================================================
            if outputs["animate"]:

                writer_H, writer_C, writer_psi_Ci = outputs["writers"]

                writer_H.grab_frame()
                writer_C.grab_frame()

    finally:
        if outputs["animate"]:
            stack.close()
    
    print(np.max(maxH))
    print(np.max(maxC))
    print(f"Tiempo total adjunto: {time.perf_counter() - t_start:.2f} s")


def create_writers(figs):
    fig_H, fig_V, fig_C, fig_C_im = figs

    writer_kwargs = dict(
        fps=20,
        codec="libx264",
        bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p"]
    )

    writer_H = FFMpegWriter(**writer_kwargs)
    writer_V = FFMpegWriter(**writer_kwargs)
    writer_C = FFMpegWriter(**writer_kwargs)
    if "mrmt" in p.model:
        writer_C_im = [FFMpegWriter(**writer_kwargs) for _ in range(p.Nr)]
    else:
        writer_C_im = []

    context = [
        writer_H.saving(fig_H, f"{p.save_video}/H.mp4", dpi=150),
        writer_V.saving(fig_V, f"{p.save_video}/V.mp4", dpi=150),
        writer_C.saving(fig_C, f"{p.save_video}/C.mp4", dpi=150)
    ]

    if "mrmt" in p.model:
        for r in range(p.Nr):
            context.append(
                writer_C_im[r].saving(
                    fig_C_im[r],
                    f"{p.save_video}/C_im_r{r}.mp4",
                    dpi=150
                )
            )

    return [writer_H, writer_V, writer_C, writer_C_im], context

def create_optimization_writers(fig):

    writer_kwargs = dict(
        fps=10,
        codec="libx264",
        bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p"]
    )

    writer = FFMpegWriter(**writer_kwargs)

    context = writer.saving(
        fig,
        f"{p.save_video}/optimization.mp4",
        dpi=150
    )

    return writer, context

def create_writers_adj(figs):

    os.makedirs(f"{p.save_video}/adjoin", exist_ok=True)

    fig_psi_H, fig_psi_C = figs[:2]

    writer_kwargs = dict(
        fps=20,
        codec="libx264",
        bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p"]
    )

    # =====================================================
    # MAIN WRITERS
    # =====================================================
    writer_psi_H = FFMpegWriter(**writer_kwargs)
    writer_psi_C = FFMpegWriter(**writer_kwargs)

    # =====================================================
    # MRMT (optional)
    # =====================================================
    if "mrmt" in p.model:
        writer_psi_Ci = [
            FFMpegWriter(**writer_kwargs)
            for _ in range(p.Nr)
        ]
    else:
        writer_psi_Ci = []

    # =====================================================
    # CONTEXT MANAGERS
    # =====================================================
    context = [
        writer_psi_H.saving(
            fig_psi_H,
            f"{p.save_video}/adjoin/psi_H.mp4",
            dpi=150
        ),
        writer_psi_C.saving(
            fig_psi_C,
            f"{p.save_video}/adjoin/psi_C.mp4",
            dpi=150
        )
    ]

    if "mrmt" in p.model:
        for r in range(p.Nr):
            context.append(
                writer_psi_Ci[r].saving(
                    figs[2][r],  # fig_Ci list
                    f"{p.save_video}/adjoin/psi_Ci_r{r}.mp4",
                    dpi=150
                )
            )

    return [writer_psi_H, writer_psi_C, writer_psi_Ci], context



def setup_outputs(figs=None, save_data=False, animate=False, nodes=None):

    outputs = {
        "save_data": save_data,
        "animate": animate,
        "H_hist": [],
        "C_hist": [],
        "U_hist": [],
        "nodes": nodes
    }
    if "mrmt" in p.model:
        outputs["Ci_hist"] = [ [] for _ in range(p.Nr) ]

    if animate:
        writers, context = create_writers(figs)
        outputs["writers"] = writers
        outputs["context"] = context

    return outputs

def setup_outputs_optimization(figs=None, save_data=False, animate=False):

    outputs = {
        "save_data": save_data,
        "animate": animate,

        # historial optimización
        "Q_hist": [],
        "zp_hist": [],
        "J_hist": [],

        "frames": []
    }

    if animate:
        writer, context = create_optimization_writers(figs)
        outputs["writer"] = writer
        outputs["context"] = context

    return outputs

def setup_outputs_adj(figs=None, save_data=False, animate=False, nodes=None):
    from collections import deque

    outputs = {
        "save_data": save_data,
        "animate": animate,

        # histories adjuntos
        "psi_H_hist": deque([]),
        "psi_C_hist": deque([]),

        "nodes": nodes
    }

    if "mrmt" in p.model:
        outputs["psi_Ci_hist"] = [[] for _ in range(p.Nr)]

    if animate:
        writers, context = create_writers_adj(figs)
        outputs["writers"] = writers
        outputs["context"] = context

    return outputs



def finalize_outputs(outputs, sim):

    if outputs["save_data"]:

        save_dict = {
            "H": np.array(outputs["H_hist"]),
            "C": np.array(outputs["C_hist"]),
            "U": np.array(outputs["U_hist"]),
            "nodes": outputs["nodes"],
            "dt": sim.dt
        }

        # agregar MRMT solo si aplica
        if "mrmt" in p.model:
            save_dict["Ci"] = np.array([np.array(hist) for hist in outputs["Ci_hist"]])

        # una sola escritura
        np.savez(f"{p.save_data}/simulation_results.npz", **save_dict)



        print("Datos guardados ✔")

def finalize_optimization_outputs(history):

    # -----------------------------------------------------
    # guardar historial
    # -----------------------------------------------------

    np.savez(
        f"{p.save_data}/optimization_history.npz",

        Q=np.array(history["Q"]),

        zp=np.array(history["zp"]),

        J=np.array(history["J"])
    )

    print("Historial de optimización guardado ✔")

def finalize_outputs_adj(outputs, sim):

    if outputs["save_data"]:

        save_dict = {
            "psi_H": np.array(outputs["psi_H_hist"]),
            "psi_C": np.array(outputs["psi_C_hist"]),
            "nodes": outputs["nodes"],
            "dt": sim.dt
        }

        if "mrmt" in p.model:
            save_dict["psi_Ci"] = np.array([
                np.array(hist) for hist in outputs["psi_Ci_hist"]
            ])

        np.savez(
            f"{p.save_data}/adjoint_results.npz",
            **save_dict
        )

        print("Datos adjuntos guardados ✔")