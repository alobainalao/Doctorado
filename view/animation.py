from contextlib import ExitStack
from funtions.step_time import step_time
from funtions.runtime import RUNTIME
from view.plot import update_supertitles
import numpy as np
import time
from matplotlib.animation import FFMpegWriter
p = RUNTIME.params

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
            self.H[k], self.U[k], self.C[k], self.C_im[k] = step_time( H=self.H[k], C=self.C[k], C_im=self.C_im[k],
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

        if visuals is not None:
            update_supertitles(figs, self.t)

        # print(self.t/3600)

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


def run_simulationOS(sim, frames, writers,
                   mask_h_cor, d, context, visuals):

    writer_H, writer_V, writer_C, writer_C_im = writers

    print("Generando MP4...")
    t_start = time.perf_counter()

    with ExitStack() as stack:

        for c in context:
            stack.enter_context(c)

        for n in range(frames):
            sim.update(n, mask_h_cor, d, visuals)

            writer_H.grab_frame()
            writer_V.grab_frame()
            writer_C.grab_frame()

            for r in range(Nr):
                writer_C_im[r].grab_frame()

    print(f"Tiempo total: {time.perf_counter() - t_start:.2f} s")


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
