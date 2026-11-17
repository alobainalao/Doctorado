from contextlib import ExitStack
from funtions.step_time import step_time
from funtions.runtime import RUNTIME
from view.plot import update_supertitles, update_optimization_frame, create_optimization_figures
from postprocessing.create_btc  import create_btc
from view.plot  import create_figures, initialize_plots, create_adj_figures, initialize_adj_plots, initialize_optimization_plots
from funtions.utils import update_pozo, get_solucion, compute_functional
import os
import numpy as np
import time
from matplotlib.animation import FFMpegWriter
p = RUNTIME.params


# =========================================================
# SOLVE ADJOINT
# =========================================================

def solve_adjoint(hist, Qout, d, animate=False, save_data=False):
    """Campos adjuntos ψ_h, ψ_C (y ψ_im en MRMT) del adjunto discreto EXACTO, el mismo del gradiente
    (funtions/adjunto_exacto.py), a partir de las historias del directo `hist` (lo que devuelve solve_forward)."""
    if getattr(p, "upwind_transport", False):
        raise NotImplementedError("adjunto BFR: el adjunto exacto no cubre el transporte upwind")
    d.psi_H = [np.zeros_like(hk) for hk in d.H]
    d.psi_C = [np.zeros_like(ck) for ck in d.C]
    h, U, C = hist["H"], hist["U"], hist["C"]

    figs = None
    visuals = None
    if animate:
        figs, axes, cmaps = create_adj_figures()
        visuals = initialize_adj_plots(d, axes, cmaps, figs)

    from funtions.adjunto_exacto import gradiente_exacto
    exacto = {}
    gradiente_exacto(d, p, np.asarray(Qout[0], float), float(d.pozo_cor[1]), h, U, C, Ci=hist.get("Ci"),
                     Csol=hist.get("Csol"), debug=exacto)
    sim = SimulationAdjoint(d.psi_H, d.psi_C, exacto=exacto)

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
            nodes=d.nodes,
            anim_stride=getattr(p, "anim_stride", 1)
        )

    run_simulation(sim, p.Nt, outputs, d, figs, visuals)

    hist = finalize_outputs(outputs, sim)

    if p.postproc:
        create_btc()
    return hist

def pack_controls(Q, zp):

    return np.concatenate([
        Q.ravel(),
        np.array([zp])
    ])

def unpack_controls(x):

    Q = x[:-1]

    zp = x[-1]

    return Q, zp

# Historias del último directo por control: functional y gradient en el mismo x (trust-constr los pide así)
# comparten un solo directo, sin escribir ni releer el npz.
_ULTIMO = {"clave": None, "hist": None}


def _directo(x, d, p, animate=False, save_data=False):
    """Directo con el control x = [Q, z_p]; devuelve las historias {H, U, C[, Ci, Csol]} (niveles 1..N_t)."""
    Q, zp = unpack_controls(np.asarray(x, float))
    clave = (np.asarray(x, float).tobytes(), p.model, id(d))
    if _ULTIMO["clave"] == clave and not animate and not save_data:
        return _ULTIMO["hist"]
    pozo_cor = np.array(p.pozo, dtype=float)   # p.pozo puede ser entero: sin esto z_p se truncaba a metros enteros
    pozo_cor[1] = zp
    d = update_pozo(d, pozo_cor)
    hist = solve_forward(d, [Q.copy()], animate, save_data=save_data)
    _ULTIMO.update(clave=clave, hist=hist)
    return hist


def functional(x, d, p, animate=False, save_data=False):
    Q, zp = unpack_controls(np.asarray(x, float))
    hist = _directo(x, d, p, animate, save_data)
    return compute_functional(Q, zp, hist["C"], d)


def gradient(x, d, p, animate=False, save_data=False):
    """∇Ĵ por el adjunto discreto exacto (funtions/adjunto_exacto.py): derivada exacta del esquema programado."""
    if getattr(p, "upwind_transport", False):
        raise NotImplementedError("gradiente BFR: el adjunto exacto no cubre el transporte upwind")
    from funtions.adjunto_exacto import gradiente_exacto
    Q, zp = unpack_controls(np.asarray(x, float))
    hist = _directo(x, d, p, animate, save_data)
    gQ, gzp = gradiente_exacto(d, p, Q, zp, hist["H"], hist["U"], hist["C"], Ci=hist.get("Ci"),
                               Csol=hist.get("Csol"))
    return pack_controls(gQ, gzp)


def optimize_bfr(d, p, max_iter=None):
    """Optimización adjunta sobre x = [Q(t), z_p] con el optimizador común (funtions/optimizador.py): variables
    escaladas, trust-constr y paro por la medida de criticidad χ; mismo algoritmo que optimize_fenicsx."""
    import json
    from funtions.optimizador import minimizar
    from funtions.water_supply import storage_trajectory

    Q0 = np.asarray(p.Qout[0], float)
    x0 = pack_controls(Q0, float(p.pozo[1]))
    zp_lo = float(getattr(p, "zp_min", float(d.nodes[:, 1].min()) + float(p.spacing)))
    zp_hi = float(getattr(p, "zp_max", float(d.nodes[:, 1].max()) - float(p.spacing)))
    if max_iter:
        p.opt_maxiter = int(max_iter)

    opt_dir = p.save_data
    os.makedirs(opt_dir, exist_ok=True)
    prog_path = os.path.join(opt_dir, "progress.jsonl")
    open(prog_path, "w").close()
    history = {"Q": [], "zp": [], "J": [], "grad_norm": []}
    t0 = time.time()

    def _log(msg):
        print(msg, flush=True)

    def fun(x):
        t_s = time.time()
        J = functional(x, d, p)
        history["Q"].append(x[:-1].copy()); history["zp"].append(float(x[-1])); history["J"].append(float(J))
        with open(prog_path, "a") as fh:
            fh.write(json.dumps({"eval": len(history["J"]), "J": float(J), "zp": float(x[-1]),
                                 "Q_mean": float(x[:-1].mean()), "iter_s": round(time.time() - t_s, 1),
                                 "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        _log(f"[bfr-opt] J={J:.6e}  z_p={x[-1]:.1f}m  Q̄={x[:-1].mean():.3e}  ({time.time() - t_s:.0f}s)")
        return J

    def jac(x):
        g = gradient(x, d, p)
        history["grad_norm"].append(float(np.linalg.norm(g)))
        _log(f"[bfr-opt grad #{len(history['grad_norm'])}] z_p={x[-1]:.1f}m  ‖∇Ĵ‖={history['grad_norm'][-1]:.3e}")
        return g

    res, chi = minimizar(fun, jac, x0, p, zp_lo, zp_hi, log=_log,
                         progreso=getattr(p, "opt_progreso", os.path.join(opt_dir, "progreso")))
    x_opt = res.x
    S_opt = storage_trajectory(x_opt[:-1], p)
    _log(f"[bfr-opt] terminado: J={res.fun:.6e}  success={res.success}  nit={res.nit}  msg={res.message}")
    _log(f"[bfr-opt] z_p*={x_opt[-1]:.2f}m  S∈[{S_opt.min():.2f}, {S_opt.max():.2f}] m³")
    np.savez(f"{opt_dir}/optimization_results.npz",
             Q=np.asarray(history["Q"]), zp=np.asarray(history["zp"]), J=np.asarray(history["J"]),
             grad_norm=np.asarray(history["grad_norm"]), chi=chi,
             x_opt=x_opt, J_opt=res.fun, S_opt=S_opt)
    _log(f"[bfr-opt] historial guardado ✔  {opt_dir}/optimization_results.npz")
    try:
        from scripts.opt_animation import render_opt_animation
        render_opt_animation(np.asarray(history["Q"]), np.asarray(history["zp"]), np.asarray(history["J"]),
                             dt=p.dt, fig_dir=opt_dir, title=f"BFR · {p.domain}/{p.model}")
    except Exception as e:
        _log(f"[bfr-opt] aviso: no se pudo generar opt_animation ({e})")
    return res


class Simulation:

    def __init__(self, H_list, C_list, C_im_list, U_list, Qout):

        self.H = [h.copy() for h in H_list]
        self.U = [u.copy() for u in U_list]
        self.C = [c.copy() for c in C_list]
        self.C_im = [m for m in C_im_list]
        self.Csol = [None for _ in C_list]   # C del paso antes del recorte C ≥ 0 (lo usa el adjunto exacto)

        self.Qout = Qout
        self.t = 0.0
        self.dt = p.dt

    def update(self, step, d, figs, visuals):
        self.t += p.dt 

        if visuals is not None:
            im_H, im_V, im_C, im_C_im, quiv_V = visuals

        for k in range(p.K): 
            self.H[k], self.U[k], self.C[k], self.C_im[k], self.Csol[k] = step_time( H=self.H[k], U=self.U[k], C=self.C[k], C_im=self.C_im[k],
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

        if visuals is not None:
            update_supertitles(figs, self.t)

        # print(self.t/3600)

class SimulationAdjoint:

    def __init__(self, psi_H_list, psi_C_list, exacto):

        self.exacto = exacto   # {'LamH','LamC','LamIm'} de gradiente_exacto (nivel l = paso+1)

        self.psi_H = [h.copy() for h in psi_H_list]
        self.psi_C = [c.copy() for c in psi_C_list]

        self.t = p.T
        self.dt = p.dt

        # ψ_C_im — condición terminal: cero. Shape (Nr, N) para K=1.
        if "mrmt" in p.model:
            N = len(psi_H_list[0])
            self.psi_C_im = np.zeros((p.Nr, N))
        else:
            self.psi_C_im = None

    def update(self, step, h, U, C, Qout, d, figs=None, visuals=None):

        self.t -= p.dt

        if visuals is not None:
            im_H, im_C = visuals

        for k in range(p.K):

            # =====================================================
            # STEP ADJUNTO
            # =====================================================
            # multiplicador nodal Λ → ψ del documento: ψ(x_i) = Λ_i/(Δt w_i) (Δt de la cuadratura temporal, w_i de la
            # espacial); en los nodos fantasma (w_i = 0, fuera del dominio) ψ = 0
            inv = np.divide(1.0, p.dt * d.wi, out=np.zeros_like(d.wi), where=d.wi > 0)
            self.psi_H[k] = self.exacto["LamH"][step + 1] * inv
            if getattr(p, "bc_outlet_mode", "pfs") == "libre":
                # Cierre (D): h|Γ_out evoluciona solo (las filas de salida de A_left/A_right no usan columnas de
                # otros nodos), su multiplicador no realimenta al interior ni entra en ∇Q ni ∇z_p, y FEniCSx lo
                # elimina (Dirichlet). Convención del documento (§salida): ψ_h = 0 en Γ_out.
                self.psi_H[k][np.asarray(d.groups["boundary:outlet"], int)] = 0.0
            self.psi_C[k] = self.exacto["LamC"][step + 1] * inv
            lam_im = self.exacto.get("LamIm")
            if lam_im is not None and lam_im[step + 1] is not None:
                self.psi_C_im = lam_im[step + 1] * inv   # solo para guardar (psi_Ci), misma escala que ψ_C
            # =====================================================
            # VISUALIZATION (optional)
            # =====================================================
            if visuals is not None:
                i, j = divmod(k, p.ncols)

                SH = d.I.dot(self.psi_H[k])
                SH[~d.mask] = np.nan
                SH = SH.reshape(d.xy_grid.shape[1:]).T
                im_H[i][j].set_data(SH)

                SC = d.I.dot(self.psi_C[k])
                SC[~d.mask] = np.nan
                SC = SC.reshape(d.xy_grid.shape[1:]).T
                im_C[i][j].set_data(SC)

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

            # 🔹 DATA (siempre en memoria: el funcional y el adjunto la leen de aquí; el npz sólo con save_data)
            outputs["H_hist"].append(sim.H.copy())
            outputs["C_hist"].append(sim.C.copy())
            outputs["U_hist"].append(sim.U.copy())
            outputs["Csol_hist"].append(sim.Csol.copy())
            if "mrmt" in p.model:
                for r in range(p.Nr):
                    outputs["Ci_hist"][r].append(np.array([sim.C_im[k][r].copy() for k in range(p.K)]))


            # 🔹 ANIMACIÓN
            if outputs["animate"] and (n % outputs.get("anim_stride", 1) == 0):
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



def setup_outputs(figs=None, save_data=False, animate=False, nodes=None, anim_stride=1):

    outputs = {
        "save_data": save_data,
        "animate": animate,
        "anim_stride": int(anim_stride),
        "H_hist": [],
        "C_hist": [],
        "U_hist": [],
        "Csol_hist": [],
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
    """Historias del directo {H, U, C[, Ci], Csol} (niveles 1..N_t); el npz sólo con save_data."""
    save_dict = {
        "H": np.array(outputs["H_hist"]),
        "C": np.array(outputs["C_hist"]),
        "U": np.array(outputs["U_hist"]),
        "nodes": outputs["nodes"],
        "dt": sim.dt
    }
    if "mrmt" in p.model:
        save_dict["Ci"] = np.array([np.array(hist) for hist in outputs["Ci_hist"]])

    if outputs["save_data"]:
        np.savez(f"{p.save_data}/simulation_results.npz", **save_dict)
        print("Datos guardados ✔")
    save_dict["Csol"] = np.array(outputs["Csol_hist"])
    return save_dict

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
                np.array(hist[::-1]) for hist in outputs["psi_Ci_hist"]   # se llenó hacia atrás: orden directo
            ])

        np.savez(
            f"{p.save_data}/adjoint_results.npz",
            **save_dict
        )

        print("Datos adjuntos guardados ✔")