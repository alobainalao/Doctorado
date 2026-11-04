import os
import json
import numpy as np
from config.geometry import build_geometry


class Parameters:

    def __init__(self, env=None, json_path="config/malla_sp60.json", config=None):

        # =========================
        # 1. CARGA BASE
        # =========================
        if config is not None:
            # 🔥 modo experimentos → dict COMPLETO (sin filtrar por VALID_KEYS:
            # el modo env descarta n_stencil, theta, phi, … y caen a defaults)
            data = dict(config)
        elif env is None:
            # 🔥 modo terminal → JSON
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            # 🔥 modo app → env
            VALID_KEYS = {
                "model", "domain", "metodo",

                "pre", "postproc", "plot_grid", "anim_stride",

                "activate_fuente", "activate_ext",

                "spacing", "min_sp", "min_sp_boundary", "s_cone", "r_circle", "r_cone", "sigma_fuente",
                "fenicsx_lc_near", "dt", "T",
                "upwind_transport",

                "phi_const", "a_l", "a_t", "D_d", "eps",

                "Nr", "Deff", "phi_im", "beta",

                "save_dat", "animate", "run_type",

                # Optimización: pesos del funcional (los controla la app, tab
                # Optimization) y cotas/iteraciones del optimizador.
                "gamma", "koppa", "z0",
                "Q_max", "Q_std", "zp_min", "zp_max", "opt_maxiter",
                # Balance de agua / suministro (funtions/water_supply.py):
                # el pozo debe cubrir un consumo diario con un tanque acotado.
                "Vscale", "demand_day", "demand_agri",
                "S0", "S_max", "t0_hour",
                "pump_cycling", "pump_ton", "pump_toff",
                "bfr_mesh",
                # Experimentos (experimentos/): BC de salida BFR, malla congelada
                # y carpeta de salida propia por caso.
                "bc_outlet_mode", "mesh_pkl", "exp_dir", "r_pozo",
                "norm_delta_pozo", "theta_C",
                # Sumidero W- del pozo (0 = anulado, 1 = modelo anterior)
                "factor_sumidero_C", "factor_almacenamiento_C",
                # Recorte de positividad C >= 0 en BFR (diagnóstico de la oscilación del pozo)
                "clip_C",
                # Fuente de contaminante: ancho vertical e intensidad (1.0 = estándar)
                "epsilon_z_fuente", "intensidad_fuente",
            }

            data = {k: v for k, v in env.items() if k in VALID_KEYS}


        # =========================
        # 2. NORMALIZAR VALORES
        # =========================
        import ast

        def cast(v):
            if isinstance(v, str):
                v = v.strip()

                # booleanos
                if v.lower() in ["true", "false"]:
                    return v.lower() == "true"

                # None
                if v.lower() == "none":
                    return None

                # listas, dicts, números, etc.
                try:
                    return ast.literal_eval(v)
                except:
                    return v

            return v


        data = {k: cast(v) for k, v in data.items()}

        # =========================
        # 3. ASIGNAR DINÁMICAMENTE
        # =========================
        for k, v in data.items():
            setattr(self, k, v)

        # =========================
        # 4. DERIVADOS PYTHON
        # =========================
        self._build_derived()

    # =====================================================
    # DERIVADOS (SIEMPRE PYTHON, NO JSON NI ENV)
    # =====================================================
    def _build_derived(self):

        # =========================
        # CONFIG GENERAL
        # =========================
        self.K = getattr(self, "K", 1)
        self.spacing = getattr(self, "spacing", 30)
        self.domain = getattr(self, "domain", "real")
        self.metodo = getattr(self, "metodo", "bfr")
        self.model = getattr(self, "model", "adr")

        # =========================
        # FLAGS
        # =========================
        self.save_dat = getattr(self, "save_dat", True)
        self.animate = getattr(self, "animate", True)

        self.pre = getattr(self, "pre", False)
        self.postproc = getattr(self, "postproc", False)
        self.anim_stride = int(getattr(self, "anim_stride", 1))
        self.plot_grid = getattr(self, "plot_grid", False)

        self.activate_fuente = getattr(self, "activate_fuente", True)
        self.activate_ext = getattr(self, "activate_ext", True)
        # Factor del sumidero de soluto en el pozo W- = χ(Q) g_ε C (transporte,
        # adjunto y gradientes, BFR y FEniCS). 0 lo anula: en la forma no
        # conservativa el soluto ya sale con el agua (exp/sumidero-W, grupo Z);
        # 1 recupera el modelo anterior.
        self.factor_sumidero_C = float(getattr(self, "factor_sumidero_C", 0.0))
        # Término de almacenamiento -S_s ∂ₜh·C del transporte (docs/adjunto_sin_sumidero.tex):
        # sobrevive al eliminar W- al despejar ∇·u del flujo. 1 lo activa (estado, adjunto
        # ψ_C y fuente -S_s ∂ₜ(Cψ_C) del adjunto del flujo); 0 = modelo reducido anterior.
        self.factor_almacenamiento_C = float(getattr(self, "factor_almacenamiento_C", 0.0))
        # Recorte C >= 0 en cada paso de BFR (docs/clip_positividad_C.md). True por
        # defecto; False sólo para diagnosticar la oscilación del nodo del pozo.
        self.clip_C = bool(getattr(self, "clip_C", True))

        # =========================
        # TIEMPO
        # =========================
        self.dt = getattr(self, "dt", 3.6e4 if self.domain == "real" else 360)
        self.T = getattr(self, "T", 2592000 if self.domain == "real" else 25920)

        # =========================
        # FÍSICOS PRINCIPALES
        # =========================
        self.phi_const = getattr(self, "phi_const", 0.1)
        self.a_l = getattr(self, "a_l", 10.0)
        self.a_t = getattr(self, "a_t", 1.0)
        self.D_d = getattr(self, "D_d", 1.2e-19)
        self.eps = getattr(self, "eps", 1e-16)

        # =========================
        # FLUIDO / PROPIEDADES
        # =========================
        self.nu = getattr(self, "nu", 1.055e-6)
        self.rho = getattr(self, "rho", 1.0)

        self.g = getattr(self, "g", 9.81)
        self.nu_C = getattr(self, "nu_C", 1.055e-6)
        self.d_z = getattr(self, "d_z", 0.001)
        self.alpha = getattr(self, "alpha", 2.0)

        # =========================
        # NUMÉRICOS AVANZADOS
        # =========================
        self.n_stencil = getattr(self, "n_stencil", 35)
        self.phi = getattr(self, "phi", "mq")
        self.order = getattr(self, "order", 2)

        self.tol = getattr(self, "tol", 1e-5)
        self.theta = getattr(self, "theta", 0.9)
        self.sp_q = getattr(self, "sp_q", 5)

        # =========================
        # OTROS
        # =========================
        self.R = getattr(self, "R", 1)
        self.landa = getattr(self, "landa", 1e-10)
        self.L = getattr(self, "L", 0.01)

        # =========================
        # GEOMETRÍA
        # =========================
        geom = build_geometry(self.domain, self.spacing, getattr(self, "sigma_fuente", 1.0),
                              getattr(self, "lado_cuadrado", 1000.0))
        for k, v in geom.items():
            setattr(self, k, v)
        # Fuente de contaminante: valores absolutos opcionales (None = los de la configuración estándar).
        #   epsilon_z_fuente [m]: ancho vertical ε_z (= epsilon_y) de la gaussiana (estándar: spacing·sigma_fuente).
        #   intensidad_fuente: coeficiente del pico temporal de la fuente (estándar: 1e-11; el de la base, 1e-12, es 1/10).
        self.epsilon_z_fuente = getattr(self, "epsilon_z_fuente", None)
        self.intensidad_fuente = getattr(self, "intensidad_fuente", None)
        if self.epsilon_z_fuente is not None:
            self.epsilon_y = float(self.epsilon_z_fuente)

        # Soporte de los kernels del pozo [m], fijo e independiente de la malla.
        # r_pozo = radio del Wendland C2 del sumidero de flujo; las gaussianas
        # del pozo (sumidero de C, observación J1, adjuntos) usan ε = r_pozo/3
        # (~99 % de la masa dentro de r_pozo). Sin r_pozo se conserva el
        # comportamiento anterior (radio 2.5·spacing, ε = epsilon_y en bfr).
        self.r_pozo = getattr(self, "r_pozo", None)
        if self.r_pozo:
            self.radio_delta_pozo = float(self.r_pozo)
            self.epsilon_pozo = float(self.r_pozo) / 3.0
        else:
            self.radio_delta_pozo = 2.5 * self.spacing
            self.epsilon_pozo = self.epsilon_y
        # Normalización de la delta del sumidero de flujo: "suma" (Σδ_i = 1,
        # histórico: el caudal efectivo escala con el área nodal) o "area"
        # (∫δ dΩ = 1: el caudal extraído es Q, independiente de la malla).
        self.norm_delta_pozo = getattr(self, "norm_delta_pozo", "suma")

        # =========================
        # GRIDS
        # =========================
        self.ncols = int(np.ceil(np.sqrt(self.K)))
        self.nrows = int(np.ceil(self.K / self.ncols))
        self.min_sp = getattr(self, "min_sp", 30)
        # Espaciado mínimo en la franja de frontera (2*min_sp_boundary metros).
        # 0 → mismo que min_sp (comportamiento clásico sin distinción frontera/interior).
        self.min_sp_boundary = getattr(self, "min_sp_boundary", 0)

        # =========================
        # PATHS
        # =========================
        # Separados por run_type ("standard"/"optimization"): así una corrida
        # de optimización no mezcla sus iteraciones con las corridas de
        # sanity-check, y cada quien vive bajo su propia carpeta.
        self.run_type = getattr(self, "run_type", "standard")
        self.save_data = f"./data/output/{self.metodo}/data/{self.domain}/{self.model}/{self.run_type}"
        self.save_video = f"./data/output/{self.metodo}/figures/{self.domain}/{self.model}/{self.run_type}"
        self.save_preprocess = f"./data/input/{self.metodo}/{self.domain}/{self.model}"

        # Experimentos: todo (datos, figuras y preproceso) vive en la carpeta
        # del caso → <exp_dir>/<metodo>/, sin pisar las rutas de arriba.
        self.exp_dir = getattr(self, "exp_dir", None)
        if self.exp_dir:
            self.save_data = os.path.join(self.exp_dir, self.metodo)
            self.save_video = os.path.join(self.exp_dir, self.metodo, "figures")
            self.save_preprocess = os.path.join(self.exp_dir, self.metodo)

        os.makedirs(self.save_data, exist_ok=True)
        os.makedirs(self.save_video, exist_ok=True)
        os.makedirs(self.save_preprocess, exist_ok=True)

        self.Nt = int(self.T / self.dt)

        # Caudal fijo de referencia para run_type="standard" (sanity check /
        # demo): mismo valor usado en las corridas de avance2 y documentado
        # como escenario base en los comentarios de water_supply más abajo
        # (Q~1e-3, dt=3.6e4). No es una restricción física, es el punto de
        # comparación con la investigación ya publicada.
        self.Q_std = getattr(self, "Q_std", 1e-3)

        # Punto de partida Q0 = Q_std para todos los run_type.
        # Para "optimization" el optimizador ajusta Q(t) libremente dentro de
        # [0, Q_max] respetando la restricción de suministro (supply_constraint);
        # usar Q_std como Q0 garantiza subsidencia de contaminante desde la
        # primera iteración (bombeo suficiente para arrastrar el contaminante
        # de z=+109 m hacia el pozo en z=−700 m) y JE ≠ 0 al arranque.
        _q0 = np.full(self.Nt, self.Q_std)

        Q_base_rest = [1e-4, 1e-5, 0]
        self.Qout = [_q0.copy()] + [np.full(self.Nt, Q_base_rest[i]) for i in range(self.K - 1)]

        # =========================
        # OPTIMIZACIÓN
        # =========================
        self.gamma = getattr(self, "gamma", 1.0)
        self.koppa = getattr(self, "koppa", 1.0)
        self.z0    = getattr(self, "z0",    0.0)
        # Cotas físicas del pozo (domain real z ∈ [−2690, 250] m)
        self.zp_min = getattr(self, "zp_min", -1400.0)
        self.zp_max = getattr(self, "zp_max",  -200.0)

        # Balance de agua / suministro (funtions/water_supply.py). Defaults
        # elegidos para que el problema sea no trivial con el escenario base
        # (Q~1e-3, dt=3.6e4): el pozo arranca con tanque vacío (S0=0) y debe
        # bombear para cubrir el consumo sin exceder S_max. Tunables por la app.
        # Vscale = L_perp [m]: anchura perpendicular al plano de la sección 2D.
        # Q [m²/s] → Q_vol [m³/s] = Q * Vscale. Por defecto = spacing = 30 m.
        self.Vscale       = getattr(self, "Vscale",       30.0)
        # 5 000 hab × 150 L/día + 250 m³/día riego → 1 000 m³/día total
        self.demand_day   = getattr(self, "demand_day",  750.0)  # doméstico [m³/día]
        self.demand_agri  = getattr(self, "demand_agri", 250.0)  # riego     [m³/día]
        self.S0           = getattr(self, "S0",          500.0)  # [m³]
        self.S_max        = getattr(self, "S_max",      2000.0)  # ≈2 días reserva [m³]
        self.t0_hour      = getattr(self, "t0_hour",       6.0)  # hora solar inicio
        self.pump_cycling = getattr(self, "pump_cycling",  True)  # ciclos bomba
        self.pump_ton     = getattr(self, "pump_ton",     20.0)  # h máx operación continua
        self.pump_toff    = getattr(self, "pump_toff",     4.0)  # h mín descanso

        # =========================
        # MRMT DERIVADOS
        # =========================
        if "mrmt" in self.model:
            self._build_mrmt()


    def _build_mrmt(self):

        if not hasattr(self, "Nr"):
            return

        
        self.Deff = np.array(getattr(self, "Deff", [1e-9, 5e-10, 1e-10]))
        self.beta = np.array(getattr(self, "beta", [0.15, 0.1, 0.05]))
        self.phi_im = np.array(getattr(self, "phi_im", [0.1, 0.05, 0.02]))

        self.R = getattr(self, "R", 1)
        self.L = getattr(self, "L", 0.01)

        self.alpha_r = np.zeros(self.Nr)
        self.alpha_sum = 0

        # Coeficiente del complemento de Schur para MRMT Block.
        # Origen: al eliminar C_im del sistema acoplado (regla trapezoidal),
        # la diagonal de la ecuación móvil acumula Σ_r 2·base·α_r y el RHS
        # acumula Σ_r 4·base·α_r·C_im_r (ver operators.py make_transport_operator).
        # Equivalencia: α_r = κ_r·β_r/dt, donde
        #   κ_r = dt·Deff_r / (2·β_r·R·L² + dt·Deff_r)
        # es la fracción de diferencia (C−C_im) transferida en un paso de tiempo.
        # Límites: κ_r → 1 cuando dt → ∞ (equilibrio instantáneo),
        #          κ_r → 0 cuando dt → 0 (sin transferencia).
        for r in range(self.Nr):
            val = (self.beta[r] * self.Deff[r]) / (
                2 * self.beta[r] * self.R * self.L**2 + self.dt * self.Deff[r]
            )
            self.alpha_r[r] = val
            self.alpha_sum += val
