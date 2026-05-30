import os
import json
import numpy as np
from config.geometry import build_geometry


class Parameters:

    def __init__(self, env=None, json_path="config/default.json"):

        # =========================
        # 1. CARGA BASE
        # =========================
        if env is None:
            # 🔥 modo terminal → JSON
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            # 🔥 modo app → env
            VALID_KEYS = {
                "model", "domain", "metodo",

                "pre", "postproc", "plot_grid",

                "activate_fuente", "activate_ext",

                "spacing", "dt", "T",

                "phi_const", "a_l", "a_t", "D_d", "eps",

                "Nr", "Deff", "phi_im", "beta",

                "save_dat", "animate", "run_type"
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
        self.plot_grid = getattr(self, "plot_grid", False)

        self.activate_fuente = getattr(self, "activate_fuente", True)
        self.activate_ext = getattr(self, "activate_ext", True)

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
        self.n_stencil = getattr(self, "n_stencil", 25)
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
        geom = build_geometry(self.domain, self.spacing)
        for k, v in geom.items():
            setattr(self, k, v)

        # =========================
        # GRIDS
        # =========================
        self.ncols = int(np.ceil(np.sqrt(self.K)))
        self.nrows = int(np.ceil(self.K / self.ncols))
        self.min_sp = self.spacing

        # =========================
        # PATHS
        # =========================
        self.save_data = f"./data/output/{self.metodo}/data/{self.domain}/{self.model}"
        self.save_video = f"./data/output/{self.metodo}/figures/{self.domain}/{self.model}"
        self.save_preprocess = f"./data/input/{self.metodo}/{self.domain}/{self.model}"

        os.makedirs(self.save_data, exist_ok=True)
        os.makedirs(self.save_video, exist_ok=True)
        os.makedirs(self.save_preprocess, exist_ok=True)

        Q_base = [1e-3, 1e-4, 1e-5, 0]
        self.Nt = int(self.T / self.dt)
        self.Qout = [np.full(self.Nt, Q_base[i]) for i in range(self.K)]

        # =========================
        # OPTIMIZACIÓN
        # =========================
        self.gamma = getattr(self, "gamma", 1.0)
        self.koppa = getattr(self, "koppa", 1.0)
        self.z0 = getattr(self, "z0", 0.0)

        self.run_type = getattr(self, "run_type", "standard")

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

        for r in range(self.Nr):
            val = (self.beta[r] * self.Deff[r]) / (
                2 * self.beta[r] * self.R * self.L**2 + self.dt * self.Deff[r]
            )
            self.alpha_r[r] = val
            self.alpha_sum += val
