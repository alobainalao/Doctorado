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
            data = dict(env)

        # =========================
        # 2. NORMALIZAR VALORES
        # =========================
        def cast(v):
            if isinstance(v, str):
                if v.lower() in ["true", "false"]:
                    return v.lower() == "true"
                try:
                    if "." in v or "e" in v.lower():
                        return float(v)
                    return int(v)
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

        # ---- defaults seguros
        self.K = getattr(self, "K", 4)
        self.spacing = getattr(self, "spacing", 30)
        self.domain = getattr(self, "domain", "real")
        self.metodo = getattr(self, "metodo", "bfr")
        self.model = getattr(self, "model", "adr")

        # ---- dt según dominio
        if self.domain == "real":
            self.dt = getattr(self, "dt", 3.6e4)
            self.T = getattr(self, "T", 2592000)
        else:
            self.dt = getattr(self, "dt", 360)
            self.T = getattr(self, "T", 25920)

        # ---- geometría
        geom = build_geometry(self.domain, self.spacing)
        for k, v in geom.items():
            setattr(self, k, v)

        # ---- grids
        self.ncols = int(np.ceil(np.sqrt(self.K)))
        self.nrows = int(np.ceil(self.K / self.ncols))
        self.min_sp = self.spacing

        # ---- paths
        self.save_data = f"./data/output/{self.metodo}/data/{self.domain}/{self.model}"
        self.save_video = f"./data/output/{self.metodo}/figures/{self.domain}/{self.model}"
        self.save_data = f"./data/output/{self.metodo}/data/{self.domain}/{self.model}"
        self.save_preprocess = f"./data/input/{self.metodo}/{self.domain}/{self.model}"

        os.makedirs(self.save_data, exist_ok=True)
        os.makedirs(self.save_video, exist_ok=True)
        os.makedirs(self.save_preprocess, exist_ok=True)

        # ---- MRMT
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
