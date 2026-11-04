"""
Cuadrado con pozo y flujo TRANSITORIO (volúmenes finitos): ¿cuánta masa crea la
forma no conservativa al omitir el término S_s (∂_t h) C_m?
Nomenclatura del Cap. I de la tesis (docR.tex, ChaptersR/CapI_ADR.tex).

Flujo (ec. flujo con ec. pozo_fuente), Euler implícito, h = h_b en ∂Ω, h(0) = h_b:
    S_s A (h^{n+1} - h^n)/dt + Σ_f F_f^{n+1} = -Q w_R,i ,     F_f = -K Δh
así que la salida neta de cada celda es  Σ_f F_f = -Q w_R,i - S_s A ∂_t h
(ec. div_u). Con ese flujo, tres formulaciones del transporte (R = 1, λ_d = 0):

  "cons"  φ ∂_t C + ∇·(u C - D∇C) = -Q w_R C                          (conservativa)
  "nc"    φ ∂_t C + u·∇C - ∇·(D∇C) = 0          (ec. transporte_adr sin W_-: omite S_s ∂_t h C)
  "ncS"   φ ∂_t C + u·∇C - ∇·(D∇C) = S_s ∂_t h C   (ec. transporte_noconservativo completa)

con φ constante. Si el contenido de agua sigue al almacenamiento, θ(t) = φ + S_s (h - h_b)
(∂_t θ = S_s ∂_t h), las versiones coherentes son
  "consT" ∂_t(θ C) + ∇·(u C - D∇C) = -Q w_R C        y        "ncT" θ ∂_t C + u·∇C - ∇·(D∇C) = 0,
que son la misma ecuación: ahí la forma no conservativa NO lleva el término S_s ∂_t h C.

"cons" y "ncS" son la misma ecuación; "nc" es la que resuelven hoy los dos
backends. La diferencia entre "nc" y "cons" es exactamente  -C S_s ∂_t h.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cuadrado_nucleo import Cuadrado, DIA


class CuadradoTransitorio(Cuadrado):
    def __init__(self, S_s=1e-5, **kw):
        self.S_s = S_s
        super().__init__(**kw)

    def _flujo(self, h_b):
        """Estado estacionario (Thiem discreto) sólo para tener la matriz y los caudales; el
        transitorio se resuelve en `correr_tr`."""
        super()._flujo(h_b)
        self.h_b = h_b
        n, K = self.n, self.K
        # matriz de conductancias (la misma de la clase base) para el paso implícito
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(n):
                k = self.idx(i, j)
                diag = 0.0
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        rows.append(k); cols.append(self.idx(ii, jj)); vals.append(-K); diag += K
                    else:
                        diag += 2 * K
                rows.append(k); cols.append(k); vals.append(diag)
        self.Mflujo = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self.b_borde = self._rhs_flujo(h_b) + self.Q * self.w      # sólo la parte de borde

    def _caudales(self, h):
        """Caudales de cara y u nodal para una carga h (misma fórmula que la clase base)."""
        n, K, h_b = self.n, self.K, self.h_b
        H = h.reshape(n, n)
        self.Fx = np.zeros((n + 1, n)); self.Fz = np.zeros((n, n + 1))
        self.Fx[1:-1, :] = -K * (H[1:, :] - H[:-1, :])
        self.Fx[0, :] = -2 * K * (H[0, :] - h_b); self.Fx[-1, :] = -2 * K * (h_b - H[-1, :])
        self.Fz[:, 1:-1] = -K * (H[:, 1:] - H[:, :-1])
        self.Fz[:, 0] = -2 * K * (H[:, 0] - h_b); self.Fz[:, -1] = -2 * K * (h_b - H[:, -1])
        return (self.Fx[1:, :] - self.Fx[:-1, :] + self.Fz[:, 1:] - self.Fz[:, :-1]).ravel()

    def correr_tr(self, forma, C0, T_fin=30 * DIA, dt=0.5 * DIA, C_b=0.0):
        """forma ∈ {'cons', 'nc', 'ncS'}. Devuelve masa, c_w, c_g, entrada/salida acumuladas."""
        n_pasos = int(round(T_fin / dt))
        h = np.full(self.N, self.h_b, float)
        Mh = (self.Mflujo + sp.diags(np.full(self.N, self.S_s * self.A / dt))).tocsc()
        luh = spla.splu(Mh)
        C = np.asarray(C0, float).ravel().copy()
        out = dict(t=[0.0], masa=[self.masa(C)], c_w=[self.w @ C], c_g=[self.A * self.g @ C],
                   crea=[0.0], sale=[0.0], div_err=[])
        crea = sale = 0.0
        forma_op = "cons" if forma in ("cons", "consT") else "nc_fv"
        theta = np.full(self.N, self.phi)           # contenido de agua (sólo en las formas T)
        for _ in range(n_pasos):
            h_old = h
            h = luh.solve(self.S_s * self.A / dt * h_old + self.b_borde - self.Q * self.w)
            div = self._caudales(h)
            dhdt = (h - h_old) / dt
            self.ux = (0.5 * (self.Fx[1:, :] + self.Fx[:-1, :]) / self.dx).ravel()
            self.uz = (0.5 * (self.Fz[:, 1:] + self.Fz[:, :-1]) / self.dx).ravel()
            out["div_err"].append(np.abs(div + self.Q * self.w + self.S_s * self.A * dhdt).max() / self.Q)
            T, b = self.operador(forma_op, f_W=0.0)
            if forma == "ncS":
                T = T - sp.diags(self.S_s * self.A * dhdt)         # + S_s ∂_t h C en el lado derecho
            if forma in ("consT", "ncT"):
                th_old, theta = theta, self.phi + self.S_s * (h - self.h_b)
                Mn = sp.diags(theta * self.A / dt)
                Mo = sp.diags((th_old if forma == "consT" else theta) * self.A / dt)   # θ^n C^n o θ^{n+1} C^n
                Cn = spla.spsolve((Mn + T).tocsc(), Mo @ C + b * C_b)
            else:
                Mt = sp.diags(np.full(self.N, self.phi * self.A / dt))
                Cn = spla.spsolve((Mt + T).tocsc(), Mt @ C + b * C_b)
            # masa que sale por el pozo en este paso (Q × concentración del agua extraída)
            sale += self.Q * (self.w @ Cn) * dt
            C = Cn
            out["t"].append(out["t"][-1] + dt / DIA); out["masa"].append(self.masa(C))
            out["c_w"].append(self.w @ C); out["c_g"].append(self.A * self.g @ C); out["sale"].append(sale)
        for k in ("t", "masa", "c_w", "c_g", "sale"):
            out[k] = np.array(out[k])
        out["C"] = C; out["h"] = h; out["theta"] = theta
        return out
