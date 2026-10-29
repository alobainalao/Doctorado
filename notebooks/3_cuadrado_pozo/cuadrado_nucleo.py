"""
Modelo mínimo del cuadrado con un pozo central (volúmenes finitos, malla
cartesiana) para decidir si el sumidero de transporte W- hace falta.

Flujo estacionario  ∇·q = -Q δ,  q = -K ∇h,  h = h_b en ∂Ω  (Thiem discreto).
Transporte, tres formulaciones con el MISMO flujo y la misma malla:

  "cons"   forma conservativa (referencia física)
           φ ∂_t C + ∇·(q C) - ∇·(D∇C) = -Q δ C
  "nc_fv"  forma no conservativa en volúmenes finitos (resta C·continuidad)
           φ ∂_t C + q·∇C - ∇·(D∇C) = -f_W κ g C
  "nc_fd"  forma no conservativa en diferencias finitas por nodos (como el
           simulador: q en el nodo y gradiente de C sin usar flujos de cara)
           φ ∂_t C + q·∇C - ∇·(D∇C) = -f_W κ g C

f_W = 0 es la forma no conservativa correcta; f_W = 1 añade W-, con
κ = Q (sumidero "físico" duplicado) o κ = 1 (lo implementado en el simulador:
W- = χ(Q) g_ε C, sin el caudal).

δ = kernel Wendland C2 de radio r_pozo (Σ w_i = 1, el del flujo) y
g = gaussiana de ancho ε = r_pozo/3 con ∫g = 1 (la de W- y de la C captada),
como en el simulador. Euler implícito, upwind de primer orden.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIA = 86400.0


class Cuadrado:
    def __init__(self, L=600.0, n=61, K=1e-3, phi=0.1, D=1e-5, Q=0.01,
                 h_b=300.0, r_pozo=20.0):
        self.L, self.n, self.K, self.phi, self.D, self.Q = L, n, K, phi, D, Q
        self.dx = L / n
        self.A = self.dx ** 2
        c = (np.arange(n) + 0.5) * self.dx - L / 2      # centros, pozo en (0, 0)
        self.X, self.Z = np.meshgrid(c, c, indexing="ij")
        r = np.hypot(self.X, self.Z).ravel()
        # δ del flujo: Wendland C2, Σ w = 1 (discrete_delta del simulador)
        s = np.clip(r / r_pozo, 0, 1)
        w = (1 - s) ** 4 * (4 * s + 1)
        self.w = w / w.sum()
        # g de W- y de la C captada: gaussiana normalizada a ∫g = 1
        eps = r_pozo / 3.0
        g = np.exp(-r ** 2 / (2 * eps ** 2))
        self.g = g / (g.sum() * self.A)
        self.N = n * n
        self._flujo(h_b)

    def idx(self, i, j):
        return i * self.n + j

    # ------------------------------------------------------------------
    def _flujo(self, h_b):
        """Carga con Dirichlet h_b en el borde (media celda) y caudales de cara."""
        n, K = self.n, self.K
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(n):
                k = self.idx(i, j)
                diag = 0.0
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        rows.append(k); cols.append(self.idx(ii, jj)); vals.append(-K)
                        diag += K
                    else:                         # cara de borde, distancia dx/2
                        diag += 2 * K
                rows.append(k); cols.append(k); vals.append(diag)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        # Σ_f K(h_i - h_j) = -Q w_i (+ el borde): la salida neta de la celda es -Q w_i
        self.h = spla.spsolve(M.tocsc(), self._rhs_flujo(h_b))
        H = self.h.reshape(n, n)
        # caudal por cara (positivo en +x / +z), por unidad de espesor: K Δh
        self.Fx = np.zeros((n + 1, n))          # caras x = i - 1/2
        self.Fz = np.zeros((n, n + 1))
        self.Fx[1:-1, :] = -K * (H[1:, :] - H[:-1, :])
        self.Fx[0, :] = -2 * K * (H[0, :] - h_b)
        self.Fx[-1, :] = -2 * K * (h_b - H[-1, :])
        self.Fz[:, 1:-1] = -K * (H[:, 1:] - H[:, :-1])
        self.Fz[:, 0] = -2 * K * (H[:, 0] - h_b)
        self.Fz[:, -1] = -2 * K * (h_b - H[:, -1])
        div = (self.Fx[1:, :] - self.Fx[:-1, :] + self.Fz[:, 1:] - self.Fz[:, :-1]).ravel()
        self.err_continuidad = np.abs(div + self.Q * self.w).max() / self.Q
        # q en el nodo (para nc_fd): media de las caras, dividido por dx
        self.qx = (0.5 * (self.Fx[1:, :] + self.Fx[:-1, :]) / self.dx).ravel()
        self.qz = (0.5 * (self.Fz[:, 1:] + self.Fz[:, :-1]) / self.dx).ravel()

    def _rhs_flujo(self, h_b):
        n, K = self.n, self.K
        b = -self.Q * self.w.copy()
        for i in range(n):
            for j in range(n):
                nb = (i == 0) + (i == n - 1) + (j == 0) + (j == n - 1)
                b[self.idx(i, j)] += 2 * K * h_b * nb
        return b

    # ------------------------------------------------------------------
    def operador(self, forma, f_W=0.0, kappa=None):
        """Matriz T y vector de entrada b del semidiscreto φA dC/dt = -T C + b·C_b
        (C_b = concentración del agua que entra por el borde)."""
        n, dx = self.n, self.dx
        kappa = self.Q if kappa is None else kappa
        Dc = self.D                                # D·(dx/dx): conductancia de cara
        T = sp.lil_matrix((self.N, self.N))
        b = np.zeros(self.N)
        for i in range(n):
            for j in range(n):
                k = self.idx(i, j)
                caras = (  # (vecino o None, caudal saliente de la celda por esa cara)
                    ((i + 1, j), self.Fx[i + 1, j]), ((i - 1, j), -self.Fx[i, j]),
                    ((i, j + 1), self.Fz[i, j + 1]), ((i, j - 1), -self.Fz[i, j]),
                )
                for (ii, jj), F in caras:
                    dentro = 0 <= ii < n and 0 <= jj < n
                    kk = self.idx(ii, jj) if dentro else None
                    if dentro:                    # difusión entre celdas
                        T[k, k] += Dc; T[k, kk] -= Dc
                    if forma in ("cons", "nc_fv"):
                        if F > 0:                 # sale: lleva C_k
                            if forma == "cons":
                                T[k, k] += F
                        else:                     # entra: trae C del vecino o del borde
                            if forma == "cons":
                                if dentro: T[k, kk] += F
                                else:      b[k] -= F
                            else:         # F (C_up - C_k), F < 0
                                if dentro: T[k, kk] += F
                                else:      b[k] -= F
                                T[k, k] -= F
                if forma == "nc_fd":
                    # A q·∇C con upwind por nodo, q del nodo (no los caudales de cara)
                    for q, (dp, dm) in ((self.qx[k], ((i + 1, j), (i - 1, j))),
                                        (self.qz[k], ((i, j + 1), (i, j - 1)))):
                        up = dm if q > 0 else dp
                        coef = abs(q) * dx        # A·|q|/dx
                        T[k, k] += coef
                        ii, jj = up
                        if 0 <= ii < n and 0 <= jj < n:
                            T[k, self.idx(ii, jj)] -= coef
                        else:
                            b[k] += coef
        T = T.tocsr()
        if forma == "cons":
            T = T + sp.diags(self.Q * self.w)                       # -Q δ C
        else:
            T = T + sp.diags(f_W * kappa * self.g * self.A)         # -f_W κ g C
        return T, b

    def correr(self, forma, C0, C_b=0.0, T_fin=90 * DIA, dt=0.25 * DIA, f_W=0.0, kappa=None):
        T, b = self.operador(forma, f_W, kappa)
        Mt = sp.diags(np.full(self.N, self.phi * self.A / dt))
        lu = spla.splu((Mt + T).tocsc())
        C = np.asarray(C0, float).ravel().copy()
        nt = int(round(T_fin / dt))
        out = dict(t=np.arange(nt + 1) * dt / DIA, masa=[self.masa(C)], c_pozo=[self.w @ C],
                   captada=[self.A * self.g @ C], C=[C.copy()])
        for _ in range(nt):
            C = lu.solve(Mt @ C + b * C_b)
            out["masa"].append(self.masa(C)); out["c_pozo"].append(self.w @ C)
            out["captada"].append(self.A * self.g @ C); out["C"].append(C.copy())
        for k in ("masa", "c_pozo", "captada"):
            out[k] = np.array(out[k])
        out["C"] = np.array(out["C"])
        # salida por el borde (caudal saliente × C de la celda) y entrada C_b
        out["borde_sale"] = np.array([self._borde_sale(c) for c in out["C"]])
        out["dt"] = dt
        return out

    def masa(self, C):
        return self.phi * self.A * C.sum()

    def _borde_sale(self, C):
        Cm = C.reshape(self.n, self.n)
        s = (np.clip(-self.Fx[0, :], 0, None) @ Cm[0, :] + np.clip(self.Fx[-1, :], 0, None) @ Cm[-1, :]
             + np.clip(-self.Fz[:, 0], 0, None) @ Cm[:, 0] + np.clip(self.Fz[:, -1], 0, None) @ Cm[:, -1])
        return s
