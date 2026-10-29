"""Peldaño 3 del caso base: control distribuido de la ecuación del calor (Manzoni–Quarteroni–Salsa 2021, §8.3).

    min J(u) = 1/2 ∫∫_{Q_T} (y - y_d)^2 + β/2 ∫∫_{Q_T} u^2          (caso 1, observación distribuida)
    min J(u) = 1/2 ∫_Ω (y(T) - z_d)^2 + β/2 ∫∫_{Q_T} u^2            (caso 2, observación final)
    y_t - Δy = u en Ω = (0,1)^2,  y = 0 en ∂Ω,  y(0) = 0,  T = 1

    y_d = -64 t e^{-|x-c|^2} χ_cruz,   z_d = -x1 e^{-|x-c|^2} χ_cruz,   cruz = {x1 ∈ (0.4,0.6) ó x2 ∈ (0.4,0.6)}

Discretizaciones RBF-FD (PHS r^3, mismos pesos que `graetz_nucleo`) y FEM P1 de referencia; esquema θ en el
tiempo (θ = 1/2: Crank–Nicolson, como el libro); control nodal en cada nivel t_0..t_Nt; J con trapecio en el tiempo
(libro ec. 8.12). Adjunto discreto (DtO) exacto y CG precondicionado con el producto L^2(Q_T) discreto.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from graetz_nucleo import rbffd_weights_unique, trap

TOL = 1e-9
T_FIN = 1.0
BOOK = {  # Tablas 8.2 y 8.3: J_h(Û) (iteraciones BFGS); malla (i) 712 vértices, Δt=0.05; (ii) 1921 vértices, Δt=0.01
    1: {"i": {1e-2: (147.6619, 36), 1e-3: (91.9129, 63), 1e-4: (55.4118, 176)},
        "ii": {1e-2: (159.5637, 46), 1e-3: (104.9392, 82), 1e-4: (67.2146, 206)}},
    2: {"i": {1e-2: (0.018837, 64), 1e-3: (0.0074174, 153), 1e-4: (0.0019042, 318)},
        "ii": {1e-2: (0.022603, 66), 1e-3: (0.012369, 151), 1e-4: (0.0050182, 308)}},
}


def cross(P, strict=True):
    """Indicador de la cruz. strict=True: intervalos abiertos (lectura literal); False: 1/2 sobre las líneas."""
    def ind(s):  # s in (0.4, 0.6)
        if strict:
            return ((s > 0.4 + TOL) & (s < 0.6 - TOL)).astype(float)
        on = (np.abs(s - 0.4) < TOL) | (np.abs(s - 0.6) < TOL)
        return np.where(on, 0.5, ((s > 0.4) & (s < 0.6)).astype(float))
    a, b = ind(P[..., 0]), ind(P[..., 1])
    return a + b - a * b


def profile(P, case, strict=True):
    """Parte espacial del blanco: caso 1 y_d = -64 t g(x); caso 2 z_d = g(x)."""
    r2 = (P[..., 0] - 0.5) ** 2 + (P[..., 1] - 0.5) ** 2
    amp = 1.0 if case == 1 else -P[..., 0]
    return amp * np.exp(-r2) * cross(P, strict)


def time_factor(t, case):
    return -64.0 * t if case == 1 else np.ones_like(t)


# Cuadratura de Dunavant de grado 5 (7 puntos) en el triángulo de referencia
_a1, _b1 = 0.059715871789770, 0.470142064105115
_a2, _b2 = 0.797426985353087, 0.101286507323456
DUN_L = np.array([[1 / 3, 1 / 3, 1 / 3], [_a1, _b1, _b1], [_b1, _a1, _b1], [_b1, _b1, _a1],
                  [_a2, _b2, _b2], [_b2, _a2, _b2], [_b2, _b2, _a2]])
DUN_W = np.array([0.225, *[0.132394152788506] * 3, *[0.125939180544827] * 3])


class FEM2:
    """FEM P1 en (0,1)^2 con n x n cuadrados partidos en triángulos (diagonales en patrón de diamante)."""

    def __init__(self, n):
        x = np.linspace(0, 1, n + 1)
        X, Y = np.meshgrid(x, x, indexing="ij")
        P = np.c_[X.ravel(), Y.ravel()]
        I, Jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); I, Jj = I.ravel(), Jj.ravel()
        k = lambda i, j: i * (n + 1) + j
        a, b, c, d = k(I, Jj), k(I + 1, Jj), k(I + 1, Jj + 1), k(I, Jj + 1)
        diag = (((I < n / 2) == (Jj < n / 2)))[:, None]  # simétrica respecto al centro
        T = np.stack([np.where(diag, np.c_[a, b, c], np.c_[a, b, d]),
                      np.where(diag, np.c_[a, c, d], np.c_[b, c, d])], 1).reshape(-1, 3)
        N = len(P)
        p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        area = 0.5 * np.abs(det)
        Gr = np.stack([np.c_[p1[:, 1] - p2[:, 1], p2[:, 0] - p1[:, 0]],
                       np.c_[p2[:, 1] - p0[:, 1], p0[:, 0] - p2[:, 0]],
                       np.c_[p0[:, 1] - p1[:, 1], p1[:, 0] - p0[:, 0]]], 1) / det[:, None, None]
        Ke = area[:, None, None] * np.einsum("tik,tjk->tij", Gr, Gr)
        Me = area[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 12
        II, JJ = np.repeat(T, 3, 1).ravel(), np.tile(T, 3).ravel()
        self.K = sp.coo_matrix((Ke.ravel(), (II, JJ)), (N, N)).tocsr()
        self.M = sp.coo_matrix((Me.ravel(), (II, JJ)), (N, N)).tocsr()
        self.bnd = (P[:, 0] < TOL) | (P[:, 0] > 1 - TOL) | (P[:, 1] < TOL) | (P[:, 1] > 1 - TOL)
        self.P, self.T, self.area, self.N, self.n = P, T, area, N, n
        # Operadores del esquema θ: Ms y' + A y = Bc u; filas de Dirichlet: y = 0
        free = sp.diags((~self.bnd).astype(float))
        self.Ms = (free @ self.M).tocsr()
        self.A = (free @ self.K + sp.diags(self.bnd.astype(float))).tocsr()  # identidad en Dirichlet
        self.Bc = self.Ms            # control distribuido en todos los vértices (N_u = N, como el libro)
        self.Mo = self.M             # observación (masa consistente)
        self.Mu = self.M             # producto L^2 del control en cada nivel
        self.Mu_solve = spla.splu(self.M.tocsc()).solve
        self.dir_rows = self.bnd

    def target_load(self, case, exact=True):
        """b = ∫ g φ_i  y  c = ∫ g^2 (exactas salvo cuadratura de grado 5 por triángulo, malla alineada con la cruz),
        o, con exact=False, interpolante nodal del blanco (lectura literal, ec. 8.12 del libro): b = M g_h, c = g_h' M g_h."""
        if not exact:
            g = profile(self.P, case)
            Mg = self.M @ g
            return Mg, float(g @ Mg)
        V = self.P[self.T]  # (nt, 3, 2)
        Xq = np.einsum("qk,tkd->tqd", DUN_L, V)
        cent = V.mean(1)
        chi = cross(cent)  # malla alineada: la cruz es constante en cada triángulo
        r2 = (Xq[..., 0] - 0.5) ** 2 + (Xq[..., 1] - 0.5) ** 2
        g = (1.0 if case == 1 else -Xq[..., 0]) * np.exp(-r2) * chi[:, None]
        w = self.area[:, None] * DUN_W[None, :]
        b = np.bincount(self.T.ravel(), np.einsum("tq,tq,qk->tk", w, g, DUN_L).ravel(), self.N)
        return b, float(np.sum(w * g * g))


class RBF2:
    """RBF-FD en la retícula uniforme de paso 1/n: Laplaciano por colocación (PHS r^3 + grado deg), Dirichlet fuerte."""

    def __init__(self, n, n_st=30, deg=3):
        h = 1.0 / n
        x = np.linspace(0, 1, n + 1)
        X, Y = np.meshgrid(x, x, indexing="ij")
        P = np.c_[X.ravel(), Y.ravel()]
        N = len(P)
        bnd = (P[:, 0] < TOL) | (P[:, 0] > 1 - TOL) | (P[:, 1] < TOL) | (P[:, 1] > 1 - TOL)
        free = np.where(~bnd)[0]
        _, idx = cKDTree(P).query(P[free], n_st)
        _, _, wl = rbffd_weights_unique(P[free], P[idx], h, deg)
        rows = [np.repeat(free, n_st), np.where(bnd)[0]]
        cols = [idx.ravel(), np.where(bnd)[0]]
        vals = [-wl.ravel(), np.ones(bnd.sum())]
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (N, N))
        self.W = np.outer(trap(x), trap(x)).ravel()
        self.Ms = sp.diags((~bnd).astype(float)).tocsr()  # colocación: y' = Δy + u en nodos interiores
        self.Bc = self.Ms
        self.Mo = sp.diags(self.W).tocsr()
        self.Mu = self.Mo
        self.Mu_solve = lambda g, W=self.W: g / W
        self.P, self.N, self.n, self.bnd, self.dir_rows = P, N, n, bnd, bnd

    def target_load(self, case, exact=True):
        """Trapecio 2D sobre el blanco nodal (1/2 en las líneas de la cruz); c exacto si exact=True."""
        g = profile(self.P, case, strict=not exact)
        b = self.W * g
        c = _exact_c(case) if exact else float(g @ b)
        return b, c


def _exact_c(case, m=2000):
    """∫_Ω g^2 con una retícula fina de punto medio alineada con la cruz (error ~1e-8 relativo)."""
    s = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(s, s, indexing="ij")
    return float((profile(np.stack([X, Y], -1), case) ** 2).mean())


def time_grid(nt, levels=0, m=4):
    """Niveles de tiempo: nt pasos uniformes Δt0 = T/nt; con levels=K, los últimos m pasos se sustituyen por una
    graduación geométrica hacia T: m pasos de Δt0/2, m de Δt0/4, ..., m de Δt0/2^K y m más de Δt0/2^K (suma exacta
    m Δt0). Resuelve la capa límite del control en t = T de la observación final (ancho ~ 1/λ_max ~ h^2) con pocos
    Δt distintos, es decir, pocas factorizaciones."""
    dt0 = T_FIN / nt
    if levels == 0:
        return np.linspace(0, T_FIN, nt + 1)
    steps = [dt0] * (nt - m) + sum(([dt0 / 2**k] * m for k in range(1, levels + 1)), []) + [dt0 / 2**levels] * m
    t = np.r_[0.0, np.cumsum(steps)]
    t[-1] = T_FIN
    return t


class HeatOCP:
    """Problema de control discreto (esquema θ) independiente de la discretización espacial.

    Paso n -> n+1:  L y^{n+1} = R y^n + Δt Bc (θ u^{n+1} + (1-θ) u^n),   L = Ms + θΔt A,  R = Ms - (1-θ)Δt A
    (filas de Dirichlet: L = I, R = 0, Bc = 0). Control U = (u^0, ..., u^Nt), N_u = N nodos por nivel."""

    def __init__(self, disc, case, beta, nt, theta=0.5, exact=True, t=None):
        """t: niveles de tiempo (por defecto nt pasos uniformes; ver `time_grid`)."""
        self.d, self.case, self.beta, self.theta = disc, case, beta, theta
        self.t = np.linspace(0, T_FIN, nt + 1) if t is None else np.asarray(t)
        self.nt = len(self.t) - 1
        self.dts = np.diff(self.t)
        self.wt = np.r_[self.dts, 0] / 2 + np.r_[0, self.dts] / 2  # pesos del trapecio en el tiempo (incluyen Δt)
        dirm = sp.diags(disc.dir_rows.astype(float))
        free = sp.diags((~disc.dir_rows).astype(float))
        self.Bc = (free @ disc.Bc).tocsr()
        self._ops = {}  # Δt -> (lu de L, R): una factorización por cada Δt distinto
        for dt in np.unique(np.round(self.dts, 14)):
            L = (free @ (disc.Ms + theta * dt * disc.A) + dirm).tocsc()
            R = (free @ (disc.Ms - (1 - theta) * dt * disc.A)).tocsr()
            self._ops[dt] = (spla.splu(L), R)
        self.op = [self._ops[dt] for dt in np.round(self.dts, 14)]
        self.b, self.c = disc.target_load(case, exact)
        self.nsweeps = 0

    # ---- estado y adjunto -----------------------------------------------------------------------------------
    def forward(self, U):
        Y = np.zeros_like(U)
        th = self.theta
        for n in range(self.nt):
            lu, R = self.op[n]
            rhs = R @ Y[n] + self.dts[n] * (self.Bc @ (th * U[n + 1] + (1 - th) * U[n]))
            Y[n + 1] = lu.solve(rhs)
        self.nsweeps += 1
        return Y

    def _obs_grad(self, Y, homog):
        """dJ/dy^n (sin el término de control)."""
        Mo = self.d.Mo
        G = np.zeros_like(Y)
        if self.case == 1:
            tf = 0.0 if homog else time_factor(self.t, 1)
            for n in range(self.nt + 1):
                G[n] = self.wt[n] * (Mo @ Y[n] - (0.0 if homog else tf[n] * self.b))
        else:
            G[-1] = Mo @ Y[-1] - (0.0 if homog else self.b)
        return G

    def backward(self, G):
        """L_{Nt-1}^T λ^{Nt} = g_Nt;  L_{n-1}^T λ^n = g_n + R_n^T λ^{n+1}  (n = Nt-1..1). λ^1..λ^Nt en Lam[1:]."""
        Lam = np.zeros_like(G)
        Lam[-1] = self.op[-1][0].solve(G[-1], trans="T")
        for n in range(self.nt - 1, 0, -1):
            Lam[n] = self.op[n - 1][0].solve(G[n] + self.op[n][1].T @ Lam[n + 1], trans="T")
        self.nsweeps += 1
        return Lam

    # ---- funcional y gradiente ------------------------------------------------------------------------------
    def J(self, U, Y=None):
        Y = self.forward(U) if Y is None else Y
        Mo, Mu = self.d.Mo, self.d.Mu
        reg = 0.5 * self.beta * sum(self.wt[n] * U[n] @ (Mu @ U[n]) for n in range(self.nt + 1))
        if self.case == 1:
            tf = time_factor(self.t, 1)
            obs = sum(self.wt[n] * (Y[n] @ (Mo @ Y[n]) - 2 * tf[n] * (Y[n] @ self.b) + tf[n] ** 2 * self.c)
                      for n in range(self.nt + 1))
            return 0.5 * obs + reg
        return 0.5 * (Y[-1] @ (Mo @ Y[-1]) - 2 * Y[-1] @ self.b + self.c) + reg

    def grad(self, U, homog=False):
        """Gradiente euclidiano de J_h (homog=True: parte lineal H U, sin el blanco)."""
        Y = self.forward(U)
        Lam = self.backward(self._obs_grad(Y, homog))
        th = self.theta
        Gu = np.empty_like(U)
        for m in range(self.nt + 1):
            s = th * self.dts[m - 1] * Lam[m] if m >= 1 else 0.0
            if m <= self.nt - 1:
                s = s + (1 - th) * self.dts[m] * Lam[m + 1]
            Gu[m] = self.beta * self.wt[m] * (self.d.Mu @ U[m]) + self.Bc.T @ s
        return Gu

    def riesz(self, G):
        """Representante de Riesz en el producto L^2(Q_T) discreto: (w_n Mu)^{-1} g_n."""
        return np.array([self.d.Mu_solve(G[n]) / self.wt[n] for n in range(self.nt + 1)])

    def zeros(self):
        return np.zeros((self.nt + 1, self.d.N))


def pcg(ocp, tol=1e-8, maxit=500):
    """CG en el producto L^2(Q_T): cada iteración = 1 barrido hacia adelante + 1 hacia atrás."""
    U = ocp.zeros()
    g = ocp.grad(U)
    z = ocp.riesz(g); d = -z
    gz = np.sum(g * z); n0 = np.sqrt(gz)
    gn = [1.0]
    for k in range(maxit):
        q = ocp.grad(d, homog=True)
        a = gz / np.sum(d * q)
        U = U + a * d
        g = g + a * q
        z = ocp.riesz(g)
        gz_new = np.sum(g * z)
        gn.append(np.sqrt(gz_new) / n0)
        if gn[-1] < tol:
            break
        d = -z + (gz_new / gz) * d
        gz = gz_new
    return U, len(gn) - 1, np.array(gn)
