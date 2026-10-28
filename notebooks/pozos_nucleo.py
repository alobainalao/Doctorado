"""Peldaño 4 del caso base: canal de Graetz + 4 pozos con tasas de extracción u_i(t) (transitorio).

Es la Parte B de `caso_base_manzoni_graetz` con término temporal: un pulso de contaminante entra por x1 = 0 y
se quiere proteger la zona aguas abajo Ω_obs = {x1 > 3.4} durante todo el intervalo [0, T],

    ∂t y − μΔy + b·∇y + Σ_i u_i(t) ψ_i(x) y = 0,   y = g(t) en x1 = 0,  μ∂n y = 0 en el resto,  y(0) = 0,
    min_{0 ≤ u ≤ u_max} J = ½ ∫_0^T ∫_Ω_obs y² + β/2 Σ_i ∫_0^T u_i(t)² dt,

con u_i constante a trozos en M intervalos iguales (como Q(t) en el simulador). Esquema θ con Δt uniforme;
una factorización LU por intervalo de control, compartida por estado y adjunto (discretizar y luego optimizar).
Las discretizaciones espaciales (RBF-FD con fantasmas y FEM P1) son las de `graetz_nucleo` con kind='canal'.
Ver `pozos_benchmark.py`.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from graetz_nucleo import FEM, RBFFD, XW, ncg_proj, wolfe_secant  # noqa: F401  (reexporta el optimizador)

T_FIN, T_PULSO = 20.0, 2.0


def pulso(t, tp=T_PULSO):
    """Concentración de entrada g(t) = sin²(π t / tp) en [0, tp] y 0 después (suave: compatible con y(0) = 0 y CN)."""
    return np.where(t < tp, np.sin(np.pi * np.minimum(t, tp) / tp) ** 2, 0.0)


def masa_fem(d):
    """Masa consistente P1 con filas de Dirichlet anuladas (las columnas multiplican valores conocidos)."""
    II, JJ = np.repeat(d.T, 3, 1).ravel(), np.tile(d.T, 3).ravel()
    M = sp.coo_matrix((d.Me.ravel(), (II, JJ)), (d.N, d.N)).tocsr()
    return (sp.diags((~d.GD).astype(float)) @ M).tocsr()


class TransientWells:
    """Problema de pozos transitorio, independiente de la discretización (RBF-FD o FEM, kind='canal').

    Paso n → n+1 en el intervalo de control m:  L_m y^{n+1} = R_m y^n + g(t_{n+1}) f,
        L_m = D/Δt + Θ K(u^m),  R_m = D/Δt − (I − Θ) K(u^m),  K(u) = A + Σ_i u_i S_i,
    D = masa de las filas diferenciales (colocación: indicador de filas de EDP; FEM: masa consistente) y
    Θ = θ en filas diferenciales, 1 en filas algebraicas (Dirichlet, condición de Neumann en el fantasma)."""

    def __init__(self, disc, T=T_FIN, nt=400, M=20, theta=0.5, beta=1e-4, umax=5.0, sigma=0.08, xw=XW):
        assert nt % M == 0
        self.d, self.beta, self.umax, self.xw, self.theta = disc, beta, umax, np.array(xw), theta
        self.T, self.nt, self.M, self.K = T, nt, M, len(xw)
        if isinstance(disc, RBFFD):
            dif = disc.pde.copy()
            self.D = sp.diags(dif.astype(float)).tocsr()
        else:
            dif = ~disc.GD
            self.D = masa_fem(disc)
        self.Th = sp.diags(np.where(dif, theta, 1.0)).tocsr()
        self.Ic = sp.diags(np.where(dif, 1.0 - theta, 0.0)).tocsr()
        psi = np.array([np.exp(-((disc.P - c) ** 2).sum(1) / (2 * sigma**2)) / (2 * np.pi * sigma**2) for c in xw])
        self.S = np.array([disc.pad(disc.node_mass()[: disc.N] * p) for p in psi])  # diagonales S_i (filas de EDP)
        self.Mobs = disc.M0("pozos")
        self.dt, self.ks, self.tau = T / nt, nt // M, T / M
        self.t = np.linspace(0, T, nt + 1)
        self.F = pulso(self.t)[:, None] * disc.f[None, :]  # dato de Dirichlet en cada t_n
        self.w = np.full(nt + 1, self.dt); self.w[[0, -1]] *= 0.5  # trapecio en tiempo
        self.nsolves = 0

    def _ops(self, U):
        ops = []
        for m in range(self.M):
            Km = self.d.A + sp.diags(U[m] @ self.S)
            L = (self.D / self.dt + self.Th @ Km).tocsc()
            ops.append((spla.splu(L), (self.D / self.dt - self.Ic @ Km).tocsr()))
        return ops

    def forward(self, U, ops=None):
        ops = self._ops(U) if ops is None else ops
        Y = np.zeros((self.nt + 1, self.d.Nt))
        for n in range(self.nt):
            lu, R = ops[n // self.ks]
            Y[n + 1] = lu.solve(R @ Y[n] + self.F[n + 1])
        self.nsolves += 1
        return Y

    def _J(self, U, Y):
        obs = np.einsum("ni,ni->n", Y, (self.Mobs @ Y.T).T)
        return 0.5 * self.w @ obs + 0.5 * self.beta * self.tau * (U**2).sum()

    def J(self, u):
        U = u.reshape(self.M, self.K)
        return self._J(U, self.forward(U))

    def Jg(self, u):
        """J y su gradiente exacto del problema discreto. Adjunto (L_{n−1})ᵀ λ^n = w_n M_obs y^n + R_nᵀ λ^{n+1};
        ∂J/∂u_i^m = β τ u_i^m − Σ_{n ∈ m} (λ^{n+1})ᵀ S_i (θ y^{n+1} + (1 − θ) y^n)."""
        U = u.reshape(self.M, self.K)
        ops = self._ops(U)
        Y = self.forward(U, ops)
        G = self.beta * self.tau * U.copy()
        lam = np.zeros(self.d.Nt)
        for n in range(self.nt, 0, -1):
            rhs = self.w[n] * (self.Mobs @ Y[n])
            if n < self.nt:
                rhs = rhs + ops[n // self.ks][1].T @ lam
            m = (n - 1) // self.ks
            lam = ops[m][0].solve(rhs, trans="T")
            G[m] -= self.S @ (lam * (self.theta * Y[n] + (1 - self.theta) * Y[n - 1]))
        self.nsolves += 1
        return self._J(U, Y), G.ravel(), Y

    def zeros(self):
        return np.zeros(self.M * self.K)


class ConstantWells:
    """Mismo problema con tasas constantes en el tiempo (u_i(t) ≡ c_i): referencia para medir lo que aporta u(t)."""

    def __init__(self, tw):
        self.tw, self.umax, self.xw, self.K, self.M = tw, tw.umax, tw.xw, tw.K, 1
        self.nsolves = 0

    def Jg(self, c):
        J, G, Y = self.tw.Jg(np.tile(c, self.tw.M)); self.nsolves += 1
        return J, G.reshape(self.tw.M, self.K).sum(0), Y

    def J(self, c):
        return self.tw.J(np.tile(c, self.tw.M))

    def zeros(self):
        return np.zeros(self.K)
