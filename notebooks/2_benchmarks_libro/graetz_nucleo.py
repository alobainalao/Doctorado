"""Núcleo vectorizado del caso base Manzoni–Quarteroni–Salsa §6.5.1 (Graetz) y de la Parte B (pozos).

Mismas discretizaciones que `2a_graetz_manzoni_6_5_1.py` (RBF-FD con nodos fantasma y FEM P1 de
referencia), con el ensamblado vectorizado para que el benchmark corra en segundos y el FEM llegue a
mallas finas (valores de referencia por extrapolación de Richardson). Ver `graetz_benchmark.py`.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

LX, LY, MU, BETA = 4.0, 1.0, 0.1, 1e-3
TOL = 1e-9
C_OUT = {"libro": MU, "conormal": 1.0, "libre": 0.0}  # c en  mu dn y - c (b·n) y = mu u
CASES = {"a": (1, 1), "b": (2, 1), "c": (1, 2), "d": (2, 2)}  # (Omega_0, z_d)
XW = [(1.5, 0.5), (2.0, 0.3), (2.0, 0.7), (2.6, 0.5)]  # pozos de la Parte B


def vel(P):
    """Campo de Graetz b = (1.5 x2 (1 - x2), 0)."""
    return np.c_[1.5 * P[:, 1] * (1 - P[:, 1]), np.zeros(len(P))]


def obs_weight(P, obs):
    """Indicador de Omega_0 en nodos con peso 1/2 sobre las líneas de interfaz (consistente con trapecio)."""
    x1, x2 = P[:, 0], P[:, 1]
    step = lambda s: np.where(np.abs(s) < TOL, 0.5, (s > 0).astype(float))  # Heaviside con H(0)=1/2
    if obs == "pozos":
        return step(x1 - 3.4)
    fx = step(x1 - 1.0)
    if obs == 1:
        return fx * (step(0.3 - x2) + step(x2 - 0.7))
    return fx


def target(P, zd):
    return 2.0 * np.ones(len(P)) if zd == 1 else 4 * P[:, 1] * (1 - P[:, 1])


# ---------------------------------------------------------------------------------------------------
# RBF-FD
# ---------------------------------------------------------------------------------------------------
def poly_terms(X, deg):
    """Monomios x^i y^j (i+j<=deg) y sus derivadas d/dx, d/dy y Laplaciano. X: (..., 2) -> (..., k)."""
    x, y = X[..., 0], X[..., 1]
    val, dx, dy, lap = [], [], [], []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            val.append(x**i * y**j)
            dx.append(i * x ** max(i - 1, 0) * y**j)
            dy.append(j * x**i * y ** max(j - 1, 0))
            l = np.zeros_like(x)
            if i >= 2:
                l = l + i * (i - 1) * x ** (i - 2) * y**j
            if j >= 2:
                l = l + j * (j - 1) * x**i * y ** (j - 2)
            lap.append(l)
    return [np.stack(t, -1) for t in (val, dx, dy, lap)]


def rbffd_weights_batch(xc, Xs, deg, m=3, chunk=4000):
    """Pesos RBF-FD (PHS r^m + polinomios de grado deg) de d/dx, d/dy y Laplaciano.

    xc: (N, 2) centros; Xs: (N, n, 2) esténciles. Devuelve wx, wy, wl de forma (N, n).
    Un solo `np.linalg.solve` por bloque de nodos en vez de uno por nodo."""
    N, n, _ = Xs.shape
    _, pdx, pdy, plap = poly_terms(np.zeros((1, 2)), deg)
    k = pdx.shape[1]
    rhs_p = np.c_[pdx[0], pdy[0], plap[0]]  # (k, 3)
    W = np.empty((N, n, 3))
    for a in range(0, N, chunk):
        s = Xs[a:a + chunk] - xc[a:a + chunk, None, :]
        sc = np.abs(s).max(axis=(1, 2))
        s = s / sc[:, None, None]  # escalar el esténcil mejora el condicionamiento
        A = np.linalg.norm(s[:, :, None, :] - s[:, None, :, :], axis=3) ** m
        P = poly_terms(s, deg)[0]
        b = len(s)
        M = np.zeros((b, n + k, n + k))
        M[:, :n, :n], M[:, :n, n:], M[:, n:, :n] = A, P, P.transpose(0, 2, 1)
        r0 = np.linalg.norm(s, axis=2)
        rp = m * r0 ** (m - 2)
        rhs = np.empty((b, n + k, 3))
        rhs[:, :n] = np.stack([-rp * s[..., 0], -rp * s[..., 1], m * rp], -1)
        rhs[:, n:] = rhs_p
        w = np.linalg.solve(M, rhs)[:, :n]
        W[a:a + b] = w / np.stack([sc, sc, sc**2], -1)[:, None, :]
    return W[..., 0], W[..., 1], W[..., 2]


def rbffd_weights_unique(xc, Xs, h, deg):
    """Como `rbffd_weights_batch`, pero resuelve sólo los esténciles distintos salvo traslación.

    En una retícula uniforme (fantasmas incluidos, a distancia h) los desplazamientos x_j - x_i son múltiplos
    enteros de h: se ordenan por fila, se agrupan con `np.unique` y se reparten los pesos. Es exacto."""
    off = np.rint((Xs - xc[:, None, :]) / h).astype(np.int64)
    key = off[..., 0] * 100003 + off[..., 1]
    perm = np.argsort(key, axis=1, kind="stable")
    ks = np.take_along_axis(key, perm, 1)
    _, first, inv = np.unique(ks, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    su = np.take_along_axis(off, perm[..., None], 1)[first] * h  # esténciles únicos (ordenados)
    wu = rbffd_weights_batch(np.zeros((len(su), 2)), su.astype(float), deg)
    iperm = np.argsort(perm, axis=1)
    return [np.take_along_axis(w[inv], iperm, 1) for w in wu]


def graded_axes(n, q):
    """Coordinadas de la retícula FEM: 4n x n celdas, graduadas con exponente q hacia x1=1 y hacia las paredes.

    Cada tramo [a, b] se recorre con a + (b-a) s^q (o el simétrico) para concentrar nodos en las uniones
    Dirichlet–Neumann (1, 0) y (1, 1), donde el estado controlado tiene una singularidad tipo r^(1/2). Las
    líneas x1=1, x2=0.3, 0.5, 0.7 son siempre de la retícula (Omega_0 alineada). q=1: retícula uniforme."""
    def seg(a, b, m, hacia):  # m celdas en [a, b], concentradas en el extremo `hacia`
        s = np.linspace(0, 1, m + 1) ** q
        return a + (b - a) * s if hacia == a else b - (b - a) * s[::-1]
    x = np.r_[seg(0, 1, n, 1)[:-1], seg(1, LX, 3 * n, 1)]
    k = int(round(0.3 * n)); r = (n - 2 * k) // 2
    y = np.r_[seg(0, 0.3, k, 0)[:-1], np.linspace(0.3, 0.5, r + 1)[:-1],
              np.linspace(0.5, 0.7, r + 1)[:-1], seg(0.7, 1, k, 1)]
    return x, y


def trap(x):
    """Pesos de la regla del trapecio en los nodos x (ordenados)."""
    d = np.diff(x)
    return np.r_[d, 0] / 2 + np.r_[0, d] / 2


def grid(h, q=1.0):
    """Retícula de nodos (uniforme de paso h, o graduada con `graded_axes`) y pesos de trapecio 2D.

    La graduación tensorial sólo se usa en el FEM: con RBF-FD los esténciles kNN quedan alineados con las
    líneas finas (anisótropos) y el esquema diverge (probado el 2026-10-27 con q = 1.5, 2)."""
    if q == 1.0:
        x, y = np.linspace(0, LX, int(round(LX / h)) + 1), np.linspace(0, LY, int(round(LY / h)) + 1)
    else:
        x, y = graded_axes(int(round(LY / h)), q)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.c_[X.ravel(), Y.ravel()], np.outer(trap(x), trap(y)).ravel(), x, y


CORNERS = ((1.0, 0.0), (1.0, LY))  # uniones Dirichlet–Neumann del problema de Graetz


def refine_corners(P, h, levels, R=4):
    """Refinamiento local isótropo: añade subretículas de paso h/2^k (k = 1..levels) en cuadrados de semilado
    R·h/2^(k-1) centrados en las esquinas singulares. Cada nivel es una retícula alineada con la gruesa, así que
    los desplazamientos entre nodos siguen siendo múltiplos enteros de h/2^levels (vale la deduplicación de pesos).

    CASO NEGATIVO (2026-10-27): el estado es tan exacto como en la retícula uniforme (solución manufacturada), pero
    el óptimo es espurio: J*_RBF cae por debajo de la referencia (0.0188 vs 0.0228 en el caso a, h=1/40) con picos
    del control de ~13 (FEM: 6) en la zona refinada, y ese control evaluado en el FEM da J 3x peor. Mismo mecanismo
    que Neumann sin fantasmas. No usar con control de frontera sin evaluación cruzada."""
    pts = [P]
    for k in range(1, levels + 1):
        hk, r = h / 2**k, R * h / 2 ** (k - 1)
        for cx, cy in CORNERS:
            xs = np.arange(max(0.0, cx - r), min(LX, cx + r) + TOL, hk)
            ys = np.arange(max(0.0, cy - r), min(LY, cy + r) + TOL, hk)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            pts.append(np.c_[X.ravel(), Y.ravel()])
    Q = np.concatenate(pts)
    hm = h / 2**levels
    _, keep = np.unique(np.rint(Q / hm).astype(np.int64), axis=0, return_index=True)
    return Q[np.sort(keep)]


def lumped_area(P):
    """Pesos de cuadratura nodales = masa concentrada P1 sobre la triangulación de Delaunay (exacta para constantes)."""
    from scipy.spatial import Delaunay
    T = Delaunay(P).simplices
    p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    return np.bincount(T.ravel(), np.repeat(area / 3, 3), len(P))


def boundary_sets(P, kind):
    """kind='graetz': Dirichlet en entrada + paredes x1<=1, control en paredes x1>1.
    kind='canal' : Dirichlet sólo en la entrada, Neumann homogéneo en el resto (Parte B)."""
    x, y = P[:, 0], P[:, 1]
    left, right, bot, top = x < TOL, x > LX - TOL, y < TOL, y > LY - TOL
    GD = left | ((bot | top) & (x <= 1 + TOL)) if kind == "graetz" else left
    Gc = (bot | top) & ~right & ~GD if kind == "graetz" else np.zeros(len(P), bool)
    Gn = (bot | top | right) & ~GD  # todos los nodos Neumann (incluye Gc)
    nrm = np.zeros((len(P), 2))
    nrm[bot, 1], nrm[top, 1], nrm[right] = -1, 1, (1, 0)  # esquinas de salida: normal (1, 0)
    return GD, Gc, Gn, nrm


class RBFFD:
    """Discretización RBF-FD. Incógnitas: [nodos reales, nodos fantasma]."""

    def __init__(self, h, kind="graetz", bc="libro", ghost=True, n_st=30, deg=3, ref=0):
        """ref: niveles de refinamiento local alrededor de las esquinas singulares (0 = retícula uniforme)."""
        P, W, xs, _ = grid(h)
        hm = h / 2**ref  # paso más fino (unidad de los desplazamientos enteros)
        if ref:
            P = refine_corners(P, h, ref)
            W = lumped_area(P)
        N = len(P)
        GD, Gc, Gn, nrm = boundary_sets(P, kind)
        neu = np.where(Gn)[0]
        hn = np.full(len(neu), h)  # fantasma a la distancia del vecino interior más cercano en la normal
        if ref:
            have = set(map(tuple, np.rint(P / hm).astype(np.int64)))
            for k in range(ref, 0, -1):
                s = h / 2**k
                cand = np.rint((P[neu] - s * nrm[neu]) / hm).astype(np.int64)
                hit = np.array([tuple(v) in have for v in cand]) & (hn == h)
                hn[hit] = s
        Pall = np.r_[P, P[neu] + hn[:, None] * nrm[neu]] if ghost else P
        Nt = len(Pall)
        gid = np.arange(N)
        if ghost:
            gid[neu] = N + np.arange(len(neu))  # fila de la condición de frontera de cada nodo Neumann
        _, idx = cKDTree(Pall).query(P, n_st)
        bvec, c = vel(P), C_OUT[bc]
        free = np.where(~GD)[0]
        wx, wy, wl = rbffd_weights_unique(P[free], Pall[idx[free]], hm, deg)
        ix = idx[free]
        rows, cols, vals = [np.where(GD)[0]], [np.where(GD)[0]], [np.ones(GD.sum())]
        pde = np.zeros(Nt, bool)  # filas donde se coloca la EDP (para fuentes/sumideros)
        ep = ghost | ~Gn[free]  # nodos libres con fila de EDP
        L = -MU * wl + bvec[free, 0:1] * wx + bvec[free, 1:2] * wy
        rows.append(np.repeat(free[ep], n_st)); cols.append(ix[ep].ravel()); vals.append(L[ep].ravel())
        pde[free[ep]] = True
        nb = Gn[free]
        fb = free[nb]
        wb = MU * (nrm[fb, 0:1] * wx[nb] + nrm[fb, 1:2] * wy[nb])
        wb[:, 0] -= c * np.einsum("ij,ij->i", bvec[fb], nrm[fb])  # idx[i][0] == i (el propio nodo)
        rows.append(np.repeat(gid[fb], n_st)); cols.append(ix[nb].ravel()); vals.append(wb.ravel())
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(Nt, Nt))
        self.f = np.r_[GD.astype(float), np.zeros(Nt - N)]
        self.ic = np.where(Gc)[0]
        rc = gid[self.ic]
        self.B = sp.csr_matrix((np.full(len(rc), MU), (rc, np.arange(len(rc)))), shape=(Nt, len(rc)))
        wc = np.zeros(len(self.ic))  # trapecio 1D sobre Gamma_c (x1 en (1, LX], el nodo x1=1 es Dirichlet)
        for wall in (0.0, LY):
            m = np.abs(P[self.ic, 1] - wall) < TOL
            xw = np.r_[1.0, P[self.ic[m], 0]]
            o = np.argsort(xw)
            t = np.empty(len(xw)); t[o] = trap(xw[o])
            wc[m] = t[1:]
        self.Mc = sp.diags(wc).tocsr()
        self.Mc_solve = lambda g, wc=wc: g / wc
        self.P, self.W, self.N, self.Nt, self.h = P, W, N, Nt, h
        self.GD, self.Gn, self.nrm, self.pde = GD, Gn, nrm, pde
        self.gid = {int(i): int(gid[i]) for i in neu}
        self.lu = spla.splu(self.A.tocsc())  # una factorización sirve para estado y adjunto

    def pad(self, v):
        return np.r_[v, np.zeros(self.Nt - self.N)]

    def M0(self, obs):
        return sp.diags(self.pad(self.W * obs_weight(self.P, obs))).tocsr()

    def node_mass(self):
        """Peso de cada fila de EDP para un término de reacción (colocación: 1 en filas de EDP)."""
        return self.pde.astype(float)


# ---------------------------------------------------------------------------------------------------
# FEM P1 de referencia
# ---------------------------------------------------------------------------------------------------
def _edge_mass(P, nodes, N, w=None):
    """Matriz de masa 1D sobre la poligonal `nodes` (Simpson por arista, exacta para P1·P1·peso cuadrático)."""
    a, b = nodes[:-1], nodes[1:]
    L = np.linalg.norm(P[a] - P[b], axis=1)
    wq = (np.ones((len(a), 3)) if w is None else
          np.stack([w(P[a]), w(0.5 * (P[a] + P[b])), w(P[b])], 1))
    q = L[:, None] / 6 * np.array([1, 4, 1]) * wq
    ph = np.array([[1, 0.5, 0], [0, 0.5, 1]])
    m = np.einsum("eq,iq,jq->eij", q, ph, ph)
    E = np.stack([a, b], 1)
    return sp.coo_matrix((m.ravel(), (np.repeat(E, 2, 1).ravel(), np.tile(E, 2).ravel())), (N, N)).tocsr()


class FEM:
    def __init__(self, n, kind="graetz", bc="libro", factor=True, q=1.0):
        x, y = graded_axes(n, q) if q != 1.0 else (np.linspace(0, LX, 4 * n + 1), np.linspace(0, LY, n + 1))
        nx, ny = len(x) - 1, len(y) - 1
        X, Y = np.meshgrid(x, y, indexing="ij")
        P = np.c_[X.ravel(), Y.ravel()]
        I, Jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        I, Jj = I.ravel(), Jj.ravel()
        k = lambda i, j: i * (ny + 1) + j
        a, b, c_, d = k(I, Jj), k(I + 1, Jj), k(I + 1, Jj + 1), k(I, Jj + 1)
        diag = ((Jj < ny / 2) == (I % 2 == 0))[:, None]  # diagonales alternadas, simétricas en x2=0.5
        T1 = np.where(diag, np.c_[a, b, c_], np.c_[a, b, d])
        T2 = np.where(diag, np.c_[a, c_, d], np.c_[b, c_, d])
        T = np.stack([T1, T2], 1).reshape(-1, 3)
        N = len(P)
        p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        area = 0.5 * np.abs(det)
        Gr = np.stack([np.c_[p1[:, 1] - p2[:, 1], p2[:, 0] - p1[:, 0]],
                       np.c_[p2[:, 1] - p0[:, 1], p0[:, 0] - p2[:, 0]],
                       np.c_[p0[:, 1] - p1[:, 1], p1[:, 0] - p0[:, 0]]], 1) / det[:, None, None]
        K = MU * area[:, None, None] * np.einsum("tik,tjk->tij", Gr, Gr)
        C = np.zeros_like(K)
        for r, s in ((0, 1), (1, 2), (0, 2)):  # cuadratura en puntos medios (exacta en grado 2)
            phi = np.zeros(3); phi[[r, s]] = 0.5
            bm = vel(0.5 * (P[T[:, r]] + P[T[:, s]]))
            C -= (area / 3)[:, None, None] * np.einsum("tk,tik->ti", bm, Gr)[:, :, None] * phi[None, None, :]
        Me = area[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 12
        II, JJ = np.repeat(T, 3, 1).ravel(), np.tile(T, 3).ravel()
        A = sp.coo_matrix(((K + C).ravel(), (II, JJ)), (N, N)).tocsr()
        GD, Gc, Gn, _ = boundary_sets(P, kind)
        srt = lambda msk, col: np.where(msk)[0][np.argsort(P[msk, col])]
        out = srt(P[:, 0] > LX - TOL, 1)
        A = A + (1 - C_OUT[bc]) * _edge_mass(P, out, N, lambda z: 1.5 * z[:, 1] * (1 - z[:, 1]))
        if kind == "graetz":
            wall = lambda yy: (np.abs(P[:, 1] - yy) < TOL) & (P[:, 0] >= 1 - TOL)
            Mw = _edge_mass(P, srt(wall(0), 0), N) + _edge_mass(P, srt(wall(LY), 0), N)
            self.ic = np.where(Mw.diagonal() > 0)[0]
            B = MU * Mw[:, self.ic]
        else:
            self.ic = np.array([], int); B = sp.csr_matrix((N, 0))
        keep = sp.diags((~GD).astype(float))  # filas de Dirichlet: identidad
        self.A = (keep @ A + sp.diags(GD.astype(float))).tocsr()
        self.A.eliminate_zeros()
        self.B = (keep @ B).tocsr()
        self.f = GD.astype(float)
        if kind == "graetz":
            self.Mc = Mw[self.ic][:, self.ic].tocsr()
            lu_c = spla.splu(self.Mc.tocsc()); self.Mc_solve = lu_c.solve
        self.P, self.T, self.area, self.Me, self.N, self.Nt, self.GD = P, T, area, Me, N, N, GD
        self.n = n
        self.ml = np.where(GD, 0.0, np.bincount(T.ravel(), np.repeat(area / 3, 3), N))  # masa concentrada
        self.lu = spla.splu(self.A.tocsc()) if factor else None

    def pad(self, v):
        return v

    def M0(self, obs):
        chi = obs_weight(self.P[self.T].mean(1), obs) > 0.5  # retículas alineadas: el centroide decide
        Tt = self.T[chi]
        return sp.coo_matrix((self.Me[chi].ravel(), (np.repeat(Tt, 3, 1).ravel(), np.tile(Tt, 3).ravel())),
                             (self.N, self.N)).tocsr()

    def node_mass(self):
        return self.ml


# ---------------------------------------------------------------------------------------------------
# Optimización (independiente de la discretización)
# ---------------------------------------------------------------------------------------------------
class OCP:
    """Problema de control lineal-cuadrático discreto (Parte A), independiente de la discretización."""

    def __init__(self, disc, obs, zd, beta=BETA):
        self.d, self.beta = disc, beta
        self.M0 = disc.M0(obs)
        self.zd = disc.pad(target(disc.P, zd))
        self.nsolves = 0

    def state(self, u):
        self.nsolves += 1
        return self.d.lu.solve(self.d.f + self.d.B @ u)

    def J(self, u, y=None):
        y = self.state(u) if y is None else y
        r = y - self.zd
        return 0.5 * r @ (self.M0 @ r) + 0.5 * self.beta * u @ (self.d.Mc @ u)

    def grad(self, u, y=None):
        y = self.state(u) if y is None else y
        self.nsolves += 1
        p = self.d.lu.solve(self.M0 @ (y - self.zd), trans="T")
        return self.d.B.T @ p + self.beta * (self.d.Mc @ u)

    def hess(self, v):
        """H v = B^T A^{-T} M0 A^{-1} B v + beta Mc v  (1 estado + 1 adjunto)."""
        self.nsolves += 2
        w = self.d.lu.solve(self.d.B @ v)
        return self.d.B.T @ self.d.lu.solve(self.M0 @ w, trans="T") + self.beta * (self.d.Mc @ v)


def pcg(ocp, tol=1e-3, maxit=500):
    """CG precondicionado con Mc (producto L2(Gamma_c)); tol sobre ‖g_k‖_{Mc^-1}/‖g_0‖_{Mc^-1}."""
    u = np.zeros(len(ocp.d.ic))
    g = ocp.grad(u); z = ocp.d.Mc_solve(g); d = -z
    gz = g @ z; n0 = np.sqrt(gz); hist = [ocp.J(u)]; gn = [1.0]
    for k in range(maxit):
        q = ocp.hess(d)
        a = gz / (d @ q)
        u = u + a * d
        g = g + a * q
        z = ocp.d.Mc_solve(g)
        gz_new = g @ z
        hist.append(ocp.J(u)); gn.append(np.sqrt(gz_new) / n0)
        if gn[-1] < tol:
            break
        d = -z + (gz_new / gz) * d
        gz = gz_new
    return u, np.array(hist), np.array(gn)


def transfer_control(src, u, dst):
    """Lleva un control nodal de Gamma_c de `src` a `dst` por interpolación lineal en x1 (pared a pared)."""
    v = np.zeros(len(dst.ic))
    for wall in (0.0, LY):
        ms = np.abs(src.P[src.ic, 1] - wall) < TOL
        md = np.abs(dst.P[dst.ic, 1] - wall) < TOL
        xs = src.P[src.ic, 0][ms]; o = np.argsort(xs)
        v[md] = np.interp(dst.P[dst.ic, 0][md], xs[o], u[ms][o])
    return v


def l2_gamma(disc, v):
    return np.sqrt(v @ (disc.Mc @ v))


# ---------------------------------------------------------------------------------------------------
# Parte B: pozos (control bilineal con cotas)
# ---------------------------------------------------------------------------------------------------
class Wells:
    """Problema de pozos (Parte B), independiente de la discretización (RBF-FD o FEM, kind='canal')."""

    def __init__(self, disc, xw=XW, sigma=0.08, beta=1e-4, umax=5.0):
        self.d, self.beta, self.xw, self.umax = disc, beta, np.array(xw), umax
        psi = np.array([np.exp(-((disc.P - c) ** 2).sum(1) / (2 * sigma**2)) / (2 * np.pi * sigma**2) for c in xw])
        self.S = np.array([disc.pad(disc.node_mass()[: disc.N] * p) for p in psi])  # diagonales S_i
        self.Mobs = disc.M0("pozos")
        self.nsolves = 0

    def _lu(self, u):
        return spla.splu((self.d.A + sp.diags(u @ self.S)).tocsc())

    def J(self, u):
        y = self._lu(u).solve(self.d.f); self.nsolves += 1
        return 0.5 * y @ (self.Mobs @ y) + 0.5 * self.beta * u @ u

    def Jg(self, u):
        lu = self._lu(u); y = lu.solve(self.d.f)
        p = lu.solve(self.Mobs @ y, trans="T"); self.nsolves += 2
        J = 0.5 * y @ (self.Mobs @ y) + 0.5 * self.beta * u @ u
        return J, -(self.S * y) @ p + self.beta * u, y


def wolfe_secant(prob, u, J, g, d, a, sigma=1e-4, c2=0.1, maxls=12):
    """Búsqueda lineal (Wolfe fuerte) por secante sobre phi'(a) = g(u+a d)·d, truncada en la cota."""
    umax = prob.umax
    proj = lambda v: np.clip(v, 0.0, umax)
    pos, neg = d > 0, d < 0
    amax = min(np.min((umax - u[pos]) / d[pos], initial=np.inf), np.min(-u[neg] / d[neg], initial=np.inf))
    dphi0 = g @ d
    lo, dlo, hi, dhi = 0.0, dphi0, None, None
    ap, dp = 0.0, dphi0  # punto evaluado anterior (para la secante)
    a = min(a, amax)
    for _ in range(maxls):
        un = proj(u + a * d)
        Jn, gn, _ = prob.Jg(un)
        dphi = gn @ d
        armijo = Jn <= J + sigma * a * dphi0
        if armijo and abs(dphi) <= c2 * abs(dphi0):
            break
        if armijo and dphi < 0:
            if a >= amax:
                break  # tocamos la cota con descenso: aceptar
            lo, dlo = a, dphi
        else:
            hi, dhi = a, dphi
        if hi is None:  # extrapolar con secante entre el punto anterior y el actual, acotado
            a_sec = a - dphi * (a - ap) / (dphi - dp) if dphi > dp else 2 * a
            ap, dp = a, dphi
            a = min(max(a_sec, 1.5 * a), 10 * a, amax)
        else:  # interpolar dentro de [lo, hi]
            a_sec = lo - dlo * (hi - lo) / (dhi - dlo) if dhi > dlo else 0.5 * (lo + hi)
            a = np.clip(a_sec, lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
    return un, Jn, gn, a, a >= amax


def ncg_proj(prob, u0, tol=1e-6, maxit=100):
    """CG no lineal Polak–Ribière+ con conjunto activo de cotas (CG proyectado)."""
    umax = prob.umax
    proj = lambda v: np.clip(v, 0.0, umax)
    active = lambda v, g: ((v <= 0) & (g > 0)) | ((v >= umax) & (g < 0))
    u = proj(u0.copy())
    J, g, _ = prob.Jg(u)
    gr = np.where(active(u, g), 0.0, g)  # gradiente reducido
    d = -gr
    hist = [J]; pg0 = np.linalg.norm(proj(u - g) - u)
    a = 1.0 / np.abs(g).max()  # primer paso: mover ~1 unidad la tasa más sensible
    for k in range(maxit):
        un, Jn, gn, a_acc, hit = wolfe_secant(prob, u, J, g, d, a)
        grn = np.where(active(un, gn), 0.0, gn)
        if hit:  # cambió el conjunto activo -> reinicio en dirección de descenso
            d_new = -grn
        else:
            bpr = max(0.0, grn @ (grn - gr) / (gr @ gr))
            d_new = -grn + bpr * d
            d_new[active(un, gn)] = 0.0
            if grn @ d_new >= 0:
                d_new = -grn
        # paso inicial siguiente (Nocedal–Wright 3.60)
        a = a_acc * (gr @ d) / (grn @ d_new) if grn @ d_new < 0 else 1.0
        u, J, g, gr, d = un, Jn, gn, grn, d_new
        hist.append(J)
        if np.linalg.norm(proj(u - g) - u) / pg0 < tol:
            break
    return u, np.array(hist)
