"""Banco 1D, complemento de runp1.py: (i) gradiente respecto de la posición del pozo x_p (análogo 1D de z_p) con el adjunto
de las fórmulas del documento; (ii) equivalencia entre la forma conservativa y la no conservativa del transporte (Prop. ADR)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from p1 import *


def ctrl(c):
    t = np.linspace(0, c['T'], c['nt'] + 1); return 3.0 * (1 + 0.5 * np.sin(2 * np.pi * t / c['T'] * 1.5)), np.cos(2 * np.pi * t / c['T'] * 2) + 0.5


def grad_xp(c, Q):
    J0, s = J_of(c, Q); _, a = adjoint(c, Q, s)
    x, m, dt, nt, g = s['x'], s['m'], s['dt'], c['nt'], s['g']
    gd = g * (x - c['xp']) / c['eps'] ** 2                      # ∂g/∂x_p
    w = np.full(nt + 1, dt); w[0] = w[-1] = dt / 2
    t1 = c['gamma'] * sum(w[k] * np.sum(m * gd * s['C'][k] ** 2) for k in range(nt + 1))
    t2 = sum(Q[k] * dt * np.sum(m * gd * a['psih'][k]) for k in range(1, nt + 1))
    return t1 + t2, t1


def forward_cons(c, Q):
    """Mismo esquema que p1.forward, pero con el transporte en forma conservativa  φR C_t + (uC - D C')' + c0 C = W + f C,
    f = -Q g (sin el término de almacenamiento: aparece solo a través de la divergencia discreta de u)."""
    x, dx, m = mesh(c); n = c['n']; N = n + 1; nt = c['nt']; dt = c['T'] / nt
    g = gk(x, c['xp'], c['eps']); h = np.zeros((nt + 1, N)); C = np.zeros((nt + 1, N))
    C[0] = np.exp(-(x - 0.7) ** 2 / (2 * 0.08 ** 2))
    Kmat = stiff(np.full(n, c['K']), dx, n); Mh = sp.diags(c['S'] / dt * m, 0, format='csc'); i = np.arange(n)
    for k in range(nt):
        A = dirich(Mh + Kmat); rhs = c['S'] / dt * m * h[k] - Q[k + 1] * g * m; rhs[0] = rhs[-1] = 0
        h[k + 1] = sla.spsolve(A, rhs); u = elem_u(h[k + 1], c, dx); D_e = c['D0'] + c['aL'] * np.abs(u)
        # -∫ u C v' por elemento (u constante, C P1): fila i: +u (C_i + C_{i+1})/2, fila i+1: -u (C_i + C_{i+1})/2
        r = np.concatenate([i, i, i + 1, i + 1]); cc = np.concatenate([i, i + 1, i, i + 1])
        v = np.concatenate([u / 2, u / 2, -u / 2, -u / 2]); Adv = sp.csc_matrix((v, (r, cc)), shape=(N, N)).tolil()
        Adv[0, 0] += -u[0]; Adv[n, n] += u[-1]                    # término de frontera [u C v] (flujo advectivo libre)
        M = sp.diags(c['phi'] * c['R'] / dt * m, 0) + stiff(D_e, dx, n) + Adv.tocsc() \
            + sp.diags(m * (c['lam0'] + Q[k + 1] * g), 0)          # -f C = +Q g C
        C[k + 1] = sla.spsolve(M.tocsc(), c['phi'] * c['R'] / dt * m * C[k])
    return C


if __name__ == "__main__":
    print("(i) gradiente respecto de x_p (S = 1)")
    for nr in (0, 2):
        for n in (100, 200, 400):
            c = cfg(S=1.0, n=n, nt=n, Nr=nr); Q, _ = ctrl(c); ad, solo_Jm = grad_xp(c, Q); errs = []
            for e in (1e-3, 1e-4, 1e-5):                                 # barrido del paso de diferencias finitas
                fd = (J_of(c, Q, c['xp'] + e)[0] - J_of(c, Q, c['xp'] - e)[0]) / (2 * e); errs.append(abs(ad - fd) / abs(fd))
            print(f"  {'MRMT Nr=2' if nr else 'ADR      '} n={n:4d} adj={ad:+.5e}  err(e=1e-3,1e-4,1e-5)=" + ", ".join(f"{x:.2e}" for x in errs)
                  + f";  control negativo (sin el término de ψ_h): {abs(solo_Jm - fd) / abs(fd):.2e}")
    print("(ii) forma conservativa vs no conservativa (ADR, S = 1): ||C(T)_cons - C(T)_nc|| / ||C(T)_nc||")
    for n in (100, 200, 400):
        c = cfg(S=1.0, n=n, nt=n); Q, _ = ctrl(c); s = forward(c, Q); Cc = forward_cons(c, Q)
        print(f"  n={n:4d} err_rel={np.linalg.norm(Cc[-1] - s['C'][-1]) / np.linalg.norm(s['C'][-1]):.2e}")
