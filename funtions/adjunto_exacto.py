"""
Adjunto discreto EXACTO del simulador BFR (RBF-FD), caso ADR.

Es el gradiente exacto (a precisión de máquina) del funcional discreto que evalúa `view.animation.functional`,
obtenido derivando el esquema tal como está programado, sin aproximaciones continuas ni factores ajustados:

    Lagrangiano  L = J_h + Σ_ℓ [ Λ_C^ℓ·r_C^ℓ + Λ_h^ℓ·r_h^ℓ ],        ℓ = 1..N_t  (nivel de tiempo al que llega el paso)

    r_h^ℓ = A_L H^ℓ − E(M_R H^{ℓ-1} − [(1-θ) q_viejo + θ q_nuevo] δ) − b_bc          (flujo, `step_time`)
    r_C^ℓ = A_C(H^ℓ,H^{ℓ-1}) C^ℓ − B_C(H^ℓ,H^{ℓ-1}) C^{ℓ-1} − F                       (transporte)

Cada fila del transporte es  Σ_k c_k,i (O_k C)_i  con O_k los estenciles RBF-FD fijos (identidad, ∂xx, ∂zz, ∂x, ∂z)
y c_k,i coeficientes puntuales que dependen de h a través de U = −K∇h, D(U), ∇·D y ∂ₜh. Por eso la derivada
respecto de h es una regla de la cadena exacta (ver `_vjp_transporte`), que se comprueba contra diferencias
finitas del constructor real en tests/test_adjunto_exacto.py.

Convención: se resuelven Λ (signo del documento docs/adjunto_sin_sumidero.tex: L = J + Λ·r) y

    A_C^ℓᵀ Λ_C^ℓ = M^ℓ ( B_C^{ℓ+1 ᵀ} Λ_C^{ℓ+1} − ∇j_ℓ )            M = máscara del recorte C ≥ 0
    A_Lᵀ  Λ_h^ℓ = M_Rᵀ E Λ_h^{ℓ+1} − σ^ℓ ,  σ^ℓ = ∂/∂H^ℓ Σ Λ_C·r_C
    ∂J/∂Q_k = θ Λ_h^{k+1}·Eδ + (1-θ) Λ_h^{ℓ'}·Eδ + 2|z_p−z_0| Q_k Δt,     ℓ' = (k+1) mod N_t + 1

Alcance: ADR, MRMT por bloque y MRMT semi-analítico; transporte centrado (sin upwind) y sin el sumidero W₋
(el modelo no lo incluye).
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows

from funtions.runtime import RUNTIME
from funtions.operators import coeficientes_transporte, peso_theta_C, factor_avance

DIFFS = [[0, 0], [2, 0], [0, 2], [1, 0], [0, 1]]


class OperadoresTransporte:
    """Estenciles fijos O_k del transporte y de sus filas de frontera (misma construcción que build_matrix)."""

    def __init__(self, d):
        p = RUNTIME.get()
        g = d.groups
        self.N = len(d.nodes)
        self.idx = np.hstack([g['interior'], g['boundary:inlet'], g['boundary:outlet'], g['boundary:wall']])
        eps = np.asarray(d.eps_M)
        self.O = [csr_matrix(weight_matrix(x=d.nodes[self.idx], p=d.nodes, n=p.n_stencil, diffs=[df],
                                           phi=p.phi, order=p.order, eps=eps[self.idx], chunk_size=None))
                  for df in DIFFS]
        self.bc = []
        for tag in ("inlet", "wall", "outlet"):            # mismo orden que make_transport_bc
            ib = g.get(f'boundary:{tag}', None)
            if ib is None or len(ib) == 0:
                continue
            gh = g.get(f'ghosts:{tag}', ib)
            Ox, Oz = [csr_matrix(weight_matrix(x=d.nodes[ib], p=d.nodes, n=p.n_stencil, diffs=[df], phi=p.phi,
                                               order=p.order, eps=eps[ib], chunk_size=None))
                      for df in ([1, 0], [0, 1])]
            self.bc.append((np.asarray(ib), np.asarray(gh), Ox, Oz))
        self.normals = d.normals

    def matriz(self, coefs, D, con_bc):
        """A (con_bc) o B (sin filas de frontera) a partir de los coeficientes puntuales."""
        A = None
        for c, O in zip(coefs, self.O):
            blk = expand_rows(diags(c[self.idx]) @ O, self.idx, self.N)
            A = blk if A is None else A + blk
        if con_bc:
            for ib, gh, Ox, Oz in self.bc:
                blk = diags(-D[ib, 0] * self.normals[ib, 0]) @ Ox + diags(-D[ib, 1] * self.normals[ib, 1]) @ Oz
                A = A + expand_rows(csr_matrix(blk), gh, self.N)
        return A.tocsc()


# ---------------------------------------------------------------------------------------------
# D(U) y su derivada
# ---------------------------------------------------------------------------------------------
def _dD_dU(p, U):
    """Jacobiana puntual de D_total respecto de U (se supone D_total(Dif, U) = Dif + disp(U))."""
    Ux, Uz = U[:, 0], U[:, 1]
    r = np.sqrt(Ux ** 2 + Uz ** 2)
    activo = r > 0
    r = np.where(activo, r, 1e-12)
    a_l, a_t = p.a_l, p.a_t
    qx = a_l * Ux ** 2 + a_t * Uz ** 2
    qz = a_t * Ux ** 2 + a_l * Uz ** 2
    # con |U| fijado en 1e-12 (U = 0) la norma es constante: sin término de |U|
    k = np.where(activo, 1.0, 0.0)
    dDx_dUx = 2 * a_l * Ux / r - k * qx * Ux / r ** 3
    dDx_dUz = 2 * a_t * Uz / r - k * qx * Uz / r ** 3
    dDz_dUx = 2 * a_t * Ux / r - k * qz * Ux / r ** 3
    dDz_dUz = 2 * a_l * Uz / r - k * qz * Uz / r ** 3
    return dDx_dUx, dDx_dUz, dDz_dUx, dDz_dUz


def _pull_D_a_U(p, U, Dbar):
    a, b, c, e = _dD_dU(p, U)
    return np.column_stack([Dbar[:, 0] * a + Dbar[:, 1] * c, Dbar[:, 0] * b + Dbar[:, 1] * e])


def _pull_U_a_H(d, p, Ubar):
    """U = −K ∘ (G H):  H̄ = −Gxᵀ(Kx Ūx) − Gzᵀ(Kz Ūz)."""
    return -(d.grad[0].T @ (d.K[:, 0] * Ubar[:, 0]) + d.grad[1].T @ (d.K[:, 1] * Ubar[:, 1]))


# ---------------------------------------------------------------------------------------------
# VJP del residuo de transporte respecto de (H^ℓ, H^{ℓ-1})
# ---------------------------------------------------------------------------------------------
def _vjp_transporte(d, p, ops, pho, lam, C_new, C_old, U_new, U_old, D_new, D_old, h_t, nivel_viejo_es_dato):
    """
    Derivada de  lam·(A C_new) − lam·(B C_old)  respecto de H^ℓ y H^{ℓ-1}  (A, B como en step_time):
      A: sig=-1, coeficientes con (V=U_new, D_new, ∇·D_new, h_t) y filas de frontera con D_new
      B: sig=+1, coeficientes con (V=U_new, D_old, ∇·D_old, h_t)
    Devuelve (H̄_new, H̄_old).
    """
    N = ops.N
    idx = ops.idx
    Ob = ops.O
    Gx, Gz = d.grad[0], d.grad[1]
    a = factor_avance()
    f_alm = getattr(p, "factor_almacenamiento_C", 0.0)

    # g_k = ∂(lam·A C_new)/∂c_k  (A)   y   −∂(lam·B C_old)/∂c_k  (B)  en las filas del dominio
    gA = [np.zeros(N) for _ in range(5)]
    gB = [np.zeros(N) for _ in range(5)]
    for k in range(5):
        gA[k][idx] = lam[idx] * (Ob[k] @ C_new)
        gB[k][idx] = -lam[idx] * (Ob[k] @ C_old)

    wA, sA = peso_theta_C(p, -1), -1.0
    wB, sB = peso_theta_C(p, +1), +1.0

    # ∂ₜh
    ht_bar = (gA[0] * (wA * sA) + gB[0] * (wB * sB)) * (f_alm * pho)

    # velocidad V = U_new (en ambos operadores)
    V_bar = np.column_stack([
        gA[3] * (-wA * sA * a) + gB[3] * (-wB * sB * a),
        gA[4] * (-wA * sA * a) + gB[4] * (-wB * sB * a),
    ])

    def D_bar(g, w, s):
        Db = np.column_stack([g[1] * (w * s), g[2] * (w * s)])
        # c_x = −w s a (V_x − (G_x D_x)),  c_z = −w s a (V_z − (G_z D_z))
        Db[:, 0] += Gx.T @ (g[3] * (w * s * a))
        Db[:, 1] += Gz.T @ (g[4] * (w * s * a))
        return Db

    DbA = D_bar(gA, wA, sA)
    DbB = D_bar(gB, wB, sB)

    # filas de frontera de A (ghosts): c = −D n  sobre [∂x, ∂z]
    for ib, gh, Ox, Oz in ops.bc:
        DbA[ib, 0] += -ops.normals[ib, 0] * lam[gh] * (Ox @ C_new)
        DbA[ib, 1] += -ops.normals[ib, 1] * lam[gh] * (Oz @ C_new)

    U_bar_new = V_bar + _pull_D_a_U(p, U_new, DbA)
    H_new = _pull_U_a_H(d, p, U_bar_new) + ht_bar / p.dt
    H_old = -ht_bar / p.dt
    if not nivel_viejo_es_dato:
        H_old = H_old + _pull_U_a_H(d, p, _pull_D_a_U(p, U_old, DbB))
    return H_new, H_old


# ---------------------------------------------------------------------------------------------
# Derivada de la delta del pozo respecto de z_p (Wendland C2 normalizada, como discrete_delta)
# ---------------------------------------------------------------------------------------------
def ddelta_dzp(d, p):
    x = d.nodes
    R = p.radio_delta_pozo
    dx, dz = x[:, 0] - d.pozo_cor[0], x[:, 1] - d.pozo_cor[1]
    r = np.hypot(dx, dz)
    q = r / R
    m = q < 1.0
    W = np.zeros(len(x)); dW = np.zeros(len(x))
    W[m] = (1 - q[m]) ** 4 * (4 * q[m] + 1)
    rp = np.where(r > 0, r, 1.0)
    dq_dz = np.where(r > 0, -dz / rp / R, 0.0)
    dW[m] = -20.0 * q[m] * (1 - q[m]) ** 3 * dq_dz[m]
    if p.norm_delta_pozo == "area":
        wi = d.wi
        S, dS = np.sum(W * wi), np.sum(dW * wi)
        Wn = W / S
        # delta = (W/ΣW)/Σ((W/ΣW) wi) = W/Σ(W wi)
        return dW / S - W * dS / S ** 2, Wn
    S, dS = np.sum(W), np.sum(dW)
    return dW / S - W * dS / S ** 2, W / S


# ---------------------------------------------------------------------------------------------
# Gradiente exacto
# ---------------------------------------------------------------------------------------------
def gradiente_exacto(d, p, Q, zp, H, U, C, Ci=None, ops=None, debug=None):
    """
    H, U, C: historias del forward (niveles 1..N_t, forma (N_t, K, N[,2])) tal como las guarda la simulación;
    los niveles 0 son d.H[0], d.U[0], d.C[0]. Devuelve (gQ, gzp) del funcional de `compute_functional`.
    """
    if p.model not in ("adr", "mrmt_block", "mrmt_semi"):
        raise NotImplementedError(f"adjunto exacto: modelo {p.model}")
    if p.model == "mrmt_block" and Ci is None:
        raise ValueError("MRMT por bloque: se necesita la historia C_im ('Ci' del npz)")
    if getattr(p, "upwind_transport", False):
        raise NotImplementedError("adjunto exacto: transporte upwind no implementado")

    Nt = len(Q)
    N = len(d.nodes)
    th = p.theta
    dt = p.dt
    pho = d.pho
    ops = ops or OperadoresTransporte(d)
    idx = ops.idx

    Hn = [np.asarray(d.H[0], float)] + [np.asarray(H[s, 0], float) for s in range(Nt)]
    Un = [np.asarray(d.U[0], float)] + [np.asarray(U[s, 0], float) for s in range(Nt)]
    Cn = [np.asarray(d.C[0], float)] + [np.asarray(C[s, 0], float) for s in range(Nt)]
    # D en cada nivel y ∇·D (los usa step_time: D_n del nivel viejo, D_N del nuevo)
    Dn = [D_total_(d, p, Un[l]) for l in range(Nt + 1)]
    ht = [None] + [(Hn[l] - Hn[l - 1]) / dt for l in range(1, Nt + 1)]

    wi, gp = d.wi, d.gauss_p
    # --- adjunto del transporte (hacia atrás) y fuentes σ del flujo -------------------------------
    # Se propagan COTANGENTES del esquema (modo reverso exacto); Λ_C^ℓ = −A_ℓ^{-T} (cotangente de C_sol^ℓ).
    #   ADR      : x^ℓ = M A^{-1} B x^{ℓ-1}
    #   bloque   : x^ℓ = M A^{-1}(B x^{ℓ-1} + Σ_r Q_r y_r^{ℓ-1}),  y_r^ℓ = 2c_r(θ_C x^ℓ + (1-θ_C) x^{ℓ-1}) + (1-2c_r) y_r^{ℓ-1}
    #   semi     : C* = A^{-1} B x^{ℓ-1};  C̃_r = (1-a_r) C̃_{r-1} + a_r y_r^{ℓ-1};  y_r^ℓ = (1-e_r) C̃_{r-1} + e_r y_r^{ℓ-1};  x^ℓ = M C̃_{Nr}
    LamC = [None] * (Nt + 2)
    LamC[Nt + 1] = np.zeros(N)
    sigma = [np.zeros(N) for _ in range(Nt + 2)]
    from scipy.sparse.linalg import splu

    Nr = p.Nr if p.model != "adr" else 0
    E = np.zeros(N); E[idx] = 1.0
    thc = float(getattr(p, "theta_C", 0.5))
    if Nr:
        beta = np.asarray(p.beta, float)
    if p.model == "mrmt_block":
        coef_r = p.dt * np.asarray(p.alpha_r, float) / beta
        Qr = [2.0 * p.R * p.alpha_r[r] * pho * E for r in range(Nr)]
        y = [np.asarray(d.C_im[0][r], float) for r in range(Nr)]
        ylev = [[np.asarray(d.C_im[0][r], float) for r in range(Nr)]] + \
               [[np.asarray(Ci[r, s, 0], float) for r in range(Nr)] for s in range(Nt)]
    if p.model == "mrmt_semi":
        e_r = np.asarray(d.exp_lam_dt, float)
        a_r = beta * (1.0 - e_r)
    ybar = [np.zeros(N) for _ in range(Nr)]       # cotangente de y_r^{ℓ} (bloque) / de C_im,r^{ℓ} (semi)
    B_next = None
    for l in range(Nt, 0, -1):
        coefA, coefB = _coefs(d, p, pho, Un[l], Dn[l], Dn[l - 1], ht[l])
        A = ops.matriz(coefA, Dn[l], True)
        B = ops.matriz(coefB, Dn[l - 1], False)
        # ∇j_ℓ = 2γ Δt w_i g_i C^ℓ   (J_1 = γ Δt Σ_ℓ Σ_i w_i g_i (C_i^ℓ)²)
        xbar = 2.0 * p.gamma * dt * wi * gp * Cn[l]
        if B_next is not None:
            xbar = xbar - B_next.T @ LamC[l + 1]
        mask = (Cn[l] > 0.0) if getattr(p, "clip_C", True) else np.ones(N, bool)
        extra = np.zeros(N)
        if p.model == "mrmt_block":
            ynew = []
            for r in range(Nr):
                yb = (1.0 - 2.0 * coef_r[r]) * ybar[r] * E - Qr[r] * LamC[l + 1]
                # y^{ℓ+1} usa x^ℓ guardado (recortado): pasa por la máscara. y^ℓ usa C_sol^ℓ SIN recortar
                # (step_time actualiza C_im antes del recorte): entra directo en la cotangente de C_sol.
                xbar = xbar + 2.0 * coef_r[r] * E * (1.0 - thc) * ybar[r]
                extra = extra + 2.0 * coef_r[r] * E * thc * yb
                ynew.append(yb)
            ybar = ynew
        ct = np.where(mask, xbar, 0.0) + extra
        if p.model == "mrmt_semi":
            nueva = [None] * Nr
            for r in range(Nr - 1, -1, -1):
                nueva[r] = a_r[r] * ct + e_r[r] * ybar[r]
                ct = (1.0 - a_r[r]) * ct + (1.0 - e_r[r]) * ybar[r]
            ybar = nueva
        LamC[l] = -splu(A.T.tocsc()).solve(ct)
        B_next = B
        # El residuo r_C^ℓ = A C_sol − B x^{ℓ-1} (− Σ Q y) se evalúa con la solución SIN recortar:
        # x^ℓ guardado = max(C_sol, 0) (step_time). Se recupera resolviendo de nuevo el paso.
        b = np.zeros(N)
        b[idx] = (B @ Cn[l - 1])[idx]
        if p.model == "mrmt_block":
            for r in range(Nr):
                b[idx] += (Qr[r] * ylev[l - 1][r])[idx]
        C_sol = splu(A.tocsc()).solve(b)
        # σ: derivada de Λ_C^ℓ·r_C^ℓ respecto de (H^ℓ, H^{ℓ-1})
        Hnew, Hold = _vjp_transporte(d, p, ops, pho, LamC[l], C_sol, Cn[l - 1], Un[l], Un[l - 1], Dn[l], Dn[l - 1],
                                     ht[l], nivel_viejo_es_dato=(l == 1))
        sigma[l] += Hnew
        sigma[l - 1] += Hold

    # --- adjunto del flujo ----------------------------------------------------------------------
    MR_T = d.A_right.T.tocsr()
    LamH = [None] * (Nt + 2)
    LamH[Nt + 1] = np.zeros(N)
    for l in range(Nt, 0, -1):
        rhs = MR_T @ (E * LamH[l + 1]) - sigma[l]
        LamH[l] = d.A_left.solve(rhs, trans='T')

    # --- gradientes -------------------------------------------------------------------------------
    dlt = d.delta_p
    Ed = E * dlt
    zdist = abs(zp - d.z0)
    gQ = np.zeros(Nt)
    if p.activate_ext:
        for k in range(Nt):
            gQ[k] += th * (LamH[k + 1] @ Ed)
            lprima = (k + 1) % Nt + 1
            gQ[k] += (1.0 - th) * (LamH[lprima] @ Ed)
    gQ += 2.0 * zdist * np.asarray(Q) * dt

    # z_p: kernel de J_1, costo económico y desplazamiento de la delta del flujo
    dgdz = d.gauss_p * (d.nodes[:, 1] - d.pozo_cor[1]) / (p.epsilon_pozo ** 2)
    g_zp = p.gamma * dt * sum(np.sum(wi * dgdz * Cn[l] ** 2) for l in range(1, Nt + 1))
    g_zp += 2.0 * p.koppa * (zp - d.z0)
    g_zp += np.sign(zp - d.z0) * np.sum(np.asarray(Q) ** 2) * dt
    if p.activate_ext:
        dd, _ = ddelta_dzp(d, p)
        for s in range(Nt):
            qn = Q[s]
            qv = Q[(s - 1) % Nt]
            g_zp += (LamH[s + 1] @ (E * dd)) * ((1.0 - th) * qv + th * qn)
    if debug is not None:
        debug.update(sigma=sigma, LamC=LamC, LamH=LamH, Hn=Hn, Un=Un, Cn=Cn)
    return gQ, g_zp


def D_total_(d, p, U):
    from funtions.utils import D_total
    return D_total(d.D_f, U)


def _coefs(d, p, pho, U_new, D_new, D_old, h_t):
    """Coeficientes (cI, cxx, czz, cx, cz) de A y de B, exactamente como step_time."""
    from funtions.utils import Div_KD
    divD_new = Div_KD(D_new, d.grad)
    divD_old = Div_KD(D_old, d.grad)
    cA = coeficientes_transporte(p, pho, U_new, D_new, divD_new, -1, h_t)
    cB = coeficientes_transporte(p, pho, U_new, D_old, divD_old, +1, h_t)
    return cA, cB
