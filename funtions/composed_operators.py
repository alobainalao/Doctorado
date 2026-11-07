"""Operadores con la difusión SIN desarrollar (diffusion_form="composed").

En vez de K∂²+∇K·∇ (un solo stencil de 2.º orden), ∇·(K∇·) se arma componiendo
las matrices RBF-FD de 1.er orden:  Gx·diag(Kx)·Gx + Gz·diag(Kz)·Gz.  Así no aparece
∇·K (ni ∇·D) y toda la matriz se construye sólo con derivadas primeras; las
condiciones de frontera ya son de 1.er orden (Neumann K∇H·n, D∇C·n).
La advección entra completa: diag(V)·G (sin el 1/2 de make_transport_operator).
"""
import numpy as np
from scipy.sparse import diags

_CACHE = {}


def grad_matrices(nodes, eps_M):
    """Gx, Gz de 1.er orden sobre todos los nodos (los mismos de build_operators)."""
    key = (id(nodes), id(eps_M), len(nodes))
    if key not in _CACHE:
        from funtions.operators import build_operators
        _CACHE.clear()
        _CACHE[key] = build_operators(nodes, eps_M)
    return _CACHE[key]


def _dominio(groups):
    return np.hstack([groups['interior'], groups['boundary:inlet'],
                      groups['boundary:outlet'], groups['boundary:wall']])


def build_H_matrix_composed(nodes, groups, normals, theta, S_s, K, Div_K,
                            eps_M, sig, p):
    """S_s/Δt·I + θ·sig·∇·(K∇·) con ∇·(K∇·) = Gx Kx Gx + Gz Kz Gz, más BC de flujo."""
    from funtions.operators import apply_boundary_conditions, make_H_bc
    modo = getattr(p, "bc_outlet_mode", "pfs")
    if modo not in ("pfs", "pfs_nn", "libre", "fem", "dirichlet"):
        raise NotImplementedError(f"diffusion_form='composed': bc_outlet_mode '{modo}' no soportado")
    N = len(nodes)
    Gx, Gz = grad_matrices(nodes, eps_M)
    idx = _dominio(groups)
    L = Gx @ diags(K[:, 0]) @ Gx + Gz @ diags(K[:, 1]) @ Gz
    A = (diags(S_s / p.dt) + theta * sig * L).tocsr()[idx]
    blocks = [(A, idx)]
    bc = None
    if sig < 0:
        bc = make_H_bc(nodes, groups, K, Div_K, normals)
        if modo == "pfs":
            # La BC de radiación de 2.º orden (n·∇)(Kx∂ₓH + Kz∂zH) = 0 tampoco se
            # desarrolla: (diag(nx)Gx + diag(nz)Gz)·(diag(Kx)Gx + diag(Kz)Gz), evaluada
            # en los nodos de borde y colocada en las filas de los fantasmas (como bc_block).
            bc = [b for b in bc if b["name"] != "outlet"]
            b_idx = np.asarray(groups.get("boundary:outlet", []), dtype=int)
            if len(b_idx):
                M = diags(K[:, 0]) @ Gx + diags(K[:, 1]) @ Gz          # s = Kx Hx + Kz Hz
                Dn = diags(normals[:, 0]) @ Gx + diags(normals[:, 1]) @ Gz
                R = -(Dn @ M).tocsr()[b_idx]                           # mismo signo que 'pfs' expandido
                blocks.append((R, np.asarray(groups.get("ghosts:outlet", b_idx), dtype=int)))
        elif modo in ("pfs_nn", "libre"):
            # ∂ₙ(u·n) = 0 con n congelada en el nodo de borde i:
            #   nx² Gx·Kx·Gx + nx nz (Gx·Kz·Gz + Gz·Kx·Gx) + nz² Gz·Kz·Gz
            # (mismos coeficientes de 2.º orden que el modo desarrollado, sin ∇K explícito).
            bc = [b for b in bc if b["name"] != "outlet"]
            b_idx = np.asarray(groups.get("boundary:outlet", []), dtype=int)
            if len(b_idx):
                Kx, Kz = diags(K[:, 0]), diags(K[:, 1])
                A1, A2 = (Gx @ Kx @ Gx)[b_idx], (Gx @ Kz @ Gz)[b_idx]
                A3, A4 = (Gz @ Kx @ Gx)[b_idx], (Gz @ Kz @ Gz)[b_idx]
                nx, nz = normals[b_idx, 0][:, None], normals[b_idx, 1][:, None]
                R = -(A1.multiply(nx ** 2) + (A2 + A3).multiply(nx * nz) + A4.multiply(nz ** 2)).tocsr()
                blocks.append((R, np.asarray(groups.get("ghosts:outlet", b_idx), dtype=int)))
    A_full = apply_boundary_conditions(blocks, bc, nodes, groups, p, eps_M, N)
    if modo == "libre":
        # fila del nodo de borde = ∂ₜH = ∂τ(Kτ ∂τ H) en 3 puntos (ya es flujo de primeras diferencias)
        from funtions.operators import ecuacion_borde_libre
        A_full = ecuacion_borde_libre(A_full, nodes, groups, normals, theta, S_s, K, sig, p.dt)
    return A_full


def build_transport_matrix_composed(V, nodes, groups, normals, pho, D, Qout,
                                    eps_M, gauss_p, sig, p, grad):
    """φR∂ₜC = ∇·(D∇C) − q·∇C − λφRC  (formas B y C; la B suma Ss∂ₜh·C aparte,
    con add_storage_term).  Escala CN idéntica a build_transport_matrix."""
    from funtions.operators import (apply_boundary_conditions, make_transport_bc,
                                    peso_theta_C)
    if getattr(p, "upwind_transport", False):
        raise NotImplementedError("diffusion_form='composed' no soporta upwind_transport")
    if p.model != "adr":
        raise NotImplementedError("diffusion_form='composed' sólo para model='adr'")
    N = len(nodes)
    w = peso_theta_C(p, sig)
    idx = _dominio(groups)
    Gx, Gz = grad
    L = (diags(V[:, 0]) @ Gx + diags(V[:, 1]) @ Gz
         - Gx @ diags(D[:, 0]) @ Gx - Gz @ diags(D[:, 1]) @ Gz)
    base = pho * p.R
    loc = base * p.landa
    if getattr(p, "extract_C", True) and p.activate_ext and Qout and p.run_type != "optimization":
        loc = loc + p.factor_sumidero_C * np.asarray(gauss_p, float)   # = sumidero W- del estándar
    A = (diags(base * (2. / p.dt)) + w * (-sig) * (L + diags(loc))).tocsr()[idx]
    bc = None if sig > 0 else make_transport_bc(normals, D)
    return apply_boundary_conditions([(A, idx)], bc, nodes, groups, p, eps_M, N)
