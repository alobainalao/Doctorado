"""
Operadores de gradiente con stencil direccional (upwinding) para RBF-FD.

Para cada nodo i, el gradiente usa solo vecinos aguas arriba de V[i].
Esto elimina oscilaciones de Gibbs en la discretización del término
advectivo sin añadir difusión artificial global.

Referencia: Shu et al. (2003), "An upwind local RBF-DQ method for
simulation of inviscid compressible flows", Computer Methods in Applied
Mechanics and Engineering.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import KDTree
from rbf.pde.fd import weight_matrix


def build_directional_gradient(nodes, V, eps_M, n, phi, order,
                                oversample=3):
    """
    Construye operadores de gradiente ∂/∂x y ∂/∂y usando stencils
    direccionales (upwind) para cada nodo.

    Para cada nodo i con velocidad V[i]:
      - Se toman oversample*n vecinos más cercanos
      - Se filtran los que están aguas arriba: (x_j - x_i)·(-V[i]) ≥ 0
      - Si quedan < n, se usan los n más cercanos sin filtro (fallback)
      - Se calculan los pesos RBF-FD con esos vecinos

    Parámetros
    ----------
    nodes    : (N, 2) nodos del dominio
    V        : (N, 2) velocidades de Darcy
    eps_M    : (N,)   parámetros de forma RBF
    n        : int    tamaño del stencil objetivo
    phi      : str    nombre del RBF (p.ej. 'mq')
    order    : int    orden del polinomio aumentado
    oversample : int  factor para el pool de vecinos previo al filtrado

    Devuelve
    --------
    Dx_upw, Dy_upw : sparse matrices (N, N) para ∂/∂x y ∂/∂y
    """
    N = len(nodes)
    pool_size = min(oversample * n, N)

    # V puede ser una lista de arrays (como d.U) o un array (N,2)
    if isinstance(V, list):
        V = V[0] if len(V) == 1 else np.vstack(V)
    V = np.asarray(V)
    assert V.shape == (N, 2), f"V.shape={V.shape} != ({N},2)"

    # eps_M puede ser (N,) o (N,k): usar el primer valor por nodo
    eps_M = np.asarray(eps_M)
    if eps_M.ndim > 1:
        eps_flat = eps_M[:, 0].astype(float)
    else:
        eps_flat = eps_M.astype(float)

    kd = KDTree(nodes)
    _, pool_idx = kd.query(nodes, k=pool_size)  # (N, pool_size)

    Dx = lil_matrix((N, N), dtype=float)
    Dy = lil_matrix((N, N), dtype=float)

    n_fallback = 0

    for i in range(N):
        vi = V[i]
        vnorm = np.linalg.norm(vi)

        if vnorm < 1e-20:
            # Sin flujo: usa los n vecinos más cercanos (centrado)
            neighbors = pool_idx[i, :n]
        else:
            upstream_dir = -vi / vnorm        # dirección hacia aguas arriba
            cands = pool_idx[i]               # pool_size candidatos
            diff = nodes[cands] - nodes[i]    # (pool, 2)
            proj = diff @ upstream_dir        # proyección sobre upstream
            up_mask = proj >= 0               # aguas arriba
            upstream = cands[up_mask]

            if len(upstream) >= n:
                # Tomar los n más cercanos entre los upstream
                d_up = np.linalg.norm(nodes[upstream] - nodes[i], axis=1)
                order_up = np.argsort(d_up)
                neighbors = upstream[order_up[:n]]
            else:
                # Fallback: n vecinos más cercanos sin dirección
                neighbors = pool_idx[i, :n]
                n_fallback += 1

        # Siempre incluir el nodo i en el stencil (evita singularidad)
        if i not in neighbors:
            neighbors = np.append(neighbors[:-1], i)

        pts = nodes[neighbors]
        k = len(neighbors)
        eps_i = float(eps_flat[i])

        try:
            wx = weight_matrix(
                x=nodes[i:i+1], p=pts, n=k,
                diffs=[[1, 0]], phi=phi, order=order, eps=eps_i,
            ).toarray().ravel()

            wy = weight_matrix(
                x=nodes[i:i+1], p=pts, n=k,
                diffs=[[0, 1]], phi=phi, order=order, eps=eps_i,
            ).toarray().ravel()
        except Exception:
            # Si falla (stencil degenerado), usar gradiente centrado estándar
            std_nb = pool_idx[i, :n]
            pts_std = nodes[std_nb]
            wx = weight_matrix(
                x=nodes[i:i+1], p=pts_std, n=n,
                diffs=[[1, 0]], phi=phi, order=order, eps=eps_i,
            ).toarray().ravel()
            wy = weight_matrix(
                x=nodes[i:i+1], p=pts_std, n=n,
                diffs=[[0, 1]], phi=phi, order=order, eps=eps_i,
            ).toarray().ravel()
            neighbors = std_nb

        for local_j, global_j in enumerate(neighbors):
            if wx[local_j] != 0.0:
                Dx[i, global_j] += wx[local_j]
            if wy[local_j] != 0.0:
                Dy[i, global_j] += wy[local_j]

    if n_fallback > 0:
        print(f"  [upwind] {n_fallback}/{N} nodos usaron fallback centrado"
              f" (insuf. vecinos upstream)")

    return Dx.tocsr(), Dy.tocsr()


def build_upwind_transport_matrix(V, nodes, groups, normals,
                                   pho, D, Div_D, Qout, eps_M,
                                   gauss_p, sig, p):
    """
    Matriz de transporte con gradiente advectivo upwind.

    Igual que build_transport_matrix pero el término advectivo u·∇C
    usa operadores de gradiente con stencil direccional.

    Diffusion term  ∇·(D∇C): usa stencil centrado (Laplaciano estándar)
    Advection term  u·∇C:     usa stencil direccional (upwind)
    """
    from funtions.operators import (
        apply_boundary_conditions, make_transport_bc,
        bc_block, build_operator_block
    )
    from funtions.operators import rbf_op, peso_theta_C
    from scipy.sparse import eye as speye

    w = peso_theta_C(p, sig)

    N = len(nodes)
    idx_dom = np.hstack([
        groups['interior'],
        groups['boundary:inlet'],
        groups['boundary:outlet'],
        groups['boundary:wall'],
    ])

    base = pho * p.R
    coef_identity = base * (2.0 / p.dt)

    if getattr(p, "extract_C", True) and p.activate_ext and Qout and p.run_type != "optimization":
        coef_identity -= w * sig * gauss_p

    if p.model == "adr":
        coef_identity -= w * base * sig * p.landa
    elif p.model == "mrmt_block":
        coef_identity -= w * 2 * base * sig * p.alpha_sum
    elif p.model != "mrmt_semi":
        raise ValueError(f"Modelo no válido: {p.model}")

    # ---- Parte difusiva + identidad: centrado, como siempre ----
    op_diff = dict(
        idx=idx_dom,
        eps=eps_M,
        diffs=[[0, 0], [2, 0], [0, 2]],
        coeffs_func=lambda i: [
            coef_identity[i],
            w * sig * D[i, 0],
            w * sig * D[i, 1],
        ]
    )
    A_diff = apply_boundary_conditions(
        [build_operator_block(nodes, p, op_diff)],
        None if sig > 0 else make_transport_bc(normals, D),
        nodes, groups, p, eps_M, N
    )

    # ---- Parte advectiva: upwind ----
    # Si están preconstruidos (en p._Dx_upw, p._Dy_upw), reusar.
    Dx_upw = getattr(p, '_Dx_upw', None)
    Dy_upw = getattr(p, '_Dy_upw', None)
    if Dx_upw is None or Dy_upw is None:
        print(f"  [upwind] construyendo gradiente direccional (N={N})...",
              flush=True)
        Dx_upw, Dy_upw = build_directional_gradient(
            nodes, V, eps_M,
            n=p.n_stencil,
            phi=p.phi,
            order=p.order,
        )

    # Coeficientes advectivos para los nodos del dominio
    # Ecuación de transporte: ∂C/∂t + U_div_D·∇C - D:∇²C = src
    # U_div_D = (V - Div_D[:,:2]) / 2  (factor 2 del esquema θ)
    U_div_D = (V - Div_D[:, :2]) / 2.0

    # Construir diagonal de coeficientes para idx_dom
    from scipy.sparse import diags as sp_diags
    cx = np.zeros(N); cy = np.zeros(N)
    cx[idx_dom] = -w * sig * U_div_D[idx_dom, 0]
    cy[idx_dom] = -w * sig * U_div_D[idx_dom, 1]

    A_adv = sp_diags(cx) @ Dx_upw + sp_diags(cy) @ Dy_upw

    return (A_diff + A_adv).tocsc()
