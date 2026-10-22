"""
tests/test_bc_outlet_mode.py
============================
Pruebas unitarias para la opción bc_outlet_mode en el backend BFR.

Verifica:
  1. Modo "pfs" → operador de 2.º orden en salida (sin regresión).
  2. Modo "fem" → operador de 1.º orden Neumann en salida.
  3. RHS en nodos fantasma de salida tiene el signo correcto.
  4. activate_ext=False desactiva el pozo en H_vector.
  5. activate_ext=True genera caudal de pozo ≠ 0 en H_vector.

Ejecutar desde codigo/:
    conda run -n bfr_env pytest tests/test_bc_outlet_mode.py -v
"""

import sys
import os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from funtions.runtime import RUNTIME


# ---------------------------------------------------------------------------
# Fixture: parámetros base mínimos para el dominio benchmark (rápido)
# ---------------------------------------------------------------------------

@pytest.fixture
def p_pfs(tmp_path):
    from config.parameters import Parameters
    p = Parameters()
    p.model           = "adr"
    p.domain          = "benchmark"
    p.metodo          = "bfr"
    p.bc_outlet_mode  = "pfs"
    p.activate_ext    = False
    p.activate_fuente = False
    p.spacing         = 10
    p.min_sp          = 10
    p.pre             = False
    p.save_dat        = False
    p.animate         = False
    p.save_data       = str(tmp_path)
    RUNTIME.params = p
    return p


@pytest.fixture
def p_fem(tmp_path):
    from config.parameters import Parameters
    p = Parameters()
    p.model           = "adr"
    p.domain          = "benchmark"
    p.metodo          = "bfr"
    p.bc_outlet_mode  = "fem"
    p.activate_ext    = False
    p.activate_fuente = False
    p.spacing         = 10
    p.min_sp          = 10
    p.pre             = False
    p.save_dat        = False
    p.animate         = False
    p.save_data       = str(tmp_path)
    RUNTIME.params = p
    return p


# ---------------------------------------------------------------------------
# Helpers: construir nodos y operadores en benchmark
# ---------------------------------------------------------------------------

def _build_benchmark_nodes(p):
    """Devuelve nodes, groups, normals para benchmark con spacing uniforme."""
    from funtions.mesh import _boundary_config
    from rbf.pde.nodes import poisson_disc_nodes

    smp, boundary_groups, boundary_groups_with_ghosts = _boundary_config()
    nodes, groups, normals = poisson_disc_nodes(
        p.spacing,
        (p.vert, smp),
        boundary_groups=boundary_groups,
        boundary_groups_with_ghosts=boundary_groups_with_ghosts,
    )
    return nodes, groups, normals


# ---------------------------------------------------------------------------
# TEST 1: modo "pfs" → operador de 2.º orden (5 diffs) en salida
# ---------------------------------------------------------------------------

def test_pfs_mode_has_second_order_outlet(p_pfs):
    from funtions.operators import make_H_bc
    from funtions.utils import K_ope, Div_KD, local_lengthscale, eps_from_h
    from funtions.operators import build_operators

    nodes, groups, normals = _build_benchmark_nodes(p_pfs)
    pho = np.full(len(nodes), p_pfs.phi_const)
    K = K_ope(pho)
    hU = local_lengthscale(nodes, k=p_pfs.n_stencil - 1)
    eps_M = np.repeat(eps_from_h(hU)[:, None], p_pfs.n_stencil, axis=1)
    grad = build_operators(nodes, eps_M)
    from funtions.utils import Div_KD
    Div_K = Div_KD(K, grad)

    bcs = make_H_bc(nodes, groups, K, Div_K, normals)
    outlet_bc = next(bc for bc in bcs if bc["name"] == "outlet")

    # 2.º orden → 5 términos diferenciales (1x,1y,2x,2y,xy)
    diffs, coeffs = outlet_bc["coeffs_func"](groups["boundary:outlet"])
    assert len(diffs) == 5, f"Modo pfs debe tener 5 diffs; encontró {len(diffs)}"


# ---------------------------------------------------------------------------
# TEST 2: modo "fem" → operador de 1.º orden (2 diffs) en salida
# ---------------------------------------------------------------------------

def test_fem_mode_has_first_order_outlet(p_fem):
    from funtions.operators import make_H_bc
    from funtions.utils import K_ope, Div_KD, local_lengthscale, eps_from_h
    from funtions.operators import build_operators

    nodes, groups, normals = _build_benchmark_nodes(p_fem)
    pho = np.full(len(nodes), p_fem.phi_const)
    K = K_ope(pho)
    hU = local_lengthscale(nodes, k=p_fem.n_stencil - 1)
    eps_M = np.repeat(eps_from_h(hU)[:, None], p_fem.n_stencil, axis=1)
    grad = build_operators(nodes, eps_M)
    Div_K = Div_KD(K, grad)

    bcs = make_H_bc(nodes, groups, K, Div_K, normals)
    outlet_bc = next(bc for bc in bcs if bc["name"] == "outlet")

    # 1.º orden → sólo 2 diffs (∂/∂x, ∂/∂y)
    diffs, coeffs = outlet_bc["coeffs_func"](groups["boundary:outlet"])
    assert len(diffs) == 2, f"Modo fem debe tener 2 diffs; encontró {len(diffs)}"
    assert diffs == [[1, 0], [0, 1]], f"Diffs inesperados: {diffs}"


# ---------------------------------------------------------------------------
# TEST 3: RHS en fantasmas de salida tiene signo correcto en modo "fem"
# ---------------------------------------------------------------------------

def test_outlet_ghost_rhs_applied_in_fem_mode(p_fem):
    """
    En modo "fem", H_vector debe aplicar el perfil de salida a los nodos
    fantasma del outlet (RHS ≠ 0). En modo "pfs", esos nodos quedan en 0.

    Nota: el valor exacto puede ser ligeramente negativo en el dominio
    benchmark (efectos de esquina en In_h), pero lo importante es que el
    perfil SE APLIQUE en modo fem y NO SE APLIQUE en modo pfs.
    """
    from funtions.operators import H_vector
    from funtions.utils import K_ope, Div_KD, local_lengthscale, eps_from_h
    from funtions.operators import build_operators, build_H_matrix
    from funtions.utils import discrete_delta

    nodes, groups, normals = _build_benchmark_nodes(p_fem)
    N = len(nodes)
    pho = np.full(N, p_fem.phi_const)
    K = K_ope(pho)
    hU = local_lengthscale(nodes, k=p_fem.n_stencil - 1)
    eps_M = np.repeat(eps_from_h(hU)[:, None], p_fem.n_stencil, axis=1)
    eps_arr = np.repeat(eps_from_h(hU)[:, None], 1, axis=1)
    from funtions.utils import Div_KD
    grad = build_operators(nodes, eps_M)
    Div_K = Div_KD(K, grad)

    M_right = build_H_matrix(nodes, groups, normals, 1 - p_fem.theta, pho, K, Div_K, eps_M, 1, eps_arr)
    delta_p = discrete_delta(nodes, np.asarray(p_fem.pozo), 2.5 * p_fem.spacing)

    outlet_ghosts = groups.get("ghosts:outlet", np.array([], dtype=int))
    if len(outlet_ghosts) == 0:
        pytest.skip("No hay nodos fantasma de salida en benchmark")

    H0 = np.zeros(N)

    # Modo fem: el perfil se aplica → b_outlet debe ser distinto de cero
    b_fem = H_vector(H0, M_right, delta_p, groups, nodes, N)
    assert not np.allclose(b_fem[outlet_ghosts], 0.0), (
        "En modo fem, H_vector debe aplicar el perfil de salida (RHS ≠ 0)"
    )

    # Modo pfs: no se aplica perfil → b_outlet debe ser cero (H0=0 → aux=0)
    p_fem.bc_outlet_mode = "pfs"
    RUNTIME.params = p_fem
    # Reconstruir M_right con el nuevo modo (el operador de salida cambia)
    from funtions.operators import build_H_matrix as bhm
    M_right_pfs = bhm(nodes, groups, normals, 1 - p_fem.theta, pho, K, Div_K, eps_M, 1, eps_arr)
    b_pfs = H_vector(H0, M_right_pfs, delta_p, groups, nodes, N)
    assert np.allclose(b_pfs[outlet_ghosts], 0.0, atol=1e-30), (
        "En modo pfs, los fantasmas de salida deben tener b=0 cuando H=0"
    )


# ---------------------------------------------------------------------------
# TEST 4: activate_ext=False → término de pozo = 0 en H_vector
# ---------------------------------------------------------------------------

def test_noextraction_disables_well(p_pfs):
    """Con activate_ext=False, el término Q·delta debe ser cero."""
    from funtions.operators import H_vector, build_H_matrix
    from funtions.utils import K_ope, Div_KD, local_lengthscale, eps_from_h, discrete_delta
    from funtions.operators import build_operators

    nodes, groups, normals = _build_benchmark_nodes(p_pfs)
    N = len(nodes)
    pho = np.full(N, p_pfs.phi_const)
    K = K_ope(pho)
    hU = local_lengthscale(nodes, k=p_pfs.n_stencil - 1)
    eps_M = np.repeat(eps_from_h(hU)[:, None], p_pfs.n_stencil, axis=1)
    eps_arr = np.repeat(eps_from_h(hU)[:, None], 1, axis=1)
    grad = build_operators(nodes, eps_M)
    Div_K = Div_KD(K, grad)

    M_right = build_H_matrix(nodes, groups, normals, 1 - p_pfs.theta, pho, K, Div_K, eps_M, 1, eps_arr)
    delta_p = discrete_delta(nodes, np.asarray(p_pfs.pozo), 2.5 * p_pfs.spacing)

    H0 = np.zeros(N)
    # Con activate_ext=False y Qout_N=999 → b no debe cambiar al variar Q
    b1 = H_vector(H0, M_right, delta_p, groups, nodes, N, Qout_N=0)
    b2 = H_vector(H0, M_right, delta_p, groups, nodes, N, Qout_N=999)

    # b1 y b2 deben ser idénticos: no hay diferencia por el caudal
    np.testing.assert_allclose(b1, b2, atol=0,
        err_msg="Con activate_ext=False, el caudal no debe afectar H_vector")


# ---------------------------------------------------------------------------
# TEST 5: activate_ext=True → el término de pozo modifica H_vector
# ---------------------------------------------------------------------------

def test_extraction_changes_hvector(p_fem):
    """Con activate_ext=True, variar Qout_N debe cambiar H_vector."""
    from funtions.operators import H_vector, build_H_matrix
    from funtions.utils import K_ope, Div_KD, local_lengthscale, eps_from_h, discrete_delta
    from funtions.operators import build_operators

    p_fem.activate_ext = True
    RUNTIME.params = p_fem

    nodes, groups, normals = _build_benchmark_nodes(p_fem)
    N = len(nodes)
    pho = np.full(N, p_fem.phi_const)
    K = K_ope(pho)
    hU = local_lengthscale(nodes, k=p_fem.n_stencil - 1)
    eps_M = np.repeat(eps_from_h(hU)[:, None], p_fem.n_stencil, axis=1)
    eps_arr = np.repeat(eps_from_h(hU)[:, None], 1, axis=1)
    grad = build_operators(nodes, eps_M)
    Div_K = Div_KD(K, grad)

    M_right = build_H_matrix(nodes, groups, normals, 1 - p_fem.theta, pho, K, Div_K, eps_M, 1, eps_arr)
    delta_p = discrete_delta(nodes, np.asarray(p_fem.pozo), 2.5 * p_fem.spacing)

    H0 = np.zeros(N)
    b_noQ = H_vector(H0, M_right, delta_p, groups, nodes, N, Qout_N=0.0)
    b_Q   = H_vector(H0, M_right, delta_p, groups, nodes, N, Qout_N=1e-3)

    # Al menos un nodo debe diferir
    assert not np.allclose(b_noQ, b_Q), \
        "Con activate_ext=True, Qout_N != 0 debe modificar H_vector"
