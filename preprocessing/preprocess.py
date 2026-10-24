import os
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
import funtions.get_phi as gph
#from funtions.mesh import build_nodes_var
from funtions.mesh_test import build_nodes_var
from funtions.utils import *
from funtions.operators import *
from scipy.sparse.linalg import splu
from funtions.runtime import RUNTIME
p = RUNTIME.params


def run_preprocess():

    # p fresco desde RUNTIME (no el global de módulo, que puede haber quedado
    # ligado a None si este módulo se importó antes de fijar RUNTIME.params —
    # p.ej. desde tests/test_gradient_checkpoint.py).
    p = RUNTIME.get()

    print(">>> Ejecutando preprocesamiento...")

    mesh_pkl = getattr(p, "mesh_pkl", None)
    if mesh_pkl:
        # Malla congelada (experimentos/): se reusan los nodos exactos y sólo se
        # reconstruyen los operadores con los parámetros de esta corrida
        # (dt, theta, sigma_fuente, bc_outlet_mode, …).
        with open(mesh_pkl, "rb") as f:
            _m = pickle.load(f)
        nodes, groups, normals = _m["nodes"], _m["groups"], _m["normals"]
        simplices, pozo_cor = _m["simplices"], _m["pozo_cor"]
        print(f">>> Malla congelada: {mesh_pkl} ({len(nodes)} nodos)")
    else:
        nodes, groups, normals, simplices, pozo_cor = build_nodes_var()

    hU = local_lengthscale(nodes, k=p.n_stencil-1)
    eps_arr_ = eps_from_h(hU, scale=1.0)

    eps_M = np.repeat(eps_arr_[:, None], p.n_stencil, axis=1)
    eps_arr = np.repeat(eps_arr_[:, None], 1, axis=1)

    if p.domain == "benchmark":
        pho = np.full(len(nodes), p.phi_const)

    elif p.domain == "real":
        pho = gph.gen_malla(nodes)

    else:
        raise ValueError(f"Dominio inválido: {p.domain}")


    K = K_ope(pho)
    D_f = diffusion_operator(pho)
    grad = build_operators(nodes, eps_M)

    div_K = Div_KD(K, grad)

    A_left = build_H_matrix(nodes, groups, normals, p.theta, pho, K, div_K, eps_M, -1)
    A_right = build_H_matrix(nodes, groups, normals, 1-p.theta, pho, K, div_K, eps_M, 1)
    

    delta_p = discrete_delta(nodes, np.array(p.pozo), p.radio_delta_pozo)
    if p.norm_delta_pozo == "area":
        delta_p = normalizar_delta_area(delta_p, rbf_integration_weights(nodes))
    # delta_p = delta_char(nodes, pozo_cor, 1e-6)
    gauss_p = gaussian_2d(nodes, pozo_cor, 1, p.epsilon_pozo)

    # Detectar si existe el nodo forzado en p.pozo y precomputar pesos IDW
    # para corregir la oscilación espuria del adjunto en ese nodo.
    _r_pozo = np.linalg.norm(nodes - np.asarray(p.pozo, float), axis=1)
    _fi = int(np.argmin(_r_pozo))
    if _r_pozo[_fi] < 0.5:
        _kd = KDTree(nodes)
        _, _nbrs = _kd.query(nodes[_fi:_fi+1], k=p.n_stencil + 1)
        _nbrs = _nbrs[0][1:]   # excluir self
        _dists = np.linalg.norm(nodes[_nbrs] - nodes[_fi], axis=1)
        _w = 1.0 / np.maximum(_dists, 1e-12)
        _w /= _w.sum()
        forced_well_node = _fi
        forced_well_nbrs = _nbrs.astype(int)
        forced_well_idw  = _w.astype(float)
    else:
        forced_well_node = -1
        forced_well_nbrs = np.array([], dtype=int)
        forced_well_idw  = np.array([], dtype=float)
    gauss_f = gaussian_2d(nodes, p.fnte, 1, p.epsilon_x, p.epsilon_y)

    
    
    if p.run_type in ("optimization", "comparison"):
        # Matrices del paso adjunto (ψ_H): necesarias para comparison y optimization.
        A_H = build_adj_H_matrix(
            nodes=nodes,
            groups=groups,
            normals=normals,
            theta=p.theta,
            S_s=pho,
            K=K,
            Div_K=div_K,
            eps_M=eps_M,
            sig=-1
        )

        B_H = build_adj_H_matrix(
            nodes=nodes,
            groups=groups,
            normals=normals,
            theta=1-p.theta,
            S_s=pho,
            K=K,
            Div_K=div_K,
            eps_M=eps_M,
            sig=+1
        )

        if p.run_type in ("optimization", "comparison"):
            wi = rbf_integration_weights(nodes, 2.0)

    N = nodes.shape[0]

    H0, C0 = get_init_values(nodes)
    H0 = new_H_init(H0, None, gauss_p, groups, N, eps_arr, 
                    nodes, normals, pho, K, div_K, eps_M)

    U0 = U_vector(H0, K, grad)

    H = [H0 for _ in range(p.K)]
    U = [U0 for _ in range(p.K)]
    C = [C0 for _ in range(p.K)]

    xy_grid = np.mgrid[p.xmin:p.xmax:400j, p.ymin:p.ymax:80j]
    xy = xy_grid.reshape(2, -1).T

    _, stencils = KDTree(nodes).query(xy)
    eps_matrix = eps_arr_[stencils]

    I = weight_matrix(
        x=xy, p=nodes, n=1, diffs=[[0,0]],
        phi=p.phi, eps=eps_matrix, chunk_size=None
    )

    segments = np.column_stack((
        np.arange(len(p.vert)),
        np.roll(np.arange(len(p.vert)), -1)
    ))

    mask = contains(xy, p.vert, segments)

    data = dict(
        nodes=nodes, groups=groups, normals=normals, simplices=simplices,
        pozo_cor=pozo_cor, hU=hU, eps_arr_=eps_arr_, eps_M=eps_M,
        eps_arr=eps_arr, pho=pho, K=K, D_f=D_f, grad=grad,
        div_K=div_K, A_left=A_left, A_right=A_right,
        delta_p=delta_p, gauss_p=gauss_p, gauss_f=gauss_f,
        H=H, C=C, U=U,
        xy_grid=xy_grid, xy=xy, I=I, mask=mask,
        stencils=stencils,
        forced_well_node=forced_well_node,
        forced_well_nbrs=forced_well_nbrs,
        forced_well_idw=forced_well_idw,
    )

    if "mrmt" in p.model:
        C_im = [np.zeros((p.Nr, len(nodes))) for _ in range(p.K)]
        lam_im = list(p.Deff)
        exp_lam_dt = np.zeros(len(p.Deff))

        for r in range(p.Nr):
            lam_im[r] /= (p.R*p.beta[r]*(p.L**2))
            exp_lam_dt[r] = np.exp(-lam_im[r]*p.dt)
        
        data["exp_lam_dt"] = exp_lam_dt
        data["C_im"] = C_im
    else:
        data["exp_lam_dt"] = None
        data["C_im"] = [None for _ in range(p.K)]

    if p.run_type in ("optimization", "comparison"):
        data["A_H"] = A_H
        data["B_H"] = B_H
        data["wi"] = wi  # necesario para la fuente adjunta ∂J/∂C = 2γ·gₚ·wi·C
        # z0 (profundidad de referencia del término económico): fuente única en
        # `p` para que la app lo controle y coincida con el backend fenicsx.
        data["z0"] = p.z0



    savefile = f"{p.save_preprocess}/preproceso.pkl"
    with open(savefile, "wb") as f:
        pickle.dump(data, f)


    print(">>> Preprocesamiento guardado en", savefile)

    return data


def load_data():
    p = RUNTIME.params
    savefile = f"{p.save_preprocess}/preproceso.pkl"

    # `data/input/*` está en .gitignore (salvo .h5): en un entorno fresco
    # (deploy, contenedor sin volumen persistente) el .pkl cacheado no existe
    # aunque `pre` no se haya marcado. En ese caso hay que generarlo sí o sí,
    # no solo cuando el usuario pide preprocesamiento explícito.
    if p.pre or not os.path.exists(savefile):
        data = run_preprocess()
    else:
        print(">>> Cargando datos...")
        with open(savefile, "rb") as f:
            data = pickle.load(f)
        # Si se necesitan matrices adjuntas pero el pkl fue generado sin ellas, regenerar.
        if p.run_type in ("optimization", "comparison") and "A_H" not in data:
            print(">>> Regenerando preproceso (A_H no disponible)...")
            data = run_preprocess()
        # Regenerar si faltan los pesos IDW del nodo forzado (pkls anteriores al fix).
        elif "forced_well_node" not in data:
            print(">>> Regenerando preproceso (forced_well_node no disponible)...")
            data = run_preprocess()

    class Data:
        pass

    d = Data()
    d.__dict__.update(data)
    d.A_left = splu(d.A_left)
    if p.run_type in ("optimization", "comparison"):
        d.A_H = splu(d.A_H)


    print(">>> Datos listos")
    return d
