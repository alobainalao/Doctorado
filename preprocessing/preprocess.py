import numpy as np
import pickle
from sklearn.neighbors import KDTree
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
import funtions.get_phi as gph
from funtions.mesh import build_nodes_var
from funtions.utils import *
from funtions.operators import *
from funtions.runtime import RUNTIME
p = RUNTIME.params


def run_preprocess():

    print(">>> Ejecutando preprocesamiento...")

    nodes, groups, normals, simplices, pozo_cor = build_nodes_var()

    hU = local_lengthscale(nodes, k=p.n_stencil-1)
    eps_arr_ = eps_from_h(hU, scale=1.0)

    eps_M = np.repeat(eps_arr_[:, None], p.n_stencil, axis=1)
    eps_arr = np.repeat(eps_arr_[:, None], 1, axis=1)

    if p.domain == "benchmark":
        pho = np.full(len(nodes), phi_const)

    elif p.domain == "real":
        pho = gph.gen_malla(nodes)

    else:
        raise ValueError(f"Dominio inválido: {p.domain}")

    lam_im = p.Deff.copy()
    exp_lam_dt = np.zeros(p.Deff.shape)

    for r in range(p.Nr):
        lam_im[r] /= (p.R*p.beta[r]*(p.L**2))
        exp_lam_dt[r] = np.exp(-lam_im[r]*p.dt)

    K = K_ope(pho)
    D_f = diffusion_operator(pho)
    grad = build_operators(nodes, eps_M)

    div_K = Div_KD(K, grad)

    A_left = build_H_matrix(nodes, groups, normals, p.theta, pho, K, div_K, eps_M, -1)
    A_right = build_H_matrix(nodes, groups, normals, 1-p.theta, pho, K, div_K, eps_M, 1)

    delta_p = delta_char(nodes, pozo_cor, 1e-6)
    gauss_p = gaussian_2d(nodes, pozo_cor, 1, p.epsilon_y)
    gauss_f = gaussian_2d(nodes, p.fnte, 1, p.epsilon_x, p.epsilon_y)

    N = nodes.shape[0]

    H0, C0 = get_init_values(nodes)
    H0 = new_H_init(H0, None, gauss_p, groups, N, eps_arr, 
                    nodes, normals, pho, K, div_K, eps_M)

    U0 = U_vector(H0, K, grad)

    H = [H0.copy() for _ in range(4)]
    U = [U0.copy() for _ in range(4)]
    C = [C0.copy() for _ in range(4)]
    C_im = [np.zeros((p.Nr, len(nodes))) for _ in range(4)]

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
        H=H, C=C, U=U, C_im=C_im,
        xy_grid=xy_grid, xy=xy, I=I, mask=mask,
        stencils=stencils, exp_lam_dt=exp_lam_dt
    )

    savefile = f"{p.save_preprocess}/preproceso.pkl"
    with open(savefile, "wb") as f:
        pickle.dump(data, f)

    print(">>> Preprocesamiento guardado en", savefile)

    return data


def load_data():
    p = RUNTIME.params
    if p.pre:
        data = run_preprocess()
    else:
        print(">>> Cargando datos...")
        with open(f"{p.save_preprocess}/preproceso.pkl", "rb") as f:
            data = pickle.load(f)

    class Data:
        pass

    d = Data()
    d.__dict__.update(data)

    print(">>> Datos listos")
    return d
