import numpy as np
import pickle
from sklearn.neighbors import KDTree
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
import funtions.get_phi as gph
from funtions.mesh import build_nodes_var
from funtions.utils import *
from funtions.operators import *
from scipy.sparse.linalg import splu
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
    

    delta_p = discrete_delta(nodes, pozo_cor, 2.5*p.spacing)
    # delta_p = delta_char(nodes, pozo_cor, 1e-6)
    gauss_p = gaussian_2d(nodes, pozo_cor, 1, p.epsilon_y)
    gauss_f = gaussian_2d(nodes, p.fnte, 1, p.epsilon_x, p.epsilon_y)

    
    
    if p.run_type == "optimization":
           # =====================================================
        # ψ_H
        # =====================================================

        # -----------------------------------------------------
        # A ψ_H^n
        # -----------------------------------------------------
        A_H = build_adj_H_matrix(
            nodes=nodes,
            groups=groups,
            normals=normals,
            theta =p.theta,
            S_s=pho,
            K=K,
            Div_K=div_K,
            eps_M=eps_M,

            sig=-1
        )

        # -----------------------------------------------------
        # B ψ_H^{n+1}
        # -----------------------------------------------------
        B_H = build_adj_H_matrix(
            nodes=nodes,
            groups=groups,
            normals=normals,
            theta =1-p.theta,
            S_s=pho,
            K=K,
            Div_K=div_K,

            eps_M=eps_M,

            sig=+1
        )

        wi=rbf_integration_weights(nodes, 2.0)

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
        stencils=stencils
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

    if p.run_type == "optimization":
        data["A_H"]= A_H
        data["B_H"]= B_H
        data["wi"]= wi
        data["z0"]= 300
        data["beta"]= 0.2
        data["gamma"] = 0.5



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
    d.A_left = splu(d.A_left)
    if p.run_type == "optimization":
        d.A_H = splu(d.A_H)


    print(">>> Datos listos")
    return d
