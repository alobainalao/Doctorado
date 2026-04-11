# import os
# import numpy as np
# from config.geometry import build_geometry

# class Parameters:
#     # ---------------------------
#     # 1) Parámetros
#     # ---------------------------
#     pre = False
#     #pre = True

#     postproc = False
#     postproc = True

#     #plot_grid =True
#     plot_grid  = False

#     activate_fuente = True
#     #activate_fuente = False

#     activate_ext = True
#     #activate_ext = False

#     model = "adr"          # Transporte clásico (sin MRMT)
#     #model = "mrmt_semi"    # MRMT semianalítico (exponencial)
#     #model = "mrmt_block"   # MRMT implícito (acoplado)

#     VALID_MODELS = {"adr", "mrmt_semi", "mrmt_block"}

#     if model not in VALID_MODELS:
#         raise ValueError(f"Modelo inválido: {model}")

#     K= 4
#     ncols = int(np.ceil(np.sqrt(K)))
#     nrows = int(np.ceil(K / ncols))

#     metodo = "bfr"
#     #metodo = "fenicx"

#     domain = "real"  
#     #domain = "benchmark"

#     phi_const = 0.1

#     save_preporcess = f"./data/input/{metodo}/{domain}/{model}"
#     os.makedirs(save_preporcess, exist_ok=True)
    
#     if domain == "real":
#         spacing = 30
#         dt, T = 3.6e4, 2592000
#     elif domain == "benchmark":
#         spacing = 1
#         dt, T = 360, 25920
#     else:
#         raise ValueError(f"Dominio inválido: {domain}")



#     min_sp = spacing
#     nu, rho = 1e-3, 1.0

#     n_stencil, phi, order = 35, 'phs7', 4
#     n_stencil, phi, order = 25, 'mq', 2
    
#     savefolder_data  = f"./data/output/{metodo}/data/{domain}/{model}"
#     savefolder_video = f"./data/output/{metodo}/figures/{domain}/{model}"
#     os.makedirs(savefolder_data, exist_ok=True)
#     os.makedirs(savefolder_video, exist_ok=True)

#     tol= 1e-5
#     theta = 0.9
#     sp_q = 5

#     geom = build_geometry(domain, spacing)
#     locals().update(geom)



#     # -------------------- PARÁMETROS FÍSICOS ----------------------
#     g, nu, d_z, alpha = 9.81, 1.055e-6, 1e-3, 2.0
#     tol = 1e-6
#     R, landa = 1, 1e-10

#     # -------------- PARÁMETROS DE DISPERSIÓN ---------------------
#     a_l, a_t, D_d, eps = 10, 1, 1.2e-19, 1e-16


#     # MRMT parameters
#     Nr = 3
#     Deff = np.array([1e-9, 5e-10, 1e-10])
#     L = 0.01

#     alpha_im = np.zeros((3,len(phi)))

#     phi_im = np.array([
#         0.1,
#         0.05,
#         0.02
#     ])

#     beta = np.array([
#         0.15,
#         0.1,
#         0.05
#     ])

#     R_im = np.ones(Nr)

#     alpha_r = np.zeros((Nr))
#     alpha_sum = 0

#     for r in range(Nr):
#         alpha_r[r] = (beta[r] * Deff[r]) / (2*beta[r]*R*L**2 + dt*Deff[r])
#         alpha_sum += alpha_r[r]

import os
import numpy as np

def parse_array(text):
    return np.array([float(x) for x in text.split(",")])

class Parameters:

    model = os.getenv("MODEL", "adr")
    domain = os.getenv("DOMAIN", "real")
    metodo = os.getenv("METODO", "bfr")

    pre = os.getenv("PRE") == "True"
    postproc = os.getenv("POST") == "True"
    plot_grid = os.getenv("PLOT") == "True"

    activate_fuente = os.getenv("FUENTE") == "True"
    activate_ext = os.getenv("EXT") == "True"

    spacing = float(os.getenv("SPACING", 30))
    dt = float(os.getenv("DT", 3600))
    T = float(os.getenv("T", 2592000))

    phi_const = float(os.getenv("PHI_CONST", 0.1))

    a_l = float(os.getenv("A_L", 10))
    a_t = float(os.getenv("A_T", 1))
    D_d = float(os.getenv("D_D", 1e-19))
    eps = float(os.getenv("EPS", 1e-16))

    Nr = int(os.getenv("NR", 3))
    Deff = parse_array(os.getenv("DEFF", "1e-9,5e-10,1e-10"))
    beta = parse_array(os.getenv("BETA", "0.15,0.1,0.05"))
    phi_im = parse_array(os.getenv("PHI_IM", "0.1,0.05,0.02"))

