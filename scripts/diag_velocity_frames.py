"""
Diagnóstico de la fluctuación de |U| cerca del pozo (backend bfr), en 5 frames.

(A) Descomposición por operador sobre el MISMO campo H de cada frame:
      - RBF-FD mq  (operador actual, config/default.json)
      - RBF-FD phs3 + polinomio grado 2 (best-practice Fornberg-Flyer)
      - lineal local (ajuste de plano por mínimos cuadrados, ~gradiente P1 de FEM)
    Aisla si el ringing de |U|=|K·∇H| viene del OPERADOR o de la física de K.

(B) Contraste BFR vs FEniCS del |U| en los mismos 5 instantes, interpolando
    ambos a puntos comunes de un anillo alrededor del pozo.

No toca el benchmark: recomputa SOLO el flujo (H no depende del transporte),
save_dat=False, y escribe la figura en /tmp.  Requiere haber corrido antes el
forward FEniCS a /tmp/diag_fenicsx (run_fenicsx_diag.py, env fenics_env).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/alex/dev/doctorado/codigo")
sys.path.insert(0, "/home/alex/dev/doctorado/codigo")

from funtions.runtime import RUNTIME
from config.parameters import Parameters
RUNTIME.params = Parameters(json_path="config/default.json")
p = RUNTIME.params
p.save_dat = False; p.animate = False; p.pre = False

from preprocessing.preprocess import load_data
from funtions.operators import H_vector
from rbf.pde.fd import weight_matrix
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

d = load_data()
nodes = d.nodes
N = len(nodes)
pozo = np.asarray(p.pozo, float)
sp = p.spacing

# ---------------------------------------------------------------------------
# Operadores de gradiente alternativos (sobre la MISMA nube de nodos)
# ---------------------------------------------------------------------------
# (1) mq actual  -> d.grad[0], d.grad[1]
# (2) phs3 + polinomio grado 2
Dx_phs = csr_matrix(weight_matrix(x=nodes, p=nodes, n=p.n_stencil, diffs=[1, 0], phi="phs3", order=2))
Dy_phs = csr_matrix(weight_matrix(x=nodes, p=nodes, n=p.n_stencil, diffs=[0, 1], phi="phs3", order=2))

# (3) gradiente lineal local (ajuste de plano MC, ~P1 FEM) — vectorizado
klin = 12
_, idx_lin = cKDTree(nodes).query(nodes, k=klin)
dxy = nodes[idx_lin] - nodes[:, None, :]              # (N,k,2)
Alin = np.concatenate([np.ones((N, klin, 1)), dxy], axis=2)  # (N,k,3)
AtA = np.einsum("nki,nkj->nij", Alin, Alin)           # (N,3,3)


def grad_linear(H):
    Atb = np.einsum("nki,nk->ni", Alin, H[idx_lin])[:, :, None]   # (N,3,1)
    coef = np.linalg.solve(AtA, Atb)[:, :, 0]                     # (N,3)
    return coef[:, 1], coef[:, 2]


def darcy_mag(gx, gy):
    """|U| = |-K·∇H| con la misma K anisótropa que usa el solver y FEniCS."""
    Ux = -d.K[:, 0] * gx
    Uy = -d.K[:, 1] * gy
    return np.hypot(Ux, Uy)


# ---------------------------------------------------------------------------
# Métrica de rugosidad (alta frecuencia nodo-a-nodo, aislada del trend radial)
# ---------------------------------------------------------------------------
kk = 9
_, nbr = cKDTree(nodes).query(nodes, k=kk)
nbr = nbr[:, 1:]                                       # descartar self
rows = np.repeat(np.arange(N), kk - 1)
A_ref = csr_matrix((np.full(rows.size, 1.0 / (kk - 1)), (rows, nbr.ravel())), shape=(N, N))

r_node = np.linalg.norm(nodes - pozo, axis=1)
ring = (r_node > 1.0 * sp) & (r_node < 5.0 * sp)


def rough(field):
    hf = (field - A_ref @ field)[ring]
    return hf.std() / (np.abs(field[ring]).mean() + 1e-30)


Kmag = np.hypot(d.K[:, 0], d.K[:, 1])
rough_K = rough(Kmag)

# ---------------------------------------------------------------------------
# Forward SOLO flujo, capturando H en 5 frames
# ---------------------------------------------------------------------------
frames = [144, 288, 432, 576, 719]   # Q tiene Nt entradas (0..Nt-1)
H = d.H[0].copy()
Q = p.Qout[0]
H_frames = {}
for n in range(1, p.Nt):
    rhs = H_vector(H, d.A_right, d.delta_p, d.groups, nodes, N, None, Q[n - 1], Q[n])
    H = d.A_left.solve(rhs)
    if n in frames:
        H_frames[n] = H.copy()

# ---------------------------------------------------------------------------
# FEniCS (para el contraste B)
# ---------------------------------------------------------------------------
FEN = "/tmp/diag_fenicsx/simulation_results.npz"
have_fen = os.path.exists(FEN)
if have_fen:
    fz = np.load(FEN)
    U_fen = fz["U"]            # (Nt, Nf, 2)
    nodes_fen = fz["nodes"]    # (Nf, 2)
    print(f">>> FEniCS cargado: U{U_fen.shape}, nodes{nodes_fen.shape}")
else:
    print(f">>> AVISO: no existe {FEN}; se omite el contraste BFR vs FEniCS (fila B).")

# Puntos comunes: anillo polar alrededor del pozo (radios × ángulos)
radii = np.linspace(1.2, 5.0, 9) * sp
theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
RR, TH = np.meshgrid(radii, theta)
pts = np.column_stack([pozo[0] + RR.ravel() * np.cos(TH.ravel()),
                       pozo[1] + RR.ravel() * np.sin(TH.ravel())])
pts_r = np.linalg.norm(pts - pozo, axis=1)


def azimuthal_cv(vals):
    """CV azimutal medio: por radio, std/mean sobre ángulos (aísla ringing)."""
    V = vals.reshape(TH.shape)          # (ntheta, nradii)
    cv = V.std(axis=0) / (np.abs(V).mean(axis=0) + 1e-30)
    return np.nanmean(cv)


# ---------------------------------------------------------------------------
# Figura + tabla
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 5, figsize=(22, 8.5), sharey="row")
print("\n=== Rugosidad de |∇H| en el anillo del pozo (mismo H por frame) ===")
print(f"{'frame t/T':>9} | {'rough K':>8} | {'mq':>8} | {'phs3':>8} | {'lineal':>8} | "
      f"{'mq/lin':>7} | {'phs/lin':>7} | {'CVmq':>6} | {'CVlin':>6} | {'CVfen':>6}")

for j, n in enumerate(frames):
    Hn = H_frames[n]
    gx_mq, gy_mq = d.grad[0] @ Hn, d.grad[1] @ Hn
    gx_ph, gy_ph = Dx_phs @ Hn, Dy_phs @ Hn
    gx_li, gy_li = grad_linear(Hn)

    Umq = darcy_mag(gx_mq, gy_mq)
    Uph = darcy_mag(gx_ph, gy_ph)
    Uli = darcy_mag(gx_li, gy_li)

    r_mq = rough(np.hypot(gx_mq, gy_mq))
    r_ph = rough(np.hypot(gx_ph, gy_ph))
    r_li = rough(np.hypot(gx_li, gy_li))

    # --- Fila A: |U| vs radio por operador (nodos del anillo) ---
    a = ax[0, j]
    rr = r_node[ring] / sp
    a.scatter(rr, Umq[ring], s=12, alpha=0.6, label=f"mq (rough {r_mq:.2f})", color="C3")
    a.scatter(rr, Uph[ring], s=12, alpha=0.6, label=f"phs3 (rough {r_ph:.2f})", color="C0")
    a.scatter(rr, Uli[ring], s=12, alpha=0.6, label=f"lineal (rough {r_li:.2f})", color="C2")
    a.set_title(f"t/T={n/p.Nt:.1f}  ·  rough K={rough_K:.2f}", fontsize=10)
    a.set_xlabel("r / spacing"); a.legend(fontsize=7)
    if j == 0:
        a.set_ylabel("(A) |U| por operador\n(mismo H)")
    a.grid(alpha=0.3)

    # --- Fila B: BFR-mq vs BFR-lineal vs FEniCS en puntos comunes del anillo ---
    b = ax[1, j]
    cv_bfr = cv_lin = cv_fen = np.nan
    Ubfr_pts = griddata(nodes, Umq, pts, method="linear")
    cv_bfr = azimuthal_cv(Ubfr_pts)
    b.scatter(pts_r / sp, Ubfr_pts, s=12, alpha=0.6, label=f"BFR-mq (CV {cv_bfr:.2f})", color="C3")
    Ulin_pts = griddata(nodes, Uli, pts, method="linear")
    cv_lin = azimuthal_cv(Ulin_pts)
    b.scatter(pts_r / sp, Ulin_pts, s=12, alpha=0.5, label=f"BFR-lineal (CV {cv_lin:.2f})", color="C2")
    if have_fen:
        Ufen_mag = np.hypot(U_fen[n - 1, :, 0], U_fen[n - 1, :, 1])
        Ufen_pts = griddata(nodes_fen, Ufen_mag, pts, method="linear")
        cv_fen = azimuthal_cv(Ufen_pts)
        b.scatter(pts_r / sp, Ufen_pts, s=12, alpha=0.6, label=f"FEniCS (CV {cv_fen:.2f})", color="C0")
    b.set_xlabel("r / spacing"); b.legend(fontsize=7)
    if j == 0:
        b.set_ylabel("(B) |U| BFR vs FEniCS\n(anillo común)")
    b.grid(alpha=0.3)

    def rel(ga, gb):
        A_ = np.hypot(*ga)[ring]; B_ = np.hypot(*gb)[ring]
        return (np.abs(A_ - B_).mean()) / (np.abs(B_).mean() + 1e-30)
    print(f"{n/p.Nt:>9.1f} | {rough_K:>8.3f} | {r_mq:>8.3f} | {r_ph:>8.3f} | {r_li:>8.3f} | "
          f"{100*rel((gx_mq,gy_mq),(gx_li,gy_li)):>6.1f}% | "
          f"{100*rel((gx_ph,gy_ph),(gx_li,gy_li)):>6.1f}% | "
          f"{cv_bfr:>6.3f} | {cv_lin:>6.3f} | {cv_fen:>6.3f}")

fig.suptitle("Fluctuación de |U| cerca del pozo — (A) por operador sobre el mismo H   "
             "(B) BFR vs FEniCS", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "/tmp/diag_velocity_frames.png"
fig.savefig(out, dpi=120)
print(f"\n>>> figura guardada en {out}")
