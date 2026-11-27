"""
Campos del modelo MRMT por bloques (FEniCSx) para la versión simplificada del avance 3 (presentaciones/
avance1-doc-simplificada, src/slides-avance3s/). Misma configuración que la optimización de la serie 33
(experimentos/optimizar_nitrato.py mrmt_block nitrato_vscale_malla_eje_g1_prom --f=nitrato_vscale_malla_eje_g1): malla con
conos y eje de 1 m, Δt = 10 h, 72 pasos, N_r = 3, θ_C = 1, escala f de la fuente y C_ref = 0.05/f.

    python -m scripts.exportar_simplificada campos <etiqueta> <iterado> [--adjunto]       # fenics_env, desde codigo/
    python -m scripts.exportar_simplificada verificar <etiqueta> <iterado> <direccion> <delta>
    python scripts/exportar_simplificada.py js [ruta_de_la_presentacion]

<iterado>: un .json con `x` o `x_opt` (p. ej. optimizacion_<V>/mrmt_block.json) o `<ruta>.jsonl:<nit>` (un iterado de
<modelo>_progreso.jsonl; nit 1 = punto de arranque factible).
campos: directo (y adjunto con --adjunto) en ese control. Guarda data/output/simplificada/<etiqueta>.npz: triangulación
P1, cada 6 pasos h, |u| por nodo (media de los triángulos), C móvil y las tres inmóviles interpolados a P1 (y ψ_h, ψ_C,
ψ_im,r); en cada paso la concentración y la carga observadas en el pozo (∫δ_p C, ∫δ_p h, el mismo núcleo de J1); los
términos de Ĵ y, con --adjunto, el gradiente.
verificar: J y J1 en x ± δ·(s⊙d), s = (Q_max…, 100 m), d unitario: «grad» = −s⊙∇Ĵ normalizado (necesita el .npz de
`campos <etiqueta> … --adjunto`) o «azar» (semilla 42). Compara, en la configuración real, la derivada direccional del
adjunto con el cociente centrado (tests/test_gradiente_modelo_nuevo.py lo hace en la configuración de prueba).
js: bandas de nivel (coordenadas de slides-avance3/escena.jsx) y series → src/slides-avance3s/camposBloque.js.
"""
import json
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
V = "nitrato_vscale_malla_eje_g1_prom"
DIR_OPT = os.path.join(CODIGO, "data", "output", f"optimizacion_{V}")
DIR_OUT = os.path.join(CODIGO, "data", "output", "simplificada")
C_LIMITE = 0.05                    # kg/m³ (50 mg/L como nitrato), el de optimizar_nitrato.py
CADA = 6                           # un cuadro cada 6 pasos (60 h)
S_Z = 100.0                        # escala de z_p [m] en la verificación


def leer_iterado(ruta):
    if ".jsonl:" in ruta:
        ruta, nit = ruta.rsplit(":", 1)
        for linea in open(ruta):
            r = json.loads(linea)
            if r["nit"] == int(nit):
                return np.asarray(r["x"], float), dict(origen=ruta, nit=int(nit))
        sys.exit(f"no está el iterado {nit} en {ruta}")
    d = json.load(open(ruta))
    return np.asarray(d.get("x", d.get("x_opt")), float), dict(origen=ruta, nit=d.get("nit"))


def preparar():
    """Contexto de la optimización de mrmt_block, como experimentos/optimizar_nitrato.py (sin escribir nada)."""
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    cfg = json.load(open(os.path.join(CODIGO, "config", f"{V}.json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": "mrmt_block", "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()
    from fenicsx.mesh import get_mesh, build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context
    mc = build_mesh_from_bfr_pkl(p.mesh_pkl) if getattr(p, "mesh_pkl", None) else get_mesh(p.pozo, p.spacing, regenerate=True)
    ctx = build_context(list(p.pozo), p, mc)
    f = json.load(open(os.path.join(DIR_OPT, "f_fuente.json")))["f"]
    p.C_ref = C_LIMITE / f
    return p, ctx, f


def a_P1(ctx, hist):
    """Filas (…, N_Q) de P2 a los nodos P1 (interpolación, el mismo soporte que h)."""
    from dolfinx import fem
    fQ, fG = fem.Function(ctx.Q), fem.Function(ctx.Gh)
    hist = np.asarray(hist, float)
    out = np.empty(hist.shape[:-1] + (fG.x.array.size,))
    for idx in np.ndindex(hist.shape[:-1]):
        fQ.x.array[:] = hist[idx]
        fG.interpolate(fQ)
        out[idx] = fG.x.array
    return out


def campos(etiqueta, iterado, adjunto=False, pasos_max=None):
    import ufl
    from dolfinx import fem
    from fenicsx.adjoint import _solve_forward, _solve_adjoint_grad, _campo_desde_dual, _gauss, _pesos_J
    x, de = leer_iterado(iterado)
    p, ctx, f = preparar()
    Q, zp = x[:-1][:pasos_max], float(x[-1])            # pasos_max: solo para probar el script en pocos pasos
    Nt = len(Q)
    print(f"[simplificada] {etiqueta}: {de} z_p={zp:.2f} f={f:.4e} C_ref={p.C_ref:.4e}", flush=True)
    r = _solve_forward(ctx, Q, zp)
    w1, w2, w3 = _pesos_J(p, Nt * p.dt)
    z0 = float(getattr(p, "z0", 0.0))
    J2 = float(getattr(p, "koppa", 1.0)) * w2 * abs(zp - z0) ** 2
    J3 = w3 * float(np.sum(Q ** 2)) * p.dt * abs(zp - z0)
    print(f"[simplificada] J={r['J']:.6e} J1={r['J1']:.6e} J2={J2:.6e} J3={J3:.6e}", flush=True)

    Gh = ctx.Gh
    xy = Gh.tabulate_dof_coordinates()[:, :2]
    tri = np.asarray(Gh.dofmap.list, np.int64).reshape(-1, 3)
    pasos = np.arange(CADA - 1, Nt, CADA)                     # t_{n+1} = (n+1)Δt
    H = np.asarray(r["h"])[pasos]
    # |u| = |K_z(α²∂x h, ∂z h)|: gradiente de h P1 constante en cada triángulo, K_z medio del triángulo; a los nodos
    # con la media pesada por área de los triángulos que los tocan
    fK = fem.Function(Gh); fK.interpolate(ctx.K_z)
    alfa = float(ctx.phys["alpha"])
    P0, P1_, P2_ = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    e1, e2 = P1_ - P0, P2_ - P0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    area = 0.5 * np.abs(det)
    Kt = fK.x.array[tri].mean(axis=1)
    peso = np.bincount(tri.ravel(), weights=np.repeat(area, 3), minlength=len(xy))
    U = []
    for h in H:
        d1, d2 = h[tri[:, 1]] - h[tri[:, 0]], h[tri[:, 2]] - h[tri[:, 0]]
        gx = (d1 * e2[:, 1] - d2 * e1[:, 1]) / det
        gz = (d2 * e1[:, 0] - d1 * e2[:, 0]) / det
        ut = Kt * np.hypot(alfa ** 2 * gx, gz)
        U.append(np.bincount(tri.ravel(), weights=np.repeat(ut * area, 3), minlength=len(xy)) / peso)
    C = a_P1(ctx, np.asarray(r["C"])[pasos])
    # C_im del registro es la del inicio de cada paso (t_n): la de t_{n+1} es la del registro n + 1
    Cim_hist = np.asarray(r["C_im"])                           # (Nt, N_r, N_Q)
    pasos_im = np.minimum(pasos + 1, Nt - 1)
    Cim = a_P1(ctx, np.moveaxis(Cim_hist[pasos_im], 1, 0))     # (N_r, cuadros, N_G)

    # observación en el pozo con el núcleo de J1 (masa 1): ∫δ_p C, ∫δ_p h en cada paso
    zp_c = fem.Constant(ctx.domain, float(zp))
    delta_p = _gauss(ufl.SpatialCoordinate(ctx.domain), ctx.xp, zp_c, ctx.ep)
    bC = fem.assemble_vector(fem.form(delta_p * ufl.TestFunction(ctx.Q) * ufl.dx)).array
    bh = fem.assemble_vector(fem.form(delta_p * ufl.TestFunction(Gh) * ufl.dx)).array
    sal = dict(xy=xy, tri=tri, t_h=(pasos + 1) * p.dt / 3600.0, t_im_h=(pasos_im) * p.dt / 3600.0,
               H=H, U=np.asarray(U), C=C, Cim=Cim,
               Cpozo=np.asarray(r["C"]) @ bC, Hpozo=np.asarray(r["h"]) @ bh, t_pasos_h=np.arange(1, Nt + 1) * p.dt / 3600.0,
               x=x, f=f, C_ref=p.C_ref, J=r["J"], J1=r["J1"], J2=J2, J3=J3, dt=p.dt, Q_max=float(p.Q_max),
               Nr=int(p.Nr), origen=json.dumps(de))
    if adjunto:
        g, adj = _solve_adjoint_grad(ctx, r)
        sal.update(grad=g, psiH=np.asarray(adj["psiH"])[pasos], psiC=a_P1(ctx, np.asarray(adj["psiC"])[pasos]),
                   psiIm=a_P1(ctx, _campo_desde_dual(ctx.Q, np.asarray(adj["psiIm"])[:, pasos])))
        print(f"[simplificada] |∇Ĵ|={np.linalg.norm(g):.4e} ∂Ĵ/∂z_p={g[-1]:.4e}", flush=True)
    os.makedirs(DIR_OUT, exist_ok=True)
    np.savez_compressed(os.path.join(DIR_OUT, f"{etiqueta}.npz"), **sal)
    print(f"[simplificada] guardado {etiqueta}.npz", flush=True)


def verificar(etiqueta, iterado, direccion, delta):
    from fenicsx.adjoint import _solve_forward
    x, de = leer_iterado(iterado)
    p, ctx, f = preparar()
    s = np.concatenate([np.full(len(x) - 1, float(p.Q_max)), [S_Z]])
    if direccion == "grad":
        g = np.load(os.path.join(DIR_OUT, f"{etiqueta}.npz"))["grad"]
        d = -s * g
    else:
        d = np.random.default_rng(42).standard_normal(len(x))
    d /= np.linalg.norm(d)
    delta = float(delta)
    res = dict(iterado=de, direccion=direccion, delta=delta, d=d.tolist(), s=s.tolist())
    for signo in (+1, -1):
        xs = x + signo * delta * s * d
        r = _solve_forward(ctx, xs[:-1], float(xs[-1]))
        res["mas" if signo > 0 else "menos"] = dict(J=float(r["J"]), J1=float(r["J1"]))
        print(f"[simplificada] {direccion} δ={delta:g} {'+' if signo > 0 else '−'}: J={r['J']:.10e} J1={r['J1']:.10e}", flush=True)
    os.makedirs(DIR_OUT, exist_ok=True)
    with open(os.path.join(DIR_OUT, f"verif_{etiqueta}_{direccion}_{delta:g}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if pos[0] == "campos":
        opc = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        campos(pos[1], pos[2], adjunto="--adjunto" in sys.argv, pasos_max=int(opc["pasos"]) if "pasos" in opc else None)
    elif pos[0] == "verificar":
        verificar(*pos[1:5])
    elif pos[0] == "js":
        from scripts.exportar_simplificada_js import main as js
        js(pos[1] if len(pos) > 1 else None)
