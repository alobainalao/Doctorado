"""
Ilustración en 2D del optimizador de las corridas para la presentación (avance3, lámina «Del punto inicial al óptimo»):
trust-constr (punto interior, BFGS, iterados admisibles) con las opciones por defecto de la serie 33 (barrera inicial
0.1, radio 1, variables en [0, 1]) sobre un problema cuadrático pequeño con la misma estructura que U_ad: cotas de
caja más dos restricciones lineales acopladas. Cada iterado es el que devuelve scipy (no se simula nada); χ se calcula
con el mismo proyector certificado que las corridas (funtions/optimizador.py:proyectar).

    python scripts/exportar_trust_constr_2d.py [ruta_de_la_presentacion]      # fenics_env (SciPy 1.17)

Escribe <presentacion>/src/slides-avance3/trustConstr2d.js
"""
import inspect
import json
import os
import sys
import textwrap

import numpy as np
import scipy
from scipy.optimize import minimize, Bounds, LinearConstraint, BFGS
import scipy.optimize._trustregion_constr.equality_constrained_sqp as _sqp
import scipy.optimize._trustregion_constr.tr_interior_point as _tip

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc")
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
from funtions.optimizador import proyectar  # noqa: E402

# Ĵ(u) = ½ (u − m)ᵀ H (u − m): elipses con el eje mayor a TH grados y razón de curvaturas K; mínimo m fuera de U_ad.
TH, K = -30.0, 6.0
c, s = np.cos(np.deg2rad(TH)), np.sin(np.deg2rad(TH))
R = np.array([[c, -s], [s, c]])
H = R @ np.diag([1.0, K]) @ R.T
M = np.array([1.0, 0.75])
# U_ad: 0 ≤ u ≤ 1 (cotas, como 0 ≤ Q_n ≤ Q̄_n) y A u ≤ b (acopladas, como el tanque: cada fila mezcla las dos variables)
LB, UB = np.zeros(2), np.ones(2)
A = np.array([[1.0, 1.0], [-0.6, 1.0]])
B = np.array([1.25, 0.62])
X0 = np.array([0.12, 0.10])
LIN = [(A, np.full(2, -10.0), B)]          # cota inferior finita: SLSQP no acepta −∞


def J(u):
    return 0.5 * (u - M) @ H @ (u - M)


def grad(u):
    return H @ (u - M)


def c_todas(u):
    """Las seis desigualdades c_i(u) ≥ 0 tal como las ve trust-constr (las cotas también llevan holgura)."""
    return np.r_[u - LB, UB - u, B - A @ u]


def camino_central(mu, u):
    """Mínimo del subproblema de barrera Ĵ(u) − μ Σ log c_i(u) (Newton amortiguado desde u, que debe ser interior)."""
    G = np.vstack([np.eye(2), -np.eye(2), -A])                    # ∇c_i (filas)
    for _ in range(200):
        c = c_todas(u)
        g = grad(u) - mu * G.T @ (1 / c)
        Hb = H + mu * G.T @ np.diag(1 / c ** 2) @ G
        d = -np.linalg.solve(Hb, g)
        t = 1.0
        while np.any(c_todas(u + t * d) <= 0) or (J(u + t * d) - mu * np.sum(np.log(c_todas(u + t * d)))
                                                   > J(u) - mu * np.sum(np.log(c)) + 1e-4 * t * g @ d):
            t *= 0.5
        u = u + t * d
        if np.linalg.norm(t * d) < 1e-13:
            break
    return u


def chi(u):
    z, residuo = proyectar(u - grad(u), LB, UB, LIN)
    return float(np.max(np.abs(u - z))), float(residuo)


# Registro de cada paso de prueba del SQP interno de trust-constr (equality_constrained_sqp, SciPy 1.17): ρ = reducción
# real / predicha de la función de mérito, radio Δ antes y después y norma escalada del paso. scipy no los devuelve:
# se copia la función con dos líneas de registro añadidas (no cambia ningún cálculo; los iterados salen idénticos).
ENSAYOS = []


def instrumentar():
    src = textwrap.dedent(inspect.getsource(_sqp.equality_constrained_sqp))
    ini, fin = "    # Normal Step - `dn`\n", "    # Update iteration\n"
    assert src.count(ini) == 1 and src.count(fin) == 1, "cambió equality_constrained_sqp: revisar el registro"
    src = src.replace(ini, "    _delta0 = trust_radius\n" + ini).replace(fin, (
        "    _ENSAYOS.append(dict(z=x.copy(), z_prueba=x_next.copy(), rho=float(reduction_ratio), delta=float(_delta0),\n"
        "                         delta_nuevo=float(trust_radius), norma_d=float(norm(d)), real=float(actual_reduction),\n"
        "                         predicha=float(predicted_reduction)))\n") + fin)
    ns = dict(vars(_sqp), _ENSAYOS=ENSAYOS)
    exec(compile(src, _sqp.__file__, "exec"), ns)
    _tip.equality_constrained_sqp = ns["equality_constrained_sqp"]


def elipse(u, delta, n=96):
    """Borde de la región de confianza en u: ‖d‖² + Σ (∇c_i·d / c_i(u))² = Δ² (paso escalado (d_x, S⁻¹d_s) con d_s = ∇c d_x,
    que es lo que impone el SQP en un punto admisible con holguras s = c(u))."""
    G = np.vstack([np.eye(2), -np.eye(2), -A])
    Mt = np.eye(2) + G.T @ np.diag(1 / c_todas(u) ** 2) @ G
    lam, V = np.linalg.eigh(Mt)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = delta * (V @ np.diag(1 / np.sqrt(lam)) @ np.vstack([np.cos(th), np.sin(th)])).T
    return (u + d), Mt


if __name__ == "__main__":
    instrumentar()
    reg = []
    res = minimize(J, X0, jac=grad, hess=BFGS(), method="trust-constr", bounds=Bounds(LB, UB, keep_feasible=True),
                   constraints=[LinearConstraint(A, -np.inf, B, keep_feasible=True)],
                   callback=lambda x, st: reg.append((x.copy(), st.fun, st.tr_radius, st.barrier_parameter)),
                   options=dict(gtol=1e-10, xtol=1e-10, maxiter=300))
    its, previo = [], None
    for k, (x, f, tr, mu) in enumerate(reg, 1):
        x_chi, residuo = chi(x)
        # 'paso': el iterado se movió; 'barrera': mismo punto, μ ← μ/5 (empieza el siguiente subproblema de barrera)
        tipo = "inicio" if previo is None else ("paso" if np.linalg.norm(x - previo[0]) > 1e-12 else
                                                 ("barrera" if mu < previo[3] else "rechazo"))
        its.append(dict(k=k, x=np.round(x, 6).tolist(), J=round(float(f), 6), chi=float(f"{x_chi:.3e}"),
                        residuo=float(f"{residuo:.1e}"), tr=float(f"{tr:.4g}"), mu=float(f"{mu:.3g}"),
                        g=np.round(grad(x), 5).tolist(), tipo=tipo))
        previo = (x, f, tr, mu)
    # ρ y Δ de cada paso aceptado (el ensayo cuyo punto de prueba es el iterado); los rechazados (ρ < 10⁻⁸) se cuentan
    rechazos = sum(e["rho"] < 1e-8 for e in ENSAYOS)
    for r in its:
        if r["tipo"] != "paso":
            continue
        e = min(ENSAYOS, key=lambda e: np.linalg.norm(e["z_prueba"][:2] - np.asarray(r["x"])))
        assert np.linalg.norm(e["z_prueba"][:2] - np.asarray(r["x"])) < 1e-5 and e["rho"] >= 1e-8
        r.update(rho=round(e["rho"], 4), delta=float(f"{e['delta']:.4g}"), delta_nuevo=float(f"{e['delta_nuevo']:.4g}"),
                 norma_escalada=round(e["norma_d"], 4))
    # lado acoplado activo en û (u₁ + u₂ ≤ 1.25): holgura c₁ y −∇Ĵ en componentes normal (n = (1,1)/√2, hacia afuera) y
    # tangencial (t = (−1,1)/√2); para cada paso, ángulo entre el paso y el lado [grados]
    n_, t_ = np.array([1, 1]) / np.sqrt(2), np.array([-1, 1]) / np.sqrt(2)
    # (con los iterados sin redondear); fracción a la frontera: menor cociente s_nueva/s_vieja de un paso (τ = 0.995 ⇒ ≥ 0.005)
    frac_min = 1.0
    for j, r in enumerate(its):
        u = reg[j][0]
        g = -grad(u)
        r.update(c1=float(f"{B[0] - A[0] @ u:.4g}"), gn=round(float(g @ n_), 5), gt=float(f"{g @ t_:.3g}"))
        if r["tipo"] == "paso":
            d = u - reg[j - 1][0]
            r["angulo_lado"] = round(float(np.degrees(np.arctan2(abs(d @ n_), abs(d @ t_)))), 1)
            frac_min = min(frac_min, float(np.min(c_todas(u) / c_todas(reg[j - 1][0]))))
    x = res.x
    activa = (A @ x - B > -1e-6).tolist()
    # camino central: mínimo de cada subproblema de barrera, para los μ que usó scipy
    mus = sorted({r["mu"] for r in its}, reverse=True)
    u, central = np.array([0.4, 0.3]), []
    for mu in mus:
        u = camino_central(mu, u)
        # en el camino central, μ/c₁ estima el multiplicador del lado activo (KKT perturbada λ₁ s₁ = μ)
        central.append(dict(mu=mu, x=np.round(u, 6).tolist(), J=round(float(J(u)), 6),
                            c1=float(f"{B[0] - A[0] @ u:.4g}"), mu_c1=round(float(mu / (B[0] - A[0] @ u)), 4)))
    # primer paso: norma del paso escalado (d_x, S⁻¹ d_s) con que trust-constr mide la región de confianza
    d = np.asarray(its[1]["x"]) - X0
    escalado = float(np.sqrt(d @ d + np.sum(((c_todas(X0 + d) - c_todas(X0)) / c_todas(X0)) ** 2)))
    # regiones de confianza de los dos primeros pasos, dibujadas en u (Δ₀ = 1 en u₀; Δ = 7 en u₁ tras ρ ≥ 0.9)
    regiones = []
    for j in (0, 1):
        u, dl = np.asarray(its[j]["x"]), its[j + 1]["delta"]
        borde, Mt = elipse(u, dl)
        semiejes = sorted(dl / np.sqrt(np.linalg.eigvalsh(Mt)))
        regiones.append(dict(desde=j + 1, delta=dl, borde=np.round(borde, 5).tolist(), semiejes=np.round(semiejes, 4).tolist()))
    out = dict(scipy=scipy.__version__, th=TH, K=K, m=M.tolist(), A=A.tolist(), b=B.tolist(), x0=X0.tolist(),
               poligono=[[0, 0], [1, 0], [1, 0.25], [0.39375, 0.85625], [0, 0.62]],
               optimo=dict(x=np.round(x, 6).tolist(), J=round(float(res.fun), 6), nit=int(res.nit),
                           lambda_lin=np.round(res.v[0], 4).tolist(), activa=activa, g=np.round(grad(x), 5).tolist(),
                           chi=float(f"{chi(x)[0]:.2e}"), msg=str(res.message)),
               iteraciones=its, camino_central=central, regiones=regiones, rechazos=int(rechazos),
               frac_min=round(frac_min, 4),
               # constantes de SciPy (tr_interior_point.py y equality_constrained_sqp.py)
               reglas=dict(tau=0.995, mu_factor=0.2, delta_barrera=5, rho_grande=0.9, rho_medio=0.3, rho_minimo=1e-8,
                           crece_grande=7.0, crece_medio=2.0),
               primer_paso=dict(norma=round(float(np.linalg.norm(d)), 4), norma_escalada=round(escalado, 3),
                                holguras_x0=np.round(c_todas(X0), 3).tolist()))
    ruta = os.path.join(PRES, "src", "slides-avance3", "trustConstr2d.js")
    with open(ruta, "w") as fh:
        fh.write("// Generado por codigo/scripts/exportar_trust_constr_2d.py: iterados reales de trust-constr sobre un\n"
                 "// problema ilustrativo en 2D (cuadrático, cotas + 2 restricciones acopladas). No editar a mano.\n"
                 f"export const TC2D = {json.dumps(out, ensure_ascii=False)};\n")
    print(f"{len(its)} iteraciones, x* = {x}, λ = {res.v[0]}, Ĵ* = {res.fun:.6f} → {ruta}")
    for r in its:
        print(r["k"], r["tipo"], r["x"], r["J"], r["chi"], r["tr"], r["mu"], r.get("rho"), r.get("delta"), r.get("delta_nuevo"),
              r.get("norma_escalada"), "c1", r["c1"], "gn", r["gn"], "gt", r["gt"], r.get("angulo_lado"))
    for r in central:
        print("central", r)
    print("primer paso", out["primer_paso"], "rechazos", rechazos, "ensayos", len(ENSAYOS))
    for g in regiones:
        print("region", g["desde"], g["delta"], g["semiejes"])
