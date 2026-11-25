"""
Discrepancia RBF-FD vs elementos finitos en TODO el horizonte (láminas 9 y 10 del avance 3, revisión de la versión 03).

La lámina 9 daba la L² relativa con pesos de área nodal en un solo instante, t = T. Aquí, con los mismos pesos y los
mismos nodos, se pasa al tiempo:
  * e(t_n) = ‖v_RBF(t_n) − v_EF(t_n)‖_h / ‖v_EF(t_n)‖_h en los 144 instantes guardados (series de comparacion/metricas.json,
    las mismas que dibujan los videos; también existen para las corridas archivadas, cuyos campos no se guardaron);
  * máximo en el tiempo, max_n e(t_n): versión discreta de la norma L^∞(0,T; L²(Ω)) relativa;
  * espacio-tiempo L²(0,T; L²(Ω)) relativa (necesita los campos: solo los casos con simulation_results.npz);
  * prueba de desfase: e_ST(τ) entre v_RBF(·, t+τ) y v_EF(·, t), con interpolación lineal en el tiempo, para ver si la
    discrepancia es un retraso (mínimo en τ ≠ 0) o una diferencia de forma (mínimo en τ = 0). min_τ e_ST(τ) no es una norma
    de la diferencia de soluciones sincronizadas: es una medida de parecido salvo traslación temporal;
  * lo mismo para la concentración captada por el pozo, c_w(t) = ∫ g_ε C dΩ (serie Cpozo de metricas.json).
Solo lectura de los resultados.

    nice -n 19 python scripts/metricas_tiempo.py [ruta_de_la_presentacion]      # fenics_env

Escribe data/output/experimentos/metricas_tiempo.json y <presentacion>/src/slides-avance3/discrepanciaTiempo.js
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

COD = "/home/alex/dev/doctorado/codigo"
EXP = os.path.join(COD, "data", "output", "experimentos")
ARCH = os.path.join(COD, "archivo", "resultados_json", "codigo", "experimentos")
PRES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(COD), "presentaciones", "avance1-doc")

# columnas de la lámina 9 (mismo caso ADR con fuente y pozo) y casos de la tabla de sensibilidad de la lámina 10
COLUMNAS = [("anterior", os.path.join(ARCH, "A_terminos", "A3_fuente_pozo")),
            ("nuevo_nube", os.path.join(ARCH, "M_modelo_nuevo", "M0_adr")),
            ("nuevo_conos", os.path.join(EXP, "M_modelo_nuevo", "M0_adr"))]
SENS = {"nube": [os.path.join(ARCH, "M_modelo_nuevo", m) for m in ("M0_adr", "M1_mrmt_semi", "M2_mrmt_block")],
        "conos": [os.path.join(EXP, "M_modelo_nuevo", m) for m in ("M0_adr", "M1_mrmt_semi", "M2_mrmt_block")],
        "eje": [os.path.join(EXP, "MJ_malla_eje", m) for m in ("MJ0_adr", "MJ1_mrmt_semi", "MJ2_mrmt_block")],
        "eje05": [os.path.join(EXP, "ME_malla_eje05", m) for m in ("ME0_adr", "ME1_mrmt_semi", "ME2_mrmt_block")]}
TAU_MAX, DTAU = 5.0, 0.05            # barrido de desfase: ±5 pasos de 5 h, cada 15 min


def series(caso):
    s = json.load(open(os.path.join(caso, "comparacion", "metricas.json")))["series"]
    return {k: np.asarray(v, float) for k, v in s.items()}


def resumen_tiempo(caso):
    """Cifras en t = T y máximas en el tiempo a partir de las series guardadas (todas las corridas las tienen)."""
    s = series(caso)
    t, eC, eV = s["t_dias"], s["C_L2_rel"], s["V_L2_rel"]
    rI = np.abs(s["masa_bfr"] / s["masa_fem"] - 1)          # ∫C dΩ (sin porosidad)
    cb, cf = s["Cpozo_bfr"], s["Cpozo_fem"]
    return dict(C_T=eC[-1], C_max=eC.max(), C_tmax=t[eC.argmax()], V_T=eV[-1], V_max=eV.max(), V_tmax=t[eV.argmax()],
                I_T=rI[-1], I_max=rI.max(), I_tmax=t[rI.argmax()], H_T=s["H_rms_m"][-1], H_max=s["H_rms_m"].max(),
                cw_T=abs(cb[-1] / cf[-1] - 1), cw_L2t=np.sqrt(((cb - cf) ** 2).sum() / (cf ** 2).sum()))


def interp_t(A, x):
    """A[n] en el índice fraccionario x (interpolación lineal entre instantes guardados)."""
    i = int(np.floor(x)); a = x - i
    return A[i] if a < 1e-12 else (1 - a) * A[i] + a * A[i + 1]


def desfase(B, F, w):
    """e_ST(τ) = (Σ_n ‖B(t_n+τ) − F(t_n)‖²_h / Σ_n ‖F(t_n)‖²_h)^½ en la ventana común n ∈ [K, N−1−K], τ en pasos."""
    N, K = len(F), int(np.ceil(TAU_MAX))
    ns = np.arange(K, N - K)
    den = sum((w * F[n] ** 2).sum() for n in ns)
    taus = np.round(np.arange(-TAU_MAX, TAU_MAX + 1e-9, DTAU), 4)
    e = [np.sqrt(sum((w * (interp_t(B, n + tau) - F[n]) ** 2).sum() for n in ns) / den) for tau in taus]
    return taus, np.asarray(e), ns


def desfase_serie(b, f):
    """Lo mismo para una serie escalar (c_w): ‖b(t+τ) − f(t)‖_{L²(t)} / ‖f‖_{L²(t)}."""
    N, K = len(f), int(np.ceil(TAU_MAX))
    ns = np.arange(K, N - K)
    taus = np.round(np.arange(-TAU_MAX, TAU_MAX + 1e-9, DTAU), 4)
    e = [np.sqrt(sum((interp_t(b, n + tau) - f[n]) ** 2 for n in ns) / sum(f[n] ** 2 for n in ns)) for tau in taus]
    return taus, np.asarray(e)


def campos(caso):
    """C y |u| de ambos backends en los nodos comunes, con los pesos de área nodal (como experimentos/comparar.py)."""
    cfg = json.load(open(os.path.join(caso, "bfr", "config.json")))
    m = pickle.load(open(os.path.join(COD, cfg["mesh_pkl"]), "rb"))
    todos = np.asarray(m["nodes"], float)
    fant = np.zeros(len(todos), bool)
    for k, idx in m["groups"].items():
        if k.startswith("ghost"):
            fant[np.asarray(idx, int)] = True
    keep = np.nonzero(~fant)[0]; o2n = np.full(len(todos), -1); o2n[keep] = np.arange(len(keep))
    nodes = todos[keep]; tris = o2n[np.asarray(m["triangles"], np.int64)]
    p = nodes[tris]
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    w = np.bincount(tris.ravel(), weights=np.repeat(area / 3, 3), minlength=len(nodes))
    zb = np.load(os.path.join(caso, "bfr", "simulation_results.npz"), allow_pickle=True)
    zf = np.load(os.path.join(caso, "fenicsx", "simulation_results.npz"), allow_pickle=True)
    dist, idx = cKDTree(zf["nodes"]).query(nodes); assert dist.max() < 1e-6
    Ub = zb["U"]; Ub = (Ub[:, 0] if Ub.ndim == 4 else Ub)[:, keep]
    return dict(w=w, dt=float(zb["dt"]), C=(zb["C"][:, 0][:, keep], zf["C"][:, 0][:, idx]),
                V=(np.linalg.norm(Ub, axis=-1), np.linalg.norm(zf["U"][:, idx], axis=-1)))


if __name__ == "__main__":
    out = {"columnas": {}, "sensibilidad": {}, "espacio_tiempo": {}, "desfase": {}}
    for nom, caso in COLUMNAS:
        out["columnas"][nom] = resumen_tiempo(caso)
    for malla, casos in SENS.items():
        out["sensibilidad"][malla] = [resumen_tiempo(c) for c in casos]
    # espacio-tiempo y desfase: solo donde hay campos (malla con conos; ADR y MRMT)
    for caso in SENS["conos"] + SENS["eje"]:
        nom = os.path.basename(caso)
        D = campos(caso); w = D["w"]
        st = {}
        for k in ("C", "V"):
            B, F = D[k]
            st[k] = float(np.sqrt(sum((w * (B[n] - F[n]) ** 2).sum() for n in range(len(F)))
                                  / sum((w * F[n] ** 2).sum() for n in range(len(F)))))
        # norma L^∞(0,T; L²) relativa en sentido estricto: max_n ‖d_n‖ / max_n ‖v_EF,n‖ (la lámina usa max_n de la razón)
        for k in ("C", "V"):
            B, F = D[k]
            st[k + "_linf_estricta"] = float(max(np.sqrt((w * (B[n] - F[n]) ** 2).sum()) for n in range(len(F)))
                                             / max(np.sqrt((w * F[n] ** 2).sum()) for n in range(len(F))))
        out["espacio_tiempo"][nom] = st
        if nom != "M0_adr":
            continue
        for k in ("C", "V"):
            taus, e, ns = desfase(*D[k], w)
            i0 = int(np.argmin(np.abs(taus))); im = int(np.argmin(e))
            out["desfase"][k] = dict(tau_pasos=taus.tolist(), e=e.tolist(), tau_opt_h=float(taus[im] * D["dt"] / 3600),
                                     e_opt=float(e[im]), e_0=float(e[i0]), ventana_dias=[float((ns[0] + 1) * D["dt"] / 86400),
                                                                                       float((ns[-1] + 1) * D["dt"] / 86400)])
        s = series(caso)
        taus, e = desfase_serie(s["Cpozo_bfr"], s["Cpozo_fem"])
        i0 = int(np.argmin(np.abs(taus))); im = int(np.argmin(e))
        # ¿retraso o amplitud? c_w^RBF escalada por el mejor factor a (sin desfase) y con el mejor par (τ, a)
        cb, cf = s["Cpozo_bfr"], s["Cpozo_fem"]; K = int(np.ceil(TAU_MAX)); ns = np.arange(K, len(cf) - K)
        a = (cb[ns] @ cf[ns]) / (cb[ns] @ cb[ns])
        e_a = np.linalg.norm(a * cb[ns] - cf[ns]) / np.linalg.norm(cf[ns])
        mejor = min(((np.linalg.norm(aa * bb - cf[ns]) / np.linalg.norm(cf[ns]), tau, aa) for tau in taus
                     for bb in [np.array([interp_t(cb, n + tau) for n in ns])] for aa in [(bb @ cf[ns]) / (bb @ bb)]))
        out["desfase"]["cw"] = dict(tau_pasos=taus.tolist(), e=e.tolist(), tau_opt_h=float(taus[im] * D["dt"] / 3600),
                                    e_opt=float(e[im]), e_0=float(e[i0]), amplitud=float(a), e_amplitud=float(e_a),
                                    e_tau_amp=float(mejor[0]), tau_amp_h=float(mejor[1] * D["dt"] / 3600), amp_tau=float(mejor[2]))
    conv = lambda o: {k: conv(v) for k, v in o.items()} if isinstance(o, dict) else \
        [conv(v) for v in o] if isinstance(o, list) else float(o) if isinstance(o, (np.floating, float)) else o
    out = conv(out)
    json.dump(out, open(os.path.join(EXP, "metricas_tiempo.json"), "w"), indent=1)

    # presentación: curvas e(t) de las tres columnas, cifras y barrido de desfase
    r4 = lambda v: [float(f"{x:.4g}") for x in v]
    cur = {nom: dict(C=r4(series(c)["C_L2_rel"]), V=r4(series(c)["V_L2_rel"]),
                     I=r4(np.abs(series(c)["masa_bfr"] / series(c)["masa_fem"] - 1))) for nom, c in COLUMNAS}
    t = r4(series(COLUMNAS[2][1])["t_dias"])
    js = dict(t_dias=t, curvas=cur, columnas=out["columnas"], sensibilidad=out["sensibilidad"],
              espacio_tiempo=out["espacio_tiempo"],
              desfase={k: dict(v, tau_pasos=r4(v["tau_pasos"][::4]), e=r4(v["e"][::4])) for k, v in out["desfase"].items()})
    ruta = os.path.join(PRES, "src", "slides-avance3", "discrepanciaTiempo.js")
    with open(ruta, "w") as fh:
        fh.write("// Generado por codigo/scripts/metricas_tiempo.py: discrepancia RBF-FD vs EF en los 144 instantes (Δt = 5 h, 30 d),\n"
                 "// de comparacion/metricas.json (columnas 1–2: corridas archivadas, solo series) y de los campos guardados\n"
                 "// (espacio-tiempo y desfase, malla con conos). No editar a mano.\n"
                 f"export const DISC = {json.dumps(js, ensure_ascii=False)};\n")
    for nom, r in out["columnas"].items():
        print(nom, {k: round(100 * v, 3) if not k.endswith("tmax") and not k.startswith("H") else round(v, 3) for k, v in r.items()})
    for malla, rs in out["sensibilidad"].items():
        print(malla, [(round(100 * r["C_max"], 2), round(100 * r["V_max"], 2), round(100 * r["I_max"], 3), round(100 * r["H_max"], 2)) for r in rs])
    print("espacio-tiempo", {k: {a: round(100 * b, 2) for a, b in v.items()} for k, v in out["espacio_tiempo"].items()})
    for k, v in out["desfase"].items():
        print("desfase", k, "τ* =", round(v["tau_opt_h"], 2), "h  e(τ*) =", round(100 * v["e_opt"], 3), "%  e(0) =", round(100 * v["e_0"], 3), "%")
    cw = out["desfase"]["cw"]
    print("c_w: amplitud", round(cw["amplitud"], 4), "→", round(100 * cw["e_amplitud"], 3), "%;  τ + amplitud:", round(cw["tau_amp_h"], 2), "h,",
          round(cw["amp_tau"], 4), "→", round(100 * cw["e_tau_amp"], 3), "%")
    print("→", ruta)
