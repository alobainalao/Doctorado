"""
Módulos JS de la versión simplificada del avance 3 (src/slides-avance3s/), solo MRMT por bloques con elementos finitos:

  optimoBloque.js  trayectoria de la optimización de mrmt_block (serie 33: optimizacion_<V>/mrmt_block_progreso.jsonl y
                   su continuación mrmt_block_sigue, con el número de iteración acumulado), controles inicial y último,
                   pruebas de consistencia (zp200, reinicio25) y la verificación del gradiente en la configuración real
                   (data/output/simplificada/verif_*.json).
  camposBloque.js  bandas de nivel de h, |u|, C y C_im,r (y de ψ_h, ψ_C, ψ_im,r) del control inicial y del último
                   iterado, en las coordenadas de slides-avance3/escena.jsx (x' = (x − 816)/9.939, z' = (250.5 − z)/9.939),
                   y las series en el pozo; de data/output/simplificada/{inicial,optimo}.npz (scripts/exportar_simplificada.py).

    python3 scripts/exportar_simplificada.py js [ruta_de_la_presentacion]
"""
import glob
import json
import os

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = "nitrato_vscale_malla_eje_g1_prom"
DIR_OPT = os.path.join(CODIGO, "data", "output", f"optimizacion_{V}")
DIR_SIM = os.path.join(CODIGO, "data", "output", "simplificada")
PRES_DEF = os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc-simplificada")
ESC = 9.939                         # m por unidad de escena


def jsonl(ruta):
    return [json.loads(l) for l in open(ruta)] if os.path.exists(ruta) else []


def trayectoria():
    """Iterados de la corrida principal y de su continuación (la 1.ª línea de «sigue» repite el último iterado)."""
    tray = jsonl(os.path.join(DIR_OPT, "mrmt_block_progreso.jsonl"))
    # continuaciones en orden (sigue2 reanuda la que mató el OOM el 26-nov); cada una repite el último iterado de la anterior
    for etq in ("sigue", "sigue2", "sigue3"):
        base = tray[-1]["nit"] - 1
        sig = [dict(r, nit=r["nit"] + base) for r in jsonl(os.path.join(DIR_OPT, f"mrmt_block_{etq}_progreso.jsonl"))]
        tray += [r for r in sig if r["nit"] > tray[-1]["nit"]]
    return tray


def optimo_js():
    tray = trayectoria()
    ult, ini = tray[-1], tray[0]
    cons = {}
    for etq, previas in (("zp200", 0), ("reinicio25", 25)):
        t = jsonl(os.path.join(DIR_OPT, f"mrmt_block_{etq}_progreso.jsonl"))
        if t:
            base = t[-1]["nit"] - 1
            t += [dict(r, nit=r["nit"] + base) for r in jsonl(os.path.join(DIR_OPT, f"mrmt_block_{etq}_sigue_progreso.jsonl"))[1:]]
        if t:
            cons[etq] = dict(nit=t[-1]["nit"] + previas, J=t[-1]["J"], chi=t[-1]["chi"], zp=t[-1]["x"][-1],
                             historia=[[r["nit"] + previas, r["J"], r["chi"], r["x"][-1]] for r in t])
    ref = os.path.join(DIR_SIM, "inicial.npz")
    z = np.load(ref) if os.path.exists(ref) else None
    g = z["grad"] if z is not None and "grad" in z.files else None
    verif = []
    if g is not None:
        # gradiente de J1 = ∇Ĵ − ∇(J2 + J3), con J2 = c2 (z_p − z0)² y J3 = c3 Σ Q² |z_p − z0| (pesos de los valores en x⁰)
        z0 = json.load(open(os.path.join(CODIGO, "config", f"{V}.json")))["z0"]
        x0 = z["x"]; Q, zp = x0[:-1], x0[-1]
        c2, c3 = float(z["J2"]) / (zp - z0) ** 2, float(z["J3"]) / (np.sum(Q ** 2) * abs(zp - z0))
        g_eco = np.concatenate([2 * c3 * Q * abs(zp - z0), [2 * c2 * (zp - z0) + c3 * np.sum(Q ** 2) * np.sign(zp - z0)]])
        for ruta in sorted(glob.glob(os.path.join(DIR_SIM, "verif_inicial_*.json"))):
            v = json.load(open(ruta))
            sd = np.asarray(v["s"]) * np.asarray(v["d"])
            fd = (v["mas"]["J"] - v["menos"]["J"]) / (2 * v["delta"])
            fd1 = (v["mas"]["J1"] - v["menos"]["J1"]) / (2 * v["delta"])
            ad, ad1 = float(g @ sd), float((g - g_eco) @ sd)
            verif.append(dict(direccion=v["direccion"], delta=v["delta"], adjunto=ad, fd=fd, err=abs(ad - fd) / abs(fd),
                              adjunto_J1=ad1, fd_J1=fd1, err_J1=abs(ad1 - fd1) / abs(fd1)))
        g = g.tolist()
    terminos = {}
    for etq in ("inicial", "optimo"):
        r = os.path.join(DIR_SIM, f"{etq}.npz")
        if os.path.exists(r):
            z = np.load(r)
            terminos[etq] = dict(J=float(z["J"]), J1=float(z["J1"]), J2=float(z["J2"]), J3=float(z["J3"]),
                                 x=z["x"].tolist(), origen=json.loads(str(z["origen"])))
    return dict(nit=ult["nit"], J=ult["J"], chi=ult["chi"], zp=ult["x"][-1], Q=ult["x"][:-1],
                Q0=ini["x"][:-1], zp0=ini["x"][-1], J0=ini["J"],
                historia=[[r["nit"], r["J"], r["chi"], r["x"][-1]] for r in tray],
                consistencia=cons, verificacion=verif, grad0=g, terminos=terminos)


def ruta_svg(path):
    """Path de matplotlib (vértices en m) → «M…L…Z» en coordenadas de escena, con un decimal."""
    v = path.vertices
    cod = path.codes if path.codes is not None else np.full(len(v), 2)
    trozos, prev = [], None
    for (px, pz), c in zip(v, cod):
        if c == 79:
            trozos.append("Z"); prev = None
            continue
        q = (round((px - 816.0) / ESC, 1), round((250.5 - pz) / ESC, 1))
        if c == 1:
            trozos.append(f"M{q[0]:g} {q[1]:g}"); prev = q
        elif q != prev:
            trozos.append(f"L{q[0]:g} {q[1]:g}"); prev = q
    return "".join(trozos)


def bandas(tri, valores, niveles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.tricontourf(tri, valores, levels=niveles, extend="neither")
    if hasattr(cs, "get_paths"):                             # matplotlib ≥ 3.8: un Path compuesto por nivel
        out = [ruta_svg(p) for p in cs.get_paths()]
    else:
        out = ["".join(ruta_svg(p) for p in c.get_paths()) for c in cs.collections]
    plt.close(fig)
    return out


T_ESTADO = (120.0, 240.0, 480.0, 720.0)   # t [h] de los cuadros del estado: 5, 10, 20 y 30 días
T_ADJ = (120.0, 360.0, 600.0)             # t [h] de los cuadros de los adjuntos: 5, 15 y 25 días


def campos_js():
    import matplotlib.tri as mtri
    sal = {}
    # mismas bandas de C en las dos simulaciones (la lámina de resultados las compara): hasta el máximo de ambas
    cmax = max([float(np.load(os.path.join(DIR_SIM, f"{e}.npz"))["C"].max() * np.load(os.path.join(DIR_SIM, f"{e}.npz"))["f"] * 1000.0)
                for e in ("inicial", "optimo") if os.path.exists(os.path.join(DIR_SIM, f"{e}.npz"))] or [1.0])
    nc = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if v < cmax] + [cmax * 1.0001]
    for etq in ("inicial", "optimo"):
        r = os.path.join(DIR_SIM, f"{etq}.npz")
        if not os.path.exists(r):
            continue
        z = np.load(r)
        tri = mtri.Triangulation(z["xy"][:, 0], z["xy"][:, 1], z["tri"])
        mgL = float(z["f"]) * 1000.0                         # C del modelo → mg/L de nitrato
        t_h = list(np.round(z["t_h"], 6))
        ke = [t_h.index(t) for t in T_ESTADO]
        H, U, C = z["H"][ke], z["U"][ke] * 86400.0, z["C"][ke] * mgL
        d = dict(t_h=list(T_ESTADO), x=z["x"].tolist(), Cpozo=(z["Cpozo"] * mgL).tolist(), Hpozo=z["Hpozo"].tolist(),
                 t_pasos_h=z["t_pasos_h"].tolist(), Cmax=float(C.max()), Cim_max=[float(c.max() * mgL) for c in z["Cim"][:, ke]],
                 J=float(z["J"]), J1=float(z["J1"]), J2=float(z["J2"]), J3=float(z["J3"]))
        nh = np.linspace(np.floor(H.min()), np.ceil(H.max()), 13)
        d["H"] = dict(niveles=nh.tolist(), unidad="m", bandas=[bandas(tri, h, nh) for h in H])
        nu = [v for v in np.logspace(-3, 3, 13) if v < U.max()] + [U.max() * 1.0001]
        d["U"] = dict(niveles=nu, unidad="m/día", log=True, bandas=[bandas(tri, np.clip(u, nu[0], None), nu) for u in U])
        d["C"] = dict(niveles=nc, unidad="mg/L", bandas=[bandas(tri, c, nc) for c in C])
        if "psiH" in z.files:
            # adjuntos: cada uno entre su máximo |ψ| en los cuadros elegidos (adimensional), bandas simétricas en escala logarítmica
            ka = [t_h.index(t) for t in T_ADJ]
            pos = [0.01, 0.03, 0.1, 0.3, 1.0001]
            niv = [-v for v in pos[::-1]] + pos

            def adj(campo):
                campo = campo[ka]
                m = float(np.max(np.abs(campo)))
                return dict(niveles=niv, bandas=[bandas(tri, c / m, niv) for c in campo])
            d["t_adj_h"] = list(T_ADJ)
            d["psiH"], d["psiC"] = adj(z["psiH"]), adj(-z["psiC"])
            d["psiIm"] = [adj(-c) for c in z["psiIm"]]
        sal[etq] = d
    return sal


def main(pres=None):
    pres = pres or PRES_DEF
    dest = os.path.join(pres, "src", "slides-avance3s")
    o = optimo_js()
    with open(os.path.join(dest, "optimoBloque.js"), "w") as fh:
        fh.write("// Generado por codigo/scripts/exportar_simplificada_js.py (serie 33, MRMT por bloques, FEniCSx). No editar a mano.\n")
        fh.write("export const BLOQUE = " + json.dumps(o, separators=(",", ":")) + ";\n")
    print(f"optimoBloque.js: iterado {o['nit']}, Ĵ={o['J']:.6f}, z_p={o['zp']:.2f}, verificaciones {len(o['verificacion'])}")
    c = campos_js()
    if c:
        ruta = os.path.join(dest, "camposBloque.js")
        with open(ruta, "w") as fh:
            fh.write("// Generado por codigo/scripts/exportar_simplificada_js.py desde data/output/simplificada/*.npz "
                     "(MRMT por bloques, FEniCSx, configuración de la optimización). No editar a mano.\n")
            fh.write("export const CAMPOS = " + json.dumps(c, separators=(",", ":")) + ";\n")
        print(f"camposBloque.js: {', '.join(c)} ({os.path.getsize(ruta) / 1e6:.1f} MB)")
