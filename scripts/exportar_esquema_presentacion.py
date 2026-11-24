"""
Esquema animado de la apertura del avance 3 (A3D01): porosidad, líneas de corriente y pluma de nitrato sin bombeo y
con el óptimo ADR (FEniCSx, malla con eje de 1 m, config nitrato_vscale_malla_eje_g1, iterado de
optimizacion_<variante>/instantanea_23nov/adr_progreso.json, el mismo que muestra la presentación).

    python scripts/exportar_esquema_presentacion.py [--simular] [--estados=RUTA] [ruta_presentacion]   # fenics_env

--simular corre los dos directos (Q = 0 y Q óptimo en su z_p) y guarda los campos en
data/output/presentacion_esquema/campos.npz; sin él se reusa ese archivo. --estados: GeoJSON de los estados de México
(p. ej. github.com/angelnmara/geojson, mexicoHigh.json) para el mapa de ubicación. Escribe
<presentacion>/src/slides-avance3/esquema.js con contornos y polilíneas ya en las coordenadas de escena.jsx.
"""
import json
import os
import sys

import numpy as np

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO not in sys.path:
    sys.path.insert(0, CODIGO)
os.chdir(CODIGO)
SIMULAR = "--simular" in sys.argv[1:]
ESTADOS = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--estados=")), None)
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
PRES = _ARGS[0] if _ARGS else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc")

VARIANTE = "nitrato_vscale_malla_eje_g1"
DIR_OPT = os.path.join(CODIGO, "data", "output", f"optimizacion_{VARIANTE}")
X0_OPT = os.path.join(DIR_OPT, "instantanea_23nov", "adr_progreso.json")
NPZ = os.path.join(CODIGO, "data", "output", "presentacion_esquema", "campos.npz")

# Coordenadas de escena.jsx: x' = (x − 816)/9.939, z' = (250.5 − z)/9.939 (dominio en [0, 1000] × [0, 296]).
ESC = 9.939
NIV_PHI = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NIV_C = [10.0, 50.0]                 # mg/L de nitrato: extensión visible y valor guía OMS
PASO_C = 2                           # un cuadro de la pluma cada 2 pasos (20 h)


# Zona de estudio digitalizada de tesis/Figures/ubicacion.png (píxeles de la ampliación de Sabinas, norte arriba): sondeos
# 1–6, traza del corte (recta roja, ajuste x = a + b·y) y poblados (posición de su etiqueta en el mapa base). La escala sale
# de proyectar los sondeos 2 y 6 sobre la recta: su distancia es la longitud del dominio, 10 755 − 816 = 9 939 m.
SONDEOS_PX = {1: (1039.4, 160.2), 2: (695.5, 201.4), 3: (590.5, 335.1), 4: (590.5, 432.7), 5: (695.5, 482.7), 6: (493.0, 527.7)}
RECTA_PX = (824.00, -0.6196, 149.0, 592.0)          # x = a + b·y para y en [y0, y1]
POBLADOS_PX = {"Cloete": (545, 163), "Agujita": (597, 268), "Sabinas": (725, 453)}
ALINEADOS = {6: 816.0, 4: 4354.0, 3: 6716.0, 2: 10755.0}   # sondeo de la figura → x del pozo de campo en el corte
SABINAS = (-101.12, 27.85)                                 # lon, lat


def zona():
    """Zona de estudio en metros (E, N) con origen en el sondeo 6 y la escala de la recta."""
    a, b, y0, y1 = RECTA_PX
    p0, u = np.array([a + b * y0, y0]), np.array([b, 1.0]) / np.hypot(b, 1.0)
    s = {k: float(np.dot(np.array(v) - p0, u)) for k, v in SONDEOS_PX.items()}
    m_px = (ALINEADOS[2] - ALINEADOS[6]) / abs(s[6] - s[2])
    o = np.array(SONDEOS_PX[6])
    en = lambda q: [round(float((q[0] - o[0]) * m_px), 1), round(float(-(q[1] - o[1]) * m_px), 1)]
    pie = {k: en(p0 + s[k] * u) for k in ALINEADOS}       # pie de la perpendicular sobre la recta
    return dict(m_por_px=round(m_px, 3),
                sondeos=[dict(n=k, e=en(v)[0], nn=en(v)[1], alineado=k in ALINEADOS, x_corte=ALINEADOS.get(k),
                              pie=pie.get(k)) for k, v in SONDEOS_PX.items()],
                recta=[en((a + b * y0, y0)), en((a + b * y1, y1))],
                poblados=[dict(n=k, e=en(v)[0], nn=en(v)[1]) for k, v in POBLADOS_PX.items()])


def mapa(ruta):
    """Estados de México simplificados en una proyección equirectangular (cos 25°), en unidades de 1/10 de grado."""
    def dp(p, tol):                                   # Douglas–Peucker
        if len(p) < 3:
            return p
        a, b = p[0], p[-1]
        n = np.hypot(*(b - a)) or 1e-12
        dist = np.abs(np.cross(b - a, p - a)) / n
        i = int(np.argmax(dist))
        if dist[i] <= tol:
            return np.array([a, b])
        return np.vstack([dp(p[:i + 1], tol)[:-1], dp(p[i:], tol)])

    d = json.load(open(ruta))
    k, c = 10.0, np.cos(np.radians(25.0))
    xy = lambda q: ((q[:, 0] + 118.0) * c * k, (33.0 - q[:, 1]) * k)
    estados = []
    for f in d["features"]:
        g = f["geometry"]
        polis = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        partes = []
        for pol in polis:
            q = np.asarray(pol[0], float)
            area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
            if area < 0.02:                            # islas pequeñas fuera
                continue
            i = int(np.argmax(np.hypot(*(q - q[0]).T)))  # anillo cerrado: se parte en el punto más lejano
            q = np.vstack([dp(q[:i + 1], 0.03)[:-1], dp(q[i:], 0.03)])
            x, y = xy(q)
            partes.append("M" + "L".join(f"{u:.1f},{v:.1f}" for u, v in zip(x, y)) + "Z")
        estados.append(dict(id=f["properties"]["id"], nombre=f["properties"]["name"], d="".join(partes)))
    sx, sy = xy(np.array([SABINAS]))
    return dict(estados=estados, sabinas=[round(float(sx[0]), 1), round(float(sy[0]), 1)],
                ancho=round((118.0 - 86.5) * c * k, 1), alto=round((33.0 - 14.3) * k, 1))


def simular():
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    cfg = json.load(open(os.path.join(CODIGO, "config", f"{VARIANTE}.json")))
    cfg.update({"metodo": "fenicsx", "run_type": "optimization", "model": "adr", "pre": True,
                "save_dat": False, "animate": False, "postproc": False, "extract_C": False})
    RUNTIME.params = Parameters(config=cfg)
    p = RUNTIME.get()

    from dolfinx import fem
    from fenicsx.mesh import build_mesh_from_bfr_pkl
    from fenicsx.adjoint import build_context, _solve_forward, _en_G

    mc = build_mesh_from_bfr_pkl(p.mesh_pkl)
    ctx = build_context(list(p.pozo), p, mc)
    f = json.load(open(os.path.join(DIR_OPT, "f_fuente.json")))["f"]
    p.C_ref = 0.05 / f
    d = json.load(open(X0_OPT))
    x = np.asarray(d["x"], float)
    Q, zp = x[:-1], float(x[-1])

    G = ctx.G
    fQ, fG = fem.Function(ctx.Q), fem.Function(G)

    def en_P1(hist):                 # C vive en P2: se interpola a los nodos (P1)
        out = []
        for c in hist:
            fQ.x.array[:] = c
            fG.interpolate(fQ)
            out.append(fG.x.array.copy())
        return np.asarray(out)

    campos = {}
    for nombre, Qn in (("sin", np.zeros_like(Q)), ("con", Q)):
        r = _solve_forward(ctx, Qn, zp)
        campos[f"h_{nombre}"] = _en_G(ctx, r["h"])
        campos[f"C_{nombre}"] = en_P1(r["C"]) * f * 1000.0         # mg/L de nitrato
        print(nombre, "J =", float(r["J"]), flush=True)
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, nodos=G.tabulate_dof_coordinates()[:, :2], tri=np.asarray(G.dofmap.list),
                        phi=ctx.phi.x.array.copy(), Kz=ctx.K_z.x.array.copy(), alpha=float(ctx.phys["alpha"]),
                        Q=Q, zp=zp, nit=int(d["nit"]), dt=float(p.dt), pozo_x=float(p.pozo[0]), **campos)


def a_escena(xy):
    return np.c_[(xy[:, 0] - 816.0) / ESC, (250.5 - xy[:, 1]) / ESC]


def ruta_svg(path, tol=0.8, dec=1, area_min=0.0):
    """matplotlib Path (coordenadas físicas) → atributo d de SVG en coordenadas de escena, sin puntos a menos de tol,
    con dec decimales y sin islas de área menor que area_min (unidades de escena²)."""
    partes, actual = [], []
    for (v, codigo) in zip(path.vertices, path.codes if path.codes is not None else [1] + [2] * (len(path.vertices) - 1)):
        if codigo == 1 and actual:
            partes.append(actual)
            actual = []
        if codigo != 79:
            actual.append(v)
    if actual:
        partes.append(actual)
    d = []
    for pts in partes:
        pts = a_escena(np.asarray(pts))
        keep = [pts[0]]
        for q in pts[1:]:
            if np.hypot(*(q - keep[-1])) >= tol:
                keep.append(q)
        k = np.asarray(keep)
        if len(keep) < 3 or 0.5 * abs(np.dot(k[:, 0], np.roll(k[:, 1], 1)) - np.dot(k[:, 1], np.roll(k[:, 0], 1))) < area_min:
            continue
        d.append("M" + "L".join(f"{a:.{dec}f},{b:.{dec}f}" for a, b in keep) + "Z")
    return "".join(d)


def bandas(tri, valores, niveles, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.tricontourf(tri, valores, levels=niveles, extend="max")
    out = [ruta_svg(pth, **kw) for pth in cs.get_paths()]
    plt.close(fig)
    return out


def velocidad(tri, h, Kz, phi, alpha):
    """Velocidad de poro v = u/φ, u = −K_z (α² ∂x h, ∂z h) (adjoint._u_expr), constante por triángulo."""
    xy = np.c_[tri.x, tri.y]
    t = tri.triangles
    p0, p1, p2 = xy[t[:, 0]], xy[t[:, 1]], xy[t[:, 2]]
    det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    h0, h1, h2 = h[t[:, 0]], h[t[:, 1]], h[t[:, 2]]
    hx = ((h1 - h0) * (p2[:, 1] - p0[:, 1]) - (h2 - h0) * (p1[:, 1] - p0[:, 1])) / det
    hz = ((h2 - h0) * (p1[:, 0] - p0[:, 0]) - (h1 - h0) * (p2[:, 0] - p0[:, 0])) / det
    K, ph = Kz[t].mean(axis=1), phi[t].mean(axis=1)
    return np.c_[-K * alpha ** 2 * hx, -K * hz] / ph[:, None]


def corrientes(tri, v, semillas, pozo=None, ds=15.0, max_pasos=4000):
    """Líneas de corriente desde las semillas (Euler con paso de longitud ds, v constante por triángulo); se cortan al
    salir del dominio o al llegar a menos de 30 m del pozo. Devuelve polilíneas en escena y su tiempo de viaje [d]."""
    buscar = tri.get_trifinder()
    out = []
    for s in semillas:
        x, t, pts, ts = np.array(s, float), 0.0, [], []
        for _ in range(max_pasos):
            k = int(buscar(*x))
            if k < 0:
                break
            pts.append(x.copy())
            ts.append(t)
            if pozo is not None and np.hypot(*(x - pozo)) < 30.0:
                break
            vk = v[k]
            n = np.hypot(*vk)
            if n < 1e-12:
                break
            x = x + ds * vk / n
            t += ds / n / 86400.0
        if len(pts) < 5:
            continue
        pts, ts = a_escena(np.asarray(pts)), np.asarray(ts)
        keep = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i] - pts[keep[-1]])) >= 2.0 or i == len(pts) - 1:
                keep.append(i)
        out.append(dict(p=[round(float(c), 1) for c in pts[keep].ravel()], t=[round(float(c), 2) for c in ts[keep]]))
    return out


def main():
    if SIMULAR or not os.path.exists(NPZ):
        simular()
    from matplotlib.tri import Triangulation
    d = np.load(NPZ)
    nodos, phi = d["nodos"], d["phi"]
    tri = Triangulation(nodos[:, 0], nodos[:, 1], d["tri"])

    # pozos de campo: perfil de capas con su porosidad (funtions/get_phi.well_layer_data)
    from funtions import get_phi
    cwd = os.getcwd()
    os.chdir(os.path.dirname(NPZ))                 # well_layer_data escribe datos_pozos.csv en el directorio actual
    puntos, por_n, _, _ = get_phi.well_layer_data()
    os.chdir(cwd)
    pozos_campo = [dict(x=round((x - 816.0) / ESC, 1), z=round((250.5 - z) / ESC, 1), phi=round(float(v), 2))
                   for (x, z), v in zip(puntos, por_n)]
    # capas A–E de cada sondeo (techo y base) desde el datos_pozos.csv que escribe well_layer_data
    import csv
    capas = [dict(x=round((float(r["X"]) - 816.0) / ESC, 1), capa=r["Capa"], z0=round((250.5 - min(float(r["Z0"]), 250.5)) / ESC, 1),
                  z1=round((250.5 - float(r["Z1"])) / ESC, 1), phi=float(r["Porosidad_n"]))
             for r in csv.DictReader(open(os.path.join(os.path.dirname(NPZ), "datos_pozos.csv"))) if float(r["Z1"]) < 250.5]

    # entrada (x = 10 755 m, z de −768.5 a 250.5): semillas repartidas en la vertical
    # (por debajo de −480 m el agua casi no se mueve y la línea muere contra el fondo; arriba de 160 m sale por la pared)
    semillas = [(10755.0 - 1.0, z) for z in np.linspace(-480.0, 160.0, 9)]
    pozo = np.array([d["pozo_x"], float(d["zp"])])
    nt = d["h_sin"].shape[0]
    v_sin = velocidad(tri, d["h_sin"][nt // 2], d["Kz"], phi, float(d["alpha"]))
    v_con = velocidad(tri, d["h_con"].mean(axis=0), d["Kz"], phi, float(d["alpha"]))
    semillas_con = semillas + [(pozo[0] + 400.0, pozo[1] + dz) for dz in (-150.0, 0.0, 150.0)]

    pluma = {k: [bandas(tri, d[f"C_{k}"][i], NIV_C, tol=1.5, dec=0, area_min=2.0) for i in range(PASO_C - 1, nt, PASO_C)]
             for k in ("sin", "con")}
    out = dict(
        fuente=dict(variante=VARIANTE, iterado=os.path.relpath(X0_OPT, CODIGO), nit=int(d["nit"]),
                    zp=float(d["zp"]), dt_cuadro_h=PASO_C * float(d["dt"]) / 3600.0),
        phi=dict(niveles=NIV_PHI, bandas=bandas(tri, phi, NIV_PHI)),
        pozos_campo=pozos_campo, capas=capas, zona=zona(),
        mapa=mapa(ESTADOS) if ESTADOS else None,
        corrientes=dict(sin=corrientes(tri, v_sin, semillas), con=corrientes(tri, v_con, semillas_con, pozo)),
        pluma=dict(niveles=NIV_C, sin=pluma["sin"], con=pluma["con"]),
    )
    ruta = os.path.join(PRES, "src", "slides-avance3", "esquema.js")
    with open(ruta, "w") as fh:
        fh.write("// Generado por codigo/scripts/exportar_esquema_presentacion.py. No editar a mano.\n")
        fh.write(f"export const ESQ = {json.dumps(out, separators=(',', ':'))};\n")
    print("escrito", ruta, f"{os.path.getsize(ruta) / 1024:.0f} KB")
    for k in ("sin", "con"):
        print(k, [(len(c["t"]), c["t"][-1]) for c in out["corrientes"][k]])


if __name__ == "__main__":
    main()
