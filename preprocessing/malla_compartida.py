"""
Malla compartida BFR + FEniCS, generada una sola vez e independiente del backend.

    python -m preprocessing.malla_compartida                 # → data/input/malla_compartida/malla.pkl
    python -m preprocessing.malla_compartida --config experimentos/base.json --figura malla.png
    python -m preprocessing.malla_compartida --diagnostico data/input/malla_compartida/malla.pkl

Corre en bfr_env (usa rbf.pde.nodes y triangle).  El .pkl resultante lo
consumen tal cual:
  * BFR     (preprocessing/preprocess.py, clave `mesh_pkl`): nodes, groups,
            normals, fantasmas;
  * FEniCS  (fenicsx/mesh.py:build_mesh_from_bfr_pkl): los MISMOS nodos sin
            fantasmas y los MISMOS triángulos (`triangles`), sin retriangular.

Pasos
-----
1. Nodos interiores con el mismo generador que BFR (`poisson_disc_nodes` con
   el campo de espaciamiento de 4 niveles de funtions/mesh_test.py).
   Además (REFINO_K), se refina donde K cambia rápido: el espaciado se limita
   para que log10 K no varíe más de REFINO_K décadas entre vecinos.
2. Saneamiento:
   a. Frontera: nodos remuestreados sobre cada lado del polígono con el
      espaciamiento local, con un nodo exacto en cada esquina. (En la malla
      anterior el suavizado de Laplace movía los nodos de frontera hasta 30 m
      hacia dentro: buscaba las claves 'wall'/'inlet'/'outlet', pero los
      grupos se llaman 'boundary:wall'…, así que nunca los fijaba.)
   b. Se quitan los nodos interiores a menos de FRAC_BORDE·h del borde y los
      que estén a menos de FRAC_POZO·h del pozo (se añade un nodo exacto en él).
   c. Suavizado de Laplace sólo de nodos interiores, con los vecinos de la
      triangulación restringida (no la de Delaunay con fantasmas).
   d. Triangulación restringida de calidad (triangle "pq<ang>Y"): ángulo
      mínimo ≥ ANG_MIN, sin puntos Steiner en la frontera; los Steiner
      interiores que haga falta se vuelven nodos de ambos backends.
3. Grupos y normales: cada nodo de frontera pertenece al grupo de su lado;
   las esquinas van a 'wall' (todas tocan una pared) con la normal bisectriz.
   Fantasmas a nodo + DELTA_FANTASMA·d_vecino·normal (rbf usa 0.5; ver la
   constante), comprobando que caen fuera del dominio.

Orden de los nodos: primero los del dominio (los vértices de `triangles`),
después los fantasmas, en el orden de sus nodos de frontera
(ghosts:X[i] ↔ boundary:X[i], como espera funtions/operators.py:bc_block).
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

SALIDA = os.path.join("data", "input", "malla_compartida", "malla.pkl")
BASE_JSON = os.path.join("experimentos", "base.json")

ANG_MIN = 28.0       # [°] ángulo mínimo de la triangulación final
FRAC_BORDE = 0.5     # nodos interiores a < FRAC_BORDE·h del borde se eliminan
FRAC_POZO = 0.5      # ídem alrededor del nodo exacto del pozo
N_SUAVE = 5          # iteraciones de Laplace (como build_nodes_var)
OMEGA = 0.35         # relajación de cada iteración
REFINO_K = 0.3      # [décadas] máx. variación de log10 K por espaciado (None = sin refinar)
H_MIN_K = 15.0       # [m] espaciado mínimo de ese refinamiento
DELTA_FANTASMA = 1.0  # fantasma a 1·d_vecino del nodo de frontera (rbf usa 0.5:
                      # con la salida "pfs", de 2.º orden, BFR diverge en 2 pasos)
MARCADOR = {"inlet": 1, "wall": 2, "outlet": 3}   # = fenicsx/mesh.py


# ── geometría ────────────────────────────────────────────────────────────────

def _lados(vert):
    """(a, b, normal exterior) de cada lado; vert en sentido antihorario."""
    out = []
    for i in range(len(vert)):
        a, b = vert[i], vert[(i + 1) % len(vert)]
        d = b - a
        n = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        out.append((a, b, n))
    return out


def _dist_borde(xy, vert):
    from funtions.mesh_test import _dist_to_boundary
    return _dist_to_boundary(xy, vert)


def _dentro(xy, vert):
    from matplotlib.path import Path
    return Path(vert).contains_points(xy)


def _espaciado_K(h, vert, decadas, h_min, paso=20.0):
    """Espaciado limitado para que log10 K (Kozeny-Carman con la porosidad de
    los pozos, como K_ope) no varíe más de `decadas` entre nodos vecinos:
    h_K = decadas / |∇log10 K|, recortado a ≥ h_min, y se usa min(h, h_K).
    Sin esto, la franja de K muy baja junto a la pared (φ ≈ 0.03, K 10⁴ veces
    menor que arriba) queda a 60 m y BFR muestra franjas oscilantes en H."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    from funtions.get_phi import gen_malla
    x = np.arange(vert[:, 0].min(), vert[:, 0].max() + paso, paso)
    z = np.arange(vert[:, 1].min(), vert[:, 1].max() + paso, paso)
    X, Z = np.meshgrid(x, z, indexing="ij")
    phi = gen_malla(np.column_stack([X.ravel(), Z.ravel()])).reshape(X.shape)
    phi = np.clip(phi, 1e-3, 0.999)
    lk = gaussian_filter(np.log10(phi ** 3 / (1 - phi) ** 2), 1.0)
    gx, gz = np.gradient(lk, paso, paso)
    hk = np.clip(decadas / np.maximum(np.hypot(gx, gz), 1e-12), h_min, 1e6)
    f = RegularGridInterpolator((x, z), hk, bounds_error=False, fill_value=None)
    return lambda xy: np.minimum(h(xy), f(np.atleast_2d(xy)))


def _frontera(vert, grupo_lado, h):
    """Nodos sobre cada lado, distribuidos con ∫ds/h(s) (esquina incluida al
    inicio de cada lado).  Devuelve puntos, grupo, normal y lado de cada uno."""
    pts, grp, nor, lado = [], [], [], []
    lados = _lados(vert)
    for i, (a, b, n) in enumerate(lados):
        L = np.linalg.norm(b - a)
        s = np.linspace(0.0, L, 4001)
        hs = h(a + np.outer(s / L, b - a))
        F = np.concatenate([[0.0], np.cumsum(0.5 * (1 / hs[1:] + 1 / hs[:-1]) * np.diff(s))])
        m = max(1, int(round(F[-1])))
        sk = np.interp(np.linspace(0.0, F[-1], m + 1)[:-1], F, s)
        for k, sv in enumerate(sk):
            pts.append(a + sv / L * (b - a))
            if k == 0:        # esquina vert[i]: entre el lado i-1 y el i
                n_prev = lados[i - 1][2]
                nb = n + n_prev
                nor.append(nb / np.linalg.norm(nb))
                grp.append("wall")
            else:
                nor.append(n)
                grp.append(grupo_lado[i])
            lado.append(i)
    return np.array(pts), np.array(grp), np.array(nor), np.array(lado)


# ── triangulación y calidad ──────────────────────────────────────────────────

def _cdt(nodos, n_bnd, opts):
    """CDT con los n_bnd primeros nodos como polígono cerrado (en orden)."""
    import triangle as tr
    segs = np.column_stack([np.arange(n_bnd), np.roll(np.arange(n_bnd), -1)])
    r = tr.triangulate({"vertices": nodos, "segments": segs}, opts)
    return r["vertices"], r["triangles"].astype(np.int64)


def calidad(nodos, tris):
    X = nodos[tris]
    L = np.stack([np.linalg.norm(X[:, (i + 1) % 3] - X[:, (i + 2) % 3], axis=1) for i in range(3)], 1)
    ang = np.empty_like(L)
    for i in range(3):
        a, b, c = L[:, i], L[:, (i + 1) % 3], L[:, (i + 2) % 3]
        ang[:, i] = np.degrees(np.arccos(np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1, 1)))
    area = 0.5 * np.abs((X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1])
                        - (X[:, 2, 0] - X[:, 0, 0]) * (X[:, 1, 1] - X[:, 0, 1]))
    asp = L.max(1) / L.min(1)
    return dict(ang_min=float(ang.min()), ang_max=float(ang.max()),
                n_ang_lt20=int((ang.min(1) < 20).sum()), n_ang_gt120=int((ang.max(1) > 120).sum()),
                aspecto_max=float(asp.max()), n_triangulos=int(len(tris)), area_m2=float(area.sum()))


def _suavizar(nodos, tris, n_bnd, fijos, vert, h, n_iter, omega):
    """Laplace sobre los vecinos de la CDT; los fijos (frontera, pozo) no se mueven
    y un candidato demasiado cerca del borde se rechaza."""
    libres = np.setdiff1d(np.arange(len(nodos)), fijos)
    for _ in range(n_iter):
        aristas = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        aristas = np.vstack([aristas, aristas[:, ::-1]])
        cnt = np.bincount(aristas[:, 0], minlength=len(nodos))
        sx = np.bincount(aristas[:, 0], weights=nodos[aristas[:, 1], 0], minlength=len(nodos))
        sy = np.bincount(aristas[:, 0], weights=nodos[aristas[:, 1], 1], minlength=len(nodos))
        cen = np.column_stack([sx, sy]) / np.maximum(cnt, 1)[:, None]
        cand = nodos[libres] + omega * (cen[libres] - nodos[libres])
        ok = _dentro(cand, vert) & (_dist_borde(cand, vert) >= FRAC_BORDE * h(cand))
        nodos = nodos.copy()
        nodos[libres[ok]] = cand[ok]
        _, tris = _cdt(nodos, n_bnd, "pY")
    return nodos, tris


# ── generación ───────────────────────────────────────────────────────────────

def generar(cfg, ang_min=ANG_MIN, n_suave=N_SUAVE, refino_k=REFINO_K):
    from funtions.runtime import RUNTIME
    from config.parameters import Parameters
    from funtions.mesh_test import _boundary_config, prepare_spacing_cache, spacing_func_cached
    from rbf.pde.nodes import poisson_disc_nodes

    RUNTIME.params = Parameters(config={**cfg, "metodo": "bfr"})
    p = RUNTIME.params
    vert = np.asarray(p.vert, float)
    pozo = np.asarray(p.pozo, float)
    smp, grupos_lado, _ = _boundary_config()
    lado_grupo = {i: g for g, lados in grupos_lado.items() for i in lados}

    prepare_spacing_cache()
    h = spacing_func_cached
    if refino_k:
        h = _espaciado_K(h, vert, refino_k, H_MIN_K)

    # 1. Nodos interiores del generador de BFR (sin fantasmas: se rehacen al final).
    nodes0, groups0, _ = poisson_disc_nodes(h, (vert, smp), boundary_groups=grupos_lado)
    interior = nodes0[groups0["interior"]]
    print(f"[malla] poisson_disc_nodes: {len(nodes0)} nodos ({len(interior)} interiores)")

    # 2a. Frontera exacta con esquinas.
    bnd, bgrp, bnor, _ = _frontera(vert, lado_grupo, h)

    # 2b. Interiores pegados al borde o al pozo.
    hi = h(interior)
    lejos = (_dist_borde(interior, vert) >= FRAC_BORDE * hi) & _dentro(interior, vert)
    lejos &= np.linalg.norm(interior - pozo, axis=1) >= FRAC_POZO * h(pozo[None])[0]
    print(f"[malla] frontera: {len(bnd)} nodos (5 esquinas exactas); "
          f"interiores eliminados por cercanía al borde/pozo: {int((~lejos).sum())}")
    nodos = np.vstack([bnd, interior[lejos], pozo[None]])
    n_bnd = len(bnd)
    i_pozo = len(nodos) - 1

    # 2c. Suavizado + 2d. CDT de calidad.
    _, tris = _cdt(nodos, n_bnd, "pY")
    print(f"[malla] CDT inicial: {calidad(nodos, tris)}")
    if n_suave:
        nodos, tris = _suavizar(nodos, tris, n_bnd, np.r_[np.arange(n_bnd), i_pozo], vert, h, n_suave, OMEGA)
    n_antes = len(nodos)
    nodos, tris = _cdt(nodos, n_bnd, f"pq{ang_min}Y")
    if not np.allclose(nodos[:n_antes], np.vstack([bnd, nodos[n_bnd:n_antes]])):
        raise RuntimeError("triangle reordenó los nodos de entrada")
    q = calidad(nodos, tris)
    print(f"[malla] CDT final (q{ang_min}, +{len(nodos) - n_antes} Steiner interiores): {q}")

    # 3. Grupos, normales y fantasmas.
    n_dom = len(nodos)
    groups = {"interior": np.arange(n_bnd, n_dom)}
    normals = np.zeros((n_dom, 2))
    normals[:n_bnd] = bnor
    for g in ("inlet", "outlet", "wall"):
        groups[f"boundary:{g}"] = np.nonzero(bgrp == g)[0]
    d_vec = cKDTree(nodos).query(nodos[:n_bnd], 2)[0][:, 1]
    fantasmas = []
    for g in ("inlet", "outlet", "wall"):
        idx = groups[f"boundary:{g}"]
        gh = nodos[idx] + DELTA_FANTASMA * d_vec[idx, None] * normals[idx]
        if _dentro(gh, vert).any():
            raise RuntimeError(f"fantasmas de {g} dentro del dominio")
        groups[f"ghosts:{g}"] = n_dom + sum(len(f) for f in fantasmas) + np.arange(len(idx))
        fantasmas.append(gh)
    todos = np.vstack([nodos] + fantasmas)
    normals = np.vstack([normals, np.full((len(todos) - n_dom, 2), np.nan)])
    d_g = cKDTree(todos).query(todos[n_dom:], 2)[0][:, 1]

    sha = hashlib.sha256(np.ascontiguousarray(todos, dtype=np.float64).tobytes()).hexdigest()
    h_nn = cKDTree(nodos).query(nodos, 2)[0][:, 1]
    meta = dict(
        version=2,
        origen=(f"preprocessing/malla_compartida.py (min_sp={p.min_sp}, s_cone={p.s_cone}, "
                f"min_sp_boundary={p.min_sp_boundary}, spacing={p.spacing}, "
                f"r_circle={p.r_circle}, r_cone={p.r_cone})"),
        fecha=datetime.date.today().isoformat(),
        n_nodos=int(len(todos)), n_dominio=int(n_dom), n_fantasmas=int(len(todos) - n_dom),
        n_frontera=int(n_bnd), n_stencil=int(p.n_stencil), nodes_sha256=sha,
        h_min=float(h_nn.min()), h_mediana=float(np.median(h_nn)),
        dist_min_fantasmas=float(d_g.min()),
        area_poligono_m2=float(_area_poligono(vert)),
        calidad=q, ang_min_pedido=float(ang_min), n_suave=int(n_suave),
        refino_k_decadas=refino_k, h_min_k=H_MIN_K if refino_k else None,
        nota=("Malla saneada: frontera exacta con esquinas, sin nodos pegados al borde, "
              f"CDT con ángulo mínimo ≥ {ang_min}°"
              + (f", refinada donde log10 K cambia > {refino_k} décadas por espaciado (h ≥ {H_MIN_K:g} m)"
                 if refino_k else "")
              + "; BFR y FEniCS usan estos nodos y triángulos tal cual."),
    )
    return dict(nodes=todos, groups=groups, normals=normals, simplices=smp,
                pozo_cor=nodos[i_pozo].copy(), triangles=tris, meta=meta)


def _area_poligono(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def diagnostico(m):
    """Resumen de una malla (nueva o anterior) para compararlas."""
    from shapely.geometry import Polygon
    nodes, g = np.asarray(m["nodes"], float), m["groups"]
    vert = np.array([[816, -2690.5], [4354, -1757.5], [10755, -768.5], [10755, 250.5], [816, 250.5]])
    out = {"n_nodos": len(nodes)}
    for k in ("boundary:inlet", "boundary:outlet", "boundary:wall"):
        out[f"dist_max_borde[{k}]"] = float(_dist_borde(nodes[g[k]], vert).max())
    out["dist_min_interior_borde"] = float(_dist_borde(nodes[g["interior"]], vert).min())
    out["esquinas_con_nodo"] = int(sum(np.linalg.norm(nodes - v, axis=1).min() < 1e-6 for v in vert))
    if "triangles" in m:
        out.update(calidad(nodes, m["triangles"]))
    out["area_poligono_m2"] = float(Polygon(vert).area)
    return out


def figura(m, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    nodes, g, tris = m["nodes"], m["groups"], m["triangles"]
    tri = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    vistas = (("Dominio completo", None), ("Esquina salida/pared (816, −2690)", (650, 1150, -2760, -2420)),
              ("Pozo (3700, −700)", (3550, 3850, -800, -600)), ("Esquina pared/entrada (10755, −768)", (10450, 10800, -900, -600)))
    for ax, (tit, lim) in zip(axs.ravel(), vistas):
        ax.triplot(tri, lw=0.25, color="0.35")
        for k, c in (("boundary:inlet", "tab:blue"), ("boundary:outlet", "tab:green"),
                     ("boundary:wall", "tab:red")):
            ax.plot(*nodes[g[k]].T, ".", ms=4 if lim else 1.5, color=c, label=k.split(":")[1])
        gh = np.concatenate([g[k] for k in g if k.startswith("ghosts")])
        ax.plot(*nodes[gh].T, "x", ms=3 if lim else 1, color="0.5", label="fantasmas")
        if lim:
            ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:])
        ax.set_aspect("equal"); ax.set_title(tit, fontsize=13)
    axs[0, 0].legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    print(f"[malla] figura → {ruta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=BASE_JSON)
    ap.add_argument("--salida", default=SALIDA)
    ap.add_argument("--ang-min", type=float, default=ANG_MIN)
    ap.add_argument("--n-suave", type=int, default=N_SUAVE)
    ap.add_argument("--refino-k", type=float, default=REFINO_K,
                    help="décadas de K por espaciado (0 = sin refinar por K)")
    ap.add_argument("--figura", help="PNG con la malla y zooms de esquinas y pozo")
    ap.add_argument("--diagnostico", metavar="PKL", help="sólo diagnosticar una malla existente")
    args = ap.parse_args()
    os.chdir(CODIGO_DIR)

    if args.diagnostico:
        with open(args.diagnostico, "rb") as f:
            print(json.dumps(diagnostico(pickle.load(f)), indent=2, ensure_ascii=False))
        return

    with open(args.config) as f:
        cfg = json.load(f)
    m = generar(cfg, args.ang_min, args.n_suave, args.refino_k or None)
    if os.path.exists(args.salida):
        raise SystemExit(f"[malla] {args.salida} ya existe: muévela antes (la malla congelada no se pisa)")
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, "wb") as f:
        pickle.dump(m, f)
    print(f"[malla] {m['meta']['n_nodos']} nodos → {args.salida}")
    print(json.dumps(diagnostico(m), indent=2, ensure_ascii=False))
    if args.figura:
        figura(m, args.figura)


if __name__ == "__main__":
    main()
