"""
Lanzador único de experimentos BFR vs FEniCS (ver experimentos/README.md).

    python -m experimentos.correr --list
    python -m experimentos.correr --todos --comparar          # grupos A–F completos
    python -m experimentos.correr --grupo A_terminos
    python -m experimentos.correr --caso A0_sin_fuente_sin_pozo --backend bfr
    python -m experimentos.correr --grupo A_terminos B_bc_salida_fem --comparar
    python -m experimentos.correr --todos --comparar --animar   # + videos y grillas
    python -m experimentos.correr --grupo P_presentacion --animar --publicar
    python -m experimentos.correr --caso A3_fuente_pozo --smoke   # prueba rápida (2 pasos)

Cada backend corre en su entorno conda (bfr_env / fenics_env) vía
`conda run`, así que este script puede lanzarse desde cualquier entorno.
Un caso ya corrido (existe simulation_results.npz) se salta salvo --force.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODIGO_DIR)

from experimentos.catalogo import BACKENDS, ENVS, GRUPOS, config_caso, iter_casos  # noqa: E402

BASE_JSON = os.path.join(CODIGO_DIR, "experimentos", "base.json")
OUT_ROOT = os.path.join("data", "output", "experimentos")
VIDEOS_PRESN = os.path.normpath(os.path.join(
    CODIGO_DIR, "..", "presentaciones", "avance1-doc", "public", "videos"))


def _conda(env, args, log_path=None, extra_env=None):
    """Corre `python <args>` en el entorno conda; salida a consola y a log."""
    cmd = ["conda", "run", "-n", env, "--no-capture-output", "python"] + list(args)
    env_vars = {**os.environ, **(extra_env or {})}
    log = open(log_path, "w") if log_path else None
    proc = subprocess.Popen(cmd, cwd=CODIGO_DIR, env=env_vars,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        if log:
            log.write(line)
    proc.wait()
    if log:
        log.close()
    return proc.returncode


def _out_root(args):
    return OUT_ROOT + "_smoke" if args.smoke else OUT_ROOT


def listar():
    for g, gspec in GRUPOS.items():
        print(f"\n{g}  —  {gspec['desc']}")
        for c, spec in gspec["casos"].items():
            bks = ",".join(spec.get("backends", BACKENDS))
            ref = f"  [ref FEM: {spec['ref_fem']}]" if "ref_fem" in spec else ""
            print(f"  {c:<28} {bks:<12} {spec['desc']}{ref}")


def _desactualizado(cfg, out_dir, res):
    """Motivo por el que un resultado existente no corresponde a `cfg` y a la
    malla congelada actual, o None si sigue vigente."""
    import numpy as np
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_path):
        return "sin config.json"
    with open(cfg_path) as f:
        prev = json.load(f)
    dif = sorted(k for k in set(cfg) | set(prev)
                 if k not in ("keep_pre",) and prev.get(k) != cfg.get(k))
    if dif:
        return "config distinta en " + ", ".join(dif)
    import pickle
    from scipy.spatial import cKDTree
    with open(os.path.join(CODIGO_DIR, cfg["mesh_pkl"]), "rb") as f:
        m = pickle.load(f)
    malla = np.asarray(m["nodes"], float)
    n_res = np.asarray(np.load(res, allow_pickle=True)["nodes"], float)
    # BFR guarda todos los nodos (con fantasmas); FEM sólo los del dominio
    # (en otro orden).  Se comparan coordenadas, no sólo el número de nodos.
    if cfg["metodo"] == "bfr":
        igual = n_res.shape == malla.shape and np.allclose(n_res, malla, atol=1e-6)
    else:
        fant = np.concatenate([m["groups"][k] for k in m["groups"] if k.startswith("ghost")])
        dom = np.delete(malla, fant, axis=0)
        igual = len(n_res) == len(dom) and cKDTree(dom).query(n_res)[0].max() < 1e-6
    if not igual:
        return f"{len(n_res)} nodos en el resultado que no son los de la malla ({len(malla)})"
    return None


def correr_caso(base, g, c, spec, backend, args):
    cfg = config_caso(base, g, c, spec, backend, _out_root(args))
    if args.smoke:
        cfg["T"] = 2 * cfg["dt"]
    if args.keep_pre:
        cfg["keep_pre"] = True
    out_dir = os.path.join(CODIGO_DIR, cfg["exp_dir"], backend)
    res = os.path.join(out_dir, "simulation_results.npz")
    tag = f"{g}/{c} [{backend}]"

    if os.path.exists(res) and not args.force:
        motivo = _desactualizado(cfg, out_dir, res)
        if not motivo:
            print(f"[correr] ⏭  {tag}: ya existe (usar --force para repetir)")
            return True
        # Un resultado de otra config o de otra malla sesga la comparación:
        # se repite en vez de reutilizarlo en silencio.
        print(f"[correr] ⚠  {tag}: resultado desactualizado ({motivo}); se repite")
    if args.dry_run:
        dif = {**spec.get("cfg", {}), **spec.get(backend, {})}
        print(f"[correr] (dry-run) {tag}: {dif or 'base'}")
        return True

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[correr] ▶  {tag}", flush=True)
    t0 = time.time()
    rc = _conda(ENVS[backend], ["-m", "experimentos._worker", cfg_path],
                log_path=os.path.join(out_dir, "run.log"))
    ok = rc == 0 and os.path.exists(res)
    print(f"[correr] {'✓' if ok else '✗'}  {tag}  ({time.time() - t0:.0f}s)", flush=True)
    return ok


def animar_presentacion(args):
    """Videos individuales + grillas 3×2 del grupo P (scripts/animate_*.py)."""
    exp_root = os.path.join(CODIGO_DIR, OUT_ROOT, "P_presentacion")
    extra = {"EXP_ROOT": exp_root}
    for m in GRUPOS["P_presentacion"]["casos"]:
        for b in BACKENDS:
            for script in ("scripts/animate_forward.py", "scripts/animate_adjunto.py"):
                _conda(ENVS[b], [script, m, b], extra_env=extra)
    for field in ("H", "C", "V", "adj_H", "adj_C"):
        _conda("bfr_env", ["scripts/animate_grid.py", field], extra_env=extra)

    if args.publicar:
        copia = {"H.mp4": "H.mp4", "C.mp4": "C.mp4", "V.mp4": "V.mp4",
                 "adjunto_psi_H.mp4": "adj_H.mp4", "adjunto_psi_C.mp4": "adj_C.mp4"}
        for m in GRUPOS["P_presentacion"]["casos"]:
            for b in BACKENDS:
                dst = os.path.join(VIDEOS_PRESN, b, m)
                os.makedirs(dst, exist_ok=True)
                for src_name, dst_name in copia.items():
                    src = os.path.join(exp_root, m, b, "figures", src_name)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(dst, dst_name))
        os.makedirs(os.path.join(VIDEOS_PRESN, "grid"), exist_ok=True)
        for field in ("H", "C", "V", "adj_H", "adj_C"):
            src = os.path.join(exp_root, "grid", f"grid_{field}.mp4")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(VIDEOS_PRESN, "grid", f"grid_{field}.mp4"))
        print(f"[correr] videos copiados a {VIDEOS_PRESN}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="listar grupos y casos")
    ap.add_argument("--todos", action="store_true",
                    help="todos los grupos de comparación (A–F); P_presentacion va aparte")
    ap.add_argument("--grupo", nargs="+", help="grupos a correr")
    ap.add_argument("--caso", nargs="+", help="casos a correr (nombre o grupo/caso)")
    ap.add_argument("--backend", nargs="+", choices=BACKENDS, help="limitar backends")
    ap.add_argument("--force", action="store_true", help="repetir aunque ya exista el resultado")
    ap.add_argument("--dry-run", action="store_true", help="sólo mostrar qué se correría")
    ap.add_argument("--smoke", action="store_true",
                    help="prueba rápida: T=2·dt, salida en data/output/experimentos_smoke/")
    ap.add_argument("--keep-pre", action="store_true", help="conservar preproceso.pkl de cada caso BFR")
    ap.add_argument("--comparar", action="store_true", help="comparar BFR vs FEM al terminar")
    ap.add_argument("--animar", action="store_true",
                    help="videos H/C/|U| por backend + grilla de comparación (grupo P: videos de la presentación)")
    ap.add_argument("--publicar", action="store_true", help="grupo P: copiar videos a la presentación")
    args = ap.parse_args()

    if args.list:
        listar()
        return
    if args.todos:
        args.grupo = [g for g, gs in GRUPOS.items() if gs.get("comparar", True)]
    if not (args.grupo or args.caso or args.animar):
        ap.error("indica --todos, --grupo, --caso o --list")

    for g in args.grupo or []:
        if g not in GRUPOS:
            ap.error(f"grupo desconocido: {g}  (ver --list)")

    if not args.dry_run and (args.grupo or args.caso):
        if _conda("bfr_env", ["-m", "experimentos.verificar"]) != 0:
            sys.exit("[correr] verificación fallida: no se corre nada")

    with open(BASE_JSON) as f:
        base = json.load(f)

    fallos, hechos = [], []
    for g, c, spec in iter_casos(args.grupo, args.caso):
        for b in spec.get("backends", BACKENDS):
            if args.backend and b not in args.backend:
                continue
            if correr_caso(base, g, c, spec, b, args):
                hechos.append((g, c))
            else:
                fallos.append(f"{g}/{c} [{b}]")

    if args.comparar and not args.dry_run:
        casos = sorted({f"{g}/{c}" for g, c in hechos if GRUPOS[g].get("comparar", True)})
        if casos:
            cmd = ["-m", "experimentos.comparar", "--caso"] + casos
            _conda("bfr_env", cmd + (["--smoke"] if args.smoke else []))

    if args.animar and not args.dry_run:
        casos = sorted({f"{g}/{c}" for g, c in hechos if GRUPOS[g].get("comparar", True)})
        if casos:
            _conda("bfr_env", ["-m", "experimentos.animar", "--caso"] + casos
                   + (["--force"] if args.force else []))
        if any(g == "P_presentacion" for g, _ in hechos) or not hechos:
            animar_presentacion(args)

    if fallos:
        print("\n[correr] FALLARON:\n  " + "\n  ".join(fallos))
        sys.exit(1)


if __name__ == "__main__":
    main()
