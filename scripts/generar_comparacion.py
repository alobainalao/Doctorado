#!/usr/bin/env python
"""
generar_comparacion.py

Genera las 30 animaciones de comparación (5 campos × 6 combinaciones):
  H, C, V, adjunto_psi_H, adjunto_psi_C  para cada uno de:
  adr/mrmt_semi/mrmt_block × bfr/fenicsx

Usa run_type=comparison para NO tocar el benchmark standard (Regla 6 CLAUDE.md).

Pipeline por caso:
  1. python run.py         → simulation_results.npz + adjoint_results.npz
  2. animate_forward.py   → H.mp4, C.mp4, V.mp4
  3. animate_adjunto.py   → adjunto_psi_H.mp4, adjunto_psi_C.mp4, adjunto_panel.mp4
  4. copia a presentación → public/videos/{backend}/{model}/{H|C|V|adj_H|adj_C}.mp4

Uso:
    python scripts/generar_comparacion.py

Opciones (filtrar):
    python scripts/generar_comparacion.py --models adr mrmt_semi --backends bfr
Opciones adicionales:
    python scripts/generar_comparacion.py --grid-only
        Solo regenera los 5 videos grid (útil si ya tienes los npz y videos individuales).
"""
import subprocess
import sys
import os
import shutil
import time

# ── Rutas absolutas ───────────────────────────────────────────────────────────
CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESN_DIR  = os.path.normpath(os.path.join(CODIGO_DIR, "..", "presentaciones", "avance1-doc"))
VIDEOS_DIR = os.path.join(PRESN_DIR, "public", "videos")

# ── Configuración ─────────────────────────────────────────────────────────────
ALL_MODELS   = ["adr", "mrmt_semi", "mrmt_block"]
ALL_BACKENDS = ["bfr", "fenicsx"]
ENVS         = {"bfr": "bfr_env", "fenicsx": "fenics_env"}
RUN_TYPE     = "comparison"

# Mapeo nombre generado → nombre en presentación
COPY_MAP = {
    "H.mp4":             "H.mp4",
    "C.mp4":             "C.mp4",
    "V.mp4":             "V.mp4",
    "adjunto_psi_H.mp4": "adj_H.mp4",
    "adjunto_psi_C.mp4": "adj_C.mp4",
}

# ── Parseo mínimo de argumentos ───────────────────────────────────────────────
args = sys.argv[1:]
models    = ALL_MODELS
backends  = ALL_BACKENDS
anim_only = False
grid_only = False
i = 0
while i < len(args):
    if args[i] == "--models":
        i += 1
        models = []
        while i < len(args) and not args[i].startswith("--"):
            models.append(args[i]); i += 1
    elif args[i] == "--backends":
        i += 1
        backends = []
        while i < len(args) and not args[i].startswith("--"):
            backends.append(args[i]); i += 1
    elif args[i] == "--anim-only":
        anim_only = True; i += 1
    elif args[i] == "--grid-only":
        grid_only = True; anim_only = True; i += 1
    else:
        i += 1

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_subprocess(env_name, script_args, label, extra_env=None):
    env = {**os.environ, **(extra_env or {})}
    cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + list(script_args)
    t0  = time.time()
    print(f"\n[gen] ▶  {label}", flush=True)
    r = subprocess.run(cmd, cwd=CODIGO_DIR, env=env)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"[gen] ✗  FALLO: {label}  (rc={r.returncode})", flush=True)
        sys.exit(r.returncode)
    print(f"[gen] ✓  {label}  ({elapsed:.0f}s)", flush=True)


def copy_to_presentation(model, backend):
    src_dir = os.path.join(
        CODIGO_DIR, "data", "output", backend,
        "figures", "real", model, RUN_TYPE,
    )
    dst_dir = os.path.join(VIDEOS_DIR, backend, model)
    os.makedirs(dst_dir, exist_ok=True)
    for src_name, dst_name in COPY_MAP.items():
        src = os.path.join(src_dir, src_name)
        dst = os.path.join(dst_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[gen]     {src_name} → …/videos/{backend}/{model}/{dst_name}", flush=True)
        else:
            print(f"[gen]  ¡falta!: {os.path.relpath(src, CODIGO_DIR)}", flush=True)


# ── Loop principal ────────────────────────────────────────────────────────────
n_total = len(models) * len(backends)
t_start = time.time()

if not grid_only:
    print(f"[gen] ═══════════════════════════════════════════════════", flush=True)
    print(f"[gen] Generando {n_total * 5} animaciones individuales", flush=True)
    print(f"[gen] Modelos  : {models}", flush=True)
    print(f"[gen] Backends : {backends}", flush=True)
    print(f"[gen] run_type : {RUN_TYPE}", flush=True)
    print(f"[gen] ═══════════════════════════════════════════════════", flush=True)

for case_i, model in (enumerate(models) if not grid_only else []):
    for backend in backends:
        env_name = ENVS[backend]
        tag = f"{backend}/{model}"
        t_case = time.time()
        print(f"\n[gen] ━━━  {tag}  ({'BFR (RBF-FD)' if backend=='bfr' else 'FEniCS (FEM)'})  ━━━",
              flush=True)

        # 1. Simulación: forward + adjunto
        if not anim_only:
            sim_env = {
                "run_type": RUN_TYPE,
                "model":    model,
                "metodo":   backend,
                "save_dat": "true",
                "animate":  "false",
                "pre":      "true",    # fuerza repreprocess: garantiza matrices consistentes
            }
            if backend == "bfr":
                # Malla con refinamiento local cerca del pozo: min_sp = spacing/2 = 15 m
                # Mismo efecto que el refinamiento local de la malla gmsh de FEniCS.
                sim_env["min_sp"] = "15"
            if "mrmt" in model:
                # Parámetros MRMT no presentes en el entorno del OS
                sim_env.update({
                    "Nr":     "3",
                    "Deff":   "[1e-09, 5e-10, 1e-10]",
                    "phi_im": "[0.1, 0.05, 0.02]",
                    "beta":   "[0.15, 0.1, 0.05]",
                })
            run_subprocess(
                env_name,
                ["python", "run.py"],
                label=f"simulación  {tag}",
                extra_env=sim_env,
            )

        # 2. Animaciones forward (H.mp4, C.mp4, V.mp4)
        run_subprocess(
            env_name,
            ["python", "scripts/animate_forward.py", model, backend, RUN_TYPE],
            label=f"animate_forward  {tag}",
        )

        # 3. Animaciones adjunto (adjunto_psi_H.mp4, adjunto_psi_C.mp4, adjunto_panel.mp4)
        run_subprocess(
            env_name,
            ["python", "scripts/animate_adjunto.py", model, backend, RUN_TYPE],
            label=f"animate_adjunto  {tag}",
        )

        # 4. Copiar a presentación
        copy_to_presentation(model, backend)

        print(f"[gen] ══  {tag} completado en {time.time() - t_case:.0f}s  ══", flush=True)

total_min = (time.time() - t_start) / 60
print(f"\n[gen] ═══════════════════════════════════════════════════", flush=True)
print(f"[gen] TODO completado en {total_min:.1f} min", flush=True)
print(f"[gen] Destino: {VIDEOS_DIR}", flush=True)
print(f"[gen] ═══════════════════════════════════════════════════", flush=True)

# ── Videos grid (un video por campo con los 6 casos en grilla 3×2) ────────
GRID_FIELDS = ["H", "C", "V", "adj_H", "adj_C"]
GRID_SRC    = os.path.join(CODIGO_DIR, "data", "output", "grid")
GRID_DST    = os.path.join(VIDEOS_DIR, "grid")
os.makedirs(GRID_DST, exist_ok=True)

print(f"\n[gen] ═══════════════════════════════════════════════════", flush=True)
print(f"[gen] Generando {len(GRID_FIELDS)} videos grid", flush=True)
print(f"[gen] ═══════════════════════════════════════════════════", flush=True)

for gf in GRID_FIELDS:
    run_subprocess(
        "bfr_env",
        ["python", "scripts/animate_grid.py", gf, RUN_TYPE],
        label=f"animate_grid  {gf}",
    )
    src = os.path.join(GRID_SRC, f"grid_{gf}.mp4")
    dst = os.path.join(GRID_DST, f"grid_{gf}.mp4")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[gen]     grid_{gf}.mp4 → …/videos/grid/grid_{gf}.mp4", flush=True)
    else:
        print(f"[gen]  ¡falta!: {os.path.relpath(src, CODIGO_DIR)}", flush=True)

grand_total = (time.time() - t_start) / 60
print(f"\n[gen] ═══════════════════════════════════════════════════", flush=True)
print(f"[gen] TODO (incluyendo grids) en {grand_total:.1f} min", flush=True)
print(f"[gen] Grids en: {GRID_DST}", flush=True)
print(f"[gen] ═══════════════════════════════════════════════════", flush=True)
