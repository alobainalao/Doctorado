# scripts/legacy

Scripts retirados el 2026-10-21 al crear `experimentos/`. Se conservan para
consulta y para reproducir resultados anteriores; no se mantienen.

| Script | Reemplazo |
|---|---|
| `generar_comparacion.py`, `run_all_6_variants.py`, `run_presentacion.{py,sh}`, `make_comparison_grid.py`, `finalize_all_videos.sh`, `sync_videos_to_public.sh` | `python -m experimentos.correr --grupo P_presentacion --animar --publicar` |
| `run_comparison.py`, `rerun_fenicsx_standard.py`, `run_adjoint_single.py` | `python -m experimentos.correr --caso …` |
| `test_kernel_fixes.py`, `diag_velocity_frames.py`, `diag_adj.py`, `test_jm.py` | diagnósticos puntuales ya cerrados (ver `docs/kernel_pozo_fix.md`) |
| `tesis_bfr_*.py`, `mrmt_new_block.py`, `mrmt_bfr_tesis.py`, `_run_scenarios_now.py` | código de la maestría / temporal |
| `genera_project.py`, `tesis_proyect/` | esqueleto vacío de proyecto (AUDIT_REPORT M1) |

Ojo: los lanzadores antiguos pasaban la config en modo env, que descarta
`n_stencil`, `theta`, `phi`, … (caen a defaults), y algunos regeneraban la
malla con `pre=true`.
