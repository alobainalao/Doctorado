# Archivo

Material que ya no forma parte del flujo de trabajo vigente, conservado por trazabilidad (limpieza del 12-nov-2026;
el estado completo anterior está en el tag `antes-limpieza-2026-11-12`). Nada de aquí lo importa el código activo.

| Carpeta | Contenido |
|---|---|
| `scripts/legacy/` | scripts de la etapa con W₋ (comparaciones, kernel del pozo, plantilla `tesis_proyect`) |
| `scripts/` | scripts de los grupos N_* y de la tesis antigua (`generate_cap4_figures.py`, figuras 6 variantes, comparaciones de gradiente) |
| `experimentos/` | estudios de optimización ya cerrados (barridos de z_p/demanda, escalamiento de Q, híbrido Galerkin) |
| `fenicsx_legacy/` | `maintesis.py` y `new_grid.py` originales de TesisCode |
| `io_legacy/`, `pipeline_legacy/`, `simulation_output_legacy/` | paquetes sin uso (el `io/` sombreaba al módulo estándar) |
| `config/` | configs sin referencias |
| `docs_auditoria/` | informes de la auditoría de oct-2026 |
| `preprocessing/malla_recuperada.py` | recuperación de las mallas de oct-2026 desde sus resultados |
| `codigo-sumidero/malla_borde/` | experimentos de malla junto a la pared (solo existían en el worktree `codigo-sumidero`, ya quitado) |
| `resultados_json/` | métricas (JSON/CSV) de los grupos de experimentos cuyos `.npz` se borraron: A–G, S, N_*, W, Z, Zo, … y las optimizaciones de `codigo-sumidero` |
