# Inventario de `data/output` (no está en git)

Tamaños medidos el 2026-11-04. «Referencias» = quién lo lee según `grep` en tesis, artículos, presentaciones (`src/`) y `codigo/` (experimentos, notebooks, scripts, tests, docs). **No es una verificación de que cada figura publicada dependa de ello**: solo indica que el código lo menciona.

## `codigo/data/output` (21 GB, legacy: no borrar sin revisar)

| Carpeta | Tamaño | Referencias | Observación |
|---|---|---|---|
| `bfr/data` | 8.1 GB | 6 scripts de figuras/animaciones | resultados de las corridas BFR (`.dat`/`.npz`) |
| `fenicsx/data` | 7.0 GB | `generate_cap4_figures`, `comparar_*`, `animate_*` | resultados FEniCSx |
| `experimentos/` | 4.0 GB | `experimentos/*`, notebook `1_resultados`, comentario de `tesis/ChaptersR/CapI_ADR.tex` («datos de `A_terminos/`») | S 875 MB, C 865 MB, A 577 MB, D 577 MB, E 429 MB, B 301 MB, G 223 MB, F 151 MB; además `informe.html` (1.3 MB) y `resumen.csv` |
| `bfr_fem`, `bfr_fem_extract` | 552 MB c/u | 3 y 0 | el segundo no lo menciona nadie; candidato a revisar |
| `experimentos_smoke` | 1.3 MB | 2 | pruebas rápidas |
| `grid`, `comparison_grid_sp60` | 13 MB, 9.8 MB | 53 / 1 | |
| PNG sueltos (`comparison_*`, `panel_adjunto_6_combinaciones`) | < 1 MB | scripts de comparación | |

## `codigo-sumidero/data/output` (7.5 GB)

| Carpeta | Tamaño | Contenido |
|---|---|---|
| `experimentos/` | 7.4 GB | incluye `M_modelo_nuevo/` (M0–M3) que respalda `docs/resultados_modelo_nuevo.md` |
| `bfr/` | 18 MB | |
| `optimizacion_{estandar,nitrato,modelo_nuevo}/` | < 100 KB | resultados JSON; los `*_jscale_prematuro.json` y `adr_25it.json` son artefactos (J_scale mal elegido), conservados solo como evidencia |
| `verificacion_adjunto/` | 68 KB | figuras Taylor / identidad del adjunto |

## Para decidir qué borrar (pendiente, nada se ha borrado)

1. Los videos de las presentaciones ya están copiados en `presentaciones/avance1-doc/public/videos/` (por `scripts/legacy/sync_videos_to_public.sh`): los originales en `bfr/` y `fenicsx/` no son necesarios para la presentación.
2. Las figuras de la tesis se generan con `scripts/generate_cap4_figures.py` desde `bfr/data` y `fenicsx/data`: antes de borrar, regenerar una figura de prueba.
3. Sin verificar: qué grupos de `experimentos/` (A–G) usa el texto final de la tesis. El comentario de `CapI_ADR.tex` solo menciona `A_terminos`.
4. Regla del usuario (2026-11-04): lo legacy no se toca; esto solo orienta una limpieza opcional.
