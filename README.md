# Simulador ADR / MRMT

Simulación numérica de flujo y transporte en medios porosos (contaminación en
acuíferos), con:

- ADR clásico
- MRMT (multi-rate mass transfer)
- Control óptimo por método adjunto

## Características

- Discretización espacial con RBF-FD
- Interfaz web con Streamlit
- Comparación de curvas de llegada (BTCs)

## Estructura

- `preprocessing/`: generación de matrices y malla
- `pipeline/`: ejecución de la simulación
- `funtions/`: operadores matemáticos y runtime
- `view/`: visualización y animación
- `io/`: carga/guardado
- `postprocessing/`: curvas de llegada (BTCs)
- `config/`: parámetros (`default.json`) y geometría
- `scripts/`: variantes usadas para figuras/datos de la tesis

## Uso

CLI (parámetros desde `config/default.json`):

```bash
python run.py
```

Interfaz web:

```bash
conda activate bfr_env
streamlit run app.py
```

`run_type` en la configuración selecciona `standard` (simulación directa) u
`optimization` (control óptimo adjunto).

## Validación del gradiente adjunto (checkpoint)

**Estado: ✅ PASADO** (2026-08-25)

El método adjunto implementado en `funtions/utils.py` (`grad_Q`, `grad_zp`) fue
validado contra diferencias finitas centradas sobre `compute_functional` (Sec. 8
de `articulos/optimizacion-adjunto/optimizacion+adjunto.tex`).

**Criterio de aceptación:** error relativo mínimo < 10⁻³ sobre la curva
error vs. δ (log-log) para cada componente del gradiente.

**Resultados obtenidos** (`tests/gradient_checkpoint.py`, `T=5 pasos`, `spacing=60`):

| Componente | Error relativo mínimo | Estado |
|---|---|---|
| `grad_Q[t_0]` | ≪ 10⁻¹⁰ | ✅ OK |
| `grad_Q[t_medio]` | ≪ 10⁻¹⁰ | ✅ OK |
| `grad_Q[t_fin]` | ~10⁻¹² | ✅ OK |
| `grad_zp` | ~10⁻¹⁵ (precisión de máquina) | ✅ OK |

Figura: `tests/gradient_checkpoint.png`.

Las corridas de `conjugate_gradient`/`nonlinear_cg`/`optimize_bfr` generadas
**antes de 2026-08-15** usan el gradiente con signo incorrecto (bug D1,
corregido en `utils.py:grad_Q,grad_zp`) — deben descartarse/repetirse.

## Scripts de análisis (sem 3)

```bash
conda activate bfr_env

# 3 escenarios de optimización base (Pista A, Mes 2)
python -m scripts.optimization_scenarios

# Barrido γ para frente de Pareto (Pista A, Mes 2-3)
python -m scripts.pareto_sweep
```
