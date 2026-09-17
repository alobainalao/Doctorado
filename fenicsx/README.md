# Backend `fenicsx` (DOLFINx) — estado

Backend FEM alternativo al RBF-FD (`bfr`), portado desde
[TesisCode](https://github.com/alobainalao/TesisCode) (script de la maestría).
Seleccionable con `metodo = "fenicsx"` (en `app.py` o `config/default.json`);
`main.py`/`run.py` despachan a `solve_forward_fenicsx` con import perezoso.
El default del proyecto sigue siendo `bfr`.

## Cómo correrlo

Requiere el entorno `fenics_env` (DOLFINx 0.11.0 instalado vía conda-forge):

```bash
# Desde la app (automático — app.py despacha al binario correcto):
streamlit run app.py   # seleccionar "fenicsx" en el sidebar

# Línea de comandos directa:
/home/alex/miniconda3/envs/fenics_env/bin/python run.py
# (con las vars de entorno normales: model=adr metodo=fenicsx domain=real ...)

# Tests del gradiente adjunto:
/home/alex/miniconda3/envs/fenics_env/bin/python -m fenicsx.test_gradient_checkpoint
/home/alex/miniconda3/envs/fenics_env/bin/python -m fenicsx.test_gradient_checkpoint_mrmt_fenicsx [semi|block]
```

## Qué hay aquí

- `mesh.py` — genera la malla real del acuífero (gmsh + DOLFINx), embebiendo
  el pozo como nodo de malla. Reusa `funtions/get_phi.py:well_layer_data()`
  (misma fuente de datos reales que ya usa `bfr`, no duplicada).
- `forward.py` — núcleo físico (flujo + transporte, esquema theta) extraído
  de `legacy/maintesis.py`, reescrito como una función de un solo run
  `solve_forward_fenicsx(Qout, pozo, p)` con Q(t) y z_p como controles
  externos — mismo contrato que `view.animation.solve_forward` del backend `bfr`.
- `adjoint.py` — optimización por método adjunto (UFL AD + scipy L-BFGS-B).
- `io.py` — animaciones y guardado de resultados.
- `legacy/` — `maintesis.py` y `new_grid.py` tal cual estaban en TesisCode
  (referencia/provenance, no usados desde el pipeline nuevo).

## Configuración desde la app (unificada con bfr)

`fenicsx` lee su física de la **misma config `p`** que `bfr` (la que arma la
app / `run.py` desde env vars, ver `config/parameters.py`), no de valores
horneados. `_config_phys(p)` en `forward.py` toma de `p` los parámetros
compartidos con el mismo significado y fórmula que `bfr`. En `FEM_DEFAULTS`
quedan sólo los específicos del FEM (geometría de fronteras, markers de malla,
anchos de los sumideros).

`main.py` valida lo no soportado: `domain='real'` (la malla usa datos de pozos
reales) y `model` ∈ {`adr`, `mrmt_semi`, `mrmt_block`}.

### Etapas pre / run / post y salidas

- **pre** (`p.pre`): genera la malla si no está cacheada (`data/fenicsx/malla_generada.msh`).
- **run**: `solve_forward_fenicsx` (siempre).
- **save_dat**: guarda `{save_data}/simulation_results.npz` con `H, C, U, nodes, dt`
  en el mismo formato que `bfr`.
- **animate**: `fenicsx/io.py:animate_results` — mp4 si hay ffmpeg, gif si no.
- **postproc**: `fenicsx/postprocess.py:create_btc` — BTC en puntos de control.

## Estado (2026-09-16)

| Modelo      | Forward | Adjunto |
|-------------|---------|---------|
| ADR         | ✓       | ◑ wip   |
| MRMT Semi   | ✓       | ○       |
| MRMT Bloque | ✓       | ○       |

Forward verificado en `fenics_env` end-to-end (ADR + MRMT Semi + MRMT Block):
`H=[307.5, 310.6]` (idéntico al `funcion.h5`), CI correcta.

Adjunto ADR: `fenicsx/adjoint.py` implementado, checkpoint FD pendiente de verificar.
Adjunto MRMT: pendiente tras validar el ADR.

### Inicialización de H y C

`fenicsx` arranca del mismo estado que `bfr`: `data/input/funcion.h5` (campos `H`, `C`
de las etapas iniciales FEM de la maestría), leído con `funtions/utils.py:get_init_values`
e interpolado a los DOF de la malla. Con esto ambos backends parten de idéntico `(H, C)`.

### Gap cuantitativo con bfr

El rango global de `H` es ~9× menor en `fenicsx` (288-311 vs 260-439 en `bfr`).
Causa: `bfr` aplica las BCs de frontera por **colocación con nodos fantasma** (RBF-FD),
que impone gradientes de carga fuertes; `fenicsx` las aplica como **flujo Neumann débil**.
Diferencia de discretización, no de constantes.

## Optimización por método adjunto (`adjoint.py`)

- **Controles**: `x = [Q_0, …, Q_{Nt−1}, z_p]`.
- **Adjunto**: recursión hacia atrás con las transpuestas de los Jacobianos del forward,
  derivados automáticamente vía `ufl.derivative`.
- **Optimizador**: `scipy.optimize.minimize` (L-BFGS-B) con cotas físicas.
- **Checkpoint** (`test_gradient_checkpoint.py`): err.rel. mín ~1e−7 para `grad_Q`
  y ~1e−8 para `grad_zp`, con forma de "V" en log-log (verificado 2026-08-25).

**Desviación deliberada** respecto al forward estándar: el sumidero de FLUJO del pozo
usa un gaussiano (como el de transporte) para que toda la dependencia en z_p sea
diferenciable en forma cerrada. Todas las formas fijan `quadrature_degree=8`.
