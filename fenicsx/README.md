# Backend `fenicsx` (DOLFINx) — estado

Backend FEM alternativo al RBF-FD (`bfr`), portado desde
[TesisCode](https://github.com/alobainalao/TesisCode) (script de la maestría).
Seleccionable con `metodo = "fenicsx"` (en `app.py` o `config/default.json`);
`main.py`/`run.py` ya despachan a `solve_forward_fenicsx` con import perezoso.
El default del proyecto sigue siendo `bfr`.

## Qué hay aquí

- `mesh.py` — genera la malla real del acuífero (gmsh + DOLFINx), embebiendo
  el pozo como nodo de malla. Reusa `funtions/get_phi.py:well_layer_data()`
  (misma fuente de datos reales que ya usa `bfr`, no duplicada).
- `forward.py` — núcleo físico (flujo + transporte, esquema theta) extraído
  de `legacy/maintesis.py`, reescrito como una función de un solo run
  `solve_forward_fenicsx(Qout, pozo, p)` con Q(t) y z_p como controles
  externos — mismo contrato que `view.animation.solve_forward` del backend
  `bfr`. Ver el docstring de ese archivo para el detalle de qué se adaptó
  respecto al original y por qué.
- `legacy/` — `maintesis.py` y `new_grid.py` tal cual estaban en TesisCode
  (el script de 7 etapas de la maestría, con spin-up y `funcion.h5`). Se
  conservan como referencia/provenance, no se usan desde el pipeline nuevo.

## Configuración desde la app (unificada con bfr)

`fenicsx` lee su física de la **misma config `p`** que `bfr` (la que arma la
app / `run.py` desde env vars, ver `config/parameters.py`), no de valores
horneados. `_config_phys(p)` en `forward.py` toma de `p` los parámetros
compartidos —`theta, g, nu, d_z, alpha, R, landa, a_l, a_t, D_d, eps`— con el
mismo significado y fórmula que `bfr`. En `FEM_DEFAULTS` quedan sólo los
específicos del FEM (geometría de fronteras, markers de malla, anchos de los
sumideros), que no existen en la config de `bfr`.

También honra los flags `activate_ext` (gatea la extracción del pozo: término
`QOut` del flujo y sumidero de transporte) y `activate_fuente` (gatea la
fuente de contaminante), igual criterio que `bfr`. `dt`, `T`/`Nt`, `spacing`,
`domain`, `pozo` y `Qout` ya venían de `p`.

`main.py` valida lo no soportado por este backend: `domain='real'` (la malla
usa datos de pozos reales) y `model='adr'` (MRMT no está portado) — error
claro si la app pide otra cosa.

Corrige una divergencia real: antes `D_d` estaba horneado en `1.2e-5`, 14
órdenes de magnitud distinto del `p.D_d` de `bfr` (`1.2e-19`).

### Etapas pre / run / post y salidas (como bfr)

Las mismas etapas configurables que `bfr`, controladas por los flags de la app:

- **pre** (`p.pre`): `fenicsx/mesh.py:get_mesh` — si `pre=True` (o no hay
  cache) genera la malla; si no, reusa `data/fenicsx/malla_generada.msh`
  (análogo a `load_data()` de bfr: reejecutar preproceso vs cargar cache).
- **run**: `solve_forward_fenicsx` (siempre).
- **save_dat** (`p.save_dat`): guarda `{save_data}/simulation_results.npz` con
  la historia `H, C, U` + `nodes` + `dt`, **mismo formato que bfr**
  (`finalize_outputs`): `H`/`C` con eje de realización `K=1`, `U` con sus dos
  componentes. `C` (Lagrange-2) y `u` (vectorial) se interpolan a `G`
  (Lagrange-1) para compartir un único `nodes` con `H`.
- **animate** (`p.animate`): `fenicsx/io.py:animate_results` renderiza `H`,
  `|U|` y `C` sobre la malla a lo largo del tiempo (análogo a los
  `H.mp4/V.mp4/C.mp4` de bfr). Usa **mp4 si hay ffmpeg** (añadido al
  `Dockerfile`) y cae a **gif** si no; robusto (no tumba la corrida si falla el
  render).
- **postproc** (`p.postproc`): `fenicsx/postprocess.py:create_btc` — BTC en los
  puntos de control, `{save_video}/btc.png`. A diferencia del `create_btc` de
  bfr (que compara dos corridas adr vs mrmt en disco), aquí grafica la BTC de
  la propia corrida.

Verificado en Docker end-to-end: `pre=True/False`, `save_dat`, `animate`
(→ gif sin ffmpeg), `postproc` generan sus archivos; `activate_ext=False`
→ `C_out=[0,0,0]`; el `.npz` sale con formato `(Nt, 1, N)` como bfr.

## Estado (verificado 2026-08-24)

Forward corriendo end-to-end (Docker `dolfinx/dolfinx:stable` + `pip install
pandas matplotlib scikit-learn h5py gmsh`, ver `Dockerfile`), wireado a
`main.py`. `fenicsx/test_forward_smoke.py` **pasa** — ejecutado dentro del
contenedor.

### Inicialización de H y C (resuelto 2026-08-24)

Antes, `fenicsx` arrancaba `H` de una constante uniforme (heredado de
`maintesis.py`), así que `u_n = −K·∇H ≈ 0`, el campo se quedaba plano
(307.9-308.1) y `C` partía de cero — condiciones iniciales distintas de `bfr`,
que hacía imposible comparar los dos backends.

Ahora `fenicsx` arranca del **mismo estado precedente que carga `bfr`**:
`data/input/funcion.h5` (los campos `H`, `C` generados por las etapas
iniciales del FEM de la maestría), leído con el **mismo `get_init_values`**
(`funtions/utils.py`) que usa `bfr` e interpolado a los DOF de la malla de
`fenicsx` (Clough-Tocher + vecino más cercano). `H` vive en `G` (Lagrange-1) y
`C` en `Q` (Lagrange-2), así que el interpolador se evalúa sobre cada espacio
por separado. Con esto ambos backends parten de idéntico `(H, C)` por
construcción, y el forward de `fenicsx` es sólo **una etapa final** aplicada
con los datos de la interfaz (Q(t), pozo), no un spin-up propio.

Verificado en el contenedor: `H=[307.5, 310.6]` (idéntico al `h` de
`funcion.h5`: 307.5-310.64, media 308.3) y `C` con el campo inicial real (≠0).
Matiz sobre el "260-439" de `bfr`: **no** es su CI (el `h` de `funcion.h5` es
307-310), sino su `H` **tras el time-stepping** — `new_H_init` lleva 307-310 a
307-342 y luego los pasos de tiempo (drawdown del pozo + acumulación en la
salida) lo abren a 260-439. Ver la sección siguiente.

### Reconciliación de las BCs de flujo con bfr (2026-08-24)

Comparación cruzada `bfr ↔ fenicsx` (mismo escenario: real/adr, spacing 200,
dt=1e5, mismo pozo y Q). Diagnóstico y ajustes:

- **Constantes de frontera**: `fenicsx` ahora toma de `p` (config/geometry.py)
  los mismos valores que usa `bfr` al aplicar `In_h`/`Out_h` — `zi_max,
  zo_max, z_min, inlet_z_t, inlet_a` (`_config_phys`). Corrige un bug real:
  `zi_max` estaba horneado en 308 vs 250 en `bfr`, lo que además descuadraba
  `outlet_a` (depende de `zi_max`).
- **Sumidero de flujo del pozo**: se reemplazó el delta "duro" por la **réplica
  del `discrete_delta` de `bfr`** (Wendland C2, Σ=1, radio 2.5·spacing). Con
  esto el **cono de abatimiento del pozo coincide en forma y ubicación** con
  `bfr`.

Resultado (`data/output/comparison_bfr_fenicsx.png`): **acuerdo cualitativo**
(cono del pozo, gradiente hacia la salida). **Gap cuantitativo que persiste**:
el rango global de `H` es ~9× menor en `fenicsx` (288-311 vs 260-439) porque
`bfr` aplica `In_h`/`Out_h` por **colocación con nodos fantasma** (RBF-FD),
que impone gradientes de carga fuertes, mientras `fenicsx` los aplica como
**flujo Neumann débil** (`inner(h_inlet, g)·ds`). Esa diferencia es de
discretización (colocación vs forma débil FEM), no de constantes — cerrarla
requeriría reformular la imposición de frontera en FEM (p.ej. Dirichlet de
carga en vez de Neumann de flujo), fuera del alcance de "reconciliar BCs".

## Cómo correrlo

No hay entorno pip-only con `dolfinx` (`bfr_env` no lo tiene, y DOLFINx no
se instala confiablemente por pip ni por conda en esta máquina — el solver
`libmamba` falla con `sqlite3.OperationalError: database is locked` y el
`classic` es impracticable). Se usa **Docker con la imagen oficial** (ya
descargada localmente):

```
docker run --rm -v "$(pwd):/workspace" -w /workspace dolfinx/dolfinx:stable bash -c \
  "pip install -q pandas matplotlib scikit-learn h5py gmsh; python -m fenicsx.test_forward_smoke"
```

o construyendo la imagen del `Dockerfile` (`fenicsx-dev`) para no reinstalar
las dependencias en cada run.

## Optimización por método adjunto (`adjoint.py`)

Implementada en `fenicsx/adjoint.py` — contraparte del adjunto de `bfr`
(`view/animation.py`), pero derivando el adjunto de las **mismas formas débiles
FEM vía diferenciación automática de UFL** (`ufl.derivative`/`ufl.adjoint`), en
vez de a mano (enfoque discretize-then-optimize, patrón de Dokken *Optimal
control in DOLFINx interfacing with scipy*).

- **Controles**: `x = [Q_0, …, Q_{Nt−1}, z_p]` (tasas de extracción por paso +
  profundidad del pozo), misma convención que `pack_controls` de `bfr`.
- **Adjunto**: recursión hacia atrás resolviendo ψ_C (transporte) y ψ_h (flujo)
  con las transpuestas de los Jacobianos del forward. El acople h→C (advección
  u=−K∇h y dispersión D(u)) es **exacto** porque el residual se escribe con la
  velocidad simbólica en `h`; la sensibilidad ∂/∂z_p es analítica porque los
  kernels del pozo se escriben simbólicos en `zp_const` (gaussiano·(x_z−z_p)/ε²).
- **Optimizador**: `scipy.optimize.minimize` (L-BFGS-B) con cotas físicas en Q y
  z_p (sin cotas, L-BFGS-B saca a z_p del dominio por diferencia de escala de
  gradientes). Se selecciona con `run_type='optimization'` (ver `main.py`);
  guarda el historial en `{save_data}/optimization_results.npz`.
- **γ (peso de la contaminación)**: con el default `γ=1` la concentración
  (~1e−11) hace despreciable el objetivo de contaminación frente a los términos
  económicos; conviene subir `p.gamma` para que el pozo persiga minimizar la
  concentración observada.

**Desviación deliberada** respecto al forward estándar: el sumidero de FLUJO del
pozo usa aquí un gaussiano (como el de transporte) en vez del Wendland C2 del
`forward.py`, para que toda la dependencia en z_p sea diferenciable en forma
cerrada. Además, todas las formas fijan `quadrature_degree=8`: los kernels son
gaussianos NO polinómicos y muy estrechos (ε∼5–30) frente a la malla, y sin grado
fijo DOLFINx estima cuadraturas distintas para una forma y su derivada, rompiendo
la consistencia gradiente↔funcional.

### Checkpoint del gradiente (`test_gradient_checkpoint.py`)

Espejo del de `bfr`: compara el gradiente adjunto contra diferencias finitas
centradas del MISMO funcional (consistencia interna, no comparación con `bfr`).
Auto-escala γ y pone κ=0 para que la observación domine y ψ_C/ψ_h queden
realmente ejercitados. Correr en Docker:

```
docker build -t fenicsx-dev -f fenicsx/Dockerfile .   # una vez
docker run --rm -v "$(pwd):/workspace" -w /workspace fenicsx-dev \
  python -m fenicsx.test_gradient_checkpoint
```

Verificado (2026-08-25): `grad_Q[·]` err.rel. mín ~1e−7 y `grad_zp` ~1e−8, con
forma de "V" en log-log. Forward smoke test y una optimización corta
(z_p migra a mayor profundidad, J decrece) también verificados en el contenedor.

## Lo que falta (en orden)

1. **Cerrar el gap cuantitativo de `H` con bfr** (ver "Reconciliación de las
   BCs"): el acuerdo es cualitativo, pero el rango global de carga es ~9× menor
   por aplicar las BCs de flujo como Neumann débil (FEM) vs colocación con
   nodos fantasma (bfr). Requiere reformular la imposición de frontera (p.ej.
   Dirichlet de carga). Conviene además comparar en horizonte largo (aquí sólo
   Nt=3, transporte muy temprano).

2. **MRMT en el adjunto**: el adjunto cubre el modelo `adr`; portar MRMT
   (transferencia de masa multirate) al forward y su adjunto sigue pendiente.

## Ya corregido (no verificado en runtime todavía)

En `_boundaries_elements` (`forward.py`), la condición Dirichlet `bc_C` usaba
`fem.locate_dofs_topological(V, ...)` (espacio vectorial) para una función
definida en `Q` (espacio escalar de concentración) — así estaba en
`maintesis.py` original. Se corrigió a `locate_dofs_topological(Q, ...)` (el
espacio correcto de `C_inlet`). Sigue sin ejecutarse ni una vez — revisar
esto primero si algo falla al depurar el forward.
