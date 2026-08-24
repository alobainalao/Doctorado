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
`funcion.h5`: 307.5-310.64, media 308.3) y `C` con el campo inicial real
(≠0). **Corrige un dato previo de este README**: el rango de `bfr` no es
"260-439" — la CI real de `bfr` (el `H` de `funcion.h5`) es 307.5-310.6; ese
"260-439" era incorrecto.

Queda pendiente la verificación cruzada cuantitativa `u(x,t)`/`C(x,t)` entre
backends para el mismo escenario (ver "Lo que falta").

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

## Lo que falta (en orden)

1. **Limitación conocida sin resolver** (documentada también en
   `forward.py`): el sumidero de la ecuación de **flujo** (`delta`) es una
   aproximación "dura" que solo es distinta de cero en el nodo de malla
   exactamente en `pozo` — depende de que `pozo` haya sido embebido al
   generar la malla. El sumidero de **transporte** (`delta_C`) sí usa un
   gaussiano suave, evaluado directamente en `pozo`, así que ese sí
   responde a z_p sin remallar. Mover z_p sin remallar deja el sumidero de
   flujo desactualizado. Antes de intentar optimizar z_p con este backend
   hay que decidir: remallar en cada evaluación (caro; el adjunto
   necesitaría derivada de forma), o sustituir `delta` por un gaussiano
   suave igual que `delta_C`/`Src` (mismo criterio que ya usa `bfr`).

2. **Sin adjunto**: no existe (ni en el original ni aquí) derivación ni
   implementación de ψ_h/ψ_C/gradiente para este backend. Es el siguiente
   paso natural una vez verificado el forward — candidato para seguir el
   patrón de [Optimal control in DOLFINx interfacing with scipy](http://jsdokken.com/FEniCS-workshop/src/applications/optimal_control.html)
   (Dokken), coherente con que el gradiente ya se compara con scipy en el
   backend `bfr`.

3. **Sin verificación cruzada bfr↔fenicsx**: con el forward ya corriendo,
   el resultado más valioso es comparar `C(x,t)`/`h(x,t)` de ambos backends
   para el mismo escenario (mismo dominio, mismo Q(t), mismo pozo) — es la
   verificación de robustez numérica (RBF-FD vs FEM) mencionada en
   ROADMAP.md.

## Ya corregido (no verificado en runtime todavía)

En `_boundaries_elements` (`forward.py`), la condición Dirichlet `bc_C` usaba
`fem.locate_dofs_topological(V, ...)` (espacio vectorial) para una función
definida en `Q` (espacio escalar de concentración) — así estaba en
`maintesis.py` original. Se corrigió a `locate_dofs_topological(Q, ...)` (el
espacio correcto de `C_inlet`). Sigue sin ejecutarse ni una vez — revisar
esto primero si algo falla al depurar el forward.
