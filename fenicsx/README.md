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

## Estado (verificado 2026-08-24)

Forward corriendo end-to-end (Docker `dolfinx/dolfinx:stable` + `pip install
pandas matplotlib scikit-learn h5py gmsh`, ver `Dockerfile`), wireado a
`main.py`. `fenicsx/test_forward_smoke.py` **pasa** — ejecutado dentro del
contenedor.

### Inicialización de H en estacionario (resuelto 2026-08-24)

Antes, `fenicsx` arrancaba `H` de una constante uniforme (heredado de
`maintesis.py`), así que `u_n = −K·∇H ≈ 0` y el campo se quedaba plano
(307.9-308.1). Ahora hace un **spin-up de flujo** (`_spinup_flow` en
`forward.py`): itera el propio solver θ-implícito del flujo con un `dt` grande
dedicado (`spinup_dt`) hasta relajar al estacionario antes de arrancar el
transporte — equivalente al `new_H_init` de `bfr`. El solve estacionario
directo no se usa porque el problema de flujo es Neumann puro (singular); el
operador transitorio sí es no-singular (término de masa `S_s/dt`) y su punto
fijo es el estacionario. El modo constante se fija re-anclando la media.
Converge en ~10 iteraciones a residuo ~1e-9, robusto en malla (spacing
200/80/50). `H` ya tiene estructura espacial (~306-310.5) y `u_n` arranca
equilibrado.

**Nota sobre el rango vs bfr**: el rango de `bfr` (260-439) *no* es el
estacionario de sus BCs — `bfr` **carga `H0` desde `funcion.h5`**, el campo
del spin-up de 7 etapas de la maestría, y `new_H_init` sólo lo refina un paso.
Ese rango absoluto viene de un setup distinto (maestría) y no es reproducible
desde las BCs de `fenicsx`. Lo que sí es comparable —y lo que importa para el
transporte— es el campo de velocidad `u = −K·∇H` (invariante a una constante
aditiva en `H`), no el nivel absoluto. Esa verificación cruzada de `u`/`C`
sigue pendiente (ver "Lo que falta").

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
