# Backend `fenicsx` (DOLFINx) — estado

Backend FEM alternativo al RBF-FD (`bfr`), portado desde
[TesisCode](https://github.com/alobainalao/TesisCode) (script de la maestría).
Seleccionable en `app.py` (`metodo = "fenicsx"`), pero **no está conectado
todavía** a `main.py`/`run.py` — falta el paso de wiring (ver Pendientes).

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

## Estado (2026-08-16)

Forward corriendo end-to-end (Docker `dolfinx/dolfinx:stable` + `pip install
pandas matplotlib scikit-learn h5py`, ver `Dockerfile`), wireado a `main.py`.
`fenicsx/test_forward_smoke.py` pasa (`h_n`≈308±0.1, `C_n` creciendo desde 0,
sin crashear).

Primera comparación contra `bfr` hecha, con causa raíz identificada (no
resuelta): **`H` da rangos muy distintos entre backends** (bfr: 260-439;
fenicsx: 307.9-308.1) porque `bfr` inicializa `H0` resolviendo la ecuación de
flujo en estado estacionario (`new_H_init`), mientras que `fenicsx` arranca
de una constante uniforme (heredado de `maintesis.py`). Se intentó portar
ese fix a DOLFINx y produjo un resultado roto (sistema Neumann puro,
singular) — se revirtió. Detalle completo, por qué falló, y la alternativa
más simple pendiente de intentar en `ROADMAP-FENICSX.md`, sección Fase 1.
**No confiar en resultados físicos de este backend hasta resolver eso.**

## Lo que falta (en orden)

1. **Entorno**: no hay ningún entorno con `dolfinx` en este proyecto —
   `bfr_env` es pip-only (`requirements.txt`) y DOLFINx no se instala
   confiablemente por pip. **Se intentó vía conda (`conda create -c
   conda-forge fenics-dolfinx ...`) y no funcionó en esta máquina**: el
   solver `libmamba` tiene un bug reproducible con su caché de "shards"
   (`sqlite3.OperationalError: database is locked`, persiste incluso
   borrando la caché), y el solver `classic` de respaldo es impracticable
   para un paquete tan pesado (>20 min sin converger, varios GB de RAM).
   En su lugar se usa **Docker con la imagen oficial**:
   ```
   docker pull dolfinx/dolfinx:stable
   docker run -it --rm -v "$(pwd):/workspace" -w /workspace dolfinx/dolfinx:stable bash
   # dentro del contenedor:
   pip install h5py gmsh pyvista pandas scipy   # lo que falte, ver abajo
   python -m fenicsx.test_forward_smoke
   ```
   Ninguno de los archivos de este módulo se ha ejecutado ni verificado —
   solo se comprobó sintaxis (`py_compile`). Antes de confiar en algo de
   aquí, hay que correrlo.

2. **Limitación conocida sin resolver** (documentada también en
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

3. **Wiring a `main.py`**: falta el `if p.metodo == "fenicsx":` que llame a
   `solve_forward_fenicsx` en vez del pipeline `bfr` actual.

4. **Sin adjunto**: no existe (ni en el original ni aquí) derivación ni
   implementación de ψ_h/ψ_C/gradiente para este backend. Es el siguiente
   paso natural una vez verificado el forward — candidato para seguir el
   patrón de [Optimal control in DOLFINx interfacing with scipy](http://jsdokken.com/FEniCS-workshop/src/applications/optimal_control.html)
   (Dokken), coherente con que el gradiente ya se compara con scipy en el
   backend `bfr`.

5. **Sin verificación cruzada bfr↔fenicsx**: una vez que el forward corra,
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
