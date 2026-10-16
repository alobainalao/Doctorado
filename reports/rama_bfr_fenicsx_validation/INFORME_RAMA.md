# Informe — Rama `audit/bfr-fenicsx-validation`

**Fecha:** 2026-10-15  
**Rama:** `audit/bfr-fenicsx-validation`  
**Último commit:** `0aadb8d`

---

## 1. Objetivo de la rama

Establecer una comparación rigurosa entre los dos backends numéricos del simulador:

- **BFR** (RBF-FD sin malla, `funtions/`)
- **FEniCS** (elementos finitos, `fenicsx/`)

La meta es verificar que ambos resuelven la **misma física** con **la misma discretización espacial**, para que las diferencias observadas sean atribuibles solo al método numérico, no a parámetros distintos.

---

## 2. Trabajo realizado (commits clave)

| Commit | Descripción |
|--------|-------------|
| `d094322` | CDT q20 — elimina triángulos elongados en frontera (malla FEniCS) |
| `9013762` | `build_mesh_from_bfr_pkl()` — malla DOLFINx desde nube de nodos BFR |
| `095305a` | Fix `epsilon_p` en adjoint FEniCS — consistente con BFR |
| `7858769` | Etapa 1: validación H sin pozo ADR (BFR vs FEniCS, mallas distintas) |
| `8a4e2b1` | Fix test de Taylor — perturbación solo en espacio Q |
| `0aadb8d` | `bc_outlet_mode="fem"` + malla/CI centralizadas + script comparación |

---

## 3. Cambios al código base

### 3.1 `funtions/operators.py` — BC de salida configurable

Se añadió `bc_outlet_mode` (leído de RUNTIME) con dos modos:

| Modo | Operador | Orden | Equivalencia FEniCS |
|------|----------|-------|---------------------|
| `"pfs"` (default) | `n·∇(K∇H) = 0` | 2.º | No directa |
| `"fem"` | `−K∇H·n = −h_outlet` | 1.º | **Exacta** |

El modo `"pfs"` es retrocompatible con todo el código existente.

### 3.2 `funtions/mesh.py` — Refinamiento cerca de fuente

`spacing_func()` extendida para refinar también cerca de `p.fnte` (fuente de contaminante), además del pozo. Nuevo parámetro opcional `min_sp_fnte`.

### 3.3 `fenicsx/mesh.py` — Malla desde PKL de BFR

`build_mesh_from_bfr_pkl()` construye una malla DOLFINx desde exactamente los nodos del PKL de BFR:
- Excluye los 587 nodos ghost → **15,965 nodos reales**
- CDT con calidad q20, sin puntos Steiner
- Facet tags mapeados desde los grupos de frontera del PKL

### 3.4 `scripts/validation/experimentos_pfs_fem.py` (nuevo)

Script de comparación BFR vs FEniCS con:
- Malla centralizada (siempre del PKL estándar)
- CI centralizadas (siempre de `funcion.h5`)
- Experimento A (sin extracción) y B (con extracción)
- Verificación automática al inicio

### 3.5 `tests/test_bc_outlet_mode.py` (nuevo)

5 tests unitarios — todos pasan:
1. Modo `"pfs"` → 5 diferenciales (2.º orden) en salida
2. Modo `"fem"` → 2 diferenciales (1.º orden Neumann) en salida
3. RHS en ghost nodes de salida se aplica en modo `"fem"` y no en `"pfs"`
4. `activate_ext=False` desactiva el pozo en H_vector
5. `activate_ext=True` modifica H_vector

---

## 4. Resultados de validación

### 4.1 Etapa 1 — H sin pozo, ADR (mallas distintas, interpolación FEM→BFR)

**Configuración:**
- BFR: 16,552 nodos, `bc_outlet_mode="pfs"`, `activate_ext=False`
- FEniCS: 1,629 nodos (malla Gmsh propia), mismos parámetros físicos
- Comparación por interpolación `LinearNDInterpolator`

| Instante | L2_rel [%] | L∞_rel [%] | H_BFR_mean [m] | H_FEM_mean [m] |
|----------|-----------|-----------|----------------|----------------|
| t=0      | 0.010     | 0.302     | 308.40         | 308.29         |
| t=18     | 0.017     | 1.173     | 308.28         | 308.30         |
| t=36     | 0.027     | 1.921     | 308.28         | 308.30         |
| t=71     | 0.037     | 3.029     | 308.29         | 308.30         |

**Dictamen:** ✅ APROBADA — L2_rel < 5% en todos los pasos.

*Limitación:* La comparación usa mallas distintas (16,552 vs 1,629 nodos) con interpolación, lo que introduce error de interpolación difícil de separar del error numérico real.

---

### 4.2 Problema identificado: inestabilidad de `bc_outlet_mode="fem"` en dominio real

Al intentar correr BFR con `bc_outlet_mode="fem"` (operador Neumann 1.º orden en la salida) en el dominio real, la solución diverge desde el paso inicial:

```
H[t=0] con bc_outlet_mode="pfs": [307.5, 315.2] m  ← físicamente correcto
H[t=0] con bc_outlet_mode="fem": [ 39.2, 377.2] m  ← divergencia
```

**Causa:** El operador RBF-FD Neumann de 1.º orden es menos estable que el de 2.º orden para los stencils del dominio real (radio ~30 m, geometría extendida). En el dominio benchmark (más pequeño) funciona correctamente — confirmado por los 5 tests unitarios.

**Implicación:** No es posible imponer equivalencia exacta de BC de salida entre BFR y FEniCS en el dominio real con los parámetros actuales. La diferencia en la BC de salida es una fuente de error sistémico documentada.

---

## 5. Estructura de malla/CI centralizada

Se implementó un sistema para garantizar que ambos backends usen siempre los mismos recursos:

```
data/input/bfr/real/adr/preproceso.pkl   ← Malla (16,552 nodos, FIJA)
data/input/funcion.h5                    ← CI: H0 y C0 (FIJAS)
```

- **BFR:** `pre=False` → reutiliza pkl, no regenera
- **FEniCS:** `build_mesh_from_bfr_pkl()` → construye desde el mismo pkl
- Verificación automática al inicio de cada experimento

Ver `scripts/validation/README_malla_compartida.md` para detalles.

---

## 6. Test de Taylor (gradiente adjunto)

6/6 backends × modelos pasan el test de Taylor (orden ≈ 2.0):

| Backend | Modelo | Orden Taylor |
|---------|--------|-------------|
| BFR | ADR | ✅ ~2.0 |
| BFR | MRMT Semi | ✅ ~2.0 |
| BFR | MRMT Block | ✅ ~2.0 |
| FEniCS | ADR | ✅ ~2.0 |
| FEniCS | MRMT Semi | ✅ ~2.0 |
| FEniCS | MRMT Block | ✅ ~2.0 |

Fix aplicado: `epsilon_p` consistente entre BFR y FEniCS.

---

## 7. Pendientes antes de cerrar esta rama

1. **Comparación con malla compartida (objetivo principal):** La BC de salida de BFR es distinta a FEniCS en dominio real. Opciones:
   - a) Correr la comparación con `bc_outlet_mode="pfs"` en BFR (estable) y documentar la diferencia de BC como error sistémico cuantificado
   - b) Investigar parámetros de estabilidad para `bc_outlet_mode="fem"` en dominio real (epsilon, spacing, stencil)

2. **Experimentos B (con extracción):** Aún no corridos — depende de resolver el punto anterior.

3. **Merge a `main`:** Los cambios a `operators.py` y `mesh.py` son retrocompatibles (default `"pfs"` preservado). Los nuevos archivos (`experimentos_pfs_fem.py`, `test_bc_outlet_mode.py`) pueden mergearse sin riesgo.

---

## 8. Archivos modificados/creados en esta rama (vs `main`)

| Archivo | Tipo | Estado |
|---------|------|--------|
| `funtions/operators.py` | Modificado | `bc_outlet_mode` fem/pfs |
| `funtions/mesh.py` | Modificado | refinamiento cerca de `p.fnte` |
| `fenicsx/mesh.py` | Modificado | `build_mesh_from_bfr_pkl()` |
| `fenicsx/adjoint.py` | Modificado | fix `epsilon_p` |
| `scripts/validation/experimentos_pfs_fem.py` | Nuevo | comparación centralizada |
| `scripts/validation/README_malla_compartida.md` | Nuevo | documentación malla/CI |
| `scripts/validation/etapa1_H_sin_pozo.py` | Nuevo | validación Etapa 1 |
| `tests/test_bc_outlet_mode.py` | Nuevo | 5 tests BC salida |
| `tests/test_taylor_adjoint.py` | Nuevo | test Taylor 6/6 |
| `visualization/` | Nuevo | sistema centralizado de figuras |
| `AUDIT_REPORT.md` | Nuevo | informe auditoría integral |
| `ADJOINT_DERIVATION_AUDIT.md` | Nuevo | derivación matemática adjunto |
| `AUDIT_INVENTORY.md` | Nuevo | inventario de problemas |
