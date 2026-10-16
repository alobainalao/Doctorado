# Corrección del kernel del pozo BFR — diagnóstico y solución

## Problema identificado

El kernel de extracción BFR (`discrete_delta`, Wendland C2) estaba centrado en
`pozo_cor = nodes[idx_p]` — el nodo más cercano al pozo real, a r ≈ 16.4 m de
[3700, −700]. FEniCS centra su kernel equivalente en la posición exacta del pozo
y además embebe un nodo ahí via `gmsh.model.mesh.embed`.

Consecuencia: el nodo `pozo_cor` absorbía ~47.6 % del caudal Q (peso máximo
Wendland C2 = 1.0 antes de normalizar), produciendo un cono de depresión más
profundo en BFR que en FEniCS.

### Comparación cuantitativa (ADR, t = 30 días)

| Variante | H_well (m) | Drawdown (m) |
|---|---|---|
| BFR baseline (pozo_cor) | 304.90 | −1.42 |
| BFR Fix 1 (centro=p.pozo, sin nodo) | 305.07 | −1.44 |
| BFR Fix 2 (centro=p.pozo + nodo forzado) | 305.92 | −0.91 |
| FEniCS (referencia) | 306.79 | −0.51 |

Fix 1 solo no cambia significativamente el cono (−1.44 vs −1.42 m) porque sin
un nodo en r = 0 el peso sigue concentrándose en el mismo nodo vecino.

Fix 2 reduce la brecha BFR↔FEniCS de 1.89 m a 0.88 m (−53 %).

La brecha residual (0.88 m) tiene otras causas:
- BC de salida: BFR flujo libre vs FEniCS flujo prescrito (offset ~0.3 m global)
- Condición inicial: diferencia de 0.48 m en t = 0
- Densidad de nodos alrededor del pozo aún distinta

## Solución aplicada (2026-10-07)

### Fix 1 — kernel centrado en la posición real del pozo

**Archivo**: `preprocessing/preprocess.py`, línea 53

```python
# Antes:
delta_p = discrete_delta(nodes, pozo_cor, 2.5*p.spacing)

# Después:
delta_p = discrete_delta(nodes, np.array(p.pozo), 2.5*p.spacing)
```

Aplica a: todos los run_type en el preprocesamiento estático.
Ya estaba implícito en `update_pozo` (optimización iterativa).

### Fix 2 — nodo forzado en la posición exacta del pozo

**Archivo**: `funtions/mesh.py`, función `build_nodes_var()`

```python
# Forzar nodo en la posición exacta del pozo (análogo a gmsh.embed en FEniCS)
well = np.asarray(p.pozo, dtype=float)
if np.linalg.norm(nodes - well, axis=1).min() > 0.5:
    new_idx = len(nodes)
    nodes = np.vstack([nodes, well])
    normals = np.vstack([normals, [[0.0, 0.0]]])
    groups['interior'] = np.append(groups['interior'], new_idx)
```

Aplica a: todos los preprocesos que usan `build_nodes_var` (dominio real, todos
los modelos). El dominio benchmark usa `build_nodes` y no se ve afectado.

### Cachés regenerados

Todos los pkl BFR dominio real fueron regenerados con el nodo forzado:
- `data/input/bfr/real/adr/preproceso.pkl`
- `data/input/bfr/real/mrmt_semi/preproceso.pkl`
- `data/input/bfr/real/mrmt_block/preproceso.pkl`

---

## Limitación en optimización y posibles correcciones

Durante la optimización (L-BFGS-B sobre Q(t) y z_p), la posición del pozo
cambia de [3700, −700] a [3700, z_p] en cada iteración. El nodo forzado queda
fijo en z = −700; cuando z_p se aleja, ese nodo se convierte en un nodo
interior ordinario (sin participación en el kernel de extracción si
|z_p − (−700)| > 75 m).

`update_pozo` ya aplica Fix 1 implícitamente (centra el kernel en la posición
real de z_p), pero Fix 2 no puede seguir a z_p sin reconstruir la malla.

### Alternativas evaluadas

**A. Remallado lazy (umbral de desplazamiento)**

Reconstruir preproceso solo si z_p se aleja más de `δ_remesh` del último nodo
forzado. Factible si el optimizador converge con pocas exploraciones grandes.

```
Costo: O(N²) por remallado, ~3 min por evento
Ventaja: Fix 2 exacto para toda la trayectoria
Problema: el Hessiano aproximado del L-BFGS-B se invalida al remanar →
  habría que reiniciar el optimizador, lo cual rompe la convergencia.
Conclusión: inviable con L-BFGS-B sin modificación del optimizador.
```

**B. Pre-malla paramétrica (nodos en rejilla de z_p)**

Generar la malla con nodos forzados en todos los z_p potenciales del rango de
búsqueda [zp_min, zp_max] = [−1400, −200] a pasos de min_sp ≈ 22 m → 55
nodos extra. El kernel siempre tiene al menos un nodo a distancia < 11 m.

```
Costo: ~55 nodos extra (negligible, +0.3 % sobre 16 551)
Ventaja: Fix 2 aproximado para toda z_p sin remallado
Riesgo: nodos alineados verticalmente → posible mal condicionamiento del
  estencil RBF-FD (todos en la misma x = 3700). Necesita verificación.
Implementación: modificar build_nodes_var para insertar z_grid = arange(zp_min,
  zp_max, min_sp) en x = p.pozo[0].
```

**C. Corrección de primer orden (Taylor del kernel)**

Para un desplazamiento pequeño δz = z_p − z_nodo_más_cercano, corregir los
pesos del kernel usando la expansión de Taylor de la función Wendland C2.
Esto daría un peso equivalente a tener un nodo exactamente en z_p sin
moverlo.

```
Costo: O(n_stencil) por iteración (negligible)
Ventaja: no requiere remallado, corrección analítica exacta al primer orden
Complejidad: requiere derivar la expansión Taylor del kernel compacto y
  modificar update_pozo. No trivial.
```

**D. Opción pragmática (implementada)**

Mantener Fix 1 (kernel centrado en z_p exacto) durante iteraciones. El error
es O(r_min / radio) donde r_min ≈ 11 m (la mitad de min_sp) y radio = 75 m →
error relativo ≈ 15 % en el peso del nodo más cercano. Aceptable para
convergencia del optimizador dado que el gradiente adjunto es sensible a la
forma del funcional, no a la precisión sub-estencil del sink.

### Recomendación

Implementar opción **B** (pre-malla paramétrica) si se observa que la
optimización converge a z_p muy distinto de −700 m y el cono BFR difiere
visiblemente del esperado. Es el mejor balance entre exactitud y costo
computacional sin tocar el optimizador.
