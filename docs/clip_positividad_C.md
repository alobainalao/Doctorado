# Recorte de positividad C ≥ 0 — dónde se aplica y cómo revertirlo

## Motivo (2026-10-21)

Los esquemas θ (BFR, RBF-FD) y Galerkin (FEniCS, P2) producen pequeños
valores negativos de C cerca de gradientes abruptos (tipo Gibbs). En
`run_type=comparison` FEniCS dejaba ~745 DOF negativos por paso
(mín ≈ −5·10⁻¹⁴ frente a máx ≈ 6·10⁻¹¹, ~0.1 %), mientras que BFR ya recortaba.
Se decidió que **las 6 variantes** (ADR, MRMT Semi, MRMT Bloque × BFR, FEniCS)
recorten C ≥ 0 en cada paso de tiempo, de forma provisional.

El archivo `data/input/funcion.h5` **no** se modificó: su campo `c` ya era ≥ 0
(0 negativos de 14 338).

## Puntos de recorte

| # | Archivo | Ubicación | Afecta a |
|---|---|---|---|
| 1 | `funtions/utils.py` | `get_init_values`, tras `CloughTocher2DInterpolator` de `c` | CI de ambos backends (BFR y FEniCS) |
| 2 | `funtions/step_time.py` | final de `step_time`, antes del `return` | BFR: ADR, MRMT Semi, MRMT Bloque; standard, comparison y optimization |
| 3 | `fenicsx/forward.py` | `solve_forward_fenicsx`, antes de `C_n.x.array[:] = C_raw` | FEniCS forward físico: 3 modelos (standard, comparison) |
| 4 | `fenicsx/adjoint.py` | `_solve_forward`, tras el bloque MRMT y antes de J₁ | FEniCS forward adjunto-consistente: 3 modelos (comparison, optimization) |

En 2–4 el recorte va **después** de la corrección MRMT; `C_im` se actualiza
con la C sin recortar (mismo orden en ambos backends).

### 1. CI — `funtions/utils.py`

```python
c_new = interp_c(new_nodes)
c_new = np.maximum(c_new, 0.0)  # CI no puede ser negativa físicamente
```

Revertir: borrar la segunda línea.

### 2. BFR — `funtions/step_time.py`

```python
    # Positividad física: C ≥ 0 en cada paso (mismo criterio que fenicsx/forward.py)
    C_new = np.maximum(C_new, 0.0)

    return H, U, C_new, C_im_new
```

Antes de este cambio el recorte existía sólo en la rama `mrmt_semi`
(`C_new = np.maximum(C_new, 0.0)` al final del bucle de splitting) y, desde
`ed9151a`, en `view/animation.py` (`Simulation.update`) tras cada
`step_time`. Ambos se consolidaron en este único punto.

Revertir:
- **todo**: borrar las dos líneas;
- **al estado previo a `ed9151a`** (sólo MRMT Semi): mover la línea al final
  de la rama `if p.model == "mrmt_semi":`.

### 3. FEniCS forward físico — `fenicsx/forward.py`

```python
        C_raw = np.maximum(C_raw, 0.0)  # positividad física (igual que BFR en funtions/step_time.py)
        C_n.x.array[:] = C_raw
```

Antes de `ed9151a` el recorte estaba justo tras el solve
(`C_raw = np.maximum(C_.x.array[:], 0)`) y otra vez tras `_mrmt_semi_step`.

Revertir: borrar la línea.

### 4. FEniCS forward adjunto — `fenicsx/adjoint.py`

```python
        # Positividad física: C ≥ 0 en cada paso, igual que el forward físico.
        # Nota: el recorte no es diferenciable donde C<0; el adjunto lo ignora.
        C_cur.x.array[:] = np.maximum(C_cur.x.array, 0.0)
```

Originalmente aquí **no** se recortaba, a propósito, para conservar la
diferenciabilidad del funcional (el comentario decía: «el forward físico
recorta C≥0; aquí NO se recorta para no romper la diferenciabilidad del
funcional — el checkpoint compara J diferenciable»).

Revertir: borrar las tres líneas.

## Impacto en el gradiente adjunto

`max(·, 0)` no es diferenciable donde C < 0 y el adjunto no modela la máscara
del recorte, así que el gradiente deja de ser exacto en esos nodos. El test de
Taylor (6/6 OK, 2026-10-14) se hizo **sin** recorte en el forward adjunto de
FEniCS: hay que volver a correrlo. Si se degrada, la primera opción es revertir
el punto 4 (y, por simetría, usar en BFR un forward sin recorte para el adjunto).

## Localizar todos los recortes

```bash
grep -n "np.maximum(C\|np.maximum(c_new\|np.maximum(C_cur" \
    funtions/utils.py funtions/step_time.py fenicsx/forward.py fenicsx/adjoint.py
```
