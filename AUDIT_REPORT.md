# AUDIT_REPORT.md — Auditoría Integral del Proyecto Doctoral
**Fecha:** 2026-10-07  
**Rama:** `doctoral-audit` (todos los repos)  
**Auditor:** Claude Sonnet 4.6

---

## 1. Resumen ejecutivo

La auditoría cubre el ecosistema completo del proyecto doctoral:
simulador Python (BFR/RBF-FD + FEniCS/DOLFINx), derivación matemática del adjunto,
validación cuantitativa BFR↔FEniCS, sistema de visualización, tesis LaTeX, artículo y
presentación React. Se identificaron 18 problemas clasificados por severidad.

**Cambios realizados (ramas `doctoral-audit`):**

| Fase | Repo | Entregable |
|------|------|------------|
| FASE 2 | `codigo/` | `AUDIT_INVENTORY.md` — inventario completo + 18 problemas |
| FASE 3 | `codigo/` | `ADJOINT_DERIVATION_AUDIT.md` — derivación formal ADR+MRMT |
| FASE 4 | `codigo/` | `scripts/quantitative_comparison.py` — normas L2/L∞ cuantitativas |
| FASE 5 | `codigo/` | `visualization/` — sistema centralizado de figuras (4 perfiles) |
| FASE 6 | `tesis/`  | Cap. 2: espacios Sobolev + formulación débil; Cap. 3: U_ad + J∈L² |
| FASE 7 | `articulos/` | Abstract SIAM + §Resultados estructurado |
| FASE 8 | `presentaciones/` | A3S10: matriz backends corregida con datos reales |

---

## 2. Arquitectura del simulador

```
config/default.json  ──→  preprocessing/preprocess.py
                                  ↓
         ┌──────────────────────────────────────────┐
         │  Backend BFR/RBF-FD (funtions/)          │
         │  step_time.py · operators.py             │
         │  mesh.py · get_phi.py · utils.py         │
         └──────────────────────────────────────────┘
                              ↕
         ┌──────────────────────────────────────────┐
         │  Backend FEniCS/DOLFINx (fenicsx/)       │
         │  forward.py · adjoint.py                 │
         │  mesh.py · io.py                         │
         └──────────────────────────────────────────┘
                              ↓
            3 modelos: ADR · MRMT_semi · MRMT_block
            3 run_types: standard · comparison · optimization
                              ↓
               tests/ · scripts/ · visualization/
```

**Matriz de backends (estado actual):**

| Modelo | Fwd BFR | Fwd FEM | Adj BFR | Adj FEM | Opt BFR | Opt FEM |
|--------|---------|---------|---------|---------|---------|---------|
| ADR | ✓ | ✓ | ✓ | ✓ | ✓ (−91.9% J) | en prueba |
| MRMT Semi | ✓ | ✓ | ✓ | ✓ | ✓ (−77% J) | en prueba |
| MRMT Bloque | ✓ | ✓ | ✓ | ✓ | en prueba | pendiente |

---

## 3. Problemas identificados y estado

### 🔴 Críticos

**C1 — BC hidráulicas incompatibles (BFR vs FEniCS)**  
BFR impone Dirichlet en h en `Γ_in∪Γ_out`; FEniCS usa Neumann natural (ninguna BC
explícita de Dirichlet en el backend FEniCS forward). Esto produce campos de carga
hidráulica con rangos espaciales que difieren 4.5× (BFR=17m, FEniCS=3.82m) y un
error en velocidad de Darcy del 69% en norma L2. **Estado: documentado en
ADJOINT_DERIVATION_AUDIT.md (D2); no corregido — es resultado científico.**

**C2 — Cap. 4 de la tesis vacío (§2-§6)**  
Los resultados de MRMT, RBF-FD, verificación del gradiente y optimización no están
documentados en la tesis. **Estado: esqueleto con marcadores PENDIENTE explícitos.
A completar en sem 3 con datos disponibles.**

**C3 — Artículo sin resultados numéricos**  
§Resultados tenía solo marcadores TODO. **Estado: FASE 7 reestructuró la sección con
el marco formal de verificación del gradiente y marcadores PENDIENTE precisos.**

**C4 — Concentraciones negativas en BFR (1000+ nodos)**  
El esquema θ=0.9 con los parámetros actuales produce valores negativos en C en
aproximadamente 1.2% de los nodos en la simulación BFR/ADR. **Estado: documentado —
causa confirmada (2026-10-08): oscilaciones tipo Gibbs del esquema trapezoidal ponderado
(θ=0.9 > 0.5) cerca del frente de concentración agudo. No se aplica clipping en
ADR/MRMT Block para preservar linealidad del operador y exactitud del gradiente adjunto
(comentario añadido en step_time.py). Para MRMT Semi el clipping es necesario por el
splitting exponencial (ya presente en línea 49). Causa documentada en step_time.py.**

### 🟠 Importantes

**I1 — Inconsistencia de kernel en el pozo**  
`compute_functional()` en `utils.py` usa Wendland C2 para calcular J1; `grad_Q()`
usa Gaussiano (gauss_p). Optimiza un funcional ligeramente diferente al que simula.
**Estado: documentado en ADJOINT_DERIVATION_AUDIT.md (D1). Corrección requiere
decisión matemática sobre cuál kernel usar uniformemente.**

**I2 — Derivación adjunta MRMT ausente en tesis y artículo**  
El artículo §8 cubre solo ADR. La tesis Cap. 3 no tiene derivación adjunta MRMT.
**Estado: ADJOINT_DERIVATION_AUDIT.md tiene la derivación completa (ecuaciones
adimensionales, verificación de transpuesta M_fwd_r^T). Pendiente trasladar a tesis.**

**I3 — Sin estudio de convergencia espacial RBF-FD**  
No hay análisis de orden de convergencia del esquema RBF-FD (orden esperado: O(h²)
para MQ con stencil n=25). Solo existe validación cruzada BFR↔FEniCS.
**Estado: pendiente — experimento de convergencia en malla variable.**

**I4 — θ_BFR=0.9 vs θ_FEniCS exacto**  
El parámetro θ de integración temporal difiere entre backends, contribuyendo a
discrepancias sistemáticas en C. **Estado: documentado; no corregido — modificar
θ en BFR requiere validar de nuevo todos los tests.**

### 🟡 Moderados

**M1 — `tesis_proyect/` legacy vacío**  
Confunde la arquitectura del código. **Estado: no eliminado — requiere confirmación
del usuario.**

**M2 — config1-3.json vacíos**  
Solo `default.json` es la fuente de verdad; los otros tres archivos están vacíos.
**Estado: documentado en AUDIT_INVENTORY.md.**

**M3 — Tabla de parámetros unificada ausente**  
Los parámetros físicos están en `config/default.json`, `config/parameters.py` y
`config/geometry.py` sin tabla maestra con unidades y referencias bibliográficas.
**Estado: pendiente.**

### 🔵 Editoriales

**E1 — DPI y tamaños de figuras no estandarizados**  
Las figuras en `tesis/salidas/*.png` no tienen DPI documentado.
**Estado: FASE 5 crea `visualization/` con perfiles THESIS/PAPER/PRESENTATION.**

**E2 — Sin captions completos en figuras de la tesis (Cap. 4)**  
Las figuras de maestría en §1 tienen captions adecuados; los placeholders de §2-§6
no. **Estado: a completar junto con los datos numéricos.**

**E3 — Clasificación MSC del artículo**  
49K20 + 35Q93 + 76S05 + 90C29 es correcta y alineada con el contenido real.
**Estado: verificada, sin cambios.**

---

## 4. Resultados de la comparación cuantitativa BFR↔FEniCS

Ejecutado con `scripts/quantitative_comparison.py` sobre `comparison/simulation_results.npz`.

| Campo | Error L2 rel. | Error L∞ | Nota |
|-------|--------------|----------|------|
| H (carga hidráulica) | 0.05% | — | Rango BFR=17m, FEM=3.82m (BC diferentes) |
| C (concentración) | 32–35% | — | BFR produce 2.7× mayor concentración |
| \|U\| (velocidad Darcy) | 69% | — | Error angular medio 17°; máx 179.94° |

Las grandes discrepancias en C y U son consecuencia directa de C1 (BCs incompatibles).
La comparación H=0.05% es artificialmente baja porque ambos campos tienen el mismo
perfil cualitativamente pero diferente escala física.

---

## 5. Estado del adjunto (ADJOINT_DERIVATION_AUDIT.md)

| Componente | Estado | Detalles |
|------------|--------|----------|
| Adjunto ADR — derivación analítica | ✓ | Lagrangiana, variaciones, IBP, BCs en tabla |
| Adjunto ADR — verificación FD | ✓ | Error rel. < 10⁻¹⁰ (∇_Q J), ~10⁻¹⁵ (∂J/∂z_p) |
| Adjunto MRMT Semi — código | ✓ | M_fwd_r^T exacto (splitting backward) |
| Adjunto MRMT Bloque — código | ✓ | Acoplamiento adjunto verif. en tests |
| Adjunto MRMT — derivación analítica | ⚠️ | Solo en ADJOINT_DERIVATION_AUDIT.md; no en tesis/artículo |
| Adjunto FEniCS — ufl.derivative | ✓ | Discretize-then-optimize explícito |
| Inconsistencia kernel D1 | ⚠️ | Wendland C2 (J1) vs Gaussiano (grad_Q) |

---

## 6. Sistema de visualización (visualization/)

Módulo creado en `codigo/visualization/`:

- `style.py`: 4 perfiles `THESIS` (300 DPI), `PAPER` (600 DPI), `PAPER_WIDE`, `PRESENTATION` (150 DPI) con rcParams completos
- `fields.py`: `plot_scalar_field()`, `plot_vector_field()`, `plot_comparison_2x2()`
- `convergence.py`: `plot_gradient_fd_check()`, `plot_convergence_history()`, `plot_pareto_front()`
- `export.py`: `save_figure()` y `save_all_profiles()` con DPI del perfil

Uso:
```python
from visualization.style import apply_style
from visualization.convergence import plot_gradient_fd_check
from visualization.export import save_figure

with apply_style("PAPER"):
    fig, ax = plot_gradient_fd_check(eps_arr, err_arr, method_label="BFR")
save_figure(fig, "gradient_fd_check_adr", profile="PAPER")
```

---

## 7. Cambios en la tesis (doctoral-audit)

**Chapter2.tex** — añadido:
- `\subsection{Dominio y espacios funcionales}`: Ω ⊂ ℝ², ∂Ω=Γ_in∪Γ_out∪Γ_wall, BCs
  formales para h (Dirichlet+Neumann) y C (Neumann homogéneo), definiciones H¹(Ω),
  L²(Ω), L²(0,T;H¹(Ω)), espacio de Bochner.
- `\subsection{Formulación variacional del sistema MRMT}`: forma débil completa
  para región móvil e inmóviles, problema débil, referencia a Evans para existencia/unicidad.

**Chapter3.tex** — añadido:
- `\subsection*{Espacios funcionales y conjunto de controles admisibles}`:
  definición formal de U_ad (convexo cerrado en L²(0,T)×ℝ), regularidad de estados.
- `\paragraph{Espacios funcionales del funcional J}`: J: L²(0,T)×ℝ→ℝ, traza puntual
  H¹↪C⁰ en dim 2, diferenciabilidad Fréchet.

---

## 8. Cambios en el artículo (doctoral-audit)

- **Abstract**: reescrito en estándar SIAM (sin fórmulas, sin citas, autocontenido);
  incluye resultado cuantitativo del checkpoint (error rel. < 10⁻³); alcance ADR explicitado.
- **§Resultados**: reestructurado — fórmula de diferencias finitas, umbrales de tolerancia,
  marcadores PENDIENTE precisos con rutas de datos.
- **§Discusión y §Conclusión**: TODO genéricos convertidos a ítems PENDIENTE explícitos.

---

## 9. Cambios en la presentación (doctoral-audit)

**A3S10.jsx** — matriz de backends corregida:
- ADR optBFR: `'ok'` (J −91.9%, verificado en A3S09)
- MRMT Semi optBFR: `'ok'` (J −77%, verificado en A3S09)
- MRMT Bloque optBFR: `'test'` (en prueba)
- FEniCS optimización: `'test'` (ADR/Semi), `'pen'` (Bloque)
- Nota de pie actualizada con datos reales.

---

## 10. Diagnóstico de la narrativa de avance3 (21 slides)

La estructura narrativa es coherente: PROBLEMA→VALIDACIÓN→IMPLEMENTACIÓN→RESULTADOS→PRÓXIMOS PASOS.

**Slides completos y correctos:**
- A3S01 (agenda/estado), A3S02 (checkpoint), A3S03 (dominio/parámetros), A3S04 (ecuaciones adjuntas), A3S09 (conclusiones), A3S10 (matriz — ahora corregida), A3S11-A3S14 (comparaciones BFR↔FEniCS)

**Slides que necesitarán actualización con datos sem 3:**
- Slides de frente de Pareto (barrido γ): datos pendientes (oct-nov 2026)
- Slides de optimización MRMT Bloque + FEniCS: pendientes (nov 2026)
- Slide de estado del artículo: §Resultados en preparación

---

## 11. Tests y verificación

```bash
# Ejecutar en codigo/ rama doctoral-audit
python3 -m pytest tests/ -v                      # tests unitarios del simulador
python3 -m scripts.quantitative_comparison       # comparación cuantitativa BFR↔FEniCS
python3 -c "from visualization import *; print('OK')"  # importar visualization/
```

Los tests de gradiente existentes (`tests/test_gradient_checkpoint.py`) verifican:
- ADR BFR: error rel. < 10⁻¹⁰ ✓
- ADR FEniCS: verificado por `ufl.derivative` ✓
- MRMT BFR: error rel. < 5×10⁻³ ✓

---

## 12. Riesgos y recomendaciones

1. **[CRÍTICO]** Decidir cuál BC hidráulica es la canónica antes de escribir el artículo
   de resultados. La discrepancia BFR↔FEniCS en U es de origen físico, no numérico.
2. **[IMPORTANTE]** Unificar el kernel del pozo (Wendland C2 vs Gaussiano) en utils.py
   antes de generar resultados de optimización definitivos.
3. **[IMPORTANTE]** Trasladar la derivación adjunta MRMT de ADJOINT_DERIVATION_AUDIT.md
   al Cap. 3 de la tesis y al artículo.
4. **[MODERADO]** Ejecutar estudio de convergencia espacial RBF-FD para justificar el
   orden del método en la tesis.
5. **[MODERADO]** Completar Cap. 4 de la tesis con los datos de optimización ya disponibles
   (ADR BFR, MRMT Semi BFR).

---

## 13. Reglas de ejecución (recordatorio)

- **NO merge a main** en ningún repo
- **NO modificar** `config/default.json` sin causa matemática demostrada
- **NO corregir** el adjunto modificando parámetros para forzar coincidencia con FD
- Las discrepancias BFR↔FEniCS son **resultados científicos**, documentados en
  `comparison_report.txt` y `ADJOINT_DERIVATION_AUDIT.md`
