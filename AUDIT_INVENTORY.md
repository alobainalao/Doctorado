# AUDIT_INVENTORY.md — Inventario y Arquitectura del Proyecto Doctoral

**Rama:** `doctoral-audit`  
**Fecha de auditoría:** 2026-10-07  
**Auditor:** Claude Code (claude-sonnet-4-6)  
**Estado:** FASE 2 — Inventario y Arquitectura (sin modificaciones al código)

---

## 0. Resumen ejecutivo

El proyecto doctoral implementa un **sistema de optimización de control PDE-constrained** para gestión de acuíferos profundos contaminados. El sistema resuelve dos PDE acopladas (flujo hidráulico + transporte de contaminantes con transferencia de masa multi-tasa) usando dos backends numéricos independientes: RBF-FD (BFR) y FEM (FEniCS/DOLFINx). El gradiente del funcional de costo respecto a los controles (caudal de extracción Q(t) y posición del pozo z_p) se obtiene por el método adjunto, validado mediante diferencias finitas.

---

## 1. Mapa de repositorios

| Repo | Ruta local | Remote | Rama activa | Uso |
|------|-----------|--------|-------------|-----|
| `codigo/` | `~/dev/doctorado/codigo/` | GitHub | `doctoral-audit` | Simulador Python |
| `tesis/` | `~/dev/doctorado/tesis/` | Overleaf | `doctoral-audit` | Tesis LaTeX |
| `articulos/optimizacion-adjunto/` | `~/dev/doctorado/articulos/optimizacion-adjunto/` | Overleaf | `doctoral-audit` | Artículo LaTeX |
| `presentaciones/avance1-doc/` | `~/dev/doctorado/presentaciones/avance1-doc/` | GitHub Pages | `doctoral-audit` | Presentación React |

---

## 2. Arquitectura del simulador

```
config/default.json          ← Parámetros estándar (fuente de verdad única)
        ↓
config/parameters.py         ← Clase Parameters (lectura JSON + derivados)
config/geometry.py           ← Geometría (real / benchmark)
        ↓
preprocessing/preprocess.py  ← Nodos, operadores RBF-FD, matrices
        ↓
    ┌────────────────────────────────────────────────────────────┐
    │  Backend BFR/RBF-FD  (funtions/)                          │
    │  ─────────────────────────────────────────────────────    │
    │  mesh.py       Poisson-disc + refinamiento near-well      │
    │  get_phi.py    Porosidad interpolada en nodos             │
    │  operators.py  Matrices RBF-FD (flujo, transporte, adj.)  │
    │  step_time.py  step_time() + step_adjoint()               │
    │  utils.py      Funcional J, gradientes, K_ope, D_total    │
    │  runtime.py    Singleton RUNTIME.params                   │
    │  water_supply.py  Restricción suministro hídrico          │
    └────────────────────────────────────────────────────────────┘
    ┌────────────────────────────────────────────────────────────┐
    │  Backend FEM/DOLFINx  (fenicsx/)                          │
    │  ─────────────────────────────────────────────────────    │
    │  mesh.py       gmsh + DOLFINx                             │
    │  forward.py    Solver forward (formas UFL)                │
    │  adjoint.py    Adjunto por ufl.derivative (automático)    │
    │  io.py         Lectura/escritura (HDF5, npz, XDMF)        │
    │  postprocess.py  BTC, campos                              │
    └────────────────────────────────────────────────────────────┘
        ↓
view/animation.py            ← solve_forward, solve_adjoint, optimize_bfr
view/plot.py                 ← funciones de graficado
        ↓
postprocessing/create_btc.py ← Breakthrough curves
        ↓
data/output/{bfr,fenicsx}/data/real/{adr,mrmt_semi,mrmt_block}/
    ├── comparison/          ← simulation_results.npz, adjoint_results.npz
    └── optimization/        ← iter_001 … iter_052/
```

---

## 3. Modelos físicos implementados

### 3.1 Ecuación de flujo (hidráulica)

**Formulación fuerte:**

```
S_s ∂h/∂t − ∇·(K ∇h) = f_Q     en Ω × (0,T)
h = h_in                          en Γ_inlet
h = h_out                         en Γ_outlet
K ∇h · n = 0                      en Γ_wall
```

donde:
- `h(x,z,t)` = carga hidráulica [m]
- `S_s` = coeficiente de almacenamiento específico [m⁻¹]
- `K` = tensor de conductividad hidráulica [m/s], diagonal anisótropo
- `f_Q` = término fuente/sumidero del pozo

**Fuente del pozo:**

```
f_Q(x,z,t) = −Q(t) · g_σ(x − x_p)
```

donde `g_σ` es un kernel:
- **BFR (forward):** Wendland C2 compacto (delta discreta, `utils.py:79–125`)
- **BFR (adjunto/gradiente):** Gaussiano 2D (`utils.py:50–72`)
- **FEniCS:** Gaussiano 2D en ambos (forward y adjunto)

> ⚠️ **Inconsistencia BFR forward ↔ adjunto:** El forward usa kernel compacto pero el gradiente usa Gaussiano. Esta diferencia es conocida y está documentada en `fenicsx/adjoint.py:28-33`. Produce una discrepancia sistemática entre BFR y FEniCS que NO invalida el checkpoint del gradiente (que verifica consistencia interna), pero SÍ debe documentarse en la tesis/artículo.

**Conductividad hidráulica K** (`utils.py:369–376`):

```python
K_y = 8.3e-3 * g * d_z² * φ³ / (ν * (1−φ)²)    # Kozeny-Carman
K_x = α² * K_y                                    # anisotropía: α=2
```

---

### 3.2 Velocidad de Darcy

```
U = −K ∇h     [m/s]
```

Calculado en `operators.py` via `U_vector()`.

---

### 3.3 Modelo ADR (Advección–Difusión–Reacción)

**Formulación fuerte:**

```
φ R ∂C/∂t − ∇·(D ∇C) + U·∇C + λ φ R C = W     en Ω × (0,T)
C = 0                                              en Γ_inlet
∂C/∂n = 0                                         en ∂Ω \ Γ_inlet
C(x,0) = C₀(x)                                    condición inicial
```

donde:
- `C(x,z,t)` = concentración del contaminante [kg/m³]
- `φ` = porosidad efectiva [adim]
- `R` = factor de retardo [adim, default=1]
- `D` = tensor de dispersión hidrodinámica [m²/s]
- `λ` = coeficiente de decaimiento [s⁻¹, default=1e-10]
- `U` = velocidad de Darcy [m/s]

**Tensor de dispersión** (`utils.py:378–414`):

```
D_xx = α_L U_x² / |U| + α_T U_z² / |U| + D_d
D_zz = α_T U_x² / |U| + α_L U_z² / |U| + D_d
```

Parámetros: α_L=10 m, α_T=1 m, D_d=1.2×10⁻⁵ m²/s.

**Extracción de contaminante:**

```
W_ext = −χ_ε(Q) · C · g_σ(x − x_p)
```

con `χ_ε(Q) = ½(1 + tanh(Q/ε))` para suavizar la condición "solo se extrae si Q>0".

---

### 3.4 Modelo MRMT_semi (Multi-Rate Mass Transfer, operador splitting)

**Formulación fuerte:**

```
Región móvil:
  φ R ∂C_m/∂t − ∇·(D ∇C_m) + U·∇C_m + λ φ R C_m + Σ_r λ_r(C_m − C_im,r) = W

Regiones inmóviles (r = 1,…,N_r):
  φ_im,r R ∂C_im,r/∂t = λ_r (C_m − C_im,r)
```

**Discretización (splitting):**

1. Resolver C_m por θ-esquema ignorando el intercambio de masa.
2. Para cada r: `C_im,r^{n+1} = C_m^{n+1} + (C_im,r^n − C_m^{n+1}) exp(−λ_r Δt)`
3. Corrección de C_m: `C_m = C_m + β_r (C_im,r^n − C_im,r^{n+1})`

Parámetros: N_r=3, λ_r derivado de Deff_r y φ_im,r.

**Precisión:** O(Δt) — splitting simple (no Strang). Justifica tolerancia más holgada (5e-3 vs 1e-3) en el test del gradiente.

---

### 3.5 Modelo MRMT_block (acoplamiento implícito)

**Formulación fuerte:** ídem MRMT_semi en ecuaciones, diferente en discretización.

**Discretización (acoplamiento implícito):**

Las N_r ecuaciones de regiones inmóviles se incorporan como diagonal aumentada en la matriz de transporte. Los coeficientes `α_r = (β_r Deff_r) / (2β_r R L² + Δt Deff_r)` se precomputan en `parameters.py`.

```
C_im_new[:, idx] = coef[:, None] * (C_new[idx] + C[idx]) + (1 − 2·coef[:, None]) * C_im[:, idx]
```

**Precisión:** O(Δt) también (esquema implícito de primer orden en C_im).

---

## 4. Tabla de variables y unidades

| Variable | Símbolo | Unidad | Archivo/Línea | Descripción |
|----------|---------|--------|---------------|-------------|
| Carga hidráulica | h | m | step_time.py:25 | Estado hidráulico |
| Velocidad Darcy | U = (Ux, Uz) | m/s | step_time.py:27 | U = −K∇h |
| Concentración móvil | C | kg/m³ | step_time.py:37 | Estado de transporte |
| Concentración inmóvil | C_im[r] | kg/m³ | step_time.py:39-58 | Por región r |
| Conductividad | K = (Kx, Kz) | m/s | utils.py:370-374 | Kozeny-Carman anisótropo |
| Tensor dispersión | D = (Dxx, Dzz) | m²/s | utils.py:378-414 | Dispersión hidrodinámica |
| Porosidad | φ | adim | default.json:phi_const=0.1 | Eficaz móvil |
| Retardo | R | adim | default.json:R=1 | Retardación química |
| Decaimiento | λ | s⁻¹ | default.json:landa=1e-10 | Degradación 1er orden |
| Dispersividad long. | α_L | m | default.json:a_l=10 | |
| Dispersividad trans. | α_T | m | default.json:a_t=1 | |
| Difusividad mol. | D_d | m²/s | default.json:D_d=1.2e-5 | |
| Caudal de extracción | Q(t) | m²/s | utils.py:grad_Q | Control temporal (2D) |
| Posición del pozo | z_p | m | utils.py:grad_zp | Control espacial |
| Variable adjunta h | ψ_h | adim | step_time.py:239 | Coestado hidráulico |
| Variable adjunta C | ψ_C | m³/kg | step_time.py:196 | Coestado de transporte |
| Variable adjunta C_im | ψ_C_im[r] | m³/kg | step_time.py:150 | Coestado inmóvil |
| Funcional J | J | kg²·m/s | utils.py:316 | J = J1 + J2 + J3 |

---

## 5. Funcional de costo J

```
J(Q, z_p) = J1 + J2 + J3

J1 = γ · Σ_n Σ_i (C_i^n)² δ_p,i w_i Δt          [contaminación en pozo]
J2 = κ · |z_p − z₀|²                               [costo de perforación]
J3 = Σ_n Q_n² Δt · |z_p − z₀|                     [costo de bombeo]
```

Parámetros: γ=0.12, κ=1.0, z₀=220 m (profundidad de referencia).

**Implementación:** `utils.py:316–340`

> ⚠️ **Nota matemática:** J1 usa `δ_p` (delta compacta Wendland C2) mientras que el gradiente `grad_Q` usa `g_σ` (Gaussiano). Esta inconsistencia es menor pero debe revisarse en la derivación del adjunto.

---

## 6. Gradientes (optimización)

### 6.1 Gradiente respecto a Q(t)

```
∂J/∂Q_n = 2|z_p − z₀| Q_n Δt
         + Σ_i [ψ_h,i^n + ψ_C,i^n · C_i^n · χ'_ε(Q_n)] g_σ,i w_i
```

**Implementación:** `utils.py:185–232`

### 6.2 Gradiente respecto a z_p

```
∂J/∂z_p = 2γ Σ_n Σ_i C_i^n (∂C/∂z)_i δ_p,i w_i Δt
          − Σ_n Q_n² Δt
          + 2κ(z_p − z₀)
          + Σ_n [Q_n ψ_h,i^n + χ_ε(Q_n) ψ_C,i^n C_i^n] · (∂g_σ/∂z_p)_i w_i Δt
```

**Implementación:** `utils.py:234–313`

---

## 7. Matriz de backends × modelos × tipo de corrida

| Modelo | Fwd BFR | Adj BFR | Opt BFR | Fwd FEniCS | Adj FEniCS | Opt FEniCS |
|--------|---------|---------|---------|-----------|-----------|-----------|
| ADR | ✓ validado | ✓ FD<1e-3 | ✓ 52 iter | ✓ validado | ✓ FD<1e-3 | ⊙ en prueba |
| MRMT_semi | ✓ validado | ✓ FD<5e-3 | ⊙ en prueba | ✓ validado | ✓ FD<5e-3 | ⊙ en prueba |
| MRMT_block | ✓ validado | ✓ FD<5e-3 | ⊙ en prueba | ✓ validado | ✓ FD<5e-3 | ⊙ en prueba |

Leyenda: ✓ completo · ⊙ en prueba · ○ pendiente

---

## 8. Sistema de tests

| Test | Archivo | Qué valida | Tolerancia |
|------|---------|-----------|-----------|
| Gradiente global ADR BFR | `tests/test_gradient_checkpoint.py` | grad_Q, grad_zp vs FD | 1e-3 |
| Gradiente paso-a-paso ADR BFR | `tests/test_adjoint_identity.py` | ∂J/∂Q_n por paso vs FD | 5e-3 |
| Gradiente MRMT BFR | `tests/test_gradient_checkpoint_mrmt.py` | MRMT_semi y MRMT_block | 5e-3 |
| Gradiente ADR FEniCS | `fenicsx/test_gradient_checkpoint.py` | grad_Q FEniCS vs FD | 1e-3 |
| Gradiente MRMT FEniCS | `fenicsx/test_gradient_checkpoint_mrmt_fenicsx.py` | MRMT_semi, MRMT_block FEniCS | 5e-3 |
| Smoke test FEniCS forward | `fenicsx/test_forward_smoke.py` | Corre sin errores | — |

---

## 9. Sistema actual de figuras (diagnóstico)

**Estado:** No existe sistema centralizado. Figuras dispersas en múltiples scripts.

| Script | Qué genera | Formato | DPI |
|--------|-----------|---------|-----|
| `view/plot.py` | Campos H, U, C | Matplotlib | no estandarizado |
| `view/animation.py` | Animaciones MP4/frames | ffmpeg | no estandarizado |
| `scripts/comparar_gradientes.py` | Gradientes BFR vs FEniCS | PNG | no estandarizado |
| `scripts/comparar_adjunto.py` | Campos adjuntos BFR vs FEniCS | PNG | no estandarizado |
| `scripts/generar_comparacion.py` | Comparación visual | PNG | no estandarizado |
| `tesis/salidas/*.png` | Figuras de maestría heredadas | PNG | desconocido |

**Faltan:**
- Perfiles THESIS / PAPER / PRESENTATION
- Colormaps estandarizados
- DPI y tamaños fijados
- Sistema de exportación centralizado

---

## 10. Flujo de datos completo

```
config/default.json
        ↓
preprocessing/preprocess.py   ← Genera: nodos, operadores, matrices sparse
        ↓
view/animation.py::solve_forward()
        ↓ guarda:
data/output/bfr/data/real/{model}/simulation_results.npz
    - H: (Nt, N)    carga hidráulica
    - U: (Nt, N, 2) velocidades Darcy
    - C: (Nt, N)    concentración móvil
    - C_im: (Nt, Nr, N)  concentración inmóvil (si MRMT)
        ↓
view/animation.py::solve_adjoint()
        ↓ guarda:
data/output/bfr/data/real/{model}/adjoint_results.npz
    - psi_H: (Nt, N)    adjunto hidráulico
    - psi_C: (Nt, N)    adjunto de concentración
    - psi_C_im: (Nt, Nr, N)  adjunto inmóvil (si MRMT)
        ↓
utils.py::grad_Q(), grad_zp()
        ↓
view/animation.py::optimize_bfr()   ← Iteraciones del optimizador
        ↓ guarda:
data/output/bfr/data/real/{model}/optimization/iter_{k:03d}/
    - simulation_results.npz
    - adjoint_results.npz
    - gradient.npz
```

**Nota:** La comparación BFR↔FEniCS usa `scripts/generar_comparacion.py` que interpola BFR → grid FEniCS o viceversa para visualización. La comparación es actualmente cualitativa (no calcula normas L²/L∞).

---

## 11. Dominio geométrico

### Dominio real

**Vértices del polígono Ω ⊂ ℝ²** (`geometry.py:36–43`):

```
[816,   -2690.5]  →  [4354, -1757.5]  →  [10755, -768.5]
[10755,  250.5]   →  [816,   250.5]   →  (cierre)
```

**Dimensiones:** Lx ≈ 9939 m, Ly ≈ 2941 m (trapecio irregular)

**Puntos especiales:**
- Pozo de extracción: x_p = (3700, −700) m
- Fuente contaminante: x_s = (9116, 109) m

**Fronteras:**
- Γ_inlet: lado izquierdo (x = 816 m), carga h prescrita (perfil vertical)
- Γ_outlet: lado derecho (x = 10755 m), carga h prescrita
- Γ_wall: lados superior e inferior, flujo nulo (Neumann homogéneo)

> ⚠️ **Falta en la tesis:** Las fronteras no están definidas formalmente con sus respectivas condiciones; el dominio se describe con una figura TikZ pero sin la notación matemática ∂Ω = Γ_inlet ∪ Γ_outlet ∪ Γ_wall.

---

## 12. Estructura de documentos científicos

### 12.1 Tesis (doc.tex)

| Capítulo | Título | Líneas | Estado |
|---------|--------|--------|--------|
| 1 | Introducción | ~200 | ✓ completo |
| 2 | Modelo Matemático y Discretización | ~100 | ⊙ parcial |
| 3 | Problema de Optimización | 1770 | ✓ sustancial |
| 4 | Resultados y Conclusiones | 312 | ✗ esqueleto |
| Apéndice | — | — | — |

**Principales vacíos matemáticos en la tesis:**
- Sin espacios de Sobolev ni formulación débil
- Fronteras ∂Ω no particionadas formalmente
- Cap. 4 §2-§6 vacíos (MRMT, adjunto, optimización)
- Figures de resultados heredadas de maestría

### 12.2 Artículo (optimizacion+adjunto.tex)

| Sección | Contenido | Estado |
|---------|-----------|--------|
| Abstract | Resumen del método | ⚠️ TODO: reescribir sin fórmulas |
| §1 Introducción | Motivación, antecedentes | ✓ |
| §2 Modelo matemático | Flujo + transporte | ✓ |
| §3 Modelación de fuentes | Pozo, extracción, fuente | ✓ |
| §4 (implícita) Espacios funcionales | — | ✗ ausente |
| §5 Funcionales de costo | J1, J2, J3, multiobjetivo | ✓ |
| §6-7 Optimización + Pareto | Scalarización, restricciones | ✓ |
| §8 Método adjunto | Lagrangiano, derivación, gradientes | ✓ ADR |
| §Resultados | Validación + optimización | ✗ TODO |

**Nota:** §8 cubre la derivación adjunta del ADR. No incluye MRMT. El artículo está clasificado para SIAM (MSC 49K20, 35Q93, 76S05, 90C29).

---

## 13. Problemas clasificados

### 🔴 CRÍTICOS (pueden invalidar resultados o publicación)

1. **Cap. 4 de la tesis vacío** — §2-§6 son esqueletos sin contenido. Sin estos, la tesis no puede presentarse.
2. **Artículo sin sección de resultados** — El abstract indica "resultados numéricos" pero la sección no existe. No enviable.
3. **Comparación BFR↔FEniCS cualitativa** — No hay normas L²/L∞ calculadas. "Los campos se parecen" no es una comparación numérica válida para un doctorado en matemáticas.
4. **Kernel inconsistente BFR forward/adjunto** — El forward usa Wendland C2 en el pozo; el gradiente usa Gaussiano. Esto produce un desacuerdo sistemático entre el funcional que se minimiza y su gradiente. El checkpoint valida la consistencia interna (funcional con Gaussiano = adjunto con Gaussiano), pero el forward produce un J diferente al que el gradiente cree minimizar.

### 🟠 IMPORTANTES (corregir antes de defensa/publicación)

5. **Sin espacios de Sobolev en la tesis** — El problema físico admite solución en H¹(Ω); no documentarlo debilita el fundamento matemático.
6. **Sin formulación débil en la tesis** — La formulación variacional existe en el código FEniCS pero no está derivada en el documento.
7. **Derivación adjunta en artículo/tesis no cubre MRMT** — §8 del artículo trata solo ADR. Los adjuntos de MRMT_semi y MRMT_block están implementados y validados pero no documentados matemáticamente.
8. **Definición formal de ∂Ω ausente** — Las BCs están definidas en código pero no en la tesis con la partición formal Γ_D ∪ Γ_N.
9. **Sin sistema centralizado de visualización** — Las figuras no tienen DPI/tamaños/colores estandarizados para thesis/paper/presentation.

### 🟡 MODERADOS (mejoran rigor sin ser bloqueantes)

10. **θ-esquema: BFR usa θ=0.9, FEniCS usa θ propio** — Pequeña diferencia sistemática en la comparación temporal.
11. **Sin análisis de convergencia RBF-FD** — No hay estudio error vs resolución para demostrar la tasa experimental de convergencia.
12. **tesis_proyect/ es legacy vacío** — Confunde la arquitectura del repositorio.
13. **configs/config1-3.json vacíos** — Solo default.json es la fuente de verdad; los otros generan confusión.
14. **Sin tabla maestra de parámetros con unidades** — Los parámetros están en default.json pero sin descripción de unidades en el documento.

### 🔵 EDITORIAL (formato y presentación)

15. **DPI, tamaños y fuentes no estandarizados** — Para SIAM se requiere mínimo 300 DPI, vectorial preferido.
16. **Sin glosario de nomenclatura maestro** — Se usan ψ_h, ψ_C, psiH, psi_H en diferentes lugares.
17. **Captions incompletos** — Varias figuras no especifican variable, unidades, escenario.
18. **Abstract del artículo contiene fórmulas** — SIAM no permite fórmulas en el abstract.

---

## 14. Fuentes de estándares relevantes

| Fuente | Aplicabilidad | URL |
|--------|--------------|-----|
| SIAM Author Guidelines | Artículo (revista objetivo) | https://epubs.siam.org/journal/siopt/instructions-for-authors |
| SIAM Journal on Control and Optimization | Revista primaria | 35 págs máx, sin fórmulas en abstract |
| AMS-LaTeX | Tipografía matemática | https://www.ams.org/arc/tex/amscls/amsthm.pdf |
| Brezis (Functional Analysis) | Espacios de Sobolev, Lax-Milgram | Libro de referencia |
| Evans (PDE) | Existencia, unicidad, regularidad | Libro de referencia |
| Tröltzsch (Optimal Control of PDEs) | Marco teórico de optimización | Libro de referencia |
| Fornberg & Flyer (2015) | Convergencia RBF-FD | A Primer on Radial Basis Functions |
| Haggerty & Gorelick (1995) | MRMT — fundamento físico | WRR 31(10) |

---

## 15. Plan de acción por prioridad

| Fase | Descripción | Repo | Prioridad |
|------|-------------|------|-----------|
| **3** | ADJOINT_DERIVATION_AUDIT.md — derivación matemática completa | codigo/ | 🔴 |
| **4** | scripts/quantitative_comparison.py — normas L², L∞ | codigo/ | 🔴 |
| **5** | visualization/ — sistema centralizado de figuras | codigo/ | 🟠 |
| **6a** | Tesis Cap. 2 — espacios funcionales + formulación débil | tesis/ | 🟠 |
| **6b** | Tesis Cap. 3 — extensión adjunto a MRMT | tesis/ | 🟠 |
| **6c** | Tesis Cap. 4 — completar resultados con datos existentes | tesis/ | 🔴 |
| **7** | Artículo — abstract + sección de resultados | articulos/ | 🔴 |
| **8** | Presentación — verificar narrativa y matriz backends | presentaciones/ | 🟡 |

---

*Documento generado en rama `doctoral-audit`. NO mezclar con `main` sin revisión del usuario.*
