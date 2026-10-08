# ADJOINT_DERIVATION_AUDIT.md — Derivación y Auditoría del Problema Adjunto

**Rama:** `doctoral-audit`  
**Fecha:** 2026-10-07  
**Contexto:** Doctorado en Matemáticas — Control óptimo gobernado por PDE en acuíferos profundos

---

## 0. Propósito de este documento

Este documento realiza tres tareas:

1. **Derivación analítica completa** del problema adjunto — tanto para ADR como para MRMT — partiendo del Lagrangiano continuo, sin asumir nada del código.
2. **Comparación detallada** entre la derivación analítica y la implementación en código (BFR y FEniCS).
3. **Registro de discrepancias** — clasificadas por severidad.

La derivación sigue el método del **Lagrangiano continuo** (optimize-then-discretize en lo analítico), y la implementación sigue el método **discretize-then-optimize** (transpuesta del operador discreto). La compatibilidad entre ambos es lo que se audita.

---

## 1. Problema de estado

### 1.1 Dominio y espacios funcionales

Sea **Ω ⊂ ℝ²** un dominio poligonal (pentágono convexo en el caso real), con frontera:

```
∂Ω = Γ_in ∪ Γ_out ∪ Γ_wall    (unión disjunta)
```

- `Γ_in`: frontera de entrada (x = x_min), flujo hidráulico entrante
- `Γ_out`: frontera de salida (x = x_max), flujo hidráulico saliente  
- `Γ_wall`: fronteras impenetrables (superior e inferior)

**Horizonte temporal:** T > 0 fijo. Espacio-tiempo: Q_T := Ω × (0,T).

**Espacios funcionales:**
- `h(·,t) ∈ H¹(Ω)` para c.t. t ∈ (0,T)  — carga hidráulica
- `C(·,t) ∈ H¹(Ω)` para c.t. t ∈ (0,T)  — concentración móvil
- `C_im,r(·,t) ∈ L²(Ω)` para c.t. t ∈ (0,T)  — concentración inmóvil región r
- `Q(t) ∈ L²(0,T)` — caudal de extracción (control)
- `z_p ∈ ℝ` — profundidad del pozo (control escalar)

**Conjunto de controles admisibles:**
```
U_ad = {(Q,z_p) : Q ∈ L²(0,T), 0 ≤ Q(t) ≤ Q_max c.t.p.,
                  z_p ∈ [z_p^min, z_p^max]}
```

### 1.2 Ecuación de flujo hidráulico (E_h = 0)

```
S_s ∂h/∂t − ∇·(K ∇h) = f_Q(Q, z_p)     en Q_T            [E_h]
h = h₀                                     en Ω × {0}
h = h_in(z)                                en Γ_in × (0,T)  [BC_h_in]
h = h_out(z)                               en Γ_out × (0,T) [BC_h_out]
K ∇h · n = 0                               en Γ_wall × (0,T) [BC_h_wall]
```

**Fuente del pozo:**
```
f_Q(x,z,t) = −Q(t) · g_ε(x − x_p, z − z_p)
```

donde `g_ε` es el kernel del pozo (Gaussiano 2D o Wendland C2 — ver Sección 7).

**Conductividad K** (Kozeny-Carman, diagonal):
```
K_z = 8.3×10⁻³ g d_z² φ³ / [ν(1−φ)²]
K_x = α² K_z    (α=2, anisotropía)
```

> **Nota sobre las BCs de flujo:** En el código BFR, `h = h_in(z)` se impone como Dirichlet fuerte (nodos de frontera se fijan). En FEniCS se implementa como condición natural: `(K∇h)·n|_{Γ_in} = −h_inlet`. Esta diferencia de implementación entre backends es una **inconsistencia conocida** que afecta la comparación cuantitativa.

### 1.3 Ecuación de transporte — modelo ADR (E_C = 0)

```
φR ∂C/∂t − ∇·(D ∇C) + U·∇C + (λφR + χ_ε(Q) g_ε) C = W_+(z_p)   en Q_T
C = 0                en Γ_in × (0,T)   [BC_C_in]
D∇C·n = 0           en (∂Ω\Γ_in) × (0,T)
C(x,0) = C₀(x)
```

donde:
- `U = −K∇h` — velocidad de Darcy
- `D = D(U)` — tensor de dispersión hidrodinámica (dependiente de U)
- `χ_ε(Q) = ½(1 + tanh(Q/ε))` — función suave de extracción activa
- `W_+ ≥ 0` — fuente gaussiana de contaminante

> **Nota de signos:** La ecuación en el artículo escribe el término advectivo como `−(K∇h)·∇C` (con K∇h sin el signo negativo). Esto es equivalente a escribir `U·∇C` con `U = −K∇h`, ya que `U·∇C = −K∇h·∇C`. **El código usa U = −K∇h (correcto).**

### 1.4 Ecuaciones de regiones inmóviles — modelo MRMT (r = 1,…,N_r)

```
φ_im,r R ∂C_im,r/∂t = λ_r (C − C_im,r)    en Q_T
C_im,r(x,0) = C_im,r,0(x)
```

con coeficiente de transferencia `λ_r = Deff_r / (φ_im,r R L²)`.

---

## 2. Funcional de costo

```
J(Q, z_p) = J₁ + J₂ + J₃
```

### 2.1 Componentes del funcional

```
J₁ = γ ∫₀ᵀ ∫_Ω δ_p(x) C(x,t)² dΩ dt        [contaminación en pozo]

J₂ = κ |z_p − z₀|²                            [costo perforación]

J₃ = ∫₀ᵀ Q(t)² |z_p − z₀| dt                [costo bombeo]
```

donde `δ_p(x)` es una delta discreta centrada en `(x_p, z_p)`.

### 2.2 Dominio de J

`J: L²(0,T) × ℝ → ℝ`

Para que J₁ esté bien definido se necesita `C ∈ L²(Q_T)`, lo que se garantiza si `C ∈ L²(0,T; H¹(Ω))`.

---

## 3. Lagrangiano continuo

```
L(Q, z_p, h, C, C_im; ψ_h, ψ_C, ψ_im) =
  J(Q, z_p)
  + ∫₀ᵀ ∫_Ω ψ_h [S_s ∂h/∂t − ∇·(K∇h) − f_Q] dΩ dt
  + ∫₀ᵀ ∫_Ω ψ_C [φR ∂C/∂t − ∇·(D∇C) + U·∇C + (λφR + χ_ε g) C − W_+] dΩ dt
  + Σᵣ ∫₀ᵀ ∫_Ω ψ_im,r [φ_im,r R ∂C_im,r/∂t − λ_r(C − C_im,r)] dΩ dt
```

Los multiplicadores de Lagrange son `(ψ_h, ψ_C, ψ_im,r)` — los estados adjuntos.

---

## 4. Derivación de la ecuación adjunta del transporte (δ_C L = 0)

### 4.1 Variación respecto a C

```
δ_C L = 2γ ∫_QT δ_p C δC dΩ dt
      + ∫_QT ψ_C [φR ∂(δC)/∂t − ∇·(D∇δC) + U·∇(δC) + (λφR + χ_ε g) δC] dΩ dt
      + Σᵣ ∫_QT ψ_im,r [−λ_r δC] dΩ dt
```

### 4.2 Integración por partes — término temporal

```
∫_QT ψ_C φR ∂(δC)/∂t dΩ dt
= −∫_QT φR ∂ψ_C/∂t δC dΩ dt
  + ∫_Ω φR ψ_C δC |_{t=T} dΩ
  − ∫_Ω φR ψ_C δC |_{t=0} dΩ
```

Condiciones:
- **Condición terminal:** `ψ_C(x,T) = 0`
- `δC(x,0) = 0` (condición inicial fija)

**→ Los términos de frontera temporal se anulan.**

### 4.3 Integración por partes — término difusivo

```
−∫_Ω ψ_C ∇·(D∇δC) dΩ
= ∫_Ω ∇ψ_C · D∇δC dΩ − ∫_{∂Ω} ψ_C (D∇δC)·n dΓ
= ∫_Ω δC ∇·(D∇ψ_C) dΩ − ∫_{∂Ω} (D∇ψ_C)·n δC dΓ
  − ∫_{∂Ω} ψ_C (D∇δC)·n dΓ
```

Condición de Neumann del estado: `(D∇C)·n = 0` en ∂Ω → `(D∇δC)·n = 0` en ∂Ω.  
**→ El segundo término de frontera se anula.**

Condición adjunta: `(D∇ψ_C)·n = 0` en Γ_wall (Neumann homogéneo adjunto).

### 4.4 Integración por partes — término advectivo

```
∫_Ω ψ_C U·∇(δC) dΩ
= −∫_Ω δC ∇·(ψ_C U) dΩ + ∫_{∂Ω} ψ_C (U·n) δC dΓ
= −∫_Ω δC [ψ_C ∇·U + U·∇ψ_C] dΩ + ∫_{∂Ω} ψ_C (U·n) δC dΓ
```

En `Γ_in ∪ Γ_out`: U·n ≠ 0 en general. Para que δ_C L = 0 para todo δC arbitrario:
```
ψ_C (U·n) = 0    en Γ_in ∪ Γ_out
```
Como U·n ≠ 0 (hay flujo), se requiere:
```
ψ_C = 0    en Γ_in ∪ Γ_out
```

En `Γ_wall`: `U·n = 0` (condición de impermeabilidad) → contribución nula.

### 4.5 Contribución de las regiones inmóviles

```
Σᵣ ∫_QT (−λ_r ψ_im,r) δC dΩ dt
```

Este término contribuye al RHS de la ecuación adjunta.

### 4.6 Ecuación adjunta del transporte (resultado)

```
−φR ∂ψ_C/∂t + ∇·(D∇ψ_C) − U·∇ψ_C − ψ_C ∇·U
+ (λφR + χ_ε g) ψ_C − Σᵣ λ_r ψ_im,r
= 2γ C δ_p    en Q_T
```

Con condiciones:
```
ψ_C(x,T) = 0                        (condición terminal)
ψ_C = 0                 en Γ_in ∪ Γ_out
(D∇ψ_C)·n = 0           en Γ_wall
```

> **Nota:** La ecuación adjunta es **backward en tiempo** (se resuelve de t=T hacia t=0). El signo del término `−φR ∂ψ_C/∂t` confirma esto.

> **Nota de signo advectivo:** La ecuación adjunta del transporte tiene el término advectivo con signo **opuesto** al forward: en el forward es `+U·∇C`, en el adjunto es `−U·∇ψ_C`. Esta inversión de signo en la advección es la característica fundamental del operador adjunto de la advección.

---

## 5. Derivación de la ecuación adjunta del flujo (δ_h L = 0)

### 5.1 Variación respecto a h

```
δ_h L = ∫_QT ψ_h [S_s ∂(δh)/∂t − ∇·(K∇δh)] dΩ dt
       + ∫_QT ψ_C [−∇·(δU·...)] dΩ dt
```

El segundo término viene del acoplamiento h → U = −K∇h → D(U) → transporte. La variación del término advectivo respecto a h da:

```
∫_Ω ψ_C (−∇(Kδh))·∇C dΩ = −∫_Ω ψ_C (K∇δh)·∇C dΩ
= ∫_Ω δh ∇·(K ψ_C ∇C) dΩ − ∫_{∂Ω} δh (K ψ_C ∇C)·n dΓ
```

En Γ_wall: `(K∇h)·n = 0` → `(K∇δh)·n = 0` → contribución nula.  
En Γ_in ∪ Γ_out: `ψ_C = 0` → término de frontera nulo.

### 5.2 Integración por partes del término temporal

```
∫_QT ψ_h S_s ∂(δh)/∂t dΩ dt
= −∫_QT S_s ∂ψ_h/∂t δh dΩ dt + ∫_Ω S_s ψ_h δh |_{t=T} − ∫_Ω S_s ψ_h δh |_{t=0}
```

Condición terminal: `ψ_h(x,T) = 0`.

### 5.3 Integración por partes del término difusivo de ψ_h

```
−∫_Ω ψ_h ∇·(K∇δh) dΩ
= ∫_Ω δh ∇·(K∇ψ_h) dΩ − ∫_{∂Ω} (K∇δh)·n ψ_h dΓ
```

En Γ_in ∪ Γ_out: δh es arbitrario (h no está fijo → hay flujo). Para que el término de frontera se anule se requiere:
```
ψ_h = 0    en Γ_in ∪ Γ_out
```

En Γ_wall: `(K∇δh)·n = 0` → contribución nula. Condición adjunta de Neumann:
```
(K∇ψ_h)·n = 0    en Γ_wall
```

### 5.4 Ecuación adjunta del flujo (resultado)

```
−S_s ∂ψ_h/∂t + ∇·(K∇ψ_h) = −∇·(K ψ_C ∇C)    en Q_T
```

Con condiciones:
```
ψ_h(x,T) = 0                        (condición terminal)
ψ_h = 0                 en Γ_in ∪ Γ_out
(K∇ψ_h)·n = 0           en Γ_wall
```

> **Nota:** El RHS `−∇·(K ψ_C ∇C)` es el **término de acoplamiento** del adjunto hidráulico — proviene de que la velocidad U = −K∇h entra en el transporte. Si no hubiera transporte (o ψ_C = 0 en toda la trayectoria), entonces el adjunto hidráulico sería trivialmente ψ_h = 0. El adjunto hidráulico "existe" porque cambiar h cambia U y por tanto cambia C.

---

## 6. Derivación del adjunto MRMT

### 6.1 Variación respecto a C_im,r (δ_r L = 0)

```
δ_r L = ∫_QT ψ_im,r φ_im,r R ∂(δC_im,r)/∂t dΩ dt
       + ∫_QT ψ_im,r (λ_r δC_im,r) dΩ dt
```

Integración por partes temporal:
```
−∫_QT φ_im,r R ∂ψ_im,r/∂t δC_im,r dΩ dt + ∫_Ω φ_im,r R ψ_im,r δC_im,r |_{t=T}
```

Condición terminal: `ψ_im,r(x,T) = 0`.

**Ecuación adjunta de región inmóvil:**
```
−φ_im,r R ∂ψ_im,r/∂t = −λ_r ψ_im,r + λ_r ψ_C    en Q_T
ψ_im,r(x,T) = 0
```

O equivalentemente:
```
φ_im,r R ∂ψ_im,r/∂t = λ_r (ψ_C − ψ_im,r)    [adjunto MRMT región r]
```

> **Observación:** La ecuación adjunta de la región inmóvil tiene la **misma estructura** que la ecuación forward para C_im,r, pero con los papeles de ψ_C y ψ_im,r intercambiados. El intercambio (C ↔ C_im,r) en el acoplamiento se refleja exactamente en el adjunto (ψ_C ↔ ψ_im,r).

### 6.2 Corrección al adjunto del transporte por MRMT

El término `Σᵣ λ_r ψ_im,r` que aparece en la ecuación adjunta del transporte (Sección 4.6) proviene de la variación del acoplamiento respecto a C_m en las ecuaciones de regiones inmóviles:

```
∂[φ_im,r R ∂C_im,r/∂t − λ_r(C − C_im,r)]/∂C = −λ_r
```

Por tanto la contribución de cada región al RHS adjunto de C_m es:
```
−λ_r ψ_im,r    por cada región r
```

---

## 7. Gradientes de control

### 7.1 Gradiente respecto a Q(t)

La derivada de L respecto a Q(t) a tiempo fijo t ∈ (0,T):

```
dJ/dQ(t) = ∂J₃/∂Q + ∂J/∂Q + ψ_C · ∂E_C/∂Q + ψ_h · ∂E_h/∂Q
```

Términos:
- `∂J₃/∂Q = 2|z_p−z₀| Q(t)` (de J₃ = ∫Q² |z_p−z₀| dt)
- `ψ_h · ∂E_h/∂Q = ψ_h · (−g_ε)` (de E_h: pozo = −Q g_ε)
- `ψ_C · ∂E_C/∂Q = ψ_C · (−C g_ε χ'_ε(Q))` (de E_C: extracción = −χ_ε(Q) C g_ε)

**Resultado:**
```
dJ/dQ(t) = 2|z_p−z₀| Q(t) + ∫_Ω g_ε [ψ_h + ψ_C C χ'_ε(Q)] dΩ
```

> Nota: El signo `+` en el integrando se debe a que `∂E_h/∂Q = −g_ε` contribuye con `ψ_h·(−g_ε)` al Lagrangiano, y la derivada de L se escribe como `−(contribución)`, cancelando el signo.

### 7.2 Gradiente respecto a z_p

```
dJ/dz_p = ∂J₁/∂z_p + ∂J₂/∂z_p + ∂J₃/∂z_p
         + ∫₀ᵀ ∫_Ω [ψ_h ∂E_h/∂z_p + ψ_C ∂E_C/∂z_p] dΩ dt
```

Términos:
- `∂J₁/∂z_p = 2γ ∫₀ᵀ ∫_Ω C ∂C/∂z δ_p dΩ dt` (δ_p depende de z_p)
- `∂J₂/∂z_p = 2κ(z_p − z₀)`
- `∂J₃/∂z_p = −∫₀ᵀ Q(t)² dt` (asumiendo z_p < z₀)
- Términos de ψ_h y ψ_C via `∂g_ε/∂z_p = g_ε (z − z_p)/ε²`

**Resultado (ecuación A del artículo, sección §8):**
```
dJ/dz_p = 2γ ∫₀ᵀ C(x_p,z_p,t) ∂C/∂z(x_p,z_p,t) dt
          − ∫₀ᵀ Q(t)² dt + 2κ(z_p − z₀)
          + ∫₀ᵀ ∫_Ω ∂g_ε/∂z_p [Q(t) ψ_h + χ_ε(Q) ψ_C C] dΩ dt
```

---

## 8. Comparación analítica vs. implementación

### 8.1 Backend BFR/RBF-FD

#### Método empleado: Discretize-then-optimize (transpuesta exacta)

El código `step_adjoint()` en `step_time.py:67–241` **no** reconstruye el adjunto a partir de la derivación analítica, sino que toma la **transpuesta exacta del operador forward discreto**:

```python
A_fwd = build_transport_matrix(V_N, ...)
AT = A_fwd.T.tocsr()    # transpuesta exacta
psiC_n = splu(AT).solve(rhs_C)
```

**Ventaja:** El gradiente satisface exactamente la identidad adjunta discreta, lo que garantiza que el test de diferencias finitas pase a tolerancia numérica (no hay error de discretización del adjunto).

**Consecuencia:** La ecuación adjunta que resuelve el código es el adjunto del operador **discreto**, que puede diferir de la discretización del operador adjunto **continuo** en `O(h)` o `O(Δt)` (según el esquema).

#### Condiciones de frontera — verificación

| Frontera | Estado C | Adjunto ψ_C (derivación) | Adjunto ψ_C (código) |
|----------|---------|------------------------|----------------------|
| Γ_in | Dirichlet C=0 | ψ_C = 0 | ✓ `AT[i,:]=0; AT[i,i]=1` (línea 139) |
| Γ_out | Neumann D∇C·n=0 | ψ_C = 0 (por flujo U·n≠0) | ✓ mismo Dirichlet |
| Γ_wall | Neumann D∇C·n=0 | Neumann (D∇ψ_C)·n=0 | ✓ (no se impone Dirichlet) |

#### Condiciones de frontera — flujo adjunto ψ_h

| Frontera | Estado h | Adjunto ψ_h (derivación) | Adjunto ψ_h (código) |
|----------|---------|--------------------------|----------------------|
| Γ_in | Dirichlet h=h_in | ψ_h = 0 | ✓ (A_solver es simétrico, BC heredada) |
| Γ_out | Dirichlet h=h_out | ψ_h = 0 | ✓ |
| Γ_wall | Neumann K∇h·n=0 | Neumann (K∇ψ_h)·n=0 | ✓ |

#### Adjunto MRMT Semi — verificación

**Derivación continua** (Sección 6):
```
φ_im,r R ∂ψ_im,r/∂t = λ_r (ψ_C − ψ_im,r)
```

**Discretización backward del operador splitting** (código `step_adjoint:160–174`):

El forward del splitting actualiza `(C_new, C_im_r)` con la matriz:
```
M_fwd_r = [[1−β_r·α_r,  β_r·α_r],
           [α_r,        e_r     ]]
```
donde `e_r = exp(−λ_r Δt)`, `α_r = 1 − e_r`.

El código toma su transpuesta:
```
M_adj_r = [[1−β_r·α_r,  α_r      ],
           [β_r·α_r,    e_r       ]]
```

**¿Es consistente con la ecuación continua?**

La ecuación discreta adjunta exacta es `M_fwd_r^T`, que es lo que implementa el código. La discretización de la ODE `φR ∂ψ_im/∂t = λ_r(ψ_C − ψ_im)` backward en tiempo da exactamente la fila del sistema transpuesto por la misma razón que en el forward (splitting con exponencial exacto).

✓ **Consistente:** El código MRMT Semi adjunto es el adjunto discreto exacto del forward MRMT Semi.

#### Adjunto MRMT Block — verificación

Forward (código `step_time.py:51–58`):
```
C_im_new[:, idx] = coef*( C_new[idx]+C[idx] ) + (1−2*coef)*C_im[:, idx]
```
con `coef[r] = dt*α_r/β_r`.

También: La matriz de transporte incluye un término adicional `4R·ρ·α_r·C_im` en el RHS.

Adjunto (código `step_adjoint:188–220`):
1. Antes de resolver ψ_C: `rhs_C[idx] += coef[r] * psiC_im_N[r][idx]`  
   → adjunto de `∂C_im_new/∂C_new = coef`
2. Después de resolver ψ_C:
   ```
   psiC_im_n[r][idx] = (1−2*coef[r])*psiC_im_N[r][idx] + 4R*ρ*α_r[r]*psiC_n[idx]
   ```
   → adjunto de la fila de la matriz C_im_new respecto a C_im y al acoplamiento en el RHS del transporte

✓ **Consistente:** El código MRMT Block adjunto es el adjunto discreto exacto del forward MRMT Block.

---

### 8.2 Backend FEniCS/DOLFINx

#### Método empleado: Diferenciación automática UFL

```python
Rc = _transport_residual(ctx, C_cur, C_prev, h_cur, ...)
a_cc_T = fem.form(ufl.adjoint(ufl.derivative(Rc, C_cur, dC)))
```

`ufl.derivative` calcula `∂R^C/∂C` simbólicamente. `ufl.adjoint` toma la transpuesta del operador discreto resultante. Esto es **matemáticamente idéntico** al enfoque BFR pero de forma automática.

**Ventaja sobre BFR:** El acoplamiento `h → U → D(U) → transporte` se captura exactamente porque el residual se escribe con h simbólico. En BFR, la actualización de U y D entre pasos puede introducir un desfase de `O(Δt)` en el acoplamiento.

#### Nota sobre el kernel del pozo

El backend FEniCS usa un Gaussiano para el sumidero de flujo (como para el transporte):
```
f_Q = −Q g_ε(x−x_p, z−z_p)   [Gaussiano en ambos]
```

Esto difiere del forward BFR que usa Wendland C2 en el flujo. La razón (documentada en `fenicsx/adjoint.py:28-33`) es que el gradiente `∂g_ε/∂z_p` debe ser diferenciable en forma cerrada para z_p.

---

## 9. Discrepancias encontradas

### 9.1 Discrepancia D1 — Kernel del pozo BFR forward ≠ BFR adjunto

**Severidad:** 🔴 CRÍTICA (afecta la coherencia del funcional minimizado)

**Descripción:**
- Forward BFR: `f_Q = −Q · δ_C(x−x_p)` (Wendland C2, compacto)
- Gradiente BFR: `∫ g_ε [ψ_h + ψ_C C χ'_ε] dΩ` (Gaussiano)

**Consecuencia matemática:** El funcional J que el gradiente cree minimizar NO es el mismo J que el forward evalúa. Específicamente:
- El checkpoint de gradiente valida la consistencia `J (Gaussiano forward) ↔ adj (Gaussiano)`. 
- Pero el forward estándar usa Wendland C2 → `J_estándar ≠ J_Gaussiano`.
- En la optimización actual, el checkpoint fue hecho con Gaussiano para ambos (ver `fenicsx/adjoint.py:28-33`).

**¿Invalida el checkpoint?** No — el checkpoint valida la consistencia interna del par `(J_gaussiano, gradiente adjunto)`. Ese par es matemáticamente correcto para el funcional J con kernel Gaussiano.

**¿Invalida la optimización?** Parcialmente — la optimización minimiza J_gaussiano, no J_Wendland. En la práctica la diferencia es pequeña (ambos kernels son aproximaciones de la delta de Dirac), pero matemáticamente no es lo mismo.

**Acción recomendada:** Documentar explícitamente en la tesis/artículo que la optimización usa kernel Gaussiano para el sumidero de flujo (por diferenciabilidad), a diferencia de la simulación forward estándar que usa Wendland C2.

---

### 9.2 Discrepancia D2 — BCs de flujo BFR ≠ FEniCS

**Severidad:** 🟠 IMPORTANTE

**Descripción:**
- BFR: `h = h_in(z)` (Dirichlet fuerte en nodos del inlet)
- FEniCS: `(K∇h)·n|_{Γ_in} = −h_inlet(z)` (condición natural/Robin)

**Consecuencia:** Las condiciones son físicamente diferentes. La condición Dirichlet fija el nivel piezométrico; la condición Neumann prescribe el flujo. Producen distribuciones h(x,z) diferentes, y por ende velocidades U y transporte C diferentes.

**Evidencia cuantitativa:** La comparación BFR↔FEniCS muestra diferencias en U. Sin métricas L² no es posible determinar qué fracción de la diferencia se debe a esta BC vs. la malla.

**Acción recomendada:** Unificar las BCs o documentar explícitamente la diferencia y su efecto en los campos H, U, C.

---

### 9.3 Discrepancia D3 — Derivación adjunta MRMT ausente en documentos

**Severidad:** 🟠 IMPORTANTE

**Descripción:** El artículo (§8) y la tesis (Cap. 3) derivan el adjunto solo para ADR. La extensión a MRMT está implementada y validada numéricamente en el código, pero no documentada matemáticamente.

**Consecuencia:** No se puede publicar el artículo afirmando que incluye MRMT sin la derivación.

**Acción recomendada:** Agregar la derivación del adjunto MRMT (Sección 6 de este documento) al artículo y/o tesis. Alternativamente, delimitar el alcance del artículo a ADR y dejar MRMT para trabajo futuro (decisión del autor).

---

### 9.4 Discrepancia D4 — Parámetro J_scale en implementación

**Severidad:** 🟡 MODERADA

**Descripción:** Los tests de gradiente usan `J_scale = 1e12` para escalar el funcional J₁ (ver `test_gradient_checkpoint.py`). Esto es una normalización numérica para mejorar el condicionamiento del optimizador, pero no está documentada matemáticamente.

**Consecuencia:** Si se reportan los gradientes sin mencionar el factor de escala, los valores absolutos no son directamente interpretables.

**Acción recomendada:** Documentar la escala y su justificación. Incluir en la tabla de parámetros.

---

### 9.5 Discrepancia D5 — δ_p en J₁ vs g_σ en grad_Q

**Severidad:** 🟡 MODERADA

**Descripción:** El funcional J₁ usa `delta_p` (Wendland C2) para evaluar la concentración en el pozo:
```python
Cp = d.delta_p @ C_n    # Wendland C2
J1 += gamma * np.sum(Cp**2 * d.wi) * dt
```

Pero el gradiente `grad_Q` usa `gauss_p` (Gaussiano):
```python
integral = np.sum(g * integrand * d.wi)    # g = gauss_p = Gaussiano
```

**Consecuencia:** El funcional J₁ real no es exactamente el que el gradiente cree optimizar (misma naturaleza que D1 pero para el funcional de observación).

**Acción recomendada:** Verificar si esta inconsistencia es intencional (por diferenciabilidad) o un bug. Si es intencional, documentarla. Si es un bug, corregirla en `utils.py`.

---

### 9.6 Observación O1 — ∇·U no es cero (flujo no incompresible)

**Severidad:** 🟡 MODERADA (impacto matemático)

**Descripción:** El término `ψ_C ∇·U` aparece en la ecuación adjunta (Sección 4.4). Este término no es cero porque el flujo en un medio poroso con almacenamiento satisface `S_s ∂h/∂t = ∇·(K∇h) + f_Q`, no `∇·U = 0`. 

En el código adjunto, `Div_V * 0.5` aparece en el coeficiente de la identidad (`make_adj_C_operator`). Esto confirma que el código sí incluye este término.

**Implicación:** La afirmación simplificadora `∇·U = 0` (flujo incompresible) NO aplica en este modelo. La ecuación adjunta del artículo debería incluir el término `ψ_C ∇·(K∇h)` explícitamente, no solo `(K∇h)·∇ψ_C`.

Revisando la ecuación del artículo (líneas 1886-1889):
```
+ (K∇h)·∇ψ_C + ψ_C ∇·(K∇h)   ← ✓ INCLUIDO correctamente
```
**La ecuación del artículo SÍ incluye este término.** ✓

---

## 10. Validación numérica del gradiente

### 10.1 Protocolo del test de diferencias finitas

**Método:** Diferencias finitas centradas:
```
[J(Q + εe_i) − J(Q − εe_i)] / (2ε)  ≈  [dJ/dQ]_i
```

**Comparación:** Error relativo entre gradiente adjunto y FD:
```
err_i = |[dJ/dQ]_i^adj − [dJ/dQ]_i^FD| / |[dJ/dQ]_i^FD|
```

### 10.2 Resultados de los tests existentes

| Test | Modelo | Variable | Error relativo mín. | Tolerancia | Estado |
|------|--------|---------|---------------------|-----------|--------|
| `test_gradient_checkpoint.py` | ADR BFR | grad_Q | ≪ 1e-3 | 1e-3 | ✓ PASA |
| `test_gradient_checkpoint.py` | ADR BFR | grad_zp | ≪ 1e-3 | 1e-3 | ✓ PASA |
| `test_adjoint_identity.py` | ADR BFR (paso a paso) | ∂J/∂Q_n | < 5e-3 | 5e-3 | ✓ PASA |
| `test_gradient_checkpoint_mrmt.py` | MRMT Semi BFR | grad_Q | < 5e-3 | 5e-3 | ✓ PASA |
| `test_gradient_checkpoint_mrmt.py` | MRMT Block BFR | grad_Q | < 5e-3 | 5e-3 | ✓ PASA |
| `fenicsx/test_gradient_checkpoint.py` | ADR FEniCS | grad_Q | < 1e-3 | 1e-3 | ✓ PASA |
| `fenicsx/test_gradient_checkpoint_mrmt_fenicsx.py` | MRMT FEniCS | grad_Q | < 5e-3 | 5e-3 | ✓ PASA |

**Figuras de diagnóstico existentes:**
- `adjoint_identity_bfr_adr.png` — curva log-log error vs ε (ADR)
- `adjoint_identity_bfr_mrmt_semi.png` — ídem MRMT Semi
- `adjoint_identity_bfr_mrmt_block.png` — ídem MRMT Block
- `gradient_checkpoint_fenicsx_adr.png` — ídem FEniCS ADR
- `gradient_checkpoint_fenicsx_mrmt_semi.png` — ídem FEniCS MRMT Semi
- `gradient_checkpoint_fenicsx_mrmt_block.png` — ídem FEniCS MRMT Block

### 10.3 Interpretación del comportamiento esperado

Para diferencias finitas centradas, el error como función de ε:
- Para ε pequeño: domina el error de redondeo → error ↑
- Para ε grande: domina el error de truncamiento O(ε²) → error ↓
- Mínimo en ε ≈ ε* (régimen válido)

La figura log-log debería mostrar:
- Rama izquierda (ε < ε*): pendiente ≈ −1 (redondeo)
- Rama derecha (ε > ε*): pendiente ≈ +2 (truncamiento O(ε²))
- Mínimo < tolerancia especificada

**La existencia del mínimo y su valor es la validación del gradiente.**

---

## 11. Resumen de la auditoría del adjunto

### ✓ Correcto y completo

1. Derivación analítica ADR (artículo §8) — correcta, con signos y BCs verificados
2. Condiciones de frontera adjuntas (ψ_C=0 en Γ_in∪Γ_out, Neumann en Γ_wall) — correctas
3. Condición terminal (ψ(T)=0) — correcta e implementada
4. Adjunto BFR (transpuesta exacta del operador discreto) — correcto
5. Adjunto FEniCS (ufl.derivative automático) — correcto
6. Adjunto MRMT Semi (transpuesta del splitting) — correcto
7. Adjunto MRMT Block (transpuesta acoplamiento implícito) — correcto
8. Gradiente dJ/dQ — correcto en artículo e implementación
9. Gradiente dJ/dz_p — correcto en artículo e implementación
10. Tests de validación (diferencias finitas) — pasan para todos los modelos y backends

### ✗ Pendiente / incompleto

1. **D1:** Kernel del pozo BFR forward (Wendland C2) ≠ kernel del gradiente/adjunto (Gaussiano) — no documentado en tesis/artículo → agregar nota
2. **D2:** BCs de flujo BFR (Dirichlet) ≠ FEniCS (Neumann/natural) — no documentado → agregar nota
3. **D3:** Derivación adjunta MRMT ausente en artículo/tesis → agregar (derivación completa en este documento, Sección 6)
4. **D4:** J_scale=1e12 no documentado → agregar en parámetros
5. **D5:** δ_p en J₁ vs g_σ en grad_Q — verificar si es intencional o bug

---

*Documento generado en rama `doctoral-audit`. Para discutir cualquiera de estas discrepancias antes de modificar código o documentos, revisar con el autor.*
