# Métodos numéricos del simulador: solvers, grados de las funciones y elección del esquema temporal

*Informe del 2026-11-16. Analicé el código de `main` en `codigo/`. La configuración de referencia es la de las
corridas actuales (`experimentos/base.json`, `MC_malla_conos`): ADR, malla `malla_conos` (20 348 nodos), Δt = 5 h,
T = 30 d, θ = θ_C = 1, salida libre, sin W₋. Las cifras de la §4 salen de
`data/output/experimentos/MC_malla_conos/MC0_adr/bfr/simulation_results.npz` (144 pasos, 21 192 nodos con fantasmas).*

---

## 0. Resumen

1. **Los sistemas lineales se resuelven todos con factorización LU directa.**
   - En BFR: SuperLU (`scipy.sparse.linalg.splu`).
   - En FEniCSx: la LU de PETSc dentro de un KSP.
   - No hay solvers iterativos efectivos. En el directo de FEniCSx el KSP es BiCGStab, pero con la LU como
     precondicionador converge en una iteración.
   - La no linealidad del problema de optimización la resuelve `scipy.optimize.minimize(method="trust-constr")` en
     ambos backends.
2. **El esquema temporal es Euler implícito (θ = 1) en todos los subproblemas.**
   - Para el intercambio MRMT hay dos variantes: la semianalítica usa la exponencial exacta con *splitting* de Lie;
     la de bloque usa un factor tipo Crank–Nicolson.
   - Ninguna pieza pasa de orden 1 en Δt.
3. **Estimación a priori: con Δt = 5 h, el error temporal dominaría al espacial.** *Las corridas del 16-nov
   (§9) la **refutan** para el directo ADR: el error temporal estimado con 5 h es ≈ 0.5 % en C y ≈ 0.6 % en la C
   del pozo, unas 6 veces menor que la diferencia BFR–FEM (3 %). Lo que sigue en este punto es el argumento a priori.*
   - Euler implícito añade una difusión numérica de $v^2\Delta t/2$.
   - En la pluma, esa difusión es **1.8 veces la dispersión física** en la mediana y 3.2 veces en el percentil 90.
     Cerca del pozo llega a 50 veces.
   - Además, Δt = 5 h es mayor que el tiempo de intercambio MRMT más rápido (τ₁ = 4.2 h).
   - Δt = 5 h no divide el ciclo de la bomba (20 h encendida / 4 h apagada), así que el apagado de 4 h no se
     resuelve.
4. **Recomendación** (a priori; **decisión del 16-nov: se mantiene Δt = 5 h con Euler implícito**, ver §9):
   - Usar **BDF2** (L-estable y de orden 2) en el flujo y en el transporte, arrancando con un paso de Euler
     implícito.
   - En MRMT semi, usar la exponencial exacta con *splitting* de **Strang**.
   - En MRMT bloque, usar BDF2 con eliminación de C_im.
   - Usar **Δt = 2 h** para producción y 1 h como control (ambos dividen el ciclo de 4 h).
   - Separar la malla de control de Q(t) de la malla temporal.
   - Crank–Nicolson queda descartado: no es L-estable y en BFR ya se observó que diverge.
5. Hallazgos colaterales de eficiencia y consistencia (§7):
   - el adjunto exacto de BFR factoriza dos veces cada matriz del transporte por paso;
   - el directo de BFR recalcula los pesos RBF-FD en cada paso;
   - en FEniCSx, la velocidad es P0 interpolada a P2.

---

## 1. Inventario de métodos

### 1.1 Discretización espacial y grado de las funciones

| Componente | BFR (RBF-FD) | FEniCSx (FEM) |
|---|---|---|
| **Carga h** | Colocación fuerte RBF-FD. Base **multicuádrica** $\phi(r)=\sqrt{1+(\varepsilon r)^2}$ (`phi="mq"`), **aumentada con polinomios de grado 2** (`order=2`: 6 monomios $1,x,z,x^2,xz,z^2$). Estencil de **n = 25** vecinos (la meta de la malla dice 35; las corridas usan 25). Forma de escala $\varepsilon_i = 1/h_i$, recortada a [0.75, 1.25] veces su rango (`eps_from_h`), es decir, el régimen estacionario $\varepsilon h\approx 1$. | Galerkin continuo, **Lagrange P1** (espacio `G`) |
| **Concentración C** | La misma RBF-FD MQ + polinomio de grado 2, n = 25, sobre la misma nube de nodos | Galerkin continuo, **Lagrange P2** (espacio `Q`), sin estabilización (no hay SUPG ni GLS) |
| **Velocidad u = −K∇h** | Gradiente RBF-FD del mismo estencil (grado 2) evaluado en los nodos | `u.interpolate(-K∇h_h)` en **P2 vectorial** (`V`). Como $h_h\in P_1$, $\nabla h_h$ es **constante por elemento** (P0 discontinuo): interpolarla a P2 no aumenta su precisión. |
| **Dispersión D(u)** | Nodal, $D=\alpha_L u^2/|u|+\dots$; $\nabla\cdot D$ con la derivada RBF-FD | Interpolada en P2 vectorial (`V`) |
| **C_im (MRMT)** | Nodal, en los mismos nodos que C | Nodal, en los DOF de P2 |
| **Neumann (entrada, paredes)** | Nodos fantasma a $\approx 1.0\,d_{\text{vecino}}$ fuera de la frontera: fila de colocación de la condición | Natural (forma débil) |
| **Salida libre $\partial_n(u\cdot n)=0$** | Fila sustituida por la ecuación de borde $S_s\partial_t h=\partial_\tau(K_\tau\partial_\tau h)$ con **diferencias de 3 puntos** (interpolación de grado 2; orden 1 en espaciado no uniforme) | Dirichlet con dato $h_\Gamma(t)$ evolucionado por la misma ecuación de 3 puntos (`salida_libre.py`) |
| **Pozo y fuente** | Gaussianas $g_\varepsilon$ evaluadas en los nodos, con la normalización `norm_delta_pozo="area"` | Gaussianas interpoladas en P2. En el adjunto se integra con cuadratura fija de grado 8 (`_QD = 8`); en el directo, con el grado que estima UFL. |
| **Cuadratura de J** | Pesos de Delaunay: 1/3 del área por triángulo, es decir, masa concentrada P1, $O(h^2)$ | Cuadratura de Gauss exacta para el integrando polinómico; la gaussiana se aproxima con grado 8 |
| **Geometría** | Nube de nodos (Poisson-disc, cápsulas y conos) | Triángulos afines (**P1 geométrico**) de la misma `malla_conos` |
| **Opcional** | `galerkin_banda`: filas Galerkin **P1** con masa concentrada en una franja (apagado; el usuario lo descartó) | — |

### 1.2 Discretización temporal

| Subproblema | BFR | FEniCSx | Orden en Δt |
|---|---|---|---|
| Flujo $S_s\partial_t h-\nabla\cdot K\nabla h=-Q\delta$ | Esquema θ con `theta = 1.0` (Euler implícito). Matriz `A_left` constante: **una LU para toda la corrida** | Esquema θ con `theta = 1.0`. `A1` constante: se ensambla y factoriza una vez | 1 |
| Transporte ADR | Esquema θ con `theta_C = 1.0`. Coeficientes en $t^{n+1}$ (izquierda) y $t^n$ (derecha). Almacenamiento $-S_s\partial_t h\,C$ con $\partial_t h=(H^{n+1}-H^n)/\Delta t$ | Igual. Forma débil con pesos `theta_C` / `1 − theta_C` | 1 |
| Acoplamiento flujo → transporte | Secuencial: $H^{n+1}\to u^{n+1}\to C^{n+1}$. El flujo no depende de C, así que **no hay error de partición** | Igual | — |
| MRMT semianalítico | Transporte, y después intercambio exacto $C_{im}^{n+1}=C+(C_{im}^n-C)e^{-\lambda_r\Delta t}$ región por región: **splitting de Lie** | Igual (`_mrmt_semi_step`) | 1 (por el splitting) |
| MRMT bloque | $C_{im}^{n+1}=2\kappa_r[\theta_C C^{n+1}+(1-\theta_C)C^n]+(1-2\kappa_r)C_{im}^n$, con $\kappa_r=\lambda_r\Delta t/(2+\lambda_r\Delta t)$. El factor $1-2\kappa_r=(2-\lambda_r\Delta t)/(2+\lambda_r\Delta t)$ **es el de Crank–Nicolson**. C_im se elimina (complemento de Schur) en la diagonal del transporte. | Igual (`_mrmt_block_step`) | 1 con θ_C = 1 (C en $t^{n+1}$, C_im explícito) |
| Ecuación de borde de la salida | θ del flujo | θ del flujo (`BordeLibre`, con su propia `splu`) | 1 |
| Adjunto | Transpuesto exacto del esquema programado, hacia atrás (`funtions/adjunto_exacto.py`) | Transpuesto de las formas ensambladas (`fenicsx/adjoint.py`) | Hereda el del directo |

### 1.3 Solvers de los sistemas

| Sistema | BFR | FEniCSx |
|---|---|---|
| Flujo directo $A_L H^{n+1}=\dots$ | `splu(A_left)` una vez (`preprocess.py:237`), reutilizada en cada paso | `KSP(BCGS)` + `PC(LU)` de PETSc, factorizado una vez (`forward.py:199`). En `adjoint.py`: `PREONLY` + `LU` |
| Transporte directo | `splu(C_solver)` **en cada paso**: la matriz cambia con $u^{n+1}$, $D^{n+1}$ y $\partial_t h$ | `A2` reensamblada y **refactorizada en cada paso** (igual) |
| Adjunto del flujo | `A_left.solve(rhs, trans='T')`: reutiliza la LU del directo | `PREONLY` + LU de $A_{hh}^T$, una vez |
| Adjunto del transporte | `splu(A.T)` en cada paso y **otra** `splu(A)` para reconstruir $C_{sol}$ sin recorte (`adjunto_exacto.py:274,285`) | LU de $A_{CC}^T$ en cada paso |
| Salida libre (1D) | Fila incluida en `A_left` | `splu(M − θL)` una vez |
| Masa (campos duales → nodales) | — | LU de $M_Q$ (`_campo_desde_dual`) |
| Optimización $\min \hat J(Q,z_p)$ | `trust-constr` (punto interior / SQP con región de confianza), con escalado por el gradiente inicial, `Bounds` y `LinearConstraint` del tanque y del ciclo de bomba (`view/animation.py:619`) | `trust-constr` con medida de criticidad χ (proyección resuelta con `SLSQP`) y parada χ < tol (`fenicsx/adjoint.py:749`) |

**Ordenamientos.** SuperLU usa por defecto `permc_spec="COLAMD"`, que es el mejor ordenamiento según el banco de
pozos ([caso-pozos-transitorio]: MMD_AT_PLUS_A fue catastrófico). PETSc usa en serie su LU nativa con el ordenamiento
por defecto (`nd`).

**¿Por qué LU y no iterativos?** En 2D, con unos 2·10⁴ incógnitas en BFR y unas 8·10⁴ en P2, la LU es la opción
correcta:
- el relleno de RBF-FD es unas 4 veces el de FEM, pero cabe en memoria;
- las matrices de RBF-FD son no simétricas, no normales y con filas de condición de frontera, que es justo donde
  ILU/GMRES suele fallar;
- la exactitud del adjunto discreto (test de Taylor) agradece residuos a precisión de máquina, no a `rtol`.

Los métodos iterativos (GMRES con ILU o AMG, BiCGStab) sólo tendrían sentido en 3D.

---

## 2. Convergencia espacial: qué orden cabe esperar

### 2.1 RBF-FD (BFR)

- **Teoría** (Flyer, Fornberg, Bayona, Barnett 2016; Bayona, Flyer, Fornberg, Barnett 2017). Con RBF y polinomios
  de grado *m* en el estencil, la derivada de orden *k* tiene un error de consistencia $O(h^{m+1-k})$, siempre que el
  estencil sea suficientemente grande ($n\gtrsim 2\binom{m+2}{2}=12$ para *m* = 2; n = 25 cumple con margen).
  En régimen estacionario ($\varepsilon h$ fijo), la MQ por sí sola **no** converge: el orden lo marca el polinomio.
  Con *m* = 2:
  - **gradiente** ($u=-K\nabla h$, advección): $O(h^2)$;
  - **segundas derivadas** (difusión de h, dispersión de C): $O(h)$. Sólo en estenciles casi simétricos
    hay cancelación a $O(h^2)$, cosa que no ocurre junto a fronteras ni en transiciones de espaciado.
- **Condiciones de frontera** con fantasmas: mismo orden que el interior. La ecuación de borde libre es de 3 puntos:
  orden 1 si el espaciado no es uniforme.
- **Estabilidad** (lo que limita en la práctica):
  - la colocación fuerte produce operadores no normales, con autovalores de parte real positiva en mallas
    desfavorables ([estabilidad-malla-bfr-minsp]);
  - no es W-simétrica: $\|WA-(WA)^T\|/\|WA\|=0.54$ ([adjunto-bfr-ruido-superficie]);
  - consecuencia: el **directo** converge (C 5.4 % → 3.0 % frente a FEM al pasar de 15 626 a 20 348 nodos), pero el
    **adjunto** de BFR **no** converge con la malla (ψ_C cambia 25–36 %).
- **Péclet de malla** $Pe=h\,|u|/D\approx h/\alpha_L$ (con $D\approx\alpha_L|u|$): mediana 1.9, p99 7.0. Junto al
  pozo $Pe\le 1$ ([condicionamiento/README]). El esquema centrado es admisible en la pluma, al límite en la zona
  gruesa ($h\sim 60$ m, $Pe\sim 6$).
- **Conclusión:** el orden espacial efectivo de BFR es **1 a 2**, dominado por el error $O(h)$ de la dispersión.

### 2.2 FEM (FEniCSx)

- **Teoría** (Ern & Guermond 2004; Brenner & Scott 2008). Con solución regular:
  - P1 para h: $\|h-h_h\|_{L^2}=O(h^2)$, $\|h-h_h\|_{H^1}=O(h)$;
  - P2 para C: $O(h^3)$ y $O(h^2)$.
- **Lo que de verdad limita** es la velocidad: $u_h=-K\nabla h_h\in P_0$ tiene error $O(h)$ en $L^2$ (lema de Strang,
  perturbación del coeficiente advectivo). Así, el error de C queda limitado a **$O(h)$ por la velocidad**, aunque C
  esté en P2. Interpolar el P0 discontinuo en los DOF de P2 (vértices y aristas compartidos) toma el valor de una de
  las celdas vecinas: no es una proyección.
- **Regularidad reducida** por:
  - $K(\phi)=8.3\cdot10^{-3}\,g\,d^2\phi^3/(\nu(1-\phi)^2)$ heterogénea;
  - las esquinas Dirichlet–Neumann (singularidad $r^{1/2}$, orden 1 observado en Graetz [caso-base-manzoni-graetz]);
  - la gaussiana estrecha del pozo.
  En la práctica se observa **orden 1 a 2**, no 3.
- **Galerkin P2 sin estabilizar:** el Péclet por elemento $h/(2\alpha_L)$ es < 1 en la mediana y llega a 3.5. Hay
  riesgo de oscilaciones moderadas en la zona gruesa.
- **Conclusión:** el orden espacial efectivo de FEniCSx también es **1 a 2**. Converge mejor que BFR en el adjunto
  (ψ_C 3–10 % entre mallas) porque Galerkin es W-simétrico.

### 2.3 Consecuencia para el tiempo

Ambos backends tienen un error espacial **de orden 1 a 2**, con $h\approx 4$–18 m en la pluma (p10–p50). El esquema
temporal debe ser al menos de **orden 2**. Con orden 1, el Δt necesario para igualar el error espacial es muy
pequeño (§4). Ir más allá (orden 3 o superior: BDF3, SDIRK3, ESDIRK) no compensa, porque el espacio no lo
aprovecharía.

---

## 3. Propiedades que debe tener el esquema temporal

1. **Estabilidad.**
   - El problema es rígido: la difusión de h con $S_s=\phi$ y la dispersión en $h\sim 4$ m dan números de difusión
     $D\Delta t/h^2$ de hasta 64 (/φ ⇒ ~640).
   - Hace falta un método **A-estable** y, por los modos rígidos sin amortiguar, **L-estable**.
   - RBF-FD añade autovalores cerca del eje imaginario o con parte real ligeramente positiva. Crank–Nicolson
     ($R(\infty)=-1$) no los amortigua y **ya divergió en BFR** (1e56–1e123, [oscilacion-pozo-bfr-inestabilidad]).
2. **Positividad.**
   - Teorema de Bolley–Crouzeix (1978): un método lineal incondicionalmente positivo tiene orden ≤ 1, y el único
     práctico es Euler implícito.
   - Cualquier esquema de orden 2 es positivo sólo bajo una restricción de Δt.
   - Consecuencia: se mantiene el recorte `clip_C` (que el adjunto exacto ya trata con la máscara) y se vigila el
     número de nodos recortados.
3. **Coste.**
   - El transporte refactoriza su matriz en cada paso. Un método con **una sola resolución lineal por paso** (BDF2)
     cuesta lo mismo que Euler implícito.
   - Los métodos RK implícitos multietapa (Radau IIA, SDIRK) multiplican las LU por el número de etapas.
4. **Adjunto discreto.** El esquema debe tener un transpuesto sencillo: multipaso lineal o un DIRK de pocas etapas.
   Ambos cumplen.
5. **Forzamiento discontinuo.** Q(t) es constante a trozos, con conmutaciones de la bomba. El orden 2 sólo se
   mantiene si **las conmutaciones caen en instantes de la malla temporal**.

---

## 4. Escalas temporales del problema (medidas en la corrida MC0_adr)

Con $v=|u|/\phi$ (φ = 0.1) y el máximo temporal de cada nodo:

| Δt | Courant $v\Delta t/h$ en la pluma, p50 / p90 | **Difusión numérica de Euler implícito / física**, $v\Delta t/(2\alpha_L)$, pluma p50 / p90 | Máximo en el pozo |
|---|---|---|---|
| 0.5 h | 0.23 / 1.1 | 0.18 / 0.32 | 5 |
| 1 h | 0.47 / 2.2 | 0.36 / 0.64 | 10 |
| 2 h | 0.93 / 4.3 | 0.72 / 1.28 | 20 |
| 4 h | 1.9 / 8.6 | 1.44 / 2.56 | 41 |
| **5 h (actual)** | **2.3 / 10.7** | **1.80 / 3.20** | **51** |
| 10 h (default.json) | 4.7 / 21.5 | 3.59 / 6.40 | 102 |

De dónde sale esa difusión: la ecuación modificada de Euler implícito para $c_t+v c_x=D c_{xx}$ es
$c_t+vc_x=(D+\tfrac12 v^2\Delta t)c_{xx}+O(\Delta t^2)$. El cociente de la difusión espuria sobre la física
$D\approx\alpha_L v$ es $v\Delta t/(2\alpha_L)$.

**Con el Δt actual, la difusión numérica es mayor que la dispersión física en la mayor parte de la pluma.** Eso
explica buena parte de que FEM y RBF-FD «disipen distinto» ([bfr-mesh-fenicsx]: diferencias del 10–34 % crecientes
con t). Para que la difusión espuria de Euler implícito fuera ≤ 10 % de la física, haría falta $\Delta t\lesssim 0.3$ h
(unos 2 400 pasos), lo que no es viable con orden 1.

**Intercambio MRMT.** Con $\lambda_r=D_{\text{eff},r}/(R\beta_rL^2)$:

| región | λ_r [s⁻¹] | τ_r = 1/λ_r | λΔt con 2 h | λΔt con 5 h | λΔt con 10 h | factor CN (bloque) $1-2\kappa$ con 10 h | exacto $e^{-\lambda\Delta t}$ |
|---|---|---|---|---|---|---|---|
| 1 | 6.7e-5 | **4.2 h** | 0.48 | 1.20 | 2.40 | **−0.091** | 0.091 |
| 2 | 5.0e-5 | 5.6 h | 0.36 | 0.90 | 1.80 | 0.053 | 0.165 |
| 3 | 2.0e-5 | 13.9 h | 0.14 | 0.36 | 0.72 | 0.471 | 0.487 |

- Con Δt ≥ 2τ₁ (10 h, el valor de `default.json`), el MRMT de bloque **cambia de signo** C_im en cada paso: la no
  monotonía típica de CN.
- Con 5 h el factor es 0.25, frente al 0.30 exacto: un error del 17 % por paso en la región 1.
- El MRMT semi integra el intercambio de forma exacta, pero su *splitting* de Lie de orden 1 con $\lambda\Delta t\sim 1$
  está fuera del régimen asintótico.

**Flujo.**
- El cambio máximo de H por paso decae como $\sim 1/t$: 0.089 m en el paso 1, 0.0086 m en el 10 y 0.0009 m en el 143.
- Es un transitorio difusivo fuerte al principio, que pide pasos más finos al arrancar.
- La matriz es constante, así que el coste de un orden mayor es casi nulo (una LU reutilizada).

**Bomba.**
- El ciclo es de 20 h encendida y 4 h apagada, y `pump_upper_bounds` evalúa la disponibilidad **en el punto medio del
  paso**.
- Con Δt = 5 h o 10 h el apagado de 4 h se pierde o se desplaza (*aliasing*).
- Los Δt compatibles son los divisores de 4 h: **4, 2, 1 y 0.5 h**.

---

## 5. Comparación de esquemas candidatos

| Esquema | Orden | A / L-estable | Resoluciones por paso | Positivo | Adjunto | Veredicto |
|---|---|---|---|---|---|---|
| Euler implícito (actual) | 1 | sí / sí | 1 | **sí** (incondicional) | trivial | Robusto, pero el error temporal domina (§4) |
| Crank–Nicolson (θ = ½) | 2 | sí / **no** ($R(\infty)=-1$) | 1 | no | trivial | **Descartado**: no amortigua modos rígidos; ya divergió en BFR; oscila con λΔt > 2 en MRMT |
| θ-método con θ = ½ + O(Δt) (CN desplazado) | ≈ 1–2 | sí / casi | 1 | no | trivial | Parche; pierde el orden 2 |
| **BDF2** | **2** | **sí / sí** | **1** (la misma estructura de matriz) | sólo con restricción de Δt | multipaso: el transpuesto se acopla con $n+1$ y $n+2$ | **Recomendado** |
| TR-BDF2 (γ = 2 − √2) | 2 | sí / sí | 2, con **la misma** matriz en ambas etapas si los coeficientes no cambian | no | DIRK de 2 etapas | Buena alternativa de un paso: reinicia sin problemas en las conmutaciones de Q. En el transporte (matriz variable) cuesta 2 LU por paso. |
| SDIRK2 / ESDIRK3 | 2–3 | sí / sí | 2–4 | no | más complejo | Coste ×2–4; el espacio no aprovecha el orden 3 |
| Radau IIA (orden 3–5) | ≥ 3 | sí / sí | sistema acoplado de *s* etapas | no | complejo | Excesivo |
| Exponencial (intercambio MRMT) | exacto | — | 0 | sí | trivial | **Recomendado para el intercambio**, con Strang |

**Por qué BDF2 frente a TR-BDF2.**
- En el **transporte**, la matriz cambia en cada paso (depende de $u^{n+1}$). BDF2 hace **una** LU por paso, el mismo
  coste que ahora; TR-BDF2 necesitaría dos.
- En el **flujo** (matriz constante) los dos cuestan lo mismo.
- BDF2 tiene la contra de su arranque y de los reinicios en las discontinuidades. Se resuelven:
  1. arrancando con un paso de Euler implícito (o dos medios pasos);
  2. alineando las conmutaciones de Q con la malla temporal. Q constante a trozos no rompe la consistencia de BDF2
     si el salto cae en un nodo temporal. Lo que se pierde localmente es un orden en ese paso, no la estabilidad.
- El paso variable de BDF2 (para graduar al inicio) tiene fórmula cerrada y es cero-estable con razones
  $\omega=\Delta t_{n+1}/\Delta t_n<1+\sqrt2$.

---

## 6. Recomendación concreta

### 6.1 Esquemas

**Flujo (los dos backends).** BDF2 con paso uniforme:
$$\Big(\tfrac{3}{2}\tfrac{S_s}{\Delta t}-\mathcal L_h\Big)H^{n+1}=\tfrac{S_s}{\Delta t}\big(2H^n-\tfrac12H^{n-1}\big)-Q^{n+1}\delta_\varepsilon+b_{bc}^{n+1}.$$
- La matriz es constante: **una LU** en toda la corrida, como ahora.
- El primer paso se hace con Euler implícito.
- La ecuación de borde de la salida libre pasa a BDF2 con la misma fórmula.
- Para el transitorio inicial ($\sim 1/t$), hay dos opciones:
  - graduar los primeros 6–8 pasos geométricamente desde $\Delta t_0\approx 15$ min hasta Δt, con BDF2 de paso
    variable y una LU por cada Δt distinto (caben en memoria);
  - o mantener Δt uniforme y comprobar en el estudio de convergencia (§6.4) que el arranque no domina.

**Transporte ADR.** BDF2 con los coeficientes en $t^{n+1}$:
$$\Big(\tfrac{3}{2}\tfrac{\phi R}{\Delta t}-\mathcal L_C(u^{n+1},D^{n+1})+\phi R\lambda-S_s\partial_t h^{n+1}\Big)C^{n+1}
=\tfrac{\phi R}{\Delta t}\big(2C^n-\tfrac12C^{n-1}\big)+f^{n+1},$$
- donde $\partial_t h^{n+1}=(3H^{n+1}-4H^n+H^{n-1})/(2\Delta t)$, para no degradar el orden;
- **una LU por paso**, como ahora;
- ya no hace falta la matriz derecha `C_right`: el lado derecho sólo lleva masa, así que se ahorra un ensamblaje
  por paso.

**MRMT semianalítico.** Strang:

    medio paso de intercambio exacto (Δt/2) → transporte BDF2 (Δt) → medio paso de intercambio exacto

Es de orden 2 y conserva la exactitud del intercambio para cualquier λΔt. El coste es el mismo, porque el
intercambio es explícito y nodal.

**MRMT bloque.** BDF2 en el sistema acoplado y eliminación de C_im:
$$C_{im,r}^{n+1}=\frac{2C_{im,r}^n-\tfrac12C_{im,r}^{n-1}+\lambda_r\Delta t\,C^{n+1}}{\tfrac32+\lambda_r\Delta t}.$$
- Es L-estable, sin la restricción λΔt < 2 de la variante tipo CN actual.
- El complemento de Schur sigue siendo diagonal: sólo cambia el coeficiente $\alpha_r$ de `config/parameters.py`.

### 6.2 Paso de tiempo

> **Decisión (2026-11-16):** se sigue con **Δt = 5 h** (144 pasos, `base.json` sin cambios). Las corridas de la §9
> muestran que en ADR el error temporal (≈ 0.5 % en C) es unas 6 veces menor que la diferencia espacial BFR–FEM.
> La tabla siguiente queda como referencia si el estudio en MRMT o en el óptimo indica lo contrario.

| Uso | Δt | Pasos (T = 30 d) | Coste aproximado frente a hoy (144 pasos) |
|---|---|---|---|
| **Producción** | **2 h** | 360 | ×2.5 (≈ ×1.6 si se aplican las mejoras de la §7) |
| Control / referencia | 1 h | 720 | ×5 |
| Exploración rápida | 4 h | 180 | ×1.25 (sólo BDF2; resuelve el ciclo de la bomba) |

Por qué 2 h:
- divide el ciclo de 4 h;
- con BDF2, el error dominante es dispersivo, $O((kv\Delta t)^2)$, y en la pluma $v\Delta t/h\approx 0.9$ (p50);
- deja λ₁Δt = 0.48 en MRMT, dentro del régimen asintótico;
- con Euler implícito, el mismo error requeriría $\Delta t\lesssim 0.3$ h.

### 6.3 Malla de control frente a malla temporal

Hoy hay un control $Q_n$ por paso (`Nt = T/dt` controles). Al bajar Δt de 5 h a 2 h, el número de variables pasaría
de 144 a 360, y el óptimo dependería de Δt. Recomiendo:
- definir Q constante a trozos en **intervalos de control fijos**, por ejemplo de 4 h, alineados con la bomba
  (180 variables), con $Q(t^n)$ por inyección;
- en el gradiente, $\partial\hat J/\partial Q_k=\sum_{n\in I_k}\partial\hat J/\partial Q^n$, una suma de los
  gradientes por paso.

Así el problema de optimización no cambia al refinar Δt, y el estudio de convergencia temporal del **óptimo** tiene
sentido. Es lo que ya se hizo con éxito en el banco de pozos transitorio (una LU por intervalo de control).

### 6.4 Verificación (pasos 6–7 de [metodologia-benchmark-ocp])

1. Fijar la malla `malla_conos` y correr el directo ADR con Δt = 4, 2, 1 y 0.5 h, con Euler implícito y con BDF2,
   en ambos backends.
2. Medir:
   - J y $J_1$;
   - la curva BTC en el pozo;
   - ‖C(T)‖ en L²;
   - el pico de C en el pozo;
   - ∇Ĵ con el control fijo.
3. Calcular el **orden observado** $p=\log_2(e_{2\Delta t}/e_{\Delta t})$ y la referencia por Richardson.
   - Esperado: p ≈ 1 con Euler implícito y p ≈ 2 con BDF2.
   - Si BDF2 da p < 2, buscar la causa en el arranque, en las conmutaciones de Q o en el recorte.
4. Rehacer el test de Taylor del adjunto BDF2 (orden 2 en ε) en los 6 casos (2 backends × 3 modelos).
5. Comprobar que el error temporal con 2 h queda por debajo de la diferencia entre mallas (C 3.0 % entre
   15 626 y 20 348 nodos). Si es así, el esquema está equilibrado.
6. **No tocar `default.json`.** Los esquemas nuevos van como parámetro opt-in (por ejemplo `"esquema_t": "bdf2"`) y el
   Δt en una config aparte (regla 6 del CLAUDE.md).

---

## 7. Hallazgos colaterales (eficiencia y consistencia)

> **Corregidos el 2026-11-16 (puntos 1, 2 y 6).** `tests/test_adjunto_exacto.py` pasa en ADR, MRMT bloque y
> MRMT semi. Las matrices son idénticas a la construcción original (1e-15) y el gradiente coincide con la línea
> base en todos los dígitos. El directo de 4 pasos en el cuadrado bajó de ~19 s a ~10 s.
> - Punto 1: una sola `splu(A)` por paso, con `solve(trans='T')`.
> - Punto 2: `EstencilesTransporte` en caché en `funtions/operators.py`. La construcción original se conserva como
>   `build_transport_matrix_directo`, sólo para la prueba.
> - Punto 6: comentarios corregidos.

1. **LU duplicada en el adjunto exacto de BFR** (`funtions/adjunto_exacto.py:274` y `:285`). Cada paso hace
   `splu(A.T)` y además `splu(A)` para reconstruir $C_{sol}$. Con una sola `lu = splu(A.tocsc())` bastan
   `lu.solve(ct, trans='T')` y `lu.solve(b)`. Es lo que ya se hace en el flujo con `A_left.solve(..., trans='T')`.
   Ahorra la mitad de las factorizaciones del gradiente.
2. **Pesos RBF-FD recalculados en cada paso del directo.** `build_transport_matrix` llama a `weight_matrix` (que
   resuelve un sistema local por nodo) dos veces por paso, aunque los estenciles $O_k$ son fijos y sólo cambian los
   coeficientes. `adjunto_exacto.py` ya los precalcula (`self.O`) y ensambla $\sum_k\mathrm{diag}(c_k)O_k$. Llevar
   ese patrón a `step_time` haría el ensamblaje casi gratis.
3. **FEniCSx, directo:** `KSP.Type.BCGS` con `PC.LU` hace al menos una iteración de BiCGStab sobre una LU exacta,
   con `rtol` por defecto. `PREONLY` + LU (lo que usa `adjoint.py`), o MUMPS, es más limpio y hace que el directo de
   `forward.py` y el de `adjoint.py` sean idénticos.
4. **FEniCSx, cuadratura:** `forward.py` deja que UFL estime el grado, mientras que `adjoint.py` lo fija en 8. Sus
   directos pueden diferir ligeramente en J. Conviene usar `_QD` en ambos.
5. **FEniCSx, velocidad:** `u.interpolate(-K∇h_h)` con $h_h\in P_1$ da un campo P0 evaluado en DOF de P2 compartidos.
   Hay dos opciones coherentes:
   - (a) $h\in P_2$, con lo que $u$ queda en P1 discontinuo y es $O(h^2)$: sube el orden espacial efectivo de C, que
     hoy está limitado por $u$;
   - (b) C en P1, más barato y de orden consistente con $u$.

   Recomiendo (a) como opción, porque J y $\partial\hat J/\partial z_p$ dependen de $\nabla h$ cerca del pozo.
6. **Comentario engañoso** en `funtions/operators.py:817` (`make_adj_H_operator`): «La ecuación de flujo es
   simétrica → A_left^T = A_left». Con colocación RBF-FD **no** lo es (asimetría relativa 0.54). El gradiente
   actual no se ve afectado, porque usa `adjunto_exacto.py`, que transpone de verdad. Pero `step_adjoint` (usado para
   animar ψ) resuelve con la matriz sin transponer y con otras condiciones de frontera.
7. **`default.json` usa Δt = 10 h:** con ese paso, el MRMT de bloque no es monótono (tabla de la §4). Es otro motivo
   para que las corridas de producción usen una config aparte, sin tocar el benchmark.

---

## 8. Referencias

- Bayona, V., Flyer, N., Fornberg, B., Barnett, G. A. (2017). On the role of polynomials in RBF-FD approximations II.
  *J. Comput. Phys.* 332, 257–273.
- Flyer, N., Fornberg, B., Bayona, V., Barnett, G. A. (2016). On the role of polynomials in RBF-FD approximations I.
  *J. Comput. Phys.* 321, 21–38.
- Fornberg, B., Flyer, N. (2015). *A Primer on Radial Basis Functions with Applications to the Geosciences.* SIAM.
- Ern, A., Guermond, J.-L. (2004). *Theory and Practice of Finite Elements.* Springer.
- Brenner, S., Scott, R. (2008). *The Mathematical Theory of Finite Element Methods*, 3.ª ed. Springer.
- Hairer, E., Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and DAE Problems.* Springer
  (A-/L-estabilidad, BDF, Radau, SDIRK).
- Bank, R. E. et al. (1985). Transient simulation of silicon devices and circuits. *IEEE Trans. CAD* 4 (TR-BDF2).
- Bolley, C., Crouzeix, M. (1978). Conservation de la positivité lors de la discrétisation des problèmes d'évolution
  paraboliques. *RAIRO Anal. Numér.* 12, 237–245.
- Hundsdorfer, W., Verwer, J. (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations.*
  Springer (splitting de Strang, positividad, ecuación modificada).
- Manzoni, A., Quarteroni, A., Salsa, S. (2021). *Optimal Control of PDEs*, §7.4–7.5 (DtO y esquemas temporales del
  adjunto).

Notas internas citadas: [estabilidad-malla-bfr-minsp], [oscilacion-pozo-bfr-inestabilidad],
[adjunto-bfr-ruido-superficie], [bfr-mesh-fenicsx], [caso-pozos-transitorio], [caso-base-manzoni-graetz],
[metodologia-benchmark-ocp], `experimentos/condicionamiento/README.md`.

---

## 9. Corridas en el dominio real (2026-11-16, grupo `MT_paso_tiempo`)

Las corridas son de ADR en `malla_conos`, con la configuración de `MC0_adr`, Euler implícito y T = 30 d. Las
diferencias relativas se miden en L² con pesos de área nodal y la C del pozo es la media gaussiana
(ε = r_pozo/3), con los mismos criterios que `experimentos/comparar.py`.

**Regresión (código nuevo frente a `MC0_adr`, Δt = 5 h).**
- BFR da el mismo resultado: C cambia ≤ 1e-13 y H y U son idénticos. El tiempo total baja de 304 s a 208 s
  (−32 %) gracias a los estenciles en caché.
- FEniCSx, con LU directa y cuadratura fija de grado 8: C cambia ≤ 2e-10 y U ≤ 9e-10. Con esta malla la cuadratura
  automática ya era suficiente.

**Δt = 2 h frente a 5 h** (360 frente a 144 pasos):

| magnitud | BFR | FEniCSx |
|---|---|---|
| C, L² relativa (t = 100 / 360 / 720 h) | 3.8e-3 / 3.0e-3 / 2.6e-3 | 3.9e-3 / 3.1e-3 / 2.8e-3 |
| \|U\|, L² relativa (final) | 1.1e-3 | 1.1e-3 |
| ΔH rms (final) | 1e-4 m | 1e-4 m |
| C del pozo, máximo en el tiempo | 2.6e-3 | 2.8e-3 |
| $\int C_{pozo}^2\,dt$ (lo que entra en $J_1$) | +3.8e-3 | +4.2e-3 |
| máx. de C (junto a la fuente), máximo en el tiempo | 2.2e-2 | 2.2e-2 |
| BFR–FEM: C L² final | 2.99 % → 3.01 % | |

- Error de Euler implícito con 5 h, por Richardson con orden 1: $e_{5h}\approx\tfrac53\,\Delta$, es decir ≈ 0.5 % en
  C, ≈ 0.6 % en la C del pozo y ≈ 3.6 % en el pico de C junto a la fuente.
- La diferencia BFR–FEM no cambia, así que **es espacial, no temporal**.
- La estimación de la §4 ($v\Delta t/2\alpha_L>1$) sobreestimó el efecto. La difusión espuria actúa sobre la
  curvatura de C, y en estos 30 días el campo es casi todo la condición inicial, suave, que se transporta poco. Sólo
  el pico junto a la fuente nota el Δt de forma apreciable.

**FEniCSx con h en P2** (`grado_h = 2`, Δt = 2 h), frente a P1:
- La discrepancia BFR–FEM en \|U\| baja de **12.5 % a 8.8 %**: la velocidad era la mayor fuente de diferencia entre
  backends y P2 la reduce, como predice la §2.2.
- La C de FEM en el pozo sube un 2 % ($\int C_{pozo}^2$ +3.7 %). La razón BFR/FEM de la C del pozo pasa de 0.951 a
  0.933, y la diferencia en C (L²) de 3.0 % a 2.9 %.
- ΔH entre P1 y P2 es ≈ 1.4e-3 m.
- Coste: 462 s frente a 555 s. P2 no lo empeora; la variación es del orden del ruido de la máquina.

**Consecuencias para la §6.**
- Para el directo **ADR**, Euler implícito con 5 h ya tiene un error temporal pequeño frente al espacial. BDF2 no es
  urgente por exactitud en ADR.
- Los motivos que siguen en pie para bajar Δt o usar BDF2 son tres:
  1. el ciclo de la bomba (Δt divisor de 4 h);
  2. el intercambio MRMT (τ₁ = 4.2 h), **no medido aquí**: falta repetir el estudio con `mrmt_semi` y `mrmt_block`;
  3. la convergencia del óptimo y del gradiente, también **no medida**.
- **Decisión:** se mantiene Δt = 5 h; los puntos 2 y 3 se revisarán con Δt = 5 h como referencia.
- El siguiente paso con más rendimiento es **espacial**: \|U\| en FEM (P2 ya ayuda) y la no convergencia del adjunto
  de BFR en malla.

## 10. Adjunto de FEniCSx con h en P2 (2026-11-16, grupo `MH_adjunto_p2`)

`fenicsx/adjoint.py` acepta ahora `grado_h = 2`, igual que el directo:
- h y ψ_h viven en Lagrange P2 (`ctx.Gh`);
- φ, K_z y las salidas siguen en P1 (`ctx.G`, los 20 348 nodos de la malla compartida);
- al guardar, h y ψ_h se interpolan a P1 (`_en_G`).

**Verificación del gradiente** (`tests.test_gradiente_modelo_nuevo fenicsx <modelo> --modulada --grado-h=2`, C = 1 modulada,
γ escalado para que domine J_m):

| Modelo | Error relativo adjunto frente a diferencias finitas |
|---|---|
| adr | 5e-9 – 7e-7 |
| mrmt_semi | 3e-9 – 9e-6 |
| mrmt_block | 4e-9 – 1e-6 |

Son los mismos niveles que con P1. En ADR, la derivada direccional según −∇J pasa de −2.987 (P1) a −3.098 (P2), un 3.7 %.
El test de Taylor (`--grado-h=2`) también pasa, pero no es informativo: en ese dominio la pluma no llega al pozo, J₁ = 0
y J lo domina el término en z_p.

**Dominio real** (MC0_adr, Δt = 5 h, `run_type=comparison`), P2 frente a P1:

| Campo | Diferencia relativa L² |
|---|---|
| h | 2.5e-6 |
| C | 3.1e-3 |
| u (en los nodos) | 17 % |
| ψ_h | 1.4 % |
| ψ_C | 4.1 % (máximo −9 %) |

El cambio en ψ_C (4 %) se acerca al de la convergencia en malla del adjunto FEniCSx (3–10 %,
[adjunto-bfr-ruido-superficie]). La velocidad, que en P1 es P0, es el factor de mayor peso en el adjunto del transporte.

Coste: 964 s con P2 frente a 714 s con P1 (×1.35).
