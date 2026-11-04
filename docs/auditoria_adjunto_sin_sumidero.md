# Auditoría matemática de `adjunto_sin_sumidero` (revisión del 4 de noviembre de 2026)

Documento auditado: `docs/adjunto_sin_sumidero.tex/.pdf`, versión del commit `6b1b056` (14 págs.).
Versión corregida: la misma ruta, 20 págs. Esta auditoría separa lo que **demuestra el análisis** de lo que **demuestra el código** y de lo que **no se ha demostrado**.

## 0. Alcance y método

- Se leyó el `.tex` completo (800 líneas) y se rehízo a mano: el Lagrangiano y sus variaciones (ADR y MRMT), las integraciones por partes temporales y espaciales, las condiciones naturales, la forma no conservativa del adjunto, el gradiente en `Q` y `z_p`, las tres recursiones discretas (ADR, bloque, semi-analítica) y la fuente del adjunto del flujo.
- Se contrastó con el código: `funtions/operators.py` (`make_H_bc`, `ecuacion_borde_libre`, `coeficientes_transporte`), `funtions/adjunto_exacto.py`, `fenicsx/forward.py`, `fenicsx/adjoint.py`, `config/parameters.py`, `funtions/step_time.py`.
- **Ejecutado hoy (4 nov 2026):** `tests/banco1d/alg.py` (V6 álgebra), `tests/banco1d/runp1.py` (V3, gradiente en `Q`), `tests/banco1d/runp1_extra.py` (nuevo: gradiente en `x_p`, forma conservativa vs no conservativa y barrido del paso de diferencias finitas), `python -m tests.test_adjunto_exacto` en `bfr_env` con ADR, MRMT bloque y semi (columnas «Q aleatoria» y «(Q,z_p)»).
- **No ejecutado hoy:** V1, V2 (FEniCSx), la columna «−∇J / Q uniforme» de V4, V4 con θ=0.9, V5 (dominio real). Se citan tal como estaban.
- No se inventaron referencias; se conservan las 4 existentes (Tröltzsch, Hinze et al., Manzoni et al., Haggerty–Gorelick).

## 1. Hallazgos

Clases: **EM** error matemático · **EF** error físico · **EV** error variacional · **EN** error numérico · **OM** omisión · **ME** mejora editorial.
«Original» = versión `6b1b056`; «Nueva» = versión corregida.

### 1.1 Frontera de salida y formulación débil del flujo

| ID | Clase | Ubicación | Expresión original | Problema | Corrección |
|---|---|---|---|---|---|
| F01 | EV | Orig. §3.2, forma `a_h` | `a_h(h,v)=∫K∇h·∇v − ∫_{Γout}(K∇h·n)v` | El término `−∫_{Γout}(K∇h·n)v` retiene el flujo normal desconocido. No está definido en `H¹×H¹` (la traza del flujo vive en `H^{-1/2}`), no es una forma bilineal acotada y no corresponde a ninguna condición de frontera. | Nueva §3.2: `a_h=∫K∇h·∇v` y `ℓ_h` con un dato (`f_out`) o con espacio de ensayo `H¹_{0,out}` (cierres A y D). |
| F02 | EF / OM | Orig. §2.5 (tabla) y §4.2 | `Γout: condición libre (∇(K∇h)·n=0)` | La condición pretendida es `∂ₙu=(∇u)n=0` con `K` variable. Son 2 ecuaciones de orden 2 sobre un escalar de orden 2: no cierra. El texto no decía qué implica ni cómo entra en la forma débil. (Mi edición del 4 nov, ítem 10 de la revisión anterior, la reescribió como «flujo libre (no se impone dato…)», que era incorrecta y se revierte.) | Nueva §2.2: análisis completo (Prop. 2.1 y componente tangencial) y cierres (A)–(D). **Sin elegir**: decisión D1. |
| F03 | EV | Orig. Teorema 4.2 | «En Γout … el adjunto queda sobredeterminado (ψ_h=0 y K∇ψ_h·n=K b·n)» | La sobredeterminación era consecuencia de F01. Con un cierre esencial (A, D), `δh=0` en `Γout` y basta `ψ_h=0`; con el cierre natural (B), basta la condición de flujo. | Nuevo Teorema 4.3 con tabla de condiciones por cierre y prueba. |
| F04 | EF | Comparación BFR–FEniCSx | (no estaba) | FEniCSx usa el cierre (B) con el perfil `f_out(z)`; BFR `pfs` (por defecto), `libre`, `fem`, `dirichlet` usan otros. Con `pfs` los dos backends resuelven **problemas continuos distintos**: error relativo de `|u|` 0.72 con `pfs` frente a 0.10 con `fem` (casos M0 y M3). | Nueva §2.2 lo documenta. Pendiente E3 (diagnóstico de `∂ₙu`). |
| F05 | OM | Orig. V4 / V5 | (no se decía) | El adjunto exacto de BFR solo se verificó con `bc_outlet_mode="libre"` (fijado en `tests/test_adjunto_exacto.py`). Nada con `pfs` (por defecto) ni `fem`. | Nueva V5: se declara; pendiente E5. |
| F06 | EN / OM | `make_H_bc`, modo `pfs` | — | En una salida vertical la fila impuesta es `∂xKx·h_x + ∂xKz·h_z + Kx·h_xx − Kz·h_xz = 0`. Equivale a `∂ₓ(Kx hₓ)+∂ₓKz h_z−Kz h_xz`; el término mixto tiene el signo contrario al de `∂ₓ(Kz h_z)`. No coincide con la componente normal ni con la vectorial. | Decisión D7 (confirmar intención). No se afirma que sea un error. |

### 1.2 Modelo de flujo y transporte

| ID | Clase | Ubicación | Original | Problema | Corrección |
|---|---|---|---|---|---|
| F07 | EF | Orig. §2.1, §3, §4 | `f=−Q g_ε`, un único núcleo | El código usa dos núcleos: Wendland para el sumidero de flujo en BFR y gaussianos de anchos distintos en FEniCSx (`ef`, `ep`). El gradiente es exacto solo si estado y gradiente usan el mismo par. | Nuevo núcleo `ϱ` (flujo) y `g_ε` (observación); fórmulas `∫ϱψ_h`, `Q∫ϱ̇ψ_h`; se reduce al caso previo si `ϱ=g_ε`. Decisión D5. |
| F08 | OM | Orig. Prop. ADR, demostración | «la cuenta es la misma probando con `v∈H¹`» | La equivalencia débil exige que la forma conservativa conserve el término de frontera advectivo `∮(u·n)Cv`; no se decía. | Texto de la prueba ampliado (Nueva Prop. 2.2). |
| F09 | OM | Orig. §2.2 | (no estaba) | No había balance global de masa de soluto ni se advertía que la cancelación `fC` es exacta solo en el continuo (con `u=−K∇h_h` discreta la divergencia es aproximada). | Ec. (2.6) y observación; verificación E4. |
| F10 | EF | Orig. §2.1, ec. de transporte | `φ` constante con almacenamiento elástico en el flujo | El término `−S_s∂ₜh C` aparece porque `φ` es constante en el tiempo mientras el flujo tiene almacenamiento. Si el almacenamiento fuese `∂ₜφ=S_s∂ₜh`, el término se cancelaría. Además `S_s=φ` numérico (0.03–0.95 m⁻¹) es físicamente absurdo (valores usuales 1e-6–1e-4) y hace que el término sea comparable con `λ_d`. | Decisión D2. No se cambió el modelo. |
| F11 | OM / EN | Orig. §2.4 (MRMT) | `c_0=Σλ_r`; «`N_r=0` recupera ADR» | Con `N_r=0` se recupera ADR **con `λ_d=0`**. El código aplica `landa` en FEniCSx a los tres modelos (`forward.py:353`) pero en BFR solo a ADR (`operators.py:490`). Con `landa=2.3e-7` (nitrato) las variantes MRMT de BFR y FEniCSx no resuelven la misma ecuación, y la optimización del nitrato con FEniCSx sí lleva decaimiento en MRMT. | Observación y decisión D3; verificación E6. |
| F12 | OM | Orig. §2.5 | `(D∇C)·n=0` en todo `∂Ω` | En `Γin` entra agua; la condición deja que arrastre la concentración local, sin prescribir concentración de entrada. | Decisión D4 con la condición alternativa y su efecto en el adjunto. |
| F13 | OM | Orig. hipótesis | (no figuraba) | `∂ₜh∈L^∞` necesario para Gårding con reacción de signo indefinido; `D` diagonal necesaria para la fórmula de `d`; regularidad de `b` para la forma fuerte del adjunto del flujo; `z_p<z_0`; datos de frontera independientes de los controles; soportes de los núcleos. | Hipótesis 3, 5, 6 y 7 (nuevas). |
| F14 | ME | Orig. §1 | tabla sin unidades | Faltaban unidades y la diferencia entre `λ_r` (móvil), `ν_r` (relajación), `η_r` (capacidades). | Tabla con unidades; Observación de parámetros (`λ_r=φ_{im,r}Rν_r`, `ν_r=D_eff/(Rη L²)`, `η_r=β_r` del código). |

### 1.3 Funcional y gradientes

| ID | Clase | Ubicación | Original | Problema | Corrección |
|---|---|---|---|---|---|
| F15 | OM | Orig. §3.1 | `J_m=∫∫g_ε C²` descrito como «concentración captada» | Es un índice cuadrático ponderado, no la masa extraída `∫Q∫ϱC`; es positivo con `Q=0`. | Observación en Nueva §3.1; no se cambia el funcional. Decisión D6. |
| F16 | OM | Orig. §3.1 | sin conjunto admisible ni unidades | No se definía `U_ad`; los pesos `γ,α,β` absorben unidades distintas. | `U_ad` y unidades en §3.1; observación de la normalización (`w₁,w₂,w₃`, solo FEniCSx). |
| F17 | OM | Orig. Lema 3.1 | solo `(h,C)` | Se usa para MRMT (tres estados) y faltaba la hipótesis de diferenciabilidad del control al estado. | Lema generalizado a `y` y con hipótesis explícita. |
| F18 | ME | Orig. §4.3 | «los controles aparecen en tres lugares» | Mezclaba `Q` (2 lugares) y `z_p` (4). | Separado. |

Se recalcularon sin hallar error: ecuación adjunta del transporte (ADR y MRMT), condición natural `(D∇ψ+uψ)·n=0`, Corolario (forma no conservativa, con `+QϱψA`), adjunto de la región inmóvil y su solución con memoria, adjunto del flujo (`−S_s∂ₜψ_h−∇·(K∇ψ_h)=−∇·(Kb)−S_s∂ₜ(Cψ)`), gradientes en `Q` y `z_p` (signos y factores), recursiones discretas ADR/bloque/semi (índices de residuos), y la parte verde de `σ`.

### 1.4 Esquemas discretos

| ID | Clase | Ubicación | Original | Problema | Corrección |
|---|---|---|---|---|---|
| F19 | EN | Orig. ec. (6.x) `disc_adr` | `(P^n)ᵀψ^n=(R^n)ᵀψ^{n+1}−∇j_n` | El texto dice que el recorte `C≥0` se incluye como máscara, pero la ecuación no la lleva. El código la aplica (`A^Tλ=M(…)`). | `M^n[…]` en las tres recursiones y en los Algoritmos 2–4. |
| F20 | ME / OM | Orig. §6 | `ω_r`, `η_r` sin definir; `α_r` colisiona con el peso `α` del funcional | Notación no definida antes de usarse. | `ν_r`, `η_r`, `κ_r=Δtν_r/(2+Δtν_r)`, `ω_r=η_rν_r/(2+Δtν_r)`, `e_r`, `a_r`; verificado contra `config/parameters.py` y `fenicsx/forward.py` (`α_código=β D_eff/(2βRL²+Δt D_eff)`). |
| F21 | ME | Orig. §6 | `g` para el vector nodal del núcleo y `g` para el gradiente | Colisión de símbolo. | `δ_p` (núcleo de flujo) y `M_g` (cuadratura del núcleo de observación). |
| F22 | OM | Orig. §6 | (no estaba) | No se explicaba la relación entre los multiplicadores discretos y continuos (factor `Δt`, pesos nodales) ni que el funcional discreto es la regla del extremo derecho. | Párrafo «Relación con el adjunto continuo»: verificado para FEM (V2), no para RBF-FD. |
| F23 | EN | Revisión previa (mía) | «las fórmulas están escritas para θ_C=1» | Era al revés: la fórmula `C^{m+1}+C^m` corresponde al trapezoidal (`θ_C=½`). Ya corregido en `6b1b056`. | Se mantiene la forma con `w_A,w_B`. |

### 1.5 Algoritmos

| ID | Clase | Ubicación | Original | Problema | Corrección |
|---|---|---|---|---|---|
| F24 | EN | Orig. Alg. 1 | `Armijo` evalúa `J(x_k+αd_k)` y acepta `Π(x_k+αd_k)` | La búsqueda lineal evalúa un punto distinto del que se acepta; el test usa `gᵀd` en lugar de `gᵀ(x⁺−x)`. | Nuevo Alg. 1: se evalúa y acepta `Π(x_k+αd_k)`, test con `gᵀ(x⁺−x_k)`. |
| F25 | EN | Orig. Alg. 1 | `‖g_k‖<ε` | Con cotas debe usarse el gradiente proyectado. Sin tratamiento si Armijo agota `m_max`. | `‖x−Π(x−g)‖<ε`; reinicio a `−g` y salida si ya era `−g`. |
| F26 | OM | Orig. Observación | `trust-constr` | Se decía que Alg. 1 «es la versión sin restricciones de suministro» de `trust-constr`, lo que sugiere equivalencia. | El Alg. 1 es el esquema teórico y **no** el usado en los experimentos; `trust-constr` se describe aparte con su criterio (`gtol` sobre `J/J_scale`). Se señala que las variables tienen escalas muy distintas y que PR⁺ + Armijo no tiene garantía global. |

### 1.6 Verificaciones V1–V6

| ID | Clase | Hallazgo |
|---|---|---|
| F27 | EN | V3, «sin el término `d` el error es 0.7–0.95»: se reejecutó; con `S_s=1` vale 0.73/0.89/0.96, pero con `S_s=4` vale **0.11**. La prueba discrimina, pero el texto exageraba. Corregido. |
| F28 | EN | V3, gradiente respecto de `x_p` (2.8, 0.9, 0.2·10⁻³, decreciente) y forma conservativa vs no conservativa (2.6, 0.7, 0.17·10⁻³): **no existía ningún guion en el repositorio** que los produjera. Se escribió `tests/banco1d/runp1_extra.py` y se reejecutó: el gradiente en `x_p` da 5·10⁻⁴–1·10⁻³ (paso 10⁻⁵) **sin decrecer con `n`** y muy sensible al paso de diferencias finitas (1.6·10⁻² con paso 10⁻³ a `n=100`); la comparación conservativa/no conservativa da **3.6·10⁻¹³, 1.9·10⁻¹³, 1.5·10⁻¹²**, es decir, coinciden a precisión de máquina con P1 consistente (esto confirma el signo y factor de `−S_s∂ₜh C`, pero no mide error de modelo). Los valores anteriores no se reprodujeron y se sustituyeron. |
| F29 | OM | V4: «`C≡1`, dispersión y recorte activos». Con `C≡1` las derivadas espaciales son nulas al inicio y «recorte habilitado» no implica «recorte activado»; no se comprobó que algún nodo se recorte. La opción `--modulada` existe y no figura en la tabla. Cierre usado: `libre`. |
| F30 | OM | V5 usa `Q=0.3` m²/s, 30 veces `Q_max=0.01`. |
| F31 | OM | V1–V2 corresponden al cierre (B) natural de FEniCSx; V3 a un cierre esencial en 1D; V4–V5 a `libre`. Ninguna prueba cubre BFR con (B) o (A). |

Lo que sí quedó confirmado hoy: V3 (columna `Q`, ADR `S_s=1,4`, MRMT) reproduce los valores del texto; V6 álgebra reproduce 1.1e-9, 8.8e-10, 4.1e-10; V4 reproduce «Q aleatoria» (1.46e-5, 1.44e-5, 2.96e-5) y «(Q,z_p)» (4.76e-6, 5.23e-6, 4.90e-6) para las tres variantes.

## 2. Decisiones pendientes del autor

| ID | Decisión | Por qué no se resolvió |
|---|---|---|
| D1 | **Cierre de Γout**: RESUELTO por el autor (4 nov): derivada normal de u nula, adoptada como cierre (D) (componente normal). Pendiente confirmar que no se exige la componente tangencial y si FEniCSx debe pasar de (B) a (D). | — |
| D2 | Modelo de almacenamiento: `φ` constante con `−S_s∂ₜh C`, o `∂ₜφ=S_s∂ₜh` (término nulo); `S_s=φ` numérico vs valores físicos. | Es una decisión de modelación. |
| D3 | Decaimiento en MRMT: con o sin `λ_dφRC_m` (hoy FEniCSx sí, BFR no). | Afecta resultados del nitrato. |
| D4 | Concentración de entrada en Γin (`C_in`). | Dato físico. |
| D5 | Núcleos: `ϱ=g_ε` o distintos. | Hoy difieren entre BFR y FEniCSx. |
| D6 | Funcional: índice `J_m` o masa extraída; formalizar la restricción de suministro. | Cambia el significado del objetivo. |
| D7 | Fila `pfs` de BFR: intención y signo del término mixto. | No se puede deducir del código. |

## 3. Trazabilidad

| Corrección | Ecuaciones y secciones afectadas (Nueva numeración) |
|---|---|
| Nueva §2.2 (frontera de salida) | Prop. 2.1; tabla de cierres; §2.6 (tabla de frontera); §3.2 (espacios `V_h`); Teorema 4.3 (condiciones de `ψ_h`); §6 (filas de frontera del flujo); V3–V5 (qué cierre ejercitan); §9 (D1, D7, E2, E3, E5) |
| `a_h` sin flujo desconocido (F01, F03) | (3.2) formas `a_h,ℓ_h`; (3.4) estado débil; Lagrangianos (3.5)–(3.6); (4.5) adjunto débil del flujo; Teorema 4.3 |
| Núcleo `ϱ` (F07) | tabla de notación; (2.1), (2.2); Cor. 4.2 (`+QϱψA`); (4.7)–(4.8) gradientes; Teorema 5.1; §6 (`δ_p`); Alg. 2–4 |
| Hipótesis 3, 5, 6, 7 (F13) | Teoremas 4.1, 4.3, 5.1; Lema 3.1; definición de `d` y `b` (4.4) |
| Decaimiento y parámetros MRMT (F11, F14) | Observación de §2.5; `c_0` en la tabla; (5.3) solución con `ν_r`; §6 `κ_r,ω_r,e_r,a_r`; Alg. 4 |
| Máscara `M^n` (F19) | (6.1), (6.2), (6.3); Alg. 2–4 |
| Algoritmo 1 (F24–F26) | §7; observación de alcance |
| Verificaciones (F27–F31) | §8 completo; `tests/banco1d/runp1_extra.py` |

## 4. Verificaciones numéricas adicionales necesarias

Copia de la §9 del PDF, en orden de prioridad:

1. **E1** Diferencias finitas del gradiente completo en la configuración tipo `main` (`Q≤Q_max`, malla compartida, fuente activa, `C` que llega al pozo), BFR y FEniCSx, ADR y MRMT, con barrido del paso y prueba de Taylor.
2. **E3** Diagnóstico de `∂ₙu=0`: evaluar `n·∂ₙu` y `τ·∂ₙu` en Γout sobre `pfs`, `libre` y `fem`.
3. **E2** Solución manufacturada 2D del flujo con `K` variable para el cierre elegido.
4. **E5** Adjunto exacto de BFR con `pfs` y `fem`, con `--modulada`, y una prueba que active el recorte.
5. **E6** Consistencia del decaimiento BFR–FEniCSx en MRMT (nitrato).
6. **E4** Balance global de masa de soluto en 2D (conservativa vs no conservativa).
7. **E7** Identidad de dualidad con perturbaciones aleatorias de `Q` en ambos backends.
8. **E8** Sensibilidad del gradiente en `z_p` al paso de diferencias finitas y por qué no decrece con la malla (F28).

## 5. Qué demuestra el código y qué el análisis

- **El análisis (este documento)** demuestra: la cancelación de `fC`, las ecuaciones adjuntas continuas y los gradientes a partir del Lagrangiano, las recursiones discretas escritas, y la proposición sobre la componente normal de `∂ₙu=0`.
- **El código** respalda: que el adjunto exacto de BFR coincide con diferencias finitas en los casos V4–V5 con `libre`; que el adjunto discreto de FEniCSx es consistente con las ecuaciones continuas (V2); los pesos `θ_C` y la parte verde de `σ` (leídos en `adjunto_exacto.py`).
- **No está demostrado:** convergencia espacial y temporal del adjunto; el comportamiento con `pfs` y `fem`; la condición `∂ₜh∈L^∞`; la relación entre multiplicadores discretos de colocación y continuos; el argumento de que `∂ₙu=0` vectorial es inconsistente en general es de recuento y de casos, no un teorema.
