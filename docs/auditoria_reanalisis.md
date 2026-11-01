# Informe de auditoría — reanálisis ADR/MRMT (rama `exp/sumidero-W`)

Documento fuente: `docs/reanalisis_adr_mrmt.tex` (PDF homónimo). El original
`docs/adjunto_sin_sumidero.{tex,pdf}` se conserva sin cambios.

## 1. Correspondencia de ecuaciones con el PDF original

| PDF original | Reanálisis | Contenido |
|---|---|---|
| (1.1) | (4.1) | flujo `S_s ∂ₜh − ∇·(K∇h) = f = −Q g_ε`, `u = −K∇h` |
| (1.2) | (4.2) | `∇·u = f − S_s ∂ₜh` |
| (1.3) | (4.3) | balance conservativo de soluto, `… = W₊ + fC` |
| (2.1) | (4.4) | forma desarrollada con `C ∇·u` |
| (N-ADR) | (4.6) | ADR sin extracción explícita, con `−S_s ∂ₜh C` |

Verificado por derivación (documento §4): (4.4) sale de (4.3) por la regla del producto;
al sustituir (4.2) el término `fC` aparece en los dos miembros y se cancela; `S_s ∂ₜh C` no.
La extracción queda implícita (balance de masa (4.7): aparece `−Q∫g_ε C`). No se reintrodujo
ningún término explícito de extracción.

## 2. Inconsistencias encontradas

| # | Dónde | Descripción | Resolución |
|---|---|---|---|
| I1 | PDF original §Lagrangiano/adjunto | Tomaba `C = 0` en Γ_D y `ψ_C = 0` en Γ_D. El código (BFR y FEniCS) impone flujo dispersivo nulo en **toda** ∂Ω, y `ChaptersR` también. | Reanálisis: `(D∇C)·n = 0` en ∂Ω; el adjunto resulta con `(D∇ψ + uψ)·n = 0` (natural), no `ψ = 0`. |
| I2 | `tesis/ChaptersD/Chapter2.tex` ec. (2.11) | El Cap. 2 doctoral enuncia `C_m = 0` en Γ_in∪Γ_out para el continuo y dice (Sec. 2.5.5) que RBF-FD lo sustituye por flujo dispersivo nulo. `ChaptersR/CapI_ADR.tex` usa flujo nulo en toda la frontera. | **Pendiente de decisión de la autora/autor de la tesis**: el reanálisis adopta flujo dispersivo nulo (coincide con el código). No se modificó la tesis. |
| I3 | `ChaptersD/Chapter2.tex` ec. (2.8) | `−(K∇h)·n = f_in` con `f_in ≥ 0` "entrante": con `u = −K∇h` da `u·n = f_in ≥ 0` (salida). El código (FEniCS `−(K∇h)·n = h_inlet = inlet_a cos² < 0`; BFR fila `−K_x ∂ₓh = h_inlet`) y `ChaptersR` dan `(K∇h)·n = f_in ≥ 0`, `u·n = −f_in ≤ 0` (entra). Malla: Γ_in en x = 10755 con n = (+1,0); Γ_out en x = 816 con n = (−1,0). | Reanálisis: `(K∇h)·n = f_in`. Forma débil con `+∫_{Γ_in} f_in v` (la del Cap. 2 tiene `−`). No se modificó la tesis. |
| I4 | `ChaptersD/Chapter2.tex` ec. (2.9) | `∇(K∇h)·n = 0` es ambigua (gradiente de un vector). BFR (`make_H_bc`, modo `pfs`) implementa una forma concreta de segundo orden; FEniCS implementa Neumann de primer orden (`bc_outlet_mode=fem` en BFR reproduce esa). Ya documentado en la tesis. | Se cita la ecuación del Cap. 2 tal cual. |
| I5 | Cap. 3 / PDF | Adjunto continuo del flujo sobredeterminado en Γ_out (`ψ_h = 0` y `K∇ψ_h·n = K b·n`). | Ya reconocido en el Cap. 3; el adjunto se define a nivel discreto y se valida numéricamente. |
| I6 | `funtions/step_time.py` `step_adjoint` | El adjunto de BFR sustituye las filas de frontera de `Aᵀ` por `ψ_C = 0` en inlet/outlet y anula las de `Bᵀ` (líneas ~150-156, 213). Con transporte de flujo dispersivo nulo en todo el borde, la transpuesta exacta no impone `ψ_C = 0`. | Ver §4 (resultado del test). |
| I7 | PDF original §código | Normalización `½` (sin `1/Δt` ni `1/w_i`) de la fuente de almacenamiento del adjunto del flujo en BFR "contrastada con diferencias finitas" (error 3–5 %, MRMT hasta 17 %). Es un ajuste, no una derivación. | Ver §4. |
| I8 | `operators.py:482`, `upwind_operators.py:172` | El sumidero `W₋` se desactiva también si `run_type == "optimization"`, aunque `extract_C` sea `True`. Irrelevante con `factor_sumidero_C = 0`. | Sin cambio. |
| I9 | Parámetros | `S_s = φ = 0.1 m⁻¹` (valor no físico), de modo que `−S_s ∂ₜh C` no es despreciable. | Es el valor de la tesis (Tabla 2.1); se mantiene. |

## 3. Cambios de código (rama de trabajo)

| Archivo | Cambio | Motivo |
|---|---|---|
| `fenicsx/forward.py` | Almacenamiento con `h_new`/`h_old` (Functions) | El término usaba la TrialFunction del flujo (error UFL) y `h_n` ya valía `h` al resolver el transporte (∂ₜh = 0) |
| `fenicsx/adjoint.py` | MRMT-bloque: `ψ_im^{n-1} += 2Rα_r M_φ ψ_C^n` | Faltaba la dependencia del residuo en `C_im^{n-1}` |
| `fenicsx/adjoint.py` | MRMT-bloque: signo del término `2(1-θ_C)coef ψ_im` en `termC_next` | Se restaba con el signo contrario (oculto con θ_C=1) |
| `fenicsx/adjoint.py` | MRMT-semi: `∂R/∂h_prev` evaluado en `C*` pre-MRMT | Inconsistencia con el almacenamiento |
| `fenicsx/adjoint.py` | Recorte C≥0 condicionado a `clip_C` | Permite desactivarlo en pruebas |
| `funtions/operators.py`, `step_time.py` | Término `∇·(K d)` (∂ᵤD) en la fuente del adjunto del flujo BFR; `adj_disp_u` | Término de la formulación que faltaba. Efecto numérico ~1e-5 relativo en la prueba |
| `funtions/step_time.py` | Opción `adj_C_bc_exacta` (por defecto False) | Prueba de la frontera ψ_C; no cambió el resultado |
| `funtions/utils.py` | `C_init_override`, `C_init_modulacion` (por defecto None) | Sólo pruebas de gradiente |
| `experimentos/catalogo.py` | Grupo `N_modelo_nuevo` | Las 6 variantes |
| `tests/test_gradiente_modelo_nuevo.py`, `experimentos/optimizar_modelo_nuevo.py`, `scripts/generar_verificaciones.py` | nuevos | Pruebas y optimización |

## 4. Resultados efectivamente ejecutados

- **FEniCSx, adjunto vs diferencias finitas** (sólo γJ_m, 3 pasos, θ_C=0.9, C uniforme y modulada, con y sin
  almacenamiento): ADR, MRMT-semi y MRMT-bloque coinciden a 1e-6…1e-10 tras las correcciones. Antes: MRMT-semi
  hasta 11 %, MRMT-bloque hasta 19 %.
- **BFR, adjunto vs diferencias finitas** (ADR, C modulada, Q≈1e-3): **no concuerdan** (factor 2–7 con
  almacenamiento; sin almacenamiento el adjunto da ≈0 y las diferencias finitas ≈ −98). Sin recorte C≥0, con
  frontera ψ_C exacta o sin el término ∂ᵤD el resultado es el mismo. Interpretación (no demostrada): el forward BFR
  tiene una sensibilidad a Q espuria en el nodo del pozo (oscilación ya documentada), por lo que las diferencias
  finitas no son una referencia fiable. **El adjunto BFR no queda validado.**
- **Optimización FEniCSx** (6 pasos, 10 iteraciones de trust-constr, C modulada): no convergió (límite de
  evaluaciones); J escalada 0.9985–0.9990; la iterada final de Q tiene componentes negativas (≈ −3e-4) aunque los
  límites son Q≥0. Prueba de funcionamiento, no un resultado.
- **No ejecutado**: balance de masa numérico, convergencia en malla/tiempo, optimización BFR.
