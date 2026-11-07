# Fase C — Derivación, unidades y condiciones de validez

Rama `experiment/transport-continuity-substitution` (base `f69afdb`). Todo lo de abajo
se contrastó con el código de esa base (`funtions/operators.py`, `fenicsx/forward.py`).

## 1. Derivación

Flujo (ambos backends): `Ss ∂h/∂t + Q·w_R − ∇·(K∇h) = 0`, `q = −K∇h`
⇒ `∇·q = −Q w_R − Ss ∂h/∂t`.   (BFR: `H_vector`; FEM: `_flow_forms`)

Referencia A:  `∂(φRC)/∂t = ∇·(D∇C) − ∇·(qC) + W₊ − Q w_R C − λφRC`

`−∇·(qC) = −q·∇C − C∇·q = −q·∇C + Q w_R C + Ss (∂h/∂t) C`; el `+Q w_R C` cancela
el sumidero de A:

B:  `∂(φRC)/∂t = ∇·(D∇C) − q·∇C + Ss (∂h/∂t) C + W₊ − λφRC`
C:  B con `Ss ∂h/∂t = 0` (válida solo si el flujo es estacionario).

A ≡ B solo si `r_h = ∇·q + Q w_R + Ss ∂h/∂t = 0`; en general
`A − B = −C r_h` (con el mismo C). Así que r_h es la medida directa de la diferencia.

## 2. Qué implementa HOY cada backend (base f69afdb)

| | Forma | Ss ∂h/∂t·C | Sumidero Q w_R C |
|---|---|---|---|
| BFR | no conservativa `D∂²+(∇·D−u)·∇` | no | `factor_sumidero_C=0` (anulado) |
| FEM | no conservativa `u·∇C` débil | no | `f_W=0` (anulado) |

⇒ la configuración estándar **ya es la formulación C**. B solo añade `Ss ∂h/∂t·C`;
A (conservativa) no existe en la base limpia en ningún backend.

## 3. Unidades (2D, por unidad de espesor)

- `C` [M/L³], `φ,R` [–], `q` [L/T], `D` [L²/T], `h` [L], `Ss` [1/L].
- `Ss ∂h/∂t` [1/T]; `Q w_R` [1/T] ⇒ `Q` [L²/T], `w_R` [1/L²].
- Con `norm_delta_pozo="area"` (estándar) `∫w_R dΩ = 1` ⇒ coherente. Con `"suma"` (Σδ=1)
  `Q w_R` no tiene unidades de 1/T: solo se admite `"area"` en estos experimentos.
- Kernels distintos: h usa Wendland (`delta_p`, radio `r_pozo`); el sumidero de C usa
  `gauss_p`. En BFR `gauss_p` tiene amplitud 1 (∫=2πε²≠1); en FEM `g_pozo` está
  normalizado (1/2πε²). Para el control "sumidero artificial" hay que usar el MISMO
  `w_R` que en h; de lo contrario A deja de ser consistente con la continuidad.

## 4. Hallazgos que condicionan los experimentos

1. **Con A o B, C uniforme NO es solución si ∂h/∂t ≠ 0**: ambas dan
   `d ln C/dt = (Ss/φR) ∂h/∂t`. Solo C (sin almacenamiento) conserva C uniforme.
   Exp. 1 debe esperar: A≈B (deriva), C constante; no es un "fallo" de A/B.
2. **Ss = φ por defecto** (`preprocess.py:61-63`) da `Ss/φR = 1/R` [1/L]: un descenso Δh
   multiplicaría C por `exp(Δh/R)` (p. ej. Δh=−5 m ⇒ ×0.007). Es no físico
   (Ss real ~1e-5–1e-3 1/m). Los experimentos B usarán `S_s_const` (ya existe) en configs
   separadas, barriendo su valor; el estándar no se toca. Probable causa también de la
   divergencia del WIP conservativo de BFR (a confirmar con r_h y el balance).
3. **Dependencia temporal de φ**: A usa `∂(φRC)/∂t` con φ fijo en el tiempo, pero el agua
   sí se almacena (`Ss h_t`). A es el modelo de la tesis; B hereda su (in)consistencia.
   No se resuelve aquí; se documenta.
4. **Fronteras**: `∇·(D∇C−qC)` genera flujo advectivo `q·n C` en ∂Ω en A (BFR usa BC
   Neumann en `make_transport_bc`, FEM natural). En B/C el término `q·∇C` no genera
   flujo de frontera; el balance de masa de B debe incluir `∮C q·n` explícito. La
   equivalencia fuerte↔débil se comprueba con el residuo de balance, no se asume.
5. **Discreto**: con θ=θ_C=1 (estándar) `∂h/∂t ≈ (H^{n+1}−H^n)/Δt`; `∇·q` discreto
   con `grad` RBF-FD (BFR) o FEM de grado igual al de h, así que `r_h ≠ 0` a nivel de
   truncamiento. Se medirá `r_h` por paso y su norma en L²/Linf, en nodos interiores.

## 5. Balance de masa (Exp. 2), por formulación

`M = ∫φRC`. `dM/dt = −∮(qC−D∇C)·n + ∫W₊ − ∫λφRC − [A: ∫Q w_R C | B/C: —] (+ B: ∫(Ss h_t + ∇·q + Q w_R)... )`.
Para B/C, `dM/dt` contiene además `−∫C ∇·q` frente a `−∮Cq·n`: se reporta el término
`∫C r_h` como residuo explícito. Se registran por separado: ΔM, flujos de entrada/salida
por ∂Ω, masa extraída `∫Q w_R C`, fuentes, decaimiento, efecto del recorte `max(C,0)`
(Σ φR·|C⁻|·área) y residuo global. Masa extraída ≠ −ΔM; no se equiparan.

## 6. Criterio de tolerancia
No se fijan tolerancias a priori: el criterio de paso es que el residuo/diferencia
decrezca con el orden observado del esquema bajo refinamiento (Richardson), y que sea
≤ el error de truncamiento medido de A frente a sí misma en malla fina.
