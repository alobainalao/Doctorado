# ¿Por qué H y |U| de BFR y FEM antes diferían tanto y ahora casi coinciden?

Datos: `scripts/comparison_report.txt` (2026-10-07, commit 75fbbe4), `data/output/experimentos/resumen.csv`
(main, malla saneada) y el grupo `H_ablacion_flujo` (2026-11-02; `python -m experimentos.tabla_ablacion_flujo`).

## Antes (2026-10-07, ADR, 144 pasos)
H: rango BFR 17.05 m frente a FEM 3.82 m; |U|: error L2 relativo 0.69, máximo 1.99e-3 frente a 1.93e-3 m/s,
error angular medio 17°. Cono de depresión BFR ≈4.5 veces más profundo.

## Ablación de la configuración (12 pasos, sólo flujo, cada caso suma un cambio al anterior)
| caso | cambio | rms(H_BFR−H_FEM) m | rango H FEM m | rms/rango | error L2 de \|U\| |
|---|---|---|---|---|---|
| H0 | malla 2026-10-22 sin sanear, Q=1e-3, kernel 2.5h por suma, θ=0.9 | 0.004 | 2.99 | 0.15 % | 0.18 |
| H1 | Q_std = 0.3 | 0.012 | 11.6 | 0.10 % | 0.04 |
| H2 | malla saneada | 0.012 | 11.9 | 0.10 % | 0.04 |
| H3 | r_pozo = 20 m | 0.015 | 19.4 | 0.08 % | 0.13 |
| H4 | delta normalizada por área | 0.008 | 3.59 | 0.24 % | 0.17 |
| H5 | θ = θ_C = 1 (configuración actual) | 0.008 | 3.59 | 0.23 % | 0.17 |
| H6 | BFR con bc_outlet_mode = fem | 0.002 | 3.59 | 0.04 % | 0.12 |

(Comparación en nodos del dominio, FEM remuestreado al vecino más cercano cuando los nodos no coinciden; las
mallas distintas de H0/H1 impiden usar `comparar.py` nodo a nodo.)

## Lectura
1. **Con la configuración actual del código, incluso la configuración antigua da H casi idéntico** (≤0.25 % del
   rango de H). Los cambios de configuración (malla, radio del kernel, normalización de la delta, θ, Q) **no** explican
   el salto de 17 m → 3.8 m. Lo que lo explicó son defectos de **código** ya corregidos, que no se pueden activar
   por configuración: delta del pozo en el nodo más cercano (47.6 % del caudal en un nodo, `docs/kernel_pozo_fix.md`),
   nodo forzado en el pozo, condición de entrada BFR no uniforme, y suavizado de malla que movía los nodos de frontera
   hasta 15–30 m hacia dentro (corregido en la malla compartida; **sigue activo en main para corridas sin `mesh_pkl`**).
2. **|U| no es «casi idéntico»**: el error L2 sigue en 4–18 % con pozo y llega a 72 % sin pozo (A0, A1 de main) con
   la condición de salida pfs; con `bc_outlet_mode = fem` baja a 10 % (B0). La velocidad ambiente es pequeña y
   la diferencia de condición de salida la domina; en H el efecto es invisible porque el cono (Q = 0.3) fija el rango.
3. La mejora aparente de H en main a partir del 2026-10-28 viene en parte de **Q_std 1e-3 → 0.3**: el cono domina el
   rango y los errores relativos al rango (y la escala de color 305–309 m de los videos) hacen invisibles 0.04–0.1 m.
