# Resultados del problema directo (grupo `M_modelo_nuevo`)

Configuración tipo `main`: spacing 60, dt 18000 s, T = 30 d (144 pasos), `Q_std=1e-3`, `factor_sumidero_C=0`, `factor_almacenamiento_C=1`.
Métricas BFR vs FEniCSx al tiempo final (`data/output/experimentos/M_modelo_nuevo/<caso>/comparacion/metricas.json`).

| Caso | ΔH rms (m) | C L2 rel | C L∞ rel | \|U\| L2 rel | masa BFR/FEM |
|---|---|---|---|---|---|
| M0 ADR (`pfs`) | 0.0731 | 0.186 | 0.248 | 0.718 | 0.846 |
| M1 MRMT semi | 0.0731 | 0.153 | 0.227 | 0.718 | 0.878 |
| M2 MRMT bloque | 0.0731 | 0.152 | 0.223 | 0.718 | 0.878 |
| M3 ADR, BFR con `bc_outlet_mode=fem` | 0.0105 | 0.086 | **0.401** | 0.100 | 1.007 |

- Rango de H del FEM: 3.10 m ⇒ ΔH rms de M0 = 2.4 % del rango. Carga en el pozo: 307.71 (BFR) vs 307.81 (FEM) con `pfs`; 307.812 vs 307.812 con `fem`.
- Concentración en el pozo (30 d): ADR 4.2e-15 (BFR) vs 4.0e-14 (FEM); MRMT 3.8e-15 vs 2.4e-14 ⇒ BFR ~10× menor.
- **Lectura:** la condición de salida de BFR (`pfs`) explica casi toda la discrepancia de velocidad (0.72 → 0.10), de C en L2 (0.186 → 0.086) y de masa (0.85 → 1.01).
- **Matiz:** C L∞ relativo *empeora* con `fem` (0.248 → 0.401). L2 mejora, pero el máximo puntual no; no se ha investigado (posible desplazamiento del frente o difusión numérica distinta). No se verificó aún C en el pozo para M3.
- Sin diferencias finitas del adjunto en esta configuración todavía (ver `PENDIENTES.md`).
