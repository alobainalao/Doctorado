# Línea de investigación: condicionamiento y oscilaciones en BFR (RBF-FD)

Pregunta: ¿el mal condicionamiento de RBF-FD causa las oscilaciones de BFR? Resultados del 12-nov-2026 (ADR,
configuración de `MC_malla_conos`/`M_modelo_nuevo`: sin W₋, almacenamiento, salida libre, Q = 0.3, Δt = 5 h, θ_C = 1).

| Hipótesis | Prueba | Resultado |
|---|---|---|
| Estenciles locales mal condicionados | `estenciles.py`: cond₂ de [Φ P; Pᵀ 0] por nodo (MQ, εh = 1, n = 25) | Mediana 4·10³; p99 0.7–1.3·10⁵; máx 1.4·10⁷ en la transición de espaciado a ~100 m de la frontera. Junto al pozo (< 30 m) es la zona **mejor** condicionada (4·10³). Laplaciano: peso central < 0 en todos los nodos. **Descartada** como causa en el pozo. |
| Péclet de malla alto (esquema centrado) | Pe = h·máx_k |u_k|/D_kk con D = D_total(D_m, u) | Junto al pozo Pe ≤ 1.0 (D ≈ α|u| crece con la velocidad). Máximo 15 lejos del pozo (h ≈ 90 m). **Descartada** en el pozo. |
| Reacción de almacenamiento −S_s∂ₜh·C con signo de crecimiento | ΔH por paso / R (la diagonal implícita es φR/Δt − φ∂ₜh) | ΔH/R ≤ 0.03 en todo el dominio; junto al pozo h solo baja. **Descartada** en esta configuración. |
| El recorte C ≥ 0 oculta un modo inestable | corrida sin recorte (`cfg_sin_recorte.json`, `clip_C=False`) | Estable; diferencia L2 con/sin recorte 0.5 %. En el dominio, mínimo −0.47 % de C_máx (cerca de la fuente, undershoot dispersivo); los negativos grandes (−16 %) están en nodos **fantasma**. Nada en el pozo. |

Conclusión: con la configuración actual el **directo** de BFR no oscila (C en el pozo crece monótona; 0 nodos con
alternancia temporal). La oscilación documentada el 30-oct (0 ↔ 1.4 C₀) era del dominio cuadrado con la configuración
anterior. Lo que sigue ruidoso es el **adjunto** de BFR (ψ_C, ψ_H), cuya causa documentada es la asimetría de la
colocación fuerte (no el condicionamiento): ver la memoria «Ruido de ψ_H del adjunto BFR».

Siguiente paso posible: espectro global (autovalores con parte real positiva) de los operadores de flujo y transporte
en las dos mallas, y del sistema transpuesto que resuelve el adjunto.
