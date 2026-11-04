# Pendientes de la línea `sin-sumidero`

Foco acordado: primero problema directo y adjunto consistentes; la optimización es secundaria.

## Directo
- [ ] Pasar BFR a `bc_outlet_mode=fem` en la configuración tipo `main` y repetir M0–M2 (M3 solo cubre ADR).
- [ ] Explicar por qué C L∞ empeora con `fem` (0.248 → 0.401) y revisar C en el pozo para M3.
- [ ] Explicar el factor ~10 de C en el pozo (BFR menor que FEM) con `pfs`.

## Adjunto
- [ ] Diferencias finitas en configuración tipo `main` (spacing 60, dt 18000, `Q_std=1e-3`, banderas del modelo nuevo): BFR exacto y FEniCSx, para ADR, MRMT semi y MRMT bloque, con una pluma que llegue al pozo o C modulada.
- [ ] Chequeo de gradiente normalizado en MRMT (solo ADR hecho: errores 1e-9–9e-7).
- [ ] Replicar `normalizar_J` en BFR (`compute_functional`, gradiente exacto, optimizador).

## Optimización (después)
- [ ] Repetir con `Q_std=1e-3`, `J_scale=1` y $f$ común (`experimentos/optimizar_nitrato.py`); las corridas se detuvieron al cambiar el foco y no hay resultados revisados.
- [ ] Línea base factible (Q constante que cumpla la restricción de suministro $0\le S\le S_{max}$, demandas 750+250) en vez de $Q=0$.
- [ ] $J(z_p)$ no es suave (saltos ~0.02 en pocos metros, por el espaciado 60 y el kernel); revisar con `trust-constr`.

## Organización
- [ ] `codigo/data/output` (21 GB): decidir qué borrar con `docs/inventario_data_output.md`.
- [ ] Commits sin subir (no hacer push sin orden): `main` de `codigo` (+61), `avance1-doc` (+2), `tesis` (+2), `articulos` (+1).
- [ ] No tocar: `codigo-exp-continuidad` (21 archivos sin commit), cambios sin commit del worktree `codigo/`, `stash@{0}`.
