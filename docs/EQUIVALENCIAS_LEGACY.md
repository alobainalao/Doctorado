# Equivalencias legacy (`main`) ↔ línea `sin-sumidero`

| Aspecto | Legacy (`main`, carpeta `codigo/`) | `sin-sumidero` (esta carpeta) |
|---|---|---|
| Término de sumidero $W_-$ | desactivado por defecto (`factor_sumidero_C=0` en el código) | desactivado; el análisis no lo considera (Prop. 2.1 de `adjunto_sin_sumidero`: el término $fC$ se cancela) |
| Almacenamiento $-S_s\partial_t h\,C$ | no (`factor_almacenamiento_C=0`) | sí (`factor_almacenamiento_C=1`) |
| Adjunto | forma anterior | $-\varphi R\partial_t\psi-\nabla\!\cdot(D\nabla\psi)-\nabla\!\cdot(u\psi)+c_0\psi-S_s\partial_t h\,\psi-\sum\lambda\psi_{im}=-2\gamma gC$; fuente del adjunto de flujo $-S_s\partial_t(C\psi)$ |
| Adjunto BFR | aproximado | discreto exacto (`funtions/adjunto_exacto.py`; ADR, MRMT bloque y semi) |
| Adjunto FEniCSx | UFL AD | UFL AD (sin cambios) |
| Normalización de $J$ | no | opt-in `normalizar_J` (solo FEniCSx; **no** está en BFR) |
| Config estándar | `config/default.json` | `config/default.json` **sin cambios** (regla 6). El modelo nuevo y el nitrato van en `config/nitrato.json` |
| Casos del catálogo | grupos A–G, S, W, Z | + `M_modelo_nuevo` (M0 ADR, M1 MRMT semi, M2 MRMT bloque, M3 ADR con `bc_outlet_mode=fem`) |
| Benchmarks del libro | `notebooks/2_benchmarks_libro/` | `benchmarks_libro/` (contenido idéntico) |
| Salidas de verificación del adjunto | raíz del repo | `data/output/verificacion_adjunto/` |

## Banderas y configs

- `Parameters(env=...)` filtra por `VALID_KEYS` y cae a los valores por defecto del código; `Parameters(config=dict)` toma el diccionario completo (lo usa `experimentos/`).
- `main.py` carga por defecto `config/malla_sp60.json` (spacing 60, dt 18000, T 30 d, n_stencil 25, sin `Q_std` ⇒ 1e-3 del código). `default.json` tiene `Q_std=0.3` (desde 2026-10-28) y dt 36000.
- `config/nitrato.json` = `default.json` + `landa=0.02/86400`, `C_ref=0.05`, `normalizar_J=true`, `factor_sumidero_C=0`, `factor_almacenamiento_C=1`, `Q_std=1e-3`.

## Cómo reproducir

```
python3 -m experimentos.correr --grupo M_modelo_nuevo --comparar      # directo, BFR vs FEniCSx
python3 -m experimentos.correr --caso M3_adr_bc_fem --comparar       # BFR con salida tipo FEM
python3 -m experimentos.optimizar_nitrato <adr|mrmt_semi|mrmt_block>  # optimización (fenics_env)
```
