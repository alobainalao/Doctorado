# Experimentos numéricos — notebooks

Dos bloques, en el orden en que conviene leerlos:

| # | Carpeta | Qué contiene | Corre simulaciones | Tiempo |
|---|---|---|---|---|
| 1 | `1_resultados/` | `1_resultados_experimentos`: los resultados de los grupos A–F de `experimentos/` (BFR vs FEniCS), ordenados por pregunta, con tablas, figuras finales y qué es fiable hoy | no (lee `data/output/experimentos/`) | ~10 s |
| 2 | `../benchmarks_libro/` (fuera de `notebooks/`) | Casos de Manzoni–Quarteroni–Salsa (2021) resueltos con RBF-FD + FEM propio + adjunto discreto. Cada uno abre con una **ficha** (el problema en palabras, cómo lo resuelve el libro, cómo lo implementamos, cómo comparamos y resultados) | sí (casos pequeños) | 1–4 min |

## Bloque 2: benchmarks del libro

| notebook | problema | referencia | benchmark rápido |
|---|---|---|---|
| `2a_graetz_manzoni_6_5_1` | Graetz: advección–difusión estacionaria, control de flujo en la pared (§6.5.1, ec. 6.65, Tabla 6.2). Parte B: 4 pozos (control bilineal con cotas) | FEM graduado + Richardson (`graetz_referencia.json`) | `python graetz_benchmark.py` (~6 s) |
| `2b_calor_manzoni_8_3` | Calor transitorio con control distribuido, observación en $Q_T$ o en $t=T$ (§8.3, Tablas 8.2–8.3) | FEM con blanco exacto + Richardson (`calor_referencia.json`) | `python calor_benchmark.py` (~6 s) |
| `2c_pozos_transitorio` | Extensión propia: Graetz + pulso + 4 pozos con $u_i(t)$ constantes a trozos | FEM $n$=20/40/80, orden 2.05 (`pozos_referencia.json`) | `python pozos_benchmark.py` (~35 s) |

Los `*_nucleo.py` tienen las discretizaciones (RBF-FD con nodos fantasma, FEM $\mathbb P_1$) y los optimizadores.
Los `*_benchmark.py` admiten `--referencia` (recalcula la referencia) y `--congelar` (instantánea de regresión).

## Cómo regenerar

La fuente de cada notebook es el `.py` (celdas `# %%`, formato «percent»). El `.ipynb` se genera:

```bash
conda activate doctorado            # nbformat + nbclient + ipykernel
python notebooks/construir.py notebooks/1_resultados/1_resultados_experimentos.py --ejecutar
```

Los resultados de `data/output/experimentos/` no están en git y cada checkout tiene los suyos.
`rutas.py` busca cada grupo en todos los worktrees del repo, o en `EXP_ROOT` si está definido.
