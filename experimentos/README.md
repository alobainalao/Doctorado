# Experimentos BFR (RBF-FD) vs FEniCS (FEM)

> **Lectura ordenada de los resultados:** `notebooks/README.md`. Allí están (1) los resultados de todos los grupos
> por pregunta, (2) los benchmarks del libro y (3) el análisis del sumidero $W_-$ en el cuadrado con pozo
> (`notebooks/3_cuadrado_pozo/`). Ese análisis concluye que en la forma no conservativa el término sobra, y que el
> implementado además no lleva $Q$.

Estructura única para definir, correr y comparar casos de estudio. Sustituye a
los lanzadores sueltos que ahora están en `scripts/legacy/`
(`generar_comparacion.py`, `run_all_6_variants.py`, `run_comparison.py`, …).

## Principios

1. **Malla congelada y compartida, saneada.** `data/input/malla_compartida/malla.pkl`
   la genera `python -m preprocessing.malla_compartida` (bfr_env), sin depender
   del backend: los nodos de BFR (4 niveles 4/16/30/60 m, `n_stencil=35`) con
   frontera exacta (nodo en cada esquina), sin nodos pegados al borde y una
   triangulación restringida con ángulo mínimo ≥ 28°. Guarda nodos, grupos,
   normales, fantasmas **y triángulos**: BFR reconstruye sus operadores sobre
   esos nodos (`mesh_pkl` en `preprocessing/preprocess.py`) y FEniCS usa esos
   mismos triángulos sin retriangular (`build_mesh_from_bfr_pkl`). La
   comparación es nodo a nodo, sin interpolación. `malla.png` muestra la malla
   y los zooms de esquinas y pozo.
2. **Misma CI.** `data/input/funcion.h5` (C ≥ 0), vía `get_init_values`.
3. **Config completa y explícita.** `base.json` contiene *todas* las claves
   y se pasa con `Parameters(config=...)`, que no descarta nada. (El modo env
   de la app sólo acepta una lista de claves: `n_stencil`, `theta`, `phi`, …
   caían a los defaults de Python).
4. **`default.json` no se toca** (Regla 6 de `CLAUDE.md`).
5. **Un caso = cambios respecto a `base.json`** (`catalogo.py`).
6. **Cada caso en su carpeta**: nada se pisa.

```
experimentos/
  base.json      config de referencia COMPLETA
  catalogo.py    grupos y casos
  correr.py      lanzador (cada backend en su entorno conda)
  _worker.py     corre un caso dentro del entorno (lo usa correr.py)
  comparar.py    UNA pasada por caso: métricas + videos de comparación con
                 los mismos arreglos (sin figuras de postproceso aparte)
  informe.py     informe.html general (lee metricas.json + videos; no recalcula)
  conclusiones.html  texto de revisión y conclusiones que inserta informe.py
  analisis_casos.json  análisis escrito de cada caso (con las métricas con que
                 se escribió: el informe avisa si quedó desactualizado)
  animar.py      videos H/C/|U| sueltos por backend (opcional)
  verificar.py   malla congelada (hash) y CI ≥ 0, antes de correr

data/output/experimentos/<grupo>/<caso>/
  bfr/          simulation_results.npz, config.json (la usada), run.log,
                figures/{H,C,V}.mp4
  fenicsx/      ídem
  comparacion/  metricas.json, comparacion_{H,C,V}.{mp4,webm} (un video por
                campo: BFR / FEM / BFR−FEM a todo el ancho, flechas de
                dirección en |U|, curvas con cursor), comparacion_{H,C,V}_final.png
                (= último cuadro de cada video)
data/output/experimentos/resumen.csv   una fila por caso comparado
data/output/experimentos/informe.html  informe general con videos
```

## Uso

```bash
cd codigo
python -m experimentos.correr --list                       # catálogo
python -m experimentos.correr --grupo A_terminos --dry-run # qué se correría
python -m experimentos.correr --todos --comparar           # todo A–F (38 corridas) + comparación
python -m experimentos.correr --grupo A_terminos B_bc_salida_fem --comparar
python -m experimentos.correr --caso A3_fuente_pozo --backend bfr --force
python -m experimentos.correr --caso A3_fuente_pozo --smoke --comparar  # 2 pasos, prueba
python -m experimentos.comparar --todos [--jobs 3]         # métricas + videos de lo que haya
python -m experimentos.informe                             # informe.html
python -m experimentos.correr --todos --comparar --animar  # + videos y grillas
python -m experimentos.animar --todos [--force] [--jobs 4] # sólo videos de lo que haya
```

No hace falta activar un entorno: `correr.py` lanza BFR en `bfr_env` y FEniCS
en `fenics_env` con `conda run`. Los casos ya corridos se saltan (`--force`
para repetir). El `preproceso.pkl` de cada caso BFR (~50 MB) se borra al
terminar salvo `--keep-pre`.

### Garantías contra sesgos (comparar.py / correr.py)

- `correr.py` **no reutiliza** un resultado si su `config.json` difiere de la
  config actual o si sus nodos no son los de la malla congelada: lo repite.
  (Así se coló en oct 2026 una comparación con resultados de la malla anterior.)
- `comparar.py` rechaza el caso si los nodos BFR ≠ malla congelada, los
  vértices FEM ≠ nodos sin fantasmas, o las config de ambos backends difieren
  en algo que no esté declarado como propio del backend en el catálogo.
- Normas y masa con pesos de área nodal (la malla es 15× más fina en pozo y
  fuente); ΔH en metros; C del pozo = media con la gaussiana del sumidero.
- Video, PNG, métricas e informe salen de los mismos arreglos.
- Paletas de `view.plot.get_colormaps()` (las de los videos de las demás
  variantes): H y |U| turbo, C inferno; |U| lineal saturado en el percentil 99.
  En curvas e informe, BFR en rojo y FEniCS en azul, como en la presentación.

## Catálogo (ADR)

Referencia común: fuente y pozo activos, `Q_std=0.3`, `sigma_fuente=1`,
`r_pozo=20 m` (soporte de los kernels del pozo, independiente de la malla),
`dt=18000 s`, `T=30 días`, `bc_outlet_mode=pfs`, sin upwind.

| Grupo | Casos | Qué aísla |
|---|---|---|
| **A_terminos** | A0 sin fuente ni pozo · A1 sólo fuente · A2 sólo pozo · A3 ambos | Contribución de cada término; A0 = CI transportada |
| **B_bc_salida_fem** | B0–B3 = A0–A3 con BFR `bc_outlet_mode=fem` (sólo BFR; FEM de referencia = A) | Peso de la BC de salida en la diferencia BFR↔FEM |
| **C_soporte_fuente** | `sigma_fuente` ∈ {0.5, 2, 4} × {sin pozo, con pozo} (σ=1 está en A1/A3) | Sensibilidad a lo localizada que está la fuente |
| **D_caudal_pozo** | `Q_std` ∈ {0.01, 0.03, 0.1, 0.7}, sin fuente (0.3 está en A2) | Cono de depresión y velocidades cerca del pozo |
| **E_paso_tiempo** | `dt` ∈ {36000, 7200} s (18000 está en A3) | Error temporal frente a espacial |
| **F_upwind_bfr** | A1 y A3 con BFR `upwind_transport` (sólo BFR) | Efecto de la estabilización upwind |
| **P_presentacion** | adr / mrmt_semi / mrmt_block, `run_type=comparison` | Videos de la presentación (forward + adjunto) |

Para añadir un caso basta una entrada en `catalogo.py`; para un grupo con
MRMT, poner `"model": "mrmt_semi"` en su `cfg`.

### Notas

- `bc_outlet_mode=fem` divergía en una esquina con las mallas anteriores
  (nodos de frontera desplazados hacia dentro); con la malla saneada es
  estable y es la variante BFR que más se parece a FEM (grupo B).
- `upwind_transport` sólo existe en BFR.
- Videos: rangos de color fijos por caso y comunes a BFR y FEM (H y C en
  [mín, máx], |U| lineal hasta el percentil 99); los nodos fantasma BFR se
  excluyen. En B y F el FEM es el de `ref_fem`.
- El recorte C ≥ 0 se aplica en cada paso en ambos backends
  (`docs/clip_positividad_C.md`).

## Presentación (grupo P)

```bash
python -m experimentos.correr --grupo P_presentacion --animar             # corre + videos
python -m experimentos.correr --grupo P_presentacion --animar --publicar  # + copia a avance1-doc/public/videos
```

Los videos usan `scripts/animate_{forward,adjunto,grid}.py` con
`EXP_ROOT=data/output/experimentos/P_presentacion` (sin `EXP_ROOT` esos
scripts leen las rutas antiguas, como antes).

## Sumidero W₋ en un cuadrado (grupo Z, rama exp/sumidero-W)

Pregunta: ¿hace falta el sumidero W₋ = χ(Q) g_ε C para que el soluto salga por
el pozo, o basta la advección? Cuadrado de 600 m con el pozo en el centro
(Q = 0.01, S_s = 10⁻⁵), condición libre ∂ₙ(u·n) = 0 en todo el borde y
h(0) = 300; C₀ = 5.94·10⁻¹¹ (máximo de la CI real) salvo una franja limpia de
60 m junto al borde; 90 días.

```bash
python -m experimentos.correr --grupo Z_cuadrado          # Z1 con W₋, Z0 sin W₋ (BFR)
python -m experimentos.sumidero_cuadrado flujo            # flujo con 3 BC + flujo.html
python -m experimentos.sumidero_cuadrado transporte       # transporte.html (lee Z0/Z1)
python -m experimentos.analisis_sumidero --cuadrado       # cuadrado.png + resumen.txt
```

Salida en `data/output/experimentos/Z_cuadrado/`. Las páginas se arman con las
plantillas de `experimentos/plantillas/` (datos incrustados, se abren sin
servidor).

**Flujo** (`bc_outlet_mode`): en un tramo recto, la EDP junto con la condición
libre da la ecuación de borde S_s ∂ₜH = ∂τ(K_τ ∂τH), que no recibe nada del
interior; con h(0) = 300 el borde queda en 300 para todo t.
- `libre`: fantasma con ∂ₙ(u·n) = 0 y nodo de borde con la ecuación de borde.
  Estable; estacionario en ~6 h; cono de Thiem; borde idéntico con Q = 0 y
  Q = 0.01 (5·10⁻⁸ m). **Es el flujo base del grupo Z.**
- `pfs_nn`: sólo el fantasma. Las plantillas del borde acoplan el pozo con los
  modos neutros {1, x, z, xz}: la carga deriva sin cota (borde ≈ −490 m a 90 d).
- `pfs`: condición programada de la maestría (término cruzado): deriva a −10⁷ m.

**Transporte**: la masa ∫φC baja igual con y sin W₋ (25.4 % y 25.6 % de la
inicial a 90 d). W₋ no saca más soluto: sustituye la salida con el agua y sólo
cambia la C del pozo (agujero C ≈ 0). Pendientes: (1) sin W₋ el nodo del pozo
alterna entre 0 y ~0.65 C₀ de un paso a otro; (2) por la anisotropía (|u| en x
≈ 4.8 × en z) el cuadrado no se limpia en 90 d; (3) la C del borde llega a
0.1 C₀ (sin flujo dispersivo en el borde no se impone C = 0 a la entrada).

Corrección asociada: `funtions/mesh_test._laplacian_smooth` dejaba mover los
nodos de frontera (comparaba con `'outlet'` en vez de `'boundary:outlet'`).

## Sumidero W₋ del pozo (grupo S)

ADR de referencia con `factor_sumidero_C = 1` (S1, modelo anterior) y `= 0`
(S0, W₋ anulado: valor por defecto desde la comparativa del cuadrado de la rama
`exp/sumidero-W`), en BFR y FEniCS.

```bash
python -m experimentos.correr --grupo S_sumidero --comparar
python -m experimentos.grilla_sumidero [--campo C|H|V]   # grilla 2×2 backend × término
```

Salida en `data/output/experimentos/S_sumidero/`: `grilla_C.mp4`,
`grilla_C_final.png`, `resumen.txt`. A 30 d, ∫C con/sin W₋ = 0.9998 (BFR) y
0.9999 (FEM): el término sólo cambia C junto al pozo (C captada 8× menor en BFR
y 4× en FEM con W₋).
