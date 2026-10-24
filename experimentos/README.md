# Experimentos BFR (RBF-FD) vs FEniCS (FEM)

Estructura única para definir, correr y comparar casos de estudio. Sustituye a
los lanzadores sueltos que ahora están en `scripts/legacy/`
(`generar_comparacion.py`, `run_all_6_variants.py`, `run_comparison.py`, …).

## Principios

1. **Malla congelada y compartida.** `data/input/malla_compartida/malla.pkl`
   (16 552 nodos, malla de 4 niveles 4/16/30/60 m, `n_stencil=35`). BFR
   reconstruye sus operadores sobre esos nodos exactos en cada caso
   (`mesh_pkl` en `preprocessing/preprocess.py`); FEniCS construye su malla
   P1 con esos mismos nodos (`build_mesh_from_bfr_pkl`). La comparación es
   nodo a nodo, sin interpolación.
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
  comparar.py    UNA pasada por caso: métricas + video de comparación con
                 los mismos arreglos (sin figuras de postproceso aparte)
  informe.py     informe.html general (lee metricas.json + videos; no recalcula)
  conclusiones.html  texto de revisión y conclusiones que inserta informe.py
  animar.py      videos H/C/|U| sueltos por backend (opcional)
  verificar.py   malla congelada (hash) y CI ≥ 0, antes de correr

data/output/experimentos/<grupo>/<caso>/
  bfr/          simulation_results.npz, config.json (la usada), run.log,
                figures/{H,C,V}.mp4
  fenicsx/      ídem
  comparacion/  metricas.json, comparacion.mp4 (filas H/C/|U| × columnas
                BFR / FEM / BFR−FEM + curvas con cursor), comparacion_final.png
                (= último cuadro del video)
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

- `bc_outlet_mode=fem` divergía en el dominio real con la malla anterior
  (informe de la rama `audit/bfr-fenicsx-validation`); el grupo B lo prueba
  con la malla actual.
- `upwind_transport` sólo existe en BFR.
- Videos: rangos de color fijos por caso y comunes a BFR y FEM (H y C en
  [mín, máx], |U| en escala log); los nodos fantasma BFR se excluyen.
  En B y F la columna FEM de la grilla es la de `ref_fem`.
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
