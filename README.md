# Simulador ADR / MRMT

Simulación numérica de flujo y transporte en medios porosos (contaminación en
acuíferos), con:

- ADR clásico
- MRMT (multi-rate mass transfer)
- Control óptimo por método adjunto

## Características

- Discretización espacial con RBF-FD
- Interfaz web con Streamlit
- Comparación de curvas de llegada (BTCs)

## Estructura

- `preprocessing/`: generación de matrices y malla
- `pipeline/`: ejecución de la simulación
- `funtions/`: operadores matemáticos y runtime
- `view/`: visualización y animación
- `io/`: carga/guardado
- `postprocessing/`: curvas de llegada (BTCs)
- `config/`: parámetros (`default.json`) y geometría
- `scripts/`: variantes usadas para figuras/datos de la tesis

## Uso

CLI (parámetros desde `config/default.json`):

```bash
python run.py
```

Interfaz web:

```bash
conda activate bfr_env
streamlit run app.py
```

`run_type` en la configuración selecciona `standard` (simulación directa) u
`optimization` (control óptimo adjunto).
