# Parámetros del caso nitrato (`config/nitrato.json`)

| Parámetro | Valor | Origen |
|---|---|---|
| Contaminante | nitrato (NO₃⁻) | elegido en la tesis |
| Valor de referencia $C_{ref}$ | 0.05 kg/m³ (50 mg/L como nitrato) | guía OMS, 4.ª ed. (tabla de sustancias químicas de relevancia para la salud) |
| Referencia alternativa | 10 mg/L como N | EPA, MCL del nitrato |
| Concentración de la fuente | 42 mg/L como N × 62/14 = 0.1859 kg/m³ de nitrato | nivel de contaminación agrícola citado en la literatura (Spalding & Exner 1993) |
| Retardo | $K_d=0$ ⇒ $R=1$ | el nitrato es conservativo, no se adsorbe |
| Decaimiento $\lambda_d$ | 0.02 /día = 2.3148e-7 s⁻¹ | desnitrificación lenta en acuífero (orden de magnitud, no calibrado) |
| `Q_std` | 1e-3 | como `main.py`; $Q_{max}=0.01$ |

**Escalado de la fuente.** El modelo es lineal en $C$ (salvo el recorte $C\ge0$). Escalar la fuente por $f$ equivale a usar $C_{ref,modelo}=C_{ref}/f$. Con $f=C_{fuente}/\text{pico}(C,\,Q=0,\,\text{ADR})=7.3663\times10^{8}$, el mismo para los tres modelos (se guarda en `data/output/optimizacion_nitrato/f_fuente.json`).

**Normalización de $J$ (`normalizar_J`).** $w_1=1/(C_{ref}^2T)$, $w_2=1/\max|z_p-z_0|^2$ sobre las cotas, $w_3=1/(Q_{max}^2\,T\,\max|z_p-z_0|)$; $J_{scale}=1$.

**Pendiente de verificar:** las fuentes de $\lambda_d$ y de los 42 mg/L salieron de una búsqueda web puntual; falta contrastarlas con la referencia primaria y citarlas en `shared/referencias.bib`.
Fuentes consultadas: WHO 4th ed. tables (ESdat); Table A3.3 (foodlegal); Oklahoma DEQ, Inorganic Chemicals (MCL EPA); Spalding & Exner 1993.
