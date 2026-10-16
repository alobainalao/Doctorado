# Malla y Condiciones Iniciales COMPARTIDAS y CENTRALIZADAS — BFR vs FEniCS

## 🔒 REGLAS FUNDAMENTALES

### 1. Malla CENTRALIZADA

```
data/input/bfr/real/adr/preproceso.pkl
│
├─ 16,552 nodos (fijos, generados con default.json)
│
├─ BFR: corre con pre=False → reutiliza estos nodos
│
└─ FEniCS: construye malla DOLFINx desde estos mismos nodos
    (sin Steiner points, mismas coordenadas exactas)
```

### 2. Condiciones Iniciales CENTRALIZADAS

```
data/input/funcion.h5  (dominio real)
│
├─ "h": H0 (carga hidráulica inicial) — nodos en "nodes_G"
├─ "c": C0 (concentración inicial) — nodos en "nodes_Q"
│
├─ BFR: get_init_values() → interpola desde H5
│
└─ FEniCS: get_init_values() → interpola desde MISMO H5
    (ambos usan la misma fuente)
```

**Garantía**: 
- Mismos nodos espaciales (malla)
- Mismos valores iniciales (H0, C0)
- Mismos BCs (frontera)

---

## Casos de uso

### ✅ Desde la app (Streamlit)
```python
# Por defecto, la app SIEMPRE usa:
mesh_path = get_shared_mesh_path()                      # → data/input/bfr/real/adr/preproceso.pkl
h5_path = get_shared_initial_conditions_path()          # → data/input/funcion.h5

# Puedes parametrizar:
# - Parámetros de dinámica (theta, dispersión, difusión)
# - Parámetros de flujo de extracción (Q(t), posición pozo)
# - Parámetros de tiempo (Nt, dt)
# - Parámetros numéricos (epsilon RBF, etc)

# PERO NUNCA CAMBIAN:
# - Malla (16,552 nodos) — ambos backends la usan idéntica
# - CI (H0, C0) — cargadas desde funcion.h5, idénticas para ambos
```

### ✅ Desde scripts de comparación
```bash
# Experimento A
conda run -n bfr_env    python -m scripts.validation.experimentos_pfs_fem bfr_A
conda run -n fenics_env python -m scripts.validation.experimentos_pfs_fem fem_A
conda run -n bfr_env    python -m scripts.validation.experimentos_pfs_fem compare_A
```

Verificación automática:
```
✓ Malla COMPARTIDA verificada: 16552 nodos (fija para BFR + FEniCS)
✓ Mallas idénticas (distancia máxima: 0.00e+00)
```

---

## 🚫 Casos prohibidos

### ❌ NO hagas esto:
1. **No modifiques `default.json`** — es el benchmark (regla 6 de CLAUDE.md)
2. **No corra BFR con `pre=True` en scripts de comparación** — regenera malla y CI
3. **No esperes mallas/CI distintas en ramas** — ambas son GLOBALES
4. **No generes nuevos archivos H5** en experimentos de comparación — reutiliza el existente
5. **No sobrescribas** `funcion.h5` o `preproceso.pkl` con parámetros distintos

### ❌ Si necesitas malla o CI distintas:
1. Crea un **nuevo script separado** (no modifiques `experimentos_pfs_fem.py`)
2. Crea un **nuevo archivo de config** (ej: `custom_config.json`)
3. Genera un **nuevo pkl + nuevo H5 separado** para ese experimento
4. **`experimentos_pfs_fem.py` SIEMPRE usa malla estándar (16,552 nodos) + CI estándar (funcion.h5)**

---

## Inicialización

### Primera vez: crear malla y CI centralizadas

```bash
# Asegurate que default.json está SIN CAMBIOS
conda run -n bfr_env python -m scripts.validation.etapa1_H_sin_pozo bfr

# Genera automáticamente AMBAS cosas:
# → data/input/bfr/real/adr/preproceso.pkl  (16,552 nodos)
# → data/input/funcion.h5  (H0, C0 iniciales)
```

Ambas se usan en TODO a partir de ese momento.

### ¿Dónde se crean las CI?

Las CI (H0, C0) se generan cuando BFR corre una simulación completa:
- BFR interpola H y C en sus nodos
- Las guarda en `funcion.h5` (automático si `save_dat=True`)
- FEniCS lee las CI desde el MISMO archivo

**Verificar que existen:**
```bash
python3 << 'EOF'
import h5py
h5 = h5py.File("data/input/funcion.h5", "r")
print(f"H0: {len(h5['h'])} nodos")
print(f"C0: {len(h5['c'])} nodos")
print(f"Nodos H: {h5['nodes_G'].shape}")
print(f"Nodos C: {h5['nodes_Q'].shape}")
h5.close()
EOF
```

### Después: NUNCA regenerar
```bash
# ✓ CORRECTO: reutiliza la malla
conda run -n bfr_env python -m scripts.validation.experimentos_pfs_fem bfr_A

# ✗ INCORRECTO: regeneraría (no hagas esto)
# (ver scripts/validation/experimentos_pfs_fem.py — pre=False fuerza reutilización)
```

---

## Integración en la app

Desde `view/app.py` o equivalent Streamlit:

```python
from scripts.validation.experimentos_pfs_fem import get_shared_mesh_path, verify_reference_mesh

# Al startup:
try:
    verify_reference_mesh()  # Falla si no existe
except FileNotFoundError:
    st.error("Malla compartida no existe. Ejecuta primero etapa1_H_sin_pozo.")
    return

# Al comparar:
mesh_path = get_shared_mesh_path()
# Pasar a BFR y FEniCS — ambos la usan idéntica
```

---

## Verificación manual

```bash
python3 << 'EOF'
import pickle
import numpy as np

pkl = "data/input/bfr/real/adr/preproceso.pkl"
with open(pkl, "rb") as f:
    data = pickle.load(f)

print(f"Nodos: {len(data['nodes'])}")
print(f"✓ Malla centralizada lista para BFR + FEniCS")
EOF
```

---

## Archivos clave

| Archivo | Rol |
|---------|-----|
| `data/input/bfr/real/adr/preproceso.pkl` | 🔒 Malla compartida (centralizada, fija) |
| `scripts/validation/experimentos_pfs_fem.py` | Verifica malla al inicio, la usa en ambos backends |
| `fenicsx/mesh.py:build_mesh_from_bfr_pkl()` | Construye malla FEniCS desde pkl compartido |

**Resumen**: Una sola malla. Dos backends. Mismo conjunto de nodos. Garantizado.
