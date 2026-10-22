#!/bin/bash
# Ejecutar DESPUÉS de que los re-runs FEniCS y BFR animate terminen.
# 1. Corre animate_forward.py para los 3 modelos FEniCS (usa fenics_env)
# 2. Corre sync_videos_to_public.sh para copiar todos los videos válidos
#
# Uso: conda activate bfr_env && bash scripts/finalize_all_videos.sh

set -e
echo "=== Generando videos FEniCS forward ==="

for model in adr mrmt_semi mrmt_block; do
    echo "--- FEniCS $model ---"
    conda run -n fenics_env python scripts/animate_forward.py "$model" fenicsx
done

echo ""
echo "=== Copiando videos a public/ ==="
bash scripts/sync_videos_to_public.sh

echo ""
echo "=== Todo listo ==="
