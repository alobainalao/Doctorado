#!/bin/bash
# Uso: barrido.sh "<etiqueta> <delta> <capas> <modo>" ...   (la etiqueta "oficial" usa malla.pkl sin generar)
# Resultados: data/output/experimentos/malla_borde/resultados.log (una línea RESULT por malla)
cd "$(dirname "$0")/../.."
R=data/output/experimentos/malla_borde; mkdir -p $R/mallas
for cfg in "$@"; do
  set -- $cfg; tag=$1
  if [ "$tag" = oficial ]; then pkl=data/input/malla_compartida/malla.pkl
  else pkl=$R/mallas/m_$tag.pkl
       [ -f $pkl ] || conda run -n bfr_env python -m experimentos.malla_borde.gen_malla $2 $3 $4 $pkl > $R/mallas/gen_$tag.log 2>&1
  fi
  conda run -n bfr_env python -m experimentos.malla_borde.prueba_malla $tag $pkl 2>&1 | grep -E "RESULT|Error|Traceback" >> $R/resultados.log
done
echo FIN >> $R/resultados.log
