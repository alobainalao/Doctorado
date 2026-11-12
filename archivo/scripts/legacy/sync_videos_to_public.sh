#!/bin/bash
# Copia todos los videos regenerados desde data/output/*/figures/real/*/
# hacia presentaciones/avance1-doc/public/videos/
# Uso: bash scripts/sync_videos_to_public.sh

set -e
PRESENTACION="/home/alex/dev/doctorado/presentaciones/avance1-doc/public/videos"

echo "=== Sincronizando videos hacia public/ ==="

for metodo in bfr fenicsx; do
    for model in adr mrmt_semi mrmt_block; do
        SRC="./data/output/${metodo}/figures/real/${model}"
        DST="${PRESENTACION}/${metodo}/${model}"

        if [ ! -d "$SRC" ]; then
            echo "  [SKIP] $SRC no existe"
            continue
        fi

        mkdir -p "$DST"

        # Videos forward
        for video in H.mp4 C.mp4 V.mp4; do
            if [ -f "${SRC}/${video}" ]; then
                if ffprobe -v error "${SRC}/${video}" 2>/dev/null; then
                    cp "${SRC}/${video}" "${DST}/${video}"
                    size=$(du -h "${DST}/${video}" | cut -f1)
                    echo "  ✓ ${metodo}/${model}/${video}  ($size)"
                else
                    echo "  ✗ ${metodo}/${model}/${video}  CORRUPTO — omitido"
                fi
            fi
        done
        # Videos adjunto: animate_adjunto.py genera adjunto_psi_{H,C}.mp4
        for pair in "adj_H.mp4:adjunto_psi_H.mp4" "adj_C.mp4:adjunto_psi_C.mp4"; do
            dst_name="${pair%%:*}"
            src_name="${pair##*:}"
            if [ -f "${SRC}/${src_name}" ]; then
                if ffprobe -v error "${SRC}/${src_name}" 2>/dev/null; then
                    cp "${SRC}/${src_name}" "${DST}/${dst_name}"
                    size=$(du -h "${DST}/${dst_name}" | cut -f1)
                    echo "  ✓ ${metodo}/${model}/${dst_name}  ($size)"
                else
                    echo "  ✗ ${metodo}/${model}/${dst_name}  CORRUPTO — omitido"
                fi
            fi
        done
    done
done

echo "=== Listo ==="
