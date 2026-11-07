#!/usr/bin/env bash
#
# Despliega el material de la Práctica 2 en el espacio de trabajo de cada equipo.
#
# Lo ejecuta EL PROFESOR en NIMROD (como azken), no los alumnos:
#
#     sudo bash desplegar_p2.sh /data1/builds/p2/plantilla
#
# Sigue la convención de P1: el material vive en el directorio ESCRIBIBLE del
# equipo (/data2/habla/equipos/<owner>/), que el workspace ve montado en
# /data/equipo y que el alumno alcanza por el enlace ~/equipo.
#
#     ~/equipo/p1-tts/     <- Práctica 1
#     ~/equipo/p2-asr/     <- Práctica 2  (lo que crea este script)
#
# NO sobreescribe el trabajo de un equipo que ya haya empezado: si el destino
# existe, lo salta y lo dice. Con --forzar hace copia de seguridad del anterior
# antes de reemplazarlo.

set -euo pipefail

ORIGEN="${1:-}"
EQUIPOS_DIR="/data2/habla/equipos"
DESTINO_REL="p2-asr"
UID_WS=1000
GID_WS=1000
FORZAR=0

for arg in "$@"; do
    [ "$arg" = "--forzar" ] && FORZAR=1
done

if [ -z "$ORIGEN" ] || [ ! -d "$ORIGEN" ]; then
    echo "uso: sudo bash desplegar_p2.sh <directorio-plantilla> [--forzar]" >&2
    echo "     el directorio debe contener p2_plantilla.ipynb y scripts/" >&2
    exit 1
fi

if [ ! -f "$ORIGEN/p2_plantilla.ipynb" ] || [ ! -d "$ORIGEN/scripts" ]; then
    echo "ERROR: $ORIGEN no parece la plantilla de P2" >&2
    echo "       faltan p2_plantilla.ipynb y/o scripts/" >&2
    exit 1
fi

if [ ! -d "$EQUIPOS_DIR" ]; then
    echo "ERROR: no existe $EQUIPOS_DIR" >&2
    exit 1
fi

echo "Origen : $ORIGEN"
echo "Destino: $EQUIPOS_DIR/<equipo>/$DESTINO_REL"
echo

copiados=0
saltados=0

for dir_equipo in "$EQUIPOS_DIR"/*/; do
    [ -d "$dir_equipo" ] || continue
    equipo="$(basename "$dir_equipo")"
    destino="${dir_equipo}${DESTINO_REL}"

    if [ -e "$destino" ]; then
        if [ "$FORZAR" -eq 1 ]; then
            respaldo="${destino}.bak-$(date +%Y%m%d-%H%M%S)"
            mv "$destino" "$respaldo"
            echo "  $equipo: anterior guardado en $(basename "$respaldo")"
        else
            echo "  $equipo: YA EXISTE, se salta (--forzar para reemplazar)"
            saltados=$((saltados + 1))
            continue
        fi
    fi

    mkdir -p "$destino"
    # --no-target-directory evitaría anidar; copiamos el contenido, no la carpeta.
    cp -r "$ORIGEN"/. "$destino"/

    # Fuera lo que no tiene sentido en el espacio del equipo:
    #   .git/.github   -> sólo valen en el repositorio
    #   eval_local/    -> en NIMROD se usa el conjunto común vía P2_EVAL_SET
    #   scripts del profesor -> despliegue y preparación de muestras
    rm -rf "$destino/.git" "$destino/.github" "$destino/.gitignore" \
           "$destino/__pycache__" "$destino/scripts/__pycache__" \
           "$destino/eval_local" \
           "$destino/scripts/desplegar_p2.sh" \
           "$destino/scripts/preparar_muestras.py"

    # El workspace corre como UID 1000; sin esto el equipo no puede escribir.
    chown -R "$UID_WS:$GID_WS" "$destino"
    find "$destino" -type d -exec chmod 755 {} +
    find "$destino" -type f -exec chmod 644 {} +

    echo "  $equipo: desplegado"
    copiados=$((copiados + 1))
done

echo
echo "$copiados desplegado(s), $saltados saltado(s)."
echo
echo "Comprobad uno antes de avisar a los alumnos:"
echo "  coder ssh <workspace>"
echo "  cd ~/equipo/$DESTINO_REL && python3 comprobar_entorno.py"
