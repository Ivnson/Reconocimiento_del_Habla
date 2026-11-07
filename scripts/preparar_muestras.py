#!/usr/bin/env python3
"""
Extrae una muestra pequeña del conjunto común para el directorio `eval_local/`.

Lo ejecuta EL PROFESOR una vez, en NIMROD, antes de publicar el repositorio:

    python3 -I scripts/preparar_muestras.py \
        --eval-set /data2/habla/datasets/material-p2/eval_set \
        --salida eval_local --n 12

El resultado se commitea al repositorio plantilla. Sirve para que un equipo
pueda ejecutar el cuaderno en su portátil sin acceso a NIMROD: las celdas
usan `eval_local/` cuando `P2_EVAL_SET` no está definida.

OJO: una muestra de 12 enunciados NO es el conjunto de evaluación. Los
números que se entregan en `resultados.csv` salen del conjunto común
completo; `eval_local/` sólo sirve para que el código arranque y se depure.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--salida", default="eval_local")
    ap.add_argument("--n", type=int, default=12,
                    help="enunciados por bloque a copiar (reparto equitativo)")
    args = ap.parse_args()

    origen = pathlib.Path(args.eval_set)
    manifiesto = origen / "manifest.tsv"
    if not manifiesto.exists():
        print(f"No encuentro {manifiesto}", file=sys.stderr)
        return 1

    with manifiesto.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f, delimiter="\t"))

    # Reparto equitativo entre bloques, tomando los primeros de cada uno para
    # que la selección sea reproducible.
    bloques: dict[str, list[dict]] = {}
    for fila in filas:
        bloques.setdefault(fila["bloque"], []).append(fila)

    por_bloque = max(1, args.n // len(bloques))
    elegidas = [fila for b in sorted(bloques) for fila in bloques[b][:por_bloque]]

    destino = pathlib.Path(args.salida)
    (destino / "audio").mkdir(parents=True, exist_ok=True)

    for fila in elegidas:
        shutil.copy2(origen / fila["ruta"], destino / fila["ruta"])

    with (destino / "manifest.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, delimiter="\t",
            fieldnames=["id", "ruta", "duracion_s", "bloque", "transcripcion_referencia"],
        )
        w.writeheader()
        w.writerows(elegidas)

    total = sum(float(r["duracion_s"]) for r in elegidas)
    print(f"{len(elegidas)} enunciados ({total:.0f} s) en {destino}")
    for b in sorted(bloques):
        print(f"  {b}: {sum(1 for r in elegidas if r['bloque'] == b)}")

    # La procedencia viaja con la muestra: son corpus con licencia que exige
    # atribución y el repositorio es público para los alumnos.
    fuentes = origen / "FUENTES.md"
    if fuentes.exists():
        shutil.copy2(fuentes, destino / "FUENTES.md")
        print(f"  FUENTES.md copiado")
    else:
        print("  AVISO: no encuentro FUENTES.md en el conjunto de origen")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
