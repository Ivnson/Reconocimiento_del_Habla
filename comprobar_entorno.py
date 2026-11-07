#!/usr/bin/env python3
"""
Comprobación del entorno — Práctica 2 (ASR).

Ejecutadlo ANTES de cargar ningún modelo, tanto en NIMROD como en vuestra
máquina:

    python3 comprobar_entorno.py

Es el mismo código que ejecuta la primera celda del cuaderno
(`scripts/entorno.py`), para que no puedan dar respuestas distintas.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scripts.entorno import FALLO, verificar_entorno  # noqa: E402


def main() -> int:
    print("Comprobación del entorno — Práctica 2 (ASR)\n")
    res = verificar_entorno(verbose=True)
    return 1 if any(c.estado == FALLO for c in res) else 0


if __name__ == "__main__":
    raise SystemExit(main())
