"""
Utilidades de entorno para la Práctica 2 (ASR).

Hereda de P1 las dos soluciones que más tiempo ahorraron:
  - `sin_ruido()`, para silenciar el vertido de stderr de ONNX Runtime
    (y ahora también de CTranslate2);
  - lectura de la memoria real del contenedor, porque `free` informa de la
    del host.

Añade la verificación de entorno que conviene ejecutar en la primera celda
de cada notebook, antes de cargar ningún modelo.
"""

from __future__ import annotations

import csv
import logging
import os
import pathlib
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Silenciar stderr
# --------------------------------------------------------------------------

@contextmanager
def sin_ruido():
    """Silencia stderr a nivel de descriptor de fichero.

    ONNX Runtime y CTranslate2 escriben desde C++, así que redirigir
    `sys.stderr` desde Python no sirve de nada: hay que redirigir el fd 2.

        with sin_ruido():
            modelo = WhisperModel("large-v3", device="cuda", compute_type="float16")

    OJO: esto NO atrapa los avisos del módulo `logging` de Python. Dentro de
    Jupyter, `sys.stderr` es un objeto del kernel que no pasa por el fd 2.
    Para esos está `silenciar_avisos_cache()`.
    """
    try:
        fd_err = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # Algunos kernels no exponen un descriptor real; sin él no hay nada
        # que redirigir, pero tampoco motivo para romper.
        yield
        return

    copia = os.dup(fd_err)
    nulo = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(nulo, fd_err)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(copia, fd_err)
        os.close(copia)
        os.close(nulo)


# Avisos que Hugging Face emite al no poder escribir metadatos en la caché
# compartida. Son esperados —el montaje es de sólo lectura a propósito— y la
# propia biblioteca los ignora y continúa, pero llenan la salida del cuaderno.
_RUIDO_CACHE = (
    "Read-only file system",
    "Could not cache non-existence",
    "Ignored error while writing commit hash",
)

_LOGGERS_RUIDOSOS = ("huggingface_hub", "transformers", "filelock", "datasets")

_filtro_instalado = False


class _FiltroCacheSoloLectura(logging.Filter):
    """Descarta sólo los avisos de escritura en la caché de sólo lectura.

    Es deliberadamente quirúrgico: filtra por el texto del mensaje en vez de
    subir el nivel del logger entero, para que cualquier otro error siga
    viéndose. Silenciar un logger completo es cómodo hasta el día en que tapa
    el fallo que llevas una hora buscando.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            mensaje = record.getMessage()
        except Exception:
            return True
        return not any(p in mensaje for p in _RUIDO_CACHE)


def silenciar_avisos_cache() -> None:
    """Quita de la salida los avisos de caché de sólo lectura. Idempotente.

    Llamadlo una vez, en la celda de arranque. No silencia ningún otro
    mensaje: si algo falla de verdad, lo seguiréis viendo.
    """
    global _filtro_instalado
    if _filtro_instalado:
        return
    filtro = _FiltroCacheSoloLectura()
    for nombre in _LOGGERS_RUIDOSOS:
        logging.getLogger(nombre).addFilter(filtro)
    _filtro_instalado = True


# --------------------------------------------------------------------------
# Dispositivo de cómputo
# --------------------------------------------------------------------------

def dispositivo(precision_gpu: str = "float16") -> tuple[str, str]:
    """Devuelve (device, compute_type) adecuados a la máquina donde corréis.

        device, precision = dispositivo()
        modelo = WhisperModel("small", device=device, compute_type=precision)

    En NIMROD devuelve ("cuda", "float16"). En un portátil sin GPU NVIDIA,
    ("cpu", "int8"), que es varias veces más rápido que float32 en CPU y la
    diferencia de calidad es despreciable para el baseline.

    Por qué NO se fija "cuda" a mano en el cuaderno: la GPU de NIMROD es
    Blackwell (sm_120) y exige float16 —int8 falla—, mientras que en CPU
    float16 no está soportado. Son combinaciones incompatibles, así que la
    elección no puede quedar escrita en una celda que se ejecuta en los dos
    sitios.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", precision_gpu
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", precision_gpu
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        import subprocess

        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return "cuda", precision_gpu

    return "cpu", "int8"


# --------------------------------------------------------------------------
# Memoria real del contenedor
# --------------------------------------------------------------------------

def memoria_limite_gb() -> float | None:
    """Memoria máxima del contenedor en GB, leída de cgroups v2.

    `free` dentro del workspace informa de los 270 GB del host, no de los
    32 GB del contenedor. Si veis 270 GB, estáis mirando el dato equivocado.
    Devuelve None si no hay límite (ejecución fuera de contenedor).
    """
    for ruta in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = pathlib.Path(ruta)
        if not p.exists():
            continue
        valor = p.read_text().strip()
        if valor in ("max", ""):
            return None
        try:
            n = int(valor)
        except ValueError:
            continue
        # Sin límite, cgroups devuelve un número absurdamente grande.
        if n > 2**60:
            return None
        return n / 1024**3
    return None


def cpus_limite() -> float | None:
    """Núcleos de CPU asignados al contenedor, leídos de cgroups v2."""
    p = pathlib.Path("/sys/fs/cgroup/cpu.max")
    if not p.exists():
        return None
    partes = p.read_text().split()
    if len(partes) != 2 or partes[0] == "max":
        return None
    return int(partes[0]) / int(partes[1])


# --------------------------------------------------------------------------
# Verificación de entorno
# --------------------------------------------------------------------------
# Verificación del entorno
# --------------------------------------------------------------------------
# Vivía aquí hasta el 28-oct-2026, duplicada con comprobar_entorno.py y ya
# desfasada (buscaba la caché en /data1 y el conjunto en una ruta relativa).
# Ahora hay una sola implementación, en scripts/entorno.py. Se reexporta para
# que el código antiguo siga funcionando.

from .entorno import Comprobacion, verificar_entorno  # noqa: E402,F401


# --------------------------------------------------------------------------
# Manifiesto
# --------------------------------------------------------------------------

@dataclass
class Enunciado:
    id: str
    ruta: str
    duracion_s: float
    bloque: str
    transcripcion_referencia: str


def cargar_manifiesto(ruta: str = "eval_set/manifest.tsv") -> list[Enunciado]:
    """Carga el conjunto común congelado. No lo filtréis ni lo recortéis."""
    p = pathlib.Path(ruta)
    if not p.exists():
        raise FileNotFoundError(
            f"No se encuentra {p}. El conjunto de evaluación se distribuye ya "
            "materializado; no se descarga desde el Hub."
        )
    with p.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f, delimiter="\t"))
    return [
        Enunciado(
            id=r["id"],
            ruta=r["ruta"],
            duracion_s=float(r["duracion_s"]),
            bloque=r["bloque"],
            transcripcion_referencia=r["transcripcion_referencia"],
        )
        for r in filas
    ]


def duracion_total_s(enunciados: list[Enunciado]) -> float:
    """Denominador del RTF: duración total del audio evaluado."""
    return sum(e.duracion_s for e in enunciados)


if __name__ == "__main__":
    verificar_entorno()
