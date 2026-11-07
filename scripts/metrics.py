"""
Métricas de la Práctica 2 (ASR): WER, CER, RTF, tiempo de carga y VRAM pico.

Pensado para importarse desde los notebooks:

    from scripts.metrics import MedidorVRAM, cronometro, evaluar, detectar_entorno

Aviso importante sobre la VRAM
------------------------------
`torch.cuda.max_memory_allocated()` NO ve la memoria que reserva CTranslate2
(el backend de faster-whisper), porque no usa el asignador de PyTorch. Un
equipo que mida así obtendrá 0 MB y no se dará cuenta hasta la defensa.

Aquí se muestrea con NVML sobre el PID del proceso, lo que funciona con
cualquier backend: PyTorch, CTranslate2, ONNX Runtime o NeMo.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import jiwer

from .normalize import normalizar


# --------------------------------------------------------------------------
# Entorno de ejecución
# --------------------------------------------------------------------------

def detectar_entorno() -> str:
    """Devuelve 'nimrod' o 'local' para la columna `entorno` de resultados.csv.

    Los workspaces de Coder en NIMROD exponen OLLAMA_HOST apuntando al bridge
    docker0 del servidor. Es la marca más fiable de estar dentro del servidor,
    porque no depende del hostname del contenedor, que es aleatorio.
    """
    if "172.17.0.1" in os.environ.get("OLLAMA_HOST", ""):
        return "nimrod"
    if os.environ.get("CODER_WORKSPACE_NAME"):
        return "nimrod"
    if socket.gethostname().startswith("nimrod"):
        return "nimrod"
    return "local"


def exigir_nimrod(metrica: str = "RTF/VRAM") -> None:
    """Aborta si se intenta medir rendimiento fuera de NIMROD (guía, 4.2.1)."""
    entorno = detectar_entorno()
    if entorno != "nimrod":
        raise RuntimeError(
            f"{metrica} debe medirse en un workspace de NIMROD, no en '{entorno}'. "
            "El WER y el CER sí podéis calcularlos donde queráis; las columnas de "
            "rendimiento solo son comparables entre equipos si salen del mismo "
            "hardware. Ver apartado 4.2.1 de la guía."
        )


# --------------------------------------------------------------------------
# Cronómetro
# --------------------------------------------------------------------------

def calentar(transcribir, ruta_audio: str, veces: int = 2) -> None:
    """Pasadas de calentamiento DESCARTADAS, antes de cronometrar nada.

    La primera inferencia de un proceso paga el contexto CUDA, la compilación
    de kernels y la lectura en frío de los pesos. Medido en NIMROD
    (21-oct-2026) con faster-whisper small: RTF 1.403 en la primera pasada
    frente a 0.053 en las siguientes. Un factor de 26 que no tiene nada que
    ver con el modelo.

    Sin descartar esas pasadas, cada equipo arrastraría un sesgo distinto
    según el modelo y el orden en que midiera, y las columnas de rendimiento
    dejarían de ser comparables.

        calentar(lambda r: transcribir(modelo, backend, r), rutas[0])
        with cronometro() as t: ...
    """
    for _ in range(veces):
        transcribir(ruta_audio)


@contextmanager
def cronometro():
    """Mide tiempo de pared en segundos.

        with cronometro() as t:
            modelo = cargar()
        print(t.segundos)

    La carga del modelo se mide SIEMPRE por separado de la inferencia: es un
    coste de una sola vez y mezclarlo con el tiempo por enunciado destroza el
    RTF, como ya se vio en P1.
    """
    class _T:
        segundos = 0.0

    t = _T()
    inicio = time.perf_counter()
    try:
        yield t
    finally:
        t.segundos = time.perf_counter() - inicio


# --------------------------------------------------------------------------
# VRAM
# --------------------------------------------------------------------------

def _hay_nvidia_smi() -> bool:
    import shutil

    return shutil.which("nvidia-smi") is not None


class MedidorVRAM:
    """Muestrea la VRAM usada por ESTE proceso mientras dura el bloque.

        with MedidorVRAM() as v:
            transcribir_todo()
        print(v.pico_mb)

    Funciona con cualquier backend porque pregunta al driver, no a PyTorch.
    Fuera de NIMROD o sin GPU devuelve None en lugar de fallar, para que el
    notebook siga corriendo en un portátil.
    """

    def __init__(self, intervalo_s: float = 0.1):
        self.intervalo_s = intervalo_s
        self.pico_mb: float | None = None
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None
        self._nvml = None

    def _leer_mb(self) -> float | None:
        if self._nvml is not None:
            return self._leer_nvml()
        return self._leer_smi()

    def _leer_nvml(self) -> float | None:
        pid = os.getpid()
        try:
            for i in range(self._nvml.nvmlDeviceGetCount()):
                h = self._nvml.nvmlDeviceGetHandleByIndex(i)
                for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(h):
                    if p.pid == pid and p.usedGpuMemory:
                        return p.usedGpuMemory / 1024 / 1024
        except Exception:
            return None
        return None

    def _leer_smi(self) -> float | None:
        """Fallback sin pynvml: parsea nvidia-smi.

        El contenedor de NIMROD trae nvidia-smi por el runtime NVIDIA aunque
        no tenga el paquete de Python, así que esto funciona en la imagen tal
        y como está hoy. Es más lento, de ahí que NVML tenga preferencia.
        """
        import subprocess

        pid = os.getpid()
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return None
            for linea in r.stdout.strip().splitlines():
                partes = [c.strip() for c in linea.split(",")]
                if len(partes) == 2 and partes[0].isdigit() and int(partes[0]) == pid:
                    return float(partes[1])
        except Exception:
            return None
        return None

    def _bucle(self):
        while not self._parar.is_set():
            mb = self._leer_mb()
            if mb is not None:
                self.pico_mb = mb if self.pico_mb is None else max(self.pico_mb, mb)
            self._parar.wait(self.intervalo_s)

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None
            # Sin pynvml probamos nvidia-smi. Si tampoco está, pico_mb se
            # queda en None y la columna del CSV va vacía, como debe.
            if self._leer_smi() is None and not _hay_nvidia_smi():
                return self
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=2)
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return False


# --------------------------------------------------------------------------
# WER / CER
# --------------------------------------------------------------------------

@dataclass
class Resultado:
    """Una fila de resultados.csv."""

    equipo: str
    modelo: str
    backend: str
    precision: str
    condicion: str
    entorno: str
    wer: float
    cer: float
    rtf_medio: float | None = None
    t_carga_s: float | None = None
    vram_pico_mb: float | None = None
    licencia: str = ""

    CAMPOS = (
        "equipo,modelo,backend,precision,condicion,entorno,"
        "wer,cer,rtf_medio,t_carga_s,vram_pico_mb,licencia"
    ).split(",")

    def fila(self) -> dict:
        def fmt(v):
            if v is None:
                return ""
            return round(v, 4) if isinstance(v, float) else v

        return {c: fmt(getattr(self, c)) for c in self.CAMPOS}


def evaluar(referencias: list[str], hipotesis: list[str]) -> tuple[float, float]:
    """Devuelve (WER, CER) sobre el corpus completo, ya normalizado.

    Se calcula sobre el corpus agregado, no promediando WER por enunciado: la
    media de tasas pondera igual una frase de 3 palabras y una de 30, y no es
    la definición estándar.
    """
    if len(referencias) != len(hipotesis):
        raise ValueError(
            f"{len(referencias)} referencias frente a {len(hipotesis)} hipótesis. "
            "Toda la evaluación debe cubrir el manifiesto completo."
        )

    refs = [normalizar(r) for r in referencias]
    hips = [normalizar(h) for h in hipotesis]

    # Un enunciado cuya referencia queda vacía tras normalizar rompe jiwer y
    # no aporta nada: se descarta y se avisa.
    pares = [(r, h) for r, h in zip(refs, hips) if r.strip()]
    if len(pares) < len(refs):
        print(f"[aviso] {len(refs) - len(pares)} referencias vacías descartadas")

    refs = [r for r, _ in pares]
    hips = [h for _, h in pares]

    return jiwer.wer(refs, hips), jiwer.cer(refs, hips)


def guardar_csv(resultados: list[Resultado], ruta: str = "results/resultados.csv") -> None:
    """Escribe resultados.csv con el esquema fijo de la guía (6.1)."""
    import csv
    import pathlib

    p = pathlib.Path(ruta)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=Resultado.CAMPOS)
        w.writeheader()
        for r in resultados:
            w.writerow(r.fila())
    print(f"{len(resultados)} filas escritas en {p}")
