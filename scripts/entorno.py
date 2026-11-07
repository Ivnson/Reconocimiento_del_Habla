"""
Comprobación del entorno — Práctica 2 (ASR).

Única fuente de verdad: la usan tanto `comprobar_entorno.py` (desde el
terminal) como la primera celda del cuaderno. No dupliquéis estas
comprobaciones en otro sitio: hasta el 28-oct-2026 había dos versiones y la
del cuaderno daba falsos FALLO en la caché y en el conjunto de evaluación.

    from scripts.entorno import verificar_entorno
    verificar_entorno()
"""

from __future__ import annotations

import importlib
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

OK, FALLO, AVISO = "OK", "FALLO", "AVISO"


@dataclass
class Comprobacion:
    estado: str
    nombre: str
    detalle: str

    @property
    def ok(self) -> bool:
        return self.estado != FALLO


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------

def _entorno_ejecucion(res: list[Comprobacion]) -> str:
    from .metrics import detectar_entorno

    entorno = detectar_entorno()
    if entorno == "nimrod":
        res.append(Comprobacion(OK, "entorno", "NIMROD — podéis medir RTF, t_carga y VRAM"))
    else:
        res.append(Comprobacion(OK, "entorno", "local — WER y CER válidos; "
                                               "RTF y VRAM hay que medirlos en NIMROD"))
    return entorno


def _python(res: list[Comprobacion]) -> None:
    v = sys.version_info
    if v >= (3, 10):
        res.append(Comprobacion(OK, "python", f"{v.major}.{v.minor}.{v.micro}"))
    else:
        res.append(Comprobacion(FALLO, "python",
                                f"{v.major}.{v.minor} — se necesita 3.10 o superior"))


def _paquetes(res: list[Comprobacion]) -> None:
    requeridos = {
        "jiwer": "métricas WER y CER",
        "num2words": "normalización de numerales",
        "soundfile": "lectura y escritura de audio",
        "numpy": "cálculo numérico",
    }
    opcionales = {
        "transformers": "backend CTC (wav2vec2, MMS)",
        "torch": "requerido por transformers",
        "pynvml": "medida de VRAM",
        "librosa": "análisis de audio",
        "pandas": "tablas de resultados",
    }

    faltan = [f"{m} ({d})" for m, d in requeridos.items() if not _importable(m)]
    if faltan:
        res.append(Comprobacion(FALLO, "paquetes", "faltan: " + ", ".join(faltan)
                                + " — pip install -r requirements.txt"))
    else:
        res.append(Comprobacion(OK, "paquetes", "todos los requeridos presentes"))

    # faster-whisper NO es opcional: el cuaderno lo usa desde la sesión 1.
    # Antes figuraba entre los opcionales y la comprobación decía «entorno
    # listo» mientras el cuaderno se estrellaba en la celda del baseline.
    if _importable("faster_whisper"):
        res.append(Comprobacion(OK, "backend", "faster-whisper disponible"))
    else:
        res.append(Comprobacion(FALLO, "backend",
                                "falta faster-whisper, que el cuaderno usa desde la "
                                "sesión 1 — pip install faster-whisper"))

    ausentes = [f"{m} ({d})" for m, d in opcionales.items() if not _importable(m)]
    if ausentes:
        res.append(Comprobacion(AVISO, "opcionales", "no instalados: " + ", ".join(ausentes)))


def _ffmpeg(res: list[Comprobacion]) -> None:
    ff = shutil.which("ffmpeg")
    if ff:
        res.append(Comprobacion(OK, "ffmpeg", ff))
    else:
        res.append(Comprobacion(FALLO, "ffmpeg",
                                "NO ENCONTRADO — necesario para la condición `codec` y "
                                "para leer audio que no sea WAV. "
                                "apt install ffmpeg / brew install ffmpeg"))


def _dispositivo(res: list[Comprobacion]) -> None:
    from .asr_util import dispositivo

    dev, prec = dispositivo()
    if dev == "cuda":
        res.append(Comprobacion(OK, "dispositivo", f'cuda / compute_type="{prec}"'))
    else:
        res.append(Comprobacion(OK, "dispositivo",
                                f'cpu / compute_type="{prec}" — funciona, pero '
                                "una pasada completa puede tardar horas"))


def _gpu(res: list[Comprobacion]) -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                res.append(Comprobacion(AVISO, "gpu", "sin GPU — sólo modelos de CPU"))
                return
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            nombre = pynvml.nvmlDeviceGetName(h)
            nombre = nombre.decode() if isinstance(nombre, bytes) else nombre
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            mayor, menor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            detalle = f"{nombre} — {mem.total / 1024**3:.1f} GB, sm_{mayor}{menor}"
            if mayor >= 12:
                detalle += " — Blackwell: compute_type='float16', NUNCA 'int8'"
            res.append(Comprobacion(OK, "gpu", detalle))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        if shutil.which("nvidia-smi"):
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                "--format=csv,noheader"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                res.append(Comprobacion(OK, "gpu", r.stdout.strip().splitlines()[0]
                                        + " (vía nvidia-smi; sin pynvml)"))
                return
        res.append(Comprobacion(AVISO, "gpu", "sin GPU detectada — sólo modelos de CPU"))


def _cache(res: list[Comprobacion], entorno: str) -> None:
    cache = os.environ.get("HF_HUB_CACHE", "")
    home = os.environ.get("HF_HOME", "")

    if entorno != "nimrod":
        res.append(Comprobacion(OK, "cache_hf", cache or home or "(por defecto, local)"))
        return

    # No se comprueba la RUTA concreta —cambió de /data1 a /data2 y volverá a
    # cambiar— sino lo que de verdad importa: que exista, que tenga modelos y
    # que HF_HOME sea escribible y esté fuera de ella.
    if not cache:
        res.append(Comprobacion(FALLO, "cache_hf",
                                "HF_HUB_CACHE sin definir — reiniciad el workspace "
                                "para recoger la plantilla actual"))
        return
    if not pathlib.Path(cache).is_dir():
        res.append(Comprobacion(FALLO, "cache_hf", f"{cache} no existe"))
        return
    if home and os.path.abspath(home).startswith(os.path.abspath(cache)):
        res.append(Comprobacion(FALLO, "cache_hf",
                                "HF_HOME está dentro de HF_HUB_CACHE, que es de sólo "
                                "lectura — así falló la primera sesión de P1"))
        return
    try:
        p = pathlib.Path(home)
        p.mkdir(parents=True, exist_ok=True)
        (p / ".escritura").touch()
        (p / ".escritura").unlink()
    except OSError as e:
        res.append(Comprobacion(FALLO, "cache_hf", f"HF_HOME no es escribible ({e.strerror})"))
        return

    modelos = sorted(d.name.replace("models--", "").replace("--", "/")
                     for d in pathlib.Path(cache).glob("models--*"))
    if not modelos:
        res.append(Comprobacion(AVISO, "cache_hf", f"{cache} está vacía"))
        return
    res.append(Comprobacion(OK, "cache_hf", f"{len(modelos)} modelos: "
                            + ", ".join(modelos[:3]) + (" …" if len(modelos) > 3 else "")))


def _eval_set(res: list[Comprobacion]) -> None:
    # En NIMROD la ruta llega por P2_EVAL_SET; en local se usa eval_local/.
    ruta = os.environ.get("P2_EVAL_SET") or "eval_local"
    manifiesto = pathlib.Path(ruta) / "manifest.tsv"

    if not manifiesto.exists():
        res.append(Comprobacion(FALLO, "eval_set",
                                f"no encuentro {manifiesto}. En NIMROD lo indica "
                                "P2_EVAL_SET; en local, usad eval_local/"))
        return
    try:
        from .asr_util import cargar_manifiesto, duracion_total_s

        enunciados = cargar_manifiesto(str(manifiesto))
        bloques = sorted({e.bloque for e in enunciados})
        res.append(Comprobacion(OK, "eval_set",
                                f"{len(enunciados)} enunciados, "
                                f"{duracion_total_s(enunciados)/60:.1f} min, "
                                f"bloques: {', '.join(bloques)}"))
    except Exception as e:
        res.append(Comprobacion(FALLO, "eval_set", f"{type(e).__name__}: {e}"))


def _limites(res: list[Comprobacion]) -> None:
    from .asr_util import cpus_limite, memoria_limite_gb

    mem, cpu = memoria_limite_gb(), cpus_limite()
    partes = []
    if mem:
        partes.append(f"{mem:.0f} GB RAM")
    if cpu:
        partes.append(f"{cpu:.0f} CPU")
    res.append(Comprobacion(OK, "límites",
                            " / ".join(partes) + "  (`free` informa del host, no de esto)"
                            if partes else "sin límite de cgroup"))


# ---------------------------------------------------------------------------

def verificar_entorno(verbose: bool = True) -> list[Comprobacion]:
    """Comprueba que todo está en su sitio ANTES de cargar ningún modelo.

    Devuelve la lista de comprobaciones. `c.ok` es False sólo en los FALLO:
    los AVISO no bloquean nada.
    """
    res: list[Comprobacion] = []
    entorno = _entorno_ejecucion(res)
    _python(res)
    _paquetes(res)
    _ffmpeg(res)
    _gpu(res)
    _dispositivo(res)
    _cache(res, entorno)
    _eval_set(res)
    _limites(res)

    if verbose:
        ancho = max(len(c.nombre) for c in res)
        for c in res:
            print(f"  {c.estado:<6} {c.nombre:<{ancho}}  {c.detalle}")
        fallos = sum(1 for c in res if c.estado == FALLO)
        print()
        if fallos:
            print(f"{fallos} comprobación(es) en FALLO. Resolvedlas antes de seguir.")
        else:
            print("Entorno listo.")
            if entorno != "nimrod":
                print("Recordad: RTF, tiempo de carga y VRAM se miden en NIMROD.")

    return res
