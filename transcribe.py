"""
Entregable de la Práctica 2: interfaz estable de transcripción.

Contrato:

    transcribe(ruta_audio: str) -> str

  - devuelve la transcripción en texto plano, SIN normalizar
    (la normalización la aplica `scripts/normalize.py` al medir);
  - acepta WAV de 16 kHz mono;
  - carga el modelo UNA SOLA VEZ y lo cachea a nivel de módulo.

La carga perezosa es parte del contrato: P3 llamará a esta función muchas
veces dentro de un bucle de diálogo y no puede recargar el modelo en cada
llamada.

EQUIPOS: rellenad la configuración de abajo y escribid el cuerpo de
`_cargar_modelo` y `_transcribir` para vuestro backend. El resto del fichero
no hace falta tocarlo.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import wave

from scripts.asr_util import dispositivo, sin_ruido

# ---------------------------------------------------------------------------
# Configuración del equipo
# ---------------------------------------------------------------------------
# Elección provisional hasta cerrar las medidas de la sesión 3. Para cambiarla,
# basta con tocar estas cinco líneas: las tres familias están implementadas más
# abajo y el resto del fichero no depende de cuál esté activa.
#
#   encoder-decoder  MODELO="large-v3-turbo"  BACKEND="faster-whisper"  MIT
#   CTC              MODELO="facebook/wav2vec2-large-xlsr-53-spanish"
#                    BACKEND="transformers"   PRECISION="float32"       Apache-2.0
#   híbrido TDNN-HMM MODELO="vosk-model-es-0.42"  BACKEND="vosk"        Apache-2.0

MODELO = "large-v3-turbo"
BACKEND = "faster-whisper"       # faster-whisper | transformers | vosk
PRECISION = "float16"            # en NIMROD (sm_120) NUNCA int8 con faster-whisper
LICENCIA = "MIT"
IDIOMA = "es"

# Dónde buscar el modelo de Vosk, que no está en la caché compartida de NIMROD.
DIR_MODELOS = pathlib.Path(__file__).resolve().parent / "modelos"


@functools.lru_cache(maxsize=1)
def _cargar_modelo():
    """Carga el modelo una sola vez. Devuelve el objeto que use vuestro backend."""
    if BACKEND == "faster-whisper":
        from faster_whisper import WhisperModel

        if PRECISION == "int8" and _en_blackwell():
            raise RuntimeError(
                "compute_type='int8' falla en la GPU de NIMROD (Blackwell, sm_120). "
                "Usad 'float16'."
            )

        # El dispositivo no se escribe a mano: en NIMROD hay GPU y hace falta
        # float16; en un portátil sin GPU, float16 ni siquiera existe. P3 va a
        # importar esto desde donde sea, así que tiene que funcionar en ambos.
        device, precision_auto = dispositivo(PRECISION)

        # El silenciado es necesario: CTranslate2 vierte avisos desde C++ y
        # tapa la salida útil del cuaderno.
        with sin_ruido():
            return WhisperModel(MODELO, device=device, compute_type=precision_auto)

    if BACKEND == "transformers":
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        device, _ = dispositivo(PRECISION)
        with sin_ruido():
            procesador = AutoProcessor.from_pretrained(MODELO)
            red = Wav2Vec2ForCTC.from_pretrained(
                MODELO,
                dtype=torch.float16 if PRECISION == "float16" else torch.float32,
            ).to(device).eval()
            # MMS publica un adaptador por idioma; sin cargarlo transcribe en inglés.
            if "mms" in MODELO.lower():
                procesador.tokenizer.set_target_lang("spa")
                red.load_adapter("spa")
        return (procesador, red, device)

    if BACKEND == "vosk":
        from vosk import Model as ModeloVosk
        from vosk import SetLogLevel

        # Vosk vuelca su log desde C++ por stderr nada más construir el modelo.
        SetLogLevel(-1)
        with sin_ruido():
            return ModeloVosk(str(_ruta_modelo_vosk(MODELO)))

    # Los tres backends de la comparativa están implementados arriba. Llegar
    # aquí significa que BACKEND tiene un valor que no existe, que es un error
    # de configuración, no una funcionalidad pendiente.
    raise ValueError(
        f"BACKEND='{BACKEND}' desconocido. Valores válidos: "
        "'faster-whisper', 'transformers', 'vosk'."
    )


def _ruta_modelo_vosk(nombre: str) -> pathlib.Path:
    """Localiza la carpeta del modelo Vosk. No descarga nada.

    El modelo no viaja con el repositorio (`modelos/` está en .gitignore): se
    descarga una vez desde https://alphacephei.com/vosk/models y se descomprime
    aquí, o se apunta a él con la variable de entorno VOSK_MODEL_PATH.
    """
    candidatas = [DIR_MODELOS / nombre,
                  pathlib.Path(nombre),
                  pathlib.Path.home() / "modelos" / nombre]
    desde_entorno = os.environ.get("VOSK_MODEL_PATH")
    if desde_entorno:
        candidatas.insert(0, pathlib.Path(desde_entorno))

    for i in range(len(candidatas)):
        if candidatas[i].is_dir():
            return candidatas[i]

    raise FileNotFoundError(
        f"No encuentro el modelo Vosk '{nombre}'. Descomprimidlo en "
        f"{DIR_MODELOS / nombre} o apuntad VOSK_MODEL_PATH a su carpeta."
    )


def _en_blackwell() -> bool:
    """True si la GPU visible es la RTX PRO 5000 de NIMROD (sm_120)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return False
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mayor, _ = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            return mayor >= 12
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


def _transcribir(modelo, ruta_audio: str) -> str:
    """Devuelve el texto crudo que produce el modelo."""
    if BACKEND == "faster-whisper":
        with sin_ruido():
            segmentos, _ = modelo.transcribe(ruta_audio, language=IDIOMA, beam_size=5)
            return " ".join(s.text for s in segmentos).strip()

    if BACKEND == "transformers":
        import soundfile as sf
        import torch

        procesador, red, device = modelo
        audio, sr = sf.read(ruta_audio, dtype="float32")
        entradas = procesador(audio, sampling_rate=sr, return_tensors="pt")
        entradas = {k: v.to(device) for k, v in entradas.items()}
        if red.dtype == torch.float16 and "input_values" in entradas:
            entradas["input_values"] = entradas["input_values"].half()
        with torch.no_grad():
            logits = red(**entradas).logits
        ids = torch.argmax(logits, dim=-1)
        return procesador.batch_decode(ids)[0].strip()

    if BACKEND == "vosk":
        from vosk import KaldiRecognizer

        # El reconocedor lleva estado: uno nuevo por fichero. El modelo, que es
        # lo caro de cargar, se reutiliza entre llamadas gracias al lru_cache.
        with wave.open(ruta_audio, "rb") as onda:
            reconocedor = KaldiRecognizer(modelo, onda.getframerate())
            reconocedor.SetWords(False)
            while True:
                trozo = onda.readframes(4000)
                if len(trozo) == 0:
                    break
                reconocedor.AcceptWaveform(trozo)
            final = json.loads(reconocedor.FinalResult())
        return final.get("text", "").strip()

    raise ValueError(
        f"BACKEND='{BACKEND}' desconocido. Valores válidos: "
        "'faster-whisper', 'transformers', 'vosk'."
    )


def transcribe(ruta_audio: str) -> str:
    """Transcribe un fichero de audio y devuelve el texto sin normalizar.

    Interfaz pública de la práctica. El modelo se carga en la primera llamada
    y se reutiliza en las siguientes.
    """
    if not os.path.exists(ruta_audio):
        raise FileNotFoundError(ruta_audio)
    return _transcribir(_cargar_modelo(), ruta_audio)


def precalentar() -> float:
    """Fuerza la carga del modelo y devuelve el tiempo en segundos.

    Llamadlo antes de cronometrar la inferencia: así el t_carga queda medido
    por separado y no contamina el RTF.
    """
    from scripts.metrics import cronometro

    with cronometro() as t:
        _cargar_modelo()
    return t.segundos


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("uso: python3 transcribe.py <fichero.wav>")
        raise SystemExit(1)
    print(f"carga: {precalentar():.2f} s")
    print(transcribe(sys.argv[1]))
