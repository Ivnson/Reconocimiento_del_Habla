#!/usr/bin/env python3
"""
Genera las condiciones acústicas degradadas del conjunto de evaluación.

    python -I scripts/degradar.py --eval-set eval_set --salida eval_set_degradado

Produce un directorio por condición, con los mismos identificadores de fichero
que el conjunto original, de modo que el manifiesto sirve igual para todas:

    eval_set_degradado/
    ├── snr20/     ruido aditivo a 20 dB
    ├── snr10/     ruido aditivo a 10 dB
    ├── snr5/      ruido aditivo a 5 dB
    ├── telefono/  banda limitada: 16k -> 8k -> 16k
    └── codec/     Opus 16 kbps, ida y vuelta

El ruido se genera sintéticamente a partir de una semilla fija: la salida es
reproducible bit a bit en cualquier máquina y no arrastra la licencia de
ningún corpus de ruido.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SEMILLA = 20262  # fija a propósito: todos los equipos degradan igual
SR = 16_000
SNRS = (20, 10, 5)


def ruido_blanco(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def ruido_rosa(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ruido 1/f por filtrado en frecuencia. Más parecido al ruido ambiente
    real que el blanco, que concentra demasiada energía en agudos."""
    blanco = rng.standard_normal(n)
    espectro = np.fft.rfft(blanco)
    frecuencias = np.fft.rfftfreq(n, d=1.0 / SR)
    escala = np.ones_like(frecuencias)
    escala[1:] = 1.0 / np.sqrt(frecuencias[1:])
    rosa = np.fft.irfft(espectro * escala, n=n)
    pico = np.max(np.abs(rosa))
    return rosa / pico if pico > 0 else rosa


def mezclar_a_snr(senal: np.ndarray, ruido: np.ndarray, snr_db: float) -> np.ndarray:
    """Escala el ruido para alcanzar el SNR pedido y lo suma."""
    p_senal = np.mean(senal**2)
    p_ruido = np.mean(ruido**2)
    if p_ruido == 0 or p_senal == 0:
        return senal
    factor = np.sqrt(p_senal / (p_ruido * 10 ** (snr_db / 10)))
    mezcla = senal + factor * ruido
    # Evitar recorte por saturación, que sería una degradación extra no
    # controlada y distinta en cada fichero.
    pico = np.max(np.abs(mezcla))
    return mezcla / pico * 0.98 if pico > 1.0 else mezcla


def _ffmpeg(args: list[str]) -> None:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {r.stderr.strip()}")


def a_telefono(entrada: pathlib.Path, salida: pathlib.Path) -> None:
    """Limitación de banda: baja a 8 kHz y vuelve a subir a 16 kHz.

    La vuelta a 16 kHz es necesaria porque los modelos esperan esa tasa; lo
    que se pierde es el contenido por encima de 4 kHz, que es justo donde
    viven las fricativas (/s/, /f/, /θ/). Es ahí donde suele aparecer la
    degradación más interesante de analizar.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        intermedio = pathlib.Path(tmp.name)
    try:
        _ffmpeg(["-i", str(entrada), "-ar", "8000", "-ac", "1", str(intermedio)])
        _ffmpeg(["-i", str(intermedio), "-ar", str(SR), "-ac", "1", str(salida)])
    finally:
        intermedio.unlink(missing_ok=True)


def a_codec(entrada: pathlib.Path, salida: pathlib.Path) -> None:
    """Opus a 16 kbps, ida y vuelta. Simula audio transmitido por VoIP."""
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tmp:
        intermedio = pathlib.Path(tmp.name)
    try:
        _ffmpeg(["-i", str(entrada), "-c:a", "libopus", "-b:a", "16k",
                 "-ac", "1", str(intermedio)])
        _ffmpeg(["-i", str(intermedio), "-ar", str(SR), "-ac", "1", str(salida)])
    finally:
        intermedio.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--salida", default="eval_set_degradado")
    args = ap.parse_args()

    origen = pathlib.Path(args.eval_set) / "audio"
    if not origen.is_dir():
        print(f"No encuentro {origen}", file=sys.stderr)
        return 1

    destino = pathlib.Path(args.salida)
    ficheros = sorted(origen.glob("*.wav"))
    if not ficheros:
        print(f"No hay WAV en {origen}", file=sys.stderr)
        return 1

    print(f"{len(ficheros)} ficheros, 5 condiciones")

    for cond in [f"snr{s}" for s in SNRS] + ["telefono", "codec"]:
        (destino / cond).mkdir(parents=True, exist_ok=True)

    # Una semilla por fichero, derivada de la global: el ruido de cada
    # enunciado es reproducible aunque se procese en otro orden o en paralelo.
    for i, f in enumerate(ficheros):
        audio, sr = sf.read(f, dtype="float64", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            print(f"  [aviso] {f.name} está a {sr} Hz, se esperaba {SR}")

        rng = np.random.default_rng(SEMILLA + i)
        # Mitad blanco, mitad rosa: da variedad espectral sin introducir
        # una variable extra que haya que controlar en el análisis.
        ruido = ruido_blanco(len(audio), rng) if i % 2 == 0 else ruido_rosa(len(audio), rng)

        for snr in SNRS:
            sf.write(destino / f"snr{snr}" / f.name,
                     mezclar_a_snr(audio, ruido, snr), SR, subtype="PCM_16")

        a_telefono(f, destino / "telefono" / f.name)
        a_codec(f, destino / "codec" / f.name)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ficheros)}")

    print(f"Listo en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
